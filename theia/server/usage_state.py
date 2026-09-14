"""Theia-owned usage accounting kept separate from general session storage."""

from __future__ import annotations

import math
import time
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from .policy import _TOKEN_USAGE_KEYS, _USAGE_DAILY_LIMIT
from .usage import (
    PROMPT_CATEGORIES,
    USAGE_TURN_LIMIT,
    estimate_api_value,
    normalize_tokens,
)
from ..core import _Session


class CodexUsageStateMixin:  # pylint: disable=no-member
    """Track provider totals and bounded local estimates for normal turns."""

    if TYPE_CHECKING:
        _usage_threads: dict[str, dict[str, int]]
        _usage_thread_fields: dict[str, set[str]]
        _usage_daily: dict[str, int]
        _usage_daily_breakdown: dict[str, dict[str, int]]
        _usage_turns: dict[str, dict[str, Any]]
        _usage_internal_turns: dict[str, dict[str, Any]]
        _usage_retries: int
        _usage_retries_daily: dict[str, int]
        _usage_failed_turns: int
        _usage_failed_daily: dict[str, int]
        _usage_tracked_since: float | None
        _usage_longest_running_turn_sec: float
        _sessions: dict[str, _Session]
        _turns: dict[str, Any]
        _persist_state: Any

    @staticmethod
    def _token_usage_breakdown(value: Any) -> dict[str, int]:
        result = {key: 0 for key in _TOKEN_USAGE_KEYS}
        if not isinstance(value, dict):
            return result
        for key in _TOKEN_USAGE_KEYS:
            number = value.get(key)
            if isinstance(number, int) and not isinstance(number, bool):
                result[key] = max(0, number)
        return result

    @staticmethod
    def _is_internal_usage_session(session: _Session) -> bool:
        return session.key.startswith("__")

    def _usage_thread_is_owned(self, thread_id: str) -> bool:
        if thread_id in self._usage_threads:
            return True
        return any(
            session.thread_id == thread_id
            and not self._is_internal_usage_session(session)
            for session in self._sessions.values()
        )

    def _claim_usage_thread(self, thread_id: str) -> None:
        if not thread_id or thread_id in self._usage_threads:
            return
        self._usage_threads[thread_id] = self._token_usage_breakdown(None)
        self._usage_thread_fields[thread_id] = set()
        if self._usage_tracked_since is None:
            self._usage_tracked_since = time.time()

    def _record_token_usage(self, params: dict[str, Any]) -> None:
        thread_id = params.get("threadId")
        if not isinstance(thread_id, str) or not thread_id:
            return
        turn_id = params.get("turnId")
        turn_state = (
            getattr(self, "_turns", {}).get(str(turn_id))
            if isinstance(turn_id, (str, int)) and not isinstance(turn_id, bool)
            else None
        )
        if not self._usage_thread_is_owned(thread_id):
            if getattr(getattr(turn_state, "session", None), "key", "").startswith(
                "__"
            ):
                self._record_internal_token_usage(
                    thread_id, turn_id, params, turn_state
                )
            return
        token_usage = params.get("tokenUsage")
        if not isinstance(token_usage, dict):
            return
        total_raw = normalize_tokens(token_usage.get("total"))
        last = normalize_tokens(token_usage.get("last"))
        current = self._token_usage_breakdown(total_raw)
        previous = self._usage_threads.setdefault(
            thread_id, self._token_usage_breakdown(None)
        )
        delta = max(0, current["totalTokens"] - previous["totalTokens"])
        self._usage_threads[thread_id] = current
        day = time.strftime("%Y-%m-%d", time.gmtime())
        if delta:
            self._usage_daily[day] = self._usage_daily.get(day, 0) + delta
            self._usage_daily = dict(
                sorted(self._usage_daily.items())[-_USAGE_DAILY_LIMIT:]
            )
        daily_breakdown = self._usage_daily_breakdown.setdefault(day, {})
        known_fields = self._usage_thread_fields.setdefault(
            thread_id,
            {key for key, value in previous.items() if value > 0},
        )
        for key, value in total_raw.items():
            increment = (
                max(0, value - previous.get(key, 0))
                if key in known_fields
                else last.get(key, 0)
            )
            daily_breakdown[key] = daily_breakdown.get(key, 0) + increment
            known_fields.add(key)
        self._usage_daily_breakdown = dict(
            sorted(self._usage_daily_breakdown.items())[-_USAGE_DAILY_LIMIT:]
        )
        if isinstance(turn_id, (str, int)) and not isinstance(turn_id, bool):
            if not last:
                last = {
                    key: max(0, value - previous.get(key, 0))
                    for key, value in total_raw.items()
                }
            model = getattr(turn_state, "model", None) or params.get("model")
            effort = getattr(turn_state, "effort", None) or params.get("effort")
            attribution = getattr(turn_state, "prompt_attribution", {})
            self._usage_turns[f"{thread_id}:{turn_id}"] = {
                "day": day,
                "recorded_at": time.time(),
                "model": str(model)[:120] if model else None,
                "effort": str(effort)[:32] if effort else None,
                "tokens": last,
                "attribution": {
                    category: max(0, int(attribution.get(category, 0)))
                    for category in PROMPT_CATEGORIES
                    if isinstance(attribution.get(category), int)
                    and not isinstance(attribution.get(category), bool)
                },
            }
            self._usage_turns = dict(
                sorted(
                    self._usage_turns.items(),
                    key=lambda item: float(item[1].get("recorded_at", 0)),
                )[-USAGE_TURN_LIMIT:]
            )
        if self._usage_tracked_since is None:
            self._usage_tracked_since = time.time()
        self._persist_state()

    def _record_internal_token_usage(
        self,
        thread_id: str,
        turn_id: Any,
        params: dict[str, Any],
        turn_state: Any,
    ) -> None:
        """Keep exact internal-worker totals separate from user-turn totals."""
        if not isinstance(turn_id, (str, int)) or isinstance(turn_id, bool):
            return
        token_usage = params.get("tokenUsage")
        if not isinstance(token_usage, dict):
            return
        tokens = normalize_tokens(token_usage.get("last"))
        if not tokens:
            tokens = normalize_tokens(token_usage.get("total"))
        if not tokens:
            return
        model = getattr(turn_state, "model", None) or params.get("model")
        effort = getattr(turn_state, "effort", None) or params.get("effort")
        self._usage_internal_turns[f"{thread_id}:{turn_id}"] = {
            "day": time.strftime("%Y-%m-%d", time.gmtime()),
            "recorded_at": time.time(),
            "model": str(model)[:120] if model else None,
            "effort": str(effort)[:32] if effort else None,
            "tokens": tokens,
        }
        self._usage_internal_turns = dict(
            sorted(
                self._usage_internal_turns.items(),
                key=lambda item: float(item[1].get("recorded_at", 0)),
            )[-USAGE_TURN_LIMIT:]
        )
        self._persist_state()

    def _record_usage_retry(self) -> None:
        self._usage_retries += 1
        day = time.strftime("%Y-%m-%d", time.gmtime())
        self._usage_retries_daily[day] = self._usage_retries_daily.get(day, 0) + 1
        self._persist_state()

    def _record_failed_usage_turn(self, state: Any, _turn: dict[str, Any]) -> None:
        session = getattr(state, "session", None)
        if session is None or self._is_internal_usage_session(session):
            return
        self._usage_failed_turns += 1
        day = time.strftime("%Y-%m-%d", time.gmtime())
        self._usage_failed_daily[day] = self._usage_failed_daily.get(day, 0) + 1
        self._persist_state()

    def _record_usage_turn_duration(self, duration: float, session: _Session) -> None:
        if self._is_internal_usage_session(session):
            return
        if math.isfinite(duration) and duration >= 0:
            self._usage_longest_running_turn_sec = max(
                self._usage_longest_running_turn_sec, duration
            )
            self._persist_state()

    def theia_usage(
        self, *, now: float | None = None, date_value: str | None = None
    ) -> dict[str, Any]:
        totals = self._token_usage_breakdown(None)
        for snapshot in self._usage_threads.values():
            for key in _TOKEN_USAGE_KEYS:
                totals[key] += snapshot.get(key, 0)
        daily = {
            day: value
            for day, value in self._usage_daily.items()
            if isinstance(value, int) and value > 0
        }
        active_days = set(daily)
        current_streak = 0
        longest_streak = 0
        if active_days:
            today = datetime.fromtimestamp(
                now if now is not None else time.time(), tz=timezone.utc
            ).date()
            cursor = today
            while cursor.isoformat() in active_days:
                current_streak += 1
                cursor -= timedelta(days=1)
            ordered_days: list[date] = []
            for day in active_days:
                try:
                    ordered_days.append(date.fromisoformat(day))
                except ValueError:
                    continue
            ordered_days.sort()
            streak = 0
            previous: date | None = None
            for active_day in ordered_days:
                if previous is not None and active_day == previous + timedelta(days=1):
                    streak += 1
                else:
                    streak = 1
                longest_streak = max(longest_streak, streak)
                previous = active_day
        selected_day = (
            datetime.fromtimestamp(
                now if now is not None else time.time(), tz=timezone.utc
            )
            .date()
            .isoformat()
        )
        if isinstance(date_value, str):
            try:
                selected_day = date.fromisoformat(date_value).isoformat()
            except ValueError:
                pass
        daily_breakdown = self._usage_daily_breakdown.get(selected_day, {})
        turn_records = [
            record
            for record in self._usage_turns.values()
            if isinstance(record, dict) and record.get("day") == selected_day
        ]
        categories: dict[str, dict[str, Any]] = {}
        for category in PROMPT_CATEGORIES:
            values = [
                record.get("attribution", {}).get(category)
                for record in turn_records
                if isinstance(record.get("attribution"), dict)
                and isinstance(record.get("attribution", {}).get(category), int)
            ]
            categories[category] = {
                "value": sum(values) if values else None,
                "estimated": bool(values),
            }
        known_input = (
            daily_breakdown.get("inputTokens"),
            daily_breakdown.get("cachedInputTokens"),
        )
        category_values = [
            value["value"]
            for value in categories.values()
            if isinstance(value.get("value"), int)
        ]
        unattributed = None
        if any(isinstance(value, int) for value in known_input) and category_values:
            unattributed = max(
                0,
                sum(value for value in known_input if isinstance(value, int))
                - sum(category_values),
            )
        estimate = estimate_api_value(turn_records)
        internal_records = [
            record
            for record in self._usage_internal_turns.values()
            if isinstance(record, dict) and record.get("day") == selected_day
        ]
        internal_totals = [
            normalize_tokens(record.get("tokens")).get("totalTokens")
            for record in internal_records
        ]
        internal_tokens = sum(value for value in internal_totals if value is not None)
        internal_usage: int | None = (
            internal_tokens
            if internal_records and all(value is not None for value in internal_totals)
            else None
        )
        daily_total = self._usage_daily.get(selected_day, 0)
        if "totalTokens" in daily_breakdown:
            daily_total_value: int | None = daily_breakdown["totalTokens"]
        elif selected_day in self._usage_daily:
            daily_total_value = daily_total
        elif daily_breakdown:
            daily_total_value = None
        else:
            daily_total_value = 0
        exact = {
            "inputTokens": daily_breakdown.get("inputTokens"),
            "cachedInputTokens": daily_breakdown.get("cachedInputTokens"),
            "outputTokens": daily_breakdown.get("outputTokens"),
            "reasoningOutputTokens": daily_breakdown.get("reasoningOutputTokens"),
            "totalTokens": daily_total_value,
        }
        return {
            "scope": "theia",
            "date": selected_day,
            "exact": exact,
            "estimate": estimate,
            "rateLimits": getattr(self, "_rate_limits", None),
            "detailed": {
                "categories": categories,
                "reasoningTokens": exact["reasoningOutputTokens"],
                "retries": self._usage_retries_daily.get(selected_day, 0),
                "failedTurns": self._usage_failed_daily.get(selected_day, 0),
                "subagentUsage": internal_usage,
                "unattributedOverhead": {
                    "value": unattributed,
                    "estimated": unattributed is not None,
                },
            },
            "summary": {
                "lifetimeTokens": totals["totalTokens"],
                "totalCumulativeTokens": totals["totalTokens"],
                "peakDailyTokens": max(daily.values(), default=0),
                "currentStreakDays": current_streak,
                "longestStreakDays": longest_streak,
                "longestRunningTurnSec": self._usage_longest_running_turn_sec,
                "date": selected_day,
                "inputTokens": exact["inputTokens"],
                "cachedInputTokens": exact["cachedInputTokens"],
                "outputTokens": exact["outputTokens"],
                "reasoningOutputTokens": exact["reasoningOutputTokens"],
                "totalTokens": exact["totalTokens"],
            },
        }
