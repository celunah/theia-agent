"""Theia-owned usage accounting kept separate from general session storage."""

from __future__ import annotations

import math
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from .policy import _TOKEN_USAGE_KEYS, _USAGE_DAILY_LIMIT
from .usage import (
    estimate_api_cost,
    PROMPT_CATEGORIES,
    USAGE_TURN_LIMIT,
    normalize_tokens,
)
from ..core import _Session

LONG_RUNNING_TURN_SECONDS = 60.0


def initialize_usage_state(server: Any) -> None:
    """Initialize usage counters added after the original state schema."""
    server._usage_api_calls = 0
    server._usage_api_calls_daily = {}
    server._usage_subagent_turns = 0
    server._usage_subagent_turns_daily = {}
    server._usage_long_running_turns_daily = {}


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
        _usage_api_calls: int
        _usage_api_calls_daily: dict[str, int]
        _usage_subagent_turns: int
        _usage_subagent_turns_daily: dict[str, int]
        _usage_long_running_turns_daily: dict[str, int]
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

    def _record_usage_api_call(self, session: _Session) -> None:
        """Count a started user or internal turn without retaining its input."""
        day = time.strftime("%Y-%m-%d", time.gmtime())
        if self._is_internal_usage_session(session):
            self._usage_subagent_turns += 1
            self._usage_subagent_turns_daily[day] = (
                self._usage_subagent_turns_daily.get(day, 0) + 1
            )
        else:
            self._usage_api_calls += 1
            self._usage_api_calls_daily[day] = (
                self._usage_api_calls_daily.get(day, 0) + 1
            )
        self._persist_state()

    def _record_usage_turn_duration(
        self,
        duration: float,
        session: _Session,
        turn_id: str | None = None,
    ) -> None:
        if self._is_internal_usage_session(session):
            return
        if math.isfinite(duration) and duration >= 0:
            self._usage_longest_running_turn_sec = max(
                self._usage_longest_running_turn_sec, duration
            )
            if duration >= LONG_RUNNING_TURN_SECONDS:
                day = time.strftime("%Y-%m-%d", time.gmtime())
                self._usage_long_running_turns_daily[day] = (
                    self._usage_long_running_turns_daily.get(day, 0) + 1
                )
            if turn_id and session.thread_id:
                record = self._usage_turns.get(f"{session.thread_id}:{turn_id}")
                if isinstance(record, dict):
                    previous = record.get("durationSec", 0.0)
                    if (
                        not isinstance(previous, (int, float))
                        or isinstance(previous, bool)
                        or not math.isfinite(float(previous))
                        or previous < 0
                    ):
                        previous = 0.0
                    record["durationSec"] = max(0.0, float(previous), duration)
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
        estimate = estimate_api_cost(turn_records)
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
        has_daily_usage = bool(daily_breakdown) or selected_day in self._usage_daily
        daily_input = daily_breakdown.get("inputTokens")
        daily_cached = daily_breakdown.get("cachedInputTokens")
        daily_output = daily_breakdown.get("outputTokens")
        if not has_daily_usage:
            daily_input = daily_cached = daily_output = 0
        processed_tokens = self._processed_tokens(
            daily_input, daily_cached, daily_output
        )
        cumulative_processed_values = [
            self._processed_snapshot(thread_id, snapshot)
            for thread_id, snapshot in self._usage_threads.items()
        ]
        cumulative_processed = (
            sum(value for value in cumulative_processed_values if value is not None)
            if all(value is not None for value in cumulative_processed_values)
            else None
        )
        api_calls = self._usage_api_calls_daily.get(selected_day)
        if api_calls is None:
            api_calls = len(turn_records)
        subagent_turns = self._usage_subagent_turns_daily.get(selected_day)
        if subagent_turns is None:
            subagent_turns = len(internal_records)
        long_running_turns = self._usage_long_running_turns_daily.get(selected_day)
        if long_running_turns is None:
            long_running_turns = sum(
                1
                for record in turn_records
                if isinstance(record.get("durationSec"), (int, float))
                and not isinstance(record.get("durationSec"), bool)
                and record["durationSec"] >= LONG_RUNNING_TURN_SECONDS
            )
        exact = {
            "inputTokens": daily_input,
            "cachedInputTokens": daily_cached,
            "outputTokens": daily_output,
            "reasoningOutputTokens": daily_breakdown.get("reasoningOutputTokens"),
            "totalTokens": daily_total_value,
            "totalProcessedTokens": processed_tokens,
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
                "apiCalls": api_calls,
                "subagentTurns": subagent_turns,
                "longRunningTurns": long_running_turns,
                "subagentUsage": internal_usage,
                "unattributedOverhead": {
                    "value": unattributed,
                    "estimated": unattributed is not None,
                },
            },
            "summary": {
                "lifetimeTokens": totals["totalTokens"],
                "totalCumulativeTokens": totals["totalTokens"],
                "totalCumulativeProcessedTokens": cumulative_processed,
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
                "totalProcessedTokens": exact["totalProcessedTokens"],
            },
        }

    @staticmethod
    def _processed_tokens(
        input_tokens: Any, cached_tokens: Any, output_tokens: Any
    ) -> int | None:
        values = (input_tokens, cached_tokens, output_tokens)
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in values
        ):
            return None
        return sum(values)

    def _processed_snapshot(
        self, thread_id: str, snapshot: dict[str, int]
    ) -> int | None:
        known_fields = self._usage_thread_fields.get(thread_id, set())
        if not {"inputTokens", "cachedInputTokens", "outputTokens"}.issubset(
            known_fields
        ):
            return None
        return self._processed_tokens(
            snapshot.get("inputTokens"),
            snapshot.get("cachedInputTokens"),
            snapshot.get("outputTokens"),
        )

    def _restore_usage_state(self, usage: Any) -> None:
        """Restore only the bounded, Theia-owned usage subsection."""
        if not isinstance(usage, dict):
            return
        usage_threads = usage.get("threads")
        if isinstance(usage_threads, dict):
            for thread_id, snapshot in usage_threads.items():
                if isinstance(thread_id, str) and isinstance(snapshot, dict):
                    self._usage_threads[thread_id] = self._token_usage_breakdown(
                        snapshot
                    )
                    self._usage_thread_fields[thread_id] = {
                        key
                        for key, value in self._usage_threads[thread_id].items()
                        if value > 0
                    }
        thread_fields = usage.get("thread_fields")
        if isinstance(thread_fields, dict):
            for thread_id, fields in thread_fields.items():
                if not isinstance(thread_id, str) or not isinstance(fields, list):
                    continue
                self._usage_thread_fields[thread_id] = {
                    field for field in fields if field in _TOKEN_USAGE_KEYS
                }
        usage_daily = usage.get("daily_tokens")
        if isinstance(usage_daily, dict):
            self._usage_daily = {
                str(day): value
                for day, value in usage_daily.items()
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(day))
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
            }
            self._usage_daily = dict(
                sorted(self._usage_daily.items())[-_USAGE_DAILY_LIMIT:]
            )
        daily_breakdown = usage.get("daily_breakdown")
        if isinstance(daily_breakdown, dict):
            restored_breakdown: dict[str, dict[str, int]] = {}
            for day, values in daily_breakdown.items():
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(day)):
                    continue
                normalized = normalize_tokens(values)
                if normalized:
                    restored_breakdown[str(day)] = normalized
            self._usage_daily_breakdown = dict(
                sorted(restored_breakdown.items())[-_USAGE_DAILY_LIMIT:]
            )
        usage_turns = usage.get("turns")
        if isinstance(usage_turns, dict):
            restored_turns: dict[str, dict[str, Any]] = {}
            for key, record in usage_turns.items():
                if not isinstance(key, str) or not isinstance(record, dict):
                    continue
                day = record.get("day")
                tokens = normalize_tokens(record.get("tokens"))
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(day)) or not tokens:
                    continue
                attribution = record.get("attribution")
                safe_attribution = {
                    category: value
                    for category in PROMPT_CATEGORIES
                    if isinstance(attribution, dict)
                    and isinstance(value := attribution.get(category), int)
                    and not isinstance(value, bool)
                    and value >= 0
                }
                recorded_at = record.get("recorded_at", 0.0)
                if not isinstance(recorded_at, (int, float)) or isinstance(
                    recorded_at, bool
                ):
                    recorded_at = 0.0
                restored_turns[key] = {
                    "day": str(day),
                    "recorded_at": max(0.0, float(recorded_at)),
                    "model": str(record.get("model"))[:120]
                    if record.get("model")
                    else None,
                    "effort": str(record.get("effort"))[:32]
                    if record.get("effort")
                    else None,
                    "tokens": tokens,
                    "attribution": safe_attribution,
                }
                duration = record.get("durationSec")
                if (
                    isinstance(duration, (int, float))
                    and not isinstance(duration, bool)
                    and math.isfinite(float(duration))
                    and duration >= 0
                ):
                    restored_turns[key]["durationSec"] = float(duration)
            self._usage_turns = dict(
                sorted(
                    restored_turns.items(),
                    key=lambda item: float(item[1].get("recorded_at", 0)),
                )[-USAGE_TURN_LIMIT:]
            )
        internal_turns = usage.get("internal_turns")
        if isinstance(internal_turns, dict):
            restored_internal: dict[str, dict[str, Any]] = {}
            for key, record in internal_turns.items():
                if not isinstance(key, str) or not isinstance(record, dict):
                    continue
                day = record.get("day")
                tokens = normalize_tokens(record.get("tokens"))
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(day)) or not tokens:
                    continue
                recorded_at = record.get("recorded_at", 0)
                restored_internal[key] = {
                    "day": str(day),
                    "recorded_at": max(
                        0.0,
                        float(recorded_at)
                        if isinstance(recorded_at, (int, float))
                        and not isinstance(recorded_at, bool)
                        else 0.0,
                    ),
                    "model": str(record.get("model"))[:120]
                    if record.get("model")
                    else None,
                    "effort": str(record.get("effort"))[:32]
                    if record.get("effort")
                    else None,
                    "tokens": tokens,
                }
            self._usage_internal_turns = dict(
                sorted(
                    restored_internal.items(),
                    key=lambda item: float(item[1].get("recorded_at", 0)),
                )[-USAGE_TURN_LIMIT:]
            )
        for attribute, key in (
            ("_usage_failed_turns", "failed_turns"),
            ("_usage_retries", "retries"),
            ("_usage_api_calls", "api_calls"),
            ("_usage_subagent_turns", "subagent_turns"),
        ):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(self, attribute, max(0, value))
        for attribute, key in (
            ("_usage_failed_daily", "failed_daily"),
            ("_usage_retries_daily", "retries_daily"),
            ("_usage_api_calls_daily", "api_calls_daily"),
            ("_usage_subagent_turns_daily", "subagent_turns_daily"),
            ("_usage_long_running_turns_daily", "long_running_turns_daily"),
        ):
            values = usage.get(key)
            if not isinstance(values, dict):
                continue
            setattr(
                self,
                attribute,
                {
                    str(day): value
                    for day, value in values.items()
                    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(day))
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                },
            )
        tracked_since = usage.get("tracked_since")
        if (
            isinstance(tracked_since, (int, float))
            and not isinstance(tracked_since, bool)
            and math.isfinite(float(tracked_since))
            and tracked_since > 0
        ):
            self._usage_tracked_since = float(tracked_since)
        longest_turn = usage.get("longest_running_turn_sec")
        if (
            isinstance(longest_turn, (int, float))
            and not isinstance(longest_turn, bool)
            and math.isfinite(float(longest_turn))
        ):
            self._usage_longest_running_turn_sec = max(0.0, float(longest_turn))

    def _serialize_usage_state(self) -> dict[str, Any]:
        """Return the bounded usage subsection for the atomic state writer."""
        return {
            "threads": {
                thread_id: dict(snapshot)
                for thread_id, snapshot in self._usage_threads.items()
            },
            "thread_fields": {
                thread_id: sorted(fields)
                for thread_id, fields in self._usage_thread_fields.items()
            },
            "daily_tokens": dict(self._usage_daily),
            "daily_breakdown": {
                day: dict(values) for day, values in self._usage_daily_breakdown.items()
            },
            "turns": {
                key: {
                    "day": record.get("day"),
                    "recorded_at": record.get("recorded_at"),
                    "model": record.get("model"),
                    "effort": record.get("effort"),
                    "tokens": dict(record.get("tokens", {})),
                    "attribution": dict(record.get("attribution", {})),
                    "durationSec": record.get("durationSec"),
                }
                for key, record in self._usage_turns.items()
            },
            "internal_turns": {
                key: {
                    "day": record.get("day"),
                    "recorded_at": record.get("recorded_at"),
                    "model": record.get("model"),
                    "effort": record.get("effort"),
                    "tokens": dict(record.get("tokens", {})),
                }
                for key, record in self._usage_internal_turns.items()
            },
            "failed_turns": self._usage_failed_turns,
            "failed_daily": dict(self._usage_failed_daily),
            "retries": self._usage_retries,
            "retries_daily": dict(self._usage_retries_daily),
            "tracked_since": self._usage_tracked_since,
            "longest_running_turn_sec": self._usage_longest_running_turn_sec,
            "api_calls": self._usage_api_calls,
            "api_calls_daily": dict(self._usage_api_calls_daily),
            "subagent_turns": self._usage_subagent_turns,
            "subagent_turns_daily": dict(self._usage_subagent_turns_daily),
            "long_running_turns_daily": dict(self._usage_long_running_turns_daily),
        }
