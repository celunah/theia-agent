"""Bounded usage records, estimates, and prompt attribution helpers."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

TOKEN_FIELDS = (
    "cacheWriteInputTokens",
    "cachedInputTokens",
    "inputTokens",
    "outputTokens",
    "reasoningOutputTokens",
    "totalTokens",
)
USAGE_TURN_LIMIT = 512
PROMPT_CATEGORIES = (
    "system_instructions",
    "identity_self_model",
    "memory_data",
    "skill_data",
    "user_history",
    "tool_definitions",
    "tool_results",
    "routing_context",
    "subagent_context",
)


@dataclass(frozen=True)
class ModelPricing:
    """Central API estimate in USD per million tokens."""

    input_miss: float
    input_hit: float
    output: float
    reasoning: float


# Codex does not expose subscription billing. These are centralized,
# replaceable API pricing rates used only for a clearly labelled estimate.
MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-5.6-luna": ModelPricing(1.25, 0.125, 10.0, 10.0),
    "gpt-5.6-terra": ModelPricing(2.0, 0.2, 15.0, 15.0),
    "gpt-5.6-sol": ModelPricing(0.25, 0.025, 2.0, 2.0),
    "gpt-6-astra": ModelPricing(5.0, 0.5, 30.0, 30.0),
}
EFFORT_MULTIPLIERS = {
    "low": 0.75,
    "medium": 1.0,
    "high": 1.25,
    "xhigh": 1.5,
    "max": 1.75,
}


def normalize_tokens(value: Any) -> dict[str, int]:
    """Keep only non-negative integer fields reported by Codex."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key in TOKEN_FIELDS:
        number = value.get(key)
        if isinstance(number, int) and not isinstance(number, bool) and number >= 0:
            result[key] = number
    return result


def estimated_tokens(text: Any) -> int:
    """Estimate prompt tokens without pretending character counts are exact."""
    if not isinstance(text, str) or not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def model_label(model: Any) -> str:
    """Return a short safe display name for a model identifier."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "", str(model or "unknown"))[:80]
    if not value:
        return "Unknown model"
    if value.casefold().startswith("gpt-"):
        prefix, _, suffix = value.partition("-")
        return (
            prefix.upper() + "-" + " ".join(part.title() for part in suffix.split("-"))
        )
    return value.replace("-", " ").title()


def _pricing_for(model: Any) -> ModelPricing | None:
    return MODEL_PRICING.get(str(model or "").strip().casefold())


def estimate_api_cost(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Estimate API cost in USD from provider-reported per-turn usage."""
    by_model: dict[str, float] = {}
    unavailable_models: set[str] = set()
    observed_usage = False
    for record in records:
        if not isinstance(record, dict):
            continue
        tokens = normalize_tokens(record.get("tokens"))
        if not tokens:
            continue
        observed_usage = True
        pricing = _pricing_for(record.get("model"))
        if pricing is None:
            unavailable_models.add(model_label(record.get("model")))
            continue
        effort = str(record.get("effort") or "medium").casefold()
        effort_multiplier = EFFORT_MULTIPLIERS.get(effort, 1.0)
        value = (
            tokens.get("inputTokens", 0) * pricing.input_miss
            + tokens.get("cachedInputTokens", 0) * pricing.input_hit
            + tokens.get("outputTokens", 0) * pricing.output
        ) / 1_000_000
        reasoning = tokens.get("reasoningOutputTokens", 0)
        if reasoning:
            value += reasoning * pricing.reasoning * effort_multiplier / 1_000_000
        elif tokens.get("outputTokens", 0):
            value *= effort_multiplier
        label = model_label(record.get("model"))
        by_model[label] = by_model.get(label, 0.0) + value
    total = sum(by_model.values())
    pricing_available = not unavailable_models
    return {
        "estimated": True,
        "currency": "USD",
        "available": pricing_available,
        "total": (
            round(total, 8)
            if observed_usage and pricing_available
            else 0.0
            if not observed_usage
            else None
        ),
        "byModel": {key: round(value, 8) for key, value in by_model.items()},
        "unavailableModels": sorted(unavailable_models),
    }


def estimate_api_value(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Backward-compatible name for the USD API cost estimate."""
    return estimate_api_cost(records)
