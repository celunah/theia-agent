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
    """Canonical model pricing in USD per million tokens."""

    input_miss: float
    input_hit: float
    output: float
    reasoning: float
    canonical_id: str = ""
    display_name: str = ""
    reasoning_token_pricing: str = "output_rate"
    pricing_source: str = ""
    pricing_version: str = ""
    effective_date: str = ""
    fast_mode_multiplier: float = 1.5

    @property
    def cache_miss_input_per_million(self) -> float:
        return self.input_miss

    @property
    def cache_hit_input_per_million(self) -> float:
        return self.input_hit

    @property
    def output_per_million(self) -> float:
        return self.output


# Codex does not expose subscription billing. These are centralized,
# replaceable API pricing rates used only for a clearly labelled estimate.
PRICING_REGISTRY: dict[str, ModelPricing] = {
    "gpt-5.6-luna": ModelPricing(
        0.20,
        0.02,
        1.20,
        1.20,
        "gpt-5.6-luna",
        "GPT-5.6 Luna",
        "output_rate",
        "user-provided Theia pricing",
        "1.0",
        "2026-09-15",
    ),
    "gpt-5.6-terra": ModelPricing(
        2.00,
        0.20,
        12.00,
        12.00,
        "gpt-5.6-terra",
        "GPT-5.6 Terra",
        "output_rate",
        "user-provided Theia pricing",
        "1.0",
        "2026-09-15",
    ),
    "gpt-5.6-sol": ModelPricing(
        4.00,
        0.40,
        20.00,
        20.00,
        "gpt-5.6-sol",
        "GPT-5.6 Sol",
        "output_rate",
        "user-provided Theia pricing",
        "1.0",
        "2026-09-15",
    ),
    "gpt-6-astra": ModelPricing(
        10.00,
        1.00,
        50.00,
        50.00,
        "gpt-6-astra",
        "GPT-6 Astra",
        "output_rate",
        "user-provided Theia pricing",
        "1.0",
        "2026-09-15",
    ),
}
# Keep the historical name available to callers that imported the registry.
MODEL_PRICING = PRICING_REGISTRY
MODEL_IDS_BY_DISPLAY_NAME = {
    pricing.display_name.casefold(): model_id
    for model_id, pricing in PRICING_REGISTRY.items()
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
    canonical = _canonical_model_id(model)
    if canonical is not None:
        return PRICING_REGISTRY[canonical].display_name
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
    canonical = _canonical_model_id(model)
    return PRICING_REGISTRY.get(canonical) if canonical else None


def _canonical_model_id(model: Any) -> str | None:
    value = str(model or "").strip().casefold()
    if value in PRICING_REGISTRY:
        return value
    return MODEL_IDS_BY_DISPLAY_NAME.get(value)


def estimate_api_cost(
    records: Iterable[dict[str, Any]], *, fallback_model: Any = None
) -> dict[str, Any]:
    """Estimate API cost from usage records, using a current-model fallback."""
    by_model: dict[str, float] = {}
    unavailable_models: set[str] = set()
    incomplete_records: set[str] = set()
    observed_usage = False
    for record in records:
        if not isinstance(record, dict):
            continue
        tokens = normalize_tokens(record.get("tokens"))
        if not tokens:
            continue
        observed_usage = True
        if not any(
            key in tokens
            for key in ("inputTokens", "cachedInputTokens", "outputTokens")
        ):
            incomplete_records.add(model_label(record.get("model")))
            continue
        model = record.get("model") or fallback_model
        pricing = _pricing_for(model)
        if pricing is None:
            unavailable_models.add(model_label(model))
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
        if (
            record.get("fastMode") is True
            or str(record.get("mode") or "").casefold() == "fast"
            or str(record.get("speed") or "").casefold() == "fast"
            or effort == "fast"
        ):
            value *= pricing.fast_mode_multiplier
        label = model_label(model)
        by_model[label] = by_model.get(label, 0.0) + value
    total = sum(by_model.values())
    pricing_available = not unavailable_models and not incomplete_records
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
        "incompleteRecords": sorted(incomplete_records),
    }


def estimate_api_value(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Backward-compatible name for the USD API cost estimate."""
    return estimate_api_cost(records)
