"""Job-attributes schema, ONET work activities and the LLM 2 result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AttributeLevel:
    id: str
    label: str
    value: float | None = None


@dataclass(frozen=True)
class AttributeDefinition:
    name: str
    label: str
    group: str
    type: str
    coding: str
    levels: tuple[AttributeLevel, ...]
    base_level_id: str | None = None


@dataclass(frozen=True)
class JobAttributesSchema:
    attributes: tuple[AttributeDefinition, ...]
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class OnetWorkActivity:
    wa_element_id: str
    wa_element_name: str
    wa_element_name_simplified: str
    wa_element_name_swahili: str | None = None


@dataclass(frozen=True)
class AttributeEstimate:
    """One attribute row from LLM2 JSON (name, level, confidence, reasoning)."""

    attribute_name: str
    selected_level_id: str
    confidence: str
    reasoning: str


@dataclass(frozen=True)
class JobAttributesResult:
    """Parsed output of LLM call 2: structured estimates + summary + ONET work activities."""

    attribute_estimates: tuple[AttributeEstimate, ...]
    summary: str
    attributes: dict[str, str | bool]
    work_activity_ids: tuple[str, ...]
    raw_response: str
