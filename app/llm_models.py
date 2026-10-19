"""Raw LLM JSON response shapes, and the shared JSON-decode step all three parsers share.

These models validate only the *shape* an LLM response is expected to have — they are the
parse boundary, not the business-rule boundary. Allowlist filtering (which attribute level
ids are legal, which ONET work-activity ids are known, …) stays in the
``app.job_ranking.*.parser`` modules' ``parse_*_response`` functions, which consume these
models and apply that filtering against separately-loaded reference data.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import Field

from app.domain_base import SourceRecord
from app.llm.errors import LlmParseError


def strip_json_fence(text: str) -> str:
    """Strip a leading/trailing Markdown ``` ```json ``` fence, if present."""
    t = text.strip()
    if not t.startswith("```"):
        return t
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _repair_json_common_llm_errors(s: str) -> str:
    """Strip trailing commas before ``}``/``]`` (frequent invalid JSON from models)."""
    t = s.strip()
    t = re.sub(r",(\s*[\]}])", r"\1", t)
    return t


def parse_llm_json_response(raw_text: str, *, repair_on_failure: bool = True) -> dict[str, Any]:
    """Strip a markdown fence, parse JSON, optionally retry once after stripping trailing commas.

    Raises :class:`~app.llm.errors.LlmParseError` (not ``json.JSONDecodeError``)
    on failure, so every parse boundary catches one consistent exception type.
    """
    cleaned = strip_json_fence(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        if not repair_on_failure:
            raise LlmParseError(f"LLM response is not valid JSON: {e}") from e
        try:
            data = json.loads(_repair_json_common_llm_errors(cleaned))
        except json.JSONDecodeError:
            raise LlmParseError(f"LLM response is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise LlmParseError("LLM response JSON must be an object")
    return data


class SkillRankerResponse(SourceRecord):
    """Raw shape of LLM1's JSON: skill labels sorted into essential / optional."""

    essential_skills: list[str] = Field(
        default_factory=list, description="Labels LLM1 ranked essential."
    )
    optional_skills: list[str] = Field(
        default_factory=list, description="Labels LLM1 ranked optional."
    )


class AttributeEstimateRaw(SourceRecord):
    """One raw attribute estimate row from LLM2's JSON, before allowlist filtering.

    Fields are loosely typed (``Any``/``None``-tolerant) on purpose: this model only
    confirms the row is an object. Business-rule filtering (does the name/level exist,
    is the value a non-blank string, …) happens afterwards in
    :mod:`app.job_ranking.job_attributes.parser`, which must tolerate the same
    malformed-row shapes the pre-Pydantic parser tolerated.
    """

    attribute_name: Any = Field(default=None, description="Attribute key the LLM is estimating.")
    selected_level_id: Any = Field(
        default=None, description="Level id the LLM selected for this attribute."
    )
    confidence: Any = Field(default=None, description="LLM's stated confidence.")
    reasoning: Any = Field(default=None, description="LLM's free-text reasoning.")


class JobAttributesLlmResponse(SourceRecord):
    """Raw shape of LLM2's JSON, before allowlist filtering against schema/ONET reference data.

    ``attributes``/``work_activities`` entries are intentionally loosely typed — see
    :class:`AttributeEstimateRaw`.
    """

    summary: Any = Field(default=None, description="LLM2's one-paragraph job summary.")
    attributes: list[Any] | dict[str, Any] = Field(
        default_factory=list,
        description="List of attribute estimates, or a legacy name->level_id object map.",
    )
    work_activities: list[Any] = Field(
        default_factory=list, description="Candidate ONET work-activity ids."
    )
    requires_post_secondary: Any = Field(
        default=None, description="Kenya eligibility flag, loosely typed."
    )


class RelevanceDecisionRaw(SourceRecord):
    """One raw relevance decision row from LLM3's JSON."""

    skill_id: str = Field(default="", description="ESCO skill id as the LLM echoed it.")
    tabiya_occupation_id: str = Field(
        default="", description="Tabiya occupation id as the LLM echoed it."
    )
    is_relevant: bool | str | None = Field(
        default=None, description="Relevance verdict, loosely typed."
    )
