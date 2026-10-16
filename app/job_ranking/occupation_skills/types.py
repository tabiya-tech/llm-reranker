"""The occupation-skills menu and the LLM 3 result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils.occupation_to_skills_mapping import (
    OccupationTaxonomyStats,
)


@dataclass(frozen=True)
class OccupationTaxonomyOutcome:
    """What the taxonomy step resolved for one job."""

    # Unique ESCO uuids linked via occupation relations (for logging / analytics).
    relation_skill_uuids: list[str]
    stats: OccupationTaxonomyStats | None
    # ``occupation_to_skills`` items, after ``RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE``.
    occupation_to_skills: list[dict[str, Any]]
    # Essential + optional skills on ``occupation_to_skills`` (the LLM3 menu before its filters).
    menu_size: int


@dataclass(frozen=True)
class OccupationSkillsRelevanceDecision:
    """One row from LLM call 3 JSON (occupation taxonomy skill vs this job ad)."""

    skill_id: str
    tabiya_occupation_id: str
    is_relevant: bool


@dataclass(frozen=True)
class OccupationSkillsRelevanceResult:
    """Parsed LLM call 3 output: flags only, no summary or reasons."""

    decisions: tuple[OccupationSkillsRelevanceDecision, ...]
    raw_response: str
