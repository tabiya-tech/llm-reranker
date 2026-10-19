"""The occupation-skills menu and the LLM 3 result."""

from __future__ import annotations

from pydantic import Field

from app.domain_base import DomainModel
from app.job_ranking.routes.types import OccupationSkillsBlock
from utils.occupation_to_skills_mapping import (
    OccupationTaxonomyStats,
)


class OccupationTaxonomyOutcome(DomainModel):
    """What the taxonomy step resolved for one job."""

    relation_skill_uuids: list[str] = Field(
        description="Unique ESCO uuids linked via occupation relations (for logging / analytics)."
    )
    stats: OccupationTaxonomyStats | None = Field(
        description="Resolution detail for terminal logs."
    )
    occupation_to_skills: list[OccupationSkillsBlock] = Field(
        description="`occupation_to_skills` items, after `RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE`."
    )
    menu_size: int = Field(
        description="Essential + optional skills on `occupation_to_skills` (the LLM3 menu before its filters)."
    )


class OccupationSkillsRelevanceDecision(DomainModel):
    """One row from LLM call 3 JSON (occupation taxonomy skill vs this job ad)."""

    skill_id: str = Field(description="ESCO skill id judged.")
    tabiya_occupation_id: str = Field(description="Tabiya occupation id the skill belongs to.")
    is_relevant: bool = Field(description="LLM3's relevance verdict for this skill.")


class OccupationSkillsRelevanceResult(DomainModel):
    """Parsed LLM call 3 output: flags only, no summary or reasons."""

    decisions: tuple[OccupationSkillsRelevanceDecision, ...] = Field(
        description="One decision per judged (occupation, skill) pair, deduplicated."
    )
    raw_response: str = Field(description="LLM3's raw response text, fence-stripped.")
