"""``POST /rank/skills`` result."""

from __future__ import annotations

from pydantic import Field

from app.domain_base import DomainModel


class ExperienceSkillsResult(DomainModel):
    """Parsed ranking of one experience's candidate skills, in the caller's ids."""

    ranked_ids: list[str] = Field(description="Caller ids of the demonstrated skills, best first.")
    discarded_ids: list[str] = Field(description="Every other caller id, in request order.")
    raw_response: str = Field(description="The model's raw response text.")
