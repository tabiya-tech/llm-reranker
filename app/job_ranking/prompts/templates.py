"""The prompt templates, as read from ``resources/prompts/``."""

from __future__ import annotations

from pydantic import Field

from app.domain_base import DomainModel


class PromptTemplates(DomainModel):
    """One template per LLM call; slots not yet filled."""

    skill_ranker: str = Field(
        description="`skill_ranker.md` (LLM1): `{full_details}`, `{skills_text}`, `{language}`."
    )
    job_attributes: str = Field(
        description="`job_attributes.md` (LLM2): `{country}`, `{opening_line}`, "
        "`{occupation_block}`, `{job_listing}`, `{listing_metadata}`, `{attributes_menu}`, "
        "`{work_activities_menu}`, `{extra_field_instructions}`, `{extra_field_json}`, "
        "`{language}`."
    )
    occupation_skills_relevance: str = Field(
        description="`occupation_skills_relevance.md` (LLM3): `{country}`, `{job_listing}`, "
        "`{listing_metadata}`, `{skills_menu}`, `{language}`."
    )
