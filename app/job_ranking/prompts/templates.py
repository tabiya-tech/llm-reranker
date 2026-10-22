"""The prompt templates, as read from ``resources/prompts/``."""

from __future__ import annotations

from pydantic import Field

from app.domain_base import DomainModel


class PromptTemplates(DomainModel):
    """One template per LLM call, plus the answer-language block; slots not yet filled."""

    skill_ranker: str = Field(
        description="`skill_ranker.md` (LLM1): `{full_details}`, `{skills_text}`, "
        "`{answer_language}`."
    )
    job_attributes: str = Field(
        description="`job_attributes.md` (LLM2): `{country}`, `{opening_line}`, "
        "`{occupation_block}`, `{job_listing}`, `{listing_metadata}`, `{attributes_menu}`, "
        "`{work_activities_menu}`, `{extra_field_instructions}`, `{extra_field_json}`, "
        "`{answer_language}`."
    )
    occupation_skills_relevance: str = Field(
        description="`occupation_skills_relevance.md` (LLM3): `{country}`, `{job_listing}`, "
        "`{listing_metadata}`, `{skills_menu}`, `{answer_language}`."
    )
    answer_language: str = Field(
        description="`answer_language.md`: `{language}`. Added to every prompt when the run's "
        "language is not English."
    )
