"""A language pack's prompt texts."""

from __future__ import annotations

from pydantic import Field

from app.domain_base import DomainModel


class PromptTemplates(DomainModel):
    """A language pack's prompt texts, with ``{country}`` / ``{language}`` already filled in."""

    skill_ranker: str = Field(
        description="`Skills_reranker_prompts.md`: `str.format(full_details=..., skills_text=...)`."
    )
    job_attributes_system: str = Field(description="`prompts.py` `JOB_ATTRIBUTES_SYSTEM_PROMPT`.")
    occupation_skills_relevance_system: str = Field(
        description="`prompts.py` `OCCUPATION_SKILLS_RELEVANCE_SYSTEM_PROMPT`."
    )
