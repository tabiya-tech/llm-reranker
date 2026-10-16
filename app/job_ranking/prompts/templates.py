"""A language pack's prompt texts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplates:
    """A language pack's prompt texts, with ``{country}`` / ``{language}`` already filled in."""

    # ``Skills_reranker_prompts.md``: ``str.format(full_details=..., skills_text=...)``.
    skill_ranker: str
    # ``prompts.py`` ``JOB_ATTRIBUTES_SYSTEM_PROMPT``.
    job_attributes_system: str
    # ``prompts.py`` ``OCCUPATION_SKILLS_RELEVANCE_SYSTEM_PROMPT``.
    occupation_skills_relevance_system: str
