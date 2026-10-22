"""
Reads the prompt templates in ``resources/prompts/``: one per LLM call, written once in
English for every country and language (see README, "Prompts").

The templates are ``str.format`` text: ``{slot}`` is filled by the call's prompt builder,
and a literal brace is doubled (``{{`` / ``}}``), as in the JSON examples.
"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.job_ranking.prompts.templates import PromptTemplates

PROMPTS_SUBDIR = "prompts"


def prompts_dir(settings: Settings) -> Path:
    """``resources/prompts/`` (under ``RESOURCES_DIR`` when that is set)."""
    return settings.resources_dir / PROMPTS_SUBDIR


def _read_template(directory: Path, name: str) -> str:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8").rstrip("\n")


class PromptRepository:
    """Reads the prompt templates; a missing one fails startup, not a job."""

    def load(self, settings: Settings) -> PromptTemplates:
        directory = prompts_dir(settings)
        return PromptTemplates(
            skill_ranker=_read_template(directory, "skill_ranker.md"),
            job_attributes=_read_template(directory, "job_attributes.md"),
            occupation_skills_relevance=_read_template(directory, "occupation_skills_relevance.md"),
            answer_language=_read_template(directory, "answer_language.md"),
        )
