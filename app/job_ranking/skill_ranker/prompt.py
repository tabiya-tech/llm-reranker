"""LLM 1 prompt: the job ad's full details plus the candidate skill menu. No IO."""

from __future__ import annotations

from app.job_ranking.prompts.templates import PromptTemplates
from utils.classifier_linked_entities import SkillPromptEntry


def _format_labeled_list(potential_skills: list[SkillPromptEntry]) -> str:
    """Format skill menu for the ranker: label, optional description; model returns labels verbatim."""
    lines: list[str] = []
    for i, s in enumerate(potential_skills, 1):
        desc = (s.description or "").strip()
        lines.append(f"{i}. {s.label}")
        if desc:
            lines.append(f"   Description: {desc}")
    return "\n".join(lines)


def format_skill_ranker_skills_text(potential_skills: list[SkillPromptEntry]) -> str:
    """The ``skills_text`` argument passed to ``str.format`` for ``Skills_reranker_prompts.md``."""
    return _format_labeled_list(potential_skills)


def build_skill_ranker_prompt(
    template: str,
    full_details: str,
    potential_skills: list[SkillPromptEntry],
) -> str:
    """
    Skill ranker user message from ``template`` (``resources/Skills_reranker_prompts.md``).

    ``potential_skills``: label and optional description (no URIs or scores).
    """
    skills_text = _format_labeled_list(potential_skills)
    return template.format(full_details=full_details, skills_text=skills_text)


class SkillRankerPromptBuilder:
    """LLM call 1: the job ad's full details plus the candidate skill menu."""

    def __init__(self, templates: PromptTemplates):
        self._template = templates.skill_ranker

    def build(self, full_details: str, potential_skills: list[SkillPromptEntry]) -> str:
        return build_skill_ranker_prompt(self._template, full_details, potential_skills)
