"""``POST /rank/skills`` prompt: the jobseeker's experience plus the candidate skill menu. No IO."""

from __future__ import annotations

from app.config import Settings
from app.job_ranking.prompts.render import render_prompt
from app.job_ranking.prompts.templates import PromptTemplates
from app.job_ranking.routes.types import ExperienceDetails, SkillCandidate


def format_experience_details(experience: ExperienceDetails) -> str:
    """The ``{experience_details}`` slot: title, company, work type and responsibilities."""
    lines = [f"Title: {experience.title.strip()}"]
    if experience.company and experience.company.strip():
        lines.append(f"Company: {experience.company.strip()}")
    if experience.work_type and experience.work_type.strip():
        lines.append(f"Work type: {experience.work_type.strip()}")
    responsibilities = [r.strip() for r in experience.responsibilities if r.strip()]
    if responsibilities:
        lines.append("Responsibilities:")
        lines.extend(f"- {r}" for r in responsibilities)
    return "\n".join(lines)


def format_skills_text(menu: list[tuple[str, SkillCandidate]]) -> str:
    """The ``{skills_text}`` slot: one ``id: label`` line per candidate, then its description.

    ``menu`` pairs each candidate with the short prompt id the model answers with; the
    caller's own ids (often UUIDs) and scores are not shown.
    """
    lines: list[str] = []
    for prompt_id, candidate in menu:
        lines.append(f"{prompt_id}: {candidate.label.strip()}")
        desc = candidate.description.strip()
        if desc:
            lines.append(f"   Description: {desc}")
    return "\n".join(lines)


class ExperienceSkillRankerPromptBuilder:
    """The jobseeker's experience plus the candidate skill menu."""

    def __init__(self, templates: PromptTemplates, settings: Settings):
        self._templates = templates
        self._settings = settings

    def build(
        self,
        experience: ExperienceDetails,
        menu: list[tuple[str, SkillCandidate]],
        max_skills: int,
    ) -> str:
        return render_prompt(
            self._templates.experience_skill_ranker,
            experience_details=format_experience_details(experience),
            skills_text=format_skills_text(menu),
            max_skills=str(max_skills),
            language=self._settings.language_name,
        )
