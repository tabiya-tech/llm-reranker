"""LLM 3 prompt: the job ad plus the occupation-linked skills to judge. No IO."""

from __future__ import annotations

from app.config import Settings
from app.job_context import JobDescriptionContext
from app.job_ranking.prompts.job_listing import (
    build_job_attributes_jd_text,
    build_job_attributes_metadata_block,
)
from app.job_ranking.prompts.render import render_prompt
from app.job_ranking.prompts.templates import PromptTemplates
from app.job_ranking.routes.types import OccupationSkillsBlock


def build_occupation_skills_relevance_prompt(
    templates: PromptTemplates,
    settings: Settings,
    ctx: JobDescriptionContext,
    occupation_to_skills_items: list[OccupationSkillsBlock],
) -> str:
    """
    LLM call 3 from ``resources/prompts/occupation_skills_relevance.md``: the job ad context and
    the occupation taxonomy skills (essential / optional only). Model returns JSON ``decisions``
    with ``is_relevant`` per skill.
    """
    jd = build_job_attributes_jd_text(ctx)
    meta = build_job_attributes_metadata_block(ctx)

    menu_lines: list[str] = []
    for block in occupation_to_skills_items:
        tid = block.tabiya_occupation_id
        o_lab = (block.occupation_label or "").strip() or "(occupation)"
        menu_lines.append(f"### Occupation: {o_lab}")
        menu_lines.append(f"- tabiya_occupation_id: `{tid}`")
        for sk in block.skills:
            rt = sk.relation_type.strip().lower()
            if rt not in ("essential", "optional"):
                continue
            sid = sk.skill_id.strip()
            if not sid or not tid:
                continue
            lab = (sk.label or "").strip() or sid
            menu_lines.append(f"- skill_id: `{sid}` | label: {lab} | taxonomy_role: {rt}")
        menu_lines.append("")

    menu = "\n".join(menu_lines).strip() or "(no essential or optional occupation skills to judge)"

    return render_prompt(
        templates.occupation_skills_relevance,
        country=settings.country_name,
        job_listing=jd,
        listing_metadata=meta,
        skills_menu=menu,
        language=settings.language_name,
    )


class OccupationSkillsRelevancePromptBuilder:
    """LLM call 3: the job ad plus the occupation-linked skills to judge."""

    def __init__(self, templates: PromptTemplates, settings: Settings):
        self._templates = templates
        self._settings = settings

    def build(
        self, ctx: JobDescriptionContext, occupation_to_skills_items: list[OccupationSkillsBlock]
    ) -> str:
        return build_occupation_skills_relevance_prompt(
            self._templates, self._settings, ctx, occupation_to_skills_items
        )
