"""LLM 3 prompt: the job ad plus the occupation-linked skills to judge. No IO."""

from __future__ import annotations

from typing import Any

from app.job_context import JobDescriptionContext
from app.job_ranking.prompts.job_listing import (
    build_job_attributes_jd_text,
    build_job_attributes_metadata_block,
)
from app.job_ranking.prompts.templates import PromptTemplates


def build_occupation_skills_relevance_prompt(
    system: str,
    ctx: JobDescriptionContext,
    occupation_to_skills_items: list[dict[str, Any]],
) -> str:
    """
    LLM call 3: ``system`` + user message listing job ad context and occupation taxonomy skills
    (essential / optional only). Model returns JSON ``decisions`` with ``is_relevant`` per skill.
    """
    jd = build_job_attributes_jd_text(ctx)
    meta = build_job_attributes_metadata_block(ctx)

    menu_lines: list[str] = []
    for block in occupation_to_skills_items:
        tid = block.get("tabiya_occupation_id")
        o_lab = (block.get("occupation_label") or "").strip() or "(occupation)"
        menu_lines.append(f"### Occupation: {o_lab}")
        menu_lines.append(f"- tabiya_occupation_id: `{tid}`")
        for sk in block.get("skills") or []:
            if not isinstance(sk, dict):
                continue
            rt = (sk.get("relation_type") or "").strip().lower()
            if rt not in ("essential", "optional"):
                continue
            sid = str(sk.get("skill_id") or "").strip()
            if not sid or not tid:
                continue
            lab = (sk.get("label") or "").strip() or sid
            menu_lines.append(f"- skill_id: `{sid}` | label: {lab} | taxonomy_role: {rt}")
        menu_lines.append("")

    menu = "\n".join(menu_lines).strip() or "(no essential or optional occupation skills to judge)"

    user = f"""## Job listing (description and requirements)

{jd}

## Listing metadata

{meta}

## Occupation-linked skills to judge

Each line below is one taxonomy skill tied to an occupation classified on this posting.
Decide whether it is **relevant to this specific job advertisement** (not only whether it belongs to the occupation in general).

{menu}

## Required response format

Respond with **only** a single JSON object (no markdown fences, no other text).

- `decisions`: array with **exactly one object per skill** listed under "Occupation-linked skills" above that has `taxonomy_role` **essential** or **optional**. Each object must have:
  - `skill_id` (string, must match exactly)
  - `tabiya_occupation_id` (string, must match exactly)
  - `is_relevant` (boolean)

Do not include `summary`, `reason`, or any other top-level keys.

**JSON shape:**
{{
  "decisions": [
    {{"skill_id": "<uuid>", "tabiya_occupation_id": "<id>", "is_relevant": true}}
  ]
}}"""
    return f"{system}\n\n---\n\n{user}"


class OccupationSkillsRelevancePromptBuilder:
    """LLM call 3: the job ad plus the occupation-linked skills to judge."""

    def __init__(self, templates: PromptTemplates):
        self._system = templates.occupation_skills_relevance_system

    def build(
        self, ctx: JobDescriptionContext, occupation_to_skills_items: list[dict[str, Any]]
    ) -> str:
        return build_occupation_skills_relevance_prompt(
            self._system, ctx, occupation_to_skills_items
        )
