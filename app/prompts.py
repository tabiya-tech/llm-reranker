from __future__ import annotations

from typing import Any, Sequence

from app.config import Settings, get_settings
from app.job_context import JobDescriptionContext
from app.prompt_resources import (
    load_job_attributes_system_prompt,
    load_skills_ranker_prompt_template,
)
from app.schemas import JobAttributesSchema, OnetWorkActivity, load_job_attributes_schema


def _format_labeled_list(potential_skills: list[dict[str, Any]]) -> str:
    """Format skill menu for the ranker: label, optional description; model returns labels verbatim."""
    lines: list[str] = []
    for i, s in enumerate(potential_skills, 1):
        label = s.get("label") or ""
        desc = (s.get("description") or "").strip()
        lines.append(f"{i}. {label}")
        if desc:
            lines.append(f"   Description: {desc}")
    return "\n".join(lines)


def format_skill_ranker_skills_text(potential_skills: list[dict[str, Any]]) -> str:
    """The ``skills_text`` argument passed to ``str.format`` for ``Skills_reranker_prompts.md``."""
    return _format_labeled_list(potential_skills)


def build_skill_ranker_prompt(
    full_details: str,
    potential_skills: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
) -> str:
    """
    Skill ranker user message from ``resources/Skills_reranker_prompts.md``.

    ``potential_skills``: dicts with ``label`` and optional ``description`` (no URIs or scores).
    """
    skills_text = _format_labeled_list(potential_skills)
    template = load_skills_ranker_prompt_template(settings or get_settings())
    return template.format(full_details=full_details, skills_text=skills_text)


def build_attribute_descriptions(schema: JobAttributesSchema) -> str:
    """Markdown-style block listing each attribute and its allowed level ids."""
    lines: list[str] = []
    for attr in schema.attributes:
        lines.append(f"\n### {attr.label} ({attr.name})")
        lines.append(f"Group: {attr.group} | Type: {attr.type}")
        lines.append("Levels:")
        for level in attr.levels:
            if level.value is not None:
                lines.append(f"  - `{level.id}`: {level.label} (value: {level.value})")
            else:
                lines.append(f"  - `{level.id}`: {level.label}")
    return "\n".join(lines)


def _format_onet_work_activity_menu(onet: Sequence[OnetWorkActivity]) -> str:
    lines: list[str] = []
    for i, t in enumerate(onet, 1):
        sw = f" | Swahili: {t.wa_element_name_swahili}" if t.wa_element_name_swahili else ""
        lines.append(
            f"{i}. `{t.wa_element_id}` — {t.wa_element_name_simplified} "
            f"({t.wa_element_name}){sw}"
        )
    return "\n".join(lines)


def build_job_attributes_jd_text(ctx: JobDescriptionContext) -> str:
    """Raw job description + requirements for the attribute classifier."""
    parts: list[str] = []
    if ctx.raw_job_description:
        parts.append(ctx.raw_job_description.strip())
    if ctx.requirements_text:
        parts.append("Requirements:\n" + ctx.requirements_text.strip())
    return "\n\n".join(parts) if parts else "(no job description text provided)"


def build_job_attributes_user_message_block(
    ctx: JobDescriptionContext,
    *,
    settings: Settings | None = None,
    schema: JobAttributesSchema | None = None,
) -> str:
    """
    Exact user message for job-attributes LLM call 2 (same block as in the full prompt, excluding system + ONET appendix).

    Uses description/requirements from ``build_job_attributes_jd_text`` and listing fields from
    ``build_job_attributes_metadata_block``.
    """
    s = settings or get_settings()
    schema = schema or load_job_attributes_schema(settings=s)
    title = (ctx.title or "").strip() or "Unknown role"
    county = (ctx.county or "").strip() or "Unknown"
    jd = build_job_attributes_jd_text(ctx)
    meta = build_job_attributes_metadata_block(ctx)
    return _build_job_attributes_user_prompt(
        preferred_label=title,
        county=county,
        job_description_text=jd,
        metadata_block=meta,
        schema=schema,
    )


def build_job_attributes_metadata_block(ctx: JobDescriptionContext) -> str:
    """Structured listing fields (salary, location, etc.) for LLM call 2."""
    rows: list[str] = []
    for label, val in (
        ("Title", ctx.title),
        ("Employer", ctx.employer),
        ("Category", ctx.category),
        ("Salary", ctx.salary),
        ("Location / city", ctx.city),
        ("County / region", ctx.county),
        ("Employment type", ctx.employment_type),
        ("Posted", ctx.posted_date),
        ("Closing", ctx.closing_date),
        ("Source platform", ctx.source_platform),
        ("Application URL", ctx.application_url),
    ):
        if val is not None and str(val).strip():
            rows.append(f"- {label}: {val}")
    return "\n".join(rows) if rows else "(no structured metadata fields available)"


def _build_job_attributes_user_prompt(
    *,
    preferred_label: str,
    county: str,
    job_description_text: str,
    metadata_block: str,
    schema: JobAttributesSchema,
) -> str:
    """User message: occupation, description, listing metadata, and attribute menu."""
    attr_desc = build_attribute_descriptions(schema)
    parts = [
        f"Estimate job attributes for the following occupation in {county} County, Kenya:",
        "",
        f"**Occupation:** {preferred_label}",
        f"**Description:** {job_description_text}",
        "",
        "**Additional listing metadata (may be incomplete):**",
        metadata_block,
        "",
        "## Attributes to Estimate",
        attr_desc,
        "",
        "For each attribute, select the most appropriate level_id and provide brief reasoning.",
    ]
    return "\n".join(parts)


def _job_attributes_onet_and_json_appendix(onet_work_activities: Sequence[OnetWorkActivity]) -> str:
    """Append ONET work-activity menu and required JSON response shape."""
    task_menu = _format_onet_work_activity_menu(onet_work_activities)
    return f"""

---

## Work activities (ONET Work Activity questionnaires)

Select **3 to 10** distinct `WA_Element_ID` values that best describe day-to-day tasks for this role.
Use **only** ids from this list (no invented ids):

{task_menu}

## Required response format

Respond with **only** a single JSON object (no markdown fences, no text before or after).

- `attributes`: object whose keys are **exactly** the attribute `name` values from "## Attributes to Estimate" above, and values are **exactly** one allowed `level.id` string for that attribute.
- `work_activities`: array of `WA_Element_ID` strings from the ONET list above, no duplicates, ordered most central to the role first.

**JSON shape:**
{{
  "attributes": {{
    "<attribute_name>": "<level_id>"
  }},
  "work_activities": ["<WA_Element_ID>", "..."]
}}"""


def build_job_attributes_prompt(
    *,
    occupation_title: str,
    county: str,
    job_description_text: str,
    metadata_block: str,
    schema: JobAttributesSchema,
    onet_work_activities: Sequence[OnetWorkActivity],
    settings: Settings | None = None,
) -> str:
    """
    Job attributes (LLM call 2): system prompt, user block, ONET menu, and JSON contract.
    """
    s = settings or get_settings()
    system = load_job_attributes_system_prompt(s)
    user = _build_job_attributes_user_prompt(
        preferred_label=occupation_title,
        county=county,
        job_description_text=job_description_text,
        metadata_block=metadata_block,
        schema=schema,
    )
    appendix = _job_attributes_onet_and_json_appendix(onet_work_activities)
    return f"{system}\n\n---\n\n{user}{appendix}"
