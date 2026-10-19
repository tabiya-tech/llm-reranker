from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from app.config import Settings, get_settings
from app.job_context import JobDescriptionContext
from app.languages import DEFAULT_LANGUAGE, language_setting
from app.prompt_resources import (
    JOB_ATTRIBUTES_RESPONSE,
    JOB_ATTRIBUTES_SYSTEM,
    OCCUPATION_SKILLS_RELEVANCE_SYSTEM,
    OCCUPATION_SKILLS_RELEVANCE_USER,
    SKILL_RANKER,
    country_field_template,
    render_prompt,
)
from app.schemas import JobAttributesSchema, OnetWorkActivity
from utils.classifier_linked_entities import group_linked_entities_by_type
from utils.occupation_to_skills_mapping import tabiya_occupation_id_for_candidate
from utils.uuid_to_tabiya_id import cell_to_esco_uuid


def prompt_label(key: str, settings: Settings | None = None) -> str:
    """One scaffolding string assembled around the prompt templates, in the run's language.

    Falls back to English for a key a language has not translated, so adding a key here
    never breaks a language pack that predates it.
    """
    s = settings or get_settings()
    labels = language_setting(s.language, "prompt_labels", {}) or {}
    if key in labels:
        return str(labels[key])
    english = language_setting(DEFAULT_LANGUAGE, "prompt_labels", {}) or {}
    return str(english.get(key, key))


def llm2_opening_line_from_context(
    ctx: JobDescriptionContext, *, settings: Settings | None = None
) -> str:
    """
    First line of the LLM2 user block: region field (county/district/etc.) when present, else city.

    Uses :attr:`Settings.country_name` from ``TARGET_COUNTRY``, phrased in the run's
    language (``prompt_labels`` in ``app/languages/{lang}_config.py``). If ``city`` equals
    the country name or slug, avoid duplicating the country in the phrasing.
    """
    s = settings or get_settings()
    country = s.country_name
    slug = s.target_country
    co = (ctx.county or "").strip()
    ci = (ctx.city or "").strip()
    cl = country.lower()
    city_is_country = ci.lower() in (slug, cl, country)

    if co:
        return prompt_label("opening_with_region", s).format(region=co, country=country)
    if ci:
        if city_is_country:
            return prompt_label("opening_country_only", s).format(country=country)
        return prompt_label("opening_with_region", s).format(region=ci, country=country)
    return prompt_label("opening_no_region", s).format(country=country)


@dataclass(frozen=True)
class OccupationPromptEnrichment:
    """Primary occupation slice for the job-attributes (LLM2) prompt; CSV-backed when resolvable."""

    occupation_label: str
    csv_definition: str | None
    tabiya_occupation_id: str | None
    esco_uuid: str | None
    classifier_label: str | None


def build_occupation_prompt_enrichment(
    classified_doc: dict[str, Any],
    settings: Settings,
    *,
    fallback_listing_title: str,
    occupations_bridge: dict[str, str] | None,
    occupations_all_ids: frozenset[str] | None,
    label_by_esco: dict[str, str],
    description_by_esco: dict[str, str],
) -> OccupationPromptEnrichment:
    """
    LLM call 2: first deduped classifier occupation → ``occupations.csv`` label/description by ESCO keys.

    When there is no occupation entity, ``occupation_label`` is ``fallback_listing_title``.
    """
    occs = group_linked_entities_by_type(classified_doc, settings).get("occupation", [])
    fb = (fallback_listing_title or "").strip() or prompt_label("unknown_occupation", settings)
    if not occs:
        return OccupationPromptEnrichment(
            occupation_label=fb,
            csv_definition=None,
            tabiya_occupation_id=None,
            esco_uuid=None,
            classifier_label=None,
        )
    c = occs[0]
    clabel = (c.label or "").strip() or None
    esco_raw = ((c.concept_uuid or "").strip() or cell_to_esco_uuid(c.uri) or "").strip()
    esco_l = esco_raw.lower() if esco_raw else None
    tid: str | None = None
    if occupations_bridge is not None and occupations_all_ids is not None:
        tid = tabiya_occupation_id_for_candidate(
            c, occupations_bridge, occupations_all_ids, settings=settings
        )

    csv_label = label_by_esco.get(esco_l, "") if esco_l else ""
    if not csv_label.strip():
        csv_label = clabel or fb

    csv_def: str | None = None
    if description_by_esco and esco_l:
        d = (description_by_esco.get(esco_l) or "").strip()
        if d:
            csv_def = d

    return OccupationPromptEnrichment(
        occupation_label=csv_label.strip(),
        csv_definition=csv_def,
        tabiya_occupation_id=tid,
        esco_uuid=esco_l,
        classifier_label=clabel,
    )


@dataclass(frozen=True)
class JobAttributesLlm2Extras:
    """
    Optional user-message fields for :func:`build_single_county_job_attributes_user_prompt`.

    ``None`` means: omit override (use :class:`OccupationPromptEnrichment` defaults where applicable).
    Wire these from your pipeline when you know sources (e.g. earnings ``context_str``).

    ``included_tasks`` is classifier / ESCO-style task text, not ONET ``WA_Element_ID`` values
    (those are chosen from the ONET appendix).
    """

    alt_labels: tuple[str, ...] | None = None
    description: str | None = None
    code: str | None = None
    context_str: str | None = None
    included_occupations: str | None = None
    included_tasks: str | None = None
    excluded_occupations: str | None = None
    notes: str | None = None


def _extra_field(extras: JobAttributesLlm2Extras | None, name: str) -> str | None:
    if extras is None:
        return None
    v = getattr(extras, name)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def resolve_llm2_single_county_user_fields(
    occupation: OccupationPromptEnrichment,
    extras: JobAttributesLlm2Extras | None,
) -> dict[str, Any]:
    """
    Resolved kwargs for :func:`build_single_county_job_attributes_user_prompt`
    (after applying ``extras`` overrides and CSV/classifier defaults).
    """
    x = extras
    alts: list[str] | None = None
    classifier_label = (occupation.classifier_label or "").strip()
    if x is not None and x.alt_labels is not None:
        alts = [str(a).strip() for a in x.alt_labels if str(a).strip()][:3] or None
    elif classifier_label and classifier_label != (occupation.occupation_label or "").strip():
        alts = [classifier_label]

    description: str | None = occupation.csv_definition
    if x is not None and x.description is not None:
        description = str(x.description).strip() or None

    code: str | None = (occupation.tabiya_occupation_id or "").strip() or None
    if x is not None and x.code is not None:
        code = str(x.code).strip() or None

    return {
        "alt_labels": alts,
        "description": description,
        "code": code,
        "context_str": _extra_field(x, "context_str"),
        "included_occupations": _extra_field(x, "included_occupations"),
        "included_tasks": _extra_field(x, "included_tasks"),
        "excluded_occupations": _extra_field(x, "excluded_occupations"),
        "notes": _extra_field(x, "notes"),
    }


def build_single_county_job_attributes_user_prompt(
    opening_line: str,
    preferred_label: str,
    schema: JobAttributesSchema,
    alt_labels: list[str] | None = None,
    description: str | None = None,
    code: str | None = None,
    context_str: str | None = None,
    included_occupations: str | None = None,
    included_tasks: str | None = None,
    excluded_occupations: str | None = None,
    notes: str | None = None,
    job_listing_body: str | None = None,
    job_listing_metadata: str | None = None,
    settings: Settings | None = None,
) -> str:
    """
    User message for LLM2: locale framing (from env), occupation-centric block, then attribute menu.

    Field headings come from the run's language (``prompt_labels`` in
    ``app/languages/{lang}_config.py``); the attribute names and level ids inside
    ``attr_desc`` are identifiers and stay as the schema defines them.

    ``included_tasks`` is **classification task text**, not ONET ``WA_Element_ID`` values (those appear in the appendix).

    When ``job_listing_body`` / ``job_listing_metadata`` are set (non-``None``), they are inserted after the occupation
    block and before the attributes heading (job-ad context for the classifier).
    """
    s = settings or get_settings()
    attr_desc = build_attribute_descriptions(schema, settings=s)
    parts = [
        opening_line,
        "",
        f"**{prompt_label('occupation', s)}:** {preferred_label}",
    ]
    alt_label_text = "; ".join(alt_labels) if alt_labels else None
    if code:
        parts.append(f"**{prompt_label('code', s)}:** {code}")
    if alt_label_text:
        parts.append(f"**{prompt_label('alt_names', s)}:** {alt_label_text}")
    if description:
        parts.append(f"**{prompt_label('description', s)}:** {description}")
    if included_occupations:
        parts.append(f"**{prompt_label('included_occupations', s)}:** {included_occupations}")
    if excluded_occupations:
        parts.append(f"**{prompt_label('excluded_occupations', s)}:** {excluded_occupations}")
    if included_tasks:
        parts.append(f"**{prompt_label('included_tasks', s)}:** {included_tasks}")
    if notes:
        parts.append(f"**{prompt_label('notes', s)}:** {notes}")
    if context_str:
        parts.append(context_str)
    if job_listing_body is not None:
        parts.extend(
            [
                "",
                f"**{prompt_label('job_listing_block', s)}**",
                job_listing_body,
            ]
        )
    if job_listing_metadata is not None:
        parts.extend(
            [
                "",
                f"**{prompt_label('listing_metadata_block', s)}**",
                job_listing_metadata,
            ]
        )
    parts.extend(
        [
            "",
            prompt_label("attributes_heading", s),
            attr_desc,
            "",
            prompt_label("attributes_instruction", s),
        ]
    )
    return "\n".join(parts)


def _format_labeled_list(
    potential_skills: list[dict[str, Any]], *, settings: Settings | None = None
) -> str:
    """Format skill menu for the ranker: label, optional description; model returns labels verbatim."""
    description = prompt_label("skill_description", settings)
    lines: list[str] = []
    for i, s in enumerate(potential_skills, 1):
        label = s.get("label") or ""
        desc = (s.get("description") or "").strip()
        lines.append(f"{i}. {label}")
        if desc:
            lines.append(f"   {description}: {desc}")
    return "\n".join(lines)


def format_skill_ranker_skills_text(
    potential_skills: list[dict[str, Any]], *, settings: Settings | None = None
) -> str:
    """The ``{skills_text}`` placeholder of ``prompts/skill_ranker.md``."""
    return _format_labeled_list(potential_skills, settings=settings)


def build_skill_ranker_prompt(
    full_details: str,
    potential_skills: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
) -> str:
    """
    Skill ranker user message from ``resources/<lang>/prompts/skill_ranker.md``.

    ``potential_skills``: dicts with ``label`` and optional ``description`` (no URIs or scores).
    """
    s = settings or get_settings()
    skills_text = _format_labeled_list(potential_skills, settings=s)
    return render_prompt(SKILL_RANKER, s, full_details=full_details, skills_text=skills_text)


def build_attribute_descriptions(
    schema: JobAttributesSchema, *, settings: Settings | None = None
) -> str:
    """Markdown-style block listing each attribute and its allowed level ids."""
    group = prompt_label("attribute_group", settings)
    type_ = prompt_label("attribute_type", settings)
    levels = prompt_label("attribute_levels", settings)
    value = prompt_label("attribute_level_value", settings)
    lines: list[str] = []
    for attr in schema.attributes:
        lines.append(f"\n### {attr.label} ({attr.name})")
        lines.append(f"{group}: {attr.group} | {type_}: {attr.type}")
        lines.append(f"{levels}:")
        for level in attr.levels:
            if level.value is not None:
                lines.append(f"  - `{level.id}`: {level.label} ({value}: {level.value})")
            else:
                lines.append(f"  - `{level.id}`: {level.label}")
    return "\n".join(lines)


def _format_onet_work_activity_menu(
    onet: Sequence[OnetWorkActivity], *, settings: Settings | None = None
) -> str:
    swahili = prompt_label("work_activity_swahili", settings)
    lines: list[str] = []
    for i, t in enumerate(onet, 1):
        sw = f" | {swahili}: {t.wa_element_name_swahili}" if t.wa_element_name_swahili else ""
        lines.append(
            f"{i}. `{t.wa_element_id}` — {t.wa_element_name_simplified} ({t.wa_element_name}){sw}"
        )
    return "\n".join(lines)


def build_job_attributes_jd_text(
    ctx: JobDescriptionContext, *, settings: Settings | None = None
) -> str:
    """Raw job description + requirements for the attribute classifier."""
    parts: list[str] = []
    if ctx.raw_job_description:
        parts.append(ctx.raw_job_description.strip())
    if ctx.requirements_text:
        parts.append(
            prompt_label("listing_requirements", settings) + "\n" + ctx.requirements_text.strip()
        )
    return "\n\n".join(parts) if parts else prompt_label("listing_no_description", settings)


def build_job_attributes_metadata_block(
    ctx: JobDescriptionContext, *, settings: Settings | None = None
) -> str:
    """Structured listing fields (salary, location, etc.) for LLM call 2."""
    rows: list[str] = []
    for key, val in (
        ("listing_title", ctx.title),
        ("listing_employer", ctx.employer),
        ("listing_category", ctx.category),
        ("listing_salary", ctx.salary),
        ("listing_city", ctx.city),
        ("listing_region", ctx.county),
        ("listing_employment_type", ctx.employment_type),
        ("listing_posted", ctx.posted_date),
        ("listing_closing", ctx.closing_date),
        ("listing_source_platform", ctx.source_platform),
        ("listing_application_url", ctx.application_url),
    ):
        if val is not None and str(val).strip():
            rows.append(f"- {prompt_label(key, settings)}: {val}")
    return "\n".join(rows) if rows else prompt_label("listing_no_metadata", settings)


def build_occupation_skills_relevance_prompt(
    ctx: JobDescriptionContext,
    occupation_to_skills_items: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
) -> str:
    """
    LLM call 3: system + user message listing job ad context and occupation taxonomy skills
    (essential / optional only). Model returns JSON ``decisions`` with ``is_relevant`` per skill.
    """
    s = settings or get_settings()
    occupation = prompt_label("relevance_occupation", s)
    unnamed = prompt_label("relevance_unnamed_occupation", s)
    menu_lines: list[str] = []
    for block in occupation_to_skills_items:
        tid = block.get("tabiya_occupation_id")
        o_lab = (block.get("occupation_label") or "").strip() or unnamed
        menu_lines.append(f"### {occupation}: {o_lab}")
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

    menu = "\n".join(menu_lines).strip() or prompt_label("relevance_no_skills", s)

    system = render_prompt(OCCUPATION_SKILLS_RELEVANCE_SYSTEM, s)
    user = render_prompt(
        OCCUPATION_SKILLS_RELEVANCE_USER,
        s,
        job_description=build_job_attributes_jd_text(ctx, settings=s),
        listing_metadata=build_job_attributes_metadata_block(ctx, settings=s),
        skills_menu=menu,
    )
    return f"{system}\n\n---\n\n{user}"


def _job_attributes_response_contract(
    onet_work_activities: Sequence[OnetWorkActivity],
    *,
    settings: Settings | None = None,
) -> str:
    """ONET work-activity menu and required JSON response shape, plus the country's fields.

    Each of :attr:`Settings.job_attributes_country_fields` adds its instructions from
    ``prompts/countries/<country>/<field>.md`` and a boolean key to the JSON shape.
    """
    s = settings or get_settings()
    fields = s.job_attributes_country_fields
    instructions = "".join(
        "\n" + render_prompt(country_field_template(s.target_country, f), s) for f in fields
    )
    json_keys = "".join(f',\n  "{f}": true' for f in fields)
    return render_prompt(
        JOB_ATTRIBUTES_RESPONSE,
        s,
        work_activities_menu=_format_onet_work_activity_menu(onet_work_activities, settings=s),
        # Must be the same heading text the user block emitted, or the model is pointed at
        # a section that does not exist in its prompt.
        attributes_heading=prompt_label("attributes_heading", s),
        country_field_instructions=instructions,
        country_field_json=json_keys,
    )


def build_job_attributes_prompt(
    *,
    opening_line: str,
    occupation: OccupationPromptEnrichment,
    schema: JobAttributesSchema,
    onet_work_activities: Sequence[OnetWorkActivity],
    extras: JobAttributesLlm2Extras | None = None,
    job_listing_body: str | None = None,
    job_listing_metadata: str | None = None,
    settings: Settings | None = None,
) -> str:
    """
    LLM call 2: ``prompts/job_attributes_system.md``, then this module's single-county-style
    user block, optional job-listing context, attribute menu, then
    ``prompts/job_attributes_response.md`` (ONET menu + JSON contract).
    """
    s = settings or get_settings()
    system = render_prompt(JOB_ATTRIBUTES_SYSTEM, s)
    fields = resolve_llm2_single_county_user_fields(occupation, extras)
    user = build_single_county_job_attributes_user_prompt(
        opening_line,
        preferred_label=occupation.occupation_label,
        schema=schema,
        job_listing_body=job_listing_body,
        job_listing_metadata=job_listing_metadata,
        settings=s,
        **fields,
    )
    response = _job_attributes_response_contract(onet_work_activities, settings=s)
    return f"{system}\n\n---\n\n{user}\n\n---\n\n{response}"
