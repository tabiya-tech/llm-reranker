"""LLM 2 prompt: occupation block, job ad, attribute menu, ONET list and JSON contract. No IO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from app.config import Settings
from app.job_context import JobDescriptionContext
from app.job_ranking.job_attributes.types import (
    JobAttributesSchema,
    OnetWorkActivity,
)
from app.job_ranking.prompts.job_listing import (
    build_job_attributes_jd_text,
    build_job_attributes_metadata_block,
)
from app.job_ranking.prompts.templates import PromptTemplates
from app.languages import DEFAULT_LANGUAGE, language_setting
from utils.classifier_linked_entities import (
    group_linked_entities_by_type,
)
from utils.occupation_to_skills_mapping import (
    tabiya_occupation_id_for_candidate,
)
from utils.uuid_to_tabiya_id import cell_to_esco_uuid


def prompt_label(key: str, settings: Settings) -> str:
    """One scaffolding string of the LLM2 user block, in the run's language.

    Falls back to English for a key a language has not translated, so adding a key here
    never breaks a language pack that predates it.
    """
    labels = language_setting(settings.language, "prompt_labels", {}) or {}
    if key in labels:
        return str(labels[key])
    english = language_setting(DEFAULT_LANGUAGE, "prompt_labels", {}) or {}
    return str(english.get(key, key))


def llm2_opening_line_from_context(ctx: JobDescriptionContext, *, settings: Settings) -> str:
    """
    First line of the LLM2 user block: region field (county/district/etc.) when present, else city.

    Uses :attr:`Settings.country_name` from ``TARGET_COUNTRY``, phrased in the run's
    language (``prompt_labels`` in ``app/languages/{lang}_config.py``). If ``city`` equals
    the country name or slug, avoid duplicating the country in the phrasing.
    """
    country = settings.country_name
    slug = settings.target_country
    co = (ctx.county or "").strip()
    ci = (ctx.city or "").strip()
    cl = country.lower()
    city_is_country = ci.lower() in (slug, cl, country)

    if co:
        return prompt_label("opening_with_region", settings).format(region=co, country=country)
    if ci:
        if city_is_country:
            return prompt_label("opening_country_only", settings).format(country=country)
        return prompt_label("opening_with_region", settings).format(region=ci, country=country)
    return prompt_label("opening_no_region", settings).format(country=country)


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
    fb = (fallback_listing_title or "").strip() or "Unknown role"
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
    *,
    settings: Settings,
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
    attr_desc = build_attribute_descriptions(schema)
    parts = [
        opening_line,
        "",
        f"**{prompt_label('occupation', settings)}:** {preferred_label}",
    ]
    alt_label_text = "; ".join(alt_labels) if alt_labels else None
    if code:
        parts.append(f"**{prompt_label('code', settings)}:** {code}")
    if alt_label_text:
        parts.append(f"**{prompt_label('alt_names', settings)}:** {alt_label_text}")
    if description:
        parts.append(f"**{prompt_label('description', settings)}:** {description}")
    if included_occupations:
        parts.append(
            f"**{prompt_label('included_occupations', settings)}:** {included_occupations}"
        )
    if excluded_occupations:
        parts.append(
            f"**{prompt_label('excluded_occupations', settings)}:** {excluded_occupations}"
        )
    if included_tasks:
        parts.append(f"**{prompt_label('included_tasks', settings)}:** {included_tasks}")
    if notes:
        parts.append(f"**{prompt_label('notes', settings)}:** {notes}")
    if context_str:
        parts.append(context_str)
    if job_listing_body is not None:
        parts.extend(
            [
                "",
                f"**{prompt_label('job_listing_block', settings)}**",
                job_listing_body,
            ]
        )
    if job_listing_metadata is not None:
        parts.extend(
            [
                "",
                f"**{prompt_label('listing_metadata_block', settings)}**",
                job_listing_metadata,
            ]
        )
    parts.extend(
        [
            "",
            prompt_label("attributes_heading", settings),
            attr_desc,
            "",
            prompt_label("attributes_instruction", settings),
        ]
    )
    return "\n".join(parts)


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
            f"{i}. `{t.wa_element_id}` — {t.wa_element_name_simplified} ({t.wa_element_name}){sw}"
        )
    return "\n".join(lines)


def _kenya_requires_post_secondary_prompt_block() -> str:
    """LLM2 JSON contract addendum: Kenya education eligibility (stored on ranked job documents)."""
    return """
- `requires_post_secondary` (boolean): whether this **specific job advertisement** requires
  education **beyond Kenya secondary school** (KCSE / Form 4 / O-Level completion).

  Set **`true`** when the listing explicitly requires any of:
  - Certificate, diploma, higher diploma, or degree from a college, university, or TVET institution
  - Named professional qualifications (e.g. CPA, ACCA, nursing registration, teaching certificate
    from a post-secondary institution)
  - "Graduate", "bachelor's", "master's", or equivalent post-secondary credentials

  Set **`false`** when:
  - Only KCSE, O-Level, Form 4, or "secondary education" is required or mentioned
  - Only a KCSE grade is specified (e.g. "KCSE C+", "D plain and above") with no college/university qualification
  - No education requirement is stated, or requirements are experience/skills only
  - Requirements are unclear or ambiguous — prefer **`false`** rather than guessing

  Do **not** treat generic phrases like "minimum qualification" or "relevant qualification" as post-secondary
  unless the ad clearly points to certificate/diploma/degree-level study after secondary school."""


def _job_attributes_onet_and_json_appendix(
    onet_work_activities: Sequence[OnetWorkActivity],
    *,
    settings: Settings,
) -> str:
    """Append ONET work-activity menu and required JSON response shape."""
    task_menu = _format_onet_work_activity_menu(onet_work_activities)
    kenya_postsec = (
        _kenya_requires_post_secondary_prompt_block() if settings.target_country == "kenya" else ""
    )
    kenya_json_field = (
        ',\n  "requires_post_secondary": true' if settings.target_country == "kenya" else ""
    )
    # Must be the same heading text the user block emitted, or the model is pointed at a
    # section that does not exist in its prompt.
    attributes_heading = prompt_label("attributes_heading", settings)
    return f"""

---

## Work activities (ONET Work Activity questionnaires)

Select **3 to 10** distinct `WA_Element_ID` values that best describe day-to-day tasks for this role.
Use **only** ids from this list (no invented ids):

{task_menu}

## Required response format

Respond with **only** a single JSON object (no markdown fences, no text before or after).

- `attributes`: **array**; include **one object per** attribute from "{attributes_heading}", each with:
  - `attribute_name` (string, must match the attribute's `name`),
  - `selected_level_id` (string, must be a listed level `id` for that attribute),
  - `confidence`: `high`, `medium`, or `low`,
  - `reasoning` (short string).
- `summary`: brief synthesis of the overall attribute profile (string).
- `work_activities`: array of `WA_Element_ID` strings from the ONET list above, no duplicates, ordered most central to the role first.{kenya_postsec}

**JSON shape:**
{{
  "attributes": [
    {{
      "attribute_name": "<name>",
      "selected_level_id": "<level_id>",
      "confidence": "high",
      "reasoning": "..."
    }}
  ],
  "summary": "...",
  "work_activities": ["<WA_Element_ID>", "..."]{kenya_json_field}
}}"""


def build_job_attributes_prompt(
    *,
    system: str,
    opening_line: str,
    occupation: OccupationPromptEnrichment,
    schema: JobAttributesSchema,
    onet_work_activities: Sequence[OnetWorkActivity],
    extras: JobAttributesLlm2Extras | None = None,
    job_listing_body: str | None = None,
    job_listing_metadata: str | None = None,
    settings: Settings,
) -> str:
    """
    LLM call 2: system text from ``resources/prompts.py`` (:data:`resources.prompts.JOB_ATTRIBUTES_SYSTEM_PROMPT`),
    then this module's single-county-style user block, optional job-listing context, attribute menu,
    then ONET + JSON contract.
    """
    fields = resolve_llm2_single_county_user_fields(occupation, extras)
    user = build_single_county_job_attributes_user_prompt(
        opening_line,
        preferred_label=occupation.occupation_label,
        schema=schema,
        job_listing_body=job_listing_body,
        job_listing_metadata=job_listing_metadata,
        settings=settings,
        **fields,
    )
    appendix = _job_attributes_onet_and_json_appendix(onet_work_activities, settings=settings)
    return f"{system}\n\n---\n\n{user}{appendix}"


class JobAttributesPromptBuilder:
    """LLM call 2: occupation block, job ad, attribute menu, ONET list and JSON contract."""

    def __init__(
        self,
        templates: PromptTemplates,
        schema: JobAttributesSchema,
        onet: tuple[OnetWorkActivity, ...],
        settings: Settings,
    ):
        self._system = templates.job_attributes_system
        self.schema = schema
        self.onet = onet
        self._settings = settings

    def build(
        self,
        ctx: JobDescriptionContext,
        occupation: OccupationPromptEnrichment,
        extras: JobAttributesLlm2Extras | None = None,
    ) -> str:
        return build_job_attributes_prompt(
            system=self._system,
            opening_line=llm2_opening_line_from_context(ctx, settings=self._settings),
            occupation=occupation,
            schema=self.schema,
            onet_work_activities=self.onet,
            extras=extras,
            job_listing_body=build_job_attributes_jd_text(ctx),
            job_listing_metadata=build_job_attributes_metadata_block(ctx),
            settings=self._settings,
        )
