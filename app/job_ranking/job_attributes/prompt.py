"""LLM 2 prompt: occupation block, job ad, attribute menu, ONET list and JSON contract. No IO."""

from __future__ import annotations

from typing import Any, Sequence

from pydantic import Field

from app.config import JobAttributeExtraField, Settings
from app.domain_base import DomainModel
from app.job_context import JobDescriptionContext
from app.job_ranking.job_attributes.types import (
    JobAttributesSchema,
    OnetWorkActivity,
)
from app.job_ranking.prompts.job_listing import (
    build_job_attributes_jd_text,
    build_job_attributes_metadata_block,
)
from app.job_ranking.prompts.render import render_prompt
from app.job_ranking.prompts.templates import PromptTemplates
from app.job_ranking.routes.types import RankRequest
from utils.classifier_linked_entities import (
    group_linked_entities_by_type,
)
from utils.occupation_to_skills_mapping import (
    tabiya_occupation_id_for_candidate,
)
from utils.uuid_to_tabiya_id import cell_to_esco_uuid

# The LLM2 opening line. Prompt text stays English for every language: the template tells the
# model which language to answer in.
_OPENING_WITH_REGION = (
    "Estimate job attributes for the following occupation in {region}, {country}:"
)
_OPENING_COUNTRY_ONLY = "Estimate job attributes for the following occupation in {country}:"
_OPENING_NO_REGION = (
    "Estimate job attributes for the following occupation in {country} (region not specified):"
)


def llm2_opening_line_from_context(ctx: JobDescriptionContext, *, settings: Settings) -> str:
    """
    First line of the LLM2 user block: region field (county/district/etc.) when present, else city.

    Uses :attr:`Settings.country_name` from ``TARGET_COUNTRY``. If ``city`` equals the country
    name or slug, avoid duplicating the country in the phrasing.
    """
    country = settings.country_name
    slug = settings.target_country
    co = (ctx.county or "").strip()
    ci = (ctx.city or "").strip()
    cl = country.lower()
    city_is_country = ci.lower() in (slug, cl, country)

    if co:
        return _OPENING_WITH_REGION.format(region=co, country=country)
    if ci:
        if city_is_country:
            return _OPENING_COUNTRY_ONLY.format(country=country)
        return _OPENING_WITH_REGION.format(region=ci, country=country)
    return _OPENING_NO_REGION.format(country=country)


class OccupationPromptEnrichment(DomainModel):
    """Primary occupation slice for the job-attributes (LLM2) prompt; CSV-backed when resolvable."""

    occupation_label: str = Field(description="Occupation label shown in the LLM2 prompt.")
    csv_definition: str | None = Field(
        description="`occupations.csv` description for this ESCO uuid, when present."
    )
    tabiya_occupation_id: str | None = Field(
        description="Tabiya occupation id from `occupations.csv`, when resolved."
    )
    esco_uuid: str | None = Field(description="ESCO occupation uuid, lowercased.")
    classifier_label: str | None = Field(
        description="Classifier label for the first deduped occupation hit, when there is one."
    )


def build_occupation_prompt_enrichment(
    classified_doc: RankRequest | dict[str, Any],
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


class JobAttributesLlm2Extras(DomainModel):
    """
    Optional user-message fields for :func:`build_occupation_block`.

    ``None`` means: omit override (use :class:`OccupationPromptEnrichment` defaults where applicable).
    Wire these from your pipeline when you know sources (e.g. earnings ``context_str``).

    ``included_tasks`` is classifier / ESCO-style task text, not ONET ``WA_Element_ID`` values
    (those are chosen from the work-activity menu).
    """

    alt_labels: tuple[str, ...] | None = Field(
        default=None, description="Override for alternate occupation labels shown in the prompt."
    )
    description: str | None = Field(
        default=None, description="Override for the occupation description."
    )
    code: str | None = Field(default=None, description="Override for the occupation code shown.")
    context_str: str | None = Field(
        default=None, description="Extra free-text context appended to the prompt (e.g. earnings)."
    )
    included_occupations: str | None = Field(
        default=None, description="Override for the included-occupations prompt field."
    )
    included_tasks: str | None = Field(
        default=None,
        description="Override for included-tasks text (classifier/ESCO task text, not "
        "ONET work-activity ids).",
    )
    excluded_occupations: str | None = Field(
        default=None, description="Override for the excluded-occupations prompt field."
    )
    notes: str | None = Field(default=None, description="Override for the notes prompt field.")


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
    Resolved kwargs for :func:`build_occupation_block`
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


def build_occupation_block(
    preferred_label: str,
    alt_labels: list[str] | None = None,
    description: str | None = None,
    code: str | None = None,
    context_str: str | None = None,
    included_occupations: str | None = None,
    included_tasks: str | None = None,
    excluded_occupations: str | None = None,
    notes: str | None = None,
) -> str:
    """
    The ``{occupation_block}`` slot of ``resources/prompts/job_attributes.md``: the occupation
    and whichever of its optional fields are known, one line each.

    ``included_tasks`` is **classification task text**, not ONET ``WA_Element_ID`` values (those
    appear in the work-activity menu).
    """
    lines = [f"**Occupation:** {preferred_label}"]
    if code:
        lines.append(f"**KESCO Code:** {code}")
    if alt_labels:
        lines.append(f"**Alternative Names:** {'; '.join(alt_labels)}")
    if description:
        lines.append(f"**Description:** {description}")
    if included_occupations:
        lines.append(f"**Included Occupations:** {included_occupations}")
    if excluded_occupations:
        lines.append(f"**Excluded Occupations:** {excluded_occupations}")
    if included_tasks:
        lines.append(f"**Included Tasks:** {included_tasks}")
    if notes:
        lines.append(f"**Notes:** {notes}")
    if context_str:
        lines.append(context_str)
    return "\n".join(lines)


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


def _extra_field_instructions(fields: Sequence[JobAttributeExtraField]) -> str:
    """The ``{extra_field_instructions}`` slot: one response-field bullet per extra field."""
    return "".join(f"\n- `{f.name}` (boolean): {f.instructions}" for f in fields)


def _extra_field_json(fields: Sequence[JobAttributeExtraField]) -> str:
    """The ``{extra_field_json}`` slot: each extra field's key in the JSON example."""
    return "".join(f',\n  "{f.name}": true' for f in fields)


def build_job_attributes_prompt(
    *,
    templates: PromptTemplates,
    opening_line: str,
    occupation: OccupationPromptEnrichment,
    schema: JobAttributesSchema,
    onet_work_activities: Sequence[OnetWorkActivity],
    extras: JobAttributesLlm2Extras | None = None,
    job_listing_body: str,
    job_listing_metadata: str,
    settings: Settings,
) -> str:
    """
    LLM call 2 from ``resources/prompts/job_attributes.md``: the occupation, the job ad, the
    attribute menu, the ONET work-activity menu and the JSON contract, plus the deployment's
    extra fields (:attr:`Settings.job_attributes_extra_fields`; none gives the plain prompt).
    """
    fields = resolve_llm2_single_county_user_fields(occupation, extras)
    extra_fields = settings.job_attributes_extra_fields
    return render_prompt(
        templates.job_attributes,
        country=settings.country_name,
        opening_line=opening_line,
        occupation_block=build_occupation_block(occupation.occupation_label, **fields),
        job_listing=job_listing_body,
        listing_metadata=job_listing_metadata,
        attributes_menu=build_attribute_descriptions(schema),
        work_activities_menu=_format_onet_work_activity_menu(onet_work_activities),
        extra_field_instructions=_extra_field_instructions(extra_fields),
        extra_field_json=_extra_field_json(extra_fields),
        language=settings.language_name,
    )


class JobAttributesPromptBuilder:
    """LLM call 2: occupation block, job ad, attribute menu, ONET list and JSON contract."""

    def __init__(
        self,
        templates: PromptTemplates,
        schema: JobAttributesSchema,
        onet: tuple[OnetWorkActivity, ...],
        settings: Settings,
    ):
        self._templates = templates
        # Read now so a misconfigured RERANKER_LLM2_EXTRA_FIELDS fails startup, not a job.
        self.extra_fields = settings.job_attributes_extra_fields
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
            templates=self._templates,
            opening_line=llm2_opening_line_from_context(ctx, settings=self._settings),
            occupation=occupation,
            schema=self.schema,
            onet_work_activities=self.onet,
            extras=extras,
            job_listing_body=build_job_attributes_jd_text(ctx),
            job_listing_metadata=build_job_attributes_metadata_block(ctx),
            settings=self._settings,
        )
