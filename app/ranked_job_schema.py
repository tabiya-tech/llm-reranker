"""Target collection document shape for LLM reranker + job attributes (``PIPELINE_SCHEMA_VERSION``)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.config import Settings, get_settings
from app.job_context import JobDescriptionContext
from app.schemas import JobAttributesResult, load_onet_work_activities
from utils.classifier_linked_entities import (
    LinkedEntityCandidate,
    MappedEscoSkills,
    group_linked_entities_by_type,
)

PIPELINE_SCHEMA_VERSION = "6"


def _classified_rows_and_uuids(
    candidates: list[LinkedEntityCandidate],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Serialize linked entities for Mongo (same shape for skills and occupations)."""
    rows: list[dict[str, Any]] = []
    uuids: list[str] = []
    for c in candidates:
        rows.append(
            {
                "label": c.label,
                "uri": c.uri,
                "esco_uuid": c.concept_uuid,
                "similarity_score": c.similarity_score,
            }
        )
        if c.concept_uuid:
            uuids.append(c.concept_uuid)
    return rows, uuids


def _classified_linked_by_type(
    by_type: dict[str, list[LinkedEntityCandidate]],
    settings: Settings,
) -> dict[str, dict[str, Any]]:
    """
    One entry per configured ``entity_type``: ``items`` (row dicts) and ``uuids`` (concept UUIDs).

    Covers every value in ``settings.classified_linked_entity_types`` (empty lists when none extracted).
    """
    out: dict[str, dict[str, Any]] = {}
    for et in settings.classified_linked_entity_types:
        rows, uuids = _classified_rows_and_uuids(by_type.get(et, []))
        out[et] = {"items": rows, "uuids": uuids}
    return out


def work_activity_names_for_ids(
    ids: tuple[str, ...],
    *,
    settings: Settings | None = None,
) -> list[str]:
    """Resolve ONET work activity ids to display names (simplified label when present)."""
    settings = settings or get_settings()
    onet = load_onet_work_activities(settings=settings)
    by_id = {x.wa_element_id: (x.wa_element_name_simplified or x.wa_element_name) for x in onet}
    return [by_id.get(wid, "") for wid in ids]


def classifier_metadata_from_context(ctx: JobDescriptionContext) -> dict[str, Any]:
    """Classifier fields for the target document, including full description and requirements text."""
    return {
        "employer": ctx.employer,
        "category": ctx.category,
        "application_url": ctx.application_url,
        "posted_date": ctx.posted_date,
        "closing_date": ctx.closing_date,
        "source_platform": ctx.source_platform,
        "city": ctx.city,
        "county": ctx.county,
        "salary": ctx.salary,
        "employment_type": ctx.employment_type,
        "job_description": ctx.raw_job_description,
        "requirements": ctx.requirements_text,
    }


def build_llm_reranker_meta(settings: Settings) -> dict[str, Any]:
    return {
        "skill_ranker_model": settings.vertex_model,
        "job_attributes_model": settings.vertex_model,
        "gcp_location": settings.gcp_location,
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "classified_linked_entity_types": list(settings.classified_linked_entity_types),
    }


def build_ranked_job_document(
    classified_doc: dict[str, Any],
    ctx: JobDescriptionContext,
    *,
    settings: Settings,
    ja: JobAttributesResult | None,
    skill_ranker: tuple[list[str], list[str], MappedEscoSkills, str] | None = None,
) -> dict[str, Any] | None:
    """
    Full ``$set`` body for the ranked-job collection document.

    Omits ``job_id`` from the written payload (fingerprint / filter keys unchanged).

    Returns ``None`` if there is no job-attributes result (nothing to persist).
    """
    if ja is None:
        return None

    now = datetime.now(timezone.utc)
    by_type = group_linked_entities_by_type(classified_doc, settings)
    classified_skills, classified_skills_uuids = _classified_rows_and_uuids(by_type.get("skill", []))
    classified_occupations, classified_occupations_uuids = _classified_rows_and_uuids(
        by_type.get("occupation", [])
    )
    classified_linked_by_type = _classified_linked_by_type(by_type, settings)
    wa_ids = list(ja.work_activity_ids)
    wa_names = work_activity_names_for_ids(ja.work_activity_ids, settings=settings)

    essential: list[str] = []
    optional: list[str] = []
    ess_uuids: list[str] = []
    opt_uuids: list[str] = []
    if skill_ranker is not None:
        essential, optional, mapped, _ = skill_ranker
        ess_uuids = [x for x in mapped.essential_skills_uuids if x]
        opt_uuids = [x for x in mapped.optional_skills_uuids if x]

    payload: dict[str, Any] = {
        "job_fingerprint": ctx.job_fingerprint,
        "title": ctx.title,
        "processed_at": now,
        "is_active": True,
        "source_classified_id": classified_doc.get("_id"),
        "classified_skills": classified_skills,
        "classified_skills_uuids": classified_skills_uuids,
        "classified_occupations": classified_occupations,
        "classified_occupations_uuids": classified_occupations_uuids,
        "classified_linked_by_type": classified_linked_by_type,
        "occupation_skills": [],
        "occupation_skills_uuids": [],
        "essential_skills": essential,
        "optional_skills": optional,
        "essential_skills_uuids": ess_uuids,
        "optional_skills_uuids": opt_uuids,
        "job_attributes": dict(ja.attributes),
        "work_activity_ids": wa_ids,
        "work_activity_names": wa_names,
        "classifier_metadata": classifier_metadata_from_context(ctx),
        "llm_reranker_meta": build_llm_reranker_meta(settings),
    }

    return payload
