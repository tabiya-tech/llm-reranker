"""Target collection document shape for LLM reranker + job attributes (``PIPELINE_SCHEMA_VERSION``)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.config import Settings, get_settings
from app.job_context import JobDescriptionContext
from app.schemas import JobAttributesResult, load_onet_work_activities
from utils.skills import MappedEscoSkills, extract_esco_skill_candidates

PIPELINE_SCHEMA_VERSION = "3"


def classified_skills_payload(
    classified_doc: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """ESCO candidates from classifier: serializable rows + deduped UUID list."""
    candidates = extract_esco_skill_candidates(classified_doc)
    skills: list[dict[str, Any]] = []
    uuids: list[str] = []
    for c in candidates:
        skills.append(
            {
                "label": c.label,
                "uri": c.uri,
                "esco_uuid": c.esco_uuid,
                "similarity_score": c.similarity_score,
            }
        )
        if c.esco_uuid:
            uuids.append(c.esco_uuid)
    return skills, uuids


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
    classified_skills, classified_skills_uuids = classified_skills_payload(classified_doc)
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
