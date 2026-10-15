"""Build the target-collection ``UpdateOne`` for one ranked job (pure: no Mongo I/O)."""

from __future__ import annotations

from typing import Any

from pymongo.operations import UpdateOne

from app.config import Settings, get_settings
from app.job_context import JobDescriptionContext


def _upsert_filter_for_llm_results(
    classified_doc: dict[str, Any],
    ctx: JobDescriptionContext,
    settings: Settings,
) -> dict[str, Any]:
    if ctx.job_id is not None:
        return {settings.mongo_field_job_id: ctx.job_id}
    if ctx.job_fingerprint:
        return {settings.mongo_field_job_fingerprint: ctx.job_fingerprint}
    oid = classified_doc.get("_id")
    if oid is not None:
        return {"source_classified_id": oid}
    raise ValueError(
        "Cannot upsert LLM results: document has no job id, fingerprint, or source _id"
    )


def ranked_job_upsert_update_one(
    classified_doc: dict[str, Any],
    ctx: JobDescriptionContext,
    ranked_document: dict[str, Any],
    *,
    settings: Settings | None = None,
) -> UpdateOne:
    """
    Build one target-collection ``UpdateOne`` from a ranked document (``JobResult.ranked_document``).

    Attaches the job embedding first (``RERANKER_ENABLE_JOB_EMBEDDING``); the API never embeds,
    so this is the one step the batch adds to the shared pipeline's output.

    May raise ``ValueError`` if the classified document cannot be matched (see ``_upsert_filter_for_llm_results``).
    """
    from utils.job_embedding import attach_job_embedding_to_payload

    settings = settings or get_settings()
    flt = _upsert_filter_for_llm_results(classified_doc, ctx, settings)
    payload = attach_job_embedding_to_payload(ranked_document, settings=settings)
    # Drop legacy nested raw-response blobs (schema v2+).
    return UpdateOne(
        flt,
        {
            "$set": payload,
            "$unset": {
                "llm_skill_ranker": "",
                "classified_linked_by_type": "",
                "occupation_skills": "",
                "occupation_skills_uuids": "",
                "classified_skills_uuids": "",
                "classified_occupations_uuids": "",
                "essential_skills": "",
                "essential_skills_uuids": "",
                "optional_skills": "",
                "optional_skills_uuids": "",
                "job_attributes": "",
                "work_activity_ids": "",
                "work_activity_names": "",
                "title": "",
                "enrichment": "",
                "embedding_meta": "",
            },
        },
        upsert=True,
    )
