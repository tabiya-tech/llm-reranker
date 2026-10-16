"""Pick the identity a ranked job is upserted on (pure: no Mongo I/O)."""

from __future__ import annotations

from typing import Any

from app.job_context import JobDescriptionContext
from app.ranked_jobs.types import RankedJobKey, RankedJobUpsert


def ranked_job_upsert(
    classified_doc: dict[str, Any],
    ctx: JobDescriptionContext,
    ranked_document: dict[str, Any],
) -> RankedJobUpsert:
    """
    The upsert for one ranked document (``JobResult.ranked_document``, with the job embedding
    already attached by ``JobEmbeddingService``), keyed on the job id, else its fingerprint, else
    the classified document's ``_id``.

    :raises ValueError: the classified document has none of the three.
    """
    if ctx.job_id is not None:
        return RankedJobUpsert(RankedJobKey.JOB_ID, ctx.job_id, ranked_document)
    if ctx.job_fingerprint:
        return RankedJobUpsert(RankedJobKey.JOB_FINGERPRINT, ctx.job_fingerprint, ranked_document)
    oid = classified_doc.get("_id")
    if oid is not None:
        return RankedJobUpsert(RankedJobKey.SOURCE_CLASSIFIED_ID, oid, ranked_document)
    raise ValueError(
        "Cannot upsert LLM results: document has no job id, fingerprint, or source _id"
    )
