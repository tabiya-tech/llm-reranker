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
        return RankedJobUpsert(
            key=RankedJobKey.JOB_ID, key_value=ctx.job_id, document=ranked_document
        )
    if ctx.job_fingerprint:
        return RankedJobUpsert(
            key=RankedJobKey.JOB_FINGERPRINT,
            key_value=ctx.job_fingerprint,
            document=ranked_document,
        )
    oid = classified_doc.get("_id")
    if oid is not None:
        return RankedJobUpsert(
            key=RankedJobKey.SOURCE_CLASSIFIED_ID, key_value=oid, document=ranked_document
        )
    raise ValueError(
        "Cannot upsert LLM results: document has no job id, fingerprint, or source _id"
    )
