"""The root trace every ranked job runs under, for both the batch CLI and ``POST /rank``."""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

from app.observability.sampling import SamplingTier
from app.observability.tracing import start_trace, update_observation

if TYPE_CHECKING:
    from app.config import Settings
    from app.job_ranking.routes.types import RankRequest

logger = logging.getLogger(__name__)

JOB_TRACE_NAME = "rank_job"


def _doc_field(doc: dict[str, Any], name: str) -> str:
    """A top-level field, else the same name under ``metadata`` — how the classifier nests them."""
    value = doc.get(name)
    if value in (None, ""):
        meta = doc.get("metadata")
        value = meta.get(name) if isinstance(meta, dict) else None
    return str(value).strip() if value not in (None, "") else ""


def _as_document(doc: RankRequest | dict[str, Any]) -> dict[str, Any]:
    from app.job_ranking.routes.types import RankRequest

    return doc.to_document() if isinstance(doc, RankRequest) else doc


def job_trace_identity(doc: RankRequest | dict[str, Any], settings: Settings) -> str:
    """The job's stable id, used as the sampling key and shown on the trace."""
    d = _as_document(doc)
    return (
        _doc_field(d, settings.mongo_field_job_id)
        or _doc_field(d, settings.mongo_field_job_fingerprint)
        or _doc_field(d, "_id")
    )


@contextmanager
def job_trace(
    doc: RankRequest | dict[str, Any],
    *,
    settings: Settings,
    invocation: str,
    session_id: object = None,
    job_label: str = "",
) -> Iterator[Any]:
    """Trace one job: every LLM generation and embedding below nests under it.

    :param invocation: ``batch`` or ``api``; picks the sampling tier and is a tag.
    :param session_id: The batch run id, so one run's jobs group as a Langfuse session.
    :return: The root observation (``None`` when untraced); the caller sets its output.
    """
    doc = _as_document(doc)
    job_id = job_trace_identity(doc, settings)
    tier = SamplingTier.API if invocation == "api" else SamplingTier.BATCH
    llm = settings.llm
    tags = [
        f"invocation:{invocation}",
        f"country:{settings.target_country}",
        f"language:{settings.language}",
        f"llm_provider:{llm.provider}",
    ]
    metadata = {
        "job_id": job_id,
        "job_label": job_label,
        "invocation": invocation,
        "country": settings.target_country,
        "language": settings.language,
        "llm_provider": llm.provider,
        "llm_model": llm.model,
        "embedding_model": (
            settings.reranker_job_embedding_model if settings.reranker_enable_job_embedding else ""
        ),
        "source_collection": settings.mongodb_source_collection if invocation == "batch" else "",
        "target_collection": settings.mongodb_target_collection if invocation == "batch" else "",
    }
    trace_input = {
        "job_id": job_id,
        "title": _doc_field(doc, settings.field_title),
        "employer": _doc_field(doc, settings.field_employer),
        "source_platform": _doc_field(doc, settings.field_source_platform),
    }
    # A job with no id at all is sampled at random rather than all sharing one decision.
    sampling_key = job_id or uuid.uuid4().hex
    with start_trace(
        name=JOB_TRACE_NAME,
        tier=tier,
        sampling_key=sampling_key,
        session_id=session_id,
        input=trace_input,
        metadata=metadata,
        tags=tags,
    ) as root:
        try:
            yield root
        except Exception as e:
            update_observation(root, level="ERROR", status_message=f"{type(e).__name__}: {e}")
            raise
