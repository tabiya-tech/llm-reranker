"""Attach Gemini concat job embeddings to ranked job documents."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.config import Settings

from utils.concat_embedding_text import ranked_job_concat_embedding_text
from utils.gemini_embeddings import (
    EMBEDDING_DIM,
    MODEL_NAME,
    TASK_TYPE,
    embed_text_list,
    l2_normalize_rows,
)

logger = logging.getLogger(__name__)

# Describes what text was embedded (not an external NPZ path).
DEFAULT_EMBEDDING_SOURCE = "llm_classified_skills"


def build_embedding_section(
    *,
    model: str = MODEL_NAME,
    dim: int = EMBEDDING_DIM,
    source: str | None = DEFAULT_EMBEDDING_SOURCE,
    loaded_at: datetime | None = None,
) -> dict[str, Any]:
    """Embedding provenance block nested under ``llm_reranker_meta.embedding``."""
    section: dict[str, Any] = {
        "model": model,
        "dim": dim,
        "loaded_at": (loaded_at or datetime.now(timezone.utc)).isoformat(),
    }
    if source:
        section["source"] = source
    return section


def merge_embedding_into_llm_reranker_meta(
    llm_reranker_meta: dict[str, Any],
    *,
    settings: Settings,
    loaded_at: datetime | None = None,
) -> dict[str, Any]:
    """Unify embedding provenance into ``llm_reranker_meta`` (replaces top-level ``embedding_meta``)."""
    out = dict(llm_reranker_meta)
    out["embedding"] = build_embedding_section(
        model=settings.reranker_job_embedding_model,
        dim=settings.reranker_job_embedding_dim,
        source=settings.reranker_job_embedding_source,
        loaded_at=loaded_at,
    )
    return out


def attach_job_embedding_to_payload(
    payload: dict[str, Any],
    *,
    settings: Settings,
) -> dict[str, Any]:
    """Embed concat skill text; set ``job_embedding`` and ``llm_reranker_meta.embedding``."""
    if not settings.reranker_enable_job_embedding:
        return payload

    api_key = (settings.gemini_api_key or "").strip()
    if not api_key:
        logger.warning("job_embedding skipped: GEMINI_API_KEY not set")
        return payload

    text = ranked_job_concat_embedding_text(payload).strip() or " "

    try:
        vec = embed_text_list(
            [text],
            api_key=api_key,
            batch_size=settings.reranker_job_embedding_batch_size,
            sleep_s=settings.reranker_job_embedding_sleep_s,
            model=settings.reranker_job_embedding_model,
            embedding_dim=settings.reranker_job_embedding_dim,
            task_type=settings.reranker_job_embedding_task_type,
        )
        if settings.reranker_job_embedding_l2_normalize:
            vec = l2_normalize_rows(vec)
        row = vec[0]
        if row.size != settings.reranker_job_embedding_dim:
            raise ValueError(
                f"expected dim {settings.reranker_job_embedding_dim}, got {row.size}"
            )
        payload["job_embedding"] = [float(x) for x in row.tolist()]
        base_meta = payload.get("llm_reranker_meta")
        if not isinstance(base_meta, dict):
            base_meta = {}
        payload["llm_reranker_meta"] = merge_embedding_into_llm_reranker_meta(
            base_meta,
            settings=settings,
        )
    except Exception as exc:
        if settings.reranker_fail_on_job_embedding_error:
            raise
        logger.warning(
            "job_embedding failed for job_id=%s: %s",
            payload.get("job_id"),
            exc,
        )
    return payload
