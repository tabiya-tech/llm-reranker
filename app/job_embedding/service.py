"""Attach concat job embeddings to ranked job documents (batch only; the API never embeds)."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from app.config import Settings
from app.job_embedding.client import EmbeddingClient, l2_normalize_rows
from app.job_embedding.gemini_client import EMBEDDING_DIM, MODEL_NAME
from app.job_embedding.whitening import ConcatWhitener
from app.job_embedding.whitening_repository import WhiteningArtifactRepository
from utils.concat_embedding_text import ranked_job_concat_embedding_text

logger = logging.getLogger(__name__)

# Describes what text was embedded (not an external NPZ path).
DEFAULT_EMBEDDING_SOURCE = "llm_classified_skills"


def build_embedding_section(
    *,
    model: str = MODEL_NAME,
    dim: int = EMBEDDING_DIM,
    source: str | None = DEFAULT_EMBEDDING_SOURCE,
    loaded_at: datetime | None = None,
    whitening: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Embedding provenance block nested under ``llm_reranker_meta.embedding``."""
    section: dict[str, Any] = {
        "model": model,
        "dim": dim,
        "loaded_at": (loaded_at or datetime.now(timezone.utc)).isoformat(),
    }
    if source:
        section["source"] = source
    if whitening:
        section["whitening"] = whitening
    return section


def merge_embedding_into_llm_reranker_meta(
    llm_reranker_meta: dict[str, Any],
    *,
    settings: Settings,
    loaded_at: datetime | None = None,
    whitening: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Unify embedding provenance into ``llm_reranker_meta`` (replaces top-level ``embedding_meta``)."""
    out = dict(llm_reranker_meta)
    out["embedding"] = build_embedding_section(
        model=settings.reranker_job_embedding_model,
        dim=settings.reranker_job_embedding_dim,
        source=settings.reranker_job_embedding_source,
        loaded_at=loaded_at,
        whitening=whitening,
    )
    return out


class JobEmbeddingService:
    """Embeds a ranked job's skill labels and writes ``job_embedding`` (``RERANKER_ENABLE_JOB_EMBEDDING``).

    ``client`` is ``None`` when no embedding credentials are configured; the step is then skipped
    with a warning. The whitening artifact is read on first use, so a missing artifact fails the
    embedding of each job (a warning, unless ``RERANKER_FAIL_ON_JOB_EMBEDDING_ERROR``), not startup.
    """

    def __init__(
        self,
        settings: Settings,
        client: EmbeddingClient | None,
        whitening: WhiteningArtifactRepository,
    ):
        self._settings = settings
        self._client = client
        self._whitening = whitening
        self._whitener: ConcatWhitener | None = None
        self._lock = threading.Lock()

    def _get_whitener(self) -> ConcatWhitener:
        if self._whitener is None:
            with self._lock:
                if self._whitener is None:
                    self._whitener = ConcatWhitener(self._whitening.load(self._settings))
        return self._whitener

    def attach(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Embed concat skill text; set ``job_embedding`` and ``llm_reranker_meta.embedding``."""
        settings = self._settings
        if not settings.reranker_enable_job_embedding:
            return payload
        if self._client is None:
            logger.warning("job_embedding skipped: GEMINI_API_KEY not set")
            return payload

        text = ranked_job_concat_embedding_text(payload).strip() or " "
        try:
            vec = self._client.embed([text])
            if settings.reranker_job_embedding_l2_normalize:
                vec = l2_normalize_rows(vec)
            raw_row = vec[0]
            if raw_row.size != settings.reranker_job_embedding_dim:
                raise ValueError(
                    f"expected dim {settings.reranker_job_embedding_dim}, got {raw_row.size}"
                )

            loaded_at = datetime.now(timezone.utc)
            whitening_meta: dict[str, Any] | None = None

            if settings.reranker_concat_whitening_enabled:
                whitener = self._get_whitener()
                whitened_row = whitener.whiten(vec)[0]
                payload["raw_embedding"] = [float(x) for x in raw_row.tolist()]
                payload["job_embedding"] = [float(x) for x in whitened_row.tolist()]
                whitening_meta = {
                    "enabled": True,
                    "artifact_sha256": whitener.artifact_sha256,
                    "target": whitener.target,
                    "applied_at": loaded_at.isoformat(),
                }
            else:
                payload["job_embedding"] = [float(x) for x in raw_row.tolist()]

            base_meta = payload.get("llm_reranker_meta")
            if not isinstance(base_meta, dict):
                base_meta = {}
            payload["llm_reranker_meta"] = merge_embedding_into_llm_reranker_meta(
                base_meta,
                settings=settings,
                loaded_at=loaded_at,
                whitening=whitening_meta,
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
