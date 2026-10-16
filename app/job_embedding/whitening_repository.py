"""Reads the concat whitening artifact (``concat_whitening_gemini.npz``)."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from app.config import Settings
from app.job_embedding.whitening import WhiteningArtifact

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_NAME = "concat_whitening_gemini.npz"


def resolve_whitening_path(settings: Settings) -> Path:
    raw = settings.reranker_concat_whitening_path
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    under_res = (settings.resources_dir / p).resolve()
    if under_res.is_file():
        return under_res
    under_root = (settings.resources_dir.parent / p).resolve()
    return under_root if under_root.is_file() else under_res


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class WhiteningArtifactRepository:
    """Loads the artifact named by ``RERANKER_CONCAT_WHITENING_PATH``."""

    def load(self, settings: Settings) -> WhiteningArtifact | None:
        """The artifact, or ``None`` when it is absent or invalid and not required.

        With ``RERANKER_CONCAT_WHITENING_REQUIRED`` a missing or incompatible artifact raises.
        """
        required = settings.reranker_concat_whitening_required
        path = resolve_whitening_path(settings)
        if not path.is_file():
            msg = f"concat whitening artifact not found at {path}"
            if required:
                raise FileNotFoundError(msg)
            logger.warning("%s; whitening disabled", msg)
            return None
        try:
            z = np.load(path)
            mu = z["mu"].astype(np.float64)
            w = z["W"].astype(np.float64)
            target = float(z["target"])
        except Exception as exc:
            if required:
                raise
            logger.error("failed to load concat whitening artifact %s: %s", path, exc)
            return None
        dim = settings.reranker_job_embedding_dim
        # Whole shapes, not ``shape[0]``: a scalar ``mu`` or ``W`` has shape ``()``.
        if mu.shape != (dim,) or w.shape != (dim, dim) or not target > 0:
            msg = (
                f"concat whitening artifact {path} incompatible "
                f"(mu_shape={mu.shape} W_shape={w.shape} target={target:.4f}, "
                f"expected mu_shape={(dim,)} W_shape={(dim, dim)}, target>0)"
            )
            if required:
                raise ValueError(msg)
            logger.error("%s; whitening disabled", msg)
            return None
        logger.info("loaded concat whitening artifact %s (target=%.4f)", path, target)
        return WhiteningArtifact(
            mu=mu, w=w, target=target, path=str(path), sha256=_file_sha256(path)
        )
