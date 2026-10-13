"""Concat embedding whitening (bit-identical to shp-matching-algorithm v4).

Loads ``concat_whitening_gemini.npz`` (mu, W=Sigma^-1/2, target) once per process
and applies the same L2-normalize -> (v - mu) @ W -> re-L2-normalize transform used
by ``match_concat_gemini_ce_service.whiten_concat_rows``.

**Re-fit contract:** Any future re-fit of the whitening artifact must use
``raw_embedding`` (pre-whitening L2-normalized Gemini vectors) from Mongo, never
``job_embedding`` (which holds the whitened vector after pipeline cutover).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from utils.gemini_embeddings import l2_normalize_rows

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_NAME = "concat_whitening_gemini.npz"
CONCAT_WHITENING_ARTIFACT_SHA256 = (
    "5d96cbdd69bc3e920976ed0273036e859e13cb7140887f8367156249a8683f18"
)


class _Uncomputed:
    """Sentinel type: the SHA-256 has not been computed yet, distinct from a real "no artifact" ``None``."""


_UNCOMPUTED = _Uncomputed()

_concat_white_lock = threading.Lock()
_concat_white: dict[str, Any] | None = None  # None = not loaded; {} = missing/disabled
_artifact_sha256: str | None | _Uncomputed = _UNCOMPUTED


def _resolve_whitening_path(settings: Settings) -> Path:
    raw = settings.reranker_concat_whitening_path
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    under_res = (settings.resources_dir / p).resolve()
    if under_res.is_file():
        return under_res
    under_root = (settings.resources_dir.parent / p).resolve()
    return under_root if under_root.is_file() else under_res


def _compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reset_concat_whitening_cache() -> None:
    """Clear lazy-loaded artifact (for tests)."""
    global _concat_white, _artifact_sha256
    with _concat_white_lock:
        _concat_white = None
        _artifact_sha256 = _UNCOMPUTED


def _get_concat_whitening(settings: Settings | None = None) -> dict[str, Any] | None:
    """Lazy-load whitening artifact. Returns None if absent or invalid."""
    global _concat_white
    settings = settings or get_settings()
    if _concat_white is None:
        with _concat_white_lock:
            if _concat_white is None:
                path = _resolve_whitening_path(settings)
                if path.is_file():
                    try:
                        z = np.load(path)
                        mu = z["mu"].astype(np.float64)
                        w = z["W"].astype(np.float64)
                        target = float(z["target"])
                        dim = settings.reranker_job_embedding_dim
                        if mu.shape[0] != dim or w.shape[0] != dim or target <= 0:
                            msg = (
                                f"concat whitening artifact {path} incompatible "
                                f"(mu_dim={mu.shape[0]} W_dim={w.shape[0]} target={target:.4f}, "
                                f"expected dim={dim}, target>0)"
                            )
                            if settings.reranker_concat_whitening_required:
                                raise ValueError(msg)
                            logger.error("%s; whitening disabled", msg)
                            _concat_white = {}
                        else:
                            _concat_white = {"mu": mu, "W": w, "target": target, "path": str(path)}
                            logger.info(
                                "loaded concat whitening artifact %s (target=%.4f)",
                                path,
                                target,
                            )
                    except Exception as exc:
                        if settings.reranker_concat_whitening_required:
                            raise
                        logger.error("failed to load concat whitening artifact %s: %s", path, exc)
                        _concat_white = {}
                else:
                    msg = f"concat whitening artifact not found at {path}"
                    if settings.reranker_concat_whitening_required:
                        raise FileNotFoundError(msg)
                    logger.warning("%s; whitening disabled", msg)
                    _concat_white = {}
    return _concat_white or None


def whiten_concat_rows(vecs: np.ndarray, *, settings: Settings | None = None) -> np.ndarray:
    """L2-normalize rows, apply concat whitening, re-normalize.

    If the artifact is unavailable and ``reranker_concat_whitening_required`` is
    False, returns L2-normalized rows unchanged (matches v4 fallback).
    """
    settings = settings or get_settings()
    v = l2_normalize_rows(np.asarray(vecs, dtype=np.float64))
    cw = _get_concat_whitening(settings)
    if cw is None:
        return v
    return l2_normalize_rows((v - cw["mu"]) @ cw["W"])


def concat_whitening_target(*, settings: Settings | None = None) -> float:
    """p99 rescale target from artifact (0.0 if unavailable)."""
    cw = _get_concat_whitening(settings)
    return float(cw["target"]) if cw else 0.0


def concat_whitening_artifact_sha256(*, settings: Settings | None = None) -> str | None:
    """SHA-256 hex digest of the artifact file, computed once per process."""
    global _artifact_sha256
    settings = settings or get_settings()
    if isinstance(_artifact_sha256, _Uncomputed):
        path = _resolve_whitening_path(settings)
        if path.is_file():
            _artifact_sha256 = _compute_file_sha256(path)
        else:
            _artifact_sha256 = None
    return _artifact_sha256


def assert_concat_whitening_artifact_sha256(
    expected: str, *, settings: Settings | None = None
) -> str:
    """Raise if on-disk artifact SHA-256 does not match ``expected``."""
    actual = concat_whitening_artifact_sha256(settings=settings)
    if actual is None:
        raise FileNotFoundError("concat whitening artifact not found; cannot verify SHA-256")
    exp = expected.strip().lower()
    if actual.lower() != exp:
        raise ValueError(
            f"concat whitening artifact SHA-256 mismatch: expected {exp}, got {actual}"
        )
    return actual
