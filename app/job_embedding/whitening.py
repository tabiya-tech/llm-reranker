"""Concat embedding whitening (bit-identical to shp-matching-algorithm v4). No IO.

Applies the same L2-normalize -> (v - mu) @ W -> re-L2-normalize transform used by
``match_concat_gemini_ce_service.whiten_concat_rows``. The artifact (``mu``, ``W = Sigma^-1/2``,
``target``) is read by :class:`app.job_embedding.whitening_repository.WhiteningArtifactRepository`.

**Re-fit contract:** Any future re-fit of the whitening artifact must use
``raw_embedding`` (pre-whitening L2-normalized Gemini vectors) from Mongo, never
``job_embedding`` (which holds the whitened vector after pipeline cutover).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.job_embedding.client import l2_normalize_rows

CONCAT_WHITENING_ARTIFACT_SHA256 = (
    "5d96cbdd69bc3e920976ed0273036e859e13cb7140887f8367156249a8683f18"
)


@dataclass(frozen=True)
class WhiteningArtifact:
    """The contents of ``concat_whitening_gemini.npz`` plus where it came from."""

    mu: np.ndarray
    w: np.ndarray
    # p99 rescale target.
    target: float
    path: str
    sha256: str


class ConcatWhitener:
    """Whitens concat job embeddings with an artifact, or only L2-normalizes without one.

    ``artifact=None`` is the "missing but not required" case and matches v4's fallback.
    """

    def __init__(self, artifact: WhiteningArtifact | None):
        self._artifact = artifact

    @property
    def target(self) -> float:
        """p99 rescale target from the artifact (0.0 if unavailable)."""
        return self._artifact.target if self._artifact else 0.0

    @property
    def artifact_sha256(self) -> str | None:
        return self._artifact.sha256 if self._artifact else None

    def whiten(self, vecs: np.ndarray) -> np.ndarray:
        """L2-normalize rows, apply concat whitening, re-normalize."""
        v = l2_normalize_rows(np.asarray(vecs, dtype=np.float64))
        if self._artifact is None:
            return v
        return l2_normalize_rows((v - self._artifact.mu) @ self._artifact.w)

    def assert_sha256(self, expected: str) -> str:
        """Raise if the artifact's SHA-256 does not match ``expected``."""
        actual = self.artifact_sha256
        if actual is None:
            raise FileNotFoundError("concat whitening artifact not found; cannot verify SHA-256")
        exp = expected.strip().lower()
        if actual.lower() != exp:
            raise ValueError(
                f"concat whitening artifact SHA-256 mismatch: expected {exp}, got {actual}"
            )
        return actual
