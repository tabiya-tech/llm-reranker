"""The embedding client interface the job-embedding step talks to."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingClient(ABC):
    """Turns texts into vectors. One instance is built per process and passed to the services."""

    #: Model id recorded on the stored embedding's provenance block.
    model: str
    #: Width of every vector :meth:`embed` returns.
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts``; returns a float32 matrix of shape ``(len(texts), dim)``."""
        raise NotImplementedError


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True).astype(np.float32)
    norms = np.where(norms > 0, norms, 1.0).astype(np.float32)
    return (mat / norms).astype(np.float32)
