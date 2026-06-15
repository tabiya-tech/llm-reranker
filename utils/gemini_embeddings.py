"""Gemini concat embeddings for job skill text (same space as shp-matching /match_v4)."""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

import numpy as np

MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIM = 3072
TASK_TYPE = "SEMANTIC_SIMILARITY"
GEMINI_EMBED_API_MAX_BATCH = 100
DEFAULT_BATCH_SIZE = 100
DEFAULT_SLEEP_S = 0.12
MAX_RETRIES = 5


def _embed_chunk(
    client: Any,
    texts: Sequence[str],
    *,
    model: str,
    embedding_dim: int,
    task_type: str,
) -> np.ndarray:
    from google.genai import types as genai_types

    payload = list(texts)
    last_err: Optional[BaseException] = None
    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.embed_content(
                model=model,
                contents=payload,
                config=genai_types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=embedding_dim,
                ),
            )
            rows = []
            for emb in result.embeddings:
                rows.append(np.asarray(emb.values, dtype=np.float32))
            return np.stack(rows, axis=0)
        except Exception as e:
            last_err = e
            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"Gemini embed_content failed after {MAX_RETRIES} attempts: {last_err}")


def embed_text_list(
    texts: list[str],
    *,
    api_key: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sleep_s: float = DEFAULT_SLEEP_S,
    model: str = MODEL_NAME,
    embedding_dim: int = EMBEDDING_DIM,
    task_type: str = TASK_TYPE,
) -> np.ndarray:
    """Embed one or more texts; returns float32 matrix (n, embedding_dim)."""
    from google import genai

    if not texts:
        return np.zeros((0, embedding_dim), dtype=np.float32)
    batch_size = min(max(1, batch_size), GEMINI_EMBED_API_MAX_BATCH)
    client = genai.Client(api_key=api_key.strip())
    safe = [t.strip() if t.strip() else " " for t in texts]
    out = np.zeros((len(safe), embedding_dim), dtype=np.float32)
    n_batches = (len(safe) + batch_size - 1) // batch_size
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, len(safe))
        out[start:end] = _embed_chunk(
            client,
            safe[start:end],
            model=model,
            embedding_dim=embedding_dim,
            task_type=task_type,
        )
        if b < n_batches - 1 and sleep_s > 0:
            time.sleep(sleep_s)
    return out


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True).astype(np.float32)
    norms = np.where(norms > 0, norms, 1.0).astype(np.float32)
    return (mat / norms).astype(np.float32)
