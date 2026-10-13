"""Gemini concat embeddings for job skill text (same space as shp-matching /match_v4)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional, Sequence, cast

import numpy as np

from app.observability import report_usage, traced_observation, update_observation

if TYPE_CHECKING:
    from google.genai import Client
    from google.genai.types import ContentListUnion, EmbedContentResponse

MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIM = 3072
TASK_TYPE = "SEMANTIC_SIMILARITY"
GEMINI_EMBED_API_MAX_BATCH = 100
DEFAULT_BATCH_SIZE = 100
DEFAULT_SLEEP_S = 0.12
MAX_RETRIES = 5


def _embed_chunk(
    client: Client,
    texts: Sequence[str],
    *,
    model: str,
    embedding_dim: int,
    task_type: str,
) -> np.ndarray:
    """One ``embed_content`` request (with retries), recorded as a Langfuse embedding."""
    from google.genai import types as genai_types

    payload = list(texts)
    with traced_observation(
        name="embed_content",
        as_type="embedding",
        input=payload,
        model=model,
        model_parameters={"task_type": task_type, "output_dimensionality": embedding_dim},
        metadata={
            "provider": "gemini",
            "texts": len(payload),
            "input_characters": sum(len(t) for t in payload),
        },
    ) as observation:
        last_err: Optional[BaseException] = None
        for attempt in range(MAX_RETRIES):
            try:
                result = client.models.embed_content(
                    model=model,
                    contents=cast("ContentListUnion", payload),
                    config=genai_types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=embedding_dim,
                    ),
                )
                if not result.embeddings:
                    raise RuntimeError("Gemini embed_content returned no embeddings")
                rows = []
                for emb in result.embeddings:
                    rows.append(np.asarray(emb.values, dtype=np.float32))
                mat = np.stack(rows, axis=0)
                # The vectors themselves are not exported: 3072 floats per text is ingest
                # cost with nothing to read. Their count and shape are what a trace needs.
                update_observation(
                    observation,
                    output={"vectors": int(mat.shape[0]), "dim": int(mat.shape[1])},
                    metadata={"attempts": attempt + 1},
                )
                report_usage(_embedding_usage(result))
                return mat
            except Exception as e:
                last_err = e
                if attempt == MAX_RETRIES - 1:
                    break
                time.sleep(2**attempt)
        update_observation(
            observation,
            level="ERROR",
            status_message=f"{type(last_err).__name__}: {last_err}",
            metadata={"attempts": MAX_RETRIES},
        )
    raise RuntimeError(f"Gemini embed_content failed after {MAX_RETRIES} attempts: {last_err}")


def _embedding_usage(result: EmbedContentResponse) -> dict[str, int] | None:
    """Input tokens, when the API reports them.

    Vertex AI returns a per-vector ``statistics.token_count``; the Gemini Developer API
    returns none, and then no usage is recorded rather than an invented one.
    """
    total = 0
    seen = False
    for emb in getattr(result, "embeddings", None) or []:
        count = getattr(getattr(emb, "statistics", None), "token_count", None)
        if isinstance(count, (int, float)) and not isinstance(count, bool):
            total += int(count)
            seen = True
    return {"input": total, "total": total} if seen else None


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
