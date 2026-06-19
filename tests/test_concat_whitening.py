"""Unit tests for concat whitening artifact load and v4 parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from utils.concat_whitening import (
    CONCAT_WHITENING_ARTIFACT_SHA256,
    assert_concat_whitening_artifact_sha256,
    concat_whitening_artifact_sha256,
    concat_whitening_target,
    reset_concat_whitening_cache,
    whiten_concat_rows,
)
from utils.gemini_embeddings import l2_normalize_rows

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACT = _REPO_ROOT / "resources" / "concat_whitening_gemini.npz"


@pytest.fixture(autouse=True)
def _reset_whitening_cache():
    reset_concat_whitening_cache()
    yield
    reset_concat_whitening_cache()


def test_artifact_sha256_pinned():
    assert _ARTIFACT.is_file(), f"missing whitening artifact at {_ARTIFACT}"
    assert concat_whitening_artifact_sha256() == CONCAT_WHITENING_ARTIFACT_SHA256
    assert_concat_whitening_artifact_sha256(CONCAT_WHITENING_ARTIFACT_SHA256)


def test_whiten_concat_rows_changes_non_trivial_vector():
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((3, 3072))
    raw = l2_normalize_rows(raw)
    out = whiten_concat_rows(raw)
    assert out.shape == raw.shape
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5, atol=1e-5)
    assert not np.allclose(out, raw)


def test_whiten_concat_rows_not_idempotent_on_whitened_vectors():
    """Applying whitening twice changes the vector (v4 applies it once only)."""
    rng = np.random.default_rng(7)
    raw = l2_normalize_rows(rng.standard_normal((1, 3072)))
    once = whiten_concat_rows(raw)
    twice = whiten_concat_rows(once)
    assert not np.allclose(once, twice, rtol=1e-6, atol=1e-6)


def test_concat_whitening_target_positive():
    target = concat_whitening_target()
    assert target > 0.0


def test_missing_artifact_required_raises(isolated_env, monkeypatch):
    isolated_env.setenv(
        "RERANKER_CONCAT_WHITENING_PATH", "/nonexistent/concat_whitening_gemini.npz"
    )
    isolated_env.setenv("RERANKER_CONCAT_WHITENING_REQUIRED", "1")
    reset_concat_whitening_cache()
    settings = Settings.from_env()
    raw = l2_normalize_rows(np.ones((1, settings.reranker_job_embedding_dim)))
    with pytest.raises(FileNotFoundError):
        whiten_concat_rows(raw, settings=settings)


def test_missing_artifact_not_required_returns_l2_only(isolated_env, monkeypatch):
    isolated_env.setenv(
        "RERANKER_CONCAT_WHITENING_PATH", "/nonexistent/concat_whitening_gemini.npz"
    )
    isolated_env.setenv("RERANKER_CONCAT_WHITENING_REQUIRED", "0")
    reset_concat_whitening_cache()
    settings = Settings.from_env()
    raw = l2_normalize_rows(np.random.default_rng(0).standard_normal((1, 3072)))
    out = whiten_concat_rows(raw, settings=settings)
    np.testing.assert_allclose(out, raw, rtol=1e-6, atol=1e-6)
