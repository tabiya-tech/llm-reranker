"""Integration tests for dual-field job embedding with concat whitening."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from app.config import Settings
from utils.concat_whitening import (
    CONCAT_WHITENING_ARTIFACT_SHA256,
    reset_concat_whitening_cache,
    whiten_concat_rows,
)
from utils.gemini_embeddings import l2_normalize_rows
from utils.job_embedding import attach_job_embedding_to_payload


@pytest.fixture(autouse=True)
def _reset_whitening_cache():
    reset_concat_whitening_cache()
    yield
    reset_concat_whitening_cache()


def _sample_payload() -> dict:
    return {
        "job_id": "job-123",
        "llm_classified_skills": {
            "essential": [{"label": "Python programming"}],
            "optional": [{"label": "Data analysis"}],
        },
    }


def test_whitening_disabled_writes_raw_job_embedding_only(isolated_env):
    isolated_env.setenv("GEMINI_API_KEY", "test-key")
    isolated_env.setenv("RERANKER_CONCAT_WHITENING_ENABLED", "0")
    settings = Settings.from_env()
    raw_vec = l2_normalize_rows(np.random.default_rng(1).standard_normal((1, 3072)))[0]

    with patch("utils.job_embedding.embed_text_list", return_value=raw_vec.reshape(1, -1)):
        out = attach_job_embedding_to_payload(_sample_payload(), settings=settings)

    assert "job_embedding" in out
    assert "raw_embedding" not in out
    assert len(out["job_embedding"]) == 3072
    assert out["llm_reranker_meta"]["embedding"].get("whitening") is None


def test_whitening_enabled_writes_dual_fields_and_provenance(isolated_env):
    isolated_env.setenv("GEMINI_API_KEY", "test-key")
    isolated_env.setenv("RERANKER_CONCAT_WHITENING_ENABLED", "1")
    settings = Settings.from_env()
    raw_vec = l2_normalize_rows(np.random.default_rng(2).standard_normal((1, 3072)))
    expected_white = whiten_concat_rows(raw_vec, settings=settings)[0]

    with patch("utils.job_embedding.embed_text_list", return_value=raw_vec):
        out = attach_job_embedding_to_payload(_sample_payload(), settings=settings)

    assert "raw_embedding" in out
    assert "job_embedding" in out
    assert len(out["raw_embedding"]) == 3072
    assert len(out["job_embedding"]) == 3072
    np.testing.assert_allclose(out["raw_embedding"], raw_vec[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out["job_embedding"], expected_white, rtol=1e-5, atol=1e-5)

    whitening = out["llm_reranker_meta"]["embedding"]["whitening"]
    assert whitening["enabled"] is True
    assert whitening["artifact_sha256"] == CONCAT_WHITENING_ARTIFACT_SHA256
    assert whitening["target"] > 0
    assert whitening["applied_at"]

    cosine = float(
        np.dot(out["job_embedding"], out["raw_embedding"])
        / (np.linalg.norm(out["job_embedding"]) * np.linalg.norm(out["raw_embedding"]))
    )
    assert cosine < 1.0
