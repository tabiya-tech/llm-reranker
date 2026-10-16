"""Integration tests for dual-field job embedding with concat whitening."""

from __future__ import annotations

import numpy as np

from app.config import Settings
from app.job_embedding.client import EmbeddingClient, l2_normalize_rows
from app.job_embedding.service import JobEmbeddingService
from app.job_embedding.whitening import CONCAT_WHITENING_ARTIFACT_SHA256, ConcatWhitener
from app.job_embedding.whitening_repository import WhiteningArtifactRepository


class _FixedEmbeddingClient(EmbeddingClient):
    """Returns one prepared matrix for every call."""

    def __init__(self, vectors: np.ndarray):
        self.model = "fixed"
        self.dim = int(vectors.shape[1])
        self._vectors = vectors

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._vectors


def _attach(settings: Settings, vectors: np.ndarray, payload: dict) -> dict:
    service = JobEmbeddingService(
        settings, _FixedEmbeddingClient(vectors), WhiteningArtifactRepository()
    )
    return service.attach(payload)


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

    out = _attach(settings, raw_vec.reshape(1, -1), _sample_payload())

    assert "job_embedding" in out
    assert "raw_embedding" not in out
    assert len(out["job_embedding"]) == 3072
    assert out["llm_reranker_meta"]["embedding"].get("whitening") is None


def test_whitening_enabled_writes_dual_fields_and_provenance(isolated_env):
    isolated_env.setenv("GEMINI_API_KEY", "test-key")
    isolated_env.setenv("RERANKER_CONCAT_WHITENING_ENABLED", "1")
    settings = Settings.from_env()
    raw_vec = l2_normalize_rows(np.random.default_rng(2).standard_normal((1, 3072)))
    expected_white = ConcatWhitener(WhiteningArtifactRepository().load(settings)).whiten(raw_vec)[0]

    out = _attach(settings, raw_vec, _sample_payload())

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


def test_scalar_artifact_not_required_falls_back_to_l2_only(isolated_env, tmp_path):
    path = tmp_path / "scalar.npz"
    with path.open("wb") as f:
        np.savez(f, mu=np.float64(0.0), W=np.float64(1.0), target=np.float64(1.0))
    isolated_env.setenv("GEMINI_API_KEY", "test-key")
    isolated_env.setenv("RERANKER_CONCAT_WHITENING_ENABLED", "1")
    isolated_env.setenv("RERANKER_CONCAT_WHITENING_REQUIRED", "0")
    isolated_env.setenv("RERANKER_CONCAT_WHITENING_PATH", str(path))
    settings = Settings.from_env()
    raw_vec = l2_normalize_rows(np.random.default_rng(3).standard_normal((1, 3072)))

    out = _attach(settings, raw_vec, _sample_payload())

    np.testing.assert_allclose(out["job_embedding"], raw_vec[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out["raw_embedding"], raw_vec[0], rtol=1e-6, atol=1e-6)
    assert out["llm_reranker_meta"]["embedding"]["whitening"]["artifact_sha256"] is None
