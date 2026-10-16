"""Unit tests for concat whitening artifact load and v4 parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.job_embedding.client import l2_normalize_rows
from app.job_embedding.whitening import CONCAT_WHITENING_ARTIFACT_SHA256, ConcatWhitener
from app.job_embedding.whitening_repository import WhiteningArtifactRepository

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACT = _REPO_ROOT / "resources" / "concat_whitening_gemini.npz"


@pytest.fixture
def whitener(isolated_env) -> ConcatWhitener:
    return ConcatWhitener(WhiteningArtifactRepository().load(Settings.from_env()))


def test_artifact_sha256_pinned(whitener):
    assert _ARTIFACT.is_file(), f"missing whitening artifact at {_ARTIFACT}"
    assert whitener.artifact_sha256 == CONCAT_WHITENING_ARTIFACT_SHA256
    whitener.assert_sha256(CONCAT_WHITENING_ARTIFACT_SHA256)


def test_sha256_mismatch_raises(whitener):
    with pytest.raises(ValueError, match="mismatch"):
        whitener.assert_sha256("0" * 64)


def test_whiten_changes_non_trivial_vector(whitener):
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((3, 3072))
    raw = l2_normalize_rows(raw)
    out = whitener.whiten(raw)
    assert out.shape == raw.shape
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5, atol=1e-5)
    assert not np.allclose(out, raw)


def test_whiten_not_idempotent_on_whitened_vectors(whitener):
    """Applying whitening twice changes the vector (v4 applies it once only)."""
    rng = np.random.default_rng(7)
    raw = l2_normalize_rows(rng.standard_normal((1, 3072)))
    once = whitener.whiten(raw)
    twice = whitener.whiten(once)
    assert not np.allclose(once, twice, rtol=1e-6, atol=1e-6)


def test_concat_whitening_target_positive(whitener):
    assert whitener.target > 0.0


def test_missing_artifact_required_raises(isolated_env, monkeypatch):
    isolated_env.setenv(
        "RERANKER_CONCAT_WHITENING_PATH", "/nonexistent/concat_whitening_gemini.npz"
    )
    isolated_env.setenv("RERANKER_CONCAT_WHITENING_REQUIRED", "1")
    settings = Settings.from_env()
    with pytest.raises(FileNotFoundError):
        WhiteningArtifactRepository().load(settings)


def test_missing_artifact_not_required_returns_l2_only(isolated_env, monkeypatch):
    isolated_env.setenv(
        "RERANKER_CONCAT_WHITENING_PATH", "/nonexistent/concat_whitening_gemini.npz"
    )
    isolated_env.setenv("RERANKER_CONCAT_WHITENING_REQUIRED", "0")
    settings = Settings.from_env()
    artifact = WhiteningArtifactRepository().load(settings)
    assert artifact is None
    raw = l2_normalize_rows(np.random.default_rng(0).standard_normal((1, 3072)))
    out = ConcatWhitener(artifact).whiten(raw)
    np.testing.assert_allclose(out, raw, rtol=1e-6, atol=1e-6)


def _write_artifact(path: Path, *, mu: np.ndarray, w: np.ndarray, target: float = 1.0) -> None:
    with path.open("wb") as f:
        np.savez(f, mu=mu, W=w, target=np.float64(target))


_DIM = 3072
_INCOMPATIBLE = {
    "scalar_mu": (np.float64(0.0), np.eye(_DIM)),
    "scalar_w": (np.zeros(_DIM), np.float64(1.0)),
    "w_one_dimensional": (np.zeros(_DIM), np.ones(_DIM)),
    "wrong_dim": (np.zeros(_DIM - 1), np.eye(_DIM - 1)),
}


@pytest.mark.parametrize("case", sorted(_INCOMPATIBLE))
def test_incompatible_artifact_not_required_returns_l2_only(isolated_env, tmp_path, case):
    mu, w = _INCOMPATIBLE[case]
    path = tmp_path / "artifact.npz"
    _write_artifact(path, mu=mu, w=w)
    isolated_env.setenv("RERANKER_CONCAT_WHITENING_PATH", str(path))
    isolated_env.setenv("RERANKER_CONCAT_WHITENING_REQUIRED", "0")

    assert WhiteningArtifactRepository().load(Settings.from_env()) is None


@pytest.mark.parametrize("case", sorted(_INCOMPATIBLE))
def test_incompatible_artifact_required_raises(isolated_env, tmp_path, case):
    mu, w = _INCOMPATIBLE[case]
    path = tmp_path / "artifact.npz"
    _write_artifact(path, mu=mu, w=w)
    isolated_env.setenv("RERANKER_CONCAT_WHITENING_PATH", str(path))
    isolated_env.setenv("RERANKER_CONCAT_WHITENING_REQUIRED", "1")

    with pytest.raises(ValueError, match="incompatible"):
        WhiteningArtifactRepository().load(Settings.from_env())
