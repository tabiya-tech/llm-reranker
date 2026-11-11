"""``POST /rank/skills`` end to end, with the LLM step stubbed."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import api
from app.config import Settings
from app.job_ranking import get_job_ranking_service
from app.job_ranking.experience_skills.types import ExperienceSkillsResult
from app.job_ranking.service import JobRankingService, JobRankingServiceFactory
from app.llm import TracedLlmClient
from app.llm.errors import LlmInvocationError, LlmParseError
from tests.ranking_fakes import FixedRankers, StubExperienceSkillRanker, build_ranking_service


def _request(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "experience": {
            "title": "Baker",
            "company": "Mama's Bakery",
            "work_type": "SELF_EMPLOYMENT",
            "responsibilities": ["I bake bread", "I sell bread at the market"],
        },
        "candidates": [
            {"id": "a1", "label": "bake goods", "description": "Bake bread.", "score": 0.82},
            {"id": "b2", "label": "sell products", "description": "Sell.", "score": 0.77},
            {"id": "c3", "label": "operate a forklift", "score": 0.4},
        ],
        "max_skills": 2,
    }
    body.update(overrides)
    return body


def _serve(monkeypatch, ranking: JobRankingService) -> None:
    def _build(settings: Settings, llm: TracedLlmClient) -> JobRankingServiceFactory:
        return FixedRankers(ranking)

    monkeypatch.setattr(get_job_ranking_service, "build_job_ranking_service_factory", _build)


@pytest.fixture
def stub_ranker():
    return StubExperienceSkillRanker()


@pytest.fixture
def client(make_settings, monkeypatch, stub_ranker):
    s = make_settings(GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya")
    _serve(monkeypatch, build_ranking_service(s, experience_skill_ranker=stub_ranker))
    with TestClient(api.app) as c:
        yield c


class TestRankSkills:
    def test_ranks_the_candidates(self, client, stub_ranker):
        resp = client.post("/rank/skills", json=_request())

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert set(body) == {"ranked_ids", "discarded_ids", "timings", "model"}
        assert body["ranked_ids"] == ["a1", "b2"]
        assert body["discarded_ids"] == ["c3"]
        assert set(body["timings"]) == {"llm_s", "total_s"}
        assert body["model"]
        # The step got the experience and every candidate, as sent.
        (experience, candidates, max_skills) = stub_ranker.calls[0]
        assert experience.title == "Baker"
        assert [c.id for c in candidates] == ["a1", "b2", "c3"]
        assert max_skills == 2

    def test_max_skills_defaults_to_five(self, client, stub_ranker):
        body = _request()
        del body["max_skills"]

        resp = client.post("/rank/skills", json=body)

        assert resp.status_code == 200, resp.text
        assert stub_ranker.calls[0][2] == 5

    def test_the_model_may_keep_nothing(self, client, stub_ranker):
        stub_ranker.answer = lambda candidates, _: ExperienceSkillsResult(
            ranked_ids=[], discarded_ids=[c.id for c in candidates], raw_response="{}"
        )

        resp = client.post("/rank/skills", json=_request())

        assert resp.status_code == 200, resp.text
        assert resp.json()["ranked_ids"] == []

    def test_rejects_an_empty_candidate_list(self, client):
        resp = client.post("/rank/skills", json=_request(candidates=[]))

        assert resp.status_code == 422
        assert "body.candidates" in resp.json()["detail"]

    def test_rejects_duplicate_candidate_ids(self, client):
        candidates = [{"id": "a1", "label": "x"}, {"id": "a1", "label": "y"}]

        resp = client.post("/rank/skills", json=_request(candidates=candidates))

        assert resp.status_code == 422
        assert "unique" in resp.json()["detail"]

    def test_rejects_a_missing_title(self, client):
        resp = client.post("/rank/skills", json=_request(experience={"responsibilities": []}))

        assert resp.status_code == 422
        assert "body.experience.title" in resp.json()["detail"]

    @pytest.mark.parametrize(
        ("error", "detail"),
        [
            (LlmInvocationError("quota"), "LLM invocation error: LlmInvocationError"),
            (LlmParseError("bad json"), "LLM response parse error: LlmParseError"),
            (RuntimeError("boom"), "Internal server error: RuntimeError"),
        ],
    )
    def test_failures_are_500s_with_class_name_only(self, client, stub_ranker, error, detail):
        def _boom(*_):
            raise error

        stub_ranker.answer = _boom

        resp = client.post("/rank/skills", json=_request())

        assert resp.status_code == 500
        assert resp.json() == {"detail": detail}
        assert str(error) not in resp.text
