"""``POST /rank`` end to end, with the three LLM steps stubbed.

The app's lifespan builds its ranking services through ``get_job_ranking_services``; the test swaps in
a service whose LLM steps are stubs. Real taxonomy resources are loaded, so request
validation, Extended JSON unwrapping, the shared pipeline and the typed response are all
exercised for real.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient
from pydantic import BaseModel

import api
from app.config import Settings
from app.job_ranking import get_job_ranking_service
from app.job_ranking.routes.types import RankedJobDocument, RankRequest
from app.job_ranking.service import JobRankingService, JobRankingServiceFactory
from app.llm import TracedLlmClient
from app.llm.errors import LlmConfigurationError, LlmInvocationError, LlmParseError
from tests.ranking_fakes import (
    FixedOccupationTaxonomy,
    FixedRankers,
    StubJobAttributes,
    StubOccupationRelevance,
    build_ranking_service,
    occupation_menu,
    relevance_result,
)
from tests.ranking_fakes import classified_doc as _classified_doc


def _extended_json_doc() -> dict[str, Any]:
    """``_classified_doc`` as Compass would export it."""
    doc = _classified_doc()
    doc["_id"] = {"$oid": "6650c1f2a1b2c3d4e5f60718"}
    doc["classification"]["entities"][0]["linked_entities"][0]["similarity_score"] = {
        "$numberDouble": "0.95"
    }
    return doc


def _serve(monkeypatch, ranking: JobRankingService) -> None:
    """Make the app's lifespan serve ``ranking`` for every language."""

    def _build(settings: Settings, llm: TracedLlmClient) -> JobRankingServiceFactory:
        return FixedRankers(ranking)

    monkeypatch.setattr(get_job_ranking_service, "build_job_ranking_service_factory", _build)


@pytest.fixture
def stub_llm2():
    return StubJobAttributes()


@pytest.fixture
def client(make_settings, monkeypatch, stub_llm2):
    s = make_settings(GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya")
    _serve(monkeypatch, build_ranking_service(s, job_attributes=stub_llm2))
    with TestClient(api.app) as c:
        yield c


class TestRank:
    def test_ranks_an_extended_json_document(self, client):
        resp = client.post("/rank", json=_extended_json_doc())

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert set(body) == {
            "ranked_document",
            "timings",
            "ran_llm1",
            "ran_llm2",
            "ran_llm3",
            "llm3_outcome",
        }
        assert (body["ran_llm1"], body["ran_llm2"], body["ran_llm3"]) == (True, True, False)
        # No occupation entities, so the LLM3 step had nothing to work with.
        assert body["llm3_outcome"] == "no_menu"
        assert "llm3_s" not in body["timings"]
        ranked = body["ranked_document"]
        assert ranked["source_classified_id"] == "6650c1f2a1b2c3d4e5f60718"
        assert ranked["classified_skills"][0]["similarity_score"] == 0.95
        assert [s["label"] for s in ranked["llm_classified_skills"]["essential"]] == ["Python"]
        # Keys the pipeline did not write stay absent rather than appearing as null.
        assert "job_embedding" not in ranked
        assert "relevant_total" in ranked["occupation_to_skills"]["meta"]["skill_counts"]

    def test_rejects_a_malformed_classification(self, client):
        doc = _classified_doc()
        doc["classification"] = {"entities": "skill: Python"}

        resp = client.post("/rank", json=doc)

        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert isinstance(detail, str)
        assert "body.classification.entities" in detail

    def test_rejects_a_non_object_body(self, client):
        resp = client.post("/rank", json=["not", "a", "document"])
        assert resp.status_code == 422
        assert isinstance(resp.json()["detail"], str)

    def test_validation_error_detail_never_echoes_the_rejected_input(self, client):
        doc = _classified_doc()
        doc["classification"] = {"entities": "secret job text should not leak"}

        resp = client.post("/rank", json=doc)

        assert resp.status_code == 422
        assert "secret job text should not leak" not in resp.text

    def test_generic_pipeline_failure_is_a_500_with_class_name_only(self, client, stub_llm2):
        def _boom():
            raise RuntimeError("model unavailable")

        stub_llm2.answer = _boom

        resp = client.post("/rank", json=_classified_doc())

        assert resp.status_code == 500
        assert resp.json() == {"detail": "Internal server error: RuntimeError"}
        assert "model unavailable" not in resp.text

    def test_llm_configuration_error_is_a_500_with_distinct_detail(self, client, stub_llm2):
        def _boom():
            raise LlmConfigurationError("missing RERANKER_LLM_API_KEY")

        stub_llm2.answer = _boom

        resp = client.post("/rank", json=_classified_doc())

        assert resp.status_code == 500
        assert resp.json() == {"detail": "LLM configuration error: LlmConfigurationError"}
        assert "missing RERANKER_LLM_API_KEY" not in resp.text

    def test_llm_invocation_error_is_a_500_with_distinct_detail(self, client, stub_llm2):
        def _boom():
            raise LlmInvocationError("Gemini API error (TimeoutError): deadline exceeded")

        stub_llm2.answer = _boom

        resp = client.post("/rank", json=_classified_doc())

        assert resp.status_code == 500
        assert resp.json() == {"detail": "LLM invocation error: LlmInvocationError"}
        assert "deadline exceeded" not in resp.text

    def test_llm_parse_error_is_a_500_with_distinct_detail(self, client, stub_llm2):
        def _boom():
            raise LlmParseError("Job attributes JSON must be an object")

        stub_llm2.answer = _boom

        resp = client.post("/rank", json=_classified_doc())

        assert resp.status_code == 500
        assert resp.json() == {"detail": "LLM response parse error: LlmParseError"}
        assert "Job attributes JSON must be an object" not in resp.text


class TestRankRequest:
    def test_unknown_fields_pass_through_unchanged(self):
        doc = _classified_doc()
        doc["metadata"] = {"salary": {"min": 1, "max": 2}}
        doc["classification"]["model_version"] = "v7"

        assert RankRequest.model_validate(doc).to_document() == doc

    def test_document_keeps_the_mongo_id_key(self):
        out = RankRequest.model_validate({"_id": {"$oid": "abc"}}).to_document()
        assert out == {"_id": "abc"}


def _extra_keys(model: BaseModel, path: str = "") -> list[str]:
    """Keys a response model accepted as extras, anywhere in the tree."""
    found = [f"{path}.{k}" for k in (model.__pydantic_extra__ or {})]
    for name in type(model).model_fields:
        value = getattr(model, name)
        children = value if isinstance(value, list) else [value]
        for child in children:
            if isinstance(child, BaseModel):
                found += _extra_keys(child, f"{path}.{name}")
    return found


class TestRankedJobDocumentModel:
    """The response model must describe exactly what ``build_ranked_job_document`` writes."""

    @pytest.mark.parametrize("mode", ["fold", "call"])
    def test_round_trips_pipeline_output(self, make_settings, mode):
        s = make_settings(
            GEMINI_API_KEY="dummy",
            TARGET_COUNTRY="kenya",
            RERANKER_ENABLE_LLM3="1",
            RERANKER_LLM3_MODE=mode,
        )
        ranking = build_ranking_service(
            s,
            taxonomy_step=FixedOccupationTaxonomy(occupation_menu()),
            occupation_relevance=StubOccupationRelevance(relevance_result),
        )
        doc = _classified_doc()
        doc["province"] = "Nairobi County"  # scraper enrichment: a named ClassifierMetadata field

        ranked = ranking.rank(doc).ranked_document
        assert ranked is not None
        # The pipeline already returns a validated RankedJobDocument; re-validating here
        # confirms it round-trips cleanly (no lossy mode="json" coercions).
        model = RankedJobDocument.model_validate(ranked.model_dump(mode="python"))

        assert model.model_dump(mode="json", exclude_unset=True) == jsonable_encoder(ranked)
        assert model.classifier_metadata.province == "Nairobi County"
        # Every field is declared and strict (extra="forbid") — nothing carries undeclared keys.
        assert _extra_keys(model) == []


class TestOpenApi:
    def test_every_schema_property_is_described(self):
        spec = api.app.openapi()
        missing = [
            f"{name}.{prop}"
            for name, schema in spec["components"]["schemas"].items()
            for prop, d in (schema.get("properties") or {}).items()
            if "description" not in d
        ]
        assert missing == []

    def test_rank_documents_request_example_and_errors(self):
        op = api.app.openapi()["paths"]["/rank"]["post"]
        schemas = api.app.openapi()["components"]["schemas"]

        ref = op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        assert ref.endswith("/RankRequest")
        assert schemas["RankRequest"]["examples"]
        assert set(op["responses"]) >= {"200", "422", "500", "503"}
        for status in ("422", "500", "503"):
            ref = op["responses"][status]["content"]["application/json"]["schema"]["$ref"]
            assert ref.endswith("/ErrorResponse")
        assert op["parameters"][0]["name"] == "language"
        assert op["parameters"][0]["description"]


class TestLanguageSelection:
    """The route picks the language; the factory hands out that language's service."""

    @pytest.fixture
    def seen(self, client):
        languages: list[str | None] = []
        services = get_job_ranking_service.get_job_ranking_services()
        real = services.for_language

        def _spy(language):
            languages.append(language)
            return real(language)

        services.for_language = _spy  # type: ignore[method-assign]
        return languages

    def test_no_language_means_the_service_default(self, client, seen):
        assert client.post("/rank", json=_classified_doc()).status_code == 200
        assert seen == [None]

    def test_document_field_is_normalised(self, client, seen):
        doc = _classified_doc()
        doc["language"] = "AR-es"
        assert client.post("/rank", json=doc).status_code == 200
        assert seen == ["es"]

    def test_query_parameter_wins_over_the_document(self, client, seen):
        doc = _classified_doc()
        doc["language"] = "es"
        assert client.post("/rank?language=english", json=doc).status_code == 200
        assert seen == ["en"]
