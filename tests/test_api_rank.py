"""``POST /rank`` end to end, with the three LLM calls stubbed at ``app.pipeline``.

Real taxonomy resources are loaded by the app's lifespan, so request validation, Extended
JSON unwrapping, the shared pipeline and the typed response are all exercised for real.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient
from pydantic import BaseModel

import api
from app import pipeline as pipeline_module
from app.api_models import RankedJobDocument, RankRequest
from app.pipeline import load_taxonomy_context, process_single_job
from tests.test_pipeline_integration import (
    _classified_doc,
    _occupation_menu,
    _relevance_result,
    _stub_run_job_attributes_for_job,
    _stub_run_occupation_skills_relevance_for_job,
    _stub_run_skill_ranker_for_job,
)


def _extended_json_doc() -> dict[str, Any]:
    """``_classified_doc`` as Compass would export it."""
    doc = _classified_doc()
    doc["_id"] = {"$oid": "6650c1f2a1b2c3d4e5f60718"}
    doc["classification"]["entities"][0]["linked_entities"][0]["similarity_score"] = {
        "$numberDouble": "0.95"
    }
    return doc


@pytest.fixture
def client(make_settings, monkeypatch):
    make_settings(GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya")
    monkeypatch.setattr(pipeline_module, "run_skill_ranker_for_job", _stub_run_skill_ranker_for_job)
    monkeypatch.setattr(
        pipeline_module, "run_job_attributes_for_job", _stub_run_job_attributes_for_job
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_occupation_skills_relevance_for_job",
        _stub_run_occupation_skills_relevance_for_job,
    )
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
        assert resp.json()["detail"][0]["loc"][:3] == ["body", "classification", "entities"]

    def test_rejects_a_non_object_body(self, client):
        assert client.post("/rank", json=["not", "a", "document"]).status_code == 422

    def test_pipeline_failure_is_a_500_with_detail(self, client, monkeypatch):
        def _boom(*a, **kw):
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(pipeline_module, "run_job_attributes_for_job", _boom)

        resp = client.post("/rank", json=_classified_doc())

        assert resp.status_code == 500
        assert resp.json() == {"detail": "model unavailable"}


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
    def test_round_trips_pipeline_output(self, make_settings, monkeypatch, mode):
        s = make_settings(
            GEMINI_API_KEY="dummy",
            TARGET_COUNTRY="kenya",
            RERANKER_ENABLE_LLM3="1",
            RERANKER_LLM3_MODE=mode,
        )
        monkeypatch.setattr(
            pipeline_module, "run_skill_ranker_for_job", _stub_run_skill_ranker_for_job
        )
        monkeypatch.setattr(
            pipeline_module, "run_job_attributes_for_job", _stub_run_job_attributes_for_job
        )
        monkeypatch.setattr(
            pipeline_module,
            "run_occupation_skills_relevance_for_job",
            lambda *a, **kw: _relevance_result(),
        )
        monkeypatch.setattr(
            pipeline_module,
            "resolve_occupation_taxonomy_for_classified_job",
            lambda *a, **kw: ([], None, _occupation_menu()),
        )
        doc = _classified_doc()
        doc["province"] = "Nairobi County"  # scraper enrichment: an extra metadata key

        ranked = process_single_job(doc, load_taxonomy_context(s), settings=s).ranked_document
        assert ranked is not None
        model = RankedJobDocument.model_validate(ranked)

        assert model.model_dump(mode="json", exclude_unset=True) == jsonable_encoder(ranked)
        # Only the documented open-ended block may carry undeclared keys.
        assert _extra_keys(model) == [".classifier_metadata.province"]


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
        assert op["parameters"][0]["name"] == "language"
        assert op["parameters"][0]["description"]
