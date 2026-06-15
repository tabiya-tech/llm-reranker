"""Integration test for ``app.pipeline.process_single_job``.

We exercise the full per-job pipeline (LLM1 -> LLM3 -> LLM2 -> document
assembly) with the three Gemini-facing functions monkeypatched at the
``app.pipeline`` module level (so the pipeline sees stubs without needing
network access). Real taxonomy/CSV resources from ``resources/`` are used to
ensure the wiring around the LLM calls — taxonomy resolution, occupation
mapping, ranked-document construction — is genuinely tested end to end.
"""

from __future__ import annotations

from typing import Any

import pytest

from app import pipeline as pipeline_module
from app.pipeline import load_taxonomy_context, process_single_job
from app.schemas import AttributeEstimate, JobAttributesResult
from utils.classifier_linked_entities import MappedEscoSkills


def _classified_doc() -> dict[str, Any]:
    """A minimal but realistic classifier document.

    Includes one ESCO-linked skill so LLM1 fires, no occupation entities so
    LLM3 stays a no-op (occupation→skills menu is empty), and enough text to
    populate the description fields LLM2 would consume.
    """
    return {
        "_id": "src-doc-1",
        "job_id": "jid-1",
        "job_fingerprint": "fp-1",
        "title": "Software Engineer",
        "employer": "ACME",
        "category": "Tech",
        "description": "Build and maintain backend services in Python.",
        "requirements": "3 years of experience.",
        "city": "Lusaka",
        "classification": {
            "entities": [
                {
                    "entity_type": "skill",
                    "linked_entities": [
                        {
                            "taxonomy": "esco",
                            "label": "Python",
                            "uri": (
                                "http://data.europa.eu/esco/skill/"
                                "5247a6ce-6ed0-4fb3-b121-758921dcc44e"
                            ),
                            "similarity_score": 0.95,
                        }
                    ],
                }
            ]
        },
    }


def _stub_run_skill_ranker_for_job(*args, **kwargs):
    """Pretend LLM1 ranked the single classifier candidate as essential."""
    cands = kwargs.get("esco_candidates") or []
    essential_uris = [c.uri for c in cands]
    essential_uuids = [c.esco_uuid or "" for c in cands]
    mapped = MappedEscoSkills(
        essential_uris=essential_uris,
        optional_uris=[],
        essential_skills_uuids=essential_uuids,
        optional_skills_uuids=[],
    )
    raw = '{"essential_skills": ["Python"], "optional_skills": []}'
    return ["Python"], [], mapped, raw


def _stub_run_job_attributes_for_job(*args, **kwargs):
    """Pretend LLM2 produced one attribute estimate."""
    return JobAttributesResult(
        attribute_estimates=(
            AttributeEstimate(
                attribute_name="earnings_per_month",
                selected_level_id="earn_15k",
                confidence="medium",
                reasoning="stub",
            ),
        ),
        summary="stubbed",
        attributes={"earnings_per_month": "earn_15k"},
        work_activity_ids=(),
        raw_response="{}",
    )


def _stub_run_occupation_skills_relevance_for_job(*args, **kwargs):
    """LLM3 is not expected to fire (no occupation entities); fail loudly if it does."""
    raise AssertionError("LLM3 should not be invoked when occupation_to_skills_nested is empty")


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    monkeypatch.setattr(pipeline_module, "run_skill_ranker_for_job", _stub_run_skill_ranker_for_job)
    monkeypatch.setattr(
        pipeline_module, "run_job_attributes_for_job", _stub_run_job_attributes_for_job
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_occupation_skills_relevance_for_job",
        _stub_run_occupation_skills_relevance_for_job,
    )
    return monkeypatch


class TestProcessSingleJob:
    def test_full_pipeline_with_stubbed_llms(self, make_settings, stubbed_pipeline):
        s = make_settings(GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya")
        taxonomy = load_taxonomy_context(s)

        result = process_single_job(_classified_doc(), taxonomy, settings=s)

        assert result.ranked_document is not None
        # LLM1 + LLM2 ran; LLM3 did not (no occupation entities).
        assert result.ran_llm1 is True
        assert result.ran_llm2 is True
        assert result.ran_llm3 is False
        assert "llm1_s" in result.timings
        assert "llm2_s" in result.timings
        assert "total_s" in result.timings
        # Ranked document carries the ESCO skill from LLM1.
        ranked = result.ranked_document
        ess = ranked["llm_classified_skills"]["essential"]
        assert len(ess) == 1
        assert ess[0]["uri"].startswith("http://data.europa.eu/esco/skill/")
        # Top-level identifiers and metadata are populated from the source doc.
        assert ranked["job_id"] == "jid-1"
        assert ranked["job_fingerprint"] == "fp-1"
        assert ranked["source_classified_id"] == "src-doc-1"
        assert ranked["classifier_metadata"]["title"] == "Software Engineer"
        assert ranked["llm_job_attributes"]["summary"] == "stubbed"

    def test_overlap_llm12_path(self, make_settings, stubbed_pipeline):
        # When RERANKER_OVERLAP_LLM1_LLM2=1, run_llm1_llm2_phase runs them
        # concurrently. The result must be identical and both flags True.
        s = make_settings(
            GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya", RERANKER_OVERLAP_LLM1_LLM2="1"
        )
        taxonomy = load_taxonomy_context(s)

        result = process_single_job(_classified_doc(), taxonomy, settings=s)

        assert result.ran_llm1 is True
        assert result.ran_llm2 is True
        assert result.ranked_document is not None

    def test_no_skill_candidates_skips_llm1(self, make_settings, monkeypatch):
        # Document with no ESCO-linked skills -> LLM1 should not fire,
        # LLM2 still runs.
        s = make_settings(GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya")

        called = {"llm1": 0, "llm2": 0}

        def _no_llm1(*a, **kw):
            called["llm1"] += 1
            raise AssertionError("LLM1 should not be invoked when no ESCO skills exist")

        def _llm2(*a, **kw):
            called["llm2"] += 1
            return _stub_run_job_attributes_for_job()

        monkeypatch.setattr(pipeline_module, "run_skill_ranker_for_job", _no_llm1)
        monkeypatch.setattr(pipeline_module, "run_job_attributes_for_job", _llm2)
        monkeypatch.setattr(
            pipeline_module,
            "run_occupation_skills_relevance_for_job",
            _stub_run_occupation_skills_relevance_for_job,
        )

        taxonomy = load_taxonomy_context(s)
        doc = _classified_doc()
        # Strip skills entirely
        doc["classification"]["entities"] = []
        result = process_single_job(doc, taxonomy, settings=s)

        assert called["llm1"] == 0
        assert called["llm2"] == 1
        assert result.ran_llm1 is False
        assert result.ran_llm2 is True
        assert result.ran_llm3 is False
        assert result.ranked_document is not None

    def test_llm3_disabled_when_env_off(self, make_settings, stubbed_pipeline):
        s = make_settings(GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya", RERANKER_ENABLE_LLM3="0")
        taxonomy = load_taxonomy_context(s)
        result = process_single_job(_classified_doc(), taxonomy, settings=s)
        assert result.ran_llm3 is False
