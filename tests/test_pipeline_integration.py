"""Integration test for ``app.pipeline.process_single_job``.

We exercise the full per-job pipeline (taxonomy -> LLM3 -> LLM1 + LLM2 -> document
assembly) with the three model-facing functions monkeypatched at the
``app.pipeline`` module level (so the pipeline sees stubs without needing
network access). Real taxonomy/CSV resources from ``resources/`` are used to
ensure the wiring around the LLM calls — taxonomy resolution, occupation
mapping, ranked-document construction — is genuinely tested end to end.

``TestPipelineOverAConfiguredModel`` goes one level deeper: instead of stubbing the
``run_*_for_job`` helpers it leaves them real and lets configuration select a fake
:class:`~app.llm.base.LlmClient`. That covers the whole chain — env vars, LlmConfig,
the provider factory, prompt assembly, response parsing, document assembly — and is
the guarantee that swapping the backing model needs no change to pipeline code.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pytest
from pymongo.operations import UpdateOne

from app import batch_rank_one as batch_module
from app import pipeline as pipeline_module
from app.llm import LlmClient
from app.pipeline import Llm3Outcome, load_taxonomy_context, process_single_job
from app.schemas import (
    AttributeEstimate,
    JobAttributesResult,
    OccupationSkillsRelevanceDecision,
    OccupationSkillsRelevanceResult,
)
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
        assert result.llm3_outcome is Llm3Outcome.DISABLED


def _occupation_menu() -> list[dict[str, Any]]:
    """One occupation block with two taxonomy skills of different signalling strength."""
    return [
        {
            "occupation_label": "Welder",
            "occupation_esco_uuid": "33333333-3333-3333-3333-333333333333",
            "occupation_uri": (
                "http://data.europa.eu/esco/occupation/33333333-3333-3333-3333-333333333333"
            ),
            "tabiya_occupation_id": "occ-1",
            "skill_counts": {"total": 2, "essential": 1, "optional": 1, "unspecified": 0},
            "skills": [
                {
                    "skill_id": "11111111-1111-1111-1111-111111111111",
                    "tabiya_skill_id": "tskill-1",
                    "label": "Weld metal",
                    "uri": "http://data.europa.eu/esco/skill/11111111-1111-1111-1111-111111111111",
                    "relation_type": "essential",
                    "signalling_label": "high",
                    "signalling_value": "0.9",
                },
                {
                    "skill_id": "22222222-2222-2222-2222-222222222222",
                    "tabiya_skill_id": "tskill-2",
                    "label": "Paint surfaces",
                    "uri": "http://data.europa.eu/esco/skill/22222222-2222-2222-2222-222222222222",
                    "relation_type": "optional",
                    "signalling_label": "low",
                    "signalling_value": "0.1",
                },
            ],
        }
    ]


@pytest.fixture
def llm1_seen(stubbed_pipeline):
    """Records LLM1's candidate labels; the taxonomy step returns :func:`_occupation_menu`."""
    seen: list[list[str]] = []

    def _recording_llm1(*args, **kwargs):
        seen.append([c.label for c in kwargs.get("esco_candidates") or []])
        return _stub_run_skill_ranker_for_job(*args, **kwargs)

    stubbed_pipeline.setattr(pipeline_module, "run_skill_ranker_for_job", _recording_llm1)
    stubbed_pipeline.setattr(
        pipeline_module,
        "resolve_occupation_taxonomy_for_classified_job",
        lambda *a, **kw: ([], None, _occupation_menu()),
    )
    return seen


def _set_body(op: UpdateOne) -> dict[str, Any]:
    """The ``$set`` document of a queued upsert."""
    update = op._doc
    assert isinstance(update, dict)
    return update["$set"]


def _batch_args() -> argparse.Namespace:
    return argparse.Namespace(
        log_llm1_prompt=False, log_llm2_prompt=False, log_llm3_prompt=False, verbose=False
    )


class TestLlm3MenuFoldedIntoLlm1:
    """``RERANKER_LLM3_MODE=fold`` (default): the LLM3 menu feeds LLM1 and LLM3 is not called."""

    def test_capped_menu_is_added_to_llm1_candidates(self, make_settings, llm1_seen):
        s = make_settings(
            GEMINI_API_KEY="dummy",
            TARGET_COUNTRY="kenya",
            RERANKER_ENABLE_LLM3="1",
            RERANKER_LLM3_MAX_MENU_SKILLS="1",
        )
        result = process_single_job(_classified_doc(), load_taxonomy_context(s), settings=s)

        # Classifier skill + the single strongest menu skill; LLM3 stub (raises) never ran.
        assert llm1_seen == [["Python", "Weld metal"]]
        assert result.ran_llm3 is False
        assert result.llm3_outcome is Llm3Outcome.FOLDED
        assert result.stats.llm3_added_llm1_candidates == 1
        assert "llm3_s" not in result.timings
        assert result.ranked_document is not None
        llm_skills = result.ranked_document["llm_classified_skills"]
        assert [r["label"] for r in llm_skills["essential"]] == ["Python", "Weld metal"]
        assert llm_skills["discarded"] == []

    def test_menu_not_added_when_llm3_disabled(self, make_settings, llm1_seen):
        s = make_settings(GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya", RERANKER_ENABLE_LLM3="0")
        result = process_single_job(_classified_doc(), load_taxonomy_context(s), settings=s)

        assert llm1_seen == [["Python"]]
        assert result.ran_llm3 is False
        assert result.llm3_outcome is Llm3Outcome.DISABLED

    def test_llm1_picked_menu_skills_reach_the_job_embedding(
        self, make_settings, llm1_seen, monkeypatch
    ):
        """Batch path (the one that embeds): LLM1-picked menu skills are in the embedded text."""
        s = make_settings(
            GEMINI_API_KEY="dummy",
            TARGET_COUNTRY="kenya",
            RERANKER_ENABLE_LLM3="1",
            RERANKER_LLM3_MAX_MENU_SKILLS="1",
            RERANKER_ENABLE_JOB_EMBEDDING="1",
            RERANKER_CONCAT_WHITENING_ENABLED="0",
        )
        embedded: list[str] = []

        def _embed(texts, **kwargs):
            embedded.extend(texts)
            return np.ones((len(texts), s.reranker_job_embedding_dim))

        monkeypatch.setattr("utils.job_embedding.embed_text_list", _embed)
        r = batch_module.rank_one_classified_document(
            _classified_doc(),
            job_label="1/1",
            args=_batch_args(),
            settings=s,
            taxonomy=load_taxonomy_context(s),
        )

        assert llm1_seen == [["Python", "Weld metal"]]
        assert r.llm3_inc == 0
        assert r.update_op is not None
        assert embedded == ["Python | Weld metal"]


def _relevance_result() -> OccupationSkillsRelevanceResult:
    """LLM3 keeps the essential menu skill and rejects the optional one."""
    return OccupationSkillsRelevanceResult(
        decisions=(
            OccupationSkillsRelevanceDecision(
                tabiya_occupation_id="occ-1",
                skill_id="11111111-1111-1111-1111-111111111111",
                is_relevant=True,
            ),
            OccupationSkillsRelevanceDecision(
                tabiya_occupation_id="occ-1",
                skill_id="22222222-2222-2222-2222-222222222222",
                is_relevant=False,
            ),
        ),
        raw_response="{}",
    )


class TestLlm3StandaloneCall:
    """``RERANKER_LLM3_MODE=call``: LLM3 judges the menu; LLM1 sees only classifier skills."""

    @pytest.fixture
    def llm3_menus(self, llm1_seen, monkeypatch):
        menus: list[list[str]] = []

        def _llm3(ctx, items, **kwargs):
            menus.append([sk["label"] for block in items for sk in block["skills"]])
            return _relevance_result()

        monkeypatch.setattr(pipeline_module, "run_occupation_skills_relevance_for_job", _llm3)
        return menus

    def _settings(self, make_settings, **extra: str):
        return make_settings(
            GEMINI_API_KEY="dummy",
            TARGET_COUNTRY="kenya",
            RERANKER_ENABLE_LLM3="1",
            RERANKER_LLM3_MODE="call",
            **extra,
        )

    def test_call_tags_relevance_and_reports_it(self, make_settings, llm1_seen, llm3_menus):
        s = self._settings(make_settings)
        result = process_single_job(_classified_doc(), load_taxonomy_context(s), settings=s)

        assert llm3_menus == [["Weld metal", "Paint surfaces"]]
        assert llm1_seen == [["Python"]]
        assert result.ran_llm3 is True
        assert result.llm3_outcome is Llm3Outcome.CALLED
        assert result.stats.llm3_decisions == 2
        assert "llm3_s" in result.timings
        assert result.ranked_document is not None
        block = result.ranked_document["occupation_to_skills"]["items"][0]
        assert [sk["llm_is_relevant"] for sk in block["skills"]] == [True, False]
        assert block["skill_counts"]["relevant_total"] == 1

    def test_batch_counts_the_call(self, make_settings, llm1_seen, llm3_menus):
        s = self._settings(make_settings, RERANKER_ENABLE_JOB_EMBEDDING="0")
        r = batch_module.rank_one_classified_document(
            _classified_doc(),
            job_label="1/1",
            args=_batch_args(),
            settings=s,
            taxonomy=load_taxonomy_context(s),
        )

        assert (r.llm1_inc, r.llm2_inc, r.llm3_inc) == (1, 1, 1)
        assert r.update_op is not None
        block = _set_body(r.update_op)["occupation_to_skills"]["items"][0]
        assert [sk["llm_is_relevant"] for sk in block["skills"]] == [True, False]

    def test_menu_filtered_out_makes_no_call(self, make_settings, llm1_seen, llm3_menus):
        # The classifier intersect keeps only menu skills the classifier also linked; the
        # job's one classifier skill (Python) is on no occupation, so the menu ends up empty.
        s = self._settings(make_settings, RERANKER_LLM3_CLASSIFIER_INTERSECT="1")
        result = process_single_job(_classified_doc(), load_taxonomy_context(s), settings=s)

        assert llm3_menus == []
        assert result.ran_llm3 is False
        assert result.llm3_outcome is Llm3Outcome.MENU_FILTERED_OUT


class TestBatchRunsTheSharedPipeline:
    """The batch adds logging and the upsert; the ranking itself is ``process_single_job``."""

    def test_batch_upserts_exactly_what_the_pipeline_ranks(
        self, make_settings, stubbed_pipeline, monkeypatch
    ):
        s = make_settings(
            GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya", RERANKER_ENABLE_JOB_EMBEDDING="0"
        )
        taxonomy = load_taxonomy_context(s)
        calls: list[dict[str, Any]] = []
        real = batch_module.process_single_job

        def _spy(doc, tx, **kwargs):
            calls.append(kwargs)
            result = real(doc, tx, **kwargs)
            calls[-1]["ranked"] = result.ranked_document
            return result

        monkeypatch.setattr(batch_module, "process_single_job", _spy)
        r = batch_module.rank_one_classified_document(
            _classified_doc(), job_label="1/1", args=_batch_args(), settings=s, taxonomy=taxonomy
        )

        assert len(calls) == 1
        assert calls[0]["settings"] is s
        assert r.update_op is not None
        assert _set_body(r.update_op) is calls[0]["ranked"]
        assert r.update_op._filter == {"job_id": "jid-1"}
        assert (r.llm1_inc, r.llm2_inc, r.llm3_inc, r.upsert_increment) == (1, 1, 0, 1)

    def test_prompt_logging_flags_reach_the_pipeline(
        self, make_settings, stubbed_pipeline, monkeypatch
    ):
        s = make_settings(
            GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya", RERANKER_ENABLE_JOB_EMBEDDING="0"
        )
        seen: dict[str, Any] = {}

        def _llm1(*args, **kwargs):
            seen["llm1"] = (kwargs["log_llm1_prompt"], kwargs["log_llm1_job_label"])
            return _stub_run_skill_ranker_for_job(*args, **kwargs)

        monkeypatch.setattr(pipeline_module, "run_skill_ranker_for_job", _llm1)
        args = _batch_args()
        args.log_llm1_prompt = True
        batch_module.rank_one_classified_document(
            _classified_doc(),
            job_label="1/1",
            args=args,
            settings=s,
            taxonomy=load_taxonomy_context(s),
        )

        # Labelled by the job's fingerprint, as before the batch reused the pipeline.
        assert seen["llm1"] == (True, "fp-1")


class ScriptedLlmClient(LlmClient):
    """A whole model backend in twenty lines — the interface really is this small.

    Answers each reranker pass from ``purpose`` alone. Records the prompts it was
    given so a test can assert the pipeline actually went through the client.
    """

    #: (purpose, prompt) for every call, across instances.
    calls: list[tuple[str, str]] = []

    def generate_json(self, prompt: str, *, purpose: str = "") -> str:
        ScriptedLlmClient.calls.append((purpose, prompt))
        if purpose == "skill_ranker":
            return json.dumps({"essential_skills": ["Python"], "optional_skills": []})
        if purpose == "job_attributes":
            return json.dumps(
                {
                    "attributes": [
                        {
                            "attribute_name": "earnings_per_month",
                            "selected_level_id": "earn_15k",
                            "confidence": "medium",
                            "reasoning": "scripted",
                        }
                    ],
                    "summary": "ranked by a configured non-Gemini model",
                    "work_activities": [],
                }
            )
        if purpose == "occupation_skills_relevance":
            return json.dumps({"decisions": []})
        raise AssertionError(f"unexpected purpose {purpose!r}")


@pytest.mark.integration
class TestPipelineOverAConfiguredModel:
    """The pipeline runs unmodified on a model selected only by configuration."""

    def test_ranked_document_is_produced_through_a_configured_client(self, make_settings):
        ScriptedLlmClient.calls = []
        s = make_settings(
            TARGET_COUNTRY="kenya",
            RERANKER_LLM_CLIENT_CLASS="tests.test_pipeline_integration:ScriptedLlmClient",
            RERANKER_LLM_MODEL="scripted-model-v1",
            # No GEMINI_API_KEY: nothing in the ranking path should need one.
        )
        taxonomy = load_taxonomy_context(s)

        result = process_single_job(_classified_doc(), taxonomy, settings=s)

        assert result.ran_llm1 is True
        assert result.ran_llm2 is True
        ranked = result.ranked_document
        assert ranked is not None
        # LLM1's answer survived parsing and label→URI mapping.
        essential = ranked["llm_classified_skills"]["essential"]
        assert [sk["uri"] for sk in essential] == [
            "http://data.europa.eu/esco/skill/5247a6ce-6ed0-4fb3-b121-758921dcc44e"
        ]
        # LLM2's answer landed in the attributes block.
        assert ranked["llm_job_attributes"]["summary"] == "ranked by a configured non-Gemini model"
        assert ranked["llm_job_attributes"]["attributes"]["earnings_per_month"] == "earn_15k"
        # The document records which model produced it.
        meta = ranked["llm_reranker_meta"]
        assert meta["skill_ranker_model"] == "scripted-model-v1"
        assert meta["job_attributes_model"] == "scripted-model-v1"
        assert meta["llm_provider"] == "custom"
        # Both passes went through the configured client, with real assembled prompts.
        purposes = [p for p, _ in ScriptedLlmClient.calls]
        assert sorted(purposes) == ["job_attributes", "skill_ranker"]
        llm1_prompt = next(pr for p, pr in ScriptedLlmClient.calls if p == "skill_ranker")
        assert "Python" in llm1_prompt
