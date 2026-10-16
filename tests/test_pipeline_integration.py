"""Integration test for :class:`app.job_ranking.service.JobRankingService`.

We exercise the full per-job pipeline (taxonomy -> LLM3 -> LLM1 + LLM2 -> document
assembly) with the three model-facing steps replaced by the stubs in
``tests/ranking_fakes.py`` (so the pipeline runs without network access). Real
taxonomy/CSV resources from ``resources/`` are used to ensure the wiring around the LLM
calls — taxonomy resolution, occupation mapping, ranked-document construction — is
genuinely tested end to end.

``TestPipelineOverAConfiguredModel`` goes one level deeper: instead of stubbing the steps
it builds the service through ``app/job_ranking/get_job_ranking_service.py`` and lets configuration select a fake
:class:`~app.llm.base.LlmClient`. That covers the whole chain — env vars, LlmConfig,
the provider factory, prompt assembly, response parsing, document assembly — and is
the guarantee that swapping the backing model needs no change to pipeline code.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from app.batch.job_ranker import BatchJobOptions, BatchJobRanker
from app.config import Settings
from app.job_context import build_job_description_context
from app.job_embedding.client import EmbeddingClient
from app.job_embedding.service import JobEmbeddingService
from app.job_embedding.whitening_repository import WhiteningArtifactRepository
from app.job_ranking.get_job_ranking_service import build_job_ranking_service_factory
from app.job_ranking.occupation_skills.parser import OccupationSkillsRelevanceResponseParser
from app.job_ranking.occupation_skills.prompt import OccupationSkillsRelevancePromptBuilder
from app.job_ranking.occupation_skills.relevance_step import OccupationSkillsRelevanceStep
from app.job_ranking.prompts.repository import PromptRepository
from app.job_ranking.service import JobRankingService
from app.job_ranking.types import NO_PROMPT_LOGGING, Llm3Outcome, PromptLogging
from app.llm import LlmClient, build_traced_llm_client
from app.ranked_jobs.types import RankedJobKey
from tests.ranking_fakes import (
    FixedOccupationTaxonomy,
    StubJobAttributes,
    StubOccupationRelevance,
    StubSkillRanker,
    build_ranking_service,
    classified_doc,
    occupation_menu,
    relevance_result,
)


class RecordingEmbeddingClient(EmbeddingClient):
    """Embeds every text as a vector of ones and records the texts."""

    def __init__(self, dim: int):
        self.model = "fake-embedding"
        self.dim = dim
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.texts.extend(texts)
        return np.ones((len(texts), self.dim), dtype=np.float32)


def _batch_ranker(
    settings: Settings,
    ranking: JobRankingService,
    *,
    embedding_client: EmbeddingClient | None = None,
    prompt_logging: PromptLogging = NO_PROMPT_LOGGING,
) -> BatchJobRanker:
    return BatchJobRanker(
        settings=settings,
        ranking=ranking,
        embedding=JobEmbeddingService(settings, embedding_client, WhiteningArtifactRepository()),
        options=BatchJobOptions(prompt_logging=prompt_logging),
    )


class TestJobRankingService:
    def test_full_pipeline_with_stubbed_llms(self, make_settings):
        s = make_settings(GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya")

        result = build_ranking_service(s).rank(classified_doc())

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

    def test_overlap_llm12_path(self, make_settings):
        # When RERANKER_OVERLAP_LLM1_LLM2=1, LLM1 and LLM2 run concurrently.
        # The result must be identical and both flags True.
        s = make_settings(
            GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya", RERANKER_OVERLAP_LLM1_LLM2="1"
        )

        result = build_ranking_service(s).rank(classified_doc())

        assert result.ran_llm1 is True
        assert result.ran_llm2 is True
        assert result.ranked_document is not None

    def test_no_skill_candidates_skips_llm1(self, make_settings):
        # Document with no ESCO-linked skills -> LLM1 should not fire, LLM2 still runs.
        s = make_settings(GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya")
        llm1, llm2 = StubSkillRanker(), StubJobAttributes()
        doc = classified_doc()
        doc["classification"]["entities"] = []

        result = build_ranking_service(s, skill_ranker=llm1, job_attributes=llm2).rank(doc)

        assert llm1.candidates == []
        assert llm2.calls == 1
        assert result.ran_llm1 is False
        assert result.ran_llm2 is True
        assert result.ran_llm3 is False
        assert result.ranked_document is not None

    def test_llm3_disabled_when_env_off(self, make_settings):
        s = make_settings(GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya", RERANKER_ENABLE_LLM3="0")
        result = build_ranking_service(s).rank(classified_doc())
        assert result.ran_llm3 is False
        assert result.llm3_outcome is Llm3Outcome.DISABLED


def _with_menu(settings: Settings, **steps: Any) -> JobRankingService:
    """The taxonomy step returns :func:`occupation_menu` for every job."""
    return build_ranking_service(
        settings, taxonomy_step=FixedOccupationTaxonomy(occupation_menu()), **steps
    )


class TestLlm3MenuFoldedIntoLlm1:
    """``RERANKER_LLM3_MODE=fold`` (default): the LLM3 menu feeds LLM1 and LLM3 is not called."""

    def test_capped_menu_is_added_to_llm1_candidates(self, make_settings):
        s = make_settings(
            GEMINI_API_KEY="dummy",
            TARGET_COUNTRY="kenya",
            RERANKER_ENABLE_LLM3="1",
            RERANKER_LLM3_MAX_MENU_SKILLS="1",
        )
        llm1 = StubSkillRanker()

        result = _with_menu(s, skill_ranker=llm1).rank(classified_doc())

        # Classifier skill + the single strongest menu skill; LLM3 stub (raises) never ran.
        assert llm1.candidates == [["Python", "Weld metal"]]
        assert result.ran_llm3 is False
        assert result.llm3_outcome is Llm3Outcome.FOLDED
        assert result.stats.llm3_added_llm1_candidates == 1
        assert "llm3_s" not in result.timings
        assert result.ranked_document is not None
        llm_skills = result.ranked_document["llm_classified_skills"]
        assert [r["label"] for r in llm_skills["essential"]] == ["Python", "Weld metal"]
        assert llm_skills["discarded"] == []

    def test_menu_not_added_when_llm3_disabled(self, make_settings):
        s = make_settings(GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya", RERANKER_ENABLE_LLM3="0")
        llm1 = StubSkillRanker()

        result = _with_menu(s, skill_ranker=llm1).rank(classified_doc())

        assert llm1.candidates == [["Python"]]
        assert result.ran_llm3 is False
        assert result.llm3_outcome is Llm3Outcome.DISABLED

    def test_llm1_picked_menu_skills_reach_the_job_embedding(self, make_settings):
        """Batch path (the one that embeds): LLM1-picked menu skills are in the embedded text."""
        s = make_settings(
            GEMINI_API_KEY="dummy",
            TARGET_COUNTRY="kenya",
            RERANKER_ENABLE_LLM3="1",
            RERANKER_LLM3_MAX_MENU_SKILLS="1",
            RERANKER_ENABLE_JOB_EMBEDDING="1",
            RERANKER_CONCAT_WHITENING_ENABLED="0",
        )
        llm1 = StubSkillRanker()
        embedder = RecordingEmbeddingClient(s.reranker_job_embedding_dim)

        r = _batch_ranker(s, _with_menu(s, skill_ranker=llm1), embedding_client=embedder).rank(
            classified_doc(), job_label="1/1"
        )

        assert llm1.candidates == [["Python", "Weld metal"]]
        assert r.llm3_inc == 0
        assert r.upsert is not None
        assert embedder.texts == ["Python | Weld metal"]


class TestLlm3StandaloneCall:
    """``RERANKER_LLM3_MODE=call``: LLM3 judges the menu; LLM1 sees only classifier skills."""

    def _settings(self, make_settings, **extra: str) -> Settings:
        return make_settings(
            GEMINI_API_KEY="dummy",
            TARGET_COUNTRY="kenya",
            RERANKER_ENABLE_LLM3="1",
            RERANKER_LLM3_MODE="call",
            **extra,
        )

    def test_call_tags_relevance_and_reports_it(self, make_settings):
        s = self._settings(make_settings)
        llm1, llm3 = StubSkillRanker(), StubOccupationRelevance(relevance_result)

        result = _with_menu(s, skill_ranker=llm1, occupation_relevance=llm3).rank(classified_doc())

        assert llm3.menus == [["Weld metal", "Paint surfaces"]]
        assert llm1.candidates == [["Python"]]
        assert result.ran_llm3 is True
        assert result.llm3_outcome is Llm3Outcome.CALLED
        assert result.stats.llm3_decisions == 2
        assert "llm3_s" in result.timings
        assert result.ranked_document is not None
        block = result.ranked_document["occupation_to_skills"]["items"][0]
        assert [sk["llm_is_relevant"] for sk in block["skills"]] == [True, False]
        assert block["skill_counts"]["relevant_total"] == 1

    def test_batch_counts_the_call(self, make_settings):
        s = self._settings(make_settings, RERANKER_ENABLE_JOB_EMBEDDING="0")
        ranking = _with_menu(s, occupation_relevance=StubOccupationRelevance(relevance_result))

        r = _batch_ranker(s, ranking).rank(classified_doc(), job_label="1/1")

        assert (r.llm1_inc, r.llm2_inc, r.llm3_inc) == (1, 1, 1)
        assert r.upsert is not None
        block = r.upsert.document["occupation_to_skills"]["items"][0]
        assert [sk["llm_is_relevant"] for sk in block["skills"]] == [True, False]

    def test_menu_filtered_out_makes_no_call(self, make_settings):
        # The classifier intersect keeps only menu skills the classifier also linked; the
        # job's one classifier skill (Python) is on no occupation, so the menu ends up empty.
        s = self._settings(make_settings, RERANKER_LLM3_CLASSIFIER_INTERSECT="1")
        llm3 = StubOccupationRelevance(relevance_result)

        result = _with_menu(s, occupation_relevance=llm3).rank(classified_doc())

        assert llm3.menus == []
        assert result.ran_llm3 is False
        assert result.llm3_outcome is Llm3Outcome.MENU_FILTERED_OUT


class TestBatchRunsTheSharedPipeline:
    """The batch adds logging, the embedding and the upsert; the ranking is ``JobRankingService``."""

    def test_batch_upserts_exactly_what_the_pipeline_ranks(self, make_settings, monkeypatch):
        s = make_settings(
            GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya", RERANKER_ENABLE_JOB_EMBEDDING="0"
        )
        ranking = build_ranking_service(s)
        ranked: list[dict[str, Any] | None] = []
        real = ranking.rank

        def _spy(doc, **kwargs):
            result = real(doc, **kwargs)
            ranked.append(result.ranked_document)
            return result

        monkeypatch.setattr(ranking, "rank", _spy)
        r = _batch_ranker(s, ranking).rank(classified_doc(), job_label="1/1")

        assert len(ranked) == 1
        assert r.upsert is not None
        assert r.upsert.document is ranked[0]
        assert (r.upsert.key, r.upsert.key_value) == (RankedJobKey.JOB_ID, "jid-1")
        assert (r.llm1_inc, r.llm2_inc, r.llm3_inc, r.upsert_increment) == (1, 1, 0, 1)

    def test_prompt_logging_flags_reach_the_steps(self, make_settings):
        s = make_settings(
            GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya", RERANKER_ENABLE_JOB_EMBEDDING="0"
        )
        llm1 = StubSkillRanker()

        _batch_ranker(
            s,
            build_ranking_service(s, skill_ranker=llm1),
            prompt_logging=PromptLogging(llm1=True),
        ).rank(classified_doc(), job_label="1/1")

        # Labelled by the job's fingerprint, as before the batch reused the pipeline.
        (seen,) = llm1.prompt_logging
        assert (seen.llm1, seen.job_label) == (True, "fp-1")


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

        result = (
            build_job_ranking_service_factory(s, build_traced_llm_client(s.llm))
            .get(s)
            .rank(classified_doc())
        )

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

    def test_llm3_call_goes_through_the_configured_client(self, make_settings):
        ScriptedLlmClient.calls = []
        s = make_settings(
            TARGET_COUNTRY="kenya",
            RERANKER_LLM_CLIENT_CLASS="tests.test_pipeline_integration:ScriptedLlmClient",
            RERANKER_LLM_MODEL="scripted-model-v1",
        )
        step = OccupationSkillsRelevanceStep(
            build_traced_llm_client(s.llm),
            OccupationSkillsRelevancePromptBuilder(PromptRepository().load(s)),
            OccupationSkillsRelevanceResponseParser(),
        )
        ctx = build_job_description_context(classified_doc(), settings=s)

        result = step.run(ctx, occupation_menu())

        assert result.decisions == ()
        ((purpose, prompt),) = ScriptedLlmClient.calls
        assert purpose == "occupation_skills_relevance"
        assert "Weld metal" in prompt and "Paint surfaces" in prompt


class TestJobRankingServiceFactory:
    """One service per language pack, cached; ``None`` means the default language."""

    def test_language_lookup_and_health(self, make_settings):
        s = make_settings(
            TARGET_COUNTRY="kenya",
            RERANKER_LLM_CLIENT_CLASS="tests.test_pipeline_integration:ScriptedLlmClient",
            RERANKER_LLM_MODEL="scripted-model-v1",
        )
        services = build_job_ranking_service_factory(s, build_traced_llm_client(s.llm))

        assert services.health().loaded_languages == []
        default = services.for_language(None)
        assert services.for_language("en") is default
        spanish = services.for_language("es")

        assert spanish is not default
        assert spanish.settings.language == "es"
        assert services.for_language("es") is spanish
        health = services.health()
        assert (health.default_language, health.taxonomy_loaded) == ("en", True)
        assert health.loaded_languages == ["en", "es"]
