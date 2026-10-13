"""Tests for the Langfuse tracing layer (``app.observability``).

Covers what the integration promises:

1. off by default, and a no-op when off — nothing to guard at call sites;
2. every LLM pass is a ``generation`` carrying prompt, output, model and token usage;
3. every Gemini ``embed_content`` call is an ``embedding`` carrying model, parameters and
   shape — but not the vectors;
4. all of a job's observations nest under one ``rank_job`` trace, including the LLM1/LLM2
   calls that run on worker threads;
5. sampling and masking behave as configured.

No network: spans go to an in-memory exporter; the model and embedding APIs are stubbed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import httpx
import numpy as np
import pytest

from app.job_context import JobDescriptionContext
from app.llm import LlmClient, LlmConfig
from app.llm.providers.anthropic import AnthropicLlmClient
from app.llm.providers.gemini import GeminiLlmClient
from app.llm.providers.ollama import OllamaLlmClient
from app.llm.providers.openai import OpenAiLlmClient
from app.observability import (
    SamplingTier,
    TracingConfig,
    job_trace,
    parse_tracing_config,
    report_usage,
    should_sample,
    start_trace,
    traced_observation,
    tracing_config_from_env,
)
from app.observability.masking import REDACTED_EMAIL, REDACTED_PHONE, mask_value
from app.observability.testing import TEST_TRACING_CONFIG, RecordedSpans, in_memory_tracing

pytestmark = pytest.mark.unit

TYPE = "langfuse.observation.type"
MODEL = "langfuse.observation.model.name"
USAGE = "langfuse.observation.usage_details"
INPUT = "langfuse.observation.input"
OUTPUT = "langfuse.observation.output"
LEVEL = "langfuse.observation.level"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class UsageReportingLlmClient(LlmClient):
    """Returns fixed JSON and reports token usage, the way the shipped providers do."""

    def generate_json(self, prompt: str, *, purpose: str = "") -> str:
        report_usage({"input": 120, "output": 30, "total": 150})
        return json.dumps({"essential_skills": [], "optional_skills": []})


class FailingLlmClient(LlmClient):
    def generate_json(self, prompt: str, *, purpose: str = "") -> str:
        raise RuntimeError("model unavailable")


def _stub_settings(make_settings, client_class: str):
    return make_settings(
        RERANKER_LLM_CLIENT_CLASS=client_class,
        RERANKER_LLM_MODEL="stub-model-1",
    )


class _FakeEmbedModels:
    def __init__(self, dim: int, *, token_count: int | None = None, fail_times: int = 0):
        self.dim = dim
        self.token_count = token_count
        self.fail_times = fail_times
        self.calls = 0

    def embed_content(self, *, model, contents, config):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("503 unavailable")
        stats = (
            SimpleNamespace(token_count=self.token_count) if self.token_count is not None else None
        )
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[0.1] * self.dim, statistics=stats) for _ in contents
            ]
        )


def _spans_by_type(spans: RecordedSpans, as_type: str) -> list:
    return [s for s in spans.all() if s.attributes.get(TYPE) == as_type]


# ---------------------------------------------------------------------------
# 1. Configuration, and the no-op path
# ---------------------------------------------------------------------------


def test_tracing_is_off_by_default(isolated_env):
    config = tracing_config_from_env()
    assert config.enabled is False


def test_env_enables_tracing_and_keys_fall_back_to_sdk_names(isolated_env):
    isolated_env.setenv("RERANKER_ENABLE_TRACING", "true")
    isolated_env.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    isolated_env.setenv("RERANKER_LANGFUSE_SECRET_KEY", "sk-lf-y")
    isolated_env.setenv("RERANKER_TRACING_ENVIRONMENT", "Kenya Prod")
    config = tracing_config_from_env()
    assert config.enabled is True
    assert (config.public_key, config.secret_key) == ("pk-lf-x", "sk-lf-y")
    assert config.environment == "kenya-prod"


def test_json_tunables_are_applied_and_bad_values_ignored():
    config = parse_tracing_config(
        enabled=True,
        public_key="pk",
        secret_key="sk",
        host=None,
        raw_config=json.dumps(
            {
                "batchSampleRate": 0.25,
                "maskPii": False,
                "apiSampleRate": 7,  # out of range
                "secretKey": "leak",  # never taken from the blob
                "nonsense": 1,
            }
        ),
    )
    assert config.batch_sample_rate == 0.25
    assert config.mask_pii is False
    assert config.api_sample_rate == 1.0
    assert config.secret_key == "sk"


def test_unparsable_json_falls_back_to_defaults():
    config = parse_tracing_config(
        enabled=True, public_key="pk", secret_key="sk", host=None, raw_config="{not json"
    )
    assert config == TracingConfig(enabled=True, public_key="pk", secret_key="sk")


def test_helpers_are_no_ops_when_tracing_is_off():
    with start_trace(name="rank_job", tier=SamplingTier.BATCH, sampling_key="j1") as root:
        with traced_observation(name="x", as_type="generation") as obs:
            report_usage({"input": 1})
    assert root is None and obs is None


# ---------------------------------------------------------------------------
# 2. LLM generations
# ---------------------------------------------------------------------------


def test_llm_call_is_recorded_as_generation_with_usage(make_settings):
    from app.llm_service import PURPOSE_SKILL_RANKER, call_skill_ranker_model

    settings = _stub_settings(make_settings, "tests.test_observability:UsageReportingLlmClient")
    with in_memory_tracing() as spans:
        out = call_skill_ranker_model("rank these skills", settings=settings)

    gen = spans.by_name(PURPOSE_SKILL_RANKER)
    assert gen is not None
    assert gen.attributes[TYPE] == "generation"
    assert gen.attributes[MODEL] == "stub-model-1"
    assert RecordedSpans.json_attribute(gen, INPUT) == "rank these skills"
    assert RecordedSpans.json_attribute(gen, OUTPUT) == json.loads(out)
    assert RecordedSpans.json_attribute(gen, USAGE) == {"input": 120, "output": 30, "total": 150}


def test_failed_llm_call_marks_generation_as_error(make_settings):
    from app.llm_service import PURPOSE_JOB_ATTRIBUTES, call_job_attributes_model

    settings = _stub_settings(make_settings, "tests.test_observability:FailingLlmClient")
    with in_memory_tracing() as spans:
        with pytest.raises(RuntimeError):
            call_job_attributes_model("prompt", settings=settings)

    gen = spans.by_name(PURPOSE_JOB_ATTRIBUTES)
    assert gen is not None
    assert gen.attributes[LEVEL] == "ERROR"


def test_http_providers_parse_usage():
    openai = OpenAiLlmClient(LlmConfig("openai", "x", model="m", api_key="k"))
    assert openai.extract_usage(
        {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    ) == {"input": 10, "output": 5, "total": 15}

    anthropic = AnthropicLlmClient(LlmConfig("anthropic", "x", model="m", api_key="k"))
    assert anthropic.extract_usage(
        {"usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100}}
    ) == {"input": 10, "output": 5, "cache_read_input_tokens": 100, "total": 115}

    ollama = OllamaLlmClient(LlmConfig("ollama", "x", model="m"))
    assert ollama.extract_usage({"prompt_eval_count": 7, "eval_count": 3}) == {
        "input": 7,
        "output": 3,
        "total": 10,
    }
    assert ollama.extract_usage({}) is None


def test_http_provider_reports_usage_onto_the_generation(monkeypatch):
    client = OpenAiLlmClient(LlmConfig("openai", "x", model="gpt-x", api_key="k"))

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
            },
        )

    monkeypatch.setattr(
        client, "_client", lambda: httpx.Client(transport=httpx.MockTransport(_handler))
    )
    with in_memory_tracing() as spans:
        with traced_observation(name="gen", as_type="generation"):
            client.generate_json("p", purpose="job_attributes")

    usage = RecordedSpans.json_attribute(spans.by_name("gen"), USAGE)
    assert usage == {"input": 11, "output": 2, "total": 13}


def test_gemini_reports_usage_including_thinking_tokens(monkeypatch):
    client = GeminiLlmClient(LlmConfig("gemini", "x", model="gemini-2.5-flash", api_key="k"))
    response = SimpleNamespace(
        text="{}",
        usage_metadata=SimpleNamespace(
            prompt_token_count=40,
            candidates_token_count=8,
            thoughts_token_count=2,
            total_token_count=50,
        ),
    )
    fake_sdk = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: response))
    monkeypatch.setattr(client, "_client", lambda: fake_sdk)

    with in_memory_tracing() as spans:
        with traced_observation(name="gen", as_type="generation"):
            client.generate_json("p")

    usage = RecordedSpans.json_attribute(spans.by_name("gen"), USAGE)
    assert usage == {"input": 40, "output": 10, "total": 50}


# ---------------------------------------------------------------------------
# 3. Embeddings
# ---------------------------------------------------------------------------


def test_embed_call_is_recorded_as_embedding_without_vectors(monkeypatch):
    from utils import gemini_embeddings

    models = _FakeEmbedModels(dim=8, token_count=6)
    monkeypatch.setattr("google.genai.Client", lambda **_: SimpleNamespace(models=models))

    with in_memory_tracing() as spans:
        out = gemini_embeddings.embed_text_list(
            ["welding; metal fabrication"],
            api_key="k",
            model="gemini-embedding-001",
            embedding_dim=8,
            task_type="SEMANTIC_SIMILARITY",
        )

    assert out.shape == (1, 8)
    (emb,) = _spans_by_type(spans, "embedding")
    assert emb.name == "embed_content"
    assert emb.attributes[MODEL] == "gemini-embedding-001"
    assert RecordedSpans.json_attribute(emb, INPUT) == ["welding; metal fabrication"]
    assert RecordedSpans.json_attribute(emb, OUTPUT) == {"vectors": 1, "dim": 8}
    params = RecordedSpans.json_attribute(emb, "langfuse.observation.model.parameters")
    assert params == {"task_type": "SEMANTIC_SIMILARITY", "output_dimensionality": 8}
    assert RecordedSpans.json_attribute(emb, USAGE) == {"input": 6, "total": 6}


def test_embed_call_without_reported_tokens_records_no_usage(monkeypatch):
    from utils import gemini_embeddings

    models = _FakeEmbedModels(dim=4)
    monkeypatch.setattr("google.genai.Client", lambda **_: SimpleNamespace(models=models))

    with in_memory_tracing() as spans:
        gemini_embeddings.embed_text_list(["a", "b"], api_key="k", embedding_dim=4)

    (emb,) = _spans_by_type(spans, "embedding")
    assert USAGE not in emb.attributes
    assert RecordedSpans.json_attribute(emb, OUTPUT) == {"vectors": 2, "dim": 4}


def test_failed_embed_call_is_recorded_as_error(monkeypatch):
    from utils import gemini_embeddings

    models = _FakeEmbedModels(dim=4, fail_times=99)
    monkeypatch.setattr("google.genai.Client", lambda **_: SimpleNamespace(models=models))
    monkeypatch.setattr(gemini_embeddings.time, "sleep", lambda _s: None)

    with in_memory_tracing() as spans:
        with pytest.raises(RuntimeError):
            gemini_embeddings.embed_text_list(["a"], api_key="k", embedding_dim=4)

    (emb,) = _spans_by_type(spans, "embedding")
    assert emb.attributes[LEVEL] == "ERROR"


def test_job_embedding_step_nests_under_the_job_trace(make_settings, monkeypatch):
    from utils.job_embedding import attach_job_embedding_to_payload

    settings = make_settings(GEMINI_API_KEY="k", RERANKER_JOB_EMBEDDING_DIM="4")
    models = _FakeEmbedModels(dim=4)
    monkeypatch.setattr("google.genai.Client", lambda **_: SimpleNamespace(models=models))
    payload = {"job_id": "job-42", "llm_classified_skills": {"essential": ["welding"]}}

    with in_memory_tracing() as spans:
        with job_trace({"job_id": "job-42"}, settings=settings, invocation="batch"):
            attach_job_embedding_to_payload(payload, settings=settings)

    (emb,) = _spans_by_type(spans, "embedding")
    root = spans.by_name("rank_job")
    assert root is not None
    assert emb.context.trace_id == root.context.trace_id
    assert emb.parent.span_id == root.context.span_id
    assert len(payload["job_embedding"]) == 4


# ---------------------------------------------------------------------------
# 4. One trace per job, across threads
# ---------------------------------------------------------------------------


def test_overlapped_llm1_llm2_nest_under_the_job_trace(make_settings, monkeypatch):
    """LLM1 and LLM2 run on a thread pool; their generations must not become orphan traces."""
    from app import pipeline
    from app.pipeline import TaxonomyContext
    from utils.classifier_linked_entities import EscoSkillCandidate

    settings = make_settings(
        RERANKER_LLM_CLIENT_CLASS="tests.test_observability:UsageReportingLlmClient",
        RERANKER_LLM_MODEL="m",
    )

    def _fake_llm1(*_a, **_k):
        with traced_observation(name="skill_ranker", as_type="generation"):
            pass
        return [], [], None, ""

    def _fake_llm2(*_a, **_k):
        with traced_observation(name="job_attributes", as_type="generation"):
            pass
        return None

    monkeypatch.setattr(pipeline, "run_skill_ranker_for_job", _fake_llm1)
    monkeypatch.setattr(pipeline, "run_job_attributes_for_job", _fake_llm2)
    monkeypatch.setattr(pipeline, "build_occupation_prompt_enrichment", lambda *a, **k: None)
    # Doubles: run_llm1_llm2_phase only reads these specific attributes once llm1/llm2
    # and the occupation-enrichment call are monkeypatched above, so a full
    # TaxonomyContext / JobDescriptionContext is unneeded ceremony here.
    taxonomy = cast(
        TaxonomyContext,
        SimpleNamespace(
            skill_desc_by_esco={},
            occ_bridge={},
            occ_all_ids=frozenset(),
            occ_label_by_esco={},
            occ_desc_by_esco={},
        ),
    )
    ctx = cast(JobDescriptionContext, SimpleNamespace(title="Welder"))
    base_skills = cast(list[EscoSkillCandidate], [object()])

    with in_memory_tracing() as spans:
        with job_trace(
            {"job_id": "job-7"}, settings=settings, invocation="batch", session_id="run-1"
        ):
            pipeline.run_llm1_llm2_phase(
                ctx, {}, settings, taxonomy, base_skills, overlap_llm12=True
            )

    root = spans.by_name("rank_job")
    assert root is not None
    for name in ("skill_ranker", "job_attributes"):
        span = spans.by_name(name)
        assert span is not None
        assert span.context.trace_id == root.context.trace_id, name
        assert span.parent.span_id == root.context.span_id, name
    assert root.attributes["session.id"] == "run-1"
    assert "invocation:batch" in root.attributes["langfuse.trace.tags"]


# ---------------------------------------------------------------------------
# 5. Sampling and masking
# ---------------------------------------------------------------------------


def test_sampling_is_deterministic_and_respects_bounds():
    half = TracingConfig(batch_sample_rate=0.5)
    decisions = {
        should_sample(tier=SamplingTier.BATCH, key=f"job-{i}", config=half) for i in range(200)
    }
    assert decisions == {True, False}
    assert should_sample(tier=SamplingTier.BATCH, key="job-1", config=half) == should_sample(
        tier=SamplingTier.BATCH, key="job-1", config=half
    )
    assert not should_sample(
        tier=SamplingTier.BATCH, key="x", config=TracingConfig(batch_sample_rate=0.0)
    )
    assert should_sample(tier=SamplingTier.API, key="x", config=TracingConfig(api_sample_rate=1.0))


def test_unsampled_job_records_nothing_beneath_it():
    from dataclasses import replace

    config = replace(TEST_TRACING_CONFIG, batch_sample_rate=0.0)
    with in_memory_tracing(config) as spans:
        with start_trace(name="rank_job", tier=SamplingTier.BATCH, sampling_key="j") as root:
            with traced_observation(name="skill_ranker", as_type="generation") as gen:
                pass
    assert root is None and gen is None
    assert spans.all() == []


def test_masking_redacts_contacts_and_truncates():
    text = "Apply to jobs@acme.co.ke or call +254 712 345 678 by 2026. " + "x" * 50
    result = mask_value({"prompt": text}, redact_pii=True, max_chars=80)
    assert isinstance(result, dict)
    masked = result["prompt"]
    assert isinstance(masked, str)
    assert REDACTED_EMAIL in masked and REDACTED_PHONE in masked
    assert "2026" in masked
    assert masked.endswith("...[TRUNCATED]")


def test_exported_generation_input_is_masked(make_settings):
    from app.llm_service import PURPOSE_SKILL_RANKER, call_skill_ranker_model

    settings = _stub_settings(make_settings, "tests.test_observability:UsageReportingLlmClient")
    with in_memory_tracing() as spans:
        call_skill_ranker_model("Contact hr@example.com for details", settings=settings)

    exported = RecordedSpans.json_attribute(spans.by_name(PURPOSE_SKILL_RANKER), INPUT)
    assert isinstance(exported, str)
    assert "hr@example.com" not in exported and REDACTED_EMAIL in exported


def test_embedding_vectors_are_float_arrays():
    """Guard: the embedding observation must not change what the caller gets back."""
    from google.genai import Client

    from utils.gemini_embeddings import _embed_chunk

    client = cast(Client, SimpleNamespace(models=_FakeEmbedModels(dim=3)))
    mat = _embed_chunk(client, ["a"], model="m", embedding_dim=3, task_type="T")
    assert mat.dtype == np.float32 and mat.shape == (1, 3)
