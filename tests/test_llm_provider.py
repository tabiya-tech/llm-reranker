"""Tests for the configurable LLM backend (``app.llm``).

Covers the four things the feature promises:

1. configuration selects the model — by provider alias or by class path;
2. the reranker's grading/enrichment code calls whatever was selected, unchanged;
3. the Gemini default path is byte-identical to the pre-abstraction behaviour;
4. a misconfigured selection fails with an actionable error rather than degrading.

No network: the Gemini SDK client and every HTTP transport are stubbed.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.llm import (
    PROVIDER_CLASS_PATHS,
    PROVIDER_DEFAULT_MODELS,
    LlmClient,
    LlmConfig,
    LlmConfigurationError,
    LlmInvocationError,
    build_llm_client,
    load_llm_client_class,
)
from app.llm.providers.anthropic import AnthropicLlmClient
from app.llm.providers.gemini import GeminiLlmClient
from app.llm.providers.ollama import OllamaLlmClient
from app.llm.providers.openai import OpenAiCompatibleLlmClient, OpenAiLlmClient
from app.llm.providers.qwen import QwenLlmClient

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class RecordingLlmClient(LlmClient):
    """Minimal in-repo implementation, used to prove a custom class path works."""

    #: Every prompt this class has been asked, across instances.
    seen: list[tuple[str, str]] = []

    def generate_json(self, prompt: str, *, purpose: str = "") -> str:
        RecordingLlmClient.seen.append((purpose, prompt))
        return json.dumps({"essential_skills": ["welding"], "optional_skills": []})


class NotAnLlmClient:
    """Deliberately does not implement the interface."""


class AbstractLlmClient(LlmClient):
    """Subclass that forgot ``generate_json``."""


def _config(**overrides) -> LlmConfig:
    base: dict[str, Any] = dict(
        provider="custom",
        client_class="tests.test_llm_provider:RecordingLlmClient",
        model="test-model",
        api_key="k",
    )
    base.update(overrides)
    return LlmConfig(**base)


# ---------------------------------------------------------------------------
# Settings → LlmConfig resolution
# ---------------------------------------------------------------------------


class TestConfigResolution:
    def test_default_is_gemini_with_the_historical_model(self, make_settings):
        """No RERANKER_LLM_* set: an existing deployment is unchanged."""
        s = make_settings()
        assert s.llm.provider == "gemini"
        assert s.llm.client_class == PROVIDER_CLASS_PATHS["gemini"]
        assert s.llm.model == PROVIDER_DEFAULT_MODELS["gemini"] == "gemini-2.5-flash"
        assert s.llm.temperature == 0.2

    def test_gemini_model_env_still_selects_the_model(self, make_settings):
        s = make_settings(GEMINI_MODEL="gemini-3-pro-preview")
        assert s.llm.model == "gemini-3-pro-preview"

    def test_vertex_model_is_the_second_gemini_fallback(self, make_settings):
        s = make_settings(VERTEX_MODEL="gemini-2.0-flash")
        assert s.llm.model == "gemini-2.0-flash"

    def test_gemini_model_wins_over_vertex_model(self, make_settings):
        s = make_settings(GEMINI_MODEL="gemini-3-pro-preview", VERTEX_MODEL="gemini-2.0-flash")
        assert s.llm.model == "gemini-3-pro-preview"

    def test_providers_without_a_default_get_no_model(self, make_settings):
        s = make_settings(RERANKER_LLM_PROVIDER="anthropic", GEMINI_MODEL="gemini-3-pro-preview")
        assert s.llm.model == ""

    @pytest.mark.parametrize(
        "model, expected",
        [
            ("gemini-2.5-flash", 0.2),
            ("gemini-3-pro-preview", 1.0),
            ("gemini-3.5-flash", 1.0),
            ("gemini-3.8-pro-preview", 1.0),
            ("models/gemini-3.5-flash", 1.0),
            ("gemini-flash-latest", 0.2),
        ],
    )
    def test_gemini_temperature_default_follows_the_model(self, make_settings, model, expected):
        assert make_settings(GEMINI_MODEL=model).llm.temperature == expected

    def test_an_explicit_temperature_wins_over_the_model_default(self, make_settings):
        s = make_settings(GEMINI_MODEL="gemini-3.5-flash", RERANKER_LLM_TEMPERATURE="0.2")
        assert s.llm.temperature == 0.2

    def test_other_providers_keep_the_historical_temperature(self, make_settings):
        s = make_settings(RERANKER_LLM_PROVIDER="openai", RERANKER_LLM_MODEL="gemini-3-lookalike")
        assert s.llm.temperature == 0.2

    def test_thinking_level_is_unset_by_default(self, make_settings):
        assert make_settings().llm.thinking_level == ""

    def test_thinking_level_is_normalised(self, make_settings):
        assert make_settings(RERANKER_LLM_THINKING_LEVEL=" High ").llm.thinking_level == "high"

    def test_an_unknown_thinking_level_fails_at_config_time(self, make_settings):
        with pytest.raises(LlmConfigurationError, match="RERANKER_LLM_THINKING_LEVEL"):
            make_settings(RERANKER_LLM_THINKING_LEVEL="extreme")

    def test_gemini_api_key_env_feeds_the_generic_config(self, make_settings):
        s = make_settings(GEMINI_API_KEY="abc123")
        assert s.llm.api_key == "abc123"
        assert s.llm.api_key_tail() == "c123"

    def test_gemini_api_key_property_and_llm_api_key_agree(self, make_settings):
        """``settings.gemini_api_key`` (read directly by the job-embedding step) and
        ``settings.llm.api_key`` (read by the LLM client) must resolve to the same value
        when the provider is Gemini — they used to disagree when ``llm`` fell back to
        reading ``os.environ`` instead of the already-resolved ``gemini_api_key``."""
        s = make_settings(GEMINI_API_KEY="abc123")
        assert s.gemini_api_key == s.llm.api_key == "abc123"

    def test_legacy_api_key_alias_still_works(self, make_settings):
        s = make_settings(API_KEY="legacy-key")
        assert s.llm.api_key == "legacy-key"

    def test_generic_model_var_overrides_the_gemini_chain(self, make_settings):
        s = make_settings(
            GEMINI_MODEL="gemini-2.5-flash", RERANKER_LLM_MODEL="gemini-3-pro-preview"
        )
        assert s.llm.model == "gemini-3-pro-preview"

    def test_provider_alias_selects_the_shipped_class(self, make_settings):
        s = make_settings(
            RERANKER_LLM_PROVIDER="anthropic",
            RERANKER_LLM_MODEL="claude-opus-5",
            ANTHROPIC_API_KEY="sk-ant-xyz",
        )
        assert s.llm.provider == "anthropic"
        assert s.llm.client_class == PROVIDER_CLASS_PATHS["anthropic"]
        assert s.llm.model == "claude-opus-5"
        assert s.llm.api_key == "sk-ant-xyz"

    def test_provider_alias_is_case_and_separator_insensitive(self, make_settings):
        s = make_settings(RERANKER_LLM_PROVIDER="OpenAI-Compatible", RERANKER_LLM_MODEL="qwen3-32b")
        assert s.llm.provider == "openai_compatible"
        assert s.llm.client_class == PROVIDER_CLASS_PATHS["openai_compatible"]

    def test_openai_key_env_is_used_for_the_openai_provider(self, make_settings):
        s = make_settings(
            RERANKER_LLM_PROVIDER="openai", RERANKER_LLM_MODEL="gpt-4.1", OPENAI_API_KEY="sk-oai"
        )
        assert s.llm.api_key == "sk-oai"

    def test_dashscope_key_env_is_used_for_the_qwen_provider(self, make_settings):
        s = make_settings(
            RERANKER_LLM_PROVIDER="qwen",
            RERANKER_LLM_MODEL="qwen-plus",
            DASHSCOPE_API_KEY="sk-dashscope",
        )
        assert s.llm.client_class == PROVIDER_CLASS_PATHS["qwen"]
        assert s.llm.api_key == "sk-dashscope"

    def test_qwen_api_key_is_the_second_dashscope_fallback(self, make_settings):
        s = make_settings(
            RERANKER_LLM_PROVIDER="qwen", RERANKER_LLM_MODEL="qwen-plus", QWEN_API_KEY="sk-qwen"
        )
        assert s.llm.api_key == "sk-qwen"

    def test_ollama_needs_only_a_model(self, make_settings):
        """A local Ollama has no key and listens on a known port — nothing else to set."""
        s = make_settings(RERANKER_LLM_PROVIDER="ollama", RERANKER_LLM_MODEL="qwen3:32b")
        assert s.llm.client_class == PROVIDER_CLASS_PATHS["ollama"]
        assert s.llm.api_key == ""
        assert s.llm.base_url == ""
        client = build_llm_client(s.llm)
        assert isinstance(client, OllamaLlmClient)
        assert client.base_url == "http://localhost:11434"

    def test_ollama_host_env_supplies_the_endpoint_and_gains_a_scheme(self, make_settings):
        """OLLAMA_HOST is the variable the Ollama CLI itself reads, and is scheme-less."""
        s = make_settings(
            RERANKER_LLM_PROVIDER="ollama",
            RERANKER_LLM_MODEL="qwen3:32b",
            OLLAMA_HOST="ollama.internal:11434",
        )
        assert s.llm.base_url == "http://ollama.internal:11434"

    def test_generic_base_url_wins_over_the_ollama_var(self, make_settings):
        s = make_settings(
            RERANKER_LLM_PROVIDER="ollama",
            RERANKER_LLM_MODEL="qwen3:32b",
            OLLAMA_HOST="ollama.internal:11434",
            RERANKER_LLM_BASE_URL="http://127.0.0.1:9999/",
        )
        assert s.llm.base_url == "http://127.0.0.1:9999"

    def test_ollama_host_is_not_read_for_another_provider(self, make_settings):
        s = make_settings(
            OLLAMA_HOST="ollama.internal:11434",
            RERANKER_LLM_PROVIDER="openai",
            RERANKER_LLM_MODEL="gpt-4.1",
            OPENAI_API_KEY="sk-oai",
        )
        assert s.llm.base_url == ""

    def test_gemini_key_is_not_leaked_to_another_provider(self, make_settings):
        """A machine with GEMINI_API_KEY exported must not send it to Anthropic."""
        s = make_settings(
            GEMINI_API_KEY="gemini-secret",
            RERANKER_LLM_PROVIDER="anthropic",
            RERANKER_LLM_MODEL="claude-opus-5",
        )
        assert s.llm.api_key == ""

    def test_generic_api_key_wins_over_provider_specific(self, make_settings):
        s = make_settings(RERANKER_LLM_API_KEY="generic", GEMINI_API_KEY="specific")
        assert s.llm.api_key == "generic"

    def test_non_gemini_provider_has_no_default_model(self, make_settings):
        """Better an explicit failure later than silently ranking with the wrong model."""
        s = make_settings(RERANKER_LLM_PROVIDER="openai")
        assert s.llm.model == ""

    def test_class_path_alone_marks_the_provider_custom(self, make_settings):
        s = make_settings(
            RERANKER_LLM_CLIENT_CLASS="tests.test_llm_provider:RecordingLlmClient",
            RERANKER_LLM_MODEL="my-model",
        )
        assert s.llm.provider == "custom"
        assert s.llm.client_class == "tests.test_llm_provider:RecordingLlmClient"

    def test_class_path_overrides_the_alias_but_alias_still_supplies_the_key(self, make_settings):
        s = make_settings(
            RERANKER_LLM_PROVIDER="anthropic",
            RERANKER_LLM_CLIENT_CLASS="tests.test_llm_provider:RecordingLlmClient",
            RERANKER_LLM_MODEL="claude-opus-5",
            ANTHROPIC_API_KEY="sk-ant",
        )
        assert s.llm.client_class == "tests.test_llm_provider:RecordingLlmClient"
        assert s.llm.provider == "anthropic"
        assert s.llm.api_key == "sk-ant"

    def test_unknown_provider_fails_at_config_time_naming_the_alternatives(self, make_settings):
        with pytest.raises(LlmConfigurationError) as exc:
            make_settings(RERANKER_LLM_PROVIDER="gpt5-turbo-ultra")
        msg = str(exc.value)
        assert "gpt5-turbo-ultra" in msg
        assert "anthropic" in msg and "gemini" in msg
        assert "RERANKER_LLM_CLIENT_CLASS" in msg

    def test_tuning_knobs_are_read(self, make_settings):
        s = make_settings(
            RERANKER_LLM_TEMPERATURE="0.7",
            RERANKER_LLM_MAX_OUTPUT_TOKENS="4096",
            RERANKER_LLM_TIMEOUT_S="45",
            RERANKER_LLM_MAX_ATTEMPTS="3",
            RERANKER_LLM_BASE_URL="http://localhost:11434/v1/",
            RERANKER_LLM_OPTIONS='{"top_p": 0.9}',
        )
        assert s.llm.temperature == 0.7
        assert s.llm.max_output_tokens == 4096
        assert s.llm.timeout_s == 45.0
        assert s.llm.max_attempts == 3
        assert s.llm.base_url == "http://localhost:11434/v1"  # trailing slash normalised
        assert s.llm.options() == {"top_p": 0.9}

    def test_max_attempts_falls_back_to_the_legacy_gemini_var(self, make_settings):
        s = make_settings(GEMINI_TRANSIENT_MAX_ATTEMPTS="7")
        assert s.llm.max_attempts == 7

    def test_max_attempts_is_clamped_like_the_legacy_path(self, make_settings):
        assert make_settings(RERANKER_LLM_MAX_ATTEMPTS="99").llm.max_attempts == 12
        assert make_settings(RERANKER_LLM_MAX_ATTEMPTS="0").llm.max_attempts == 1

    def test_config_is_hashable_so_it_can_key_the_client_cache(self, make_settings):
        s = make_settings()
        assert hash(s.llm) == hash(Settings.from_env().llm)


class TestOptionsParsing:
    def test_empty_options_is_an_empty_dict(self):
        assert _config().options() == {}

    def test_malformed_json_names_the_variable_and_shows_the_value(self):
        cfg = _config(options_json="{top_p: 0.9}")
        with pytest.raises(LlmConfigurationError) as exc:
            cfg.options()
        assert "RERANKER_LLM_OPTIONS" in str(exc.value)
        assert "top_p" in str(exc.value)

    def test_json_that_is_not_an_object_is_rejected(self):
        with pytest.raises(LlmConfigurationError, match="must be a JSON object"):
            _config(options_json="[1, 2]").options()


# ---------------------------------------------------------------------------
# The factory: loading a class by path
# ---------------------------------------------------------------------------


class TestClassLoading:
    def test_colon_form_loads(self):
        assert load_llm_client_class("tests.test_llm_provider:RecordingLlmClient") is (
            RecordingLlmClient
        )

    def test_dot_form_loads(self):
        assert load_llm_client_class("tests.test_llm_provider.RecordingLlmClient") is (
            RecordingLlmClient
        )

    @pytest.mark.parametrize("path", sorted(PROVIDER_CLASS_PATHS.values()))
    def test_every_shipped_alias_resolves_to_a_concrete_client(self, path):
        cls = load_llm_client_class(path)
        assert issubclass(cls, LlmClient)

    def test_empty_path_explains_both_ways_to_configure(self):
        with pytest.raises(LlmConfigurationError) as exc:
            load_llm_client_class("")
        assert "RERANKER_LLM_PROVIDER" in str(exc.value)
        assert "RERANKER_LLM_CLIENT_CLASS" in str(exc.value)

    def test_path_without_a_separator_is_rejected(self):
        with pytest.raises(LlmConfigurationError, match="not a class path"):
            load_llm_client_class("GeminiLlmClient")

    def test_missing_module_names_the_module(self):
        with pytest.raises(LlmConfigurationError) as exc:
            load_llm_client_class("app.llm.providers.no_such_vendor:Client")
        assert "no_such_vendor" in str(exc.value)

    def test_missing_attribute_lists_the_clients_that_do_exist(self):
        with pytest.raises(LlmConfigurationError) as exc:
            load_llm_client_class("app.llm.providers.gemini:TypoClient")
        msg = str(exc.value)
        assert "TypoClient" in msg
        assert "GeminiLlmClient" in msg

    def test_non_class_target_is_rejected(self):
        with pytest.raises(LlmConfigurationError, match="not a class"):
            load_llm_client_class("app.llm.providers.gemini:logger")

    def test_class_that_does_not_implement_the_interface_is_rejected(self):
        with pytest.raises(LlmConfigurationError) as exc:
            load_llm_client_class("tests.test_llm_provider:NotAnLlmClient")
        assert "not a subclass" in str(exc.value)
        assert "generate_json" in str(exc.value)

    def test_abstract_subclass_names_the_unimplemented_method(self):
        with pytest.raises(LlmConfigurationError) as exc:
            load_llm_client_class("tests.test_llm_provider:AbstractLlmClient")
        assert "abstract" in str(exc.value)
        assert "generate_json" in str(exc.value)


class TestClientConstruction:
    def test_each_build_is_a_fresh_instance(self):
        # No process-wide cache: the composition root builds one client and passes it on.
        cfg = _config()
        assert build_llm_client(cfg) is not build_llm_client(cfg)


# ---------------------------------------------------------------------------
# The seam: grading code calls whatever was configured
# ---------------------------------------------------------------------------


class TestGradingCodeUsesTheConfiguredModel:
    def test_skill_ranker_call_reaches_a_custom_client(self, make_settings):
        """The configured client is provider-agnostic — no Gemini import involved."""
        from app.job_ranking.skill_ranker.step import PURPOSE_SKILL_RANKER
        from app.llm import build_traced_llm_client

        RecordingLlmClient.seen = []
        s = make_settings(
            RERANKER_LLM_CLIENT_CLASS="tests.test_llm_provider:RecordingLlmClient",
            RERANKER_LLM_MODEL="my-model",
        )
        out = build_traced_llm_client(s.llm).generate_json(
            "rank these skills", purpose=PURPOSE_SKILL_RANKER
        )
        assert json.loads(out)["essential_skills"] == ["welding"]
        assert RecordingLlmClient.seen == [(PURPOSE_SKILL_RANKER, "rank these skills")]

    def test_ranked_document_metadata_records_the_configured_model(self, make_settings):
        from app.job_ranking.ranked_document import build_llm_reranker_meta

        s = make_settings(
            RERANKER_LLM_PROVIDER="anthropic",
            RERANKER_LLM_MODEL="claude-opus-5",
            ANTHROPIC_API_KEY="sk-ant",
        )
        meta = build_llm_reranker_meta(s)
        assert meta.llm_provider == "anthropic"
        assert meta.skill_ranker_model == "claude-opus-5"
        assert meta.job_attributes_model == "claude-opus-5"
        assert meta.occupation_skills_relevance_model == "claude-opus-5"

    def test_gemini_metadata_is_unchanged_from_the_legacy_path(self, make_settings):
        """No-regression guard: the three model fields still carry the Gemini model id."""
        from app.job_ranking.ranked_document import build_llm_reranker_meta

        s = make_settings(GEMINI_API_KEY="k", GEMINI_MODEL="gemini-2.5-flash")
        meta = build_llm_reranker_meta(s)
        assert meta.skill_ranker_model == "gemini-2.5-flash"
        assert meta.occupation_skills_relevance_model == "gemini-2.5-flash"
        assert meta.job_attributes_model == "gemini-2.5-flash"


class SchemaAwareLlmClient(LlmClient):
    """Opts into structured output; records the schema each pass sent."""

    supports_response_schema = True
    seen: list[tuple[str, dict | None]] = []
    reply = "{}"

    def generate_json(self, prompt: str, *, purpose: str = "", response_schema=None) -> str:
        SchemaAwareLlmClient.seen.append((purpose, response_schema))
        return SchemaAwareLlmClient.reply


class TestResponseSchemaRouting:
    SCHEMA = {"type": "object"}

    def _call_all(self, s) -> None:
        from app.llm import build_traced_llm_client

        llm = build_traced_llm_client(s.llm)
        for purpose in ("skill_ranker", "job_attributes", "occupation_skills_relevance"):
            llm.generate_json("p", purpose=purpose, response_schema=self.SCHEMA)

    def test_schema_reaches_a_client_that_supports_it(self, make_settings):
        SchemaAwareLlmClient.seen = []
        self._call_all(
            make_settings(
                RERANKER_LLM_CLIENT_CLASS="tests.test_llm_provider:SchemaAwareLlmClient",
                RERANKER_LLM_MODEL="m",
            )
        )
        assert [schema for _, schema in SchemaAwareLlmClient.seen] == [self.SCHEMA] * 3

    def test_schema_is_withheld_when_switched_off(self, make_settings):
        SchemaAwareLlmClient.seen = []
        self._call_all(
            make_settings(
                RERANKER_LLM_CLIENT_CLASS="tests.test_llm_provider:SchemaAwareLlmClient",
                RERANKER_LLM_MODEL="m",
                RERANKER_LLM_RESPONSE_SCHEMA="false",
            )
        )
        assert [schema for _, schema in SchemaAwareLlmClient.seen] == [None] * 3

    def test_a_client_without_support_keeps_the_two_argument_call(self, make_settings):
        """``RecordingLlmClient.generate_json`` takes no ``response_schema``; it must not get one."""
        RecordingLlmClient.seen = []
        self._call_all(
            make_settings(
                RERANKER_LLM_CLIENT_CLASS="tests.test_llm_provider:RecordingLlmClient",
                RERANKER_LLM_MODEL="m",
            )
        )
        assert len(RecordingLlmClient.seen) == 3

    def test_skill_ranker_sends_the_candidate_labels_as_an_enum(self, make_settings, monkeypatch):
        from app.job_context import JobDescriptionContext
        from app.job_ranking.prompts.repository import PromptRepository
        from app.job_ranking.routes.types import RankRequest
        from app.job_ranking.skill_ranker.parser import SkillRankerResponseParser
        from app.job_ranking.skill_ranker.prompt import SkillRankerPromptBuilder
        from app.job_ranking.skill_ranker.step import SkillRankerStep
        from app.llm import build_traced_llm_client
        from tests.ranking_fakes import empty_taxonomy
        from utils.classifier_linked_entities import EscoSkillCandidate

        SchemaAwareLlmClient.seen = []
        monkeypatch.setattr(
            SchemaAwareLlmClient, "reply", '{"essential_skills": ["Python"], "optional_skills": []}'
        )
        s = make_settings(
            RERANKER_LLM_CLIENT_CLASS="tests.test_llm_provider:SchemaAwareLlmClient",
            RERANKER_LLM_MODEL="m",
        )
        step = SkillRankerStep(
            build_traced_llm_client(s.llm),
            SkillRankerPromptBuilder(PromptRepository().load(s)),
            SkillRankerResponseParser(),
            empty_taxonomy(),
        )
        ctx = JobDescriptionContext(
            job_id="J-1",
            job_fingerprint=None,
            title="Engineer",
            employer=None,
            category=None,
            application_url=None,
            posted_date=None,
            closing_date=None,
            source_platform=None,
            raw_job_description="Write Python.",
            requirements_text=None,
            salary=None,
            city=None,
            county=None,
            employment_type=None,
            classified=RankRequest.model_validate({"_id": "doc-1"}),
        )
        result = step.run(
            ctx,
            [
                EscoSkillCandidate(label="Python", uri="u1"),
                EscoSkillCandidate(label="Weld metal", uri="u2"),
            ],
        )
        assert result.essential == ["Python"]
        ((purpose, schema),) = SchemaAwareLlmClient.seen
        assert purpose == "skill_ranker"
        assert schema is not None
        items = schema["properties"]["essential_skills"]["items"]
        assert items["enum"] == ["Python", "Weld metal"]


# ---------------------------------------------------------------------------
# Gemini provider (SDK stubbed)
# ---------------------------------------------------------------------------


class _FakeGeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeGeminiModels:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeGeminiResponse(outcome) if isinstance(outcome, str) else outcome


def _stub_gemini(monkeypatch, client: GeminiLlmClient, outcomes: list) -> _FakeGeminiModels:
    """Replace the SDK client with a scripted double, keeping the retry loop real."""
    models = _FakeGeminiModels(outcomes)
    fake_sdk = type("FakeSdk", (), {"models": models})()
    monkeypatch.setattr(client, "_client", lambda: fake_sdk)
    monkeypatch.setattr("app.llm.retry.time.sleep", lambda _s: None)
    return models


class TestGeminiClient:
    def test_requires_an_api_key_and_says_which_variable(self):
        cfg = LlmConfig(
            provider="gemini",
            client_class=PROVIDER_CLASS_PATHS["gemini"],
            model="gemini-2.5-flash",
            api_key="",
        )
        with pytest.raises(LlmConfigurationError) as exc:
            build_llm_client(cfg)
        assert "GEMINI_API_KEY" in str(exc.value)

    def test_requires_a_model_id(self):
        cfg = LlmConfig(
            provider="gemini",
            client_class=PROVIDER_CLASS_PATHS["gemini"],
            model="",
            api_key="k",
        )
        with pytest.raises(LlmConfigurationError, match="RERANKER_LLM_MODEL"):
            build_llm_client(cfg)

    def test_requests_json_at_the_historical_temperature(self, monkeypatch):
        """No-regression guard on the request Gemini actually receives."""
        client = GeminiLlmClient(
            LlmConfig(
                provider="gemini",
                client_class=PROVIDER_CLASS_PATHS["gemini"],
                model="gemini-2.5-flash",
                api_key="k",
            )
        )
        models = _stub_gemini(monkeypatch, client, ['{"ok": true}'])
        assert client.generate_json("prompt", purpose="skill_ranker") == '{"ok": true}'
        (call,) = models.calls
        assert call["model"] == "gemini-2.5-flash"
        assert call["contents"] == "prompt"
        assert call["config"].temperature == 0.2
        assert call["config"].response_mime_type == "application/json"

    def test_response_schema_is_sent_as_response_json_schema(self, monkeypatch):
        client = GeminiLlmClient(
            LlmConfig(
                provider="gemini",
                client_class=PROVIDER_CLASS_PATHS["gemini"],
                model="gemini-2.5-flash",
                api_key="k",
            )
        )
        schema = {"type": "object", "properties": {"a": {"type": "string", "enum": ["x"]}}}
        models = _stub_gemini(monkeypatch, client, ['{"a": "x"}', '{"a": "x"}'])
        client.generate_json("p", response_schema=schema)
        client.generate_json("p")
        with_schema, without = models.calls
        assert with_schema["config"].response_json_schema == schema
        assert with_schema["config"].response_mime_type == "application/json"
        assert with_schema["config"].temperature == 0.2
        assert without["config"].response_json_schema is None
        assert client._generate_config.response_json_schema is None

    def test_thinking_level_reaches_the_generate_config(self):
        client = GeminiLlmClient(
            LlmConfig(
                provider="gemini",
                client_class=PROVIDER_CLASS_PATHS["gemini"],
                model="gemini-3.5-flash",
                api_key="k",
                thinking_level="low",
            )
        )
        thinking = client._generate_config.thinking_config
        assert thinking is not None
        assert thinking.thinking_level is not None
        assert thinking.thinking_level.name == "LOW"

    def test_no_thinking_level_leaves_thinking_to_the_model(self):
        client = GeminiLlmClient(
            LlmConfig(
                provider="gemini",
                client_class=PROVIDER_CLASS_PATHS["gemini"],
                model="gemini-3.5-flash",
                api_key="k",
            )
        )
        assert client._generate_config.thinking_config is None

    def test_options_thinking_config_overrides_the_level(self):
        client = GeminiLlmClient(
            LlmConfig(
                provider="gemini",
                client_class=PROVIDER_CLASS_PATHS["gemini"],
                model="gemini-2.5-flash",
                api_key="k",
                thinking_level="high",
                options_json='{"thinking_config": {"thinking_budget": 0}}',
            )
        )
        thinking = client._generate_config.thinking_config
        assert thinking is not None
        assert thinking.thinking_budget == 0
        assert thinking.thinking_level is None

    def test_options_and_token_cap_reach_the_generate_config(self):
        client = GeminiLlmClient(
            LlmConfig(
                provider="gemini",
                client_class=PROVIDER_CLASS_PATHS["gemini"],
                model="gemini-2.5-flash",
                api_key="k",
                max_output_tokens=2048,
                options_json='{"top_p": 0.5}',
            )
        )
        assert client._generate_config.max_output_tokens == 2048
        assert client._generate_config.top_p == 0.5

    def test_transient_error_is_retried_and_the_client_recreated(self, monkeypatch):
        client = GeminiLlmClient(
            LlmConfig(
                provider="gemini",
                client_class=PROVIDER_CLASS_PATHS["gemini"],
                model="gemini-2.5-flash",
                api_key="k",
                max_attempts=3,
            )
        )
        models = _stub_gemini(
            monkeypatch, client, [httpx.ReadError("connection reset"), '{"ok": true}']
        )
        resets: list[int] = []
        monkeypatch.setattr(client, "_reset_client", lambda: resets.append(1))
        assert client.generate_json("p") == '{"ok": true}'
        assert len(models.calls) == 2
        assert resets == [1]

    def test_exhausted_retries_raise_an_invocation_error(self, monkeypatch):
        client = GeminiLlmClient(
            LlmConfig(
                provider="gemini",
                client_class=PROVIDER_CLASS_PATHS["gemini"],
                model="gemini-2.5-flash",
                api_key="k",
                max_attempts=2,
            )
        )
        _stub_gemini(
            monkeypatch,
            client,
            [httpx.ReadError("connection reset"), httpx.ReadError("connection reset")],
        )
        with pytest.raises(LlmInvocationError, match="Gemini API error"):
            client.generate_json("p")

    @staticmethod
    def _api_error(code: int, retry_delay: str | None = None):
        from google.genai import errors as genai_errors

        details = (
            [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}]
            if retry_delay
            else []
        )
        body = {"error": {"code": code, "message": "m", "status": "S", "details": details}}
        cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
        return cls(code, body)

    def _retrying_client(self) -> GeminiLlmClient:
        return GeminiLlmClient(
            LlmConfig(
                provider="gemini",
                client_class=PROVIDER_CLASS_PATHS["gemini"],
                model="gemini-2.5-flash",
                api_key="k",
                max_attempts=3,
            )
        )

    @pytest.mark.parametrize("code", [429, 500, 503, 504])
    def test_rate_limits_and_server_errors_are_retried(self, monkeypatch, code):
        client = self._retrying_client()
        models = _stub_gemini(monkeypatch, client, [self._api_error(code), '{"ok": true}'])
        assert client.generate_json("p") == '{"ok": true}'
        assert len(models.calls) == 2

    def test_a_bad_request_is_not_retried(self, monkeypatch):
        client = self._retrying_client()
        models = _stub_gemini(monkeypatch, client, [self._api_error(400)])
        with pytest.raises(LlmInvocationError, match="Gemini API error"):
            client.generate_json("p")
        assert len(models.calls) == 1

    def test_rate_limit_waits_at_least_the_server_retry_delay(self, monkeypatch):
        client = self._retrying_client()
        _stub_gemini(monkeypatch, client, [self._api_error(429, "17s"), '{"ok": true}'])
        slept: list[float] = []
        monkeypatch.setattr("app.llm.retry.time.sleep", slept.append)
        client.generate_json("p")
        assert slept == [17.0]

    def test_server_retry_delay_is_capped(self, monkeypatch):
        from app.llm.retry import MAX_BACKOFF_S

        client = self._retrying_client()
        _stub_gemini(monkeypatch, client, [self._api_error(429, "3600s"), '{"ok": true}'])
        slept: list[float] = []
        monkeypatch.setattr("app.llm.retry.time.sleep", slept.append)
        client.generate_json("p")
        assert slept == [MAX_BACKOFF_S]

    def test_unknown_model_is_a_configuration_error_not_a_retry(self, monkeypatch):
        client = GeminiLlmClient(
            LlmConfig(
                provider="gemini",
                client_class=PROVIDER_CLASS_PATHS["gemini"],
                model="gemini-9-ultra",
                api_key="k",
                max_attempts=5,
            )
        )
        models = _stub_gemini(monkeypatch, client, [RuntimeError("404 model not found")])
        with pytest.raises(LlmConfigurationError) as exc:
            client.generate_json("p")
        assert "gemini-9-ultra" in str(exc.value)
        assert len(models.calls) == 1  # not retried

    @staticmethod
    def _sdk_response(text: str, finish_reason: str, **extra):
        from google.genai import types as genai_types

        return genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)]),
                    finish_reason=genai_types.FinishReason(finish_reason),
                )
            ],
            **extra,
        )

    def test_a_normal_stop_returns_the_text(self, monkeypatch):
        client = self._retrying_client()
        _stub_gemini(monkeypatch, client, [self._sdk_response('{"ok": 1}', "STOP")])
        assert client.generate_json("p") == '{"ok": 1}'

    def test_truncation_at_the_token_limit_is_reported_and_not_retried(self, monkeypatch):
        client = self._retrying_client()
        models = _stub_gemini(
            monkeypatch, client, [self._sdk_response('{"essential_skills": ["Py', "MAX_TOKENS")]
        )
        with pytest.raises(LlmInvocationError, match="MAX_TOKENS") as exc:
            client.generate_json("p", purpose="skill_ranker")
        assert "RERANKER_LLM_MAX_OUTPUT_TOKENS" in str(exc.value)
        assert "skill_ranker" in str(exc.value)
        assert len(models.calls) == 1

    def test_a_safety_stop_names_its_reason(self, monkeypatch):
        client = self._retrying_client()
        _stub_gemini(monkeypatch, client, [self._sdk_response("", "SAFETY")])
        with pytest.raises(LlmInvocationError, match="finish_reason=SAFETY"):
            client.generate_json("p")

    def test_a_blocked_prompt_names_its_reason(self, monkeypatch):
        from google.genai import types as genai_types

        client = self._retrying_client()
        blocked = genai_types.GenerateContentResponse(
            prompt_feedback=genai_types.GenerateContentResponsePromptFeedback(
                block_reason=genai_types.BlockedReason.PROHIBITED_CONTENT
            )
        )
        _stub_gemini(monkeypatch, client, [blocked])
        with pytest.raises(LlmInvocationError, match="blocked .PROHIBITED_CONTENT."):
            client.generate_json("p")

    def test_thought_parts_are_never_returned_as_the_answer(self):
        from google.genai import types as genai_types

        from app.llm.providers.gemini import _response_text

        response = genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    content=genai_types.Content(
                        role="model",
                        parts=[
                            genai_types.Part(text="let me think", thought=True),
                            genai_types.Part(text='{"ok": 1}'),
                        ],
                    ),
                    finish_reason=genai_types.FinishReason.STOP,
                )
            ]
        )
        assert _response_text(response) == '{"ok": 1}'

    def test_empty_completion_is_an_error_not_an_empty_string(self, monkeypatch):
        client = GeminiLlmClient(
            LlmConfig(
                provider="gemini",
                client_class=PROVIDER_CLASS_PATHS["gemini"],
                model="gemini-2.5-flash",
                api_key="k",
            )
        )
        _stub_gemini(monkeypatch, client, ["   "])
        with pytest.raises(LlmInvocationError, match="Empty response"):
            client.generate_json("p")


# ---------------------------------------------------------------------------
# HTTP providers (transport stubbed)
# ---------------------------------------------------------------------------


def _stub_http(monkeypatch, client, handler) -> list[httpx.Request]:
    """Route the client's httpx pool through ``handler``; return the captured requests."""
    seen: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    transport = httpx.MockTransport(_handler)
    monkeypatch.setattr(client, "_client", lambda: httpx.Client(transport=transport))
    monkeypatch.setattr("app.llm.retry.time.sleep", lambda _s: None)
    return seen


def _json_response(status: int, payload: dict) -> httpx.Response:
    return httpx.Response(status, json=payload)


class TestAnthropicClient:
    def _client(self, **overrides) -> AnthropicLlmClient:
        cfg = LlmConfig(
            provider="anthropic",
            client_class=PROVIDER_CLASS_PATHS["anthropic"],
            model="claude-opus-5",
            api_key="sk-ant",
            **overrides,
        )
        return AnthropicLlmClient(cfg)

    def test_posts_a_messages_request_with_auth_and_version(self, monkeypatch):
        client = self._client()
        seen = _stub_http(
            monkeypatch,
            client,
            lambda r: _json_response(200, {"content": [{"type": "text", "text": '{"ok": 1}'}]}),
        )
        assert client.generate_json("prompt", purpose="job_attributes") == '{"ok": 1}'
        (request,) = seen
        assert str(request.url) == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == "sk-ant"
        assert request.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(request.content)
        assert body["model"] == "claude-opus-5"
        assert body["temperature"] == 0.2
        assert body["max_tokens"] > 0
        assert body["messages"] == [{"role": "user", "content": "prompt"}]
        assert "JSON" in body["system"]

    def test_text_blocks_are_joined_and_thinking_blocks_ignored(self, monkeypatch):
        client = self._client()
        _stub_http(
            monkeypatch,
            client,
            lambda r: _json_response(
                200,
                {
                    "content": [
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "text", "text": '{"a":'},
                        {"type": "text", "text": " 1}"},
                    ]
                },
            ),
        )
        assert json.loads(client.generate_json("p")) == {"a": 1}

    def test_options_override_max_tokens_and_pass_extras_through(self, monkeypatch):
        client = self._client(max_output_tokens=1234, options_json='{"top_k": 7}')
        seen = _stub_http(
            monkeypatch,
            client,
            lambda r: _json_response(200, {"content": [{"type": "text", "text": "{}"}]}),
        )
        client.generate_json("p")
        body = json.loads(seen[0].content)
        assert body["max_tokens"] == 1234
        assert body["top_k"] == 7

    def test_bad_key_is_a_configuration_error_and_is_not_retried(self, monkeypatch):
        client = self._client(max_attempts=5)
        seen = _stub_http(
            monkeypatch,
            client,
            lambda r: _json_response(401, {"error": {"message": "invalid x-api-key"}}),
        )
        with pytest.raises(LlmConfigurationError) as exc:
            client.generate_json("p")
        assert "invalid x-api-key" in str(exc.value)
        assert len(seen) == 1

    def test_rate_limit_is_retried_then_succeeds(self, monkeypatch):
        client = self._client(max_attempts=3)
        responses = [
            _json_response(429, {"error": {"message": "rate limited"}}),
            _json_response(200, {"content": [{"type": "text", "text": "{}"}]}),
        ]
        seen = _stub_http(monkeypatch, client, lambda r: responses.pop(0))
        assert client.generate_json("p") == "{}"
        assert len(seen) == 2

    def test_truncation_before_any_text_is_reported_actionably(self, monkeypatch):
        client = self._client()
        _stub_http(
            monkeypatch,
            client,
            lambda r: _json_response(200, {"content": [], "stop_reason": "max_tokens"}),
        )
        with pytest.raises(LlmInvocationError, match="RERANKER_LLM_MAX_OUTPUT_TOKENS"):
            client.generate_json("p")

    def test_missing_key_names_the_env_var(self):
        cfg = LlmConfig(
            provider="anthropic",
            client_class=PROVIDER_CLASS_PATHS["anthropic"],
            model="claude-opus-5",
            api_key="",
        )
        with pytest.raises(LlmConfigurationError, match="ANTHROPIC_API_KEY"):
            build_llm_client(cfg)


class TestOpenAiClient:
    def _client(self, cls=OpenAiLlmClient, **overrides):
        cfg = LlmConfig(
            provider="openai",
            client_class=PROVIDER_CLASS_PATHS["openai"],
            model="gpt-4.1",
            api_key="sk-oai",
            **overrides,
        )
        return cls(cfg)

    def test_posts_chat_completions_in_json_mode(self, monkeypatch):
        client = self._client()
        seen = _stub_http(
            monkeypatch,
            client,
            lambda r: _json_response(200, {"choices": [{"message": {"content": '{"ok": 1}'}}]}),
        )
        assert client.generate_json("prompt") == '{"ok": 1}'
        (request,) = seen
        assert str(request.url) == "https://api.openai.com/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer sk-oai"
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert body["temperature"] == 0.2
        assert body["messages"] == [{"role": "user", "content": "prompt"}]

    def test_json_mode_can_be_disabled_for_servers_that_reject_it(self, monkeypatch):
        client = self._client(options_json='{"json_mode": false}')
        seen = _stub_http(
            monkeypatch,
            client,
            lambda r: _json_response(200, {"choices": [{"message": {"content": "{}"}}]}),
        )
        client.generate_json("p")
        assert "response_format" not in json.loads(seen[0].content)

    def test_temperature_can_be_omitted_for_reasoning_models(self, monkeypatch):
        client = self._client(
            options_json='{"send_temperature": false, "max_tokens_field": "max_completion_tokens"}',
            max_output_tokens=999,
        )
        seen = _stub_http(
            monkeypatch,
            client,
            lambda r: _json_response(200, {"choices": [{"message": {"content": "{}"}}]}),
        )
        client.generate_json("p")
        body = json.loads(seen[0].content)
        assert "temperature" not in body
        assert body["max_completion_tokens"] == 999

    def test_length_truncation_is_reported_actionably(self, monkeypatch):
        client = self._client()
        _stub_http(
            monkeypatch,
            client,
            lambda r: _json_response(200, {"choices": [{"finish_reason": "length"}]}),
        )
        with pytest.raises(LlmInvocationError, match="RERANKER_LLM_MAX_OUTPUT_TOKENS"):
            client.generate_json("p")

    def test_unknown_model_is_a_configuration_error(self, monkeypatch):
        client = self._client()
        _stub_http(
            monkeypatch,
            client,
            lambda r: _json_response(404, {"error": {"message": "model not found"}}),
        )
        with pytest.raises(LlmConfigurationError, match="model not found"):
            client.generate_json("p")

    def test_non_json_body_is_reported_with_an_excerpt(self, monkeypatch):
        client = self._client()
        _stub_http(monkeypatch, client, lambda r: httpx.Response(200, text="<html>gateway</html>"))
        with pytest.raises(LlmInvocationError, match="non-JSON response body"):
            client.generate_json("p")


class TestOpenAiCompatibleClient:
    def test_local_server_needs_no_api_key(self, monkeypatch):
        client = OpenAiCompatibleLlmClient(
            LlmConfig(
                provider="openai_compatible",
                client_class=PROVIDER_CLASS_PATHS["openai_compatible"],
                model="qwen3-32b",
                api_key="",
                base_url="http://localhost:8000/v1",
            )
        )
        seen = _stub_http(
            monkeypatch,
            client,
            lambda r: _json_response(200, {"choices": [{"message": {"content": "{}"}}]}),
        )
        assert client.generate_json("p") == "{}"
        (request,) = seen
        assert str(request.url) == "http://localhost:8000/v1/chat/completions"
        assert "authorization" not in request.headers

    def test_missing_base_url_says_what_to_set(self):
        cfg = LlmConfig(
            provider="openai_compatible",
            client_class=PROVIDER_CLASS_PATHS["openai_compatible"],
            model="qwen3-32b",
        )
        with pytest.raises(LlmConfigurationError, match="RERANKER_LLM_BASE_URL"):
            build_llm_client(cfg)

    def test_base_url_without_a_scheme_is_rejected(self):
        cfg = LlmConfig(
            provider="openai_compatible",
            client_class=PROVIDER_CLASS_PATHS["openai_compatible"],
            model="qwen3-32b",
            base_url="localhost:8000/v1",
        )
        with pytest.raises(LlmConfigurationError, match="http://"):
            build_llm_client(cfg)


class TestOllamaClient:
    def _client(self, **overrides) -> OllamaLlmClient:
        cfg = LlmConfig(
            provider="ollama",
            client_class=PROVIDER_CLASS_PATHS["ollama"],
            model="qwen3:32b",
            **overrides,
        )
        return OllamaLlmClient(cfg)

    def _ok(self, content: str = "{}") -> httpx.Response:
        return _json_response(200, {"message": {"role": "assistant", "content": content}})

    def test_posts_the_native_chat_endpoint_with_json_format(self, monkeypatch):
        client = self._client()
        seen = _stub_http(monkeypatch, client, lambda r: self._ok('{"ok": 1}'))
        assert client.generate_json("prompt", purpose="skill_ranker") == '{"ok": 1}'
        (request,) = seen
        assert str(request.url) == "http://localhost:11434/api/chat"
        body = json.loads(request.content)
        assert body["model"] == "qwen3:32b"
        assert body["messages"] == [{"role": "user", "content": "prompt"}]
        assert body["stream"] is False
        assert body["format"] == "json"
        assert body["options"] == {"temperature": 0.2}

    def test_local_server_needs_no_api_key(self, monkeypatch):
        client = self._client()
        seen = _stub_http(monkeypatch, client, lambda r: self._ok())
        client.generate_json("p")
        assert "authorization" not in seen[0].headers

    def test_a_configured_key_is_sent_as_a_bearer_token(self, monkeypatch):
        client = self._client(api_key="ollama-cloud-key", base_url="https://ollama.com")
        seen = _stub_http(monkeypatch, client, lambda r: self._ok())
        client.generate_json("p")
        assert str(seen[0].url) == "https://ollama.com/api/chat"
        assert seen[0].headers["authorization"] == "Bearer ollama-cloud-key"

    def test_openai_compat_base_url_is_tolerated(self, monkeypatch):
        """Operators paste the /v1 URL from the compat docs; the native API is one level up."""
        client = self._client(base_url="http://ollama.internal:11434/v1")
        seen = _stub_http(monkeypatch, client, lambda r: self._ok())
        client.generate_json("p")
        assert str(seen[0].url) == "http://ollama.internal:11434/api/chat"

    def test_token_cap_becomes_num_predict_and_extras_land_in_options(self, monkeypatch):
        client = self._client(
            max_output_tokens=4096,
            options_json='{"top_p": 0.9, "num_ctx": 8192, "keep_alive": "10m"}',
        )
        seen = _stub_http(monkeypatch, client, lambda r: self._ok())
        client.generate_json("p")
        body = json.loads(seen[0].content)
        assert body["options"] == {
            "temperature": 0.2,
            "num_predict": 4096,
            "top_p": 0.9,
            "num_ctx": 8192,
        }
        # keep_alive is a request field, not a runtime option.
        assert body["keep_alive"] == "10m"

    def test_format_can_be_a_schema_or_disabled(self, monkeypatch):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        client = self._client(options_json=json.dumps({"format": schema}))
        seen = _stub_http(monkeypatch, client, lambda r: self._ok())
        client.generate_json("p")
        assert json.loads(seen[0].content)["format"] == schema

        off = self._client(options_json='{"format": null}')
        seen = _stub_http(monkeypatch, off, lambda r: self._ok())
        off.generate_json("p")
        assert "format" not in json.loads(seen[0].content)

    def test_a_model_that_was_never_pulled_is_a_configuration_error(self, monkeypatch):
        client = self._client(max_attempts=5)
        seen = _stub_http(
            monkeypatch,
            client,
            lambda r: _json_response(404, {"error": 'model "qwen3:32b" not found, try pulling it'}),
        )
        with pytest.raises(LlmConfigurationError, match="try pulling it"):
            client.generate_json("p")
        assert len(seen) == 1

    def test_truncation_before_any_text_is_reported_actionably(self, monkeypatch):
        client = self._client()
        _stub_http(
            monkeypatch,
            client,
            lambda r: _json_response(200, {"message": {"content": ""}, "done_reason": "length"}),
        )
        with pytest.raises(LlmInvocationError, match="RERANKER_LLM_MAX_OUTPUT_TOKENS"):
            client.generate_json("p")

    def test_a_response_from_something_that_is_not_ollama_says_so(self, monkeypatch):
        client = self._client()
        _stub_http(monkeypatch, client, lambda r: _json_response(200, {"choices": []}))
        with pytest.raises(LlmInvocationError, match="RERANKER_LLM_BASE_URL"):
            client.generate_json("p")

    def test_requires_a_model_id(self):
        cfg = LlmConfig(
            provider="ollama",
            client_class=PROVIDER_CLASS_PATHS["ollama"],
            model="",
        )
        with pytest.raises(LlmConfigurationError, match="RERANKER_LLM_MODEL"):
            build_llm_client(cfg)

    def test_a_cold_local_model_gets_a_longer_default_timeout(self):
        assert self._client().timeout_s > 120.0
        assert self._client(timeout_s=30.0).timeout_s == 30.0


class TestQwenClient:
    def _client(self, **overrides) -> QwenLlmClient:
        cfg = LlmConfig(
            provider="qwen",
            client_class=PROVIDER_CLASS_PATHS["qwen"],
            model="qwen-plus",
            api_key="sk-dashscope",
            **overrides,
        )
        return QwenLlmClient(cfg)

    def _ok(self, content: str = "{}") -> httpx.Response:
        return _json_response(200, {"choices": [{"message": {"content": content}}]})

    def test_posts_dashscope_compatible_mode_with_thinking_off(self, monkeypatch):
        client = self._client()
        seen = _stub_http(monkeypatch, client, lambda r: self._ok('{"ok": 1}'))
        assert client.generate_json("prompt", purpose="job_attributes") == '{"ok": 1}'
        (request,) = seen
        assert str(request.url) == (
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
        )
        assert request.headers["authorization"] == "Bearer sk-dashscope"
        body = json.loads(request.content)
        assert body["model"] == "qwen-plus"
        assert body["temperature"] == 0.2
        assert body["response_format"] == {"type": "json_object"}
        # A non-streaming DashScope call rejects qwen3's default thinking mode.
        assert body["enable_thinking"] is False

    def test_thinking_can_be_re_enabled_or_omitted(self, monkeypatch):
        on = self._client(options_json='{"enable_thinking": true}')
        seen = _stub_http(monkeypatch, on, lambda r: self._ok())
        on.generate_json("p")
        assert json.loads(seen[0].content)["enable_thinking"] is True

        off = self._client(options_json='{"enable_thinking": null}')
        seen = _stub_http(monkeypatch, off, lambda r: self._ok())
        off.generate_json("p")
        assert "enable_thinking" not in json.loads(seen[0].content)

    def test_the_beijing_region_is_a_base_url_override(self, monkeypatch):
        client = self._client(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        seen = _stub_http(monkeypatch, client, lambda r: self._ok())
        client.generate_json("p")
        assert str(seen[0].url).startswith("https://dashscope.aliyuncs.com/compatible-mode/v1")

    def test_missing_key_names_the_dashscope_var(self):
        cfg = LlmConfig(
            provider="qwen",
            client_class=PROVIDER_CLASS_PATHS["qwen"],
            model="qwen-plus",
            api_key="",
        )
        with pytest.raises(LlmConfigurationError, match="DASHSCOPE_API_KEY"):
            build_llm_client(cfg)

    def test_bad_key_is_a_configuration_error_and_is_not_retried(self, monkeypatch):
        client = self._client(max_attempts=5)
        seen = _stub_http(
            monkeypatch,
            client,
            lambda r: _json_response(401, {"error": {"message": "Incorrect API key"}}),
        )
        with pytest.raises(LlmConfigurationError, match="Incorrect API key"):
            client.generate_json("p")
        assert len(seen) == 1
