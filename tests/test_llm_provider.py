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

import httpx
import pytest

from app.config import Settings
from app.llm import (
    PROVIDER_CLASS_PATHS,
    LlmClient,
    LlmConfig,
    LlmConfigurationError,
    LlmInvocationError,
    build_llm_client,
    get_llm_client,
    load_llm_client_class,
    reset_llm_clients,
)
from app.llm.providers.anthropic import AnthropicLlmClient
from app.llm.providers.gemini import GeminiLlmClient
from app.llm.providers.openai import OpenAiCompatibleLlmClient, OpenAiLlmClient

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
    base = dict(
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
        assert s.llm.model == "gemini-2.5-flash"
        assert s.llm.model == s.gemini_model
        assert s.llm.temperature == 0.2

    def test_gemini_model_env_still_selects_the_model(self, make_settings):
        s = make_settings(GEMINI_MODEL="gemini-3-pro-preview")
        assert s.llm.model == "gemini-3-pro-preview"

    def test_vertex_model_is_the_second_gemini_fallback(self, make_settings):
        s = make_settings(VERTEX_MODEL="gemini-2.0-flash")
        assert s.llm.model == "gemini-2.0-flash"

    def test_gemini_api_key_env_feeds_the_generic_config(self, make_settings):
        s = make_settings(GEMINI_API_KEY="abc123")
        assert s.llm.api_key == "abc123"
        assert s.llm.api_key_tail() == "c123"

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


class TestClientCache:
    def test_same_config_returns_the_same_instance(self):
        reset_llm_clients()
        cfg = _config()
        assert get_llm_client(cfg) is get_llm_client(cfg)

    def test_different_config_builds_a_separate_instance(self):
        reset_llm_clients()
        a = get_llm_client(_config(model="a"))
        b = get_llm_client(_config(model="b"))
        assert a is not b
        assert (a.model, b.model) == ("a", "b")

    def test_reset_drops_the_cache(self):
        reset_llm_clients()
        cfg = _config()
        first = get_llm_client(cfg)
        reset_llm_clients()
        assert get_llm_client(cfg) is not first


# ---------------------------------------------------------------------------
# The seam: grading code calls whatever was configured
# ---------------------------------------------------------------------------


class TestGradingCodeUsesTheConfiguredModel:
    def test_skill_ranker_call_reaches_a_custom_client(self, make_settings):
        """``call_skill_ranker_model`` is provider-agnostic — no Gemini import involved."""
        from app.llm_service import PURPOSE_SKILL_RANKER, call_skill_ranker_model

        RecordingLlmClient.seen = []
        s = make_settings(
            RERANKER_LLM_CLIENT_CLASS="tests.test_llm_provider:RecordingLlmClient",
            RERANKER_LLM_MODEL="my-model",
        )
        out = call_skill_ranker_model("rank these skills", settings=s)
        assert json.loads(out)["essential_skills"] == ["welding"]
        assert RecordingLlmClient.seen == [(PURPOSE_SKILL_RANKER, "rank these skills")]

    def test_each_pass_labels_itself(self, make_settings):
        from app.llm_service import (
            PURPOSE_JOB_ATTRIBUTES,
            PURPOSE_OCCUPATION_SKILLS_RELEVANCE,
            PURPOSE_SKILL_RANKER,
            call_job_attributes_model,
            call_occupation_skills_relevance_model,
            call_skill_ranker_model,
        )

        RecordingLlmClient.seen = []
        s = make_settings(
            RERANKER_LLM_CLIENT_CLASS="tests.test_llm_provider:RecordingLlmClient",
            RERANKER_LLM_MODEL="my-model",
        )
        call_skill_ranker_model("p1", settings=s)
        call_job_attributes_model("p2", settings=s)
        call_occupation_skills_relevance_model("p3", settings=s)
        assert [p for p, _ in RecordingLlmClient.seen] == [
            PURPOSE_SKILL_RANKER,
            PURPOSE_JOB_ATTRIBUTES,
            PURPOSE_OCCUPATION_SKILLS_RELEVANCE,
        ]

    def test_ranked_document_metadata_records_the_configured_model(self, make_settings):
        from app.ranked_job_schema import build_llm_reranker_meta

        s = make_settings(
            RERANKER_LLM_PROVIDER="anthropic",
            RERANKER_LLM_MODEL="claude-opus-5",
            ANTHROPIC_API_KEY="sk-ant",
        )
        meta = build_llm_reranker_meta(s)
        assert meta["llm_provider"] == "anthropic"
        assert meta["skill_ranker_model"] == "claude-opus-5"
        assert meta["job_attributes_model"] == "claude-opus-5"
        assert meta["occupation_skills_relevance_model"] == "claude-opus-5"

    def test_gemini_metadata_is_unchanged_from_the_legacy_path(self, make_settings):
        """No-regression guard: the three model fields still carry the Gemini model id."""
        from app.ranked_job_schema import build_llm_reranker_meta

        s = make_settings(GEMINI_API_KEY="k", GEMINI_MODEL="gemini-2.5-flash")
        meta = build_llm_reranker_meta(s)
        assert meta["skill_ranker_model"] == "gemini-2.5-flash"
        assert meta["occupation_skills_relevance_model"] == "gemini-2.5-flash"
        assert meta["job_attributes_model"] == "gemini-2.5-flash"


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
        return _FakeGeminiResponse(outcome)


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
