"""The provider-agnostic seam between the reranker and whatever model backs it.

The reranker asks a model for exactly one thing, three times per job (skill ranking,
job attributes, occupation–skill relevance): *given this prompt, return a JSON
document*. That single operation is the whole interface. Everything
provider-specific — SDK or REST shape, auth, retry policy, response unwrapping —
lives behind :class:`LlmClient`, so swapping Gemini for Claude, GPT, Qwen or a
locally hosted model is a configuration change and touches no grading or
enrichment code.

To add a provider: subclass :class:`LlmClient`, implement ``generate_json``, and
point ``RERANKER_LLM_CLIENT_CLASS`` at it (``my_pkg.my_module:MyLlmClient``). No
registration step, no edit to this file — the class path *is* the registration.
The aliases in :data:`PROVIDER_CLASS_PATHS` are only shorthand for the
implementations that ship with the repo.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import Field

from app.domain_base import DomainModel
from app.llm.errors import LlmConfigurationError

# Short ``RERANKER_LLM_PROVIDER`` names → dotted path of the shipped implementation.
# Anything not listed here can still be selected by its full class path.
PROVIDER_CLASS_PATHS: dict[str, str] = {
    "gemini": "app.llm.providers.gemini:GeminiLlmClient",
    "anthropic": "app.llm.providers.anthropic:AnthropicLlmClient",
    "openai": "app.llm.providers.openai:OpenAiLlmClient",
    # Hosted Qwen through Alibaba DashScope's OpenAI-compatible mode.
    "qwen": "app.llm.providers.qwen:QwenLlmClient",
    # A local (or Ollama Cloud) Ollama, over its native /api/chat.
    "ollama": "app.llm.providers.ollama:OllamaLlmClient",
    # Any other OpenAI-Chat-Completions-compatible server: vLLM, SGLang, LM Studio,
    # llama.cpp, Together, Groq, … — set RERANKER_LLM_BASE_URL.
    "openai_compatible": "app.llm.providers.openai:OpenAiCompatibleLlmClient",
}

# Provider alias → env vars consulted for the API key when ``RERANKER_LLM_API_KEY``
# is unset. Keeps existing deployments working: a Gemini deployment that only sets
# ``GEMINI_API_KEY``/``API_KEY`` needs no new variable.
PROVIDER_API_KEY_ENV_VARS: dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY", "API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "qwen": ("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
    # A local Ollama has no auth; Ollama Cloud / a proxied instance takes a key.
    "ollama": ("OLLAMA_API_KEY",),
    "openai_compatible": (),
}

# Provider alias → env vars consulted for the API endpoint when
# ``RERANKER_LLM_BASE_URL`` is unset. ``OLLAMA_HOST`` is the variable the Ollama
# CLI itself reads, so a host that already runs Ollama elsewhere needs no new one;
# it may omit the scheme (``127.0.0.1:11434``), which :mod:`app.config` adds.
PROVIDER_BASE_URL_ENV_VARS: dict[str, tuple[str, ...]] = {
    "ollama": ("OLLAMA_BASE_URL", "OLLAMA_HOST"),
}

# Provider alias → model used when neither ``RERANKER_LLM_MODEL`` nor the provider's own
# variable (Gemini: ``GEMINI_MODEL`` / ``VERTEX_MODEL``) names one. The single place a default
# model lives. Only Gemini, the historical backend, has one: for any other provider a
# missing model id is a configuration error, not a silent fallback.
PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
}

# Provider name recorded for a selection made purely by class path.
CUSTOM_PROVIDER = "custom"

# Matches the temperature the Gemini call used before this abstraction existed;
# changing it would change model output, so it stays the default for every provider
# and model except those listed in :func:`default_temperature`.
DEFAULT_TEMPERATURE = 0.2

# Gemini 3 and later are tuned for their default temperature of 1.0; Google's guidance
# is that lowering it can cause looping or degraded reasoning.
GEMINI_3_TEMPERATURE = 1.0

_GEMINI_MAJOR_VERSION = re.compile(r"^(?:models/)?gemini-(\d+)")

# Values ``RERANKER_LLM_THINKING_LEVEL`` accepts (Gemini 3+ ``thinking_level``).
THINKING_LEVELS = ("minimal", "low", "medium", "high")


def default_temperature(provider: str, model: str) -> float:
    """Temperature used when ``RERANKER_LLM_TEMPERATURE`` is unset.

    :data:`GEMINI_3_TEMPERATURE` for Gemini models of major version 3 or later
    (``gemini-3-pro-preview``, ``gemini-3.5-flash``, …), :data:`DEFAULT_TEMPERATURE`
    for everything else. Aliases without a version (``gemini-flash-latest``) get the
    historical default; set the variable explicitly for those.
    see: https://ai.google.dev/gemini-api/docs/gemini-3
    """
    if normalise_provider(provider) == "gemini":
        m = _GEMINI_MAJOR_VERSION.match((model or "").strip().lower())
        if m and int(m.group(1)) >= 3:
            return GEMINI_3_TEMPERATURE
    return DEFAULT_TEMPERATURE


def normalise_provider(raw: str | None) -> str:
    """Lowercase provider alias with ``-``/space folded to ``_`` (``"Open AI"`` → ``open_ai``)."""
    t = (raw or "").strip().lower()
    return t.replace("-", "_").replace(" ", "_")


def resolve_provider_class_path(provider: str, *, raw: str | None = None) -> str:
    """Dotted class path for a provider alias, or a clear error listing the known ones.

    ``raw`` is echoed in the error message when given, so the operator sees the value
    they actually typed rather than its normalised form.
    """
    key = normalise_provider(provider)
    try:
        return PROVIDER_CLASS_PATHS[key]
    except KeyError:
        known = ", ".join(sorted(PROVIDER_CLASS_PATHS))
        raise LlmConfigurationError(
            f"Unknown RERANKER_LLM_PROVIDER={(raw if raw is not None else provider)!r}. "
            f"Known providers: {known}. "
            f"To use a provider that does not ship with this repo, leave "
            f"RERANKER_LLM_PROVIDER unset and set RERANKER_LLM_CLIENT_CLASS to the "
            f"dotted path of an app.llm.base.LlmClient subclass "
            f"(e.g. my_pkg.my_module:MyLlmClient)."
        ) from None


class LlmConfig(DomainModel):
    """Resolved, provider-agnostic model selection for one run.

    Built once by :meth:`app.config.Settings.from_env` and carried on
    ``Settings.llm``; frozen and hashable so it can key the process-wide client
    cache in :mod:`app.llm.factory`.

    ``0`` / ``""`` means "let the provider decide" for the optional numeric and
    URL fields, so a config that sets nothing beyond provider and model behaves
    exactly as that provider's implementation intends.
    """

    provider: str = Field(
        description="Alias the deployment asked for, or `custom` when only a class path was given."
    )
    client_class: str = Field(
        description="Dotted path (`module:Class` or `module.Class`) of the LlmClient to load."
    )
    model: str = Field(
        description="Provider-specific model id, e.g. `gemini-2.5-flash`, `claude-opus-5`, `qwen3:32b`."
    )
    api_key: str = Field(default="", description="API key for the provider, when required.")
    base_url: str = Field(
        default="",
        description="Override the provider's API host — required for self-hosted servers.",
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE, description="Sampling temperature for all calls."
    )
    max_output_tokens: int = Field(
        default=0,
        description="Max output tokens; `0` = provider default (Gemini has none; Anthropic requires "
        "a value, see its client).",
    )
    timeout_s: float = Field(
        default=0.0, description="Request timeout in seconds; `0` = provider/SDK default."
    )
    max_attempts: int = Field(
        default=5,
        description="Total attempts per call, including the first, for retryable transport failures.",
    )
    options_json: str = Field(
        default="",
        description='Provider-specific extras as a JSON object string (e.g. `{"top_p": 0.9}`).',
    )
    thinking_level: str = Field(
        default="",
        description="Reasoning depth for models that take a level (`THINKING_LEVELS`); `\"\"` = the "
        "model's own default. Only the Gemini client reads it.",
    )
    response_schema: bool = Field(
        default=True,
        description="Send each pass's JSON Schema to clients that support structured output "
        "(`LlmClient.supports_response_schema`). Off = prose contract + JSON mode only.",
    )

    def options(self) -> dict[str, Any]:
        """Parse ``options_json`` into a dict; a malformed value is a configuration error."""
        raw = (self.options_json or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LlmConfigurationError(
                f"RERANKER_LLM_OPTIONS is not valid JSON ({e}). "
                f'Expected a JSON object, e.g. {{"top_p": 0.9}}. Got: {raw[:200]!r}'
            ) from e
        if not isinstance(parsed, dict):
            raise LlmConfigurationError(
                f"RERANKER_LLM_OPTIONS must be a JSON object (key/value map), "
                f"got {type(parsed).__name__}: {raw[:200]!r}"
            )
        return parsed

    def require_model(self) -> str:
        """Model id, or a configuration error naming the variable to set."""
        m = (self.model or "").strip()
        if not m:
            raise LlmConfigurationError(
                f"No model id configured for provider {self.provider!r}. "
                f"Set RERANKER_LLM_MODEL to a model this provider serves."
            )
        return m

    def require_api_key(self, *, env_hint: str) -> str:
        """API key, or a configuration error naming the variable to set."""
        k = (self.api_key or "").strip()
        if not k:
            raise LlmConfigurationError(
                f"No API key configured for provider {self.provider!r}. "
                f"Set {env_hint} (or RERANKER_LLM_API_KEY) in .env."
            )
        return k

    def describe(self) -> str:
        """Compact ``provider/model`` label for log lines."""
        return f"{self.provider}/{self.model or '(no model)'}"

    def api_key_tail(self) -> str:
        """Last 4 characters of the key, for logging that must not leak the secret."""
        return self.api_key[-4:] if self.api_key else "?"


class LlmClient(ABC):
    """One configured model, ready to answer prompts with JSON.

    Implementations are constructed once per process by the composition root
    (the ``get_*_service`` providers) and passed to the services that use them, so they may hold
    connection pools. They must be safe to call from several threads at once —
    the batch CLI runs up to ``RERANKER_PARALLEL_WORKERS`` jobs concurrently and
    overlaps two calls within a single job.

    Contract for ``generate_json``:

    * return the model's raw text — the caller parses it (fenced ```json blocks
      are tolerated downstream, so stripping them here is optional);
    * never return an empty/whitespace-only string — raise instead;
    * raise :class:`~app.llm.errors.LlmConfigurationError` for anything a retry
      cannot fix (bad credentials, unknown model id) and
      :class:`~app.llm.errors.LlmInvocationError` for a failed call.

    Structured output is opt-in: a client that sets ``supports_response_schema`` is
    called with ``response_schema=`` (a JSON Schema dict, or ``None``) and should
    constrain decoding to it. Clients that leave it ``False`` keep the two-argument
    ``generate_json`` signature and are never passed a schema.
    """

    #: Whether ``generate_json`` accepts a ``response_schema`` keyword (see above).
    supports_response_schema: bool = False

    def __init__(self, config: LlmConfig) -> None:
        self.config = config

    @property
    def model(self) -> str:
        """Model id this client is bound to — recorded in ``classifier_metadata``."""
        return self.config.model

    @abstractmethod
    def generate_json(self, prompt: str, *, purpose: str = "") -> str:
        """Send ``prompt``, return the raw JSON text.

        ``purpose`` is a free-form label for the reranker pass making the call
        (``skill_ranker``, ``job_attributes``, ``occupation_skills_relevance``);
        implementations use it only for logs and error messages.
        """

    def close(self) -> None:  # noqa: B027 — intentional no-op default, not required to override
        """Release sockets / clients. Default is a no-op; override when needed."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(provider={self.config.provider!r}, model={self.model!r})"
