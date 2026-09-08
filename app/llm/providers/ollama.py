from __future__ import annotations

from typing import Any, Mapping

from app.llm.base import LlmConfig
from app.llm.errors import LlmInvocationError
from app.llm.providers._http import HttpJsonLlmClient

# Ollama's own default listen address. A deployment that runs it as a sidecar or
# behind a proxy overrides this with RERANKER_LLM_BASE_URL / OLLAMA_HOST.
DEFAULT_BASE_URL = "http://localhost:11434"

# A local model is often cold: the first call pays for loading weights off disk,
# and generation runs at a fraction of a hosted model's tokens/s. The shared
# 120 s default times out on a 32B model's first request, so this is 10 minutes.
DEFAULT_TIMEOUT_S = 600.0

# Body fields of ``/api/chat`` that are *not* runtime options. Everything else in
# RERANKER_LLM_OPTIONS is a sampling/runtime knob and belongs in the nested
# ``options`` object, which is where Ollama looks for top_p, num_ctx, seed, ….
TOP_LEVEL_OPTION_KEYS = ("keep_alive", "think", "tools")


class OllamaLlmClient(HttpJsonLlmClient):
    """Ollama JSON completions (``POST /api/chat``).

    Talks Ollama's native API rather than its OpenAI-compatible shim, so
    ``format`` (constrained JSON decoding) and the nested ``options`` block are
    available — the shim exposes neither. For a Qwen, Llama or Mistral model
    pulled into a local Ollama this is the provider to use; point
    :class:`~app.llm.providers.qwen.QwenLlmClient` at hosted Qwen instead.

    No API key is required, because a local Ollama has no auth. One is sent as a
    bearer token when configured, which covers Ollama Cloud
    (``RERANKER_LLM_BASE_URL=https://ollama.com``) and reverse-proxied instances.
    """

    provider_label = "Ollama"
    api_key_env_hint = "RERANKER_LLM_API_KEY"
    default_base_url = DEFAULT_BASE_URL
    default_timeout_s = DEFAULT_TIMEOUT_S
    requires_api_key = False

    def __init__(self, config: LlmConfig) -> None:
        super().__init__(config)
        options = dict(config.options())
        # ``"json"`` is free-form JSON; a JSON-schema object constrains the decode.
        # ``null`` / ``false`` sends no format at all, for a server that rejects it.
        self.response_format = options.pop("format", "json")
        self.extra_body = {k: options.pop(k) for k in TOP_LEVEL_OPTION_KEYS if k in options}
        # Whatever is left is a runtime option (top_p, top_k, num_ctx, seed, …).
        self.runtime_options = options

    def _resolve_base_url(self, config: LlmConfig) -> str:
        """Base URL, tolerating the ``/v1`` suffix from Ollama's OpenAI-compat docs.

        ``http://localhost:11434/v1`` is the URL every OpenAI-compatible example
        shows, so operators paste it here too; the native endpoints live one level
        up and would 404 with it left on.
        """
        base = super()._resolve_base_url(config)
        return base[: -len("/v1")].rstrip("/") if base.endswith("/v1") else base

    def endpoint(self) -> str:
        return f"{self.base_url}/api/chat"

    def headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        if self.api_key:
            h["authorization"] = f"Bearer {self.api_key}"
        return h

    def request_body(self, prompt: str) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": self.config.temperature}
        if self.config.max_output_tokens > 0:
            options["num_predict"] = self.config.max_output_tokens
        options.update(self.runtime_options)
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            # The reranker reads one whole document per call; streaming would only
            # add reassembly for no benefit.
            "stream": False,
            "options": options,
        }
        if self.response_format:
            body["format"] = self.response_format
        body.update(self.extra_body)
        return body

    def extract_text(self, payload: Mapping[str, Any]) -> str:
        message = payload.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                if not content.strip() and payload.get("done_reason") == "length":
                    raise LlmInvocationError(
                        "Ollama stopped at the token limit before emitting any text — "
                        "raise RERANKER_LLM_MAX_OUTPUT_TOKENS, or the model's num_ctx "
                        'via RERANKER_LLM_OPTIONS={"num_ctx": 8192}.'
                    )
                return content
            raise LlmInvocationError(
                f"Ollama message.content is a {type(content).__name__}, expected a string."
            )
        # ``/api/generate`` shape, in case the base URL points at a proxy that maps to it.
        legacy = payload.get("response")
        if isinstance(legacy, str):
            return legacy
        raise LlmInvocationError(
            f"Ollama response has no message.content (keys: {sorted(payload)}). "
            f"Check that RERANKER_LLM_BASE_URL points at an Ollama server."
        )
