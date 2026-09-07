from __future__ import annotations

from typing import Any, Mapping

from app.llm.base import LlmConfig
from app.llm.errors import LlmInvocationError
from app.llm.providers._http import HttpJsonLlmClient

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAiLlmClient(HttpJsonLlmClient):
    """OpenAI JSON completions (``POST /chat/completions``)."""

    provider_label = "OpenAI"
    api_key_env_hint = "OPENAI_API_KEY"
    default_base_url = DEFAULT_BASE_URL
    requires_api_key = True

    def __init__(self, config: LlmConfig) -> None:
        super().__init__(config)
        options = dict(config.options())
        self.json_mode = bool(options.pop("json_mode", True))
        # Reasoning models reject `temperature`; drop it with {"send_temperature": false}.
        self.send_temperature = bool(options.pop("send_temperature", True))
        # Newer models take `max_completion_tokens` instead of `max_tokens`.
        self.max_tokens_field = str(options.pop("max_tokens_field", "max_tokens"))
        self.extra_body = options

    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        if self.api_key:
            h["authorization"] = f"Bearer {self.api_key}"
        return h

    def request_body(self, prompt: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.send_temperature:
            body["temperature"] = self.config.temperature
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.config.max_output_tokens > 0:
            body[self.max_tokens_field] = self.config.max_output_tokens
        body.update(self.extra_body)
        return body

    def extract_text(self, payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LlmInvocationError(
                f"{self.provider_label} response has no choices (keys: {sorted(payload)})."
            )
        first = choices[0]
        if not isinstance(first, dict):
            raise LlmInvocationError(
                f"{self.provider_label} choices[0] is a {type(first).__name__}, expected an object."
            )
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            # Some compatible servers return the multimodal block list shape.
            if isinstance(content, list):
                return "".join(
                    str(b.get("text") or "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") in ("text", "output_text")
                )
        if first.get("finish_reason") == "length":
            raise LlmInvocationError(
                f"{self.provider_label} truncated the response at the token limit — "
                f"raise RERANKER_LLM_MAX_OUTPUT_TOKENS."
            )
        # ``text`` is the legacy completions field; a few compatible servers still use it.
        legacy = first.get("text")
        return legacy if isinstance(legacy, str) else ""


class OpenAiCompatibleLlmClient(OpenAiLlmClient):
    """A non-OpenAI server speaking the Chat Completions API.

    Differs from :class:`OpenAiLlmClient` in exactly two ways: the base URL must be
    given explicitly (there is no sensible default for "someone else's server"), and
    no API key is required, because a local vLLM or Ollama usually has none.
    """

    provider_label = "OpenAI-compatible"
    api_key_env_hint = "RERANKER_LLM_API_KEY"
    default_base_url = ""
    requires_api_key = False
