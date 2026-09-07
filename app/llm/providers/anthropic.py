from __future__ import annotations

from typing import Any, Mapping

from app.llm.base import LlmConfig
from app.llm.errors import LlmInvocationError
from app.llm.providers._http import HttpJsonLlmClient

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_API_VERSION = "2023-06-01"

# The Messages API requires ``max_tokens``. LLM2 returns a full attribute schema plus
# ONET ids, so the ceiling has to be generous or responses truncate mid-JSON.
DEFAULT_MAX_OUTPUT_TOKENS = 16384

JSON_SYSTEM_INSTRUCTION = (
    "You are a data extraction service. Respond with a single valid JSON object and "
    "nothing else: no prose, no explanation, no markdown code fences."
)


class AnthropicLlmClient(HttpJsonLlmClient):
    """Claude JSON completions (``POST /v1/messages``)."""

    provider_label = "Anthropic"
    api_key_env_hint = "ANTHROPIC_API_KEY"
    default_base_url = DEFAULT_BASE_URL

    def __init__(self, config: LlmConfig) -> None:
        super().__init__(config)
        options = dict(config.options())
        self.api_version = str(options.pop("anthropic_version", DEFAULT_API_VERSION))
        # ``system`` is overridable because a self-selected model may need different
        # coaxing to stay in pure JSON.
        self.system_instruction = str(options.pop("system", JSON_SYSTEM_INSTRUCTION))
        self.max_output_tokens = (
            config.max_output_tokens if config.max_output_tokens > 0 else DEFAULT_MAX_OUTPUT_TOKENS
        )
        # Anything left over is passed through verbatim (top_p, top_k, thinking, …).
        self.extra_body = options

    def endpoint(self) -> str:
        return f"{self.base_url}/v1/messages"

    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

    def request_body(self, prompt: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.max_output_tokens,
            "temperature": self.config.temperature,
            "system": self.system_instruction,
            "messages": [{"role": "user", "content": prompt}],
        }
        body.update(self.extra_body)
        return body

    def extract_text(self, payload: Mapping[str, Any]) -> str:
        """Concatenate the ``text`` blocks; ignore ``thinking`` and tool blocks."""
        content = payload.get("content")
        if not isinstance(content, list):
            raise LlmInvocationError(
                f"Anthropic response has no content list (keys: {sorted(payload)})."
            )
        parts = [
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "".join(parts)
        if not text.strip() and payload.get("stop_reason") == "max_tokens":
            raise LlmInvocationError(
                "Anthropic stopped at max_tokens before emitting any text — raise "
                "RERANKER_LLM_MAX_OUTPUT_TOKENS."
            )
        return text
