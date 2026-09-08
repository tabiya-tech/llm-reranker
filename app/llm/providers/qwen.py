from __future__ import annotations

from typing import Any

from app.llm.base import LlmConfig
from app.llm.providers.openai import OpenAiLlmClient

# DashScope's OpenAI-compatible endpoint, international region. The Beijing region
# is https://dashscope.aliyuncs.com/compatible-mode/v1 — an API key is issued for
# one region only, so a Beijing key needs RERANKER_LLM_BASE_URL set to it.
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class QwenLlmClient(OpenAiLlmClient):
    """Qwen JSON completions via Alibaba DashScope's compatible mode.

    DashScope serves Qwen behind the Chat Completions API, so the OpenAI client
    does the work; this subclass supplies the endpoint, the key variable, and the
    one behavioural difference that matters — see ``enable_thinking`` below.

    For a Qwen model served by something else, use the provider matching that
    server: ``ollama`` for a local Ollama, ``openai_compatible`` for vLLM,
    Together, Groq or SGLang.
    """

    provider_label = "Qwen"
    api_key_env_hint = "DASHSCOPE_API_KEY"
    default_base_url = DEFAULT_BASE_URL
    requires_api_key = True

    def __init__(self, config: LlmConfig) -> None:
        super().__init__(config)
        # Qwen3 hybrid models think by default, and DashScope serves thinking only
        # over streaming — a non-streaming call like this one is rejected outright.
        # Turning it off is what makes `qwen3-*` usable here at all. Override with
        # RERANKER_LLM_OPTIONS={"enable_thinking": true} (needs streaming support),
        # or {"enable_thinking": null} to omit the field entirely for a server
        # that does not know it.
        extra: dict[str, Any] = self.extra_body
        extra.setdefault("enable_thinking", False)
        if extra.get("enable_thinking", False) is None:
            extra.pop("enable_thinking")
