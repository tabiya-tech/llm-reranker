from __future__ import annotations

import logging
import threading
from typing import Any

from google import genai
from google.genai import types as genai_types

from app.llm.base import LlmClient, LlmConfig
from app.llm.errors import LlmConfigurationError, LlmInvocationError
from app.llm.retry import call_with_transient_retry, is_transient_network_error
from app.observability import report_usage

logger = logging.getLogger(__name__)


def _is_transient(exc: BaseException) -> bool:
    """Network-level transients plus the SDK's own 503/overload wording."""
    if is_transient_network_error(exc):
        return True
    msg = str(exc).lower()
    return "503" in msg and ("unavailable" in msg or "overloaded" in msg)


class GeminiLlmClient(LlmClient):
    """Gemini JSON completions.

    Holds one ``genai.Client`` for the process, rebuilt after a transient failure
    so a retry does not reuse a half-dead connection pool.
    """

    supports_response_schema = True

    def __init__(self, config: LlmConfig) -> None:
        super().__init__(config)
        self._api_key = config.require_api_key(env_hint="GEMINI_API_KEY")
        config.require_model()
        self._sdk: genai.Client | None = None
        self._sdk_lock = threading.Lock()
        self._generate_config = self._build_generate_config(config)

    @staticmethod
    def _build_generate_config(config: LlmConfig) -> genai_types.GenerateContentConfig:
        """``response_mime_type=application/json`` is what makes the model emit parseable JSON."""
        kwargs: dict[str, object] = {
            "temperature": config.temperature,
            "response_mime_type": "application/json",
        }
        if config.max_output_tokens > 0:
            kwargs["max_output_tokens"] = config.max_output_tokens
        kwargs.update(config.options())
        return genai_types.GenerateContentConfig(**kwargs)  # type: ignore[arg-type]

    def _http_options(self) -> genai_types.HttpOptions | None:
        """``None`` unless the config overrides the endpoint or timeout (SDK defaults win)."""
        opts: dict[str, object] = {}
        if self.config.base_url:
            opts["base_url"] = self.config.base_url
        if self.config.timeout_s > 0:
            opts["timeout"] = int(self.config.timeout_s * 1000)  # SDK expects milliseconds
        return genai_types.HttpOptions(**opts) if opts else None  # type: ignore[arg-type]

    def _client(self) -> genai.Client:
        """Lazily create (and memoise) the SDK client."""
        sdk = self._sdk
        if sdk is not None:
            return sdk
        with self._sdk_lock:
            if self._sdk is None:
                http_options = self._http_options()
                self._sdk = (
                    genai.Client(api_key=self._api_key, http_options=http_options)
                    if http_options is not None
                    else genai.Client(api_key=self._api_key)
                )
            return self._sdk

    def _reset_client(self) -> None:
        """Drop the cached SDK client so the next call opens a fresh HTTP pool."""
        with self._sdk_lock:
            self._sdk = None

    def close(self) -> None:
        self._reset_client()

    def _config_for(
        self, response_schema: dict[str, Any] | None
    ) -> genai_types.GenerateContentConfig:
        """The shared config, plus ``response_json_schema`` when the call has one."""
        if response_schema is None:
            return self._generate_config
        return self._generate_config.model_copy(update={"response_json_schema": response_schema})

    def generate_json(
        self,
        prompt: str,
        *,
        purpose: str = "",
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        model = self.config.model
        label = f"Gemini[{purpose or 'json'}]"
        generate_config = self._config_for(response_schema)

        def _once() -> object:
            client = self._client()
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=generate_config,
                )
            except Exception as e:
                msg = str(e)
                if "404" in msg or "not found" in msg.lower():
                    raise LlmConfigurationError(
                        f"Model {model!r} not found (or project has no access). "
                        f"Check the model id at https://ai.google.dev/gemini-api/docs/models — "
                        f"preview models use a -preview suffix (e.g. gemini-3-pro-preview)."
                    ) from e
                raise
            return response

        try:
            response = call_with_transient_retry(
                _once,
                max_attempts=self.config.max_attempts,
                label=label,
                is_transient=_is_transient,
                on_retry=self._reset_client,
            )
        except LlmConfigurationError:
            raise
        except Exception as e:
            raise LlmInvocationError(f"Gemini API error ({type(e).__name__}): {e}") from e

        text = _response_text(response)
        report_usage(_usage_details(response))
        if not text.strip():
            raise LlmInvocationError("Empty response from Gemini")
        return text


def _response_text(response: object) -> str:
    """Pull text out of a GenAI response, falling back to the first candidate part."""
    t = getattr(response, "text", None)
    if isinstance(t, str) and t.strip():
        return t
    cands = getattr(response, "candidates", None) or []
    if not cands:
        return ""
    parts = getattr(cands[0].content, "parts", None) or []
    if not parts:
        return ""
    p0 = parts[0]
    return getattr(p0, "text", "") or ""


def _usage_details(response: object) -> dict[str, int] | None:
    """Token counts from ``usage_metadata``; thinking tokens are billed as output."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None

    def _count(name: str) -> int | None:
        value = getattr(meta, name, None)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    prompt = _count("prompt_token_count")
    candidates = _count("candidates_token_count")
    thoughts = _count("thoughts_token_count")
    total = _count("total_token_count")
    out: dict[str, int] = {}
    if prompt is not None:
        out["input"] = prompt
    if candidates is not None or thoughts is not None:
        out["output"] = (candidates or 0) + (thoughts or 0)
    if total is not None:
        out["total"] = total
    return out or None
