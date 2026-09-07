from __future__ import annotations

import json
import logging
import threading
from typing import Any, Mapping

import httpx

from app.llm.base import LlmClient, LlmConfig
from app.llm.errors import LlmConfigurationError, LlmInvocationError
from app.llm.retry import call_with_transient_retry, is_transient_network_error

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 120.0

# HTTP statuses worth retrying: rate limits and transient server-side failures.
RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

# Statuses that no retry can fix and that mean the deployment is wired wrong.
CONFIG_ERROR_STATUSES = frozenset({401, 403, 404})


class _RetryableStatus(RuntimeError):
    """Internal marker so a retryable HTTP status flows through the transient retry."""


class HttpJsonLlmClient(LlmClient):
    """Base for providers driven by a single JSON POST.

    Subclasses supply the endpoint, headers and request body, and pull the
    completion text out of the parsed response.
    """

    #: Used in log lines and error messages.
    provider_label = "LLM"
    #: Consulted by ``require_api_key`` when no key is configured.
    api_key_env_hint = "RERANKER_LLM_API_KEY"
    #: Endpoint when the deployment does not set ``RERANKER_LLM_BASE_URL``.
    default_base_url = ""
    #: Whether a usable client requires an API key (local servers often do not).
    requires_api_key = True

    def __init__(self, config: LlmConfig) -> None:
        super().__init__(config)
        config.require_model()
        self.api_key = (
            config.require_api_key(env_hint=self.api_key_env_hint)
            if self.requires_api_key
            else (config.api_key or "").strip()
        )
        self.base_url = self._resolve_base_url(config)
        self.timeout_s = config.timeout_s if config.timeout_s > 0 else DEFAULT_TIMEOUT_S
        self._http: httpx.Client | None = None
        self._http_lock = threading.Lock()

    def _resolve_base_url(self, config: LlmConfig) -> str:
        base = (config.base_url or self.default_base_url or "").strip().rstrip("/")
        if not base:
            raise LlmConfigurationError(
                f"No API endpoint configured for provider {config.provider!r}. "
                f"Set RERANKER_LLM_BASE_URL — for a self-hosted or OpenAI-compatible "
                f"server this is the base of its API, e.g. http://localhost:11434/v1."
            )
        if not base.startswith(("http://", "https://")):
            raise LlmConfigurationError(
                f"RERANKER_LLM_BASE_URL={base!r} must start with http:// or https://."
            )
        return base

    # ------------------------------------------------------------------ transport

    def _client(self) -> httpx.Client:
        """One connection pool per client, recreated after a transient failure."""
        http = self._http
        if http is not None:
            return http
        with self._http_lock:
            if self._http is None:
                self._http = httpx.Client(timeout=self.timeout_s)
            return self._http

    def _reset_client(self) -> None:
        with self._http_lock:
            http, self._http = self._http, None
        if http is not None:
            try:
                http.close()
            except Exception:
                pass

    def close(self) -> None:
        self._reset_client()

    # ------------------------------------------------------- subclass entry points

    def endpoint(self) -> str:
        """Full URL to POST to."""
        raise NotImplementedError

    def headers(self) -> dict[str, str]:
        """Request headers, including auth."""
        raise NotImplementedError

    def request_body(self, prompt: str) -> dict[str, Any]:
        """JSON body asking the model for a single JSON document."""
        raise NotImplementedError

    def extract_text(self, payload: Mapping[str, Any]) -> str:
        """Completion text from the parsed response body."""
        raise NotImplementedError

    # ------------------------------------------------------------------- the call

    def generate_json(self, prompt: str, *, purpose: str = "") -> str:
        label = f"{self.provider_label}[{purpose or 'json'}]"
        url = self.endpoint()
        headers = self.headers()
        body = self.request_body(prompt)

        def _once() -> Mapping[str, Any]:
            response = self._client().post(url, headers=headers, json=body)
            if response.status_code in CONFIG_ERROR_STATUSES:
                raise LlmConfigurationError(
                    f"{self.provider_label} rejected the request with HTTP "
                    f"{response.status_code} for model {self.config.model!r} at {url}: "
                    f"{_error_snippet(response)}. Check the API key, the model id, and "
                    f"that this deployment is allowed to use that model."
                )
            if response.status_code in RETRYABLE_STATUSES:
                raise _RetryableStatus(f"HTTP {response.status_code}: {_error_snippet(response)}")
            if response.status_code >= 400:
                raise LlmInvocationError(
                    f"{self.provider_label} API error (HTTP {response.status_code}) "
                    f"at {url}: {_error_snippet(response)}"
                )
            try:
                payload = response.json()
            except ValueError as e:
                raise LlmInvocationError(
                    f"{self.provider_label} returned a non-JSON response body: "
                    f"{_error_snippet(response)}"
                ) from e
            if not isinstance(payload, dict):
                raise LlmInvocationError(
                    f"{self.provider_label} response body is a "
                    f"{type(payload).__name__}, expected an object."
                )
            return payload

        def _is_transient(exc: BaseException) -> bool:
            return isinstance(exc, _RetryableStatus) or is_transient_network_error(exc)

        try:
            payload = call_with_transient_retry(
                _once,
                max_attempts=self.config.max_attempts,
                label=label,
                is_transient=_is_transient,
                on_retry=self._reset_client,
            )
        except (LlmConfigurationError, LlmInvocationError):
            raise
        except Exception as e:
            raise LlmInvocationError(
                f"{self.provider_label} API error ({type(e).__name__}): {e}"
            ) from e

        text = self.extract_text(payload)
        if not text.strip():
            raise LlmInvocationError(
                f"Empty response from {self.provider_label} "
                f"(model={self.config.model!r}, purpose={purpose or 'json'}). "
                f"Response keys: {sorted(payload)}"
            )
        return text


def _error_snippet(response: httpx.Response, *, limit: int = 500) -> str:
    """Short, log-safe excerpt of an error body — the vendor message, when there is one."""
    try:
        data = response.json()
    except ValueError:
        return response.text[:limit]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("type")
            if msg:
                return str(msg)[:limit]
        if isinstance(err, str):
            return err[:limit]
        msg = data.get("message")
        if msg:
            return str(msg)[:limit]
    return json.dumps(data)[:limit]
