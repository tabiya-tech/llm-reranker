"""``LlmClient`` wrapper that records every call as a Langfuse generation."""

from __future__ import annotations

import logging
from typing import Any

from app.llm.base import LlmClient, LlmConfig
from app.observability import traced_observation, update_observation

logger = logging.getLogger(__name__)


class TracedLlmClient:
    """The one place the reranker's LLM passes reach the model.

    Wraps the configured :class:`LlmClient`; retries and error translation stay the
    client's job. Every pass goes through :meth:`generate_json`, which is why the
    Langfuse generation lives here rather than in each provider. Providers only add the
    token usage they parse out of their response (``app.observability.report_usage``).
    """

    def __init__(self, client: LlmClient):
        self._client = client

    @property
    def client(self) -> LlmClient:
        return self._client

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def config(self) -> LlmConfig:
        return self._client.config

    def generate_json(
        self, prompt: str, *, purpose: str, response_schema: dict[str, Any] | None = None
    ) -> str:
        """Ask the model for JSON, recording the call as a generation named ``purpose``.

        ``response_schema`` reaches the client only when it supports structured output and
        ``RERANKER_LLM_RESPONSE_SCHEMA`` is on; otherwise the prompt's prose contract and
        the provider's JSON mode are all the model gets.
        """
        cfg = self._client.config
        use_schema = (
            response_schema is not None
            and cfg.response_schema
            and self._client.supports_response_schema
        )
        logger.debug("LLM call (%s): invoking %s", purpose, cfg.describe())
        with traced_observation(
            name=purpose or "generate_json",
            as_type="generation",
            input=prompt,
            model=self._client.model,
            model_parameters={
                "temperature": cfg.temperature,
                "max_output_tokens": cfg.max_output_tokens or None,
            },
            metadata={
                "purpose": purpose,
                "provider": cfg.provider,
                "client_class": cfg.client_class,
                "response_schema": use_schema,
            },
        ) as generation:
            try:
                if use_schema:
                    # Opt-in keyword, not in the base signature: only clients that set
                    # supports_response_schema are called with it (see LlmClient).
                    out = self._client.generate_json(
                        prompt,
                        purpose=purpose,
                        response_schema=response_schema,  # type: ignore[call-arg]
                    )
                else:
                    out = self._client.generate_json(prompt, purpose=purpose)
            except Exception as e:
                update_observation(
                    generation, level="ERROR", status_message=f"{type(e).__name__}: {e}"
                )
                raise
            update_observation(generation, output=out)
        logger.debug("LLM call (%s): completed", purpose)
        return out

    def close(self) -> None:
        self._client.close()
