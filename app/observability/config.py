"""Configuration for the LLM tracing layer.

Mirrors the compass-connect backend: a handful of dedicated environment variables for
the credentials plus one JSON blob (``RERANKER_TRACING_CONFIG``) for the tunables, so a
deployment can be re-tuned without a code change.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any, TypeVar, cast

from pydantic import Field

from app.domain_base import DomainModel

_T = TypeVar("_T")

logger = logging.getLogger(__name__)

DEFAULT_LANGFUSE_HOST = "https://cloud.langfuse.com"

# Langfuse restricts the environment name to lowercase alphanumerics, hyphens and
# underscores, and reserves the "langfuse" prefix for its own use.
_ENVIRONMENT_ALLOWED_CHARS = re.compile(r"[^a-z0-9_-]")
_ENVIRONMENT_MAX_LENGTH = 40

_TRUE = ("1", "true", "yes", "on")


class TracingConfig(DomainModel):
    """The resolved configuration of the tracing layer.

    An instance with ``enabled=False`` (the default) makes every helper in
    :mod:`app.observability.tracing` a no-op, so tracing can be shipped dark.
    """

    enabled: bool = Field(
        default=False,
        description="Master on/off switch. Everything else is ignored when this is False.",
    )
    public_key: str = Field(
        default="",
        description="Langfuse project public key. The secret key belongs in Secret Manager.",
    )
    secret_key: str = Field(default="", description="Langfuse project secret key.")
    host: str = Field(
        default=DEFAULT_LANGFUSE_HOST,
        description="The Langfuse ingestion host (cloud, or self-hosted).",
    )
    environment: str | None = Field(
        default=None,
        description="The Langfuse environment (dev, prod, …), used to separate traces.",
    )
    release: str | None = Field(
        default=None, description="The deployed build, so a trace can be tied back to the image."
    )
    debug: bool = Field(default=False, description="Turn on the Langfuse SDK's own debug logging.")

    batch_sample_rate: float = Field(
        default=1.0,
        description="Fraction of batch-CLI jobs traced. Sampling is deterministic per job id, so "
        "a re-run of the same job gets the same decision.",
    )
    api_sample_rate: float = Field(
        default=1.0, description="Fraction of `POST /rank` requests traced."
    )

    mask_pii: bool = Field(
        default=True,
        description="Redact e-mail addresses, phone numbers and long digit runs before export.",
    )
    max_payload_chars: int = Field(
        default=100_000,
        description="Truncate any single string in a payload to this many characters.",
    )

    flush_at: int = Field(
        default=512, description="SDK buffering: batch size before an automatic flush."
    )
    flush_interval: float = Field(
        default=5.0, description="SDK buffering: max seconds between flushes."
    )
    timeout: int = Field(
        default=10, description="Hard timeout, in seconds, on the exporter's HTTP calls."
    )


# Field → (type, min, max) for the values the JSON blob may override.
_TUNABLES: dict[str, tuple[type, float | None, float | None]] = {
    "debug": (bool, None, None),
    "batch_sample_rate": (float, 0.0, 1.0),
    "api_sample_rate": (float, 0.0, 1.0),
    "mask_pii": (bool, None, None),
    "max_payload_chars": (int, 1, None),
    "flush_at": (int, 1, None),
    "flush_interval": (float, 0.001, None),
    "timeout": (int, 1, None),
}


def sanitize_environment_name(name: str | None) -> str | None:
    """Coerce an environment name into the shape Langfuse accepts, or ``None``."""
    if not name:
        return None
    sanitized = _ENVIRONMENT_ALLOWED_CHARS.sub("-", name.strip().lower()).strip("-")
    if not sanitized:
        return None
    if sanitized.startswith("langfuse"):
        # "langfuse" is a reserved prefix; keep the name meaningful rather than dropping it.
        sanitized = f"env-{sanitized}"
    return sanitized[:_ENVIRONMENT_MAX_LENGTH]


def parse_tracing_config(
    *,
    enabled: bool,
    public_key: str | None,
    secret_key: str | None,
    host: str | None,
    environment: str | None = None,
    release: str | None = None,
    raw_config: str | None = None,
) -> TracingConfig:
    """Build a :class:`TracingConfig` from already-read environment values.

    The JSON blob uses camelCase keys (``{"batchSampleRate": 0.2}``). Unknown keys and
    out-of-range values are logged and ignored, so a bad value can never stop the
    reranker from starting — tracing is optional.
    """
    config = TracingConfig(
        enabled=enabled,
        public_key=(public_key or "").strip(),
        secret_key=(secret_key or "").strip(),
        host=(host or "").strip() or DEFAULT_LANGFUSE_HOST,
        environment=sanitize_environment_name(environment),
        release=(release or "").strip() or None,
    )
    if not raw_config or not raw_config.strip():
        return config
    try:
        parsed = json.loads(raw_config)
    except json.JSONDecodeError as e:
        logger.warning("Invalid RERANKER_TRACING_CONFIG JSON. Using defaults. Error: %s", e)
        return config
    if not isinstance(parsed, dict):
        logger.warning("RERANKER_TRACING_CONFIG must be a JSON object. Using defaults.")
        return config
    return config.model_copy(update=_overrides_from_json(parsed))


def tracing_config_from_env() -> TracingConfig:
    """Read the tracing configuration from the process environment.

    Tracing is off unless ``RERANKER_ENABLE_TRACING`` opts in, so a deployment that has
    not been given Langfuse credentials behaves exactly as before. The keys fall back to
    the Langfuse SDK's own variable names, so a host that already exports those needs
    nothing new.
    """
    enabled = (os.getenv("RERANKER_ENABLE_TRACING") or "").strip().lower() in _TRUE
    return parse_tracing_config(
        enabled=enabled,
        public_key=_first_env("RERANKER_LANGFUSE_PUBLIC_KEY", "LANGFUSE_PUBLIC_KEY"),
        secret_key=_first_env("RERANKER_LANGFUSE_SECRET_KEY", "LANGFUSE_SECRET_KEY"),
        host=_first_env("RERANKER_LANGFUSE_HOST", "LANGFUSE_HOST", "LANGFUSE_BASE_URL"),
        environment=_first_env("RERANKER_TRACING_ENVIRONMENT"),
        # Cloud Run sets K_REVISION on a service; a job has no equivalent.
        release=_first_env("RERANKER_TRACING_RELEASE", "K_REVISION"),
        raw_config=os.getenv("RERANKER_TRACING_CONFIG", ""),
    )


def _first_env(*names: str) -> str | None:
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


_CAMEL_CASE_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _overrides_from_json(source: dict[str, Any]) -> dict[str, Any]:
    """camelCase JSON keys → validated ``TracingConfig`` field values.

    Credentials, host, environment and release come from dedicated variables and are
    never taken from the blob.
    """
    known = set(TracingConfig.model_fields)
    out: dict[str, Any] = {}
    for key, value in source.items():
        name = _CAMEL_CASE_BOUNDARY.sub("_", str(key)).lower()
        if name not in _TUNABLES:
            if name not in known:
                logger.warning("Ignoring unknown RERANKER_TRACING_CONFIG key %r.", key)
            else:
                logger.warning(
                    "Ignoring RERANKER_TRACING_CONFIG key %r — set it with its own variable.",
                    key,
                )
            continue
        typ, lo, hi = _TUNABLES[name]
        coerced = _coerce(value, typ)
        if (
            coerced is None
            or (lo is not None and coerced < lo)
            or (hi is not None and coerced > hi)
        ):
            logger.warning("Ignoring invalid RERANKER_TRACING_CONFIG value %r=%r.", key, value)
            continue
        out[name] = coerced
    return out


def _coerce(value: object, typ: type[_T]) -> _T | None:
    if typ is bool:
        if isinstance(value, bool):
            return cast(_T, value)
        if isinstance(value, str) and value.strip().lower() in (*_TRUE, "0", "false", "no", "off"):
            return cast(_T, value.strip().lower() in _TRUE)
        return None
    if isinstance(value, bool):
        return None
    try:
        return cast(Callable[[object], _T], typ)(value)
    except (TypeError, ValueError):
        return None
