"""LLM tracing for the reranker, on Langfuse.

Ported from the compass-connect backend. ``tracing`` holds the lifecycle and the span
helpers, ``config`` the settings, ``masking`` the pre-export redaction and ``sampling``
the per-tier sample-rate logic. ``job_trace`` is the reranker-specific root trace every
ranked job runs under.

What gets recorded, per job: one ``rank_job`` trace, a ``generation`` for each LLM pass
(prompt, raw JSON output, model, token usage) and an ``embedding`` for each Gemini
``embed_content`` call (input text, model, dimensionality, token usage when the API
reports it).
"""

from app.observability.config import TracingConfig, parse_tracing_config, tracing_config_from_env
from app.observability.job_trace import job_trace
from app.observability.sampling import SamplingTier, should_sample
from app.observability.tracing import (
    current_trace_id,
    get_tracing_client,
    get_tracing_config,
    init_tracing,
    is_tracing_enabled,
    report_usage,
    shutdown_tracing,
    start_trace,
    suppress_tracing,
    traced_observation,
    update_current_observation,
    update_observation,
)

__all__ = [
    "SamplingTier",
    "TracingConfig",
    "current_trace_id",
    "get_tracing_client",
    "get_tracing_config",
    "init_tracing",
    "is_tracing_enabled",
    "job_trace",
    "parse_tracing_config",
    "report_usage",
    "should_sample",
    "shutdown_tracing",
    "start_trace",
    "suppress_tracing",
    "traced_observation",
    "tracing_config_from_env",
    "update_current_observation",
    "update_observation",
]
