"""The LLM tracing layer, built on Langfuse.

Ported from the compass-connect backend (``common_libs/observability``). Everything in
this module is safe to call unconditionally: when tracing is disabled — the default,
and always the case under pytest — every helper degrades to a cheap no-op that yields
``None``, so call sites never guard with ``if tracing_enabled``.

Nothing here is allowed to break the job it is observing. The Langfuse client batches
and exports on its own threads, and every entry point is wrapped so that a tracing
failure is logged and swallowed rather than propagated onto the ranking path.

Typical use::

    with start_trace(name="rank_job", tier=SamplingTier.BATCH, sampling_key=job_id):
        with traced_observation(name="skill_ranker", as_type="generation", input=prompt):
            ...

Nesting follows the current context. Threads do **not** inherit it: work submitted to a
``ThreadPoolExecutor`` must be run through ``contextvars.copy_context().run`` to stay
attached to its parent (see ``app.pipeline.run_llm1_llm2_phase``).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from langfuse import Langfuse, propagate_attributes
from opentelemetry import trace as otel_trace_api

from app.observability.config import TracingConfig
from app.observability.masking import build_mask_function
from app.observability.sampling import SamplingTier, should_sample

logger = logging.getLogger(__name__)

_client: Langfuse | None = None
_config: TracingConfig = TracingConfig()

# Set while inside a ``suppress_tracing()`` block, i.e. a unit of work sampling skipped.
_suppressed_ctx_var: ContextVar[bool] = ContextVar("tracing_suppressed", default=False)

# The innermost observation opened by ``traced_observation``, so code below it — an LLM
# provider parsing token counts out of its response — can annotate it without the
# observation being threaded through every signature.
_current_observation_ctx_var: ContextVar[Any] = ContextVar("current_observation", default=None)


def init_tracing(config: TracingConfig) -> None:
    """Initialise the tracing layer. Call once, at process startup.

    Missing credentials or a bad host are logged and otherwise ignored: the reranker
    starts and runs untraced rather than failing to boot.
    """
    global _client, _config
    _config = config

    if not config.enabled:
        logger.info(
            "LLM tracing is disabled (RERANKER_ENABLE_TRACING). No traces will be exported."
        )
        _client = None
        return

    if not config.public_key or not config.secret_key:
        logger.warning(
            "LLM tracing is enabled but the Langfuse keys are not set "
            "(RERANKER_LANGFUSE_PUBLIC_KEY / RERANKER_LANGFUSE_SECRET_KEY). Tracing will stay off."
        )
        _client = None
        return

    try:
        _client = Langfuse(
            public_key=config.public_key,
            secret_key=config.secret_key,
            host=config.host,
            environment=config.environment,
            release=config.release,
            debug=config.debug,
            timeout=config.timeout,
            flush_at=config.flush_at,
            flush_interval=config.flush_interval,
            mask=build_mask_function(config),
            tracing_enabled=True,
        )
    except Exception as e:
        logger.error(
            "Failed to initialize the Langfuse client. Tracing will stay off. Error: %s",
            e,
            exc_info=True,
        )
        _client = None
        return

    logger.info(
        "LLM tracing initialized | host=%s | environment=%s | release=%s | "
        "batch_sample_rate=%s | api_sample_rate=%s | mask_pii=%s",
        config.host,
        config.environment,
        config.release,
        config.batch_sample_rate,
        config.api_sample_rate,
        config.mask_pii,
    )


def shutdown_tracing() -> None:
    """Flush buffered observations and shut the client down.

    Must run on exit. A Cloud Run Job's container is torn down as soon as the process
    ends, and without an explicit flush the last jobs' spans are simply lost.
    """
    global _client
    if _client is None:
        return
    try:
        _client.flush()
        _client.shutdown()
        logger.info("LLM tracing flushed and shut down.")
    except Exception as e:
        logger.warning("Failed to shut the Langfuse client down cleanly. Error: %s", e)
    finally:
        _client = None


def is_tracing_enabled() -> bool:
    """Whether spans created right now would actually be recorded."""
    return _client is not None and not _suppressed_ctx_var.get()


def get_tracing_config() -> TracingConfig:
    """The configuration the tracing layer was initialised with (defaults if never)."""
    return _config


def get_tracing_client() -> Langfuse | None:
    """The underlying Langfuse client, for the rare call site that needs the raw SDK."""
    return _client


@contextmanager
def suppress_tracing() -> Iterator[None]:
    """Suppress span creation for the block, including in contexts copied from inside it."""
    token = _suppressed_ctx_var.set(True)
    try:
        yield
    finally:
        _suppressed_ctx_var.reset(token)


def _has_active_span() -> bool:
    """Whether an OpenTelemetry span is active in the current context.

    Asked of OpenTelemetry directly: the Langfuse client's ``…_current_span`` helpers
    log a warning when there is none, which is noise on every untraced call.
    """
    return otel_trace_api.get_current_span() is not otel_trace_api.INVALID_SPAN


@contextmanager
def traced_observation(
    *,
    name: str,
    as_type: str = "span",
    input: Any = None,
    metadata: dict | None = None,
    model: str | None = None,
    model_parameters: dict | None = None,
) -> Iterator[Any]:
    """Open an observation nested under whatever is currently active.

    :param as_type: The Langfuse observation type — "span", "generation", "embedding", ….
    :param input: The input payload; masked and truncated before export.
    :return: The Langfuse observation, or ``None`` when tracing is off — handle ``None``.
    """
    if not is_tracing_enabled():
        yield None
        return

    try:
        manager = _client.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=input,
            metadata=metadata or None,
            model=model,
            model_parameters=_scalar_only(model_parameters),
        )
    except Exception as e:
        logger.warning(
            "Failed to start the %r observation. Continuing untraced. Error: %s", name, e
        )
        yield None
        return

    with manager as observation:
        token = _current_observation_ctx_var.set(observation)
        try:
            yield observation
        finally:
            _current_observation_ctx_var.reset(token)


@contextmanager
def start_trace(
    *,
    name: str,
    tier: SamplingTier,
    sampling_key: str,
    session_id: Any = None,
    user_id: Any = None,
    input: Any = None,
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> Iterator[Any]:
    """Open a root trace for one unit of work, if it is in the sample.

    Work that sampling skips runs inside :func:`suppress_tracing`, so none of its nested
    generations and embeddings are recorded either.

    :param sampling_key: A stable key (the job id); the same key always gets the same decision.
    :param session_id: Groups traces in the Langfuse session view (the batch run id).
    :param metadata: Trace metadata; values are coerced to strings, as Langfuse requires.
    :return: The root observation, or ``None`` when tracing is off or the work was not sampled.
    """
    if not is_tracing_enabled():
        yield None
        return

    if not should_sample(tier=tier, key=sampling_key, config=_config):
        with suppress_tracing():
            yield None
        return

    trace_metadata = {k: str(v) for k, v in (metadata or {}).items() if v not in (None, "")}
    try:
        manager = _client.start_as_current_observation(
            name=name,
            as_type="span",
            input=input,
            metadata=trace_metadata or None,
        )
    except Exception as e:
        logger.warning("Failed to start the %r trace. Continuing untraced. Error: %s", name, e)
        yield None
        return

    with manager as root:
        with propagate_attributes(
            trace_name=name,
            session_id=_clean(session_id),
            user_id=_clean(user_id),
            tags=list(tags or []),
            metadata=trace_metadata or None,
        ):
            yield root


def update_observation(observation: Any, **fields: Any) -> None:
    """Update an observation, tolerating both a ``None`` observation and a failing SDK call."""
    if observation is None:
        return
    try:
        observation.update(**fields)
    except Exception as e:
        logger.warning("Failed to update an observation. Error: %s", e)


def update_current_observation(**fields: Any) -> None:
    """Update the innermost observation opened by :func:`traced_observation`, if any."""
    update_observation(_current_observation_ctx_var.get(), **fields)


def report_usage(usage_details: dict[str, Any] | None) -> None:
    """Attach token counts to the generation or embedding currently being recorded.

    Called by the LLM providers after they parse a response. Non-integer and negative
    counts are dropped rather than sent, since Langfuse rejects them.
    """
    if not usage_details or _current_observation_ctx_var.get() is None:
        return
    clean = {
        str(k): int(v)
        for k, v in usage_details.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
    }
    if clean:
        update_current_observation(usage_details=clean)


def current_trace_id() -> str | None:
    """The id of the trace in progress, for correlating logs with traces."""
    if _client is None or not _has_active_span():
        return None
    try:
        return _client.get_current_trace_id()
    except Exception:
        return None


def _clean(value: Any) -> str | None:
    """Coerce an identity to the string Langfuse requires; empty becomes ``None``.

    The batch run id is a Mongo ``ObjectId``, and Langfuse drops (with a warning) any
    propagated attribute that is not a string — which silently empties the session view.
    """
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text or None


def _scalar_only(parameters: dict | None) -> dict | None:
    """Keep only the values Langfuse accepts as model parameters."""
    if not parameters:
        return None
    kept = {
        key: value
        for key, value in parameters.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    return kept or None
