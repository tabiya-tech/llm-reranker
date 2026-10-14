from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from app.api_models import (
    ErrorResponse,
    HealthResponse,
    RankedJobDocument,
    RankRequest,
    RankResponse,
)
from app.config import get_settings, get_settings_for_language, log_ranker_run_configuration
from app.languages import LANGUAGE_REGISTRY, normalise_language
from app.llm import get_llm_client
from app.logging_config import configure_logging
from app.observability import (
    init_tracing,
    job_trace,
    shutdown_tracing,
    tracing_config_from_env,
    update_observation,
)
from app.pipeline import TaxonomyContext, load_taxonomy_context, process_single_job

logger = logging.getLogger(__name__)

_taxonomy: TaxonomyContext | None = None
# One taxonomy context per language: the CSV label packs differ per language, so a
# Spanish request cannot be served from the English lookups. Loaded on first use.
_taxonomies: dict[str, TaxonomyContext] = {}
_taxonomy_lock = threading.Lock()


def get_taxonomy(language: str) -> TaxonomyContext:
    """Taxonomy context for a language, loading its CSV pack on first use.

    :raises HTTPException: 503 if the service has not finished starting up.
    """
    if _taxonomy is None:
        raise HTTPException(status_code=503, detail="Service not ready — taxonomy not loaded.")
    cached = _taxonomies.get(language)
    if cached is not None:
        return cached
    with _taxonomy_lock:
        cached = _taxonomies.get(language)
        if cached is not None:
            return cached
        settings = get_settings_for_language(language)
        logger.info("Loading taxonomy data for language=%s (CSV lookups)…", language)
        context = load_taxonomy_context(settings)
        _taxonomies[language] = context
        logger.info(
            "Taxonomy ready | language=%s | resources=%s | occupations=%s",
            language,
            settings.language_resources_dir.name,
            len(context.occ_map),
        )
        return context


@asynccontextmanager
async def lifespan(application: FastAPI):
    configure_logging()
    global _taxonomy
    settings = get_settings()
    # Build the configured model up front: a misconfigured selection must fail the
    # container's startup probe, not the first request that reaches it. This also
    # warms the process-wide client that every request will reuse.
    llm = get_llm_client(settings.llm)
    init_tracing(tracing_config_from_env())
    logger.info("Loading taxonomy data (CSV lookups)…")
    _taxonomy = load_taxonomy_context(settings)
    _taxonomies[settings.language] = _taxonomy
    logger.info(
        "API ready | llm=%s (%s) | language=%s | taxonomy_occupations=%s | supported_languages=%s",
        llm.model,
        settings.llm.provider,
        settings.language,
        len(_taxonomy.occ_map),
        ",".join(LANGUAGE_REGISTRY),
    )
    yield
    # Flush buffered traces before Cloud Run tears the instance down.
    shutdown_tracing()
    _taxonomy = None
    _taxonomies.clear()


app = FastAPI(
    title="LLM Re-ranker",
    description=(
        "Single-job ranking: a classified job document in, the ranked job document out. "
        "Runs the same per-job pipeline as the batch CLI (`python main.py`), without the "
        "job embedding or the MongoDB write. See README.md, \"API\"."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.post(
    "/rank",
    response_model=RankResponse,
    # Optional keys the pipeline did not write (e.g. ``llm_is_relevant`` outside LLM3
    # ``call`` mode) stay absent instead of appearing as null.
    response_model_exclude_unset=True,
    responses={
        422: {"description": "The body is not a classified job document (see `detail`)."},
        500: {"model": ErrorResponse, "description": "The pipeline failed, e.g. an LLM call."},
        503: {"model": ErrorResponse, "description": "The service is still starting up."},
    },
)
async def rank_job(
    body: RankRequest,
    language: Annotated[
        str | None,
        Query(
            description="Language pack to rank against: a code (`en`, `es`), a locale "
            "(`AR-es`, `es_AR`), or a name (`spanish`). Wins over the document's `language` "
            "field. Unknown values fall back to `en` with a warning.",
            examples=["es"],
        ),
    ] = None,
) -> RankResponse:
    """Rank a single classified job document. Send the document directly as the JSON body.

    ``?language=es`` (or a ``language`` field on the document) picks the resource pack the
    job is ranked against — taxonomy labels, prompts, job attributes. Omitted, the
    service default applies, so existing callers are unaffected.
    """
    doc = body.to_document()
    requested = language or body.language
    lang = normalise_language(requested) if requested else get_settings().language
    taxonomy = get_taxonomy(lang)
    settings = get_settings_for_language(lang)
    log_ranker_run_configuration(
        logger,
        settings,
        bulk_write_batch_size=settings.bulk_write_batch_size,
        cli_limit=0,
        log_prompt_debug=False,
        invocation="api",
    )
    try:
        with job_trace(doc, settings=settings, invocation="api") as trace:
            result = process_single_job(doc, taxonomy, settings=settings)
            update_observation(
                trace,
                output={
                    "ran_llm1": result.ran_llm1,
                    "ran_llm2": result.ran_llm2,
                    "ran_llm3": result.ran_llm3,
                    "llm3_outcome": result.llm3_outcome.value,
                    "ranked": result.ranked_document is not None,
                    "timings": result.timings,
                },
            )
    except Exception as exc:
        logger.exception("Error processing job: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RankResponse(
        ranked_document=(
            RankedJobDocument.model_validate(result.ranked_document)
            if result.ranked_document is not None
            else None
        ),
        timings=result.timings,
        ran_llm1=result.ran_llm1,
        ran_llm2=result.ran_llm2,
        ran_llm3=result.ran_llm3,
        llm3_outcome=result.llm3_outcome,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Readiness check, plus which language packs this instance serves and has loaded."""
    return HealthResponse(
        status="ok",
        taxonomy_loaded=_taxonomy is not None,
        default_language=get_settings().language,
        supported_languages=list(LANGUAGE_REGISTRY),
        loaded_languages=sorted(_taxonomies),
    )
