from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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
from app.pipeline import JobResult, TaxonomyContext, load_taxonomy_context, process_single_job

logger = logging.getLogger(__name__)

JsonValue = Union[str, int, float, bool, None, dict[str, "JsonValue"], list["JsonValue"]]


def _normalize_mongo_extended_json(obj: JsonValue) -> JsonValue:
    """Convert MongoDB Extended JSON (``$oid``, ``$date``) to plain Python types.

    Accepts documents copy-pasted from MongoDB Compass / Atlas / mongoexport.
    """
    if isinstance(obj, dict):
        if len(obj) == 1:
            if "$oid" in obj:
                return str(obj["$oid"])
            if "$date" in obj:
                return str(obj["$date"])
            if "$numberInt" in obj:
                return int(str(obj["$numberInt"]))
            if "$numberLong" in obj:
                return int(str(obj["$numberLong"]))
            if "$numberDouble" in obj:
                return float(str(obj["$numberDouble"]))
        return {k: _normalize_mongo_extended_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_mongo_extended_json(item) for item in obj]
    return obj


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
    description="Single-job ranking: classified document in, ranked document out.",
    version="1.0.0",
    lifespan=lifespan,
)


class RankResponse(BaseModel):
    ranked_document: dict[str, Any] | None = Field(
        None,
        description="The ranked job payload (same shape written to the target collection), "
        "or null if no attributes could be produced.",
    )
    timings: dict[str, float] = Field(
        default_factory=dict,
        description="Per-step wall-clock times in seconds.",
    )
    ran_llm1: bool = False
    ran_llm2: bool = False
    ran_llm3: bool = False


@app.post("/rank", response_model=RankResponse)
async def rank_job(body: dict[str, Any], language: str | None = None) -> RankResponse:
    """Rank a single classified job document. Send the document directly as the JSON body.

    ``?language=es`` (or a ``language`` field on the document) picks the resource pack the
    job is ranked against — taxonomy labels, prompts, job attributes. Omitted, the
    service default applies, so existing callers are unaffected.
    """
    doc = _normalize_mongo_extended_json(body)
    assert isinstance(doc, dict)  # body is a dict, so normalizing it returns one
    doc_language = doc.get("language")
    requested = language or (doc_language if isinstance(doc_language, str) else None)
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
            result: JobResult = process_single_job(
                doc,
                taxonomy,
                settings=settings,
            )
            update_observation(
                trace,
                output={
                    "ran_llm1": result.ran_llm1,
                    "ran_llm2": result.ran_llm2,
                    "ran_llm3": result.ran_llm3,
                    "ranked": result.ranked_document is not None,
                    "timings": result.timings,
                },
            )
    except Exception as exc:
        logger.exception("Error processing job: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RankResponse(
        ranked_document=result.ranked_document,
        timings=result.timings,
        ran_llm1=result.ran_llm1,
        ran_llm2=result.ran_llm2,
        ran_llm3=result.ran_llm3,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "taxonomy_loaded": _taxonomy is not None,
        "default_language": get_settings().language,
        "supported_languages": list(LANGUAGE_REGISTRY),
        "loaded_languages": sorted(_taxonomies),
    }
