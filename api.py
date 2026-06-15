from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings, log_ranker_run_configuration
from app.logging_config import configure_logging
from app.pipeline import JobResult, TaxonomyContext, load_taxonomy_context, process_single_job

logger = logging.getLogger(__name__)


def _normalize_mongo_extended_json(obj: Any) -> Any:
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
                return int(obj["$numberInt"])
            if "$numberLong" in obj:
                return int(obj["$numberLong"])
            if "$numberDouble" in obj:
                return float(obj["$numberDouble"])
        return {k: _normalize_mongo_extended_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_mongo_extended_json(item) for item in obj]
    return obj


_taxonomy: TaxonomyContext | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    configure_logging()
    global _taxonomy
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured — cannot start API server.")
    logger.info("Loading taxonomy data (CSV lookups)…")
    _taxonomy = load_taxonomy_context(settings)
    logger.info(
        "API ready | model=%s | taxonomy_occupations=%s",
        settings.gemini_model,
        len(_taxonomy.occ_map),
    )
    yield
    _taxonomy = None


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
async def rank_job(body: dict[str, Any]) -> RankResponse:
    """Rank a single classified job document. Send the document directly as the JSON body."""
    if _taxonomy is None:
        raise HTTPException(status_code=503, detail="Service not ready — taxonomy not loaded.")
    settings = get_settings()
    log_ranker_run_configuration(
        logger,
        settings,
        bulk_write_batch_size=settings.bulk_write_batch_size,
        cli_limit=0,
        log_prompt_debug=False,
        invocation="api",
    )
    doc = _normalize_mongo_extended_json(body)
    try:
        result: JobResult = process_single_job(
            doc,
            _taxonomy,
            settings=settings,
        )
    except Exception as exc:
        logger.exception("Error processing job: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return RankResponse(
        ranked_document=result.ranked_document,
        timings=result.timings,
        ran_llm1=result.ran_llm1,
        ran_llm2=result.ran_llm2,
        ran_llm3=result.ran_llm3,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "taxonomy_loaded": _taxonomy is not None}
