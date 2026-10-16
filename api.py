"""HTTP entrypoint (Cloud Run Service): ``uvicorn api:app``.

Mounts the use-case routes and warms their services at startup.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.constants.errors import ErrorResponse
from app.job_ranking.get_job_ranking_service import (
    get_job_ranking_services,
    reset_job_ranking_services,
)
from app.job_ranking.routes import add_job_ranking_routes
from app.logging_config import configure_logging, reconfigure_uvicorn_logging
from app.observability import init_tracing, shutdown_tracing, tracing_config_from_env

configure_logging()
logger = logging.getLogger(__name__)


def _error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=ErrorResponse(detail=detail).model_dump())


@asynccontextmanager
async def lifespan(application: FastAPI):
    # uvicorn installs its own log handlers at startup — reroute them too.
    reconfigure_uvicorn_logging()
    settings = get_settings()
    init_tracing(tracing_config_from_env())
    # Build the model and load the default language pack up front: a misconfiguration must
    # fail the container's startup probe, not the first request that reaches it.
    status = get_job_ranking_services().health()
    logger.info(
        "API ready | llm=%s | language=%s | supported_languages=%s",
        settings.llm.describe(),
        status.default_language,
        ",".join(status.supported_languages),
    )
    yield
    reset_job_ranking_services()
    # Flush buffered traces before Cloud Run tears the instance down.
    shutdown_tracing()


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


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 as ``ErrorResponse``: FastAPI's error list flattened to one ``detail`` string.

    Never includes the rejected ``input``, which can carry job text.
    """
    errors = exc.errors()
    detail = "; ".join(
        f"{'.'.join(str(part) for part in err.get('loc', ()))}: {err.get('msg', '')}"
        for err in errors
    )
    logger.warning("Request validation failed: %s", detail)
    return _error_response(422, detail or "Request validation failed.")


add_job_ranking_routes(app)
