"""``POST /rank``, ``POST /rank/skills`` and ``GET /health``."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query

from app.config import log_ranker_run_configuration
from app.constants.errors import error_response_doc
from app.job_ranking.get_job_ranking_service import get_job_ranking_services
from app.job_ranking.routes.types import (
    HealthResponse,
    RankRequest,
    RankResponse,
    RankSkillsRequest,
    RankSkillsResponse,
)
from app.job_ranking.service import JobRankingServiceFactory
from app.languages import normalise_language
from app.llm.errors import LlmConfigurationError, LlmInvocationError, LlmParseError
from app.observability import SamplingTier, job_trace, start_trace, update_observation


def add_job_ranking_routes(app: FastAPI) -> None:
    """Register ``POST /rank``, ``POST /rank/skills`` and ``GET /health``."""
    app.include_router(router)


logger = logging.getLogger(__name__)

router = APIRouter()

_LANGUAGE_QUERY_DESCRIPTION = (
    "Language pack to rank against: a code (`en`, `es`), a locale "
    "(`AR-es`, `es_AR`), or a name (`spanish`). Wins over the body's `language` "
    "field. Unknown values fall back to `en` with a warning."
)

_ERROR_RESPONSES = {
    500: error_response_doc(
        "The pipeline failed, e.g. an LLM call.", "LLM invocation error: LlmInvocationError"
    ),
    503: error_response_doc(
        "The ranking services could not be built.", "Service not ready: FileNotFoundError"
    ),
}


@contextmanager
def _errors_as_http() -> Iterator[None]:
    """Map a pipeline failure to a 500 whose ``detail`` names only the error class."""
    try:
        yield
    except LlmConfigurationError as exc:
        logger.exception("LLM configuration error: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"LLM configuration error: {exc.__class__.__name__}"
        ) from exc
    except LlmInvocationError as exc:
        logger.exception("LLM invocation error: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"LLM invocation error: {exc.__class__.__name__}"
        ) from exc
    except LlmParseError as exc:
        logger.exception("LLM response parse error: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"LLM response parse error: {exc.__class__.__name__}"
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error processing request: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {exc.__class__.__name__}"
        ) from exc


@router.post(
    "/rank",
    response_model=RankResponse,
    # Optional keys the pipeline did not write (e.g. ``llm_is_relevant`` outside LLM3
    # ``call`` mode) stay absent instead of appearing as null.
    response_model_exclude_unset=True,
    responses={
        422: error_response_doc(
            "The body is not a classified job document (see `detail`).",
            "body.classification.entities: Input should be a valid list",
        ),
        **_ERROR_RESPONSES,
    },
)
async def rank_job(
    body: RankRequest,
    services: Annotated[JobRankingServiceFactory, Depends(get_job_ranking_services)],
    language: Annotated[
        str | None,
        Query(description=_LANGUAGE_QUERY_DESCRIPTION, examples=["es"]),
    ] = None,
) -> RankResponse:
    """Rank a single classified job document. Send the document directly as the JSON body.

    ``?language=es`` (or a ``language`` field on the document) picks the resource pack the
    job is ranked against (taxonomy labels, job attributes) and the language the model
    answers in. Omitted, the
    service default applies, so existing callers are unaffected.
    """
    # The query parameter wins over the document's field; neither means the service default.
    requested = language or body.language
    document = body.to_document()
    with _errors_as_http():
        ranking = services.for_language(normalise_language(requested) if requested else None)
        settings = ranking.settings
        log_ranker_run_configuration(
            logger,
            settings,
            bulk_write_batch_size=settings.bulk_write_batch_size,
            cli_limit=0,
            log_prompt_debug=False,
            invocation="api",
        )
        with job_trace(document, settings=settings, invocation="api") as trace:
            result = ranking.rank(document)
            update_observation(trace, output=result.trace_output())
    return RankResponse.from_job_result(result)


@router.post(
    "/rank/skills",
    response_model=RankSkillsResponse,
    responses={
        422: error_response_doc(
            "The body is not a valid skills ranking request (see `detail`).",
            "body.candidates: List should have at least 1 item after validation, not 0",
        ),
        **_ERROR_RESPONSES,
    },
)
def rank_experience_skills(
    body: RankSkillsRequest,
    services: Annotated[JobRankingServiceFactory, Depends(get_job_ranking_services)],
    language: Annotated[
        str | None,
        Query(description=_LANGUAGE_QUERY_DESCRIPTION, examples=["es"]),
    ] = None,
) -> RankSkillsResponse:
    """Rank a jobseeker experience's candidate skills (e.g. classifier matches).

    One LLM call picks the candidates the experience shows, best first, up to
    ``max_skills``. Unlike ``POST /rank`` there is no taxonomy lookup and no job attributes
    call: the caller's candidate ids, labels and descriptions are used as given.
    """
    # A plain ``def``: FastAPI runs it in the threadpool, so the blocking LLM call does not
    # hold up the event loop.
    requested = language or body.language
    t0 = time.perf_counter()
    with _errors_as_http():
        ranking = services.for_language(normalise_language(requested) if requested else None)
        settings = ranking.settings
        with start_trace(
            name="rank_skills",
            tier=SamplingTier.API,
            sampling_key=uuid.uuid4().hex,
            input={"title": body.experience.title, "candidates": len(body.candidates)},
            metadata={
                "language": settings.language,
                "llm_provider": settings.llm.provider,
                "llm_model": settings.llm.model,
            },
            tags=["invocation:api", f"language:{settings.language}"],
        ) as trace:
            t_llm = time.perf_counter()
            result = ranking.rank_experience_skills(
                body.experience, body.candidates, body.max_skills
            )
            llm_s = time.perf_counter() - t_llm
            update_observation(trace, output={"ranked_ids": result.ranked_ids})
    return RankSkillsResponse(
        ranked_ids=result.ranked_ids,
        discarded_ids=result.discarded_ids,
        timings={"llm_s": llm_s, "total_s": time.perf_counter() - t0},
        model=settings.llm.model,
    )


@router.get("/health", response_model=HealthResponse)
async def health(
    services: Annotated[JobRankingServiceFactory, Depends(get_job_ranking_services)],
) -> HealthResponse:
    """Readiness check, plus which language packs this instance serves and has loaded."""
    return HealthResponse.from_status(services.health())
