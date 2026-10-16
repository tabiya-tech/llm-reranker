"""Builds the job-ranking services; ``get_job_ranking_services`` is the API's dependency provider."""

from __future__ import annotations

import logging
import threading

from fastapi import HTTPException

from app.config import Settings, get_settings
from app.job_ranking.job_attributes.repository import JobAttributesRepository
from app.job_ranking.prompts.repository import PromptRepository
from app.job_ranking.service import JobRankingServiceFactory
from app.llm import TracedLlmClient, build_traced_llm_client
from app.taxonomy.repository import TaxonomyRepository
from app.taxonomy.service import TaxonomyService

logger = logging.getLogger(__name__)

_job_ranking_services_singleton: JobRankingServiceFactory | None = None
_job_ranking_services_lock = threading.Lock()


def build_job_ranking_service_factory(
    settings: Settings, llm: TracedLlmClient
) -> JobRankingServiceFactory:
    """Per-language ranking services sharing ``llm``; each language pack is read on first use."""
    return JobRankingServiceFactory(
        default_settings=settings,
        llm=llm,
        taxonomy=TaxonomyService(TaxonomyRepository()),
        prompts=PromptRepository(),
        job_attributes=JobAttributesRepository(),
    )


def get_job_ranking_services() -> JobRankingServiceFactory:
    """Return the process-wide ranking services, instantiating them on first use.

    The API's lifespan calls this at startup, so a misconfigured model or a broken default
    language pack fails the container's startup probe rather than the first request.

    :raises HTTPException: 503 if the services cannot be built.
    """
    global _job_ranking_services_singleton  # pylint: disable=global-statement

    if _job_ranking_services_singleton is None:
        with _job_ranking_services_lock:
            if _job_ranking_services_singleton is None:
                settings = get_settings()
                try:
                    services = build_job_ranking_service_factory(
                        settings, build_traced_llm_client(settings.llm)
                    )
                    services.warm_up()
                except Exception as e:
                    logger.exception("Job ranking services failed to start: %s", e)
                    raise HTTPException(
                        status_code=503, detail=f"Service not ready: {e.__class__.__name__}"
                    ) from e
                _job_ranking_services_singleton = services
    return _job_ranking_services_singleton


def reset_job_ranking_services() -> None:
    """Close and forget the singleton (API shutdown, tests)."""
    global _job_ranking_services_singleton  # pylint: disable=global-statement

    with _job_ranking_services_lock:
        if _job_ranking_services_singleton is not None:
            _job_ranking_services_singleton.close()
        _job_ranking_services_singleton = None
