"""Builds the batch run around the run's database."""

from __future__ import annotations

from typing import Any

from pymongo.database import Database

from app.batch.post_steps import BatchPostSteps
from app.batch.service import BatchRankingService
from app.classified_jobs.mongo_repository import MongoClassifiedJobsRepository
from app.config import Settings
from app.job_embedding.gemini_client import GeminiEmbeddingClient
from app.job_embedding.service import JobEmbeddingService
from app.job_embedding.whitening_repository import WhiteningArtifactRepository
from app.job_ranking.get_job_ranking_service import build_job_ranking_service_factory
from app.llm import build_traced_llm_client
from app.ranked_jobs.mongo_repository import MongoRankedJobsRepository
from app.run_logs.mongo_repository import MongoRunLogsRepository
from app.run_state.mongo_repository import MongoRunStateRepository
from app.work_activities_enrichment.get_work_activities_enrichment_service import (
    build_work_activities_enrichment_service,
)


def build_batch_ranking_service(
    settings: Settings, db: Database[dict[str, Any]]
) -> BatchRankingService:
    """The batch run over ``db`` (the run's database; its client is closed by the caller).

    :raises LlmConfigurationError: the configured model cannot be built.
    """
    llm = build_traced_llm_client(settings.llm)
    ranked_jobs = MongoRankedJobsRepository(db, settings)
    return BatchRankingService(
        settings=settings,
        llm_model=llm.model,
        rankers=build_job_ranking_service_factory(settings, llm),
        embedding=JobEmbeddingService(
            settings, GeminiEmbeddingClient.from_settings(settings), WhiteningArtifactRepository()
        ),
        classified_jobs=MongoClassifiedJobsRepository(db, settings),
        ranked_jobs=ranked_jobs,
        run_logs=MongoRunLogsRepository(db, settings),
        run_state=MongoRunStateRepository(db, settings),
        post_steps=BatchPostSteps(
            settings,
            ranked_jobs,
            enrichment=lambda: build_work_activities_enrichment_service(settings, db),
        ),
    )
