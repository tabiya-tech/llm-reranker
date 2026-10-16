"""Builds the work-activities enrichment over the configured target collection."""

from __future__ import annotations

from typing import Any

from pymongo.database import Database

from app.config import Settings
from app.ranked_jobs.mongo_repository import MongoRankedJobsRepository
from app.work_activities_enrichment.repository import WorkActivityTaxonomyRepository
from app.work_activities_enrichment.service import WorkActivitiesEnrichmentService


def build_work_activities_enrichment_service(
    settings: Settings, db: Database[dict[str, Any]]
) -> WorkActivitiesEnrichmentService:
    """The enrichment over ``ENRICH_WA_DATABASE.ENRICH_WA_TARGET_COLLECTION`` on ``db``'s cluster."""
    target = MongoRankedJobsRepository(
        db.client[settings.enrich_wa_database],
        settings,
        collection_name=settings.enrich_wa_target_collection,
    )
    return WorkActivitiesEnrichmentService(settings, target, WorkActivityTaxonomyRepository())
