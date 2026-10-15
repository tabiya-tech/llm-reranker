from __future__ import annotations

import logging
from datetime import date
from typing import Any, Literal

from pymongo.database import Database
from pymongo.operations import UpdateOne

from app.config import Settings
from app.ranked_jobs.repository import IRankedJobsRepository
from app.server_dependencies.database_collections import Collections
from utils.update_is_active_from_closing_date import (
    IsActiveSyncStats,
    sync_is_active_from_closing_date,
)


class MongoRankedJobsRepository(IRankedJobsRepository):
    """
    Mongo implementation of ``IRankedJobsRepository``: the ranked job documents the batch writes.
    """

    def __init__(self, db: Database[dict[str, Any]], settings: Settings):
        self._database_name = db.name
        self._collection_name = Collections.from_settings(settings).TARGET
        self._collection = db.get_collection(self._collection_name)
        self._job_id_field = settings.mongo_field_job_id
        self._logger = logging.getLogger(self.__class__.__name__)

    def distinct_job_ids(self) -> list[Any]:
        field = self._job_id_field
        ids = self._collection.distinct(field, {field: {"$exists": True, "$ne": None}})
        return [x for x in ids if x is not None]

    def bulk_upsert(self, operations: list[UpdateOne]) -> None:
        if not operations:
            return
        result = self._collection.bulk_write(operations, ordered=False)
        self._logger.info(
            "Bulk upsert LLM results → %s.%s: ops=%s matched=%s modified=%s upserted=%s",
            self._database_name,
            self._collection_name,
            len(operations),
            result.matched_count,
            result.modified_count,
            result.upserted_count,
        )

    def sync_is_active_from_closing_date(
        self,
        *,
        today: date | None = None,
        missing_policy: Literal["keep", "active", "inactive"] = "keep",
        dry_run: bool = False,
    ) -> IsActiveSyncStats:
        return sync_is_active_from_closing_date(
            self._collection, today=today, missing_policy=missing_policy, dry_run=dry_run
        )
