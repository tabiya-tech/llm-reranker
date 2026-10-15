from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from bson import ObjectId
from pymongo.database import Database

from app.config import Settings
from app.run_logs.repository import IRunLogsRepository, RunStats
from app.server_dependencies.database_collections import Collections


def _stats_fields(stats: RunStats) -> dict[str, int]:
    return {
        "stats.jobs_processed": stats.jobs_processed,
        "stats.upserts_queued": stats.upserts_queued,
        "stats.llm1_calls": stats.llm1_calls,
        "stats.llm2_calls": stats.llm2_calls,
        "stats.llm3_calls": stats.llm3_calls,
    }


class MongoRunLogsRepository(IRunLogsRepository):
    """
    Mongo implementation of ``IRunLogsRepository``: the ``reranker_logs`` collection.
    """

    def __init__(self, db: Database[dict[str, Any]], settings: Settings):
        self._collection = db.get_collection(Collections.from_settings(settings).LOGS)
        self._logger = logging.getLogger(self.__class__.__name__)

    def insert_run(
        self,
        *,
        limit: int,
        model: str,
        source_collection: str,
        target_collection: str,
    ) -> ObjectId:
        doc = {
            "started_at": datetime.utcnow(),
            "completed_at": None,
            "status": "running",
            "success": None,
            "error_message": None,
            "config": {
                "model": model,
                "limit": limit,
                "source_collection": source_collection,
                "target_collection": target_collection,
            },
            "stats": {
                "jobs_processed": 0,
                "upserts_queued": 0,
                "llm1_calls": 0,
                "llm2_calls": 0,
                "llm3_calls": 0,
            },
            "elapsed_s": None,
            "updated_at": datetime.utcnow(),
        }
        result = self._collection.insert_one(doc)
        self._logger.info("ranking-logs | run started | _id=%s", result.inserted_id)
        return result.inserted_id

    def update_progress(self, run_id: ObjectId, stats: RunStats) -> None:
        self._collection.update_one(
            {"_id": run_id},
            {"$set": {**_stats_fields(stats), "updated_at": datetime.utcnow()}},
        )

    def complete_run(
        self,
        run_id: ObjectId,
        stats: RunStats,
        *,
        success: bool,
        elapsed_s: float,
        error_message: str | None = None,
    ) -> None:
        now = datetime.utcnow()
        self._collection.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "completed_at": now,
                    "status": "completed" if success else "failed",
                    "success": success,
                    "error_message": error_message,
                    **_stats_fields(stats),
                    "elapsed_s": round(elapsed_s, 3),
                    "updated_at": now,
                }
            },
        )
        self._logger.info(
            "ranking-logs | run %s | _id=%s | jobs=%s upserts=%s elapsed=%.3fs",
            "completed" if success else "failed",
            run_id,
            stats.jobs_processed,
            stats.upserts_queued,
            elapsed_s,
        )
