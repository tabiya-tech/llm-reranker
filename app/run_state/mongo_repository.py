from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo.database import Database

from app.config import Settings
from app.run_state.repository import IRunStateRepository
from app.server_dependencies.database_collections import Collections


class MongoRunStateRepository(IRunStateRepository):
    """
    Mongo implementation of ``IRunStateRepository``: the ``reranker_state`` collection.
    """

    def __init__(self, db: Database[dict[str, Any]], settings: Settings):
        self._collection = db.get_collection(Collections.from_settings(settings).STATE)

    def get_last_cutoff_date(self, country: str) -> str | None:
        state_doc = self._collection.find_one({"country": country}) or {}
        last_cutoff = state_doc.get("last_cutoff_date")
        if not last_cutoff:
            return None
        if isinstance(last_cutoff, datetime):
            dt = last_cutoff if last_cutoff.tzinfo else last_cutoff.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
        return datetime.fromisoformat(str(last_cutoff).strip().replace("Z", "+00:00")).strftime(
            "%Y-%m-%d"
        )

    def save(self, country: str, *, jobs_processed: int) -> None:
        now = datetime.now(timezone.utc)
        self._collection.update_one(
            {"country": country},
            {
                "$set": {
                    "country": country,
                    "last_cutoff_date": now.strftime("%Y-%m-%d"),
                    "last_run_at": now,
                    "jobs_processed": int(jobs_processed),
                }
            },
            upsert=True,
        )
