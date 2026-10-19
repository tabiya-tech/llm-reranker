from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator
from typing import Any

from pydantic import ValidationError
from pymongo.database import Database

from app.classified_jobs.repository import IClassifiedJobsRepository
from app.classified_jobs.types import ClassifiedJobsFilter
from app.config import Settings
from app.job_context import stringify_top_level_object_ids
from app.job_ranking.routes.types import RankRequest
from app.server_dependencies.database_collections import Collections

logger = logging.getLogger(__name__)


class MongoClassifiedJobsRepository(IClassifiedJobsRepository):
    """
    Mongo implementation of ``IClassifiedJobsRepository``: reads the classifier's job ads.
    """

    def __init__(self, db: Database[dict[str, Any]], settings: Settings):
        self._collection = db.get_collection(Collections.from_settings(settings).SOURCE)
        self._job_id_field = settings.mongo_field_job_id

    def build_query(self, job_filter: ClassifiedJobsFilter | None) -> dict[str, Any]:
        """The Mongo filter for ``job_filter`` (``{}`` matches every document)."""
        if job_filter is None:
            return {}
        clauses: list[dict[str, Any]] = []
        if job_filter.classified_since is not None:
            # Anchor on immutable `classified_at` only (set once, on first classifier insert).
            # Avoids matching docs whose `updated_at` gets bumped by re-classification.
            clauses.append({"classified_at": {"$gte": job_filter.classified_since}})
        if job_filter.exclude_job_ids:
            clauses.append({self._job_id_field: {"$nin": list(job_filter.exclude_job_ids)}})
        if not clauses:
            return {}
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def iter_jobs(
        self,
        *,
        limit: int = 0,
        skip: int = 0,
        sort_field: str = "_id",
        sort_direction: int = -1,
        job_filter: ClassifiedJobsFilter | None = None,
    ) -> Iterator[RankRequest]:
        # Long parallel batches can exceed MongoDB's default ~10m idle cursor timeout between chunk reads.
        cur = (
            self._collection.find(self.build_query(job_filter), no_cursor_timeout=True)
            .sort(sort_field, sort_direction)
            .skip(skip)
        )
        if limit > 0:
            cur = cur.limit(limit)
        for raw_doc in cur:
            try:
                yield RankRequest.model_validate(stringify_top_level_object_ids(raw_doc))
            except ValidationError as e:
                logger.warning(
                    "Skipping invalid classified document _id=%s: %s", raw_doc.get("_id"), e
                )

    def iter_jobs_chunks(
        self,
        chunk_size: int,
        *,
        limit: int = 0,
        skip: int = 0,
        sort_field: str = "_id",
        sort_direction: int = -1,
        job_filter: ClassifiedJobsFilter | None = None,
    ) -> Iterator[list[RankRequest]]:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        it = self.iter_jobs(
            limit=limit,
            skip=skip,
            sort_field=sort_field,
            sort_direction=sort_direction,
            job_filter=job_filter,
        )
        while True:
            chunk = list(itertools.islice(it, chunk_size))
            if not chunk:
                break
            yield chunk
