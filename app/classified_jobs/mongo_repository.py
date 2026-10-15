from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

from pymongo.database import Database

from app.classified_jobs.repository import IClassifiedJobsRepository
from app.config import Settings
from app.server_dependencies.database_collections import Collections


class MongoClassifiedJobsRepository(IClassifiedJobsRepository):
    """
    Mongo implementation of ``IClassifiedJobsRepository``: reads the classifier's job ads.
    """

    def __init__(self, db: Database[dict[str, Any]], settings: Settings):
        self._collection = db.get_collection(Collections.from_settings(settings).SOURCE)

    def iter_jobs(
        self,
        *,
        limit: int = 0,
        skip: int = 0,
        sort_field: str = "_id",
        sort_direction: int = -1,
        query: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        # Long parallel batches can exceed MongoDB's default ~10m idle cursor timeout between chunk reads.
        cur = (
            self._collection.find(query or {}, no_cursor_timeout=True)
            .sort(sort_field, sort_direction)
            .skip(skip)
        )
        if limit > 0:
            cur = cur.limit(limit)
        yield from cur

    def iter_jobs_chunks(
        self,
        chunk_size: int,
        *,
        limit: int = 0,
        skip: int = 0,
        sort_field: str = "_id",
        sort_direction: int = -1,
        query: dict[str, Any] | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        it = self.iter_jobs(
            limit=limit,
            skip=skip,
            sort_field=sort_field,
            sort_direction=sort_direction,
            query=query,
        )
        while True:
            chunk = list(itertools.islice(it, chunk_size))
            if not chunk:
                break
            yield chunk
