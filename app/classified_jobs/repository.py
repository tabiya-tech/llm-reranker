from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from app.classified_jobs.types import ClassifiedJobsFilter


class IClassifiedJobsRepository(ABC):
    """
    Interface for the classified jobs (source) repository.
    Allows to mock the repository in tests.
    """

    @abstractmethod
    def iter_jobs(
        self,
        *,
        limit: int = 0,
        skip: int = 0,
        sort_field: str = "_id",
        sort_direction: int = -1,
        job_filter: ClassifiedJobsFilter | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Stream classified job documents matching ``job_filter``, newest first by default.

        ``limit`` <= 0 means no limit (entire collection). Results are streamed one document at a time.
        """
        raise NotImplementedError

    @abstractmethod
    def iter_jobs_chunks(
        self,
        chunk_size: int,
        *,
        limit: int = 0,
        skip: int = 0,
        sort_field: str = "_id",
        sort_direction: int = -1,
        job_filter: ClassifiedJobsFilter | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """
        Yield lists of documents in the same order as ``iter_jobs``.

        Use for parallel batch processing without loading unlimited collections into memory.
        ``chunk_size`` is typically ``>= RERANKER_PARALLEL_WORKERS``. Raises ``ValueError`` if < 1.
        """
        raise NotImplementedError
