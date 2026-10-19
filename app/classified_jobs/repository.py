from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from app.classified_jobs.types import ClassifiedJobsFilter
from app.job_ranking.routes.types import RankRequest


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
    ) -> Iterator[RankRequest]:
        """
        Stream classified job documents matching ``job_filter``, newest first by default.

        ``limit`` <= 0 means no limit (entire collection). Each document is validated as a
        :class:`~app.job_ranking.routes.types.RankRequest` before being yielded; a document that
        fails validation is logged and skipped rather than aborting the whole stream.
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
    ) -> Iterator[list[RankRequest]]:
        """
        Yield lists of documents in the same order as ``iter_jobs``.

        Use for parallel batch processing without loading unlimited collections into memory.
        ``chunk_size`` is typically ``>= RERANKER_PARALLEL_WORKERS``. Raises ``ValueError`` if < 1.
        """
        raise NotImplementedError
