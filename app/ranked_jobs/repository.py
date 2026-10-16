from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import date
from typing import Any

from app.ranked_jobs.is_active import IsActiveSyncStats, MissingPolicy
from app.ranked_jobs.types import RankedJobUpsert


class IRankedJobsRepository(ABC):
    """
    Interface for the ranked jobs (target) repository.
    Allows to mock the repository in tests.
    """

    @property
    @abstractmethod
    def location(self) -> str:
        """``database.collection`` this repository reads and writes, for log lines."""
        raise NotImplementedError

    @abstractmethod
    def distinct_job_ids(self) -> list[Any]:
        """Return ``job_id`` values already stored on the target (for resume without redoing the LLM calls)."""
        raise NotImplementedError

    @abstractmethod
    def bulk_upsert(self, upserts: list[RankedJobUpsert]) -> None:
        """Write ranked documents in one unordered batch, inserting those not yet stored.

        A no-op for an empty list.
        """
        raise NotImplementedError

    @abstractmethod
    def sync_is_active_from_closing_date(
        self,
        *,
        today: date | None = None,
        missing_policy: MissingPolicy = "keep",
        dry_run: bool = False,
    ) -> IsActiveSyncStats:
        """Set ``is_active`` on every ranked job from ``classifier_metadata.closing_date``."""
        raise NotImplementedError

    @abstractmethod
    def iter_for_work_activities_enrichment(
        self, *, only_missing: bool = True, limit: int = 0
    ) -> Iterator[dict[str, Any]]:
        """Ranked jobs to enrich, newest first; ``only_missing`` skips those already enriched.

        ``limit`` <= 0 means no limit.
        """
        raise NotImplementedError

    @abstractmethod
    def iter_whitening_backfill_candidates(
        self, *, limit: int | None = None
    ) -> Iterator[dict[str, Any]]:
        """Ranked jobs with a ``job_embedding`` not yet whitened (embedding fields and meta only)."""
        raise NotImplementedError

    @abstractmethod
    def bulk_set(self, updates: list[tuple[Any, dict[str, Any]]]) -> int:
        """``$set`` fields per ``(_id, fields)`` (no upsert); returns the modified count."""
        raise NotImplementedError
