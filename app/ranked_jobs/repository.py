from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Literal

from pymongo.operations import UpdateOne

from utils.update_is_active_from_closing_date import IsActiveSyncStats


class IRankedJobsRepository(ABC):
    """
    Interface for the ranked jobs (target) repository.
    Allows to mock the repository in tests.
    """

    @abstractmethod
    def distinct_job_ids(self) -> list[Any]:
        """Return ``job_id`` values already stored on the target (for resume without redoing the LLM calls)."""
        raise NotImplementedError

    @abstractmethod
    def bulk_upsert(self, operations: list[UpdateOne]) -> None:
        """Run an unordered ``bulk_write`` of ranked-job upserts (see ``ranked_job_upsert_update_one``).

        A no-op for an empty list.
        """
        raise NotImplementedError

    @abstractmethod
    def sync_is_active_from_closing_date(
        self,
        *,
        today: date | None = None,
        missing_policy: Literal["keep", "active", "inactive"] = "keep",
        dry_run: bool = False,
    ) -> IsActiveSyncStats:
        """Set ``is_active`` on every ranked job from ``classifier_metadata.closing_date``."""
        raise NotImplementedError
