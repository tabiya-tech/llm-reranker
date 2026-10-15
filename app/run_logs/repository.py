from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from bson import ObjectId


@dataclass(frozen=True)
class RunStats:
    """Counters a batch run reports on its log document."""

    jobs_processed: int
    upserts_queued: int
    llm1_calls: int
    llm2_calls: int
    llm3_calls: int


class IRunLogsRepository(ABC):
    """
    Interface for the batch run log repository (one document per batch run).
    Allows to mock the repository in tests.
    """

    @abstractmethod
    def insert_run(
        self,
        *,
        limit: int,
        model: str,
        source_collection: str,
        target_collection: str,
    ) -> ObjectId:
        """Insert a new run document with status ``running`` and return its ``_id``."""
        raise NotImplementedError

    @abstractmethod
    def update_progress(self, run_id: ObjectId, stats: RunStats) -> None:
        """Update stats and ``updated_at`` on the run document (called periodically)."""
        raise NotImplementedError

    @abstractmethod
    def complete_run(
        self,
        run_id: ObjectId,
        stats: RunStats,
        *,
        success: bool,
        elapsed_s: float,
        error_message: str | None = None,
    ) -> None:
        """Mark the run as ``completed`` or ``failed`` with final stats."""
        raise NotImplementedError
