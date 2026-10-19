from __future__ import annotations

from abc import ABC, abstractmethod

from bson import ObjectId
from pydantic import Field

from app.domain_base import DomainModel


class RunStats(DomainModel):
    """Counters a batch run reports on its log document."""

    jobs_processed: int = Field(description="Jobs processed so far.")
    upserts_queued: int = Field(description="MongoDB upserts queued so far.")
    llm1_calls: int = Field(description="LLM1 (skill ranker) calls made so far.")
    llm2_calls: int = Field(description="LLM2 (job attributes) calls made so far.")
    llm3_calls: int = Field(description="LLM3 (occupation-skills relevance) calls made so far.")


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
