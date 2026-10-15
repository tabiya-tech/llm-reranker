from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings


@dataclass(frozen=True)
class Collections:
    """Collection names in the reranker database.

    All four are env-driven (via ``Settings``) because each deployment stack reads and writes its own
    collections.
    """

    # Classified job ads written by the classifier; the batch reads them (MONGODB_SOURCE_COLLECTION).
    SOURCE: str
    # Ranked job documents the batch upserts; the matching engine reads them (MONGODB_TARGET_COLLECTION).
    TARGET: str
    # One document per batch run: status, config and stats (RERANKER_LOGS_COLLECTION).
    LOGS: str
    # One document per country: the incremental-fetch cutoff (RERANKER_STATE_COLLECTION).
    STATE: str

    @classmethod
    def from_settings(cls, settings: Settings) -> Collections:
        return cls(
            SOURCE=settings.mongodb_source_collection,
            TARGET=settings.mongodb_target_collection,
            LOGS=settings.mongodb_logs_collection,
            STATE=settings.mongodb_state_collection,
        )
