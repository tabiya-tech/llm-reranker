from __future__ import annotations

from pydantic import Field

from app.config import Settings
from app.domain_base import DomainModel


class Collections(DomainModel):
    """Collection names in the reranker database.

    All four are env-driven (via ``Settings``) because each deployment stack reads and writes its own
    collections.
    """

    SOURCE: str = Field(
        description="Classified job ads written by the classifier; the batch reads them "
        "(`MONGODB_SOURCE_COLLECTION`)."
    )
    TARGET: str = Field(
        description="Ranked job documents the batch upserts; the matching engine reads them "
        "(`MONGODB_TARGET_COLLECTION`)."
    )
    LOGS: str = Field(
        description="One document per batch run: status, config and stats "
        "(`RERANKER_LOGS_COLLECTION`)."
    )
    STATE: str = Field(
        description="One document per country: the incremental-fetch cutoff "
        "(`RERANKER_STATE_COLLECTION`)."
    )

    @classmethod
    def from_settings(cls, settings: Settings) -> Collections:
        return cls(
            SOURCE=settings.mongodb_source_collection,
            TARGET=settings.mongodb_target_collection,
            LOGS=settings.reranker_logs_collection,
            STATE=settings.reranker_state_collection,
        )
