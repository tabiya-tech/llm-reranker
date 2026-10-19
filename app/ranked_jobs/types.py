from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from app.domain_base import DomainModel


class RankedJobKey(str, Enum):
    """Which identity a ranked job is matched on in the target collection (first one present wins)."""

    JOB_ID = "job_id"
    JOB_FINGERPRINT = "job_fingerprint"
    SOURCE_CLASSIFIED_ID = "source_classified_id"


class RankedJobUpsert(DomainModel):
    """One ranked document to write, matched on ``key`` = ``key_value``."""

    key: RankedJobKey = Field(description="Which identity field `key_value` matches on.")
    key_value: Any = Field(description="The value to match `key` against in the target collection.")
    document: dict[str, Any] = Field(
        description="The ranked document's `$set` payload, with the job embedding attached."
    )
