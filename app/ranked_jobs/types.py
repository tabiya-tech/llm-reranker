from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RankedJobKey(str, Enum):
    """Which identity a ranked job is matched on in the target collection (first one present wins)."""

    JOB_ID = "job_id"
    JOB_FINGERPRINT = "job_fingerprint"
    SOURCE_CLASSIFIED_ID = "source_classified_id"


@dataclass(frozen=True)
class RankedJobUpsert:
    """One ranked document to write, matched on ``key`` = ``key_value``."""

    key: RankedJobKey
    key_value: Any
    document: dict[str, Any]
