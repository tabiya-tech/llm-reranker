from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ClassifiedJobsFilter:
    """Which classified jobs a batch reads. The default matches the whole collection."""

    # Only jobs first classified at or after this instant (the incremental cutoff).
    classified_since: datetime | None = None
    # Skip jobs whose job id is one of these (already ranked, for resume).
    exclude_job_ids: tuple[Any, ...] = ()
