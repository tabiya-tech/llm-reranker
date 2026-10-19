from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from app.domain_base import DomainModel


class ClassifiedJobsFilter(DomainModel):
    """Which classified jobs a batch reads. The default matches the whole collection."""

    classified_since: datetime | None = Field(
        default=None,
        description="Only jobs first classified at or after this instant (the incremental cutoff).",
    )
    exclude_job_ids: tuple[Any, ...] = Field(
        default=(),
        description="Skip jobs whose job id is one of these (already ranked, for resume).",
    )
