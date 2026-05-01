"""Shared job listing context (Classifier document fields + full ``classified`` doc)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JobDescriptionContext:
    """
    Fields needed for downstream steps, sourced from classified job documents.

    Field names used to populate this object come from ``Settings`` (environment variables).

    The full Mongo document is always available on ``classified`` (includes ``classification``,
    ``metadata``, etc.).
    """

    job_id: Any | None
    job_fingerprint: str | None
    title: str | None
    employer: str | None
    category: str | None
    application_url: str | None
    posted_date: str | None
    closing_date: str | None
    source_platform: str | None
    raw_job_description: str | None
    requirements_text: str | None
    salary: str | None
    city: str | None
    county: str | None
    employment_type: str | None
    classified: dict[str, Any]
