"""Scraper/classifier enrichment fields (Kenya + Zambia post-processing)."""

from __future__ import annotations

from typing import Any

# Mirror tabiya-livelihoods-classifier util/classified_job_schema.py
SCRAPER_ENRICHMENT_FIELD_NAMES: tuple[str, ...] = (
    "city",
    "province",
    "sector",
    "domain",
    "isco_major_group",
    "isco_major_group_label",
    "isco_sub_major_group",
    "isco_sub_major_group_label",
    "is_priority_sector",
    "zqf_min",
    "zqf_max",
    "zqf_min_label",
    "zqf_max_label",
)


def copy_scraper_enrichment_fields(doc: dict[str, Any]) -> dict[str, Any]:
    """Copy enrichment keys from a classified (or scraped) job when present."""
    out: dict[str, Any] = {}
    for key in SCRAPER_ENRICHMENT_FIELD_NAMES:
        val = doc.get(key)
        if val is not None:
            out[key] = val
    return out
