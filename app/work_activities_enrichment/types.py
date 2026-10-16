"""Work-activities enrichment data and counters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkActivityTaxonomy:
    """``combined_occupation_database_with_wa.json``, indexed.

    ``occ_lookup``: lowercase occupation label → WA code → ``{importance, level}``.
    ``wa_averages``: WA code → cross-occupation average ``{importance, level}``.
    """

    occ_lookup: dict[str, dict[str, dict[str, float]]]
    wa_averages: dict[str, dict[str, float]]


@dataclass
class WorkActivitiesEnrichmentStats:
    scanned: int = 0
    with_occ_match: int = 0
    with_avg_match: int = 0
    with_default: int = 0
    no_data: int = 0
    queued: int = 0
    applied: int = 0
