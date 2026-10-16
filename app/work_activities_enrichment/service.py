"""Sets ``onet_work_activities`` on ranked jobs (a batch post-step, or ``scripts/``)."""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.ranked_jobs.repository import IRankedJobsRepository
from app.work_activities_enrichment.repository import WorkActivityTaxonomyRepository
from app.work_activities_enrichment.rules import (
    build_onet_work_activity_rows,
)
from app.work_activities_enrichment.types import WorkActivitiesEnrichmentStats

logger = logging.getLogger(__name__)


class WorkActivitiesEnrichmentService:
    """Sets ``onet_work_activities`` on ranked jobs from their ``work_activity.items``.

    For each work-activity code: the matched occupation's importance / level from
    ``combined_occupation_database_with_wa.json``, else the cross-occupation average, else
    "code is not available".
    """

    def __init__(
        self,
        settings: Settings,
        target: IRankedJobsRepository,
        taxonomy: WorkActivityTaxonomyRepository,
    ):
        self._settings = settings
        self._target = target
        self._taxonomy = taxonomy

    def run(self) -> WorkActivitiesEnrichmentStats:
        s = self._settings
        logger.info(
            "work_activities_enrichment | target=%s | taxonomy=%s | only_missing=%s | "
            "dry_run=%s | limit=%s",
            self._target.location,
            s.enrich_wa_occupation_db_path,
            s.enrich_wa_only_missing,
            s.enrich_wa_dry_run,
            s.enrich_wa_limit or "all",
        )
        taxonomy = self._taxonomy.load(s.enrich_wa_occupation_db_path)
        logger.info(
            "work_activities_enrichment | taxonomy loaded | occupations_with_WA=%s "
            "wa_codes_with_averages=%s",
            len(taxonomy.occ_lookup),
            len(taxonomy.wa_averages),
        )

        stats = WorkActivitiesEnrichmentStats()
        pending: list[tuple[Any, dict[str, Any]]] = []

        def flush() -> None:
            if pending and not s.enrich_wa_dry_run:
                stats.applied += self._target.bulk_set(pending)
            pending.clear()

        for doc in self._target.iter_for_work_activities_enrichment(
            only_missing=s.enrich_wa_only_missing, limit=s.enrich_wa_limit
        ):
            stats.scanned += 1
            rows, used_occ_match = build_onet_work_activity_rows(doc, taxonomy=taxonomy)
            if not rows:
                stats.no_data += 1
                continue
            if used_occ_match:
                stats.with_occ_match += 1
            elif any(str(x.get("WA_code") or "").strip() in taxonomy.wa_averages for x in rows):
                stats.with_avg_match += 1
            else:
                stats.with_default += 1
            pending.append((doc["_id"], {"onet_work_activities": rows}))
            stats.queued += 1
            if len(pending) >= s.enrich_wa_bulk_batch_size:
                flush()
        flush()

        logger.info(
            "work_activities_enrichment | done | scanned=%s with_occ_match=%s with_avg_match=%s "
            "with_default=%s no_data=%s updates_queued=%s updates_applied=%s",
            stats.scanned,
            stats.with_occ_match,
            stats.with_avg_match,
            stats.with_default,
            stats.no_data,
            stats.queued,
            stats.applied,
        )
        return stats
