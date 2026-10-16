"""What the batch runs over the whole target collection after ranking."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from app.config import Settings
from app.ranked_jobs.repository import IRankedJobsRepository
from app.work_activities_enrichment.service import WorkActivitiesEnrichmentService

logger = logging.getLogger(__name__)


class BatchPostSteps:
    """The batch's post-steps, each behind its ``RERANKER_*`` switch and failure policy.

    ``enrichment`` is a factory: the enrichment's configuration and target are resolved when
    the step runs, so a misconfiguration is handled by the step's failure policy.
    """

    def __init__(
        self,
        settings: Settings,
        ranked_jobs: IRankedJobsRepository,
        enrichment: Callable[[], WorkActivitiesEnrichmentService],
    ):
        self._settings = settings
        self._ranked_jobs = ranked_jobs
        self._enrichment = enrichment

    def run(self, *, jobs_processed: int, run_error: str | None) -> str | None:
        """Run the enabled post-steps; returns the run's error, which enrichment may set."""
        if self._settings.reranker_run_wa_enrichment and jobs_processed > 0:
            run_error = self.enrich_work_activities(
                jobs_processed=jobs_processed, run_error=run_error
            )
        if self._settings.reranker_sync_is_active_from_closing_date and run_error is None:
            self.sync_is_active()
        return run_error

    def enrich_work_activities(self, *, jobs_processed: int, run_error: str | None) -> str | None:
        """Runs even after a failed batch (the persisted jobs still need it).

        A failure is a warning, or the run's error with ``RERANKER_FAIL_ON_WA_ENRICHMENT_ERROR``.
        """
        target = self._settings.mongodb_target_collection
        try:
            t0 = time.perf_counter()
            if run_error is not None:
                logger.warning(
                    "Post-step | work_activities_enrichment | batch had errors — "
                    "running anyway on target=%s for %s persisted job(s)",
                    target,
                    jobs_processed,
                )
            logger.info("Post-step | work_activities_enrichment | start | target=%s", target)
            self._enrichment().run()
            logger.info(
                "Post-step | work_activities_enrichment | done | elapsed_s=%.3f",
                time.perf_counter() - t0,
            )
        except Exception as e:
            if self._settings.reranker_fail_on_wa_enrichment_error:
                run_error = f"RuntimeError: post enrichment failed: {e}"
                logger.exception(run_error)
            else:
                logger.warning("Post-step | work_activities_enrichment | skipped-on-error | %s", e)
        return run_error

    def sync_is_active(self) -> None:
        """Set ``is_active`` from ``closing_date``; a failure is only a warning."""
        try:
            t0 = time.perf_counter()
            logger.info(
                "Post-step | is_active from closing_date | start | target=%s",
                self._settings.mongodb_target_collection,
            )
            stats = self._ranked_jobs.sync_is_active_from_closing_date(
                missing_policy="keep", dry_run=False
            )
            logger.info(
                "Post-step | is_active from closing_date | done | total=%s parsed=%s "
                "missing_or_bad=%s set_true=%s set_false=%s unchanged=%s modified=%s elapsed_s=%.3f",
                stats["total"],
                stats["parsed"],
                stats["missing_or_bad"],
                stats["will_set_true"],
                stats["will_set_false"],
                stats["unchanged"],
                stats["modified"],
                time.perf_counter() - t0,
            )
        except Exception as e:
            logger.warning("Post-step | is_active from closing_date | failed | %s", e)
