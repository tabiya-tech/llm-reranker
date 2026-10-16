"""The batch run: classified jobs → per-job pipeline → ranked jobs upsert → post-steps."""

from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.batch.job_ranker import BatchJobOptions, BatchJobRanker, RankOneJobResult, elapsed_s
from app.batch.post_steps import BatchPostSteps
from app.classified_jobs.repository import IClassifiedJobsRepository
from app.classified_jobs.types import ClassifiedJobsFilter
from app.config import Settings, log_ranker_run_configuration
from app.job_embedding.service import JobEmbeddingService
from app.job_ranking.service import JobRankingServiceFactory
from app.job_ranking.types import NO_PROMPT_LOGGING, PromptLogging
from app.observability import job_trace, update_observation
from app.ranked_jobs.repository import IRankedJobsRepository
from app.ranked_jobs.types import RankedJobUpsert
from app.run_logs.repository import IRunLogsRepository, RunStats
from app.run_state.repository import IRunStateRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchOptions:
    """What the batch CLI asks for (``main.py --help``)."""

    # Max documents from the source collection; 0 = the whole collection.
    limit: int = 0
    verbose: bool = False
    prompt_logging: PromptLogging = NO_PROMPT_LOGGING
    # Overrides ``BULK_WRITE_BATCH_SIZE`` when set.
    bulk_batch_size: int | None = None
    # ``--skip-already-ranked`` (``RERANKER_SKIP_ALREADY_RANKED`` also turns it on).
    skip_already_ranked: bool = False


@dataclass
class _Counters:
    jobs: int = 0
    upserts: int = 0
    llm1_calls: int = 0
    llm2_calls: int = 0
    llm3_calls: int = 0
    job_errors: int = 0

    def run_stats(self) -> RunStats:
        return RunStats(
            jobs_processed=self.jobs,
            upserts_queued=self.upserts,
            llm1_calls=self.llm1_calls,
            llm2_calls=self.llm2_calls,
            llm3_calls=self.llm3_calls,
        )


class BatchRankingService:
    """Owns one batch run end to end; ``run`` returns whether it succeeded."""

    def __init__(
        self,
        *,
        settings: Settings,
        llm_model: str,
        rankers: JobRankingServiceFactory,
        embedding: JobEmbeddingService,
        classified_jobs: IClassifiedJobsRepository,
        ranked_jobs: IRankedJobsRepository,
        run_logs: IRunLogsRepository,
        run_state: IRunStateRepository,
        post_steps: BatchPostSteps,
    ):
        self._settings = settings
        self._llm_model = llm_model
        self._rankers = rankers
        self._embedding = embedding
        self._classified_jobs = classified_jobs
        self._ranked_jobs = ranked_jobs
        self._run_logs = run_logs
        self._run_state = run_state
        self._post_steps = post_steps

    # -- source filter ---------------------------------------------------------

    def resolve_cutoff_date(self) -> str | None:
        """YYYY-MM-DD lower bound for the incremental fetch.

        Order: per-country env (e.g. ``KENYA_RERANKER_CUTOFF_DATE``) or ``RERANKER_CUTOFF_DATE``
        (used as-is); else ``reranker_state.last_cutoff_date`` for this country (used as-is, no
        day overlap).
        """
        country = self._settings.target_country
        env_specific = f"{country.upper()}_RERANKER_CUTOFF_DATE"
        env_override = (os.getenv(env_specific) or os.getenv("RERANKER_CUTOFF_DATE") or "").strip()
        if env_override:
            logger.info(
                "[%s] Cutoff: env override (%s=%s)", country.upper(), env_specific, env_override
            )
            return env_override

        cutoff_str = self._run_state.get_last_cutoff_date(country)
        if cutoff_str:
            logger.info(
                "[%s] Cutoff: incremental from state last_cutoff_date=%s (exact)",
                country.upper(),
                cutoff_str,
            )
            return cutoff_str

        logger.info("[%s] Cutoff: no previous state (full scan)", country.upper())
        return None

    def build_source_filter(self, *, skip_already_ranked: bool) -> ClassifiedJobsFilter:
        """Which classified jobs to read: the incremental cutoff, minus already-ranked jobs."""
        settings = self._settings
        classified_since: datetime | None = None
        cutoff_date = self.resolve_cutoff_date()
        if cutoff_date:
            try:
                classified_since = datetime.strptime(cutoff_date, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                logger.warning(
                    "[%s] Invalid cutoff_date=%s, falling back to full scan",
                    settings.target_country.upper(),
                    cutoff_date,
                )

        exclude_job_ids: tuple[Any, ...] = ()
        if skip_already_ranked or settings.reranker_skip_already_ranked:
            exclude_job_ids = tuple(self._ranked_jobs.distinct_job_ids())
            logger.info(
                "Resume | skip_already_ranked | excluding %s %s already in %s",
                len(exclude_job_ids),
                settings.mongo_field_job_id,
                self._ranked_jobs.location,
            )
        return ClassifiedJobsFilter(
            classified_since=classified_since, exclude_job_ids=exclude_job_ids
        )

    # -- run -------------------------------------------------------------------

    def run(self, options: BatchOptions) -> bool:
        settings = self._settings
        prompt_debug = options.prompt_logging.any
        batch_size = (
            options.bulk_batch_size
            if options.bulk_batch_size is not None
            else settings.bulk_write_batch_size
        )
        batch_size = max(1, min(batch_size, 10_000))

        t_startup = time.perf_counter()
        ranking = self._rankers.get(settings)
        taxonomy = self._rankers.taxonomy.get(settings)
        startup_s = elapsed_s(t_startup)
        job_ranker = BatchJobRanker(
            settings=settings,
            ranking=ranking,
            embedding=self._embedding,
            options=BatchJobOptions(prompt_logging=options.prompt_logging, verbose=options.verbose),
        )

        logger.info(
            "Batch start | db=%s | limit=%s | target_collection=%s bulk_batch_size=%s | "
            "startup_load_s=%.3f | taxonomy_relation_occupations=%s | llm=%s",
            settings.mongodb_database,
            "all" if options.limit <= 0 else str(options.limit),
            settings.mongodb_target_collection,
            batch_size,
            startup_s,
            len(taxonomy.occ_map),
            settings.llm.describe(),
        )
        logger.debug(
            "Batch config detail | job_attributes_schema=%s occupation_relations_csv=%s "
            "occupations_csv=%s skills_csv=%s classified_linked_entity_types=%s",
            settings.job_attributes_schema_path,
            settings.occupation_to_skill_relations_path,
            settings.occupations_csv_path,
            settings.skills_csv_path,
            list(settings.classified_linked_entity_types),
        )
        log_ranker_run_configuration(
            logger,
            settings,
            bulk_write_batch_size=batch_size,
            cli_limit=options.limit,
            log_prompt_debug=prompt_debug,
            invocation="batch",
        )

        job_filter = self.build_source_filter(skip_already_ranked=options.skip_already_ranked)
        run_id = self._run_logs.insert_run(
            limit=options.limit,
            model=self._llm_model,
            source_collection=settings.mongodb_source_collection,
            target_collection=settings.mongodb_target_collection,
        )

        counters = _Counters()
        pending: list[RankedJobUpsert] = []
        run_error: str | None = None
        t_batch = time.perf_counter()

        def apply_result(r: RankOneJobResult) -> None:
            counters.llm1_calls += r.llm1_inc
            counters.llm2_calls += r.llm2_inc
            counters.llm3_calls += r.llm3_inc
            counters.upserts += r.upsert_increment
            if r.upsert is not None:
                pending.append(r.upsert)
                if len(pending) >= batch_size:
                    self._ranked_jobs.bulk_upsert(pending)
                    pending.clear()
            counters.jobs += 1
            if counters.jobs % 10 == 0:
                self._run_logs.update_progress(run_id, counters.run_stats())

        def rank_one(doc: dict[str, Any], job_label: str) -> RankOneJobResult:
            # One Langfuse trace per job, grouped into a session per batch run.
            with job_trace(
                doc,
                settings=settings,
                invocation="batch",
                session_id=run_id,
                job_label=job_label,
            ) as trace:
                r = job_ranker.rank(doc, job_label=job_label)
                update_observation(
                    trace,
                    output={
                        "ran_llm1": bool(r.llm1_inc),
                        "ran_llm2": bool(r.llm2_inc),
                        "ran_llm3": bool(r.llm3_inc),
                        "upsert_queued": bool(r.upsert_increment),
                    },
                )
                return r

        try:
            workers = settings.reranker_parallel_workers
            if workers > 1 and prompt_debug:
                logger.warning(
                    "RERANKER_PARALLEL_WORKERS=%s disabled when --log-llm*-prompt is set.",
                    workers,
                )
            if workers > 1 and not prompt_debug:
                self._run_parallel(options, job_filter, workers, rank_one, apply_result, counters)
            else:
                self._run_sequential(options, job_filter, rank_one, apply_result, counters)

            if pending:
                self._ranked_jobs.bulk_upsert(pending)
                pending.clear()

            if counters.jobs > 0 and counters.upserts == 0:
                logger.warning(
                    "No MongoDB upserts ran for %s document(s). Check 'Upsert skipped' messages above "
                    "(each document needs job id, fingerprint, or source _id for the target upsert).",
                    counters.jobs,
                )
        except Exception as exc:
            run_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Batch failed: %s", run_error)
            logger.debug("Reranker traceback:\n%s", traceback.format_exc())
        finally:
            self._flush_on_exit(pending)
            batch_s = elapsed_s(t_batch)
            logger.info(
                "Batch %s | documents=%s upserts_queued=%s job_errors=%s elapsed_s=%.3f",
                "complete" if run_error is None else "FAILED",
                counters.jobs,
                counters.upserts,
                counters.job_errors,
                batch_s,
            )
            if counters.job_errors:
                logger.warning(
                    "%s job(s) failed (see stack traces above); Mongo writes from successful jobs are kept.",
                    counters.job_errors,
                )
            self._run_logs.complete_run(
                run_id,
                counters.run_stats(),
                success=run_error is None,
                elapsed_s=batch_s,
                error_message=run_error,
            )
            run_error = self._post_steps.run(jobs_processed=counters.jobs, run_error=run_error)
            if run_error is None:
                self._save_run_state(counters.jobs)
        return run_error is None

    def _run_parallel(
        self,
        options: BatchOptions,
        job_filter: ClassifiedJobsFilter,
        workers: int,
        rank_one: Callable[[dict[str, Any], str], RankOneJobResult],
        apply_result: Callable[[RankOneJobResult], None],
        counters: _Counters,
    ) -> None:
        chunk_size = max(1, workers)
        logger.info(
            "Parallel batch | workers=%s | fetch_chunk=%s | limit=%s | "
            "documents may complete out of order in logs.",
            workers,
            chunk_size,
            str(options.limit) if options.limit > 0 else "0 (whole collection, chunked fetch)",
        )
        lock = threading.Lock()
        seq = 0

        def next_label() -> str:
            nonlocal seq
            seq += 1
            return f"{seq}/{options.limit}" if options.limit > 0 else f"#{seq}"

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for chunk in self._classified_jobs.iter_jobs_chunks(
                chunk_size, limit=options.limit, job_filter=job_filter
            ):
                futures = [executor.submit(rank_one, d, next_label()) for d in chunk]
                for fut in as_completed(futures):
                    try:
                        r = fut.result()
                        with lock:
                            apply_result(r)
                    except Exception as exc:
                        with lock:
                            counters.job_errors += 1
                        logger.exception("Parallel worker job failed (continuing): %s", exc)

    def _run_sequential(
        self,
        options: BatchOptions,
        job_filter: ClassifiedJobsFilter,
        rank_one: Callable[[dict[str, Any], str], RankOneJobResult],
        apply_result: Callable[[RankOneJobResult], None],
        counters: _Counters,
    ) -> None:
        for doc in self._classified_jobs.iter_jobs(limit=options.limit, job_filter=job_filter):
            n = counters.jobs + 1
            job_label = f"{n}/{options.limit}" if options.limit > 0 else f"#{n}"
            try:
                apply_result(rank_one(doc, job_label))
            except Exception as exc:
                counters.job_errors += 1
                logger.exception("Sequential job failed (continuing): %s", exc)

    def _flush_on_exit(self, pending: list[RankedJobUpsert]) -> None:
        if not pending:
            return
        n_pending = len(pending)
        try:
            self._ranked_jobs.bulk_upsert(pending)
            logger.info(
                "MongoDB flush on exit | wrote %s pending upsert(s) "
                "(interrupt or early exit before in-loop flush)",
                n_pending,
            )
        except Exception as exc:
            logger.warning(
                "MongoDB flush on exit failed | %s pending upsert(s) not written: %s",
                n_pending,
                exc,
            )
        pending.clear()

    def _save_run_state(self, jobs_processed: int) -> None:
        country = self._settings.target_country
        try:
            self._run_state.save(country, jobs_processed=jobs_processed)
            logger.info("[%s] Reranker state saved (jobs=%s)", country.upper(), jobs_processed)
        except Exception as e:
            logger.warning("[%s] Could not persist reranker state: %s", country.upper(), e)
