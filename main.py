#!/usr/bin/env python3
"""Batch CLI: ``classified_jobs`` → per-job pipeline → upsert ranked job schema (e.g. ``ranked_jobs``).

Each job runs :func:`app.pipeline.process_single_job`, the same pipeline as ``POST /rank`` in
``api.py``; the batch adds per-step logging, the job embedding, and the MongoDB upsert.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError
from pymongo.operations import UpdateOne

from app.batch_rank_one import RankOneJobResult, rank_one_classified_document
from app.classified_jobs.mongo_repository import MongoClassifiedJobsRepository
from app.config import Settings, get_settings, log_ranker_run_configuration
from app.languages import LANGUAGE_REGISTRY
from app.llm import (
    LlmClient,
    LlmConfig,
    LlmConfigurationError,
    describe_available_providers,
    get_llm_client,
)
from app.logging_config import configure_logging
from app.observability import (
    init_tracing,
    job_trace,
    shutdown_tracing,
    tracing_config_from_env,
    update_observation,
)
from app.pipeline import load_taxonomy_context
from app.prompt_resources import prompt_provenance
from app.ranked_jobs.mongo_repository import MongoRankedJobsRepository
from app.run_logs.mongo_repository import MongoRunLogsRepository
from app.run_logs.repository import RunStats
from app.run_state.mongo_repository import MongoRunStateRepository
from app.run_state.repository import IRunStateRepository
from app.server_dependencies.db_dependencies import RerankerDBProvider
from utils.work_activities_enrichment import run as run_work_activities_enrichment

logger = logging.getLogger(__name__)


def _elapsed_s(start: float) -> float:
    return time.perf_counter() - start


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="classified_jobs batch: Mongo → per-job pipeline (as POST /rank) → upsert target collection"
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Max documents from the source collection. Default: 0 (entire collection, newest _id first). "
        "With RERANKER_PARALLEL_WORKERS>1, the full collection is streamed in chunks (not loaded entirely into memory).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Per job: extra counts on stderr",
    )
    p.add_argument(
        "--log-llm1-prompt",
        action="store_true",
        help="Per job: log LLM1 (skill ranker) full prompt; skip other per-job INFO logs",
    )
    p.add_argument(
        "--log-llm2-prompt",
        action="store_true",
        help="Per job: log LLM2 job-attributes full prompt; skip other per-job INFO logs",
    )
    p.add_argument(
        "--log-llm3-prompt",
        action="store_true",
        help="Per job: log LLM3 occupation-skills relevance full prompt (only RERANKER_LLM3_MODE=call "
        "makes that call); skip other per-job INFO logs",
    )
    p.add_argument(
        "--bulk-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Flush MongoDB bulk writes every N upserts (overrides BULK_WRITE_BATCH_SIZE; default from env or 100)",
    )
    p.add_argument(
        "--skip-already-ranked",
        action="store_true",
        help=(
            "Only process classified_jobs whose job_id is not yet in MONGODB_TARGET_COLLECTION "
            "(same as RERANKER_SKIP_ALREADY_RANKED=1)."
        ),
    )
    p.add_argument(
        "--language",
        default=None,
        metavar="LANG",
        help=(
            f"Language of the job ads: {', '.join(LANGUAGE_REGISTRY)} (a locale like AR-es also "
            "works). Selects resources/<language>/ — taxonomy labels, prompts, job attributes. "
            "Same as RERANKER_LANGUAGE. Default: derived from TARGET_COUNTRY."
        ),
    )
    return p.parse_args()


def _settings_and_llm_or_exit() -> tuple[Settings, LlmClient]:
    settings: Settings | None = None
    llm_config: LlmConfig | None = None
    try:
        settings = get_settings()
        llm_config = settings.llm
        return settings, get_llm_client(llm_config)
    except LlmConfigurationError as e:
        selection = (
            f"Configured selection: provider={llm_config.provider} "
            f"model={llm_config.model or '(unset)'} class={llm_config.client_class}\n\n"
            if llm_config is not None
            else ""
        )
        print(
            f"LLM configuration error: {e}\n\n"
            f"{selection}"
            f"Available providers (RERANKER_LLM_PROVIDER):\n"
            f"{describe_available_providers()}",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    except ValidationError as e:
        print(f"Settings configuration error:\n\n{e}", file=sys.stderr)
        raise SystemExit(1) from e


def _resolve_reranker_cutoff(*, country: str, run_state: IRunStateRepository) -> str | None:
    """Resolve YYYY-MM-DD lower bound for incremental classified_jobs fetch.

    Order: per-country env (e.g. ``KENYA_RERANKER_CUTOFF_DATE``) or ``RERANKER_CUTOFF_DATE`` (used as-is);
    else ``reranker_state.last_cutoff_date`` for this country (used as-is, no day overlap).
    """
    env_specific = f"{country.upper()}_RERANKER_CUTOFF_DATE"
    env_override = (os.getenv(env_specific) or os.getenv("RERANKER_CUTOFF_DATE") or "").strip()
    if env_override:
        logger.info(
            "[%s] Cutoff: env override (%s=%s)", country.upper(), env_specific, env_override
        )
        return env_override

    cutoff_str = run_state.get_last_cutoff_date(country)
    if cutoff_str:
        logger.info(
            "[%s] Cutoff: incremental from state last_cutoff_date=%s (exact)",
            country.upper(),
            cutoff_str,
        )
        return cutoff_str

    logger.info("[%s] Cutoff: no previous state (full scan)", country.upper())
    return None


def run() -> None:
    configure_logging()
    args = _parse_args()
    if args.language:
        # Exported before the first get_settings() so every resource path — taxonomy
        # labels, prompts, job attributes — resolves inside this language's pack.
        os.environ["RERANKER_LANGUAGE"] = str(args.language).strip()
    settings, llm = _settings_and_llm_or_exit()
    # After settings, so the repo .env has been loaded into the environment.
    init_tracing(tracing_config_from_env())
    batch_size = (
        args.bulk_batch_size if args.bulk_batch_size is not None else settings.bulk_write_batch_size
    )
    batch_size = max(1, min(batch_size, 10_000))
    pending: list[UpdateOne] = []

    t_startup = time.perf_counter()
    # Resolves every prompt template now: a pack missing one fails the run here, not
    # each job in turn.
    prompts = prompt_provenance(settings)
    taxonomy = load_taxonomy_context(settings)
    startup_s = _elapsed_s(t_startup)

    limit_log = "all" if args.limit <= 0 else str(args.limit)
    logger.info(
        "Batch start | db=%s | limit=%s | target_collection=%s bulk_batch_size=%s | "
        "startup_load_s=%.3f | taxonomy_relation_occupations=%s | llm=%s | prompts=%s (%s)",
        settings.mongodb_database,
        limit_log,
        settings.mongodb_target_collection,
        batch_size,
        startup_s,
        len(taxonomy.occ_map),
        settings.llm.describe(),
        prompts["prompt_version"],
        prompts["prompt_digest"],
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
        cli_limit=args.limit,
        log_prompt_debug=(args.log_llm1_prompt or args.log_llm2_prompt or args.log_llm3_prompt),
        invocation="batch",
    )

    country = settings.target_country
    # The run's one MongoDB client: closed in ``main()`` when the run ends.
    db = RerankerDBProvider.get_db(settings)
    classified_jobs = MongoClassifiedJobsRepository(db, settings)
    ranked_jobs = MongoRankedJobsRepository(db, settings)
    run_logs = MongoRunLogsRepository(db, settings)
    run_state = MongoRunStateRepository(db, settings)
    cutoff_date = _resolve_reranker_cutoff(country=country, run_state=run_state)
    query: dict[str, Any] = {}
    if cutoff_date:
        try:
            cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            # Anchor on immutable `classified_at` only (set once, on first classifier insert).
            # Avoids matching docs whose `updated_at` gets bumped by re-classification.
            query = {"classified_at": {"$gte": cutoff_dt}}
        except ValueError:
            logger.warning(
                "[%s] Invalid cutoff_date=%s, falling back to full scan",
                country.upper(),
                cutoff_date,
            )

    skip_ranked = args.skip_already_ranked or settings.reranker_skip_already_ranked
    if skip_ranked:
        ranked_ids = ranked_jobs.distinct_job_ids()
        jid = settings.mongo_field_job_id
        if ranked_ids:
            nin_clause = {jid: {"$nin": ranked_ids}}
            query = {**nin_clause} if not query else {"$and": [dict(query), nin_clause]}
        logger.info(
            "Resume | skip_already_ranked | excluding %s %s already in %s.%s",
            len(ranked_ids),
            jid,
            settings.mongodb_database,
            settings.mongodb_target_collection,
        )

    run_id = run_logs.insert_run(
        limit=args.limit,
        model=llm.model,
        source_collection=settings.mongodb_source_collection,
        target_collection=settings.mongodb_target_collection,
    )

    t_batch = time.perf_counter()
    n = 0
    upsert_ops = 0
    llm1_calls = 0
    llm2_calls = 0
    llm3_calls = 0
    job_errors = 0
    _run_error: str | None = None

    def _run_stats() -> RunStats:
        return RunStats(
            jobs_processed=n,
            upserts_queued=upsert_ops,
            llm1_calls=llm1_calls,
            llm2_calls=llm2_calls,
            llm3_calls=llm3_calls,
        )

    enrichment_enabled = settings.reranker_run_wa_enrichment
    enrichment_fail_hard = settings.reranker_fail_on_wa_enrichment_error
    is_active_sync_enabled = settings.reranker_sync_is_active_from_closing_date

    try:
        log_prompt_debug = args.log_llm1_prompt or args.log_llm2_prompt or args.log_llm3_prompt
        workers = settings.reranker_parallel_workers
        use_parallel = workers > 1 and not log_prompt_debug
        if workers > 1 and log_prompt_debug:
            logger.warning(
                "RERANKER_PARALLEL_WORKERS=%s disabled when --log-llm*-prompt is set.",
                workers,
            )
        if use_parallel:
            chunk_sz = max(1, workers)
            limit_msg = str(args.limit) if args.limit > 0 else "0 (whole collection, chunked fetch)"
            logger.info(
                "Parallel batch | workers=%s | fetch_chunk=%s | limit=%s | "
                "documents may complete out of order in logs.",
                workers,
                chunk_sz,
                limit_msg,
            )

        counter_lock = threading.Lock()

        def _apply_rank_result(r: RankOneJobResult) -> None:
            nonlocal n, upsert_ops, llm1_calls, llm2_calls, llm3_calls, pending
            llm1_calls += r.llm1_inc
            llm2_calls += r.llm2_inc
            llm3_calls += r.llm3_inc
            upsert_ops += r.upsert_increment
            if r.update_op is not None:
                pending.append(r.update_op)
                if len(pending) >= batch_size:
                    ranked_jobs.bulk_upsert(pending)
                    pending.clear()
            n += 1
            if n % 10 == 0:
                run_logs.update_progress(run_id, _run_stats())

        job_seq = 0
        job_seq_lock = threading.Lock()

        def _next_job_label() -> str:
            nonlocal job_seq
            with job_seq_lock:
                job_seq += 1
                j = job_seq
            if args.limit > 0:
                return f"{j}/{args.limit}"
            return f"#{j}"

        def _run_one(doc: dict[str, Any], job_label: str) -> RankOneJobResult:
            # One Langfuse trace per job, grouped into a session per batch run.
            with job_trace(
                doc,
                settings=settings,
                invocation="batch",
                session_id=run_id,
                job_label=job_label,
            ) as trace:
                r = rank_one_classified_document(
                    doc,
                    job_label=job_label,
                    args=args,
                    settings=settings,
                    taxonomy=taxonomy,
                )
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

        if use_parallel:
            chunk_sz = max(1, workers)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for chunk in classified_jobs.iter_jobs_chunks(
                    chunk_sz,
                    limit=args.limit,
                    query=query,
                ):
                    futures = [executor.submit(_run_one, d, _next_job_label()) for d in chunk]
                    for fut in as_completed(futures):
                        try:
                            r = fut.result()
                            with counter_lock:
                                _apply_rank_result(r)
                        except Exception as exc:
                            with counter_lock:
                                job_errors += 1
                            logger.exception(
                                "Parallel worker job failed (continuing): %s",
                                exc,
                            )
        else:
            for doc in classified_jobs.iter_jobs(limit=args.limit, query=query):
                if args.limit > 0:
                    job_label = f"{n + 1}/{args.limit}"
                else:
                    job_label = f"#{n + 1}"
                try:
                    r = _run_one(doc, job_label)
                    _apply_rank_result(r)
                except Exception as exc:
                    job_errors += 1
                    logger.exception("Sequential job failed (continuing): %s", exc)

        if pending:
            ranked_jobs.bulk_upsert(pending)
            pending.clear()

        if n > 0 and upsert_ops == 0:
            logger.warning(
                "No MongoDB upserts ran for %s document(s). Check 'Upsert skipped' messages above "
                "(each document needs job id, fingerprint, or source _id for the target upsert).",
                n,
            )
    except Exception as exc:
        _run_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Batch failed: %s", _run_error)
        logger.debug("Reranker traceback:\n%s", traceback.format_exc())
    finally:
        if pending:
            n_pending = len(pending)
            try:
                ranked_jobs.bulk_upsert(pending)
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
        batch_s = _elapsed_s(t_batch)
        logger.info(
            "Batch %s | documents=%s upserts_queued=%s job_errors=%s elapsed_s=%.3f",
            "complete" if _run_error is None else "FAILED",
            n,
            upsert_ops,
            job_errors,
            batch_s,
        )
        if job_errors:
            logger.warning(
                "%s job(s) failed (see stack traces above); Mongo writes from successful jobs are kept.",
                job_errors,
            )
        run_logs.complete_run(
            run_id,
            _run_stats(),
            success=_run_error is None,
            elapsed_s=batch_s,
            error_message=_run_error,
        )
        if enrichment_enabled and n > 0:
            try:
                t0 = time.perf_counter()
                if _run_error is not None:
                    logger.warning(
                        "Post-step | work_activities_enrichment | batch had errors — "
                        "running anyway on target=%s for %s persisted job(s)",
                        settings.mongodb_target_collection,
                        n,
                    )
                logger.info(
                    "Post-step | work_activities_enrichment | start | target=%s",
                    settings.mongodb_target_collection,
                )
                code = run_work_activities_enrichment(db.client)
                elapsed = _elapsed_s(t0)
                if code != 0:
                    msg = (
                        "Post-step | work_activities_enrichment | failed | "
                        f"exit_code={code} elapsed_s={elapsed:.3f}"
                    )
                    if enrichment_fail_hard:
                        raise RuntimeError(msg)
                    logger.warning(msg)
                else:
                    logger.info(
                        "Post-step | work_activities_enrichment | done | elapsed_s=%.3f",
                        elapsed,
                    )
            except Exception as e:
                if enrichment_fail_hard:
                    _run_error = f"RuntimeError: post enrichment failed: {e}"
                    logger.exception(_run_error)
                else:
                    logger.warning(
                        "Post-step | work_activities_enrichment | skipped-on-error | %s", e
                    )
        if is_active_sync_enabled and _run_error is None:
            try:
                t0 = time.perf_counter()
                logger.info(
                    "Post-step | is_active from closing_date | start | target=%s",
                    settings.mongodb_target_collection,
                )
                stats = ranked_jobs.sync_is_active_from_closing_date(
                    missing_policy="keep", dry_run=False
                )
                elapsed = _elapsed_s(t0)
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
                    elapsed,
                )
            except Exception as e:
                logger.warning("Post-step | is_active from closing_date | failed | %s", e)
        if _run_error is None:
            try:
                run_state.save(country, jobs_processed=n)
                logger.info("[%s] Reranker state saved (jobs=%s)", country.upper(), n)
            except Exception as e:
                logger.warning("[%s] Could not persist reranker state: %s", country.upper(), e)
        if _run_error is not None:
            raise SystemExit(1)


def main() -> None:
    try:
        run()
    finally:
        RerankerDBProvider.close()
        # The container goes away with the process: flush the last jobs' traces first.
        shutdown_tracing()


if __name__ == "__main__":
    main()
