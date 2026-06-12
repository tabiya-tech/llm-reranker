#!/usr/bin/env python3
"""Batch CLI: ``classified_jobs`` → LLM 1, 3, 2 → upsert ranked job schema (e.g. ``ranked_jobs``).

For single-job HTTP processing, see ``api.py`` (FastAPI).
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

from pymongo.operations import UpdateOne

from app.batch_rank_one import RankOneJobResult, rank_one_classified_document
from app.config import get_settings, log_ranker_run_configuration, reranker_enable_llm3_from_env
from app.database import (
    bulk_write_llm_upserts,
    complete_ranking_run_log,
    distinct_ranked_job_ids,
    get_database,
    insert_ranking_run_log,
    iter_classified_jobs,
    iter_classified_jobs_chunks,
    update_ranking_run_progress,
)
from app.logging_config import configure_logging
from app.pipeline import load_taxonomy_context
from utils.update_is_active_from_closing_date import sync_is_active_from_closing_date
from utils.work_activities_enrichment import run as run_work_activities_enrichment

logger = logging.getLogger(__name__)


def _elapsed_s(start: float) -> float:
    return time.perf_counter() - start


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="classified_jobs batch: Mongo → LLM 1, 3, 2 → upsert target collection"
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
        help="Per job: log LLM3 occupation-skills relevance full prompt; skip other per-job INFO logs",
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
    return p.parse_args()


def _gemini_or_exit(settings) -> None:
    if not settings.gemini_api_key:
        print(
            "No Gemini API key configured. Set GEMINI_API_KEY in .env.\n"
            "Create one at https://aistudio.google.com/apikey",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _resolve_reranker_cutoff(*, country: str, db) -> str | None:
    """Resolve YYYY-MM-DD lower bound for incremental classified_jobs fetch.

    Order: per-country env (e.g. ``KENYA_RERANKER_CUTOFF_DATE``) or ``RERANKER_CUTOFF_DATE`` (used as-is);
    else ``reranker_state.last_cutoff_date`` for this country (used as-is, no day overlap).
    """
    env_specific = f"{country.upper()}_RERANKER_CUTOFF_DATE"
    env_override = (os.getenv(env_specific) or os.getenv("RERANKER_CUTOFF_DATE") or "").strip()
    if env_override:
        logger.info("[%s] Cutoff: env override (%s=%s)", country.upper(), env_specific, env_override)
        return env_override

    state_coll_name = (os.getenv("RERANKER_STATE_COLLECTION") or "reranker_state").strip()
    state_doc = db[state_coll_name].find_one({"country": country}) or {}
    last_cutoff = state_doc.get("last_cutoff_date")
    if last_cutoff:
        if isinstance(last_cutoff, datetime):
            dt = (
                last_cutoff
                if last_cutoff.tzinfo
                else last_cutoff.replace(tzinfo=timezone.utc)
            )
            cutoff_str = dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
        else:
            cutoff_str = datetime.fromisoformat(
                str(last_cutoff).strip().replace("Z", "+00:00")
            ).strftime("%Y-%m-%d")
        logger.info(
            "[%s] Cutoff: incremental from state last_cutoff_date=%s (exact)",
            country.upper(),
            cutoff_str,
        )
        return cutoff_str

    logger.info("[%s] Cutoff: no previous state (full scan)", country.upper())
    return None


def _set_reranker_state(*, country: str, db, jobs_processed: int) -> None:
    state_coll_name = (os.getenv("RERANKER_STATE_COLLECTION") or "reranker_state").strip()
    db[state_coll_name].update_one(
        {"country": country},
        {"$set": {
            "country": country,
            "last_cutoff_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "last_run_at": datetime.now(timezone.utc),
            "jobs_processed": int(jobs_processed),
        }},
        upsert=True,
    )
    logger.info("[%s] Reranker state saved (jobs=%s)", country.upper(), jobs_processed)


def run() -> None:
    configure_logging()
    args = _parse_args()
    settings = get_settings()
    batch_size = (
        args.bulk_batch_size if args.bulk_batch_size is not None else settings.bulk_write_batch_size
    )
    batch_size = max(1, min(batch_size, 10_000))
    pending: list[UpdateOne] = []

    t_startup = time.perf_counter()
    taxonomy = load_taxonomy_context(settings)
    startup_s = _elapsed_s(t_startup)

    limit_log = "all" if args.limit <= 0 else str(args.limit)
    logger.info(
        "Batch start | db=%s | limit=%s | target_collection=%s bulk_batch_size=%s | "
        "startup_load_s=%.3f | taxonomy_relation_occupations=%s | model=%s",
        settings.mongodb_database,
        limit_log,
        settings.mongodb_target_collection,
        batch_size,
        startup_s,
        len(taxonomy.occ_map),
        settings.gemini_model,
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

    _gemini_or_exit(settings)
    enable_llm3 = reranker_enable_llm3_from_env()
    log_ranker_run_configuration(
        logger,
        settings,
        bulk_write_batch_size=batch_size,
        cli_limit=args.limit,
        log_prompt_debug=(
            args.log_llm1_prompt or args.log_llm2_prompt or args.log_llm3_prompt
        ),
        invocation="batch",
    )

    country = settings.target_country
    db = get_database(settings=settings)
    cutoff_date = _resolve_reranker_cutoff(country=country, db=db)
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

    skip_ranked = args.skip_already_ranked or _env_bool("RERANKER_SKIP_ALREADY_RANKED", False)
    if skip_ranked:
        ranked_ids = distinct_ranked_job_ids(db=db, settings=settings)
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

    run_id = insert_ranking_run_log(
        settings=settings,
        limit=args.limit,
        model=settings.gemini_model,
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
    enrichment_enabled = _env_bool("RERANKER_RUN_WA_ENRICHMENT", True)
    enrichment_fail_hard = _env_bool("RERANKER_FAIL_ON_WA_ENRICHMENT_ERROR", False)
    is_active_sync_enabled = _env_bool("RERANKER_SYNC_IS_ACTIVE_FROM_CLOSING_DATE", True)

    try:
        log_prompt_debug = (
            args.log_llm1_prompt or args.log_llm2_prompt or args.log_llm3_prompt
        )
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
                    bulk_write_llm_upserts(pending, settings=settings)
                    pending.clear()
            n += 1
            if n % 10 == 0:
                update_ranking_run_progress(
                    run_id,
                    jobs_processed=n,
                    upserts_queued=upsert_ops,
                    llm1_calls=llm1_calls,
                    llm2_calls=llm2_calls,
                    llm3_calls=llm3_calls,
                    settings=settings,
                )

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
            return rank_one_classified_document(
                doc,
                job_label=job_label,
                args=args,
                settings=settings,
                taxonomy=taxonomy,
                enable_llm3=enable_llm3,
            )

        if use_parallel:
            chunk_sz = max(1, workers)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for chunk in iter_classified_jobs_chunks(
                    chunk_sz,
                    limit=args.limit,
                    query=query,
                    settings=settings,
                ):
                    futures = [
                        executor.submit(_run_one, d, _next_job_label()) for d in chunk
                    ]
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
            for doc in iter_classified_jobs(limit=args.limit, query=query, settings=settings):
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
            bulk_write_llm_upserts(pending, settings=settings)
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
                bulk_write_llm_upserts(pending, settings=settings)
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
            n, upsert_ops, job_errors, batch_s,
        )
        if job_errors:
            logger.warning(
                "%s job(s) failed (see stack traces above); Mongo writes from successful jobs are kept.",
                job_errors,
            )
        complete_ranking_run_log(
            run_id,
            success=_run_error is None,
            jobs_processed=n, upserts_queued=upsert_ops,
            llm1_calls=llm1_calls, llm2_calls=llm2_calls, llm3_calls=llm3_calls,
            elapsed_s=batch_s, error_message=_run_error,
            settings=settings,
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
                code = run_work_activities_enrichment()
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
                    logger.warning("Post-step | work_activities_enrichment | skipped-on-error | %s", e)
        if is_active_sync_enabled and _run_error is None:
            try:
                t0 = time.perf_counter()
                coll = db[settings.mongodb_target_collection]
                logger.info(
                    "Post-step | is_active from closing_date | start | target=%s",
                    settings.mongodb_target_collection,
                )
                stats = sync_is_active_from_closing_date(coll, missing_policy="keep", dry_run=False)
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
                _set_reranker_state(country=country, db=db, jobs_processed=n)
            except Exception as e:
                logger.warning("[%s] Could not persist reranker state: %s", country.upper(), e)
        if _run_error is not None:
            raise SystemExit(1)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
