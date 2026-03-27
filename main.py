#!/usr/bin/env python3
"""Batch: ClassifiedJobs → Vertex LLM 1 & 2 → upsert ranked job schema to ``MONGODB_TARGET_COLLECTION``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config import get_settings, vertex_project_unset_message
from pymongo.operations import UpdateOne

from app.database import (
    bulk_write_llm_upserts,
    build_job_description_context,
    fetch_classified_jobs,
    llm_pipeline_upsert_update_one,
)
from app.logging_config import configure_logging
from app.llm_service import run_job_attributes_for_job, run_skill_ranker_for_job
from app.prompt_debug import dump_prompt_debug_first_row
from app.schemas import JobAttributesResult
from utils.classifier_linked_entities import (
    MappedEscoSkills,
    extract_esco_skill_candidates,
    job_identifier,
    skills_cell,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ClassifiedJobs batch: Mongo → Vertex LLM 1 & 2 → upsert target collection"
    )
    p.add_argument("--limit", type=int, default=5, help="Documents to fetch from ClassifiedJobs")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Vertex calls and do not write to the target collection",
    )
    p.add_argument(
        "--dump-prompts",
        type=Path,
        default=None,
        metavar="DIR",
        help="First document only: write skills + job_attributes prompt files under DIR",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Per job: extra counts on stderr",
    )
    p.add_argument(
        "--bulk-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Flush MongoDB bulk writes every N upserts (overrides BULK_WRITE_BATCH_SIZE; default from env or 100)",
    )
    return p.parse_args()


def _vertex_or_exit(settings) -> None:
    if not settings.gcp_project_id.strip():
        print(vertex_project_unset_message(), file=sys.stderr)
        raise SystemExit(1)


def run() -> None:
    configure_logging()
    args = _parse_args()
    settings = get_settings()
    first = True
    batch_size = (
        args.bulk_batch_size if args.bulk_batch_size is not None else settings.bulk_write_batch_size
    )
    batch_size = max(1, min(batch_size, 10_000))
    pending: list[UpdateOne] = []

    logger.info(
        "Batch start: limit=%s dry_run=%s target=%s bulk_batch_size=%s job_attributes_schema=%s "
        "classified_linked_entity_types=%s",
        args.limit,
        args.dry_run,
        settings.mongodb_target_collection,
        batch_size,
        settings.job_attributes_schema_path,
        list(settings.classified_linked_entity_types),
    )

    n = 0
    upsert_ops = 0
    for doc in fetch_classified_jobs(limit=args.limit):
        n += 1
        ctx = build_job_description_context(doc)
        title = ctx.title or ""
        ident = job_identifier(ctx, doc)

        if args.dump_prompts is not None and first:
            try:
                dump_prompt_debug_first_row(ctx, doc, args.dump_prompts, settings=settings)
            except FileNotFoundError as e:
                print(f"dump-prompts failed (missing resource file?): {e}", file=sys.stderr)
                raise SystemExit(1) from e

        ran_skill = False
        ran_ja = False
        essential: list[str] = []
        optional: list[str] = []
        skill_mapped: MappedEscoSkills | None = None
        skill_raw = ""
        ja: JobAttributesResult | None = None

        if not args.dry_run:
            _vertex_or_exit(settings)
            if extract_esco_skill_candidates(doc):
                essential, optional, skill_mapped, skill_raw = run_skill_ranker_for_job(
                    ctx, settings=settings
                )
                ran_skill = True
            elif args.verbose:
                logger.info(
                    "Skill ranker skipped (no ESCO candidates on document): id=%s",
                    ident,
                )

            ja = run_job_attributes_for_job(ctx, settings=settings)
            ran_ja = True

        logger.info(
            "Processing job: %s (id=%s, esco_candidates=%s)",
            title or "(no title)",
            ident,
            skills_cell(doc) or "—",
        )

        if args.verbose:
            parts: list[str] = []
            if ran_skill:
                parts.append(f"LLM1: ess={len(essential)} opt={len(optional)}")
            if ran_ja and ja is not None:
                parts.append(
                    f"LLM2: attrs={len(ja.attributes)} w_acts={len(ja.work_activity_ids)}"
                )
            if parts:
                print("  " + " | ".join(parts), file=sys.stderr)

        if not args.dry_run and (ran_skill or ran_ja):
            try:
                op = llm_pipeline_upsert_update_one(
                    doc,
                    ctx,
                    settings=settings,
                    ja=ja if ran_ja and ja is not None else None,
                    skill_ranker=(essential, optional, skill_mapped, skill_raw)
                    if ran_skill and skill_mapped is not None
                    else None,
                )
                if op is not None:
                    upsert_ops += 1
                    pending.append(op)
                    if len(pending) >= batch_size:
                        bulk_write_llm_upserts(pending, settings=settings)
                        pending.clear()
            except ValueError as e:
                logger.warning("Upsert skipped: %s", e)

        first = False

    if pending:
        bulk_write_llm_upserts(pending, settings=settings)

    if not args.dry_run and n > 0 and upsert_ops == 0:
        logger.warning(
            "No MongoDB upserts ran for %s document(s). Check 'Upsert skipped' messages above "
            "(each document needs job id, fingerprint, or source _id for the target upsert).",
            n,
        )

    logger.info("Batch complete: processed %s document(s)", n)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
