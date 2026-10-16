#!/usr/bin/env python3
"""Batch entrypoint (Cloud Run Job): ``classified_jobs`` → per-job pipeline → upsert ranked jobs.

Parses the arguments and runs :class:`app.batch.service.BatchRankingService`, which
ranks each job with the same pipeline as ``POST /rank`` and adds the job embedding, the MongoDB
upsert and the post-steps.
"""

from __future__ import annotations

import argparse
import os
import sys

from pydantic import ValidationError

from app.batch.get_batch_service import build_batch_ranking_service
from app.batch.service import BatchOptions
from app.config import Settings, get_settings
from app.job_ranking.types import PromptLogging
from app.languages import LANGUAGE_REGISTRY
from app.llm import (
    LlmConfigurationError,
    describe_llm_configuration_error,
)
from app.logging_config import configure_logging
from app.observability import init_tracing, shutdown_tracing, tracing_config_from_env
from app.server_dependencies.db_dependencies import RerankerDBProvider


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


def _batch_options(args: argparse.Namespace) -> BatchOptions:
    return BatchOptions(
        limit=args.limit,
        verbose=args.verbose,
        prompt_logging=PromptLogging(
            llm1=args.log_llm1_prompt, llm2=args.log_llm2_prompt, llm3=args.log_llm3_prompt
        ),
        bulk_batch_size=args.bulk_batch_size,
        skip_already_ranked=args.skip_already_ranked,
    )


def run() -> None:
    configure_logging()
    args = _parse_args()
    if args.language:
        # Exported before the first get_settings() so every resource path — taxonomy
        # labels, prompts, job attributes — resolves inside this language's pack.
        os.environ["RERANKER_LANGUAGE"] = str(args.language).strip()
    settings: Settings | None = None
    try:
        settings = get_settings()
        # The run's one MongoDB client (no I/O until first use): closed in ``main()``.
        service = build_batch_ranking_service(settings, RerankerDBProvider.get_db(settings))
    except LlmConfigurationError as e:
        print(
            describe_llm_configuration_error(settings.llm if settings else None, e),
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    except ValidationError as e:
        print(f"Settings configuration error:\n\n{e}", file=sys.stderr)
        raise SystemExit(1) from e
    # After settings, so the repo .env has been loaded into the environment.
    init_tracing(tracing_config_from_env())
    if not service.run(_batch_options(args)):
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
