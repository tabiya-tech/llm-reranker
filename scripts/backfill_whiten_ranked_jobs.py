"""Backfill whitened ``job_embedding`` on historical ``ranked_jobs`` documents.

For each doc with a raw L2-normalized ``job_embedding`` and no whitening provenance:
  1. Copy existing ``job_embedding`` to ``raw_embedding``
  2. Overwrite ``job_embedding`` with ``ConcatWhitener.whiten(raw_embedding)``
  3. Set ``llm_reranker_meta.embedding.whitening`` provenance

Does NOT call Gemini. Idempotent: skips docs already migrated.

``--mongodb-uri`` / ``--db`` override ``MONGODB_URI`` / ``MONGODB_DATABASE``; the connection is
opened through ``standalone_db()`` (see ``app/server_dependencies/README.md``).

Usage (from llm-reranker repo root):

  # Dry run — count candidates only:
  python scripts/backfill_whiten_ranked_jobs.py \\
      --mongodb-uri "$MONGODB_URI" \\
      --db ZambiaJobs_V2 \\
      --dry-run

  # Bounded run with SHA-256 assertion:
  python scripts/backfill_whiten_ranked_jobs.py \\
      --mongodb-uri "$MONGODB_URI" \\
      --db ZambiaJobs_V2 \\
      --expected-artifact-sha256 5d96cbdd69bc3e920976ed0273036e859e13cb7140887f8367156249a8683f18 \\
      --limit 100
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings  # noqa: E402
from app.job_embedding.whitening import CONCAT_WHITENING_ARTIFACT_SHA256, ConcatWhitener
from app.job_embedding.whitening_repository import WhiteningArtifactRepository
from app.ranked_jobs.mongo_repository import MongoRankedJobsRepository
from app.ranked_jobs.repository import IRankedJobsRepository
from app.server_dependencies.db_dependencies import standalone_db  # noqa: E402

DEFAULT_COLLECTION = "ranked_jobs"
DEFAULT_BATCH_SIZE = 200

log = logging.getLogger("backfill_whiten_ranked_jobs")


class BackfillStats(BaseModel):
    """Per-run counters for one ``run_backfill`` invocation."""

    model_config = ConfigDict(strict=True, extra="forbid")

    scanned: int = Field(default=0, description="Candidate documents read from the cursor.")
    skipped_already_migrated: int = Field(
        default=0, description="Documents that already have whitening provenance."
    )
    skipped_missing_embedding: int = Field(
        default=0, description="Documents with no usable raw `job_embedding`."
    )
    skipped_wrong_dim: int = Field(
        default=0, description="Documents whose `job_embedding` length doesn't match expected_dim."
    )
    whitened: int = Field(default=0, description="Documents successfully whitened in-memory.")
    written: int = Field(default=0, description="Documents actually written back to MongoDB.")
    dry_run_would_write: int = Field(
        default=0, description="Documents that would have been written, in `--dry-run` mode."
    )


def _already_migrated(doc: dict[str, Any]) -> bool:
    meta = doc.get("llm_reranker_meta") or {}
    embedding = meta.get("embedding") or {}
    whitening = embedding.get("whitening") or {}
    return bool(whitening.get("enabled")) and isinstance(doc.get("raw_embedding"), list)


def _build_whitening_provenance(
    *, whitener: ConcatWhitener, applied_at: datetime
) -> dict[str, Any]:
    return {
        "enabled": True,
        "artifact_sha256": whitener.artifact_sha256,
        "target": whitener.target,
        "applied_at": applied_at.isoformat(),
    }


def _process_doc(
    doc: dict[str, Any],
    *,
    whitener: ConcatWhitener,
    expected_dim: int,
) -> tuple[dict[str, Any] | None, str | None]:
    if _already_migrated(doc):
        return None, "already_migrated"

    je = doc.get("job_embedding")
    if not isinstance(je, list) or not je:
        return None, "missing_embedding"
    if len(je) != expected_dim:
        return None, "wrong_dim"

    raw = np.asarray(je, dtype=np.float64)
    whitened = whitener.whiten(raw.reshape(1, -1))[0]
    whitening = _build_whitening_provenance(
        whitener=whitener, applied_at=datetime.now(timezone.utc)
    )

    update: dict[str, Any] = {
        "raw_embedding": [float(x) for x in raw.tolist()],
        "job_embedding": [float(x) for x in whitened.tolist()],
        "llm_reranker_meta.embedding.whitening": whitening,
    }
    return update, None


def run_backfill(
    *,
    ranked_jobs: IRankedJobsRepository,
    settings: Settings,
    limit: int | None,
    batch_size: int,
    dry_run: bool,
    expected_artifact_sha256: str | None,
) -> BackfillStats:
    """Scan the target collection, whiten each candidate's ``job_embedding``, and write it back."""
    whitener = ConcatWhitener(WhiteningArtifactRepository().load(settings))
    if expected_artifact_sha256:
        whitener.assert_sha256(expected_artifact_sha256)
    elif not dry_run:
        whitener.assert_sha256(CONCAT_WHITENING_ARTIFACT_SHA256)

    stats = BackfillStats()
    expected_dim = settings.reranker_job_embedding_dim

    ops: list[tuple[Any, dict[str, Any]]] = []
    for doc in ranked_jobs.iter_whitening_backfill_candidates(limit=limit):
        stats.scanned += 1
        update, reason = _process_doc(doc, whitener=whitener, expected_dim=expected_dim)
        if reason == "already_migrated":
            stats.skipped_already_migrated += 1
            continue
        if reason == "missing_embedding":
            stats.skipped_missing_embedding += 1
            continue
        if reason == "wrong_dim":
            stats.skipped_wrong_dim += 1
            continue
        assert update is not None
        stats.whitened += 1

        if dry_run:
            stats.dry_run_would_write += 1
            continue

        ops.append((doc["_id"], update))
        if len(ops) >= batch_size:
            stats.written += ranked_jobs.bulk_set(ops)
            ops.clear()

    if ops and not dry_run:
        stats.written += ranked_jobs.bulk_set(ops)

    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mongodb-uri", default=os.environ.get("MONGODB_URI", ""))
    p.add_argument("--db", required=True, help="MongoDB database name (e.g. ZambiaJobs_V2)")
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--expected-artifact-sha256",
        default=os.environ.get(
            "RERANKER_CONCAT_WHITENING_EXPECTED_SHA256", CONCAT_WHITENING_ARTIFACT_SHA256
        ),
    )
    p.add_argument(
        "--concat-whitening-path",
        default=os.environ.get(
            "RERANKER_CONCAT_WHITENING_PATH", "resources/concat_whitening_gemini.npz"
        ),
    )
    return p.parse_args()


def main() -> int:
    """CLI entry point: parse args, run the backfill, log a summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    if not args.mongodb_uri.strip():
        log.error("--mongodb-uri or MONGODB_URI is required")
        return 2

    # After app.config has loaded .env, so the CLI values win over it.
    os.environ["MONGODB_URI"] = args.mongodb_uri.strip()
    os.environ["MONGODB_DATABASE"] = args.db
    os.environ["RERANKER_CONCAT_WHITENING_ENABLED"] = "1"
    os.environ["RERANKER_CONCAT_WHITENING_PATH"] = args.concat_whitening_path
    os.environ["RERANKER_CONCAT_WHITENING_REQUIRED"] = "1"
    settings = Settings.from_env()

    t0 = time.time()
    with standalone_db(settings) as db:
        stats = run_backfill(
            ranked_jobs=MongoRankedJobsRepository(db, settings, collection_name=args.collection),
            settings=settings,
            limit=args.limit,
            batch_size=max(1, args.batch_size),
            dry_run=args.dry_run,
            expected_artifact_sha256=(args.expected_artifact_sha256 or "").strip() or None,
        )
    elapsed = time.time() - t0

    log.info(
        "backfill complete db=%s collection=%s dry_run=%s scanned=%d already_migrated=%d "
        "missing_embedding=%d wrong_dim=%d whitened=%d written=%d dry_run_would_write=%d elapsed=%.1fs",
        args.db,
        args.collection,
        args.dry_run,
        stats.scanned,
        stats.skipped_already_migrated,
        stats.skipped_missing_embedding,
        stats.skipped_wrong_dim,
        stats.whitened,
        stats.written,
        stats.dry_run_would_write,
        elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
