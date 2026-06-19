"""Backfill whitened ``job_embedding`` on historical ``ranked_jobs`` documents.

For each doc with a raw L2-normalized ``job_embedding`` and no whitening provenance:
  1. Copy existing ``job_embedding`` to ``raw_embedding``
  2. Overwrite ``job_embedding`` with ``whiten_concat_rows(raw_embedding)``
  3. Set ``llm_reranker_meta.embedding.whitening`` provenance

Does NOT call Gemini. Idempotent: skips docs already migrated.

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pymongo import MongoClient, UpdateOne

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings  # noqa: E402
from utils.concat_whitening import (  # noqa: E402
    CONCAT_WHITENING_ARTIFACT_SHA256,
    assert_concat_whitening_artifact_sha256,
    concat_whitening_artifact_sha256,
    concat_whitening_target,
    reset_concat_whitening_cache,
    whiten_concat_rows,
)

DEFAULT_COLLECTION = "ranked_jobs"
DEFAULT_BATCH_SIZE = 200

log = logging.getLogger("backfill_whiten_ranked_jobs")


@dataclass
class BackfillStats:
    scanned: int = 0
    skipped_already_migrated: int = 0
    skipped_missing_embedding: int = 0
    skipped_wrong_dim: int = 0
    whitened: int = 0
    written: int = 0
    dry_run_would_write: int = 0


def _already_migrated(doc: dict[str, Any]) -> bool:
    meta = doc.get("llm_reranker_meta") or {}
    embedding = meta.get("embedding") or {}
    whitening = embedding.get("whitening") or {}
    return bool(whitening.get("enabled")) and isinstance(doc.get("raw_embedding"), list)


def _candidate_filter() -> dict[str, Any]:
    return {
        "job_embedding": {"$exists": True, "$type": "array", "$ne": []},
        "$or": [
            {"llm_reranker_meta.embedding.whitening.enabled": {"$ne": True}},
            {"raw_embedding": {"$exists": False}},
        ],
    }


def _build_whitening_provenance(*, artifact_sha256: str | None, applied_at: datetime) -> dict[str, Any]:
    return {
        "enabled": True,
        "artifact_sha256": artifact_sha256,
        "target": concat_whitening_target(),
        "applied_at": applied_at.isoformat(),
    }


def _process_doc(
    doc: dict[str, Any],
    *,
    settings: Settings,
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
    whitened = whiten_concat_rows(raw.reshape(1, -1), settings=settings)[0]
    applied_at = datetime.now(timezone.utc)
    artifact_sha = concat_whitening_artifact_sha256(settings=settings)
    whitening = _build_whitening_provenance(artifact_sha256=artifact_sha, applied_at=applied_at)

    update: dict[str, Any] = {
        "raw_embedding": [float(x) for x in raw.tolist()],
        "job_embedding": [float(x) for x in whitened.tolist()],
        "llm_reranker_meta.embedding.whitening": whitening,
    }
    return update, None


def run_backfill(
    *,
    mongodb_uri: str,
    db_name: str,
    collection: str,
    settings: Settings,
    limit: int | None,
    batch_size: int,
    dry_run: bool,
    expected_artifact_sha256: str | None,
) -> BackfillStats:
    reset_concat_whitening_cache()
    if expected_artifact_sha256:
        assert_concat_whitening_artifact_sha256(expected_artifact_sha256, settings=settings)
    elif not dry_run:
        assert_concat_whitening_artifact_sha256(CONCAT_WHITENING_ARTIFACT_SHA256, settings=settings)

    stats = BackfillStats()
    client = MongoClient(mongodb_uri)
    coll = client[db_name][collection]
    expected_dim = settings.reranker_job_embedding_dim

    cursor = coll.find(_candidate_filter(), projection={"job_embedding": 1, "raw_embedding": 1, "llm_reranker_meta": 1})
    if limit is not None:
        cursor = cursor.limit(limit)

    ops: list[UpdateOne] = []
    for doc in cursor:
        stats.scanned += 1
        update, reason = _process_doc(doc, settings=settings, expected_dim=expected_dim)
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

        ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": update}))
        if len(ops) >= batch_size:
            result = coll.bulk_write(ops, ordered=False)
            stats.written += result.modified_count
            ops.clear()

    if ops and not dry_run:
        result = coll.bulk_write(ops, ordered=False)
        stats.written += result.modified_count

    client.close()
    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mongodb-uri", default=os.environ.get("MONGODB_URI", ""))
    p.add_argument("--db", required=True, help="MongoDB database name (e.g. ZambiaJobs_V2)")
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--expected-artifact-sha256",
        default=os.environ.get("RERANKER_CONCAT_WHITENING_EXPECTED_SHA256", CONCAT_WHITENING_ARTIFACT_SHA256),
    )
    p.add_argument(
        "--concat-whitening-path",
        default=os.environ.get("RERANKER_CONCAT_WHITENING_PATH", "resources/concat_whitening_gemini.npz"),
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    if not args.mongodb_uri.strip():
        log.error("--mongodb-uri or MONGODB_URI is required")
        return 2

    os.environ["RERANKER_CONCAT_WHITENING_ENABLED"] = "1"
    os.environ["RERANKER_CONCAT_WHITENING_PATH"] = args.concat_whitening_path
    os.environ["RERANKER_CONCAT_WHITENING_REQUIRED"] = "1"
    reset_concat_whitening_cache()
    settings = Settings.from_env()

    t0 = time.time()
    stats = run_backfill(
        mongodb_uri=args.mongodb_uri.strip(),
        db_name=args.db,
        collection=args.collection,
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
