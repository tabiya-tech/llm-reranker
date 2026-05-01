
from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection


def _parse_date(raw: object) -> Optional[date]:
    """Best-effort parser for common closing_date formats."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Try ISO first (supports "YYYY-MM-DD" and many datetime variants).
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).date()
    except ValueError:
        pass

    # Common fallback formats seen in mixed datasets.
    patterns = (
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d %b %Y",
        "%d %B %Y",
    )
    for p in patterns:
        try:
            return datetime.strptime(s, p).date()
        except ValueError:
            continue
    return None


class IsActiveSyncStats(TypedDict):
    total: int
    parsed: int
    missing_or_bad: int
    will_set_true: int
    will_set_false: int
    unchanged: int
    modified: int


def sync_is_active_from_closing_date(
    coll: Collection[Any],
    *,
    today: Optional[date] = None,
    missing_policy: Literal["keep", "active", "inactive"] = "keep",
    dry_run: bool = False,
) -> IsActiveSyncStats:
    """
    Scan ``coll`` and set ``is_active`` from ``classifier_metadata.closing_date``.

    When ``dry_run`` is True, no writes are performed and ``modified`` is 0.
    """
    day = today if today is not None else date.today()

    total = 0
    parsed = 0
    missing_or_bad = 0
    will_set_true = 0
    will_set_false = 0
    unchanged = 0
    updates: list[UpdateOne] = []

    proj = {"_id": 1, "is_active": 1, "classifier_metadata.closing_date": 1}
    for doc in coll.find({}, proj):
        total += 1
        current = bool(doc.get("is_active", False))
        closing_raw = (doc.get("classifier_metadata") or {}).get("closing_date")
        d = _parse_date(closing_raw)

        if d is None:
            missing_or_bad += 1
            if missing_policy == "keep":
                target = current
            elif missing_policy == "active":
                target = True
            else:
                target = False
        else:
            parsed += 1
            target = d >= day

        if target == current:
            unchanged += 1
            continue

        if target:
            will_set_true += 1
        else:
            will_set_false += 1
        updates.append(UpdateOne({"_id": doc["_id"]}, {"$set": {"is_active": target}}))

    modified = 0
    if updates and not dry_run:
        res = coll.bulk_write(updates, ordered=False)
        modified = int(res.modified_count)

    return IsActiveSyncStats(
        total=total,
        parsed=parsed,
        missing_or_bad=missing_or_bad,
        will_set_true=will_set_true,
        will_set_false=will_set_false,
        unchanged=unchanged,
        modified=modified,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Persist updates to Mongo (default: print counts only)")
    parser.add_argument(
        "--missing-policy",
        choices=("keep", "active", "inactive"),
        default="keep",
        help="How to handle missing/unparseable closing_date",
    )
    parser.add_argument(
        "--today",
        default=None,
        help="Override today (YYYY-MM-DD) for deterministic runs",
    )
    args = parser.parse_args()

    # Default behavior is dry-run unless --apply is explicitly provided.
    do_apply = bool(args.apply)

    _root = Path(__file__).resolve().parents[1]
    load_dotenv(_root / ".env", override=False)
    mongo_url = (os.getenv("MONGODB_URI") or os.getenv("MONGO_URL") or "").strip()
    db_name = (os.getenv("MONGODB_DATABASE") or os.getenv("MONGO_DB_NAME") or "").strip()
    coll_name = (
        os.getenv("MONGODB_TARGET_COLLECTION")
        or os.getenv("MONGO_JOBS_COLLECTION")
        or "ranked_jobs_enriched"
    ).strip()
    if not mongo_url or not db_name:
        raise SystemExit(
            "Set MONGODB_URI and MONGODB_DATABASE in llm-reranker/.env "
            "(or legacy MONGO_URL / MONGO_DB_NAME)."
        )

    if args.today:
        today = datetime.strptime(args.today, "%Y-%m-%d").date()
    else:
        today = date.today()

    client = MongoClient(mongo_url, serverSelectionTimeoutMS=15000)
    coll = client[db_name][coll_name]

    stats = sync_is_active_from_closing_date(
        coll,
        today=today,
        missing_policy=args.missing_policy,
        dry_run=not do_apply,
    )

    print(f"Collection: {db_name}.{coll_name}")
    print(f"Today: {today.isoformat()}")
    print(f"Total docs scanned: {stats['total']}")
    print(f"Parsed closing_date: {stats['parsed']}")
    print(f"Missing/unparseable closing_date: {stats['missing_or_bad']}")
    print(f"Would set is_active=True: {stats['will_set_true']}")
    print(f"Would set is_active=False: {stats['will_set_false']}")
    print(f"Unchanged: {stats['unchanged']}")

    if not do_apply:
        print("Dry run complete. Re-run with --apply to persist changes.")
        client.close()
        return 0

    if stats["will_set_true"] + stats["will_set_false"] == 0:
        print("No updates to apply.")
    else:
        print(
            "Applied updates:",
            f"modified={stats['modified']}",
        )
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
