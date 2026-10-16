"""Set ``is_active`` on ranked jobs from ``classifier_metadata.closing_date`` (dry run by default).

    python -m app.ranked_jobs.is_active_cli [--apply] [--missing-policy keep]

Targets ``MONGODB_DATABASE.MONGODB_TARGET_COLLECTION``. The batch runs the same sync as a
post-step (``RERANKER_SYNC_IS_ACTIVE_FROM_CLOSING_DATE``).
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

from app.config import get_settings
from app.ranked_jobs.mongo_repository import MongoRankedJobsRepository
from app.server_dependencies.db_dependencies import standalone_db


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply", action="store_true", help="Persist updates to Mongo (default: print counts only)"
    )
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

    settings = get_settings()

    if args.today:
        today = datetime.strptime(args.today, "%Y-%m-%d").date()
    else:
        today = date.today()

    with standalone_db(settings) as db:
        ranked_jobs = MongoRankedJobsRepository(db, settings)
        location = ranked_jobs.location
        stats = ranked_jobs.sync_is_active_from_closing_date(
            today=today,
            missing_policy=args.missing_policy,
            dry_run=not do_apply,
        )

    print(f"Collection: {location}")
    print(f"Today: {today.isoformat()}")
    print(f"Total docs scanned: {stats['total']}")
    print(f"Parsed closing_date: {stats['parsed']}")
    print(f"Missing/unparseable closing_date: {stats['missing_or_bad']}")
    print(f"Would set is_active=True: {stats['will_set_true']}")
    print(f"Would set is_active=False: {stats['will_set_false']}")
    print(f"Unchanged: {stats['unchanged']}")

    if not do_apply:
        print("Dry run complete. Re-run with --apply to persist changes.")
        return 0

    if stats["will_set_true"] + stats["will_set_false"] == 0:
        print("No updates to apply.")
    else:
        print(
            "Applied updates:",
            f"modified={stats['modified']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
