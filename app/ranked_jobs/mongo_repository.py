from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import date
from typing import Any

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.operations import UpdateOne

from app.config import Settings
from app.ranked_jobs.is_active import IsActiveSyncStats, MissingPolicy, target_is_active
from app.ranked_jobs.repository import IRankedJobsRepository
from app.ranked_jobs.types import RankedJobKey, RankedJobUpsert
from app.server_dependencies.database_collections import Collections


def sync_is_active_from_closing_date(
    coll: Collection[Any],
    *,
    today: date | None = None,
    missing_policy: MissingPolicy = "keep",
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
        target, was_parsed = target_is_active(
            current, closing_raw, today=day, missing_policy=missing_policy
        )
        if was_parsed:
            parsed += 1
        else:
            missing_or_bad += 1

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


# Missing, null, or empty list all count as "needs enrichment" ($exists:false alone misses null/[]).
_MISSING_ONET_WORK_ACTIVITIES = {
    "$or": [
        {"onet_work_activities": {"$exists": False}},
        {"onet_work_activities": None},
        {"onet_work_activities": []},
    ]
}

# Jobs with a job embedding but no whitening provenance (or no raw embedding kept).
_WHITENING_BACKFILL_CANDIDATES = {
    "job_embedding": {"$exists": True, "$type": "array", "$ne": []},
    "$or": [
        {"llm_reranker_meta.embedding.whitening.enabled": {"$ne": True}},
        {"raw_embedding": {"$exists": False}},
    ],
}


# Legacy top-level fields from before schema v2, dropped whenever a job is re-ranked.
_LEGACY_FIELDS = {
    name: ""
    for name in (
        "llm_skill_ranker",
        "classified_linked_by_type",
        "occupation_skills",
        "occupation_skills_uuids",
        "classified_skills_uuids",
        "classified_occupations_uuids",
        "essential_skills",
        "essential_skills_uuids",
        "optional_skills",
        "optional_skills_uuids",
        "job_attributes",
        "work_activity_ids",
        "work_activity_names",
        "title",
        "enrichment",
        "embedding_meta",
    )
}


class MongoRankedJobsRepository(IRankedJobsRepository):
    """
    Mongo implementation of ``IRankedJobsRepository``: the ranked job documents the batch writes.

    ``collection_name`` overrides ``MONGODB_TARGET_COLLECTION`` (the work-activities enrichment
    may target another collection, see ``ENRICH_WA_TARGET_COLLECTION``).
    """

    def __init__(
        self,
        db: Database[dict[str, Any]],
        settings: Settings,
        *,
        collection_name: str | None = None,
    ):
        self._database_name = db.name
        self._collection_name = collection_name or Collections.from_settings(settings).TARGET
        self._collection = db.get_collection(self._collection_name)
        self._job_id_field = settings.mongo_field_job_id
        self._key_fields = {
            RankedJobKey.JOB_ID: settings.mongo_field_job_id,
            RankedJobKey.JOB_FINGERPRINT: settings.mongo_field_job_fingerprint,
            RankedJobKey.SOURCE_CLASSIFIED_ID: "source_classified_id",
        }
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def location(self) -> str:
        """``database.collection``, for log lines."""
        return f"{self._database_name}.{self._collection_name}"

    def distinct_job_ids(self) -> list[Any]:
        field = self._job_id_field
        ids = self._collection.distinct(field, {field: {"$exists": True, "$ne": None}})
        return [x for x in ids if x is not None]

    def upsert_operation(self, upsert: RankedJobUpsert) -> UpdateOne:
        """The ``UpdateOne`` that writes ``upsert``: ``$set`` its document, drop legacy fields."""
        return UpdateOne(
            {self._key_fields[upsert.key]: upsert.key_value},
            {"$set": upsert.document, "$unset": _LEGACY_FIELDS},
            upsert=True,
        )

    def bulk_upsert(self, upserts: list[RankedJobUpsert]) -> None:
        if not upserts:
            return
        operations = [self.upsert_operation(u) for u in upserts]
        result = self._collection.bulk_write(operations, ordered=False)
        self._logger.info(
            "Bulk upsert LLM results → %s.%s: ops=%s matched=%s modified=%s upserted=%s",
            self._database_name,
            self._collection_name,
            len(operations),
            result.matched_count,
            result.modified_count,
            result.upserted_count,
        )

    def sync_is_active_from_closing_date(
        self,
        *,
        today: date | None = None,
        missing_policy: MissingPolicy = "keep",
        dry_run: bool = False,
    ) -> IsActiveSyncStats:
        return sync_is_active_from_closing_date(
            self._collection, today=today, missing_policy=missing_policy, dry_run=dry_run
        )

    def iter_for_work_activities_enrichment(
        self, *, only_missing: bool = True, limit: int = 0
    ) -> Iterator[dict[str, Any]]:
        cur = self._collection.find(_MISSING_ONET_WORK_ACTIVITIES if only_missing else {}).sort(
            "_id", -1
        )
        if limit > 0:
            cur = cur.limit(limit)
        yield from cur

    def iter_whitening_backfill_candidates(
        self, *, limit: int | None = None
    ) -> Iterator[dict[str, Any]]:
        cur = self._collection.find(
            _WHITENING_BACKFILL_CANDIDATES,
            projection={"job_embedding": 1, "raw_embedding": 1, "llm_reranker_meta": 1},
        )
        if limit is not None:
            cur = cur.limit(limit)
        yield from cur

    def bulk_set(self, updates: list[tuple[Any, dict[str, Any]]]) -> int:
        if not updates:
            return 0
        ops = [UpdateOne({"_id": _id}, {"$set": fields}, upsert=False) for _id, fields in updates]
        result = self._collection.bulk_write(ops, ordered=False)
        return int(result.modified_count)
