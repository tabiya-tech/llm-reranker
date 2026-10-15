"""Tests for the Mongo repositories against an in-process fake database (no Mongo required)."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from bson import ObjectId
from pymongo.operations import UpdateOne

from app.classified_jobs.mongo_repository import MongoClassifiedJobsRepository
from app.ranked_jobs.mongo_repository import MongoRankedJobsRepository
from app.run_logs.mongo_repository import MongoRunLogsRepository
from app.run_logs.repository import RunStats
from app.run_state.mongo_repository import MongoRunStateRepository


class FakeCursor:
    """Records ``sort`` / ``skip`` / ``limit`` and applies them to an in-memory list."""

    def __init__(self, docs: list[dict[str, Any]]):
        self._docs = docs
        self.sort_args: tuple[str, int] | None = None
        self.skip_n = 0
        self.limit_n = 0

    def sort(self, field: str, direction: int) -> FakeCursor:
        self.sort_args = (field, direction)
        self._docs = sorted(self._docs, key=lambda d: d[field], reverse=direction < 0)
        return self

    def skip(self, n: int) -> FakeCursor:
        self.skip_n = n
        return self

    def limit(self, n: int) -> FakeCursor:
        self.limit_n = n
        return self

    def __iter__(self):
        docs = self._docs[self.skip_n :]
        if self.limit_n:
            docs = docs[: self.limit_n]
        return iter(docs)


class FakeCollection:
    """Minimal in-memory pymongo Collection that records every call."""

    def __init__(self, docs: list[dict[str, Any]] | None = None):
        self.docs = list(docs or [])
        self.find_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.cursors: list[FakeCursor] = []
        self.bulk_write_calls: list[tuple[list[UpdateOne], bool]] = []
        self.update_one_calls: list[tuple[dict[str, Any], dict[str, Any], bool]] = []

    def find(self, query: dict[str, Any], **kwargs: Any) -> FakeCursor:
        self.find_calls.append((query, kwargs))
        cur = FakeCursor(self.docs)
        self.cursors.append(cur)
        return cur

    def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        return next(
            (d for d in self.docs if all(d.get(k) == v for k, v in query.items())),
            None,
        )

    def distinct(self, field: str, _query: dict[str, Any]) -> list[Any]:
        return list(dict.fromkeys(d.get(field) for d in self.docs))

    def bulk_write(self, ops: list[UpdateOne], ordered: bool = True) -> SimpleNamespace:
        self.bulk_write_calls.append((ops, ordered))
        return SimpleNamespace(matched_count=0, modified_count=0, upserted_count=len(ops))

    def insert_one(self, doc: dict[str, Any]) -> SimpleNamespace:
        doc = {"_id": ObjectId(), **doc}
        self.docs.append(doc)
        return SimpleNamespace(inserted_id=doc["_id"])

    def update_one(
        self, query: dict[str, Any], update: dict[str, Any], upsert: bool = False
    ) -> None:
        self.update_one_calls.append((query, update, upsert))


class FakeDatabase:
    name = "TestJobs"

    def __init__(self, collections: dict[str, FakeCollection] | None = None):
        self.collections = collections or {}

    def get_collection(self, name: str) -> FakeCollection:
        return self.collections.setdefault(name, FakeCollection())


@pytest.fixture
def settings(make_settings):
    return make_settings(
        MONGODB_SOURCE_COLLECTION="src",
        MONGODB_TARGET_COLLECTION="dst",
        RERANKER_LOGS_COLLECTION="logs",
        RERANKER_STATE_COLLECTION="state",
    )


class TestClassifiedJobsRepository:
    def _repo(self, settings, n: int = 5):
        db = FakeDatabase({"src": FakeCollection([{"_id": i} for i in range(n)])})
        return MongoClassifiedJobsRepository(db, settings), db.collections["src"]  # type: ignore[arg-type]

    def test_iter_jobs_newest_first_without_cursor_timeout(self, settings):
        repo, coll = self._repo(settings)
        docs = list(repo.iter_jobs(query={"classified_at": {"$gte": 1}}))
        assert [d["_id"] for d in docs] == [4, 3, 2, 1, 0]
        assert coll.find_calls == [({"classified_at": {"$gte": 1}}, {"no_cursor_timeout": True})]
        assert coll.cursors[0].sort_args == ("_id", -1)

    def test_iter_jobs_limit_zero_means_all(self, settings):
        repo, coll = self._repo(settings)
        assert len(list(repo.iter_jobs(limit=0))) == 5
        assert coll.cursors[0].limit_n == 0
        assert len(list(repo.iter_jobs(limit=2))) == 2

    def test_iter_jobs_chunks(self, settings):
        repo, _ = self._repo(settings)
        chunks = list(repo.iter_jobs_chunks(2))
        assert [[d["_id"] for d in c] for c in chunks] == [[4, 3], [2, 1], [0]]

    def test_iter_jobs_chunks_rejects_zero(self, settings):
        repo, _ = self._repo(settings)
        with pytest.raises(ValueError, match="chunk_size"):
            list(repo.iter_jobs_chunks(0))


class TestRankedJobsRepository:
    def test_distinct_job_ids_drops_none(self, settings):
        db = FakeDatabase(
            {"dst": FakeCollection([{"job_id": "a"}, {"job_id": None}, {"job_id": "b"}])}
        )
        repo = MongoRankedJobsRepository(db, settings)  # type: ignore[arg-type]
        assert repo.distinct_job_ids() == ["a", "b"]

    def test_bulk_upsert_is_unordered_and_skips_empty(self, settings):
        db = FakeDatabase()
        repo = MongoRankedJobsRepository(db, settings)  # type: ignore[arg-type]
        repo.bulk_upsert([])
        op = UpdateOne({"job_id": "a"}, {"$set": {"x": 1}}, upsert=True)
        repo.bulk_upsert([op])
        assert db.collections["dst"].bulk_write_calls == [([op], False)]


class TestRunLogsRepository:
    def test_run_lifecycle(self, settings):
        db = FakeDatabase()
        repo = MongoRunLogsRepository(db, settings)  # type: ignore[arg-type]
        coll = db.collections["logs"]

        run_id = repo.insert_run(
            limit=10, model="m", source_collection="src", target_collection="dst"
        )
        (doc,) = coll.docs
        assert doc["_id"] == run_id
        assert doc["status"] == "running"
        assert doc["config"] == {
            "model": "m",
            "limit": 10,
            "source_collection": "src",
            "target_collection": "dst",
        }

        stats = RunStats(
            jobs_processed=3, upserts_queued=2, llm1_calls=3, llm2_calls=3, llm3_calls=1
        )
        repo.update_progress(run_id, stats)
        repo.complete_run(run_id, stats, success=False, elapsed_s=1.23456, error_message="E")

        (q1, u1, _), (q2, u2, _) = coll.update_one_calls
        assert q1 == q2 == {"_id": run_id}
        assert u1["$set"]["stats.jobs_processed"] == 3
        assert u2["$set"]["status"] == "failed"
        assert u2["$set"]["success"] is False
        assert u2["$set"]["error_message"] == "E"
        assert u2["$set"]["elapsed_s"] == 1.235
        assert u2["$set"]["stats.llm3_calls"] == 1


class TestRunStateRepository:
    def _repo(self, settings, docs):
        db = FakeDatabase({"state": FakeCollection(docs)})
        return MongoRunStateRepository(db, settings), db.collections["state"]  # type: ignore[arg-type]

    def test_no_state_returns_none(self, settings):
        repo, _ = self._repo(settings, [])
        assert repo.get_last_cutoff_date("kenya") is None

    def test_string_cutoff(self, settings):
        repo, _ = self._repo(settings, [{"country": "kenya", "last_cutoff_date": "2026-03-04"}])
        assert repo.get_last_cutoff_date("kenya") == "2026-03-04"

    def test_datetime_cutoff_is_converted_to_utc_date(self, settings):
        naive = datetime(2026, 3, 4, 23, 30)
        repo, _ = self._repo(settings, [{"country": "kenya", "last_cutoff_date": naive}])
        assert repo.get_last_cutoff_date("kenya") == "2026-03-04"

    def test_save_upserts_today(self, settings):
        repo, coll = self._repo(settings, [])
        repo.save("kenya", jobs_processed=7)
        ((query, update, upsert),) = coll.update_one_calls
        assert query == {"country": "kenya"}
        assert upsert is True
        assert update["$set"]["jobs_processed"] == 7
        assert update["$set"]["last_cutoff_date"] == datetime.now(timezone.utc).strftime("%Y-%m-%d")
