"""The batch run, its post-steps and the taxonomy cache, over in-memory repositories."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from bson import ObjectId

from app.batch.post_steps import BatchPostSteps
from app.batch.service import BatchOptions, BatchRankingService
from app.classified_jobs.repository import IClassifiedJobsRepository
from app.classified_jobs.types import ClassifiedJobsFilter
from app.config import Settings
from app.job_embedding.service import JobEmbeddingService
from app.job_embedding.whitening_repository import WhiteningArtifactRepository
from app.ranked_jobs.is_active import IsActiveSyncStats, MissingPolicy
from app.ranked_jobs.repository import IRankedJobsRepository
from app.ranked_jobs.types import RankedJobKey, RankedJobUpsert
from app.run_logs.repository import IRunLogsRepository, RunStats
from app.run_state.repository import IRunStateRepository
from app.taxonomy.repository import TaxonomyRepository
from app.taxonomy.service import TaxonomyService
from app.taxonomy.types import TaxonomyContext
from app.work_activities_enrichment.repository import WorkActivityTaxonomyRepository
from app.work_activities_enrichment.service import WorkActivitiesEnrichmentService
from app.work_activities_enrichment.types import WorkActivitiesEnrichmentStats, WorkActivityTaxonomy
from tests.ranking_fakes import (
    FixedRankers,
    StubJobAttributes,
    build_ranking_service,
    classified_doc,
)


class InMemoryClassifiedJobs(IClassifiedJobsRepository):
    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs
        self.filters: list[ClassifiedJobsFilter | None] = []

    def iter_jobs(
        self,
        *,
        limit: int = 0,
        skip: int = 0,
        sort_field: str = "_id",
        sort_direction: int = -1,
        job_filter: ClassifiedJobsFilter | None = None,
    ) -> Iterator[dict[str, Any]]:
        self.filters.append(job_filter)
        yield from (self.docs[:limit] if limit > 0 else self.docs)

    def iter_jobs_chunks(
        self,
        chunk_size: int,
        *,
        limit: int = 0,
        skip: int = 0,
        sort_field: str = "_id",
        sort_direction: int = -1,
        job_filter: ClassifiedJobsFilter | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        docs = list(self.iter_jobs(limit=limit, job_filter=job_filter))
        for i in range(0, len(docs), chunk_size):
            yield docs[i : i + chunk_size]


class InMemoryRankedJobs(IRankedJobsRepository):
    def __init__(self, docs: list[dict[str, Any]] | None = None):
        self.docs = docs or []
        self.upserts: list[RankedJobUpsert] = []
        self.is_active_syncs = 0
        self.set_calls: list[tuple[Any, dict[str, Any]]] = []

    @property
    def location(self) -> str:
        return "memory.ranked_jobs"

    def distinct_job_ids(self) -> list[Any]:
        return [d["job_id"] for d in self.docs if d.get("job_id") is not None]

    def bulk_upsert(self, upserts: list[RankedJobUpsert]) -> None:
        self.upserts.extend(upserts)

    def sync_is_active_from_closing_date(
        self,
        *,
        today: date | None = None,
        missing_policy: MissingPolicy = "keep",
        dry_run: bool = False,
    ) -> IsActiveSyncStats:
        self.is_active_syncs += 1
        return IsActiveSyncStats(
            total=0,
            parsed=0,
            missing_or_bad=0,
            will_set_true=0,
            will_set_false=0,
            unchanged=0,
            modified=0,
        )

    def iter_for_work_activities_enrichment(
        self, *, only_missing: bool = True, limit: int = 0
    ) -> Iterator[dict[str, Any]]:
        yield from self.docs

    def iter_whitening_backfill_candidates(
        self, *, limit: int | None = None
    ) -> Iterator[dict[str, Any]]:
        yield from ()

    def bulk_set(self, updates: list[tuple[Any, dict[str, Any]]]) -> int:
        self.set_calls.extend(updates)
        return len(updates)


class InMemoryRunLogs(IRunLogsRepository):
    def __init__(self):
        self.completed: list[tuple[RunStats, bool, str | None]] = []

    def insert_run(
        self, *, limit: int, model: str, source_collection: str, target_collection: str
    ) -> ObjectId:
        return ObjectId()

    def update_progress(self, run_id: ObjectId, stats: RunStats) -> None:
        pass

    def complete_run(
        self,
        run_id: ObjectId,
        stats: RunStats,
        *,
        success: bool,
        elapsed_s: float,
        error_message: str | None = None,
    ) -> None:
        self.completed.append((stats, success, error_message))


class InMemoryRunState(IRunStateRepository):
    def __init__(self, cutoff: str | None = None):
        self.cutoff = cutoff
        self.saved: list[tuple[str, int]] = []

    def get_last_cutoff_date(self, country: str) -> str | None:
        return self.cutoff

    def save(self, country: str, *, jobs_processed: int) -> None:
        self.saved.append((country, jobs_processed))


class _Enrichment(WorkActivitiesEnrichmentService):
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.runs = 0

    def run(self) -> WorkActivitiesEnrichmentStats:
        self.runs += 1
        if self.error is not None:
            raise self.error
        return WorkActivitiesEnrichmentStats()


def _settings(make_settings, **env: str) -> Settings:
    return make_settings(
        GEMINI_API_KEY="dummy",
        TARGET_COUNTRY="kenya",
        RERANKER_ENABLE_JOB_EMBEDDING="0",
        **env,
    )


def _service(
    settings: Settings,
    *,
    docs: list[dict[str, Any]],
    ranked_jobs: InMemoryRankedJobs | None = None,
    run_state: InMemoryRunState | None = None,
    enrichment: _Enrichment | None = None,
    llm2: StubJobAttributes | None = None,
) -> tuple[BatchRankingService, dict[str, Any]]:
    ranked = ranked_jobs or InMemoryRankedJobs()
    parts: dict[str, Any] = {
        "classified_jobs": InMemoryClassifiedJobs(docs),
        "ranked_jobs": ranked,
        "run_logs": InMemoryRunLogs(),
        "run_state": run_state or InMemoryRunState(),
        "enrichment": enrichment or _Enrichment(),
    }
    service = BatchRankingService(
        settings=settings,
        llm_model="stub",
        rankers=FixedRankers(build_ranking_service(settings, job_attributes=llm2)),
        embedding=JobEmbeddingService(settings, None, WhiteningArtifactRepository()),
        classified_jobs=parts["classified_jobs"],
        ranked_jobs=ranked,
        run_logs=parts["run_logs"],
        run_state=parts["run_state"],
        post_steps=BatchPostSteps(settings, ranked, enrichment=lambda: parts["enrichment"]),
    )
    return service, parts


def _doc(n: int) -> dict[str, Any]:
    doc = classified_doc()
    doc.update(_id=f"src-{n}", job_id=f"jid-{n}", job_fingerprint=f"fp-{n}")
    return doc


class TestBatchRankingService:
    def test_ranks_upserts_and_runs_post_steps(self, make_settings):
        s = _settings(make_settings)
        service, parts = _service(s, docs=[_doc(1), _doc(2)])

        assert service.run(BatchOptions()) is True

        assert [(u.key, u.key_value) for u in parts["ranked_jobs"].upserts] == [
            (RankedJobKey.JOB_ID, "jid-1"),
            (RankedJobKey.JOB_ID, "jid-2"),
        ]
        ((stats, success, error),) = parts["run_logs"].completed
        assert (stats.jobs_processed, stats.upserts_queued, success, error) == (2, 2, True, None)
        assert parts["enrichment"].runs == 1
        assert parts["ranked_jobs"].is_active_syncs == 1
        assert parts["run_state"].saved == [("kenya", 2)]

    def test_parallel_workers_rank_every_job(self, make_settings):
        s = _settings(make_settings, RERANKER_PARALLEL_WORKERS="3")
        service, parts = _service(s, docs=[_doc(i) for i in range(5)])

        assert service.run(BatchOptions()) is True
        assert len(parts["ranked_jobs"].upserts) == 5

    def test_a_failed_job_is_counted_and_the_run_continues(self, make_settings):
        s = _settings(make_settings)
        llm2 = StubJobAttributes()
        answers = iter([RuntimeError("model down"), None])

        def _flaky():
            err = next(answers)
            if err is not None:
                raise err
            return StubJobAttributes().answer()

        llm2.answer = _flaky
        service, parts = _service(s, docs=[_doc(1), _doc(2)], llm2=llm2)

        assert service.run(BatchOptions()) is True
        assert len(parts["ranked_jobs"].upserts) == 1
        ((stats, success, _),) = parts["run_logs"].completed
        assert (stats.jobs_processed, success) == (1, True)

    def test_enrichment_failure_fails_the_run_only_when_configured(self, make_settings):
        s = _settings(make_settings, RERANKER_FAIL_ON_WA_ENRICHMENT_ERROR="1")
        service, parts = _service(s, docs=[_doc(1)], enrichment=_Enrichment(RuntimeError("x")))

        assert service.run(BatchOptions()) is False
        # A failed run neither syncs is_active nor moves the cutoff forward.
        assert parts["ranked_jobs"].is_active_syncs == 0
        assert parts["run_state"].saved == []

    def test_enrichment_failure_is_a_warning_by_default(self, make_settings):
        s = _settings(make_settings)
        service, parts = _service(s, docs=[_doc(1)], enrichment=_Enrichment(RuntimeError("x")))

        assert service.run(BatchOptions()) is True
        assert parts["run_state"].saved == [("kenya", 1)]

    def test_post_steps_can_be_switched_off(self, make_settings):
        s = _settings(
            make_settings,
            RERANKER_RUN_WA_ENRICHMENT="0",
            RERANKER_SYNC_IS_ACTIVE_FROM_CLOSING_DATE="0",
        )
        service, parts = _service(s, docs=[_doc(1)])

        assert service.run(BatchOptions()) is True
        assert parts["enrichment"].runs == 0
        assert parts["ranked_jobs"].is_active_syncs == 0


class TestSourceFilter:
    def test_cutoff_from_run_state(self, make_settings):
        s = _settings(make_settings)
        service, _ = _service(s, docs=[], run_state=InMemoryRunState("2026-01-02"))

        job_filter = service.build_source_filter(skip_already_ranked=False)

        assert job_filter == ClassifiedJobsFilter(
            classified_since=datetime(2026, 1, 2, tzinfo=timezone.utc)
        )

    def test_country_env_override_wins_over_state(self, make_settings):
        s = _settings(make_settings, KENYA_RERANKER_CUTOFF_DATE="2025-05-05")
        service, _ = _service(s, docs=[], run_state=InMemoryRunState("2026-01-02"))

        assert service.resolve_cutoff_date() == "2025-05-05"

    def test_skip_already_ranked_excludes_ranked_ids(self, make_settings):
        s = _settings(make_settings)
        service, _ = _service(s, docs=[], ranked_jobs=InMemoryRankedJobs([{"job_id": "jid-9"}]))

        assert service.build_source_filter(skip_already_ranked=True) == ClassifiedJobsFilter(
            exclude_job_ids=("jid-9",)
        )

    def test_no_state_is_a_full_scan(self, make_settings):
        service, _ = _service(_settings(make_settings), docs=[])
        assert service.build_source_filter(skip_already_ranked=False) == ClassifiedJobsFilter()


class _WaTaxonomy(WorkActivityTaxonomyRepository):
    def load(self, path: Path) -> WorkActivityTaxonomy:
        return WorkActivityTaxonomy(
            occ_lookup={"welder": {"4.A.1": {"importance": 4.0, "level": 3.0}}},
            wa_averages={"4.A.2": {"importance": 2.5, "level": 2.0}},
        )


class TestWorkActivitiesEnrichmentService:
    def test_sets_rows_from_occupation_then_averages(self, make_settings):
        target = InMemoryRankedJobs(
            [
                {
                    "_id": 1,
                    "classified_occupations": [{"label": "Welder"}],
                    "work_activity": {"items": [{"id": "4.A.1", "name": "Weld"}]},
                },
                {
                    "_id": 2,
                    "work_activity": {"items": [{"id": "4.A.2", "name": "Paint"}]},
                },
                {"_id": 3},
            ]
        )
        settings = make_settings(ENRICH_WA_OCCUPATION_DB_PATH="unused.json")

        stats = WorkActivitiesEnrichmentService(settings, target, _WaTaxonomy()).run()

        assert (stats.scanned, stats.with_occ_match, stats.with_avg_match, stats.no_data) == (
            3,
            1,
            1,
            1,
        )
        assert stats.applied == 2
        rows = {_id: fields["onet_work_activities"] for _id, fields in target.set_calls}
        assert rows[1][0]["WA_Importance"] == 4.0
        assert rows[2][0]["WA_Level"] == 2.0


class _CountingTaxonomyRepository(TaxonomyRepository):
    def __init__(self):
        self.loads: list[str] = []

    def load(self, settings: Settings) -> TaxonomyContext:
        self.loads.append(settings.language)
        return super().load(settings)


class TestTaxonomyService:
    def test_loads_each_language_once(self, make_settings):
        from app.config import get_settings_for_language

        make_settings(TARGET_COUNTRY="kenya")
        repo = _CountingTaxonomyRepository()
        service = TaxonomyService(repo)

        en = service.get(get_settings_for_language("en"))
        assert service.get(get_settings_for_language("en")) is en
        service.get(get_settings_for_language("es"))

        assert repo.loads == ["en", "es"]
        assert service.loaded_languages == ["en", "es"]
        assert service.is_loaded("en")


def test_invalid_cutoff_falls_back_to_full_scan(make_settings):
    service, _ = _service(_settings(make_settings, RERANKER_CUTOFF_DATE="not-a-date"), docs=[])
    assert service.build_source_filter(skip_already_ranked=False) == ClassifiedJobsFilter()
