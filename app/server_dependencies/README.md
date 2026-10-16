# Database access

The batch reads classified job ads from MongoDB, writes ranked job documents back, and keeps a run
log and a per-country cutoff state. This document covers who owns the connection, which collections
exist, and how to add a query. The layout follows Compass's backend (`app/server_dependencies/` +
per-collection repositories), as the matching engine does.

```
app/
├── server_dependencies/
│   ├── db_dependencies.py       RerankerDBProvider: the one client; build_client_kwargs; standalone_db
│   └── database_collections.py  Collections: collection names (from Settings)
├── classified_jobs/             source collection (read)
│   ├── repository.py            IClassifiedJobsRepository: the interface the batch service depends on (and tests fake)
│   ├── types.py                 ClassifiedJobsFilter: which jobs a batch reads (no Mongo syntax)
│   └── mongo_repository.py      MongoClassifiedJobsRepository: every query on the source collection
├── ranked_jobs/                 target collection (write; also the is_active sync and enrichment queries)
│   ├── repository.py            IRankedJobsRepository
│   ├── mongo_repository.py      MongoRankedJobsRepository: builds every UpdateOne it writes
│   ├── types.py                 RankedJobUpsert: a ranked document and the key it is matched on
│   ├── upsert.py                ranked_job_upsert: picks that key for one ranked document (pure)
│   └── is_active.py             is_active from closing_date (pure rules)
├── run_logs/                    reranker_logs: IRunLogsRepository / MongoRunLogsRepository, RunStats
├── run_state/                   reranker_state: IRunStateRepository / MongoRunStateRepository
├── job_context.py               build_job_description_context: classified doc → JobDescriptionContext (pure)
└── translation.py               English machine-translation overlay for the listing text (pure)
```

## Connection lifecycle

There is exactly one `MongoClient` per process, owned by `RerankerDBProvider`. No module creates a
client at import time, so importing the app never needs `MONGODB_URI`, and no function takes an
optional client and opens its own.

### Batch run (`main.py`)

1. **Run start**: `run()` calls `RerankerDBProvider.get_db(settings)`, which reads the settings and
   creates the client. pymongo connects lazily, so this does no I/O. `build_batch_ranking_service`
   (`app/batch/get_batch_service.py`) builds the four repositories around that database, and
   `BatchRankingService` uses only them.
2. **During the run**: the worker threads (`RERANKER_PARALLEL_WORKERS`) share the client.
   `MongoClient` is thread-safe and pools connections. The post-steps get the same connection: the
   `is_active` sync runs through `MongoRankedJobsRepository`, and the work-activities enrichment gets
   a `MongoRankedJobsRepository` on the run's client (`ENRICH_WA_DATABASE` may name another database
   on the same cluster).
3. **Run end**: `main()` calls `RerankerDBProvider.close()` in its `finally`, so the client is closed
   on success, on failure, and on the `SystemExit(1)` path.

`get_client()` creates the client under a `threading.Lock` with a double check, so concurrent first
callers share one client.

### CLIs

CLIs have no run around them. They open a short-lived client with `standalone_db()`, which closes it
when the block exits:

```python
from app.ranked_jobs.mongo_repository import MongoRankedJobsRepository
from app.server_dependencies.db_dependencies import standalone_db

settings = get_settings()
with standalone_db(settings) as db:
    stats = MongoRankedJobsRepository(db, settings).sync_is_active_from_closing_date()
```

`app/ranked_jobs/is_active_cli.py`, `app/work_activities_enrichment/cli.py` and
`scripts/backfill_whiten_ranked_jobs.py` work this way. Never use `standalone_db()` inside the batch
run: it would close the run's client.

### API (`api.py`)

The API holds **no** client. `POST /rank` takes the classified document in the body and returns the
ranked document; it does not read or write MongoDB. If a route ever needs the database, open the
client in the `lifespan` (`RerankerDBProvider.get_db()` at startup, `RerankerDBProvider.close()` at
shutdown), and give the route a service that holds the repository, through the module's
`get_<x>_service.py` provider (as compass does):

```python
# app/<use_case>/get_<x>_service.py
def get_x_service() -> IXService:
    settings = get_settings()
    return XService(MongoRankedJobsRepository(RerankerDBProvider.get_db(settings), settings))

# app/<use_case>/routes.py
@router.get("/…")
async def route(service: IXService = Depends(get_x_service)): ...
```

pymongo is synchronous: run its calls off the event loop (a plain `def` route, or
`run_in_threadpool`).

### Settings

The settings are read from the environment on first use (`Settings.from_env`). See `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `MONGODB_URI` | `mongodb://localhost:27017` | Connection string |
| `MONGODB_DATABASE` | `ZambiaJobs` | Database (one per country/stack) |
| `MONGODB_SOURCE_COLLECTION` | `classified_jobs` | Source collection (legacy alias `MONGODB_CLASSIFIED_COLLECTION`) |
| `MONGODB_TARGET_COLLECTION` | `ranked_jobs` | Target collection |
| `RERANKER_LOGS_COLLECTION` | `reranker_logs` | Run log collection |
| `RERANKER_STATE_COLLECTION` | `reranker_state` | Cutoff state collection |
| `MONGODB_TLS_CA_FILE` | — | CA bundle path |
| `MONGODB_TLS_INSECURE` | off | Skip TLS certificate verification (dev only) |

A TLS URI (`mongodb+srv://`, `tls=true`, `ssl=true`) without `MONGODB_TLS_CA_FILE` uses the
`certifi` bundle. `MONGODB_TLS_INSECURE` wins over both.

The work-activities enrichment can target another database or collection on the same cluster
(`ENRICH_WA_DATABASE`, `ENRICH_WA_TARGET_COLLECTION`). It uses the same client.

## Collections

| `Collections.` | Default name | Written by | Read by |
|---|---|---|---|
| `SOURCE` | `classified_jobs` | job classifier | `MongoClassifiedJobsRepository` |
| `TARGET` | `ranked_jobs` | `MongoRankedJobsRepository` (upserts), WA enrichment, `is_active` sync | matching engine; `distinct_job_ids` for `--skip-already-ranked` |
| `LOGS` | `reranker_logs` | `MongoRunLogsRepository` | people (run history); the `_id` names the Langfuse session |
| `STATE` | `reranker_state` | `MongoRunStateRepository` | `MongoRunStateRepository` (next run's cutoff) |

- **Source**: the batch streams it newest `_id` first, with `no_cursor_timeout` because long parallel
  runs can sit between chunk reads. With a cutoff, the filter is
  `{"classified_at": {"$gte": cutoff}}`; with `--skip-already-ranked`, the `job_id`s already in the
  target are excluded. `build_job_description_context` maps a document to the listing fields; its
  field names are configured in `Settings`.
- **Target**: each job is upserted by `job_id`, else `job_fingerprint`, else `source_classified_id`
  (see `ranked_jobs/upsert.py`). The document shape is the contract with the matching engine, defined
  by `app/job_ranking/ranked_document.py`.
- **Logs**: one document per run, holding `status` (`running` → `completed` / `failed`), `config`
  (model, limit, collections), `stats` (jobs, upserts, LLM calls), `elapsed_s` and `error_message`.
  Progress is written every 10 jobs.
- **State**: one document per `country` (`TARGET_COUNTRY`), holding `last_cutoff_date`
  (`YYYY-MM-DD`), `last_run_at` and `jobs_processed`. It is saved only after a successful run, and
  the next run fetches from that date unless `RERANKER_CUTOFF_DATE` /
  `<COUNTRY>_RERANKER_CUTOFF_DATE` overrides it.

## Adding a query

1. **Declare it on the interface.** Add an abstract method to the collection's `I…Repository`
   (`repository.py`), with a docstring that states its filter, sort and return shape. The services
   depend on the interface, so this is also what tests fake.
2. **Implement it in the `Mongo…Repository`** (`mongo_repository.py`). Use `self._collection`. Do
   not open a client or reach for the provider.
3. **Keep policy out of it, and Mongo syntax in it.** If the filter depends on run options or
   business rules (like the cutoff resolution in `BatchRankingService.build_source_filter`), decide
   it in the service and pass it in as a plain type from the collection's `types.py`, as
   `iter_jobs(job_filter=ClassifiedJobsFilter(…))` does; the repository turns it into the query.
   Writes work the same way: the service hands over `RankedJobUpsert`s and the repository builds
   the `UpdateOne`s. Services never import `pymongo` (`tests/test_architecture.py` enforces it).
4. **Use it** from the service that receives the repository (built in the module's `get_<x>_service.py`).
5. **Test it offline.** `tests/test_repositories.py` runs each repository against an in-process
   fake database that records `find` / `sort` / `limit` / `bulk_write` / `update_one`.

### Adding a collection

Add a `Settings` field (env-driven, with a default) and a `Collections` entry, then create a
repository package like `app/run_state/` (an interface plus a Mongo implementation taking
`(db, settings)`). Build it in the provider of the service that uses it (e.g.
`app/batch/get_batch_service.py`). If it lives in a different database,
take it from `RerankerDBProvider.get_client()[name]`. Never create a second client.
