# LLM Re-ranker

Batch pipeline and HTTP API that processes classified job documents from MongoDB through Google Gemini, producing ranked skill profiles, job attributes, and ONET work activities.

## Pipeline

1. **Skill ranker (LLM 1)** — Classifier ESCO skills ranked as essential, optional, or discarded.
2. **Job attributes (LLM 2)** — Structured attribute estimates and ONET work activities.
3. **Occupation–skills relevance (LLM 3)** — Taxonomy skills scored for relevance to the specific job ad.


## Repository layout

| Path | Purpose |
|------|---------|
| `api.py` | FastAPI service — single-job HTTP endpoint (Cloud Run Service) |
| `main.py` | Batch CLI — process full collection (Cloud Run Job) |
| `app/pipeline.py` | Shared per-job processing core |
| `app/` | Config, MongoDB, prompts, Gemini client, schemas, logging |
| `utils/` | Classifier entities, ESCO–Tabiya mapping, occupation–skill relations |
| `resources/` | Prompt templates, job-attribute schema, ONET reference data |

## Prerequisites

- Python 3.11+
- MongoDB
- [Gemini API key](https://aistudio.google.com/apikey)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with: `MONGODB_URI`, `MONGODB_DATABASE`, `MONGODB_SOURCE_COLLECTION`, `MONGODB_TARGET_COLLECTION`, and `GEMINI_API_KEY`. See `.env.example` for all available options.

## API — single job

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**POST `/rank`** — send a classified job document directly as the JSON body:

```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{ "_id": "...", "title": "...", "classification": { ... } }'
```

The endpoint accepts MongoDB Extended JSON (`$oid`, `$date`) so documents can be pasted directly from Compass or Atlas.

**GET `/health`** — readiness check.

Interactive docs at `http://localhost:8000/docs`.

## Batch CLI — full collection

```bash
python main.py
```

| Flag | Effect |
|------|--------|
| `--limit N` | Process at most N documents. Default: all (streamed, newest first). |
| `--verbose` | Extra per-job detail on stderr |
| `--log-llm1-prompt` | Log skill ranker prompt |
| `--log-llm2-prompt` | Log job attributes prompt |
| `--log-llm3-prompt` | Log occupation–skills relevance prompt |
| `--bulk-batch-size N` | Flush MongoDB writes every N upserts |

## Origin UUIDs, work activities, and job status

1. **`skill_groups_origin_uuids`** — After ranking, skill groups are derived from the skill hierarchy so each ranked job can carry the **origin UUIDs** of the relevant skill groups (see `.env` / `resources/` for the hierarchy CSV).
2. **`onet_work_activities`** — Model output includes work-activity context; after a successful batch, optional enrichment can fill or adjust **ONET-style work activities** (importance/level) using the occupation taxonomy reference JSON, when enabled in `.env` (`RERANKER_RUN_WA_ENRICHMENT` and related variables in `.env.example`).
3. **`is_active` and closing dates** — **`is_active`** can be synced from **`classifier_metadata.closing_date`** (open vs closed relative to today) after a successful batch when enabled in `.env`, or by running `python utils/update_is_active_from_closing_date.py` manually. No documents are deleted; only the flag is updated.

## Countries

`TARGET_COUNTRY` (`zambia` | `kenya` | `south_africa` | `argentina`) is the batch/state
key and picks the country name used in the prompts. Everything else that varies per
country is env-driven, so **one image serves every country** — the GCP Workflow supplies
the differences as env overrides on each Cloud Run Job execution.

### Argentina

Argentina's jobs and taxonomy are in Spanish. No code path differs; only configuration:

```bash
TARGET_COUNTRY=argentina
MONGODB_DATABASE=ArgentinaJobs_V2
MONGODB_SOURCE_COLLECTION=classified_jobs

# Argentine Spanish taxonomy — Tabiya (ESCO 1.1.1), locale AR-es
OCCUPATIONS_CSV=argentina/occupations.csv
SKILLS_CSV=argentina/skills.csv
OCCUPATION_TO_SKILL_RELATIONS_CSV=argentina/occupation_to_skill_relations.csv
SKILL_HIERARCHY_CSV=argentina/skill_hierarchy.csv
```

Those four files live in `resources/argentina/`. Their ESCO relation graph is identical
to the default files (the same 123,788 `escooccupation` rows, the same signalling-value
distribution); what changes is the Spanish `PREFERREDLABEL` text that LLM1's skill menu
and LLM2's occupation line are built from. `tests/test_argentina_resources.py` guards
the drop — presence, required columns, and the `AR-es` locale in `model_info.csv`.

`RERANKER_PREFER_TRANSLATION` (default `1`) makes the LLM prompts use
`translation.text_en` / `translation.fields_en` from `classified_jobs` when a document
carries them. Documents without a `translation` field fall through to the Spanish text
unchanged, so the flag is safe to leave on either way.

## Deployment

| Mode | Entrypoint | GCP |
|------|-----------|-----|
| HTTP (single job) | `uvicorn api:app --host 0.0.0.0 --port 8000` | Cloud Run **Service** |
| Batch (full collection) | `python main.py` | Cloud Run **Job** |

Both use the **same** image (`llm-reranker/Dockerfile`); only the process/command differs.

### Cloud Run Service (HTTPS API)

1. Build and push the image (from repo root):

   ```bash
   gcloud builds submit --config=llm-reranker/cloudbuild.yaml .
   ```

2. Deploy the service (edit script env or export variables as needed):

   ```bash
   chmod +x llm-reranker/deploy-cloud-run-service.sh
   # Recommended: set secrets (example — use your Secret Manager names)
   export GCP_PROJECT=horizon-dev-481316
   export RERANKER_SERVICE_SECRETS="GEMINI_API_KEY=GEMINI_API_KEY:latest"
   # Add Mongo and other keys as secrets or env vars your server needs at runtime.
   ./llm-reranker/deploy-cloud-run-service.sh
   ```

   The script runs `python -m uvicorn api:app --host 0.0.0.0 --port 8080`, sets request timeout to 3600s, and deploys with **authenticated** invocation by default (`--no-allow-unauthenticated`). Grant **Cloud Run Invoker** to teammates or the service account that calls `/rank`.

3. Call the API with an identity token (same pattern as `rerank_classified_json.py`):

   ```bash
   URL="$(gcloud run services describe llm-reranker-api --region=us-central1 --format='value(status.url)')"
   curl -sS -X POST "${URL}/rank" \
     -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
     -H "Content-Type: application/json" \
     -d @classified.json
   ```

4. Optional: `gcloud run services update llm-reranker-api --region=...` to add `--update-env-vars`, `--update-secrets`, memory, or `--allow-unauthenticated` (not recommended if unauthenticated public access is a concern).

## Run logs

Each batch run writes a document to the `ranking-logs` collection (created automatically) tracking status, config, stats, and timing. Progress is updated every 10 jobs. Failed runs include the error message.
# LLM Re-ranker

Batch pipeline and HTTP API that processes classified job documents from MongoDB through Google Gemini, producing ranked skill profiles, job attributes, and ONET work activities.

## Pipeline

1. **Skill ranker (LLM 1)** — Classifier ESCO skills ranked as essential, optional, or discarded.
2. **Job attributes (LLM 2)** — Structured attribute estimates and ONET work activities.
3. **Occupation–skills relevance (LLM 3)** — Taxonomy skills scored for relevance to the specific job ad.


## Repository layout

| Path | Purpose |
|------|---------|
| `api.py` | FastAPI service — single-job HTTP endpoint (Cloud Run Service) |
| `main.py` | Batch CLI — process full collection (Cloud Run Job) |
| `app/pipeline.py` | Shared per-job processing core |
| `app/` | Config, MongoDB, prompts, Gemini client, schemas, logging |
| `utils/` | Classifier entities, ESCO–Tabiya mapping, occupation–skill relations |
| `resources/` | Prompt templates, job-attribute schema, ONET reference data |

## Prerequisites

- Python 3.11+
- MongoDB
- [Gemini API key](https://aistudio.google.com/apikey)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with: `MONGODB_URI`, `MONGODB_DATABASE`, `MONGODB_SOURCE_COLLECTION`, `MONGODB_TARGET_COLLECTION`, and `GEMINI_API_KEY`. See `.env.example` for all available options.

## API — single job

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**POST `/rank`** — send a classified job document directly as the JSON body:

```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{ "_id": "...", "title": "...", "classification": { ... } }'
```

The endpoint accepts MongoDB Extended JSON (`$oid`, `$date`) so documents can be pasted directly from Compass or Atlas.

**GET `/health`** — readiness check.

Interactive docs at `http://localhost:8000/docs`.

## Batch CLI — full collection

```bash
python main.py
```

| Flag | Effect |
|------|--------|
| `--limit N` | Process at most N documents. Default: all (streamed, newest first). |
| `--verbose` | Extra per-job detail on stderr |
| `--log-llm1-prompt` | Log skill ranker prompt |
| `--log-llm2-prompt` | Log job attributes prompt |
| `--log-llm3-prompt` | Log occupation–skills relevance prompt |
| `--bulk-batch-size N` | Flush MongoDB writes every N upserts |

## Origin UUIDs, work activities, and job status

1. **`skill_groups_origin_uuids`** — After ranking, skill groups are derived from the skill hierarchy so each ranked job can carry the **origin UUIDs** of the relevant skill groups (see `.env` / `resources/` for the hierarchy CSV).
2. **`onet_work_activities`** — Model output includes work-activity context; after a successful batch, optional enrichment can fill or adjust **ONET-style work activities** (importance/level) using the occupation taxonomy reference JSON, when enabled in `.env` (`RERANKER_RUN_WA_ENRICHMENT` and related variables in `.env.example`).
3. **`is_active` and closing dates** — **`is_active`** can be synced from **`classifier_metadata.closing_date`** (open vs closed relative to today) after a successful batch when enabled in `.env`, or by running `python utils/update_is_active_from_closing_date.py` manually. No documents are deleted; only the flag is updated.

## Deployment

| Mode | Entrypoint | GCP |
|------|-----------|-----|
| HTTP (single job) | `uvicorn api:app --host 0.0.0.0 --port 8000` | Cloud Run Service |
| Batch (full collection) | `python main.py` | Cloud Run Job |

Both share the same Docker image with different entrypoint commands.

## Run logs

Each batch run writes a document to the `ranking-logs` collection (created automatically) tracking status, config, stats, and timing. Progress is updated every 10 jobs. Failed runs include the error message.
