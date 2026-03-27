# LLM re-ranker

Batch pipeline over classified job documents in MongoDB: Vertex AI (Gemini) reranks ESCO skills (essential vs optional) and infers job attributes plus ONET work activities.

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | CLI and orchestration |
| `app/` | Config, MongoDB (read + target upsert), prompts, Vertex, schemas, logging |
| `utils/` | ESCO skill extraction and label mapping |
| `resources/` | Prompt templates, job attribute schema JSON(s), ONET reference data |

## Prerequisites

- Python 3.11+
- MongoDB
- GCP: Vertex AI API enabled, billing, credentials with Vertex access

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Configure `.env`: `MONGODB_URI`, `MONGODB_DATABASE`, `MONGODB_SOURCE_COLLECTION`, `MONGODB_TARGET_COLLECTION`, optional `BULK_WRITE_BATCH_SIZE` (default `100`), optional `JOB_ATTRIBUTES_SCHEMA` (path to job-attributes JSON; default `resources/job_attributes_reduced.json` under `RESOURCES_DIR`), Vertex (`GCP_PROJECT_ID`, `GCP_LOCATION`, `VERTEX_MODEL`, credentials). Set `LOG_LEVEL` as needed.

## Usage

```bash
python main.py 
```

Each run performs **both** LLM steps (skill ranker when ESCO candidates exist, then job attributes) and upserts to the target collection unless you use **`--dry-run`**.

| Flag | Effect |
|------|--------|
| `--dry-run` | No Vertex calls; no DB writes |
| `--verbose` | Extra per-job counts on stderr |
| `--dump-prompts DIR` | Write prompts for the first document only (debug) |
| `--bulk-batch-size N` | Flush MongoDB writes every N upserts (overrides `BULK_WRITE_BATCH_SIZE`) |

