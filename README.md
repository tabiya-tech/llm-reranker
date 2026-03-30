# LLM re-ranker

Batch pipeline over classified job documents in MongoDB: Vertex AI (Gemini) reranks ESCO skills (essential vs optional) and infers job attributes plus ONET work activities.

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | CLI and orchestration |
| `app/` | Config, MongoDB (read + target upsert), prompts, Vertex, schemas, logging |
| `utils/` | `classifier_linked_entities.py`, `uuid_to_tabiya_id.py` (ESCO ↔ Tabiya from `skills.csv`), `occupation_to_skills_mapping.py` (relations CSV → `occupation_to_skills`) |
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

Configure `.env`: `MONGODB_URI`, `MONGODB_DATABASE`, `MONGODB_SOURCE_COLLECTION`, `MONGODB_TARGET_COLLECTION`, optional `BULK_WRITE_BATCH_SIZE` (default `100`), optional `JOB_ATTRIBUTES_SCHEMA`, optional `CLASSIFIED_LINKED_ENTITY_TYPES` (default `skill,occupation`; linked taxonomy is always ESCO), Vertex (`GCP_PROJECT_ID`, `GCP_LOCATION`, `VERTEX_MODEL`, credentials). Set `LOG_LEVEL` as needed.

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

Target documents (see `app/ranked_job_schema.py`) store **`classified_skills`** / **`classified_occupations`** (with Tabiya ids where mapped), **`occupation_to_skills`** as `{ meta, items }` (relations CSV + `skills.csv` ESCO mapping, per-occupation `skill_counts`), **`llm_classified_skills`** (essential / optional / discarded from the ranker), **`llm_job_attributes`**, **`work_activity`**, **`classifier_metadata`** (incl. title), and **`llm_reranker_meta`** (`schema_version`).

