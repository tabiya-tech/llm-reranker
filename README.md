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

## Deployment

| Mode | Entrypoint | GCP |
|------|-----------|-----|
| HTTP (single job) | `uvicorn api:app --host 0.0.0.0 --port 8000` | Cloud Run Service |
| Batch (full collection) | `python main.py` | Cloud Run Job |

Both share the same Docker image with different entrypoint commands.

## Run logs

Each batch run writes a document to the `ranking-logs` collection (created automatically) tracking status, config, stats, and timing. Progress is updated every 10 jobs. Failed runs include the error message.
