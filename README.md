# LLM Re-ranker

Batch pipeline that reads classified job documents from MongoDB, runs Google Gemini to enrich and rank skills and attributes, and upserts structured results into a target collection.

## What it does

1. **Skill ranker** : Classifier ESCO skills are ranked as essential, optional, or discarded for the listing.
2. **Occupation–skills relevance** : Taxonomy skills linked to occupations are scored for relevance to this specific job ad (`llm_is_relevant` and rollups in stored data).
3. **Job attributes** : Structured attribute estimates and ONET work activities from your job-attribute schema.


## Repository layout

| Path | Purpose |
|------|---------|
| `main.py` | CLI entrypoint and orchestration |
| `app/` | Settings, MongoDB access, prompts, Gemini client, schemas, logging |
| `utils/` | Classifier entities, ESCO ↔ Tabiya mapping, occupation–skill relations from CSV |
| `resources/` | Prompt text, job-attribute JSON schema, ONET reference data |

## Prerequisites

- Python 3.11+
- MongoDB
- A [Gemini API key](https://aistudio.google.com/apikey) (`GEMINI_API_KEY`)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with at least: `MONGODB_URI`, `MONGODB_DATABASE`, `MONGODB_SOURCE_COLLECTION`, `MONGODB_TARGET_COLLECTION`, and `GEMINI_API_KEY`. Set `GEMINI_MODEL` (for example `gemini-3-pro-preview` or `gemini-2.5-flash`). Optional variables cover bulk write batch size, CSV paths, job-attribute schema file, classifier entity types, occupation CSV columns for LLM2 context, and `LOG_LEVEL`.

## Usage

```bash
python main.py
```

| Flag | Effect |
|------|--------|
| `--limit N` | Process at most `N` source documents. Default: `0` (entire collection, streamed newest `_id` first). |
| `--verbose` | Extra per-job detail |
| `--log-llm1-prompt` / `--log-llm2-prompt` / `--log-llm3-prompt` | Log assembled prompts for that step (useful for debugging) |
| `--bulk-batch-size N` | Flush MongoDB bulk writes every `N` upserts |

Ranked documents are built in `app/ranked_job_schema.py` (see `PIPELINE_SCHEMA_VERSION` in metadata). They include classifier skills and occupations, `occupation_to_skills` as `{ meta, items }` with per-occupation skill counts and listing relevance rollups, LLM-ranked skills, job attributes, work activities, and classifier metadata.
