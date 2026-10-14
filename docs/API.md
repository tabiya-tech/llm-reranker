# LLM Re-ranker API

An HTTP service that ranks one classified job: send a document from the classifier's
`classified_jobs` collection, get back the ranked job document. It runs the same per-job
pipeline as the batch CLI (`python main.py`). The API does not embed the job and does not
write to MongoDB. See [How the batch CLI relates](#how-the-batch-cli-relates).

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The OpenAPI spec is at `/openapi.json`, with interactive docs at `/docs`. Every request
and response field below is also described there, with examples. The models are in
`app/api_models.py`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/rank` | Rank one classified job document |
| `GET` | `/health` | Readiness, plus which language packs are supported and loaded |

On Cloud Run the service deploys with authenticated invocation. Send an identity token as
shown in the main [README](../README.md#cloud-run-service-https-api).

## `POST /rank`

```bash
curl -X POST 'http://localhost:8000/rank?language=en' \
  -H 'Content-Type: application/json' \
  -d @classified.json
```

### Query parameters

| Name | Description |
|---|---|
| `language` | Optional. The language pack to rank against: taxonomy labels, prompts and job attributes. Accepts a code (`en`, `es`), a locale (`AR-es`, `es_AR`) or a name (`spanish`). Takes precedence over the document's `language` field. When both are missing, the service default applies (`RERANKER_LANGUAGE`, then `TARGET_COUNTRY`). An unknown value falls back to `en` and logs a warning. Each language's taxonomy loads on the first request that needs it, so that request is slower. |

### Request body

The body is the classified job document itself, not wrapped in another object. The
following fields are typed and validated. Any other field, such as salary, location,
dates, `metadata`, `translation` or the scraper enrichment fields, is accepted and passed
to the pipeline unchanged.

| Field | Type | Description |
|---|---|---|
| `_id` | string \| int | Source document id. Becomes `source_classified_id`. The batch also uses it as the upsert key when the document has no job id or fingerprint. |
| `job_id` | string \| int | Job id (`CLASSIFIED_MONGO_FIELD_JOB_ID`). |
| `job_fingerprint` | string | Listing fingerprint (`CLASSIFIED_KEYS_FINGERPRINT`). |
| `title`, `employer`, `description`, `requirements` | string | Listing text given to the LLMs. |
| `language` | string | Language pack, for when the `?language=` parameter is not sent. |
| `classification.entities[]` | object | Entities the classifier extracted. |
| `…entities[].entity_type` | string | `skill` links are ranked by LLM1. `occupation` links select the occupation-to-skill menu. |
| `…entities[].linked_entities[]` | object | `taxonomy` (only `esco` is used), `label`, `uri` (the ESCO UUID is its last path segment), `similarity_score`. |

Deployments can rename the listing fields (`CLASSIFIED_FIELD_*`, `CLASSIFIED_KEYS_*`).
The pipeline also falls back to `metadata.<field>`. That is why the listing fields are all
optional: a document with none of them is still ranked, from whatever text it has.
Without `classification`, LLM1 has nothing to rank and there is no occupation menu, but
LLM2 still runs.

**MongoDB Extended JSON** is accepted, so a document copied from Compass, Atlas or
`mongoexport` can be sent as is. `{"$oid": …}` and `{"$date": …}` become strings, and
`{"$numberInt"|"$numberLong"|"$numberDouble": …}` become numbers. These values are
converted when the request is validated, so the pipeline never sees Extended JSON.

### What runs

1. **Taxonomy.** The classifier's ESCO skill links become LLM1's candidates. Its
   occupation links are looked up in the occupation-to-skill relations, which gives
   `occupation_to_skills` and the **occupation-skill menu** (essential and optional
   skills, capped and filtered by `RERANKER_LLM3_*`).
2. **LLM3.** What happens to the menu depends on `RERANKER_ENABLE_LLM3` and
   `RERANKER_LLM3_MODE`, and is reported as `llm3_outcome`:

   | `llm3_outcome` | When | Effect |
   |---|---|---|
   | `disabled` | `RERANKER_ENABLE_LLM3=0` | The menu is not used |
   | `no_menu` | The job's occupations have no essential or optional skills | Nothing to do |
   | `menu_filtered_out` | The `RERANKER_LLM3_*` filters emptied the menu | Nothing to do |
   | `folded` | `RERANKER_LLM3_MODE=fold` (default) | The menu skills are added to LLM1's candidates. No LLM3 call is made. |
   | `called` | `RERANKER_LLM3_MODE=call` | A standalone LLM3 call judges each menu skill. Its verdict is stored as `llm_is_relevant` on `occupation_to_skills`. |

3. **LLM1 (skill ranker).** Ranks the candidates as essential, optional or discarded. It
   is skipped when there are no candidates. `RERANKER_OVERLAP_LLM1_LLM2=1` runs LLM1 and
   LLM2 concurrently.
4. **LLM2 (job attributes).** Produces the attribute estimates, a summary and the ONET
   work activities.
5. **Assembly.** The results are built into the ranked job document.

### Response — `200`

| Field | Type | Description |
|---|---|---|
| `ranked_document` | object \| null | The ranked job (below). It is `null` only if LLM2 returned no attributes. |
| `timings` | object | Seconds per step that ran: `taxonomy_s`, `llm1_s`, `llm2_s`, `llm3_s` (only when the LLM3 call ran) and `total_s`. |
| `ran_llm1` | bool | LLM1 was called. This is false when the job has no ESCO skill links. |
| `ran_llm2` | bool | LLM2 was called. |
| `ran_llm3` | bool | The standalone LLM3 call was made. This can only be true in `call` mode. When the menu was folded into LLM1, it is false. |
| `llm3_outcome` | string | What the LLM3 step did (see the table above). |

`ranked_document` is the document the batch writes to the target collection, without
the embedding fields:

| Field | Description |
|---|---|
| `job_id`, `job_fingerprint`, `source_classified_id` | Identifiers from the source document |
| `is_active` | Always `true` at ranking time |
| `processed_at` | When the job was ranked (UTC, ISO 8601) |
| `classified_skills`, `classified_occupations` | The classifier's ESCO links after deduplication, with Tabiya ids where the CSVs map them |
| `occupation_to_skills` | `{meta, items}`: one block per classified occupation with its taxonomy skills and `skill_counts`. `llm_is_relevant` and the per-block `relevant_total` are present only in `call` mode. |
| `work_activity.items[]` | ONET work activities chosen by LLM2 (`id`, `name`) |
| `skill_groups_origin_uuids` | Skill-group ids derived from the essential and optional skills |
| `llm_classified_skills` | LLM1's ranking: `essential`, `optional`, `discarded`, plus `unmatched_*` labels that matched no candidate |
| `llm_job_attributes` | LLM2's `summary`, its `estimates`, and an `attributes` map from attribute name to level |
| `classifier_metadata` | Listing fields copied from the source document, plus any scraper enrichment fields (`province`, `sector`, `isco_*`, `zqf_*`) |
| `llm_reranker_meta` | `schema_version`, `llm_provider`, and the model behind each pass. `occupation_skills_relevance_model` is recorded even when no LLM3 call ran, because the matching service reads this shape. Use `llm3_outcome` to tell whether the call happened. |

Optional keys the pipeline did not write are left out of the response rather than sent as
`null`.

### Errors

| Status | When | Body |
|---|---|---|
| `422` | The body is not a JSON object, or a typed field has the wrong shape (for example `classification.entities` is not a list) | FastAPI's validation error: `detail[]` with `loc`, `msg`, `type` |
| `500` | The pipeline failed, usually because an LLM call kept failing after its retries (`RERANKER_LLM_MAX_ATTEMPTS`) | `{"detail": "<error message>"}` |
| `503` | The service is still starting up and the taxonomy is not loaded | `{"detail": "Service not ready — taxonomy not loaded."}` |

A misconfigured model (unknown provider, missing key or model id) fails at **startup**,
not on a request: the container fails its startup probe.

## `GET /health`

```json
{
  "status": "ok",
  "taxonomy_loaded": true,
  "default_language": "en",
  "supported_languages": ["en", "es"],
  "loaded_languages": ["en"]
}
```

## How the batch CLI relates

`python main.py` and `POST /rank` call the same function for each job:
`app.pipeline.process_single_job`. Given the same document and configuration, both run
the same steps, make the same LLM calls with the same prompts, and build the same ranked
document. The two differ only in what happens around that call:

| | API | Batch CLI |
|---|---|---|
| Input | One document in the request body | Documents streamed from `MONGODB_SOURCE_COLLECTION` |
| Per-job pipeline | `process_single_job` | `process_single_job`, via `app/batch_rank_one.py` |
| Per-step log lines | — | One line per step, with counts and timings |
| `--log-llm{1,2,3}-prompt` | — | Logs each assembled prompt |
| Job embedding (`RERANKER_ENABLE_JOB_EMBEDDING`) | Not computed | Adds `job_embedding` (plus `raw_embedding` when whitening is on) and `llm_reranker_meta.embedding` |
| Output | The JSON response | Upserted into `MONGODB_TARGET_COLLECTION` |
| Afterwards | — | Run log in `ranking-logs`, work-activity enrichment, `is_active` sync |

Every setting that changes the ranking applies to both: the model (`RERANKER_LLM_*`),
the language pack, the taxonomy filters, and `RERANKER_ENABLE_LLM3` /
`RERANKER_LLM3_MODE`. A document ranked through the API therefore matches what the
batch would write for it, apart from the embedding fields and `processed_at`.
