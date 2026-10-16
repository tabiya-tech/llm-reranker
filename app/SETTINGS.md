# Settings reference

`app/config.py` defines `Settings`, a [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
model that reads every environment variable the reranker depends on. Each row below maps
one env var to its field with its type and default. Where the Field column names
something that isn't a 1:1 match for its env var (e.g. `language` for
`RERANKER_LANGUAGE` / `TARGET_LANGUAGE`, or any `*_path`), that's a `@cached_property`
derived from one or more raw fields rather than a direct binding.

Build it with `app.config.get_settings()` (or `get_settings_for_language(language)`),
which cache the result. A bad value — an out-of-range float, an unrecognised enum — fails
with a `pydantic.ValidationError` as soon as `Settings` is built, so a misconfigured
deployment fails to start instead of silently mis-ranking jobs.

`.env` is loaded once at import, without overriding variables already exported in the
shell — so CI/Cloud Run/a developer's `export FOO=` always wins over `.env`.

See `.env.example` for a copy-pasteable `.env`; this document is the authoritative list of
defaults and types.

## Mongo / job retrieval

| Env var | Field | Type | Default |
|---|---|---|---|
| `MONGODB_URI` | `mongodb_uri` | str | `mongodb://localhost:27017` |
| `MONGODB_DATABASE` | `mongodb_database` | str | `ZambiaJobs` |
| `MONGODB_SOURCE_COLLECTION` | `mongodb_source_collection` | str | `classified_jobs` |
| `MONGODB_TARGET_COLLECTION` | `mongodb_target_collection` | str | `ranked_jobs` |
| `MONGODB_TLS_CA_FILE` | `mongodb_tls_ca_file` | str \| none | unset |
| `MONGODB_TLS_INSECURE` | `mongodb_tls_insecure` | bool | `false` |
| `RERANKER_LOGS_COLLECTION` | `reranker_logs_collection` | str | `reranker_logs` |
| `RERANKER_STATE_COLLECTION` | `reranker_state_collection` | str | `reranker_state` |
| `BULK_WRITE_BATCH_SIZE` | `bulk_write_batch_size` | int, range `[1, 10000]` | `100` |

## Classified job field / key names

Field-name mapping for reading the classifier's `classified_jobs` documents. Comma lists
are first-match-wins.

| Env var | Field | Type | Default |
|---|---|---|---|
| `CLASSIFIED_MONGO_FIELD_JOB_ID` | `mongo_field_job_id` | str | `job_id` |
| `CLASSIFIED_MONGO_FIELD_JOB_FINGERPRINT` | `mongo_field_job_fingerprint` | str | `job_fingerprint` |
| `CLASSIFIED_KEYS_DESCRIPTION` | `keys_description` | comma list | `description,job_description,body,text,content,summary,details,snippet,html` |
| `CLASSIFIED_KEYS_REQUIREMENTS` | `keys_requirements` | comma list | `requirements,job_requirements,qualifications` |
| `CLASSIFIED_KEYS_SALARY` | `keys_salary` | comma list | `salary_text,salary,compensation,pay` |
| `CLASSIFIED_KEYS_LOCATION` | `keys_location` | comma list | `city,location,town` |
| `CLASSIFIED_KEYS_COUNTY` | `keys_county` | comma list | `province,county,district,region` |
| `CLASSIFIED_KEYS_EMPLOYMENT` | `keys_employment` | comma list | `employment_type,employmentType,contract_type` |
| `CLASSIFIED_KEYS_FINGERPRINT` | `keys_fingerprint` | comma list | `job_fingerprint,jobFingerprint` |
| `CLASSIFIED_FIELD_TITLE` | `field_title` | str | `title` |
| `CLASSIFIED_FIELD_EMPLOYER` | `field_employer` | str | `employer` |
| `CLASSIFIED_FIELD_CATEGORY` | `field_category` | str | `category` |
| `CLASSIFIED_FIELD_APPLICATION_URL` | `field_application_url` | str | `application_url` |
| `CLASSIFIED_FIELD_POSTED_DATE` | `field_posted_date` | str | `posted_date` |
| `CLASSIFIED_FIELD_CLOSING_DATE` | `field_closing_date` | str | `closing_date` |
| `CLASSIFIED_FIELD_SOURCE_PLATFORM` | `field_source_platform` | str | `source_platform` |

## The backing LLM

Which model runs the three grading/enrichment passes. Everything here is optional: with
none of it set, the reranker uses Gemini exactly as before, from `GEMINI_API_KEY` /
`GEMINI_MODEL`. The resolved selection is `settings.llm` (an `app.llm.base.LlmConfig`) —
read `settings.llm.provider` / `.model` rather than the raw fields below.

| Env var | Field | Type | Default |
|---|---|---|---|
| `RERANKER_LLM_PROVIDER` | `reranker_llm_provider` | str (shipped alias) | unset (→ gemini) |
| `RERANKER_LLM_CLIENT_CLASS` | `reranker_llm_client_class` | str (dotted path) | unset |
| `RERANKER_LLM_MODEL` | `reranker_llm_model` | str | unset |
| `RERANKER_LLM_API_KEY` | `reranker_llm_api_key` | str | unset (falls back per-provider) |
| `RERANKER_LLM_BASE_URL` | `reranker_llm_base_url` | str | unset (falls back per-provider) |
| `RERANKER_LLM_TEMPERATURE` | `reranker_llm_temperature` | float, range `[0, 2]` | `0.2` |
| `RERANKER_LLM_MAX_OUTPUT_TOKENS` | `reranker_llm_max_output_tokens` | int, range `[0, 1000000]` | `0` (provider default) |
| `RERANKER_LLM_TIMEOUT_S` | `reranker_llm_timeout_s` | float, range `[0, 3600]` | `0.0` (provider default) |
| `RERANKER_LLM_MAX_ATTEMPTS` | `reranker_llm_max_attempts` | int, clamped to `[1, 12]` | unset (→ `GEMINI_TRANSIENT_MAX_ATTEMPTS` → `5`) |
| `GEMINI_TRANSIENT_MAX_ATTEMPTS` | `gemini_transient_max_attempts` | int, clamped to `[1, 12]` | historical alias for `RERANKER_LLM_MAX_ATTEMPTS` |
| `RERANKER_LLM_OPTIONS` | `reranker_llm_options` | JSON object string | `""` (no extras) |

`max_attempts` is the one field that's **clamped** rather than rejected when out of range
(`0`/negative → `1`, `99` → `12`), to match pre-refactor behaviour.

### Gemini specifics

| Env var | Field | Type | Default |
|---|---|---|---|
| `GEMINI_API_KEY` | `gemini_api_key` | str | `""` |
| `API_KEY` | — | str | legacy alias for `GEMINI_API_KEY` — see [Legacy aliases](#legacy-aliases) |
| `GEMINI_MODEL` | `gemini_model` | str | `gemini-2.5-flash` |
| `VERTEX_MODEL` | — | str | legacy alias for `GEMINI_MODEL` — see [Legacy aliases](#legacy-aliases) |

`gemini_api_key` is read directly by the job-embedding step, which always uses Gemini
regardless of `RERANKER_LLM_PROVIDER`.

## LLM3: the occupation-skills menu

| Env var | Field | Type | Default |
|---|---|---|---|
| `RERANKER_ENABLE_LLM3` | `reranker_enable_llm3` | bool | `true` |
| `RERANKER_LLM3_MODE` | `reranker_llm3_mode` | `"fold"` \| `"call"` | `fold` |
| `RERANKER_LLM3_MAX_MENU_SKILLS` | `reranker_llm3_max_menu_skills` | int, range `[0, 50000]` | `0` (no cap) |
| `RERANKER_LLM3_CLASSIFIER_INTERSECT` | `reranker_llm3_classifier_intersect` | bool | `false` |
| `RERANKER_LLM3_MENU_SIGNAL_FILTER` | `reranker_llm3_menu_signal_filter` | `"off"` \| `"high"` \| `"medium"` \| `"low"` | `off` |
| `RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE` | `reranker_occupation_skills_signal_include` | comma subset of `high,medium,low,unlabelled` → `frozenset \| none` | `none` (all) |

`RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE` applies independently of
`RERANKER_ENABLE_LLM3` — filtering applies even when LLM3 is off.

## Occupation ↔ skill relations

| Env var | Field | Type | Default |
|---|---|---|---|
| `OCCUPATION_RELATION_LOOKUP` | `occupation_relation_lookup` | `"tabiya_bridge"` \| `"esco_uuid_bridge"` | `tabiya_bridge` |
| `OCCUPATION_RELATION_ESCO_FALLBACK` | `occupation_relation_esco_fallback` | bool | `true` |
| `OCCUPATION_SKILL_RELATION_OCCUPATION_TYPES` | `occupation_skill_relation_occupation_types` | comma list → `frozenset \| none` | `none` (all types) |

`OCCUPATION_RELATION_LOOKUP` also accepts `local`/`tabiya` and `esco`/`esco_uuid`/`esco_bridge`
as aliases; an unrecognised value falls back to `tabiya_bridge`.

## Batch CLI / job embedding / concat whitening

| Env var | Field | Type | Default |
|---|---|---|---|
| `RERANKER_PARALLEL_WORKERS` | `reranker_parallel_workers` | int, range `[1, 32]` | `1` |
| `RERANKER_OVERLAP_LLM1_LLM2` | `reranker_overlap_llm1_llm2` | bool | `false` |
| `RERANKER_SKIP_ALREADY_RANKED` | `reranker_skip_already_ranked` | bool | `false` (ORed with `main.py --skip-already-ranked`) |
| `RERANKER_ENABLE_JOB_EMBEDDING` | `reranker_enable_job_embedding` | bool | `true` |
| `RERANKER_FAIL_ON_JOB_EMBEDDING_ERROR` | `reranker_fail_on_job_embedding_error` | bool | `false` |
| `RERANKER_JOB_EMBEDDING_MODEL` | `reranker_job_embedding_model` | str | `gemini-embedding-001` |
| `RERANKER_JOB_EMBEDDING_DIM` | `reranker_job_embedding_dim` | int, range `[1, 10000]` | `3072` |
| `RERANKER_JOB_EMBEDDING_TASK_TYPE` | `reranker_job_embedding_task_type` | str | `SEMANTIC_SIMILARITY` |
| `RERANKER_JOB_EMBEDDING_L2_NORMALIZE` | `reranker_job_embedding_l2_normalize` | bool | `true` |
| `RERANKER_JOB_EMBEDDING_BATCH_SIZE` | `reranker_job_embedding_batch_size` | int, range `[1, 10000]` | `100` |
| `RERANKER_JOB_EMBEDDING_SLEEP_S` | `reranker_job_embedding_sleep_s` | float, range `[0, 3600]` | `0.12` |
| `RERANKER_JOB_EMBEDDING_SOURCE` | `reranker_job_embedding_source` | str \| none | `llm_classified_skills` |
| `RERANKER_CONCAT_WHITENING_ENABLED` | `reranker_concat_whitening_enabled` | bool | `false` |
| `RERANKER_CONCAT_WHITENING_PATH` | `reranker_concat_whitening_path` | path | `resources/concat_whitening_gemini.npz` |
| `RERANKER_CONCAT_WHITENING_REQUIRED` | `reranker_concat_whitening_required` | bool | `true` |

## Classifier linked entities

| Env var | Field | Type | Default |
|---|---|---|---|
| `CLASSIFIED_LINKED_ENTITY_TYPES` | `classified_linked_entity_types` | comma list → tuple | `(skill, occupation)` |
| `CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY` | `classified_linked_entity_min_similarity` | float, range `[0, 1]` \| none | `none` (no threshold) |
| `CLASSIFIER_LINKED_ENTITY_MIN_SIMILARITY` | — | float, range `[0, 1]` | legacy alias — see [Legacy aliases](#legacy-aliases) |
| `CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY_STRICT` | `classified_linked_entity_min_similarity_strict` | bool | `false` |
| `CLASSIFIED_LINKED_ENTITY_LABEL_DEDUPE_TYPES` | `classified_linked_entity_label_dedupe_types` | comma subset of `skill,occupation` → `frozenset \| none` | `none` (no label dedupe) |

A non-numeric `CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY` fails validation; a numeric value
outside `[0, 1]` is accepted but logged and ignored (falls back to "no threshold").

## Country / language

| Env var | Field | Type | Default |
|---|---|---|---|
| `TARGET_COUNTRY` | `target_country` | `zambia` \| `kenya` \| `south_africa` \| `argentina` | `zambia` |
| — | `country_name` | str | `Zambia` |
| `RERANKER_LANGUAGE` / `TARGET_LANGUAGE` | `language` | registered language code | derived from `target_country`, else `en` |
| — | `language_name` | str | `English` |
| — | `language_resources_dir` | path | `resources/en/` |

An unrecognised `TARGET_COUNTRY` falls back to `zambia` rather than failing a batch.

## Data files / resource resolution

| Env var | Field | Type | Default |
|---|---|---|---|
| `RESOURCES_DIR` | `resources_dir` | path | `resources/` |
| `JOB_ATTRIBUTES_SCHEMA` / `JOB_ATTRIBUTES_SCHEMA_FILE` | `job_attributes_schema_path` | path | `resources/<language>/job_attributes_reduced.json` |
| `OCCUPATION_TO_SKILL_RELATIONS_CSV` | `occupation_to_skill_relations_path` | path | `resources/<language>/occupation_to_skill_relations.csv` |
| `OCCUPATIONS_CSV` | `occupations_csv_path` | path | `resources/<language>/occupations.csv` |
| `OCCUPATIONS_CSV_LABEL_COLUMN` | `occupations_csv_label_column` | str | `PREFERREDLABEL` |
| `OCCUPATIONS_CSV_INCLUDE_DESCRIPTIONS` | `occupations_csv_include_descriptions` | bool | `false` |
| `OCCUPATIONS_CSV_DESCRIPTION_COLUMN` | `occupations_csv_description_column` | str | `""` |
| `SKILLS_CSV` | `skills_csv_path` | path | `resources/<language>/skills.csv` |
| `SKILL_HIERARCHY_CSV` | `skill_hierarchy_csv_path` | path | `resources/<language>/skill_hierarchy.csv` (falls back to a sibling repo checkout) |
| `SKILLS_CSV_INCLUDE_DESCRIPTIONS` | `skills_csv_include_descriptions` | bool | `false` |
| `SKILLS_CSV_DESCRIPTION_COLUMN` | `skills_csv_description_column` | str | `""` |

A relative `*_path` value resolves against `resources/<language>/` → `resources/en/` →
`resources/`, then the repo root; an absolute path is used as given.

## Work-activities enrichment (`app/work_activities_enrichment/`)

Runs after every batch when `reranker_run_wa_enrichment` (below) is on.

| Env var | Field | Type | Default |
|---|---|---|---|
| `ENRICH_WA_DATABASE` | `enrich_wa_database` | str | `mongodb_database` |
| `ENRICH_WA_TARGET_COLLECTION` | `enrich_wa_target_collection` | str | `mongodb_target_collection` |
| `ENRICH_WA_OCCUPATION_DB_PATH` | `enrich_wa_occupation_db_path` | path | the run's language pack's WA database (falls back to English) |
| `ENRICH_WA_BULK_BATCH_SIZE` | `enrich_wa_bulk_batch_size` | int, range `[1, 100000]` | `500` |
| `ENRICH_WA_LIMIT` | `enrich_wa_limit` | int, `>= 0` | `0` (no limit) |
| `ENRICH_WA_DRY_RUN` | `enrich_wa_dry_run` | bool | `false` |
| `ENRICH_WA_ONLY_MISSING` | `enrich_wa_only_missing` | bool | `true` |
| `ENRICH_WA_DEFAULT_IMPORTANCE` | `enrich_wa_default_importance` | float | `3.5` |
| `ENRICH_WA_DEFAULT_LEVEL` | `enrich_wa_default_level` | float | `3.5` |

## Misc / post-batch steps

| Env var | Field | Type | Default |
|---|---|---|---|
| `RERANKER_RUN_WA_ENRICHMENT` | `reranker_run_wa_enrichment` | bool | `true` |
| `RERANKER_FAIL_ON_WA_ENRICHMENT_ERROR` | `reranker_fail_on_wa_enrichment_error` | bool | `false` |
| `RERANKER_SYNC_IS_ACTIVE_FROM_CLOSING_DATE` | `reranker_sync_is_active_from_closing_date` | bool | `true` |
| `LOG_LEVEL` | `log_level` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` \| `CRITICAL` | `INFO` |

Not on `Settings` (read live from the environment by the code that uses them):
`RERANKER_PREFER_TRANSLATION` (`app/translation.py`); `RERANKER_CUTOFF_DATE` and its
per-country override `<COUNTRY>_RERANKER_CUTOFF_DATE` (`app/batch/service.py`, e.g.
`KENYA_RERANKER_CUTOFF_DATE`) — the variable name is built from `TARGET_COUNTRY` at
runtime, so it can't be a static field; and Langfuse tracing (`RERANKER_ENABLE_TRACING`
and friends — see `app/observability/config.py`).

## Legacy aliases

| Variable | Alias for | Status |
|---|---|---|
| `API_KEY` | `GEMINI_API_KEY` | Kept for deployments that predate `GEMINI_API_KEY`. Remove on or after **2027-01-01**. |
| `VERTEX_MODEL` | `GEMINI_MODEL` | Kept, also used as a second model-name fallback. Remove on or after **2027-01-01**. |
| `CLASSIFIER_LINKED_ENTITY_MIN_SIMILARITY` | `CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY` | Kept, but not known to be used by any deployment. Remove on or after **2027-01-01**. |
| `GEMINI_TRANSIENT_MAX_ATTEMPTS` | `RERANKER_LLM_MAX_ATTEMPTS` | Kept, no removal date set. |
| `MONGODB_CLASSIFIED_COLLECTION` | `MONGODB_SOURCE_COLLECTION` | **Removed 2026-10-14.** |
| `LLM_RERANKER_RESOURCES_DIR` | `RESOURCES_DIR` | **Removed 2026-10-14.** |
| `MONGODB_DATABASE_REFERENCE` | `ENRICH_WA_DATABASE` | **Removed 2026-10-14.** Was a tertiary fallback in the pre-refactor script; now unreachable since `mongodb_database` always has a default. Set `ENRICH_WA_DATABASE` (or `MONGODB_DATABASE`). |
| `MONGODB_TARGET_COLLECTION_New` | `ENRICH_WA_TARGET_COLLECTION` | **Removed 2026-10-14.** Same pattern as above. Set `ENRICH_WA_TARGET_COLLECTION` (or `MONGODB_TARGET_COLLECTION`). |

## Removed (no readers)

These existed as `Settings` fields before this refactor but had no reader anywhere in the
codebase — Vertex AI isn't used by this service:

`vertex_model`, `gcp_location`, `gcp_project_id`, `google_application_credentials_resolved`,
and the env vars they read (`GCP_PROJECT_ID`, `GOOGLE_CLOUD_PROJECT`, `GCP_LOCATION`,
`GOOGLE_APPLICATION_CREDENTIALS`, `VERTEX_SERVICE_ACCOUNT_JSON`).

## Per-stack overrides

Deployed env vars live in GCP Secret Manager (Pulumi stack config, not GitHub
environments), injected into the Cloud Run Job/Service at deploy time
(`iac/infra/__main__.py`, `cloud_run_job.py`, `cloud_run_service.py`). Per-country and
per-language values (`TARGET_COUNTRY`, `MONGODB_DATABASE`, `RERANKER_LANGUAGE`) normally
arrive as per-execution overrides from the GCP Workflow instead — one deployed image
serves every country and language.

Every Cloud Run **Job** (batch) run sets these 8 variables from Pulumi config, falling
back to the value shown when the stack doesn't set its own:

| Variable | Pulumi key | Falls back to |
|---|---|---|
| `RERANKER_ENABLE_LLM3` | `enableLlm3` | `0` |
| `RERANKER_PARALLEL_WORKERS` | `parallelWorkers` | `4` |
| `RERANKER_OVERLAP_LLM1_LLM2` | `overlapLlm1Llm2` | `1` |
| `CLASSIFIED_LINKED_ENTITY_LABEL_DEDUPE_TYPES` | `classifiedLinkedEntityLabelDedupeTypes` | `occupation,skill` |
| `RERANKER_ENABLE_JOB_EMBEDDING` | `enableJobEmbedding` | `1` |
| `RERANKER_CONCAT_WHITENING_ENABLED` | `concatWhiteningEnabled` | `0` |
| `RERANKER_CONCAT_WHITENING_PATH` | `concatWhiteningPath` | `resources/concat_whitening_gemini.npz` |
| `RERANKER_CONCAT_WHITENING_REQUIRED` | `concatWhiteningRequired` | `1` |

Note the Job's fallback for `RERANKER_ENABLE_LLM3` (`0`) and `RERANKER_OVERLAP_LLM1_LLM2`
(`1`) both differ from the code default above — every deployed batch job runs with LLM3
off and LLM1/LLM2 overlap on unless a stack's Pulumi config says otherwise. The Job also
always sets `MONGODB_URI` and `GEMINI_API_KEY` from Secret Manager. The **Service**
(`POST /rank`) only ever sets `GEMINI_API_KEY`.

Anything else — a non-Gemini `RERANKER_LLM_PROVIDER`, Langfuse tracing credentials, a
country/language pin — goes through a stack's freeform `extraEnv` / `extraSecrets` Pulumi
config maps instead of a dedicated key.

To see what a stack actually overrides (requires Secret Manager access):

```bash
python iac/scripts/configure.py --stack <stack> --project <project-id> --list
```

`iac/templates/stack_config.infra.template.yaml` documents every settable key.
