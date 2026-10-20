# LLM Re-ranker

Batch pipeline and HTTP API that processes classified job documents from MongoDB through Google Gemini, producing ranked skill profiles, job attributes, and ONET work activities.

## Pipeline

One per-job pipeline, `JobRankingService.rank` (`app/job_ranking/service.py`), used by both the API and the batch CLI:

1. **Taxonomy** — The classifier's ESCO skill links become skill candidates. Its occupation links are looked up in the occupation-to-skill relations, which gives `occupation_to_skills` and an **occupation-skill menu**.
2. **LLM 3: occupation–skill menu** — Controlled by `RERANKER_ENABLE_LLM3` and `RERANKER_LLM3_MODE`. In `fold` mode (the default) the menu skills are added to LLM 1's candidates and no LLM 3 call is made. In `call` mode a standalone LLM 3 call judges each menu skill's relevance to the job ad.
3. **Skill ranker (LLM 1)** — Ranks the candidates as essential, optional or discarded.
4. **Job attributes (LLM 2)** — Structured attribute estimates and ONET work activities.

In the default configuration, each job makes two LLM calls: LLM 1 and LLM 2.


## Architecture

The code is organized **one module per use case**, as in compass's backend. A module holds
everything its use case needs, and each file's name gives its role:

| File | Role |
|---|---|
| `routes.py` / `routes/` | HTTP only: parse the request, call the service, shape the response (`routes/types.py`) |
| `service.py`, `*_service.py` | Orchestration. Receives its repositories, clients and steps through its constructor |
| `step.py`, `*_step.py` | One stage of the ranking pipeline (a reusable class run by a service) |
| `repository.py`, `mongo_repository.py`, `*_repository.py` | All IO: one MongoDB collection, or one kind of resource file |
| `types.py`, `prompt.py`, `parser.py`, `rules.py`, `menu.py` | Pure: data types, prompt building, response parsing, business rules — no IO |
| `get_<x>_service.py` | The provider that builds the module's service and its dependencies (the FastAPI dependency for routes) |

Dependencies point one way: routes → services / steps → repositories. The pure files can be used
from any of them. Services never import routes or providers, repositories never import services,
and pure files import no IO library. Only the `get_*` providers construct an `LlmClient` or an
`EmbeddingClient`; everything else receives instances. `tests/test_architecture.py` checks these
rules.

### Modules

```
api.py                               HTTP entrypoint: mounts the routes, warms the ranking service
main.py                              Batch entrypoint: parses arguments, calls BatchRankingService.run
scripts/                             Local-only tools (whitening backfill); not in the image
app/
├── job_ranking/                     Rank one classified job (used by the API and the batch)
│   ├── routes/                      POST /rank, GET /health; request/response models (types.py)
│   ├── get_job_ranking_service.py   provider: the per-language JobRankingServiceFactory (the API's dependency)
│   ├── service.py                   JobRankingService (the pipeline) + JobRankingServiceFactory (one per language)
│   ├── types.py                     JobResult, JobStats, Llm3Outcome, PromptLogging
│   ├── skill_ranker/                LLM 1: prompt.py, parser.py, step.py, types.py
│   ├── job_attributes/              LLM 2: prompt.py, parser.py, step.py, types.py, repository.py (schema, ONET)
│   ├── occupation_skills/           taxonomy_step.py, menu.py (LLM 3 menu), LLM 3 prompt / parser / relevance_step
│   ├── prompts/                     PromptTemplates, PromptRepository (prompt files), shared job-ad text
│   └── ranked_document.py           RankedJobDocumentAssembler: the ranked job document
├── taxonomy/                        Per-language taxonomy: CSV loaders, TaxonomyRepository, TaxonomyService (cache)
├── batch/                           A batch run: BatchRankingService, BatchJobRanker, BatchPostSteps, provider
├── job_embedding/                   JobEmbeddingService, EmbeddingClient / GeminiEmbeddingClient, ConcatWhitener
├── work_activities_enrichment/      onet_work_activities post-step: service, rules, config, repository, provider, cli.py
├── classified_jobs/                 source collection repository
├── ranked_jobs/                     target collection repository, upsert builder, is_active rules + is_active_cli.py
├── run_logs/, run_state/            batch run log and cutoff state repositories
├── llm/                             LlmClient, providers, factory, TracedLlmClient
├── observability/                   Langfuse tracing
├── server_dependencies/             the MongoDB connection provider and collection names
├── languages/                       language registry, one config module per language
└── config.py, job_context.py, translation.py, logging_config.py
utils/                               pure helpers: classifier entities, ESCO–Tabiya mapping
resources/<lang>/, resources/        language packs (taxonomy CSVs, prompts, job attributes); whitening artifact
```

### Reusable classes

| Concern | Classes | Module |
|---|---|---|
| Ranking steps | `SkillRankerStep`, `JobAttributesStep`, `OccupationTaxonomyStep`, `OccupationSkillsRelevanceStep` | `app/job_ranking/<pass>/…step.py` |
| Prompt building | `SkillRankerPromptBuilder`, `JobAttributesPromptBuilder`, `OccupationSkillsRelevancePromptBuilder` | `app/job_ranking/<pass>/prompt.py` |
| Response parsing | `SkillRankerResponseParser`, `JobAttributesResponseParser`, `OccupationSkillsRelevanceResponseParser` | `app/job_ranking/<pass>/parser.py` |
| Ranked document | `RankedJobDocumentAssembler` | `app/job_ranking/ranked_document.py` |
| LLM | `LlmClient` (one per provider), wrapped by `TracedLlmClient` | `app/llm/` |
| Embeddings | `EmbeddingClient`, `GeminiEmbeddingClient`, `ConcatWhitener` | `app/job_embedding/` |

Each LLM step takes a `TracedLlmClient`, a prompt builder and a parser. The same step therefore
runs on any configured model, and a test can swap one step for a stub (see
`tests/ranking_fakes.py`).

### Where new code goes

| You are adding… | Put it in… |
|---|---|
| A new use case | A new module `app/<use_case>/` with its `routes.py`, `service.py`, `repository.py`, `types.py` and `get_<x>_service.py`; register its routes in `api.py` |
| An HTTP endpoint for ranking | `app/job_ranking/routes/`, taking `JobRankingServiceFactory` from `get_job_ranking_services` |
| A pipeline stage | A new sub-module of `app/job_ranking/` (prompt, parser, step, types), run by `JobRankingService` and wired in `JobRankingServiceFactory` |
| A prompt change | The language pack under `resources/<lang>/`, or the pass's `prompt.py` |
| A field on the ranked document | `app/job_ranking/ranked_document.py` (and `routes/types.py`, `PIPELINE_SCHEMA_VERSION`) |
| A batch flag | `main.py` (argument) → `BatchOptions` (`app/batch/service.py`) |
| A post-step after the batch | A use-case module for the work, called from `BatchPostSteps` (`app/batch/post_steps.py`) |
| A MongoDB query | The collection module's repository interface and Mongo implementation; see [Data access](#data-access) |
| A new LLM provider | `app/llm/providers/`; see [Choosing the model](#choosing-the-model) |
| A maintenance CLI | A `cli.py` (or `<x>_cli.py`) in the use-case module, run with `python -m app.<module>.cli`: parse arguments, build through the module's provider, call its service. It ships in the image; top-level `scripts/` is git-ignored and local-only |

## Prerequisites

- Python 3.11+
- MongoDB
- An API key for whichever model backs the reranker — a [Gemini API key](https://aistudio.google.com/apikey) by default (see [Choosing the model](#choosing-the-model))

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with: `MONGODB_URI`, `MONGODB_DATABASE`, `MONGODB_SOURCE_COLLECTION`, `MONGODB_TARGET_COLLECTION`, and `GEMINI_API_KEY`. See `.env.example` for all available options, and [`app/SETTINGS.md`](app/SETTINGS.md) for the full settings reference: every field's type and default, and which variables each deployment stack overrides.

Every variable is validated by `app.config.Settings` (a [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) model), built once on the first `get_settings()` call (at the start of `main.py` / `api.py`'s startup) — a bad value (an out-of-range float, an unrecognised enum) fails that call with a `ValidationError` rather than silently misconfiguring a batch run.

## Choosing the model

The LLM passes talk to a model only through `LlmClient` (`app/llm/base.py`), and configuration decides which implementation answers. Nothing in the grading or enrichment code — the prompts, parsers and steps in `app/job_ranking/` — knows which provider is in use: the services receive one `TracedLlmClient` built by the module's `get_*` provider, so swapping models is an env change.

**Default (unchanged).** Set nothing and the reranker runs on Gemini from `GEMINI_API_KEY` / `GEMINI_MODEL`, with the same temperature, JSON mode, and retry policy it always had.

**Pick a shipped provider** with `RERANKER_LLM_PROVIDER`:

| Alias | Backend | Needs |
|---|---|---|
| `gemini` | Google GenAI SDK | `GEMINI_API_KEY`; model defaults to `gemini-2.5-flash` |
| `anthropic` | Claude Messages API | `ANTHROPIC_API_KEY`, `RERANKER_LLM_MODEL` |
| `openai` | OpenAI Chat Completions | `OPENAI_API_KEY`, `RERANKER_LLM_MODEL` |
| `qwen` | Qwen on Alibaba DashScope (compatible mode) | `DASHSCOPE_API_KEY`, `RERANKER_LLM_MODEL` |
| `ollama` | Ollama's native `/api/chat` — any pulled model (Qwen, Llama, Mistral, …) | `RERANKER_LLM_MODEL`; endpoint defaults to `http://localhost:11434` |
| `openai_compatible` | Any other server speaking Chat Completions — vLLM, SGLang, LM Studio, llama.cpp, Together, Groq | `RERANKER_LLM_BASE_URL`, `RERANKER_LLM_MODEL`; API key optional |

```bash
# Claude
RERANKER_LLM_PROVIDER=anthropic
RERANKER_LLM_MODEL=claude-opus-5
ANTHROPIC_API_KEY=sk-ant-...

# Hosted Qwen (DashScope)
RERANKER_LLM_PROVIDER=qwen
RERANKER_LLM_MODEL=qwen-plus
DASHSCOPE_API_KEY=sk-...

# Qwen on a local Ollama — `ollama pull qwen3:32b` first
RERANKER_LLM_PROVIDER=ollama
RERANKER_LLM_MODEL=qwen3:32b
# RERANKER_LLM_BASE_URL=http://ollama.internal:11434   # or OLLAMA_HOST, if already set

# Qwen on a local vLLM / SGLang
RERANKER_LLM_PROVIDER=openai_compatible
RERANKER_LLM_BASE_URL=http://localhost:8000/v1
RERANKER_LLM_MODEL=Qwen/Qwen3-32B
```

Every provider except Gemini needs no extra dependency — they use the documented REST APIs over `httpx`, which is already installed. See `.env.example` for the tuning knobs (`RERANKER_LLM_TEMPERATURE`, `RERANKER_LLM_MAX_OUTPUT_TOKENS`, `RERANKER_LLM_TIMEOUT_S`, `RERANKER_LLM_MAX_ATTEMPTS`) and `RERANKER_LLM_OPTIONS`, a JSON object merged into the provider's request body for anything not covered by those.

Two provider-specific behaviours are worth knowing. **Ollama** is addressed by its native API, so `RERANKER_LLM_OPTIONS` carries Ollama's runtime options (`{"num_ctx": 8192, "top_p": 0.9}`) into the request's `options` block, `{"format": {…}}` swaps free-form JSON for a constrained schema decode, and the default timeout is 600 s because a cold local model spends the first minutes loading weights; a `/v1` suffix on the base URL is stripped, so pasting the OpenAI-compat URL still works. **Qwen** on DashScope sends `enable_thinking=false`: Qwen3 hybrid models think by default and DashScope serves thinking only over streaming, which this non-streaming call is not — override with `RERANKER_LLM_OPTIONS={"enable_thinking": true}`.

**Add a provider** by implementing one method and pointing config at the class — no registration, no edit to `app/llm/`:

```python
# my_pkg/my_model.py
from app.llm import LlmClient, LlmConfigurationError, LlmInvocationError

class MyLlmClient(LlmClient):
    def __init__(self, config):
        super().__init__(config)
        self.key = config.require_api_key(env_hint="MY_VENDOR_API_KEY")
        config.require_model()          # raises LlmConfigurationError if unset

    def generate_json(self, prompt: str, *, purpose: str = "") -> str:
        ...                             # return the model's raw text
```

```bash
RERANKER_LLM_CLIENT_CLASS=my_pkg.my_model:MyLlmClient
RERANKER_LLM_MODEL=my-model-v1
RERANKER_LLM_API_KEY=...
```

**Structured output.** Each pass also builds a JSON Schema for its reply (`app/job_ranking/response_schemas.py`), with the prompt's menus — candidate skill labels, attribute level ids, ONET ids — as enums, so the model can only answer with values it was offered. A client that sets `supports_response_schema = True` receives it as `generate_json(..., response_schema=...)`; Gemini sends it as `response_json_schema`. Clients that leave the flag unset keep the two-argument signature above and get the prose contract only. `RERANKER_LLM_RESPONSE_SCHEMA=false` withholds the schema everywhere.

The contract is in `app/llm/base.py`: return raw text (fenced ```json is tolerated — every parser strips fences), never return an empty string, raise `LlmConfigurationError` for anything a retry cannot fix and `LlmInvocationError` for a failed call. Clients are built once per process and shared across `RERANKER_PARALLEL_WORKERS` threads, so they must be thread-safe. `app/llm/retry.py` provides the transient-failure retry the shipped providers use. A third exception, `LlmParseError` (`app/llm/errors.py`), is raised by the reranker itself — not by providers — when a model responds but its JSON does not parse or match the expected shape; see [Errors and logging](#errors-and-logging).

**Misconfiguration fails at startup**, not one job at a time: both `main.py` and the API build the client before reading MongoDB, so an unknown provider, an unimportable class, a class that is not an `LlmClient`, a missing model id, or a missing key stops the run with a message naming the variable to fix.

**Not covered by this setting:** the job-embedding step (`RERANKER_JOB_EMBEDDING_MODEL`) stays on `gemini-embedding-001` and still needs `GEMINI_API_KEY` when enabled. It writes into match_v4's stage-1 vector space, whose whitening artifact was fitted on that model — a different embedding model would not be a swap but a re-fit across two repos.

The model that actually ran is recorded per document in `classifier_metadata` (`llm_provider`, `skill_ranker_model`, `job_attributes_model`, `occupation_skills_relevance_model`) and in each run log.

## API — single job

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**POST `/rank`** — send a classified job document directly as the JSON body. MongoDB
Extended JSON (`$oid`, `$date`) is accepted, so documents can be pasted from Compass or
Atlas. The response is the ranked job document, together with `ran_llm1`, `ran_llm2`,
`ran_llm3` and `llm3_outcome`, which report the steps that actually ran:

```bash
curl -X POST 'http://localhost:8000/rank?language=es' \
  -H "Content-Type: application/json" \
  -d '{ "_id": "...", "title": "...", "classification": { ... } }'
```

**GET `/health`** — readiness check. It also reports the default, supported and
currently loaded languages.

**[docs/API.md](docs/API.md)** covers the request and response fields, the errors, and
how the API relates to the batch CLI. In short, the batch runs the same per-job pipeline
and adds logging, the job embedding and the MongoDB upsert. The interactive OpenAPI docs
are at `http://localhost:8000/docs`.

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
| `--log-llm3-prompt` | Log occupation–skills relevance prompt (only `RERANKER_LLM3_MODE=call` makes that call) |
| `--bulk-batch-size N` | Flush MongoDB writes every N upserts |
| `--skip-already-ranked` | Skip classified jobs already in the target collection (also `RERANKER_SKIP_ALREADY_RANKED=1`) |
| `--language LANG` | Language pack to rank against (`en`, `es`, or a locale like `AR-es`). Default: from `TARGET_COUNTRY`. |

## Data access

Only the batch and the maintenance CLIs touch MongoDB; the API does not. One `MongoClient` per
process is owned by `RerankerDBProvider`: the batch opens it at the start of a run and closes it at
the end, and the CLIs open it with `standalone_db()`. Code reaches the four collections (source,
target, `reranker_logs`, `reranker_state`) only through the repositories, which receive the database
by injection.

See [`app/server_dependencies/README.md`](app/server_dependencies/README.md) for the connection
lifecycle, the collections, and how to add a query.

## Origin UUIDs, work activities, and job status

1. **`skill_groups_origin_uuids`** — After ranking, skill groups are derived from the skill hierarchy so each ranked job can carry the **origin UUIDs** of the relevant skill groups (see `.env` / `resources/` for the hierarchy CSV).
2. **`onet_work_activities`** — Model output includes work-activity context; after a successful batch, optional enrichment can fill or adjust **ONET-style work activities** (importance/level) using the occupation taxonomy reference JSON, when enabled in `.env` (`RERANKER_RUN_WA_ENRICHMENT` and related variables in `.env.example`).
3. **`is_active` and closing dates** — **`is_active`** can be synced from **`classifier_metadata.closing_date`** (open vs closed relative to today) after a successful batch when enabled in `.env`, or by running `python -m app.ranked_jobs.is_active_cli` manually (`--apply` to write). Work-activities enrichment can be run on its own with `python -m app.work_activities_enrichment.cli`. No documents are deleted; only the flag is updated.

## Countries and languages

`TARGET_COUNTRY` (`zambia` | `kenya` | `south_africa` | `argentina`) is the batch/state
key and picks the country name used in the prompts. `RERANKER_LANGUAGE` (`en` | `es`) picks
the **language pack**: the taxonomy label CSVs, the prompt files, and the job-attribute
schema. Everything else that varies is env-driven, so **one image serves every country and
every language** — the GCP Workflow supplies the differences as env overrides on each Cloud
Run Job execution.

Language resolution, most specific first:

1. `--language es` on the batch CLI, or `?language=es` on `POST /rank`
2. `RERANKER_LANGUAGE`, then `TARGET_LANGUAGE`
3. the language of `TARGET_COUNTRY` (`argentina` → `es`, everything else → `en`)
4. `en`

A locale spelling works anywhere a code does: `AR-es`, `es_AR`, `spanish`.

### Language packs

```
resources/
  concat_whitening_gemini.npz     language-neutral (embeddings, not text)
  en/                             English pack
    occupations.csv  skills.csv  occupation_to_skill_relations.csv  skill_hierarchy.csv
    prompts.py  Skills_reranker_prompts.md  job_attributes_reduced.json
    ONET_tasks_WAs_translated.json  combined_occupation_database_with_wa.json
  es/                             Argentine Spanish pack (Tabiya ESCO 1.1.1, locale AR-es)
    occupations.csv  skills.csv  occupation_to_skill_relations.csv  skill_hierarchy.csv
    prompts.py  Skills_reranker_prompts.md  job_attributes_reduced.json  model_info.csv
```

Each file is looked up in `resources/<language>/`, then `resources/en/`, then `resources/`.
A language therefore only ships what actually differs — `es` ships no ONET reference data
and is served English's, whose `WA_Element_ID` values are language-neutral anyway.

The Spanish taxonomy's ESCO relation graph is identical to English's; what changes is the
`PREFERREDLABEL` text LLM1's skill menu and LLM2's occupation line are built from.
`UUIDHISTORY`'s oldest entry is the same in both packs, which is what lets a
Spanish-linked skill resolve to the same entity downstream —
`tests/test_language_resources.py` asserts that the two packs share one skill id space,
along with presence, required columns, and the declared locale.

What is *not* translated, on purpose: the JSON response contract in the LLM2/LLM3 prompts.
Its keys (`attribute_name`, `selected_level_id`, …), the attribute `name`s, and the level
ids are identifiers the model must copy verbatim, and a translated contract invites
translated keys back. The human-facing framing around it is translated
(`prompt_labels` in `app/languages/<lang>_config.py`).

### Adding a language

1. Add the code to `LANGUAGE_REGISTRY` in `app/languages/__init__.py`.
2. Copy `app/languages/es_config.py` to `<code>_config.py`; set its locales,
   `taxonomy_locale`, `prompt_language` and `prompt_labels`.
3. Build the taxonomy half of the pack from a taxonomy CSV export:

```bash
python scripts/build_language_resources.py --taxonomy-dir <export-dir> --language <code>
```

   The script validates the columns each loader reads by name and reports which prompt
   files still need translating.

4. Translate `prompts.py`, `Skills_reranker_prompts.md` and `job_attributes_reduced.json`
   from `resources/en/`. Until you do, that language falls back to the English prompts —
   the model would be instructed in English over a non-English skill menu.
5. If the language belongs to a new country, add it to `LANGUAGE_BY_COUNTRY`.

### Argentina

Configuration is now just the database and the country:

```bash
TARGET_COUNTRY=argentina          # implies RERANKER_LANGUAGE=es
MONGODB_DATABASE=ArgentinaJobs_V2
MONGODB_SOURCE_COLLECTION=classified_jobs
```

The four `*_CSV` overrides that used to be required are not needed any more — the language
selects `resources/es/`. Set them only to pin something unusual, e.g. `SKILLS_CSV=en/skills.csv`
to rank Argentine jobs against English labels.

`RERANKER_PREFER_TRANSLATION` makes the prompts use `translation.text_en` /
`translation.fields_en` from `classified_jobs` when a document carries them. Unset, the
language decides: `en` prefers the English machine translation, `es` reads the original
Spanish, because prompting in Spanish over English MT text would mix the two. Set it
explicitly to `1` or `0` to override.

Note the job-attribute earnings levels: `resources/es/job_attributes_reduced.json` keeps the
English level **ids** (they are the contract with the matching service) and expresses the
labels as multiples of the Salario Mínimo Vital y Móvil rather than nominal ARS amounts,
which go stale with inflation. Replace them with ARS bands once Empujar supplies a local
calibration.

## LLM tracing (Langfuse)

Every model call the reranker makes can be traced to [Langfuse](https://langfuse.com). The
layer in `app/observability/` is the one the compass-connect backend runs, adapted to jobs
instead of conversations. It is **off by default**; a deployment with no Langfuse
credentials behaves exactly as before.

What is recorded, per ranked job:

| Observation | Type | Carries |
|---|---|---|
| `rank_job` | trace | job id, title, employer, platform; tags `invocation:`, `country:`, `language:`, `llm_provider:` |
| `skill_ranker`, `job_attributes`, `occupation_skills_relevance` (`RERANKER_LLM3_MODE=call` only) | generation | prompt, raw JSON output, model, temperature, token usage |
| `embed_content` | embedding | the text embedded, model, `task_type`, `output_dimensionality`, vector count/dim, token usage when reported |

Batch jobs from one run share a Langfuse **session**, named by the `reranker_logs` run `_id`.
Token usage comes from every shipped provider (Gemini counts thinking tokens as output).
The Gemini Developer API does not report embedding tokens, so embedding usage appears only
on Vertex AI. The vectors are never exported, only their shape.

Turn it on in `.env`:

```bash
RERANKER_ENABLE_TRACING=1
RERANKER_LANGFUSE_HOST=https://cloud.langfuse.com
RERANKER_LANGFUSE_PUBLIC_KEY=pk-lf-...
RERANKER_LANGFUSE_SECRET_KEY=sk-lf-...
RERANKER_TRACING_ENVIRONMENT=dev
RERANKER_TRACING_CONFIG={"batchSampleRate": 1.0, "apiSampleRate": 1.0, "maskPii": true, "maxPayloadChars": 100000}
```

Sampling is deterministic per job id, so a sampled job is always traced whole. `maskPii`
redacts e-mail addresses, phone numbers and long digit runs before export. It is a
heuristic: for job ads that must not leave your infrastructure, point the host at a
self-hosted Langfuse.

On the Cloud Run **Job**, wire the keys without an IaC change through the existing stack
config: `extraEnv` for `RERANKER_ENABLE_TRACING` / `RERANKER_LANGFUSE_HOST`, and
`extraSecrets` for the keys (see `iac/templates/stack_config.infra.template.yaml`).

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

Each batch run writes a document to the `reranker_logs` collection (`RERANKER_LOGS_COLLECTION`, created automatically) tracking status, config, stats, and timing. Progress is updated every 10 jobs. Failed runs include the error message.

## Errors and logging

### Error responses

Every error `/rank` returns has one body: `ErrorResponse` (`app/constants/errors.py`), `{"detail": "..."}`.
Routes declare their error statuses with this model, so it shows up in `/docs` and `/openapi.json`.
A global handler in `api.py` flattens FastAPI's request-validation errors (422) to one string;
`/rank`'s own `except` ladder (`app/job_ranking/routes/routes.py`) maps the three LLM failure categories to distinct `detail` prefixes.
In every case, `detail` carries a category label and the exception's **class name**, never the raw
exception message — the full message is logged server-side via `logger.exception`, not returned to
the caller.

| Status | When | Example `detail` |
|---|---|---|
| 422 | The body is not a classified job document, or a query parameter is invalid | `"body.classification.entities: Input should be a valid list"` |
| 500 | `LlmConfigurationError` — the configured model/provider is unusable | `"LLM configuration error: LlmConfigurationError"` |
| 500 | `LlmInvocationError` — a configured provider was called and the call failed | `"LLM invocation error: LlmInvocationError"` |
| 500 | `LlmParseError` — the model responded, but its JSON didn't parse or match the expected shape | `"LLM response parse error: LlmParseError"` |
| 500 | Any other unexpected error | `"Internal server error: RuntimeError"` |
| 503 | The ranking services could not be built (model or default language pack) | `"Service not ready: FileNotFoundError"` |

See [Choosing the model](#choosing-the-model) for what each `LlmError` subclass means and where
providers raise them.

### Log format

`LOG_LEVEL` (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`; default `INFO`) sets verbosity for
both the API and the batch CLI. An invalid value logs a warning and falls back to `INFO`.

`LOG_FORMAT` selects the line shape: `text` (default — the existing
`timestamp | LEVEL | logger | message` format) or `json`, one JSON object per line, for log
aggregators that index structured fields. An invalid value logs a warning and falls back to `text`.
On the API, uvicorn's own lines (access log, startup/shutdown) are rerouted through the same
formatter once the app finishes starting, so `LOG_FORMAT=json` covers them too; the first two
lines uvicorn prints before the app starts (`Started server process`, `Waiting for application
startup`) are always plain text — there is no earlier hook to format them from.

JSON fields:

| Field | Meaning |
|---|---|
| `timestamp` | ISO 8601, UTC |
| `severity` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `logger` | Logger name (module) |
| `message` | The log message |
| `exception` | Traceback, when the record carries one |

Call sites may add their own fields via `extra={"fields": {...}}`; they appear as top-level keys
alongside the ones above, e.g.:

```json
{"timestamp": "2026-10-15T09:14:03.120+00:00", "severity": "INFO", "logger": "app.batch.job_ranker",
 "message": "  step | LLM call 1 — skill ranker (anthropic/claude-opus-5) | 2.103s | menu=12 essential=5 optional=4"}
```

The three LLM step labels (`app/logging_config.py`) name the provider and model that actually ran
(`LlmConfig.describe()`, `"{provider}/{model}"`) instead of a hardcoded vendor name.

### What must not be logged

These must never be logged, at any level, in any format:

- **Full job description/requirements text.** Assembled LLM prompts are logged in full only
  under the explicit `--log-llm1-prompt` / `--log-llm2-prompt` / `--log-llm3-prompt` batch CLI
  flags — never at a default log level, and never reachable from the API (`/rank` never sets
  these flags).
- **Raw LLM response text** at default log level.
- **LLM API keys** — log only `LlmConfig.api_key_tail()` (last 4 characters), as
  `log_ranker_run_configuration` already does.
- **MongoDB URI credentials** — mask with `sanitize_mongodb_uri` (`app/logging_config.py`)
  before logging a connection string.
- **Langfuse keys.**
- **The rejected request body** on a validation error — the 422 handler keeps only the field
  path (`loc`) and message, never the input value.

Safe to log: counts, timings, ids/fingerprints, the configured provider and model name,
exception class names, and validation field locations (never the rejected value).

## Type checking and linting

CI runs three checks on every PR, all enforcing: `ruff check`, `ruff format --check`, and `pyright`. Run them locally before pushing:

```bash
pip install -r requirements.txt "ruff>=0.15,<0.16" "pyright==1.1.414"
ruff check .
ruff format --check --diff .   # drop --check --diff to apply fixes
pyright
```

**Ruff** (`ruff.toml`) selects pyflakes (`F`), a pragmatic subset of pycodestyle (`E4`, `E7`, `E9`), isort (`I`), flake8-bugbear (`B`), and `ANN401`. `E701` is ignored (compact one-line `if cond: return X` mappings are intentional here); nothing else is. `ANN401` fails the build on a `typing.Any` in a function parameter or return type — it does not reach `dict[str, Any]` used as a body/field annotation (e.g. a loosely-structured Mongo document or raw JSON payload), which remains a legitimate type here. There is no suppression mechanism for it: a parameter or return type genuinely needs a real type, not `Any`.

**Pyright** (`pyrightconfig.json`) runs in `basic` mode: it catches real defects — calling a method on a value that can be `None`, passing the wrong argument type, unreachable `isinstance` narrowing — without requiring every third-party dependency to carry complete type information.
