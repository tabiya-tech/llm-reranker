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
| `app/llm/` | The backing LLM: `LlmClient` interface, provider factory, provider implementations |
| `app/observability/` | LLM tracing to Langfuse — generations and embeddings, per job |
| `app/` | Config, MongoDB, prompts, schemas, logging |
| `utils/` | Classifier entities, ESCO–Tabiya mapping, occupation–skill relations |
| `app/languages/` | Language registry — one config module per language |
| `resources/<lang>/` | Per-language pack: taxonomy label CSVs, prompts, job-attribute schema |
| `resources/` | Language-neutral artifacts (embedding whitening) |

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

Edit `.env` with: `MONGODB_URI`, `MONGODB_DATABASE`, `MONGODB_SOURCE_COLLECTION`, `MONGODB_TARGET_COLLECTION`, and `GEMINI_API_KEY`. See `.env.example` for all available options.

## Choosing the model

The three LLM passes talk to a model only through `LlmClient` (`app/llm/base.py`), and configuration decides which implementation answers. Nothing in the grading or enrichment code — prompts, parsing, `pipeline.py`, `ranked_job_schema.py` — knows which provider is in use, so swapping models is an env change.

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

The contract is in `app/llm/base.py`: return raw text (fenced ```json is tolerated — every parser strips fences), never return an empty string, raise `LlmConfigurationError` for anything a retry cannot fix and `LlmInvocationError` for a failed call. Clients are built once per process and shared across `RERANKER_PARALLEL_WORKERS` threads, so they must be thread-safe. `app/llm/retry.py` provides the transient-failure retry the shipped providers use.

**Misconfiguration fails at startup**, not one job at a time: both `main.py` and the API build the client before reading MongoDB, so an unknown provider, an unimportable class, a class that is not an `LlmClient`, a missing model id, or a missing key stops the run with a message naming the variable to fix.

**Not covered by this setting:** the job-embedding step (`RERANKER_JOB_EMBEDDING_MODEL`) stays on `gemini-embedding-001` and still needs `GEMINI_API_KEY` when enabled. It writes into match_v4's stage-1 vector space, whose whitening artifact was fitted on that model — a different embedding model would not be a swap but a re-fit across two repos.

The model that actually ran is recorded per document in `classifier_metadata` (`llm_provider`, `skill_ranker_model`, `job_attributes_model`, `occupation_skills_relevance_model`) and in each run log.

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

`?language=es` (or a `language` field on the document) ranks the job against that
language's pack — taxonomy labels, prompts, job attributes. Each language's taxonomy is
loaded on the first request that needs it, so that request is slower. Omitted, the service
default applies.

**GET `/health`** — readiness check; also reports the default, supported, and
currently-loaded languages.

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
| `--language LANG` | Language pack to rank against (`en`, `es`, or a locale like `AR-es`). Default: from `TARGET_COUNTRY`. |

## Origin UUIDs, work activities, and job status

1. **`skill_groups_origin_uuids`** — After ranking, skill groups are derived from the skill hierarchy so each ranked job can carry the **origin UUIDs** of the relevant skill groups (see `.env` / `resources/` for the hierarchy CSV).
2. **`onet_work_activities`** — Model output includes work-activity context; after a successful batch, optional enrichment can fill or adjust **ONET-style work activities** (importance/level) using the occupation taxonomy reference JSON, when enabled in `.env` (`RERANKER_RUN_WA_ENRICHMENT` and related variables in `.env.example`).
3. **`is_active` and closing dates** — **`is_active`** can be synced from **`classifier_metadata.closing_date`** (open vs closed relative to today) after a successful batch when enabled in `.env`, or by running `python utils/update_is_active_from_closing_date.py` manually. No documents are deleted; only the flag is updated.

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
| `skill_ranker`, `job_attributes`, `occupation_skills_relevance` | generation | prompt, raw JSON output, model, temperature, token usage |
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

Each batch run writes a document to the `ranking-logs` collection (created automatically) tracking status, config, stats, and timing. Progress is updated every 10 jobs. Failed runs include the error message.

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
