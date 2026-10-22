# LLM Reranker IaC

Infrastructure-as-Code for the job-pipeline LLM Reranker, using **Pulumi (Python)**.

## What this manages

| Resource | Pulumi Stack |
|----------|-------------|
| GCP API enablement | `enable-services/` |
| Secret Manager refs (`MONGODB_URI`, `GEMINI_API_KEY`), Cloud Run Job (`llm-reranker`), Cloud Run Service (`llm-reranker-api`) | `infra/` |

**Not managed here** (shared infra, lives outside this repo):
- Artifact Registry repository (`job-pipeline`) — managed by the scraper IaC; this repo only pushes images to it.
- GCP Workflow `job-pipeline` (orchestrates scraper → classifier → reranker)
- Cloud Scheduler jobs
- Service account `job-pipeline-workflow`
- Scraper and classifier Cloud Run Jobs

## CI/CD

GitHub Actions workflows in `.github/workflows/`:

1. **`main.yml`** — Orchestrator:
   - On push to any branch or PR to main: runs tests
   - On merge to main: tests + build image + deploy via Pulumi
2. **`reranker-ci.yml`** — Tests + conditional Docker image build/push (three tags: `:sha`, `:latest`, `:service`)
3. **`deploy.yml`** — Pulumi deploy (called by main.yml or manually via workflow_dispatch)

### Image tagging

The single image is pushed with three tags:
- `:${{ github.sha }}` — used by the Cloud Run Job
- `:latest` — convenience tag
- `:service` — used by the Cloud Run Service (so the Service can be updated independently if needed)

Pulumi updates the Cloud Run Job and Service to reference the exact `:sha` / `:service` tags.

## Prerequisites

### GitHub Environment secrets (per stack)

| Secret | Description |
|--------|-------------|
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | WIF provider resource name |
| `GCP_SERVICE_ACCOUNT` | SA email for CI impersonation |
| `GCP_PROJECT_ID` | GCP project ID (e.g. `horizon-dev-481316`) |
| `GCP_REGION` | GCP region (e.g. `us-central1`) |
| `PULUMI_ACCESS_TOKEN` | Pulumi Cloud auth token |

### GCP Secret Manager secrets

| Secret | Content |
|--------|---------|
| `stack-config-reranker-enable-services` | Pulumi config for enable-services stack |
| `stack-config-reranker-infra` | Pulumi config for infra stack |
| `MONGODB_URI` | MongoDB connection string (consumed by Cloud Run Job) |
| `GEMINI_API_KEY` | Google Gemini API key (consumed by Job + Service) |

Backing the reranker with a non-Gemini model needs no change here: put the selection in
`extraEnv` (`RERANKER_LLM_PROVIDER`, `RERANKER_LLM_MODEL`, `RERANKER_LLM_BASE_URL`) and the
credential in `extraSecrets`, which references any secret in the project by id. See
`../README.md#choosing-the-model`. `GEMINI_API_KEY` stays required either way — the
job-embedding step is pinned to Gemini.

These stack config secrets are reranker-specific (separate from the scraper's `stack-config-*` secrets) to avoid naming collisions when both IaC stacks share a GCP project.

Bootstrap stack configs with:
```bash
python iac/scripts/configure.py --stack dev --project horizon-dev-481316
```

## Per-country configuration

Nothing country- or language-specific is baked into the deployed Job. `TARGET_COUNTRY` and
`MONGODB_DATABASE` are supplied by the `job-pipeline` GCP Workflow as env overrides on each
execution, so one image serves Zambia, Kenya, and Argentina:

```bash
gcloud run jobs execute llm-reranker --region us-central1 \
  --update-env-vars \
TARGET_COUNTRY=argentina,MONGODB_DATABASE=ArgentinaJobs_V2
```

The language follows from the country (`argentina` → `es`) and selects the whole
`resources/es/` pack (taxonomy labels and the Spanish job-attributes schema), and the shared
prompts are told to answer in Spanish.
Override with `RERANKER_LANGUAGE` (or `--language`) when you want a different pairing.
Every language pack ships in the image under `resources/<lang>/` — see the *Countries and
languages* section of the repo README.

If a stack should *default* to one country instead, set the optional `extraEnv` mapping
in its stack config (see `iac/templates/stack_config.infra.template.yaml`). Leaving it
unset produces the same Job definition as before the option existed.

## Local deployment

```bash
# 1. Authenticate
gcloud auth login
gcloud config set project horizon-dev-481316

# 2. Prepare configs (fetch from Secret Manager)
python iac/scripts/prepare.py \
  --stack dev \
  --project horizon-dev-481316 \
  --reranker-image us-central1-docker.pkg.dev/horizon-dev-481316/job-pipeline/llm-reranker:latest \
  --reranker-service-image us-central1-docker.pkg.dev/horizon-dev-481316/job-pipeline/llm-reranker:service

# 3. Deploy
python iac/scripts/up.py --stack dev

# Or deploy a single module
python iac/scripts/up.py --stack dev --module infra
```

## Deploy order

```
enable-services → infra
```

APIs must be enabled before any other resources can be created.

## File structure

```
iac/
  templates/           # Config schema templates (committed)
  scripts/             # Deploy orchestration (prepare.py, up.py, configure.py)
  enable-services/     # Pulumi stack: GCP API enablement
  infra/               # Pulumi stack: Secret refs + Cloud Run Job + Cloud Run Service
```
