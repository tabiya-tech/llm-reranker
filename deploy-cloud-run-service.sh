#!/usr/bin/env bash
# Deploy the FastAPI reranker (POST /rank, GET /health) as a Cloud Run *Service*.
# Uses the same container image as the batch Job; only the start command differs.
#
# Prerequisites:
#   - gcloud auth and project
#   - Image already in Artifact Registry (e.g. gcloud builds submit --config=llm-reranker/cloudbuild.yaml .)
#   - Secret Manager secrets (recommended) or use --update-env-vars for non-sensitive config only
#
# Usage:
#   chmod +x llm-reranker/deploy-cloud-run-service.sh
#   ./llm-reranker/deploy-cloud-run-service.sh
#
# Then grant invokers (example: one user):
#   gcloud run services add-iam-policy-binding llm-reranker-api \
#     --region=us-central1 --member=user:someone@example.com --role=roles/run.invoker
#
set -euo pipefail

PROJECT="${GCP_PROJECT:-horizon-dev-481316}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${RERANKER_SERVICE_NAME:-llm-reranker-api}"
# Default image tag :service (same image as :latest from cloudbuild; use this for Cloud Run API).
IMAGE="${RERANKER_IMAGE:-us-central1-docker.pkg.dev/${PROJECT}/job-pipeline/llm-reranker:service}"

# Container listens on 8080 (Cloud Run default PORT).
CONTAINER_PORT=8080

# --- Non-secret env vars (add more with repeated --update-env-vars or use Console) ---
# Minimal set; mirror your local .env for full behavior (database, model, paths).
ENV_VARS="PYTHONUNBUFFERED=1"

# REQUIRED in practice: mount GEMINI_API_KEY (API exits on startup without it; uvicorn never binds → Cloud Run error).
#   export RERANKER_SERVICE_SECRETS="GEMINI_API_KEY=YOUR_SECRET_NAME:latest"
# Cloud Run's service account needs secretmanager.secretAccessor on that secret.
# Optional extra: ,MONGODB_URI=...
SECRETS="${RERANKER_SERVICE_SECRETS:-}"

if [[ -z "${SECRETS}" ]]; then
  echo "WARNING: RERANKER_SERVICE_SECRETS is empty — deploy often fails (no GEMINI_API_KEY)." >&2
  echo "  Example: export RERANKER_SERVICE_SECRETS='GEMINI_API_KEY=GEMINI_API_KEY:latest'" >&2
fi

gcloud run deploy "${SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --image="${IMAGE}" \
  --platform=managed \
  --port="${CONTAINER_PORT}" \
  --cpu=2 \
  --memory=4Gi \
  --timeout=3600 \
  --cpu-boost \
  --concurrency="${RERANKER_CONCURRENCY:-4}" \
  --min-instances="${RERANKER_MIN_INSTANCES:-0}" \
  --max-instances="${RERANKER_MAX_INSTANCES:-10}" \
  --no-cpu-throttling \
  --command=python \
  --args=-m,uvicorn,api:app,--host,0.0.0.0,--port,${CONTAINER_PORT} \
  --set-env-vars="${ENV_VARS}" \
  ${SECRETS:+--set-secrets="${SECRETS}"} \
  --no-allow-unauthenticated

echo ""
echo "Deployed: ${SERVICE}"
gcloud run services describe "${SERVICE}" --project="${PROJECT}" --region="${REGION}" \
  --format='value(status.url)'
