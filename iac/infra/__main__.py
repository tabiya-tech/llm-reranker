"""LLM Reranker — Infra Stack

Deploys:
  - Secret Manager references (MONGODB_URI, GEMINI_API_KEY)
  - Cloud Run Job (llm-reranker) for batch processing
  - Cloud Run Service (llm-reranker-api) for single-document HTTP API

Note: Artifact Registry is shared with the scraper and managed by the scraper's
IaC stack. This stack references images by URI only.

Required Pulumi config:
  tabiya-horizon-reranker-infra:project                — GCP project ID
  tabiya-horizon-reranker-infra:region                 — GCP region (default: us-central1)
  tabiya-horizon-reranker-infra:rerankerImage          — Job image URI (injected by CI)
  tabiya-horizon-reranker-infra:rerankerServiceImage   — Service image URI (injected by CI)
  tabiya-horizon-reranker-infra:mongodbSecretId        — Secret ID for MONGODB_URI (default: MONGODB_URI)
  tabiya-horizon-reranker-infra:geminiSecretId         — Secret ID for GEMINI_API_KEY (default: GEMINI_API_KEY)

  # Job env / resources
  tabiya-horizon-reranker-infra:enableLlm3                            — "0" or "1" (default: 0)
  tabiya-horizon-reranker-infra:parallelWorkers                       — int as string (default: 4)
  tabiya-horizon-reranker-infra:overlapLlm1Llm2                       — "0" or "1" (default: 1)
  tabiya-horizon-reranker-infra:classifiedLinkedEntityLabelDedupeTypes — comma list (default: occupation,skill)
  tabiya-horizon-reranker-infra:enableJobEmbedding                    — "0" or "1" (default: 1)
  tabiya-horizon-reranker-infra:jobMemory                             — default: 2Gi
  tabiya-horizon-reranker-infra:jobCpu                                — default: 1000m
  tabiya-horizon-reranker-infra:jobTimeout                            — default: 14400s
  tabiya-horizon-reranker-infra:jobMaxRetries                         — default: 3

  # Service resources
  tabiya-horizon-reranker-infra:serviceMemory                         — default: 4Gi
  tabiya-horizon-reranker-infra:serviceCpu                            — default: 2
  tabiya-horizon-reranker-infra:serviceTimeoutSeconds                 — default: 3600
  tabiya-horizon-reranker-infra:serviceContainerConcurrency           — default: 4
  tabiya-horizon-reranker-infra:serviceMaxScale                       — default: 10
  tabiya-horizon-reranker-infra:serviceContainerPort                  — default: 8080
  tabiya-horizon-reranker-infra:serviceStartupProbePeriodSeconds      — default: 240
  tabiya-horizon-reranker-infra:serviceStartupProbeTimeoutSeconds     — default: 240
"""

import pulumi

from secrets import get_secret_references
from cloud_run_job import create_reranker_job
from cloud_run_service import create_reranker_service

config = pulumi.Config()
project = config.require("project")
region = config.get("region") or "us-central1"
reranker_image = config.require("rerankerImage")
reranker_service_image = config.require("rerankerServiceImage")

mongodb_secret_id = config.get("mongodbSecretId") or "MONGODB_URI"
gemini_secret_id = config.get("geminiSecretId") or "GEMINI_API_KEY"

enable_llm3 = config.get("enableLlm3") or "0"
parallel_workers = config.get("parallelWorkers") or "4"
overlap_llm1_llm2 = config.get("overlapLlm1Llm2") or "1"
classified_linked_entity_label_dedupe_types = (
    config.get("classifiedLinkedEntityLabelDedupeTypes") or "occupation,skill"
)
enable_job_embedding = config.get("enableJobEmbedding") or "1"

job_memory = config.get("jobMemory") or "2Gi"
job_cpu = config.get("jobCpu") or "1000m"
job_timeout = config.get("jobTimeout") or "14400s"
job_max_retries = int(config.get("jobMaxRetries") or "3")

service_memory = config.get("serviceMemory") or "4Gi"
service_cpu = config.get("serviceCpu") or "2"
service_timeout_seconds = int(config.get("serviceTimeoutSeconds") or "3600")
service_container_concurrency = int(config.get("serviceContainerConcurrency") or "4")
service_max_scale = int(config.get("serviceMaxScale") or "10")
service_container_port = int(config.get("serviceContainerPort") or "8080")
service_startup_probe_period_seconds = int(
    config.get("serviceStartupProbePeriodSeconds") or "240"
)
service_startup_probe_timeout_seconds = int(
    config.get("serviceStartupProbeTimeoutSeconds") or "240"
)

secrets = get_secret_references(
    project=project,
    mongodb_secret_id=mongodb_secret_id,
    gemini_secret_id=gemini_secret_id,
)

reranker_job = create_reranker_job(
    project=project,
    region=region,
    reranker_image=reranker_image,
    mongodb_uri_secret=secrets["mongodb_uri"],
    gemini_api_key_secret=secrets["gemini_api_key"],
    enable_llm3=enable_llm3,
    parallel_workers=parallel_workers,
    overlap_llm1_llm2=overlap_llm1_llm2,
    classified_linked_entity_label_dedupe_types=classified_linked_entity_label_dedupe_types,
    enable_job_embedding=enable_job_embedding,
    memory=job_memory,
    cpu=job_cpu,
    timeout=job_timeout,
    max_retries=job_max_retries,
)
pulumi.export("rerankerJobName", reranker_job.name)

reranker_service = create_reranker_service(
    project=project,
    region=region,
    reranker_service_image=reranker_service_image,
    gemini_api_key_secret=secrets["gemini_api_key"],
    memory=service_memory,
    cpu=service_cpu,
    timeout_seconds=service_timeout_seconds,
    container_concurrency=service_container_concurrency,
    max_scale=service_max_scale,
    container_port=service_container_port,
    startup_probe_period_seconds=service_startup_probe_period_seconds,
    startup_probe_timeout_seconds=service_startup_probe_timeout_seconds,
)
pulumi.export("rerankerServiceName", reranker_service.name)
pulumi.export("rerankerServiceUrl", reranker_service.uri)
