"""LLM Reranker — Enable Services Stack

Enables all GCP APIs required by the llm-reranker Cloud Run Job, Cloud Run
Service, and supporting infrastructure. Must be deployed before any other
stack.

Required Pulumi config:
  tabiya-horizon-reranker-enable-services:project — GCP project ID
"""

import pulumi
import pulumi_gcp as gcp

config = pulumi.Config()
project = config.require("project")

REQUIRED_SERVICES = [
    "cloudresourcemanager.googleapis.com",
    "artifactregistry.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",  # Workload Identity Federation (CI login)
]

for service in REQUIRED_SERVICES:
    resource_name = service.split(".")[0]
    gcp.projects.Service(
        f"enable-{resource_name}",
        project=project,
        service=service,
        disable_dependent_services=False,
        disable_on_destroy=False,
    )
