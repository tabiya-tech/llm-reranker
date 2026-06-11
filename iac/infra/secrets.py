"""Secret Manager references for the llm-reranker job and service.

We reference existing secrets (not create them) since they are shared across
the pipeline and were created outside this IaC. Pulumi imports them as
read-only references so we can wire them into the Cloud Run Job + Service env.
"""

import pulumi_gcp as gcp


def get_secret_references(
    project: str,
    mongodb_secret_id: str,
    gemini_secret_id: str,
) -> dict:
    mongodb_uri_secret = gcp.secretmanager.Secret(
        "mongodb-uri",
        project=project,
        secret_id=mongodb_secret_id,
        replication=gcp.secretmanager.SecretReplicationArgs(auto={}),
    )

    gemini_api_key_secret = gcp.secretmanager.Secret(
        "gemini-api-key",
        project=project,
        secret_id=gemini_secret_id,
        replication=gcp.secretmanager.SecretReplicationArgs(auto={}),
    )

    return {
        "mongodb_uri": mongodb_uri_secret,
        "gemini_api_key": gemini_api_key_secret,
    }
