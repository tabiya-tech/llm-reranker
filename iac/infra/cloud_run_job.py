"""Cloud Run Job definition for the llm-reranker batch worker.

All configuration is parameterized via Pulumi config — nothing is hardcoded.
The image URI is injected by CI with the commit SHA tag at deploy time.

Per-country settings (``TARGET_COUNTRY``, ``MONGODB_DATABASE``, the taxonomy CSV
overrides) are deliberately **not** baked in as named parameters: one deployed image
serves every country, and the GCP Workflow supplies them as env overrides on each
execution. ``extra_env`` exists for the cases where a stack wants a baked-in default —
see ``extraEnv`` in ``__main__.py``.
"""

from typing import Mapping, Optional

import pulumi_gcp as gcp


def create_reranker_job(
    project: str,
    region: str,
    reranker_image: str,
    mongodb_uri_secret: gcp.secretmanager.Secret,
    gemini_api_key_secret: gcp.secretmanager.Secret,
    enable_llm3: str,
    parallel_workers: str,
    overlap_llm1_llm2: str,
    classified_linked_entity_label_dedupe_types: str,
    enable_job_embedding: str,
    concat_whitening_enabled: str,
    concat_whitening_path: str,
    concat_whitening_required: str,
    memory: str,
    cpu: str,
    timeout: str,
    max_retries: int,
    extra_env: Optional[Mapping[str, str]] = None,
):
    # Names this function already sets — from Secret Manager or a dedicated Pulumi
    # config key. extra_env must not shadow them.
    reserved = frozenset(
        {
            "MONGODB_URI",
            "GEMINI_API_KEY",
            "RERANKER_ENABLE_LLM3",
            "RERANKER_PARALLEL_WORKERS",
            "RERANKER_OVERLAP_LLM1_LLM2",
            "CLASSIFIED_LINKED_ENTITY_LABEL_DEDUPE_TYPES",
            "RERANKER_ENABLE_JOB_EMBEDDING",
            "RERANKER_CONCAT_WHITENING_ENABLED",
            "RERANKER_CONCAT_WHITENING_PATH",
            "RERANKER_CONCAT_WHITENING_REQUIRED",
        }
    )

    envs = [
        gcp.cloudrunv2.JobTemplateTemplateContainerEnvArgs(
            name="MONGODB_URI",
            value_source=gcp.cloudrunv2.JobTemplateTemplateContainerEnvValueSourceArgs(
                secret_key_ref=gcp.cloudrunv2.JobTemplateTemplateContainerEnvValueSourceSecretKeyRefArgs(
                    secret=mongodb_uri_secret.secret_id,
                    version="latest",
                ),
            ),
        ),
        gcp.cloudrunv2.JobTemplateTemplateContainerEnvArgs(
            name="GEMINI_API_KEY",
            value_source=gcp.cloudrunv2.JobTemplateTemplateContainerEnvValueSourceArgs(
                secret_key_ref=gcp.cloudrunv2.JobTemplateTemplateContainerEnvValueSourceSecretKeyRefArgs(
                    secret=gemini_api_key_secret.secret_id,
                    version="latest",
                ),
            ),
        ),
        gcp.cloudrunv2.JobTemplateTemplateContainerEnvArgs(
            name="RERANKER_ENABLE_LLM3",
            value=enable_llm3,
        ),
        gcp.cloudrunv2.JobTemplateTemplateContainerEnvArgs(
            name="RERANKER_PARALLEL_WORKERS",
            value=parallel_workers,
        ),
        gcp.cloudrunv2.JobTemplateTemplateContainerEnvArgs(
            name="RERANKER_OVERLAP_LLM1_LLM2",
            value=overlap_llm1_llm2,
        ),
        gcp.cloudrunv2.JobTemplateTemplateContainerEnvArgs(
            name="CLASSIFIED_LINKED_ENTITY_LABEL_DEDUPE_TYPES",
            value=classified_linked_entity_label_dedupe_types,
        ),
        gcp.cloudrunv2.JobTemplateTemplateContainerEnvArgs(
            name="RERANKER_ENABLE_JOB_EMBEDDING",
            value=enable_job_embedding,
        ),
        gcp.cloudrunv2.JobTemplateTemplateContainerEnvArgs(
            name="RERANKER_CONCAT_WHITENING_ENABLED",
            value=concat_whitening_enabled,
        ),
        gcp.cloudrunv2.JobTemplateTemplateContainerEnvArgs(
            name="RERANKER_CONCAT_WHITENING_PATH",
            value=concat_whitening_path,
        ),
        gcp.cloudrunv2.JobTemplateTemplateContainerEnvArgs(
            name="RERANKER_CONCAT_WHITENING_REQUIRED",
            value=concat_whitening_required,
        ),
    ]

    # Appended last, and only for non-empty values, so an unset extraEnv leaves the
    # deployed job byte-for-byte identical to before this option existed.
    for name, value in sorted((extra_env or {}).items()):
        if not str(name).strip() or value is None or not str(value).strip():
            continue
        if name in reserved:
            raise ValueError(
                f"extraEnv may not override {name!r} — it is set from Secret Manager or "
                "a dedicated Pulumi config key."
            )
        envs.append(gcp.cloudrunv2.JobTemplateTemplateContainerEnvArgs(name=name, value=str(value)))

    job = gcp.cloudrunv2.Job(
        "llm-reranker",
        project=project,
        location=region,
        name="llm-reranker",
        template=gcp.cloudrunv2.JobTemplateArgs(
            task_count=1,
            template=gcp.cloudrunv2.JobTemplateTemplateArgs(
                max_retries=max_retries,
                timeout=timeout,
                containers=[
                    gcp.cloudrunv2.JobTemplateTemplateContainerArgs(
                        image=reranker_image,
                        resources=gcp.cloudrunv2.JobTemplateTemplateContainerResourcesArgs(
                            limits={"cpu": cpu, "memory": memory},
                        ),
                        envs=envs,
                    )
                ],
            ),
        ),
        launch_stage="GA",
    )

    return job
