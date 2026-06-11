"""Cloud Run Job definition for the llm-reranker batch worker.

All configuration is parameterized via Pulumi config — nothing is hardcoded.
The image URI is injected by CI with the commit SHA tag at deploy time.
"""

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
    memory: str,
    cpu: str,
    timeout: str,
    max_retries: int,
):
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
                        envs=[
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
                        ],
                    )
                ],
            ),
        ),
        launch_stage="GA",
    )

    return job
