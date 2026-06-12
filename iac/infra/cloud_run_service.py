"""Cloud Run Service definition for the llm-reranker HTTP API.

All configuration is parameterized via Pulumi config — nothing is hardcoded.
The image URI is injected by CI at deploy time. The same image as the batch
Job is used with a command override to launch uvicorn on the configured port.
"""

import pulumi
import pulumi_gcp as gcp


def create_reranker_service(
    project: str,
    region: str,
    reranker_service_image: str,
    gemini_api_key_secret: gcp.secretmanager.Secret,
    memory: str,
    cpu: str,
    timeout_seconds: int,
    container_concurrency: int,
    max_scale: int,
    container_port: int,
    startup_probe_period_seconds: int,
    startup_probe_timeout_seconds: int,
):
    service = gcp.cloudrunv2.Service(
        "llm-reranker-api",
        project=project,
        location=region,
        name="llm-reranker-api",
        ingress="INGRESS_TRAFFIC_ALL",
        template=gcp.cloudrunv2.ServiceTemplateArgs(
            max_instance_request_concurrency=container_concurrency,
            timeout=f"{timeout_seconds}s",
            scaling=gcp.cloudrunv2.ServiceTemplateScalingArgs(
                max_instance_count=max_scale,
            ),
            containers=[
                gcp.cloudrunv2.ServiceTemplateContainerArgs(
                    image=reranker_service_image,
                    commands=["python"],
                    args=[
                        "-m",
                        "uvicorn",
                        "api:app",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        str(container_port),
                    ],
                    ports=[
                        gcp.cloudrunv2.ServiceTemplateContainerPortArgs(
                            name="http1",
                            container_port=container_port,
                        ),
                    ],
                    resources=gcp.cloudrunv2.ServiceTemplateContainerResourcesArgs(
                        limits={"cpu": cpu, "memory": memory},
                        cpu_idle=False,  # CPU throttling disabled (always-on CPU)
                        startup_cpu_boost=True,
                    ),
                    envs=[
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="PYTHONUNBUFFERED",
                            value="1",
                        ),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="GEMINI_API_KEY",
                            value_source=gcp.cloudrunv2.ServiceTemplateContainerEnvValueSourceArgs(
                                secret_key_ref=gcp.cloudrunv2.ServiceTemplateContainerEnvValueSourceSecretKeyRefArgs(
                                    secret=gemini_api_key_secret.secret_id,
                                    version="latest",
                                ),
                            ),
                        ),
                    ],
                    startup_probe=gcp.cloudrunv2.ServiceTemplateContainerStartupProbeArgs(
                        failure_threshold=1,
                        period_seconds=startup_probe_period_seconds,
                        timeout_seconds=startup_probe_timeout_seconds,
                        tcp_socket=gcp.cloudrunv2.ServiceTemplateContainerStartupProbeTcpSocketArgs(
                            port=container_port,
                        ),
                    ),
                ),
            ],
        ),
        traffics=[
            gcp.cloudrunv2.ServiceTrafficArgs(
                type="TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST",
                percent=100,
            ),
        ],
        launch_stage="GA",
        opts=pulumi.ResourceOptions(
            # Cloud Run returns service-level scaling.minInstanceCount: 0 even
            # when unset (server-side default). Ignore it to avoid perpetual
            # drift detection.
            ignore_changes=["scaling"],
        ),
    )

    return service
