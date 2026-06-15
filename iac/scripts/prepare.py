#!/usr/bin/env python3
"""prepare.py — fetch environment config from GCP Secret Manager and write
local files that pulumi up needs for each stack.

Usage (called by CI before pulumi up):
    python iac/scripts/prepare.py \\
        --stack dev \\
        --project horizon-dev-481316 \\
        --reranker-image us-central1-docker.pkg.dev/.../llm-reranker:SHA \\
        --reranker-service-image us-central1-docker.pkg.dev/.../llm-reranker:service

What it does:
  1. Fetches one secret per stack ("stack-config-enable-services", "stack-config-infra")
  2. Validates each against its template in iac/templates/
  3. Merges the reranker image URIs into the infra stack config
  4. Writes Pulumi.{stack}.yaml into each stack's directory

GCP project must be authenticated before running.
"""

import argparse
import os
import re
import sys

import yaml
from google.cloud import secretmanager
from google.api_core.exceptions import NotFound

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
IAC_DIR = os.path.join(REPO_ROOT, "iac")
TEMPLATES_DIR = os.path.join(IAC_DIR, "templates")

STACKS = {
    "enable-services": (
        "iac/enable-services",
        "stack_config.enable-services.template.yaml",
        "tabiya-horizon-reranker-enable-services",
    ),
    "infra": (
        "iac/infra",
        "stack_config.infra.template.yaml",
        "tabiya-horizon-reranker-infra",
    ),
}


def _fetch_secret(project: str, secret_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project}/secrets/{secret_id}/versions/latest"
    try:
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("utf-8")
    except NotFound:
        print(f"error: secret '{secret_id}' not found in project '{project}'.", file=sys.stderr)
        print("       Create it with:", file=sys.stderr)
        print(f"         gcloud secrets create {secret_id} --project {project}", file=sys.stderr)
        print(f"         gcloud secrets versions add {secret_id} --project {project} --data-file=-", file=sys.stderr)
        sys.exit(1)


def _is_regex(value: str) -> bool:
    return isinstance(value, str) and value.startswith("/") and value.endswith("/")


def _validate_against_template(template: dict, actual: dict, parent: str = "root") -> bool:
    ok = True
    for key, template_value in template.items():
        actual_value = actual.get(key, "")
        if isinstance(template_value, dict):
            if not isinstance(actual_value, dict):
                print(f"error: {parent}.{key} must be a mapping.")
                ok = False
            else:
                ok = _validate_against_template(template_value, actual_value, f"{parent}.{key}") and ok
        elif isinstance(template_value, str):
            if _is_regex(template_value):
                pattern = re.compile(template_value[1:-1], re.DOTALL)
            else:
                pattern = re.compile(r"[\s\S]+")
            if not re.fullmatch(pattern, str(actual_value)):
                print(f"error: {parent}.{key} value {actual_value!r} does not match expected pattern {template_value!r}.")
                ok = False
    for key in actual:
        if key not in template:
            print(f"warning: {parent}.{key} is present but not in the template (will be passed through).")
    return ok


def _validate_stack_config(logical_stack: str, config_content: str) -> dict:
    _, template_file, _ = STACKS[logical_stack]
    template_path = os.path.join(TEMPLATES_DIR, template_file)
    template = yaml.safe_load(open(template_path))
    actual = yaml.safe_load(config_content)
    if not _validate_against_template(template, actual):
        print(f"error: stack-config-reranker-{logical_stack} does not satisfy the template. Aborting.", file=sys.stderr)
        sys.exit(1)
    print(f"info: stack-config-reranker-{logical_stack} validated.")
    return actual


def _write_pulumi_yaml(
    logical_stack: str,
    stack: str,
    config: dict,
    reranker_image: str = "",
    reranker_service_image: str = "",
):
    stack_dir, _, _ = STACKS[logical_stack]
    if logical_stack == "infra":
        config.setdefault("config", {})
        if reranker_image:
            config["config"]["tabiya-horizon-reranker-infra:rerankerImage"] = reranker_image
        if reranker_service_image:
            config["config"]["tabiya-horizon-reranker-infra:rerankerServiceImage"] = reranker_service_image

    path = os.path.join(REPO_ROOT, stack_dir, f"Pulumi.{stack}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"info: wrote {path}")


def _main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stack", required=True, help="Pulumi stack name (dev / staging / prod)")
    parser.add_argument("--project", required=True, help="GCP project ID for this environment")
    parser.add_argument("--reranker-image", required=True, help="Reranker Job Docker image URI (commit SHA tag)")
    parser.add_argument("--reranker-service-image", required=True, help="Reranker Service Docker image URI")
    parser.add_argument(
        "--stacks",
        default="all",
        help="Comma-separated logical stacks to prepare (default: all). "
             "Options: enable-services,infra",
    )
    args = parser.parse_args()

    logical_stacks = list(STACKS.keys()) if args.stacks == "all" else args.stacks.split(",")
    for s in logical_stacks:
        if s not in STACKS:
            print(f"error: unknown logical stack '{s}'. Valid options: {list(STACKS)}", file=sys.stderr)
            sys.exit(1)

    print(f"info: preparing stack '{args.stack}' from project '{args.project}'")
    print(f"info: logical stacks: {logical_stacks}")

    for logical_stack in logical_stacks:
        secret_id = f"stack-config-reranker-{logical_stack}"
        config_content = _fetch_secret(args.project, secret_id)
        config = _validate_stack_config(logical_stack, config_content)
        _write_pulumi_yaml(
            logical_stack,
            args.stack,
            config,
            reranker_image=args.reranker_image,
            reranker_service_image=args.reranker_service_image,
        )

    print(f"info: preparation complete for stack '{args.stack}'.")


if __name__ == "__main__":
    _main()
