"""Vertex AI (Gemini) calls — JSON responses for skill ranking and job attributes."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import vertexai
from google.api_core import exceptions as google_api_exceptions
from vertexai.generative_models import GenerativeModel, GenerationConfig

from app.config import (
    Settings,
    get_settings,
    repo_root,
    vertex_project_unset_message,
)
from app.job_context import JobDescriptionContext
from app.prompts import (
    build_job_attributes_jd_text,
    build_job_attributes_metadata_block,
    build_job_attributes_prompt,
    build_skill_ranker_prompt,
)
from app.schemas import (
    JobAttributesResult,
    JobAttributesSchema,
    OnetWorkActivity,
    load_job_attributes_schema,
    load_onet_work_activities,
    parse_job_attributes_response,
)
from utils.classifier_linked_entities import (
    MappedEscoSkills,
    extract_esco_skill_candidates,
    full_details_from_context,
    map_skill_labels_to_esco_uris,
    parse_skill_ranker_json,
    skills_to_prompt_dicts,
)

logger = logging.getLogger(__name__)


def apply_vertex_credentials(settings: Settings) -> None:
    """Point ADC at the resolved service-account JSON when configured."""
    if settings.google_application_credentials_resolved:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
            settings.google_application_credentials_resolved
        )


def _vertex_credentials_missing_message(settings: Settings) -> str:
    raw = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip() or (
        os.environ.get("VERTEX_SERVICE_ACCOUNT_JSON") or ""
    ).strip()
    hint = ""
    if settings.google_application_credentials_resolved:
        hint = f"\nResolved: {settings.google_application_credentials_resolved!r}"
    elif raw:
        tried = (repo_root() / raw) if not Path(raw).is_absolute() else Path(raw)
        hint = f"\nEnv has {raw!r} but file not found (tried {tried!r})."
    else:
        hint = "\nSet VERTEX_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS in .env, or run gcloud auth application-default login."
    return (
        "No Google credentials for Vertex. Set VERTEX_SERVICE_ACCOUNT_JSON or "
        "GOOGLE_APPLICATION_CREDENTIALS in .env (repo-relative paths OK), or: "
        "gcloud auth application-default login."
        f"{hint}"
    )


def _resolve_gac_path(gac: str) -> Path | None:
    """Accept absolute paths, cwd-relative, or paths relative to repo root."""
    p = Path(gac.strip())
    try:
        if p.is_file():
            return p.resolve()
        if not p.is_absolute():
            under_root = (repo_root() / p).resolve()
            if under_root.is_file():
                return under_root
    except OSError:
        return None
    return None


def _ensure_google_credentials_or_raise(settings: Settings) -> None:
    """Fail fast with a clear message instead of DefaultCredentialsError deep in Vertex."""
    apply_vertex_credentials(settings)
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and gac.strip():
        resolved = _resolve_gac_path(gac)
        if resolved is not None:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(resolved)
            return
        raise RuntimeError(
            f"GOOGLE_APPLICATION_CREDENTIALS is set but file not found: {gac!r}\n"
            + _vertex_credentials_missing_message(settings)
        )
    try:
        import google.auth

        google.auth.default()
    except Exception as e:
        raise RuntimeError(_vertex_credentials_missing_message(settings)) from e


def _call_vertex_json_model(prompt: str, *, settings: Settings) -> str:
    """Single Gemini call with JSON MIME type."""
    project = settings.gcp_project_id.strip()
    if not project:
        raise ValueError(vertex_project_unset_message())
    _ensure_google_credentials_or_raise(settings)
    vertexai.init(project=project, location=settings.gcp_location)
    model = GenerativeModel(settings.vertex_model)
    gen = GenerationConfig(
        temperature=0.2,
        response_mime_type="application/json",
    )
    try:
        response = model.generate_content(prompt, generation_config=gen)
    except google_api_exceptions.NotFound as e:
        raise RuntimeError(
            f"Vertex model {settings.vertex_model!r} not found in "
            f"{settings.gcp_location!r} (or your project has no access). "
            "Set VERTEX_MODEL in .env to a model id from Vertex AI Model Garden for "
            "that region (e.g. gemini-2.0-flash-001 in us-central1). "
            "See: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions"
        ) from e
    text = _response_text(response)
    if not text.strip():
        raise RuntimeError("Empty response from Vertex Gemini")
    return text


def call_skill_ranker_model(prompt: str, *, settings: Settings | None = None) -> str:
    """Send the skill-ranker prompt to Gemini on Vertex; expect JSON in the response body."""
    s = settings or get_settings()
    logger.info(
        "LLM call 1 (skill ranker): invoking model=%s project=%s",
        s.vertex_model,
        s.gcp_project_id or "(unset)",
    )
    out = _call_vertex_json_model(prompt, settings=s)
    logger.info("LLM call 1 (skill ranker): completed")
    return out


def call_job_attributes_model(prompt: str, *, settings: Settings | None = None) -> str:
    """Send the job-attributes (LLM call 2) prompt; expect JSON with ``attributes`` and ``work_activities``."""
    s = settings or get_settings()
    logger.info(
        "LLM call 2 (job attributes): invoking model=%s project=%s",
        s.vertex_model,
        s.gcp_project_id or "(unset)",
    )
    out = _call_vertex_json_model(prompt, settings=s)
    logger.info("LLM call 2 (job attributes): completed")
    return out


def build_job_attributes_prompt_for_job(
    ctx: JobDescriptionContext,
    *,
    schema: JobAttributesSchema | None = None,
    onet: tuple[OnetWorkActivity, ...] | None = None,
    settings: Settings | None = None,
) -> str:
    """
    Inject JD text, parsed metadata, and schema / ONET constraints into the second LLM prompt.
    """
    s = settings or get_settings()
    schema = schema or load_job_attributes_schema(settings=s)
    onet = onet or load_onet_work_activities(settings=s)
    jd = build_job_attributes_jd_text(ctx)
    meta = build_job_attributes_metadata_block(ctx)
    title = (ctx.title or "").strip() or "Unknown role"
    county = (ctx.county or "").strip() or "Unknown"
    return build_job_attributes_prompt(
        occupation_title=title,
        county=county,
        job_description_text=jd,
        metadata_block=meta,
        schema=schema,
        onet_work_activities=onet,
        settings=s,
    )


def run_job_attributes_for_job(
    ctx: JobDescriptionContext,
    *,
    settings: Settings | None = None,
    schema: JobAttributesSchema | None = None,
    onet: tuple[OnetWorkActivity, ...] | None = None,
) -> JobAttributesResult:
    """
    LLM call 2: unified attributes + ONET work-activity ids; parse JSON into validated dicts.
    """
    s = settings or get_settings()
    prompt = build_job_attributes_prompt_for_job(ctx, schema=schema, onet=onet, settings=s)
    raw = call_job_attributes_model(prompt, settings=s)
    schema_resolved = schema or load_job_attributes_schema(settings=s)
    onet_resolved = onet or load_onet_work_activities(settings=s)
    attrs, wa_ids = parse_job_attributes_response(raw, schema_resolved, onet_resolved)
    return JobAttributesResult(
        attributes=attrs,
        work_activity_ids=wa_ids,
        raw_response=raw,
    )


def _response_text(response: Any) -> str:
    t = getattr(response, "text", None)
    if isinstance(t, str) and t.strip():
        return t
    cands = getattr(response, "candidates", None) or []
    if not cands:
        return ""
    parts = getattr(cands[0].content, "parts", None) or []
    if not parts:
        return ""
    p0 = parts[0]
    return getattr(p0, "text", "") or ""


def run_skill_ranker_for_job(
    ctx: JobDescriptionContext,
    *,
    settings: Settings | None = None,
) -> tuple[list[str], list[str], MappedEscoSkills, str]:
    """
    Skill ranker: ESCO candidates from ``ctx.classified``, JD text, LLM call, JSON parse, label→URI/UUID map.

    Returns essential labels, optional labels, ``MappedEscoSkills``, and raw model text.
    """
    classified_doc = ctx.classified
    candidates = extract_esco_skill_candidates(classified_doc)
    full_details = full_details_from_context(ctx)
    s = settings or get_settings()
    prompt = build_skill_ranker_prompt(full_details, skills_to_prompt_dicts(candidates), settings=s)
    raw = call_skill_ranker_model(prompt, settings=s)
    essential, optional = parse_skill_ranker_json(raw)
    mapped = map_skill_labels_to_esco_uris(essential, optional, candidates)
    return essential, optional, mapped, raw
