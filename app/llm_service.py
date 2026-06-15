"""Gemini calls — JSON responses for skill ranking, job attributes, and occupation-skills relevance.

Prefers the Google GenAI SDK (``GEMINI_API_KEY``) when an API key is configured;
falls back to Vertex AI SDK (service account) otherwise.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import asdict, replace
from typing import Any

import httpx
from google import genai
from google.genai import types as genai_types

from app.config import Settings, get_settings
from app.job_context import JobDescriptionContext
from app.prompts import (
    JobAttributesLlm2Extras,
    OccupationPromptEnrichment,
    build_job_attributes_jd_text,
    build_job_attributes_metadata_block,
    build_job_attributes_prompt,
    build_occupation_skills_relevance_prompt,
    llm2_opening_line_from_context,
    resolve_llm2_single_county_user_fields,
    build_skill_ranker_prompt,
    format_skill_ranker_skills_text,
)
from app.schemas import (
    JobAttributesResult,
    JobAttributesSchema,
    OccupationSkillsRelevanceResult,
    OnetWorkActivity,
    load_job_attributes_schema,
    load_onet_work_activities,
    parse_job_attributes_response,
    parse_occupation_skills_relevance_response,
)
from utils.classifier_linked_entities import (
    EscoSkillCandidate,
    MappedEscoSkills,
    extract_esco_skill_candidates,
    full_details_from_context,
    map_skill_labels_to_esco_uris,
    parse_skill_ranker_json,
    skills_to_prompt_dicts,
)

logger = logging.getLogger(__name__)

_genai_client: genai.Client | None = None


def _get_genai_client(settings: Settings) -> genai.Client:
    """Return a process-scoped GenAI client (created once, reused across calls)."""
    global _genai_client
    if _genai_client is not None:
        return _genai_client
    api_key = settings.gemini_api_key
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY (or API_KEY) in .env.\n"
            "Create one at https://aistudio.google.com/apikey for your GCP project."
        )
    _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def _reset_genai_client() -> None:
    """Drop cached client so the next call opens a fresh HTTP pool (after connection drops)."""
    global _genai_client
    _genai_client = None


def _transient_gemini_network_error(exc: BaseException) -> bool:
    """True for errors where retrying the same request may succeed (load balancers, idle TCP, Wi‑Fi)."""
    if isinstance(
        exc,
        (
            httpx.ReadError,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            httpx.TimeoutException,
            ConnectionResetError,
            BrokenPipeError,
            TimeoutError,
        ),
    ):
        return True
    msg = str(exc).lower()
    for needle in (
        "connection reset",
        "connection aborted",
        "broken pipe",
        "econnreset",
        "readerror",
        "temporarily unavailable",
    ):
        if needle in msg:
            return True
    errno = getattr(exc, "errno", None)
    if errno == 54:  # macOS ECONNRESET / "connection reset by peer" on some runs
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return _transient_gemini_network_error(cause)
    return False


def _gemini_transient_max_attempts() -> int:
    raw = (os.environ.get("GEMINI_TRANSIENT_MAX_ATTEMPTS") or "5").strip()
    try:
        n = int(raw)
    except ValueError:
        return 5
    return max(1, min(n, 12))


def _call_gemini_json(prompt: str, *, settings: Settings) -> str:
    """Gemini call via GenAI SDK with JSON MIME; retries transient network failures across batches."""
    config = genai_types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
    )
    max_attempts = _gemini_transient_max_attempts()
    last_exc: BaseException | None = None

    for attempt in range(max_attempts):
        client = _get_genai_client(settings)
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            last_exc = e
            msg = str(e)
            if "404" in msg or "not found" in msg.lower():
                raise RuntimeError(
                    f"Model {settings.gemini_model!r} not found (or project has no access). "
                    f"Check the model id at https://ai.google.dev/gemini-api/docs/models — "
                    f"preview models use a -preview suffix (e.g. gemini-3-pro-preview)."
                ) from e
            if attempt + 1 < max_attempts and _transient_gemini_network_error(e):
                delay = min(60.0, (2**attempt) + random.uniform(0.0, 1.5))
                logger.warning(
                    "Gemini transient network error (attempt %s/%s): %s — retry in %.1fs; "
                    "recreating API client",
                    attempt + 1,
                    max_attempts,
                    e,
                    delay,
                )
                time.sleep(delay)
                _reset_genai_client()
                continue
            ename = type(e).__name__
            raise RuntimeError(f"Gemini API error ({ename}): {msg}") from e

        text = _response_text(response)
        if not text.strip():
            raise RuntimeError("Empty response from Gemini")
        return text

    raise RuntimeError(f"Gemini API error: exhausted retries ({last_exc})") from last_exc


def call_skill_ranker_model(prompt: str, *, settings: Settings | None = None) -> str:
    """LLM call 1 — skill ranker; expect JSON."""
    s = settings or get_settings()
    logger.debug("LLM call 1 (skill ranker): invoking model=%s", s.gemini_model)
    out = _call_gemini_json(prompt, settings=s)
    logger.debug("LLM call 1 (skill ranker): completed")
    return out


def call_job_attributes_model(prompt: str, *, settings: Settings | None = None) -> str:
    """LLM call 2 — job attributes; expect JSON with ``attributes`` and ``work_activities``."""
    s = settings or get_settings()
    logger.debug("LLM call 2 (job attributes): invoking model=%s", s.gemini_model)
    out = _call_gemini_json(prompt, settings=s)
    logger.debug("LLM call 2 (job attributes): completed")
    return out


def call_occupation_skills_relevance_model(prompt: str, *, settings: Settings | None = None) -> str:
    """LLM call 3 — occupation-skills listing relevance; expect valid JSON."""
    s = settings or get_settings()
    logger.debug("LLM call 3 (occupation skills relevance): invoking model=%s", s.gemini_model)
    out = _call_gemini_json(prompt, settings=s)
    logger.debug("LLM call 3 (occupation skills relevance): completed")
    return out


def run_occupation_skills_relevance_for_job(
    ctx: JobDescriptionContext,
    occupation_to_skills_items: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
    log_llm3_prompt: bool = False,
    log_llm3_job_label: str = "",
) -> OccupationSkillsRelevanceResult:
    """LLM call 3: judge essential/optional occupation-linked skills against the listing."""
    from app.ranked_job_schema import occupation_skills_relevance_menu_size

    s = settings or get_settings()
    n_menu = occupation_skills_relevance_menu_size(occupation_to_skills_items)
    prompt = build_occupation_skills_relevance_prompt(ctx, occupation_to_skills_items, settings=s)
    if log_llm3_prompt:
        sep = "-" * 80
        label = (log_llm3_job_label or "").strip() or "(job)"
        logger.info("%s", sep)
        logger.info(
            "LLM3 occupation skills relevance | job=%s | menu_skills=%s",
            label,
            n_menu,
        )
        logger.info("LLM3 assembled prompt (%s chars):\n%s", len(prompt), prompt)
        logger.info("%s", sep)
    raw = call_occupation_skills_relevance_model(prompt, settings=s)
    return parse_occupation_skills_relevance_response(raw)


def build_job_attributes_prompt_for_job(
    ctx: JobDescriptionContext,
    *,
    occupation: OccupationPromptEnrichment,
    extras: JobAttributesLlm2Extras | None = None,
    schema: JobAttributesSchema | None = None,
    onet: tuple[OnetWorkActivity, ...] | None = None,
    settings: Settings | None = None,
) -> str:
    """LLM2: single-county occupation user block + ONET/JSON appendix; system from ``resources/prompts.py``."""
    s = settings or get_settings()
    schema = schema or load_job_attributes_schema(settings=s)
    onet = onet or load_onet_work_activities(settings=s)
    jd = build_job_attributes_jd_text(ctx)
    meta = build_job_attributes_metadata_block(ctx)
    return build_job_attributes_prompt(
        opening_line=llm2_opening_line_from_context(ctx, settings=s),
        occupation=occupation,
        schema=schema,
        onet_work_activities=onet,
        extras=extras,
        job_listing_body=jd,
        job_listing_metadata=meta,
        settings=s,
    )


def run_job_attributes_for_job(
    ctx: JobDescriptionContext,
    *,
    occupation: OccupationPromptEnrichment,
    settings: Settings | None = None,
    schema: JobAttributesSchema | None = None,
    onet: tuple[OnetWorkActivity, ...] | None = None,
    extras: JobAttributesLlm2Extras | None = None,
    log_llm2_prompt: bool = False,
    log_llm2_job_label: str = "",
) -> JobAttributesResult:
    """LLM call 2: structured attribute estimates + summary + ONET work-activity ids."""
    s = settings or get_settings()
    schema_resolved = schema or load_job_attributes_schema(settings=s)
    onet_resolved = onet or load_onet_work_activities(settings=s)
    county = (ctx.county or "").strip()
    city = (ctx.city or "").strip()
    prompt = build_job_attributes_prompt_for_job(
        ctx,
        occupation=occupation,
        extras=extras,
        schema=schema_resolved,
        onet=onet_resolved,
        settings=s,
    )

    if log_llm2_prompt:
        sep = "-" * 80
        label = (log_llm2_job_label or "").strip() or "(job)"
        resolved = resolve_llm2_single_county_user_fields(occupation, extras)
        logger.info("%s", sep)
        logger.info(
            "LLM2 job attributes | job=%s | system=resources/prompts.py | user=client single-county | "
            "onet_rows=%s | schema_attrs=%s | county=%r | city=%r",
            label,
            len(onet_resolved),
            len(schema_resolved.attributes),
            county or None,
            city or None,
        )
        logger.info(
            "LLM2 `occupation` enrichment (JSON):\n%s",
            json.dumps(asdict(occupation), ensure_ascii=False, indent=2),
        )
        if extras is not None:
            logger.info(
                "LLM2 `extras` (JSON):\n%s",
                json.dumps(asdict(extras), ensure_ascii=False, indent=2),
            )
        logger.info(
            "LLM2 resolved single-county user fields (JSON):\n%s",
            json.dumps(
                {
                    "preferred_label": occupation.occupation_label,
                    "county": county,
                    **{k: v for k, v in resolved.items() if v},
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        jd = build_job_attributes_jd_text(ctx)
        meta = build_job_attributes_metadata_block(ctx)
        logger.info("LLM2 `job_listing_body` (%s chars):\n%s", len(jd), jd)
        logger.info("LLM2 `job_listing_metadata` (%s chars):\n%s", len(meta), meta)
        logger.info("LLM2 assembled prompt (%s chars):\n%s", len(prompt), prompt)
        logger.info("%s", sep)

    raw = call_job_attributes_model(prompt, settings=s)
    parsed = parse_job_attributes_response(raw, schema_resolved, onet_resolved, settings=s)
    return replace(parsed, raw_response=raw)


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
    esco_candidates: list[EscoSkillCandidate] | None = None,
    log_llm1_prompt: bool = False,
    log_llm1_job_label: str = "",
    description_by_esco: dict[str, str] | None = None,
) -> tuple[list[str], list[str], MappedEscoSkills, str]:
    """
    Skill ranker: ESCO candidates, JD text, LLM call, JSON parse, label->URI/UUID map.

    Returns essential labels, optional labels, ``MappedEscoSkills``, and raw model text.
    """
    classified_doc = ctx.classified
    candidates = (
        esco_candidates
        if esco_candidates is not None
        else extract_esco_skill_candidates(classified_doc)
    )
    s = settings or get_settings()
    full_details = full_details_from_context(ctx)
    skill_dicts = skills_to_prompt_dicts(candidates, description_by_esco=description_by_esco)
    skills_text = format_skill_ranker_skills_text(skill_dicts)
    prompt = build_skill_ranker_prompt(full_details, skill_dicts, settings=s)

    if log_llm1_prompt:
        sep = "-" * 80
        label = (log_llm1_job_label or "").strip() or "(job)"
        logger.info("%s", sep)
        logger.info(
            "LLM1 skill ranker | job=%s | template=resources/Skills_reranker_prompts.md | candidates=%s",
            label,
            len(skill_dicts),
        )
        logger.info(
            "LLM1 parameter `full_details` (%s chars):\n%s", len(full_details), full_details
        )
        logger.info(
            "LLM1 parameter `potential_skills` (JSON, %s rows):\n%s",
            len(skill_dicts),
            json.dumps(skill_dicts, ensure_ascii=False, indent=2),
        )
        logger.info("LLM1 derived `skills_text` (%s chars):\n%s", len(skills_text), skills_text)
        logger.info("LLM1 assembled user prompt (%s chars):\n%s", len(prompt), prompt)
        logger.info("%s", sep)

    raw = call_skill_ranker_model(prompt, settings=s)
    essential, optional = parse_skill_ranker_json(raw)
    mapped = map_skill_labels_to_esco_uris(essential, optional, candidates)
    return essential, optional, mapped, raw
