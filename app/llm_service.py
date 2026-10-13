"""The reranker's three LLM passes: skill ranking, job attributes, occupation-skill relevance.

Prompt assembly and response parsing live here; *which model answers* does not. Every
call goes through :class:`app.llm.base.LlmClient`, loaded from configuration by
:mod:`app.llm.factory`, so pointing a deployment at Claude, GPT, Qwen or a local
server changes no code in this module.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from typing import Any

from app.config import Settings, get_settings
from app.job_context import JobDescriptionContext
from app.llm import LlmClient, get_llm_client
from app.observability import traced_observation, update_observation
from app.prompts import (
    JobAttributesLlm2Extras,
    OccupationPromptEnrichment,
    build_job_attributes_jd_text,
    build_job_attributes_metadata_block,
    build_job_attributes_prompt,
    build_occupation_skills_relevance_prompt,
    build_skill_ranker_prompt,
    format_skill_ranker_skills_text,
    llm2_opening_line_from_context,
    resolve_llm2_single_county_user_fields,
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

# Labels passed to the client as ``purpose`` — they identify the pass in provider logs
# and error messages, and are the same strings used in ``classifier_metadata``.
PURPOSE_SKILL_RANKER = "skill_ranker"
PURPOSE_JOB_ATTRIBUTES = "job_attributes"
PURPOSE_OCCUPATION_SKILLS_RELEVANCE = "occupation_skills_relevance"


def llm_client(settings: Settings | None = None) -> LlmClient:
    """The configured model for this run (built once per process, per configuration)."""
    s = settings or get_settings()
    return get_llm_client(s.llm)


def _generate_json(prompt: str, *, purpose: str, settings: Settings) -> str:
    """Ask the configured model for JSON; retries and error translation are the client's job.

    Every LLM pass goes through here, which is why the Langfuse generation lives here
    rather than in each provider. Providers only add the token usage they parse out of
    their response (``app.observability.report_usage``).
    """
    client = llm_client(settings)
    cfg = settings.llm
    with traced_observation(
        name=purpose or "generate_json",
        as_type="generation",
        input=prompt,
        model=client.model,
        model_parameters={
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_output_tokens or None,
        },
        metadata={"purpose": purpose, "provider": cfg.provider, "client_class": cfg.client_class},
    ) as generation:
        try:
            out = client.generate_json(prompt, purpose=purpose)
        except Exception as e:
            update_observation(generation, level="ERROR", status_message=f"{type(e).__name__}: {e}")
            raise
        update_observation(generation, output=out)
        return out


def call_skill_ranker_model(prompt: str, *, settings: Settings | None = None) -> str:
    """LLM call 1 — skill ranker; expect JSON."""
    s = settings or get_settings()
    logger.debug("LLM call 1 (skill ranker): invoking %s", s.llm.describe())
    out = _generate_json(prompt, purpose=PURPOSE_SKILL_RANKER, settings=s)
    logger.debug("LLM call 1 (skill ranker): completed")
    return out


def call_job_attributes_model(prompt: str, *, settings: Settings | None = None) -> str:
    """LLM call 2 — job attributes; expect JSON with ``attributes`` and ``work_activities``."""
    s = settings or get_settings()
    logger.debug("LLM call 2 (job attributes): invoking %s", s.llm.describe())
    out = _generate_json(prompt, purpose=PURPOSE_JOB_ATTRIBUTES, settings=s)
    logger.debug("LLM call 2 (job attributes): completed")
    return out


def call_occupation_skills_relevance_model(prompt: str, *, settings: Settings | None = None) -> str:
    """LLM call 3 — occupation-skills listing relevance; expect valid JSON."""
    s = settings or get_settings()
    logger.debug("LLM call 3 (occupation skills relevance): invoking %s", s.llm.describe())
    out = _generate_json(prompt, purpose=PURPOSE_OCCUPATION_SKILLS_RELEVANCE, settings=s)
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
