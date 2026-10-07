"""Shared per-job processing core — used by both the batch CLI and the FastAPI endpoint."""

from __future__ import annotations

import contextvars
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from app.config import (
    FOLD_LLM3_MENU_INTO_LLM1,
    Settings,
    get_settings,
    reranker_enable_llm3_from_env,
)
from app.database import build_job_description_context
from app.job_context import JobDescriptionContext
from app.llm_service import (
    run_job_attributes_for_job,
    run_occupation_skills_relevance_for_job,
    run_skill_ranker_for_job,
)
from app.prompts import build_occupation_prompt_enrichment
from app.ranked_job_schema import (
    apply_occupation_skills_listing_relevance,
    build_llm3_occupation_menu,
    build_ranked_job_document,
    filter_occupation_skills_nested_by_signal_include,
    llm3_menu_skill_candidates,
    occupation_skills_relevance_menu_size,
)
from app.schemas import JobAttributesResult
from utils.classifier_linked_entities import (
    EscoSkillCandidate,
    MappedEscoSkills,
    extract_esco_skill_candidates,
)
from utils.occupation_to_skills_mapping import resolve_occupation_taxonomy_for_classified_job

logger = logging.getLogger(__name__)


@dataclass
class TaxonomyContext:
    """Pre-loaded CSV / lookup data shared across all jobs in a process."""

    occ_map: dict[str, frozenset[str]]
    csv_skill_labels: dict[str, str] | None
    occ_skill_relation: dict[str, dict[str, str]] | None
    occ_skill_signalling: dict[str, dict[str, dict[str, str]]] | None
    occ_bridge: dict[str, str]
    occ_all_ids: frozenset[str]
    skills_uuid_tabiya_map: Any
    skill_desc_by_esco: dict[str, str]
    occ_label_by_esco: dict[str, str]
    occ_desc_by_esco: dict[str, str]


@dataclass
class JobResult:
    """Output of processing a single classified job document."""

    ranked_document: dict[str, Any] | None
    timings: dict[str, float] = field(default_factory=dict)
    ran_llm1: bool = False
    ran_llm2: bool = False
    ran_llm3: bool = False


def run_llm1_llm2_phase(
    ctx: JobDescriptionContext,
    classified_doc: dict[str, Any],
    settings: Settings,
    taxonomy: TaxonomyContext,
    base_skills: list[EscoSkillCandidate],
    *,
    overlap_llm12: bool,
    log_llm1_prompt: bool = False,
    log_llm2_prompt: bool = False,
    log_llm1_job_label: str = "",
    log_llm2_job_label: str = "",
) -> tuple[
    list[str],
    list[str],
    MappedEscoSkills | None,
    str,
    JobAttributesResult,
    float,
    float,
    bool,
    bool,
]:
    """
    Run skill ranker (LLM1) and job attributes (LLM2).

    When ``overlap_llm12`` and there are classifier skills and no prompt-debug logging, the
    two Gemini calls run concurrently (~max latency instead of sum). Beware combined with
    ``RERANKER_PARALLEL_WORKERS`` — total in-flight requests can be workers×2.
    """
    essential: list[str] = []
    optional: list[str] = []
    skill_mapped: MappedEscoSkills | None = None
    skill_raw = ""
    llm1_s = 0.0
    llm2_s = 0.0
    ran_skill = False

    def _llm1() -> tuple[float, list[str], list[str], MappedEscoSkills | None, str]:
        t0 = time.perf_counter()
        e, o, sm, sr = run_skill_ranker_for_job(
            ctx,
            settings=settings,
            esco_candidates=base_skills,
            description_by_esco=taxonomy.skill_desc_by_esco or None,
            log_llm1_prompt=log_llm1_prompt,
            log_llm1_job_label=log_llm1_job_label,
        )
        return time.perf_counter() - t0, e, o, sm, sr

    def _llm2() -> tuple[float, JobAttributesResult]:
        t0 = time.perf_counter()
        occ_enrich = build_occupation_prompt_enrichment(
            classified_doc,
            settings,
            fallback_listing_title=(ctx.title or "").strip() or "Unknown role",
            occupations_bridge=taxonomy.occ_bridge,
            occupations_all_ids=taxonomy.occ_all_ids,
            label_by_esco=taxonomy.occ_label_by_esco,
            description_by_esco=taxonomy.occ_desc_by_esco,
        )
        j = run_job_attributes_for_job(
            ctx,
            settings=settings,
            occupation=occ_enrich,
            log_llm2_prompt=log_llm2_prompt,
            log_llm2_job_label=log_llm2_job_label,
        )
        return time.perf_counter() - t0, j

    use_overlap = (
        overlap_llm12 and bool(base_skills) and not log_llm1_prompt and not log_llm2_prompt
    )

    if use_overlap:
        with ThreadPoolExecutor(max_workers=2) as pool:
            # Each call runs in a copy of this thread's context, so its Langfuse
            # generation nests under the job's trace instead of starting a new one.
            # One copy per call: a Context cannot be entered by two threads at once.
            f1 = pool.submit(contextvars.copy_context().run, _llm1)
            f2 = pool.submit(contextvars.copy_context().run, _llm2)
            llm1_s, essential, optional, skill_mapped, skill_raw = f1.result()
            llm2_s, ja = f2.result()
        ran_skill = True
        return (
            essential,
            optional,
            skill_mapped,
            skill_raw,
            ja,
            llm1_s,
            llm2_s,
            True,
            True,
        )

    if base_skills:
        llm1_s, essential, optional, skill_mapped, skill_raw = _llm1()
        ran_skill = True
    llm2_s, ja = _llm2()
    return (
        essential,
        optional,
        skill_mapped,
        skill_raw,
        ja,
        llm1_s,
        llm2_s,
        ran_skill,
        True,
    )


def load_taxonomy_context(settings: Settings | None = None) -> TaxonomyContext:
    """Load all CSV / lookup data once per process. Results are lru_cache'd internally."""
    from utils.occupation_description import load_occupation_labels_and_descriptions_by_esco
    from utils.occupation_to_skills_mapping import (
        load_occupations_lookup,
        occupation_skill_taxonomy_for_settings,
    )
    from utils.skill_descriptions import load_skill_description_by_esco
    from utils.uuid_to_tabiya_id import load_skills_uuid_tabiya_map

    settings = settings or get_settings()

    occ_map, csv_skill_labels, occ_skill_relation, occ_skill_signalling = (
        occupation_skill_taxonomy_for_settings(settings)
    )
    occ_bridge, occ_all_ids = load_occupations_lookup(str(settings.occupations_csv_path.resolve()))
    skills_uuid_tabiya_map = load_skills_uuid_tabiya_map(str(settings.skills_csv_path.resolve()))
    skill_desc_by_esco: dict[str, str] = {}
    if (
        settings.skills_csv_include_descriptions
        and (settings.skills_csv_description_column or "").strip()
    ):
        skill_desc_by_esco = load_skill_description_by_esco(
            str(settings.skills_csv_path.resolve()),
            settings.skills_csv_description_column.strip(),
        )
    occ_label_by_esco, occ_desc_by_esco = load_occupation_labels_and_descriptions_by_esco(
        str(settings.occupations_csv_path.resolve()),
        label_column=settings.occupations_csv_label_column,
        include_descriptions=settings.occupations_csv_include_descriptions,
        description_column=(settings.occupations_csv_description_column or "").strip() or None,
    )
    return TaxonomyContext(
        occ_map=occ_map,
        csv_skill_labels=csv_skill_labels,
        occ_skill_relation=occ_skill_relation,
        occ_skill_signalling=occ_skill_signalling,
        occ_bridge=occ_bridge,
        occ_all_ids=occ_all_ids,
        skills_uuid_tabiya_map=skills_uuid_tabiya_map,
        skill_desc_by_esco=skill_desc_by_esco,
        occ_label_by_esco=occ_label_by_esco,
        occ_desc_by_esco=occ_desc_by_esco,
    )


def process_single_job(
    classified_doc: dict[str, Any],
    taxonomy: TaxonomyContext,
    settings: Settings | None = None,
) -> JobResult:
    """
    Run the full LLM pipeline (1 → 3 → 2) on one classified job document.

    Returns a ``JobResult`` with the ranked payload (ready for MongoDB ``$set`` or API response)
    and step timings. Does NOT write to MongoDB — the caller decides what to do with the result.
    """
    settings = settings or get_settings()
    t_total = time.perf_counter()
    result = JobResult(ranked_document=None)
    enable_llm3 = reranker_enable_llm3_from_env()

    ctx = build_job_description_context(classified_doc, settings=settings)

    t0 = time.perf_counter()
    base_skills: list[EscoSkillCandidate] = extract_esco_skill_candidates(classified_doc)
    (
        occupation_relation_skill_uuids,
        occ_stats,
        occupation_to_skills_nested,
    ) = resolve_occupation_taxonomy_for_classified_job(
        classified_doc,
        base_skills,
        taxonomy.occ_map,
        taxonomy.csv_skill_labels,
        taxonomy.occ_skill_relation,
        taxonomy.skills_uuid_tabiya_map,
        occupation_skill_signalling=taxonomy.occ_skill_signalling,
        settings=settings,
    )
    result.timings["taxonomy_s"] = time.perf_counter() - t0

    occupation_to_skills_nested = filter_occupation_skills_nested_by_signal_include(
        occupation_to_skills_nested,
        include_tiers=settings.reranker_occupation_skills_signal_include,
    )

    llm3_prompt_nested: list[dict[str, Any]] = []
    llm3_prompt_n = 0
    n_llm3_menu = occupation_skills_relevance_menu_size(occupation_to_skills_nested)
    if enable_llm3 and n_llm3_menu > 0:
        llm3_prompt_nested, _, llm3_prompt_n = build_llm3_occupation_menu(
            occupation_to_skills_nested, base_skills, settings
        )
    llm1_candidates = base_skills
    if FOLD_LLM3_MENU_INTO_LLM1 and enable_llm3 and llm3_prompt_n > 0:
        llm1_candidates = base_skills + llm3_menu_skill_candidates(llm3_prompt_nested, base_skills)
    elif enable_llm3 and occupation_to_skills_nested and llm3_prompt_n > 0:
        t0 = time.perf_counter()
        rel_result = run_occupation_skills_relevance_for_job(
            ctx,
            llm3_prompt_nested,
            settings=settings,
        )
        occupation_to_skills_nested = apply_occupation_skills_listing_relevance(
            occupation_to_skills_nested,
            rel_result,
        )
        result.timings["llm3_s"] = time.perf_counter() - t0
        result.ran_llm3 = True

    (
        essential,
        optional,
        skill_mapped,
        skill_raw,
        ja,
        llm1_s,
        llm2_s,
        ran_skill,
        ran_ja,
    ) = run_llm1_llm2_phase(
        ctx,
        classified_doc,
        settings,
        taxonomy,
        llm1_candidates,
        overlap_llm12=settings.reranker_overlap_llm1_llm2,
    )
    if ran_skill:
        result.timings["llm1_s"] = llm1_s
        result.ran_llm1 = True
    if ran_ja:
        result.timings["llm2_s"] = llm2_s
        result.ran_llm2 = True

    ranked = build_ranked_job_document(
        classified_doc,
        ctx,
        settings=settings,
        ja=ja,
        skill_ranker=(essential, optional, skill_mapped, skill_raw)
        if skill_mapped is not None
        else None,
        occupation_to_skills=occupation_to_skills_nested or None,
        skills_uuid_tabiya_map=taxonomy.skills_uuid_tabiya_map,
        occupations_bridge=taxonomy.occ_bridge or None,
        occupations_all_ids=taxonomy.occ_all_ids or None,
        ranker_candidates=llm1_candidates,
        classifier_skill_candidates=base_skills,
    )
    result.ranked_document = ranked
    result.timings["total_s"] = time.perf_counter() - t_total
    return result
