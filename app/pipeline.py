"""The per-job pipeline — the one both the batch CLI and the FastAPI endpoint run."""

from __future__ import annotations

import contextvars
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.config import LLM3_MODE_FOLD, Settings, get_settings
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
    job_identifier,
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


class Llm3Outcome(StrEnum):
    """What the LLM3 step did for one job (see ``RERANKER_ENABLE_LLM3`` / ``RERANKER_LLM3_MODE``)."""

    DISABLED = "disabled"
    """``RERANKER_ENABLE_LLM3=0``: the occupation-skills menu was not used."""
    NO_MENU = "no_menu"
    """No essential/optional occupation skills to put on a menu."""
    MENU_FILTERED_OUT = "menu_filtered_out"
    """The ``RERANKER_LLM3_*`` filters left the menu empty."""
    FOLDED = "folded"
    """``fold`` mode: the menu went to LLM1 as extra candidates; no LLM3 call."""
    CALLED = "called"
    """``call`` mode: the standalone LLM3 relevance call ran."""


@dataclass
class JobStats:
    """Per-step counts for one job — what the batch CLI logs; the API leaves them out."""

    classifier_skill_candidates: int = 0
    occupation_blocks: int = 0
    taxonomy_menu_skills: int = 0
    llm3_prompt_skills: int = 0
    llm3_added_llm1_candidates: int = 0
    llm3_decisions: int = 0
    llm1_menu: int = 0
    llm1_essential: int = 0
    llm1_optional: int = 0
    # (essential, optional, unranked) occupation-relation skills, when LLM1 ran and there were any.
    occupation_ranker_counts: tuple[int, int, int] | None = None
    attribute_estimates: int = 0
    work_activity_ids: int = 0
    taxonomy_detail: list[str] = field(default_factory=list)


@dataclass
class JobResult:
    """Output of processing a single classified job document."""

    ranked_document: dict[str, Any] | None
    ctx: JobDescriptionContext
    timings: dict[str, float] = field(default_factory=dict)
    ran_llm1: bool = False
    ran_llm2: bool = False
    ran_llm3: bool = False
    llm3_outcome: Llm3Outcome = Llm3Outcome.DISABLED
    stats: JobStats = field(default_factory=JobStats)


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


def _occupation_ranker_counts(
    occ_uuids: list[str],
    mapped: MappedEscoSkills | None,
) -> tuple[int, int, int]:
    """How many occupation-relation skill uuids LLM1 ranked essential / optional / not at all."""
    if not occ_uuids or mapped is None:
        return 0, 0, len(occ_uuids)
    oset = {u.strip().lower() for u in occ_uuids if u}
    ess = {u.strip().lower() for u in mapped.essential_skills_uuids if u}
    opt = {u.strip().lower() for u in mapped.optional_skills_uuids if u}
    return len(oset & ess), len(oset & opt), len(oset - ess - opt)


def process_single_job(
    classified_doc: dict[str, Any],
    taxonomy: TaxonomyContext,
    settings: Settings | None = None,
    *,
    log_llm1_prompt: bool = False,
    log_llm2_prompt: bool = False,
    log_llm3_prompt: bool = False,
) -> JobResult:
    """
    Run the per-job pipeline on one classified job document: taxonomy → LLM3 → LLM1 + LLM2.

    LLM3 depends on ``RERANKER_ENABLE_LLM3`` and ``RERANKER_LLM3_MODE``: in ``fold`` mode the
    occupation-skills menu becomes extra LLM1 candidates and no LLM3 call is made; in ``call``
    mode the standalone relevance call tags ``occupation_to_skills``. ``JobResult.llm3_outcome``
    says which happened.

    Returns a ``JobResult`` with the ranked payload (ready for MongoDB ``$set`` or API response),
    step timings, and per-step counts. Does NOT write to MongoDB or embed the job — the caller
    decides what to do with the result. The ``log_*_prompt`` flags log each assembled prompt,
    labelled with the job's fingerprint / id.
    """
    settings = settings or get_settings()
    t_total = time.perf_counter()

    ctx = build_job_description_context(classified_doc, settings=settings)
    result = JobResult(ranked_document=None, ctx=ctx)
    stats = result.stats
    label = job_identifier(ctx, classified_doc)

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
    occupation_to_skills_nested = filter_occupation_skills_nested_by_signal_include(
        occupation_to_skills_nested,
        include_tiers=settings.reranker_occupation_skills_signal_include,
    )
    result.timings["taxonomy_s"] = time.perf_counter() - t0
    stats.classifier_skill_candidates = len(base_skills)
    stats.occupation_blocks = len(occupation_to_skills_nested)
    stats.taxonomy_menu_skills = occupation_skills_relevance_menu_size(occupation_to_skills_nested)
    if occ_stats is not None:
        stats.taxonomy_detail = occ_stats.format_log_lines()

    llm3_prompt_nested: list[dict[str, Any]] = []
    if settings.reranker_enable_llm3 and stats.taxonomy_menu_skills > 0:
        llm3_prompt_nested, _, stats.llm3_prompt_skills = build_llm3_occupation_menu(
            occupation_to_skills_nested, base_skills, settings
        )

    llm1_candidates = base_skills
    if not settings.reranker_enable_llm3:
        result.llm3_outcome = Llm3Outcome.DISABLED
    elif stats.taxonomy_menu_skills == 0:
        result.llm3_outcome = Llm3Outcome.NO_MENU
    elif stats.llm3_prompt_skills == 0:
        result.llm3_outcome = Llm3Outcome.MENU_FILTERED_OUT
    elif settings.reranker_llm3_mode == LLM3_MODE_FOLD:
        llm3_extra = llm3_menu_skill_candidates(llm3_prompt_nested, base_skills)
        llm1_candidates = base_skills + llm3_extra
        stats.llm3_added_llm1_candidates = len(llm3_extra)
        result.llm3_outcome = Llm3Outcome.FOLDED
    else:
        t0 = time.perf_counter()
        rel_result = run_occupation_skills_relevance_for_job(
            ctx,
            llm3_prompt_nested,
            settings=settings,
            log_llm3_prompt=log_llm3_prompt,
            log_llm3_job_label=label,
        )
        occupation_to_skills_nested = apply_occupation_skills_listing_relevance(
            occupation_to_skills_nested,
            rel_result,
        )
        result.timings["llm3_s"] = time.perf_counter() - t0
        stats.llm3_decisions = len(rel_result.decisions)
        result.ran_llm3 = True
        result.llm3_outcome = Llm3Outcome.CALLED

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
        log_llm1_prompt=log_llm1_prompt,
        log_llm2_prompt=log_llm2_prompt,
        log_llm1_job_label=label,
        log_llm2_job_label=label,
    )
    stats.llm1_menu = len(llm1_candidates)
    if ran_skill:
        result.timings["llm1_s"] = llm1_s
        result.ran_llm1 = True
        stats.llm1_essential = len(essential)
        stats.llm1_optional = len(optional)
        if skill_mapped is not None and occupation_relation_skill_uuids:
            stats.occupation_ranker_counts = _occupation_ranker_counts(
                occupation_relation_skill_uuids, skill_mapped
            )
    if ran_ja:
        result.timings["llm2_s"] = llm2_s
        result.ran_llm2 = True
        stats.attribute_estimates = len(ja.attribute_estimates)
        stats.work_activity_ids = len(ja.work_activity_ids)

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
