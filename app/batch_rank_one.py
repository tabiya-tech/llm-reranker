"""Single-document ranking step for ``main.py`` (sequential or thread-pool batch)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

from pymongo.operations import UpdateOne

from app.config import FOLD_LLM3_MENU_INTO_LLM1, Settings
from app.database import build_job_description_context, llm_pipeline_upsert_update_one
from app.llm_service import (
    run_occupation_skills_relevance_for_job,
)
from app.logging_config import (
    STEP_LLM1_SKILL_RANKER,
    STEP_LLM2_JOB_ATTRIBUTES,
    STEP_LLM3_OCC_SKILLS,
    STEP_MONGO_UPSERT,
    STEP_OCCUPATION_TAXONOMY,
)
from app.pipeline import TaxonomyContext, run_llm1_llm2_phase
from app.ranked_job_schema import (
    apply_occupation_skills_listing_relevance,
    build_llm3_occupation_menu,
    filter_occupation_skills_nested_by_signal_include,
    llm3_menu_skill_candidates,
    occupation_skills_relevance_menu_size,
)
from app.schemas import JobAttributesResult
from utils.classifier_linked_entities import (
    EscoSkillCandidate,
    MappedEscoSkills,
    extract_esco_skill_candidates,
    group_linked_entities_by_type,
    job_identifier,
)
from utils.occupation_to_skills_mapping import resolve_occupation_taxonomy_for_classified_job

logger = logging.getLogger(__name__)


def _elapsed_s(start: float) -> float:
    return time.perf_counter() - start


def _classified_occupation_labels_for_log(
    classified_doc: dict[str, Any], settings: Settings
) -> str:
    by_t = group_linked_entities_by_type(classified_doc, settings).get("occupation", [])
    if not by_t:
        return "—"
    parts: list[str] = []
    for i, c in enumerate(by_t, start=1):
        lab = (c.label or "").strip() or "(no label)"
        cu = (c.concept_uuid or "").strip()
        parts.append(
            f"{i}:{lab}" + (f" ({cu[:8]}…)" if len(cu) >= 8 else (f" ({cu})" if cu else ""))
        )
    return "; ".join(parts)


def _occupation_ranker_counts(
    occ_uuids: list[str],
    mapped: MappedEscoSkills | None,
) -> tuple[int, int, int]:
    if not occ_uuids or mapped is None:
        return 0, 0, len(occ_uuids)
    oset = {u.strip().lower() for u in occ_uuids if u}
    ess = {u.strip().lower() for u in mapped.essential_skills_uuids if u}
    opt = {u.strip().lower() for u in mapped.optional_skills_uuids if u}
    e = len(oset & ess)
    o = len(oset & opt)
    un = len(oset - ess - opt)
    return e, o, un


@dataclass(frozen=True)
class RankOneJobResult:
    """Return value from :func:`rank_one_classified_document`."""

    update_op: UpdateOne | None
    upsert_increment: int
    llm1_inc: int
    llm2_inc: int
    llm3_inc: int


def rank_one_classified_document(
    doc: dict[str, Any],
    *,
    job_label: str,
    args: argparse.Namespace,
    settings: Settings,
    taxonomy: TaxonomyContext,
    enable_llm3: bool,
) -> RankOneJobResult:
    """
    Run taxonomy → LLM3 → LLM1 → LLM2 → build upsert for one classified document.

    Thread-safe when each call uses a distinct ``doc`` and shared read-only ``taxonomy`` / ``settings``.
    """
    t_job = time.perf_counter()
    ctx = build_job_description_context(doc)
    title = ctx.title or ""
    ident = job_identifier(ctx, doc)
    title_short = (title or "(no title)").strip()
    if len(title_short) > 100:
        title_short = title_short[:97] + "..."

    ran_skill = False
    ran_llm3_occ = False
    ran_ja = False
    essential: list[str] = []
    optional: list[str] = []
    skill_mapped: MappedEscoSkills | None = None
    skill_raw = ""
    ja: JobAttributesResult | None = None
    occupation_relation_skill_uuids: list[str] = []
    occupation_to_skills_nested: list[dict[str, Any]] = []
    occ_stats = None
    base_skills: list[EscoSkillCandidate] | None = None

    log_prompt_debug = args.log_llm1_prompt or args.log_llm2_prompt or args.log_llm3_prompt

    if log_prompt_debug:
        logger.info(
            "---- Job %s | id=%s | title=%s | (prompt-debug mode)",
            job_label,
            ident,
            title_short,
        )
    else:
        occ_preview = _classified_occupation_labels_for_log(doc, settings)
        logger.info(
            "---- Job %s | id=%s | title=%s | classifier_occupations=%s",
            job_label,
            ident,
            title_short,
            occ_preview,
        )

    t0 = time.perf_counter()
    base_skills = extract_esco_skill_candidates(doc)
    (
        occupation_relation_skill_uuids,
        occ_stats,
        occupation_to_skills_nested,
    ) = resolve_occupation_taxonomy_for_classified_job(
        doc,
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
    n_blocks = len(occupation_to_skills_nested)
    n_llm3_full = occupation_skills_relevance_menu_size(occupation_to_skills_nested)
    if enable_llm3 and n_llm3_full > 0:
        llm3_prompt_nested, _, llm3_prompt_n = build_llm3_occupation_menu(
            occupation_to_skills_nested, base_skills, settings
        )
    else:
        llm3_prompt_nested = []
        llm3_prompt_n = 0
    if not log_prompt_debug:
        if not enable_llm3:
            llm3_note = " | llm3=skipped"
        elif n_llm3_full == 0:
            llm3_note = ""
        else:
            llm3_note = f" llm3_prompt_skills={llm3_prompt_n}"
        logger.info(
            "  step | %s | %.3fs | classifier_skill_candidates=%s occupation_blocks=%s "
            "taxonomy_occupation_skills=%s%s",
            STEP_OCCUPATION_TAXONOMY,
            _elapsed_s(t0),
            len(base_skills),
            n_blocks,
            n_llm3_full,
            llm3_note,
        )
    if occ_stats is not None:
        lines = occ_stats.format_log_lines()
        if lines:
            logger.debug("  taxonomy detail | %s", lines[0])
            for line in lines[1:]:
                logger.debug("%s", line)

    llm3_s = 0.0
    llm3_inc = 0
    llm1_candidates = base_skills
    if FOLD_LLM3_MENU_INTO_LLM1 and enable_llm3 and llm3_prompt_n > 0:
        llm3_extra = llm3_menu_skill_candidates(llm3_prompt_nested, base_skills)
        llm1_candidates = base_skills + llm3_extra
        if not log_prompt_debug:
            logger.info(
                "  step | %s | folded into LLM1 | llm3_prompt_skills=%s added_llm1_candidates=%s",
                STEP_LLM3_OCC_SKILLS,
                llm3_prompt_n,
                len(llm3_extra),
            )
    elif enable_llm3 and occupation_to_skills_nested and llm3_prompt_n > 0:
        if not log_prompt_debug:
            if llm3_prompt_n != n_llm3_full:
                logger.info(
                    "  step | %s | starting | taxonomy_menu_skills=%s llm3_prompt_skills=%s",
                    STEP_LLM3_OCC_SKILLS,
                    n_llm3_full,
                    llm3_prompt_n,
                )
            else:
                logger.info(
                    "  step | %s | starting | menu_skills=%s",
                    STEP_LLM3_OCC_SKILLS,
                    llm3_prompt_n,
                )
        t0 = time.perf_counter()
        rel_result = run_occupation_skills_relevance_for_job(
            ctx,
            llm3_prompt_nested,
            settings=settings,
            log_llm3_prompt=args.log_llm3_prompt,
            log_llm3_job_label=ident,
        )
        occupation_to_skills_nested = apply_occupation_skills_listing_relevance(
            occupation_to_skills_nested,
            rel_result,
        )
        llm3_s = _elapsed_s(t0)
        llm3_inc = 1
        ran_llm3_occ = True
        if not log_prompt_debug:
            logger.info(
                "  step | %s | %.3fs | decisions=%s",
                STEP_LLM3_OCC_SKILLS,
                llm3_s,
                len(rel_result.decisions),
            )
    elif not enable_llm3 and not log_prompt_debug:
        logger.info("  step | %s | skipped | RERANKER_ENABLE_LLM3=0", STEP_LLM3_OCC_SKILLS)
    elif not log_prompt_debug:
        if occupation_to_skills_nested and n_llm3_full > 0 and llm3_prompt_n == 0:
            logger.info(
                "  step | %s | skipped | llm3 prompt menu empty after filters "
                "(taxonomy_menu=%s; try RERANKER_LLM3_CLASSIFIER_INTERSECT=0 or more classifier skills)",
                STEP_LLM3_OCC_SKILLS,
                n_llm3_full,
            )
        else:
            logger.info(
                "  step | %s | skipped | no essential/optional occupation skills to judge",
                STEP_LLM3_OCC_SKILLS,
            )

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
        doc,
        settings,
        taxonomy,
        llm1_candidates,
        overlap_llm12=settings.reranker_overlap_llm1_llm2,
        log_llm1_prompt=args.log_llm1_prompt,
        log_llm2_prompt=args.log_llm2_prompt,
        log_llm1_job_label=ident,
        log_llm2_job_label=ident,
    )
    llm1_inc = 1 if ran_skill else 0
    if not log_prompt_debug:
        if ran_skill:
            logger.info(
                "  step | %s | %.3fs | menu=%s essential=%s optional=%s",
                STEP_LLM1_SKILL_RANKER,
                llm1_s,
                len(llm1_candidates),
                len(essential),
                len(optional),
            )
        else:
            logger.info(
                "  step | %s | skipped | no ESCO skill candidates for job",
                STEP_LLM1_SKILL_RANKER,
            )

    if ran_skill and skill_mapped is not None and not log_prompt_debug:
        if occupation_relation_skill_uuids:
            oe, oo, ou = _occupation_ranker_counts(occupation_relation_skill_uuids, skill_mapped)
            logger.debug(
                "  ranker vs occupation-relation UUIDs | occ_essential=%s occ_optional=%s occ_unranked=%s",
                oe,
                oo,
                ou,
            )
        else:
            logger.debug(
                "  ranker | total_essential=%s total_optional=%s (no occupation-relation UUIDs)",
                len(essential),
                len(optional),
            )

    if not log_prompt_debug and ran_ja:
        logger.info(
            "  step | %s | %.3fs | attribute_estimates=%s work_activity_ids=%s",
            STEP_LLM2_JOB_ATTRIBUTES,
            llm2_s,
            len(ja.attribute_estimates),
            len(ja.work_activity_ids),
        )

    mongo_s = 0.0
    upsert_queued = False
    update_op: UpdateOne | None = None
    upsert_increment = 0
    if ran_skill or ran_ja:
        t0 = time.perf_counter()
        try:
            update_op = llm_pipeline_upsert_update_one(
                doc,
                ctx,
                settings=settings,
                ja=ja if ran_ja and ja is not None else None,
                skill_ranker=(essential, optional, skill_mapped, skill_raw)
                if ran_skill and skill_mapped is not None
                else None,
                occupation_to_skills=occupation_to_skills_nested or None,
                skills_uuid_tabiya_map=taxonomy.skills_uuid_tabiya_map,
                occupations_bridge=taxonomy.occ_bridge or None,
                occupations_all_ids=taxonomy.occ_all_ids or None,
                ranker_candidates=llm1_candidates,
                classifier_skill_candidates=base_skills,
            )
            mongo_s = _elapsed_s(t0)
            if update_op is not None:
                upsert_increment = 1
                upsert_queued = True
        except ValueError as e:
            mongo_s = _elapsed_s(t0)
            logger.warning("Upsert skipped: %s", e)
        if not log_prompt_debug:
            logger.info(
                "  step | %s | %.3fs | queued=%s",
                STEP_MONGO_UPSERT,
                mongo_s,
                "yes" if upsert_queued else "no",
            )

    if not log_prompt_debug:
        logger.info(
            "  job complete | total_s=%.3fs | llm1=%.3fs llm2=%.3fs llm3=%.3fs mongo=%.3fs",
            _elapsed_s(t_job),
            llm1_s,
            llm2_s,
            llm3_s,
            mongo_s,
        )
    else:
        logger.info("  job complete | total_s=%.3fs (see LLM logs above)", _elapsed_s(t_job))

    if args.verbose and not log_prompt_debug:
        parts: list[str] = []
        if ran_skill:
            parts.append(f"LLM1: ess={len(essential)} opt={len(optional)}")
        if ran_llm3_occ:
            parts.append("LLM3: occ-skill relevance")
        if ran_ja and ja is not None:
            parts.append(
                f"LLM2: attrs={len(ja.attribute_estimates)} w_acts={len(ja.work_activity_ids)}"
            )
        if parts:
            print("  " + " | ".join(parts), file=sys.stderr)

    return RankOneJobResult(
        update_op=update_op,
        upsert_increment=upsert_increment,
        llm1_inc=llm1_inc,
        llm2_inc=1 if ran_ja else 0,
        llm3_inc=llm3_inc,
    )
