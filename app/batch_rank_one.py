"""Single-document ranking step for ``main.py`` (sequential or thread-pool batch).

Runs the same per-job pipeline as the API (:func:`app.pipeline.process_single_job`) and adds
only the batch's per-step log lines and the MongoDB upsert (with the job embedding).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

from pymongo.operations import UpdateOne

from app.config import Settings
from app.job_context import build_job_description_context
from app.logging_config import (
    STEP_LLM1_SKILL_RANKER,
    STEP_LLM2_JOB_ATTRIBUTES,
    STEP_LLM3_OCC_SKILLS,
    STEP_MONGO_UPSERT,
    STEP_OCCUPATION_TAXONOMY,
)
from app.pipeline import JobResult, Llm3Outcome, TaxonomyContext, process_single_job
from app.ranked_jobs.upsert import ranked_job_upsert_update_one
from utils.classifier_linked_entities import group_linked_entities_by_type, job_identifier

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


@dataclass(frozen=True)
class RankOneJobResult:
    """Return value from :func:`rank_one_classified_document`."""

    update_op: UpdateOne | None
    upsert_increment: int
    llm1_inc: int
    llm2_inc: int
    llm3_inc: int


def _log_pipeline_steps(result: JobResult) -> None:
    """Per-step INFO lines for one job, from what :func:`process_single_job` reports."""
    st = result.stats
    outcome = result.llm3_outcome
    if outcome is Llm3Outcome.DISABLED:
        llm3_note = " | llm3=skipped"
    elif outcome is Llm3Outcome.NO_MENU:
        llm3_note = ""
    else:
        llm3_note = f" llm3_prompt_skills={st.llm3_prompt_skills}"
    logger.info(
        "  step | %s | %.3fs | classifier_skill_candidates=%s occupation_blocks=%s "
        "taxonomy_occupation_skills=%s%s",
        STEP_OCCUPATION_TAXONOMY,
        result.timings.get("taxonomy_s", 0.0),
        st.classifier_skill_candidates,
        st.occupation_blocks,
        st.taxonomy_menu_skills,
        llm3_note,
    )
    if st.taxonomy_detail:
        logger.debug("  taxonomy detail | %s", st.taxonomy_detail[0])
        for line in st.taxonomy_detail[1:]:
            logger.debug("%s", line)

    if outcome is Llm3Outcome.FOLDED:
        logger.info(
            "  step | %s | folded into LLM1 | llm3_prompt_skills=%s added_llm1_candidates=%s",
            STEP_LLM3_OCC_SKILLS,
            st.llm3_prompt_skills,
            st.llm3_added_llm1_candidates,
        )
    elif outcome is Llm3Outcome.CALLED:
        logger.info(
            "  step | %s | %.3fs | taxonomy_menu_skills=%s llm3_prompt_skills=%s decisions=%s",
            STEP_LLM3_OCC_SKILLS,
            result.timings.get("llm3_s", 0.0),
            st.taxonomy_menu_skills,
            st.llm3_prompt_skills,
            st.llm3_decisions,
        )
    elif outcome is Llm3Outcome.DISABLED:
        logger.info("  step | %s | skipped | RERANKER_ENABLE_LLM3=0", STEP_LLM3_OCC_SKILLS)
    elif outcome is Llm3Outcome.MENU_FILTERED_OUT:
        logger.info(
            "  step | %s | skipped | llm3 prompt menu empty after filters "
            "(taxonomy_menu=%s; try RERANKER_LLM3_CLASSIFIER_INTERSECT=0 or more classifier skills)",
            STEP_LLM3_OCC_SKILLS,
            st.taxonomy_menu_skills,
        )
    else:
        logger.info(
            "  step | %s | skipped | no essential/optional occupation skills to judge",
            STEP_LLM3_OCC_SKILLS,
        )

    if result.ran_llm1:
        logger.info(
            "  step | %s | %.3fs | menu=%s essential=%s optional=%s",
            STEP_LLM1_SKILL_RANKER,
            result.timings.get("llm1_s", 0.0),
            st.llm1_menu,
            st.llm1_essential,
            st.llm1_optional,
        )
        if st.occupation_ranker_counts is not None:
            oe, oo, ou = st.occupation_ranker_counts
            logger.debug(
                "  ranker vs occupation-relation UUIDs | occ_essential=%s occ_optional=%s occ_unranked=%s",
                oe,
                oo,
                ou,
            )
        else:
            logger.debug(
                "  ranker | total_essential=%s total_optional=%s (no occupation-relation UUIDs)",
                st.llm1_essential,
                st.llm1_optional,
            )
    else:
        logger.info(
            "  step | %s | skipped | no ESCO skill candidates for job",
            STEP_LLM1_SKILL_RANKER,
        )

    if result.ran_llm2:
        logger.info(
            "  step | %s | %.3fs | attribute_estimates=%s work_activity_ids=%s",
            STEP_LLM2_JOB_ATTRIBUTES,
            result.timings.get("llm2_s", 0.0),
            st.attribute_estimates,
            st.work_activity_ids,
        )


def rank_one_classified_document(
    doc: dict[str, Any],
    *,
    job_label: str,
    args: argparse.Namespace,
    settings: Settings,
    taxonomy: TaxonomyContext,
) -> RankOneJobResult:
    """
    Run :func:`process_single_job` on one classified document, log its steps, and build the upsert.

    Thread-safe when each call uses a distinct ``doc`` and shared read-only ``taxonomy`` / ``settings``.
    """
    t_job = time.perf_counter()
    log_prompt_debug = args.log_llm1_prompt or args.log_llm2_prompt or args.log_llm3_prompt

    ctx = build_job_description_context(doc, settings=settings)
    ident = job_identifier(ctx, doc)
    title_short = (ctx.title or "(no title)").strip()
    if len(title_short) > 100:
        title_short = title_short[:97] + "..."
    if log_prompt_debug:
        logger.info(
            "---- Job %s | id=%s | title=%s | (prompt-debug mode)",
            job_label,
            ident,
            title_short,
        )
    else:
        logger.info(
            "---- Job %s | id=%s | title=%s | classifier_occupations=%s",
            job_label,
            ident,
            title_short,
            _classified_occupation_labels_for_log(doc, settings),
        )

    result = process_single_job(
        doc,
        taxonomy,
        settings=settings,
        log_llm1_prompt=args.log_llm1_prompt,
        log_llm2_prompt=args.log_llm2_prompt,
        log_llm3_prompt=args.log_llm3_prompt,
    )
    if not log_prompt_debug:
        _log_pipeline_steps(result)

    mongo_s = 0.0
    update_op: UpdateOne | None = None
    if result.ranked_document is None:
        logger.warning(
            "MongoDB upsert skipped: ranked job payload is empty (job attributes / LLM2 result missing)."
        )
    else:
        t0 = time.perf_counter()
        try:
            update_op = ranked_job_upsert_update_one(
                doc, result.ctx, result.ranked_document, settings=settings
            )
        except ValueError as e:
            logger.warning("Upsert skipped: %s", e)
        mongo_s = _elapsed_s(t0)
        if not log_prompt_debug:
            logger.info(
                "  step | %s | %.3fs | queued=%s",
                STEP_MONGO_UPSERT,
                mongo_s,
                "yes" if update_op is not None else "no",
            )

    if log_prompt_debug:
        logger.info("  job complete | total_s=%.3fs (see LLM logs above)", _elapsed_s(t_job))
    else:
        logger.info(
            "  job complete | total_s=%.3fs | llm1=%.3fs llm2=%.3fs llm3=%.3fs mongo=%.3fs",
            _elapsed_s(t_job),
            result.timings.get("llm1_s", 0.0),
            result.timings.get("llm2_s", 0.0),
            result.timings.get("llm3_s", 0.0),
            mongo_s,
        )

    if args.verbose and not log_prompt_debug:
        st = result.stats
        parts: list[str] = []
        if result.ran_llm1:
            parts.append(f"LLM1: ess={st.llm1_essential} opt={st.llm1_optional}")
        if result.ran_llm3:
            parts.append("LLM3: occ-skill relevance")
        if result.ran_llm2:
            parts.append(f"LLM2: attrs={st.attribute_estimates} w_acts={st.work_activity_ids}")
        if parts:
            print("  " + " | ".join(parts), file=sys.stderr)

    return RankOneJobResult(
        update_op=update_op,
        upsert_increment=1 if update_op is not None else 0,
        llm1_inc=int(result.ran_llm1),
        llm2_inc=int(result.ran_llm2),
        llm3_inc=int(result.ran_llm3),
    )
