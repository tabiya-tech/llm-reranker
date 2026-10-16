"""Single-document ranking step for the batch (sequential or thread-pool).

Runs the same per-job pipeline as the API (:class:`app.job_ranking.service.JobRankingService`)
and adds only the batch's per-step log lines, the job embedding and the MongoDB upsert.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.job_context import build_job_description_context
from app.job_embedding.service import JobEmbeddingService
from app.job_ranking.service import JobRankingService
from app.job_ranking.types import NO_PROMPT_LOGGING, JobResult, Llm3Outcome, PromptLogging
from app.llm.base import LlmConfig
from app.logging_config import (
    STEP_MONGO_UPSERT,
    STEP_OCCUPATION_TAXONOMY,
    step_llm1_skill_ranker,
    step_llm2_job_attributes,
    step_llm3_occ_skills,
)
from app.ranked_jobs.types import RankedJobUpsert
from app.ranked_jobs.upsert import ranked_job_upsert
from utils.classifier_linked_entities import group_linked_entities_by_type, job_identifier

logger = logging.getLogger(__name__)


def elapsed_s(start: float) -> float:
    """Seconds since ``start`` (a ``time.perf_counter()`` reading)."""
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
    """Return value from :meth:`BatchJobRanker.rank`."""

    upsert: RankedJobUpsert | None
    upsert_increment: int
    llm1_inc: int
    llm2_inc: int
    llm3_inc: int


def _log_pipeline_steps(result: JobResult, *, llm: LlmConfig) -> None:
    """Per-step INFO lines for one job, from what :meth:`JobRankingService.rank` reports."""
    st = result.stats
    outcome = result.llm3_outcome
    l1 = step_llm1_skill_ranker(llm)
    l2 = step_llm2_job_attributes(llm)
    l3 = step_llm3_occ_skills(llm)
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
            l3,
            st.llm3_prompt_skills,
            st.llm3_added_llm1_candidates,
        )
    elif outcome is Llm3Outcome.CALLED:
        logger.info(
            "  step | %s | %.3fs | taxonomy_menu_skills=%s llm3_prompt_skills=%s decisions=%s",
            l3,
            result.timings.get("llm3_s", 0.0),
            st.taxonomy_menu_skills,
            st.llm3_prompt_skills,
            st.llm3_decisions,
        )
    elif outcome is Llm3Outcome.DISABLED:
        logger.info("  step | %s | skipped | RERANKER_ENABLE_LLM3=0", l3)
    elif outcome is Llm3Outcome.MENU_FILTERED_OUT:
        logger.info(
            "  step | %s | skipped | llm3 prompt menu empty after filters "
            "(taxonomy_menu=%s; try RERANKER_LLM3_CLASSIFIER_INTERSECT=0 or more classifier skills)",
            l3,
            st.taxonomy_menu_skills,
        )
    else:
        logger.info(
            "  step | %s | skipped | no essential/optional occupation skills to judge",
            l3,
        )

    if result.ran_llm1:
        logger.info(
            "  step | %s | %.3fs | menu=%s essential=%s optional=%s",
            l1,
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
            l1,
        )

    if result.ran_llm2:
        logger.info(
            "  step | %s | %.3fs | attribute_estimates=%s work_activity_ids=%s",
            l2,
            result.timings.get("llm2_s", 0.0),
            st.attribute_estimates,
            st.work_activity_ids,
        )


@dataclass(frozen=True)
class BatchJobOptions:
    """Per-job switches from the batch CLI."""

    prompt_logging: PromptLogging = NO_PROMPT_LOGGING
    # ``--verbose``: extra per-job counts on stderr.
    verbose: bool = False


class BatchJobRanker:
    """Ranks one classified document for the batch, logs its steps and builds its upsert.

    Thread-safe when each call uses a distinct ``doc``: the ranking service, embedding service
    and settings are shared read-only.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        ranking: JobRankingService,
        embedding: JobEmbeddingService,
        options: BatchJobOptions,
    ):
        self._settings = settings
        self._ranking = ranking
        self._embedding = embedding
        self._options = options

    def rank(self, doc: dict[str, Any], *, job_label: str) -> RankOneJobResult:
        settings = self._settings
        prompt_logging = self._options.prompt_logging
        log_prompt_debug = prompt_logging.any
        t_job = time.perf_counter()

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

        result = self._ranking.rank(doc, prompt_logging=prompt_logging)
        if not log_prompt_debug:
            _log_pipeline_steps(result, llm=settings.llm)

        mongo_s = 0.0
        upsert: RankedJobUpsert | None = None
        if result.ranked_document is None:
            logger.warning(
                "MongoDB upsert skipped: ranked job payload is empty (job attributes / LLM2 result missing)."
            )
        else:
            t0 = time.perf_counter()
            try:
                upsert = ranked_job_upsert(
                    doc, result.ctx, self._embedding.attach(result.ranked_document)
                )
            except ValueError as e:
                logger.warning("Upsert skipped: %s", e)
            mongo_s = elapsed_s(t0)
            if not log_prompt_debug:
                logger.info(
                    "  step | %s | %.3fs | queued=%s",
                    STEP_MONGO_UPSERT,
                    mongo_s,
                    "yes" if upsert is not None else "no",
                )

        if log_prompt_debug:
            logger.info("  job complete | total_s=%.3fs (see LLM logs above)", elapsed_s(t_job))
        else:
            logger.info(
                "  job complete | total_s=%.3fs | llm1=%.3fs llm2=%.3fs llm3=%.3fs mongo=%.3fs",
                elapsed_s(t_job),
                result.timings.get("llm1_s", 0.0),
                result.timings.get("llm2_s", 0.0),
                result.timings.get("llm3_s", 0.0),
                mongo_s,
            )

        if self._options.verbose and not log_prompt_debug:
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
            upsert=upsert,
            upsert_increment=1 if upsert is not None else 0,
            llm1_inc=int(result.ran_llm1),
            llm2_inc=int(result.ran_llm2),
            llm3_inc=int(result.ran_llm3),
        )
