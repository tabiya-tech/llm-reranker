#!/usr/bin/env python3
"""Batch CLI: ClassifiedJobs → LLM 1, 3, 2 → upsert ranked job schema to target collection.

For single-job HTTP processing, see ``api.py`` (FastAPI).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any

from pymongo.operations import UpdateOne

from app.config import get_settings
from app.database import (
    bulk_write_llm_upserts,
    build_job_description_context,
    complete_ranking_run_log,
    insert_ranking_run_log,
    iter_classified_jobs,
    llm_pipeline_upsert_update_one,
    update_ranking_run_progress,
)
from app.logging_config import (
    STEP_LLM1_SKILL_RANKER,
    STEP_LLM2_JOB_ATTRIBUTES,
    STEP_LLM3_OCC_SKILLS,
    STEP_MONGO_UPSERT,
    STEP_OCCUPATION_TAXONOMY,
    configure_logging,
)
from app.pipeline import load_taxonomy_context
from app.llm_service import (
    run_job_attributes_for_job,
    run_occupation_skills_relevance_for_job,
    run_skill_ranker_for_job,
)
from app.prompts import build_occupation_prompt_enrichment
from app.ranked_job_schema import (
    apply_occupation_skills_listing_relevance,
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


def _classified_occupation_labels_for_log(classified_doc: dict[str, Any], settings) -> str:
    by_t = group_linked_entities_by_type(classified_doc, settings)
    occ = by_t.get("occupation", [])
    if not occ:
        return "—"
    parts: list[str] = []
    for i, c in enumerate(occ, start=1):
        lab = (c.label or "").strip() or "(no label)"
        cu = (c.concept_uuid or "").strip()
        parts.append(f"{i}:{lab}" + (f" ({cu[:8]}…)" if len(cu) >= 8 else (f" ({cu})" if cu else "")))
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ClassifiedJobs batch: Mongo → LLM 1, 3, 2 → upsert target collection"
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Max documents from the source collection. Default: 0 (entire collection, "
        "streamed newest _id first).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Per job: extra counts on stderr",
    )
    p.add_argument(
        "--log-llm1-prompt",
        action="store_true",
        help="Per job: log LLM1 (skill ranker) full prompt; skip other per-job INFO logs",
    )
    p.add_argument(
        "--log-llm2-prompt",
        action="store_true",
        help="Per job: log LLM2 job-attributes full prompt; skip other per-job INFO logs",
    )
    p.add_argument(
        "--log-llm3-prompt",
        action="store_true",
        help="Per job: log LLM3 occupation-skills relevance full prompt; skip other per-job INFO logs",
    )
    p.add_argument(
        "--bulk-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Flush MongoDB bulk writes every N upserts (overrides BULK_WRITE_BATCH_SIZE; default from env or 100)",
    )
    return p.parse_args()


def _gemini_or_exit(settings) -> None:
    if not settings.gemini_api_key:
        print(
            "No Gemini API key configured. Set GEMINI_API_KEY in .env.\n"
            "Create one at https://aistudio.google.com/apikey",
            file=sys.stderr,
        )
        raise SystemExit(1)


def run() -> None:
    configure_logging()
    args = _parse_args()
    settings = get_settings()
    batch_size = (
        args.bulk_batch_size if args.bulk_batch_size is not None else settings.bulk_write_batch_size
    )
    batch_size = max(1, min(batch_size, 10_000))
    pending: list[UpdateOne] = []

    t_startup = time.perf_counter()
    taxonomy = load_taxonomy_context(settings)
    startup_s = _elapsed_s(t_startup)

    limit_log = "all" if args.limit <= 0 else str(args.limit)
    logger.info(
        "Batch start | limit=%s | target_collection=%s bulk_batch_size=%s | "
        "startup_load_s=%.3f | taxonomy_relation_occupations=%s | model=%s",
        limit_log,
        settings.mongodb_target_collection,
        batch_size,
        startup_s,
        len(taxonomy.occ_map),
        settings.gemini_model,
    )
    logger.debug(
        "Batch config detail | job_attributes_schema=%s occupation_relations_csv=%s "
        "occupations_csv=%s skills_csv=%s classified_linked_entity_types=%s",
        settings.job_attributes_schema_path,
        settings.occupation_to_skill_relations_path,
        settings.occupations_csv_path,
        settings.skills_csv_path,
        list(settings.classified_linked_entity_types),
    )

    _gemini_or_exit(settings)
    logger.info(
        "Gemini | model=%s | api_key=...%s",
        settings.gemini_model,
        settings.gemini_api_key[-4:] if settings.gemini_api_key else "?",
    )

    run_id = insert_ranking_run_log(
        settings=settings,
        limit=args.limit,
        model=settings.gemini_model,
        source_collection=settings.mongodb_source_collection,
        target_collection=settings.mongodb_target_collection,
    )

    t_batch = time.perf_counter()
    n = 0
    upsert_ops = 0
    llm1_calls = 0
    llm2_calls = 0
    llm3_calls = 0
    _run_error: str | None = None

    try:
      for doc in iter_classified_jobs(limit=args.limit):
        n += 1
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

        if args.limit > 0:
            job_label = f"{n}/{args.limit}"
        else:
            job_label = f"#{n}"

        if log_prompt_debug:
            logger.info(
                "---- Job %s | id=%s | title=%s | (prompt-debug mode)",
                job_label, ident, title_short,
            )
        else:
            occ_preview = _classified_occupation_labels_for_log(doc, settings)
            logger.info(
                "---- Job %s | id=%s | title=%s | classifier_occupations=%s",
                job_label, ident, title_short, occ_preview,
            )

        # -- Taxonomy --
        t0 = time.perf_counter()
        base_skills = extract_esco_skill_candidates(doc)
        (
            occupation_relation_skill_uuids,
            occ_stats,
            occupation_to_skills_nested,
        ) = resolve_occupation_taxonomy_for_classified_job(
            doc, base_skills,
            taxonomy.occ_map, taxonomy.csv_skill_labels, taxonomy.occ_skill_relation,
            taxonomy.skills_uuid_tabiya_map,
            settings=settings,
        )
        n_blocks = len(occupation_to_skills_nested)
        n_llm3_menu = occupation_skills_relevance_menu_size(occupation_to_skills_nested)
        if not log_prompt_debug:
            logger.info(
                "  step | %s | %.3fs | classifier_skill_candidates=%s occupation_blocks=%s "
                "llm3_menu_skills=%s",
                STEP_OCCUPATION_TAXONOMY, _elapsed_s(t0),
                len(base_skills), n_blocks, n_llm3_menu,
            )
        if occ_stats is not None:
            lines = occ_stats.format_log_lines()
            if lines:
                logger.debug("  taxonomy detail | %s", lines[0])
                for line in lines[1:]:
                    logger.debug("%s", line)

        # -- LLM 3: occupation-skills relevance --
        llm3_s = 0.0
        if occupation_to_skills_nested and n_llm3_menu > 0:
            if not log_prompt_debug:
                logger.info(
                    "  step | %s | starting | menu_skills=%s",
                    STEP_LLM3_OCC_SKILLS, n_llm3_menu,
                )
            t0 = time.perf_counter()
            rel_result = run_occupation_skills_relevance_for_job(
                ctx, occupation_to_skills_nested,
                settings=settings,
                log_llm3_prompt=args.log_llm3_prompt,
                log_llm3_job_label=ident,
            )
            occupation_to_skills_nested = apply_occupation_skills_listing_relevance(
                occupation_to_skills_nested, rel_result,
            )
            llm3_s = _elapsed_s(t0)
            llm3_calls += 1
            ran_llm3_occ = True
            if not log_prompt_debug:
                logger.info(
                    "  step | %s | %.3fs | decisions=%s",
                    STEP_LLM3_OCC_SKILLS, llm3_s, len(rel_result.decisions),
                )
        elif not log_prompt_debug:
            logger.info(
                "  step | %s | skipped | no essential/optional occupation skills to judge",
                STEP_LLM3_OCC_SKILLS,
            )

        # -- LLM 1: skill ranker --
        llm1_s = 0.0
        if base_skills:
            t0 = time.perf_counter()
            essential, optional, skill_mapped, skill_raw = run_skill_ranker_for_job(
                ctx, settings=settings, esco_candidates=base_skills,
                log_llm1_prompt=args.log_llm1_prompt, log_llm1_job_label=ident,
                description_by_esco=taxonomy.skill_desc_by_esco or None,
            )
            llm1_s = _elapsed_s(t0)
            llm1_calls += 1
            ran_skill = True
            if not log_prompt_debug:
                logger.info(
                    "  step | %s | %.3fs | menu=%s essential=%s optional=%s",
                    STEP_LLM1_SKILL_RANKER, llm1_s,
                    len(base_skills), len(essential), len(optional),
                )
        elif not log_prompt_debug:
            logger.info(
                "  step | %s | skipped | no classifier ESCO skills on document",
                STEP_LLM1_SKILL_RANKER,
            )

        if ran_skill and skill_mapped is not None and not log_prompt_debug:
            if occupation_relation_skill_uuids:
                oe, oo, ou = _occupation_ranker_counts(occupation_relation_skill_uuids, skill_mapped)
                logger.debug(
                    "  ranker vs occupation-relation UUIDs | occ_essential=%s occ_optional=%s occ_unranked=%s",
                    oe, oo, ou,
                )
            else:
                logger.debug(
                    "  ranker | total_essential=%s total_optional=%s (no occupation-relation UUIDs)",
                    len(essential), len(optional),
                )

        # -- LLM 2: job attributes --
        t0 = time.perf_counter()
        occ_enrich = build_occupation_prompt_enrichment(
            doc, settings,
            fallback_listing_title=(ctx.title or "").strip() or "Unknown role",
            occupations_bridge=taxonomy.occ_bridge,
            occupations_all_ids=taxonomy.occ_all_ids,
            label_by_esco=taxonomy.occ_label_by_esco,
            description_by_esco=taxonomy.occ_desc_by_esco,
        )
        ja = run_job_attributes_for_job(
            ctx, settings=settings, occupation=occ_enrich,
            log_llm2_prompt=args.log_llm2_prompt, log_llm2_job_label=ident,
        )
        llm2_s = _elapsed_s(t0)
        llm2_calls += 1
        ran_ja = True
        if not log_prompt_debug:
            logger.info(
                "  step | %s | %.3fs | attribute_estimates=%s work_activity_ids=%s",
                STEP_LLM2_JOB_ATTRIBUTES, llm2_s,
                len(ja.attribute_estimates), len(ja.work_activity_ids),
            )

        # -- MongoDB upsert --
        mongo_s = 0.0
        upsert_queued = False
        if ran_skill or ran_ja:
            t0 = time.perf_counter()
            try:
                op = llm_pipeline_upsert_update_one(
                    doc, ctx, settings=settings,
                    ja=ja if ran_ja and ja is not None else None,
                    skill_ranker=(essential, optional, skill_mapped, skill_raw)
                    if ran_skill and skill_mapped is not None else None,
                    occupation_to_skills=occupation_to_skills_nested or None,
                    skills_uuid_tabiya_map=taxonomy.skills_uuid_tabiya_map,
                    occupations_bridge=taxonomy.occ_bridge or None,
                    occupations_all_ids=taxonomy.occ_all_ids or None,
                    ranker_candidates=base_skills,
                    classifier_skill_candidates=base_skills,
                )
                mongo_s = _elapsed_s(t0)
                if op is not None:
                    upsert_ops += 1
                    upsert_queued = True
                    pending.append(op)
                    if len(pending) >= batch_size:
                        bulk_write_llm_upserts(pending, settings=settings)
                        pending.clear()
            except ValueError as e:
                mongo_s = _elapsed_s(t0)
                logger.warning("Upsert skipped: %s", e)
            if not log_prompt_debug:
                logger.info(
                    "  step | %s | %.3fs | queued=%s",
                    STEP_MONGO_UPSERT, mongo_s, "yes" if upsert_queued else "no",
                )

        # -- Job summary --
        if not log_prompt_debug:
            logger.info(
                "  job complete | total_s=%.3fs | llm1=%.3fs llm2=%.3fs llm3=%.3fs mongo=%.3fs",
                _elapsed_s(t_job), llm1_s, llm2_s, llm3_s, mongo_s,
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

        if n % 10 == 0:
            update_ranking_run_progress(
                run_id,
                jobs_processed=n, upserts_queued=upsert_ops,
                llm1_calls=llm1_calls, llm2_calls=llm2_calls, llm3_calls=llm3_calls,
                settings=settings,
            )

      if pending:
        bulk_write_llm_upserts(pending, settings=settings)

      if n > 0 and upsert_ops == 0:
        logger.warning(
            "No MongoDB upserts ran for %s document(s). Check 'Upsert skipped' messages above "
            "(each document needs job id, fingerprint, or source _id for the target upsert).",
            n,
        )
    except Exception as exc:
        _run_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Batch failed: %s", _run_error)
    finally:
        batch_s = _elapsed_s(t_batch)
        logger.info(
            "Batch %s | documents=%s upserts_queued=%s elapsed_s=%.3f",
            "complete" if _run_error is None else "FAILED",
            n, upsert_ops, batch_s,
        )
        complete_ranking_run_log(
            run_id,
            success=_run_error is None,
            jobs_processed=n, upserts_queued=upsert_ops,
            llm1_calls=llm1_calls, llm2_calls=llm2_calls, llm3_calls=llm3_calls,
            elapsed_s=batch_s, error_message=_run_error,
            settings=settings,
        )
        if _run_error is not None:
            raise SystemExit(1)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
