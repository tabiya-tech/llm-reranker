"""Optional on-disk prompt dumps for debugging (first row only)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import Settings
from app.job_context import JobDescriptionContext
from app.llm_service import build_job_attributes_prompt_for_job
from app.prompts import build_skill_ranker_prompt
from utils.classifier_linked_entities import (
    extract_esco_skill_candidates,
    full_details_from_context,
    skills_to_prompt_dicts,
)
from utils.occupation_to_skills_mapping import (
    merge_esco_candidates_with_occupation_taxonomy,
    occupation_skill_taxonomy_for_settings,
)
from utils.skill_descriptions import load_skill_description_by_esco
from utils.uuid_to_tabiya_id import load_skills_uuid_tabiya_map

logger = logging.getLogger(__name__)


def dump_prompt_debug_first_row(
    ctx: JobDescriptionContext,
    doc: dict[str, Any],
    out_dir: Path,
    *,
    settings: Settings,
) -> None:
    """Write full skills + job-attributes prompts for the first document only."""
    out_dir.mkdir(parents=True, exist_ok=True)
    full = full_details_from_context(ctx)
    occ_map, csv_skill_labels, occ_skill_relation = occupation_skill_taxonomy_for_settings(settings)
    skills_uuid_tabiya_map = load_skills_uuid_tabiya_map(
        str(settings.skills_csv_path.resolve())
    )
    skill_desc_by_esco: dict[str, str] = {}
    if settings.skills_csv_include_descriptions and (
        settings.skills_csv_description_column or ""
    ).strip():
        skill_desc_by_esco = load_skill_description_by_esco(
            str(settings.skills_csv_path.resolve()),
            settings.skills_csv_description_column.strip(),
        )
    base = extract_esco_skill_candidates(doc)
    merge_esco_candidates_with_occupation_taxonomy(
        doc,
        base,
        occ_map,
        csv_skill_labels,
        occ_skill_relation,
        skills_uuid_tabiya_map,
        settings=settings,
    )
    skills_prompt = build_skill_ranker_prompt(
        full,
        skills_to_prompt_dicts(
            base,
            description_by_esco=skill_desc_by_esco if skill_desc_by_esco else None,
        ),
        settings=settings,
    )
    attrs_prompt = build_job_attributes_prompt_for_job(ctx, settings=settings)
    (out_dir / "01_skills_prompt.txt").write_text(skills_prompt, encoding="utf-8")
    (out_dir / "02_job_attributes_prompt.txt").write_text(attrs_prompt, encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "01_skills_prompt.txt — resources/Skills_reranker_prompts.md\n"
        "02_job_attributes_prompt.txt — resources/prompts.py + user block + ONET/JSON appendix\n",
        encoding="utf-8",
    )
    logger.info("Prompt debug files written to %s", out_dir.resolve())
