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
    candidates = extract_esco_skill_candidates(doc)
    skills_prompt = build_skill_ranker_prompt(
        full, skills_to_prompt_dicts(candidates), settings=settings
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
