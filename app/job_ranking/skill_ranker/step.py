"""LLM 1: rank the candidate skills as essential, optional or discarded."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.job_context import JobDescriptionContext
from app.job_ranking.response_schemas import skill_ranker_response_schema
from app.job_ranking.skill_ranker.parser import SkillRankerResponseParser
from app.job_ranking.skill_ranker.prompt import (
    SkillRankerPromptBuilder,
    format_skill_ranker_skills_text,
)
from app.job_ranking.skill_ranker.types import SkillRankerResult
from app.job_ranking.types import (
    NO_PROMPT_LOGGING,
    PROMPT_LOG_SEPARATOR,
    PromptLogging,
)
from app.llm import TracedLlmClient
from app.taxonomy.types import TaxonomyContext
from utils.classifier_linked_entities import (
    EscoSkillCandidate,
    full_details_from_context,
    skills_to_prompt_dicts,
)

logger = logging.getLogger(__name__)


# Labels passed to the client as ``purpose`` — they identify the pass in provider logs,
# error messages and Langfuse generations.
PURPOSE_SKILL_RANKER = "skill_ranker"


class SkillRankerStep:
    """LLM call 1: rank the candidate skills as essential, optional or discarded."""

    def __init__(
        self,
        llm: TracedLlmClient,
        prompts: SkillRankerPromptBuilder,
        parser: SkillRankerResponseParser,
        taxonomy: TaxonomyContext,
    ):
        self._llm = llm
        self._prompts = prompts
        self._parser = parser
        self._description_by_esco = taxonomy.skill_desc_by_esco or None

    def run(
        self,
        ctx: JobDescriptionContext,
        candidates: list[EscoSkillCandidate],
        *,
        prompt_logging: PromptLogging = NO_PROMPT_LOGGING,
    ) -> SkillRankerResult:
        full_details = full_details_from_context(ctx)
        skill_dicts = skills_to_prompt_dicts(
            candidates, description_by_esco=self._description_by_esco
        )
        prompt = self._prompts.build(full_details, skill_dicts)
        if prompt_logging.llm1:
            self._log_prompt(prompt_logging.label, full_details, skill_dicts, prompt)
        raw = self._llm.generate_json(
            prompt,
            purpose=PURPOSE_SKILL_RANKER,
            response_schema=skill_ranker_response_schema(d["label"] for d in skill_dicts),
        )
        return self._parser.parse(raw, candidates)

    @staticmethod
    def _log_prompt(
        label: str, full_details: str, skill_dicts: list[dict[str, Any]], prompt: str
    ) -> None:
        skills_text = format_skill_ranker_skills_text(skill_dicts)
        logger.info("%s", PROMPT_LOG_SEPARATOR)
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
        logger.info("%s", PROMPT_LOG_SEPARATOR)
