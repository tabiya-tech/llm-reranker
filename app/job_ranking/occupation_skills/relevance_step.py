"""LLM 3: judge each occupation-linked skill's relevance to the job ad."""

from __future__ import annotations

import logging

from app.job_context import JobDescriptionContext
from app.job_ranking.occupation_skills.menu import (
    occupation_skills_relevance_menu_size,
)
from app.job_ranking.occupation_skills.parser import (
    OccupationSkillsRelevanceResponseParser,
)
from app.job_ranking.occupation_skills.prompt import (
    OccupationSkillsRelevancePromptBuilder,
)
from app.job_ranking.occupation_skills.types import (
    OccupationSkillsRelevanceResult,
)
from app.job_ranking.response_schemas import occupation_skills_relevance_response_schema
from app.job_ranking.routes.types import OccupationSkillsBlock
from app.job_ranking.types import (
    NO_PROMPT_LOGGING,
    PROMPT_LOG_SEPARATOR,
    PromptLogging,
)
from app.llm import TracedLlmClient

logger = logging.getLogger(__name__)


PURPOSE_OCCUPATION_SKILLS_RELEVANCE = "occupation_skills_relevance"


class OccupationSkillsRelevanceStep:
    """LLM call 3: judge each occupation-linked skill's relevance to the job ad."""

    def __init__(
        self,
        llm: TracedLlmClient,
        prompts: OccupationSkillsRelevancePromptBuilder,
        parser: OccupationSkillsRelevanceResponseParser,
    ):
        self._llm = llm
        self._prompts = prompts
        self._parser = parser

    def run(
        self,
        ctx: JobDescriptionContext,
        menu_items: list[OccupationSkillsBlock],
        *,
        prompt_logging: PromptLogging = NO_PROMPT_LOGGING,
    ) -> OccupationSkillsRelevanceResult:
        prompt = self._prompts.build(ctx, menu_items)
        if prompt_logging.llm3:
            logger.info("%s", PROMPT_LOG_SEPARATOR)
            logger.info(
                "LLM3 occupation skills relevance | job=%s | menu_skills=%s",
                prompt_logging.label,
                occupation_skills_relevance_menu_size(menu_items),
            )
            logger.info("LLM3 assembled prompt (%s chars):\n%s", len(prompt), prompt)
            logger.info("%s", PROMPT_LOG_SEPARATOR)
        raw = self._llm.generate_json(
            prompt,
            purpose=PURPOSE_OCCUPATION_SKILLS_RELEVANCE,
            response_schema=occupation_skills_relevance_response_schema(menu_items),
        )
        return self._parser.parse(raw)
