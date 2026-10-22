"""LLM 2: structured attribute estimates, summary and ONET work-activity ids."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from app.config import Settings
from app.job_context import JobDescriptionContext
from app.job_ranking.job_attributes.parser import (
    JobAttributesResponseParser,
)
from app.job_ranking.job_attributes.prompt import (
    JobAttributesPromptBuilder,
    OccupationPromptEnrichment,
    build_occupation_prompt_enrichment,
    resolve_llm2_single_county_user_fields,
)
from app.job_ranking.job_attributes.types import (
    JobAttributesResult,
)
from app.job_ranking.prompts.job_listing import (
    build_job_attributes_jd_text,
    build_job_attributes_metadata_block,
)
from app.job_ranking.response_schemas import job_attributes_response_schema
from app.job_ranking.types import (
    NO_PROMPT_LOGGING,
    PROMPT_LOG_SEPARATOR,
    PromptLogging,
)
from app.llm import TracedLlmClient
from app.taxonomy.types import TaxonomyContext

logger = logging.getLogger(__name__)


PURPOSE_JOB_ATTRIBUTES = "job_attributes"


class JobAttributesStep:
    """LLM call 2: structured attribute estimates, summary and ONET work-activity ids."""

    def __init__(
        self,
        llm: TracedLlmClient,
        prompts: JobAttributesPromptBuilder,
        parser: JobAttributesResponseParser,
        settings: Settings,
        taxonomy: TaxonomyContext,
    ):
        self._llm = llm
        self._prompts = prompts
        self._parser = parser
        self._settings = settings
        self._taxonomy = taxonomy

    def run(
        self,
        ctx: JobDescriptionContext,
        classified_doc: dict[str, Any],
        *,
        prompt_logging: PromptLogging = NO_PROMPT_LOGGING,
    ) -> JobAttributesResult:
        tx = self._taxonomy
        occupation = build_occupation_prompt_enrichment(
            classified_doc,
            self._settings,
            fallback_listing_title=(ctx.title or "").strip() or "Unknown role",
            occupations_bridge=tx.occ_bridge,
            occupations_all_ids=tx.occ_all_ids,
            label_by_esco=tx.occ_label_by_esco,
            description_by_esco=tx.occ_desc_by_esco,
        )
        prompt = self._prompts.build(ctx, occupation)
        if prompt_logging.llm2:
            self._log_prompt(prompt_logging.label, ctx, occupation, prompt)
        raw = self._llm.generate_json(
            prompt,
            purpose=PURPOSE_JOB_ATTRIBUTES,
            response_schema=job_attributes_response_schema(
                self._prompts.schema,
                self._prompts.onet,
                requires_post_secondary=self._settings.target_country == "kenya",
            ),
        )
        return self._parser.parse(raw)

    def _log_prompt(
        self,
        label: str,
        ctx: JobDescriptionContext,
        occupation: OccupationPromptEnrichment,
        prompt: str,
    ) -> None:
        county = (ctx.county or "").strip()
        city = (ctx.city or "").strip()
        resolved = resolve_llm2_single_county_user_fields(occupation, None)
        logger.info("%s", PROMPT_LOG_SEPARATOR)
        logger.info(
            "LLM2 job attributes | job=%s | template=resources/prompts/job_attributes.md | "
            "onet_rows=%s | schema_attrs=%s | county=%r | city=%r",
            label,
            len(self._prompts.onet),
            len(self._prompts.schema.attributes),
            county or None,
            city or None,
        )
        logger.info(
            "LLM2 `occupation` enrichment (JSON):\n%s",
            json.dumps(asdict(occupation), ensure_ascii=False, indent=2),
        )
        logger.info(
            "LLM2 resolved single-county user fields (JSON):\n%s",
            json.dumps(
                {
                    "preferred_label": occupation.occupation_label,
                    "county": county,
                    **{k: v for k, v in resolved.items() if v},
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        jd = build_job_attributes_jd_text(ctx)
        meta = build_job_attributes_metadata_block(ctx)
        logger.info("LLM2 `job_listing_body` (%s chars):\n%s", len(jd), jd)
        logger.info("LLM2 `job_listing_metadata` (%s chars):\n%s", len(meta), meta)
        logger.info("LLM2 assembled prompt (%s chars):\n%s", len(prompt), prompt)
        logger.info("%s", PROMPT_LOG_SEPARATOR)
