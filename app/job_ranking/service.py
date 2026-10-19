"""The per-job pipeline — the one both the batch CLI and the FastAPI endpoint run."""

from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from app.config import LLM3_MODE_FOLD, Settings, get_settings_for_language
from app.job_context import JobDescriptionContext, build_job_description_context
from app.job_ranking.job_attributes.parser import JobAttributesResponseParser
from app.job_ranking.job_attributes.prompt import JobAttributesPromptBuilder
from app.job_ranking.job_attributes.repository import JobAttributesRepository
from app.job_ranking.job_attributes.step import JobAttributesStep
from app.job_ranking.job_attributes.types import JobAttributesResult
from app.job_ranking.occupation_skills.menu import (
    apply_occupation_skills_listing_relevance,
    build_llm3_occupation_menu,
    llm3_menu_skill_candidates,
)
from app.job_ranking.occupation_skills.parser import OccupationSkillsRelevanceResponseParser
from app.job_ranking.occupation_skills.prompt import OccupationSkillsRelevancePromptBuilder
from app.job_ranking.occupation_skills.relevance_step import OccupationSkillsRelevanceStep
from app.job_ranking.occupation_skills.taxonomy_step import OccupationTaxonomyStep
from app.job_ranking.prompts.repository import PromptRepository
from app.job_ranking.ranked_document import RankedJobDocumentAssembler
from app.job_ranking.routes.types import OccupationSkillsBlock
from app.job_ranking.skill_ranker.parser import SkillRankerResponseParser
from app.job_ranking.skill_ranker.prompt import SkillRankerPromptBuilder
from app.job_ranking.skill_ranker.step import SkillRankerStep
from app.job_ranking.skill_ranker.types import SkillRankerResult
from app.job_ranking.types import (
    NO_PROMPT_LOGGING,
    HealthStatus,
    JobResult,
    Llm1Llm2Result,
    Llm3Outcome,
    PromptLogging,
)
from app.languages import LANGUAGE_REGISTRY
from app.llm import TracedLlmClient
from app.taxonomy.service import TaxonomyService
from utils.classifier_linked_entities import (
    EscoSkillCandidate,
    MappedEscoSkills,
    extract_esco_skill_candidates,
    job_identifier,
)

_T = TypeVar("_T")


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


def _timed(fn: Callable[[], _T]) -> tuple[float, _T]:
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


class JobRankingService:
    """Ranks one classified job document: taxonomy → LLM3 menu → LLM1 + LLM2 → ranked document.

    Built for one language pack (:class:`JobRankingServiceFactory`); thread-safe, so the batch's
    worker threads share one instance.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        taxonomy_step: OccupationTaxonomyStep,
        skill_ranker: SkillRankerStep,
        job_attributes: JobAttributesStep,
        occupation_relevance: OccupationSkillsRelevanceStep,
        assembler: RankedJobDocumentAssembler,
    ):
        self.settings = settings
        self._taxonomy_step = taxonomy_step
        self._skill_ranker = skill_ranker
        self._job_attributes = job_attributes
        self._occupation_relevance = occupation_relevance
        self._assembler = assembler

    def rank(
        self,
        classified_doc: dict[str, Any],
        *,
        prompt_logging: PromptLogging | None = None,
    ) -> JobResult:
        """
        Run the pipeline on one classified job document.

        LLM3 depends on ``RERANKER_ENABLE_LLM3`` and ``RERANKER_LLM3_MODE``: in ``fold`` mode the
        occupation-skills menu becomes extra LLM1 candidates and no LLM3 call is made; in ``call``
        mode the standalone relevance call tags ``occupation_to_skills``. ``JobResult.llm3_outcome``
        says which happened.

        Returns the ranked payload (ready for MongoDB ``$set`` or API response), step timings, and
        per-step counts. Does NOT write to MongoDB or embed the job — the caller decides what to do
        with the result. ``prompt_logging`` logs each assembled prompt, labelled with the job's
        fingerprint / id.
        """
        settings = self.settings
        t_total = time.perf_counter()

        ctx = build_job_description_context(classified_doc, settings=settings)
        from app.job_ranking.routes.types import RankedJobDocument  # noqa: F401

        JobResult.model_rebuild()
        result = JobResult(ranked_document=None, ctx=ctx)
        stats = result.stats
        flags = prompt_logging or NO_PROMPT_LOGGING
        logging_for_job = PromptLogging(
            llm1=flags.llm1,
            llm2=flags.llm2,
            llm3=flags.llm3,
            job_label=job_identifier(ctx, classified_doc),
        )

        base_skills: list[EscoSkillCandidate] = extract_esco_skill_candidates(
            classified_doc, settings
        )
        taxonomy_s, taxonomy = _timed(lambda: self._taxonomy_step.run(classified_doc, base_skills))
        occupation_to_skills = taxonomy.occupation_to_skills
        result.timings["taxonomy_s"] = taxonomy_s
        stats.classifier_skill_candidates = len(base_skills)
        stats.occupation_blocks = len(occupation_to_skills)
        stats.taxonomy_menu_skills = taxonomy.menu_size
        if taxonomy.stats is not None:
            stats.taxonomy_detail = taxonomy.stats.format_log_lines()

        llm3_menu: list[OccupationSkillsBlock] = []
        if settings.reranker_enable_llm3 and stats.taxonomy_menu_skills > 0:
            llm3_subset = build_llm3_occupation_menu(occupation_to_skills, base_skills, settings)
            llm3_menu = llm3_subset.items
            stats.llm3_prompt_skills = llm3_subset.subset_menu_count

        llm1_candidates = base_skills
        if not settings.reranker_enable_llm3:
            result.llm3_outcome = Llm3Outcome.DISABLED
        elif stats.taxonomy_menu_skills == 0:
            result.llm3_outcome = Llm3Outcome.NO_MENU
        elif stats.llm3_prompt_skills == 0:
            result.llm3_outcome = Llm3Outcome.MENU_FILTERED_OUT
        elif settings.reranker_llm3_mode == LLM3_MODE_FOLD:
            llm3_extra = llm3_menu_skill_candidates(llm3_menu, base_skills)
            llm1_candidates = base_skills + llm3_extra
            stats.llm3_added_llm1_candidates = len(llm3_extra)
            result.llm3_outcome = Llm3Outcome.FOLDED
        else:
            llm3_s, relevance = _timed(
                lambda: self._occupation_relevance.run(
                    ctx, llm3_menu, prompt_logging=logging_for_job
                )
            )
            occupation_to_skills = apply_occupation_skills_listing_relevance(
                occupation_to_skills, relevance
            )
            result.timings["llm3_s"] = llm3_s
            stats.llm3_decisions = len(relevance.decisions)
            result.ran_llm3 = True
            result.llm3_outcome = Llm3Outcome.CALLED

        llm12 = self._run_llm1_llm2(
            ctx, classified_doc, llm1_candidates, logging_for_job, result.timings
        )
        skill_ranker, job_attributes = llm12.skill_ranker, llm12.job_attributes
        stats.llm1_menu = len(llm1_candidates)
        if skill_ranker is not None:
            result.ran_llm1 = True
            stats.llm1_essential = len(skill_ranker.essential)
            stats.llm1_optional = len(skill_ranker.optional)
            if taxonomy.relation_skill_uuids:
                stats.occupation_ranker_counts = _occupation_ranker_counts(
                    taxonomy.relation_skill_uuids, skill_ranker.mapped
                )
        result.ran_llm2 = True
        stats.attribute_estimates = len(job_attributes.attribute_estimates)
        stats.work_activity_ids = len(job_attributes.work_activity_ids)

        result.ranked_document = self._assembler.assemble(
            classified_doc,
            ctx,
            job_attributes=job_attributes,
            skill_ranker=skill_ranker,
            occupation_to_skills=occupation_to_skills or None,
            ranker_candidates=llm1_candidates,
            classifier_skill_candidates=base_skills,
        )
        result.timings["total_s"] = time.perf_counter() - t_total
        return result

    def _run_llm1_llm2(
        self,
        ctx: JobDescriptionContext,
        classified_doc: dict[str, Any],
        candidates: list[EscoSkillCandidate],
        prompt_logging: PromptLogging,
        timings: dict[str, float],
    ) -> Llm1Llm2Result:
        """
        Run the skill ranker (LLM1, only with candidates) and job attributes (LLM2).

        With ``RERANKER_OVERLAP_LLM1_LLM2``, candidates and no prompt-debug logging, the two calls
        run concurrently (~max latency instead of sum). Beware combined with
        ``RERANKER_PARALLEL_WORKERS`` — total in-flight requests can be workers×2.
        """

        def llm1() -> SkillRankerResult:
            return self._skill_ranker.run(ctx, candidates, prompt_logging=prompt_logging)

        def llm2() -> JobAttributesResult:
            return self._job_attributes.run(ctx, classified_doc, prompt_logging=prompt_logging)

        overlap = (
            self.settings.reranker_overlap_llm1_llm2
            and bool(candidates)
            and not prompt_logging.llm1
            and not prompt_logging.llm2
        )
        skill_ranker: SkillRankerResult | None = None
        if overlap:
            with ThreadPoolExecutor(max_workers=2) as pool:
                # Each call runs in a copy of this thread's context, so its Langfuse
                # generation nests under the job's trace instead of starting a new one.
                # One copy per call: a Context cannot be entered by two threads at once.
                f1 = pool.submit(contextvars.copy_context().run, _timed, llm1)
                f2 = pool.submit(contextvars.copy_context().run, _timed, llm2)
                timings["llm1_s"], skill_ranker = f1.result()
                timings["llm2_s"], job_attributes = f2.result()
            return Llm1Llm2Result(skill_ranker=skill_ranker, job_attributes=job_attributes)

        if candidates:
            timings["llm1_s"], skill_ranker = _timed(llm1)
        timings["llm2_s"], job_attributes = _timed(llm2)
        return Llm1Llm2Result(skill_ranker=skill_ranker, job_attributes=job_attributes)


class JobRankingServiceFactory:
    """Builds and caches one :class:`JobRankingService` per language pack.

    Every service shares the one LLM client; each gets its language's taxonomy, prompts and
    job-attribute reference data, read through the repositories on first use.
    ``default_settings`` is the language used when a caller names none.
    """

    def __init__(
        self,
        *,
        default_settings: Settings,
        llm: TracedLlmClient,
        taxonomy: TaxonomyService,
        prompts: PromptRepository,
        job_attributes: JobAttributesRepository,
    ):
        self._default_settings = default_settings
        self._llm = llm
        self._taxonomy = taxonomy
        self._prompts = prompts
        self._job_attributes = job_attributes
        self._services: dict[str, JobRankingService] = {}
        self._lock = threading.Lock()

    @property
    def taxonomy(self) -> TaxonomyService:
        return self._taxonomy

    def close(self) -> None:
        """Release the model client shared by every service built here."""
        self._llm.close()

    def for_language(self, language: str | None) -> JobRankingService:
        """The service for a normalized language code; ``None`` is the default language."""
        if not language or language == self._default_settings.language:
            return self.get(self._default_settings)
        return self.get(get_settings_for_language(language))

    def warm_up(self) -> None:
        """Load the default language pack (a broken pack must fail startup, not a request)."""
        self.get(self._default_settings)

    def health(self) -> HealthStatus:
        default = self._default_settings.language
        return HealthStatus(
            taxonomy_loaded=self.taxonomy.is_loaded(default),
            default_language=default,
            supported_languages=list(LANGUAGE_REGISTRY),
            loaded_languages=self.taxonomy.loaded_languages,
        )

    def get(self, settings: Settings) -> JobRankingService:
        cached = self._services.get(settings.language)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._services.get(settings.language)
            if cached is None:
                cached = self._build(settings)
                self._services[settings.language] = cached
            return cached

    def _build(self, settings: Settings) -> JobRankingService:
        taxonomy = self._taxonomy.get(settings)
        templates = self._prompts.load(settings)
        schema = self._job_attributes.load_schema(settings)
        onet = self._job_attributes.load_onet_work_activities(settings)
        return JobRankingService(
            settings=settings,
            taxonomy_step=OccupationTaxonomyStep(settings, taxonomy),
            skill_ranker=SkillRankerStep(
                self._llm,
                SkillRankerPromptBuilder(templates),
                SkillRankerResponseParser(),
                taxonomy,
            ),
            job_attributes=JobAttributesStep(
                self._llm,
                JobAttributesPromptBuilder(templates, schema, onet, settings),
                JobAttributesResponseParser(schema, onet, settings),
                settings,
                taxonomy,
            ),
            occupation_relevance=OccupationSkillsRelevanceStep(
                self._llm,
                OccupationSkillsRelevancePromptBuilder(templates),
                OccupationSkillsRelevanceResponseParser(),
            ),
            assembler=RankedJobDocumentAssembler(settings, taxonomy, onet),
        )
