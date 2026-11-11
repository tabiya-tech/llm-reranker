"""Test doubles for the ranking services: stub steps and a builder that wires them.

The stubs subclass the real steps and replace only ``run``, so a test injects exactly the
step it wants to control and keeps every other part of the pipeline real.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.job_context import JobDescriptionContext
from app.job_ranking.experience_skills.step import ExperienceSkillRankerStep
from app.job_ranking.experience_skills.types import ExperienceSkillsResult
from app.job_ranking.job_attributes.repository import JobAttributesRepository
from app.job_ranking.job_attributes.step import JobAttributesStep
from app.job_ranking.job_attributes.types import AttributeEstimate, JobAttributesResult
from app.job_ranking.occupation_skills.menu import occupation_skills_relevance_menu_size
from app.job_ranking.occupation_skills.relevance_step import OccupationSkillsRelevanceStep
from app.job_ranking.occupation_skills.taxonomy_step import OccupationTaxonomyStep
from app.job_ranking.occupation_skills.types import (
    OccupationSkillsRelevanceDecision,
    OccupationSkillsRelevanceResult,
    OccupationTaxonomyOutcome,
)
from app.job_ranking.ranked_document import RankedJobDocumentAssembler
from app.job_ranking.routes.types import (
    ExperienceDetails,
    OccupationSkill,
    OccupationSkillsBlock,
    SkillCandidate,
    SkillCounts,
)
from app.job_ranking.service import JobRankingService, JobRankingServiceFactory
from app.job_ranking.skill_ranker.step import SkillRankerStep
from app.job_ranking.skill_ranker.types import SkillRankerResult
from app.job_ranking.types import NO_PROMPT_LOGGING, PromptLogging
from app.taxonomy.repository import TaxonomyRepository
from app.taxonomy.service import TaxonomyService
from app.taxonomy.types import TaxonomyContext
from utils.classifier_linked_entities import EscoSkillCandidate, MappedEscoSkills
from utils.uuid_to_tabiya_id import SkillsUuidTabiyaMap


def empty_taxonomy() -> TaxonomyContext:
    """A taxonomy with no rows: enough to build the service and its steps."""
    return TaxonomyContext(
        occ_map={},
        csv_skill_labels=None,
        occ_skill_relation=None,
        occ_skill_signalling=None,
        occ_bridge={},
        occ_all_ids=frozenset(),
        skills_uuid_tabiya_map=SkillsUuidTabiyaMap(),
        skill_desc_by_esco={},
        occ_label_by_esco={},
        occ_desc_by_esco={},
    )


def classified_doc() -> dict[str, Any]:
    """A minimal but realistic classifier document.

    Includes one ESCO-linked skill so LLM1 fires, no occupation entities so
    LLM3 stays a no-op (occupation→skills menu is empty), and enough text to
    populate the description fields LLM2 would consume.
    """
    return {
        "_id": "src-doc-1",
        "job_id": "jid-1",
        "job_fingerprint": "fp-1",
        "title": "Software Engineer",
        "employer": "ACME",
        "category": "Tech",
        "description": "Build and maintain backend services in Python.",
        "requirements": "3 years of experience.",
        "city": "Lusaka",
        "classification": {
            "entities": [
                {
                    "entity_type": "skill",
                    "linked_entities": [
                        {
                            "taxonomy": "esco",
                            "label": "Python",
                            "uri": (
                                "http://data.europa.eu/esco/skill/"
                                "5247a6ce-6ed0-4fb3-b121-758921dcc44e"
                            ),
                            "similarity_score": 0.95,
                        }
                    ],
                }
            ]
        },
    }


def occupation_menu() -> list[OccupationSkillsBlock]:
    """One occupation block with two taxonomy skills of different signalling strength."""
    return [
        OccupationSkillsBlock(
            occupation_label="Welder",
            occupation_esco_uuid="33333333-3333-3333-3333-333333333333",
            occupation_uri=(
                "http://data.europa.eu/esco/occupation/33333333-3333-3333-3333-333333333333"
            ),
            tabiya_occupation_id="occ-1",
            skill_counts=SkillCounts(total=2, essential=1, optional=1, unspecified=0),
            skills=[
                OccupationSkill(
                    skill_id="11111111-1111-1111-1111-111111111111",
                    tabiya_skill_id="tskill-1",
                    label="Weld metal",
                    uri="http://data.europa.eu/esco/skill/11111111-1111-1111-1111-111111111111",
                    relation_type="essential",
                    signalling_label="high",
                    signalling_value="0.9",
                ),
                OccupationSkill(
                    skill_id="22222222-2222-2222-2222-222222222222",
                    tabiya_skill_id="tskill-2",
                    label="Paint surfaces",
                    uri="http://data.europa.eu/esco/skill/22222222-2222-2222-2222-222222222222",
                    relation_type="optional",
                    signalling_label="low",
                    signalling_value="0.1",
                ),
            ],
        )
    ]


def relevance_result() -> OccupationSkillsRelevanceResult:
    """LLM3 keeps the essential menu skill and rejects the optional one."""
    return OccupationSkillsRelevanceResult(
        decisions=(
            OccupationSkillsRelevanceDecision(
                tabiya_occupation_id="occ-1",
                skill_id="11111111-1111-1111-1111-111111111111",
                is_relevant=True,
            ),
            OccupationSkillsRelevanceDecision(
                tabiya_occupation_id="occ-1",
                skill_id="22222222-2222-2222-2222-222222222222",
                is_relevant=False,
            ),
        ),
        raw_response="{}",
    )


def job_attributes_result() -> JobAttributesResult:
    """Pretend LLM2 produced one attribute estimate."""
    return JobAttributesResult(
        attribute_estimates=(
            AttributeEstimate(
                attribute_name="earnings_per_month",
                selected_level_id="earn_15k",
                confidence="medium",
                reasoning="stub",
            ),
        ),
        summary="stubbed",
        attributes={"earnings_per_month": "earn_15k"},
        work_activity_ids=(),
        raw_response="{}",
    )


def rank_all_essential(candidates: list[EscoSkillCandidate]) -> SkillRankerResult:
    """Pretend LLM1 ranked every candidate as essential."""
    mapped = MappedEscoSkills(
        essential_uris=[c.uri for c in candidates],
        optional_uris=[],
        essential_skills_uuids=[c.esco_uuid or "" for c in candidates],
        optional_skills_uuids=[],
    )
    return SkillRankerResult(
        essential=[c.label for c in candidates],
        optional=[],
        mapped=mapped,
        raw_response='{"essential_skills": ["Python"], "optional_skills": []}',
    )


class StubSkillRanker(SkillRankerStep):
    """LLM1 without a model: records each call's candidate labels and prompt logging."""

    def __init__(
        self,
        answer: Callable[[list[EscoSkillCandidate]], SkillRankerResult] = rank_all_essential,
    ):
        self.answer = answer
        self.candidates: list[list[str]] = []
        self.prompt_logging: list[PromptLogging] = []

    def run(
        self,
        ctx: JobDescriptionContext,
        candidates: list[EscoSkillCandidate],
        *,
        prompt_logging: PromptLogging = NO_PROMPT_LOGGING,
    ) -> SkillRankerResult:
        self.candidates.append([c.label for c in candidates])
        self.prompt_logging.append(prompt_logging)
        return self.answer(candidates)


class StubJobAttributes(JobAttributesStep):
    """LLM2 without a model."""

    def __init__(self, answer: Callable[[], JobAttributesResult] = job_attributes_result):
        self.answer = answer
        self.calls = 0

    def run(
        self,
        ctx: JobDescriptionContext,
        classified_doc: dict[str, Any],
        *,
        prompt_logging: PromptLogging = NO_PROMPT_LOGGING,
    ) -> JobAttributesResult:
        self.calls += 1
        return self.answer()


def _llm3_must_not_run() -> OccupationSkillsRelevanceResult:
    raise AssertionError("LLM3 should not be invoked when occupation_to_skills_nested is empty")


class StubOccupationRelevance(OccupationSkillsRelevanceStep):
    """LLM3 without a model: records each menu's skill labels. Fails loudly by default."""

    def __init__(self, answer: Callable[[], OccupationSkillsRelevanceResult] = _llm3_must_not_run):
        self.answer = answer
        self.menus: list[list[str]] = []

    def run(
        self,
        ctx: JobDescriptionContext,
        menu_items: list[OccupationSkillsBlock],
        *,
        prompt_logging: PromptLogging = NO_PROMPT_LOGGING,
    ) -> OccupationSkillsRelevanceResult:
        self.menus.append([sk.label for block in menu_items for sk in block.skills])
        return self.answer()


def keep_candidates_in_order(
    candidates: list[SkillCandidate], max_skills: int
) -> ExperienceSkillsResult:
    """Pretend the model kept the first ``max_skills`` candidates, in request order."""
    ids = [c.id for c in candidates]
    return ExperienceSkillsResult(
        ranked_ids=ids[:max_skills], discarded_ids=ids[max_skills:], raw_response="{}"
    )


class StubExperienceSkillRanker(ExperienceSkillRankerStep):
    """``POST /rank/skills`` without a model: records each call's experience and candidates."""

    def __init__(
        self,
        answer: Callable[
            [list[SkillCandidate], int], ExperienceSkillsResult
        ] = keep_candidates_in_order,
    ):
        self.answer = answer
        self.calls: list[tuple[ExperienceDetails, list[SkillCandidate], int]] = []

    def run(
        self,
        experience: ExperienceDetails,
        candidates: list[SkillCandidate],
        max_skills: int,
    ) -> ExperienceSkillsResult:
        self.calls.append((experience, candidates, max_skills))
        return self.answer(candidates, max_skills)


class FixedOccupationTaxonomy(OccupationTaxonomyStep):
    """The taxonomy step, answering every job with the same ``occupation_to_skills`` blocks."""

    def __init__(self, blocks: list[OccupationSkillsBlock]):
        self.blocks = blocks

    def run(
        self, classified_doc: dict[str, Any], classifier_skills: list[EscoSkillCandidate]
    ) -> OccupationTaxonomyOutcome:
        return OccupationTaxonomyOutcome(
            relation_skill_uuids=[],
            stats=None,
            occupation_to_skills=self.blocks,
            menu_size=occupation_skills_relevance_menu_size(self.blocks),
        )


def build_ranking_service(
    settings: Settings,
    *,
    taxonomy: TaxonomyContext | None = None,
    taxonomy_step: OccupationTaxonomyStep | None = None,
    skill_ranker: SkillRankerStep | None = None,
    job_attributes: JobAttributesStep | None = None,
    occupation_relevance: OccupationSkillsRelevanceStep | None = None,
    experience_skill_ranker: ExperienceSkillRankerStep | None = None,
) -> JobRankingService:
    """A :class:`JobRankingService` on the real resources, with the given steps swapped in."""
    tx = taxonomy if taxonomy is not None else TaxonomyRepository().load(settings)
    onet = JobAttributesRepository().load_onet_work_activities(settings)
    return JobRankingService(
        settings=settings,
        taxonomy_step=taxonomy_step or OccupationTaxonomyStep(settings, tx),
        skill_ranker=skill_ranker or StubSkillRanker(),
        job_attributes=job_attributes or StubJobAttributes(),
        occupation_relevance=occupation_relevance or StubOccupationRelevance(),
        experience_skill_ranker=experience_skill_ranker or StubExperienceSkillRanker(),
        assembler=RankedJobDocumentAssembler(settings, tx, onet),
    )


class FixedRankers(JobRankingServiceFactory):
    """A factory that hands out one prepared service, for every language."""

    def __init__(
        self,
        service: JobRankingService,
        taxonomy: TaxonomyService | None = None,
    ):
        self._service = service
        self._default_settings = service.settings
        self._fixed_taxonomy = taxonomy or TaxonomyService(TaxonomyRepository())

    @property
    def taxonomy(self) -> TaxonomyService:
        return self._fixed_taxonomy

    def get(self, settings: Settings) -> JobRankingService:
        self._fixed_taxonomy.get(settings)
        return self._service

    def close(self) -> None:
        pass
