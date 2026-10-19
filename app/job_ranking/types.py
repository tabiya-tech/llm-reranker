"""Results and options of the per-job pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain_base import DomainModel
from app.job_context import JobDescriptionContext
from app.job_ranking.job_attributes.types import JobAttributesResult
from app.job_ranking.skill_ranker.types import SkillRankerResult

if TYPE_CHECKING:
    # Deferred: app.job_ranking.routes.types imports HealthStatus/JobResult/Llm3Outcome from this
    # module — a real top-level import of RankedJobDocument here would be circular. JobResult
    # calls model_rebuild() in app/job_ranking/service.py (its single production construction
    # site) once the real class is resolvable.
    from app.job_ranking.routes.types import RankedJobDocument


class Llm3Outcome(StrEnum):
    """What the LLM3 step did for one job (see ``RERANKER_ENABLE_LLM3`` / ``RERANKER_LLM3_MODE``)."""

    DISABLED = "disabled"
    """``RERANKER_ENABLE_LLM3=0``: the occupation-skills menu was not used."""
    NO_MENU = "no_menu"
    """No essential/optional occupation skills to put on a menu."""
    MENU_FILTERED_OUT = "menu_filtered_out"
    """The ``RERANKER_LLM3_*`` filters left the menu empty."""
    FOLDED = "folded"
    """``fold`` mode: the menu went to LLM1 as extra candidates; no LLM3 call."""
    CALLED = "called"
    """``call`` mode: the standalone LLM3 relevance call ran."""


class JobStats(BaseModel):
    """Per-step counts for one job — what the batch CLI logs; the API leaves them out.

    Mutated field-by-field as each pipeline step completes, so this is a plain ``BaseModel``
    with assignment validation, not the frozen :class:`~app.domain_base.DomainModel`.
    """

    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)

    classifier_skill_candidates: int = Field(
        default=0, description="Classifier ESCO skill links considered."
    )
    occupation_blocks: int = Field(
        default=0, description="Occupation blocks in `occupation_to_skills`."
    )
    taxonomy_menu_skills: int = Field(
        default=0, description="Essential+optional taxonomy skills before LLM3 menu filters."
    )
    llm3_prompt_skills: int = Field(
        default=0, description="Essential+optional taxonomy skills actually sent to LLM3."
    )
    llm3_added_llm1_candidates: int = Field(
        default=0, description="LLM3 menu skills folded into LLM1's candidates (`fold` mode)."
    )
    llm3_decisions: int = Field(default=0, description="Relevance decisions LLM3 returned.")
    llm1_menu: int = Field(default=0, description="Candidate skills LLM1 was offered.")
    llm1_essential: int = Field(default=0, description="Skills LLM1 ranked essential.")
    llm1_optional: int = Field(default=0, description="Skills LLM1 ranked optional.")
    occupation_ranker_counts: tuple[int, int, int] | None = Field(
        default=None,
        description="(essential, optional, unranked) occupation-relation skills, when LLM1 ran "
        "and there were any.",
    )
    attribute_estimates: int = Field(default=0, description="LLM2 attribute estimates returned.")
    work_activity_ids: int = Field(default=0, description="ONET work activity ids LLM2 selected.")
    taxonomy_detail: list[str] = Field(
        default_factory=list, description="Per-classifier-hit taxonomy resolution log lines."
    )


class Llm1Llm2Result(DomainModel):
    """LLM1 (skill ranker) and LLM2 (job attributes) outcomes for one job.

    Returned by :meth:`JobRankingService._run_llm1_llm2`, which runs the two calls
    sequentially or concurrently (``RERANKER_OVERLAP_LLM1_LLM2``) depending on settings.
    """

    skill_ranker: SkillRankerResult | None = Field(
        description="LLM1's result, or None when there were no candidate skills to rank."
    )
    job_attributes: JobAttributesResult = Field(description="LLM2's result; always runs.")


class JobResult(BaseModel):
    """Output of processing a single classified job document.

    Mutated field-by-field as each pipeline step completes, so this is a plain ``BaseModel``
    with assignment validation, not the frozen :class:`~app.domain_base.DomainModel`.
    """

    model_config = ConfigDict(
        strict=True, extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
    )

    ranked_document: RankedJobDocument | None = Field(
        description="The ranked job, or None until LLM2's result is assembled into one."
    )
    ctx: JobDescriptionContext = Field(description="The job's extracted listing context.")
    timings: dict[str, float] = Field(
        default_factory=dict, description="Wall-clock seconds per step that ran."
    )
    ran_llm1: bool = Field(default=False, description="LLM1 (skill ranker) was called.")
    ran_llm2: bool = Field(default=False, description="LLM2 (job attributes) was called.")
    ran_llm3: bool = Field(
        default=False, description="The standalone LLM3 relevance call was made."
    )
    llm3_outcome: Llm3Outcome = Field(
        default=Llm3Outcome.DISABLED, description="What the LLM3 step did for this job."
    )
    stats: JobStats = Field(default_factory=JobStats, description="Per-step counts for this job.")

    def trace_output(self) -> dict[str, Any]:
        """What a job's Langfuse trace records about the run."""
        return {
            "ran_llm1": self.ran_llm1,
            "ran_llm2": self.ran_llm2,
            "ran_llm3": self.ran_llm3,
            "llm3_outcome": self.llm3_outcome.value,
            "ranked": self.ranked_document is not None,
            "timings": self.timings,
        }


class PromptLogging(DomainModel):
    """Which assembled prompts to log for one job (``main.py --log-llm*-prompt``)."""

    llm1: bool = Field(default=False, description="Log LLM1's assembled prompt.")
    llm2: bool = Field(default=False, description="Log LLM2's assembled prompt.")
    llm3: bool = Field(default=False, description="Log LLM3's assembled prompt.")
    job_label: str = Field(default="", description="Label for this job in log lines.")

    @property
    def any(self) -> bool:
        return self.llm1 or self.llm2 or self.llm3

    @property
    def label(self) -> str:
        return (self.job_label or "").strip() or "(job)"


NO_PROMPT_LOGGING = PromptLogging()


PROMPT_LOG_SEPARATOR = "-" * 80


class HealthStatus(DomainModel):
    """Which language packs this process serves and has loaded."""

    taxonomy_loaded: bool = Field(description="The default language's taxonomy is loaded.")
    default_language: str = Field(description="Language used when a request names none.")
    supported_languages: list[str] = Field(description="Language packs this image ships.")
    loaded_languages: list[str] = Field(description="Language packs loaded so far.")
