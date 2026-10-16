"""Results and options of the per-job pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.job_context import JobDescriptionContext


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


@dataclass
class JobStats:
    """Per-step counts for one job — what the batch CLI logs; the API leaves them out."""

    classifier_skill_candidates: int = 0
    occupation_blocks: int = 0
    taxonomy_menu_skills: int = 0
    llm3_prompt_skills: int = 0
    llm3_added_llm1_candidates: int = 0
    llm3_decisions: int = 0
    llm1_menu: int = 0
    llm1_essential: int = 0
    llm1_optional: int = 0
    # (essential, optional, unranked) occupation-relation skills, when LLM1 ran and there were any.
    occupation_ranker_counts: tuple[int, int, int] | None = None
    attribute_estimates: int = 0
    work_activity_ids: int = 0
    taxonomy_detail: list[str] = field(default_factory=list)


@dataclass
class JobResult:
    """Output of processing a single classified job document."""

    ranked_document: dict[str, Any] | None
    ctx: JobDescriptionContext
    timings: dict[str, float] = field(default_factory=dict)
    ran_llm1: bool = False
    ran_llm2: bool = False
    ran_llm3: bool = False
    llm3_outcome: Llm3Outcome = Llm3Outcome.DISABLED
    stats: JobStats = field(default_factory=JobStats)

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


@dataclass(frozen=True)
class PromptLogging:
    """Which assembled prompts to log for one job (``main.py --log-llm*-prompt``)."""

    llm1: bool = False
    llm2: bool = False
    llm3: bool = False
    job_label: str = ""

    @property
    def any(self) -> bool:
        return self.llm1 or self.llm2 or self.llm3

    @property
    def label(self) -> str:
        return (self.job_label or "").strip() or "(job)"


NO_PROMPT_LOGGING = PromptLogging()


PROMPT_LOG_SEPARATOR = "-" * 80


@dataclass(frozen=True)
class HealthStatus:
    """Which language packs this process serves and has loaded."""

    taxonomy_loaded: bool
    default_language: str
    supported_languages: list[str]
    loaded_languages: list[str]
