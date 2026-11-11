"""``POST /rank/skills``: rank one experience's candidate skills with a single LLM call."""

from __future__ import annotations

from app.job_ranking.experience_skills.parser import ExperienceSkillRankerResponseParser
from app.job_ranking.experience_skills.prompt import ExperienceSkillRankerPromptBuilder
from app.job_ranking.experience_skills.types import ExperienceSkillsResult
from app.job_ranking.response_schemas import experience_skill_ranker_response_schema
from app.job_ranking.routes.types import ExperienceDetails, SkillCandidate
from app.llm import TracedLlmClient

# Label passed to the client as ``purpose`` (provider logs, errors, Langfuse generations).
PURPOSE_EXPERIENCE_SKILL_RANKER = "experience_skill_ranker"


def prompt_menu(candidates: list[SkillCandidate]) -> list[tuple[str, SkillCandidate]]:
    """Give each candidate a short prompt id (``S1``, ``S2``, ...).

    Short ids are cheaper to copy than the caller's (often UUIDs) and never collide with
    a label, so the answer maps back without fuzzy matching.
    """
    return [(f"S{i}", c) for i, c in enumerate(candidates, 1)]


class ExperienceSkillRankerStep:
    """One LLM call: the candidate skills the experience shows, best first."""

    def __init__(
        self,
        llm: TracedLlmClient,
        prompts: ExperienceSkillRankerPromptBuilder,
        parser: ExperienceSkillRankerResponseParser,
    ):
        self._llm = llm
        self._prompts = prompts
        self._parser = parser

    def run(
        self,
        experience: ExperienceDetails,
        candidates: list[SkillCandidate],
        max_skills: int,
    ) -> ExperienceSkillsResult:
        menu = prompt_menu(candidates)
        prompt = self._prompts.build(experience, menu, max_skills)
        raw = self._llm.generate_json(
            prompt,
            purpose=PURPOSE_EXPERIENCE_SKILL_RANKER,
            response_schema=experience_skill_ranker_response_schema(pid for pid, _ in menu),
        )
        return self._parser.parse(raw, {pid: c.id for pid, c in menu}, max_skills)
