"""The ``POST /rank/skills`` step: prompt, response schema and parser, with a fake model."""

from __future__ import annotations

import json

import pytest

from app.job_ranking.experience_skills.parser import ExperienceSkillRankerResponseParser
from app.job_ranking.experience_skills.prompt import (
    ExperienceSkillRankerPromptBuilder,
    format_experience_details,
)
from app.job_ranking.experience_skills.step import (
    PURPOSE_EXPERIENCE_SKILL_RANKER,
    ExperienceSkillRankerStep,
)
from app.job_ranking.prompts.repository import PromptRepository
from app.job_ranking.routes.types import ExperienceDetails, SkillCandidate
from app.llm.errors import LlmParseError

pytestmark = pytest.mark.unit


class FakeLlm:
    """Answers every call with ``reply``; records the prompt and response schema."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[tuple[str, str, dict | None]] = []

    def generate_json(self, prompt: str, *, purpose: str, response_schema=None) -> str:
        self.calls.append((prompt, purpose, response_schema))
        return self.reply


def _experience() -> ExperienceDetails:
    return ExperienceDetails(
        title="Baker",
        company="Mama's Bakery",
        work_type="SELF_EMPLOYMENT",
        responsibilities=["I bake bread", "  ", "I sell bread"],
    )


def _candidates() -> list[SkillCandidate]:
    return [
        SkillCandidate(id="uuid-a", label="bake goods", description="Bake bread and cakes."),
        SkillCandidate(id="uuid-b", label="sell products", score=0.7),
        SkillCandidate(id="uuid-c", label="operate a forklift"),
    ]


@pytest.fixture
def step_for(make_settings):
    def _build(reply: str) -> tuple[ExperienceSkillRankerStep, FakeLlm]:
        s = make_settings()
        llm = FakeLlm(reply)
        step = ExperienceSkillRankerStep(
            llm,  # type: ignore[arg-type]
            ExperienceSkillRankerPromptBuilder(PromptRepository().load(s), s),
            ExperienceSkillRankerResponseParser(),
        )
        return step, llm

    return _build


class TestStep:
    def test_maps_prompt_ids_back_to_caller_ids(self, step_for):
        step, llm = step_for(json.dumps({"ranked_skill_ids": ["S2", "S1"]}))

        result = step.run(_experience(), _candidates(), max_skills=5)

        assert result.ranked_ids == ["uuid-b", "uuid-a"]
        assert result.discarded_ids == ["uuid-c"]
        (prompt, purpose, schema) = llm.calls[0]
        assert purpose == PURPOSE_EXPERIENCE_SKILL_RANKER
        assert schema["properties"]["ranked_skill_ids"]["items"]["enum"] == ["S1", "S2", "S3"]

    def test_prompt_shows_short_ids_labels_and_descriptions_but_not_caller_ids(self, step_for):
        step, llm = step_for(json.dumps({"ranked_skill_ids": []}))

        step.run(_experience(), _candidates(), max_skills=3)

        prompt = llm.calls[0][0]
        assert "S1: bake goods" in prompt
        assert "Description: Bake bread and cakes." in prompt
        assert "S3: operate a forklift" in prompt
        assert "Return at most 3 skills." in prompt
        assert "uuid-a" not in prompt
        assert "0.7" not in prompt


class TestParser:
    ID_MAP = {"S1": "a", "S2": "b", "S3": "c"}

    def test_drops_unknown_and_repeated_ids_and_caps_at_max(self):
        raw = json.dumps({"ranked_skill_ids": ["S3", "S9", "S3", " S1 ", 7, "S2"]})

        result = ExperienceSkillRankerResponseParser().parse(raw, self.ID_MAP, max_skills=2)

        assert result.ranked_ids == ["c", "a"]
        assert result.discarded_ids == ["b"]

    def test_rejects_a_reply_without_the_list(self):
        with pytest.raises(LlmParseError, match="ranked_skill_ids"):
            ExperienceSkillRankerResponseParser().parse('{"skills": []}', self.ID_MAP, 5)


def test_experience_details_skip_blank_fields():
    text = format_experience_details(ExperienceDetails(title=" Baker ", company=" "))

    assert text == "Title: Baker"


def test_experience_details_list_the_responsibilities():
    text = format_experience_details(_experience())

    assert text.splitlines() == [
        "Title: Baker",
        "Company: Mama's Bakery",
        "Work type: SELF_EMPLOYMENT",
        "Responsibilities:",
        "- I bake bread",
        "- I sell bread",
    ]
