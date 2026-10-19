"""LLM 1 (skill ranker) result."""

from __future__ import annotations

from pydantic import Field

from app.domain_base import DomainModel
from utils.classifier_linked_entities import (
    MappedEscoSkills,
)


class SkillRankerResult(DomainModel):
    """Parsed LLM call 1: essential / optional labels and their mapping back to the candidates."""

    essential: list[str] = Field(description="Skill labels LLM1 ranked essential.")
    optional: list[str] = Field(description="Skill labels LLM1 ranked optional.")
    mapped: MappedEscoSkills = Field(description="Essential/optional labels mapped to ESCO URIs.")
    raw_response: str = Field(description="LLM1's raw response text.")
