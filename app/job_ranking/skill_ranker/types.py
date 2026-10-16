"""LLM 1 (skill ranker) result."""

from __future__ import annotations

from dataclasses import dataclass

from utils.classifier_linked_entities import (
    MappedEscoSkills,
)


@dataclass(frozen=True)
class SkillRankerResult:
    """Parsed LLM call 1: essential / optional labels and their mapping back to the candidates."""

    essential: list[str]
    optional: list[str]
    mapped: MappedEscoSkills
    raw_response: str
