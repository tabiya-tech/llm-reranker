"""LLM 1 response: label lists mapped back onto the candidate menu. No IO."""

from __future__ import annotations

from app.job_ranking.skill_ranker.types import SkillRankerResult
from utils.classifier_linked_entities import (
    EscoSkillCandidate,
    map_skill_labels_to_esco_uris,
    parse_skill_ranker_json,
)


class SkillRankerResponseParser:
    """LLM call 1: essential / optional label lists, mapped back onto the candidate menu."""

    def parse(self, raw: str, candidates: list[EscoSkillCandidate]) -> SkillRankerResult:
        essential, optional = parse_skill_ranker_json(raw)
        mapped = map_skill_labels_to_esco_uris(essential, optional, candidates)
        return SkillRankerResult(
            essential=essential, optional=optional, mapped=mapped, raw_response=raw
        )
