"""LLM 3 response: one ``is_relevant`` flag per occupation-linked skill. No IO."""

from __future__ import annotations

from app.job_ranking.occupation_skills.types import (
    OccupationSkillsRelevanceDecision,
    OccupationSkillsRelevanceResult,
)
from app.llm.errors import LlmParseError
from app.llm_models import RelevanceDecisionRaw, parse_llm_json_response, strip_json_fence
from utils.classifier_linked_entities import json_bool


def parse_occupation_skills_relevance_response(raw_text: str) -> OccupationSkillsRelevanceResult:
    """
    Parse LLM call 3 JSON: ``{ "decisions": [ { "skill_id", "tabiya_occupation_id", "is_relevant" } ] }``.

    Extra fields on objects are ignored. Duplicate (tabiya_occupation_id, skill_id) keys: last wins.
    """
    data = parse_llm_json_response(raw_text)
    raw_list = data.get("decisions")
    if not isinstance(raw_list, list):
        raise LlmParseError("Occupation skills relevance JSON must contain a decisions array")
    merged: dict[tuple[str, str], OccupationSkillsRelevanceDecision] = {}
    for raw_item in raw_list:
        if not isinstance(raw_item, dict):
            continue
        item = RelevanceDecisionRaw.model_validate(raw_item)
        sid = item.skill_id.strip()
        tid = item.tabiya_occupation_id.strip()
        if not sid or not tid:
            continue
        merged[(tid.lower(), sid.lower())] = OccupationSkillsRelevanceDecision(
            skill_id=sid,
            tabiya_occupation_id=tid,
            is_relevant=json_bool(item.is_relevant),
        )
    decisions = tuple(merged.values())
    return OccupationSkillsRelevanceResult(
        decisions=decisions, raw_response=strip_json_fence(raw_text)
    )


class OccupationSkillsRelevanceResponseParser:
    """LLM call 3: one ``is_relevant`` flag per occupation-linked skill."""

    def parse(self, raw: str) -> OccupationSkillsRelevanceResult:
        return parse_occupation_skills_relevance_response(raw)
