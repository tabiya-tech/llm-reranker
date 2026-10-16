"""LLM 3 response: one ``is_relevant`` flag per occupation-linked skill. No IO."""

from __future__ import annotations

import json

from app.job_ranking.occupation_skills.types import (
    OccupationSkillsRelevanceDecision,
    OccupationSkillsRelevanceResult,
)
from app.llm.errors import LlmParseError
from utils.classifier_linked_entities import (
    json_bool,
    strip_json_fence,
)


def parse_occupation_skills_relevance_response(raw_text: str) -> OccupationSkillsRelevanceResult:
    """
    Parse LLM call 3 JSON: ``{ "decisions": [ { "skill_id", "tabiya_occupation_id", "is_relevant" } ] }``.

    Extra fields on objects are ignored. Duplicate (tabiya_occupation_id, skill_id) keys: last wins.
    """
    cleaned = strip_json_fence(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LlmParseError(f"Occupation skills relevance JSON could not be parsed: {e}") from e
    if not isinstance(data, dict):
        raise LlmParseError("Occupation skills relevance JSON must be an object")
    raw_list = data.get("decisions")
    if not isinstance(raw_list, list):
        raise LlmParseError("Occupation skills relevance JSON must contain a decisions array")
    merged: dict[tuple[str, str], OccupationSkillsRelevanceDecision] = {}
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("skill_id") or "").strip()
        tid = str(item.get("tabiya_occupation_id") or "").strip()
        if not sid or not tid:
            continue
        merged[(tid.lower(), sid.lower())] = OccupationSkillsRelevanceDecision(
            skill_id=sid,
            tabiya_occupation_id=tid,
            is_relevant=json_bool(item.get("is_relevant")),
        )
    decisions = tuple(merged.values())
    return OccupationSkillsRelevanceResult(decisions=decisions, raw_response=cleaned)


class OccupationSkillsRelevanceResponseParser:
    """LLM call 3: one ``is_relevant`` flag per occupation-linked skill."""

    def parse(self, raw: str) -> OccupationSkillsRelevanceResult:
        return parse_occupation_skills_relevance_response(raw)
