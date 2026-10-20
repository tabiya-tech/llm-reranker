"""JSON Schemas for the three LLM passes, sent as the provider's structured-output schema.

Each prompt already spells out its JSON shape in prose; these schemas say the same thing
in a form the model is decoded against (Gemini's ``response_json_schema``), so the reply
always parses and every id/label it carries is one the prompt offered. The pass parsers
(``app.job_ranking.*.parser``) still validate, because a
provider without structured output gets only the prose contract.

Menus become ``enum`` lists, built per call from the menu the prompt shows. Above
:data:`MAX_ENUM_VALUES` an ``enum`` is dropped in favour of a plain string: very large
enums make the schema too big for the API to accept, and the parsers already discard
values outside the menu.
"""

from __future__ import annotations

from typing import Any, Iterable

from app.job_ranking.job_attributes.types import JobAttributesSchema, OnetWorkActivity

# Largest menu sent as an ``enum``; beyond it the field is a plain string.
MAX_ENUM_VALUES = 500

CONFIDENCE_LEVELS = ("high", "medium", "low")


def _string_field(values: Iterable[str]) -> dict[str, Any]:
    """``{"type": "string", "enum": [...]}`` over the distinct non-empty values, in order."""
    distinct = list(dict.fromkeys(v for v in values if v))
    if not distinct or len(distinct) > MAX_ENUM_VALUES:
        return {"type": "string"}
    return {"type": "string", "enum": distinct}


def skill_ranker_response_schema(skill_labels: Iterable[str]) -> dict[str, Any]:
    """LLM1: ``essential_skills`` / ``optional_skills``, each a list of menu labels."""
    label = _string_field(skill_labels)
    return {
        "type": "object",
        "properties": {
            "essential_skills": {"type": "array", "items": label},
            "optional_skills": {"type": "array", "items": label},
        },
        "required": ["essential_skills", "optional_skills"],
    }


def job_attributes_response_schema(
    schema: JobAttributesSchema,
    onet: Iterable[OnetWorkActivity],
    *,
    requires_post_secondary: bool = False,
) -> dict[str, Any]:
    """LLM2: one estimate per attribute, a summary, 3–10 ONET ids (and the Kenya flag).

    Level ids are the union across attributes; whether a level belongs to *its*
    attribute is left to :func:`app.job_ranking.job_attributes.parser.parse_job_attributes_response`, which keeps
    the schema small and free of per-attribute branches.
    """
    n_attrs = len(schema.attributes)
    attributes: dict[str, Any] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "attribute_name": _string_field(a.name for a in schema.attributes),
                "selected_level_id": _string_field(
                    lev.id for a in schema.attributes for lev in a.levels
                ),
                "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
                "reasoning": {"type": "string"},
            },
            "required": ["attribute_name", "selected_level_id", "confidence", "reasoning"],
        },
    }
    if n_attrs:
        attributes["minItems"] = n_attrs
        attributes["maxItems"] = n_attrs
    properties: dict[str, Any] = {
        "attributes": attributes,
        "summary": {"type": "string"},
        "work_activities": {
            "type": "array",
            "items": _string_field(x.wa_element_id for x in onet),
            "minItems": 3,
            "maxItems": 10,
        },
    }
    required = ["attributes", "summary", "work_activities"]
    if requires_post_secondary:
        properties["requires_post_secondary"] = {"type": "boolean"}
        required.append("requires_post_secondary")
    return {"type": "object", "properties": properties, "required": required}


def occupation_skills_relevance_response_schema(
    occupation_to_skills_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """LLM3: one ``{skill_id, tabiya_occupation_id, is_relevant}`` per judged skill."""
    occupation_ids: list[str] = []
    skill_ids: list[str] = []
    for block in occupation_to_skills_items:
        tid = str(block.get("tabiya_occupation_id") or "").strip()
        if not tid:
            continue
        for sk in block.get("skills") or []:
            if not isinstance(sk, dict):
                continue
            if (sk.get("relation_type") or "").strip().lower() not in ("essential", "optional"):
                continue
            sid = str(sk.get("skill_id") or "").strip()
            if sid:
                occupation_ids.append(tid)
                skill_ids.append(sid)
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "skill_id": _string_field(skill_ids),
                        "tabiya_occupation_id": _string_field(occupation_ids),
                        "is_relevant": {"type": "boolean"},
                    },
                    "required": ["skill_id", "tabiya_occupation_id", "is_relevant"],
                },
            },
        },
        "required": ["decisions"],
    }
