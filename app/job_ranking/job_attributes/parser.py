"""LLM 2 response: estimates validated against the schema, ONET ids against the list. No IO."""

from __future__ import annotations

import json
from dataclasses import replace

from app.config import Settings
from app.job_ranking.job_attributes.types import (
    AttributeEstimate,
    JobAttributesResult,
    JobAttributesSchema,
    OnetWorkActivity,
)
from app.llm.errors import LlmParseError
from utils.classifier_linked_entities import (
    json_bool,
    strip_json_fence,
)


def allowed_level_ids_by_attribute(schema: JobAttributesSchema) -> dict[str, frozenset[str]]:
    return {a.name: frozenset(lev.id for lev in a.levels) for a in schema.attributes}


def _normalize_confidence(raw: object) -> str:
    s = str(raw or "").strip().lower()
    if s in ("high", "medium", "low"):
        return s
    return "medium"


def parse_requires_post_secondary(raw: object) -> bool:
    """Kenya eligibility flag from LLM2 JSON; omitted or unparseable -> ``false``."""
    if raw is None:
        return False
    return json_bool(raw, default=False)


def parse_job_attributes_response(
    raw_text: str,
    schema: JobAttributesSchema,
    onet: tuple[OnetWorkActivity, ...],
    *,
    settings: Settings,
) -> JobAttributesResult:
    """
    Parse and validate LLM JSON.

    Expects ``attributes`` as a **list** of objects with ``attribute_name``, ``selected_level_id``,
    ``confidence``, ``reasoning``. Also accepts a legacy **object** map
    ``name -> level_id`` and synthesizes estimates with medium confidence.

    ``summary`` is optional text; ``work_activities`` is a list of allowed ONET ids.

    When present, ``requires_post_secondary`` (boolean) is stored on the flat ``attributes`` map
    under that key for downstream matching eligibility (Kenya pipeline).
    """
    cleaned = strip_json_fence(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LlmParseError(f"Job attributes JSON could not be parsed: {e}") from e
    if not isinstance(data, dict):
        raise LlmParseError("Job attributes JSON must be an object")

    allowed_levels = allowed_level_ids_by_attribute(schema)
    attr_names = frozenset(allowed_levels.keys())

    by_name: dict[str, AttributeEstimate] = {}

    raw_attrs = data.get("attributes")
    if isinstance(raw_attrs, list):
        for item in raw_attrs:
            if not isinstance(item, dict):
                continue
            name = str(item.get("attribute_name") or "").strip()
            lid_raw = item.get("selected_level_id")
            if name not in attr_names or not isinstance(lid_raw, str) or not lid_raw.strip():
                continue
            lid = lid_raw.strip()
            if lid not in allowed_levels[name]:
                continue
            by_name[name] = AttributeEstimate(
                attribute_name=name,
                selected_level_id=lid,
                confidence=_normalize_confidence(item.get("confidence")),
                reasoning=str(item.get("reasoning") or "").strip(),
            )
    elif isinstance(raw_attrs, dict):
        for name, level_id in raw_attrs.items():
            if name not in attr_names:
                continue
            if not isinstance(level_id, str) or not level_id.strip():
                continue
            lid = level_id.strip()
            if lid not in allowed_levels[name]:
                continue
            by_name[name] = AttributeEstimate(
                attribute_name=name,
                selected_level_id=lid,
                confidence="medium",
                reasoning="",
            )

    estimates = tuple(by_name[a.name] for a in schema.attributes if a.name in by_name)
    flat: dict[str, str | bool] = {e.attribute_name: e.selected_level_id for e in estimates}
    if settings.target_country == "kenya":
        flat["requires_post_secondary"] = parse_requires_post_secondary(
            data.get("requires_post_secondary")
        )
    summary_raw = data.get("summary")
    summary = str(summary_raw).strip() if summary_raw is not None else ""

    wa_allowed = frozenset(x.wa_element_id for x in onet)
    wa_out: list[str] = []
    raw_wa = data.get("work_activities")
    if isinstance(raw_wa, list):
        for x in raw_wa:
            if isinstance(x, str) and x.strip() in wa_allowed:
                wid = x.strip()
                if wid not in wa_out:
                    wa_out.append(wid)

    return JobAttributesResult(
        attribute_estimates=tuple(estimates),
        summary=summary,
        attributes=flat,
        work_activity_ids=tuple(wa_out),
        raw_response=cleaned,
    )


class JobAttributesResponseParser:
    """LLM call 2: attribute estimates validated against the schema, ONET ids against the list."""

    def __init__(
        self,
        schema: JobAttributesSchema,
        onet: tuple[OnetWorkActivity, ...],
        settings: Settings,
    ):
        self._schema = schema
        self._onet = onet
        self._settings = settings

    def parse(self, raw: str) -> JobAttributesResult:
        parsed = parse_job_attributes_response(
            raw, self._schema, self._onet, settings=self._settings
        )
        return replace(parsed, raw_response=raw)
