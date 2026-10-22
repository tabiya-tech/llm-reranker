"""LLM 2 response: estimates validated against the schema, ONET ids against the list. No IO."""

from __future__ import annotations

from pydantic import ValidationError

from app.config import Settings
from app.job_ranking.job_attributes.types import (
    AttributeEstimate,
    JobAttributesResult,
    JobAttributesSchema,
    OnetWorkActivity,
)
from app.llm.errors import LlmParseError
from app.llm_models import (
    AttributeEstimateRaw,
    JobAttributesLlmResponse,
    parse_llm_json_response,
    strip_json_fence,
)
from utils.classifier_linked_entities import json_bool


def allowed_level_ids_by_attribute(schema: JobAttributesSchema) -> dict[str, frozenset[str]]:
    return {a.name: frozenset(lev.id for lev in a.levels) for a in schema.attributes}


def _normalize_confidence(raw: object) -> str:
    s = str(raw or "").strip().lower()
    if s in ("high", "medium", "low"):
        return s
    return "medium"


def parse_extra_field_flag(raw: object) -> bool:
    """One of the deployment's extra LLM2 booleans (e.g. ``requires_post_secondary``);
    omitted or unparseable -> ``false``."""
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

    Each of the deployment's extra fields (:attr:`Settings.job_attributes_extra_fields`, e.g.
    Kenya's ``requires_post_secondary``) is stored as a boolean on the flat ``attributes`` map
    under its own key, for downstream matching eligibility.
    """
    cleaned = strip_json_fence(raw_text)
    data = parse_llm_json_response(raw_text)
    try:
        parsed = JobAttributesLlmResponse.model_validate(data)
    except ValidationError as e:
        raise LlmParseError(f"Job attributes JSON has an unexpected shape: {e}") from e

    allowed_levels = allowed_level_ids_by_attribute(schema)
    attr_names = frozenset(allowed_levels.keys())

    by_name: dict[str, AttributeEstimate] = {}

    raw_attrs = parsed.attributes
    if isinstance(raw_attrs, list):
        for raw_item in raw_attrs:
            if not isinstance(raw_item, dict):
                continue
            item = AttributeEstimateRaw.model_validate(raw_item)
            name = str(item.attribute_name or "").strip()
            lid_raw = item.selected_level_id
            if name not in attr_names or not isinstance(lid_raw, str) or not lid_raw.strip():
                continue
            lid = lid_raw.strip()
            if lid not in allowed_levels[name]:
                continue
            by_name[name] = AttributeEstimate(
                attribute_name=name,
                selected_level_id=lid,
                confidence=_normalize_confidence(item.confidence),
                reasoning=str(item.reasoning or "").strip(),
            )
    else:
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
    for field in settings.job_attributes_extra_fields:
        flat[field.name] = parse_extra_field_flag(data.get(field.name))
    summary_raw = parsed.summary
    summary = str(summary_raw).strip() if summary_raw is not None else ""

    wa_allowed = frozenset(x.wa_element_id for x in onet)
    wa_out: list[str] = []
    for x in parsed.work_activities:
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
        return parsed.model_copy(update={"raw_response": raw})
