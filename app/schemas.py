"""Load job attribute enums and ONET work-activity lists from JSON (path from ``Settings.job_attributes_schema_path``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings, language_resource_dirs
from app.llm.errors import LlmParseError
from utils.classifier_linked_entities import strip_json_fence


@dataclass(frozen=True)
class AttributeLevel:
    id: str
    label: str
    value: float | None = None


@dataclass(frozen=True)
class AttributeDefinition:
    name: str
    label: str
    group: str
    type: str
    coding: str
    levels: tuple[AttributeLevel, ...]
    base_level_id: str | None = None


@dataclass(frozen=True)
class JobAttributesSchema:
    attributes: tuple[AttributeDefinition, ...]
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class OnetWorkActivity:
    wa_element_id: str
    wa_element_name: str
    wa_element_name_simplified: str
    wa_element_name_swahili: str | None = None


@dataclass(frozen=True)
class AttributeEstimate:
    """One attribute row from LLM2 JSON (name, level, confidence, reasoning)."""

    attribute_name: str
    selected_level_id: str
    confidence: str
    reasoning: str


@dataclass(frozen=True)
class JobAttributesResult:
    """Parsed output of LLM call 2: structured estimates + summary + ONET work activities."""

    attribute_estimates: tuple[AttributeEstimate, ...]
    summary: str
    attributes: dict[str, str | bool]
    work_activity_ids: tuple[str, ...]
    raw_response: str


@dataclass(frozen=True)
class OccupationSkillsRelevanceDecision:
    """One row from LLM call 3 JSON (occupation taxonomy skill vs this job ad)."""

    skill_id: str
    tabiya_occupation_id: str
    is_relevant: bool


@dataclass(frozen=True)
class OccupationSkillsRelevanceResult:
    """Parsed LLM call 3 output: flags only, no summary or reasons."""

    decisions: tuple[OccupationSkillsRelevanceDecision, ...]
    raw_response: str


def job_attributes_schema_path(settings: Settings | None = None) -> Path:
    """Resolved path from env (see ``JOB_ATTRIBUTES_SCHEMA`` in ``app.config``)."""
    settings = settings or get_settings()
    return settings.job_attributes_schema_path


def onet_work_activities_path(settings: Settings | None = None) -> Path:
    """``ONET_tasks_WAs_translated.json`` from the run's language pack.

    Falls back to the English pack: a language that has not translated the ONET work
    activity descriptors still gets them, and the ``WA_Element_ID`` values downstream
    stages key on are language-neutral either way.
    """
    settings = settings or get_settings()
    name = "ONET_tasks_WAs_translated.json"
    for directory in language_resource_dirs(settings.resources_dir, settings.language):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return language_resource_dirs(settings.resources_dir, settings.language)[0] / name


def load_job_attributes_schema(
    path: Path | None = None, *, settings: Settings | None = None
) -> JobAttributesSchema:
    p = path or job_attributes_schema_path(settings)
    return _load_job_attributes_schema_cached(str(p.resolve()))


@lru_cache(maxsize=4)
def _load_job_attributes_schema_cached(resolved_path: str) -> JobAttributesSchema:
    p = Path(resolved_path)
    if not p.is_file():
        raise FileNotFoundError(f"Job attributes schema not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    attrs_out: list[AttributeDefinition] = []
    for a in data.get("attributes") or []:
        levels: list[AttributeLevel] = []
        for lev in a.get("levels") or []:
            raw_v = lev.get("value")
            val = float(raw_v) if raw_v is not None else None
            levels.append(
                AttributeLevel(
                    id=str(lev["id"]),
                    label=str(lev["label"]),
                    value=val,
                )
            )
        attrs_out.append(
            AttributeDefinition(
                name=str(a["name"]),
                label=str(a["label"]),
                group=str(a["group"]),
                type=str(a["type"]),
                coding=str(a["coding"]),
                levels=tuple(levels),
                base_level_id=a.get("base_level_id"),
            )
        )
    meta = data.get("metadata")
    if meta is not None and not isinstance(meta, dict):
        meta = None
    return JobAttributesSchema(attributes=tuple(attrs_out), metadata=meta)


def load_onet_work_activities(
    path: Path | None = None, *, settings: Settings | None = None
) -> tuple[OnetWorkActivity, ...]:
    p = path or onet_work_activities_path(settings)
    return _load_onet_work_activities_cached(str(p.resolve()))


@lru_cache(maxsize=4)
def _load_onet_work_activities_cached(resolved_path: str) -> tuple[OnetWorkActivity, ...]:
    p = Path(resolved_path)
    if not p.is_file():
        raise FileNotFoundError(f"ONET work activities JSON not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("ONET_tasks_WAs_translated.json must be a JSON array")
    out: list[OnetWorkActivity] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        out.append(
            OnetWorkActivity(
                wa_element_id=str(row.get("WA_Element_ID") or "").strip(),
                wa_element_name=str(row.get("WA_Element_Name") or "").strip(),
                wa_element_name_simplified=str(
                    row.get("WA_Element_Name_simplified") or row.get("WA_Element_Name") or ""
                ).strip(),
                wa_element_name_swahili=(
                    str(row["WA_Element_Name_swahili"]).strip()
                    if row.get("WA_Element_Name_swahili")
                    else None
                ),
            )
        )
    return tuple(x for x in out if x.wa_element_id)


def allowed_level_ids_by_attribute(schema: JobAttributesSchema) -> dict[str, frozenset[str]]:
    return {a.name: frozenset(lev.id for lev in a.levels) for a in schema.attributes}


def _normalize_confidence(raw: object) -> str:
    s = str(raw or "").strip().lower()
    if s in ("high", "medium", "low"):
        return s
    return "medium"


def _json_bool(v: object, default: bool = True) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v or "").strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return default


def parse_country_field_flag(raw: object) -> bool:
    """A country's extra LLM2 boolean (e.g. Kenya's ``requires_post_secondary``); omitted or
    unparseable -> ``false``."""
    if raw is None:
        return False
    return _json_bool(raw, default=False)


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
            is_relevant=_json_bool(item.get("is_relevant")),
        )
    decisions = tuple(merged.values())
    return OccupationSkillsRelevanceResult(decisions=decisions, raw_response=cleaned)


def parse_job_attributes_response(
    raw_text: str,
    schema: JobAttributesSchema,
    onet: tuple[OnetWorkActivity, ...],
    *,
    settings: Settings | None = None,
) -> JobAttributesResult:
    """
    Parse and validate LLM JSON.

    Expects ``attributes`` as a **list** of objects with ``attribute_name``, ``selected_level_id``,
    ``confidence``, ``reasoning``. Also accepts a legacy **object** map
    ``name -> level_id`` and synthesizes estimates with medium confidence.

    ``summary`` is optional text; ``work_activities`` is a list of allowed ONET ids.

    Each of the country's extra fields (:attr:`Settings.job_attributes_country_fields`, e.g.
    Kenya's ``requires_post_secondary``) is stored as a boolean on the flat ``attributes``
    map under its own key, for downstream matching eligibility.
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
    s = settings or get_settings()
    for field in s.job_attributes_country_fields:
        flat[field] = parse_country_field_flag(data.get(field))
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
