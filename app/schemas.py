"""Load job attribute enums and ONET work-activity lists from JSON (path from ``Settings.job_attributes_schema_path``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
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
class JobAttributesResult:
    """Parsed output of LLM call 2 (unified attributes + task classification)."""

    attributes: dict[str, str]
    work_activity_ids: tuple[str, ...]
    raw_response: str


def job_attributes_schema_path(settings: Settings | None = None) -> Path:
    """Resolved path from env (see ``JOB_ATTRIBUTES_SCHEMA`` in ``app.config``)."""
    settings = settings or get_settings()
    return settings.job_attributes_schema_path


def onet_work_activities_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.resources_dir / "ONET_tasks_WAs_translated.json"


def load_job_attributes_schema(path: Path | None = None, *, settings: Settings | None = None) -> JobAttributesSchema:
    p = path or job_attributes_schema_path(settings)
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


def load_onet_work_activities(path: Path | None = None, *, settings: Settings | None = None) -> tuple[OnetWorkActivity, ...]:
    p = path or onet_work_activities_path(settings)
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


def parse_job_attributes_response(
    raw_text: str,
    schema: JobAttributesSchema,
    onet: tuple[OnetWorkActivity, ...],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """
    Parse and validate LLM JSON: ``attributes`` (name -> level id) and ``work_activities`` (WA_Element_ID list).

    Drops invalid attribute keys, unknown level ids, and unknown work-activity ids.
    """
    cleaned = strip_json_fence(raw_text)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Job attributes JSON must be an object")

    allowed_levels = allowed_level_ids_by_attribute(schema)
    attr_names = frozenset(allowed_levels.keys())
    out_attrs: dict[str, str] = {}

    raw_attrs = data.get("attributes")
    if isinstance(raw_attrs, dict):
        for name, level_id in raw_attrs.items():
            if name not in attr_names:
                continue
            if not isinstance(level_id, str) or not level_id.strip():
                continue
            lid = level_id.strip()
            if lid in allowed_levels[name]:
                out_attrs[name] = lid

    wa_allowed = frozenset(x.wa_element_id for x in onet)
    wa_out: list[str] = []
    raw_wa = data.get("work_activities")
    if isinstance(raw_wa, list):
        for x in raw_wa:
            if isinstance(x, str) and x.strip() in wa_allowed:
                wid = x.strip()
                if wid not in wa_out:
                    wa_out.append(wid)

    return out_attrs, tuple(wa_out)
