"""Reads the job-attributes schema and the ONET work-activity list of a language pack."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings, language_resource_path
from app.job_ranking.job_attributes.types import (
    AttributeDefinition,
    AttributeLevel,
    JobAttributesSchema,
    OnetWorkActivity,
)


def onet_work_activities_path(settings: Settings) -> Path:
    """``ONET_tasks_WAs_translated.json`` from the run's language pack.

    Falls back to the English pack: a language that has not translated the ONET work
    activity descriptors still gets them, and the ``WA_Element_ID`` values downstream
    stages key on are language-neutral either way.
    """
    return language_resource_path(
        settings.resources_dir, settings.language, "ONET_tasks_WAs_translated.json"
    )


def load_job_attributes_schema(path: Path) -> JobAttributesSchema:
    p = path.resolve()
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


def load_onet_work_activities(path: Path) -> tuple[OnetWorkActivity, ...]:
    p = path.resolve()
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


class JobAttributesRepository:
    """The LLM2 reference data of a language pack. Every call re-reads the files."""

    def load_schema(self, settings: Settings) -> JobAttributesSchema:
        """``JOB_ATTRIBUTES_SCHEMA``: attributes and their allowed levels."""
        return load_job_attributes_schema(settings.job_attributes_schema_path)

    def load_onet_work_activities(self, settings: Settings) -> tuple[OnetWorkActivity, ...]:
        """``ONET_tasks_WAs_translated.json`` (falls back to the English pack)."""
        return load_onet_work_activities(onet_work_activities_path(settings))
