"""Shared job listing context (Classifier document fields + full ``classified`` doc).

``build_job_description_context`` extracts it from a classified job document. Pure: no Mongo I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.config import Settings, get_settings
from app.translation import overlay_translation_listing_fields


@dataclass(frozen=True)
class JobDescriptionContext:
    """
    Fields needed for downstream steps, sourced from classified job documents.

    Field names used to populate this object come from ``Settings`` (environment variables).

    The full Mongo document is always available on ``classified`` (includes ``classification``,
    ``metadata``, etc.).
    """

    job_id: Any | None
    job_fingerprint: str | None
    title: str | None
    employer: str | None
    category: str | None
    application_url: str | None
    posted_date: str | None
    closing_date: str | None
    source_platform: str | None
    raw_job_description: str | None
    requirements_text: str | None
    salary: str | None
    city: str | None
    county: str | None
    employment_type: str | None
    classified: dict[str, Any]


def _metadata(doc: dict[str, Any]) -> dict[str, Any]:
    m = doc.get("metadata")
    return m if isinstance(m, dict) else {}


def _scalar_to_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            s = bytes(value).decode("utf-8").strip()
            return s or None
        except UnicodeDecodeError:
            return None
    return None


def _first_str(d: dict[str, Any], *keys: str) -> str | None:
    for k in keys:
        s = _scalar_to_str(d.get(k))
        if s is not None:
            return s
    return None


def _field_or_metadata(
    classified_doc: dict[str, Any],
    key: str,
    *,
    meta_key: str | None = None,
) -> str | None:
    """Prefer top-level ``key``, then ``metadata[meta_key or key]``."""
    v = _first_str(classified_doc, key)
    if v is not None:
        return v
    return _first_str(_metadata(classified_doc), meta_key or key)


def _get_dotted(doc: dict[str, Any], path: str) -> str | None:
    """Resolve ``a.b.c`` style paths; returns None if any segment is missing or not a dict."""
    if "." not in path:
        return _scalar_to_str(doc.get(path))
    cur: Any = doc
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return _scalar_to_str(cur)


def _multi_key_top_or_metadata(doc: dict[str, Any], *keys: str) -> str | None:
    """Try flat keys on top-level, then ``metadata``, then dotted paths on each."""
    flat = [k for k in keys if "." not in k]
    dotted = [k for k in keys if "." in k]
    v = _first_str(doc, *flat) if flat else None
    if v is not None:
        return v
    meta = _metadata(doc)
    v = _first_str(meta, *flat) if flat else None
    if v is not None:
        return v
    for path in dotted:
        v = _get_dotted(doc, path)
        if v is not None:
            return v
        v = _get_dotted(meta, path)
        if v is not None:
            return v
    return None


def build_job_description_context(
    classified_doc: dict[str, Any],
    *,
    settings: Settings | None = None,
) -> JobDescriptionContext:
    """
    Extract listing fields from a classified_jobs document.

    All Mongo field names are configured via ``Settings`` / environment variables.
    """
    settings = settings or get_settings()
    fp = _first_str(classified_doc, *settings.keys_fingerprint)
    job_id = classified_doc.get(settings.mongo_field_job_id)

    title = _field_or_metadata(classified_doc, settings.field_title)
    employer = _field_or_metadata(classified_doc, settings.field_employer)
    category = _field_or_metadata(classified_doc, settings.field_category)
    application_url = _field_or_metadata(classified_doc, settings.field_application_url)
    posted_date = _field_or_metadata(classified_doc, settings.field_posted_date)
    closing_date = _field_or_metadata(classified_doc, settings.field_closing_date)
    source_platform = _field_or_metadata(classified_doc, settings.field_source_platform)

    raw_jd = _multi_key_top_or_metadata(classified_doc, *settings.keys_description)
    req_text = _multi_key_top_or_metadata(classified_doc, *settings.keys_requirements)
    salary = _multi_key_top_or_metadata(classified_doc, *settings.keys_salary)
    city = _multi_key_top_or_metadata(classified_doc, *settings.keys_location)
    county = _multi_key_top_or_metadata(classified_doc, *settings.keys_county)
    employment = _multi_key_top_or_metadata(classified_doc, *settings.keys_employment)

    title, raw_jd, req_text = overlay_translation_listing_fields(
        classified_doc, title=title, raw_jd=raw_jd, req_text=req_text, settings=settings
    )

    return JobDescriptionContext(
        job_id=job_id,
        job_fingerprint=fp,
        title=title,
        employer=employer,
        category=category,
        application_url=application_url,
        posted_date=posted_date,
        closing_date=closing_date,
        source_platform=source_platform,
        raw_job_description=raw_jd,
        requirements_text=req_text,
        salary=salary,
        city=city,
        county=county,
        employment_type=employment,
        classified=dict(classified_doc),
    )
