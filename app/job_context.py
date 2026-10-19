"""Shared job listing context (Classifier document fields + full ``classified`` doc).

``build_job_description_context`` extracts it from a classified job document. Pure: no Mongo I/O.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from bson import ObjectId
from pydantic import Field

from app.config import Settings
from app.domain_base import DomainModel
from app.translation import overlay_translation_listing_fields

if TYPE_CHECKING:
    # Deferred: app.job_ranking.routes.types -> app.job_ranking.types -> app.job_context
    # (this module) — a real top-level import here would be a circular import.
    from app.job_ranking.routes.types import RankRequest


def stringify_top_level_object_ids(doc: dict[str, Any]) -> dict[str, Any]:
    """``ObjectId`` values at the top level -> ``str``, so ``RankRequest`` can validate raw
    pymongo documents (which carry real BSON types, unlike the API's JSON/Extended-JSON input).
    """
    return {k: (str(v) if isinstance(v, ObjectId) else v) for k, v in doc.items()}


class JobDescriptionContext(DomainModel):
    """
    Fields needed for downstream steps, sourced from classified job documents.

    Field names used to populate this object come from ``Settings`` (environment variables).

    The full source document is always available on ``classified`` (includes
    ``classification``, scraper enrichment, etc.) as a validated, permissive-extra
    :class:`~app.job_ranking.routes.types.RankRequest` — call ``.to_document()`` to get it
    back as a plain dict, e.g. for dynamic/configured key lookups.
    """

    job_id: str | int | None = Field(description="Job id, normalized from the source document.")
    job_fingerprint: str | None = Field(description="Listing fingerprint.")
    title: str | None = Field(description="Listing title.")
    employer: str | None = Field(description="Employer name.")
    category: str | None = Field(description="Listing category.")
    application_url: str | None = Field(description="Where to apply.")
    posted_date: str | None = Field(description="Posting date, as the source wrote it.")
    closing_date: str | None = Field(description="Closing date, as the source wrote it.")
    source_platform: str | None = Field(description="Job board the listing came from.")
    raw_job_description: str | None = Field(description="Full job description text.")
    requirements_text: str | None = Field(description="Requirements text.")
    salary: str | None = Field(description="Salary text.")
    city: str | None = Field(description="City; a list in the source is joined with `, `.")
    county: str | None = Field(description="County/region (older documents; see `province`).")
    employment_type: str | None = Field(description="Employment type text.")
    classified: RankRequest = Field(description="The full source document, validated.")

    @classmethod
    def _ensure_rebuilt(cls) -> None:
        """Resolve the deferred ``RankRequest`` forward ref, idempotently.

        Call this before constructing a ``JobDescriptionContext`` directly (outside
        ``build_job_description_context``, which already calls it) — e.g. in tests —
        so construction doesn't depend on some other module having imported
        ``app.job_ranking.routes.types`` first in this process.
        """
        if not cls.__pydantic_complete__:
            from app.job_ranking.routes.types import RankRequest  # noqa: F401

            cls.model_rebuild()


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


def _normalize_job_id(raw: object) -> str | int | None:
    """Mongo ``ObjectId`` -> ``str``; ``str``/``int`` pass through; anything else -> ``None``."""
    if raw is None:
        return None
    if isinstance(raw, (str, int)):
        return raw
    return str(raw)


def build_job_description_context(
    classified: RankRequest | dict[str, Any],
    *,
    settings: Settings,
) -> JobDescriptionContext:
    """
    Extract listing fields from a classified_jobs document.

    All Mongo field names are configured via ``Settings`` / environment variables.
    """
    from app.job_ranking.routes.types import RankRequest

    JobDescriptionContext._ensure_rebuilt()
    classified = (
        classified
        if isinstance(classified, RankRequest)
        else RankRequest.model_validate(stringify_top_level_object_ids(classified))
    )
    classified_doc = classified.to_document()

    fp = _first_str(classified_doc, *settings.keys_fingerprint)
    job_id = _normalize_job_id(classified_doc.get(settings.mongo_field_job_id))

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
        classified=classified,
    )
