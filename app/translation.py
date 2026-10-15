"""Listing-text translation overlay: prefer the classifier's English machine translation.

Pure: no Mongo I/O. Used by ``app.job_context.build_job_description_context``.
"""

from __future__ import annotations

import os
from typing import Any

from app.config import Settings, get_settings
from app.languages import language_setting


def reranker_prefer_translation(settings: Settings | None = None) -> bool:
    """Use ``translation`` on classified_jobs for LLM listing text (English MT path).

    ``RERANKER_PREFER_TRANSLATION`` wins when set. Otherwise the language decides: English
    prompts want the English machine translation, while a language with its own prompt and
    taxonomy pack wants the original text — prompting in Spanish over English MT text would
    mix the two. See ``prefer_translation`` in ``app/languages/{lang}_config.py``.
    """
    raw = (os.getenv("RERANKER_PREFER_TRANSLATION") or "").strip().lower()
    if raw:
        return raw not in ("0", "false", "no")
    settings = settings or get_settings()
    return bool(language_setting(settings.language, "prefer_translation", True))


def overlay_translation_listing_fields(
    classified_doc: dict[str, Any],
    *,
    title: str | None,
    raw_jd: str | None,
    req_text: str | None,
    settings: Settings | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Prefer ``translation.text_en`` / ``fields_en`` when present (e.g. Argentina)."""
    if not reranker_prefer_translation(settings):
        return title, raw_jd, req_text
    tr = classified_doc.get("translation")
    if not isinstance(tr, dict) or not tr:
        return title, raw_jd, req_text
    fields_en_raw = tr.get("fields_en")
    fe: dict[str, Any] = fields_en_raw if isinstance(fields_en_raw, dict) else {}
    text_en = (tr.get("text_en") or "").strip()
    if text_en:
        return (
            ((fe.get("title") or "").strip() or title),
            text_en,
            None,
        )
    if not fe:
        return title, raw_jd, req_text
    t_en = (fe.get("title") or "").strip()
    d_en = (fe.get("description") or "").strip()
    req_parts: list[str] = []
    for k in ("exclusive_requirement", "desirable_requirement"):
        v = (fe.get(k) or "").strip()
        if v:
            req_parts.append(v)
    requirements_en = " ".join(req_parts).strip()
    out_title = t_en or title
    out_desc = d_en or raw_jd
    out_req = requirements_en or req_text
    return out_title, out_desc, out_req
