"""Concat skill labels into one embedding string (aligned with shp match_v4)."""

from __future__ import annotations

from typing import Any, Sequence

DEFAULT_SEPARATOR = " | "


def _dedupe_stable(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in labels:
        s = str(raw).strip()
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def build_concat_embedding_text(
    labels: Sequence[str],
    *,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    return separator.join(_dedupe_stable(list(labels)))


def ranked_job_skill_labels_for_concat(ranked_doc: dict[str, Any]) -> list[str]:
    """Essential ∪ optional labels from ``llm_classified_skills``."""
    out: list[str] = []
    lcs = ranked_doc.get("llm_classified_skills")
    if not isinstance(lcs, dict):
        return out

    def _take(items: object) -> None:
        if not isinstance(items, list):
            return
        for s in items:
            if isinstance(s, dict):
                lab = s.get("label")
                if lab:
                    out.append(str(lab))

    _take(lcs.get("essential"))
    _take(lcs.get("optional"))
    return _dedupe_stable(out)


def ranked_job_concat_embedding_text(
    ranked_doc: dict[str, Any],
    *,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    return build_concat_embedding_text(
        ranked_job_skill_labels_for_concat(ranked_doc),
        separator=separator,
    )
