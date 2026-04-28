from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from app.config import Settings, get_settings


@lru_cache(maxsize=4)
def _load_skill_to_groups_map(csv_path: str) -> dict[str, frozenset[str]]:
    p = Path(csv_path)
    if not p.is_file():
        return {}
    out: dict[str, set[str]] = {}
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("PARENTOBJECTTYPE") != "skillgroup":
                continue
            if row.get("CHILDOBJECTTYPE") != "skill":
                continue
            child = str(row.get("CHILDID") or "").strip()
            parent = str(row.get("PARENTID") or "").strip()
            if not child or not parent:
                continue
            out.setdefault(child, set()).add(parent)
    return {k: frozenset(v) for k, v in out.items()}


def derive_skill_groups_origin_uuids(
    skill_ids: list[str] | set[str] | tuple[str, ...],
    *,
    settings: Settings | None = None,
) -> list[str]:
    """Derive parent skill-group IDs from a collection of skill IDs.

    Uses the same mapping rule as SHP matching: rows where
    ``PARENTOBJECTTYPE=skillgroup`` and ``CHILDOBJECTTYPE=skill``.
    """
    settings = settings or get_settings()
    m = _load_skill_to_groups_map(str(settings.skill_hierarchy_csv_path.resolve()))
    groups: set[str] = set()
    for sid in skill_ids:
        s = str(sid or "").strip()
        if not s:
            continue
        groups.update(m.get(s, frozenset()))
    return sorted(groups)

