"""
Optional ESCO skill **definitions** from ``skills.csv`` for the skill ranker prompt.

Enable with ``SKILLS_CSV_INCLUDE_DESCRIPTIONS`` and ``SKILLS_CSV_DESCRIPTION_COLUMN``
(see :class:`app.config.Settings`). Maps lowercase ESCO
concept uuid → description text; same row may contribute multiple keys via ``ORIGINURI`` /
``UUIDHISTORY`` (aligned with :mod:`utils.uuid_to_tabiya_id`).
Occupation analog: :mod:`utils.occupation_description`.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from utils.uuid_to_tabiya_id import cell_to_esco_uuid, parse_uuid_history_cell

logger = logging.getLogger(__name__)


def load_skill_description_by_esco(
    path_str: str,
    column_name: str | None,
) -> dict[str, str]:
    """
    Build ``esco_uuid_lower → description`` from ``skills.csv``.

    ``column_name``: header name (case-insensitive), e.g. ``SCOPEDESCRIPTION`` or ``definition``.
    If empty or file missing, returns an empty dict.
    """
    col = (column_name or "").strip()
    if not col:
        return {}
    path = Path(path_str)
    if not path.is_file():
        logger.warning("skills.csv not found for descriptions: %s", path)
        return {}

    out: dict[str, str] = {}
    n_rows = 0
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        lower = {str(h).lower().strip(): h for h in reader.fieldnames}
        desc_col = lower.get(col.lower())
        if not desc_col:
            logger.warning(
                "skills.csv has no column %r for descriptions (headers=%s)",
                col,
                reader.fieldnames,
            )
            return out
        id_col = lower.get("id")
        hist_col = lower.get("uuidhistory")
        origin_col = lower.get("originuri")
        if not id_col:
            logger.warning("skills.csv needs ID column for skill descriptions")
            return out

        for row in reader:
            n_rows += 1
            desc = (row.get(desc_col) or "").strip()
            if not desc:
                continue

            def add_for_uuid(u: str | None) -> None:
                if u:
                    out[u.lower()] = desc

            if origin_col:
                ou = cell_to_esco_uuid(row.get(origin_col) or "")
                add_for_uuid(ou)
            if hist_col:
                for esco_u in parse_uuid_history_cell(row.get(hist_col)):
                    add_for_uuid(esco_u)

    logger.info(
        "Loaded skill descriptions from %s column=%s rows=%s esco_keys=%s",
        path.name,
        col,
        n_rows,
        len(out),
    )
    return out
