"""
LLM2 text from ``occupations.csv``: **labels** and optional **definitions** per ESCO key.

Parallel to :func:`utils.skill_descriptions.load_skill_description_by_esco`, but occupations also need
a **label** column (e.g. ``PREFERREDLABEL``) for every row.

Keys use ``ID``, ``UUIDHISTORY``, ``ORIGINURI`` (see :mod:`utils.occupation_to_skills_mapping`).

Env: ``OCCUPATIONS_CSV_LABEL_COLUMN``, ``OCCUPATIONS_CSV_INCLUDE_DESCRIPTIONS``,
``OCCUPATIONS_CSV_DESCRIPTION_COLUMN`` (see :class:`app.config.Settings`).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from utils.uuid_to_tabiya_id import cell_to_esco_uuid, parse_uuid_history_cell

logger = logging.getLogger(__name__)


def load_occupation_labels_and_descriptions_by_esco(
    path_str: str,
    *,
    label_column: str,
    include_descriptions: bool = False,
    description_column: str | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    One pass over ``occupations.csv``.

    Returns ``(label_by_esco, description_by_esco)``. The description map is empty when
    ``include_descriptions`` is False or ``description_column`` is unset / missing.
    """
    label_col = (label_column or "").strip()
    if not label_col:
        return {}, {}
    desc_col = (description_column or "").strip() if include_descriptions else ""
    path = Path(path_str)
    if not path.is_file():
        logger.warning("occupations.csv not found for LLM2: %s", path)
        return {}, {}

    label_out: dict[str, str] = {}
    desc_out: dict[str, str] = {}
    n_rows = 0
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}, {}
        lower = {str(h).lower().strip(): h for h in reader.fieldnames}
        lbl_hdr = lower.get(label_col.lower())
        if not lbl_hdr:
            logger.warning(
                "occupations.csv has no label column %r (headers=%s)",
                label_col,
                reader.fieldnames,
            )
            return {}, {}
        desc_hdr = lower.get(desc_col.lower()) if desc_col else None
        if include_descriptions and desc_col and not desc_hdr:
            logger.warning(
                "occupations.csv has no description column %r (headers=%s)",
                desc_col,
                reader.fieldnames,
            )
        id_col = lower.get("id")
        hist_col = lower.get("uuidhistory")
        origin_col = lower.get("originuri")
        if not id_col or not hist_col:
            logger.warning(
                "occupations.csv must have ID and UUIDHISTORY (headers=%s)",
                reader.fieldnames,
            )
            return {}, {}

        def add_for_key(k: str | None, label: str, raw_desc: str) -> None:
            if not k:
                return
            kl = k.lower()
            label_out[kl] = label
            if raw_desc and desc_hdr:
                desc_out[kl] = raw_desc

        for row in reader:
            n_rows += 1
            label = (row.get(lbl_hdr) or "").strip()
            if not label:
                continue
            raw_desc = (row.get(desc_hdr) or "").strip() if desc_hdr else ""

            raw_id = (row.get(id_col) or "").strip()
            if raw_id:
                add_for_key(raw_id, label, raw_desc)
            if origin_col:
                ou = cell_to_esco_uuid(row.get(origin_col) or "")
                add_for_key(ou, label, raw_desc)
            if hist_col:
                for esco_u in parse_uuid_history_cell(row.get(hist_col)):
                    add_for_key(esco_u, label, raw_desc)

    logger.info(
        "Loaded occupation labels/descriptions from %s label_col=%s desc=%s rows=%s label_keys=%s desc_keys=%s",
        path.name,
        label_col,
        bool(desc_hdr),
        n_rows,
        len(label_out),
        len(desc_out),
    )
    return label_out, desc_out
