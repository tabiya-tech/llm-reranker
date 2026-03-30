"""
**ESCO skill UUID ↔ Tabiya skill ID** mapping from a single ``skills.csv`` pass, plus preferred labels.

Classifier / LLM outputs use ESCO concept UUIDs. Occupation→skill relations use **Tabiya skill IDs**
(24-char hex). This module unifies both sides: ``skill_id`` / ``esco_uuid`` in stored documents and in
``EscoSkillCandidate`` should be the **same ESCO UUID** whenever ``skills.csv`` maps the Tabiya row.

Maps (all lookup keys lowercased):

- ``preferred_label``: Tabiya ID or any ESCO uuid from the row → ``PREFERREDLABEL``
- ``primary_esco_by_tabiya``: Tabiya skill ``ID`` → primary ESCO uuid (from ``ORIGINURI``, else first
  parsable uuid in ``UUIDHISTORY``)
- ``tabiya_by_esco``: ESCO uuid → Tabiya ``ID`` (inverse; last row wins on collision)
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_ESCO_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def cell_to_esco_uuid(cell: str) -> str | None:
    s = (cell or "").strip()
    if not s:
        return None
    if _ESCO_UUID.match(s):
        return s.lower()
    part = s.rstrip("/").rsplit("/", 1)[-1]
    if _ESCO_UUID.match(part):
        return part.lower()
    return None


def parse_uuid_history_cell(raw: str | None) -> list[str]:
    if raw is None or not str(raw).strip():
        return []
    text = str(raw).replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        u = cell_to_esco_uuid(s)
        if u:
            out.append(u)
        else:
            u2 = cell_to_esco_uuid(s.rstrip(","))
            if u2:
                out.append(u2)
    return out


@dataclass
class SkillsUuidTabiyaMap:
    """In-memory ESCO uuid ↔ Tabiya id maps built from one ``skills.csv`` pass."""

    preferred_label: dict[str, str] = field(default_factory=dict)
    primary_esco_by_tabiya: dict[str, str] = field(default_factory=dict)
    tabiya_by_esco: dict[str, str] = field(default_factory=dict)

    def skill_key_as_esco(self, raw: str) -> str | None:
        """
        Normalize a relations **SKILLID** (Tabiya hex) or an ESCO uuid string to **canonical ESCO uuid**.

        Returns ``None`` if ``raw`` is Tabiya and no row maps it (caller may skip or log).
        """
        r = (raw or "").strip().lower()
        if not r:
            return None
        if _ESCO_UUID.match(r):
            return r
        esco = self.primary_esco_by_tabiya.get(r)
        return esco if esco else None

    def label_for_skill_key(self, esco_or_tabiya: str) -> str | None:
        """Preferred label for either ESCO uuid or Tabiya id (after lowercasing)."""
        k = (esco_or_tabiya or "").strip().lower()
        if not k:
            return None
        lab = self.preferred_label.get(k)
        return lab.strip() if lab and str(lab).strip() else None

    def tabiya_for_esco(self, esco_uuid: str | None) -> str | None:
        """Tabiya ``ID`` from ``skills.csv`` for an ESCO concept uuid, if mapped."""
        u = (esco_uuid or "").strip().lower()
        if not u:
            return None
        return self.tabiya_by_esco.get(u)


def load_skills_uuid_tabiya_map(path_str: str) -> SkillsUuidTabiyaMap:
    path = Path(path_str)
    idx = SkillsUuidTabiyaMap()
    if not path.is_file():
        logger.warning("skills.csv not found: %s", path)
        return idx

    n_rows = 0
    unmapped_primary = 0
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return idx
        lower = {str(h).lower().strip(): h for h in reader.fieldnames}
        id_col = lower.get("id")
        pref_col = lower.get("preferredlabel")
        hist_col = lower.get("uuidhistory")
        origin_col = lower.get("originuri")
        if not id_col or not pref_col:
            logger.warning(
                "skills.csv must have ID and PREFERREDLABEL (headers=%s)",
                reader.fieldnames,
            )
            return idx

        for row in reader:
            n_rows += 1
            raw_id = (row.get(id_col) or "").strip()
            label = (row.get(pref_col) or "").strip()
            if not raw_id or not label:
                continue
            tid = raw_id.lower()
            idx.preferred_label[tid] = label

            primary: str | None = None
            if origin_col:
                primary = cell_to_esco_uuid(row.get(origin_col) or "")
            if not primary and hist_col:
                hist = parse_uuid_history_cell(row.get(hist_col))
                primary = hist[0] if hist else None
            if primary:
                idx.primary_esco_by_tabiya[tid] = primary
                idx.tabiya_by_esco[primary] = tid
            else:
                unmapped_primary += 1

            if hist_col:
                for esco_u in parse_uuid_history_cell(row.get(hist_col)):
                    idx.preferred_label[esco_u.lower()] = label
                    if esco_u.lower() not in idx.tabiya_by_esco:
                        idx.tabiya_by_esco[esco_u.lower()] = tid
            if origin_col:
                origin = (row.get(origin_col) or "").strip()
                if origin:
                    ou = cell_to_esco_uuid(origin)
                    if ou:
                        idx.preferred_label[ou.lower()] = label

    logger.info(
        "Loaded skills.csv %s: rows=%s label_keys=%s tabiya→esco=%s esco→tabiya=%s tabiya_rows_missing_primary_esco=%s",
        path.name,
        n_rows,
        len(idx.preferred_label),
        len(idx.primary_esco_by_tabiya),
        len(idx.tabiya_by_esco),
        unmapped_primary,
    )
    return idx
