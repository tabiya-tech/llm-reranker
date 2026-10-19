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

import logging
import re

from pydantic import Field

from app.domain_base import DomainModel

logger = logging.getLogger(__name__)

_ESCO_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def cell_to_esco_uuid(cell: str) -> str | None:
    """Extract an ESCO concept uuid from a bare uuid or a URI ending in one, else ``None``."""
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
    """Split a ``UUIDHISTORY`` cell (newline-separated ESCO uuids/URIs) into parsed uuids."""
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


class SkillsUuidTabiyaMap(DomainModel):
    """In-memory ESCO uuid <-> Tabiya id maps built from one ``skills.csv`` pass."""

    preferred_label: dict[str, str] = Field(
        default_factory=dict, description="ESCO uuid or Tabiya id (lowercased) -> PREFERREDLABEL."
    )
    primary_esco_by_tabiya: dict[str, str] = Field(
        default_factory=dict, description="Tabiya skill ID -> primary ESCO uuid."
    )
    tabiya_by_esco: dict[str, str] = Field(
        default_factory=dict, description="ESCO uuid -> Tabiya ID (inverse; last row wins)."
    )

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
