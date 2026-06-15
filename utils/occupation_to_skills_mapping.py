"""
Classifier occupations → ``occupations.csv`` (Tabiya ``ID``) → ``occupation_to_skill_relations.csv`` (skills).

Flow:
1. Same occupation rows as ``classified_occupations``: deduped ESCO hits from
   :func:`utils.classifier_linked_entities.group_linked_entities_by_type`.
2. Match each hit to ``occupations.csv`` via ``UUIDHISTORY`` / ``ORIGINURI`` / row ``ID``
   (see ``OCCUPATION_RELATION_LOOKUP``: full bridge scan vs ESCO-uuid-first).
3. Tabiya occupation ids → skill ids in ``occupation_to_skill_relations.csv`` (optional
   ``OCCUPATION_SKILL_RELATION_OCCUPATION_TYPES`` filter on ``OCCUPATIONTYPE``).
4. Persist ``occupation_to_skills`` (``meta`` + ``items``) with ``RELATIONTYPE`` from the relations CSV,
   plus optional ``SIGNALLINGVALUE`` / ``SIGNALLINGVALUELABEL`` on each skill when the CSV has them.
   Classifier ESCO skills and occupation-taxonomy skills stay separate; LLM1 sees classifier candidates only.

Tabiya ``SKILLID`` → ESCO via :mod:`utils.uuid_to_tabiya_id` (``skills.csv``).
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Mapping
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from utils.classifier_linked_entities import (
    EscoSkillCandidate,
    LinkedEntityCandidate,
    group_linked_entities_by_type,
    linked_entity_preferred_taxonomy_id,
)
from utils.uuid_to_tabiya_id import SkillsUuidTabiyaMap, cell_to_esco_uuid, parse_uuid_history_cell

logger = logging.getLogger(__name__)

_HEX_ID_24 = re.compile(r"^[0-9a-f]{24}$", re.IGNORECASE)
_HEX24_STANDALONE = re.compile(r"(?<![0-9a-f])[0-9a-f]{24}(?![0-9a-f])", re.IGNORECASE)


def cell_to_taxonomy_id(cell: str) -> str | None:
    """Tabiya hex id or ESCO uuid for matching CSV cells and URIs."""
    s = (cell or "").strip()
    if not s:
        return None
    u = cell_to_esco_uuid(s)
    if u:
        return u
    if _HEX_ID_24.match(s):
        return s.lower()
    part = s.rstrip("/").rsplit("/", 1)[-1]
    if _HEX_ID_24.match(part):
        return part.lower()
    v = cell_to_esco_uuid(part)
    if v:
        return v
    m = _HEX24_STANDALONE.search(s)
    if m:
        return m.group(0).lower()
    return None


def esco_skill_uri_for_uuid(uuid: str) -> str:
    u = uuid.strip().lower()
    return f"http://data.europa.eu/esco/skill/{u}"


def _infer_skill_label_column(fieldnames: list[str]) -> str | None:
    """
    Column holding the **skill** display name — not occupation text or ``SIGNALLINGVALUELABEL``
    (that field describes relation strength, e.g. \"high\", not the skill concept title).
    """
    lower = {h.lower().replace(" ", "_"): h for h in fieldnames if h and str(h).strip()}
    for key in (
        "skill_label",
        "skill_preferred_label",
        "skillpreferredlabel",
        "relatedskill_label",
        "skill_name",
        "preferred_label",
        "preferredlabel",
        "skill_title",
        "label",
    ):
        if key in lower:
            return lower[key]
    return None


def _infer_relation_type_column(fieldnames: list[str]) -> str | None:
    lower = {h.lower().replace(" ", "_"): h for h in fieldnames if h and str(h).strip()}
    for key in ("relationtype", "relation_type", "rel_type", "relation_kind"):
        if key in lower:
            return lower[key]
    for h in fieldnames:
        if h and str(h).strip():
            hl = h.lower().replace(" ", "_")
            if "relation" in hl and "type" in hl:
                return h
    return None


def _infer_occupation_type_column(fieldnames: list[str]) -> str | None:
    """Column for relation row kind, e.g. ``escooccupation`` / ``localoccupation`` (``OCCUPATIONTYPE``)."""
    lower = {h.lower().replace(" ", "_"): h for h in fieldnames if h and str(h).strip()}
    for key in ("occupationtype", "occupation_type", "occ_type", "occupation_kind"):
        if key in lower:
            return lower[key]
    for h in fieldnames:
        if h and str(h).strip():
            hl = h.lower().replace(" ", "_")
            if "occupation" in hl and "type" in hl and "relation" not in hl:
                return h
    return None


def _infer_signalling_value_column(fieldnames: list[str]) -> str | None:
    lower = {h.lower().replace(" ", "_"): h for h in fieldnames if h and str(h).strip()}
    for key in ("signallingvalue", "signalingvalue", "signal_value", "signalling_value"):
        if key in lower:
            return lower[key]
    return None


def _infer_signalling_label_column(fieldnames: list[str]) -> str | None:
    lower = {h.lower().replace(" ", "_"): h for h in fieldnames if h and str(h).strip()}
    for key in (
        "signallingvaluelabel",
        "signalingvaluelabel",
        "signal_label",
        "signalling_value_label",
    ):
        if key in lower:
            return lower[key]
    return None


def _signalling_strength_key(
    signalling_label: str,
    signalling_value: str,
) -> tuple[int, float]:
    """
    Lower tuple sorts earlier = stronger relation for menu ordering.

    Uses ``SIGNALLINGVALUELABEL`` (high / medium / low) when present, then numeric
    ``SIGNALLINGVALUE`` when parseable (higher is stronger).
    """
    lab = (signalling_label or "").strip().lower()
    lab_rank = {"high": 0, "medium": 1, "low": 2}.get(lab, 3)
    raw = (signalling_value or "").strip()
    try:
        v = float(raw)
    except ValueError:
        v = float("-inf")
    return (lab_rank, -v)


def _merge_signalling_cells(
    prev: dict[str, str] | None,
    value_cell: str,
    label_cell: str,
) -> dict[str, str]:
    """Pick the stronger of two CSV rows for the same occupation–skill pair."""
    cur = {
        "signalling_value": (value_cell or "").strip(),
        "signalling_label": (label_cell or "").strip().lower(),
    }
    if prev is None:
        return cur
    if not cur["signalling_value"] and not cur["signalling_label"]:
        return prev
    if not prev.get("signalling_value") and not prev.get("signalling_label"):
        return cur
    ka = _signalling_strength_key(
        prev.get("signalling_label", ""), prev.get("signalling_value", "")
    )
    kc = _signalling_strength_key(cur["signalling_label"], cur["signalling_value"])
    return prev if ka <= kc else cur


_RELTYPE_PRIORITY = {"essential": 0, "optional": 1, "unspecified": 99}


def _merge_relation_types(prev: str | None, new: str) -> str:
    """Prefer *essential* over *optional* over other strings when duplicates exist for one skill."""
    new_n = (new or "").strip().lower() or "unspecified"
    if prev is None:
        return new_n
    pa = _RELTYPE_PRIORITY.get(prev, 50)
    na = _RELTYPE_PRIORITY.get(new_n, 50)
    if na < pa:
        return new_n
    return prev


def _infer_occ_skill_columns(fieldnames: list[str]) -> tuple[str | None, str | None]:
    if not fieldnames:
        return None, None
    norm_map = {h: h.strip() for h in fieldnames if h and str(h).strip()}
    lower = {h.lower().replace(" ", "_"): h for h in norm_map.values()}

    def pick(*candidates: str) -> str | None:
        for c in candidates:
            if c in lower:
                return lower[c]
        return None

    occ = pick(
        "occupationid",
        "occupation_uuid",
        "occupation_id",
        "occupation_uri",
        "occupationconcepturi",
        "occupation_concept_uri",
    )
    sk = pick(
        "skillid",
        "skill_uuid",
        "skill_id",
        "skill_uri",
        "relatedskill",
        "skillconcepturi",
        "skill_concept_uri",
    )
    if occ is None or sk is None:
        for h in norm_map.values():
            hl = h.lower()
            if occ is None and "occupation" in hl and "type" not in hl:
                occ = h
            elif sk is None and "skill" in hl and "occupation" not in hl and "type" not in hl:
                sk = h
    if len(norm_map) == 2 and (occ is None or sk is None):
        vals = list(norm_map.values())
        occ, sk = vals[0], vals[1]
    return occ, sk


def load_occupation_skill_relations(
    path_str: str,
    *,
    accepted_occupation_types: frozenset[str] | None = None,
) -> tuple[
    dict[str, frozenset[str]],
    dict[str, str],
    dict[str, dict[str, str]],
    dict[str, dict[str, dict[str, str]]],
]:
    """
    - ``OCCUPATIONID`` → set of ``SKILLID`` (union for reranker).
    - Optional per-skill display labels.
    - ``occupation_id`` → ``skill_id`` → ``RELATIONTYPE`` (e.g. essential / optional), normalized lowercase.
    - ``occupation_id`` → ``skill_id`` → ``{signalling_value, signalling_label}`` from CSV signalling
      columns when present (used to prioritize the LLM3 menu when capping).

    When ``accepted_occupation_types`` is set, only CSV rows whose ``OCCUPATIONTYPE`` cell (case-insensitive)
    is in that set are loaded (e.g. ``escooccupation`` vs ``localoccupation``). If the column is missing,
    a warning is logged once and **no** type filtering is applied.
    """
    empty = {}, {}, {}, {}
    path = Path(path_str)
    if not path.is_file():
        logger.warning("occupation_to_skill_relations.csv not found: %s", path)
        return empty

    occupation_to_skills: dict[str, set[str]] = {}
    skill_labels: dict[str, str] = {}
    occ_skill_relation: dict[str, dict[str, str]] = {}
    occ_skill_signalling: dict[str, dict[str, dict[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return empty
        fn = list(reader.fieldnames)
        occ_col, sk_col = _infer_occ_skill_columns(fn)
        label_col = _infer_skill_label_column(fn)
        relation_col = _infer_relation_type_column(fn)
        sig_val_col = _infer_signalling_value_column(fn)
        sig_lab_col = _infer_signalling_label_column(fn)
        occ_type_col = _infer_occupation_type_column(fn)
        missing_occ_type_warned = False
        if not relation_col:
            logger.warning(
                "No RELATIONTYPE-like column in %s (headers=%s); using relation_type=%r for all rows",
                path.name,
                reader.fieldnames,
                "unspecified",
            )
        if not occ_col or not sk_col:
            logger.warning(
                "Could not infer occupation/skill columns in %s (headers=%s)",
                path,
                reader.fieldnames,
            )
            return empty
        for row in reader:
            if accepted_occupation_types:
                if not occ_type_col:
                    if not missing_occ_type_warned:
                        logger.warning(
                            "occupation_to_skill_relations: OCCUPATION_SKILL_RELATION_OCCUPATION_TYPES is set "
                            "but no OCCUPATIONTYPE-like column in %s (headers=%s); loading all rows",
                            path.name,
                            reader.fieldnames,
                        )
                        missing_occ_type_warned = True
                else:
                    ot = (row.get(occ_type_col) or "").strip().lower()
                    if ot not in accepted_occupation_types:
                        continue
            ou = cell_to_taxonomy_id(row.get(occ_col) or "")
            su = cell_to_taxonomy_id(row.get(sk_col) or "")
            if not ou or not su:
                continue
            occupation_to_skills.setdefault(ou, set()).add(su)
            if label_col:
                lab = (row.get(label_col) or "").strip()
                if lab:
                    skill_labels[su] = lab
            if relation_col:
                raw_rel = (row.get(relation_col) or "").strip().lower()
                rel = raw_rel if raw_rel else "unspecified"
            else:
                rel = "unspecified"
            inner = occ_skill_relation.setdefault(ou, {})
            inner[su] = _merge_relation_types(inner.get(su), rel)
            if sig_val_col or sig_lab_col:
                raw_v = (row.get(sig_val_col) or "").strip() if sig_val_col else ""
                raw_l = (row.get(sig_lab_col) or "").strip() if sig_lab_col else ""
                if raw_v or raw_l:
                    sig_map = occ_skill_signalling.setdefault(ou, {})
                    sig_map[su] = _merge_signalling_cells(sig_map.get(su), raw_v, raw_l)

    occ_frozen = {k: frozenset(v) for k, v in occupation_to_skills.items()}
    if accepted_occupation_types:
        n_pairs = sum(len(v) for v in occ_frozen.values())
        logger.info(
            "occupation_to_skill_relations %s | OCCUPATIONTYPE filter=%s | occupations=%s pairs=%s",
            path.name,
            ",".join(sorted(accepted_occupation_types)),
            len(occ_frozen),
            n_pairs,
        )

    return (
        occ_frozen,
        skill_labels,
        occ_skill_relation,
        occ_skill_signalling,
    )


def load_occupations_lookup(path_str: str) -> tuple[dict[str, str], frozenset[str]]:
    """
    From ``occupations.csv``: map lookup keys (ESCO uuid, Tabiya ``ID``, …) → row ``ID``.

    Parsed once per resolved path per process (cached); treat returned ``bridge`` as read-only.

    Each row contributes:
    - ``ID`` → itself;
    - every UUID in ``UUIDHISTORY`` → ``ID``;
    - ESCO uuid from ``ORIGINURI`` → ``ID`` (when parseable).
    """
    path = Path(path_str)
    try:
        key = str(path.resolve())
    except OSError:
        key = str(path)
    return _load_occupations_lookup_cached(key)


@lru_cache(maxsize=8)
def _load_occupations_lookup_cached(resolved_path: str) -> tuple[dict[str, str], frozenset[str]]:
    path = Path(resolved_path)
    if not path.is_file():
        logger.warning("occupations.csv not found: %s", path)
        return {}, frozenset()

    bridge: dict[str, str] = {}
    all_ids: set[str] = set()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}, frozenset()
        lower = {str(h).lower().strip(): h for h in reader.fieldnames}
        id_col = lower.get("id")
        hist_col = lower.get("uuidhistory")
        origin_col = lower.get("originuri")
        if not id_col or not hist_col:
            logger.warning(
                "occupations.csv must have ID and UUIDHISTORY (headers=%s)",
                reader.fieldnames,
            )
            return {}, frozenset()
        for row in reader:
            raw_id = (row.get(id_col) or "").strip()
            if not raw_id:
                continue
            tid = raw_id.lower()
            all_ids.add(tid)
            bridge[tid] = tid
            for esco_u in parse_uuid_history_cell(row.get(hist_col)):
                bridge[esco_u.lower()] = tid
            if origin_col:
                origin = (row.get(origin_col) or "").strip()
                if origin:
                    ou = cell_to_esco_uuid(origin)
                    if ou:
                        bridge[ou.lower()] = tid

    logger.info(
        "Loaded occupations.csv %s: %s distinct occupation id(s), %s lookup key(s)",
        path.name,
        len(all_ids),
        len(bridge),
    )
    return bridge, frozenset(all_ids)


def _classified_occupation_dicts_for_taxonomy(
    classified_doc: dict[str, Any],
    settings: Settings,
) -> list[dict[str, Any]]:
    """
    Build lookup payloads from the **same** occupation list as ``classified_occupations`` / ``*_uuids``.

    Uses :func:`group_linked_entities_by_type` (dedupe by ``(entity_type, uri)``, label+uri required),
    not every nested ``linked_entities`` row under ``classification.entities`` (which can repeat URIs).
    """
    occ = group_linked_entities_by_type(classified_doc, settings).get("occupation", [])
    out: list[dict[str, Any]] = []
    for c in occ:
        le = {
            "uri": c.uri,
            "label": c.label,
            "taxonomy": "esco",
        }
        if c.similarity_score is not None:
            le["similarity_score"] = c.similarity_score
        if c.concept_uuid:
            le["originUUID"] = c.concept_uuid
        out.append(le)
    return out


def _esco_uuid_from_occupation_linked_entity(le: dict[str, Any]) -> str | None:
    uri = (le.get("uri") or "").strip()
    if uri:
        u = cell_to_esco_uuid(uri)
        if u:
            return u
    for ok in ("originUUID", "origin_uuid", "originUuid"):
        v = le.get(ok)
        if v is not None and str(v).strip():
            u = cell_to_esco_uuid(str(v))
            if u:
                return u
    return None


def _resolve_first_tabiya_id_for_linked_entity(
    le: dict[str, Any],
    bridge: Mapping[str, str],
    all_ids: frozenset[str],
) -> str | None:
    """First occupations.csv match for this classifier row (for per-hit logging)."""
    for k in _lookup_keys_for_occupation_row(le):
        tid = bridge.get(k)
        if tid is None and k in all_ids:
            tid = k
        if tid:
            return tid
    return None


def _esco_bridge_keys_for_occupation_row(le: dict[str, Any]) -> list[str]:
    """
    Lowercased keys to try against ``occupations.csv`` derived **only** from ESCO uris / uuids.

    Used when ``occupation_relation_lookup`` is ``esco_uuid_bridge`` so relations resolution
    prefers the classifier's ESCO occupation identity over other taxonomy ids on the same row.
    """
    keys: list[str] = []
    seen: set[str] = set()

    def add(raw: str | None) -> None:
        if not raw or not str(raw).strip():
            return
        u = cell_to_esco_uuid(str(raw).strip())
        if not u:
            return
        xl = u.lower()
        if xl not in seen:
            seen.add(xl)
            keys.append(xl)

    add((le.get("uri") or "").strip())
    for ok in ("originUUID", "origin_uuid", "originUuid"):
        v = le.get(ok)
        if v is not None and str(v).strip():
            add(str(v).strip())
    return keys


def _bridge_tabiya_for_key(
    k: str, bridge: Mapping[str, str], all_ids: frozenset[str]
) -> str | None:
    tid = bridge.get(k)
    if tid is None and k in all_ids:
        tid = k
    return tid


def resolve_occupation_tabiya_id_for_skill_relations(
    le: dict[str, Any],
    bridge: Mapping[str, str],
    all_ids: frozenset[str],
    *,
    settings: Settings | None = None,
) -> str | None:
    """
    Map one classifier occupation row → Tabiya ``ID`` used as ``OCCUPATIONID`` in relations CSV.

    - ``tabiya_bridge`` (default): first hit from the full key order (preferred id, URI, …).
    - ``esco_uuid_bridge``: try ESCO uuid keys first; if ``occupation_relation_esco_fallback`` is
      True and no hit, fall back to ``tabiya_bridge`` behaviour.

    Relations rows remain keyed by Tabiya ids; this only changes **which** bridge key wins.
    """
    s = settings or get_settings()
    mode = (s.occupation_relation_lookup or "tabiya_bridge").strip().lower()
    if mode == "esco_uuid_bridge":
        for k in _esco_bridge_keys_for_occupation_row(le):
            tid = _bridge_tabiya_for_key(k, bridge, all_ids)
            if tid:
                return tid
        if s.occupation_relation_esco_fallback:
            return _resolve_first_tabiya_id_for_linked_entity(le, bridge, all_ids)
        return None
    return _resolve_first_tabiya_id_for_linked_entity(le, bridge, all_ids)


def tabiya_occupation_id_for_candidate(
    c: LinkedEntityCandidate,
    bridge: Mapping[str, str],
    all_ids: frozenset[str],
    *,
    settings: Settings | None = None,
) -> str | None:
    """Resolve Tabiya occupation ``ID`` for one classifier occupation row (``occupations.csv``)."""
    le: dict[str, Any] = {
        "uri": c.uri,
        "label": c.label,
        "taxonomy": "esco",
    }
    if c.similarity_score is not None:
        le["similarity_score"] = c.similarity_score
    if c.concept_uuid:
        le["originUUID"] = c.concept_uuid
    return resolve_occupation_tabiya_id_for_skill_relations(le, bridge, all_ids, settings=settings)


@dataclass
class OccupationTaxonomyStats:
    """
    One row per ``classified_occupations`` entry: lookup → Tabiya id → |skills| in relations.

    ``sum_relation_skills_per_classifier_hit`` double-counts when two hits share the same Tabiya id.
    ``merged_unique_skills_from_occupations`` is the unique ESCO skill ids from occupation relations only
    (deduped across Tabiya occupation ids); classifier skills are not included in this count.
    """

    classified_skill_count: int
    occupation_classifier_hits: int
    per_classifier_hit: list[dict[str, Any]] = field(default_factory=list)
    deduped_tabiya_occupation_ids: list[str] = field(default_factory=list)
    sum_relation_skills_per_classifier_hit: int = 0
    merged_unique_skills_from_occupations: int = 0

    def format_log_lines(self) -> list[str]:
        lines = [
            "classifier_skills=%s occupation_classifier_hits=%s deduped_tabiya_ids=%s "
            "sum_skills_per_hit=%s taxonomy_unique_occupation_skills=%s"
            % (
                self.classified_skill_count,
                self.occupation_classifier_hits,
                len(self.deduped_tabiya_occupation_ids),
                self.sum_relation_skills_per_classifier_hit,
                self.merged_unique_skills_from_occupations,
            )
        ]
        for row in self.per_classifier_hit:
            lines.append(
                "  occ[%s] label=%r esco_uuid=%s tabiya_id=%s relation_skills=%s resolved=%s"
                % (
                    row["index"],
                    row.get("label") or "",
                    row.get("esco_uuid") or "—",
                    row.get("tabiya_id") or "—",
                    row.get("relation_skill_count"),
                    row.get("resolved"),
                )
            )
        return lines


def _build_occupation_taxonomy_stats(
    classified_doc: dict[str, Any],
    settings: Settings,
    bridge: dict[str, str],
    all_ids: frozenset[str],
    occupation_to_skills: Mapping[str, frozenset[str]],
    classified_skill_count: int,
    deduped_tabiya_ids: list[str],
    merged_unique_skill_count: int,
) -> OccupationTaxonomyStats:
    per_hit: list[dict[str, Any]] = []
    sum_counts = 0
    hits = _classified_occupation_dicts_for_taxonomy(classified_doc, settings)
    for i, le in enumerate(hits, start=1):
        label = (le.get("label") or "").strip()
        esco_u = _esco_uuid_from_occupation_linked_entity(le)
        tid = resolve_occupation_tabiya_id_for_skill_relations(
            le, bridge, all_ids, settings=settings
        )
        if tid:
            tid_l = tid.strip().lower()
            n = len(occupation_to_skills.get(tid_l, frozenset()))
        else:
            n = 0
        sum_counts += n
        per_hit.append(
            {
                "index": i,
                "label": label,
                "esco_uuid": esco_u or None,
                "tabiya_id": tid,
                "relation_skill_count": n,
                "resolved": tid is not None,
            }
        )
    return OccupationTaxonomyStats(
        classified_skill_count=classified_skill_count,
        occupation_classifier_hits=len(hits),
        per_classifier_hit=per_hit,
        deduped_tabiya_occupation_ids=list(deduped_tabiya_ids),
        sum_relation_skills_per_classifier_hit=sum_counts,
        merged_unique_skills_from_occupations=merged_unique_skill_count,
    )


def _lookup_keys_for_occupation_row(le: dict[str, Any]) -> list[str]:
    """Normalized keys to try against ``load_occupations_lookup`` bridge."""
    keys: list[str] = []
    seen: set[str] = set()

    def add(raw: str | None) -> None:
        if not raw or not str(raw).strip():
            return
        xl = str(raw).strip().lower()
        if xl not in seen:
            seen.add(xl)
            keys.append(xl)

    pref = linked_entity_preferred_taxonomy_id(le)
    if pref:
        add(pref)
    uri = (le.get("uri") or "").strip()
    if uri:
        t = cell_to_taxonomy_id(uri)
        if t:
            add(t)
        eu = cell_to_esco_uuid(uri)
        if eu:
            add(eu)
    for ok in ("originUUID", "origin_uuid", "originUuid"):
        v = le.get(ok)
        if v is not None and str(v).strip():
            t = cell_to_taxonomy_id(str(v))
            if t:
                add(t)
    return keys


def resolve_tabiya_occupation_ids_for_relations(
    classified_doc: dict[str, Any],
    occupations_csv_path: str,
    *,
    settings: Settings | None = None,
    bridge: dict[str, str] | None = None,
    all_ids: frozenset[str] | None = None,
) -> list[str]:
    """
    Tabiya occupation ``ID`` values (as in ``occupation_to_skill_relations.OCCUPATIONID``)
    for each ``classified_occupations`` row (same source as ``group_linked_entities_by_type``).

    With ``occupation_relation_lookup=esco_uuid_bridge``, each row yields at most one id (ESCO-first
    bridge + optional fallback). With ``tabiya_bridge``, multiple keys on the same classifier row
    can still append multiple distinct ids when each maps through the bridge.
    """
    s = settings or get_settings()
    if bridge is None or all_ids is None:
        bridge, all_ids = load_occupations_lookup(occupations_csv_path)
    if not bridge:
        return []

    out: list[str] = []
    seen: set[str] = set()
    mode = (s.occupation_relation_lookup or "tabiya_bridge").strip().lower()
    for le in _classified_occupation_dicts_for_taxonomy(classified_doc, s):
        if mode == "esco_uuid_bridge":
            tid = resolve_occupation_tabiya_id_for_skill_relations(le, bridge, all_ids, settings=s)
            if tid:
                tl = tid.strip().lower()
                if tl not in seen:
                    seen.add(tl)
                    out.append(tid)
            continue
        for k in _lookup_keys_for_occupation_row(le):
            tid = bridge.get(k)
            if tid is None and k in all_ids:
                tid = k
            if tid and tid not in seen:
                seen.add(tid)
                out.append(tid)
    return out


def occupation_skill_taxonomy_for_settings(
    settings: Settings | None = None,
) -> tuple[
    dict[str, frozenset[str]],
    dict[str, str],
    dict[str, dict[str, str]],
    dict[str, dict[str, dict[str, str]]],
]:
    s = settings or get_settings()
    return load_occupation_skill_relations(
        str(s.occupation_to_skill_relations_path.resolve()),
        accepted_occupation_types=s.occupation_skill_relation_occupation_types,
    )


def occupation_block_skill_counts(skills: list[dict[str, Any]]) -> dict[str, int]:
    """
    Count ``total`` / ``essential`` / ``optional`` / ``unspecified`` for a block's ``skills`` rows
    (``relation_type`` from the relations CSV). Used after listing-relevance filtering.
    """
    return _skill_relation_counts_after_lookup(skills)


def _skill_relation_counts_after_lookup(skills: list[dict[str, Any]]) -> dict[str, int]:
    """Counts on skills persisted under one occupation (ESCO-mapped rows only, ``relation_type`` from CSV)."""
    essential = optional = unspecified = 0
    for s in skills:
        rt = (s.get("relation_type") or "").strip().lower() or "unspecified"
        if rt == "essential":
            essential += 1
        elif rt == "optional":
            optional += 1
        else:
            unspecified += 1
    return {
        "total": len(skills),
        "essential": essential,
        "optional": optional,
        "unspecified": unspecified,
    }


def build_occupation_to_skills_nested(
    classified_doc: dict[str, Any],
    settings: Settings,
    occupation_to_skills: Mapping[str, frozenset[str]],
    occupation_skill_relation: Mapping[str, Mapping[str, str]],
    skill_labels: Mapping[str, str],
    bridge: dict[str, str],
    all_ids: frozenset[str],
    skills_uuid_tabiya_map: SkillsUuidTabiyaMap | None = None,
    occupation_skill_signalling: Mapping[str, Mapping[str, Mapping[str, str]]] | None = None,
) -> list[dict[str, Any]]:
    """
    One object per ``classified_occupations`` row: Tabiya id, skills from relations with ``relation_type``
    (from ``RELATIONTYPE`` in the CSV).

    Each skill's ``skill_id`` and ``uri`` use the **canonical ESCO uuid** from ``skills.csv`` so they
    align with classifier / ranker UUIDs. ``tabiya_skill_id`` keeps the relations-file key for auditing.
    ``skill_counts`` summarizes ``skills`` by ``relation_type`` (taxonomy CSV) after ESCO mapping
    (rows without a ``skills.csv`` mapping are omitted from both ``skills`` and counts).

    Optional ``occupation_skill_signalling`` adds ``signalling_value`` / ``signalling_label`` from the
    relations CSV when present (used to prioritize skills for LLM3 when the menu is capped).
    """
    sig_root = occupation_skill_signalling or {}
    out: list[dict[str, Any]] = []
    for c in group_linked_entities_by_type(classified_doc, settings).get("occupation", []):
        le = {
            "uri": c.uri,
            "label": c.label,
            "taxonomy": "esco",
        }
        if c.similarity_score is not None:
            le["similarity_score"] = c.similarity_score
        if c.concept_uuid:
            le["originUUID"] = c.concept_uuid
        tid = resolve_occupation_tabiya_id_for_skill_relations(
            le, bridge, all_ids, settings=settings
        )
        tid_l = tid.strip().lower() if tid else ""
        skills_raw: list[dict[str, Any]] = []
        if tid_l:
            rel_by_skill = occupation_skill_relation.get(tid_l, {})
            for sid in sorted(occupation_to_skills.get(tid_l, frozenset())):
                rel_t = rel_by_skill.get(sid, "unspecified")
                sid_l = sid.strip().lower()
                if skills_uuid_tabiya_map:
                    esco = skills_uuid_tabiya_map.skill_key_as_esco(sid)
                    if not esco:
                        logger.debug(
                            "skills.csv: no ESCO for Tabiya SKILLID %s; omitted from occupation_to_skills",
                            sid_l,
                        )
                        continue
                else:
                    esco = sid_l
                lab = None
                if skills_uuid_tabiya_map:
                    lab = skills_uuid_tabiya_map.label_for_skill_key(esco)
                if not lab:
                    lab = skill_labels.get(sid_l) or skill_labels.get(esco) or skill_labels.get(sid)
                if lab:
                    lab = str(lab).strip()
                if not lab:
                    lab = f"Skill ({esco[:8]}…)" if len(esco) >= 8 else esco
                row: dict[str, Any] = {
                    "skill_id": esco,
                    "tabiya_skill_id": sid_l,
                    "label": lab,
                    "uri": esco_skill_uri_for_uuid(esco),
                    "relation_type": rel_t,
                }
                sig = (sig_root.get(tid_l, {}) or {}).get(sid_l) or (
                    sig_root.get(tid_l, {}) or {}
                ).get(sid.strip().lower())
                if sig:
                    sv = (sig.get("signalling_value") or "").strip()
                    sl = (sig.get("signalling_label") or "").strip()
                    if sv:
                        row["signalling_value"] = sv
                    if sl:
                        row["signalling_label"] = sl.lower()
                skills_raw.append(row)

        def _sk_key(d: dict[str, Any]) -> tuple[int, str]:
            order = {"essential": 0, "optional": 1, "unspecified": 3}
            return (order.get(d.get("relation_type") or "", 2), (d.get("label") or "").lower())

        skills_sorted = sorted(skills_raw, key=_sk_key)
        out.append(
            {
                "occupation_label": c.label,
                "occupation_esco_uuid": c.concept_uuid,
                "occupation_uri": c.uri,
                "tabiya_occupation_id": tid,
                "skill_counts": _skill_relation_counts_after_lookup(skills_sorted),
                "skills": skills_sorted,
            }
        )
    return out


def _label_for_taxonomy_skill(
    skill_id: str,
    csv_labels: Mapping[str, str],
    base_by_uuid: dict[str, EscoSkillCandidate],
    skills_uuid_tabiya_map: SkillsUuidTabiyaMap | None = None,
) -> str:
    """``skill_id`` is canonical **ESCO** uuid when a ``SkillsUuidTabiyaMap`` is provided upstream."""
    u = skill_id.strip().lower()
    if skills_uuid_tabiya_map:
        lab = skills_uuid_tabiya_map.label_for_skill_key(u)
        if lab:
            return lab
    if u in csv_labels and csv_labels[u].strip():
        return csv_labels[u].strip()
    if u in base_by_uuid:
        return base_by_uuid[u].label
    return f"ESCO skill ({u[:8]}…)"


def build_occupation_taxonomy_skill_rows(
    tabiya_occupation_ids: list[str],
    occupation_to_skills: Mapping[str, frozenset[str]],
    csv_skill_labels: Mapping[str, str],
    base_candidates: list[EscoSkillCandidate],
    skills_uuid_tabiya_map: SkillsUuidTabiyaMap | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    base_by_uuid: dict[str, EscoSkillCandidate] = {}
    for c in base_candidates:
        u = (c.esco_uuid or "").strip().lower()
        if u:
            base_by_uuid[u] = c

    esco_ids: list[str] = []
    for oid in tabiya_occupation_ids:
        oid_l = oid.strip().lower()
        for su in occupation_to_skills.get(oid_l, frozenset()):
            if skills_uuid_tabiya_map:
                esco = skills_uuid_tabiya_map.skill_key_as_esco(su)
                if esco:
                    esco_ids.append(esco)
                else:
                    logger.debug(
                        "skills.csv: no ESCO for Tabiya SKILLID %s; skipped for ranker merge",
                        str(su).strip().lower(),
                    )
            else:
                esco_ids.append(str(su).strip().lower())
    skill_ids = sorted(frozenset(esco_ids))

    rows: list[dict[str, Any]] = []
    for su in skill_ids:
        rows.append(
            {
                "label": _label_for_taxonomy_skill(
                    su, csv_skill_labels, base_by_uuid, skills_uuid_tabiya_map
                ),
                "uri": esco_skill_uri_for_uuid(su),
                "esco_uuid": su,
                "similarity_score": None,
            }
        )
    return rows, skill_ids


def resolve_occupation_taxonomy_for_classified_job(
    classified_doc: dict[str, Any],
    classifier_skill_candidates: list[EscoSkillCandidate],
    occupation_to_skills: Mapping[str, frozenset[str]],
    csv_skill_labels: Mapping[str, str] | None = None,
    occupation_skill_relation: Mapping[str, Mapping[str, str]] | None = None,
    skills_uuid_tabiya_map: SkillsUuidTabiyaMap | None = None,
    occupation_skill_signalling: Mapping[str, Mapping[str, Mapping[str, str]]] | None = None,
    *,
    settings: Settings | None = None,
) -> tuple[list[str], OccupationTaxonomyStats, list[dict[str, Any]]]:
    """
    Resolve classifier occupations via ``occupations.csv`` and skills from the relations CSV.

    Returns occupation taxonomy **only** (no union with classifier skills):

    - ``taxonomy_skill_ids``: unique ESCO uuids linked via occupation relations (for logging / analytics).
    - ``OccupationTaxonomyStats`` for terminal logs.
    - ``occupation_to_skills_nested``: Mongo shape (``relation_type`` from CSV per skill).

    The skill ranker (LLM1) continues to use **classifier** candidates only; pass those separately.
    """
    labels = csv_skill_labels or {}
    occ_skill_rel = occupation_skill_relation or {}
    occ_sig = occupation_skill_signalling or {}
    s = settings or get_settings()
    occ_path = str(s.occupations_csv_path.resolve())
    bridge, all_ids = load_occupations_lookup(occ_path)
    tabiya_occ_ids = resolve_tabiya_occupation_ids_for_relations(
        classified_doc, occ_path, settings=s, bridge=bridge, all_ids=all_ids
    )

    _taxonomy_rows, taxonomy_skill_ids = build_occupation_taxonomy_skill_rows(
        tabiya_occ_ids,
        occupation_to_skills,
        labels,
        classifier_skill_candidates,
        skills_uuid_tabiya_map,
    )

    occ_stats = _build_occupation_taxonomy_stats(
        classified_doc,
        s,
        bridge,
        all_ids,
        occupation_to_skills,
        classified_skill_count=len(classifier_skill_candidates),
        deduped_tabiya_ids=tabiya_occ_ids,
        merged_unique_skill_count=len(taxonomy_skill_ids),
    )

    occupation_to_skills_nested = build_occupation_to_skills_nested(
        classified_doc,
        s,
        occupation_to_skills,
        occ_skill_rel,
        labels,
        bridge,
        all_ids,
        skills_uuid_tabiya_map,
        occupation_skill_signalling=occ_sig,
    )

    if tabiya_occ_ids and logger.isEnabledFor(logging.DEBUG):
        per_id = [
            (oid, len(occupation_to_skills.get(oid.strip().lower(), frozenset())))
            for oid in tabiya_occ_ids
        ]
        logger.debug(
            "Occupation→skills: tabiya_occupation_ids=%s per_id_unique_skill_counts=%s "
            "taxonomy_unique_total=%s (relations map keys=%s)",
            tabiya_occ_ids,
            per_id,
            len(taxonomy_skill_ids),
            len(occupation_to_skills),
        )

    if tabiya_occ_ids and occupation_to_skills and not taxonomy_skill_ids:
        logger.warning(
            "occupations→skills resolved no taxonomy skills: relations present=%s; "
            "skills.csv tabiya→esco mappings=%s. Check OCCUPATIONID/SKILLID and that each Tabiya SKILLID "
            "has ORIGINURI or UUIDHISTORY in skills.csv.",
            tabiya_occ_ids[0] in occupation_to_skills if tabiya_occ_ids else False,
            len(skills_uuid_tabiya_map.primary_esco_by_tabiya) if skills_uuid_tabiya_map else 0,
        )

    return taxonomy_skill_ids, occ_stats, occupation_to_skills_nested
