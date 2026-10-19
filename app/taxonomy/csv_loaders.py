"""CSV parsers for the taxonomy files of a language pack.

Each function reads one file and returns plain lookup maps;
:class:`app.taxonomy.repository.TaxonomyRepository` calls them once per
language and assembles the result into a ``TaxonomyContext``.

- ``occupation_to_skill_relations.csv``: occupation → skills, relation type, signalling.
- ``occupations.csv``: ESCO / Tabiya keys → occupation ``ID``; labels and descriptions.
- ``skills.csv``: ESCO uuid ↔ Tabiya skill ``ID``; optional skill descriptions.
- ``skill_hierarchy.csv``: skill → parent skill groups.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from pydantic import ValidationError

from utils.occupation_csv_models import (
    OccupationCsvRow,
    OccupationSkillRelationCsvRow,
    SkillsCsvRow,
)
from utils.occupation_to_skills_mapping import cell_to_taxonomy_id
from utils.uuid_to_tabiya_id import SkillsUuidTabiyaMap, cell_to_esco_uuid, parse_uuid_history_cell

logger = logging.getLogger(__name__)


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
        for raw_row in reader:
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
                    ot = (raw_row.get(occ_type_col) or "").strip().lower()
                    if ot not in accepted_occupation_types:
                        continue
            row = OccupationSkillRelationCsvRow.model_validate(
                {
                    "occupation_id": raw_row.get(occ_col) or "",
                    "skill_id": raw_row.get(sk_col) or "",
                    "relation_type": raw_row.get(relation_col) if relation_col else None,
                    "occupation_type": raw_row.get(occ_type_col) if occ_type_col else None,
                    "skill_label": raw_row.get(label_col) if label_col else None,
                    "signalling_value": raw_row.get(sig_val_col) if sig_val_col else None,
                    "signalling_label": raw_row.get(sig_lab_col) if sig_lab_col else None,
                }
            )
            ou = cell_to_taxonomy_id(row.occupation_id)
            su = cell_to_taxonomy_id(row.skill_id)
            if not ou or not su:
                continue
            occupation_to_skills.setdefault(ou, set()).add(su)
            lab = (row.skill_label or "").strip()
            if lab:
                skill_labels[su] = lab
            rel = (row.relation_type or "").strip().lower() or "unspecified"
            inner = occ_skill_relation.setdefault(ou, {})
            inner[su] = _merge_relation_types(inner.get(su), rel)
            raw_v = (row.signalling_value or "").strip()
            raw_l = (row.signalling_label or "").strip()
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
        for raw_row in reader:
            try:
                row = OccupationCsvRow.model_validate(
                    {
                        "id": raw_row.get(id_col),
                        "uuid_history": raw_row.get(hist_col),
                        "origin_uri": raw_row.get(origin_col) if origin_col else None,
                    }
                )
            except ValidationError:
                continue
            raw_id = row.id.strip()
            if not raw_id:
                continue
            tid = raw_id.lower()
            all_ids.add(tid)
            bridge[tid] = tid
            for esco_u in parse_uuid_history_cell(row.uuid_history):
                bridge[esco_u.lower()] = tid
            if row.origin_uri:
                origin = row.origin_uri.strip()
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

        for raw_row in reader:
            n_rows += 1
            canon = {
                "id": raw_row.get(id_col),
                "preferred_label": raw_row.get(pref_col),
                "uuid_history": raw_row.get(hist_col) if hist_col else None,
                "origin_uri": raw_row.get(origin_col) if origin_col else None,
            }
            try:
                row = SkillsCsvRow.model_validate(canon)
            except ValidationError:
                continue
            raw_id = row.id.strip()
            label = row.preferred_label.strip()
            if not raw_id or not label:
                continue
            tid = raw_id.lower()
            idx.preferred_label[tid] = label

            primary: str | None = None
            if row.origin_uri:
                primary = cell_to_esco_uuid(row.origin_uri)
            if not primary and row.uuid_history:
                hist = parse_uuid_history_cell(row.uuid_history)
                primary = hist[0] if hist else None
            if primary:
                idx.primary_esco_by_tabiya[tid] = primary
                idx.tabiya_by_esco[primary] = tid
            else:
                unmapped_primary += 1

            if row.uuid_history:
                for esco_u in parse_uuid_history_cell(row.uuid_history):
                    idx.preferred_label[esco_u.lower()] = label
                    if esco_u.lower() not in idx.tabiya_by_esco:
                        idx.tabiya_by_esco[esco_u.lower()] = tid
            if row.origin_uri:
                origin = row.origin_uri.strip()
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

        def add_for_uuid(u: str | None, desc: str) -> None:
            if u:
                out[u.lower()] = desc

        for row in reader:
            n_rows += 1
            desc = (row.get(desc_col) or "").strip()
            if not desc:
                continue

            if origin_col:
                ou = cell_to_esco_uuid(row.get(origin_col) or "")
                add_for_uuid(ou, desc)
            if hist_col:
                for esco_u in parse_uuid_history_cell(row.get(hist_col)):
                    add_for_uuid(esco_u, desc)

    logger.info(
        "Loaded skill descriptions from %s column=%s rows=%s esco_keys=%s",
        path.name,
        col,
        n_rows,
        len(out),
    )
    return out


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


def load_skill_to_groups_map(path_str: str) -> dict[str, frozenset[str]]:
    """``skill_hierarchy.csv``: skill ``ID`` → parent skill-group ``ID`` values.

    Uses the same mapping rule as SHP matching: rows where ``PARENTOBJECTTYPE=skillgroup``
    and ``CHILDOBJECTTYPE=skill``.
    """
    p = Path(path_str)
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
