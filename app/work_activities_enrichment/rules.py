"""O*NET work-activity importance / level for ranked jobs (the enrichment post-step). No IO."""

from __future__ import annotations

from typing import Any

from app.work_activities_enrichment.types import WorkActivityTaxonomy


def build_work_activity_taxonomy(raw: object) -> WorkActivityTaxonomy:
    """Index the parsed JSON list (one entry per occupation with its ``onet_work_activities``)."""
    if not isinstance(raw, list):
        raise ValueError("combined_occupation_database_with_wa.json must be a JSON list")

    occ_lookup: dict[str, dict[str, dict[str, float]]] = {}
    sums: dict[str, dict[str, float]] = {}

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        occ = entry.get("occupation") or {}
        label = str(occ.get("preferred_label") or "").strip().lower()
        if not label:
            continue
        wa_rows = entry.get("onet_work_activities") or []
        wa_map: dict[str, dict[str, float]] = {}
        for w in wa_rows:
            if not isinstance(w, dict):
                continue
            code = str(w.get("WA_code") or "").strip()
            if not code:
                continue
            raw_imp = w.get("WA_Importance")
            raw_lvl = w.get("WA_Level")
            if raw_imp is None or raw_lvl is None:
                continue
            try:
                imp = float(raw_imp)
                lvl = float(raw_lvl)
            except (TypeError, ValueError):
                continue
            wa_map[code] = {"importance": imp, "level": lvl}
            agg = sums.setdefault(code, {"imp": 0.0, "lvl": 0.0, "n": 0.0})
            agg["imp"] += imp
            agg["lvl"] += lvl
            agg["n"] += 1.0
        if wa_map:
            occ_lookup[label] = wa_map

    wa_avg: dict[str, dict[str, float]] = {}
    for code, s in sums.items():
        if s["n"] <= 0:
            continue
        wa_avg[code] = {
            "importance": round(s["imp"] / s["n"], 2),
            "level": round(s["lvl"] / s["n"], 2),
        }
    return WorkActivityTaxonomy(occ_lookup=occ_lookup, wa_averages=wa_avg)


def build_onet_work_activity_rows(
    doc: dict[str, Any],
    *,
    taxonomy: WorkActivityTaxonomy,
) -> tuple[list[dict[str, Any]], bool]:
    """``onet_work_activities`` rows for one ranked job, and whether its occupation matched.

    Per ``work_activity.items`` code: the matched occupation's importance / level, else the
    cross-occupation average, else "code is not available".
    """
    occ_lookup = taxonomy.occ_lookup
    wa_averages = taxonomy.wa_averages
    out: list[dict[str, Any]] = []
    used_occ_match = False
    wa = doc.get("work_activity")
    items = wa.get("items") if isinstance(wa, dict) else None
    if not isinstance(items, list):
        return out, used_occ_match

    occ_labels: list[str] = []
    for row in doc.get("classified_occupations") or []:
        if not isinstance(row, dict):
            continue
        lab = str(row.get("label") or "").strip().lower()
        if lab:
            occ_labels.append(lab)
    title = str(((doc.get("classifier_metadata") or {}).get("title")) or "").strip().lower()
    if title:
        occ_labels.append(title)

    occ_wa: dict[str, dict[str, float]] | None = None
    for lab in occ_labels:
        if lab in occ_lookup:
            occ_wa = occ_lookup[lab]
            used_occ_match = True
            break

    for item in items:
        if not isinstance(item, dict):
            continue
        code = str(item.get("id") or "").strip()
        if not code:
            continue
        if occ_wa and code in occ_wa:
            imp = float(occ_wa[code]["importance"])
            lvl = float(occ_wa[code]["level"])
            out.append(
                {
                    "WA_code": code,
                    "WA_label": str(item.get("name") or "").strip(),
                    "WA_Importance": round(imp, 2),
                    "WA_Level": round(lvl, 2),
                }
            )
        elif code in wa_averages:
            imp = float(wa_averages[code]["importance"])
            lvl = float(wa_averages[code]["level"])
            out.append(
                {
                    "WA_code": code,
                    "WA_label": str(item.get("name") or "").strip(),
                    "WA_Importance": round(imp, 2),
                    "WA_Level": round(lvl, 2),
                }
            )
        else:
            out.append(
                {
                    "WA_code": code,
                    "WA_label": str(item.get("name") or "").strip(),
                    "WA_Importance": "code is not available",
                    "WA_Level": "code is not available",
                }
            )
    return out, used_occ_match
