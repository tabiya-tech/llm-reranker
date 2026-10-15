#!/usr/bin/env python3
"""Enrich ``onet_work_activities`` on ranked job documents.

Primary behavior:
  - Read taxonomy WA importance/level from
    ``resources/combined_occupation_database_with_wa.json``.
  - For each job's ``work_activity.items``:
      * If a classified occupation label matches taxonomy occupation -> use that
        occupation's WA values.
      * Else fallback to cross-occupation WA-code averages from taxonomy.
      * Else use configurable defaults.

The MongoDB client is injected: the batch passes its run's client, and the CLI entry point opens
one through ``standalone_db()``. See ``app/server_dependencies/README.md``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.operations import UpdateOne

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    db_name: str
    target_collection: str
    occupation_db_path: Path
    batch_size: int
    limit: int
    dry_run: bool
    only_missing: bool
    default_importance: float
    default_level: float


def _as_bool(raw: str | None, *, default: bool = False) -> bool:
    v = (raw or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _load_config() -> Config:
    load_dotenv()
    # Main pipeline should default to current reranker DB; reference DB is fallback only.
    db_name = (
        (os.getenv("ENRICH_WA_DATABASE") or "").strip()
        or (os.getenv("MONGODB_DATABASE") or "").strip()
        or (os.getenv("MONGODB_DATABASE_REFERENCE") or "").strip()
    )
    if not db_name:
        raise ValueError("MONGODB_DATABASE_REFERENCE or MONGODB_DATABASE is required.")

    target_collection = (
        (os.getenv("ENRICH_WA_TARGET_COLLECTION") or "").strip()
        or (os.getenv("MONGODB_TARGET_COLLECTION") or "").strip()
        or (os.getenv("MONGODB_TARGET_COLLECTION_New") or "").strip()
    )
    if not target_collection:
        raise ValueError("MONGODB_TARGET_COLLECTION_New or MONGODB_TARGET_COLLECTION is required.")

    occ_db_raw = (os.getenv("ENRICH_WA_OCCUPATION_DB_PATH") or "").strip()
    if occ_db_raw:
        p = Path(occ_db_raw)
        occupation_db_path = p if p.is_absolute() else (Path(__file__).resolve().parents[1] / p)
    else:
        # Language pack first, then the English pack, then the flat pre-language layout.
        # The occupation→work-activity mapping is keyed on ISCO codes and WA_Element_IDs,
        # so a language that ships no translated copy is served correctly by English.
        from app.config import get_settings, language_resource_dirs

        settings = get_settings()
        name = "combined_occupation_database_with_wa.json"
        candidates = language_resource_dirs(settings.resources_dir, settings.language)
        occupation_db_path = next(
            (d / name for d in candidates if (d / name).is_file()), candidates[0] / name
        )

    return Config(
        db_name=db_name,
        target_collection=target_collection,
        occupation_db_path=occupation_db_path.resolve(),
        batch_size=max(1, int((os.getenv("ENRICH_WA_BULK_BATCH_SIZE") or "500").strip())),
        limit=max(0, int((os.getenv("ENRICH_WA_LIMIT") or "0").strip())),
        dry_run=_as_bool(os.getenv("ENRICH_WA_DRY_RUN"), default=False),
        only_missing=_as_bool(os.getenv("ENRICH_WA_ONLY_MISSING"), default=True),
        default_importance=float((os.getenv("ENRICH_WA_DEFAULT_IMPORTANCE") or "3.5").strip()),
        default_level=float((os.getenv("ENRICH_WA_DEFAULT_LEVEL") or "3.5").strip()),
    )


def _iter_docs(coll: Collection, *, filt: dict[str, Any], limit: int):
    cur = coll.find(filt).sort("_id", -1)
    if limit > 0:
        cur = cur.limit(limit)
    for d in cur:
        yield d


def _target_filter(cfg: Config) -> dict[str, Any]:
    if not cfg.only_missing:
        return {}
    # Treat missing, null, or empty list as “needs enrichment” ($exists:false alone misses null/[]).
    return {
        "$or": [
            {"onet_work_activities": {"$exists": False}},
            {"onet_work_activities": None},
            {"onet_work_activities": []},
        ]
    }


def _build_wa_rows_from_work_activity(
    doc: dict[str, Any],
    *,
    occ_lookup: dict[str, dict[str, dict[str, float]]],
    wa_averages: dict[str, dict[str, float]],
    default_importance: float,
    default_level: float,
) -> tuple[list[dict[str, Any]], bool]:
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


def _load_wa_taxonomy(
    path: Path,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, float]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Occupation WA taxonomy JSON not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
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
    return occ_lookup, wa_avg


def run(client: MongoClient[dict[str, Any]]) -> int:
    """Enrich the target collection. ``client`` is the caller's; it is not closed here.

    The database is resolved from the env (``ENRICH_WA_DATABASE`` may name another database on the
    same cluster), so this takes the client rather than the run's database.
    """
    cfg = _load_config()
    msg = (
        f"work_activities_enrichment | db={cfg.db_name} | target={cfg.target_collection} | "
        f"taxonomy={cfg.occupation_db_path} | only_missing={cfg.only_missing} | "
        f"dry_run={cfg.dry_run} | limit={cfg.limit or 'all'}"
    )
    logger.info(msg)
    print(msg)

    occ_lookup, wa_averages = _load_wa_taxonomy(cfg.occupation_db_path)
    tax_msg = (
        f"work_activities_enrichment | taxonomy loaded | occupations_with_WA={len(occ_lookup)} "
        f"wa_codes_with_averages={len(wa_averages)}"
    )
    logger.info(tax_msg)
    print(tax_msg)

    target = client[cfg.db_name][cfg.target_collection]

    scanned = 0
    with_occ_match = 0
    with_avg_match = 0
    with_default = 0
    no_data = 0
    queued = 0
    applied = 0
    ops: list[UpdateOne] = []

    for doc in _iter_docs(target, filt=_target_filter(cfg), limit=cfg.limit):
        scanned += 1
        wa_items, used_occ_match = _build_wa_rows_from_work_activity(
            doc,
            occ_lookup=occ_lookup,
            wa_averages=wa_averages,
            default_importance=cfg.default_importance,
            default_level=cfg.default_level,
        )
        if not wa_items:
            no_data += 1
            continue

        if used_occ_match:
            with_occ_match += 1
        elif any(
            (str(x.get("WA_code") or "").strip() in wa_averages)
            for x in wa_items
            if isinstance(x, dict)
        ):
            with_avg_match += 1
        else:
            with_default += 1

        payload = {
            "onet_work_activities": wa_items,
        }
        ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": payload}, upsert=False))
        queued += 1

        if len(ops) >= cfg.batch_size:
            if not cfg.dry_run:
                res = target.bulk_write(ops, ordered=False)
                applied += int(res.modified_count)
            ops.clear()

    if ops:
        if not cfg.dry_run:
            res = target.bulk_write(ops, ordered=False)
            applied += int(res.modified_count)
        ops.clear()

    done_msg = (
        "work_activities_enrichment | done | scanned={scanned} with_occ_match={with_occ_match} "
        "with_avg_match={with_avg_match} with_default={with_default} no_data={no_data} "
        "updates_queued={queued} updates_applied={applied}".format(
            scanned=scanned,
            with_occ_match=with_occ_match,
            with_avg_match=with_avg_match,
            with_default=with_default,
            no_data=no_data,
            queued=queued,
            applied=applied,
        )
    )
    logger.info(done_msg)
    print(done_msg)
    return 0


def main() -> int:
    from app.server_dependencies.db_dependencies import standalone_db

    with standalone_db() as db:
        return run(db.client)


if __name__ == "__main__":
    raise SystemExit(main())
