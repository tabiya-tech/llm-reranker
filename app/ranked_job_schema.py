"""Target collection document shape for LLM reranker + job attributes (``PIPELINE_SCHEMA_VERSION``)."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from app.config import Settings, get_settings
from app.job_context import JobDescriptionContext
from app.schemas import (
    JobAttributesResult,
    OccupationSkillsRelevanceResult,
    load_onet_work_activities,
)
from utils.classifier_linked_entities import (
    EscoSkillCandidate,
    LinkedEntityCandidate,
    MappedEscoSkills,
    group_linked_entities_by_type,
)
from utils.occupation_to_skills_mapping import (
    occupation_block_skill_counts,
    tabiya_occupation_id_for_candidate,
)
from utils.uuid_to_tabiya_id import SkillsUuidTabiyaMap, cell_to_esco_uuid
from utils.scraper_enrichment_fields import copy_scraper_enrichment_fields
from utils.skill_groups import derive_skill_groups_origin_uuids

PIPELINE_SCHEMA_VERSION = "32"


def _classified_skill_rows(
    candidates: list[LinkedEntityCandidate],
    skills_uuid_tabiya_map: SkillsUuidTabiyaMap | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in candidates:
        esco = (c.concept_uuid or "").strip().lower() or cell_to_esco_uuid(c.uri) or None
        rows.append(
            {
                "label": c.label,
                "uri": c.uri,
                "esco_uuid": esco,
                "similarity_score": c.similarity_score,
                "tabiya_skill_id": (
                    skills_uuid_tabiya_map.tabiya_for_esco(esco)
                    if skills_uuid_tabiya_map is not None and esco
                    else None
                ),
            }
        )
    return rows


def _classified_occupation_rows(
    candidates: list[LinkedEntityCandidate],
    occupations_bridge: dict[str, str] | None,
    occupations_all_ids: frozenset[str] | None,
    *,
    settings: Settings,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in candidates:
        esco = (c.concept_uuid or "").strip().lower() or cell_to_esco_uuid(c.uri) or None
        tabiya_occ: str | None = None
        if occupations_bridge is not None and occupations_all_ids is not None:
            tabiya_occ = tabiya_occupation_id_for_candidate(
                c, occupations_bridge, occupations_all_ids, settings=settings
            )
        rows.append(
            {
                "label": c.label,
                "uri": c.uri,
                "esco_uuid": esco,
                "similarity_score": c.similarity_score,
                "tabiya_occupation_id": tabiya_occ,
            }
        )
    return rows


def _work_activity_payload(
    wa_ids: tuple[str, ...],
    *,
    settings: Settings,
) -> dict[str, Any]:
    names = work_activity_names_for_ids(wa_ids, settings=settings)
    return {
        "items": [{"id": wid, "name": name} for wid, name in zip(list(wa_ids), names)],
    }


def _skill_row_from_candidate(
    c: EscoSkillCandidate,
    *,
    skills_uuid_tabiya_map: SkillsUuidTabiyaMap | None,
    include_score: bool = True,
) -> dict[str, Any]:
    uri = (c.uri or "").strip()
    esco_l = (c.esco_uuid or "").strip().lower() or cell_to_esco_uuid(uri) or None
    row: dict[str, Any] = {
        "label": c.label,
        "uri": uri or None,
        "esco_uuid": esco_l,
        "tabiya_skill_id": (
            skills_uuid_tabiya_map.tabiya_for_esco(esco_l)
            if skills_uuid_tabiya_map is not None and esco_l
            else None
        ),
    }
    if include_score and c.similarity_score is not None:
        row["similarity_score"] = c.similarity_score
    return row


def _llm_classified_skills_payload(
    mapped: MappedEscoSkills | None,
    ranker_candidates: list[EscoSkillCandidate] | None,
    skills_uuid_tabiya_map: SkillsUuidTabiyaMap | None,
    classifier_skill_candidates: list[EscoSkillCandidate] | None,
) -> dict[str, Any]:
    """
    ``discarded`` = classifier skills (same menu as LLM 1) whose ``uri`` was not mapped into essential or
    optional. Taxonomy skills belong in ``occupation_to_skills``, not here.
    """
    empty: dict[str, Any] = {
        "essential": [],
        "optional": [],
        "discarded": [],
        "unmatched_essential": [],
        "unmatched_optional": [],
    }
    if mapped is None:
        return empty

    by_uri = {c.uri: c for c in (ranker_candidates or [])}

    def rows(uris: list[str], uuids: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for uri, uid in zip(uris, uuids):
            cand = by_uri.get(uri)
            label = cand.label if cand else ""
            esco_raw = (uid or "").strip()
            esco = esco_raw.lower() if esco_raw else cell_to_esco_uuid(uri)
            esco_l = esco.lower() if esco else None
            out.append(
                {
                    "label": label,
                    "uri": uri or None,
                    "esco_uuid": esco_l,
                    "tabiya_skill_id": (
                        skills_uuid_tabiya_map.tabiya_for_esco(esco_l)
                        if skills_uuid_tabiya_map is not None and esco_l
                        else None
                    ),
                }
            )
        return out

    ranked_uris = set(mapped.essential_uris) | set(mapped.optional_uris)

    discarded_rows: list[dict[str, Any]] = []
    seen_classifier: set[str] = set()
    for c in classifier_skill_candidates or []:
        uri = (c.uri or "").strip()
        if not uri or uri in ranked_uris or uri in seen_classifier:
            continue
        seen_classifier.add(uri)
        merged = by_uri.get(uri, c)
        discarded_rows.append(
            _skill_row_from_candidate(merged, skills_uuid_tabiya_map=skills_uuid_tabiya_map)
        )

    return {
        "essential": rows(mapped.essential_uris, mapped.essential_skills_uuids),
        "optional": rows(mapped.optional_uris, mapped.optional_skills_uuids),
        "discarded": discarded_rows,
        "unmatched_essential": list(mapped.unmatched_essential),
        "unmatched_optional": list(mapped.unmatched_optional),
    }


def work_activity_names_for_ids(
    ids: tuple[str, ...],
    *,
    settings: Settings | None = None,
) -> list[str]:
    """Resolve ONET work activity ids to display names (simplified label when present)."""
    settings = settings or get_settings()
    onet = load_onet_work_activities(settings=settings)
    by_id = {x.wa_element_id: (x.wa_element_name_simplified or x.wa_element_name) for x in onet}
    return [by_id.get(wid, "") for wid in ids]


def _metadata_scalar_location(val: Any) -> str | None:
    """Normalize city/province enrichment (may be ``str`` or ``list[str]`` on classified docs)."""
    if val is None:
        return None
    if isinstance(val, list):
        parts = [str(x).strip() for x in val if str(x).strip()]
        return ", ".join(parts) if parts else None
    s = str(val).strip()
    return s or None


def classifier_metadata_from_context(ctx: JobDescriptionContext) -> dict[str, Any]:
    """Classifier-sourced listing fields, enrichment, and full job text (``title`` lives here, not top-level)."""
    meta: dict[str, Any] = {
        "title": ctx.title,
        "employer": ctx.employer,
        "category": ctx.category,
        "application_url": ctx.application_url,
        "posted_date": ctx.posted_date,
        "closing_date": ctx.closing_date,
        "source_platform": ctx.source_platform,
        "city": ctx.city,
        "salary": ctx.salary,
        "employment_type": ctx.employment_type,
        "job_description": ctx.raw_job_description,
        "requirements": ctx.requirements_text,
    }
    enrich = copy_scraper_enrichment_fields(ctx.classified)
    if enrich.get("city") is not None:
        meta["city"] = _metadata_scalar_location(enrich["city"]) or meta.get("city")
    # ``province`` replaces the old ``county`` field (same semantic: region / admin area).
    province_val = _metadata_scalar_location(enrich.get("province"))
    if not province_val and ctx.county:
        province_val = _metadata_scalar_location(ctx.county)
    if province_val:
        meta["province"] = province_val
    for key, val in enrich.items():
        if key in ("city", "province"):
            continue
        meta[key] = val
    return meta


def occupation_skills_relevance_menu_size(items: list[dict[str, Any]]) -> int:
    """How many taxonomy skills (essential + optional) LLM call 3 must judge."""
    n = 0
    for block in items:
        tid = block.get("tabiya_occupation_id")
        if tid is None or str(tid).strip() == "":
            continue
        for sk in block.get("skills") or []:
            if not isinstance(sk, dict):
                continue
            rt = (sk.get("relation_type") or "").strip().lower()
            if rt not in ("essential", "optional"):
                continue
            sid = str(sk.get("skill_id") or "").strip()
            if sid:
                n += 1
    return n


def classifier_skill_esco_uuids_lower(candidates: list[EscoSkillCandidate]) -> frozenset[str]:
    """Lowercased ESCO uuid strings from classifier skill linked entities (for LLM3 menu filtering)."""
    return frozenset(
        (c.esco_uuid or "").strip().lower() for c in candidates if (c.esco_uuid or "").strip()
    )


def occupation_skill_signal_tier(skill: dict[str, Any]) -> str:
    """
    Normalized CSV ``SIGNALLINGVALUELABEL`` tier for one skill row: ``high``, ``medium``,
    ``low``, or ``unlabelled`` (empty / other).
    """
    lab = (skill.get("signalling_label") or "").strip().lower()
    if lab in ("high", "medium", "low"):
        return lab
    return "unlabelled"


def _llm3_signalling_label_rank(skill: dict[str, Any]) -> int:
    """
    Tier for CSV ``SIGNALLINGVALUELABEL`` (lower = stronger).

    Empty or unknown labels map to a rank above ``low`` so they are dropped when
    ``RERANKER_LLM3_MENU_SIGNAL_FILTER`` is ``high``, ``medium``, or ``low``.
    """
    t = occupation_skill_signal_tier(skill)
    return {"high": 0, "medium": 1, "low": 2}.get(t, 99)


def filter_occupation_skills_nested_by_signal_include(
    items: list[dict[str, Any]],
    *,
    include_tiers: frozenset[str] | None,
) -> list[dict[str, Any]]:
    """
    Drop essential/optional skills whose signalling tier is not in ``include_tiers``.

    Non–essential/optional rows in a block's ``skills`` list are kept unchanged.
    Blocks that end up with no skills are removed.

    ``include_tiers``: ``None`` → return ``items`` unchanged (no copy).
    Otherwise use names ``high``, ``medium``, ``low``, ``unlabelled`` (see
    ``RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE``).
    """
    if not include_tiers:
        return items
    out: list[dict[str, Any]] = []
    for block in items:
        skills = block.get("skills") or []
        if not isinstance(skills, list):
            continue
        kept: list[dict[str, Any]] = []
        for sk in skills:
            if not isinstance(sk, dict):
                continue
            rt = (sk.get("relation_type") or "").strip().lower()
            if rt not in ("essential", "optional"):
                kept.append(deepcopy(sk))
                continue
            if occupation_skill_signal_tier(sk) in include_tiers:
                kept.append(deepcopy(sk))
        if not kept:
            continue
        nb = deepcopy(block)
        nb["skills"] = kept
        nb["skill_counts"] = occupation_block_skill_counts(kept)
        out.append(nb)
    return out


def _llm3_menu_skill_sort_tuple(
    skill: dict[str, Any], bi: int, si: int
) -> tuple[int, float, int, int, int]:
    """
    Sort key for LLM3 menu rows: stronger CSV signalling first, then essential over optional,
    then document order. Ascending sort = higher priority first.
    """
    lab_rank = _llm3_signalling_label_rank(skill)
    raw = (skill.get("signalling_value") or "").strip()
    try:
        v = float(raw)
    except ValueError:
        v = float("-inf")
    rt = (skill.get("relation_type") or "").strip().lower()
    rt_rank = 0 if rt == "essential" else 1
    return (lab_rank, -v, rt_rank, bi, si)


def subset_occupation_nested_for_llm3_prompt(
    full_items: list[dict[str, Any]],
    *,
    max_menu_skills: int,
    classifier_esco_lower: frozenset[str] | None,
    signal_menu_filter: str = "off",
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Build a smaller occupation→skills nesting for the LLM3 prompt only.

    Returns ``(subset_items, full_menu_count, subset_menu_count)``. Callers should pass
    ``subset_items`` to the relevance model, then ``apply_occupation_skills_listing_relevance``
    on the **original** ``full_items`` with the parsed result: skills omitted from the prompt
    keep the default ``llm_is_relevant`` of ``True``.

    When ``signal_menu_filter`` is ``high``, ``medium``, or ``low``, only relations whose
    CSV ``SIGNALLINGVALUELABEL`` is at least that strong are kept (empty labels are dropped).
    That yields a small, high-signal menu without an arbitrary count cap; combine with
    ``max_menu_skills`` if you still want a hard ceiling.

    When ``max_menu_skills`` > 0, skills are chosen in **CSV signalling strength** order
    (``signalling_label`` high/medium/low, then numeric ``signalling_value``, then essential
    before optional, then listing order) so the cap favours taxonomy rows that are explicitly
    weighted in the relations file—not the first rows in file order.

    ``max_menu_skills`` <= 0: no numeric cap after the signal filter (classifier filter may
    still apply).
    ``classifier_esco_lower``: when not ``None``, only skills whose ``skill_id`` is in this set.
    """
    full_n = occupation_skills_relevance_menu_size(full_items)
    if full_n == 0:
        return [], 0, 0

    budget = full_n if max_menu_skills <= 0 else min(max_menu_skills, full_n)

    entries: list[dict[str, Any]] = []
    for bi, block in enumerate(full_items):
        tid = block.get("tabiya_occupation_id")
        if tid is None or str(tid).strip() == "":
            continue
        skills = block.get("skills") or []
        if not isinstance(skills, list):
            continue
        for si, sk in enumerate(skills):
            if not isinstance(sk, dict):
                continue
            rt = (sk.get("relation_type") or "").strip().lower()
            if rt not in ("essential", "optional"):
                continue
            sid = str(sk.get("skill_id") or "").strip()
            if not sid:
                continue
            if classifier_esco_lower is not None and sid.lower() not in classifier_esco_lower:
                continue
            entries.append(
                {
                    "bi": bi,
                    "si": si,
                    "skill": deepcopy(sk),
                    "_sort": _llm3_menu_skill_sort_tuple(sk, bi, si),
                }
            )

    filt = (signal_menu_filter or "off").strip().lower()
    if filt != "off":
        max_rank = {"high": 0, "medium": 1, "low": 2}.get(filt)
        if max_rank is not None:
            entries = [e for e in entries if _llm3_signalling_label_rank(e["skill"]) <= max_rank]

    entries.sort(key=lambda e: e["_sort"])
    entries = entries[:budget]
    entries.sort(key=lambda e: (e["bi"], e["si"]))

    by_bi: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for e in entries:
        by_bi[e["bi"]].append((e["si"], e["skill"]))

    out_blocks: list[dict[str, Any]] = []
    for bi in sorted(by_bi.keys()):
        block = full_items[bi]
        chunk = [sk for _, sk in sorted(by_bi[bi], key=lambda x: x[0])]
        nb = deepcopy(block)
        nb["skills"] = chunk
        nb["skill_counts"] = occupation_block_skill_counts(chunk)
        out_blocks.append(nb)

    subset_n = occupation_skills_relevance_menu_size(out_blocks)
    return out_blocks, full_n, subset_n


def build_llm3_occupation_menu(
    occupation_to_skills_nested: list[dict[str, Any]],
    classifier_skill_candidates: list[EscoSkillCandidate],
    settings: Settings,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Apply ``Settings`` LLM3 menu controls (signal tier filter, max size, classifier intersect).

    Returns ``(menu_for_llm3, full_menu_count, prompt_menu_count)``.
    """
    clf: frozenset[str] | None = None
    if settings.reranker_llm3_classifier_intersect:
        s = classifier_skill_esco_uuids_lower(classifier_skill_candidates)
        clf = s if s else None
    return subset_occupation_nested_for_llm3_prompt(
        occupation_to_skills_nested,
        max_menu_skills=settings.reranker_llm3_max_menu_skills,
        classifier_esco_lower=clf,
        signal_menu_filter=settings.reranker_llm3_menu_signal_filter,
    )


def apply_occupation_skills_listing_relevance(
    items: list[dict[str, Any]],
    result: OccupationSkillsRelevanceResult,
) -> list[dict[str, Any]]:
    """
    Copy ``items``, set ``llm_is_relevant`` on each skill.

    Each block's ``skill_counts`` keeps taxonomy totals (``total``, ``essential``, ``optional``,
    ``unspecified``) and adds ``relevant_total``: count of skills with ``llm_is_relevant`` not false.
    """
    decision_map: dict[tuple[str, str], bool] = {}
    for d in result.decisions:
        decision_map[(d.tabiya_occupation_id.strip().lower(), d.skill_id.strip().lower())] = (
            d.is_relevant
        )

    out = deepcopy(items)
    for block in out:
        tid = str(block.get("tabiya_occupation_id") or "").strip().lower()
        skills = block.get("skills")
        if not isinstance(skills, list):
            continue
        for sk in skills:
            if not isinstance(sk, dict):
                continue
            sid = str(sk.get("skill_id") or "").strip().lower()
            rt = (sk.get("relation_type") or "").strip().lower()
            if rt in ("essential", "optional") and tid and sid:
                sk["llm_is_relevant"] = decision_map.get((tid, sid), True)
            else:
                sk["llm_is_relevant"] = True
        rel_only = [
            x for x in skills if isinstance(x, dict) and x.get("llm_is_relevant") is not False
        ]
        tax = occupation_block_skill_counts(skills)
        block["skill_counts"] = {
            **tax,
            "relevant_total": len(rel_only),
        }
    return out


def occupation_to_skills_with_meta(items: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Stored ``occupation_to_skills``: ``meta`` + ``items`` only.

    ``meta.skill_counts`` rolls up ``total``, ``essential``, ``optional``, ``unspecified``, and
    ``relevant_total`` (sum of per-item ``relevant_total`` when present, else 0).
    """
    keys_tax = ("total", "essential", "optional", "unspecified")
    empty = {k: 0 for k in keys_tax}
    empty_with_rel = {**empty, "relevant_total": 0}
    if not items:
        return {
            "meta": {
                "occupation_count": 0,
                "skill_counts": dict(empty_with_rel),
            },
            "items": [],
        }
    agg: dict[str, int] = {k: 0 for k in keys_tax}
    agg_rel = 0
    for block in items:
        sc = block.get("skill_counts")
        if not isinstance(sc, dict):
            sc = {}
        for k in keys_tax:
            agg[k] += int(sc.get(k) or 0)
        agg_rel += int(sc.get("relevant_total") or 0)
    meta_out = {
        "occupation_count": len(items),
        "skill_counts": {**agg, "relevant_total": agg_rel},
    }
    return {
        "meta": meta_out,
        "items": items,
    }


def build_llm_reranker_meta(settings: Settings) -> dict[str, Any]:
    """``classifier_metadata`` block: schema version plus the model behind each pass.

    All three passes share one configured client, so the three model fields are equal;
    they stay separate because the shape is a published contract read by the matching
    service. ``llm_provider`` is additive — a Gemini deployment still writes the same
    model string it always did.
    """
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "llm_provider": settings.llm.provider,
        "skill_ranker_model": settings.llm.model,
        "occupation_skills_relevance_model": settings.llm.model,
        "job_attributes_model": settings.llm.model,
        "classified_linked_entity_types": list(settings.classified_linked_entity_types),
    }


def build_ranked_job_document(
    classified_doc: dict[str, Any],
    ctx: JobDescriptionContext,
    *,
    settings: Settings,
    ja: JobAttributesResult | None,
    skill_ranker: tuple[list[str], list[str], MappedEscoSkills, str] | None = None,
    occupation_to_skills: list[dict[str, Any]] | None = None,
    skills_uuid_tabiya_map: SkillsUuidTabiyaMap | None = None,
    occupations_bridge: dict[str, str] | None = None,
    occupations_all_ids: frozenset[str] | None = None,
    ranker_candidates: list[EscoSkillCandidate] | None = None,
    classifier_skill_candidates: list[EscoSkillCandidate] | None = None,
) -> dict[str, Any] | None:
    """
    Full ``$set`` body for the ranked-job collection document.

    Key order: ``job_id``, ``job_fingerprint``, activity flags, classifier taxonomies, occupation skills,
    work activities, LLM skill ranking, LLM job attributes, classifier metadata (incl. title), reranker meta.
    ``occupation_to_skills`` is ``{ meta, items }``; ``meta.skill_counts`` includes taxonomy totals and
    ``relevant_total`` (sum of listing-relevant skills) after LLM call 3.

    Omits Mongo ``_id`` (server-owned). Parallel ``*_uuids`` arrays and top-level ``title`` are not written;
    use ``classified_*`` / ``llm_classified_skills`` and ``classifier_metadata.title``.

    Returns ``None`` if there is no job-attributes result (nothing to persist).
    """
    if ja is None:
        return None

    now = datetime.now(timezone.utc)
    by_type = group_linked_entities_by_type(classified_doc, settings)
    classified_skills = _classified_skill_rows(
        by_type.get("skill", []),
        skills_uuid_tabiya_map,
    )
    classified_occupations = _classified_occupation_rows(
        by_type.get("occupation", []),
        occupations_bridge,
        occupations_all_ids,
        settings=settings,
    )
    occ_to_skills_nested = list(occupation_to_skills) if occupation_to_skills else []
    occ_to_skills_payload = occupation_to_skills_with_meta(occ_to_skills_nested)

    mapped: MappedEscoSkills | None = None
    if skill_ranker is not None:
        _, _, mapped, _ = skill_ranker

    llm_skills = _llm_classified_skills_payload(
        mapped,
        ranker_candidates,
        skills_uuid_tabiya_map,
        classifier_skill_candidates,
    )
    grouped_skill_ids: list[str] = []
    for row in llm_skills.get("essential", []) + llm_skills.get("optional", []):
        if not isinstance(row, dict):
            continue
        sid = str(row.get("tabiya_skill_id") or "").strip()
        if sid:
            grouped_skill_ids.append(sid)
    skill_groups_origin_uuids = derive_skill_groups_origin_uuids(
        grouped_skill_ids,
        settings=settings,
    )
    work_activity = _work_activity_payload(ja.work_activity_ids, settings=settings)

    # Order matches target schema (see module docstring). ``_id`` is not $set.
    payload: dict[str, Any] = {
        "job_id": ctx.job_id,
        "job_fingerprint": ctx.job_fingerprint,
        "is_active": True,
        "processed_at": now,
        "source_classified_id": classified_doc.get("_id"),
        "classified_skills": classified_skills,
        "classified_occupations": classified_occupations,
        "occupation_to_skills": occ_to_skills_payload,
        "work_activity": work_activity,
        "skill_groups_origin_uuids": skill_groups_origin_uuids,
        "llm_classified_skills": llm_skills,
        "llm_job_attributes": {
            "summary": ja.summary,
            "estimates": [
                {
                    "attribute_name": e.attribute_name,
                    "selected_level_id": e.selected_level_id,
                    "confidence": e.confidence,
                    "reasoning": e.reasoning,
                }
                for e in ja.attribute_estimates
            ],
            "attributes": dict(ja.attributes),
        },
        "classifier_metadata": classifier_metadata_from_context(ctx),
        "llm_reranker_meta": build_llm_reranker_meta(settings),
    }

    return payload
