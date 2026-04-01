"""Target collection document shape for LLM reranker + job attributes (``PIPELINE_SCHEMA_VERSION``)."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from app.config import Settings, get_settings
from app.job_context import JobDescriptionContext
from app.schemas import JobAttributesResult, OccupationSkillsRelevanceResult, load_onet_work_activities
from utils.classifier_linked_entities import (
    EscoSkillCandidate,
    LinkedEntityCandidate,
    MappedEscoSkills,
    group_linked_entities_by_type,
)
from utils.occupation_to_skills_mapping import occupation_block_skill_counts, tabiya_occupation_id_for_candidate
from utils.uuid_to_tabiya_id import SkillsUuidTabiyaMap, cell_to_esco_uuid

PIPELINE_SCHEMA_VERSION = "32"


def _classified_skill_rows(
    candidates: list[LinkedEntityCandidate],
    skills_uuid_tabiya_map: SkillsUuidTabiyaMap | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in candidates:
        esco = ((c.concept_uuid or "").strip().lower() or cell_to_esco_uuid(c.uri) or None)
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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in candidates:
        esco = ((c.concept_uuid or "").strip().lower() or cell_to_esco_uuid(c.uri) or None)
        tabiya_occ: str | None = None
        if occupations_bridge is not None and occupations_all_ids is not None:
            tabiya_occ = tabiya_occupation_id_for_candidate(
                c, occupations_bridge, occupations_all_ids
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
    esco_l = ((c.esco_uuid or "").strip().lower() or cell_to_esco_uuid(uri) or None)
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


def classifier_metadata_from_context(ctx: JobDescriptionContext) -> dict[str, Any]:
    """Classifier-sourced listing fields and full job text blocks (``title`` lives here, not top-level)."""
    return {
        "title": ctx.title,
        "employer": ctx.employer,
        "category": ctx.category,
        "application_url": ctx.application_url,
        "posted_date": ctx.posted_date,
        "closing_date": ctx.closing_date,
        "source_platform": ctx.source_platform,
        "city": ctx.city,
        "county": ctx.county,
        "salary": ctx.salary,
        "employment_type": ctx.employment_type,
        "job_description": ctx.raw_job_description,
        "requirements": ctx.requirements_text,
    }


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
        decision_map[(d.tabiya_occupation_id.strip().lower(), d.skill_id.strip().lower())] = d.is_relevant

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
    return {
        "skill_ranker_model": settings.gemini_model,
        "occupation_skills_relevance_model": settings.gemini_model,
        "job_attributes_model": settings.gemini_model,
        "schema_version": PIPELINE_SCHEMA_VERSION,
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
