"""Target collection document shape for LLM reranker + job attributes (``PIPELINE_SCHEMA_VERSION``)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.config import Settings, get_settings
from app.job_context import JobDescriptionContext
from app.schemas import JobAttributesResult, load_onet_work_activities
from utils.classifier_linked_entities import (
    EscoSkillCandidate,
    LinkedEntityCandidate,
    MappedEscoSkills,
    group_linked_entities_by_type,
)
from utils.occupation_to_skills_mapping import tabiya_occupation_id_for_candidate
from utils.uuid_to_tabiya_id import SkillsUuidTabiyaMap, cell_to_esco_uuid

PIPELINE_SCHEMA_VERSION = "27"


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


def occupation_to_skills_with_meta(items: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Stored ``occupation_to_skills`` shape: rollup ``meta`` plus ``items`` (full per-occupation blocks).

    ``meta.skill_counts`` sums ``items[*].skill_counts`` after taxonomy / ESCO mapping.
    ``meta.by_occupation`` lists the same counts per classified occupation (compact index).
    """
    keys = ("total", "essential", "optional", "unspecified")
    empty = {k: 0 for k in keys}
    if not items:
        return {
            "meta": {
                "occupation_count": 0,
                "skill_counts": dict(empty),
                "by_occupation": [],
            },
            "items": [],
        }
    agg = dict(empty)
    by_occ: list[dict[str, Any]] = []
    for block in items:
        sc = block.get("skill_counts")
        if not isinstance(sc, dict):
            sc = {}
        row = {k: int(sc.get(k) or 0) for k in keys}
        for k in keys:
            agg[k] += row[k]
        by_occ.append(
            {
                "tabiya_occupation_id": block.get("tabiya_occupation_id"),
                "occupation_label": block.get("occupation_label"),
                "occupation_esco_uuid": block.get("occupation_esco_uuid"),
                "skill_counts": row,
            }
        )
    return {
        "meta": {
            "occupation_count": len(items),
            "skill_counts": agg,
            "by_occupation": by_occ,
        },
        "items": items,
    }


def build_llm_reranker_meta(settings: Settings) -> dict[str, Any]:
    return {
        "skill_ranker_model": settings.vertex_model,
        "job_attributes_model": settings.vertex_model,
        "gcp_location": settings.gcp_location,
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
    ``occupation_to_skills`` is ``{ meta, items }`` (rollup + full occupation blocks).

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
        "llm_job_attributes": dict(ja.attributes),
        "classifier_metadata": classifier_metadata_from_context(ctx),
        "llm_reranker_meta": build_llm_reranker_meta(settings),
    }

    return payload
