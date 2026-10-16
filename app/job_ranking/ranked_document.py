"""Target collection document shape for LLM reranker + job attributes (``PIPELINE_SCHEMA_VERSION``).

Pure assembly: :class:`RankedJobDocumentAssembler` receives every lookup it needs. No IO.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from app.config import Settings
from app.job_context import JobDescriptionContext
from app.job_ranking.job_attributes.types import JobAttributesResult, OnetWorkActivity
from app.job_ranking.skill_ranker.types import SkillRankerResult
from app.taxonomy.types import TaxonomyContext
from utils.classifier_linked_entities import (
    EscoSkillCandidate,
    LinkedEntityCandidate,
    MappedEscoSkills,
    group_linked_entities_by_type,
)
from utils.occupation_to_skills_mapping import tabiya_occupation_id_for_candidate
from utils.scraper_enrichment_fields import copy_scraper_enrichment_fields
from utils.uuid_to_tabiya_id import SkillsUuidTabiyaMap, cell_to_esco_uuid

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
    onet: tuple[OnetWorkActivity, ...],
) -> dict[str, Any]:
    names = work_activity_names_for_ids(wa_ids, onet)
    return {
        "items": [{"id": wid, "name": name} for wid, name in zip(list(wa_ids), names, strict=True)],
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
        for uri, uid in zip(uris, uuids, strict=True):
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
    onet: tuple[OnetWorkActivity, ...],
) -> list[str]:
    """Resolve ONET work activity ids to display names (simplified label when present)."""
    by_id = {x.wa_element_id: (x.wa_element_name_simplified or x.wa_element_name) for x in onet}
    return [by_id.get(wid, "") for wid in ids]


def derive_skill_groups_origin_uuids(
    skill_ids: Iterable[str],
    skill_groups_by_skill: Mapping[str, frozenset[str]],
) -> list[str]:
    """Parent skill-group ids of ``skill_ids`` (``skill_hierarchy.csv``), sorted."""
    groups: set[str] = set()
    for sid in skill_ids:
        s = str(sid or "").strip()
        if not s:
            continue
        groups.update(skill_groups_by_skill.get(s, frozenset()))
    return sorted(groups)


def _metadata_scalar_location(val: object) -> str | None:
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
    onet_work_activities: tuple[OnetWorkActivity, ...],
    skill_groups_by_skill: Mapping[str, frozenset[str]] | None = None,
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
        grouped_skill_ids, skill_groups_by_skill or {}
    )
    work_activity = _work_activity_payload(ja.work_activity_ids, onet_work_activities)

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


class RankedJobDocumentAssembler:
    """Builds the ranked job document for one language pack's lookups."""

    def __init__(
        self,
        settings: Settings,
        taxonomy: TaxonomyContext,
        onet_work_activities: tuple[OnetWorkActivity, ...],
    ):
        self._settings = settings
        self._taxonomy = taxonomy
        self._onet = onet_work_activities

    def assemble(
        self,
        classified_doc: dict[str, Any],
        ctx: JobDescriptionContext,
        *,
        job_attributes: JobAttributesResult | None,
        skill_ranker: SkillRankerResult | None,
        occupation_to_skills: list[dict[str, Any]] | None,
        ranker_candidates: list[EscoSkillCandidate],
        classifier_skill_candidates: list[EscoSkillCandidate],
    ) -> dict[str, Any] | None:
        """The ``$set`` body for the target collection; ``None`` without a job-attributes result."""
        tx = self._taxonomy
        return build_ranked_job_document(
            classified_doc,
            ctx,
            settings=self._settings,
            ja=job_attributes,
            skill_ranker=(
                (
                    skill_ranker.essential,
                    skill_ranker.optional,
                    skill_ranker.mapped,
                    skill_ranker.raw_response,
                )
                if skill_ranker is not None
                else None
            ),
            occupation_to_skills=occupation_to_skills,
            onet_work_activities=self._onet,
            skill_groups_by_skill=tx.skill_groups_by_skill,
            skills_uuid_tabiya_map=tx.skills_uuid_tabiya_map,
            occupations_bridge=tx.occ_bridge or None,
            occupations_all_ids=tx.occ_all_ids or None,
            ranker_candidates=ranker_candidates,
            classifier_skill_candidates=classifier_skill_candidates,
        )
