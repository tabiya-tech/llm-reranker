"""Classifier output: ESCO ``linked_entities`` extraction (skills, occupations, …) and skill-ranker helpers."""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from pydantic import Field

from app.config import Settings
from app.domain_base import DomainModel
from app.llm.errors import LlmParseError
from app.llm_models import SkillRankerResponse, parse_llm_json_response

if TYPE_CHECKING:
    # Deferred: app.job_context imports app.job_ranking.routes.types -> app.job_ranking.ranked_document,
    # which imports this module — a real top-level import here would be a circular import.
    from app.job_context import JobDescriptionContext
    from app.job_ranking.routes.types import RankRequest

# Linked entities always use ESCO taxonomy in classifier output.
LINKED_ENTITY_TAXONOMY_ESCO = "esco"


def _norm_label(s: str) -> str:
    return " ".join(s.strip().lower().split())


def normalize_linked_entity_type(raw: str) -> str:
    """
    Align classifier ``entity_type`` with env / storage keys.

    Classifier payloads may use ``skills`` / ``occupations``; we store ``skill`` / ``occupation``.
    """
    t = raw.strip().lower()
    if t == "skills":
        return "skill"
    if t == "occupations":
        return "occupation"
    return t


def linked_entity_preferred_taxonomy_id(le: dict[str, Any]) -> str | None:
    """
    Classifier-supplied id for a linked ESCO hit (uuid, Tabiya hex id, or uri-like string).

    Used to resolve rows in ``occupations.csv`` together with ``uri`` / ``originUUID``.
    """
    for key in (
        "linked_entity_preferred_taxonomy_id",
        "preferred_taxonomy_id",
        "taxonomy_preferred_id",
        "taxonomyId",
        "taxonomy_id",
        "concept_id",
        "conceptId",
        "esco_uuid",
        "escoUuid",
        "concept_uuid",
        "conceptUuid",
    ):
        v = le.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


# ESCO HTTP URIs end with the concept UUID, e.g. .../skill/5247a6ce-6ed0-4fb3-b121-758921dcc44e
_ESCO_URI_LAST_SEGMENT_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _esco_concept_uuid_from_uri(uri: str) -> str | None:
    """Return the ESCO concept UUID from the last path segment of an ESCO ``uri``, if present."""
    if not uri or not uri.strip():
        return None
    part = uri.strip().rstrip("/").rsplit("/", 1)[-1]
    if _ESCO_URI_LAST_SEGMENT_UUID.match(part):
        return part.lower()
    return None


class LinkedEntityCandidate(DomainModel):
    """One linked taxonomy hit from classifier ``classification.entities[].linked_entities``."""

    entity_type: str = Field(description="Normalized entity type (`skill`, `occupation`, …).")
    label: str = Field(description="Classifier label for the linked concept.")
    uri: str = Field(description="Concept URI.")
    similarity_score: float | None = Field(
        default=None, description="Classifier similarity score, when present."
    )
    concept_uuid: str | None = Field(
        default=None,
        description="Parsed from `uri`'s last segment when it looks like a concept UUID (e.g. ESCO).",
    )


class EscoSkillCandidate(DomainModel):
    """One ESCO-linked skill candidate offered to the skill ranker."""

    label: str = Field(description="Classifier label for the skill.")
    uri: str = Field(description="ESCO skill URI.")
    similarity_score: float | None = Field(
        default=None, description="Classifier similarity score, when present."
    )
    esco_uuid: str | None = Field(
        default=None, description="ESCO concept UUID, parsed from `uri` only."
    )


class MappedEscoSkills(DomainModel):
    """LLM1's essential/optional skill labels, mapped back to ESCO URIs and UUIDs."""

    essential_uris: list[str] = Field(
        default_factory=list, description="ESCO URIs matched from LLM1's essential labels."
    )
    optional_uris: list[str] = Field(
        default_factory=list, description="ESCO URIs matched from LLM1's optional labels."
    )
    essential_skills_uuids: list[str] = Field(
        default_factory=list,
        description="ESCO UUIDs parallel to `essential_uris` (empty string if unparseable).",
    )
    optional_skills_uuids: list[str] = Field(
        default_factory=list,
        description="ESCO UUIDs parallel to `optional_uris` (empty string if unparseable).",
    )
    unmatched_essential: list[str] = Field(
        default_factory=list, description="Essential labels LLM1 returned that match no candidate."
    )
    unmatched_optional: list[str] = Field(
        default_factory=list, description="Optional labels LLM1 returned that match no candidate."
    )


def _pick_best_linked_entity_by_score(items: list[LinkedEntityCandidate]) -> LinkedEntityCandidate:
    """Prefer highest ``similarity_score``; ``None`` loses to any float; ties → smallest ``uri``."""

    def score_val(c: LinkedEntityCandidate) -> float:
        s = c.similarity_score
        return float("-inf") if s is None else float(s)

    best = max(score_val(c) for c in items)
    tied = [c for c in items if score_val(c) == best]
    return min(tied, key=lambda c: c.uri)


def _dedupe_linked_entities_by_label(
    candidates: list[LinkedEntityCandidate],
    dedupe_types: frozenset[str] | None,
) -> list[LinkedEntityCandidate]:
    if not dedupe_types:
        return candidates
    groups: dict[tuple[str, str], list[LinkedEntityCandidate]] = defaultdict(list)
    for c in candidates:
        if c.entity_type in dedupe_types:
            groups[(c.entity_type, _norm_label(c.label))].append(c)
    winners = {k: _pick_best_linked_entity_by_score(v) for k, v in groups.items()}
    out: list[LinkedEntityCandidate] = []
    for c in candidates:
        if c.entity_type not in dedupe_types:
            out.append(c)
            continue
        w = winners[(c.entity_type, _norm_label(c.label))]
        if c.uri == w.uri:
            out.append(c)
    return out


def extract_linked_entity_candidates(
    classified: RankRequest | dict[str, Any],
    settings: Settings,
) -> list[LinkedEntityCandidate]:
    """
    Collect linked taxonomy candidates for configured ``entity_type`` values (e.g. skill, occupation).

    Uses ``Settings.classified_linked_entity_types`` and linked_entities with taxonomy ``esco`` only.
    Optional ``Settings.classified_linked_entity_min_similarity`` drops low ``similarity_score`` rows;
    ``classified_linked_entity_min_similarity_strict`` controls whether missing scores are dropped too.
    Optional ``Settings.classified_linked_entity_label_dedupe_types`` keeps one row per normalized
    ``label`` (per type), preferring highest score.
    Dedupes by (entity_type, uri), keeping best similarity_score; then applies label dedupe if configured.
    """
    from app.job_ranking.routes.types import RankRequest

    s = settings
    wanted = frozenset(s.classified_linked_entity_types)
    by_key: dict[tuple[str, str], LinkedEntityCandidate] = {}
    doc = (
        classified
        if isinstance(classified, RankRequest)
        else RankRequest.model_validate(classified)
    )
    entities = doc.classification.entities if doc.classification else []
    for ent in entities:
        et_raw = ent.entity_type
        if not et_raw or not et_raw.strip():
            continue
        et = normalize_linked_entity_type(et_raw)
        if et not in wanted:
            continue
        for le in ent.linked_entities:
            if (le.taxonomy or "").lower() != LINKED_ENTITY_TAXONOMY_ESCO:
                continue
            label = (le.label or "").strip()
            uri = (le.uri or "").strip()
            if not label or not uri:
                continue
            score = le.similarity_score
            min_sim = s.classified_linked_entity_min_similarity
            if min_sim is not None:
                if s.classified_linked_entity_min_similarity_strict and score is None:
                    continue
                if score is not None and score < min_sim:
                    continue
            cu = _esco_concept_uuid_from_uri(uri)
            key = (et, uri)
            prev = by_key.get(key)
            cand = LinkedEntityCandidate(
                entity_type=et,
                label=label,
                uri=uri,
                similarity_score=score,
                concept_uuid=cu,
            )
            if prev is None:
                by_key[key] = cand
            elif score is not None and (
                prev.similarity_score is None or score > prev.similarity_score
            ):
                by_key[key] = cand
    merged = list(by_key.values())
    return _dedupe_linked_entities_by_label(merged, s.classified_linked_entity_label_dedupe_types)


def group_linked_entities_by_type(
    classified: RankRequest | dict[str, Any],
    settings: Settings,
) -> dict[str, list[LinkedEntityCandidate]]:
    """Candidates grouped by ``entity_type`` (lowercase keys)."""
    out: dict[str, list[LinkedEntityCandidate]] = {}
    for c in extract_linked_entity_candidates(classified, settings):
        out.setdefault(c.entity_type, []).append(c)
    return out


def extract_esco_skill_candidates(
    classified: RankRequest | dict[str, Any],
    settings: Settings,
) -> list[EscoSkillCandidate]:
    """
    ESCO-linked **skills** only (``entity_type`` skill), for the skill ranker.

    Delegates to :func:`extract_linked_entity_candidates` and maps to ``EscoSkillCandidate``.
    """
    out: list[EscoSkillCandidate] = []
    for c in extract_linked_entity_candidates(classified, settings):
        if c.entity_type != "skill":
            continue
        out.append(
            EscoSkillCandidate(
                label=c.label,
                uri=c.uri,
                similarity_score=c.similarity_score,
                esco_uuid=c.concept_uuid,
            )
        )
    return out


def job_identifier(ctx: JobDescriptionContext, doc: dict[str, Any] | None = None) -> str:
    """Stable id for logging: fingerprint, job id, or Mongo ``_id``."""
    if ctx.job_fingerprint:
        return ctx.job_fingerprint
    if ctx.job_id is not None:
        return str(ctx.job_id)
    if ctx.classified.id is not None:
        return str(ctx.classified.id)
    if doc is not None:
        return str(doc.get("_id", ""))
    return ""


class SkillPromptEntry(DomainModel):
    """One skill-ranker menu row, as sent to the LLM1 prompt (label + optional description only)."""

    label: str = Field(description="Classifier label for the skill.")
    description: str | None = Field(
        default=None, description="`skills.csv` description for this ESCO uuid, when present."
    )


def skills_to_prompt_dicts(
    candidates: list[EscoSkillCandidate],
    *,
    description_by_esco: dict[str, str] | None = None,
) -> list[SkillPromptEntry]:
    """
    Serialize candidates for :func:`app.job_ranking.skill_ranker.prompt.build_skill_ranker_prompt`.

    ``description_by_esco``: optional lowercase ESCO uuid → text from ``skills.csv`` (see
    :func:`app.taxonomy.csv_loaders.load_skill_description_by_esco`). URIs and classifier scores
    are omitted from the prompt.
    """
    out: list[SkillPromptEntry] = []
    for c in candidates:
        description: str | None = None
        u = (c.esco_uuid or "").strip().lower()
        if description_by_esco and u:
            desc = description_by_esco.get(u)
            if desc and str(desc).strip():
                description = str(desc).strip()
        out.append(SkillPromptEntry(label=c.label, description=description))
    return out


def full_details_from_context(ctx: JobDescriptionContext) -> str:
    """Single text block for the skill ranker (title, employer, JD, requirements, key metadata)."""
    parts: list[str] = []
    if ctx.title:
        parts.append(f"Title: {ctx.title}")
    if ctx.employer:
        parts.append(f"Employer: {ctx.employer}")
    if ctx.category:
        parts.append(f"Category: {ctx.category}")
    if ctx.employment_type:
        parts.append(f"Employment type: {ctx.employment_type}")
    if ctx.city:
        parts.append(f"City: {ctx.city}")
    if ctx.county:
        parts.append(f"Region: {ctx.county}")
    if ctx.salary:
        parts.append(f"Salary: {ctx.salary}")
    if ctx.raw_job_description:
        parts.append(f"Description:\n{ctx.raw_job_description}")
    if ctx.requirements_text:
        parts.append(f"Requirements:\n{ctx.requirements_text}")
    return "\n\n".join(parts) if parts else "(no opportunity details)"


def json_bool(v: object, default: bool = True) -> bool:
    """A boolean from model JSON: real bools, or "true"/"1"/"yes" and "false"/"0"/"no"."""
    if isinstance(v, bool):
        return v
    s = str(v or "").strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return default


def parse_skill_ranker_json(raw_text: str) -> SkillRankerResponse:
    """Parse ``essential_skills`` and ``optional_skills`` from model output (JSON object)."""
    data = parse_llm_json_response(raw_text)
    ess_raw = data.get("essential_skills")
    opt_raw = data.get("optional_skills")
    if not isinstance(ess_raw, list) or not isinstance(opt_raw, list):
        raise LlmParseError("essential_skills and optional_skills must be lists")
    essential = [str(x) for x in ess_raw if isinstance(x, str) and x.strip()]
    optional = [str(x) for x in opt_raw if isinstance(x, str) and x.strip()]
    return SkillRankerResponse(essential_skills=essential, optional_skills=optional)


# Minimum similarity for fuzzy label match (0–1)
FUZZY_LABEL_MIN_RATIO = 0.82


def _best_fuzzy_label(
    query: str,
    candidates: list[EscoSkillCandidate],
) -> EscoSkillCandidate | None:
    nq = _norm_label(query)
    if not nq:
        return None
    best: EscoSkillCandidate | None = None
    best_ratio = 0.0
    for c in candidates:
        r = SequenceMatcher(None, nq, _norm_label(c.label)).ratio()
        if r > best_ratio:
            best_ratio = r
            best = c
    if best is not None and best_ratio >= FUZZY_LABEL_MIN_RATIO:
        return best
    return None


def _resolve_label_to_candidate(
    lab: str,
    candidates: list[EscoSkillCandidate],
    by_norm: dict[str, EscoSkillCandidate],
) -> EscoSkillCandidate | None:
    cand = by_norm.get(_norm_label(lab))
    if cand is not None:
        return cand
    return _best_fuzzy_label(lab, candidates)


def map_skill_labels_to_esco_uris(
    essential: list[str],
    optional: list[str],
    candidates: list[EscoSkillCandidate],
) -> MappedEscoSkills:
    """Map model skill labels to ESCO URIs and concept UUIDs (from each candidate uri). Exact match, then fuzzy; dedupe by URI."""
    if not candidates:
        return MappedEscoSkills(
            unmatched_essential=list(essential),
            unmatched_optional=list(optional),
        )

    by_norm: dict[str, EscoSkillCandidate] = {}
    for c in candidates:
        by_norm[_norm_label(c.label)] = c

    seen: set[str] = set()
    out_ess: list[str] = []
    out_opt: list[str] = []
    ess_uuid: list[str] = []
    opt_uuid: list[str] = []
    un_ess: list[str] = []
    un_opt: list[str] = []

    for lab in essential:
        cand = _resolve_label_to_candidate(lab, candidates, by_norm)
        if cand is None:
            un_ess.append(lab)
            continue
        uri = cand.uri
        if uri in seen:
            continue
        seen.add(uri)
        out_ess.append(uri)
        ess_uuid.append(cand.esco_uuid or "")

    for lab in optional:
        cand = _resolve_label_to_candidate(lab, candidates, by_norm)
        if cand is None:
            un_opt.append(lab)
            continue
        uri = cand.uri
        if uri in seen:
            continue
        seen.add(uri)
        out_opt.append(uri)
        opt_uuid.append(cand.esco_uuid or "")

    return MappedEscoSkills(
        essential_uris=out_ess,
        optional_uris=out_opt,
        essential_skills_uuids=ess_uuid,
        optional_skills_uuids=opt_uuid,
        unmatched_essential=un_ess,
        unmatched_optional=un_opt,
    )
