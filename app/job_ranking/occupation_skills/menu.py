"""Builds and filters the occupation-skills menu sent to LLM3 (or folded into LLM1)."""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

from pydantic import Field

from app.config import Settings
from app.domain_base import DomainModel
from app.job_ranking.occupation_skills.types import OccupationSkillsRelevanceResult
from app.job_ranking.routes.types import OccupationSkill, OccupationSkillsBlock, SkillCounts
from utils.classifier_linked_entities import EscoSkillCandidate, _norm_label
from utils.occupation_to_skills_mapping import esco_skill_uri_for_uuid


def occupation_block_skill_counts_from_models(skills: list[OccupationSkill]) -> SkillCounts:
    """Count ``total``/``essential``/``optional``/``unspecified`` over a block's typed skill rows."""
    essential = optional = unspecified = 0
    for sk in skills:
        if sk.relation_type == "essential":
            essential += 1
        elif sk.relation_type == "optional":
            optional += 1
        else:
            unspecified += 1
    return SkillCounts(
        total=len(skills),
        essential=essential,
        optional=optional,
        unspecified=unspecified,
    )


def occupation_skills_relevance_menu_size(items: list[OccupationSkillsBlock]) -> int:
    """How many taxonomy skills (essential + optional) LLM call 3 must judge."""
    n = 0
    for block in items:
        if not str(block.tabiya_occupation_id or "").strip():
            continue
        for sk in block.skills:
            if sk.relation_type not in ("essential", "optional"):
                continue
            if sk.skill_id.strip():
                n += 1
    return n


def classifier_skill_esco_uuids_lower(candidates: list[EscoSkillCandidate]) -> frozenset[str]:
    """Lowercased ESCO uuid strings from classifier skill linked entities (for LLM3 menu filtering)."""
    return frozenset(
        (c.esco_uuid or "").strip().lower() for c in candidates if (c.esco_uuid or "").strip()
    )


def occupation_skill_signal_tier(skill: OccupationSkill) -> str:
    """
    Normalized CSV ``SIGNALLINGVALUELABEL`` tier for one skill row: ``high``, ``medium``,
    ``low``, or ``unlabelled`` (empty / other).
    """
    lab = (skill.signalling_label or "").strip().lower()
    if lab in ("high", "medium", "low"):
        return lab
    return "unlabelled"


def _llm3_signalling_label_rank(skill: OccupationSkill) -> int:
    """
    Tier for CSV ``SIGNALLINGVALUELABEL`` (lower = stronger).

    Empty or unknown labels map to a rank above ``low`` so they are dropped when
    ``RERANKER_LLM3_MENU_SIGNAL_FILTER`` is ``high``, ``medium``, or ``low``.
    """
    t = occupation_skill_signal_tier(skill)
    return {"high": 0, "medium": 1, "low": 2}.get(t, 99)


def filter_occupation_skills_nested_by_signal_include(
    items: list[OccupationSkillsBlock],
    *,
    include_tiers: frozenset[str] | None,
) -> list[OccupationSkillsBlock]:
    """
    Drop essential/optional skills whose signalling tier is not in ``include_tiers``.

    Non-essential/optional rows in a block's ``skills`` list are kept unchanged.
    Blocks that end up with no skills are removed.

    ``include_tiers``: ``None`` -> return ``items`` unchanged (no copy).
    Otherwise use names ``high``, ``medium``, ``low``, ``unlabelled`` (see
    ``RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE``).
    """
    if not include_tiers:
        return items
    out: list[OccupationSkillsBlock] = []
    for block in items:
        kept: list[OccupationSkill] = []
        for sk in block.skills:
            if sk.relation_type not in ("essential", "optional"):
                kept.append(sk)
                continue
            if occupation_skill_signal_tier(sk) in include_tiers:
                kept.append(sk)
        if not kept:
            continue
        nb = block.model_copy(
            update={
                "skills": kept,
                "skill_counts": occupation_block_skill_counts_from_models(kept),
            }
        )
        out.append(nb)
    return out


def _llm3_menu_skill_sort_tuple(
    skill: OccupationSkill, bi: int, si: int
) -> tuple[int, float, int, int, int]:
    """
    Sort key for LLM3 menu rows: stronger CSV signalling first, then essential over optional,
    then document order. Ascending sort = higher priority first.
    """
    lab_rank = _llm3_signalling_label_rank(skill)
    raw = (skill.signalling_value or "").strip()
    try:
        v = float(raw)
    except ValueError:
        v = float("-inf")
    rt_rank = 0 if skill.relation_type == "essential" else 1
    return (lab_rank, -v, rt_rank, bi, si)


class Llm3MenuSubset(DomainModel):
    """A (possibly capped/filtered) occupation-skills menu built for the LLM3 prompt."""

    items: list[OccupationSkillsBlock] = Field(
        description="Occupation blocks to send to LLM3, trimmed to the prompt budget."
    )
    full_menu_count: int = Field(
        description="Essential+optional taxonomy skills in the untrimmed menu."
    )
    subset_menu_count: int = Field(description="Essential+optional taxonomy skills in `items`.")


def subset_occupation_nested_for_llm3_prompt(
    full_items: list[OccupationSkillsBlock],
    *,
    max_menu_skills: int,
    classifier_esco_lower: frozenset[str] | None,
    signal_menu_filter: str = "off",
) -> Llm3MenuSubset:
    """
    Build a smaller occupation->skills nesting for the LLM3 prompt only.

    Callers should pass the result's ``items`` to the relevance model, then
    ``apply_occupation_skills_listing_relevance`` on the **original** ``full_items`` with the
    parsed result: skills omitted from the prompt keep the default ``llm_is_relevant`` of
    ``True``.

    When ``signal_menu_filter`` is ``high``, ``medium``, or ``low``, only relations whose
    CSV ``SIGNALLINGVALUELABEL`` is at least that strong are kept (empty labels are dropped).
    That yields a small, high-signal menu without an arbitrary count cap; combine with
    ``max_menu_skills`` if you still want a hard ceiling.

    When ``max_menu_skills`` > 0, skills are chosen in **CSV signalling strength** order
    (``signalling_label`` high/medium/low, then numeric ``signalling_value``, then essential
    before optional, then listing order) so the cap favours taxonomy rows that are explicitly
    weighted in the relations file-not the first rows in file order.

    ``max_menu_skills`` <= 0: no numeric cap after the signal filter (classifier filter may
    still apply).
    ``classifier_esco_lower``: when not ``None``, only skills whose ``skill_id`` is in this set.
    """
    full_n = occupation_skills_relevance_menu_size(full_items)
    if full_n == 0:
        return Llm3MenuSubset(items=[], full_menu_count=0, subset_menu_count=0)

    budget = full_n if max_menu_skills <= 0 else min(max_menu_skills, full_n)

    class _MenuEntry(NamedTuple):
        bi: int
        si: int
        skill: OccupationSkill
        sort_key: tuple[int, float, int, int, int]

    entries: list[_MenuEntry] = []
    for bi, block in enumerate(full_items):
        if not str(block.tabiya_occupation_id or "").strip():
            continue
        for si, sk in enumerate(block.skills):
            if sk.relation_type not in ("essential", "optional"):
                continue
            sid = sk.skill_id.strip()
            if not sid:
                continue
            if classifier_esco_lower is not None and sid.lower() not in classifier_esco_lower:
                continue
            entries.append(
                _MenuEntry(bi=bi, si=si, skill=sk, sort_key=_llm3_menu_skill_sort_tuple(sk, bi, si))
            )

    filt = (signal_menu_filter or "off").strip().lower()
    if filt != "off":
        max_rank = {"high": 0, "medium": 1, "low": 2}.get(filt)
        if max_rank is not None:
            entries = [e for e in entries if _llm3_signalling_label_rank(e.skill) <= max_rank]

    entries.sort(key=lambda e: e.sort_key)
    entries = entries[:budget]
    entries.sort(key=lambda e: (e.bi, e.si))

    by_bi: dict[int, list[tuple[int, OccupationSkill]]] = defaultdict(list)
    for e in entries:
        by_bi[e.bi].append((e.si, e.skill))

    out_blocks: list[OccupationSkillsBlock] = []
    for bi in sorted(by_bi.keys()):
        block = full_items[bi]
        chunk = [sk for _, sk in sorted(by_bi[bi], key=lambda x: x[0])]
        nb = block.model_copy(
            update={
                "skills": chunk,
                "skill_counts": occupation_block_skill_counts_from_models(chunk),
            }
        )
        out_blocks.append(nb)

    subset_n = occupation_skills_relevance_menu_size(out_blocks)
    return Llm3MenuSubset(items=out_blocks, full_menu_count=full_n, subset_menu_count=subset_n)


def build_llm3_occupation_menu(
    occupation_to_skills_nested: list[OccupationSkillsBlock],
    classifier_skill_candidates: list[EscoSkillCandidate],
    settings: Settings,
) -> Llm3MenuSubset:
    """Apply ``Settings`` LLM3 menu controls (signal tier filter, max size, classifier intersect)."""
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


def llm3_menu_skill_candidates(
    menu_items: list[OccupationSkillsBlock],
    existing: list[EscoSkillCandidate],
) -> list[EscoSkillCandidate]:
    """
    LLM3 menu skills (essential/optional) as extra LLM1 candidates, in menu order.

    Skips skills whose ESCO uuid or normalized label is already in ``existing`` (or was added
    from an earlier block), so a classifier candidate is never shadowed in label->URI mapping.
    """
    seen_uuids = {(c.esco_uuid or "").strip().lower() for c in existing if c.esco_uuid}
    seen_labels = {_norm_label(c.label) for c in existing}
    out: list[EscoSkillCandidate] = []
    for block in menu_items:
        if not str(block.tabiya_occupation_id or "").strip():
            continue
        for sk in block.skills:
            if sk.relation_type not in ("essential", "optional"):
                continue
            sid = sk.skill_id.strip().lower()
            label = sk.label.strip()
            if not sid or not label:
                continue
            if sid in seen_uuids or _norm_label(label) in seen_labels:
                continue
            seen_uuids.add(sid)
            seen_labels.add(_norm_label(label))
            out.append(
                EscoSkillCandidate(
                    label=label,
                    uri=(sk.uri or "").strip() or esco_skill_uri_for_uuid(sid),
                    esco_uuid=sid,
                )
            )
    return out


def apply_occupation_skills_listing_relevance(
    items: list[OccupationSkillsBlock],
    result: OccupationSkillsRelevanceResult,
) -> list[OccupationSkillsBlock]:
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

    out: list[OccupationSkillsBlock] = []
    for block in items:
        tid = str(block.tabiya_occupation_id or "").strip().lower()
        new_skills: list[OccupationSkill] = []
        for sk in block.skills:
            sid = sk.skill_id.strip().lower()
            if sk.relation_type in ("essential", "optional") and tid and sid:
                is_relevant = decision_map.get((tid, sid), True)
            else:
                is_relevant = True
            new_skills.append(sk.model_copy(update={"llm_is_relevant": is_relevant}))
        rel_only = [sk for sk in new_skills if sk.llm_is_relevant is not False]
        tax = occupation_block_skill_counts_from_models(new_skills)
        out.append(
            block.model_copy(
                update={
                    "skills": new_skills,
                    "skill_counts": tax.model_copy(update={"relevant_total": len(rel_only)}),
                }
            )
        )
    return out
