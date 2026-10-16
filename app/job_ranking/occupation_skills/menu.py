"""The occupation-skills menu: filtering, capping, folding into LLM1 and applying LLM3. No IO."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from app.config import Settings
from app.job_ranking.occupation_skills.types import OccupationSkillsRelevanceResult
from utils.classifier_linked_entities import EscoSkillCandidate, _norm_label
from utils.occupation_to_skills_mapping import (
    esco_skill_uri_for_uuid,
    occupation_block_skill_counts,
)


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


def llm3_menu_skill_candidates(
    menu_items: list[dict[str, Any]],
    existing: list[EscoSkillCandidate],
) -> list[EscoSkillCandidate]:
    """
    LLM3 menu skills (essential/optional) as extra LLM1 candidates, in menu order.

    Skips skills whose ESCO uuid or normalized label is already in ``existing`` (or was added
    from an earlier block), so a classifier candidate is never shadowed in label→URI mapping.
    """
    seen_uuids = {(c.esco_uuid or "").strip().lower() for c in existing if c.esco_uuid}
    seen_labels = {_norm_label(c.label) for c in existing}
    out: list[EscoSkillCandidate] = []
    for block in menu_items:
        tid = block.get("tabiya_occupation_id")
        if tid is None or str(tid).strip() == "":
            continue
        for sk in block.get("skills") or []:
            if not isinstance(sk, dict):
                continue
            rt = (sk.get("relation_type") or "").strip().lower()
            if rt not in ("essential", "optional"):
                continue
            sid = str(sk.get("skill_id") or "").strip().lower()
            label = (sk.get("label") or "").strip()
            if not sid or not label:
                continue
            if sid in seen_uuids or _norm_label(label) in seen_labels:
                continue
            seen_uuids.add(sid)
            seen_labels.add(_norm_label(label))
            out.append(
                EscoSkillCandidate(
                    label=label,
                    uri=(sk.get("uri") or "").strip() or esco_skill_uri_for_uuid(sid),
                    esco_uuid=sid,
                )
            )
    return out


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
