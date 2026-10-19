"""Unit + regression tests for ``app.job_ranking.ranked_document`` and ``app.job_ranking.occupation_skills.menu``.

This module produces every document written to ``ranked_jobs``. The tests:

- Pin ``PIPELINE_SCHEMA_VERSION`` (regression).
- Pin the top-level keys of ``build_ranked_job_document`` output (regression).
- Cover the menu / filtering / aggregation pure functions.

We do NOT test downstream consumers' field reads here — only the contract of
this builder. Bumping the schema version is the legitimate way to change the
pin; tests must be updated together with the version.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.job_context import JobDescriptionContext
from app.job_ranking.job_attributes.repository import JobAttributesRepository
from app.job_ranking.job_attributes.types import AttributeEstimate, JobAttributesResult
from app.job_ranking.occupation_skills.menu import (
    _llm3_signalling_label_rank,
    apply_occupation_skills_listing_relevance,
    build_llm3_occupation_menu,
    classifier_skill_esco_uuids_lower,
    filter_occupation_skills_nested_by_signal_include,
    llm3_menu_skill_candidates,
    occupation_block_skill_counts_from_models,
    occupation_skill_signal_tier,
    occupation_skills_relevance_menu_size,
    subset_occupation_nested_for_llm3_prompt,
)
from app.job_ranking.occupation_skills.types import (
    OccupationSkillsRelevanceDecision,
    OccupationSkillsRelevanceResult,
)
from app.job_ranking.ranked_document import (
    PIPELINE_SCHEMA_VERSION,
    build_llm_reranker_meta,
    build_ranked_job_document,
    classifier_metadata_from_context,
    derive_skill_groups_origin_uuids,
    location_keys_from_metadata,
    occupation_to_skills_with_meta,
    work_activity_names_for_ids,
)
from app.job_ranking.routes.types import (
    OccupationSkill,
    OccupationSkillsBlock,
    RankRequest,
    SkillCounts,
)
from utils.classifier_linked_entities import EscoSkillCandidate, MappedEscoSkills

# ``JobDescriptionContext.classified`` is typed as the forward ref ``RankRequest`` (deferred
# to avoid a circular import between app.job_context and app.job_ranking.routes.types). The
# rebuild normally happens as a side effect of calling ``build_job_description_context``, but
# this module constructs ``JobDescriptionContext`` directly, so it must trigger the rebuild
# itself rather than relying on another test module having imported/run first.
JobDescriptionContext.model_rebuild()

_DEFAULT_URI = "__default__"


def _skill(
    skill_id: str,
    relation_type: str,
    *,
    label: str = "Skill",
    uri: str = _DEFAULT_URI,
    signalling_label: str | None = None,
    signalling_value: str | None = None,
) -> OccupationSkill:
    resolved_uri = f"http://data.europa.eu/esco/skill/{skill_id}" if uri == _DEFAULT_URI else uri
    return OccupationSkill(
        skill_id=skill_id,
        tabiya_skill_id=skill_id,
        label=label,
        uri=resolved_uri,
        relation_type=relation_type,
        signalling_label=signalling_label,
        signalling_value=signalling_value,
    )


def _menu_block(
    tabiya_occupation_id: str | None, skills: list[OccupationSkill]
) -> OccupationSkillsBlock:
    return OccupationSkillsBlock(
        occupation_label="Welder",
        occupation_esco_uuid=None,
        occupation_uri="http://data.europa.eu/esco/occupation/x",
        tabiya_occupation_id=tabiya_occupation_id,
        skills=skills,
        skill_counts=SkillCounts(
            total=len(skills),
            essential=sum(1 for s in skills if s.relation_type == "essential"),
            optional=sum(1 for s in skills if s.relation_type == "optional"),
            unspecified=sum(1 for s in skills if s.relation_type not in ("essential", "optional")),
        ),
    )


def _ctx(**overrides):
    base: dict[str, Any] = dict(
        job_id="J-1",
        job_fingerprint="fp-1",
        title="Engineer",
        employer="ACME",
        category="Tech",
        application_url=None,
        posted_date=None,
        closing_date=None,
        source_platform=None,
        raw_job_description="Build things.",
        requirements_text="3 years",
        salary=None,
        city="Lusaka",
        county=None,
        employment_type="Full time",
        classified=RankRequest.model_validate({"_id": "doc-1"}),
    )
    base.update(overrides)
    return JobDescriptionContext(**base)


def _ja_minimal() -> JobAttributesResult:
    return JobAttributesResult(
        attribute_estimates=(
            AttributeEstimate(
                attribute_name="earnings_per_month",
                selected_level_id="earn_15k",
                confidence="high",
                reasoning="r1",
            ),
        ),
        summary="ok",
        attributes={"earnings_per_month": "earn_15k"},
        work_activity_ids=(),
        raw_response="{}",
    )


# ---------------------------------------------------------------------------
# Regression pins
# ---------------------------------------------------------------------------


class TestPipelineSchemaVersion:
    def test_version_is_pinned(self):
        # Update intentionally when bumping the document schema.
        assert PIPELINE_SCHEMA_VERSION == "33"


# ---------------------------------------------------------------------------
# Tier helpers
# ---------------------------------------------------------------------------


class TestOccupationSkillSignalTier:
    @pytest.mark.parametrize(
        "label, expected",
        [
            ("high", "high"),
            ("HIGH", "high"),
            ("medium", "medium"),
            ("low", "low"),
            ("", "unlabelled"),
            (None, "unlabelled"),
            ("mystery", "unlabelled"),
        ],
    )
    def test_tier(self, label, expected):
        sk = _skill("s1", "essential", signalling_label=label)
        assert occupation_skill_signal_tier(sk) == expected


class TestLlm3SignallingLabelRank:
    @pytest.mark.parametrize(
        "label, rank",
        [("high", 0), ("medium", 1), ("low", 2), ("", 99), ("mystery", 99)],
    )
    def test_rank(self, label, rank):
        sk = _skill("s1", "essential", signalling_label=label)
        assert _llm3_signalling_label_rank(sk) == rank


# ---------------------------------------------------------------------------
# occupation_skills_relevance_menu_size
# ---------------------------------------------------------------------------


class TestOccupationSkillsRelevanceMenuSize:
    def test_counts_essential_and_optional_only(self):
        items = [
            _menu_block(
                "t1",
                [
                    _skill("s1", "essential"),
                    _skill("s2", "optional"),
                    _skill("s3", "unspecified"),  # excluded
                    # Note: a blank skill_id or a non-skill entry can no longer be constructed —
                    # OccupationSkill.skill_id is a required str field, so those invalid-input
                    # cases from the old dict-based test are no longer representable/applicable.
                ],
            ),
            _menu_block(
                None,  # excluded: no tabiya_occupation_id
                [_skill("x", "essential")],
            ),
        ]
        assert occupation_skills_relevance_menu_size(items) == 2


# ---------------------------------------------------------------------------
# filter_occupation_skills_nested_by_signal_include
# ---------------------------------------------------------------------------


class TestFilterOccupationSkillsNestedBySignalInclude:
    def _items(self):
        return [
            _menu_block(
                "t1",
                [
                    _skill("s1", "essential", signalling_label="high"),
                    _skill("s2", "optional", signalling_label="low"),
                    _skill("s3", "unspecified", signalling_label="low"),
                ],
            ),
            _menu_block(
                "t2",
                [_skill("s4", "optional", signalling_label="low")],
            ),
        ]

    def test_none_returns_input_identity(self):
        items = self._items()
        out = filter_occupation_skills_nested_by_signal_include(items, include_tiers=None)
        assert out is items

    def test_keeps_only_matching_tiers(self):
        out = filter_occupation_skills_nested_by_signal_include(
            self._items(), include_tiers=frozenset({"high"})
        )
        # Only block t1 has a high-tier skill; unspecified rows are kept untouched.
        assert len(out) == 1
        assert out[0].tabiya_occupation_id == "t1"
        kept = out[0].skills
        kept_ids = [s.skill_id for s in kept]
        # s1 essential high stays; s2 optional low filtered; s3 unspecified is non-essential/optional → kept
        assert "s1" in kept_ids
        assert "s2" not in kept_ids
        assert "s3" in kept_ids
        assert out[0].skill_counts is not None

    def test_drops_blocks_with_no_kept_skills(self):
        out = filter_occupation_skills_nested_by_signal_include(
            self._items(), include_tiers=frozenset({"medium"})
        )
        # No skill has medium tier; t2 has only optional/low → dropped; t1 keeps
        # only its unspecified row "s3".
        assert len(out) == 1
        assert [s.skill_id for s in out[0].skills] == ["s3"]


# ---------------------------------------------------------------------------
# subset_occupation_nested_for_llm3_prompt and build_llm3_occupation_menu
# ---------------------------------------------------------------------------


def _menu_items():
    return [
        _menu_block(
            "t1",
            [
                _skill("s_high", "essential", signalling_label="high", signalling_value="0.9"),
                _skill("s_med", "optional", signalling_label="medium", signalling_value="0.5"),
                _skill("s_low", "optional", signalling_label="low", signalling_value="0.1"),
                _skill("s_blank", "essential", signalling_label=""),
            ],
        ),
    ]


class TestSubsetOccupationNestedForLlm3Prompt:
    def test_full_menu_returns_all_when_no_cap(self):
        items = _menu_items()
        result = subset_occupation_nested_for_llm3_prompt(
            items, max_menu_skills=0, classifier_esco_lower=None
        )
        assert result.full_menu_count == 4
        assert result.subset_menu_count == 4
        assert result.items[0].skills[0].skill_id in {"s_high", "s_med", "s_low", "s_blank"}

    def test_signal_filter_drops_low_and_blank(self):
        items = _menu_items()
        result = subset_occupation_nested_for_llm3_prompt(
            items, max_menu_skills=0, classifier_esco_lower=None, signal_menu_filter="medium"
        )
        # high + medium kept; low + blank dropped
        assert result.full_menu_count == 4
        assert result.subset_menu_count == 2
        kept_ids = [s.skill_id for s in result.items[0].skills]
        assert set(kept_ids) == {"s_high", "s_med"}

    def test_max_menu_skills_caps_in_strength_order(self):
        items = _menu_items()
        result = subset_occupation_nested_for_llm3_prompt(
            items, max_menu_skills=2, classifier_esco_lower=None
        )
        assert result.subset_menu_count == 2
        kept = [s.skill_id for s in result.items[0].skills]
        assert kept == ["s_high", "s_med"]  # highest signal strength first

    def test_classifier_intersect_filters_to_subset(self):
        items = _menu_items()
        result = subset_occupation_nested_for_llm3_prompt(
            items,
            max_menu_skills=0,
            classifier_esco_lower=frozenset({"s_high", "s_low"}),
        )
        assert result.subset_menu_count == 2
        kept = [s.skill_id for s in result.items[0].skills]
        assert set(kept) == {"s_high", "s_low"}

    def test_empty_items_returns_zero(self):
        result = subset_occupation_nested_for_llm3_prompt(
            [], max_menu_skills=10, classifier_esco_lower=None
        )
        assert result.items == []
        assert result.full_menu_count == 0
        assert result.subset_menu_count == 0


class TestClassifierSkillEscoUuidsLower:
    def test_returns_lowercased_frozenset(self):
        cands = [
            EscoSkillCandidate(label="A", uri="u1", esco_uuid="ABC"),
            EscoSkillCandidate(label="B", uri="u2", esco_uuid=""),
            EscoSkillCandidate(label="C", uri="u3", esco_uuid="dEf"),
        ]
        out = classifier_skill_esco_uuids_lower(cands)
        assert out == frozenset({"abc", "def"})


class TestBuildLlm3OccupationMenu:
    def test_passes_settings_through(self, make_settings):
        s = make_settings(RERANKER_LLM3_MAX_MENU_SKILLS="2")
        result = build_llm3_occupation_menu(_menu_items(), [], s)
        assert result.subset_menu_count == 2
        # Default RERANKER_LLM3_CLASSIFIER_INTERSECT=False → classifier list ignored
        kept = [x.skill_id for x in result.items[0].skills]
        assert kept == ["s_high", "s_med"]


# ---------------------------------------------------------------------------
# llm3_menu_skill_candidates
# ---------------------------------------------------------------------------


class TestLlm3MenuSkillCandidates:
    def _items(self):
        return [
            _menu_block(
                "t1",
                [
                    _skill(
                        "AAA",
                        "essential",
                        label="Welding",
                        uri="http://data.europa.eu/esco/skill/aaa",
                    ),
                    _skill("bbb", "optional", label="Painting", uri=""),
                    _skill("ccc", "unspecified", label="Unspecified"),
                    # Note: a blank skill_id row ("No id" in the old dict-based test) can no
                    # longer be constructed — OccupationSkill.skill_id is a required str field.
                ],
            ),
            _menu_block(
                "t2",
                [
                    # Same skill under a second occupation → added once.
                    _skill("aaa", "optional", label="Welding"),
                    _skill("ddd", "essential", label="Python"),
                    _skill("eee", "essential", label="  painting "),
                ],
            ),
            _menu_block(
                None,
                [_skill("fff", "essential", label="Orphan")],
            ),
        ]

    def test_flattens_menu_in_order_and_builds_candidates(self):
        out = llm3_menu_skill_candidates(self._items(), [])
        assert [c.label for c in out] == ["Welding", "Painting", "Python"]
        assert out[0] == EscoSkillCandidate(
            label="Welding", uri="http://data.europa.eu/esco/skill/aaa", esco_uuid="aaa"
        )
        # Missing uri falls back to the canonical ESCO skill URI.
        assert out[1].uri == "http://data.europa.eu/esco/skill/bbb"
        assert out[1].similarity_score is None

    def test_skips_skills_already_in_existing_by_uuid_or_label(self):
        existing = [
            EscoSkillCandidate(label="Something else", uri="u1", esco_uuid="AAA"),
            EscoSkillCandidate(label="python", uri="u2", esco_uuid="zzz"),
        ]
        out = llm3_menu_skill_candidates(self._items(), existing)
        assert [c.esco_uuid for c in out] == ["bbb"]

    def test_empty_menu(self):
        assert llm3_menu_skill_candidates([], []) == []


# ---------------------------------------------------------------------------
# apply_occupation_skills_listing_relevance
# ---------------------------------------------------------------------------


class TestApplyOccupationSkillsListingRelevance:
    def test_marks_skills_and_rolls_up_relevant_total(self):
        items = [
            _menu_block(
                "t1",
                [
                    _skill("s1", "essential"),
                    _skill("s2", "optional"),
                    _skill("s3", "unspecified"),
                ],
            )
        ]
        result = OccupationSkillsRelevanceResult(
            decisions=(
                OccupationSkillsRelevanceDecision(
                    skill_id="s1", tabiya_occupation_id="t1", is_relevant=True
                ),
                OccupationSkillsRelevanceDecision(
                    skill_id="s2", tabiya_occupation_id="t1", is_relevant=False
                ),
                # s3 has no decision; default True
            ),
            raw_response="{}",
        )
        out = apply_occupation_skills_listing_relevance(items, result)
        skills = out[0].skills
        flags = {s.skill_id: s.llm_is_relevant for s in skills}
        assert flags == {"s1": True, "s2": False, "s3": True}
        assert out[0].skill_counts.relevant_total == 2  # s1 + s3
        # Original taxonomy counts preserved
        assert out[0].skill_counts.total == 3
        # Original items list is untouched (new blocks/skills, not in-place mutation).
        assert items[0].skills[0].llm_is_relevant is None


# ---------------------------------------------------------------------------
# occupation_block_skill_counts_from_models
# ---------------------------------------------------------------------------


class TestOccupationBlockSkillCountsFromModels:
    def test_counts_relation_types(self):
        skills = [
            _skill("s1", "essential"),
            _skill("s2", "essential"),
            _skill("s3", "optional"),
            _skill("s4", "unspecified"),
            _skill("s5", "weird"),  # falls into unspecified bucket
        ]
        counts = occupation_block_skill_counts_from_models(skills)
        assert counts == SkillCounts(total=5, essential=2, optional=1, unspecified=2)


# ---------------------------------------------------------------------------
# occupation_to_skills_with_meta
# ---------------------------------------------------------------------------


def _occupation_block(
    tabiya_occupation_id: str, skill_counts: dict[str, int]
) -> OccupationSkillsBlock:
    return OccupationSkillsBlock(
        occupation_label="Welder",
        occupation_esco_uuid=None,
        occupation_uri="http://data.europa.eu/esco/occupation/x",
        tabiya_occupation_id=tabiya_occupation_id,
        skills=[],
        skill_counts=SkillCounts(**skill_counts),
    )


class TestOccupationToSkillsWithMeta:
    def test_empty_input_returns_zeros(self):
        out = occupation_to_skills_with_meta([])
        assert out.meta.occupation_count == 0
        assert out.meta.skill_counts.model_dump() == {
            "total": 0,
            "essential": 0,
            "optional": 0,
            "unspecified": 0,
            "relevant_total": 0,
        }
        assert out.items == []

    def test_aggregates_skill_counts(self):
        items = [
            _occupation_block(
                "t1",
                {"total": 5, "essential": 2, "optional": 2, "unspecified": 1, "relevant_total": 3},
            ),
            _occupation_block(
                "t2",
                {"total": 1, "essential": 1, "optional": 0, "unspecified": 0, "relevant_total": 0},
            ),
        ]
        out = occupation_to_skills_with_meta(items)
        assert out.meta.occupation_count == 2
        assert out.meta.skill_counts.model_dump() == {
            "total": 6,
            "essential": 3,
            "optional": 2,
            "unspecified": 1,
            "relevant_total": 3,
        }


# ---------------------------------------------------------------------------
# work_activity_names_for_ids
# ---------------------------------------------------------------------------


class TestWorkActivityNamesForIds:
    def test_resolves_real_resource_ids(self, make_settings):
        s = make_settings()
        onet = JobAttributesRepository().load_onet_work_activities(s)
        names = work_activity_names_for_ids(("4.A.4.b.4",), onet)
        assert names and isinstance(names[0], str) and names[0]

    def test_unknown_id_returns_empty_string(self, make_settings):
        s = make_settings()
        onet = JobAttributesRepository().load_onet_work_activities(s)
        assert work_activity_names_for_ids(("zzz.unknown",), onet) == [""]


class TestDeriveSkillGroupsOriginUuids:
    def test_unions_and_sorts_parent_groups(self):
        groups = {"s1": frozenset({"g2", "g1"}), "s2": frozenset({"g1"})}
        assert derive_skill_groups_origin_uuids(["s1", "s2", "", "unknown"], groups) == [
            "g1",
            "g2",
        ]


# ---------------------------------------------------------------------------
# classifier_metadata_from_context + build_llm_reranker_meta
# ---------------------------------------------------------------------------


class TestClassifierMetadataFromContext:
    def test_includes_title_and_text_blocks(self):
        ctx = _ctx()
        meta = classifier_metadata_from_context(ctx)
        assert meta.title == "Engineer"
        assert meta.job_description == "Build things."
        assert meta.requirements == "3 years"


class TestLocationKeysFromMetadata:
    def test_keys_of_city_and_enriched_province(self):
        ctx = _ctx(
            city="Ndola, Lusaka",
            classified={"_id": "doc-1", "province": ["Copperbelt", "Lusaka"]},
        )
        meta = classifier_metadata_from_context(ctx)
        assert location_keys_from_metadata(meta) == ["copperbelt", "lusaka", "ndola"]

    def test_county_falls_back_to_province(self):
        meta = classifier_metadata_from_context(_ctx(city="Thika", county="Nairobi"))
        assert location_keys_from_metadata(meta) == ["nairobi", "thika"]

    def test_remote_anywhere_adds_the_remote_key(self):
        meta = classifier_metadata_from_context(_ctx(city=None, county="National/Remote"))
        assert location_keys_from_metadata(meta) == ["national", "remote"]


class TestBuildLlmRerankerMeta:
    def test_includes_schema_version(self, make_settings):
        s = make_settings(GEMINI_API_KEY="x", GEMINI_MODEL="m1")
        meta = build_llm_reranker_meta(s)
        assert meta.schema_version == PIPELINE_SCHEMA_VERSION
        assert meta.skill_ranker_model == "m1"
        assert meta.job_attributes_model == "m1"
        assert meta.occupation_skills_relevance_model == "m1"
        assert isinstance(meta.classified_linked_entity_types, list)


# ---------------------------------------------------------------------------
# build_ranked_job_document
# ---------------------------------------------------------------------------

EXPECTED_TOP_LEVEL_KEYS = {
    "job_id",
    "job_fingerprint",
    "is_active",
    "processed_at",
    "source_classified_id",
    "classified_skills",
    "classified_occupations",
    "occupation_to_skills",
    "work_activity",
    "skill_groups_origin_uuids",
    "llm_classified_skills",
    "llm_job_attributes",
    "classifier_metadata",
    "location_keys",
    "llm_reranker_meta",
}


class TestBuildRankedJobDocument:
    def test_returns_none_when_ja_missing(self, make_settings):
        s = make_settings()
        ctx = _ctx()
        out = build_ranked_job_document(
            classified_doc={"_id": "doc-1"}, ctx=ctx, settings=s, ja=None, onet_work_activities=()
        )
        assert out is None

    def test_returns_pinned_top_level_keys(self, make_settings):
        s = make_settings()
        ctx = _ctx()
        out = build_ranked_job_document(
            classified_doc={"_id": "doc-1"},
            ctx=ctx,
            settings=s,
            ja=_ja_minimal(),
            onet_work_activities=(),
        )
        assert out is not None
        # Regression: any new top-level key needs an explicit test update.
        assert set(out.model_dump().keys()) == EXPECTED_TOP_LEVEL_KEYS
        assert out.job_id == "J-1"
        assert out.job_fingerprint == "fp-1"
        assert out.is_active is True
        assert out.source_classified_id == "doc-1"
        assert out.llm_reranker_meta.schema_version == PIPELINE_SCHEMA_VERSION
        assert out.location_keys == ["lusaka"]
        # Classified skills/occupations default to empty lists when no entities given.
        assert out.classified_skills == []
        assert out.classified_occupations == []
        # llm_classified_skills always has the skeleton, even with no skill_ranker.
        assert set(out.llm_classified_skills.model_dump().keys()) == {
            "essential",
            "optional",
            "discarded",
            "unmatched_essential",
            "unmatched_optional",
        }

    def test_carries_skill_ranker_uris_to_essential(self, make_settings):
        s = make_settings()
        ctx = _ctx()
        cand = EscoSkillCandidate(
            label="Python",
            uri="http://data.europa.eu/esco/skill/abc",
            esco_uuid="abc",
        )
        mapped = MappedEscoSkills(
            essential_uris=[cand.uri],
            optional_uris=[],
            essential_skills_uuids=["abc"],
            optional_skills_uuids=[],
        )
        out = build_ranked_job_document(
            classified_doc={"_id": "doc-1"},
            ctx=ctx,
            settings=s,
            ja=_ja_minimal(),
            onet_work_activities=(),
            skill_ranker=(["Python"], [], mapped, ""),
            ranker_candidates=[cand],
            classifier_skill_candidates=[cand],
        )
        assert out is not None
        ess = out.llm_classified_skills.essential
        assert len(ess) == 1
        assert ess[0].uri == cand.uri
        # Ranked uri is ranked, so discarded should be empty.
        assert out.llm_classified_skills.discarded == []
