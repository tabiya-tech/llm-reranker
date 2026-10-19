"""Unit tests for ``utils.classifier_linked_entities``.

This module is the highest-risk pure-logic surface in the reranker:
- LLM JSON parsing for the skill ranker (LLM1)
- ESCO concept-UUID extraction from URIs
- Classifier ``linked_entities`` extraction with min-similarity filtering and
  URI / label dedupe
- Fuzzy label-to-URI mapping (drives what gets stored as ``essential`` vs
  ``optional`` skills on every ranked document)

Tests pin the existing behavior; the implementation is the source of truth.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from app.job_context import JobDescriptionContext
from app.job_ranking.routes.types import RankRequest
from app.llm.errors import LlmParseError
from utils.classifier_linked_entities import (
    FUZZY_LABEL_MIN_RATIO,
    EscoSkillCandidate,
    LinkedEntityCandidate,
    SkillPromptEntry,
    _esco_concept_uuid_from_uri,
    _norm_label,
    extract_esco_skill_candidates,
    extract_linked_entity_candidates,
    full_details_from_context,
    group_linked_entities_by_type,
    job_identifier,
    linked_entity_preferred_taxonomy_id,
    map_skill_labels_to_esco_uris,
    normalize_linked_entity_type,
    parse_skill_ranker_json,
    skills_to_prompt_dicts,
)

# ---------------------------------------------------------------------------
# Small helpers / pure utilities
# ---------------------------------------------------------------------------


class TestNormLabel:
    def test_collapses_whitespace_and_lowercases(self):
        assert _norm_label("  Python   Programming  ") == "python programming"

    def test_empty_after_strip(self):
        assert _norm_label("   ") == ""


class TestNormalizeLinkedEntityType:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("skill", "skill"),
            ("Skill", "skill"),
            ("skills", "skill"),
            ("SKILLS", "skill"),
            ("occupation", "occupation"),
            ("Occupations", "occupation"),
            ("custom", "custom"),
        ],
    )
    def test_plural_normalization(self, raw, expected):
        assert normalize_linked_entity_type(raw) == expected


class TestEscoConceptUuidFromUri:
    def test_extracts_uuid_tail(self):
        u = "http://data.europa.eu/esco/skill/5247a6ce-6ed0-4fb3-b121-758921dcc44e"
        assert _esco_concept_uuid_from_uri(u) == "5247a6ce-6ed0-4fb3-b121-758921dcc44e"

    def test_strips_trailing_slash(self):
        u = "http://data.europa.eu/esco/skill/5247a6ce-6ed0-4fb3-b121-758921dcc44e/"
        assert _esco_concept_uuid_from_uri(u) == "5247a6ce-6ed0-4fb3-b121-758921dcc44e"

    def test_lowercases_uuid(self):
        u = "http://example/AAAAAAAA-1111-2222-3333-AAAAAAAAAAAA"
        assert _esco_concept_uuid_from_uri(u) == "aaaaaaaa-1111-2222-3333-aaaaaaaaaaaa"

    def test_returns_none_for_non_uuid_tail(self):
        assert _esco_concept_uuid_from_uri("http://example/skill/python") is None

    def test_returns_none_for_blank(self):
        assert _esco_concept_uuid_from_uri("") is None
        assert _esco_concept_uuid_from_uri("   ") is None


# ---------------------------------------------------------------------------
# Skill-ranker JSON parser (LLM1)
# ---------------------------------------------------------------------------


class TestParseSkillRankerJson:
    def test_happy_path(self):
        raw = '{"essential_skills": ["a", "b"], "optional_skills": ["c"]}'
        result = parse_skill_ranker_json(raw)
        assert result.essential_skills == ["a", "b"]
        assert result.optional_skills == ["c"]

    def test_handles_fence(self):
        raw = '```json\n{"essential_skills": ["a"], "optional_skills": []}\n```'
        result = parse_skill_ranker_json(raw)
        assert result.essential_skills == ["a"]
        assert result.optional_skills == []

    def test_filters_non_string_and_blank(self):
        raw = '{"essential_skills": ["a", "", 1, null, "b"], "optional_skills": ["  "]}'
        result = parse_skill_ranker_json(raw)
        assert result.essential_skills == ["a", "b"]
        assert result.optional_skills == []

    def test_top_level_must_be_object(self):
        with pytest.raises(LlmParseError, match="must be an object"):
            parse_skill_ranker_json('["a", "b"]')

    def test_lists_required(self):
        with pytest.raises(LlmParseError, match="must be lists"):
            parse_skill_ranker_json('{"essential_skills": "a", "optional_skills": []}')


# ---------------------------------------------------------------------------
# Linked-entity preferred-id helper
# ---------------------------------------------------------------------------


class TestLinkedEntityPreferredTaxonomyId:
    def test_returns_first_non_empty_known_key(self):
        le = {"esco_uuid": "  abc  ", "preferred_taxonomy_id": "ignored"}
        # Order from the implementation: linked_entity_preferred_taxonomy_id starts with
        # "linked_entity_preferred_taxonomy_id" then "preferred_taxonomy_id" before "esco_uuid".
        assert linked_entity_preferred_taxonomy_id(le) == "ignored"

    def test_uses_linked_entity_preferred_first(self):
        le = {"linked_entity_preferred_taxonomy_id": "winner", "esco_uuid": "later"}
        assert linked_entity_preferred_taxonomy_id(le) == "winner"

    def test_returns_none_when_all_empty(self):
        le = {"esco_uuid": "  ", "concept_id": ""}
        assert linked_entity_preferred_taxonomy_id(le) is None


# ---------------------------------------------------------------------------
# extract_linked_entity_candidates
#
# These tests use ``isolated_env`` to ensure ``Settings.from_env`` returns
# defaults that we control (no real ``.env`` leakage).
# ---------------------------------------------------------------------------

ESCO_URI = "http://data.europa.eu/esco/skill/5247a6ce-6ed0-4fb3-b121-758921dcc44e"
ESCO_URI_2 = "http://data.europa.eu/esco/skill/aaaaaaaa-1111-2222-3333-aaaaaaaaaaaa"
ESCO_URI_OCC = "http://data.europa.eu/esco/occupation/bbbbbbbb-2222-3333-4444-bbbbbbbbbbbb"


def _doc_with_entities(entities):
    return {"classification": {"entities": entities}}


class TestExtractLinkedEntityCandidates:
    def test_returns_empty_for_no_classification(self, make_settings):
        s = make_settings()
        assert extract_linked_entity_candidates({}, settings=s) == []

    def test_only_esco_taxonomy_kept(self, make_settings):
        s = make_settings()
        doc = _doc_with_entities(
            [
                {
                    "entity_type": "skill",
                    "linked_entities": [
                        {
                            "taxonomy": "ESCO",
                            "label": "Python",
                            "uri": ESCO_URI,
                            "similarity_score": 0.9,
                        },
                        {
                            "taxonomy": "other",
                            "label": "X",
                            "uri": "http://x",
                            "similarity_score": 0.95,
                        },
                    ],
                }
            ]
        )
        cands = extract_linked_entity_candidates(doc, settings=s)
        assert len(cands) == 1
        assert cands[0].label == "Python"
        assert cands[0].entity_type == "skill"
        assert cands[0].similarity_score == pytest.approx(0.9)
        assert cands[0].concept_uuid == "5247a6ce-6ed0-4fb3-b121-758921dcc44e"

    def test_drops_blank_label_or_uri(self, make_settings):
        s = make_settings()
        doc = _doc_with_entities(
            [
                {
                    "entity_type": "skill",
                    "linked_entities": [
                        {"taxonomy": "esco", "label": "", "uri": ESCO_URI},
                        {"taxonomy": "esco", "label": "Python", "uri": "  "},
                    ],
                }
            ]
        )
        assert extract_linked_entity_candidates(doc, settings=s) == []

    def test_filters_unwanted_entity_types(self, make_settings):
        # Default classified_linked_entity_types is ("skill", "occupation").
        s = make_settings()
        doc = _doc_with_entities(
            [
                {
                    "entity_type": "language",
                    "linked_entities": [
                        {"taxonomy": "esco", "label": "Swahili", "uri": ESCO_URI},
                    ],
                }
            ]
        )
        assert extract_linked_entity_candidates(doc, settings=s) == []

    def test_normalizes_plural_entity_type(self, make_settings):
        s = make_settings()
        doc = _doc_with_entities(
            [
                {
                    "entity_type": "Skills",
                    "linked_entities": [
                        {"taxonomy": "esco", "label": "Python", "uri": ESCO_URI},
                    ],
                }
            ]
        )
        cands = extract_linked_entity_candidates(doc, settings=s)
        assert len(cands) == 1
        assert cands[0].entity_type == "skill"

    def test_uri_dedupe_keeps_best_score(self, make_settings):
        s = make_settings()
        doc = _doc_with_entities(
            [
                {
                    "entity_type": "skill",
                    "linked_entities": [
                        {
                            "taxonomy": "esco",
                            "label": "Python",
                            "uri": ESCO_URI,
                            "similarity_score": 0.5,
                        },
                        {
                            "taxonomy": "esco",
                            "label": "PYTHON",
                            "uri": ESCO_URI,
                            "similarity_score": 0.9,
                        },
                    ],
                }
            ]
        )
        cands = extract_linked_entity_candidates(doc, settings=s)
        assert len(cands) == 1
        assert cands[0].similarity_score == pytest.approx(0.9)

    def test_min_similarity_filters_low_scores(self, make_settings):
        s = make_settings(CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY="0.7")
        doc = _doc_with_entities(
            [
                {
                    "entity_type": "skill",
                    "linked_entities": [
                        {
                            "taxonomy": "esco",
                            "label": "Low",
                            "uri": ESCO_URI,
                            "similarity_score": 0.5,
                        },
                        {
                            "taxonomy": "esco",
                            "label": "High",
                            "uri": ESCO_URI_2,
                            "similarity_score": 0.8,
                        },
                    ],
                }
            ]
        )
        cands = extract_linked_entity_candidates(doc, settings=s)
        assert [c.label for c in cands] == ["High"]

    def test_min_similarity_keeps_unscored_when_not_strict(self, make_settings):
        # default strict flag is False
        s = make_settings(CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY="0.7")
        doc = _doc_with_entities(
            [
                {
                    "entity_type": "skill",
                    "linked_entities": [
                        {"taxonomy": "esco", "label": "Unscored", "uri": ESCO_URI},
                    ],
                }
            ]
        )
        cands = extract_linked_entity_candidates(doc, settings=s)
        assert [c.label for c in cands] == ["Unscored"]

    def test_min_similarity_strict_drops_unscored(self, make_settings):
        s = make_settings(
            CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY="0.7",
            CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY_STRICT="1",
        )
        doc = _doc_with_entities(
            [
                {
                    "entity_type": "skill",
                    "linked_entities": [
                        {"taxonomy": "esco", "label": "Unscored", "uri": ESCO_URI},
                        {
                            "taxonomy": "esco",
                            "label": "Scored",
                            "uri": ESCO_URI_2,
                            "similarity_score": 0.9,
                        },
                    ],
                }
            ]
        )
        cands = extract_linked_entity_candidates(doc, settings=s)
        assert [c.label for c in cands] == ["Scored"]

    def test_label_dedupe_keeps_best_per_label(self, make_settings):
        s = make_settings(CLASSIFIED_LINKED_ENTITY_LABEL_DEDUPE_TYPES="skill")
        doc = _doc_with_entities(
            [
                {
                    "entity_type": "skill",
                    "linked_entities": [
                        # Same normalized label, two different URIs and scores
                        {
                            "taxonomy": "esco",
                            "label": "Python",
                            "uri": ESCO_URI,
                            "similarity_score": 0.6,
                        },
                        {
                            "taxonomy": "esco",
                            "label": "  python  ",
                            "uri": ESCO_URI_2,
                            "similarity_score": 0.95,
                        },
                    ],
                }
            ]
        )
        cands = extract_linked_entity_candidates(doc, settings=s)
        assert len(cands) == 1
        assert cands[0].uri == ESCO_URI_2
        assert cands[0].similarity_score == pytest.approx(0.95)


class TestGroupLinkedEntitiesByType:
    def test_groups_skill_and_occupation(self, make_settings):
        s = make_settings()
        doc = _doc_with_entities(
            [
                {
                    "entity_type": "skill",
                    "linked_entities": [
                        {"taxonomy": "esco", "label": "Python", "uri": ESCO_URI},
                    ],
                },
                {
                    "entity_type": "occupation",
                    "linked_entities": [
                        {"taxonomy": "esco", "label": "Engineer", "uri": ESCO_URI_OCC},
                    ],
                },
            ]
        )
        grouped = group_linked_entities_by_type(doc, s)
        assert set(grouped.keys()) == {"skill", "occupation"}
        assert len(grouped["skill"]) == 1
        assert len(grouped["occupation"]) == 1


class TestExtractEscoSkillCandidates:
    def test_returns_only_skill_entries(self, make_settings):
        s = make_settings()
        doc = _doc_with_entities(
            [
                {
                    "entity_type": "skill",
                    "linked_entities": [
                        {
                            "taxonomy": "esco",
                            "label": "Python",
                            "uri": ESCO_URI,
                            "similarity_score": 0.9,
                        },
                    ],
                },
                {
                    "entity_type": "occupation",
                    "linked_entities": [
                        {"taxonomy": "esco", "label": "Engineer", "uri": ESCO_URI_OCC},
                    ],
                },
            ]
        )
        out = extract_esco_skill_candidates(doc, s)
        assert len(out) == 1
        assert isinstance(out[0], EscoSkillCandidate)
        assert out[0].uri == ESCO_URI
        assert out[0].esco_uuid == "5247a6ce-6ed0-4fb3-b121-758921dcc44e"


# ---------------------------------------------------------------------------
# map_skill_labels_to_esco_uris (fuzzy mapping is core to LLM1 output)
# ---------------------------------------------------------------------------


class TestMapSkillLabelsToEscoUris:
    @staticmethod
    def _candidates():
        return [
            EscoSkillCandidate(
                label="Python programming",
                uri=ESCO_URI,
                esco_uuid="5247a6ce-6ed0-4fb3-b121-758921dcc44e",
            ),
            EscoSkillCandidate(
                label="Project management",
                uri=ESCO_URI_2,
                esco_uuid="aaaaaaaa-1111-2222-3333-aaaaaaaaaaaa",
            ),
        ]

    def test_no_candidates_returns_unmatched(self):
        result = map_skill_labels_to_esco_uris(["A"], ["B"], [])
        assert result.essential_uris == []
        assert result.optional_uris == []
        assert result.unmatched_essential == ["A"]
        assert result.unmatched_optional == ["B"]

    def test_exact_match_after_normalization(self):
        result = map_skill_labels_to_esco_uris(
            ["python programming"], ["Project Management"], self._candidates()
        )
        assert result.essential_uris == [ESCO_URI]
        assert result.optional_uris == [ESCO_URI_2]
        assert result.essential_skills_uuids == ["5247a6ce-6ed0-4fb3-b121-758921dcc44e"]
        assert result.optional_skills_uuids == ["aaaaaaaa-1111-2222-3333-aaaaaaaaaaaa"]
        assert result.unmatched_essential == []
        assert result.unmatched_optional == []

    def test_fuzzy_match_above_ratio(self):
        # "Python programing" is similar enough to "Python programming" (>=0.82).
        result = map_skill_labels_to_esco_uris(["Python programing"], [], self._candidates())
        assert result.essential_uris == [ESCO_URI]
        assert result.unmatched_essential == []
        assert FUZZY_LABEL_MIN_RATIO >= 0.8

    def test_unmatched_label_is_recorded(self):
        result = map_skill_labels_to_esco_uris(
            ["completely unrelated thing"], [], self._candidates()
        )
        assert result.essential_uris == []
        assert result.unmatched_essential == ["completely unrelated thing"]

    def test_dedupes_uri_across_essential_and_optional(self):
        # Same label resolved to the same URI in both lists; second occurrence dropped.
        result = map_skill_labels_to_esco_uris(
            ["Python programming"], ["python programming"], self._candidates()
        )
        assert result.essential_uris == [ESCO_URI]
        assert result.optional_uris == []  # dropped because uri already seen


# ---------------------------------------------------------------------------
# job_identifier
# ---------------------------------------------------------------------------


def _ctx(**overrides):
    base: dict[str, Any] = dict(
        job_id=None,
        job_fingerprint=None,
        title=None,
        employer=None,
        category=None,
        application_url=None,
        posted_date=None,
        closing_date=None,
        source_platform=None,
        raw_job_description=None,
        requirements_text=None,
        salary=None,
        city=None,
        county=None,
        employment_type=None,
        classified=RankRequest.model_validate({}),
    )
    base.update(overrides)
    return JobDescriptionContext(**base)


class TestJobIdentifier:
    def test_prefers_fingerprint(self):
        ctx = _ctx(
            job_fingerprint="fp-1", job_id=42, classified=RankRequest.model_validate({"_id": "z"})
        )
        assert job_identifier(ctx) == "fp-1"

    def test_falls_back_to_job_id(self):
        ctx = _ctx(job_id=42, classified=RankRequest.model_validate({"_id": "z"}))
        assert job_identifier(ctx) == "42"

    def test_falls_back_to_mongo_id(self):
        ctx = _ctx(classified=RankRequest.model_validate({"_id": "abc"}))
        assert job_identifier(ctx) == "abc"

    def test_empty_when_nothing_is_known(self):
        assert job_identifier(_ctx()) == ""


# ---------------------------------------------------------------------------
# skills_to_prompt_dicts
# ---------------------------------------------------------------------------


class TestSkillsToPromptDicts:
    def test_outputs_label_only_when_no_descriptions(self):
        cands = [EscoSkillCandidate(label="Python", uri=ESCO_URI, esco_uuid="abc")]
        out = skills_to_prompt_dicts(cands)
        assert out == [SkillPromptEntry(label="Python")]

    def test_attaches_description_via_esco_uuid(self):
        cands = [
            EscoSkillCandidate(
                label="Python",
                uri=ESCO_URI,
                esco_uuid="5247a6ce-6ed0-4fb3-b121-758921dcc44e",
            )
        ]
        descs = {"5247a6ce-6ed0-4fb3-b121-758921dcc44e": "Programming language"}
        out = skills_to_prompt_dicts(cands, description_by_esco=descs)
        assert out == [SkillPromptEntry(label="Python", description="Programming language")]

    def test_skips_missing_description(self):
        cands = [EscoSkillCandidate(label="Python", uri=ESCO_URI, esco_uuid="other")]
        out = skills_to_prompt_dicts(cands, description_by_esco={"different": "..."})
        assert out == [SkillPromptEntry(label="Python")]


# ---------------------------------------------------------------------------
# full_details_from_context
# ---------------------------------------------------------------------------


class TestFullDetailsFromContext:
    def test_no_fields_returns_placeholder(self):
        assert full_details_from_context(_ctx()) == "(no opportunity details)"

    def test_orders_and_labels_fields(self):
        ctx = _ctx(
            title="Engineer",
            employer="ACME",
            category="Tech",
            employment_type="Full time",
            city="Lusaka",
            county="Lusaka District",
            salary="$$$",
            raw_job_description="Build things.",
            requirements_text="3 years",
        )
        out = full_details_from_context(ctx)
        # Must include labels in order; double newlines separate sections.
        assert out.startswith("Title: Engineer")
        assert "Employer: ACME" in out
        assert "Category: Tech" in out
        assert "Employment type: Full time" in out
        assert "City: Lusaka" in out
        assert "Region: Lusaka District" in out
        assert "Salary: $$$" in out
        assert "Description:\nBuild things." in out
        assert "Requirements:\n3 years" in out
        # Sections separated by blank lines
        assert "\n\n" in out


# ---------------------------------------------------------------------------
# LinkedEntityCandidate dataclass invariants
# ---------------------------------------------------------------------------


class TestLinkedEntityCandidateDataclass:
    def test_is_frozen(self):
        c = LinkedEntityCandidate(entity_type="skill", label="Python", uri=ESCO_URI)
        with pytest.raises(ValidationError):
            c.label = "Other"  # type: ignore[misc]

    def test_default_score_and_uuid_are_none(self):
        c = LinkedEntityCandidate(entity_type="skill", label="Python", uri=ESCO_URI)
        assert c.similarity_score is None
        assert c.concept_uuid is None
