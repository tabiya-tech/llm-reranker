"""Unit tests for ``app.schemas``: LLM2 / LLM3 JSON parsers and resource loaders.

The parsers are the second-most critical surface in the pipeline (after
classifier-linked-entity extraction): every stored ranked-job document
runs through ``parse_job_attributes_response``, and every LLM3 call's
output runs through ``parse_occupation_skills_relevance_response``.
"""

from __future__ import annotations

import json

import pytest

from app.schemas import (
    AttributeDefinition,
    AttributeLevel,
    JobAttributesResult,
    JobAttributesSchema,
    OccupationSkillsRelevanceDecision,
    OccupationSkillsRelevanceResult,
    OnetWorkActivity,
    _json_bool,
    _normalize_confidence,
    allowed_level_ids_by_attribute,
    job_attributes_schema_path,
    load_job_attributes_schema,
    load_onet_work_activities,
    onet_work_activities_path,
    parse_job_attributes_response,
    parse_occupation_skills_relevance_response,
    parse_requires_post_secondary,
)


# ---------------------------------------------------------------------------
# Resource paths and loaders (use the real, committed resources)
# ---------------------------------------------------------------------------


class TestResourceLoaders:
    def test_paths_resolve_under_resources_dir(self, make_settings):
        s = make_settings()
        assert job_attributes_schema_path(s).is_file()
        assert onet_work_activities_path(s).is_file()
        # Default basename matches what app.config resolves to.
        assert job_attributes_schema_path(s).name == "job_attributes_reduced.json"

    def test_load_job_attributes_schema_returns_typed_definitions(self, make_settings):
        s = make_settings()
        schema = load_job_attributes_schema(settings=s)
        assert isinstance(schema, JobAttributesSchema)
        assert len(schema.attributes) > 0
        for a in schema.attributes:
            assert isinstance(a, AttributeDefinition)
            assert a.name and a.label
            assert isinstance(a.levels, tuple) and len(a.levels) > 0
            for lev in a.levels:
                assert isinstance(lev, AttributeLevel)
                assert lev.id and lev.label

    def test_load_onet_work_activities_returns_tuple(self, make_settings):
        s = make_settings()
        wa = load_onet_work_activities(settings=s)
        assert isinstance(wa, tuple)
        assert len(wa) > 0
        for x in wa:
            assert isinstance(x, OnetWorkActivity)
            assert x.wa_element_id  # non-empty

    def test_load_job_attributes_schema_missing_file_raises(self, tmp_path, make_settings):
        # Steer the loader at a path that does not exist by pointing JOB_ATTRIBUTES_SCHEMA
        # to a non-existent file under a temporary RESOURCES_DIR. The resolver still
        # resolves to that absolute path because it is absolute.
        bogus = tmp_path / "missing.json"
        s = make_settings(JOB_ATTRIBUTES_SCHEMA=str(bogus))
        with pytest.raises(FileNotFoundError):
            load_job_attributes_schema(settings=s)

    def test_load_onet_work_activities_missing_file_raises(self, tmp_path, make_settings):
        bogus_resources = tmp_path / "no_resources"
        bogus_resources.mkdir()
        s = make_settings(RESOURCES_DIR=str(bogus_resources))
        with pytest.raises(FileNotFoundError):
            load_onet_work_activities(settings=s)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestNormalizeConfidence:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("high", "high"),
            ("HIGH", "high"),
            ("  Medium ", "medium"),
            ("low", "low"),
            ("", "medium"),
            (None, "medium"),
            ("bogus", "medium"),
        ],
    )
    def test_values(self, raw, expected):
        assert _normalize_confidence(raw) == expected


class TestJsonBool:
    @pytest.mark.parametrize(
        "raw, default, expected",
        [
            (True, False, True),
            (False, True, False),
            ("true", False, True),
            ("YES", False, True),
            ("1", False, True),
            ("false", True, False),
            ("NO", True, False),
            ("0", True, False),
            ("garbage", True, True),
            ("garbage", False, False),
            (None, True, True),
        ],
    )
    def test_bool_parsing(self, raw, default, expected):
        assert _json_bool(raw, default) == expected


class TestParseRequiresPostSecondary:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (None, False),
            (True, True),
            (False, False),
            ("yes", True),
            ("no", False),
            ("garbage", False),
        ],
    )
    def test_kenya_eligibility_flag(self, raw, expected):
        assert parse_requires_post_secondary(raw) == expected


class TestAllowedLevelIdsByAttribute:
    def test_returns_frozenset_per_attribute(self, make_settings):
        s = make_settings()
        schema = load_job_attributes_schema(settings=s)
        allowed = allowed_level_ids_by_attribute(schema)
        for a in schema.attributes:
            assert a.name in allowed
            assert allowed[a.name] == frozenset(lev.id for lev in a.levels)


# ---------------------------------------------------------------------------
# Job attributes parser (LLM2)
# ---------------------------------------------------------------------------


def _build_minimal_schema() -> JobAttributesSchema:
    """A small, deterministic schema for parser tests (independent of resources)."""
    return JobAttributesSchema(
        attributes=(
            AttributeDefinition(
                name="earnings_per_month",
                label="Earnings",
                group="Financial",
                type="ordered",
                coding="linear",
                levels=(
                    AttributeLevel(id="earn_low", label="Low"),
                    AttributeLevel(id="earn_high", label="High"),
                ),
            ),
            AttributeDefinition(
                name="career_growth",
                label="Growth",
                group="Future",
                type="ordered",
                coding="dummy",
                levels=(
                    AttributeLevel(id="growth_low", label="Low"),
                    AttributeLevel(id="growth_high", label="High"),
                ),
            ),
        )
    )


_ONET_FIXTURE = (
    OnetWorkActivity(
        wa_element_id="4.A.1.a.1",
        wa_element_name="Activity One",
        wa_element_name_simplified="One",
    ),
    OnetWorkActivity(
        wa_element_id="4.A.1.a.2",
        wa_element_name="Activity Two",
        wa_element_name_simplified="Two",
    ),
)


class TestParseJobAttributesResponse:
    def test_list_form_happy_path(self, make_settings):
        s = make_settings()
        schema = _build_minimal_schema()
        raw = json.dumps(
            {
                "attributes": [
                    {
                        "attribute_name": "earnings_per_month",
                        "selected_level_id": "earn_high",
                        "confidence": "high",
                        "reasoning": "context says so",
                    },
                    {
                        "attribute_name": "career_growth",
                        "selected_level_id": "growth_low",
                        "confidence": "Medium",
                        "reasoning": "",
                    },
                ],
                "summary": "  ok ",
                "work_activities": ["4.A.1.a.1"],
            }
        )
        result = parse_job_attributes_response(raw, schema, _ONET_FIXTURE, settings=s)
        assert isinstance(result, JobAttributesResult)
        assert {e.attribute_name for e in result.attribute_estimates} == {
            "earnings_per_month",
            "career_growth",
        }
        # Order follows schema definition, not input order.
        assert [e.attribute_name for e in result.attribute_estimates] == [
            "earnings_per_month",
            "career_growth",
        ]
        assert result.attributes["earnings_per_month"] == "earn_high"
        assert result.attributes["career_growth"] == "growth_low"
        assert result.summary == "ok"
        assert result.work_activity_ids == ("4.A.1.a.1",)
        # Confidence normalization
        confs = {e.attribute_name: e.confidence for e in result.attribute_estimates}
        assert confs["career_growth"] == "medium"
        assert confs["earnings_per_month"] == "high"

    def test_legacy_dict_form_synthesizes_medium(self, make_settings):
        s = make_settings()
        schema = _build_minimal_schema()
        raw = json.dumps(
            {
                "attributes": {
                    "earnings_per_month": "earn_low",
                    "career_growth": "growth_high",
                }
            }
        )
        result = parse_job_attributes_response(raw, schema, _ONET_FIXTURE, settings=s)
        assert all(e.confidence == "medium" for e in result.attribute_estimates)
        assert all(e.reasoning == "" for e in result.attribute_estimates)
        assert result.attributes["earnings_per_month"] == "earn_low"

    def test_unknown_level_is_filtered(self, make_settings):
        s = make_settings()
        schema = _build_minimal_schema()
        raw = json.dumps(
            {
                "attributes": [
                    {
                        "attribute_name": "earnings_per_month",
                        "selected_level_id": "earn_low",
                    },
                    {
                        "attribute_name": "earnings_per_month",  # duplicate name, last wins
                        "selected_level_id": "not_a_level",
                    },
                    {
                        "attribute_name": "unknown_attribute",
                        "selected_level_id": "earn_low",
                    },
                ]
            }
        )
        result = parse_job_attributes_response(raw, schema, _ONET_FIXTURE, settings=s)
        # Only the valid earnings_per_month estimate survives.
        assert len(result.attribute_estimates) == 1
        assert result.attribute_estimates[0].attribute_name == "earnings_per_month"
        assert result.attribute_estimates[0].selected_level_id == "earn_low"

    def test_work_activities_filter_and_dedupe(self, make_settings):
        s = make_settings()
        schema = _build_minimal_schema()
        raw = json.dumps(
            {
                "attributes": [],
                "work_activities": [
                    "4.A.1.a.1",
                    "4.A.1.a.1",  # duplicate
                    "4.A.1.a.2",
                    "unknown.id",  # filtered out
                    None,  # ignored
                ],
            }
        )
        result = parse_job_attributes_response(raw, schema, _ONET_FIXTURE, settings=s)
        assert result.work_activity_ids == ("4.A.1.a.1", "4.A.1.a.2")

    def test_summary_optional_and_trimmed(self, make_settings):
        s = make_settings()
        schema = _build_minimal_schema()
        raw1 = json.dumps({"attributes": []})
        raw2 = json.dumps({"attributes": [], "summary": "   trimmed   "})
        assert parse_job_attributes_response(raw1, schema, _ONET_FIXTURE, settings=s).summary == ""
        assert (
            parse_job_attributes_response(raw2, schema, _ONET_FIXTURE, settings=s).summary
            == "trimmed"
        )

    def test_top_level_must_be_object(self, make_settings):
        s = make_settings()
        schema = _build_minimal_schema()
        with pytest.raises(ValueError, match="must be an object"):
            parse_job_attributes_response("[]", schema, _ONET_FIXTURE, settings=s)

    def test_kenya_requires_post_secondary_attached(self, make_settings):
        s = make_settings(TARGET_COUNTRY="kenya")
        schema = _build_minimal_schema()
        raw = json.dumps({"attributes": [], "requires_post_secondary": True})
        result = parse_job_attributes_response(raw, schema, _ONET_FIXTURE, settings=s)
        assert result.attributes.get("requires_post_secondary") is True

    def test_kenya_requires_post_secondary_default_false(self, make_settings):
        s = make_settings(TARGET_COUNTRY="kenya")
        schema = _build_minimal_schema()
        raw = json.dumps({"attributes": []})
        result = parse_job_attributes_response(raw, schema, _ONET_FIXTURE, settings=s)
        assert result.attributes.get("requires_post_secondary") is False

    def test_zambia_does_not_attach_kenya_field(self, make_settings):
        s = make_settings(TARGET_COUNTRY="zambia")
        schema = _build_minimal_schema()
        raw = json.dumps({"attributes": [], "requires_post_secondary": True})
        result = parse_job_attributes_response(raw, schema, _ONET_FIXTURE, settings=s)
        assert "requires_post_secondary" not in result.attributes

    def test_handles_fenced_response(self, make_settings):
        s = make_settings()
        schema = _build_minimal_schema()
        raw = "```json\n" + json.dumps({"attributes": [], "summary": "fenced"}) + "\n```"
        result = parse_job_attributes_response(raw, schema, _ONET_FIXTURE, settings=s)
        assert result.summary == "fenced"


# ---------------------------------------------------------------------------
# Occupation-skills relevance parser (LLM3)
# ---------------------------------------------------------------------------


class TestParseOccupationSkillsRelevanceResponse:
    def test_happy_path(self):
        raw = json.dumps(
            {
                "decisions": [
                    {
                        "skill_id": "s1",
                        "tabiya_occupation_id": "t1",
                        "is_relevant": True,
                    },
                    {
                        "skill_id": "s2",
                        "tabiya_occupation_id": "t1",
                        "is_relevant": False,
                    },
                ]
            }
        )
        result = parse_occupation_skills_relevance_response(raw)
        assert isinstance(result, OccupationSkillsRelevanceResult)
        assert len(result.decisions) == 2
        for d in result.decisions:
            assert isinstance(d, OccupationSkillsRelevanceDecision)
        assert {d.skill_id for d in result.decisions} == {"s1", "s2"}

    def test_dedupe_last_wins_case_insensitive_keys(self):
        raw = json.dumps(
            {
                "decisions": [
                    {"skill_id": "S1", "tabiya_occupation_id": "T1", "is_relevant": True},
                    {"skill_id": "s1", "tabiya_occupation_id": "t1", "is_relevant": False},
                ]
            }
        )
        result = parse_occupation_skills_relevance_response(raw)
        # Same (tid.lower(), sid.lower()) → last wins
        assert len(result.decisions) == 1
        assert result.decisions[0].is_relevant is False
        # The original casing of skill_id and tabiya_occupation_id is preserved
        # from the winning row, not lowercased.
        assert result.decisions[0].skill_id == "s1"
        assert result.decisions[0].tabiya_occupation_id == "t1"

    def test_skips_rows_missing_ids(self):
        raw = json.dumps(
            {
                "decisions": [
                    {"skill_id": "", "tabiya_occupation_id": "t1", "is_relevant": True},
                    {"skill_id": "s1", "tabiya_occupation_id": "  ", "is_relevant": True},
                    {"skill_id": "s1", "tabiya_occupation_id": "t1", "is_relevant": True},
                ]
            }
        )
        result = parse_occupation_skills_relevance_response(raw)
        assert len(result.decisions) == 1

    def test_top_level_must_be_object(self):
        with pytest.raises(ValueError, match="must be an object"):
            parse_occupation_skills_relevance_response("[]")

    def test_decisions_must_be_array(self):
        with pytest.raises(ValueError, match="decisions array"):
            parse_occupation_skills_relevance_response('{"decisions": {}}')

    def test_handles_fenced_response(self):
        raw = "```json\n{\"decisions\": []}\n```"
        result = parse_occupation_skills_relevance_response(raw)
        assert result.decisions == ()

    def test_is_relevant_default_when_missing(self):
        # _json_bool default is True
        raw = json.dumps({"decisions": [{"skill_id": "s1", "tabiya_occupation_id": "t1"}]})
        result = parse_occupation_skills_relevance_response(raw)
        assert result.decisions[0].is_relevant is True
