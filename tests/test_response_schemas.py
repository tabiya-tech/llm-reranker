"""Tests for the per-pass structured-output schemas (``app.job_ranking.response_schemas``)."""

from __future__ import annotations

import json

import pytest

from app.job_ranking.job_attributes.parser import parse_job_attributes_response
from app.job_ranking.job_attributes.repository import JobAttributesRepository
from app.job_ranking.job_attributes.types import (
    AttributeDefinition,
    AttributeLevel,
    JobAttributesSchema,
    OnetWorkActivity,
)
from app.job_ranking.occupation_skills.parser import parse_occupation_skills_relevance_response
from app.job_ranking.response_schemas import (
    MAX_ENUM_VALUES,
    job_attributes_response_schema,
    occupation_skills_relevance_response_schema,
    skill_ranker_response_schema,
)
from utils.classifier_linked_entities import parse_skill_ranker_json

pytestmark = pytest.mark.unit


def _attrs_schema() -> JobAttributesSchema:
    return JobAttributesSchema(
        attributes=(
            AttributeDefinition(
                name="earnings_per_month",
                label="Earnings",
                group="g",
                type="ordinal",
                coding="c",
                levels=(
                    AttributeLevel(id="earn_low", label="Low"),
                    AttributeLevel(id="earn_high", label="High"),
                ),
            ),
            AttributeDefinition(
                name="shift",
                label="Shift",
                group="g",
                type="nominal",
                coding="c",
                levels=(
                    AttributeLevel(id="day", label="Day"),
                    AttributeLevel(id="night", label="Night"),
                ),
            ),
        )
    )


def _onet() -> tuple[OnetWorkActivity, ...]:
    return tuple(OnetWorkActivity(f"4.A.{i}", f"WA {i}", f"WA {i}") for i in range(5))


class TestSkillRankerSchema:
    def test_labels_are_an_enum_in_menu_order_without_duplicates(self):
        schema = skill_ranker_response_schema(["Python", "Weld metal", "Python", ""])
        for key in ("essential_skills", "optional_skills"):
            assert schema["properties"][key]["items"] == {
                "type": "string",
                "enum": ["Python", "Weld metal"],
            }
        assert schema["required"] == ["essential_skills", "optional_skills"]

    def test_oversized_menu_falls_back_to_a_plain_string(self):
        labels = [f"skill {i}" for i in range(MAX_ENUM_VALUES + 1)]
        items = skill_ranker_response_schema(labels)["properties"]["essential_skills"]["items"]
        assert items == {"type": "string"}

    def test_empty_menu_is_a_plain_string(self):
        items = skill_ranker_response_schema([])["properties"]["optional_skills"]["items"]
        assert items == {"type": "string"}

    def test_a_conforming_reply_parses(self):
        reply = json.dumps({"essential_skills": ["Python"], "optional_skills": ["Weld metal"]})
        assert parse_skill_ranker_json(reply) == (["Python"], ["Weld metal"])


class TestJobAttributesSchema:
    def test_names_levels_and_work_activities_are_enums(self):
        schema = job_attributes_response_schema(_attrs_schema(), _onet())
        item = schema["properties"]["attributes"]["items"]["properties"]
        assert item["attribute_name"]["enum"] == ["earnings_per_month", "shift"]
        assert item["selected_level_id"]["enum"] == ["earn_low", "earn_high", "day", "night"]
        assert item["confidence"]["enum"] == ["high", "medium", "low"]
        wa = schema["properties"]["work_activities"]
        assert wa["items"]["enum"] == [f"4.A.{i}" for i in range(5)]
        assert (wa["minItems"], wa["maxItems"]) == (3, 10)

    def test_one_estimate_per_attribute(self):
        attrs = job_attributes_response_schema(_attrs_schema(), _onet())["properties"]["attributes"]
        assert (attrs["minItems"], attrs["maxItems"]) == (2, 2)

    def test_kenya_flag_only_when_asked_for(self):
        plain = job_attributes_response_schema(_attrs_schema(), _onet())
        assert "requires_post_secondary" not in plain["properties"]
        assert "requires_post_secondary" not in plain["required"]
        kenya = job_attributes_response_schema(
            _attrs_schema(), _onet(), requires_post_secondary=True
        )
        assert kenya["properties"]["requires_post_secondary"] == {"type": "boolean"}
        assert "requires_post_secondary" in kenya["required"]

    def test_a_conforming_reply_parses(self, make_settings):
        reply = json.dumps(
            {
                "attributes": [
                    {
                        "attribute_name": "earnings_per_month",
                        "selected_level_id": "earn_high",
                        "confidence": "high",
                        "reasoning": "r",
                    },
                    {
                        "attribute_name": "shift",
                        "selected_level_id": "night",
                        "confidence": "low",
                        "reasoning": "r",
                    },
                ],
                "summary": "s",
                "work_activities": ["4.A.0", "4.A.1", "4.A.2"],
            }
        )
        out = parse_job_attributes_response(
            reply, _attrs_schema(), _onet(), settings=make_settings(TARGET_COUNTRY="zambia")
        )
        assert out.attributes == {"earnings_per_month": "earn_high", "shift": "night"}
        assert out.work_activity_ids == ("4.A.0", "4.A.1", "4.A.2")

    def test_shipped_language_packs_build_a_schema(self, make_settings):
        for language in ("en", "es"):
            s = make_settings(RERANKER_LANGUAGE=language)
            schema = job_attributes_response_schema(
                JobAttributesRepository().load_schema(s),
                JobAttributesRepository().load_onet_work_activities(s),
            )
            item = schema["properties"]["attributes"]["items"]["properties"]
            assert item["attribute_name"]["enum"]
            assert schema["properties"]["work_activities"]["items"]["enum"]


class TestOccupationSkillsRelevanceSchema:
    MENU = [
        {
            "tabiya_occupation_id": "occ-1",
            "skills": [
                {"skill_id": "s-1", "relation_type": "essential"},
                {"skill_id": "s-2", "relation_type": "optional"},
                {"skill_id": "s-3", "relation_type": "other"},
            ],
        },
        {"tabiya_occupation_id": "", "skills": [{"skill_id": "s-9", "relation_type": "essential"}]},
    ]

    def test_ids_are_enums_over_the_judged_skills_only(self):
        item = occupation_skills_relevance_response_schema(self.MENU)["properties"]["decisions"][
            "items"
        ]
        assert item["properties"]["skill_id"]["enum"] == ["s-1", "s-2"]
        assert item["properties"]["tabiya_occupation_id"]["enum"] == ["occ-1"]
        assert item["properties"]["is_relevant"] == {"type": "boolean"}

    def test_a_conforming_reply_parses(self):
        reply = json.dumps(
            {
                "decisions": [
                    {"skill_id": "s-1", "tabiya_occupation_id": "occ-1", "is_relevant": True}
                ]
            }
        )
        (d,) = parse_occupation_skills_relevance_response(reply).decisions
        assert (d.skill_id, d.tabiya_occupation_id, d.is_relevant) == ("s-1", "occ-1", True)
