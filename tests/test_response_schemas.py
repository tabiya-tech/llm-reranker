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
from app.job_ranking.routes.types import OccupationSkill, OccupationSkillsBlock, SkillCounts
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
    return tuple(
        OnetWorkActivity(
            wa_element_id=f"4.A.{i}",
            wa_element_name=f"WA {i}",
            wa_element_name_simplified=f"WA {i}",
        )
        for i in range(5)
    )


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
        parsed = parse_skill_ranker_json(reply)
        assert (parsed.essential_skills, parsed.optional_skills) == (["Python"], ["Weld metal"])


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

    def test_extra_fields_only_when_configured(self):
        plain = job_attributes_response_schema(_attrs_schema(), _onet())
        assert "requires_post_secondary" not in plain["properties"]
        assert "requires_post_secondary" not in plain["required"]
        extra = job_attributes_response_schema(
            _attrs_schema(), _onet(), extra_fields=["requires_post_secondary", "needs_license"]
        )
        for name in ("requires_post_secondary", "needs_license"):
            assert extra["properties"][name] == {"type": "boolean"}
            assert name in extra["required"]

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


def _skill(skill_id: str, relation_type: str) -> OccupationSkill:
    return OccupationSkill(
        skill_id=skill_id,
        tabiya_skill_id=skill_id,
        label=skill_id,
        uri=f"http://data.europa.eu/esco/skill/{skill_id}",
        relation_type=relation_type,
    )


def _block(tabiya_occupation_id: str, skills: list[OccupationSkill]) -> OccupationSkillsBlock:
    return OccupationSkillsBlock(
        occupation_label="Occupation",
        occupation_esco_uuid=None,
        occupation_uri="http://data.europa.eu/esco/occupation/occ",
        tabiya_occupation_id=tabiya_occupation_id or None,
        skill_counts=SkillCounts(total=len(skills), essential=0, optional=0, unspecified=0),
        skills=skills,
    )


class TestOccupationSkillsRelevanceSchema:
    MENU = [
        _block(
            "occ-1",
            [
                _skill("s-1", "essential"),
                _skill("s-2", "optional"),
                _skill("s-3", "other"),
            ],
        ),
        _block("", [_skill("s-9", "essential")]),
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
