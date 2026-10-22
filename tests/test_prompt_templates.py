"""The prompt templates in ``resources/prompts/`` and how their slots are filled."""

from __future__ import annotations

import re

import pytest

from app.job_ranking.prompts.render import render_prompt
from app.job_ranking.prompts.repository import PromptRepository, prompts_dir

pytestmark = pytest.mark.unit

# Every slot each template may use; the builders fill exactly these.
SLOTS = {
    "skill_ranker": {"full_details", "skills_text", "answer_language"},
    "job_attributes": {
        "country",
        "opening_line",
        "occupation_block",
        "job_listing",
        "listing_metadata",
        "attributes_menu",
        "work_activities_menu",
        "extra_field_instructions",
        "extra_field_json",
        "answer_language",
    },
    "occupation_skills_relevance": {
        "country",
        "job_listing",
        "listing_metadata",
        "skills_menu",
        "answer_language",
    },
    "answer_language": {"language"},
}


@pytest.mark.parametrize("name", sorted(SLOTS))
def test_template_renders_with_its_slots(make_settings, name):
    templates = PromptRepository().load(make_settings())
    template = getattr(templates, name)

    out = render_prompt(template, **{slot: f"<{slot}>" for slot in SLOTS[name]})

    # Every slot the builder supplies is used, and no placeholder is left behind.
    for slot in SLOTS[name]:
        assert f"<{slot}>" in out
    assert not re.search(r"(?<!\{)\{[a-z_]+\}(?!\})", out)


def test_unknown_slot_fails_loudly():
    with pytest.raises(KeyError, match="contry"):
        render_prompt("Jobs in {contry}", country="Kenya")


def test_values_are_inserted_verbatim():
    # Braces in job text reach the model unchanged instead of being read as slots.
    assert render_prompt("{job}", job="Cook {daily}") == "Cook {daily}"


def test_missing_template_fails_at_load(make_settings, tmp_path):
    s = make_settings(RESOURCES_DIR=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="skill_ranker.md"):
        PromptRepository().load(s)
    assert prompts_dir(s) == tmp_path / "prompts"
