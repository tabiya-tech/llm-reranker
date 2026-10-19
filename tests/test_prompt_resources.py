"""Prompt templates under ``resources/<language>/prompts/``.

Keeps the packs complete (every template resolves for every language and every country
variant), and pins the behaviour moved out of ``app/prompts.py``: country blocks come from
settings, and the prompt version lands next to the model in ``llm_reranker_meta``.
"""

import pytest

from app.config import _JOB_ATTRIBUTES_COUNTRY_FIELDS, repo_root
from app.languages import LANGUAGE_REGISTRY, get_language_config
from app.prompt_resources import (
    PROMPT_TEMPLATES,
    PROMPTS_SUBDIR,
    VERSION_FILE,
    country_field_template,
    format_prompt_template,
    prompt_provenance,
    resolve_prompt_file,
)

pytestmark = pytest.mark.unit

RESOURCES = repo_root() / "resources"


def _pack_prompts_dir(language: str):
    subdir = get_language_config(language).get("resources_subdir") or language
    return RESOURCES / subdir / PROMPTS_SUBDIR


class TestPacks:
    @pytest.mark.parametrize("name", PROMPT_TEMPLATES)
    def test_english_ships_every_template(self, name):
        # English is the fallback for every other language, so it must be complete itself.
        assert (_pack_prompts_dir("en") / name).is_file()

    @pytest.mark.parametrize("language", LANGUAGE_REGISTRY)
    def test_every_pack_versions_its_prompts(self, language):
        # Without its own VERSION a pack would report English's version for its own text.
        version = _pack_prompts_dir(language) / VERSION_FILE
        assert version.is_file()
        assert version.read_text(encoding="utf-8").strip()

    @pytest.mark.parametrize(
        ("language", "country", "field"),
        [
            (language, country, field)
            for language in LANGUAGE_REGISTRY
            for country, fields in _JOB_ATTRIBUTES_COUNTRY_FIELDS.items()
            for field in fields
        ],
    )
    def test_every_country_field_has_a_prompt_block(self, make_settings, language, country, field):
        s = make_settings(RERANKER_LANGUAGE=language, TARGET_COUNTRY=country)
        assert resolve_prompt_file(s, country_field_template(country, field)).is_file()

    def test_missing_template_fails_loudly(self, make_settings):
        s = make_settings()
        with pytest.raises(FileNotFoundError, match="no_such_prompt.md"):
            resolve_prompt_file(s, "no_such_prompt.md")


class TestFormatting:
    def test_unknown_placeholder_is_left_alone(self):
        assert format_prompt_template("{country} {other}", {"country": "Kenya"}) == "Kenya {other}"

    def test_doubled_braces_are_literal(self):
        assert format_prompt_template('{{"k": "{v}"}}', {"v": "x"}) == '{"k": "x"}'

    def test_stray_brace_does_not_raise(self):
        assert format_prompt_template("{country} {", {"country": "Kenya"}) == "Kenya {"


class TestCountryFields:
    def _llm2_prompt(self, s):
        from app.prompts import OccupationPromptEnrichment, build_job_attributes_prompt
        from app.schemas import load_job_attributes_schema, load_onet_work_activities

        return build_job_attributes_prompt(
            opening_line="Estimate:",
            occupation=OccupationPromptEnrichment("Cook", None, None, None, None),
            schema=load_job_attributes_schema(settings=s),
            onet_work_activities=load_onet_work_activities(settings=s)[:3],
            settings=s,
        )

    def test_kenya_asks_for_requires_post_secondary(self, make_settings):
        s = make_settings(TARGET_COUNTRY="kenya")
        assert s.job_attributes_country_fields == ("requires_post_secondary",)
        prompt = self._llm2_prompt(s)
        assert "education **beyond Kenya secondary school**" in prompt
        assert '"requires_post_secondary": true' in prompt

    def test_other_countries_do_not(self, make_settings):
        s = make_settings(TARGET_COUNTRY="zambia")
        assert s.job_attributes_country_fields == ()
        prompt = self._llm2_prompt(s)
        assert "requires_post_secondary" not in prompt
        assert "{country_field" not in prompt


class TestProvenance:
    def test_version_names_the_language_pack(self, make_settings):
        assert prompt_provenance(make_settings())["prompt_version"].startswith("en/")
        es = make_settings(RERANKER_LANGUAGE="es")
        assert prompt_provenance(es)["prompt_version"].startswith("es/")

    def test_digest_covers_country_blocks(self, make_settings):
        kenya = prompt_provenance(make_settings(TARGET_COUNTRY="kenya"))
        zambia = prompt_provenance(make_settings(TARGET_COUNTRY="zambia"))
        assert kenya["prompt_version"] == zambia["prompt_version"]
        assert kenya["prompt_digest"] != zambia["prompt_digest"]

    def test_recorded_alongside_the_model(self, make_settings):
        from app.ranked_job_schema import build_llm_reranker_meta

        s = make_settings(GEMINI_API_KEY="dummy", TARGET_COUNTRY="kenya")
        meta = build_llm_reranker_meta(s)
        assert meta["job_attributes_model"] == s.llm.model
        assert meta["prompt_version"] == prompt_provenance(s)["prompt_version"]
        assert len(meta["prompt_digest"]) == 12
