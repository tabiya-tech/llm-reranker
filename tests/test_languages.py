"""Language selection: registry, resource resolution, prompts, translation preference."""

import pytest

from app.languages import (
    LANGUAGE_BY_COUNTRY,
    LANGUAGE_REGISTRY,
    get_language_config,
    language_from_env,
    normalise_language,
)

pytestmark = pytest.mark.unit


class TestRegistry:
    def test_every_registered_language_has_a_config(self):
        for code in LANGUAGE_REGISTRY:
            cfg = get_language_config(code)
            assert cfg["language"] == code
            for key in ("name", "locales", "resources_subdir", "prompt_language"):
                assert cfg.get(key), f"{code} config is missing {key}"

    def test_unregistered_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            get_language_config("fr")

    def test_every_country_maps_to_a_registered_language(self):
        for country, language in LANGUAGE_BY_COUNTRY.items():
            assert language in LANGUAGE_REGISTRY, f"{country} maps to unregistered {language}"


class TestNormaliseLanguage:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("en", "en"),
            ("English", "en"),
            ("es", "es"),
            ("ES", "es"),
            ("AR-es", "es"),  # taxonomy export locale
            ("es_AR", "es"),
            ("es_AR.UTF-8", "es"),
            ("spanish", "es"),
            ("argentina", "es"),  # TARGET_COUNTRY spelling
        ],
    )
    def test_resolves_locale_spellings(self, value, expected):
        assert normalise_language(value) == expected

    def test_unknown_value_falls_back_instead_of_raising(self):
        # A bad value should not fail a batch that would otherwise run.
        assert normalise_language("klingon") == "en"
        assert normalise_language("klingon", fallback="es") == "es"


class TestLanguageFromEnv:
    def test_defaults_to_english(self, isolated_env):
        assert language_from_env() == "en"

    def test_reranker_language_wins_over_country(self, isolated_env):
        isolated_env.setenv("RERANKER_LANGUAGE", "en")
        assert language_from_env("argentina") == "en"

    def test_target_language_is_accepted(self, isolated_env):
        isolated_env.setenv("TARGET_LANGUAGE", "AR-es")
        assert language_from_env() == "es"

    def test_country_decides_when_no_language_is_set(self, isolated_env):
        # This is what the GCP Workflow relies on: it already sets TARGET_COUNTRY.
        assert language_from_env("argentina") == "es"
        assert language_from_env("kenya") == "en"

    def test_unmapped_country_is_english(self, isolated_env):
        assert language_from_env("atlantis") == "en"


class TestSettingsLanguage:
    def test_default_settings_are_english(self, make_settings):
        s = make_settings()
        assert s.language == "en"
        assert s.language_name == "English"
        assert s.language_resources_dir.name == "en"

    def test_spanish_settings_point_at_the_spanish_pack(self, make_settings):
        s = make_settings(RERANKER_LANGUAGE="es")
        assert s.language == "es"
        assert s.language_name == "Spanish"
        assert s.skills_csv_path.parent.name == "es"
        assert s.job_attributes_schema_path.parent.name == "es"

    def test_explicit_csv_override_still_wins(self, make_settings):
        # Deployments that pin a path must keep working.
        s = make_settings(RERANKER_LANGUAGE="es", SKILLS_CSV="en/skills.csv")
        assert s.skills_csv_path.parent.name == "en"

    def test_files_absent_from_a_pack_fall_back_to_english(self, make_settings):
        # The Spanish pack ships no ONET file; English's is used rather than failing.
        from app.job_ranking.job_attributes.repository import onet_work_activities_path

        s = make_settings(RERANKER_LANGUAGE="es")
        path = onet_work_activities_path(s)
        assert path.is_file()
        assert path.parent.name == "en"

    def test_concat_whitening_stays_language_neutral(self, make_settings):
        # It transforms embeddings, not text, so it lives at resources/ for every language.
        s = make_settings(RERANKER_LANGUAGE="es")
        assert s.reranker_concat_whitening_path.parent.name == "resources"


class TestPrompts:
    @pytest.mark.parametrize(
        ("env", "language"),
        [({"RERANKER_LANGUAGE": "es"}, "Spanish"), ({"TARGET_COUNTRY": "kenya"}, "English")],
    )
    def test_prompts_say_which_language_to_answer_in(self, make_settings, env, language):
        from app.job_context import JobDescriptionContext
        from app.job_ranking.occupation_skills.prompt import OccupationSkillsRelevancePromptBuilder
        from app.job_ranking.prompts.repository import PromptRepository
        from app.job_ranking.routes.types import RankRequest
        from app.job_ranking.skill_ranker.prompt import SkillRankerPromptBuilder

        s = make_settings(**env)
        templates = PromptRepository().load(s)
        ctx = JobDescriptionContext.model_validate(
            {
                **{f: None for f in JobDescriptionContext.model_fields if f != "classified"},
                "classified": RankRequest.model_validate({}),
            }
        )
        prompts = [
            SkillRankerPromptBuilder(templates, s).build("details", []),
            OccupationSkillsRelevancePromptBuilder(templates, s).build(ctx, []),
        ]
        for prompt in prompts:
            assert f"Answer in {language}." in prompt

    def test_every_language_shares_the_english_templates(self, make_settings):
        from app.job_ranking.prompts.repository import PromptRepository

        en = PromptRepository().load(make_settings(TARGET_COUNTRY="kenya"))
        es = PromptRepository().load(make_settings(RERANKER_LANGUAGE="es"))
        assert en == es
        assert en.job_attributes.startswith("You are an expert labor market analyst")


class TestTranslationPreference:
    def test_english_prefers_the_translation_overlay(self, make_settings):
        from app.translation import reranker_prefer_translation

        s = make_settings()
        assert reranker_prefer_translation(s) is True

    def test_spanish_reads_the_original_text(self, make_settings):
        # With Spanish prompts and a Spanish taxonomy there is no reason to prompt over
        # machine-translated English.
        from app.translation import reranker_prefer_translation

        s = make_settings(RERANKER_LANGUAGE="es")
        assert reranker_prefer_translation(s) is False

    def test_env_override_wins_for_any_language(self, make_settings):
        from app.translation import reranker_prefer_translation

        s = make_settings(RERANKER_LANGUAGE="es", RERANKER_PREFER_TRANSLATION="1")
        assert reranker_prefer_translation(s) is True

    def test_spanish_listing_text_is_not_replaced_by_the_english_translation(self, make_settings):
        from app.job_context import build_job_description_context

        s = make_settings(RERANKER_LANGUAGE="es")
        doc = {
            "title": "Cocinero",
            "description": "Se busca cocinero con experiencia en cocina.",
            "translation": {
                "text_en": "Cook wanted with kitchen experience.",
                "fields_en": {"title": "Cook"},
            },
        }
        ctx = build_job_description_context(doc, settings=s)
        assert ctx.title == "Cocinero"
        assert ctx.raw_job_description == "Se busca cocinero con experiencia en cocina."

    def test_english_run_still_uses_the_translation(self, make_settings):
        from app.job_context import build_job_description_context

        s = make_settings()
        doc = {
            "title": "Cocinero",
            "description": "Se busca cocinero con experiencia en cocina.",
            "translation": {
                "text_en": "Cook wanted with kitchen experience.",
                "fields_en": {"title": "Cook"},
            },
        }
        ctx = build_job_description_context(doc, settings=s)
        assert ctx.title == "Cook"
        assert ctx.raw_job_description == "Cook wanted with kitchen experience."
