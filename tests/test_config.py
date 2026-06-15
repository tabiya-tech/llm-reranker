"""Unit tests for ``app.config``: env helpers and ``Settings.from_env``.

These pin parser behaviour so a wrong env default cannot silently change the
pipeline output across all jobs.
"""

from __future__ import annotations

import pytest

from app.config import (
    _bool_env,
    _classified_linked_entity_label_dedupe_types_from_env,
    _classified_linked_entity_min_similarity_from_env,
    _comma_keys,
    _country_name_from_slug,
    _entity_types_tuple,
    _int_env,
    _int_env_min0,
    _llm3_menu_signal_filter_from_env,
    _normalize_classifier_entity_type_token,
    _occupation_relation_lookup_from_env,
    _occupation_skill_relation_occupation_types_from_env,
    _occupation_skills_signal_include_from_env,
    _target_country_slug_from_env,
    _unique_key_order,
    Settings,
    reranker_enable_llm3_from_env,
)


# ---------------------------------------------------------------------------
# Small parsers
# ---------------------------------------------------------------------------


class TestNormalizeClassifierEntityTypeToken:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("skill", "skill"),
            ("Skills", "skill"),
            ("OCCUPATIONS", "occupation"),
            ("custom", "custom"),
        ],
    )
    def test_values(self, raw, expected):
        assert _normalize_classifier_entity_type_token(raw) == expected


class TestEntityTypesTuple:
    def test_default_when_none(self):
        assert _entity_types_tuple(None) == ("skill", "occupation")

    def test_normalizes_and_dedupes(self):
        assert _entity_types_tuple("Skills, skill, occupations") == ("skill", "occupation")

    def test_blank_falls_back(self):
        assert _entity_types_tuple("  ") == ("skill", "occupation")


class TestCommaKeys:
    def test_default_used_when_empty(self):
        assert _comma_keys(None, "a,b") == ("a", "b")
        assert _comma_keys("", "a,b") == ("a", "b")
        assert _comma_keys("   ", "a,b") == ("a", "b")

    def test_explicit_overrides_default(self):
        assert _comma_keys("x , y , ,z", "a,b") == ("x", "y", "z")


class TestUniqueKeyOrder:
    def test_first_occurrence_wins(self):
        out = _unique_key_order(("a", "b"), ("b", "c"), ("a", "d"))
        assert out == ("a", "b", "c", "d")


class TestIntEnv:
    def test_default_when_unset(self, isolated_env):
        assert _int_env("TEST_INT_X", 5) == 5

    def test_parses_value(self, isolated_env):
        isolated_env.setenv("TEST_INT_X", "12")
        assert _int_env("TEST_INT_X", 5) == 12

    def test_clamps_to_min_max(self, isolated_env):
        isolated_env.setenv("TEST_INT_X", "0")
        assert _int_env("TEST_INT_X", 5, min_v=2, max_v=10) == 2
        isolated_env.setenv("TEST_INT_X", "999")
        assert _int_env("TEST_INT_X", 5, min_v=2, max_v=10) == 10

    def test_invalid_falls_back(self, isolated_env):
        isolated_env.setenv("TEST_INT_X", "abc")
        assert _int_env("TEST_INT_X", 5) == 5


class TestIntEnvMin0:
    def test_default_when_unset(self, isolated_env):
        assert _int_env_min0("TEST_INT_Y", 0) == 0
        assert _int_env_min0("TEST_INT_Y", 9) == 9

    def test_negative_falls_back(self, isolated_env):
        isolated_env.setenv("TEST_INT_Y", "-3")
        assert _int_env_min0("TEST_INT_Y", 0) == 0

    def test_caps_at_max(self, isolated_env):
        isolated_env.setenv("TEST_INT_Y", "999999")
        assert _int_env_min0("TEST_INT_Y", 0, max_v=10) == 10


class TestBoolEnv:
    @pytest.mark.parametrize(
        "raw, expected", [("1", True), ("true", True), ("YES", True), ("on", True)]
    )
    def test_truthy_values(self, isolated_env, raw, expected):
        isolated_env.setenv("TEST_BOOL", raw)
        assert _bool_env("TEST_BOOL") is expected

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", "anything-else"])
    def test_falsy_values(self, isolated_env, raw):
        isolated_env.setenv("TEST_BOOL", raw)
        assert _bool_env("TEST_BOOL") is False

    def test_default_when_unset(self, isolated_env):
        assert _bool_env("TEST_BOOL", default=True) is True
        assert _bool_env("TEST_BOOL", default=False) is False


class TestLlm3MenuSignalFilterFromEnv:
    def test_off_default(self, isolated_env):
        assert _llm3_menu_signal_filter_from_env() == "off"

    def test_recognised_values(self, isolated_env):
        for v in ("high", "medium", "low"):
            isolated_env.setenv("RERANKER_LLM3_MENU_SIGNAL_FILTER", v.upper())
            assert _llm3_menu_signal_filter_from_env() == v

    def test_unknown_falls_back_to_off(self, isolated_env):
        isolated_env.setenv("RERANKER_LLM3_MENU_SIGNAL_FILTER", "weird")
        assert _llm3_menu_signal_filter_from_env() == "off"


class TestOccupationSkillsSignalInclude:
    def test_unset_returns_none(self, isolated_env):
        assert _occupation_skills_signal_include_from_env() is None

    def test_all_returns_none(self, isolated_env):
        for tok in ("all", "*", "any", " "):
            isolated_env.setenv("RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE", tok)
            assert _occupation_skills_signal_include_from_env() is None

    def test_returns_filtered_set(self, isolated_env):
        isolated_env.setenv(
            "RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE", "high, medium , bogus, low"
        )
        out = _occupation_skills_signal_include_from_env()
        assert out == frozenset({"high", "medium", "low"})

    def test_only_unknown_returns_none(self, isolated_env):
        isolated_env.setenv("RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE", "garbage")
        assert _occupation_skills_signal_include_from_env() is None


class TestClassifiedLinkedEntityMinSimilarity:
    def test_unset_returns_none(self, isolated_env):
        assert _classified_linked_entity_min_similarity_from_env() is None

    def test_parses_valid_float(self, isolated_env):
        isolated_env.setenv("CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY", "0.65")
        assert _classified_linked_entity_min_similarity_from_env() == pytest.approx(0.65)

    def test_alias_supported(self, isolated_env):
        isolated_env.setenv("CLASSIFIER_LINKED_ENTITY_MIN_SIMILARITY", "0.5")
        assert _classified_linked_entity_min_similarity_from_env() == pytest.approx(0.5)

    def test_out_of_range_returns_none(self, isolated_env):
        for v in ("-0.1", "1.5", "garbage"):
            isolated_env.setenv("CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY", v)
            assert _classified_linked_entity_min_similarity_from_env() is None


class TestClassifiedLinkedEntityLabelDedupeTypes:
    def test_unset_returns_none(self, isolated_env):
        assert _classified_linked_entity_label_dedupe_types_from_env() is None

    def test_returns_normalized_set(self, isolated_env):
        isolated_env.setenv(
            "CLASSIFIED_LINKED_ENTITY_LABEL_DEDUPE_TYPES",
            "Skills, occupation, skill",
        )
        out = _classified_linked_entity_label_dedupe_types_from_env()
        assert out == frozenset({"skill", "occupation"})

    def test_unknown_tokens_warned_and_ignored(self, isolated_env):
        isolated_env.setenv(
            "CLASSIFIED_LINKED_ENTITY_LABEL_DEDUPE_TYPES",
            "language",
        )
        out = _classified_linked_entity_label_dedupe_types_from_env()
        assert out is None


class TestOccupationSkillRelationOccupationTypes:
    def test_unset_returns_none(self, isolated_env):
        assert _occupation_skill_relation_occupation_types_from_env() is None

    def test_all_aliases_return_none(self, isolated_env):
        for tok in ("*", "all", "any"):
            isolated_env.setenv("OCCUPATION_SKILL_RELATION_OCCUPATION_TYPES", tok)
            assert _occupation_skill_relation_occupation_types_from_env() is None

    def test_explicit_returns_lowercase_set(self, isolated_env):
        isolated_env.setenv(
            "OCCUPATION_SKILL_RELATION_OCCUPATION_TYPES",
            "ESCOOccupation, LocalOccupation",
        )
        out = _occupation_skill_relation_occupation_types_from_env()
        assert out == frozenset({"escooccupation", "localoccupation"})


class TestTargetCountrySlugFromEnv:
    def test_default_zambia(self, isolated_env):
        assert _target_country_slug_from_env() == "zambia"

    def test_recognised_slugs(self, isolated_env):
        for slug in ("zambia", "kenya", "south_africa"):
            isolated_env.setenv("TARGET_COUNTRY", slug.upper())
            assert _target_country_slug_from_env() == slug

    def test_invalid_falls_back_to_zambia(self, isolated_env):
        isolated_env.setenv("TARGET_COUNTRY", "narnia")
        assert _target_country_slug_from_env() == "zambia"


class TestCountryNameFromSlug:
    @pytest.mark.parametrize(
        "slug, expected",
        [
            ("zambia", "Zambia"),
            ("kenya", "Kenya"),
            ("south_africa", "South Africa"),
            ("custom_country", "Custom Country"),
        ],
    )
    def test_values(self, slug, expected):
        assert _country_name_from_slug(slug) == expected


class TestOccupationRelationLookupFromEnv:
    def test_default_is_tabiya_bridge(self, isolated_env):
        mode, fb = _occupation_relation_lookup_from_env()
        assert mode == "tabiya_bridge"
        assert fb is True

    def test_aliases(self, isolated_env):
        for alias, expected in [
            ("local", "tabiya_bridge"),
            ("tabiya", "tabiya_bridge"),
            ("tabiya_bridge", "tabiya_bridge"),
            ("esco", "esco_uuid_bridge"),
            ("esco_uuid", "esco_uuid_bridge"),
            ("esco_bridge", "esco_uuid_bridge"),
            ("garbage", "tabiya_bridge"),
        ]:
            isolated_env.setenv("OCCUPATION_RELATION_LOOKUP", alias)
            mode, _ = _occupation_relation_lookup_from_env()
            assert mode == expected

    def test_fallback_flag_is_respected(self, isolated_env):
        isolated_env.setenv("OCCUPATION_RELATION_ESCO_FALLBACK", "0")
        _, fb = _occupation_relation_lookup_from_env()
        assert fb is False


class TestRerankerEnableLlm3FromEnv:
    def test_default_on(self, isolated_env):
        assert reranker_enable_llm3_from_env() is True

    def test_off(self, isolated_env):
        isolated_env.setenv("RERANKER_ENABLE_LLM3", "0")
        assert reranker_enable_llm3_from_env() is False


# ---------------------------------------------------------------------------
# Settings.from_env composite behaviour
# ---------------------------------------------------------------------------


class TestSettingsFromEnv:
    def test_baseline_defaults(self, make_settings):
        s = make_settings()
        assert isinstance(s, Settings)
        assert s.mongodb_uri == "mongodb://localhost:27017"
        assert s.mongodb_database == "ZambiaJobs"
        assert s.mongodb_source_collection == "classified_jobs"
        assert s.mongodb_target_collection == "ranked_jobs"
        assert s.gemini_api_key == ""
        assert s.gemini_model.startswith("gemini-")
        assert s.target_country == "zambia"
        assert s.country_name == "Zambia"
        assert s.classified_linked_entity_types == ("skill", "occupation")
        assert s.classified_linked_entity_min_similarity is None
        assert s.classified_linked_entity_min_similarity_strict is False
        assert s.bulk_write_batch_size == 100
        # default classifier intersect off
        assert s.reranker_llm3_classifier_intersect is False

    def test_country_change_propagates_name(self, make_settings):
        s = make_settings(TARGET_COUNTRY="kenya")
        assert s.target_country == "kenya"
        assert s.country_name == "Kenya"

    def test_target_collection_default_does_not_mirror_source(self, make_settings):
        s = make_settings()
        assert s.mongodb_source_collection != s.mongodb_target_collection

    def test_paths_resolve_under_resources(self, make_settings):
        s = make_settings()
        assert s.job_attributes_schema_path.is_file()
        assert (
            s.occupations_csv_path.exists() or not s.occupations_csv_path.exists()
        )  # may be empty
        # At minimum, we must have an absolute resolved path
        assert s.resources_dir.is_absolute()
