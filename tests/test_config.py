"""Unit tests for ``app.config.Settings``: validation, defaults, and derived fields.

These pin env-parsing behaviour so a wrong env default cannot silently change the
pipeline output across all jobs.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import (
    Settings,
    _country_name_from_slug,
    _normalize_classifier_entity_type_token,
)
from app.llm.errors import LlmConfigurationError

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


# ---------------------------------------------------------------------------
# Env-driven Settings fields
# ---------------------------------------------------------------------------


class TestEntityTypesField:
    def test_default_when_unset(self, make_settings):
        assert make_settings().classified_linked_entity_types == ("skill", "occupation")

    def test_normalizes_and_dedupes(self, make_settings):
        s = make_settings(CLASSIFIED_LINKED_ENTITY_TYPES="Skills, skill, occupations")
        assert s.classified_linked_entity_types == ("skill", "occupation")

    def test_blank_falls_back(self, make_settings):
        assert make_settings(
            CLASSIFIED_LINKED_ENTITY_TYPES="  "
        ).classified_linked_entity_types == (
            "skill",
            "occupation",
        )


class TestClassifiedKeysCommaLists:
    """``keys_salary`` / ``keys_location`` / ``keys_county`` / ``keys_employment`` /
    ``keys_fingerprint`` bind a plain ``str`` field and split it in a property —
    binding ``tuple[str, ...]`` directly from env would crash construction, since
    pydantic-settings JSON-decodes "complex" field types before any validator runs
    (regression: ``CLASSIFIED_KEYS_SALARY=pay,wage`` used to raise ``SettingsError``).
    """

    def test_default_when_unset(self, make_settings):
        s = make_settings()
        assert s.keys_salary == ("salary_text", "salary", "compensation", "pay")
        assert s.keys_location == ("city", "location", "town")

    def test_comma_separated_value_is_split_and_stripped(self, make_settings):
        s = make_settings(CLASSIFIED_KEYS_SALARY="pay, wage")
        assert s.keys_salary == ("pay", "wage")

    def test_blank_falls_back_to_default(self, make_settings):
        s = make_settings(CLASSIFIED_KEYS_COUNTY="  ")
        assert s.keys_county == ("province", "county", "district", "region")

    def test_keys_description_prepends_fixed_first_choice(self, make_settings):
        s = make_settings(CLASSIFIED_KEYS_DESCRIPTION="body,summary")
        assert s.keys_description[:2] == ("description", "job_description")
        assert "body" in s.keys_description and "summary" in s.keys_description


class TestLlm3MenuSignalFilter:
    def test_off_default(self, make_settings):
        assert make_settings().reranker_llm3_menu_signal_filter == "off"

    def test_recognised_values(self, make_settings):
        for v in ("high", "medium", "low"):
            s = make_settings(RERANKER_LLM3_MENU_SIGNAL_FILTER=v.upper())
            assert s.reranker_llm3_menu_signal_filter == v

    def test_unknown_rejected(self, isolated_env):
        isolated_env.setenv("RERANKER_LLM3_MENU_SIGNAL_FILTER", "weird")
        with pytest.raises(ValidationError):
            Settings.from_env()


class TestOccupationSkillsSignalInclude:
    def test_unset_returns_none(self, make_settings):
        assert make_settings().reranker_occupation_skills_signal_include is None

    def test_all_returns_none(self, make_settings):
        for tok in ("all", "*", "any", " "):
            s = make_settings(RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE=tok)
            assert s.reranker_occupation_skills_signal_include is None

    def test_returns_filtered_set(self, make_settings):
        s = make_settings(RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE="high, medium , bogus, low")
        assert s.reranker_occupation_skills_signal_include == frozenset({"high", "medium", "low"})

    def test_only_unknown_returns_none(self, make_settings):
        s = make_settings(RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE="garbage")
        assert s.reranker_occupation_skills_signal_include is None


class TestClassifiedLinkedEntityMinSimilarity:
    def test_unset_returns_none(self, make_settings):
        assert make_settings().classified_linked_entity_min_similarity is None

    def test_parses_valid_float(self, make_settings):
        s = make_settings(CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY="0.65")
        assert s.classified_linked_entity_min_similarity == pytest.approx(0.65)

    def test_alias_supported(self, make_settings):
        s = make_settings(CLASSIFIER_LINKED_ENTITY_MIN_SIMILARITY="0.5")
        assert s.classified_linked_entity_min_similarity == pytest.approx(0.5)

    def test_out_of_range_returns_none(self, make_settings):
        for v in ("-0.1", "1.5"):
            s = make_settings(CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY=v)
            assert s.classified_linked_entity_min_similarity is None

    def test_garbage_rejected(self, isolated_env):
        isolated_env.setenv("CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY", "garbage")
        with pytest.raises(ValidationError):
            Settings.from_env()


class TestClassifiedLinkedEntityLabelDedupeTypes:
    def test_unset_returns_none(self, make_settings):
        assert make_settings().classified_linked_entity_label_dedupe_types is None

    def test_returns_normalized_set(self, make_settings):
        s = make_settings(CLASSIFIED_LINKED_ENTITY_LABEL_DEDUPE_TYPES="Skills, occupation, skill")
        assert s.classified_linked_entity_label_dedupe_types == frozenset({"skill", "occupation"})

    def test_unknown_tokens_warned_and_ignored(self, make_settings):
        s = make_settings(CLASSIFIED_LINKED_ENTITY_LABEL_DEDUPE_TYPES="language")
        assert s.classified_linked_entity_label_dedupe_types is None


class TestOccupationSkillRelationOccupationTypes:
    def test_unset_returns_none(self, make_settings):
        assert make_settings().occupation_skill_relation_occupation_types is None

    def test_all_aliases_return_none(self, make_settings):
        for tok in ("*", "all", "any"):
            s = make_settings(OCCUPATION_SKILL_RELATION_OCCUPATION_TYPES=tok)
            assert s.occupation_skill_relation_occupation_types is None

    def test_explicit_returns_lowercase_set(self, make_settings):
        s = make_settings(
            OCCUPATION_SKILL_RELATION_OCCUPATION_TYPES="ESCOOccupation, LocalOccupation"
        )
        assert s.occupation_skill_relation_occupation_types == frozenset(
            {"escooccupation", "localoccupation"}
        )


class TestTargetCountry:
    def test_default_zambia(self, make_settings):
        assert make_settings().target_country == "zambia"

    def test_recognised_slugs(self, make_settings):
        for slug in ("zambia", "kenya", "south_africa"):
            assert make_settings(TARGET_COUNTRY=slug.upper()).target_country == slug

    def test_invalid_falls_back_to_zambia(self, make_settings):
        assert make_settings(TARGET_COUNTRY="narnia").target_country == "zambia"


class TestJobAttributesExtraFields:
    def test_none_by_default(self, make_settings):
        # Kenya too: the extra question comes from the deployment's env, not the country.
        assert make_settings(TARGET_COUNTRY="kenya").job_attributes_extra_fields == ()

    def test_names_with_their_instructions(self, make_settings):
        s = make_settings(
            RERANKER_LLM2_EXTRA_FIELDS=" requires_post_secondary , needs_license,",
            RERANKER_LLM2_EXTRA_FIELD_REQUIRES_POST_SECONDARY="\nwhether a degree is needed.\n",
            RERANKER_LLM2_EXTRA_FIELD_NEEDS_LICENSE="whether a licence is needed.",
        )
        fields = s.job_attributes_extra_fields
        assert [f.name for f in fields] == ["requires_post_secondary", "needs_license"]
        # Surrounding newlines are trimmed; the text is otherwise kept as written.
        assert fields[0].instructions == "whether a degree is needed."

    def test_listed_field_without_instructions_fails(self, make_settings):
        s = make_settings(RERANKER_LLM2_EXTRA_FIELDS="requires_post_secondary")
        with pytest.raises(ValueError, match="RERANKER_LLM2_EXTRA_FIELD_REQUIRES_POST_SECONDARY"):
            _ = s.job_attributes_extra_fields

    def test_name_must_be_a_lowercase_identifier(self, make_settings):
        s = make_settings(RERANKER_LLM2_EXTRA_FIELDS="Requires-Degree")
        with pytest.raises(ValueError, match="lowercase identifier"):
            _ = s.job_attributes_extra_fields


class TestOccupationRelationLookup:
    def test_default_is_tabiya_bridge(self, make_settings):
        s = make_settings()
        assert s.occupation_relation_lookup == "tabiya_bridge"
        assert s.occupation_relation_esco_fallback is True

    def test_aliases(self, make_settings):
        for alias, expected in [
            ("local", "tabiya_bridge"),
            ("tabiya", "tabiya_bridge"),
            ("tabiya_bridge", "tabiya_bridge"),
            ("esco", "esco_uuid_bridge"),
            ("esco_uuid", "esco_uuid_bridge"),
            ("esco_bridge", "esco_uuid_bridge"),
            ("garbage", "tabiya_bridge"),
        ]:
            assert (
                make_settings(OCCUPATION_RELATION_LOOKUP=alias).occupation_relation_lookup
                == expected
            )

    def test_fallback_flag_is_respected(self, make_settings):
        assert (
            make_settings(OCCUPATION_RELATION_ESCO_FALLBACK="0").occupation_relation_esco_fallback
            is False
        )


class TestRerankerEnableLlm3:
    def test_default_on(self, make_settings):
        assert make_settings().reranker_enable_llm3 is True

    def test_off(self, make_settings):
        assert make_settings(RERANKER_ENABLE_LLM3="0").reranker_enable_llm3 is False


class TestLlm3Mode:
    def test_default_folds_into_llm1(self, make_settings):
        assert make_settings().reranker_llm3_mode == "fold"

    def test_call_is_accepted_case_insensitively(self, make_settings):
        assert make_settings(RERANKER_LLM3_MODE=" Call ").reranker_llm3_mode == "call"

    def test_unknown_value_rejected(self, isolated_env):
        isolated_env.setenv("RERANKER_LLM3_MODE", "standalone")
        with pytest.raises(ValidationError):
            Settings.from_env()


class TestLogLevel:
    def test_default_info(self, make_settings):
        assert make_settings().log_level == "INFO"

    def test_accepts_known_levels(self, make_settings):
        assert make_settings(LOG_LEVEL="debug").log_level == "DEBUG"

    def test_warn_is_accepted_as_an_alias_for_warning(self, make_settings):
        """``logging.WARN`` is Python's own alias for ``logging.WARNING`` — a config
        value every other logging-aware tool treats as valid must not crash startup."""
        assert make_settings(LOG_LEVEL="warn").log_level == "WARNING"

    def test_unknown_rejected(self, isolated_env):
        isolated_env.setenv("LOG_LEVEL", "VERBOSE")
        with pytest.raises(ValidationError):
            Settings.from_env()


# ---------------------------------------------------------------------------
# Settings.from_env composite behaviour
# ---------------------------------------------------------------------------


class TestRerankerSkipAlreadyRanked:
    def test_default_off(self, make_settings):
        assert make_settings().reranker_skip_already_ranked is False

    def test_env_var_enables_it(self, make_settings):
        assert make_settings(RERANKER_SKIP_ALREADY_RANKED="1").reranker_skip_already_ranked is True


class TestSettingsFromEnv:
    def test_baseline_defaults(self, make_settings):
        s = make_settings()
        assert isinstance(s, Settings)
        assert s.mongodb_uri == "mongodb://localhost:27017"
        assert s.mongodb_database == "ZambiaJobs"
        assert s.mongodb_source_collection == "classified_jobs"
        assert s.mongodb_target_collection == "ranked_jobs"
        assert s.gemini_api_key == ""
        assert s.llm.model.startswith("gemini-")
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


class TestSettingsWorkActivitiesEnrichment:
    """``utils.work_activities_enrichment.Config`` folded into ``Settings``."""

    def test_defaults_mirror_main_mongo_settings(self, make_settings):
        s = make_settings(MONGODB_DATABASE="Kenya", MONGODB_TARGET_COLLECTION="ranked_jobs_kenya")
        assert s.enrich_wa_database == "Kenya"
        assert s.enrich_wa_target_collection == "ranked_jobs_kenya"

    def test_enrich_wa_overrides_win(self, make_settings):
        s = make_settings(
            MONGODB_DATABASE="Kenya",
            ENRICH_WA_DATABASE="KenyaReference",
            MONGODB_TARGET_COLLECTION="ranked_jobs",
            ENRICH_WA_TARGET_COLLECTION="ranked_jobs_enriched",
        )
        assert s.enrich_wa_database == "KenyaReference"
        assert s.enrich_wa_target_collection == "ranked_jobs_enriched"

    def test_occupation_db_path_defaults_to_language_pack(self, make_settings):
        s = make_settings()
        assert s.enrich_wa_occupation_db_path.is_file()
        assert s.enrich_wa_occupation_db_path.name == "combined_occupation_database_with_wa.json"

    def test_batch_and_flag_defaults(self, make_settings):
        s = make_settings()
        assert s.enrich_wa_bulk_batch_size == 500
        assert s.enrich_wa_limit == 0
        assert s.enrich_wa_dry_run is False
        assert s.enrich_wa_only_missing is True


class TestBlankEnvValuesAreUnset:
    """A blank env var (``FOO=``) means "unset", not an empty string — a deployment that
    comments out a value but leaves the ``KEY=`` line should fall back to the field's
    default rather than fail validation on an empty string."""

    def test_blank_bool_falls_back_to_default(self, make_settings):
        s = make_settings(RERANKER_ENABLE_LLM3="")
        assert s.reranker_enable_llm3 is True

    def test_blank_int_falls_back_to_default(self, make_settings):
        s = make_settings(BULK_WRITE_BATCH_SIZE="")
        assert s.bulk_write_batch_size == 100

    def test_blank_literal_falls_back_to_default(self, make_settings):
        s = make_settings(LOG_LEVEL="")
        assert s.log_level == "INFO"


class TestFieldLevelRejection:
    """Fields with an explicit type/range reject a bad value with a ``ValidationError``
    at construction — the whole point of moving off the hand-rolled parsers, which
    silently fell back to a default instead."""

    def test_unparseable_bool_is_rejected(self, isolated_env):
        isolated_env.setenv("RERANKER_ENABLE_LLM3", "enabled")
        with pytest.raises(ValidationError):
            Settings.from_env()

    def test_unparseable_int_is_rejected(self, isolated_env):
        isolated_env.setenv("RERANKER_LLM3_MAX_MENU_SKILLS", "not-a-number")
        with pytest.raises(ValidationError):
            Settings.from_env()

    @pytest.mark.parametrize(
        "env_var, out_of_range_value",
        [
            ("RERANKER_LLM_TEMPERATURE", "5.0"),
            ("RERANKER_LLM_MAX_OUTPUT_TOKENS", "-1"),
            ("RERANKER_LLM_TIMEOUT_S", "99999"),
            ("RERANKER_PARALLEL_WORKERS", "999"),
            ("RERANKER_LLM3_MAX_MENU_SKILLS", "-1"),
            ("BULK_WRITE_BATCH_SIZE", "0"),
            ("RERANKER_JOB_EMBEDDING_DIM", "0"),
            ("RERANKER_JOB_EMBEDDING_BATCH_SIZE", "99999"),
            ("RERANKER_JOB_EMBEDDING_SLEEP_S", "-1"),
            ("ENRICH_WA_BULK_BATCH_SIZE", "0"),
            ("ENRICH_WA_LIMIT", "-1"),
        ],
    )
    def test_out_of_range_value_is_rejected(self, isolated_env, env_var, out_of_range_value):
        isolated_env.setenv(env_var, out_of_range_value)
        with pytest.raises(ValidationError):
            Settings.from_env()


class TestSettingsImmutability:
    def test_cannot_reassign_a_field_after_construction(self, make_settings):
        s = make_settings()
        with pytest.raises(AttributeError):
            s.mongodb_uri = "mongodb://elsewhere:27017"

    def test_cannot_reassign_a_cached_property_after_construction(self, make_settings):
        """``model_config``'s ``frozen=True`` only guards declared fields — a
        ``@cached_property`` (``llm``, ``language``, every ``*_path``, …) writes straight
        to ``instance.__dict__`` on first access and is otherwise an ordinary attribute,
        so without ``Settings.__setattr__`` this would silently succeed and corrupt the
        shared ``get_settings()`` singleton for every other caller."""
        s = make_settings()
        s.language  # noqa: B018 — populate the cache before trying to overwrite it
        with pytest.raises(AttributeError):
            s.language = "es"
        with pytest.raises(AttributeError):
            s.llm = None

    def test_llm_misconfiguration_fails_at_construction_not_first_access(self, isolated_env):
        """``llm`` is a ``@cached_property`` (to avoid double-binding ``RERANKER_LLM_*``
        fields), but an unknown provider must still fail fast at startup rather than on
        the first request that happens to read ``settings.llm`` — ``Settings.__init__``
        forces it eagerly."""
        isolated_env.setenv("RERANKER_LLM_PROVIDER", "not-a-real-provider")
        with pytest.raises(LlmConfigurationError, match="not-a-real-provider"):
            Settings.from_env()
