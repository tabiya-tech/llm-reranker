"""Tests for ``app.database`` pure helpers (no Mongo required).

We focus on:

- ``build_job_description_context`` field-resolution precedence rules
  (top-level vs metadata, dotted paths, multi-key first-match-wins).
- ``_scalar_to_str`` value coercion.
- ``_upsert_filter_for_llm_results`` selector logic (covers all three
  fall-back paths and the error case).
- ``_parse_object_id`` happy + error path.

Mongo connectivity, bulk_write and the cursor iterators are intentionally
out of scope here — they need a Mongo replica or an extensive fake.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from bson import ObjectId

from app.database import (
    _field_or_metadata,
    _first_str,
    _get_dotted,
    _multi_key_top_or_metadata,
    _parse_object_id,
    _scalar_to_str,
    _upsert_filter_for_llm_results,
    build_job_description_context,
)
from app.job_context import JobDescriptionContext

# ---------------------------------------------------------------------------
# _scalar_to_str
# ---------------------------------------------------------------------------


class TestScalarToStr:
    def test_none_returns_none(self):
        assert _scalar_to_str(None) is None

    def test_string_strips_and_blanks_returned_as_none(self):
        assert _scalar_to_str("  hello  ") == "hello"
        assert _scalar_to_str("   ") is None

    def test_int_and_float_become_str(self):
        assert _scalar_to_str(42) == "42"
        assert _scalar_to_str(3.5) == "3.5"

    def test_datetime_becomes_isoformat(self):
        dt = datetime(2026, 6, 12, 10, 0, tzinfo=timezone.utc)
        assert _scalar_to_str(dt) == dt.isoformat()

    def test_bytes_decoded_when_utf8(self):
        assert _scalar_to_str(b"hi") == "hi"
        assert _scalar_to_str(b"   ") is None

    def test_bytes_undecodable_returns_none(self):
        assert _scalar_to_str(b"\xff\xfe\x00") is None

    def test_other_types_return_none(self):
        assert _scalar_to_str([1, 2]) is None
        assert _scalar_to_str({"a": 1}) is None


# ---------------------------------------------------------------------------
# _first_str / _field_or_metadata / _get_dotted / _multi_key_top_or_metadata
# ---------------------------------------------------------------------------


class TestFirstStr:
    def test_returns_first_non_empty(self):
        assert _first_str({"a": "  ", "b": "x", "c": "y"}, "a", "b", "c") == "x"

    def test_returns_none_when_all_blank(self):
        assert _first_str({"a": "", "b": "  "}, "a", "b") is None


class TestFieldOrMetadata:
    def test_top_level_wins(self):
        doc = {"title": "T", "metadata": {"title": "META"}}
        assert _field_or_metadata(doc, "title") == "T"

    def test_falls_back_to_metadata(self):
        doc = {"title": "  ", "metadata": {"title": "META"}}
        assert _field_or_metadata(doc, "title") == "META"

    def test_meta_key_alias(self):
        doc = {"metadata": {"jt": "META"}}
        assert _field_or_metadata(doc, "title", meta_key="jt") == "META"

    def test_returns_none_when_missing(self):
        assert _field_or_metadata({}, "title") is None


class TestGetDotted:
    def test_resolves_dotted_path(self):
        doc = {"a": {"b": {"c": "leaf"}}}
        assert _get_dotted(doc, "a.b.c") == "leaf"

    def test_returns_none_when_segment_missing(self):
        assert _get_dotted({"a": {"b": {}}}, "a.b.c") is None

    def test_returns_none_when_intermediate_not_dict(self):
        assert _get_dotted({"a": {"b": "x"}}, "a.b.c") is None

    def test_handles_top_level_only(self):
        assert _get_dotted({"plain": "v"}, "plain") == "v"


class TestMultiKeyTopOrMetadata:
    def test_top_level_first_match(self):
        doc = {"city": "  ", "town": "Lusaka", "metadata": {"town": "META"}}
        assert _multi_key_top_or_metadata(doc, "city", "town") == "Lusaka"

    def test_metadata_fallback_when_top_blank(self):
        doc = {"city": "  ", "metadata": {"town": "META"}}
        assert _multi_key_top_or_metadata(doc, "city", "town") == "META"

    def test_dotted_path_at_top_level(self):
        doc = {"location": {"primary": "TopCity"}}
        assert _multi_key_top_or_metadata(doc, "location.primary") == "TopCity"

    def test_dotted_path_in_metadata(self):
        doc = {"metadata": {"location": {"primary": "MetaCity"}}}
        assert _multi_key_top_or_metadata(doc, "location.primary") == "MetaCity"


# ---------------------------------------------------------------------------
# build_job_description_context
# ---------------------------------------------------------------------------


class TestBuildJobDescriptionContext:
    def test_top_level_fields(self, make_settings):
        s = make_settings()
        doc = {
            "_id": "doc-1",
            "job_id": ObjectId(),
            "job_fingerprint": "fp-x",
            "title": "Engineer",
            "employer": "ACME",
            "category": "Tech",
            "application_url": "https://x",
            "posted_date": "2026-01-01",
            "closing_date": "2026-02-01",
            "source_platform": "BrighterMonday",
            "description": "JD body",
            "requirements": "3 years",
            "salary_text": "K1000",
            "city": "Lusaka",
            "county": "Lusaka District",
            "employment_type": "Full time",
        }
        ctx = build_job_description_context(doc, settings=s)
        assert isinstance(ctx, JobDescriptionContext)
        assert ctx.job_fingerprint == "fp-x"
        assert ctx.title == "Engineer"
        assert ctx.employer == "ACME"
        assert ctx.application_url == "https://x"
        assert ctx.posted_date == "2026-01-01"
        assert ctx.closing_date == "2026-02-01"
        assert ctx.raw_job_description == "JD body"
        assert ctx.requirements_text == "3 years"
        assert ctx.salary == "K1000"
        assert ctx.city == "Lusaka"
        assert ctx.county == "Lusaka District"
        assert ctx.employment_type == "Full time"
        # classified is a copy of the doc
        assert ctx.classified["_id"] == "doc-1"
        assert ctx.classified is not doc

    def test_metadata_fallback(self, make_settings):
        s = make_settings()
        doc = {
            "metadata": {
                "title": "Meta Title",
                "salary_text": "MetaSalary",
                "city": "MetaCity",
            }
        }
        ctx = build_job_description_context(doc, settings=s)
        assert ctx.title == "Meta Title"
        assert ctx.salary == "MetaSalary"
        assert ctx.city == "MetaCity"

    def test_alias_keys_first_match_wins(self, make_settings):
        s = make_settings()
        doc = {
            "job_description": "JD-alias",  # alias of `description`
            "qualifications": "QUAL",  # alias of `requirements`
        }
        ctx = build_job_description_context(doc, settings=s)
        assert ctx.raw_job_description == "JD-alias"
        assert ctx.requirements_text == "QUAL"

    def test_missing_optional_fields_default_to_none(self, make_settings):
        s = make_settings()
        ctx = build_job_description_context({}, settings=s)
        assert ctx.title is None
        assert ctx.raw_job_description is None
        assert ctx.requirements_text is None
        assert ctx.city is None


# ---------------------------------------------------------------------------
# _upsert_filter_for_llm_results
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
        classified={},
    )
    base.update(overrides)
    return JobDescriptionContext(**base)


class TestUpsertFilterForLlmResults:
    def test_prefers_job_id(self, make_settings):
        s = make_settings()
        ctx = _ctx(job_id="jid-1", job_fingerprint="fp")
        flt = _upsert_filter_for_llm_results({"_id": "doc-1"}, ctx, s)
        assert flt == {s.mongo_field_job_id: "jid-1"}

    def test_falls_back_to_fingerprint(self, make_settings):
        s = make_settings()
        ctx = _ctx(job_fingerprint="fp")
        flt = _upsert_filter_for_llm_results({"_id": "doc-1"}, ctx, s)
        assert flt == {s.mongo_field_job_fingerprint: "fp"}

    def test_falls_back_to_source_classified_id(self, make_settings):
        s = make_settings()
        ctx = _ctx()
        flt = _upsert_filter_for_llm_results({"_id": "doc-1"}, ctx, s)
        assert flt == {"source_classified_id": "doc-1"}

    def test_raises_when_no_identifier(self, make_settings):
        s = make_settings()
        ctx = _ctx()
        with pytest.raises(ValueError, match="no job id, fingerprint, or source _id"):
            _upsert_filter_for_llm_results({}, ctx, s)


# ---------------------------------------------------------------------------
# _parse_object_id
# ---------------------------------------------------------------------------


class TestParseObjectId:
    def test_passes_through_existing_oid(self):
        oid = ObjectId()
        assert _parse_object_id(oid) is oid

    def test_parses_hex_string(self):
        oid = ObjectId()
        assert _parse_object_id(str(oid)) == oid

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid ObjectId"):
            _parse_object_id("not-an-oid")
