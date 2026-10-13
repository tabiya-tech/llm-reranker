"""Tests for the FastAPI service in ``api.py``.

We cover the two safest behaviours that don't require Gemini or Mongo:

1. ``_normalize_mongo_extended_json`` — the conversion that lets the API accept
   Compass / Atlas / mongoexport output verbatim.
2. ``GET /health`` — a smoke test that the FastAPI ``app`` boots and responds
   when the dependent taxonomy + Gemini key are stubbed.

The ``/rank`` endpoint is exercised in ``test_pipeline_integration.py`` where
the LLM service is monkeypatched.
"""

from __future__ import annotations

import api
from api import JsonValue, _normalize_mongo_extended_json

# ---------------------------------------------------------------------------
# _normalize_mongo_extended_json
# ---------------------------------------------------------------------------


class TestNormalizeMongoExtendedJson:
    def test_oid_unwraps_to_string(self):
        assert _normalize_mongo_extended_json({"$oid": "abc"}) == "abc"

    def test_date_unwraps_to_string(self):
        assert (
            _normalize_mongo_extended_json({"$date": "2026-01-01T00:00:00Z"})
            == "2026-01-01T00:00:00Z"
        )

    def test_number_int_unwraps_to_int(self):
        assert _normalize_mongo_extended_json({"$numberInt": "42"}) == 42

    def test_number_long_unwraps_to_int(self):
        assert _normalize_mongo_extended_json({"$numberLong": "9999999999"}) == 9999999999

    def test_number_double_unwraps_to_float(self):
        assert _normalize_mongo_extended_json({"$numberDouble": "1.5"}) == 1.5

    def test_recursive_dict_normalization(self):
        body = {
            "_id": {"$oid": "abc"},
            "metadata": {"posted_date": {"$date": "2026-01-01T00:00:00Z"}},
        }
        out = _normalize_mongo_extended_json(body)
        assert out == {
            "_id": "abc",
            "metadata": {"posted_date": "2026-01-01T00:00:00Z"},
        }

    def test_list_normalization(self):
        body = [{"$oid": "a"}, {"$oid": "b"}, {"plain": 1}]
        assert _normalize_mongo_extended_json(body) == ["a", "b", {"plain": 1}]

    def test_only_unwraps_singleton_special_keys(self):
        # If the dict has more than one key, even one of which is "$oid", do not unwrap.
        body = {"$oid": "abc", "other": 1}
        assert _normalize_mongo_extended_json(body) == {"$oid": "abc", "other": 1}

    def test_passthrough_for_primitives(self):
        assert _normalize_mongo_extended_json("string") == "string"
        assert _normalize_mongo_extended_json(7) == 7
        assert _normalize_mongo_extended_json(None) is None
        assert _normalize_mongo_extended_json(True) is True

    def test_nested_inside_lists_inside_dicts(self):
        body: dict[str, JsonValue] = {"items": [{"id": {"$oid": "x"}, "n": {"$numberInt": "3"}}]}
        out = _normalize_mongo_extended_json(body)
        assert out == {"items": [{"id": "x", "n": 3}]}


# ---------------------------------------------------------------------------
# Smoke: GET /health works with taxonomy stubbed
# ---------------------------------------------------------------------------


class _FakeTaxonomy:
    """Minimal stand-in for ``app.pipeline.TaxonomyContext``.

    Only ``occ_map`` is read in the ``api.lifespan`` log line.
    """

    def __init__(self):
        self.occ_map = {}


def test_health_endpoint_returns_ok(make_settings, monkeypatch):
    """FastAPI lifespan boots when taxonomy load is stubbed."""
    # Real loaders need full resources/CSVs; we replace with a fake taxonomy
    # and a Gemini key (lifespan refuses to start without one).
    make_settings(GEMINI_API_KEY="dummy-key", TARGET_COUNTRY="kenya")
    fake_tx = _FakeTaxonomy()
    monkeypatch.setattr(api, "load_taxonomy_context", lambda settings: fake_tx)

    from fastapi.testclient import TestClient

    with TestClient(api.app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    # Health also reports which languages this image serves and which packs are loaded.
    assert resp.json() == {
        "status": "ok",
        "taxonomy_loaded": True,
        "default_language": "en",
        "supported_languages": ["en", "es"],
        "loaded_languages": ["en"],
    }
