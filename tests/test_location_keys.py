"""``utils.location_keys`` against the cases shared with the matching service's copy of the rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.location_keys import canonical_location, location_keys_for

_VECTORS = Path(__file__).resolve().parent / "fixtures" / "location_key_vectors.json"


@pytest.mark.parametrize(
    "case",
    json.loads(_VECTORS.read_text(encoding="utf-8"))["cases"],
    ids=lambda c: " | ".join(str(v) for v in c["values"]),
)
def test_shared_key_vectors(case):
    assert list(location_keys_for(*case["values"])) == case["keys"]


def test_canonical_location_trims_collapses_and_casefolds():
    assert canonical_location("  Taita   Taveta. ") == "taita taveta"
    assert canonical_location(None) == ""
