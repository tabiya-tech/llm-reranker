"""P2 unit tests for utility parsers and the is_active sync helper.

Covers:
- ``utils.uuid_to_tabiya_id`` cell parser, history parser, and the in-memory map
  loader (driven by a tiny fixture CSV in a tmp path).
- ``utils.occupation_to_skills_mapping`` low-level helpers
  (``cell_to_taxonomy_id``, ``esco_skill_uri_for_uuid``).
- ``app.ranked_jobs.is_active.parse_closing_date`` and the sync
  function against an in-memory fake collection.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, cast

import pytest
from pymongo.collection import Collection

from app.ranked_jobs.is_active import parse_closing_date as _parse_date
from app.ranked_jobs.mongo_repository import sync_is_active_from_closing_date
from app.taxonomy.csv_loaders import load_skills_uuid_tabiya_map
from utils.occupation_to_skills_mapping import (
    cell_to_taxonomy_id,
    esco_skill_uri_for_uuid,
)
from utils.uuid_to_tabiya_id import (
    cell_to_esco_uuid,
    parse_uuid_history_cell,
)

# ---------------------------------------------------------------------------
# uuid_to_tabiya_id helpers
# ---------------------------------------------------------------------------

UUID_A = "5247a6ce-6ed0-4fb3-b121-758921dcc44e"
UUID_B = "aaaaaaaa-1111-2222-3333-aaaaaaaaaaaa"


class TestCellToEscoUuid:
    def test_lowercases_uuid(self):
        assert cell_to_esco_uuid(UUID_A.upper()) == UUID_A

    def test_extracts_from_uri(self):
        u = f"http://data.europa.eu/esco/skill/{UUID_A}"
        assert cell_to_esco_uuid(u) == UUID_A

    def test_returns_none_for_invalid(self):
        assert cell_to_esco_uuid("garbage") is None
        assert cell_to_esco_uuid("") is None
        assert cell_to_esco_uuid(None) is None  # type: ignore[arg-type]


class TestParseUuidHistoryCell:
    def test_parses_multiple_lines(self):
        raw = f"{UUID_A}\n{UUID_B}\n"
        assert parse_uuid_history_cell(raw) == [UUID_A, UUID_B]

    def test_handles_crlf(self):
        raw = f"{UUID_A}\r\n{UUID_B}"
        assert parse_uuid_history_cell(raw) == [UUID_A, UUID_B]

    def test_strips_trailing_commas(self):
        # "..." cell with trailing comma should still parse.
        assert parse_uuid_history_cell(f"{UUID_A},") == [UUID_A]

    def test_skips_garbage_lines(self):
        raw = f"{UUID_A}\nnot-a-uuid\n{UUID_B}"
        assert parse_uuid_history_cell(raw) == [UUID_A, UUID_B]

    def test_blank_returns_empty(self):
        assert parse_uuid_history_cell("") == []
        assert parse_uuid_history_cell(None) == []


class TestLoadSkillsUuidTabiyaMap:
    def _write_csv(self, tmp_path: Path) -> Path:
        path = tmp_path / "skills.csv"
        path.write_text(
            "ID,PREFERREDLABEL,UUIDHISTORY,ORIGINURI\n"
            f'tabiyahex000000000000000a,Python,"{UUID_A}\n{UUID_B}",http://example/{UUID_A}\n'
            "tabiyahex000000000000000b,Other,,\n",
            encoding="utf-8",
        )
        return path

    def test_round_trip(self, tmp_path):
        p = self._write_csv(tmp_path)
        m = load_skills_uuid_tabiya_map(str(p))
        # Tabiya-by-ESCO mapping populated from origin or uuid history
        assert m.tabiya_by_esco[UUID_A] == "tabiyahex000000000000000a"
        # Preferred label keyed by tabiya id and esco uuids
        assert m.preferred_label["tabiyahex000000000000000a"] == "Python"
        assert m.preferred_label[UUID_A] == "Python"
        # Helper API
        assert m.tabiya_for_esco(UUID_A) == "tabiyahex000000000000000a"
        assert m.label_for_skill_key(UUID_A) == "Python"

    def test_missing_file_returns_empty(self, tmp_path):
        m = load_skills_uuid_tabiya_map(str(tmp_path / "does-not-exist.csv"))
        assert m.preferred_label == {}
        assert m.tabiya_by_esco == {}


# ---------------------------------------------------------------------------
# occupation_to_skills_mapping helpers
# ---------------------------------------------------------------------------


class TestCellToTaxonomyId:
    def test_returns_esco_uuid_when_possible(self):
        assert cell_to_taxonomy_id(UUID_A) == UUID_A

    def test_returns_24_char_hex_id(self):
        # 24 lowercase hex chars (typical Tabiya ID)
        tid = "abcdef0123456789abcdef01"
        assert cell_to_taxonomy_id(tid) == tid

    def test_returns_none_for_invalid(self):
        assert cell_to_taxonomy_id("garbage") is None
        assert cell_to_taxonomy_id("") is None


class TestEscoSkillUriForUuid:
    def test_lowercases_and_prefixes(self):
        assert (
            esco_skill_uri_for_uuid(UUID_A.upper()) == f"http://data.europa.eu/esco/skill/{UUID_A}"
        )


# ---------------------------------------------------------------------------
# is_active.parse_closing_date
# ---------------------------------------------------------------------------


class TestParseDate:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("2026-06-12", date(2026, 6, 12)),
            ("12-06-2026", date(2026, 6, 12)),
            ("12/06/2026", date(2026, 6, 12)),
            ("2026/06/12", date(2026, 6, 12)),
            ("12 Jun 2026", date(2026, 6, 12)),
            ("12 June 2026", date(2026, 6, 12)),
        ],
    )
    def test_recognised_formats(self, raw, expected):
        assert _parse_date(raw) == expected

    def test_iso_with_z_suffix(self):
        assert _parse_date("2026-06-12T15:30:00Z") == date(2026, 6, 12)

    def test_datetime_object_via_str(self):
        dt = datetime(2026, 6, 12, tzinfo=timezone.utc)
        assert _parse_date(dt) == date(2026, 6, 12)

    def test_blank_or_none(self):
        assert _parse_date(None) is None
        assert _parse_date("") is None
        assert _parse_date("   ") is None

    def test_garbage_returns_none(self):
        assert _parse_date("not-a-date") is None


# ---------------------------------------------------------------------------
# sync_is_active_from_closing_date with a fake collection
# ---------------------------------------------------------------------------


class FakeCollection:
    """Minimal in-memory pymongo Collection for sync tests."""

    def __init__(self, docs):
        self._docs = list(docs)
        self.bulk_write_calls: list[Any] = []

    def find(self, _filter, _projection=None) -> Iterator[dict[str, Any]]:
        for d in self._docs:
            yield dict(d)

    def bulk_write(self, ops, ordered=True):
        # Apply ops to internal docs and record the call.
        self.bulk_write_calls.append(ops)
        modified = 0
        for op in ops:
            # We only emit UpdateOne({_id: x}, {"$set": {...}})
            f = op._filter
            new = op._doc.get("$set", {}) if hasattr(op, "_doc") else {}
            for d in self._docs:
                if d.get("_id") == f.get("_id"):
                    if any(d.get(k) != v for k, v in new.items()):
                        d.update(new)
                        modified += 1
                    break

        class _Result:
            modified_count = modified

        return _Result()


class TestSyncIsActiveFromClosingDate:
    def test_marks_active_inactive_relative_to_today(self):
        today = date(2026, 6, 12)
        coll = FakeCollection(
            [
                {
                    "_id": "future",
                    "is_active": False,
                    "classifier_metadata": {"closing_date": "2026-12-31"},
                },
                {
                    "_id": "past",
                    "is_active": True,
                    "classifier_metadata": {"closing_date": "2026-01-01"},
                },
                {
                    "_id": "stable",
                    "is_active": True,
                    "classifier_metadata": {"closing_date": "2026-12-31"},
                },
            ]
        )
        stats = sync_is_active_from_closing_date(cast(Collection, coll), today=today, dry_run=False)
        assert stats.total == 3
        assert stats.parsed == 3
        assert stats.will_set_true == 1
        assert stats.will_set_false == 1
        assert stats.unchanged == 1
        assert stats.modified == 2
        # Verify in-memory state
        by_id = {d["_id"]: d for d in coll._docs}
        assert by_id["future"]["is_active"] is True
        assert by_id["past"]["is_active"] is False
        assert by_id["stable"]["is_active"] is True

    def test_dry_run_does_not_persist(self):
        today = date(2026, 6, 12)
        coll = FakeCollection(
            [
                {
                    "_id": "x",
                    "is_active": False,
                    "classifier_metadata": {"closing_date": "2026-12-31"},
                }
            ]
        )
        stats = sync_is_active_from_closing_date(cast(Collection, coll), today=today, dry_run=True)
        assert stats.will_set_true == 1
        assert stats.modified == 0
        assert coll.bulk_write_calls == []
        # Document untouched
        assert coll._docs[0]["is_active"] is False

    def test_missing_policy_keep(self):
        # No closing_date and missing_policy='keep' (default) -> unchanged.
        today = date(2026, 6, 12)
        coll = FakeCollection([{"_id": "x", "is_active": True, "classifier_metadata": {}}])
        stats = sync_is_active_from_closing_date(
            cast(Collection, coll), today=today, missing_policy="keep", dry_run=False
        )
        assert stats.missing_or_bad == 1
        assert stats.unchanged == 1
        assert stats.modified == 0

    def test_missing_policy_inactive(self):
        today = date(2026, 6, 12)
        coll = FakeCollection([{"_id": "x", "is_active": True, "classifier_metadata": {}}])
        stats = sync_is_active_from_closing_date(
            cast(Collection, coll), today=today, missing_policy="inactive", dry_run=False
        )
        assert stats.missing_or_bad == 1
        assert stats.will_set_false == 1
        assert stats.modified == 1
        assert coll._docs[0]["is_active"] is False
