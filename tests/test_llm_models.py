"""Unit tests for ``app.llm_models``: the shared LLM JSON parse boundary."""

from __future__ import annotations

import pytest

from app.llm.errors import LlmParseError
from app.llm_models import parse_llm_json_response, strip_json_fence


class TestStripJsonFence:
    def test_no_fence_returns_stripped(self):
        assert strip_json_fence('  {"a": 1}  ') == '{"a": 1}'

    def test_strips_json_fence(self):
        s = '```json\n{"a": 1}\n```'
        assert strip_json_fence(s) == '{"a": 1}'

    def test_strips_bare_fence(self):
        s = '```\n{"a": 1}\n```'
        assert strip_json_fence(s) == '{"a": 1}'

    def test_case_insensitive_json_tag(self):
        assert strip_json_fence('```JSON\n{"a":1}\n```') == '{"a":1}'


class TestParseLlmJsonResponse:
    def test_happy_path(self):
        assert parse_llm_json_response('{"a": 1}') == {"a": 1}

    def test_handles_fence(self):
        assert parse_llm_json_response('```json\n{"a": 1}\n```') == {"a": 1}

    def test_repairs_trailing_comma_by_default(self):
        assert parse_llm_json_response('{"a": 1,}') == {"a": 1}

    def test_no_repair_raises_when_disabled(self):
        with pytest.raises(LlmParseError, match="not valid JSON"):
            parse_llm_json_response('{"a": 1,}', repair_on_failure=False)

    def test_unrepairable_json_raises(self):
        with pytest.raises(LlmParseError, match="not valid JSON"):
            parse_llm_json_response("not json at all")

    def test_top_level_must_be_object(self):
        with pytest.raises(LlmParseError, match="must be an object"):
            parse_llm_json_response("[1, 2, 3]")
