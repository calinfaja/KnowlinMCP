"""Tests for search output formatters."""

from __future__ import annotations

import json

import pytest

from knowlin_mcp.search import (
    FORMATTERS,
    format_compact,
    format_detailed,
    format_inject,
    format_json,
    format_single_entry,
)


@pytest.fixture
def sample_results():
    return [
        {
            "id": "abc-123",
            "title": "JWT Validation",
            "type": "warning",
            "insight": "Always validate server-side",
            "date": "2026-01-15",
            "score": 0.85,
            "priority": "high",
            "source": "conv:2026-01-15",
            "keywords": ["auth", "jwt"],
        },
        {
            "id": "def-456",
            "title": "Redis Caching",
            "type": "pattern",
            "insight": "Use TTL with sliding window",
            "date": "2026-02-01",
            "score": 0.72,
            "source": "doc:cache.md",
        },
    ]


class TestFormatCompact:
    def test_includes_type_and_title(self, sample_results):
        output = format_compact(sample_results)
        assert "[warning]" in output
        assert "JWT Validation" in output

    def test_includes_date_and_id_prefix(self, sample_results):
        output = format_compact(sample_results)
        assert "2026-01-15" in output
        assert "abc-123" in output[:100]

    def test_truncates_long_insight(self):
        results = [
            {
                "title": "Test",
                "type": "finding",
                "insight": "x" * 200,
                "date": "2026-01-01",
                "id": "t1",
            },
        ]
        output = format_compact(results)
        assert "..." in output

    def test_empty_results(self):
        assert format_compact([]) == "No results found."

    def test_includes_source_label(self):
        results = [
            {
                "title": "Test",
                "type": "finding",
                "_source": "kb",
                "date": "2026-01-01",
                "id": "t1",
                "score": 0.5,
            },
        ]
        output = format_compact(results)
        assert "kb:finding" in output

    def test_includes_score(self, sample_results):
        output = format_compact(sample_results)
        assert "85%" in output


class TestFormatDetailed:
    def test_includes_all_metadata(self, sample_results):
        output = format_detailed(sample_results)
        assert "JWT Validation" in output
        assert "85%" in output
        assert "Type: warning" in output
        assert "Priority: high" in output
        assert "Source: conv:2026-01-15" in output
        assert "Keywords: auth, jwt" in output

    def test_includes_insight(self, sample_results):
        output = format_detailed(sample_results)
        assert "Always validate server-side" in output

    def test_empty_results(self):
        assert format_detailed([]) == "No results found."

    def test_includes_source_label(self):
        results = [
            {
                "title": "Test",
                "type": "finding",
                "_source": "docs",
                "date": "2026-01-01",
                "id": "t1",
                "score": 0.7,
            },
        ]
        output = format_detailed(results)
        assert "[docs]" in output


class TestFormatInject:
    def test_includes_header(self, sample_results):
        output = format_inject(sample_results)
        assert "RELEVANT PRIOR KNOWLEDGE" in output

    def test_low_score_still_included(self):
        """Low-score results are no longer filtered -- reranking handles relevance."""
        results = [{"title": "Low score", "score": 0.1}]
        output = format_inject(results)
        assert "Low score" in output

    def test_empty_results(self):
        output = format_inject([])
        assert "No relevant prior knowledge" in output

    def test_high_score_included(self, sample_results):
        output = format_inject(sample_results)
        assert "JWT Validation" in output


class TestFormatJson:
    def test_valid_json(self, sample_results):
        output = format_json(sample_results)
        parsed = json.loads(output)
        assert len(parsed) == 2
        assert parsed[0]["id"] == "abc-123"


class TestFormatSingleEntry:
    def test_includes_all_fields(self):
        entry = {
            "id": "abc-123",
            "title": "Test Entry",
            "type": "warning",
            "priority": "high",
            "date": "2026-01-15",
            "source": "conv:2026-01-15",
            "branch": "main",
            "insight": "Detailed insight text",
            "keywords": ["auth", "jwt"],
            "pinned": True,
            "related_to": ["def-456"],
        }
        output = format_single_entry(entry)
        assert "Test Entry" in output
        assert "warning" in output
        assert "high" in output
        assert "2026-01-15" in output
        assert "main" in output
        assert "Detailed insight text" in output
        assert "auth, jwt" in output
        assert "Pinned: yes" in output
        assert "def-456" in output


class TestFormatterRegistry:
    def test_all_formats_registered(self):
        assert set(FORMATTERS.keys()) == {"compact", "detailed", "inject", "json"}
