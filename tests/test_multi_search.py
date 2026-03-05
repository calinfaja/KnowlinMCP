"""Tests for multi-source search with weighted RRF fusion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from knowlin_mcp.multi_search import MultiSourceSearch


@pytest.fixture
def mock_kb_entries():
    return [
        {"title": "KB Entry 1", "insight": "Authentication pattern", "score": 0.9},
        {"title": "KB Entry 2", "insight": "Database setup", "score": 0.7},
    ]


@pytest.fixture
def mock_session_entries():
    return [
        {"title": "Session Finding", "insight": "Debug auth error", "score": 0.85},
    ]


@pytest.fixture
def mock_doc_entries():
    return [
        {"title": "Doc Chunk", "insight": "API authentication docs", "score": 0.8},
    ]


class TestMultiSourceSearch:
    """Tests for MultiSourceSearch."""

    def test_search_returns_list(self, tmp_path):
        ms = MultiSourceSearch(str(tmp_path))
        results = ms.search("test query")
        assert isinstance(results, list)

    def test_search_empty_when_no_stores(self, tmp_path):
        ms = MultiSourceSearch(str(tmp_path))
        results = ms.search("anything")
        assert results == []

    @patch("knowlin_mcp.multi_search.KnowledgeDB")
    def test_search_with_single_source(self, mock_db_cls, tmp_path, mock_kb_entries):
        mock_db = MagicMock()
        mock_db.count.return_value = 2
        mock_db.search.return_value = mock_kb_entries
        mock_db_cls.return_value = mock_db

        ms = MultiSourceSearch(str(tmp_path))
        results = ms.search("auth", sources=["kb"], limit=5)

        assert len(results) > 0
        assert all("_source" in r for r in results)
        assert all(r["_source"] == "kb" for r in results)

    @patch("knowlin_mcp.multi_search.KnowledgeDB")
    def test_search_applies_source_weights(self, mock_db_cls, tmp_path):
        mock_db = MagicMock()
        mock_db.count.return_value = 1
        mock_db.search.return_value = [
            {"title": "Result", "insight": "content", "score": 0.8}
        ]
        mock_db_cls.return_value = mock_db

        ms = MultiSourceSearch(str(tmp_path))
        results = ms.search("error trace", sources=["kb"], limit=5)

        # Should have weighted scores
        assert len(results) > 0
        assert "score" in results[0]

    @patch("knowlin_mcp.multi_search.KnowledgeDB")
    def test_deduplicates_by_title(self, mock_db_cls, tmp_path):
        mock_db = MagicMock()
        mock_db.count.return_value = 2
        # Same title from two "different" sources (simulated by returning same results)
        mock_db.search.return_value = [
            {"title": "Same Title", "insight": "content A", "score": 0.9},
            {"title": "Same Title", "insight": "content B", "score": 0.7},
        ]
        mock_db_cls.return_value = mock_db

        ms = MultiSourceSearch(str(tmp_path))
        results = ms.search("query", sources=["kb"], limit=10)

        titles = [r["title"] for r in results]
        assert titles.count("Same Title") == 1

    @patch("knowlin_mcp.multi_search.KnowledgeDB")
    def test_results_sorted_by_score(self, mock_db_cls, tmp_path):
        mock_db = MagicMock()
        mock_db.count.return_value = 3
        mock_db.search.return_value = [
            {"title": "First", "insight": "a", "score": 0.9},
            {"title": "Second", "insight": "b", "score": 0.7},
            {"title": "Third", "insight": "c", "score": 0.5},
        ]
        mock_db_cls.return_value = mock_db

        ms = MultiSourceSearch(str(tmp_path))
        results = ms.search("query", sources=["kb"], limit=10)

        scores = [r.get("score", 0) for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch("knowlin_mcp.multi_search.KnowledgeDB")
    def test_respects_limit(self, mock_db_cls, tmp_path):
        mock_db = MagicMock()
        mock_db.count.return_value = 10
        mock_db.search.return_value = [
            {"title": f"Entry {i}", "insight": f"content {i}", "score": 0.9 - i * 0.1}
            for i in range(10)
        ]
        mock_db_cls.return_value = mock_db

        ms = MultiSourceSearch(str(tmp_path))
        results = ms.search("query", sources=["kb"], limit=3)

        assert len(results) <= 3

    @patch("knowlin_mcp.multi_search.KnowledgeDB")
    def test_search_meta_contains_intent(self, mock_db_cls, tmp_path):
        mock_db = MagicMock()
        mock_db.count.return_value = 1
        mock_db.search.return_value = [
            {"title": "Result", "insight": "content"}
        ]
        mock_db_cls.return_value = mock_db

        ms = MultiSourceSearch(str(tmp_path))
        results = ms.search("how to setup auth", sources=["kb"])

        assert len(results) > 0
        meta = results[0].get("_search_meta", {})
        assert "intent" in meta


class TestAvailableSources:
    """Tests for available_sources()."""

    def test_no_sources_when_empty(self, tmp_path):
        ms = MultiSourceSearch(str(tmp_path))
        assert ms.available_sources() == []

    @patch("knowlin_mcp.multi_search.KnowledgeDB")
    def test_lists_available_sources(self, mock_db_cls, tmp_path):
        mock_db = MagicMock()
        mock_db.count.return_value = 5
        mock_db_cls.return_value = mock_db

        ms = MultiSourceSearch(str(tmp_path))
        sources = ms.available_sources()
        assert "kb" in sources


class TestStats:
    """Tests for stats()."""

    def test_stats_empty(self, tmp_path):
        ms = MultiSourceSearch(str(tmp_path))
        st = ms.stats()
        assert "kb" in st
        assert st["kb"]["count"] == 0
        assert st["kb"]["available"] is False
