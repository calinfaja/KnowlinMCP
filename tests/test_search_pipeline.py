"""Search pipeline tests - cover hybrid fusion, filtering, and reranking paths.

Uses real embeddings for dense search but tests sparse/rerank code paths
that are normally skipped when models aren't loaded.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from knowlin_mcp.db import KnowledgeDB


@pytest.fixture
def search_db(tmp_path):
    """Create a KB with entries that have diverse types and dates."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".knowledge-db").mkdir()

    db = KnowledgeDB(str(project))
    db.batch_add(
        [
            {
                "title": "JWT token validation server-side",
                "insight": "Always validate JWT tokens server-side, never trust client",
                "type": "warning",
                "date": "2026-01-15",
                "branch": "main",
            },
            {
                "title": "Redis caching best practices",
                "insight": "Use Redis with TTL for session caching, avoid thundering herd",
                "type": "pattern",
                "date": "2026-02-01",
                "branch": "feature/cache",
            },
            {
                "title": "Database connection pool sizing",
                "insight": "Pool size = (2 * cpu_count) + disk_spindles for OLTP workloads",
                "type": "solution",
                "date": "2026-03-01",
                "branch": "main",
                "priority": "high",
            },
            {
                "title": "Python asyncio event loop best practices",
                "insight": "Never block the event loop with synchronous IO calls",
                "type": "warning",
                "date": "2025-12-01",
                "branch": "main",
            },
            {
                "title": "Docker multi-stage build optimization",
                "insight": "Use multi-stage builds to reduce image size by 10x",
                "type": "pattern",
                "date": "2026-01-20",
                "branch": "main",
            },
        ],
        check_duplicates=False,
    )
    return db


class TestDenseSearch:
    """Dense (semantic) search returns relevant results."""

    def test_semantic_match(self, search_db):
        results = search_db.search("authentication token security", rerank=False, limit=3)
        assert len(results) > 0
        # JWT entry should be top result for auth-related query
        titles = [r["title"] for r in results]
        assert "JWT token validation server-side" in titles[:2]

    def test_empty_db_returns_empty(self, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        (project / ".knowledge-db").mkdir()
        db = KnowledgeDB(str(project))
        assert db.search("anything", rerank=False) == []

    def test_results_sorted_by_score_descending(self, search_db):
        results = search_db.search("database performance", rerank=False, limit=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_limit_respected(self, search_db):
        results = search_db.search("best practices", rerank=False, limit=2)
        assert len(results) <= 2


class TestPostRRFFiltering:
    """Post-RRF filters on date, type, and branch."""

    def test_date_from_filter(self, search_db):
        results = search_db.search(
            "best practices", rerank=False, limit=10, date_from="2026-02-01"
        )
        for r in results:
            assert r.get("date", "") >= "2026-02-01"

    def test_date_to_filter(self, search_db):
        results = search_db.search(
            "best practices", rerank=False, limit=10, date_to="2026-01-31"
        )
        for r in results:
            assert r.get("date", "") <= "2026-01-31"

    def test_date_range_filter(self, search_db):
        results = search_db.search(
            "optimization", rerank=False, limit=10,
            date_from="2026-01-01", date_to="2026-01-31",
        )
        for r in results:
            assert "2026-01-01" <= r.get("date", "") <= "2026-01-31"

    def test_type_filter(self, search_db):
        results = search_db.search(
            "practices", rerank=False, limit=10, entry_type="warning"
        )
        for r in results:
            assert r["type"] == "warning"

    def test_branch_filter(self, search_db):
        results = search_db.search(
            "caching", rerank=False, limit=10, branch="feature/cache"
        )
        for r in results:
            assert r["branch"] == "feature/cache"


class TestRerankPath:
    """Test the cross-encoder reranking code path with a mock."""

    def test_rerank_reorders_results(self, search_db):
        """Mock the reranker to verify the reranking code path executes."""
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [0.1, 0.2, 0.3, 0.9, 0.5]

        # Patch the private attribute (property reads from _reranker)
        search_db._reranker = mock_reranker
        try:
            results = search_db.search("test query", rerank=True, limit=3)
        finally:
            search_db._reranker = None

        mock_reranker.rerank.assert_called_once()
        assert len(results) <= 3
        for r in results:
            assert "rerank_score" in r

    def test_rerank_none_falls_back(self, search_db):
        """When reranker is None, search still works (dense-only)."""
        with patch.object(search_db, "_reranker", None):
            results = search_db.search("database connection", rerank=True, limit=3)
        assert len(results) > 0


class TestSparseSearchPath:
    """Test the BM42 sparse search code path with mocked sparse vectors."""

    def test_sparse_search_with_vectors(self, search_db):
        """Inject fake sparse vectors to exercise the sparse search path."""
        # Add fake sparse vectors for all entries
        for row_idx in range(len(search_db._entries)):
            search_db._sparse_vectors[row_idx] = {"1": 0.5, "2": 0.3, "3": 0.8}

        # Mock sparse model to return a query vector
        mock_sparse = MagicMock()
        sparse_result = MagicMock()
        sparse_result.indices = [1, 3]
        sparse_result.values = [0.6, 0.9]
        mock_sparse.embed.return_value = iter([sparse_result])

        with patch.object(search_db, "_sparse_model", mock_sparse):
            results = search_db._sparse_search("test query", limit=5)

        assert len(results) > 0
        # Results are sorted by score descending
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_fusion_with_both_rankings(self, search_db):
        """Test RRF fusion when both dense and sparse return results."""
        # Add sparse vectors
        for row_idx in range(len(search_db._entries)):
            search_db._sparse_vectors[row_idx] = {"10": 0.5}

        mock_sparse = MagicMock()
        sparse_result = MagicMock()
        sparse_result.indices = [10]
        sparse_result.values = [0.5]
        mock_sparse.embed.return_value = iter([sparse_result])

        with patch.object(search_db, "_sparse_model", mock_sparse):
            results = search_db.search("database performance", rerank=False, limit=3)

        assert len(results) > 0
        # Results should have search metadata
        for r in results:
            assert "_search_meta" in r
            meta = r["_search_meta"]
            assert "dense_rank" in meta
            assert "sparse_rank" in meta


class TestDBMethods:
    """Test auxiliary DB methods that need coverage."""

    def test_search_by_date(self, search_db):
        results = search_db.search_by_date("2026-01-01", "2026-01-31")
        assert len(results) > 0
        for r in results:
            assert "2026-01-01" <= r.get("date", "") <= "2026-01-31"

    def test_get_timeline(self, search_db):
        results = search_db.get_timeline("2026-01-15")
        assert len(results) >= 1
        assert results[0].get("date") == "2026-01-15"

    def test_get_entry_by_id(self, search_db):
        # Get an ID from the entries
        entry_id = search_db._entries[0].get("id")
        result = search_db.get(entry_id)
        assert result is not None
        assert result["id"] == entry_id

    def test_get_nonexistent_entry(self, search_db):
        result = search_db.get("nonexistent-id-12345")
        assert result is None

    def test_get_related_bidirectional(self, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        (project / ".knowledge-db").mkdir()

        db = KnowledgeDB(str(project))
        db.batch_add(
            [
                {
                    "id": "a1",
                    "title": "Entry A links to B",
                    "insight": "A links forward",
                    "related_to": ["b1"],
                },
                {
                    "id": "b1",
                    "title": "Entry B standalone here",
                    "insight": "B has no explicit links",
                },
            ],
            check_duplicates=False,
        )

        related = db.get_related("a1")
        assert any(r["id"] == "b1" for r in related)

        # Bidirectional: searching from B should find A
        related_b = db.get_related("b1")
        assert any(r["id"] == "a1" for r in related_b)

    def test_count(self, search_db):
        assert search_db.count() == 5

    def test_stats_returns_required_fields(self, search_db):
        st = search_db.stats()
        assert "count" in st
        assert "size_bytes" in st
        assert "backend" in st
        assert st["count"] == 5
