"""Performance benchmarks using pytest-benchmark.

Run with: pytest tests/test_benchmarks.py --benchmark-enable -v
Save baseline: pytest tests/test_benchmarks.py --benchmark-enable --benchmark-json=bench.json
Compare: pytest tests/test_benchmarks.py --benchmark-enable --benchmark-compare=bench.json
"""

from __future__ import annotations

import pytest

from knowlin_mcp.db import KnowledgeDB


@pytest.fixture(scope="module")
def bench_db(tmp_path_factory):
    """Create a KB with enough entries for meaningful benchmarks."""
    tmp = tmp_path_factory.mktemp("bench")
    project = tmp / "proj"
    project.mkdir()
    (project / ".knowledge-db").mkdir()

    db = KnowledgeDB(str(project))
    entries = [
        {
            "title": f"Knowledge entry number {i} about topic {i % 10}",
            "insight": (
                f"Detailed content for entry {i} covering various aspects"
                f" of software engineering topic {i % 10}"
            ),
        }
        for i in range(50)
    ]
    db.batch_add(entries, check_duplicates=False)
    return db


class TestSearchLatency:
    """Search latency benchmarks."""

    def test_dense_search_latency(self, benchmark, bench_db):
        """Target: <100ms for dense-only search."""
        result = benchmark(
            bench_db.search, "database optimization performance", limit=5, rerank=False
        )
        assert len(result) > 0

    def test_search_with_filters_latency(self, benchmark, bench_db):
        """Search with post-RRF filters."""
        result = benchmark(
            bench_db.search,
            "software engineering",
            limit=5,
            rerank=False,
            date_from="2026-01-01",
        )
        assert isinstance(result, list)


class TestBatchAddThroughput:
    """Batch embedding throughput."""

    def test_batch_add_10_entries(self, benchmark, tmp_path_factory):
        """Measure batch_add throughput for 10 entries."""
        def add_batch():
            tmp = tmp_path_factory.mktemp("batch")
            project = tmp / "proj"
            project.mkdir()
            (project / ".knowledge-db").mkdir()
            db = KnowledgeDB(str(project))
            entries = [
                {"title": f"Batch test entry {i}", "insight": f"Content for entry {i} with details"}
                for i in range(10)
            ]
            return db.batch_add(entries, check_duplicates=False)

        ids = benchmark.pedantic(add_batch, iterations=1, rounds=3)
        assert len(ids) == 10
