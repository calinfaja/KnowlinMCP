"""Property-based tests using Hypothesis.

Verify invariants that must hold for ALL inputs:
- Search determinism
- Score monotonicity
- Limit monotonicity
- No crashes on adversarial input
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from kln_knowledge.db import KnowledgeDB


@pytest.fixture(scope="module")
def prop_db(tmp_path_factory):
    """Shared KB for property tests (module-scoped for speed)."""
    tmp = tmp_path_factory.mktemp("prop_db")
    project = tmp / "proj"
    project.mkdir()
    (project / ".knowledge-db").mkdir()

    db = KnowledgeDB(str(project))
    db.batch_add(
        [
            {"title": "Authentication with JWT tokens", "insight": "Use JWT for stateless auth"},
            {"title": "Database connection pooling", "insight": "Pool size tuning for OLTP"},
            {"title": "Docker container optimization", "insight": "Multi-stage builds reduce size"},
            {"title": "Python async programming guide", "insight": "Asyncio event loop patterns"},
            {"title": "Redis caching strategies explained", "insight": "TTL and eviction policies"},
        ],
        check_duplicates=False,
    )
    return db


# Use printable ASCII strings to avoid embedding model edge cases with unicode
_query_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
    min_size=3,
    max_size=50,
).filter(lambda s: len(s.strip()) >= 3)


class TestSearchDeterminism:
    """Same query must always return same results."""

    @given(query=_query_strategy)
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_deterministic_results(self, prop_db, query):
        r1 = prop_db.search(query, limit=3, rerank=False)
        r2 = prop_db.search(query, limit=3, rerank=False)
        ids1 = [r["id"] for r in r1]
        ids2 = [r["id"] for r in r2]
        assert ids1 == ids2


class TestScoreOrdering:
    """Results must be sorted by score descending."""

    @given(query=_query_strategy)
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_scores_descending(self, prop_db, query):
        results = prop_db.search(query, limit=5, rerank=False)
        if len(results) < 2:
            return
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), f"Not sorted: {scores}"


class TestLimitMonotonicity:
    """Larger limit must return at least as many results as smaller limit."""

    @given(
        query=_query_strategy,
        k_small=st.integers(min_value=1, max_value=2),
        k_large=st.integers(min_value=3, max_value=5),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_larger_k_returns_more(self, prop_db, query, k_small, k_large):
        small = prop_db.search(query, limit=k_small, rerank=False)
        large = prop_db.search(query, limit=k_large, rerank=False)
        assert len(large) >= len(small)


class TestAdversarialInput:
    """Search must not crash on weird inputs."""

    @pytest.mark.parametrize(
        "query",
        [
            "",
            " ",
            "a",
            "a" * 10000,
            "```python\nprint('hello')\n```",
            "SELECT * FROM users;",
            "<script>alert(1)</script>",
            "\x00\x01\x02",
            "日本語クエリ",
            "🔥🎉🚀",
        ],
    )
    def test_no_crash_on_weird_input(self, prop_db, query):
        # Must not raise - can return empty list
        results = prop_db.search(query, limit=3, rerank=False)
        assert isinstance(results, list)


class TestRRFMath:
    """Verify RRF scoring math properties."""

    def test_rrf_single_rank(self):
        assert KnowledgeDB.rrf_score([1], k=60) == pytest.approx(1.0 / 61)

    def test_rrf_two_ranks(self):
        expected = 1.0 / 61 + 1.0 / 62
        assert KnowledgeDB.rrf_score([1, 2], k=60) == pytest.approx(expected)

    def test_rrf_skips_zero_ranks(self):
        """Rank 0 means "not ranked" and should be skipped."""
        assert KnowledgeDB.rrf_score([0, 1], k=60) == pytest.approx(1.0 / 61)

    def test_rrf_higher_rank_is_better(self):
        """Rank 1 should contribute more than rank 5."""
        score_high = KnowledgeDB.rrf_score([1], k=60)
        score_low = KnowledgeDB.rrf_score([5], k=60)
        assert score_high > score_low

    def test_rrf_two_rankings_beat_one(self):
        """An entry ranked in both dense and sparse should score higher."""
        both = KnowledgeDB.rrf_score([1, 1], k=60)
        one = KnowledgeDB.rrf_score([1, 0], k=60)
        assert both > one
