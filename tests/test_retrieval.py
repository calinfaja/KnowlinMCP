"""Integration tests - golden query regression with real embeddings.

These load real fastembed models and verify retrieval quality.
Run with: pytest --integration tests/test_retrieval.py -v
"""

from __future__ import annotations

import pytest

from knowlin_mcp.db import KnowledgeDB

# Golden corpus: diverse topics for testing retrieval quality
GOLDEN_CORPUS = [
    {
        "title": "BLE power optimization techniques",
        "insight": (
            "Nordic nRF52 sleep modes reduce current draw to 2uA."
            " Use system OFF mode when possible."
        ),
    },
    {
        "title": "OAuth2 token refresh patterns",
        "insight": (
            "Use refresh tokens with sliding window expiry. Rotate refresh tokens on each use."
        ),
    },
    {
        "title": "Python asyncio event loop guide",
        "insight": (
            "asyncio event loop handles IO bound concurrency. Never block with synchronous calls."
        ),
    },
    {
        "title": "RRF fusion for hybrid search",
        "insight": (
            "Reciprocal Rank Fusion combines dense and sparse rankings"
            " with k=60 smoothing constant."
        ),
    },
    {
        "title": "Docker multi-stage build patterns",
        "insight": (
            "Order Dockerfile instructions by change frequency."
            " Use multi-stage to reduce image size."
        ),
    },
    {
        "title": "JWT authentication security practices",
        "insight": (
            "Always validate JWT server-side. Check issuer, audience, and expiry."
            " Never trust client tokens."
        ),
    },
    {
        "title": "PostgreSQL connection pool sizing",
        "insight": (
            "Pool size = (2 * cpu_count) + disk_spindles for OLTP."
            " Monitor active vs idle connections."
        ),
    },
    {
        "title": "Redis caching with TTL strategies",
        "insight": (
            "Set appropriate TTL to prevent stale data."
            " Use cache-aside pattern for read-heavy workloads."
        ),
    },
]

# Golden queries: each should find its matching corpus entry at rank 1
GOLDEN_QUERIES = [
    {
        "query": "bluetooth battery drain embedded firmware",
        "expected_title": "BLE power optimization techniques",
    },
    {
        "query": "oauth session expiry refresh token rotation",
        "expected_title": "OAuth2 token refresh patterns",
    },
    {
        "query": "python concurrent IO networking",
        "expected_title": "Python asyncio event loop guide",
    },
    {
        "query": "combine search results ranking algorithm",
        "expected_title": "RRF fusion for hybrid search",
    },
    {
        "query": "dockerfile build cache layers optimization",
        "expected_title": "Docker multi-stage build patterns",
    },
    {
        "query": "token validation authentication security",
        "expected_title": "JWT authentication security practices",
    },
    {
        "query": "database connection pool performance tuning",
        "expected_title": "PostgreSQL connection pool sizing",
    },
    {
        "query": "cache eviction stale data TTL",
        "expected_title": "Redis caching with TTL strategies",
    },
]

REQUIRED_HIT_AT_1 = 0.75  # 6/8 queries must hit at rank 1
REQUIRED_HIT_AT_3 = 0.875  # 7/8 queries must be in top 3


@pytest.fixture(scope="module")
def golden_db(tmp_path_factory):
    """Create KB with golden corpus (real embeddings)."""
    tmp = tmp_path_factory.mktemp("golden")
    project = tmp / "proj"
    project.mkdir()
    (project / ".knowledge-db").mkdir()

    db = KnowledgeDB(str(project))
    db.batch_add(GOLDEN_CORPUS, check_duplicates=False)
    return db


@pytest.mark.integration
class TestGoldenRetrieval:
    """Golden query regression tests with real BGE-small embeddings."""

    @pytest.mark.parametrize(
        "case",
        GOLDEN_QUERIES,
        ids=[q["query"][:40] for q in GOLDEN_QUERIES],
    )
    def test_golden_hit_at_3(self, golden_db, case):
        """Each golden query must find its expected entry in top 3."""
        results = golden_db.search(case["query"], limit=3, rerank=False)
        titles = [r["title"] for r in results]
        assert case["expected_title"] in titles, (
            f"Query '{case['query']}' expected '{case['expected_title']}' "
            f"in top 3, got: {titles}"
        )

    def test_aggregate_hit_at_1(self, golden_db):
        """Aggregate: Hit@1 across all golden queries must meet threshold."""
        hits = 0
        for case in GOLDEN_QUERIES:
            results = golden_db.search(case["query"], limit=1, rerank=False)
            if results and results[0]["title"] == case["expected_title"]:
                hits += 1

        hit_rate = hits / len(GOLDEN_QUERIES)
        assert hit_rate >= REQUIRED_HIT_AT_1, (
            f"Hit@1 = {hit_rate:.0%} ({hits}/{len(GOLDEN_QUERIES)}), "
            f"threshold = {REQUIRED_HIT_AT_1:.0%}"
        )

    def test_aggregate_hit_at_3(self, golden_db):
        """Aggregate: Hit@3 across all golden queries must meet threshold."""
        hits = 0
        for case in GOLDEN_QUERIES:
            results = golden_db.search(case["query"], limit=3, rerank=False)
            titles = [r["title"] for r in results]
            if case["expected_title"] in titles:
                hits += 1

        hit_rate = hits / len(GOLDEN_QUERIES)
        assert hit_rate >= REQUIRED_HIT_AT_3, (
            f"Hit@3 = {hit_rate:.0%} ({hits}/{len(GOLDEN_QUERIES)}), "
            f"threshold = {REQUIRED_HIT_AT_3:.0%}"
        )


@pytest.mark.integration
class TestRetrievalWithRanx:
    """IR metrics using ranx (Recall@k, MRR@k, NDCG@k)."""

    def test_retrieval_metrics(self, golden_db):
        """Compute standard IR metrics on golden dataset."""
        try:
            from ranx import Qrels, Run, evaluate
        except ImportError:
            pytest.skip("ranx not installed")

        # Build qrels (ground truth)
        qrels_dict = {}
        for i, case in enumerate(GOLDEN_QUERIES):
            qid = f"q{i:03d}"
            # Find the entry ID for the expected title
            for entry in golden_db._entries:
                if entry.get("title") == case["expected_title"]:
                    qrels_dict[qid] = {entry["id"]: 1}
                    break

        # Build run (system output)
        run_dict = {}
        for i, case in enumerate(GOLDEN_QUERIES):
            qid = f"q{i:03d}"
            results = golden_db.search(case["query"], limit=5, rerank=False)
            run_dict[qid] = {r["id"]: float(r["score"]) for r in results}

        qrels = Qrels(qrels_dict)
        run = Run(run_dict)

        scores = evaluate(qrels, run, ["recall@5", "mrr@5", "ndcg@5"])

        # Report metrics (visible in pytest -v output)
        print(f"\n  Recall@5: {scores['recall@5']:.3f}")
        print(f"  MRR@5:    {scores['mrr@5']:.3f}")
        print(f"  NDCG@5:   {scores['ndcg@5']:.3f}")

        # Minimum quality gates
        assert scores["recall@5"] >= 0.70, f"Recall@5 = {scores['recall@5']:.3f} < 0.70"
        assert scores["mrr@5"] >= 0.60, f"MRR@5 = {scores['mrr@5']:.3f} < 0.60"
