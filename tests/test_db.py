"""Tests for knowlin_mcp.db module.

Tests the hybrid search system with RRF fusion:
- Dense search (semantic similarity via BGE)
- Sparse search (keyword matching via SPLADE++)
- RRF scoring
- Semantic deduplication
- batch_add and remove_entries
- sub_store support
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta

import numpy as np
import pytest


class KeywordDenseModel:
    """Small fake dense model for deterministic unit tests."""

    def __init__(self, vectors_by_keyword):
        self.vectors_by_keyword = vectors_by_keyword
        self.default = np.zeros(len(next(iter(vectors_by_keyword.values()))), dtype=np.float32)

    def embed(self, texts):
        for text in texts:
            lowered = text.lower()
            for keyword, vector in self.vectors_by_keyword.items():
                if keyword in lowered:
                    yield vector.copy()
                    break
            else:
                yield self.default.copy()


# =============================================================================
# TestRRFScore
# =============================================================================


class TestRRFScore:
    """Tests for RRF (Reciprocal Rank Fusion) scoring."""

    def test_rrf_single_rank(self):
        from knowlin_mcp.db import KnowledgeDB

        score = KnowledgeDB.rrf_score([1], k=60)
        assert abs(score - (1.0 / 61)) < 0.0001

    def test_rrf_multiple_ranks(self):
        from knowlin_mcp.db import KnowledgeDB

        score = KnowledgeDB.rrf_score([1, 1], k=60)
        expected = 2.0 * (1.0 / 61)
        assert abs(score - expected) < 0.0001

    def test_rrf_ignores_zero_ranks(self):
        from knowlin_mcp.db import KnowledgeDB

        score1 = KnowledgeDB.rrf_score([1, 0], k=60)
        score2 = KnowledgeDB.rrf_score([1], k=60)
        assert abs(score1 - score2) < 0.0001

    def test_rrf_higher_rank_lower_score(self):
        from knowlin_mcp.db import KnowledgeDB

        score_rank1 = KnowledgeDB.rrf_score([1], k=60)
        score_rank10 = KnowledgeDB.rrf_score([10], k=60)
        assert score_rank1 > score_rank10

    def test_rrf_different_k_values(self):
        from knowlin_mcp.db import KnowledgeDB

        score_k60 = KnowledgeDB.rrf_score([1], k=60)
        score_k20 = KnowledgeDB.rrf_score([1], k=20)
        assert score_k20 > score_k60


# =============================================================================
# TestSparseIndexPersistence
# =============================================================================


class TestSparseIndexPersistence:
    """Tests for sparse index persistence."""

    def test_sparse_index_path_set(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(temp_kb_dir.parent))
        assert db.sparse_index_path == temp_kb_dir / "sparse_index.json"

    def test_loads_sparse_vectors_if_present(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        sparse_data = {
            "0": {"token1": 0.5, "token2": 0.3},
            "1": {"token3": 0.8},
        }
        (temp_kb_dir / "sparse_index.json").write_text(json.dumps(sparse_data))
        np.save(str(temp_kb_dir / "embeddings.npy"), np.zeros((2, 384)))
        (temp_kb_dir / "index.json").write_text(json.dumps({"id1": 0, "id2": 1}))

        db = KnowledgeDB(str(temp_kb_dir.parent))

        assert len(db._sparse_vectors) == 2
        assert 0 in db._sparse_vectors
        assert db._sparse_vectors[0]["token1"] == 0.5

    def test_loads_embeddings_when_size_within_limit(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        np.save(str(temp_kb_dir / "embeddings.npy"), embeddings)
        (temp_kb_dir / "index.json").write_text(json.dumps({"id1": 0}))
        (temp_kb_dir / "entries.jsonl").write_text(
            json.dumps({"id": "id1", "title": "Entry Title", "insight": "Entry insight"}) + "\n"
        )

        db = KnowledgeDB(str(temp_kb_dir.parent))

        np.testing.assert_array_equal(db._embeddings, embeddings)


# =============================================================================
# TestHybridSearch
# =============================================================================


class TestHybridSearch:
    """Tests for hybrid search with RRF fusion."""

    def test_search_returns_empty_for_empty_db(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(temp_kb_dir.parent))
        results = db.search("test query")
        assert results == []

    def test_search_returns_results_with_scores(self, kb_with_entries):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(kb_with_entries.parent))
        db.rebuild_index()
        results = db.search("BLE power")

        assert len(results) > 0
        assert all("score" in r for r in results)
        assert all(r["score"] > 0 for r in results)

    def test_search_includes_meta_breakdown(self, kb_with_entries):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(kb_with_entries.parent))
        db.rebuild_index()
        results = db.search("OAuth security")

        assert len(results) > 0
        assert "_search_meta" in results[0]
        meta = results[0]["_search_meta"]
        assert "dense_rank" in meta
        assert "rrf_score" in meta

    def test_search_respects_limit(self, kb_with_entries):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(kb_with_entries.parent))
        db.rebuild_index()
        results = db.search("python", limit=2)
        assert len(results) <= 2


# =============================================================================
# TestDenseSearch
# =============================================================================


class TestDenseSearch:
    """Tests for dense (semantic) search."""

    def test_dense_search_returns_tuples(self, kb_with_entries):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(kb_with_entries.parent))
        db.rebuild_index()
        results = db._dense_search("power optimization", limit=5)

        assert isinstance(results, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_dense_search_scores_sorted_descending(self, kb_with_entries):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(kb_with_entries.parent))
        db.rebuild_index()
        results = db._dense_search("test query", limit=10)

        if len(results) > 1:
            scores = [r[1] for r in results]
            assert scores == sorted(scores, reverse=True)


# =============================================================================
# TestStats
# =============================================================================


class TestStats:
    """Tests for database statistics."""

    def test_stats_includes_hybrid_backend(self, kb_with_entries):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(kb_with_entries.parent))
        db.rebuild_index()
        stats = db.stats()

        assert stats["backend"] == "fastembed-hybrid"
        assert "has_sparse_index" in stats
        assert "sparse_entries" in stats


# =============================================================================
# TestRebuildIndex
# =============================================================================


class TestRebuildIndex:
    """Tests for index rebuilding."""

    def test_rebuild_creates_dense_embeddings(self, kb_with_entries):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(kb_with_entries.parent))
        count = db.rebuild_index()

        assert count == 3
        assert db._embeddings is not None
        assert db._embeddings.shape[0] == 3

    def test_rebuild_saves_files(self, kb_with_entries):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(kb_with_entries.parent))
        db.rebuild_index()

        assert (kb_with_entries / "embeddings.npy").exists()
        assert (kb_with_entries / "index.json").exists()


class TestAtomicIndexWrites:
    """Tests for atomic index persistence."""

    def test_save_index_cleans_up_tmp_files(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(temp_kb_dir.parent))
        db._embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        db._id_to_row = {"entry-1": 0}
        db._row_to_id = {0: "entry-1"}
        db._entries = [{"id": "entry-1", "title": "Atomic entry", "insight": "Persist safely"}]
        db._sparse_vectors = {0: {"token": 0.5}}

        db._save_index()

        assert list(temp_kb_dir.glob("*.tmp")) == []
        np.testing.assert_array_equal(
            np.load(temp_kb_dir / "embeddings.npy"),
            np.array([[1.0, 0.0]], dtype=np.float32),
        )
        assert json.loads((temp_kb_dir / "index.json").read_text()) == {"entry-1": 0}
        assert json.loads((temp_kb_dir / "sparse_index.json").read_text()) == {"0": {"token": 0.5}}


# =============================================================================
# TestAddEntry
# =============================================================================


class TestAddEntry:
    """Tests for adding entries."""

    def test_add_generates_dense_embedding(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(temp_kb_dir.parent))
        entry_id = db.add({"title": "Test Entry", "summary": "Test summary content"})

        assert entry_id is not None
        assert db._embeddings is not None
        assert db._embeddings.shape[0] == 1

    def test_add_appends_to_jsonl(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(temp_kb_dir.parent))
        db.add({"title": "Entry 1", "summary": "Summary 1"})
        db.add({"title": "Entry 2", "summary": "Summary 2"})

        with open(temp_kb_dir / "entries.jsonl") as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 2


# =============================================================================
# TestBatchAdd
# =============================================================================


class TestBatchAdd:
    """Tests for batch_add."""

    def test_batch_add_multiple_entries(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(temp_kb_dir.parent))
        entries = [
            {"title": "Batch Entry 1", "insight": "First batch entry"},
            {"title": "Batch Entry 2", "insight": "Second batch entry"},
            {"title": "Batch Entry 3", "insight": "Third batch entry"},
        ]
        ids = db.batch_add(entries)

        assert len(ids) == 3
        assert db._embeddings.shape[0] == 3
        assert db.count() == 3

    def test_batch_add_skips_invalid(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(temp_kb_dir.parent))
        entries = [
            {"title": "Valid Entry", "insight": "Good content"},
            {"title": "X", "insight": "Too short title"},
            {"title": "Another valid entry", "insight": "More good content"},
        ]
        ids = db.batch_add(entries)

        assert len(ids) == 3
        assert ids[1] is None
        assert db.count() == 2

    def test_batch_add_returns_none_for_rejected(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(temp_kb_dir.parent))
        entries = [
            {"title": "Accepted entry", "insight": "Good content"},
            {"title": "Singleword", "insight": "Rejected because title has one word"},
            {"title": "Another accepted entry", "insight": "More good content"},
        ]

        ids = db.batch_add(entries)

        assert len(ids) == 3
        assert ids[0] is not None
        assert ids[1] is None
        assert ids[2] is not None
        assert db.count() == 2

    def test_batch_add_empty_list(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(temp_kb_dir.parent))
        ids = db.batch_add([])
        assert ids == []

    def test_concurrent_adds(self, monkeypatch, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        dense_model = KeywordDenseModel({"worker": np.array([1.0, 0.0], dtype=np.float32)})
        monkeypatch.setattr("knowlin_mcp.db.get_dense_model", lambda: dense_model)
        monkeypatch.setattr("knowlin_mcp.db.get_sparse_model", lambda: None)

        db = KnowledgeDB(str(temp_kb_dir.parent))
        start_barrier = threading.Barrier(5)
        errors = []

        def worker(worker_id):
            try:
                start_barrier.wait()
                for entry_idx in range(2):
                    db.add(
                        {
                            "title": f"Worker {worker_id} entry {entry_idx}",
                            "insight": f"Concurrent add {worker_id}-{entry_idx}",
                        },
                        check_duplicates=False,
                    )
            except Exception as exc:  # pragma: no cover - assertion covers this path
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(worker_id,)) for worker_id in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3.0)

        expected_titles = {
            f"Worker {worker_id} entry {entry_idx}"
            for worker_id in range(5)
            for entry_idx in range(2)
        }

        assert errors == []
        assert all(not thread.is_alive() for thread in threads)
        assert db.count() == 10
        assert db._embeddings is not None
        assert db._embeddings.shape[0] == 10
        assert len(db._id_to_row) == 10
        assert {entry["title"] for entry in db._entries} == expected_titles

        with open(temp_kb_dir / "entries.jsonl") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 10
        assert {entry["title"] for entry in lines} == expected_titles


# =============================================================================
# TestRemoveEntries
# =============================================================================


class TestRemoveEntries:
    """Tests for remove_entries."""

    def test_remove_cleans_up_state(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(temp_kb_dir.parent))
        db.add({"title": "Keep this entry", "insight": "Should remain"})
        id2 = db.add({"title": "Remove this entry", "insight": "Should be removed"})

        removed = db.remove_entries([id2])

        assert removed == 1
        # Entry should be gone from in-memory state
        assert id2 not in db._id_to_row
        assert db.get(id2) is None

    def test_remove_rewrites_jsonl(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(temp_kb_dir.parent))
        id1 = db.add({"title": "Keep this entry", "insight": "Should remain"})
        id2 = db.add({"title": "Remove this entry", "insight": "Should be removed"})

        db.remove_entries([id2])

        # JSONL should not contain removed entry
        with open(temp_kb_dir / "entries.jsonl") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        ids_in_file = [e.get("id") for e in lines]
        assert id2 not in ids_in_file
        assert id1 in ids_in_file

    def test_remove_middle_entry_preserves_others(self, monkeypatch, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        dense_model = KeywordDenseModel(
            {
                "alpha": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "beta": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "gamma": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            }
        )
        monkeypatch.setattr("knowlin_mcp.db.get_dense_model", lambda: dense_model)
        monkeypatch.setattr("knowlin_mcp.db.get_sparse_model", lambda: None)

        db = KnowledgeDB(str(temp_kb_dir.parent))
        id1 = db.add({"title": "Alpha entry data", "insight": "Alpha details"})
        id2 = db.add({"title": "Beta entry data", "insight": "Beta details"})
        id3 = db.add({"title": "Gamma entry data", "insight": "Gamma details"})

        removed = db.remove_entries([id2])

        assert removed == 1
        assert db.get(id1)["title"] == "Alpha entry data"
        assert db.get(id2) is None
        assert db.get(id3)["title"] == "Gamma entry data"
        assert db.search("alpha", limit=3, rerank=False)[0]["id"] == id1
        assert db.search("gamma", limit=3, rerank=False)[0]["id"] == id3

    def test_add_after_remove(self, monkeypatch, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        dense_model = KeywordDenseModel(
            {
                "alpha": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "beta": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                "gamma": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
                "delta": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            }
        )
        monkeypatch.setattr("knowlin_mcp.db.get_dense_model", lambda: dense_model)
        monkeypatch.setattr("knowlin_mcp.db.get_sparse_model", lambda: None)

        db = KnowledgeDB(str(temp_kb_dir.parent))
        id1 = db.add({"title": "Alpha entry data", "insight": "Alpha details"})
        id2 = db.add({"title": "Beta entry data", "insight": "Beta details"})
        id3 = db.add({"title": "Gamma entry data", "insight": "Gamma details"})

        db.remove_entries([id2])
        id4 = db.add({"title": "Delta entry data", "insight": "Delta details"})

        assert db._embeddings is not None
        assert db._embeddings.shape[0] == 3
        assert db.search("alpha", limit=3, rerank=False)[0]["id"] == id1
        assert db.search("gamma", limit=3, rerank=False)[0]["id"] == id3
        assert db.search("delta", limit=3, rerank=False)[0]["id"] == id4


# =============================================================================
# TestSubStore
# =============================================================================


class TestSubStore:
    """Tests for sub_store parameter."""

    def test_sub_store_uses_separate_directory(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(temp_kb_dir.parent), sub_store="sessions")

        assert db.sub_store == "sessions"
        assert "sessions" in str(db.jsonl_path)
        assert (temp_kb_dir / "sessions").is_dir()

    def test_sub_store_rejects_escape(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        with pytest.raises(ValueError, match="Sub-store path escapes knowledge DB"):
            KnowledgeDB(str(temp_kb_dir.parent), sub_store="../../escape")

    def test_sub_stores_are_independent(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        db_main = KnowledgeDB(str(temp_kb_dir.parent))
        db_sessions = KnowledgeDB(str(temp_kb_dir.parent), sub_store="sessions")

        db_main.add({"title": "Main KB entry", "insight": "In the main store"})
        db_sessions.add({"title": "Session entry", "insight": "In the sessions store"})

        assert db_main.count() == 1
        assert db_sessions.count() == 1

        # Files are separate
        assert (temp_kb_dir / "entries.jsonl").exists()
        assert (temp_kb_dir / "sessions" / "entries.jsonl").exists()


# =============================================================================
# TestInferType
# =============================================================================


class TestInferType:
    """Tests for type inference from content."""

    def test_infers_warning_from_signals(self):
        from knowlin_mcp.utils import infer_type

        assert infer_type("Don't use global state", "") == "warning"
        assert infer_type("", "This bug causes crashes") == "warning"
        assert infer_type("Watch out for race conditions", "") == "warning"
        assert infer_type("Deprecated API", "avoid using this") == "warning"

    def test_infers_solution_from_signals(self):
        from knowlin_mcp.utils import infer_type

        assert infer_type("Fixed memory leak", "") == "solution"
        assert infer_type("", "The workaround is to restart") == "solution"
        assert infer_type("Resolved the timeout", "") == "solution"

    def test_infers_pattern_from_signals(self):
        from knowlin_mcp.utils import infer_type

        assert infer_type("Use dependency injection", "") == "pattern"
        assert infer_type("", "The best way to handle requests") == "pattern"
        assert infer_type("Prefer composition", "") == "pattern"

    def test_defaults_to_finding(self):
        from knowlin_mcp.utils import infer_type

        assert infer_type("Random observation", "some details") == "finding"
        assert infer_type("", "") == "finding"


# =============================================================================
# TestExponentialTimeDecay
# =============================================================================


class TestExponentialTimeDecay:
    """Tests for exponential time decay in get_recent_important()."""

    def test_recent_entries_score_higher(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(temp_kb_dir.parent))

        old_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        db.add(
            {
                "title": "Old finding",
                "insight": "This is old",
                "date": old_date,
                "priority": "medium",
            },
            check_duplicates=False,
        )
        today = datetime.now().strftime("%Y-%m-%d")
        db.add(
            {
                "title": "New finding",
                "insight": "This is new",
                "date": today,
                "priority": "medium",
            },
            check_duplicates=False,
        )

        results = db.get_recent_important(limit=2)
        assert len(results) == 2
        assert results[0]["title"] == "New finding"

    def test_warnings_decay_faster(self, temp_kb_dir):
        import math

        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(temp_kb_dir.parent))

        today = datetime.now().strftime("%Y-%m-%d")
        old_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

        db.add(
            {
                "title": "Fresh warning today",
                "insight": "Don't use this pattern",
                "type": "warning",
                "date": today,
                "priority": "medium",
            },
            check_duplicates=False,
        )
        db.add(
            {
                "title": "Old warning two weeks ago",
                "insight": "Avoid this other thing",
                "type": "warning",
                "date": old_date,
                "priority": "medium",
            },
            check_duplicates=False,
        )
        db.add(
            {
                "title": "Fresh finding today",
                "insight": "Discovered this behavior",
                "type": "finding",
                "date": today,
                "priority": "medium",
            },
            check_duplicates=False,
        )
        db.add(
            {
                "title": "Old finding two weeks ago",
                "insight": "Discovered that behavior",
                "type": "finding",
                "date": old_date,
                "priority": "medium",
            },
            check_duplicates=False,
        )

        results = db.get_recent_important(limit=4)
        assert len(results) == 4

        scores = {r["title"]: i for i, r in enumerate(results)}

        assert scores["Fresh warning today"] < scores["Old warning two weeks ago"]
        assert scores["Fresh finding today"] < scores["Old finding two weeks ago"]

        warning_decay_14d = math.exp(-14 * 0.693 / 7)
        finding_decay_14d = math.exp(-14 * 0.693 / 30)
        assert warning_decay_14d < 0.30
        assert finding_decay_14d > 0.70


# =============================================================================
# TestSemanticDeduplication
# =============================================================================


class TestSemanticDeduplication:
    """Tests for semantic deduplication in add()."""

    def test_allows_distinct_entries(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(temp_kb_dir.parent))

        db.add(
            {
                "title": "BLE power optimization",
                "insight": "Use sleep modes for better battery life",
            }
        )
        db.add(
            {"title": "Python async patterns", "insight": "Use asyncio for I/O bound operations"}
        )

        assert db._embeddings.shape[0] == 2

    def test_skip_dedup_check_flag(self, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(temp_kb_dir.parent))

        db.add({"title": "Test entry", "insight": "Test content"}, check_duplicates=False)
        db.add({"title": "Test entry", "insight": "Test content"}, check_duplicates=False)

        assert db._embeddings.shape[0] == 2

    def test_dedup_rejects_near_identical(self, monkeypatch, temp_kb_dir):
        from knowlin_mcp.db import KnowledgeDB

        base_vec = np.array([1.0, 0.0], dtype=np.float32)

        dense_model = KeywordDenseModel({"duplicate": base_vec})
        monkeypatch.setattr("knowlin_mcp.db.get_dense_model", lambda: dense_model)
        monkeypatch.setattr("knowlin_mcp.db.get_dense_embedding", lambda text: base_vec)
        monkeypatch.setattr("knowlin_mcp.db.get_sparse_model", lambda: None)

        db = KnowledgeDB(str(temp_kb_dir.parent))
        first_id = db.add({"title": "Duplicate entry data", "insight": "Duplicate insight content"})
        second_id = db.add(
            {"title": "Duplicate variant data", "insight": "Duplicate insight content"}
        )

        with open(temp_kb_dir / "entries.jsonl") as f:
            lines = [line for line in f if line.strip()]

        assert second_id == first_id
        assert len(lines) == 1
