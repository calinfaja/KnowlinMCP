"""Data integrity tests - verify embedding/JSONL/index alignment.

These catch silent correctness bugs where files desync.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from knowlin_mcp.db import KnowledgeDB


@pytest.fixture
def populated_db(tmp_path):
    """Create a KB with 3 real entries (uses real fastembed)."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".knowledge-db").mkdir()

    db = KnowledgeDB(str(project))
    db.add(
        {"title": "BLE power optimization", "insight": "Use Nordic sleep modes to reduce current draw"},
        check_duplicates=False,
    )
    db.add(
        {"title": "OAuth2 token refresh", "insight": "Use refresh tokens to maintain sessions"},
        check_duplicates=False,
    )
    db.add(
        {"title": "Python async patterns", "insight": "asyncio event loop for IO bound work"},
        check_duplicates=False,
    )
    return project


class TestFileAlignment:
    """Embedding, JSONL, and index files must stay in sync."""

    def test_embedding_rows_match_index_count(self, populated_db):
        db_path = populated_db / ".knowledge-db"
        embeddings = np.load(str(db_path / "embeddings.npy"))
        index = json.loads((db_path / "index.json").read_text())
        assert embeddings.shape[0] == len(index)

    def test_embedding_rows_match_jsonl_count(self, populated_db):
        db_path = populated_db / ".knowledge-db"
        embeddings = np.load(str(db_path / "embeddings.npy"))
        lines = [l for l in (db_path / "entries.jsonl").read_text().splitlines() if l.strip()]
        assert embeddings.shape[0] == len(lines)

    def test_index_row_values_are_contiguous(self, populated_db):
        db_path = populated_db / ".knowledge-db"
        index = json.loads((db_path / "index.json").read_text())
        rows = sorted(index.values())
        assert rows == list(range(len(rows)))

    def test_index_ids_match_jsonl_ids(self, populated_db):
        db_path = populated_db / ".knowledge-db"
        index = json.loads((db_path / "index.json").read_text())
        jsonl_ids = set()
        for line in (db_path / "entries.jsonl").read_text().splitlines():
            if line.strip():
                jsonl_ids.add(json.loads(line).get("id", ""))
        for entry_id in index:
            assert entry_id in jsonl_ids

    def test_embedding_dimension_is_384(self, populated_db):
        """BGE-small produces 384-dim vectors."""
        db_path = populated_db / ".knowledge-db"
        embeddings = np.load(str(db_path / "embeddings.npy"))
        assert embeddings.shape[1] == 384

    def test_embeddings_are_unit_normalized(self, populated_db):
        """Dot-product cosine requires L2-normalized vectors."""
        db_path = populated_db / ".knowledge-db"
        embeddings = np.load(str(db_path / "embeddings.npy"))
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_sparse_index_rows_valid(self, populated_db):
        db_path = populated_db / ".knowledge-db"
        sparse_path = db_path / "sparse_index.json"
        if not sparse_path.exists():
            pytest.skip("No sparse index")
        embeddings = np.load(str(db_path / "embeddings.npy"))
        sparse = json.loads(sparse_path.read_text())
        for key in sparse:
            assert 0 <= int(key) < embeddings.shape[0]


class TestPersistence:
    """Entries survive reload from disk."""

    def test_add_then_reload_finds_entry(self, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        (project / ".knowledge-db").mkdir()

        db1 = KnowledgeDB(str(project))
        entry_id = db1.add(
            {"title": "Persist test entry", "insight": "Must survive reload from disk"}
        )
        assert entry_id

        # New instance loads from disk
        db2 = KnowledgeDB(str(project))
        results = db2.search("persist survive reload", limit=1, rerank=False)
        assert results
        assert results[0]["id"] == entry_id

    def test_sequential_adds_dont_corrupt(self, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        (project / ".knowledge-db").mkdir()

        db = KnowledgeDB(str(project))
        id1 = db.add(
            {"title": "First test entry here", "insight": "First content data"},
            check_duplicates=False,
        )
        id2 = db.add(
            {"title": "Second test entry here", "insight": "Second content data"},
            check_duplicates=False,
        )

        assert db._embeddings.shape[0] == 2
        assert id1 in db._id_to_row
        assert id2 in db._id_to_row
        assert db._id_to_row[id1] != db._id_to_row[id2]

    def test_batch_add_alignment(self, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        (project / ".knowledge-db").mkdir()

        db = KnowledgeDB(str(project))
        entries = [
            {"title": f"Batch entry number {i}", "insight": f"Content for batch entry {i}"}
            for i in range(5)
        ]
        ids = db.batch_add(entries, check_duplicates=False)
        assert len(ids) == 5
        assert db._embeddings.shape[0] == 5

        # Verify index alignment
        db_path = project / ".knowledge-db"
        reloaded_index = json.loads((db_path / "index.json").read_text())
        assert len(reloaded_index) == 5

    def test_remove_then_rebuild_compacts(self, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        (project / ".knowledge-db").mkdir()

        db = KnowledgeDB(str(project))
        ids = db.batch_add(
            [
                {"title": "Keep this entry please", "insight": "Should survive removal"},
                {"title": "Remove this entry now", "insight": "Should be deleted"},
                {"title": "Also keep this entry", "insight": "Should also survive"},
            ],
            check_duplicates=False,
        )

        db.remove_entries([ids[1]])

        # Rebuild compacts the index
        count = db.rebuild_index()
        assert count == 2
        assert db._embeddings.shape[0] == 2
