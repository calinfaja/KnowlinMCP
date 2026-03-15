"""Tests for CLI commands (doctor, stats)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from knowlin_mcp.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def healthy_project(tmp_path):
    """Create a project with aligned KB data."""
    db_path = tmp_path / ".knowledge-db"
    db_path.mkdir()

    # Main KB store
    entries = [
        {"id": "e1", "title": "Test entry", "insight": "Some insight", "type": "finding"},
        {"id": "e2", "title": "Another entry", "insight": "More insight", "type": "warning"},
    ]
    with open(db_path / "entries.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    # 2 embeddings (384-dim)
    embeddings = np.random.rand(2, 384).astype(np.float32)
    np.save(str(db_path / "embeddings.npy"), embeddings)

    # Index mapping
    index = {"e1": 0, "e2": 1}
    (db_path / "index.json").write_text(json.dumps(index))

    return tmp_path


class TestDoctor:
    """Tests for the doctor command."""

    def test_doctor_healthy_project(self, runner, healthy_project):
        result = runner.invoke(main, ["doctor", "-p", str(healthy_project)])
        assert "OK" in result.output
        assert "kb: 2 entries aligned" in result.output

    def test_doctor_no_db_dir(self, runner, tmp_path):
        # Need .git so _resolve_project succeeds, but no .knowledge-db
        (tmp_path / ".git").mkdir()
        result = runner.invoke(main, ["doctor", "-p", str(tmp_path)])
        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_doctor_missing_embeddings(self, runner, tmp_path):
        """JSONL exists but no embeddings."""
        db_path = tmp_path / ".knowledge-db"
        db_path.mkdir()
        (db_path / "entries.jsonl").write_text(
            '{"id": "e1", "title": "Test", "insight": "Data"}\n'
        )
        result = runner.invoke(main, ["doctor", "-p", str(tmp_path)])
        assert "WARN" in result.output
        assert "no embeddings" in result.output

    def test_doctor_misaligned_counts(self, runner, tmp_path):
        """Embedding count differs from JSONL count."""
        db_path = tmp_path / ".knowledge-db"
        db_path.mkdir()

        # 1 entry in JSONL
        (db_path / "entries.jsonl").write_text(
            '{"id": "e1", "title": "Test entry one", "insight": "Data"}\n'
        )

        # 3 embeddings (mismatch)
        embeddings = np.random.rand(3, 384).astype(np.float32)
        np.save(str(db_path / "embeddings.npy"), embeddings)

        index = {"e1": 0, "e2": 1, "e3": 2}
        (db_path / "index.json").write_text(json.dumps(index))

        result = runner.invoke(main, ["doctor", "-p", str(tmp_path)])
        # Should report orphaned embeddings
        assert "orphaned" in result.output or "misaligned" in result.output

    def test_doctor_checks_registries(self, runner, healthy_project):
        db_path = healthy_project / ".knowledge-db"
        reg = {"test.jsonl": {"hash": "abc", "entry_ids": ["e1"]}}
        (db_path / "session-registry.json").write_text(json.dumps(reg))

        result = runner.invoke(main, ["doctor", "-p", str(healthy_project)])
        assert "session-registry: 1 tracked files" in result.output

    def test_doctor_corrupted_registry(self, runner, healthy_project):
        db_path = healthy_project / ".knowledge-db"
        (db_path / "doc-registry.json").write_text("not json{{{")

        result = runner.invoke(main, ["doctor", "-p", str(healthy_project)])
        assert "corrupted" in result.output

    def test_doctor_checks_sources_yaml(self, runner, healthy_project):
        db_path = healthy_project / ".knowledge-db"
        # Write valid YAML
        (db_path / "sources.yaml").write_text("docs:\n  paths:\n    - docs/\n")

        result = runner.invoke(main, ["doctor", "-p", str(healthy_project)])
        assert "sources.yaml: valid" in result.output

    def test_doctor_missing_configured_path(self, runner, healthy_project):
        db_path = healthy_project / ".knowledge-db"
        (db_path / "sources.yaml").write_text(
            "docs:\n  paths:\n    - /nonexistent/path/\n"
        )

        result = runner.invoke(main, ["doctor", "-p", str(healthy_project)])
        assert "configured doc path missing" in result.output


class TestStatsCommand:
    """Tests for the multi-source stats command."""

    @patch("knowlin_mcp.multi_search.MultiSourceSearch")
    def test_stats_json_output(self, mock_ms_cls, runner, tmp_path):
        # Create a .knowledge-db so _resolve_project succeeds
        (tmp_path / ".knowledge-db").mkdir()

        mock_ms = MagicMock()
        mock_ms.stats.return_value = {
            "kb": {
                "count": 10, "available": True,
                "size_human": "5.2 KB", "last_updated": "2026-03-04",
            },
            "sessions": {
                "count": 5, "available": True,
                "size_human": "2.1 KB", "last_updated": "2026-03-04",
            },
            "docs": {"count": 0, "available": False},
        }
        mock_ms_cls.return_value = mock_ms

        result = runner.invoke(main, ["stats", "--json", "-p", str(tmp_path)])
        data = json.loads(result.output)
        assert data["kb"]["count"] == 10
        assert data["sessions"]["count"] == 5
        assert data["docs"]["available"] is False

    @patch("knowlin_mcp.multi_search.MultiSourceSearch")
    def test_stats_table_output(self, mock_ms_cls, runner, tmp_path):
        (tmp_path / ".knowledge-db").mkdir()

        mock_ms = MagicMock()
        mock_ms.stats.return_value = {
            "kb": {
                "count": 10, "available": True,
                "size_human": "5.2 KB", "last_updated": "2026-03-04",
            },
            "sessions": {"count": 0, "available": False},
            "docs": {"count": 0, "available": False},
        }
        mock_ms_cls.return_value = mock_ms

        result = runner.invoke(main, ["stats", "-p", str(tmp_path)])
        assert "10" in result.output
        assert "not initialized" in result.output
        assert "Total entries" in result.output
