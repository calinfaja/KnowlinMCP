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
        (db_path / "entries.jsonl").write_text('{"id": "e1", "title": "Test", "insight": "Data"}\n')
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

    def test_doctor_detects_corrupt_sparse_index(self, runner, healthy_project):
        db_path = healthy_project / ".knowledge-db"
        (db_path / "sparse_index.json").write_text("{not-json")

        result = runner.invoke(main, ["doctor", "-p", str(healthy_project)])

        assert result.exit_code == 1
        assert "corrupt sparse_index.json" in result.output

    def test_doctor_checks_sources_yaml(self, runner, healthy_project):
        db_path = healthy_project / ".knowledge-db"
        # Write valid YAML
        (db_path / "sources.yaml").write_text("docs:\n  paths:\n    - docs/\n")

        result = runner.invoke(main, ["doctor", "-p", str(healthy_project)])
        assert "sources.yaml: valid" in result.output

    def test_doctor_missing_configured_path(self, runner, healthy_project):
        db_path = healthy_project / ".knowledge-db"
        (db_path / "sources.yaml").write_text("docs:\n  paths:\n    - /nonexistent/path/\n")

        result = runner.invoke(main, ["doctor", "-p", str(healthy_project)])
        assert "configured doc path missing" in result.output


class TestSourcesCommand:
    """Tests for the sources command."""

    def test_sources_init_template_includes_codex_section(self, runner, tmp_path):
        (tmp_path / ".knowledge-db").mkdir()

        result = runner.invoke(main, ["sources", "--init", "-p", str(tmp_path)])

        assert result.exit_code == 0
        contents = (tmp_path / ".knowledge-db" / "sources.yaml").read_text()
        assert "codex:" in contents
        assert "~/.codex/sessions/" in contents


class TestStatsCommand:
    """Tests for the multi-source stats command."""

    @patch("knowlin_mcp.multi_search.MultiSourceSearch")
    def test_stats_json_output(self, mock_ms_cls, runner, tmp_path):
        # Create a .knowledge-db so _resolve_project succeeds
        (tmp_path / ".knowledge-db").mkdir()

        mock_ms = MagicMock()
        mock_ms.stats.return_value = {
            "kb": {
                "count": 10,
                "available": True,
                "size_human": "5.2 KB",
                "last_updated": "2026-03-04",
            },
            "sessions": {
                "count": 5,
                "available": True,
                "size_human": "2.1 KB",
                "last_updated": "2026-03-04",
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
                "count": 10,
                "available": True,
                "size_human": "5.2 KB",
                "last_updated": "2026-03-04",
            },
            "sessions": {"count": 0, "available": False},
            "docs": {"count": 0, "available": False},
        }
        mock_ms_cls.return_value = mock_ms

        result = runner.invoke(main, ["stats", "-p", str(tmp_path)])
        assert "10" in result.output
        assert "not initialized" in result.output
        assert "Total entries" in result.output


class TestCaptureCommand:
    """Tests for the capture command."""

    @patch("knowlin_mcp.capture.log_to_timeline")
    @patch("knowlin_mcp.capture.save_entry", return_value=False)
    @patch("knowlin_mcp.capture.create_entry")
    def test_capture_reports_save_failure(self, mock_create, mock_save, mock_log, runner, tmp_path):
        (tmp_path / ".git").mkdir()
        mock_create.return_value = {
            "id": "entry-1",
            "title": "Useful knowledge entry",
            "type": "finding",
            "insight": "A useful insight",
        }

        result = runner.invoke(main, ["capture", "A useful insight", "-p", str(tmp_path)])

        assert result.exit_code == 1
        assert "Failed to save entry" in result.output
        mock_save.assert_called_once()
        mock_log.assert_not_called()


class TestSourceRouting:
    """Tests for CLI source routing."""

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_list_uses_root_store_for_kb_source(self, mock_db_cls, runner, tmp_path):
        mock_db = MagicMock()
        mock_db.list_recent.return_value = [
            {
                "id": "entry-1",
                "title": "Root KB entry",
                "type": "finding",
                "date": "2026-03-04",
                "timestamp": "2026-03-04T10:00:00",
            }
        ]
        mock_db_cls.return_value = mock_db

        (tmp_path / ".knowledge-db").mkdir()

        result = runner.invoke(main, ["list", "--source", "kb", "-p", str(tmp_path)])

        assert result.exit_code == 0
        mock_db_cls.assert_called_once_with(str(tmp_path), sub_store=None)
        assert "Root KB entry" in result.output

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_delete_uses_root_store_for_kb_source(self, mock_db_cls, runner, tmp_path):
        mock_db = MagicMock()
        mock_db.get.return_value = {"id": "entry-1", "title": "Root KB entry"}
        mock_db.remove_entries.return_value = 1
        mock_db_cls.return_value = mock_db

        (tmp_path / ".knowledge-db").mkdir()

        result = runner.invoke(main, ["delete", "entry-1", "--source", "kb", "-p", str(tmp_path)])

        assert result.exit_code == 0
        mock_db_cls.assert_called_once_with(str(tmp_path), sub_store=None)
        mock_db.remove_entries.assert_called_once_with(["entry-1"])
        assert "Deleted from kb: Root KB entry" in result.output


class TestServerStop:
    """Tests for server stop safety checks."""

    @patch("knowlin_mcp.cli.kill_process_tree")
    @patch("knowlin_mcp.cli.is_process_running", return_value=True)
    @patch("knowlin_mcp.cli.psutil.Process")
    def test_server_stop_skips_non_knowlin_pid(
        self, mock_process_cls, mock_is_running, mock_kill, runner, tmp_path
    ):
        project = tmp_path / "project"
        project.mkdir()
        (project / ".knowledge-db").mkdir()

        pid_file = tmp_path / "server.pid"
        pid_file.write_text("1234")
        port_file = tmp_path / "server.port"
        port_file.write_text("14000")

        proc = MagicMock()
        proc.cmdline.return_value = ["python", "other-service.py"]
        mock_process_cls.return_value = proc

        with (
            patch("knowlin_mcp.cli.get_kb_pid_file", return_value=pid_file),
            patch("knowlin_mcp.cli.get_kb_port_file", return_value=port_file),
            patch("knowlin_mcp.cli.read_pid_file", return_value=1234),
        ):
            result = runner.invoke(main, ["server", "stop", "-p", str(project)])

        assert result.exit_code == 0
        mock_kill.assert_not_called()
        assert not pid_file.exists()
        assert not port_file.exists()
