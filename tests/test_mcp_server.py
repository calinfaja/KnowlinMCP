"""Tests for the MCP server tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from knowlin_mcp import mcp_server
from knowlin_mcp.mcp_server import (
    _format_full_entry,
    _parse_sources,
    knowlin_capture,
    knowlin_get,
    knowlin_ingest,
    knowlin_search,
    knowlin_stats,
)

try:
    from mcp.server.fastmcp.exceptions import ToolError
except ImportError:
    ToolError = Exception  # type: ignore[misc,assignment]


@pytest.fixture(autouse=True)
def reset_project_root():
    """Reset cached project root between tests."""
    mcp_server._project_root = None
    yield
    mcp_server._project_root = None


# --- _parse_sources ---


class TestParseSources:
    def test_all_string(self):
        assert _parse_sources("all") == ["kb", "sessions", "docs"]

    def test_all_caps(self):
        assert _parse_sources("ALL") == ["kb", "sessions", "docs"]

    def test_empty_string(self):
        assert _parse_sources("") == ["kb", "sessions", "docs"]

    def test_single_source(self):
        assert _parse_sources("docs") == ["docs"]

    def test_comma_separated(self):
        assert _parse_sources("kb,sessions") == ["kb", "sessions"]

    def test_comma_with_spaces(self):
        assert _parse_sources("kb, docs") == ["kb", "docs"]


# --- knowlin_search ---


class TestKnowlinSearch:
    @patch("knowlin_mcp.mcp_server._get_project_root")
    @patch("knowlin_mcp.multi_search.MultiSourceSearch")
    def test_basic_search(self, mock_mss_cls, mock_root):
        mock_root.return_value = "/fake/project"
        mock_mss = MagicMock()
        mock_mss_cls.return_value = mock_mss
        mock_mss.search.return_value = [
            {
                "id": "e1",
                "title": "Test Entry",
                "score": 0.85,
                "_source": "kb",
                "type": "finding",
                "date": "2024-12-01",
                "insight": "Some insight text",
            }
        ]

        result = knowlin_search("test query", limit=5)

        assert "1 result(s)" in result
        assert "Test Entry" in result
        assert "85%" in result
        assert "e1" in result
        mock_mss.search.assert_called_once_with(
            "test query",
            sources=["kb", "sessions", "docs"],
            limit=5,
            date_from=None,
            date_to=None,
            entry_type=None,
        )

    @patch("knowlin_mcp.mcp_server._get_project_root")
    @patch("knowlin_mcp.multi_search.MultiSourceSearch")
    def test_no_results(self, mock_mss_cls, mock_root):
        mock_root.return_value = "/fake/project"
        mock_mss = MagicMock()
        mock_mss_cls.return_value = mock_mss
        mock_mss.search.return_value = []

        result = knowlin_search("obscure query")
        assert "No results found" in result

    @patch("knowlin_mcp.mcp_server._get_project_root")
    @patch("knowlin_mcp.multi_search.MultiSourceSearch")
    def test_source_filtering(self, mock_mss_cls, mock_root):
        mock_root.return_value = "/fake/project"
        mock_mss = MagicMock()
        mock_mss_cls.return_value = mock_mss
        mock_mss.search.return_value = []

        knowlin_search("query", sources="docs,sessions")

        mock_mss.search.assert_called_once()
        call_kwargs = mock_mss.search.call_args[1]
        assert call_kwargs["sources"] == ["docs", "sessions"]

    @patch("knowlin_mcp.mcp_server._get_project_root")
    @patch("knowlin_mcp.multi_search.MultiSourceSearch")
    def test_date_filters(self, mock_mss_cls, mock_root):
        mock_root.return_value = "/fake/project"
        mock_mss = MagicMock()
        mock_mss_cls.return_value = mock_mss
        mock_mss.search.return_value = []

        knowlin_search("query", since="2024-01-01", until="2024-12-31")

        call_kwargs = mock_mss.search.call_args[1]
        assert call_kwargs["date_from"] == "2024-01-01"
        assert call_kwargs["date_to"] == "2024-12-31"

    @patch("knowlin_mcp.mcp_server._get_project_root")
    @patch("knowlin_mcp.multi_search.MultiSourceSearch")
    def test_entry_type_filter(self, mock_mss_cls, mock_root):
        mock_root.return_value = "/fake/project"
        mock_mss = MagicMock()
        mock_mss_cls.return_value = mock_mss
        mock_mss.search.return_value = []

        knowlin_search("query", entry_type="warning")

        call_kwargs = mock_mss.search.call_args[1]
        assert call_kwargs["entry_type"] == "warning"

    @patch("knowlin_mcp.mcp_server._get_project_root")
    @patch("knowlin_mcp.multi_search.MultiSourceSearch")
    def test_long_insight_truncated(self, mock_mss_cls, mock_root):
        mock_root.return_value = "/fake/project"
        mock_mss = MagicMock()
        mock_mss_cls.return_value = mock_mss
        mock_mss.search.return_value = [
            {
                "id": "e1",
                "title": "Long Entry",
                "score": 0.9,
                "_source": "kb",
                "insight": "x" * 500,
            }
        ]

        result = knowlin_search("query")
        # 297 chars + "..."
        assert "..." in result
        assert "x" * 298 not in result

    @patch("knowlin_mcp.mcp_server._get_project_root")
    def test_limit_clamped(self, mock_root):
        mock_root.return_value = "/fake/project"
        with patch("knowlin_mcp.multi_search.MultiSourceSearch") as mock_mss_cls:
            mock_mss = MagicMock()
            mock_mss_cls.return_value = mock_mss
            mock_mss.search.return_value = []

            knowlin_search("query", limit=100)
            call_kwargs = mock_mss.search.call_args[1]
            assert call_kwargs["limit"] == 20

    @patch("knowlin_mcp.mcp_server._get_project_root", side_effect=RuntimeError("no root"))
    def test_error_raises_tool_error(self, _):
        """Search raises ToolError when project root not found."""
        with pytest.raises(ToolError, match="Search error"):
            knowlin_search("query")


# --- knowlin_get ---


class TestKnowlinGet:
    @patch("knowlin_mcp.mcp_server._get_project_root")
    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_found_in_kb(self, mock_db_cls, mock_root):
        mock_root.return_value = "/fake/project"
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_db.get.return_value = {
            "id": "e1",
            "title": "Test Entry",
            "type": "finding",
            "date": "2024-12-01",
            "insight": "An insight",
            "keywords": ["python", "testing"],
        }

        result = knowlin_get("e1")

        assert "# Test Entry" in result
        assert "An insight" in result
        assert "python" in result

    @patch("knowlin_mcp.mcp_server._get_project_root")
    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_not_found(self, mock_db_cls, mock_root):
        mock_root.return_value = "/fake/project"
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_db.get.return_value = None

        result = knowlin_get("nonexistent")
        assert "not found" in result.lower()

    @patch("knowlin_mcp.mcp_server._get_project_root")
    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_searches_all_substores(self, mock_db_cls, mock_root):
        mock_root.return_value = "/fake/project"
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        # Return None for first two stores, found in third
        mock_db.get.side_effect = [
            None,
            None,
            {"id": "e1", "title": "Found in docs"},
        ]

        result = knowlin_get("e1")
        assert "Found in docs" in result
        assert mock_db.get.call_count == 3

    @patch("knowlin_mcp.mcp_server._get_project_root", side_effect=RuntimeError("no root"))
    def test_error_raises_tool_error(self, _):
        with pytest.raises(ToolError, match="Get error"):
            knowlin_get("e1")


# --- knowlin_stats ---


class TestKnowlinStats:
    @patch("knowlin_mcp.mcp_server._get_project_root")
    @patch("knowlin_mcp.multi_search.MultiSourceSearch")
    def test_stats_output(self, mock_mss_cls, mock_root):
        mock_root.return_value = "/fake/project"
        mock_mss = MagicMock()
        mock_mss_cls.return_value = mock_mss
        mock_mss.stats.return_value = {
            "kb": {
                "count": 10,
                "size_human": "5.2 KB",
                "last_updated": "2024-12-01T10:00:00",
                "available": True,
            },
            "sessions": {
                "count": 25,
                "size_human": "12.1 KB",
                "last_updated": "2024-12-02T10:00:00",
                "available": True,
            },
            "docs": {"count": 0, "available": False},
        }

        result = knowlin_stats()

        assert "Knowledge DB Stats" in result
        assert "| kb |" in result
        assert "| sessions |" in result
        assert "| docs |" in result
        assert "10" in result
        assert "25" in result
        assert "**Total entries:** 35" in result

    @patch("knowlin_mcp.mcp_server._get_project_root", side_effect=RuntimeError("no root"))
    def test_error_raises_tool_error(self, _):
        with pytest.raises(ToolError, match="Stats error"):
            knowlin_stats()


# --- knowlin_ingest ---


class TestKnowlinIngest:
    @patch("knowlin_mcp.mcp_server._get_project_root")
    @patch("knowlin_mcp.ingest_docs.DocsIngester")
    @patch("knowlin_mcp.ingest_sessions.SessionIngester")
    def test_ingest_all(self, mock_si_cls, mock_di_cls, mock_root):
        mock_root.return_value = "/fake/project"
        mock_di_cls.return_value.ingest.return_value = 5
        mock_si_cls.return_value.ingest.return_value = 3

        result = knowlin_ingest("all")

        assert "Docs: 5" in result
        assert "Sessions: 3" in result

    @patch("knowlin_mcp.mcp_server._get_project_root")
    @patch("knowlin_mcp.ingest_docs.DocsIngester")
    def test_ingest_docs_only(self, mock_di_cls, mock_root):
        mock_root.return_value = "/fake/project"
        mock_di_cls.return_value.ingest.return_value = 2

        result = knowlin_ingest("docs")

        assert "Docs: 2" in result
        assert "Sessions" not in result

    @patch("knowlin_mcp.mcp_server._get_project_root")
    @patch("knowlin_mcp.ingest_sessions.SessionIngester")
    def test_ingest_sessions_only(self, mock_si_cls, mock_root):
        mock_root.return_value = "/fake/project"
        mock_si_cls.return_value.ingest.return_value = 4

        result = knowlin_ingest("sessions")

        assert "Sessions: 4" in result
        assert "Docs" not in result

    @patch("knowlin_mcp.mcp_server._get_project_root")
    def test_invalid_source(self, mock_root):
        mock_root.return_value = "/fake/project"

        result = knowlin_ingest("invalid")
        assert "Invalid source" in result

    @patch("knowlin_mcp.mcp_server._get_project_root", side_effect=RuntimeError("no root"))
    def test_error_raises_tool_error(self, _):
        with pytest.raises(ToolError, match="Ingest error"):
            knowlin_ingest()


# --- _format_full_entry ---


class TestFormatFullEntry:
    def test_basic_format(self):
        entry = {
            "id": "e1",
            "title": "My Entry",
            "type": "solution",
            "date": "2024-12-01",
            "insight": "Key insight here",
            "keywords": ["python", "mcp"],
        }
        result = _format_full_entry(entry, "kb")

        assert "# My Entry" in result
        assert "Source: kb" in result
        assert "Type: solution" in result
        assert "Key insight here" in result
        assert "python, mcp" in result

    def test_v2_compat_fields(self):
        """V2 entries use summary/tags/found_date."""
        entry = {
            "id": "e1",
            "title": "V2 Entry",
            "summary": "Old summary",
            "tags": ["old"],
            "found_date": "2024-01-01",
        }
        result = _format_full_entry(entry, "sessions")

        assert "Old summary" in result
        assert "old" in result
        assert "2024-01-01" in result

    def test_minimal_entry(self):
        entry = {"title": "Bare Entry"}
        result = _format_full_entry(entry, "docs")

        assert "# Bare Entry" in result
        assert "Source: docs" in result


# --- knowlin_capture ---


class TestKnowlinCapture:
    @patch("knowlin_mcp.mcp_server._get_project_root")
    @patch("knowlin_mcp.mcp_server.find_project_root")
    @patch("knowlin_mcp.capture.save_entry")
    @patch("knowlin_mcp.capture.create_entry_from_json")
    def test_capture_success(self, mock_create, mock_save, mock_find, mock_root, tmp_path):
        mock_root.return_value = str(tmp_path)
        mock_find.return_value = tmp_path
        mock_create.return_value = {"id": "test-id", "title": "Test Insight"}
        mock_save.return_value = True

        result = knowlin_capture(
            title="Test Insight",
            insight="This is a test insight",
            entry_type="finding",
            keywords="test,mcp",
            priority="medium",
        )

        assert "Saved" in result
        assert "test-id" in result
        mock_create.assert_called_once()
        mock_save.assert_called_once()

    def test_capture_writes_entry_to_entries_jsonl(self, monkeypatch, tmp_path):
        project_root = tmp_path / "project"
        kb_dir = project_root / ".knowledge-db"
        kb_dir.mkdir(parents=True)

        class FakeDenseModel:
            def embed(self, texts):
                for _ in texts:
                    yield np.array([1.0, 0.0], dtype=np.float32)

        monkeypatch.setattr("knowlin_mcp.capture.send_entry_to_server", lambda *_: False)
        monkeypatch.setattr("knowlin_mcp.capture._notify_server_reload", lambda *_: None)
        monkeypatch.setattr("knowlin_mcp.db.get_dense_model", lambda: FakeDenseModel())
        monkeypatch.setattr("knowlin_mcp.db.get_sparse_model", lambda: None)

        with patch("knowlin_mcp.mcp_server._get_project_root", return_value=str(project_root)):
            with patch("knowlin_mcp.mcp_server.find_project_root", return_value=project_root):
                result = knowlin_capture(
                    title="Captured MCP insight",
                    insight="This should land in the project knowledge store",
                    entry_type="finding",
                    keywords="mcp,test",
                    priority="medium",
                )

        entries_path = kb_dir / "entries.jsonl"
        assert entries_path.exists()
        with open(entries_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        assert len(entries) == 1
        assert entries[0]["title"] == "Captured MCP insight"
        assert "Saved" in result

    @patch("knowlin_mcp.mcp_server._get_project_root")
    def test_capture_invalid_type(self, mock_root, tmp_path):
        mock_root.return_value = str(tmp_path)

        result = knowlin_capture(title="Test", insight="Test", entry_type="invalid")

        assert "Invalid type" in result

    @patch("knowlin_mcp.mcp_server._get_project_root")
    def test_capture_invalid_priority(self, mock_root, tmp_path):
        mock_root.return_value = str(tmp_path)

        result = knowlin_capture(title="Test", insight="Test", priority="urgent")

        assert "Invalid priority" in result
