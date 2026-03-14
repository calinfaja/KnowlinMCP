"""Tests for Codex CLI JSONL session ingestion."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from knowlin_mcp.ingest_codex import CodexIngester


def _codex_meta_record(cwd="/home/user/project"):
    """Build a Codex session_meta record."""
    return {
        "timestamp": "2026-03-05T10:00:00Z",
        "type": "session_meta",
        "payload": {"id": "abc-123", "cwd": cwd},
    }


def _codex_user_record(message):
    """Build a Codex event_msg user_message record."""
    return {
        "timestamp": "2026-03-05T10:00:01Z",
        "type": "event_msg",
        "payload": {"type": "user_message", "message": message},
    }


def _codex_assistant_record(text):
    """Build a Codex response_item record with standard text content."""
    return {
        "timestamp": "2026-03-05T10:00:02Z",
        "type": "response_item",
        "item": {
            "role": "assistant",
            "content": [{"text": text}],
        },
    }


def _codex_assistant_output_text(text):
    """Build a Codex response_item with OutputText variant."""
    return {
        "timestamp": "2026-03-05T10:00:02Z",
        "type": "response_item",
        "item": {
            "role": "assistant",
            "content": [{"OutputText": {"text": text}}],
        },
    }


def _codex_skip_record():
    """Build a Codex event_msg that should be skipped (token_count)."""
    return {
        "timestamp": "2026-03-05T10:00:03Z",
        "type": "event_msg",
        "payload": {"type": "token_count", "total": 500},
    }


def _codex_turn_context_record():
    """Build a Codex turn_context record (skip)."""
    return {
        "timestamp": "2026-03-05T10:00:04Z",
        "type": "turn_context",
        "payload": {},
    }


SUBSTANTIVE_ANSWER = (
    "The root cause of the authentication issue was that the JWT token "
    "validation was checking the wrong issuer field. The fix is to update "
    "the issuer config in auth.py. I found that the token was being validated "
    "against 'old-issuer' instead of 'new-issuer'. This is a common gotcha "
    "when migrating authentication providers."
)


@pytest.fixture
def codex_dir(tmp_path):
    """Create a mock Codex sessions directory with date-nested structure."""
    sessions = tmp_path / "codex_sessions" / "2026" / "03" / "05"
    sessions.mkdir(parents=True)
    return sessions


@pytest.fixture
def sample_codex_jsonl(codex_dir):
    """Create a sample Codex JSONL session file."""
    path = codex_dir / "rollout-abc123.jsonl"
    records = [
        _codex_meta_record(),
        _codex_user_record("How do I fix the authentication bug?"),
        _codex_assistant_record(SUBSTANTIVE_ANSWER),
        _codex_user_record("Thanks"),
        _codex_assistant_record("You're welcome!"),
    ]
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def project_with_codex(tmp_path, codex_dir, sample_codex_jsonl):
    """Create a project directory with Codex sessions."""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".knowledge-db").mkdir()
    return project


class TestExtractFromCodexJsonl:
    """Tests for Codex JSONL parsing and extraction."""

    def test_extracts_high_value_entries(self, sample_codex_jsonl):
        ingester = CodexIngester.__new__(CodexIngester)
        ingester._registry = {}

        entries = ingester._extract_from_codex_jsonl(sample_codex_jsonl)
        assert len(entries) >= 1
        # Short "You're welcome!" should be filtered
        titles = [e["title"] for e in entries]
        assert not any("welcome" in t.lower() for t in titles)

    def test_pairs_user_question_with_response(self, codex_dir):
        path = codex_dir / "pair.jsonl"
        records = [
            _codex_user_record("What is the fix for the auth bug?"),
            _codex_assistant_record(SUBSTANTIVE_ANSWER),
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ingester = CodexIngester.__new__(CodexIngester)
        ingester._registry = {}

        entries = ingester._extract_from_codex_jsonl(path)
        assert len(entries) >= 1
        assert "auth" in entries[0]["title"].lower()

    def test_handles_output_text_variant(self, codex_dir):
        path = codex_dir / "output-text.jsonl"
        records = [
            _codex_user_record("Explain the architecture"),
            _codex_assistant_output_text(SUBSTANTIVE_ANSWER),
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ingester = CodexIngester.__new__(CodexIngester)
        ingester._registry = {}

        entries = ingester._extract_from_codex_jsonl(path)
        assert len(entries) >= 1

    def test_skips_session_meta(self, codex_dir):
        path = codex_dir / "meta-only.jsonl"
        records = [_codex_meta_record()]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ingester = CodexIngester.__new__(CodexIngester)
        ingester._registry = {}

        entries = ingester._extract_from_codex_jsonl(path)
        assert len(entries) == 0

    def test_skips_non_user_message_events(self, codex_dir):
        path = codex_dir / "skip-events.jsonl"
        records = [
            _codex_skip_record(),
            _codex_turn_context_record(),
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ingester = CodexIngester.__new__(CodexIngester)
        ingester._registry = {}

        entries = ingester._extract_from_codex_jsonl(path)
        assert len(entries) == 0

    def test_entry_has_required_fields(self, sample_codex_jsonl):
        ingester = CodexIngester.__new__(CodexIngester)
        ingester._registry = {}

        entries = ingester._extract_from_codex_jsonl(sample_codex_jsonl)
        if entries:
            entry = entries[0]
            assert "title" in entry
            assert "insight" in entry
            assert "type" in entry
            assert "priority" in entry
            assert "source" in entry
            assert entry["source"].startswith("codex:")

    def test_handles_malformed_jsonl(self, codex_dir):
        path = codex_dir / "bad.jsonl"
        with open(path, "w") as f:
            f.write("not json\n")
            f.write(json.dumps(_codex_assistant_record("valid but short")) + "\n")
            f.write("{malformed}\n")

        ingester = CodexIngester.__new__(CodexIngester)
        ingester._registry = {}

        entries = ingester._extract_from_codex_jsonl(path)
        assert isinstance(entries, list)

    def test_insight_includes_user_question(self, sample_codex_jsonl):
        ingester = CodexIngester.__new__(CodexIngester)
        ingester._registry = {}

        entries = ingester._extract_from_codex_jsonl(sample_codex_jsonl)
        if entries:
            assert "---" in entries[0]["insight"]


class TestExtractDate:
    """Tests for date extraction from Codex file paths."""

    def test_extracts_date_from_directory_structure(self, tmp_path):
        ingester = CodexIngester.__new__(CodexIngester)
        path = tmp_path / "sessions" / "2026" / "03" / "05" / "rollout.jsonl"
        path.parent.mkdir(parents=True)
        path.touch()
        assert ingester._extract_date(path) == "2026-03-05"

    def test_extracts_date_from_filename(self, tmp_path):
        ingester = CodexIngester.__new__(CodexIngester)
        path = tmp_path / "2026-03-04_session.jsonl"
        path.touch()
        assert ingester._extract_date(path) == "2026-03-04"

    def test_falls_back_to_mtime(self, tmp_path):
        ingester = CodexIngester.__new__(CodexIngester)
        path = tmp_path / "random-id.jsonl"
        path.write_text("test")
        date = ingester._extract_date(path)
        assert len(date) == 10
        assert date[4] == "-"


class TestCodexIngest:
    """Tests for the main Codex ingest pipeline."""

    def test_returns_zero_when_no_codex_dir(self, tmp_path):
        ingester = CodexIngester(str(tmp_path))
        assert ingester.ingest() == 0

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_ingest_processes_files(self, mock_db_cls, project_with_codex, codex_dir):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        ingester = CodexIngester(
            str(project_with_codex),
            codex_dir=str(codex_dir.parent.parent.parent),  # point to sessions root
        )
        count = ingester.ingest()
        assert count >= 1
        mock_db.batch_add.assert_called_once()

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_incremental_skips_processed(self, mock_db_cls, project_with_codex, codex_dir):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        ingester = CodexIngester(
            str(project_with_codex),
            codex_dir=str(codex_dir.parent.parent.parent),
        )
        ingester.ingest()
        count = ingester.ingest()
        assert count == 0

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_full_reprocesses_all(self, mock_db_cls, project_with_codex, codex_dir):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        ingester = CodexIngester(
            str(project_with_codex),
            codex_dir=str(codex_dir.parent.parent.parent),
        )
        ingester.ingest()
        count = ingester.ingest(full=True)
        assert count >= 1

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_discovers_date_nested_files(self, mock_db_cls, project_with_codex, tmp_path):
        """JSONL files in YYYY/MM/DD subdirectories should be discovered."""
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        sessions_root = tmp_path / "codex_root"
        date_dir = sessions_root / "2026" / "03" / "04"
        date_dir.mkdir(parents=True)
        path = date_dir / "rollout-xyz.jsonl"
        records = [
            _codex_user_record("Analyze this codebase"),
            _codex_assistant_record(SUBSTANTIVE_ANSWER),
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ingester = CodexIngester(
            str(project_with_codex),
            codex_dir=str(sessions_root),
        )
        count = ingester.ingest()
        assert count >= 1


class TestCodexRegistry:
    """Tests for Codex session registry persistence."""

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_registry_saved_after_ingest(self, mock_db_cls, project_with_codex, codex_dir):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        ingester = CodexIngester(
            str(project_with_codex),
            codex_dir=str(codex_dir.parent.parent.parent),
        )
        ingester.ingest()

        registry_path = project_with_codex / ".knowledge-db" / "codex-registry.json"
        assert registry_path.exists()

        registry = json.loads(registry_path.read_text())
        # Registry keys are now full paths (not just filenames)
        assert len(registry) == 1
        key = next(iter(registry))
        assert key.endswith("rollout-abc123.jsonl")
        assert "entry_ids" in registry[key]


class TestCodexCleanup:
    """Tests for stale entry cleanup."""

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_deleted_codex_entries_removed(
        self, mock_db_cls, project_with_codex, codex_dir, sample_codex_jsonl,
    ):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db.remove_entries.return_value = 1
        mock_db_cls.return_value = mock_db

        codex_root = codex_dir.parent.parent.parent
        ingester = CodexIngester(str(project_with_codex), codex_dir=str(codex_root))
        ingester.ingest()

        sample_codex_jsonl.unlink()

        ingester2 = CodexIngester(str(project_with_codex), codex_dir=str(codex_root))
        ingester2.ingest()

        mock_db.remove_entries.assert_called_with(["id1"])
        assert len(ingester2._registry) == 0
