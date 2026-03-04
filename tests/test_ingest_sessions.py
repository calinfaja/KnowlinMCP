"""Tests for Claude Code JSONL session ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kln_knowledge.ingest_sessions import SessionIngester, _VALUE_SIGNALS, _SKIP_PATTERNS


@pytest.fixture
def session_dir(tmp_path):
    """Create a mock sessions directory with JSONL files."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    return sessions


@pytest.fixture
def sample_jsonl(session_dir):
    """Create a sample JSONL session file."""
    path = session_dir / "test-session.jsonl"
    messages = [
        {"role": "user", "content": "How do I fix the authentication bug?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "The root cause of the authentication issue was that the JWT token "
                        "validation was checking the wrong issuer field. The fix is to update "
                        "the issuer config in auth.py. I found that the token was being validated "
                        "against 'old-issuer' instead of 'new-issuer'. This is a common gotcha "
                        "when migrating authentication providers."
                    ),
                }
            ],
        },
        {"role": "user", "content": "Thanks"},
        {
            "role": "assistant",
            "content": "You're welcome!",
        },
    ]
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return path


@pytest.fixture
def project_with_sessions(tmp_path, session_dir, sample_jsonl):
    """Create a project directory with sessions."""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".knowledge-db").mkdir()
    return project


class TestScoreContent:
    """Tests for content importance scoring."""

    def test_high_value_content_scores_high(self):
        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        text = (
            "The root cause was a race condition in the connection pool. "
            "The fix is to add a mutex lock around the connection checkout. "
            "Be careful with the timeout setting - it can cause deadlocks."
        )
        score, entry_type = ingester._score_content(text)
        assert score > 0.5
        assert entry_type in ("solution", "warning")

    def test_low_value_content_scores_low(self):
        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        text = "Here is the updated code with the changes applied."
        score, _ = ingester._score_content(text)
        assert score < 0.3

    def test_skip_patterns_return_zero(self):
        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        for text in ["I'll check that for you", "Let me look at the code", "Sure, I can do that", "Done."]:
            score, _ = ingester._score_content(text)
            assert score == 0.0, f"Should skip: {text!r}"

    def test_code_blocks_get_bonus(self):
        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        text_no_code = "The solution is to update the config with the correct values for production."
        text_with_code = text_no_code + "\n```python\nconfig['key'] = 'value'\n```"

        score_no_code, _ = ingester._score_content(text_no_code)
        score_with_code, _ = ingester._score_content(text_with_code)
        assert score_with_code > score_no_code

    def test_decision_type_detected(self):
        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        text = "We decided to go with Redis instead of Memcached because of the persistence features."
        _, entry_type = ingester._score_content(text)
        assert entry_type == "decision"

    def test_discovery_type_detected(self):
        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        text = "I found that the API endpoint was returning cached data. Turns out the CDN layer was caching aggressively."
        _, entry_type = ingester._score_content(text)
        assert entry_type == "discovery"


class TestExtractFromJsonl:
    """Tests for JSONL parsing and extraction."""

    def test_extracts_high_value_entries(self, sample_jsonl):
        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        entries = ingester._extract_from_jsonl(sample_jsonl)
        # The substantive assistant message should be extracted
        assert len(entries) >= 1
        # The "You're welcome!" should be filtered out (too short)
        titles = [e["title"] for e in entries]
        assert not any("welcome" in t.lower() for t in titles)

    def test_skips_user_messages(self, session_dir):
        path = session_dir / "user-only.jsonl"
        messages = [
            {"role": "user", "content": "Do something"},
            {"role": "user", "content": "Do something else"},
        ]
        with open(path, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        entries = ingester._extract_from_jsonl(path)
        assert len(entries) == 0

    def test_handles_content_blocks(self, session_dir):
        path = session_dir / "blocks.jsonl"
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "The root cause of the problem was "},
                    {"type": "text", "text": "an incorrect configuration. The fix resolved the issue because it updated the correct field."},
                ],
            },
        ]
        with open(path, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        entries = ingester._extract_from_jsonl(path)
        assert len(entries) >= 1

    def test_handles_malformed_jsonl(self, session_dir):
        path = session_dir / "bad.jsonl"
        with open(path, "w") as f:
            f.write("not json\n")
            f.write('{"role": "assistant", "content": "valid but short"}\n')
            f.write("{malformed json}\n")

        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        # Should not raise
        entries = ingester._extract_from_jsonl(path)
        assert isinstance(entries, list)

    def test_entry_has_required_fields(self, sample_jsonl):
        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        entries = ingester._extract_from_jsonl(sample_jsonl)
        if entries:
            entry = entries[0]
            assert "title" in entry
            assert "insight" in entry
            assert "type" in entry
            assert entry["type"] == "session"
            assert "priority" in entry
            assert "source" in entry
            assert entry["source"].startswith("session:")

    def test_insight_includes_user_question(self, sample_jsonl):
        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        entries = ingester._extract_from_jsonl(sample_jsonl)
        if entries:
            # The insight should contain the user's question
            assert "---" in entries[0]["insight"]


class TestExtractDate:
    """Tests for date extraction from file paths."""

    def test_extracts_date_from_filename(self, tmp_path):
        ingester = SessionIngester.__new__(SessionIngester)
        path = tmp_path / "2026-03-04_session.jsonl"
        path.touch()
        assert ingester._extract_date(path) == "2026-03-04"

    def test_falls_back_to_mtime(self, tmp_path):
        ingester = SessionIngester.__new__(SessionIngester)
        path = tmp_path / "random-session-id.jsonl"
        path.write_text("test")
        date = ingester._extract_date(path)
        # Should be a valid date string
        assert len(date) == 10
        assert date[4] == "-"


class TestIngest:
    """Tests for the main ingest pipeline."""

    def test_returns_zero_when_no_sessions_dir(self, tmp_path):
        ingester = SessionIngester(str(tmp_path))
        assert ingester.ingest() == 0

    @patch("kln_knowledge.db.KnowledgeDB")
    def test_ingest_processes_files(self, mock_db_cls, project_with_sessions, session_dir, sample_jsonl):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        ingester = SessionIngester(
            str(project_with_sessions),
            sessions_dir=str(session_dir),
        )
        count = ingester.ingest()
        assert count >= 1
        mock_db.batch_add.assert_called_once()

    @patch("kln_knowledge.db.KnowledgeDB")
    def test_incremental_skips_processed(self, mock_db_cls, project_with_sessions, session_dir, sample_jsonl):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        ingester = SessionIngester(
            str(project_with_sessions),
            sessions_dir=str(session_dir),
        )
        # First ingest
        ingester.ingest()
        # Second ingest should skip
        count = ingester.ingest()
        assert count == 0

    @patch("kln_knowledge.db.KnowledgeDB")
    def test_full_reprocesses_all(self, mock_db_cls, project_with_sessions, session_dir, sample_jsonl):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        ingester = SessionIngester(
            str(project_with_sessions),
            sessions_dir=str(session_dir),
        )
        # First ingest
        ingester.ingest()
        # Full re-ingest should process again
        count = ingester.ingest(full=True)
        assert count >= 1


class TestRegistry:
    """Tests for session registry persistence."""

    @patch("kln_knowledge.db.KnowledgeDB")
    def test_registry_saved_after_ingest(self, mock_db_cls, project_with_sessions, session_dir, sample_jsonl):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        ingester = SessionIngester(
            str(project_with_sessions),
            sessions_dir=str(session_dir),
        )
        ingester.ingest()

        registry_path = project_with_sessions / ".knowledge-db" / "session-registry.json"
        assert registry_path.exists()

        import json
        registry = json.loads(registry_path.read_text())
        assert "test-session.jsonl" in registry
