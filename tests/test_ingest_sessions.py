"""Tests for Claude Code JSONL session ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knowlin_mcp.ingest_sessions import SessionIngester


def _user_record(content):
    """Build a Claude Code JSONL user record."""
    return {
        "type": "user",
        "message": {"role": "user", "content": content},
        "uuid": "u1",
        "timestamp": "2026-03-05T10:00:00Z",
    }


def _assistant_record(content):
    """Build a Claude Code JSONL assistant record."""
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": content},
        "uuid": "a1",
        "timestamp": "2026-03-05T10:00:01Z",
    }


def _progress_record():
    """Build a progress record (noise)."""
    return {"type": "progress", "data": {"toolUseID": "t1"}, "timestamp": "2026-03-05T10:00:00Z"}


def _tool_result_user(tool_use_id="t1"):
    """Build a user record containing only tool_result blocks (noise)."""
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": "file contents..."}
            ],
        },
        "uuid": "u2",
        "timestamp": "2026-03-05T10:00:02Z",
    }


def _queue_enqueue(text):
    """Build a queue-operation enqueue record."""
    return {"type": "queue-operation", "operation": "enqueue", "content": text}


@pytest.fixture
def session_dir(tmp_path):
    """Create a mock sessions directory with JSONL files."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    return sessions


@pytest.fixture
def sample_jsonl(session_dir):
    """Create a sample JSONL session file using real Claude Code format."""
    path = session_dir / "test-session.jsonl"
    records = [
        _user_record("How do I fix the authentication bug?"),
        _assistant_record([
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
        ]),
        _user_record("Thanks"),
        _assistant_record("You're welcome!"),
    ]
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
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

    def test_skip_patterns_return_zero_for_short_messages(self):
        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        for text in ["I'll check that for you", "Let me look at the code", "Sure, I can do that", "Done."]:
            score, _ = ingester._score_content(text)
            assert score == 0.0, f"Should skip short message: {text!r}"

    def test_skip_patterns_do_not_apply_to_long_messages(self):
        """Messages >= SUBSTANTIVE_LENGTH should not be killed by skip patterns."""
        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        long_text = "I'll start by reviewing the exact state of the codebase. " + "x" * 300
        score, _ = ingester._score_content(long_text)
        assert score > 0.0, "Long message starting with 'I'll' should not be skipped"

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

    def test_markdown_structure_bonus(self):
        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        plain = "The solution is to update the config. " * 5
        structured = "## Solution\n\n| Step | Action |\n|---|---|\n| 1 | Update config |\n" + plain

        score_plain, _ = ingester._score_content(plain)
        score_structured, _ = ingester._score_content(structured)
        assert score_structured > score_plain


class TestIsRealUserMessage:
    """Tests for user message noise filtering."""

    def test_plain_string_accepted(self):
        ingester = SessionIngester.__new__(SessionIngester)
        assert ingester._is_real_user_message("How do I fix this bug?") == "How do I fix this bug?"

    def test_tool_result_filtered(self):
        ingester = SessionIngester.__new__(SessionIngester)
        content = [{"type": "tool_result", "tool_use_id": "t1", "content": "file data"}]
        assert ingester._is_real_user_message(content) == ""

    def test_command_name_filtered(self):
        ingester = SessionIngester.__new__(SessionIngester)
        text = '<command-name>/compact</command-name><command-message>compact</command-message>'
        assert ingester._is_real_user_message(text) == ""

    def test_system_reminder_filtered(self):
        ingester = SessionIngester.__new__(SessionIngester)
        text = '<system-reminder>Some system context here</system-reminder>'
        assert ingester._is_real_user_message(text) == ""

    def test_continuation_summary_filtered(self):
        ingester = SessionIngester.__new__(SessionIngester)
        text = "This session is being continued from a previous conversation that ran out of context."
        assert ingester._is_real_user_message(text) == ""

    def test_text_blocks_extracted(self):
        ingester = SessionIngester.__new__(SessionIngester)
        content = [{"type": "text", "text": "What does this function do?"}]
        assert ingester._is_real_user_message(content) == "What does this function do?"

    def test_slash_command_definitions_filtered(self):
        ingester = SessionIngester.__new__(SessionIngester)
        content = [{"type": "text", "text": "## Triggers\n- When the user asks..."}]
        assert ingester._is_real_user_message(content) == ""


class TestExtractFromJsonl:
    """Tests for JSONL parsing and extraction."""

    def test_extracts_high_value_entries(self, sample_jsonl):
        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        entries = ingester._extract_from_jsonl(sample_jsonl)
        assert len(entries) >= 1
        # The "You're welcome!" should be filtered out (too short)
        titles = [e["title"] for e in entries]
        assert not any("welcome" in t.lower() for t in titles)

    def test_uses_record_type_not_role(self, session_dir):
        """Verify we dispatch on record['type'], not record['role']."""
        path = session_dir / "type-dispatch.jsonl"
        records = [
            _user_record("What is the cause of this error?"),
            _assistant_record([{
                "type": "text",
                "text": "The root cause was a missing configuration value. The fix is to add "
                        "the DATABASE_URL environment variable. I discovered that the default "
                        "fallback was removed in the latest migration.",
            }]),
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        entries = ingester._extract_from_jsonl(path)
        assert len(entries) >= 1, "Should extract entries using record['type'] dispatch"

    def test_skips_progress_and_system_records(self, session_dir):
        path = session_dir / "noise.jsonl"
        records = [
            _progress_record(),
            {"type": "system", "subtype": "turn_duration", "durationMs": 5000},
            {"type": "file-history-snapshot", "snapshot": {}},
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        entries = ingester._extract_from_jsonl(path)
        assert len(entries) == 0

    def test_filters_tool_result_user_turns(self, session_dir):
        """tool_result user records should not overwrite last_user_text."""
        path = session_dir / "tool-results.jsonl"
        records = [
            _user_record("What is the fix for the auth bug?"),
            _assistant_record([{"type": "tool_use", "name": "Read", "input": {"file": "auth.py"}}]),
            _tool_result_user("t1"),  # Should NOT overwrite last_user_text
            _assistant_record([{
                "type": "text",
                "text": "The root cause of the auth bug was that the JWT token validation "
                        "was checking the wrong issuer field. I found that the token was being "
                        "validated against the old provider instead of the new one.",
            }]),
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        entries = ingester._extract_from_jsonl(path)
        assert len(entries) >= 1
        # Title should come from the real user question, not be empty
        assert "auth" in entries[0]["title"].lower()

    def test_queue_enqueue_captures_user_text(self, session_dir):
        """queue-operation enqueue events provide clean user input."""
        path = session_dir / "queue.jsonl"
        records = [
            _queue_enqueue("How do I configure the database connection pool?"),
            _assistant_record([{
                "type": "text",
                "text": "The root cause of slow queries was the connection pool size. "
                        "The fix is to increase max_connections in the pool config. "
                        "I found that the default of 5 was too low for production load.",
            }]),
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        entries = ingester._extract_from_jsonl(path)
        assert len(entries) >= 1
        assert "database" in entries[0]["title"].lower()

    def test_skips_tool_use_only_assistant_records(self, session_dir):
        """Assistant records with only tool_use blocks have no extractable text."""
        path = session_dir / "tool-only.jsonl"
        records = [
            _assistant_record([{"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}]),
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        entries = ingester._extract_from_jsonl(path)
        assert len(entries) == 0

    def test_handles_content_blocks(self, session_dir):
        path = session_dir / "blocks.jsonl"
        records = [
            _assistant_record([
                {"type": "text", "text": "The root cause of the problem was "},
                {
                    "type": "text",
                    "text": (
                        "an incorrect configuration. The fix resolved the issue because "
                        "it updated the correct field. I discovered the old value was stale."
                    ),
                },
            ]),
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        entries = ingester._extract_from_jsonl(path)
        assert len(entries) >= 1

    def test_handles_malformed_jsonl(self, session_dir):
        path = session_dir / "bad.jsonl"
        with open(path, "w") as f:
            f.write("not json\n")
            f.write(json.dumps(_assistant_record("valid but short")) + "\n")
            f.write("{malformed json}\n")

        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

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
            assert "priority" in entry
            assert "source" in entry
            assert entry["source"].startswith("session:")

    def test_entry_type_inferred_from_content(self, session_dir):
        """Entry type should be inferred from value signals, not always 'session'."""
        path = session_dir / "typed.jsonl"
        records = [
            _assistant_record([{
                "type": "text",
                "text": "The root cause was a race condition. The fix is to add a mutex lock "
                        "around the critical section. This resolved the intermittent failures.",
            }]),
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        entries = ingester._extract_from_jsonl(path)
        if entries:
            assert entries[0]["type"] == "solution"

    def test_insight_includes_user_question(self, sample_jsonl):
        ingester = SessionIngester.__new__(SessionIngester)
        ingester._registry = {}

        entries = ingester._extract_from_jsonl(sample_jsonl)
        if entries:
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
        assert len(date) == 10
        assert date[4] == "-"


class TestFindSessionsDir:
    """Tests for session directory discovery."""

    def test_exact_match_on_mangled_path(self, tmp_path):
        ingester = SessionIngester.__new__(SessionIngester)
        ingester.project_path = tmp_path / "home" / "user" / "project"
        ingester.project_path.mkdir(parents=True)

        claude_dir = tmp_path / ".claude" / "projects"
        mangled = str(ingester.project_path).replace("/", "-")
        project_dir = claude_dir / mangled
        project_dir.mkdir(parents=True)
        (project_dir / "session.jsonl").touch()

        with patch.object(Path, "home", return_value=tmp_path):
            result = ingester._find_sessions_dir()
            assert result == project_dir


class TestIngest:
    """Tests for the main ingest pipeline."""

    def test_returns_zero_when_no_sessions_dir(self, tmp_path):
        ingester = SessionIngester(str(tmp_path))
        assert ingester.ingest() == 0

    @patch("knowlin_mcp.db.KnowledgeDB")
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

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_incremental_skips_processed(self, mock_db_cls, project_with_sessions, session_dir, sample_jsonl):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        ingester = SessionIngester(
            str(project_with_sessions),
            sessions_dir=str(session_dir),
        )
        ingester.ingest()
        count = ingester.ingest()
        assert count == 0

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_full_reprocesses_all(self, mock_db_cls, project_with_sessions, session_dir, sample_jsonl):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        ingester = SessionIngester(
            str(project_with_sessions),
            sessions_dir=str(session_dir),
        )
        ingester.ingest()
        count = ingester.ingest(full=True)
        assert count >= 1

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_discovers_subagent_files(self, mock_db_cls, project_with_sessions, session_dir):
        """JSONL files in subdirectories should be discovered."""
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        subdir = session_dir / "subagents"
        subdir.mkdir()
        path = subdir / "sub-session.jsonl"
        records = [
            _user_record("Analyze this codebase"),
            _assistant_record([{
                "type": "text",
                "text": "I found that the codebase uses a layered architecture. "
                        "The root cause of the complexity was the tight coupling between "
                        "the service and repository layers. The fix would be to introduce "
                        "proper dependency injection.",
            }]),
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ingester = SessionIngester(
            str(project_with_sessions),
            sessions_dir=str(session_dir),
        )
        count = ingester.ingest()
        assert count >= 1


class TestRegistry:
    """Tests for session registry persistence."""

    @patch("knowlin_mcp.db.KnowledgeDB")
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

        registry = json.loads(registry_path.read_text())
        assert "test-session.jsonl" in registry
        assert "entry_ids" in registry["test-session.jsonl"]


class TestCleanup:
    """Tests for stale entry cleanup on file deletion/modification."""

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_deleted_session_entries_removed(self, mock_db_cls, project_with_sessions, session_dir, sample_jsonl):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db.remove_entries.return_value = 1
        mock_db_cls.return_value = mock_db

        ingester = SessionIngester(
            str(project_with_sessions),
            sessions_dir=str(session_dir),
        )
        ingester.ingest()

        sample_jsonl.unlink()

        ingester2 = SessionIngester(
            str(project_with_sessions),
            sessions_dir=str(session_dir),
        )
        ingester2.ingest()

        mock_db.remove_entries.assert_called_with(["id1"])
        assert len(ingester2._registry) == 0

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_modified_session_old_entries_replaced(self, mock_db_cls, project_with_sessions, session_dir, sample_jsonl):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db.remove_entries.return_value = 1
        mock_db_cls.return_value = mock_db

        ingester = SessionIngester(
            str(project_with_sessions),
            sessions_dir=str(session_dir),
        )
        ingester.ingest()

        # Append a new substantive record
        with open(sample_jsonl, "a") as f:
            f.write(json.dumps(_assistant_record([{
                "type": "text",
                "text": "The root cause of the problem was a configuration error. "
                        "The fix resolved the timeout issue completely. I discovered "
                        "this by checking the connection pool settings.",
            }])) + "\n")

        mock_db.batch_add.return_value = ["id2", "id3"]

        ingester2 = SessionIngester(
            str(project_with_sessions),
            sessions_dir=str(session_dir),
        )
        ingester2.ingest()

        mock_db.remove_entries.assert_called_with(["id1"])

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_backward_compat_no_entry_ids(self, mock_db_cls, project_with_sessions, session_dir, sample_jsonl):
        """Old registries without entry_ids should not crash."""
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        registry_path = project_with_sessions / ".knowledge-db" / "session-registry.json"
        old_registry = {
            "old-session.jsonl": {
                "hash": "abc123",
                "processed": "2026-01-01T00:00:00",
                "entries_extracted": 5,
            }
        }
        registry_path.write_text(json.dumps(old_registry))

        ingester = SessionIngester(
            str(project_with_sessions),
            sessions_dir=str(session_dir),
        )
        ingester.ingest()
