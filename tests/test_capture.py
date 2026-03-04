"""Tests for capture.py - entry creation and save fallback chain."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from kln_knowledge.capture import (
    create_entry,
    create_entry_from_json,
    log_to_timeline,
    save_entry,
)


class TestCreateEntry:
    """Tests for simple entry creation."""

    def test_basic_entry(self):
        entry = create_entry("JWT tokens must be validated server-side", entry_type="warning")
        assert entry["type"] == "warning"
        assert entry["insight"] == "JWT tokens must be validated server-side"
        assert "id" in entry
        assert entry["id"].startswith("warning-")
        assert "date" in entry
        assert "timestamp" in entry

    def test_tags_from_string(self):
        entry = create_entry("test", entry_type="finding", tags="auth, security, jwt")
        assert entry["keywords"] == ["auth", "security", "jwt"]

    def test_tags_from_list(self):
        entry = create_entry("test", entry_type="finding", tags=["a", "b"])
        assert entry["keywords"] == ["a", "b"]

    def test_url_sets_source(self):
        entry = create_entry("test", url="https://example.com/doc")
        assert entry["source"] == "https://example.com/doc"

    def test_title_truncated_at_100(self):
        long_text = "x" * 200
        entry = create_entry(long_text)
        assert len(entry["title"]) <= 100
        assert entry["title"].endswith("...")

    def test_legacy_type_normalized(self):
        entry = create_entry("test", entry_type="lesson")
        assert entry["type"] == "finding"

    def test_priority_preserved(self):
        entry = create_entry("test", priority="critical")
        assert entry["priority"] == "critical"


class TestCreateEntryFromJson:
    """Tests for structured JSON entry creation."""

    def test_basic_json_entry(self):
        data = {
            "title": "Use connection pooling",
            "insight": "Pool size = 2 * cpu_count",
            "type": "solution",
        }
        entry = create_entry_from_json(data)
        assert entry["title"] == "Use connection pooling"
        assert entry["type"] == "solution"

    def test_infers_type_when_missing(self):
        data = {"title": "Be careful with timeouts", "insight": "Watch out for TCP timeout"}
        entry = create_entry_from_json(data)
        assert entry["type"] in ("finding", "warning", "solution", "pattern", "decision", "discovery")

    def test_merges_tags_and_concepts(self):
        data = {
            "title": "Test entry here",
            "insight": "Details",
            "type": "finding",
            "tags": ["a", "b"],
            "key_concepts": ["c", "d"],
        }
        entry = create_entry_from_json(data)
        assert "a" in entry["keywords"]
        assert "c" in entry["keywords"]

    def test_v2_summary_field(self):
        data = {"title": "Old format entry", "summary": "V2 summary text", "type": "finding"}
        entry = create_entry_from_json(data)
        assert entry["insight"] == "V2 summary text"

    def test_title_from_insight_when_missing(self):
        data = {"insight": "Long insight text that should become the title", "type": "finding"}
        entry = create_entry_from_json(data)
        assert entry["title"] == "Long insight text that should become the title"


class TestSaveEntry:
    """Tests for the save fallback chain."""

    def test_jsonl_fallback(self, tmp_path):
        """When server and DB both fail, falls back to JSONL append."""
        kb_dir = tmp_path / ".knowledge-db"
        kb_dir.mkdir()
        project_path = str(tmp_path)

        entry = create_entry("Test fallback save entry content")

        # Mock server and DB to fail
        with patch("kln_knowledge.capture.send_entry_to_server", return_value=False), \
             patch("kln_knowledge.db.KnowledgeDB.add", side_effect=Exception("no DB")), \
             patch("kln_knowledge.capture._notify_server_reload"):
            result = save_entry(entry, kb_dir)

        assert result is True
        # Entry should be in JSONL
        jsonl = kb_dir / "entries.jsonl"
        assert jsonl.exists()
        saved = json.loads(jsonl.read_text().strip())
        assert saved["id"] == entry["id"]


class TestLogToTimeline:
    """Tests for timeline logging."""

    def test_appends_to_timeline(self, tmp_path):
        log_to_timeline("Test insight content", "finding", tmp_path)
        timeline = tmp_path / "timeline.txt"
        assert timeline.exists()
        content = timeline.read_text()
        assert "finding" in content
        assert "Test insight" in content

    def test_multiple_entries(self, tmp_path):
        log_to_timeline("First entry", "finding", tmp_path)
        log_to_timeline("Second entry", "warning", tmp_path)
        lines = (tmp_path / "timeline.txt").read_text().strip().split("\n")
        assert len(lines) == 2
