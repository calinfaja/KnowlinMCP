"""Tests for V2 -> V3 schema migration in utils.py."""

from __future__ import annotations

from knowlin_mcp.utils import migrate_entry


class TestMigrateInsight:
    """summary/atomic_insight -> insight."""

    def test_summary_becomes_insight(self):
        entry = migrate_entry({"title": "Test", "summary": "My summary"})
        assert entry["insight"] == "My summary"

    def test_atomic_insight_becomes_insight(self):
        entry = migrate_entry({"title": "Test", "atomic_insight": "Atomic text"})
        assert entry["insight"] == "Atomic text"

    def test_both_merged_when_different(self):
        entry = migrate_entry({
            "title": "Test",
            "atomic_insight": "Atomic",
            "summary": "Summary text",
        })
        assert entry["insight"] == "Atomic Summary text"

    def test_both_deduped_when_atomic_in_summary(self):
        entry = migrate_entry({
            "title": "Test",
            "atomic_insight": "Key insight",
            "summary": "Key insight with more detail",
        })
        assert entry["insight"] == "Key insight with more detail"

    def test_falls_back_to_title_when_empty(self):
        entry = migrate_entry({"title": "My Title"})
        assert entry["insight"] == "My Title"

    def test_existing_insight_not_overwritten(self):
        entry = migrate_entry({
            "title": "Test",
            "insight": "Already V3",
            "summary": "Old V2",
        })
        assert entry["insight"] == "Already V3"


class TestMigrateKeywords:
    """tags/key_concepts -> keywords."""

    def test_tags_become_keywords(self):
        entry = migrate_entry({"title": "Test", "tags": ["a", "b"]})
        assert entry["keywords"] == ["a", "b"]

    def test_concepts_become_keywords(self):
        entry = migrate_entry({"title": "Test", "key_concepts": ["x", "y"]})
        assert entry["keywords"] == ["x", "y"]

    def test_merged_and_deduped(self):
        entry = migrate_entry({
            "title": "Test",
            "tags": ["auth", "jwt"],
            "key_concepts": ["JWT", "security"],
        })
        kw = entry["keywords"]
        assert "auth" in kw
        assert "security" in kw
        # jwt/JWT should be deduped (case-insensitive)
        jwt_count = sum(1 for k in kw if k.lower() == "jwt")
        assert jwt_count == 1

    def test_existing_keywords_not_overwritten(self):
        entry = migrate_entry({
            "title": "Test",
            "keywords": ["existing"],
            "tags": ["ignored"],
        })
        assert entry["keywords"] == ["existing"]

    def test_empty_tags_gives_empty_keywords(self):
        entry = migrate_entry({"title": "Test"})
        assert entry["keywords"] == []


class TestMigrateSource:
    """url/source_path -> source."""

    def test_http_url_becomes_source(self):
        entry = migrate_entry({"title": "Test", "url": "https://example.com"})
        assert entry["source"] == "https://example.com"

    def test_source_path_gets_file_prefix(self):
        entry = migrate_entry({"title": "Test", "source_path": "/path/to/file.py"})
        assert entry["source"] == "file:/path/to/file.py"

    def test_source_path_already_prefixed(self):
        entry = migrate_entry({"title": "Test", "source_path": "file:already.py"})
        assert entry["source"] == "file:already.py"

    def test_http_source_path(self):
        entry = migrate_entry({"title": "Test", "source_path": "https://docs.rs"})
        assert entry["source"] == "https://docs.rs"

    def test_falls_back_to_conv_date(self):
        entry = migrate_entry({"title": "Test", "found_date": "2024-12-01T10:00:00"})
        assert entry["source"] == "conv:2024-12-01"

    def test_falls_back_to_conv_unknown(self):
        entry = migrate_entry({"title": "Test"})
        assert entry["source"] == "conv:unknown"

    def test_existing_real_source_not_overwritten(self):
        entry = migrate_entry({"title": "Test", "source": "file:my.py"})
        assert entry["source"] == "file:my.py"

    def test_generic_source_manual_overwritten(self):
        entry = migrate_entry({"title": "Test", "source": "manual", "url": "https://r.com"})
        assert entry["source"] == "https://r.com"

    def test_generic_source_conversation_overwritten(self):
        entry = migrate_entry({"title": "Test", "source": "conversation"})
        assert entry["source"].startswith("conv:")

    def test_generic_source_review_overwritten(self):
        entry = migrate_entry({"title": "Test", "source": "review"})
        assert entry["source"].startswith("conv:")


class TestMigrateDate:
    """found_date -> date."""

    def test_found_date_truncated_to_date(self):
        entry = migrate_entry({"title": "Test", "found_date": "2024-12-01T10:30:00"})
        assert entry["date"] == "2024-12-01"

    def test_missing_found_date_gives_empty(self):
        entry = migrate_entry({"title": "Test"})
        assert entry["date"] == ""

    def test_existing_date_not_overwritten(self):
        entry = migrate_entry({
            "title": "Test",
            "date": "2025-01-01",
            "found_date": "2024-12-01",
        })
        assert entry["date"] == "2025-01-01"


class TestMigratePriority:
    """quality/confidence_score/relevance_score -> priority."""

    def test_high_quality(self):
        entry = migrate_entry({"title": "Test", "quality": "high"})
        assert entry["priority"] == "high"

    def test_low_quality(self):
        entry = migrate_entry({"title": "Test", "quality": "low"})
        assert entry["priority"] == "low"

    def test_high_confidence(self):
        entry = migrate_entry({"title": "Test", "confidence_score": 0.95})
        assert entry["priority"] == "high"

    def test_low_confidence(self):
        entry = migrate_entry({"title": "Test", "confidence_score": 0.3})
        assert entry["priority"] == "low"

    def test_medium_default(self):
        entry = migrate_entry({"title": "Test"})
        assert entry["priority"] == "medium"

    def test_high_relevance(self):
        entry = migrate_entry({"title": "Test", "relevance_score": 0.95})
        assert entry["priority"] == "high"

    def test_low_relevance(self):
        entry = migrate_entry({"title": "Test", "relevance_score": 0.3})
        assert entry["priority"] == "low"

    def test_existing_priority_not_overwritten(self):
        entry = migrate_entry({
            "title": "Test",
            "priority": "critical",
            "quality": "low",
        })
        assert entry["priority"] == "critical"


class TestMigrateTimestamp:
    """found_date -> timestamp."""

    def test_iso_found_date_preserved(self):
        entry = migrate_entry({"title": "Test", "found_date": "2024-12-01T10:30:00"})
        assert entry["timestamp"] == "2024-12-01T10:30:00"

    def test_date_only_gets_midnight(self):
        entry = migrate_entry({"title": "Test", "found_date": "2024-12-01"})
        assert entry["timestamp"] == "2024-12-01T00:00:00"

    def test_no_found_date_gives_empty(self):
        entry = migrate_entry({"title": "Test"})
        assert entry["timestamp"] == ""


class TestMigrateV3Passthrough:
    """Already-V3 entries should pass through unchanged."""

    def test_v3_entry_unchanged(self):
        v3 = {
            "id": "abc",
            "title": "V3 Entry",
            "insight": "Already migrated",
            "type": "finding",
            "priority": "high",
            "keywords": ["test"],
            "source": "conv:2025-01-01",
            "date": "2025-01-01",
            "timestamp": "2025-01-01T12:00:00",
            "branch": "main",
            "related_to": [],
        }
        result = migrate_entry(v3.copy())
        assert result["insight"] == "Already migrated"
        assert result["keywords"] == ["test"]
        assert result["source"] == "conv:2025-01-01"
        assert result["date"] == "2025-01-01"
        assert result["priority"] == "high"


class TestMigrateType:
    """Generic types (lesson, best-practice) get re-inferred."""

    def test_lesson_type_replaced(self):
        entry = migrate_entry({"title": "Fixed the auth bug", "type": "lesson"})
        assert entry["type"] != "lesson"

    def test_best_practice_type_replaced(self):
        entry = migrate_entry({"title": "Always validate input", "type": "best-practice"})
        assert entry["type"] != "best-practice"

    def test_real_type_preserved(self):
        entry = migrate_entry({"title": "Test", "type": "warning"})
        assert entry["type"] == "warning"


class TestMigrateDefaults:
    """Minimal entry gets V3 defaults."""

    def test_minimal_entry_gets_defaults(self):
        entry = migrate_entry({"title": "Bare"})
        assert "branch" in entry
        assert "related_to" in entry
        assert isinstance(entry["related_to"], list)
        assert entry["type"] in (
            "finding", "solution", "pattern", "warning", "decision", "discovery",
        )
