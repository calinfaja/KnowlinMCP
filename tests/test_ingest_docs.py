"""Tests for markdown/PDF document ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knowlin_mcp.ingest_docs import (
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    DocsIngester,
    _resolve_paths,
    load_sources_config,
)


@pytest.fixture
def docs_dir(tmp_path):
    """Create a docs directory with sample markdown files."""
    docs = tmp_path / "docs"
    docs.mkdir()
    return docs


@pytest.fixture
def sample_markdown(docs_dir):
    """Create a sample markdown file."""
    content = """# Getting Started

This is the introduction to the project. It covers the basic setup
and configuration needed to get up and running.

## Installation

Install the package using pip:

```bash
pip install my-package
```

Make sure you have Python 3.9 or later installed.

## Configuration

Create a config file at `~/.config/myapp/config.yaml`:

```yaml
database:
  host: localhost
  port: 5432
```

### Advanced Configuration

For production deployments, you should also configure:

- Connection pooling
- SSL certificates
- Logging levels

# API Reference

## Authentication

All API requests require a Bearer token in the Authorization header.

## Endpoints

### GET /api/users

Returns a list of users.

### POST /api/users

Creates a new user.
"""
    path = docs_dir / "guide.md"
    path.write_text(content)
    return path


@pytest.fixture
def project_with_docs(tmp_path, docs_dir):
    """Create a project directory with docs."""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".knowledge-db").mkdir()
    # Symlink docs dir into project
    (project / "docs").symlink_to(docs_dir)
    return project


class TestChunkByHeadings:
    """Tests for heading-based markdown chunking."""

    def test_splits_at_headings(self):
        ingester = DocsIngester.__new__(DocsIngester)
        text = (
            "# Title\n\nIntro paragraph with enough content to pass minimum."
            "\n\n## Section\n\nSection content that is also long enough to be meaningful here."
        )
        chunks = ingester._chunk_by_headings(text, "test.md")
        assert len(chunks) >= 1

    def test_contextual_enrichment_stores_hierarchy_separately(self):
        """Heading hierarchy is stored as context_prefix, not baked into insight."""
        ingester = DocsIngester.__new__(DocsIngester)
        text = (
            "# Main\n\nIntro text that is long enough to be a chunk.\n\n"
            "## Sub\n\nSub content that is also long enough to pass the minimum threshold.\n\n"
            "### Subsub\n\nDeep content that should have full hierarchy"
            " prepended for contextual retrieval."
        )
        chunks = ingester._chunk_by_headings(text, "test.md")
        # Find the deepest chunk - hierarchy is in context_prefix, content in insight
        deep_chunks = [c for c in chunks if "Deep content" in c["insight"]]
        assert len(deep_chunks) == 1
        assert deep_chunks[0]["context_prefix"] == "Main > Sub > Subsub"
        assert deep_chunks[0]["insight"].startswith("Deep content")

    def test_skips_tiny_chunks(self):
        ingester = DocsIngester.__new__(DocsIngester)
        text = (
            "# Title\n\nOK\n\n## Next\n\nThis section has enough content to be meaningful"
            " and pass the minimum character threshold for chunking."
        )
        chunks = ingester._chunk_by_headings(text, "test.md")
        # "OK" is too short, should be skipped
        for chunk in chunks:
            assert len(chunk["insight"]) >= MIN_CHUNK_CHARS

    def test_empty_text_returns_empty(self):
        ingester = DocsIngester.__new__(DocsIngester)
        assert ingester._chunk_by_headings("", "test.md") == []
        assert ingester._chunk_by_headings("   \n  ", "test.md") == []

    def test_chunk_has_required_fields(self, sample_markdown):
        ingester = DocsIngester.__new__(DocsIngester)
        text = sample_markdown.read_text()
        chunks = ingester._chunk_by_headings(text, "guide.md")

        assert len(chunks) > 0
        for chunk in chunks:
            assert "title" in chunk
            assert "insight" in chunk
            assert "type" in chunk
            assert "source" in chunk
            assert chunk["source"].startswith("doc:")

    def test_chunk_insight_capped_at_max(self):
        ingester = DocsIngester.__new__(DocsIngester)
        # Create a very long section
        long_text = (
            "# Title\n\n" + "x" * (MAX_CHUNK_CHARS * 3)
            + "\n\n## Next\n\nShort content that is long enough to pass"
            " the minimum threshold for chunking in tests."
        )
        chunks = ingester._chunk_by_headings(long_text, "test.md")
        for chunk in chunks:
            assert len(chunk["insight"]) <= MAX_CHUNK_CHARS


class TestSubSplit:
    """Tests for oversized chunk splitting."""

    def test_splits_large_chunks(self):
        ingester = DocsIngester.__new__(DocsIngester)
        chunk = {
            "title": "Big Section",
            "insight": "Word " * (MAX_CHUNK_CHARS // 3),
            "type": "document",
            "source": "doc:test.md",
            "_content_hash": "abc123",
        }
        sub_chunks = ingester._sub_split(chunk)
        assert len(sub_chunks) >= 2
        for sc in sub_chunks:
            assert "(part" in sc["title"]

    def test_preserves_chunk_metadata(self):
        ingester = DocsIngester.__new__(DocsIngester)
        chunk = {
            "title": "Section",
            "insight": "Content " * 500,
            "type": "document",
            "source": "doc:test.md",
            "priority": "medium",
            "_content_hash": "abc123",
        }
        sub_chunks = ingester._sub_split(chunk)
        for sc in sub_chunks:
            assert sc["type"] == "document"
            assert sc["source"] == "doc:test.md"


class TestRecursiveSplit:
    """Tests for recursive text splitting."""

    def test_returns_text_if_under_limit(self):
        ingester = DocsIngester.__new__(DocsIngester)
        text = "Short text"
        result = ingester._recursive_split(text, ["\n\n", "\n", ". "], 1000)
        assert result == [text]

    def test_splits_on_paragraph_boundaries(self):
        ingester = DocsIngester.__new__(DocsIngester)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = ingester._recursive_split(text, ["\n\n", "\n", ". "], 30)
        assert len(result) >= 2

    def test_falls_back_to_finer_separators(self):
        ingester = DocsIngester.__new__(DocsIngester)
        text = (
            "This is a single paragraph with no double newlines but several sentences."
            " It should split on sentences. Like this one here."
        )
        result = ingester._recursive_split(text, ["\n\n", "\n", ". "], 60)
        assert len(result) >= 2


class TestMakeChunk:
    """Tests for chunk creation."""

    def test_creates_valid_entry(self):
        ingester = DocsIngester.__new__(DocsIngester)
        chunk = ingester._make_chunk(
            content="This is the content of the chunk",
            heading="My Section",
            heading_stack={1: "Main", 2: "My Section"},
            source_path="guide.md",
        )
        assert chunk["title"] == "My Section"
        assert chunk["source"] == "doc:guide.md"
        assert chunk["type"] == "document"
        assert "_content_hash" in chunk

    def test_truncates_long_titles(self):
        ingester = DocsIngester.__new__(DocsIngester)
        chunk = ingester._make_chunk(
            content="Content",
            heading="A" * 200,
            heading_stack={},
            source_path="test.md",
        )
        assert len(chunk["title"]) <= 100

    def test_uses_content_as_fallback_title(self):
        ingester = DocsIngester.__new__(DocsIngester)
        chunk = ingester._make_chunk(
            content="This is a chunk without a heading",
            heading="",
            heading_stack={},
            source_path="test.md",
        )
        assert chunk["title"].startswith("This is a chunk")


class TestContentHash:
    """Tests for content hashing."""

    def test_same_content_same_hash(self):
        ingester = DocsIngester.__new__(DocsIngester)
        h1 = ingester._content_hash("hello world")
        h2 = ingester._content_hash("hello world")
        assert h1 == h2

    def test_different_content_different_hash(self):
        ingester = DocsIngester.__new__(DocsIngester)
        h1 = ingester._content_hash("hello world")
        h2 = ingester._content_hash("goodbye world")
        assert h1 != h2

    def test_normalizes_line_endings(self):
        ingester = DocsIngester.__new__(DocsIngester)
        h1 = ingester._content_hash("line1\nline2")
        h2 = ingester._content_hash("line1\r\nline2")
        assert h1 == h2

    def test_strips_whitespace(self):
        ingester = DocsIngester.__new__(DocsIngester)
        h1 = ingester._content_hash("hello")
        h2 = ingester._content_hash("  hello  ")
        assert h1 == h2


class TestFindDocFiles:
    """Tests for document file discovery."""

    def test_finds_markdown_files(self, tmp_path, docs_dir):
        (docs_dir / "readme.md").write_text("# Test")
        (docs_dir / "guide.txt").write_text("Guide")
        (docs_dir / "image.png").write_bytes(b"\x89PNG")

        ingester = DocsIngester.__new__(DocsIngester)
        ingester.docs_dirs = [docs_dir]
        ingester._include_globs = None
        ingester._exclude_globs = []

        files = ingester._find_doc_files()
        suffixes = {f.suffix.lower() for f in files}
        assert ".md" in suffixes
        assert ".txt" in suffixes
        assert ".png" not in suffixes

    def test_finds_pdf_files(self, docs_dir):
        (docs_dir / "spec.pdf").write_bytes(b"%PDF-1.4 fake")

        ingester = DocsIngester.__new__(DocsIngester)
        ingester.docs_dirs = [docs_dir]
        ingester._include_globs = None
        ingester._exclude_globs = []

        files = ingester._find_doc_files()
        assert any(f.suffix == ".pdf" for f in files)


class TestReadFile:
    """Tests for file reading."""

    def test_reads_markdown(self, docs_dir):
        md = docs_dir / "test.md"
        md.write_text("# Hello\n\nWorld")

        ingester = DocsIngester.__new__(DocsIngester)
        content = ingester._read_file(md)
        assert "Hello" in content

    def test_reads_txt(self, docs_dir):
        txt = docs_dir / "test.txt"
        txt.write_text("Plain text content")

        ingester = DocsIngester.__new__(DocsIngester)
        content = ingester._read_file(txt)
        assert "Plain text" in content

    def test_returns_empty_for_unknown_types(self, docs_dir):
        img = docs_dir / "test.png"
        img.write_bytes(b"\x89PNG")

        ingester = DocsIngester.__new__(DocsIngester)
        content = ingester._read_file(img)
        assert content == ""


class TestIngest:
    """Tests for the main ingest pipeline."""

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_ingest_processes_docs(self, mock_db_cls, tmp_path, docs_dir, sample_markdown):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1", "id2"]
        mock_db_cls.return_value = mock_db

        project = tmp_path / "project"
        project.mkdir()
        (project / ".knowledge-db").mkdir()

        ingester = DocsIngester(str(project), docs_path=str(docs_dir))
        count = ingester.ingest()
        assert count >= 1
        mock_db.batch_add.assert_called_once()

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_incremental_skips_unchanged(self, mock_db_cls, tmp_path, docs_dir, sample_markdown):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        project = tmp_path / "project"
        project.mkdir()
        (project / ".knowledge-db").mkdir()

        ingester = DocsIngester(str(project), docs_path=str(docs_dir))
        ingester.ingest()

        # Second ingest should skip unchanged files
        count = ingester.ingest()
        assert count == 0

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_full_reprocesses_changed_content(
        self, mock_db_cls, tmp_path, docs_dir, sample_markdown
    ):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        project = tmp_path / "project"
        project.mkdir()
        (project / ".knowledge-db").mkdir()

        ingester = DocsIngester(str(project), docs_path=str(docs_dir))
        ingester.ingest()

        # Modify the file so chunks are different
        sample_markdown.write_text("# New Content\n\n" + "Updated content " * 50)

        count = ingester.ingest(full=True)
        assert count >= 1

    def test_returns_zero_when_no_docs(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        (project / ".knowledge-db").mkdir()

        ingester = DocsIngester(str(project), docs_path=str(tmp_path / "nonexistent"))
        count = ingester.ingest()
        assert count == 0


class TestRegistry:
    """Tests for document registry persistence."""

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_registry_saved_after_ingest(self, mock_db_cls, tmp_path, docs_dir, sample_markdown):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        project = tmp_path / "project"
        project.mkdir()
        (project / ".knowledge-db").mkdir()

        ingester = DocsIngester(str(project), docs_path=str(docs_dir))
        ingester.ingest()

        registry_path = project / ".knowledge-db" / "doc-registry.json"
        assert registry_path.exists()

        registry = json.loads(registry_path.read_text())
        assert len(registry) > 0

        # Check registry entry structure
        for key, entry in registry.items():
            assert "file_hash" in entry
            assert "processed" in entry
            assert "chunk_hashes" in entry
            assert "chunk_count" in entry
            assert "entry_ids" in entry
            assert isinstance(entry["entry_ids"], list)


class TestCleanup:
    """Tests for stale entry cleanup on file deletion/modification."""

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_deleted_file_entries_removed(self, mock_db_cls, tmp_path, docs_dir):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db.remove_entries.return_value = 1
        mock_db_cls.return_value = mock_db

        project = tmp_path / "project"
        project.mkdir()
        (project / ".knowledge-db").mkdir()

        # Create and ingest a doc
        doc = docs_dir / "guide.md"
        doc.write_text("# Guide\n\nThis is a guide with enough content to be chunked properly.")

        ingester = DocsIngester(str(project), docs_path=str(docs_dir))
        ingester.ingest()

        # Verify registry has the file with entry_ids
        assert len(ingester._registry) == 1
        file_key = list(ingester._registry.keys())[0]
        assert ingester._registry[file_key]["entry_ids"] == ["id1"]

        # Delete the file
        doc.unlink()

        # Re-ingest -- should detect deletion and remove entries
        ingester2 = DocsIngester(str(project), docs_path=str(docs_dir))
        ingester2.ingest()

        # Registry should be empty now
        assert len(ingester2._registry) == 0

        # remove_entries should have been called with the old IDs
        mock_db.remove_entries.assert_called_with(["id1"])

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_modified_file_old_entries_replaced(self, mock_db_cls, tmp_path, docs_dir):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db.remove_entries.return_value = 1
        mock_db_cls.return_value = mock_db

        project = tmp_path / "project"
        project.mkdir()
        (project / ".knowledge-db").mkdir()

        doc = docs_dir / "guide.md"
        doc.write_text(
            "# Original\n\nOriginal content that is long enough to pass minimum threshold."
        )

        ingester = DocsIngester(str(project), docs_path=str(docs_dir))
        ingester.ingest()

        # Modify the file
        doc.write_text(
            "# Updated\n\nCompletely different content that replaces the original version entirely."
        )

        # New batch_add returns new IDs
        mock_db.batch_add.return_value = ["id2"]

        ingester2 = DocsIngester(str(project), docs_path=str(docs_dir))
        ingester2.ingest()

        # Old entries should be removed
        mock_db.remove_entries.assert_called_with(["id1"])

        # New IDs should be in registry
        registry = json.loads((project / ".knowledge-db" / "doc-registry.json").read_text())
        for entry in registry.values():
            assert entry["entry_ids"] == ["id2"]

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_entry_ids_stored_in_registry(self, mock_db_cls, tmp_path, docs_dir):
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["abc", "def", "ghi"]
        mock_db_cls.return_value = mock_db

        project = tmp_path / "project"
        project.mkdir()
        (project / ".knowledge-db").mkdir()

        doc = docs_dir / "multi.md"
        doc.write_text(
            "# Section One\n\nFirst section with enough content to be a chunk.\n\n"
            "## Section Two\n\nSecond section with enough content to be another chunk.\n\n"
            "## Section Three\n\nThird section with enough content for yet another chunk."
        )

        ingester = DocsIngester(str(project), docs_path=str(docs_dir))
        ingester.ingest()

        registry = json.loads((project / ".knowledge-db" / "doc-registry.json").read_text())
        for entry in registry.values():
            assert "entry_ids" in entry
            assert len(entry["entry_ids"]) > 0

    @patch("knowlin_mcp.db.KnowledgeDB")
    def test_backward_compat_no_entry_ids(self, mock_db_cls, tmp_path, docs_dir):
        """Old registries without entry_ids should not crash."""
        mock_db = MagicMock()
        mock_db.batch_add.return_value = ["id1"]
        mock_db_cls.return_value = mock_db

        project = tmp_path / "project"
        project.mkdir()
        db_dir = project / ".knowledge-db"
        db_dir.mkdir()

        # Write an old-format registry (no entry_ids)
        old_registry = {
            "/old/file.md": {
                "file_hash": "abc123",
                "processed": "2026-01-01T00:00:00",
                "chunk_hashes": ["h1"],
                "chunk_count": 1,
            }
        }
        (db_dir / "doc-registry.json").write_text(json.dumps(old_registry))

        # Create a real doc to process
        doc = docs_dir / "new.md"
        doc.write_text("# New\n\nNew content that is long enough.")

        # Should not crash
        ingester = DocsIngester(str(project), docs_path=str(docs_dir))
        ingester.ingest()


class TestSourcesConfig:
    """Tests for sources.yaml configuration."""

    def test_load_returns_none_when_no_file(self, tmp_path):
        assert load_sources_config(tmp_path) is None

    def test_load_parses_yaml(self, tmp_path):
        config_path = tmp_path / "sources.yaml"
        config_path.write_text("docs:\n  paths:\n    - docs/\n    - ~/INFOS/\n")
        result = load_sources_config(tmp_path)
        assert result is not None
        assert result["docs"]["paths"] == ["docs/", "~/INFOS/"]

    def test_load_handles_empty_file(self, tmp_path):
        (tmp_path / "sources.yaml").write_text("")
        result = load_sources_config(tmp_path)
        assert result == {}

    def test_load_handles_malformed_yaml(self, tmp_path):
        (tmp_path / "sources.yaml").write_text(": : : invalid")
        result = load_sources_config(tmp_path)
        # Should not crash -- returns None on error
        assert result is None

    def test_resolve_relative_paths(self, tmp_path):
        paths = _resolve_paths(["docs/", "src/notes/"], tmp_path)
        assert paths[0] == (tmp_path / "docs").resolve()
        assert paths[1] == (tmp_path / "src" / "notes").resolve()

    def test_resolve_absolute_paths(self, tmp_path):
        paths = _resolve_paths(["/tmp/shared-docs"], tmp_path)
        assert paths[0] == Path("/tmp/shared-docs")

    def test_resolve_tilde_expansion(self, tmp_path):
        paths = _resolve_paths(["~/my-docs"], tmp_path)
        assert str(paths[0]).startswith(str(Path.home()))

    def test_docs_ingester_reads_config_paths(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        db_dir = project / ".knowledge-db"
        db_dir.mkdir()

        custom_docs = tmp_path / "custom-docs"
        custom_docs.mkdir()
        (custom_docs / "guide.md").write_text("# Guide\n\nSome content here.")

        config_path = db_dir / "sources.yaml"
        config_path.write_text(f"docs:\n  paths:\n    - {custom_docs}\n")

        ingester = DocsIngester(str(project))
        assert custom_docs.resolve() in [d.resolve() for d in ingester.docs_dirs]

    def test_docs_path_arg_overrides_config(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        db_dir = project / ".knowledge-db"
        db_dir.mkdir()

        # Config says docs/
        config_path = db_dir / "sources.yaml"
        config_path.write_text("docs:\n  paths:\n    - docs/\n")

        # But CLI says override/
        override = tmp_path / "override"
        override.mkdir()

        ingester = DocsIngester(str(project), docs_path=str(override))
        assert ingester.docs_dirs == [override.resolve()]

    def test_include_globs_filter_files(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "readme.md").write_text("# Readme")
        (docs / "notes.txt").write_text("Notes")
        (docs / "data.csv").write_text("a,b,c")

        ingester = DocsIngester.__new__(DocsIngester)
        ingester.docs_dirs = [docs]
        ingester._include_globs = ["*.md"]
        ingester._exclude_globs = []

        files = ingester._find_doc_files()
        assert len(files) == 1
        assert files[0].name == "readme.md"

    def test_exclude_globs_filter_files(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        drafts = docs / "drafts"
        drafts.mkdir()
        (docs / "readme.md").write_text("# Readme")
        (drafts / "wip.md").write_text("# WIP")

        ingester = DocsIngester.__new__(DocsIngester)
        ingester.docs_dirs = [docs]
        ingester._include_globs = None
        ingester._exclude_globs = ["drafts/*"]

        files = ingester._find_doc_files()
        names = [f.name for f in files]
        assert "readme.md" in names
        assert "wip.md" not in names

    def test_fallback_to_convention_when_no_config(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        (project / ".knowledge-db").mkdir()
        (project / "docs").mkdir()
        (project / "docs" / "guide.md").write_text("# Guide")

        ingester = DocsIngester(str(project))
        assert any("docs" in str(d) for d in ingester.docs_dirs)


class TestSessionSourcesConfig:
    """Tests for session ingester reading sources.yaml."""

    def test_session_auto_discover_default(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        (project / ".knowledge-db").mkdir()

        from knowlin_mcp.ingest_sessions import SessionIngester
        SessionIngester(str(project))  # should not raise
        # auto_discover is True by default, sessions_dir may be None
        # if no claude projects dir exists -- that's fine

    def test_session_explicit_path_from_config(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        db_dir = project / ".knowledge-db"
        db_dir.mkdir()

        custom_sessions = tmp_path / "my-sessions"
        custom_sessions.mkdir()

        config = db_dir / "sources.yaml"
        config.write_text(f"sessions:\n  path: {custom_sessions}\n")

        from knowlin_mcp.ingest_sessions import SessionIngester
        ingester = SessionIngester(str(project))
        assert ingester.sessions_dir == custom_sessions.resolve()

    def test_session_arg_overrides_config(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        db_dir = project / ".knowledge-db"
        db_dir.mkdir()

        config = db_dir / "sources.yaml"
        config.write_text("sessions:\n  path: /some/path\n")

        override = tmp_path / "override-sessions"
        override.mkdir()

        from knowlin_mcp.ingest_sessions import SessionIngester
        ingester = SessionIngester(str(project), sessions_dir=str(override))
        assert ingester.sessions_dir == override

    def test_session_auto_discover_false_disables(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        db_dir = project / ".knowledge-db"
        db_dir.mkdir()

        config = db_dir / "sources.yaml"
        config.write_text("sessions:\n  auto_discover: false\n")

        from knowlin_mcp.ingest_sessions import SessionIngester
        ingester = SessionIngester(str(project))
        assert ingester.sessions_dir is None
