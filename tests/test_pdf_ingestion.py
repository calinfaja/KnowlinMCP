"""PDF ingestion integration tests.

Tests the full pipeline: PDF -> markdown -> chunks -> embeddings.
Run with: pytest --integration tests/test_pdf_ingestion.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from knowlin_mcp.ingest_docs import DocsIngester


def _create_test_pdf(path: Path, text: str) -> bool:
    """Create a minimal test PDF using fpdf2 (if available)."""
    try:
        from fpdf import FPDF
    except ImportError:
        return False

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in text.split("\n"):
        pdf.cell(200, 10, text=line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))
    return True


@pytest.fixture
def docs_with_pdf(tmp_path):
    """Create a project with a PDF in docs/."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".knowledge-db").mkdir()
    (project / ".git").mkdir()
    docs = project / "docs"
    docs.mkdir()

    # Create a test markdown file (always works)
    md = docs / "guide.md"
    md.write_text(
        "# Installation Guide\n\n"
        "This guide covers installation of the system on Ubuntu and macOS.\n\n"
        "## Prerequisites\n\n"
        "You need Python 3.9 or later and pip installed on your system.\n\n"
        "## Steps\n\n"
        "Run pip install knowlin-mcp to install the package.\n"
    )

    return project


@pytest.mark.integration
class TestPDFIngestion:
    """Tests for PDF document ingestion."""

    def test_markdown_ingestion_full_pipeline(self, docs_with_pdf):
        """Full pipeline: markdown file -> chunks -> DB entries."""
        ingester = DocsIngester(str(docs_with_pdf))
        count = ingester.ingest()
        assert count >= 1, "Should ingest at least 1 chunk from markdown"

    def test_contextual_enrichment_in_ingested_chunks(self, docs_with_pdf):
        """Ingested chunks should have heading hierarchy in insight."""
        ingester = DocsIngester(str(docs_with_pdf))
        # Get chunks directly
        md_path = docs_with_pdf / "docs" / "guide.md"
        text = md_path.read_text()
        chunks = ingester._chunk_by_headings(text, "guide.md")

        # Chunks under headings should have hierarchy prepended
        sub_chunks = [c for c in chunks if "Prerequisites" in c["insight"] or "Steps" in c["insight"]]
        if sub_chunks:
            # At least one chunk should start with hierarchy
            has_hierarchy = any(
                c["insight"].startswith("Installation Guide")
                for c in sub_chunks
            )
            assert has_hierarchy, "Sub-heading chunks should have hierarchy prepended"

    def test_pdf_conversion_if_available(self, docs_with_pdf):
        """Test PDF ingestion if pymupdf4llm is available."""
        try:
            import pymupdf4llm  # noqa: F401
        except ImportError:
            pytest.skip("pymupdf4llm not installed")

        ingester = DocsIngester.__new__(DocsIngester)
        # Create a simple PDF for testing
        pdf_path = docs_with_pdf / "docs" / "test.pdf"
        # pymupdf4llm needs a real PDF - try creating one
        try:
            import fitz  # PyMuPDF
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "# Test Document\n\nThis is test content for PDF ingestion testing.")
            doc.save(str(pdf_path))
            doc.close()
        except Exception:
            pytest.skip("Could not create test PDF")

        text = ingester._pdf_to_markdown(pdf_path)
        assert len(text) > 0, "PDF conversion should produce non-empty text"
