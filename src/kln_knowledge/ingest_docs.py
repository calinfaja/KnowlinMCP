"""Document Ingester - Ingest markdown and PDF files into knowledge DB.

Pipeline:
1. File-level SHA-256 change detection (skip unchanged)
2. PDF -> markdown conversion (via pymupdf4llm, optional)
3. Heading-based chunking (h1/h2/h3 sections)
4. Chunk-level hash diff against registry
5. Batch embed new/changed chunks into docs sub-store

Registry tracks processed files and their chunk hashes.
Sources config (.knowledge-db/sources.yaml) declares the corpus.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from kln_knowledge.utils import debug_log

# BGE-small-en-v1.5 max input is 512 tokens (~2000 chars)
MAX_CHUNK_CHARS = 1600   # ~400 tokens
MIN_CHUNK_CHARS = 50     # Merge tiny chunks with neighbors
OVERLAP_CHARS = 200      # ~50 tokens overlap for sub-splits

SOURCES_CONFIG_FILE = "sources.yaml"


def load_sources_config(db_path: Path) -> dict | None:
    """Load .knowledge-db/sources.yaml if it exists.

    Returns parsed dict or None if no config file.
    """
    config_path = db_path / SOURCES_CONFIG_FILE
    if not config_path.exists():
        return None
    try:
        import yaml
        return yaml.safe_load(config_path.read_text()) or {}
    except ImportError:
        debug_log("pyyaml not installed, ignoring sources.yaml")
        return None
    except Exception as e:
        debug_log(f"Failed to read {config_path}: {e}")
        return None


def _resolve_paths(paths: list[str], project_root: Path) -> list[Path]:
    """Resolve a list of path strings to absolute Path objects.

    Supports: relative (to project root), absolute, ~ expansion.
    """
    resolved = []
    for p in paths:
        expanded = Path(p).expanduser()
        if not expanded.is_absolute():
            expanded = (project_root / expanded).resolve()
        else:
            expanded = expanded.resolve()
        resolved.append(expanded)
    return resolved


class DocsIngester:
    """Ingest markdown and PDF documents into knowledge DB."""

    def __init__(
        self,
        project_path: str,
        docs_path: str | None = None,
    ):
        """Initialize docs ingester.

        Args:
            project_path: Project root directory
            docs_path: Path to docs directory. If None, reads from
                       sources.yaml or scans for common doc dirs.
        """
        self.project_path = Path(project_path).resolve()
        self.db_path = self.project_path / ".knowledge-db"
        self.registry_path = self.db_path / "doc-registry.json"

        # Load sources config (may be None)
        self._sources_config = load_sources_config(self.db_path)
        docs_config = (self._sources_config or {}).get("docs", {})

        if docs_path:
            # CLI --path flag always wins
            self.docs_dirs = [Path(docs_path).resolve()]
        elif docs_config.get("paths"):
            self.docs_dirs = _resolve_paths(
                docs_config["paths"], self.project_path
            )
        else:
            self.docs_dirs = self._find_docs_dirs()

        # Include/exclude globs from config
        self._include_globs = docs_config.get("include")
        self._exclude_globs = docs_config.get("exclude", [])

        self._registry: dict[str, dict] = self._load_registry()

    def _find_docs_dirs(self) -> list[Path]:
        """Find documentation directories in the project (convention-based)."""
        candidates = ["docs", "doc", "INFOS", "documentation"]
        dirs = []
        for name in candidates:
            d = self.project_path / name
            if d.is_dir():
                dirs.append(d)
        return dirs

    def _load_registry(self) -> dict[str, dict]:
        """Load the document processing registry."""
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_registry(self) -> None:
        """Save the document processing registry."""
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(self._registry, indent=2))

    def _file_hash(self, path: Path) -> str:
        """SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _content_hash(self, text: str) -> str:
        """SHA-256 hash of normalized text content."""
        normalized = text.strip().replace("\r\n", "\n")
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _read_file(self, path: Path) -> str:
        """Read a file as markdown. Converts PDF if needed."""
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._pdf_to_markdown(path)
        elif suffix in (".md", ".markdown", ".txt", ".rst"):
            return path.read_text(encoding="utf-8", errors="replace")
        else:
            return ""

    def _pdf_to_markdown(self, path: Path) -> str:
        """Convert PDF to markdown using pymupdf4llm."""
        try:
            import pymupdf4llm
            return pymupdf4llm.to_markdown(str(path))
        except ImportError:
            debug_log("pymupdf4llm not installed. Run: pip install 'kln-knowledge-system[pdf]'")
            return ""
        except Exception as e:
            debug_log(f"PDF conversion failed for {path}: {e}")
            return ""

    def _chunk_by_headings(self, text: str, source_path: str) -> list[dict[str, Any]]:
        """Split markdown text into chunks by heading structure.

        Splits at h1/h2/h3 boundaries, preserving heading hierarchy
        as context metadata. Sub-splits oversized sections.
        """
        if not text.strip():
            return []

        # Split by heading lines
        heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

        chunks = []
        heading_stack = {}  # level -> heading text
        last_pos = 0
        last_heading = ""

        for match in heading_pattern.finditer(text):
            # Save content before this heading
            content = text[last_pos:match.start()].strip()
            if content and len(content) >= MIN_CHUNK_CHARS:
                chunks.append(self._make_chunk(
                    content, last_heading, heading_stack, source_path
                ))

            # Update heading stack
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            heading_stack[level] = heading_text
            # Clear deeper levels
            for l in list(heading_stack.keys()):
                if l > level:
                    del heading_stack[l]

            last_heading = heading_text
            last_pos = match.end()

        # Don't forget the last section
        content = text[last_pos:].strip()
        if content and len(content) >= MIN_CHUNK_CHARS:
            chunks.append(self._make_chunk(
                content, last_heading, heading_stack, source_path
            ))

        # Sub-split oversized chunks
        final_chunks = []
        for chunk in chunks:
            if len(chunk["insight"]) > MAX_CHUNK_CHARS:
                sub_chunks = self._sub_split(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _make_chunk(
        self,
        content: str,
        heading: str,
        heading_stack: dict[int, str],
        source_path: str,
    ) -> dict[str, Any]:
        """Create an entry dict from a chunk of text."""
        # Build heading hierarchy as context
        hierarchy = " > ".join(
            heading_stack[level]
            for level in sorted(heading_stack.keys())
        )

        title = heading or content[:80].split("\n")[0]
        if len(title) > 100:
            title = title[:97] + "..."

        return {
            "title": title,
            "insight": content[:MAX_CHUNK_CHARS],
            "type": "document",
            "priority": "medium",
            "keywords": [],
            "source": f"doc:{source_path}",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat(),
            "branch": "",
            "related_to": [],
            "_heading_hierarchy": hierarchy,
            "_content_hash": self._content_hash(content),
        }

    def _sub_split(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        """Split an oversized chunk into smaller pieces with overlap."""
        text = chunk["insight"]
        sub_chunks = []

        # Split on paragraph boundaries first, then sentences
        separators = ["\n\n", "\n", ". ", " "]

        pieces = self._recursive_split(text, separators, MAX_CHUNK_CHARS)

        for i, piece in enumerate(pieces):
            sub = chunk.copy()
            sub["insight"] = piece
            sub["title"] = f"{chunk['title']} (part {i + 1})"
            sub["_content_hash"] = self._content_hash(piece)
            sub_chunks.append(sub)

        return sub_chunks

    def _recursive_split(
        self, text: str, separators: list[str], max_size: int
    ) -> list[str]:
        """Recursively split text using separator hierarchy."""
        if len(text) <= max_size:
            return [text]

        for sep in separators:
            parts = text.split(sep)
            if len(parts) <= 1:
                continue

            result = []
            current = ""
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) > max_size and current:
                    result.append(current.strip())
                    # Add overlap from end of previous chunk
                    overlap = current[-OVERLAP_CHARS:] if len(current) > OVERLAP_CHARS else ""
                    current = overlap + part
                else:
                    current = candidate

            if current.strip():
                result.append(current.strip())

            if len(result) > 1:
                return result

        # Last resort: hard split
        return [text[i:i + max_size] for i in range(0, len(text), max_size - OVERLAP_CHARS)]

    def _find_doc_files(self) -> list[Path]:
        """Find document files using include/exclude globs from config."""
        files = []

        if self._include_globs:
            # Config specifies include patterns -- use those as the extension filter
            extensions = None  # globs handle filtering
        else:
            extensions = {".md", ".markdown", ".txt", ".pdf", ".rst"}

        for docs_dir in self.docs_dirs:
            if not docs_dir.exists():
                continue
            for path in sorted(docs_dir.rglob("*")):
                if not path.is_file():
                    continue
                rel = str(path.relative_to(docs_dir))

                # Include filter
                if self._include_globs:
                    if not any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(path.name, g)
                               for g in self._include_globs):
                        continue
                elif extensions and path.suffix.lower() not in extensions:
                    continue

                # Exclude filter
                if any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(path.name, g)
                       for g in self._exclude_globs):
                    continue

                files.append(path)

        return files

    def _cleanup_deleted_files(self, current_file_keys: set[str]) -> int:
        """Remove DB entries for files no longer on disk."""
        stale_keys = set(self._registry.keys()) - current_file_keys
        if not stale_keys:
            return 0

        from kln_knowledge.db import KnowledgeDB

        db = KnowledgeDB(str(self.project_path), sub_store="docs")
        removed = 0

        for key in stale_keys:
            entry_ids = self._registry[key].get("entry_ids", [])
            if entry_ids:
                db.remove_entries(entry_ids)
                removed += len(entry_ids)
                debug_log(f"Removed {len(entry_ids)} stale entries from {key}")
            del self._registry[key]

        if removed:
            self._save_registry()
        return removed

    def ingest(self, full: bool = False) -> int:
        """Ingest documents into the docs sub-store.

        Handles the full sync lifecycle:
        - Detects deleted files and removes their entries
        - Detects modified files and replaces old entries
        - Skips unchanged files (by file hash + chunk hash)

        Args:
            full: If True, re-process all docs regardless of registry.

        Returns:
            Number of entries ingested.
        """
        doc_files = self._find_doc_files()

        # Cleanup entries for deleted files
        current_file_keys = {str(p) for p in doc_files}
        self._cleanup_deleted_files(current_file_keys)

        if not doc_files:
            debug_log("No document files found")
            return 0

        # Filter to changed files
        to_process = []
        for path in doc_files:
            file_key = str(path)
            if not full:
                current_hash = self._file_hash(path)
                if (
                    file_key in self._registry
                    and self._registry[file_key].get("file_hash") == current_hash
                ):
                    continue
            to_process.append(path)

        if not to_process:
            debug_log("All documents already processed")
            return 0

        debug_log(f"Processing {len(to_process)} document files...")

        from kln_knowledge.db import KnowledgeDB

        db = KnowledgeDB(str(self.project_path), sub_store="docs")

        # Remove old entries for files being re-processed
        for path in to_process:
            file_key = str(path)
            old_ids = self._registry.get(file_key, {}).get("entry_ids", [])
            if old_ids:
                db.remove_entries(old_ids)
                debug_log(f"Removed {len(old_ids)} old entries for {path.name}")

        # Chunk and collect new entries, tracking per-file boundaries
        all_entries = []
        file_entry_counts: list[tuple[str, int]] = []  # (file_key, count)

        for path in to_process:
            content = self._read_file(path)
            if not content:
                file_entry_counts.append((str(path), 0))
                continue

            source_path = str(path.relative_to(self.docs_dirs[0]) if self.docs_dirs else path.name)
            chunks = self._chunk_by_headings(content, source_path)

            # Diff against existing chunks if file was previously processed
            file_key = str(path)
            old_hashes = set(
                self._registry.get(file_key, {}).get("chunk_hashes", [])
            )
            new_chunks = []
            new_hashes = []

            for chunk in chunks:
                chunk_hash = chunk.get("_content_hash", "")
                new_hashes.append(chunk_hash)
                if chunk_hash not in old_hashes:
                    new_chunks.append(chunk)

            # Clean internal fields before adding
            for chunk in new_chunks:
                chunk.pop("_content_hash", None)
                chunk.pop("_heading_hierarchy", None)

            file_entry_counts.append((file_key, len(new_chunks)))
            all_entries.extend(new_chunks)

            # Pre-populate registry (entry_ids filled after batch_add)
            self._registry[file_key] = {
                "file_hash": self._file_hash(path),
                "processed": datetime.now().isoformat(),
                "chunk_hashes": new_hashes,
                "chunk_count": len(chunks),
                "new_chunks": len(new_chunks),
                "entry_ids": [],  # filled below
            }

        if not all_entries:
            self._save_registry()
            return 0

        # Batch add and distribute IDs back to registry
        ids = db.batch_add(all_entries, check_duplicates=False)

        offset = 0
        for file_key, count in file_entry_counts:
            if count > 0:
                self._registry[file_key]["entry_ids"] = ids[offset:offset + count]
                offset += count

        self._save_registry()
        debug_log(f"Ingested {len(ids)} entries from {len(to_process)} documents")
        return len(ids)
