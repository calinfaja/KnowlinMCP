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

from knowlin_mcp.utils import debug_log

# BGE-small-en-v1.5 max input is 512 tokens (~2000 chars)
MAX_CHUNK_CHARS = 1600  # ~400 tokens
MIN_CHUNK_CHARS = 50  # Merge tiny chunks with neighbors
OVERLAP_CHARS = 200  # ~50 tokens overlap for sub-splits

SOURCES_CONFIG_FILE = "sources.yaml"
VALID_SOURCE_KEYS = {"docs", "sessions", "codex"}


def _validate_sources_config(config: dict) -> None:
    """Validate sources.yaml structure."""
    if not isinstance(config, dict):
        raise ValueError("sources.yaml must be a YAML mapping")

    unknown = sorted(set(config.keys()) - VALID_SOURCE_KEYS)
    if unknown:
        raise ValueError(f"Unknown keys in sources.yaml: {unknown}")

    for key in config:
        section = config[key]
        if not isinstance(section, dict):
            raise ValueError(f"sources.yaml '{key}' must be a mapping")


def load_sources_config(db_path: Path) -> dict | None:
    """Load .knowledge-db/sources.yaml if it exists.

    Returns parsed dict or None if no config file.
    """
    config_path = db_path / SOURCES_CONFIG_FILE
    if not config_path.exists():
        return None
    try:
        import yaml
    except ImportError:
        debug_log("pyyaml not installed, ignoring sources.yaml")
        return None

    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as e:
        debug_log(f"Failed to read {config_path}: {e}")
        return None
    except OSError as e:
        debug_log(f"Failed to read {config_path}: {e}")
        return None

    _validate_sources_config(config)
    return config


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
            self.docs_dirs = _resolve_paths(docs_config["paths"], self.project_path)
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
        """Save the document processing registry (atomic write)."""
        self.db_path.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._registry, indent=2))
        tmp.rename(self.registry_path)

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
            debug_log("pymupdf4llm not installed. Run: pip install 'knowlin-mcp[pdf]'")
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

        heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

        chunks = []
        heading_stack: dict[int, str] = {}  # level -> heading text
        last_pos = 0
        last_heading = ""

        def _make_doc_chunk(content: str) -> dict[str, Any]:
            # Contextual enrichment: heading hierarchy in context_prefix improves
            # embedding recall (Anthropic contextual retrieval: +20-49%)
            hierarchy = " > ".join(heading_stack[lvl] for lvl in sorted(heading_stack))
            title = last_heading or content[:80].split("\n")[0]
            return self._make_chunk(content, title, hierarchy, source_path)

        for match in heading_pattern.finditer(text):
            content = text[last_pos : match.start()].strip()
            if content and len(content) >= MIN_CHUNK_CHARS:
                chunks.append(_make_doc_chunk(content))

            level = len(match.group(1))
            heading_text = match.group(2).strip()
            heading_stack[level] = heading_text
            for lvl in list(heading_stack.keys()):
                if lvl > level:
                    del heading_stack[lvl]

            last_heading = heading_text
            last_pos = match.end()

        content = text[last_pos:].strip()
        if content and len(content) >= MIN_CHUNK_CHARS:
            chunks.append(_make_doc_chunk(content))

        # Sub-split sections that exceed the embedding window
        final_chunks = []
        for chunk in chunks:
            if len(chunk["insight"]) > MAX_CHUNK_CHARS:
                final_chunks.extend(self._sub_split(chunk))
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _make_chunk(
        self,
        content: str,
        title: str,
        context_prefix: str,
        source_path: str,
        source_prefix: str = "doc",
    ) -> dict[str, Any]:
        """Create an entry dict from a content chunk.

        Content is stored as-is; callers are responsible for size management
        (sub-splitting via _sub_split or hard-truncating large bodies).
        """
        if len(title) > 100:
            title = title[:97] + "..."

        return {
            "title": title,
            "insight": content,
            "context_prefix": context_prefix,
            "type": "document",
            "priority": "medium",
            "keywords": [],
            "source": f"{source_prefix}:{source_path}",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat(),
            "branch": "",
            "related_to": [],
            "_content_hash": self._content_hash(content),
        }

    # Regex patterns for code function/class boundaries.
    # C/C++ pattern requires opening brace on the same line as the signature;
    # keyword guard avoids matching control-flow statements (if/for/while/switch).
    _C_FUNC_RE = re.compile(
        r"^(?!(?:if|for|while|switch|return)\b)[a-zA-Z_]\w*[\s\*]+\w[\w\s\*]*\([^)]*\)\s*\{",
        re.MULTILINE,
    )
    _PY_DEF_RE = re.compile(r"^(def |class )", re.MULTILINE)
    _CODE_EXTENSIONS = {".c", ".h", ".py", ".cpp", ".hpp", ".cc"}

    def _chunk_code_file(self, text: str, source_path: str) -> list[dict[str, Any]]:
        """Split source code into chunks by function/class boundaries."""
        if not text.strip():
            return []

        ext = Path(source_path).suffix.lower()
        pattern = self._PY_DEF_RE if ext == ".py" else self._C_FUNC_RE
        filename = Path(source_path).name

        split_points = [m.start() for m in pattern.finditer(text)]

        context_prefix = f"source:{source_path}"

        def _add_code_chunk(body: str, title: str) -> dict[str, Any]:
            # Hard-truncate: splitting code mid-function is not useful
            return self._make_chunk(
                body[:MAX_CHUNK_CHARS], title, context_prefix, source_path, "code"
            )

        if not split_points:
            # No recognizable function boundaries -- treat whole file as one chunk
            if len(text) >= MIN_CHUNK_CHARS:
                return [_add_code_chunk(text, filename)]
            return []

        chunks: list[dict[str, Any]] = []

        # Preamble: includes, defines, globals before first function
        preamble = text[: split_points[0]].strip()
        if preamble and len(preamble) >= MIN_CHUNK_CHARS:
            chunks.append(_add_code_chunk(preamble, f"{filename} (preamble)"))

        # Each function/class as its own chunk
        for i, start in enumerate(split_points):
            end = split_points[i + 1] if i + 1 < len(split_points) else len(text)
            body = text[start:end].strip()
            if not body or len(body) < MIN_CHUNK_CHARS:
                continue

            first_line = body.split("\n")[0].strip()
            if ext == ".py":
                # "def func_name(..." or "class ClassName(..." -- strip trailing colon
                title = first_line.rstrip(":").strip()
            else:
                # "static int func_name(params) {" -> extract "func_name"
                m = re.match(r".*?(\w+)\s*\(", first_line)
                title = m.group(1) if m else first_line[:80]

            chunks.append(_add_code_chunk(body, title))

        return chunks

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

    def _recursive_split(self, text: str, separators: list[str], max_size: int) -> list[str]:
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
        return [text[i : i + max_size] for i in range(0, len(text), max_size - OVERLAP_CHARS)]

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
            resolved_root = docs_dir.resolve()
            for path in sorted(docs_dir.rglob("*")):
                if path.is_symlink():
                    continue
                if not path.is_file():
                    continue
                try:
                    if not path.resolve().is_relative_to(resolved_root):
                        continue
                except ValueError:
                    continue
                rel = str(path.relative_to(docs_dir))

                # Include filter
                if self._include_globs:
                    if not any(
                        fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(path.name, g)
                        for g in self._include_globs
                    ):
                        continue
                elif extensions and path.suffix.lower() not in extensions:
                    continue

                # Exclude filter
                if any(
                    fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(path.name, g)
                    for g in self._exclude_globs
                ):
                    continue

                files.append(path)

        return files

    def _cleanup_deleted_files(self, current_file_keys: set[str]) -> int:
        """Remove DB entries for files no longer on disk."""
        stale_keys = set(self._registry.keys()) - current_file_keys
        if not stale_keys:
            return 0

        from knowlin_mcp.db import KnowledgeDB

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

        # Filter to changed files, caching hashes
        to_process = []
        file_hashes: dict[str, str] = {}
        for path in doc_files:
            file_key = str(path)
            if not full:
                current_hash = self._file_hash(path)
                file_hashes[file_key] = current_hash
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

        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(self.project_path), sub_store="docs")

        # Collect old IDs to remove AFTER batch_add succeeds (crash-safe)
        old_ids_by_file: dict[str, list[str]] = {}
        for path in to_process:
            file_key = str(path)
            old_ids = self._registry.get(file_key, {}).get("entry_ids", [])
            if old_ids:
                old_ids_by_file[file_key] = old_ids

        # Chunk and collect new entries, tracking per-file boundaries
        all_entries = []
        file_entry_counts: list[tuple[str, int]] = []  # (file_key, count)
        file_chunk_hashes: dict[str, list[str]] = {}
        file_chunk_counts: dict[str, int] = {}
        file_new_chunk_counts: dict[str, int] = {}

        for path in to_process:
            file_key = str(path)
            content = self._read_file(path)
            if not content:
                # Register empty/unreadable files so skip-check works next run
                self._registry[file_key] = {
                    "file_hash": file_hashes.get(file_key) or self._file_hash(path),
                    "processed": datetime.now().isoformat(),
                    "chunk_hashes": [],
                    "chunk_count": 0,
                    "new_chunks": 0,
                    "entry_ids": [],
                }
                continue

            # Find which docs_dir this path belongs to
            source_path = path.name
            for docs_dir in self.docs_dirs:
                try:
                    source_path = str(path.relative_to(docs_dir))
                    break
                except ValueError:
                    continue

            ext = path.suffix.lower()
            if ext in self._CODE_EXTENSIONS:
                chunks = self._chunk_code_file(content, source_path)
            else:
                chunks = self._chunk_by_headings(content, source_path)

            old_hashes = set(self._registry.get(file_key, {}).get("chunk_hashes", []))
            current_hashes = [chunk.get("_content_hash", "") for chunk in chunks]
            new_chunk_count = sum(
                1 for chunk_hash in current_hashes if chunk_hash not in old_hashes
            )

            if old_hashes:
                unchanged_chunk_count = len(chunks) - new_chunk_count
                debug_log(
                    f"{Path(file_key).name}: {new_chunk_count} changed/new chunks, "
                    f"{unchanged_chunk_count} unchanged chunks re-added"
                )

            # Clean internal fields before adding
            for chunk in chunks:
                chunk.pop("_content_hash", None)

            file_entry_counts.append((file_key, len(chunks)))
            file_chunk_hashes[file_key] = current_hashes
            file_chunk_counts[file_key] = len(chunks)
            file_new_chunk_counts[file_key] = new_chunk_count
            all_entries.extend(chunks)

        if not all_entries:
            # Still update registry for empty/unchanged files
            self._save_registry()
            return 0

        # Batch add new entries
        ids = db.batch_add(all_entries, check_duplicates=False)
        accepted_count = sum(1 for entry_id in ids if entry_id is not None)

        # Only NOW remove old entries (after batch_add succeeded)
        for file_key, old_ids in old_ids_by_file.items():
            db.remove_entries(old_ids)
            debug_log(f"Removed {len(old_ids)} old entries for {Path(file_key).name}")

        # Distribute IDs back to registry per file
        offset = 0
        for file_key, count in file_entry_counts:
            raw_file_ids = ids[offset : offset + count] if count > 0 else []
            file_ids = [entry_id for entry_id in raw_file_ids if entry_id is not None]
            self._registry[file_key] = {
                "file_hash": file_hashes.get(file_key) or self._file_hash(Path(file_key)),
                "processed": datetime.now().isoformat(),
                "chunk_hashes": file_chunk_hashes.get(file_key, []),
                "chunk_count": file_chunk_counts.get(file_key, 0),
                "new_chunks": file_new_chunk_counts.get(file_key, 0),
                "entry_ids": file_ids,
            }
            offset += count

        self._save_registry()
        debug_log(f"Ingested {accepted_count} entries from {len(to_process)} documents")
        return accepted_count
