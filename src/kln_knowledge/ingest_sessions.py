"""Session Ingester - Extract knowledge from Claude Code JSONL transcripts.

Two-phase pipeline:
1. Parse JSONL files, extract assistant messages with high-value content
2. Score by importance, batch-embed into sessions sub-store

Registry tracks which files have been processed (SHA-256 hash).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from kln_knowledge.utils import debug_log

# Minimum content length worth indexing (skip trivial responses)
MIN_CONTENT_LENGTH = 100

# Signals that indicate high-value content in assistant messages
_VALUE_SIGNALS = {
    "decision": [
        "decided to", "chose", "went with", "instead of", "trade-off",
        "the reason", "because", "approach is",
    ],
    "solution": [
        "the fix", "fixed by", "resolved", "workaround", "solution is",
        "to fix this", "the issue was", "root cause",
    ],
    "warning": [
        "be careful", "watch out", "gotcha", "caveat", "don't forget",
        "important:", "note:", "warning:",
    ],
    "pattern": [
        "best practice", "pattern", "convention", "recommended",
        "the approach", "typically", "standard way",
    ],
    "discovery": [
        "found that", "turns out", "discovered", "realized",
        "interesting", "unexpected", "TIL",
    ],
}

# Content to skip (tool output, status messages, etc.)
_SKIP_PATTERNS = [
    r"^```\n[\s\S]{0,50}\n```$",  # Very short code blocks
    r"^I'll ",  # "I'll do X" filler
    r"^Let me ",  # "Let me check" filler
    r"^Sure,? ",  # Acknowledgments
    r"^OK,? ",
    r"^Done\.?\s*$",
    r"^Here's the ",  # Generic intros
]


class SessionIngester:
    """Ingest Claude Code JSONL session transcripts into knowledge DB."""

    def __init__(self, project_path: str, sessions_dir: str | None = None):
        """Initialize session ingester.

        Args:
            project_path: Project root directory
            sessions_dir: Custom sessions directory. If None, uses
                          ~/.claude/projects/ matching the project.
        """
        self.project_path = Path(project_path).resolve()
        self.db_path = self.project_path / ".knowledge-db"
        self.registry_path = self.db_path / "session-registry.json"

        if sessions_dir:
            self.sessions_dir = Path(sessions_dir)
        else:
            self.sessions_dir = self._find_sessions_dir()

        self._registry: dict[str, dict] = self._load_registry()

    def _find_sessions_dir(self) -> Path | None:
        """Find the Claude Code sessions directory for this project."""
        claude_dir = Path.home() / ".claude" / "projects"
        if not claude_dir.exists():
            return None

        # Claude Code stores projects by path hash
        # Look for directories that contain JSONL files
        project_str = str(self.project_path)

        for d in claude_dir.iterdir():
            if not d.is_dir():
                continue
            # Check if this directory has JSONL files
            jsonl_files = list(d.glob("*.jsonl"))
            if jsonl_files:
                # Heuristic: check if the directory name matches our project
                # Claude Code uses a mangled path as the directory name
                dir_name = d.name
                # Convert /home/user/Project/foo to -home-user-Project-foo
                mangled = project_str.replace("/", "-").lstrip("-")
                if mangled in dir_name or dir_name in mangled:
                    return d

        return None

    def _load_registry(self) -> dict[str, dict]:
        """Load the session processing registry."""
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_registry(self) -> None:
        """Save the session processing registry."""
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(self._registry, indent=2))

    def _file_hash(self, path: Path) -> str:
        """SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _score_content(self, text: str) -> tuple[float, str]:
        """Score content by importance and infer type.

        Returns (score, type) where score is 0-1.
        """
        text_lower = text.lower()

        # Check for skip patterns
        for pattern in _SKIP_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                return (0.0, "")

        best_score = 0.0
        best_type = "finding"

        for entry_type, signals in _VALUE_SIGNALS.items():
            matches = sum(1 for s in signals if s in text_lower)
            if matches > 0:
                # More signal matches = higher score
                score = min(1.0, 0.3 + matches * 0.2)
                if score > best_score:
                    best_score = score
                    best_type = entry_type

        # Length bonus (longer = more substantive, up to a point)
        length_bonus = min(0.2, len(text) / 2000)
        best_score += length_bonus

        # Code block bonus (concrete examples are valuable)
        if "```" in text:
            best_score += 0.1

        return (min(1.0, best_score), best_type)

    def _extract_from_jsonl(self, path: Path) -> list[dict[str, Any]]:
        """Extract high-value entries from a JSONL transcript.

        Phase 1: Parse messages, filter to assistant content.
        Phase 2: Score and filter by importance threshold.
        """
        entries = []

        try:
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Only process assistant messages
                    if msg.get("role") != "assistant":
                        continue

                    # Extract text content
                    content = ""
                    if isinstance(msg.get("content"), str):
                        content = msg["content"]
                    elif isinstance(msg.get("content"), list):
                        # Content blocks (text, tool_use, etc.)
                        for block in msg["content"]:
                            if isinstance(block, dict) and block.get("type") == "text":
                                content += block.get("text", "") + "\n"

                    content = content.strip()
                    if len(content) < MIN_CONTENT_LENGTH:
                        continue

                    # Score the content
                    score, entry_type = self._score_content(content)
                    if score < 0.3:
                        continue

                    # Extract title from first meaningful line
                    lines = content.split("\n")
                    title = ""
                    for line_text in lines:
                        clean = line_text.strip().lstrip("#").strip()
                        if len(clean) > 10 and not clean.startswith("```"):
                            title = clean[:100]
                            break
                    if not title:
                        title = content[:100]

                    # Get date from file name or modification time
                    date_str = self._extract_date(path)

                    entries.append({
                        "title": title,
                        "insight": content[:500],  # Cap at 500 chars
                        "type": entry_type,
                        "priority": "high" if score > 0.7 else "medium",
                        "keywords": [],
                        "source": f"session:{path.name}",
                        "date": date_str,
                        "timestamp": datetime.now().isoformat(),
                        "branch": "",
                        "related_to": [],
                        "_importance_score": score,
                    })

        except Exception as e:
            debug_log(f"Failed to parse {path}: {e}")

        return entries

    def _extract_date(self, path: Path) -> str:
        """Extract date from session file path or mtime."""
        # Try to extract from filename (e.g., 2026-03-04_session.jsonl)
        name = path.stem
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", name)
        if date_match:
            return date_match.group(1)

        # Fall back to file modification time
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

    def ingest(self, full: bool = False) -> int:
        """Ingest session transcripts into the sessions sub-store.

        Args:
            full: If True, re-process all sessions regardless of registry.

        Returns:
            Number of entries ingested.
        """
        if self.sessions_dir is None or not self.sessions_dir.exists():
            debug_log("No sessions directory found")
            return 0

        # Find JSONL files
        jsonl_files = sorted(self.sessions_dir.glob("*.jsonl"))
        if not jsonl_files:
            debug_log("No JSONL files found")
            return 0

        # Filter to unprocessed files
        to_process = []
        for path in jsonl_files:
            file_key = str(path.name)
            if not full:
                current_hash = self._file_hash(path)
                if (
                    file_key in self._registry
                    and self._registry[file_key].get("hash") == current_hash
                ):
                    continue
            to_process.append(path)

        if not to_process:
            debug_log("All sessions already processed")
            return 0

        debug_log(f"Processing {len(to_process)} session files...")

        # Extract entries from all files
        all_entries = []
        for path in to_process:
            entries = self._extract_from_jsonl(path)
            all_entries.extend(entries)

            # Update registry
            self._registry[path.name] = {
                "hash": self._file_hash(path),
                "processed": datetime.now().isoformat(),
                "entries_extracted": len(entries),
            }

        if not all_entries:
            self._save_registry()
            return 0

        # Sort by importance and take top entries
        all_entries.sort(key=lambda x: x.get("_importance_score", 0), reverse=True)

        # Remove internal scoring field
        for e in all_entries:
            e.pop("_importance_score", None)

        # Batch add to sessions sub-store
        from kln_knowledge.db import KnowledgeDB

        db = KnowledgeDB(str(self.project_path), sub_store="sessions")
        ids = db.batch_add(all_entries, check_duplicates=False)

        self._save_registry()
        debug_log(f"Ingested {len(ids)} entries from {len(to_process)} sessions")
        return len(ids)
