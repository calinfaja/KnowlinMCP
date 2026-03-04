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
            sessions_dir: Custom sessions directory. If None, reads from
                          sources.yaml or auto-discovers from ~/.claude/projects/.
        """
        self.project_path = Path(project_path).resolve()
        self.db_path = self.project_path / ".knowledge-db"
        self.registry_path = self.db_path / "session-registry.json"

        # Load sources config
        from kln_knowledge.ingest_docs import load_sources_config, _resolve_paths
        sources_config = load_sources_config(self.db_path)
        sessions_config = (sources_config or {}).get("sessions", {})

        if sessions_dir:
            # Explicit arg always wins
            self.sessions_dir = Path(sessions_dir)
        elif sessions_config.get("path"):
            paths = _resolve_paths([sessions_config["path"]], self.project_path)
            self.sessions_dir = paths[0] if paths else None
        elif sessions_config.get("auto_discover", True):
            self.sessions_dir = self._find_sessions_dir()
        else:
            self.sessions_dir = None

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

    def _extract_text(self, msg: dict) -> str:
        """Extract text content from a message."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts).strip()
        return ""

    def _extract_from_jsonl(self, path: Path) -> list[dict[str, Any]]:
        """Extract high-value entries from a JSONL transcript.

        Pairs user+assistant messages into turns.
        Scores each turn by importance and filters low-value content.
        """
        entries = []
        last_user_text = ""

        try:
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    role = msg.get("role")

                    # Track last user message for pairing
                    if role == "user":
                        text = self._extract_text(msg)
                        if text and len(text) > 10:
                            last_user_text = text
                        continue

                    if role != "assistant":
                        continue

                    # Extract assistant text
                    assistant_text = self._extract_text(msg)
                    if len(assistant_text) < MIN_CONTENT_LENGTH:
                        continue

                    # Score the assistant content
                    score, entry_type = self._score_content(assistant_text)
                    if score < 0.3:
                        continue

                    # Build title from user question or first meaningful line
                    title = ""
                    if last_user_text:
                        title = last_user_text[:100]
                    if not title:
                        for line_text in assistant_text.split("\n"):
                            clean = line_text.strip().lstrip("#").strip()
                            if len(clean) > 10 and not clean.startswith("```"):
                                title = clean[:100]
                                break
                    if not title:
                        title = assistant_text[:100]

                    # Combine user question + assistant response for searchability
                    insight_parts = []
                    if last_user_text:
                        insight_parts.append(last_user_text[:300])
                        insight_parts.append("---")
                    insight_parts.append(assistant_text[:2000])
                    insight = "\n".join(insight_parts)[:2500]

                    date_str = self._extract_date(path)

                    entries.append({
                        "title": title,
                        "insight": insight,
                        "type": "session",
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

    def _cleanup_deleted_files(self, current_file_keys: set[str]) -> int:
        """Remove DB entries for session files no longer on disk."""
        stale_keys = set(self._registry.keys()) - current_file_keys
        if not stale_keys:
            return 0

        from kln_knowledge.db import KnowledgeDB

        db = KnowledgeDB(str(self.project_path), sub_store="sessions")
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
        """Ingest session transcripts into the sessions sub-store.

        Handles the full sync lifecycle:
        - Detects deleted session files and removes their entries
        - Detects modified sessions and replaces old entries
        - Skips unchanged sessions (by file hash)

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

        # Cleanup entries for deleted files
        current_file_keys = {p.name for p in jsonl_files}
        self._cleanup_deleted_files(current_file_keys)

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

        from kln_knowledge.db import KnowledgeDB

        db = KnowledgeDB(str(self.project_path), sub_store="sessions")

        # Remove old entries for files being re-processed
        for path in to_process:
            old_ids = self._registry.get(path.name, {}).get("entry_ids", [])
            if old_ids:
                db.remove_entries(old_ids)
                debug_log(f"Removed {len(old_ids)} old entries for {path.name}")

        # Extract entries from all files, track per-file counts
        all_entries = []
        file_entry_counts: list[tuple[str, int]] = []

        for path in to_process:
            entries = self._extract_from_jsonl(path)
            file_entry_counts.append((path.name, len(entries)))
            all_entries.extend(entries)

        if not all_entries:
            # Update registry even if no entries extracted (marks files as processed)
            for path in to_process:
                self._registry[path.name] = {
                    "hash": self._file_hash(path),
                    "processed": datetime.now().isoformat(),
                    "entries_extracted": 0,
                    "entry_ids": [],
                }
            self._save_registry()
            return 0

        # Sort by importance and take top entries
        all_entries.sort(key=lambda x: x.get("_importance_score", 0), reverse=True)

        # Remove internal scoring field
        for e in all_entries:
            e.pop("_importance_score", None)

        # Batch add to sessions sub-store
        ids = db.batch_add(all_entries, check_duplicates=False)

        # Distribute IDs back to registry per file
        offset = 0
        for file_key, count in file_entry_counts:
            self._registry[file_key] = {
                "hash": self._file_hash(self.sessions_dir / file_key),
                "processed": datetime.now().isoformat(),
                "entries_extracted": count,
                "entry_ids": ids[offset:offset + count] if count > 0 else [],
            }
            offset += count

        self._save_registry()
        debug_log(f"Ingested {len(ids)} entries from {len(to_process)} sessions")
        return len(ids)
