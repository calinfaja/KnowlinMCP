"""Session Ingester - Extract knowledge from Claude Code JSONL transcripts.

Claude Code JSONL format (per-line records):
- Top-level `type` field: "user", "assistant", "progress", "system",
  "file-history-snapshot", "queue-operation"
- Actual message content at record["message"]["content"]
- Content is a list of typed blocks: text, tool_use, tool_result, thinking

Two-phase pipeline:
1. Parse JSONL files, extract assistant text blocks with high-value content
2. Score by importance, batch-embed into sessions sub-store

Registry tracks which files have been processed (SHA-256 hash).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from knowlin_mcp.utils import debug_log

# Minimum content length worth indexing (skip trivial responses)
MIN_CONTENT_LENGTH = 150

# Substantive content length -- always accept if above this threshold
SUBSTANTIVE_LENGTH = 300

# Signals that indicate high-value content in assistant messages
_VALUE_SIGNALS = {
    "decision": [
        "decided to",
        "chose",
        "went with",
        "instead of",
        "trade-off",
        "the reason",
        "because",
        "approach is",
    ],
    "solution": [
        "the fix",
        "fixed by",
        "resolved",
        "workaround",
        "solution is",
        "to fix this",
        "the issue was",
        "root cause",
    ],
    "warning": [
        "be careful",
        "watch out",
        "gotcha",
        "caveat",
        "don't forget",
        "important:",
        "note:",
        "warning:",
    ],
    "pattern": [
        "best practice",
        "pattern",
        "convention",
        "recommended",
        "the approach",
        "typically",
        "standard way",
    ],
    "discovery": [
        "found that",
        "turns out",
        "discovered",
        "realized",
        "interesting",
        "unexpected",
        "TIL",
    ],
}

# Content to skip (only applied to short messages < SUBSTANTIVE_LENGTH)
_SKIP_PATTERNS = [
    r"^```\n[\s\S]{0,50}\n```$",  # Very short code blocks
    r"^I'll ",  # "I'll do X" filler
    r"^Let me ",  # "Let me check" filler
    r"^Sure,? ",  # Acknowledgments
    r"^OK,? ",
    r"^Done\.?\s*$",
    r"^Here's the ",  # Generic intros
]

# Noise markers in user messages that indicate non-human content
_USER_NOISE_MARKERS = [
    "<command-name>",
    "<local-command",
    "<system-reminder>",
    "[Request interrupted",
    "This session is being continued",
    "## Triggers",  # Slash command definitions injected as user text
    "## When to",
]


def score_content(text: str) -> tuple[float, str]:
    """Score content by importance and infer type.

    Returns (score, type) where score is 0-1.
    Substantive messages (>= SUBSTANTIVE_LENGTH chars) always pass.
    """
    # Skip patterns only apply to short messages
    if len(text) < SUBSTANTIVE_LENGTH:
        for pattern in _SKIP_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                return (0.0, "")

    text_lower = text.lower()
    best_score = 0.0
    best_type = "finding"

    for entry_type, signals in _VALUE_SIGNALS.items():
        matches = sum(1 for s in signals if s in text_lower)
        if matches > 0:
            score = min(1.0, 0.3 + matches * 0.2)
            if score > best_score:
                best_score = score
                best_type = entry_type

    # Length bonus (longer = more substantive)
    length_bonus = min(0.3, len(text) / 1500)
    best_score += length_bonus

    # Code block bonus
    if "```" in text:
        best_score += 0.1

    # Markdown structure bonus (headers, tables, lists indicate organized content)
    if re.search(r"^#{1,3}\s", text, re.MULTILINE):
        best_score += 0.1
    if "|" in text and "---" in text:
        best_score += 0.05

    return (min(1.0, best_score), best_type)


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
        from knowlin_mcp.ingest_docs import _resolve_paths, load_sources_config

        sources_config = load_sources_config(self.db_path)
        sessions_config = (sources_config or {}).get("sessions", {})

        self.sessions_dir: Path | None = None
        if sessions_dir:
            # Explicit arg always wins
            self.sessions_dir = Path(sessions_dir)
        elif sessions_config.get("path"):
            paths = _resolve_paths([sessions_config["path"]], self.project_path)
            self.sessions_dir = paths[0] if paths else None
        elif sessions_config.get("auto_discover", True):
            self.sessions_dir = self._find_sessions_dir()

        self._registry: dict[str, dict] = self._load_registry()

    def _find_sessions_dir(self) -> Path | None:
        """Find the Claude Code sessions directory for this project.

        Claude Code stores sessions at ~/.claude/projects/<mangled-path>/
        where the directory name is the project path with / replaced by -.
        """
        claude_dir = Path.home() / ".claude" / "projects"
        if not claude_dir.exists():
            return None

        project_str = str(self.project_path)
        # Claude Code mangles: /home/user/Project/foo -> -home-user-Project-foo
        expected = project_str.replace("/", "-")

        for d in claude_dir.iterdir():
            if not d.is_dir():
                continue
            # Exact match on mangled path
            if d.name == expected:
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
        """Save the session processing registry (atomic write)."""
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

    def _score_content(self, text: str) -> tuple[float, str]:
        """Score content by importance and infer type."""
        return score_content(text)

    def _extract_text_from_content(self, content: Any) -> str:
        """Extract text from a message content field.

        Handles both string content and list-of-blocks content.
        Only extracts 'text' type blocks (skips tool_use, tool_result, thinking).
        """
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts).strip()
        return ""

    def _is_real_user_message(self, content: Any) -> str:
        """Extract clean text from a real human message, or empty string for noise.

        Filters out:
        - tool_result blocks (92% of user records)
        - Slash command definitions injected as text
        - Compact continuation summaries
        - System reminders and local command output
        """
        if isinstance(content, str):
            text = content.strip()
            if any(marker in text for marker in _USER_NOISE_MARKERS):
                return ""
            return text

        if isinstance(content, list):
            # If any block is a tool_result, this is a tool return -- not human input
            if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                return ""
            # Extract text blocks only
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            text = "\n".join(parts).strip()
            if any(marker in text for marker in _USER_NOISE_MARKERS):
                return ""
            return text

        return ""

    def _extract_from_jsonl(self, path: Path) -> list[dict[str, Any]]:
        """Extract high-value entries from a Claude Code JSONL transcript.

        Claude Code JSONL records have:
        - Top-level `type`: "user", "assistant", "progress", etc.
        - Message content at record["message"]["content"]
        - Content is list of blocks: {type: "text"}, {type: "tool_use"}, etc.

        Pairs user questions with assistant text responses.
        Scores each response by importance and filters low-value content.
        """
        entries: list[dict[str, Any]] = []
        last_user_text = ""

        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    record_type = record.get("type")

                    # Track last real user question for pairing
                    if record_type == "user":
                        msg = record.get("message", {})
                        text = self._is_real_user_message(msg.get("content", ""))
                        if text and len(text) > 10:
                            last_user_text = text
                        continue

                    # queue-operation enqueue = cleaner user input
                    if record_type == "queue-operation":
                        content = record.get("content", "")
                        if (
                            record.get("operation") == "enqueue"
                            and isinstance(content, str)
                            and len(content) > 10
                            and not any(m in content for m in _USER_NOISE_MARKERS)
                        ):
                            last_user_text = content.strip()
                        continue

                    # Only process assistant records
                    if record_type != "assistant":
                        continue

                    # Extract text from the message (skip tool_use, thinking blocks)
                    msg = record.get("message", {})
                    content = msg.get("content", [])
                    assistant_text = self._extract_text_from_content(content)

                    if len(assistant_text) < MIN_CONTENT_LENGTH:
                        continue

                    # Score the assistant content
                    score, entry_type = self._score_content(assistant_text)
                    if score < 0.2:
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

                    # Combine user question + assistant response
                    insight_parts = []
                    if last_user_text:
                        insight_parts.append(last_user_text[:300])
                        insight_parts.append("---")
                    insight_parts.append(assistant_text[:2000])
                    insight = "\n".join(insight_parts)[:2500]

                    date_str = self._extract_date(path)

                    entries.append(
                        {
                            "title": title,
                            "insight": insight,
                            "type": entry_type or "session",
                            "priority": "high" if score > 0.7 else "medium",
                            "keywords": [],
                            "source": f"session:{path.name}",
                            "date": date_str,
                            "timestamp": datetime.now().isoformat(),
                            "branch": "",
                            "related_to": [],
                            "_importance_score": score,
                        }
                    )

        except Exception as e:
            debug_log(f"Failed to parse {path}: {e}")

        return entries

    def _extract_date(self, path: Path) -> str:
        """Extract date from session file path or mtime."""
        name = path.stem
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", name)
        if date_match:
            return date_match.group(1)
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

    def _cleanup_deleted_files(self, current_file_keys: set[str]) -> int:
        """Remove DB entries for session files no longer on disk."""
        stale_keys = set(self._registry.keys()) - current_file_keys
        if not stale_keys:
            return 0

        from knowlin_mcp.db import KnowledgeDB

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

        # Find JSONL files (top-level + subagent subdirectories)
        jsonl_files = sorted(self.sessions_dir.glob("*.jsonl"))
        for subdir in self.sessions_dir.iterdir():
            if subdir.is_dir():
                jsonl_files.extend(sorted(subdir.glob("*.jsonl")))

        # Cleanup entries for deleted files
        current_file_keys = {str(p) for p in jsonl_files}
        self._cleanup_deleted_files(current_file_keys)

        if not jsonl_files:
            debug_log("No JSONL files found")
            return 0

        # Filter to unprocessed files, caching hashes
        to_process = []
        file_hashes: dict[str, str] = {}
        for path in jsonl_files:
            file_key = str(path)
            if not full:
                current_hash = self._file_hash(path)
                file_hashes[file_key] = current_hash
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

        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(self.project_path), sub_store="sessions")

        # Extract entries from all files
        all_entries = []
        file_entry_counts: list[tuple[str, int]] = []

        for path in to_process:
            entries = self._extract_from_jsonl(path)
            entries.sort(key=lambda x: x.get("_importance_score", 0), reverse=True)
            file_entry_counts.append((str(path), len(entries)))
            all_entries.extend(entries)

        if not all_entries:
            for path in to_process:
                fk = str(path)
                self._registry[fk] = {
                    "hash": file_hashes.get(fk) or self._file_hash(path),
                    "processed": datetime.now().isoformat(),
                    "entries_extracted": 0,
                    "entry_ids": [],
                }
            self._save_registry()
            return 0

        # Remove internal scoring field before DB add
        for e in all_entries:
            e.pop("_importance_score", None)

        # Batch add first, then remove old entries (crash-safe ordering)
        ids = db.batch_add(all_entries, check_duplicates=False)
        accepted_count = sum(1 for entry_id in ids if entry_id is not None)

        for path in to_process:
            old_ids = self._registry.get(str(path), {}).get("entry_ids", [])
            if old_ids:
                db.remove_entries(old_ids)
                debug_log(f"Removed {len(old_ids)} old entries for {path.name}")

        rejected_count = sum(1 for entry_id in ids if entry_id is None)
        if rejected_count:
            debug_log(
                "Warning: DB validation rejected "
                f"{rejected_count} session entries during batch_add."
            )

        # Distribute IDs back to registry per file
        offset = 0
        for file_key, count in file_entry_counts:
            raw_file_ids = ids[offset : offset + count] if count > 0 else []
            file_ids = [entry_id for entry_id in raw_file_ids if entry_id is not None]
            self._registry[file_key] = {
                "hash": file_hashes.get(file_key) or self._file_hash(Path(file_key)),
                "processed": datetime.now().isoformat(),
                "entries_extracted": count,
                "entry_ids": file_ids,
            }
            offset += count

        self._save_registry()
        debug_log(f"Ingested {accepted_count} entries from {len(to_process)} sessions")
        return accepted_count
