"""Codex CLI Session Ingester - Extract knowledge from Codex JSONL transcripts.

Codex CLI JSONL envelope format (per-line records):
- "session_meta": session metadata (id, cwd) -- skip
- "event_msg" with payload.type "user_message": user input
- "response_item": assistant response with content blocks
- "turn_context", "compacted", other event_msg types: skip

Uses the same sessions sub-store and entry format as SessionIngester.
Registry file: codex-registry.json (separate from session-registry.json).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from knowlin_mcp.ingest_sessions import score_content
from knowlin_mcp.utils import debug_log

# Minimum content length worth indexing
MIN_CONTENT_LENGTH = 150


class CodexIngester:
    """Ingest Codex CLI session transcripts into the sessions sub-store."""

    def __init__(self, project_path: str, codex_dir: str | None = None):
        self.project_path = Path(project_path).resolve()
        self.db_path = self.project_path / ".knowledge-db"
        self.registry_path = self.db_path / "codex-registry.json"

        from knowlin_mcp.ingest_docs import _resolve_paths, load_sources_config

        sources_config = load_sources_config(self.db_path)
        codex_config = (sources_config or {}).get("codex", {})

        self.codex_dir: Path | None = None
        if codex_dir:
            self.codex_dir = Path(codex_dir)
        elif codex_config.get("path"):
            paths = _resolve_paths([codex_config["path"]], self.project_path)
            self.codex_dir = paths[0] if paths else None
        elif codex_config.get("auto_discover", True):
            self.codex_dir = self._find_codex_dir()

        self._registry: dict[str, dict] = self._load_registry()

    def _find_codex_dir(self) -> Path | None:
        """Find Codex CLI sessions directory (~/.codex/sessions/)."""
        codex_dir = Path.home() / ".codex" / "sessions"
        if codex_dir.exists():
            return codex_dir
        return None

    def _load_registry(self) -> dict[str, dict]:
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_registry(self) -> None:
        self.db_path.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._registry, indent=2))
        tmp.rename(self.registry_path)

    def _file_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _extract_from_codex_jsonl(self, path: Path) -> list[dict[str, Any]]:
        """Extract high-value entries from a Codex CLI JSONL transcript."""
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

                    # User message from event_msg envelope
                    if record_type == "event_msg":
                        payload = record.get("payload", {})
                        if payload.get("type") == "user_message":
                            msg = payload.get("message", "")
                            if isinstance(msg, str) and len(msg) > 10:
                                last_user_text = msg.strip()
                        continue

                    # Assistant response from response_item envelope
                    if record_type != "response_item":
                        continue

                    item = record.get("item", {})
                    content = item.get("content", [])
                    assistant_text = self._extract_assistant_text(content)

                    if len(assistant_text) < MIN_CONTENT_LENGTH:
                        continue

                    score, entry_type = score_content(assistant_text)
                    if score < 0.2:
                        continue

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
                            "source": f"codex:{path.name}",
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

    def _extract_assistant_text(self, content: Any) -> str:
        """Extract text from Codex response_item content blocks.

        Handles two variants:
        - {"text": "..."} (standard)
        - {"OutputText": {"text": "..."}} (structured output)
        """
        if not isinstance(content, list):
            return ""
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if "text" in block:
                parts.append(block["text"])
            elif "OutputText" in block:
                ot = block["OutputText"]
                if isinstance(ot, dict) and "text" in ot:
                    parts.append(ot["text"])
        return "\n".join(parts).strip()

    def _extract_date(self, path: Path) -> str:
        """Extract date from file path (YYYY/MM/DD dirs) or mtime."""
        # Codex stores files under YYYY/MM/DD directories
        parts = path.parts
        for i, part in enumerate(parts):
            if re.match(r"^\d{4}$", part) and i + 2 < len(parts):
                mm = parts[i + 1]
                dd = parts[i + 2]
                if re.match(r"^\d{2}$", mm) and re.match(r"^\d{2}$", dd):
                    return f"{part}-{mm}-{dd}"
        # Fallback: check filename for date pattern
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path.stem)
        if date_match:
            return date_match.group(1)
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

    def _cleanup_deleted_files(self, current_file_keys: set[str]) -> int:
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
                debug_log(f"Removed {len(entry_ids)} stale codex entries from {key}")
            del self._registry[key]

        if removed:
            self._save_registry()
        return removed

    def ingest(self, full: bool = False) -> int:
        """Ingest Codex CLI session transcripts into the sessions sub-store."""
        if self.codex_dir is None or not self.codex_dir.exists():
            debug_log("No Codex sessions directory found")
            return 0

        # Find JSONL files (recursively through date-nested dirs)
        jsonl_files = sorted(self.codex_dir.rglob("*.jsonl"))

        current_file_keys = {str(p) for p in jsonl_files}
        self._cleanup_deleted_files(current_file_keys)

        if not jsonl_files:
            debug_log("No Codex JSONL files found")
            return 0

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
            debug_log("All Codex sessions already processed")
            return 0

        debug_log(f"Processing {len(to_process)} Codex session files...")

        from knowlin_mcp.db import KnowledgeDB

        db = KnowledgeDB(str(self.project_path), sub_store="sessions")

        all_entries = []
        file_entry_counts: list[tuple[str, int]] = []

        for path in to_process:
            entries = self._extract_from_codex_jsonl(path)
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

        for e in all_entries:
            e.pop("_importance_score", None)

        # Batch add first, then remove old entries (crash-safe ordering)
        ids = db.batch_add(all_entries, check_duplicates=False)
        accepted_count = sum(1 for entry_id in ids if entry_id is not None)

        rejected_count = sum(1 for entry_id in ids if entry_id is None)
        if rejected_count:
            debug_log(
                f"Warning: DB validation rejected {rejected_count} Codex entries during batch_add."
            )

        for path in to_process:
            old_ids = self._registry.get(str(path), {}).get("entry_ids", [])
            if old_ids:
                db.remove_entries(old_ids)

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
        debug_log(f"Ingested {accepted_count} entries from {len(to_process)} Codex sessions")
        return accepted_count
