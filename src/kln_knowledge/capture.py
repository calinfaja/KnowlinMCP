"""Knowledge Capture - Save entries to the knowledge database.

Supports simple CLI input and structured JSON. Entry creation with
fallback chain: server -> direct DB -> JSONL-only.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from kln_knowledge.platform import KB_DIR_NAME, HOST, get_kb_port_file
from kln_knowledge.utils import debug_log, infer_type, is_server_running


def _get_current_branch() -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def create_entry(
    content: str,
    entry_type: str = "finding",
    tags: list[str] | str | None = None,
    priority: str = "medium",
    url: str | None = None,
) -> dict:
    """Create a knowledge entry from simple input (V3 schema)."""
    if tags is None:
        tags = []
    elif isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    if entry_type in ("lesson", "best-practice"):
        entry_type = "finding"

    entry_id = f"{entry_type}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    return {
        "id": entry_id,
        "title": content[:100] if len(content) <= 100 else content[:97] + "...",
        "insight": content,
        "type": entry_type,
        "priority": priority,
        "keywords": tags[:10],
        "source": url or f"conv:{datetime.now().strftime('%Y-%m-%d')}",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "branch": _get_current_branch(),
        "related_to": [],
    }


def create_entry_from_json(data: dict) -> dict:
    """Create a knowledge entry from structured JSON (V3 schema, accepts V2)."""
    entry_type = data.get("type", "")

    if not entry_type or entry_type in ("lesson", "best-practice"):
        title = data.get("title", "")
        insight = (
            data.get("insight")
            or data.get("atomic_insight")
            or data.get("summary")
            or ""
        )
        entry_type = infer_type(title, insight)

    entry_id = data.get("id") or f"{entry_type}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    insight = (
        data.get("insight")
        or data.get("atomic_insight")
        or data.get("summary")
        or ""
    )

    keywords = data.get("keywords")
    if not keywords:
        tags = data.get("tags", [])
        concepts = data.get("key_concepts", [])
        keywords = list(dict.fromkeys(tags + concepts))

    source = data.get("source", "")
    if not source or source in ("manual", "conversation", "review"):
        source = (
            data.get("url")
            or data.get("source_path")
            or f"conv:{datetime.now().strftime('%Y-%m-%d')}"
        )

    entry = {
        "id": entry_id,
        "title": data.get("title", ""),
        "insight": insight,
        "type": entry_type,
        "priority": data.get("priority", "medium"),
        "keywords": keywords[:10],
        "source": source,
        "date": data.get("date") or datetime.now().strftime("%Y-%m-%d"),
        "timestamp": data.get("timestamp") or datetime.now().isoformat(),
        "branch": data.get("branch") or _get_current_branch(),
        "related_to": data.get("related_to", []),
    }

    if not entry["title"] and entry["insight"]:
        entry["title"] = entry["insight"][:100]

    return entry


def send_entry_to_server(entry: dict, project_path: str) -> bool:
    """Send entry to running KB server via TCP (preferred method)."""
    import socket

    port_file = get_kb_port_file(Path(project_path))
    if not port_file.exists():
        debug_log("KB server not running (no port file)")
        return False

    try:
        port = int(port_file.read_text().strip())
    except (ValueError, OSError):
        debug_log("Invalid port file")
        return False

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((HOST, port))
        sock.sendall(json.dumps({"cmd": "add", "entry": entry}).encode("utf-8"))
        response = sock.recv(65536).decode("utf-8")
        sock.close()

        result = json.loads(response)
        if result.get("status") == "ok":
            debug_log(f"Entry added via server: {result.get('id')}")
            return True
        else:
            debug_log(f"Server rejected entry: {result.get('error')}")
            return False
    except Exception as e:
        debug_log(f"Failed to send to server: {e}")
        return False


def _notify_server_reload(project_path: str) -> None:
    """Send reload command to running KB server (best-effort)."""
    import socket

    port_file = get_kb_port_file(Path(project_path))
    if not port_file.exists():
        return

    try:
        port = int(port_file.read_text().strip())
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((HOST, port))
        sock.sendall(json.dumps({"cmd": "reload"}).encode("utf-8"))
        sock.recv(4096)
        sock.close()
        debug_log("Notified server to reload index")
    except Exception as e:
        debug_log(f"Server reload notification failed (non-fatal): {e}")


def save_entry(entry: dict, knowledge_dir: Path) -> bool:
    """Save entry with fallback chain: server -> direct DB -> JSONL-only."""
    project_path = str(knowledge_dir.parent)

    # Method 1: Try server (immediate sync)
    if send_entry_to_server(entry, project_path):
        return True

    # Method 2: Direct KnowledgeDB
    try:
        from kln_knowledge.db import KnowledgeDB

        db = KnowledgeDB(project_path)
        db.add(entry)
        debug_log("Entry added via direct KnowledgeDB")
        _notify_server_reload(project_path)
        return True
    except ImportError:
        debug_log("KnowledgeDB not available, falling back to JSONL-only")
    except Exception as e:
        debug_log(f"KnowledgeDB.add() failed: {e}, falling back to JSONL-only")

    # Method 3: JSONL-only fallback
    entries_file = knowledge_dir / "entries.jsonl"
    with open(entries_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    debug_log("Entry appended to JSONL")
    _notify_server_reload(project_path)

    return True


def log_to_timeline(content: str, entry_type: str, knowledge_dir: Path) -> None:
    """Log to timeline for chronological tracking."""
    timeline_file = knowledge_dir / "timeline.txt"
    timestamp = datetime.now().strftime("%m-%d %H:%M")
    short_content = content[:80].replace("\n", " ")
    timeline_entry = f"{timestamp} | {entry_type} | {short_content}"

    with open(timeline_file, "a") as f:
        f.write(timeline_entry + "\n")
