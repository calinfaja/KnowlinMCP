"""Shared utilities for KLN Knowledge System.

TCP communication, schema constants, type inference, and debug logging.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

from kln_knowledge.platform import (
    HOST,
    KB_DIR_NAME,
    get_kb_pid_file,
    get_kb_port_file,
)

# =============================================================================
# Debug Logging
# =============================================================================


def debug_log(msg: str, category: str = "kb") -> None:
    """Log debug message if KLEAN_DEBUG is set."""
    if os.environ.get("KLEAN_DEBUG"):
        print(f"[{category}] {msg}", file=sys.stderr)


# =============================================================================
# V3 Schema Constants
# =============================================================================

SCHEMA_V3_FIELDS = [
    "id",
    "title",
    "insight",
    "type",
    "priority",
    "keywords",
    "source",
    "date",
    # V3.1 extensions
    "timestamp",
    "branch",
    "related_to",
]

SCHEMA_V3_DEFAULTS = {
    "type": "finding",
    "priority": "medium",
    "keywords": [],
    "source": "",
    "timestamp": "",
    "branch": "",
    "related_to": [],
}

SCHEMA_V2_DEFAULTS = {
    "confidence_score": 0.7,
    "tags": [],
    "usage_count": 0,
    "last_used": None,
    "source_quality": "medium",
    "atomic_insight": "",
    "key_concepts": [],
    "quality": "medium",
    "source": "manual",
    "source_path": "",
}


# =============================================================================
# Type Inference
# =============================================================================

TYPE_SIGNALS = {
    "warning": [
        "don't", "dont", "avoid", "never", "careful", "bug", "broken",
        "fails", "failed", "failure", "gotcha", "watch out", "issue",
        "problem", "error", "warning", "deprecated", "regression",
        "won't work", "doesn't work", "not work",
    ],
    "solution": [
        "fixed", "solved", "fix by", "fix:", "solution", "resolved",
        "workaround", "the fix", "to fix",
    ],
    "pattern": [
        "use ", "prefer", "always", "best way", "pattern", "approach",
        "technique", "how to", "should ", "recommended", "best practice",
        "convention",
    ],
    "decision": [
        "chose", "decided", "picked", "selected", "went with",
        "instead of", "decision", "trade-off", "tradeoff",
    ],
    "discovery": [
        "found that", "discovered", "turns out", "TIL", "realized",
        "learned that", "it turns out", "surprisingly",
    ],
}


def infer_type(title: str, insight: str) -> str:
    """Infer entry type from content using signal words.

    Checks in priority order: warning > solution > pattern > decision > discovery.
    Returns 'finding' if no signals match.
    """
    text = f"{title} {insight}".lower()

    for entry_type, signals in TYPE_SIGNALS.items():
        if any(signal in text for signal in signals):
            return entry_type

    return "finding"


# =============================================================================
# Schema Migration
# =============================================================================


def migrate_entry(entry: dict) -> dict:
    """Migrate entry to V3 schema.

    Handles V2 -> V3 field mappings:
    - summary/atomic_insight -> insight
    - tags/key_concepts -> keywords
    - url/source_path -> source
    - found_date -> date
    """
    # Merge summary + atomic_insight -> insight
    if "insight" not in entry:
        atomic = entry.get("atomic_insight", "")
        summary = entry.get("summary", "")
        if atomic and summary:
            entry["insight"] = f"{atomic} {summary}" if atomic not in summary else summary
        else:
            entry["insight"] = atomic or summary or entry.get("title", "")

    # Merge tags + key_concepts -> keywords
    if "keywords" not in entry:
        tags = entry.get("tags", [])
        concepts = entry.get("key_concepts", [])
        seen = set()
        keywords = []
        for kw in tags + concepts:
            if kw and kw.lower() not in seen:
                seen.add(kw.lower())
                keywords.append(kw)
        entry["keywords"] = keywords

    # Unify source fields
    if not entry.get("source") or entry.get("source") in ["manual", "conversation", "review"]:
        url = entry.get("url", "")
        source_path = entry.get("source_path", "")
        if url and url.startswith("http"):
            entry["source"] = url
        elif source_path and source_path.startswith("http"):
            entry["source"] = source_path
        elif source_path:
            if not source_path.startswith(("file:", "git:", "conv:")):
                entry["source"] = f"file:{source_path}"
            else:
                entry["source"] = source_path
        else:
            found_date = (entry.get("found_date") or "")[:10]
            entry["source"] = f"conv:{found_date}" if found_date else "conv:unknown"

    # Map found_date -> date
    if "date" not in entry:
        entry["date"] = (entry.get("found_date") or "")[:10]

    # Infer type if not set or generic
    if entry.get("type") in [None, "", "lesson", "best-practice"]:
        entry["type"] = infer_type(entry.get("title", ""), entry.get("insight", ""))

    # Map quality scores -> priority
    if "priority" not in entry:
        relevance = entry.get("relevance_score", 0.5)
        confidence = entry.get("confidence_score", 0.5)
        quality = entry.get("quality", "medium")

        if quality == "high" or relevance >= 0.9 or confidence >= 0.9:
            entry["priority"] = "high"
        elif quality == "low" or relevance < 0.5 or confidence < 0.5:
            entry["priority"] = "low"
        else:
            entry["priority"] = "medium"

    # V3.1: Set timestamp from found_date
    if "timestamp" not in entry:
        found_date = entry.get("found_date") or ""
        if found_date and "T" in found_date:
            entry["timestamp"] = found_date
        elif found_date:
            entry["timestamp"] = f"{found_date[:10]}T00:00:00"
        else:
            entry["timestamp"] = ""

    # Apply V3 defaults for any missing fields
    for field, default in SCHEMA_V3_DEFAULTS.items():
        if field not in entry:
            if isinstance(default, list):
                entry[field] = list(default)
            else:
                entry[field] = default

    return entry


# =============================================================================
# Port File / Server Status
# =============================================================================


def get_server_port(project_path: str | Path) -> int | None:
    """Get KB server port for a project from its port file."""
    port_file = get_kb_port_file(Path(project_path))
    try:
        if port_file.exists():
            return int(port_file.read_text().strip())
    except (ValueError, OSError):
        pass
    return None


def get_pid_path(project_path: str | Path) -> str:
    """Get KB server PID file path for a project."""
    return str(get_kb_pid_file(Path(project_path)))


def is_kb_initialized(project_path: str | Path) -> bool:
    """Check if knowledge DB is initialized (has .knowledge-db dir)."""
    if not project_path:
        return False
    return (Path(project_path) / KB_DIR_NAME).is_dir()


def is_server_running(project_path: str | Path, timeout: float = 0.5) -> bool:
    """Check if KB server is running and responding to ping."""
    port = get_server_port(project_path)
    if not port:
        return False

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((HOST, port))
            sock.sendall(b'{"cmd":"ping"}')
            sock.shutdown(socket.SHUT_WR)
            response = sock.recv(1024).decode()
            return '"pong"' in response
    except Exception:
        return False


def clean_stale_socket(project_path: str | Path) -> bool:
    """Remove stale port/pid files if server not responding."""
    project = Path(project_path)
    port_file = get_kb_port_file(project)
    pid_file = get_kb_pid_file(project)

    if not port_file.exists():
        return False

    if not is_server_running(project_path):
        try:
            port_file.unlink(missing_ok=True)
            pid_file.unlink(missing_ok=True)
            debug_log(f"Cleaned stale files for: {project_path}")
            return True
        except Exception as e:
            debug_log(f"Failed to clean files: {e}")
    return False


# =============================================================================
# TCP Communication
# =============================================================================


def send_command(
    project_path: str | Path, cmd_data: dict, timeout: float = 5.0
) -> dict | None:
    """Send command to KB server for a project. Returns response dict or None."""
    port = get_server_port(project_path)
    if not port:
        return None

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((HOST, port))
            sock.sendall(json.dumps(cmd_data).encode("utf-8"))
            sock.shutdown(socket.SHUT_WR)
            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
            return json.loads(b"".join(chunks).decode("utf-8"))
    except Exception as e:
        debug_log(f"Send command failed: {e}")
        return None


def search(
    project_path: str | Path,
    query: str,
    limit: int = 5,
    date_from: str | None = None,
    date_to: str | None = None,
    entry_type: str | None = None,
    branch: str | None = None,
) -> dict | None:
    """Perform semantic search via KB server with optional filters."""
    cmd = {"cmd": "search", "query": query, "limit": limit}
    if date_from:
        cmd["date_from"] = date_from
    if date_to:
        cmd["date_to"] = date_to
    if entry_type:
        cmd["entry_type"] = entry_type
    if branch:
        cmd["branch"] = branch
    return send_command(project_path, cmd)
