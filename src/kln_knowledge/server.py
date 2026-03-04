"""Knowledge Server - Per-project fastembed daemon for fast searches.

Each project gets its own server with a dedicated TCP port.
Eliminates cold start by keeping embeddings loaded in memory.
Auto-shuts down after 1 hour of inactivity.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

from kln_knowledge.platform import (
    HOST,
    find_project_root,
    get_kb_pid_file,
    get_kb_port,
    get_kb_port_file,
    get_project_hash,
    get_runtime_dir,
    is_process_running,
    write_pid_file,
)

IDLE_TIMEOUT = 3600  # 1 hour


def find_available_port(start_port: int, max_attempts: int = 100) -> int:
    """Find an available TCP port starting from start_port."""
    for offset in range(max_attempts):
        port = start_port + offset
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((HOST, port))
            sock.close()
            return port
        except OSError:
            continue
    raise RuntimeError(
        f"No available port found in range {start_port}-{start_port + max_attempts}"
    )


def read_port_file(project_path: Path) -> int | None:
    """Read port from project's port file."""
    port_file = get_kb_port_file(project_path)
    try:
        if port_file.exists():
            return int(port_file.read_text().strip())
    except (ValueError, OSError):
        pass
    return None


def write_port_file(project_path: Path, port: int) -> None:
    """Write port to project's port file."""
    port_file = get_kb_port_file(project_path)
    port_file.write_text(str(port))


def list_running_servers() -> list[dict]:
    """List all running knowledge servers."""
    servers = []
    runtime_dir = get_runtime_dir()

    for port_file in runtime_dir.glob("kb-*.port"):
        pid_file = port_file.with_suffix(".pid")
        if not pid_file.exists():
            continue

        try:
            port = int(port_file.read_text().strip())
            pid = int(pid_file.read_text().strip())

            if not is_process_running(pid):
                port_file.unlink(missing_ok=True)
                pid_file.unlink(missing_ok=True)
                continue

            info = send_command_to_port(port, {"cmd": "status"})
            if info and "project" in info:
                servers.append({
                    "port": port,
                    "pid": pid,
                    "project": info.get("project", "unknown"),
                    "load_time": info.get("load_time", 0),
                })
        except (ValueError, OSError):
            pass

    return servers


class KnowledgeServer:
    """TCP-based knowledge server for fast semantic search."""

    def __init__(self, project_path: str | Path | None = None):
        self.project_root = find_project_root(
            Path(project_path) if project_path else None
        )
        if not self.project_root:
            raise ValueError(
                f"No .knowledge-db found from {project_path or os.getcwd()}"
            )

        self.port = 0
        self.db = None
        self.running = False
        self.load_time = 0.0
        self.last_activity = time.time()

    def load_index(self) -> bool:
        """Load fastembed-based knowledge index."""
        db_path = self.project_root / ".knowledge-db"
        has_fastembed = (db_path / "embeddings.npy").exists()
        has_entries = (db_path / "entries.jsonl").exists()

        if not has_fastembed and not has_entries:
            db_path.mkdir(parents=True, exist_ok=True)
            (db_path / "entries.jsonl").touch()
            print(f"Auto-initialized empty Knowledge DB at {db_path}")

        print(f"Loading index from {db_path}...")
        start = time.time()

        from kln_knowledge.db import KnowledgeDB

        self.db = KnowledgeDB(str(self.project_root))
        self.load_time = time.time() - start
        count = self.db.count()
        print(f"Index loaded in {self.load_time:.2f}s ({count} entries)")
        return True

    def search(self, query: str, limit: int = 5) -> dict:
        """Perform semantic search."""
        self.last_activity = time.time()

        if not self.db:
            return {"error": "No index loaded"}

        start = time.time()
        results = self.db.search(query, limit)
        search_time = time.time() - start

        return {
            "results": results,
            "search_time_ms": round(search_time * 1000, 2),
            "query": query,
        }

    def handle_client(self, conn: socket.socket) -> None:
        """Handle a client connection."""
        try:
            conn.settimeout(30.0)
            # Read until EOF (client must shutdown(SHUT_WR) after sending)
            chunks = []
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if len(b"".join(chunks)) > 1024 * 1024:  # 1 MiB cap
                    conn.sendall(json.dumps({"error": "Request too large"}).encode())
                    return
            data = b"".join(chunks).decode("utf-8")
            if not data:
                return

            request = json.loads(data)
            cmd = request.get("cmd", "search")

            # Only update idle timer for meaningful commands (not ping/status)
            if cmd not in ("ping", "status"):
                self.last_activity = time.time()

            if cmd == "search":
                query = request.get("query", "")
                limit = request.get("limit", 5)
                date_from = request.get("date_from")
                date_to = request.get("date_to")
                entry_type = request.get("entry_type")
                branch = request.get("branch")
                sources = request.get("sources")

                # Multi-source search
                if sources and len(sources) > 1:
                    from kln_knowledge.multi_search import MultiSourceSearch
                    ms = MultiSourceSearch(str(self.project_root))
                    results = ms.search(
                        query, sources=sources, limit=limit,
                        date_from=date_from, date_to=date_to,
                        entry_type=entry_type, branch=branch,
                    )
                    response = {
                        "results": results,
                        "query": query,
                        "multi_source": True,
                    }
                elif any([date_from, date_to, entry_type, branch]):
                    self.last_activity = time.time()
                    if not self.db:
                        response = {"error": "No index loaded"}
                    else:
                        start = time.time()
                        results = self.db.search(
                            query, limit,
                            date_from=date_from, date_to=date_to,
                            entry_type=entry_type, branch=branch,
                        )
                        search_time = time.time() - start
                        response = {
                            "results": results,
                            "search_time_ms": round(search_time * 1000, 2),
                            "query": query,
                        }
                else:
                    response = self.search(query, limit)

            elif cmd == "status":
                idle_time = time.time() - self.last_activity
                response = {
                    "status": "running",
                    "project": str(self.project_root),
                    "port": self.port,
                    "load_time": self.load_time,
                    "index_loaded": self.db is not None,
                    "idle_seconds": int(idle_time),
                    "entries": self.db.count() if self.db else 0,
                    "backend": "fastembed",
                }

            elif cmd == "ping":
                response = {
                    "pong": True,
                    "project": str(self.project_root),
                    "port": self.port,
                }

            elif cmd == "add":
                entry = request.get("entry")
                if not entry:
                    response = {"error": "No entry provided"}
                elif not self.db:
                    response = {"error": "No index loaded"}
                else:
                    try:
                        entry_id = self.db.add(entry)
                        response = {"status": "ok", "id": entry_id}
                    except Exception as e:
                        response = {"error": f"Failed to add entry: {e}"}

            elif cmd == "update_usage":
                entry_ids = request.get("ids", [])
                if not entry_ids:
                    response = {"error": "No ids provided"}
                elif not self.db:
                    response = {"error": "No index loaded"}
                else:
                    try:
                        updated = self.db.update_usage(entry_ids)
                        response = {"status": "ok", "updated": updated}
                    except Exception as e:
                        response = {"error": f"Failed to update usage: {e}"}

            elif cmd == "recent":
                limit = request.get("limit", 3)
                if not self.db:
                    response = {"error": "No index loaded"}
                else:
                    try:
                        entries = self.db.get_recent_important(limit)
                        response = {"status": "ok", "entries": entries}
                    except Exception as e:
                        response = {"error": f"Failed to get recent: {e}"}

            elif cmd == "search_by_date":
                start_date = request.get("start", "")
                end_date = request.get("end")
                limit = request.get("limit", 50)
                if not self.db:
                    response = {"error": "No index loaded"}
                elif not start_date:
                    response = {"error": "Missing 'start' date parameter"}
                else:
                    try:
                        entries = self.db.search_by_date(start_date, end_date, limit)
                        response = {"status": "ok", "entries": entries, "count": len(entries)}
                    except Exception as e:
                        response = {"error": f"search_by_date failed: {e}"}

            elif cmd == "get_timeline":
                date = request.get("date", "")
                if not self.db:
                    response = {"error": "No index loaded"}
                elif not date:
                    response = {"error": "Missing 'date' parameter"}
                else:
                    try:
                        entries = self.db.get_timeline(date)
                        response = {"status": "ok", "entries": entries, "count": len(entries)}
                    except Exception as e:
                        response = {"error": f"get_timeline failed: {e}"}

            elif cmd == "get_related":
                entry_id = request.get("id", "")
                if not self.db:
                    response = {"error": "No index loaded"}
                elif not entry_id:
                    response = {"error": "Missing 'id' parameter"}
                else:
                    try:
                        entries = self.db.get_related(entry_id)
                        response = {"status": "ok", "entries": entries, "count": len(entries)}
                    except Exception as e:
                        response = {"error": f"get_related failed: {e}"}

            elif cmd == "get":
                entry_id = request.get("id", "")
                if not self.db:
                    response = {"error": "No index loaded"}
                elif not entry_id:
                    response = {"error": "Missing 'id' parameter"}
                else:
                    entry = self.db.get(entry_id)
                    if entry:
                        response = {"status": "ok", "entry": entry}
                    else:
                        response = {"error": f"Entry not found: {entry_id}"}

            elif cmd == "ingest":
                source = request.get("source", "all")
                full = request.get("full", False)
                try:
                    counts = {}
                    if source in ("sessions", "all"):
                        from kln_knowledge.ingest_sessions import SessionIngester
                        si = SessionIngester(str(self.project_root))
                        counts["sessions"] = si.ingest(full=full)
                    if source in ("docs", "all"):
                        from kln_knowledge.ingest_docs import DocsIngester
                        di = DocsIngester(str(self.project_root))
                        counts["docs"] = di.ingest(full=full)
                    total = sum(counts.values())
                    # Reload main index if entries were added
                    if total > 0 and self.db:
                        self.db._load_index()
                    response = {"status": "ok", "counts": counts, "total": total}
                except Exception as e:
                    response = {"error": f"Ingest failed: {e}"}

            elif cmd == "reload":
                if not self.db:
                    response = {"error": "No index loaded"}
                else:
                    try:
                        old_count = self.db.count()
                        self.db._load_index()
                        new_count = self.db.count()
                        response = {
                            "status": "ok",
                            "old_count": old_count,
                            "new_count": new_count,
                        }
                    except Exception as e:
                        response = {"error": f"Reload failed: {e}"}
            else:
                response = {"error": f"Unknown command: {cmd}"}

            conn.sendall(json.dumps(response).encode("utf-8"))
        except Exception as e:
            try:
                conn.sendall(json.dumps({"error": str(e)}).encode("utf-8"))
            except OSError:
                pass
        finally:
            conn.close()

    def check_idle_timeout(self) -> bool:
        """Check if server should shut down due to inactivity."""
        idle_time = time.time() - self.last_activity
        if idle_time > IDLE_TIMEOUT:
            print(f"\nIdle timeout ({IDLE_TIMEOUT}s) reached. Shutting down...")
            return True
        return False

    def start(self) -> None:
        """Start the TCP server."""
        if not self.load_index():
            print("ERROR: No index found in .knowledge-db/")
            sys.exit(1)

        base_port = get_kb_port()
        project_hash = get_project_hash(self.project_root)
        hash_offset = int(project_hash[:2], 16) % 256
        self.port = find_available_port(base_port + hash_offset)

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, self.port))
        server.listen(5)

        pid_file = get_kb_pid_file(self.project_root)
        port_file = get_kb_port_file(self.project_root)
        write_pid_file(pid_file, os.getpid())
        write_port_file(self.project_root, self.port)

        print("Knowledge server started")
        print(f"  Port:    {HOST}:{self.port}")
        print(f"  Project: {self.project_root}")
        print(f"  Timeout: {IDLE_TIMEOUT}s idle")
        print("Ready for queries (Ctrl+C to stop)")

        self.running = True

        def signal_handler(sig, frame):
            print("\nShutting down...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        while self.running:
            try:
                server.settimeout(60.0)
                conn, _ = server.accept()
                threading.Thread(
                    target=self.handle_client, args=(conn,), daemon=True
                ).start()
            except socket.timeout:
                if self.check_idle_timeout():
                    break
                continue
            except Exception as e:
                if self.running:
                    print(f"Error: {e}")

        server.close()
        pid_file.unlink(missing_ok=True)
        port_file.unlink(missing_ok=True)
        print("Server stopped")


def send_command_to_port(
    port: int, cmd_data: dict, timeout: float = 5.0
) -> dict | None:
    """Send command to server on specified port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect((HOST, port))
            client.sendall(json.dumps(cmd_data).encode("utf-8"))
            client.shutdown(socket.SHUT_WR)  # signal end of request
            # Read until EOF
            chunks = []
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
            return json.loads(b"".join(chunks).decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}


def send_command(project_path: Path, cmd_data: dict) -> dict | None:
    """Send command to the server for a project."""
    port = read_port_file(project_path)
    if not port:
        return None
    return send_command_to_port(port, cmd_data)
