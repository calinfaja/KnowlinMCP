"""KLN Knowledge System CLI.

Entry point: kln-kb
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from kln_knowledge.platform import (
    KB_DIR_NAME,
    cleanup_stale_files,
    find_project_root,
    get_kb_pid_file,
    get_kb_port_file,
    is_process_running,
    kill_process_tree,
    read_pid_file,
    spawn_background,
)
from kln_knowledge.search import FORMATTERS, format_single_entry
from kln_knowledge.utils import is_kb_initialized, is_server_running

console = Console()


def _resolve_project(project: str | None = None) -> Path:
    """Resolve project root, exit with error if not found."""
    root = find_project_root(Path(project) if project else None)
    if not root:
        console.print("[red]No project root found.[/red]")
        console.print("Run from a directory with .git, .claude, or .knowledge-db")
        raise SystemExit(1)
    return root


@click.group()
@click.version_option(package_name="kln-knowledge-system")
def main():
    """KLN Knowledge System - Hybrid semantic knowledge database."""
    pass


# =============================================================================
# search
# =============================================================================


@main.command()
@click.argument("query", required=False, default="")
@click.option("--source", "-s", multiple=True, help="Sources to search (kb, sessions, docs)")
@click.option("--format", "-f", "fmt", type=click.Choice(["compact", "detailed", "inject", "json"]), default="detailed")
@click.option("--limit", "-n", type=int, default=5)
@click.option("--since", help="Entries from this date (YYYY-MM-DD)")
@click.option("--until", "until_date", help="Entries up to this date (YYYY-MM-DD)")
@click.option("--type", "entry_type", help="Filter by type (warning, solution, etc)")
@click.option("--branch", help="Filter by git branch")
@click.option("--min-score", type=float, default=0.0, help="Minimum relevance score")
@click.option("--id", "entry_id", help="Get specific entry by ID")
@click.option("--project", "-p", help="Project path")
def search(query, source, fmt, limit, since, until_date, entry_type, branch, min_score, entry_id, project):
    """Search the knowledge database."""
    root = _resolve_project(project)

    # Determine which sources to search
    sources = list(source) if source else ["kb"]

    if "kb" in sources or not source:
        from kln_knowledge.db import KnowledgeDB

        try:
            db = KnowledgeDB(str(root))
        except Exception as e:
            if fmt == "json":
                click.echo(json.dumps({"error": str(e), "results": []}))
            else:
                console.print(f"[red]ERROR: {e}[/red]")
            raise SystemExit(1)

        # Detail retrieval mode
        if entry_id:
            entry = db.get(entry_id)
            if not entry:
                for e in db._entries:
                    if e.get("id", "").startswith(entry_id):
                        entry = e
                        break
            if entry:
                if fmt == "json":
                    click.echo(json.dumps(entry, indent=2))
                else:
                    click.echo(format_single_entry(entry))
                return
            else:
                if fmt == "json":
                    click.echo(json.dumps({"error": f"Entry not found: {entry_id}"}))
                else:
                    console.print(f"[red]Entry not found: {entry_id}[/red]")
                raise SystemExit(1)

        if not query:
            console.print("[red]Query is required (or use --id for detail retrieval)[/red]")
            raise SystemExit(1)

        # Multi-source search
        all_results = []

        if "kb" in sources:
            results = db.search(
                query, limit=limit,
                date_from=since, date_to=until_date,
                entry_type=entry_type, branch=branch,
            )
            for r in results:
                r["_source"] = "kb"
            all_results.extend(results)

        if "sessions" in sources:
            try:
                sessions_db = KnowledgeDB(str(root), sub_store="sessions")
                results = sessions_db.search(query, limit=limit, date_from=since, date_to=until_date)
                for r in results:
                    r["_source"] = "sessions"
                all_results.extend(results)
            except Exception:
                pass

        if "docs" in sources:
            try:
                docs_db = KnowledgeDB(str(root), sub_store="docs")
                results = docs_db.search(query, limit=limit, date_from=since, date_to=until_date)
                for r in results:
                    r["_source"] = "docs"
                all_results.extend(results)
            except Exception:
                pass

        # Sort combined results by score, take top N
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        all_results = all_results[:limit]

        if min_score > 0:
            all_results = [r for r in all_results if r.get("score", 0) >= min_score]

        formatter = FORMATTERS[fmt]
        click.echo(formatter(all_results))
        raise SystemExit(0 if all_results else 1)


# =============================================================================
# capture
# =============================================================================


@main.command()
@click.argument("content", required=False, default="")
@click.option("--type", "entry_type", type=click.Choice(["finding", "solution", "pattern", "warning", "decision", "discovery"]), default="finding")
@click.option("--tags", default="", help="Comma-separated keywords")
@click.option("--priority", type=click.Choice(["low", "medium", "high", "critical"]), default="medium")
@click.option("--url", default="", help="Source URL")
@click.option("--json-input", help="Structured JSON entry")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--project", "-p", help="Project path")
def capture(content, entry_type, tags, priority, url, json_input, json_output, project):
    """Capture knowledge to the database."""
    if not content and not json_input:
        console.print("[red]Either content or --json-input is required[/red]")
        raise SystemExit(1)

    from kln_knowledge.capture import (
        create_entry,
        create_entry_from_json,
        log_to_timeline,
        save_entry,
    )

    root = _resolve_project(project)
    knowledge_dir = root / KB_DIR_NAME
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    if json_input:
        try:
            data = json.loads(json_input)
        except json.JSONDecodeError as e:
            if json_output:
                click.echo(json.dumps({"error": f"Invalid JSON: {e}"}))
            else:
                console.print(f"[red]Invalid JSON input: {e}[/red]")
            raise SystemExit(1)
        entry = create_entry_from_json(data)
    else:
        entry = create_entry(
            content=content,
            entry_type=entry_type,
            tags=tags,
            priority=priority,
            url=url if url else None,
        )

    save_entry(entry, knowledge_dir)
    log_to_timeline(entry.get("insight", content[:60]), entry.get("type", entry_type), knowledge_dir)

    if json_output:
        click.echo(json.dumps({
            "status": "success",
            "id": entry["id"],
            "title": entry["title"],
            "type": entry.get("type", entry_type),
        }))
    else:
        display = entry.get("title", "")[:60]
        console.print(f"[green]Captured {entry.get('type', entry_type)}: {display}[/green]")


# =============================================================================
# server
# =============================================================================


@main.group()
def server():
    """Manage the knowledge server."""
    pass


@server.command("start")
@click.option("--project", "-p", help="Project path")
def server_start(project):
    """Start the knowledge server."""
    root = _resolve_project(project)

    cleanup_stale_files(root)

    if is_server_running(str(root)):
        console.print(f"Server already running for {root}")
        return

    # Start inline (foreground)
    from kln_knowledge.server import KnowledgeServer
    srv = KnowledgeServer(str(root))
    srv.start()


@server.command("stop")
@click.option("--project", "-p", help="Project path")
def server_stop(project):
    """Stop the knowledge server."""
    root = _resolve_project(project)

    pid_file = get_kb_pid_file(root)
    port_file = get_kb_port_file(root)
    pid = read_pid_file(pid_file)

    if pid and is_process_running(pid):
        kill_process_tree(pid)
        console.print(f"Stopped server for {root} (PID {pid})")
        pid_file.unlink(missing_ok=True)
        port_file.unlink(missing_ok=True)
    else:
        console.print(f"No server running for {root}")
        cleanup_stale_files(root)


@server.command("status")
@click.option("--project", "-p", help="Project path")
def server_status(project):
    """Show server status."""
    from kln_knowledge.server import list_running_servers, send_command

    if not project:
        servers = list_running_servers()
        if servers:
            table = Table(title="Running Knowledge Servers")
            table.add_column("Project", style="cyan")
            table.add_column("Port")
            table.add_column("PID")
            table.add_column("Load Time")
            for s in servers:
                table.add_row(
                    s["project"],
                    str(s["port"]),
                    str(s["pid"]),
                    f"{s['load_time']:.1f}s",
                )
            console.print(table)
        else:
            console.print("No knowledge servers running")
        return

    root = _resolve_project(project)
    result = send_command(root, {"cmd": "status"})
    if result and "error" not in result:
        console.print(f"Status: {result.get('status', 'unknown')}")
        console.print(f"Project: {result.get('project', 'none')}")
        console.print(f"Port: {result.get('port', 'none')}")
        console.print(f"Entries: {result.get('entries', 0)}")
        console.print(f"Load time: {result.get('load_time', 0):.2f}s")
        console.print(f"Idle: {result.get('idle_seconds', 0)}s")
    else:
        console.print(f"Server not running for {root}")


# =============================================================================
# stats
# =============================================================================


@main.command()
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--project", "-p", help="Project path")
def stats(json_output, project):
    """Show database statistics."""
    from kln_knowledge.db import KnowledgeDB

    root = _resolve_project(project)

    try:
        db = KnowledgeDB(str(root))
    except Exception as e:
        console.print(f"[red]ERROR: {e}[/red]")
        raise SystemExit(1)

    st = db.stats()

    if json_output:
        click.echo(json.dumps(st, indent=2))
    else:
        console.print(f"Knowledge DB: {st['db_path']}")
        console.print(f"Backend: {st['backend']}")
        console.print(f"Entries: {st['count']}")
        console.print(f"Size: {st['size_human']}")
        console.print(f"Last updated: {st['last_updated']}")
        if st.get("sub_store"):
            console.print(f"Sub-store: {st['sub_store']}")


# =============================================================================
# rebuild
# =============================================================================


@main.command()
@click.option("--dense-only", is_flag=True, help="Skip sparse embeddings")
@click.option("--batch-size", type=int, default=50, help="Batch size for sparse embeddings")
@click.option("--project", "-p", help="Project path")
def rebuild(dense_only, batch_size, project):
    """Rebuild the search index from JSONL backup."""
    from kln_knowledge.db import KnowledgeDB

    root = _resolve_project(project)

    try:
        db = KnowledgeDB(str(root))
    except Exception as e:
        console.print(f"[red]ERROR: {e}[/red]")
        raise SystemExit(1)

    mode = "dense-only" if dense_only else "hybrid (dense+sparse)"
    console.print(f"Rebuilding index (mode: {mode})...")
    count = db.rebuild_index(dense_only=dense_only, batch_size=batch_size)
    console.print(f"Rebuilt index with {count} entries")


# =============================================================================
# ingest
# =============================================================================


@main.group()
def ingest():
    """Ingest content from external sources."""
    pass


@ingest.command("sessions")
@click.option("--full", is_flag=True, help="Process all sessions (not just new)")
@click.option("--project", "-p", help="Project path")
def ingest_sessions(full, project):
    """Ingest Claude Code session transcripts."""
    root = _resolve_project(project)

    from kln_knowledge.ingest_sessions import SessionIngester

    ingester = SessionIngester(str(root))
    count = ingester.ingest(full=full)
    console.print(f"Ingested {count} entries from sessions")


@ingest.command("docs")
@click.option("--path", "docs_path", help="Path to docs directory")
@click.option("--full", is_flag=True, help="Re-process all docs")
@click.option("--project", "-p", help="Project path")
def ingest_docs(docs_path, full, project):
    """Ingest markdown/PDF documentation."""
    root = _resolve_project(project)

    from kln_knowledge.ingest_docs import DocsIngester

    ingester = DocsIngester(str(root), docs_path=docs_path)
    count = ingester.ingest(full=full)
    console.print(f"Ingested {count} entries from docs")


@ingest.command("all")
@click.option("--full", is_flag=True, help="Re-process everything")
@click.option("--project", "-p", help="Project path")
def ingest_all(full, project):
    """Ingest from all sources."""
    root = _resolve_project(project)
    total = 0

    from kln_knowledge.ingest_sessions import SessionIngester

    try:
        ingester = SessionIngester(str(root))
        count = ingester.ingest(full=full)
        console.print(f"Sessions: {count} entries")
        total += count
    except Exception as e:
        console.print(f"[yellow]Sessions skipped: {e}[/yellow]")

    from kln_knowledge.ingest_docs import DocsIngester

    try:
        ingester = DocsIngester(str(root))
        count = ingester.ingest(full=full)
        console.print(f"Docs: {count} entries")
        total += count
    except Exception as e:
        console.print(f"[yellow]Docs skipped: {e}[/yellow]")

    console.print(f"Total: {total} entries ingested")


if __name__ == "__main__":
    main()
