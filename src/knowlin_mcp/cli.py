"""KnowlinMCP CLI.

Entry point: knowlin
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import click
import psutil
from rich.console import Console
from rich.table import Table

from knowlin_mcp.platform import (
    KB_DIR_NAME,
    cleanup_stale_files,
    find_project_root,
    get_kb_pid_file,
    get_kb_port_file,
    is_process_running,
    kill_process_tree,
    read_pid_file,
)
from knowlin_mcp.search import FORMATTERS, format_single_entry
from knowlin_mcp.utils import debug_log, is_server_running

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
@click.version_option(package_name="knowlin-mcp")
def main():
    """KnowlinMCP -- Hybrid semantic knowledge database."""
    pass


# =============================================================================
# search
# =============================================================================


@main.command()
@click.argument("query", required=False, default="")
@click.option("--source", "-s", multiple=True, help="Sources to search (kb, sessions, docs)")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["compact", "detailed", "inject", "json"]),
    default="compact",
)
@click.option("--limit", "-n", type=int, default=5)
@click.option("--since", help="Entries from this date (YYYY-MM-DD)")
@click.option("--until", "until_date", help="Entries up to this date (YYYY-MM-DD)")
@click.option(
    "--type",
    "entry_type",
    type=click.Choice(
        [
            "finding",
            "solution",
            "pattern",
            "warning",
            "decision",
            "discovery",
            "document",
            "session",
        ],
        case_sensitive=False,
    ),
    help="Filter by entry type",
)
@click.option("--branch", help="Filter by git branch")
@click.option("--min-score", type=float, default=0.0, help="Minimum relevance score")
@click.option("--id", "entry_id", help="Get specific entry by ID")
@click.option("--project", "-p", help="Project path")
def search(
    query,
    source,
    fmt,
    limit,
    since,
    until_date,
    entry_type,
    branch,
    min_score,
    entry_id,
    project,
):
    """Search the knowledge database."""
    root = _resolve_project(project)

    sources = list(source) if source else None

    # Detail retrieval mode (single entry by ID)
    if entry_id:
        from knowlin_mcp.db import KnowledgeDB

        try:
            db = KnowledgeDB(str(root))
        except Exception as e:
            if fmt == "json":
                click.echo(json.dumps({"error": str(e)}))
            else:
                console.print(f"[red]ERROR: {e}[/red]")
            raise SystemExit(1)

        entry = db.get(entry_id)
        if not entry:
            for candidate in db._entries:
                if candidate and candidate.get("id", "").startswith(entry_id):
                    entry = candidate
                    break
        if entry:
            if fmt == "json":
                click.echo(json.dumps(entry, indent=2))
            else:
                click.echo(format_single_entry(entry))
            return
        if fmt == "json":
            click.echo(json.dumps({"error": f"Entry not found: {entry_id}"}))
        else:
            console.print(f"[red]Entry not found: {entry_id}[/red]")
        raise SystemExit(1)

    if not query:
        console.print("[red]Query is required (or use --id for detail retrieval)[/red]")
        raise SystemExit(1)

    # Use MultiSourceSearch for unified weighted RRF search
    from knowlin_mcp.multi_search import MultiSourceSearch

    ms = MultiSourceSearch(str(root))

    all_results = ms.search(
        query,
        sources=sources,
        limit=limit,
        date_from=since,
        date_to=until_date,
        entry_type=entry_type,
        branch=branch,
    )

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
@click.option(
    "--type",
    "entry_type",
    type=click.Choice(["finding", "solution", "pattern", "warning", "decision", "discovery"]),
    default="finding",
)
@click.option("--tags", default="", help="Comma-separated keywords")
@click.option(
    "--priority",
    type=click.Choice(["low", "medium", "high", "critical"]),
    default="medium",
)
@click.option("--url", default="", help="Source URL")
@click.option("--json-input", help="Structured JSON entry")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--project", "-p", help="Project path")
def capture(content, entry_type, tags, priority, url, json_input, json_output, project):
    """Capture knowledge to the database."""
    if not content and not json_input:
        console.print("[red]Either content or --json-input is required[/red]")
        raise SystemExit(1)

    from knowlin_mcp.capture import (
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

    saved = save_entry(entry, knowledge_dir)
    if not saved:
        if json_output:
            click.echo(json.dumps({"error": "Failed to save entry"}))
        else:
            console.print("[red]Failed to save entry[/red]")
        raise SystemExit(1)

    log_to_timeline(
        entry.get("insight", content[:60]),
        entry.get("type", entry_type),
        knowledge_dir,
    )

    if json_output:
        click.echo(
            json.dumps(
                {
                    "status": "success",
                    "id": entry["id"],
                    "title": entry["title"],
                    "type": entry.get("type", entry_type),
                }
            )
        )
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
    from knowlin_mcp.server import KnowledgeServer

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
        try:
            proc = psutil.Process(pid)
            cmdline = " ".join(proc.cmdline())
            if "knowlin" not in cmdline.lower():
                debug_log(f"PID {pid} is not a knowlin process: {cmdline}")
                pid_file.unlink(missing_ok=True)
                port_file.unlink(missing_ok=True)
                console.print(f"No server running for {root}")
                return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pid_file.unlink(missing_ok=True)
            port_file.unlink(missing_ok=True)
            console.print(f"No server running for {root}")
            return

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
    from knowlin_mcp.server import list_running_servers, send_command

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
    """Show database statistics for all sources."""
    from knowlin_mcp.multi_search import MultiSourceSearch

    root = _resolve_project(project)

    try:
        ms = MultiSourceSearch(str(root))
        all_stats = ms.stats()
    except Exception as e:
        console.print(f"[red]ERROR: {e}[/red]")
        raise SystemExit(1)

    if json_output:
        click.echo(json.dumps(all_stats, indent=2))
        return

    table = Table(title="Knowledge DB Statistics")
    table.add_column("Source", style="cyan")
    table.add_column("Entries", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Last Updated")

    total_count = 0
    for source in ["kb", "sessions", "docs"]:
        st = all_stats.get(source, {})
        count = st.get("count", 0)
        total_count += count
        if not st.get("available", count > 0):
            table.add_row(source, "[dim]--[/dim]", "[dim]--[/dim]", "[dim]not initialized[/dim]")
        else:
            table.add_row(
                source,
                str(count),
                st.get("size_human", "?"),
                st.get("last_updated", "?"),
            )

    console.print(table)
    console.print(f"\nTotal entries: {total_count}")
    console.print(f"DB path: {root / '.knowledge-db'}")


# =============================================================================
# export
# =============================================================================


@main.command()
@click.option("--source", "-s", multiple=True, help="Sources to export (kb, sessions, docs)")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["jsonl", "json"]),
    default="jsonl",
    help="Output format",
)
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path")
@click.option("--project", "-p", help="Project path")
def export(source, fmt, output, project):
    """Export knowledge entries to stdout or a file."""
    from knowlin_mcp.db import KnowledgeDB

    root = _resolve_project(project)
    sub_stores = (
        [(s if s != "kb" else None) for s in source] if source else [None, "sessions", "docs"]
    )

    entries = []
    for sub in sub_stores:
        try:
            db = KnowledgeDB(str(root), sub_store=sub)
            entries.extend(db._read_jsonl(migrate=True))
        except Exception:
            pass

    if fmt == "json":
        text = json.dumps(entries, indent=2)
    else:
        text = "\n".join(json.dumps(e) for e in entries)

    if output:
        Path(output).write_text(text + "\n")
        console.print(f"Exported {len(entries)} entries to {output}")
    else:
        click.echo(text)


# =============================================================================
# list
# =============================================================================


@main.command("list")
@click.option("--limit", "-n", default=10, help="Number of entries to show")
@click.option("--source", "-s", multiple=True, help="Source: kb, sessions, docs")
@click.option("--project", "-p", help="Project path")
def list_entries(limit, source, project):
    """List recent knowledge entries."""
    from knowlin_mcp.db import KnowledgeDB

    root = _resolve_project(project)
    if source:
        sources = [None if sub == "kb" else sub for sub in source]
    else:
        sources = [None, "sessions", "docs"]

    entries = []
    for sub in sources:
        label = sub or "kb"
        try:
            db = KnowledgeDB(str(root), sub_store=sub)
            for e in db.list_recent(limit=limit):
                e["_source"] = label
                entries.append(e)
        except Exception:
            pass

    # Sort by date descending, take top N
    entries.sort(
        key=lambda x: (x.get("date", "") or "", x.get("timestamp", "") or ""),
        reverse=True,
    )
    entries = entries[:limit]

    if not entries:
        console.print("[dim]No entries found.[/dim]")
        return

    for e in entries:
        src = e.get("_source", "?")
        title = e.get("title", "Untitled")
        date = (e.get("date") or "")[:10]
        etype = e.get("type", "")
        eid = e.get("id", "")[:12]
        console.print(f"  [{src}] {title}  [dim]{etype} | {date} | {eid}[/dim]")


# =============================================================================
# get
# =============================================================================


@main.command()
@click.argument("entry_id")
@click.option("--project", "-p", help="Project path")
def get(entry_id, project):
    """Get full details of a knowledge entry by ID."""
    from knowlin_mcp.db import KnowledgeDB

    root = _resolve_project(project)

    for sub in (None, "sessions", "docs"):
        try:
            db = KnowledgeDB(str(root), sub_store=sub)
            entry = db.get(entry_id)
            if entry:
                source = sub or "kb"
                console.print(f"\n[bold]{entry.get('title', 'Untitled')}[/bold]")
                console.print(
                    f"[dim]{source} | {entry.get('type', '')} | "
                    f"{(entry.get('date') or '')[:10]} | {entry.get('id', '')}[/dim]"
                )
                console.print()
                insight = entry.get("insight") or entry.get("summary") or ""
                if insight:
                    console.print(insight)
                kw = entry.get("keywords") or entry.get("tags")
                if kw:
                    console.print(f"\n[dim]Keywords: {', '.join(kw)}[/dim]")
                return
        except Exception:
            pass

    console.print(f"[red]Entry not found: {entry_id}[/red]")
    raise SystemExit(1)


# =============================================================================
# delete
# =============================================================================


@main.command()
@click.argument("entry_id")
@click.option("--source", "-s", default=None, help="Source: kb, sessions, docs")
@click.option("--project", "-p", help="Project path")
def delete(entry_id, source, project):
    """Delete a knowledge entry by ID."""
    from knowlin_mcp.db import KnowledgeDB

    root = _resolve_project(project)

    # If source specified, only look there
    stores = [None if source == "kb" else source] if source else [None, "sessions", "docs"]

    for sub in stores:
        try:
            db = KnowledgeDB(str(root), sub_store=sub)
            entry = db.get(entry_id)
            if entry:
                label = sub or "kb"
                title = entry.get("title", "Untitled")
                removed = db.remove_entries([entry_id])
                if removed:
                    console.print(f"Deleted from {label}: {title}")
                    return
        except Exception:
            pass

    console.print(f"[red]Entry not found: {entry_id}[/red]")
    raise SystemExit(1)


# =============================================================================
# rebuild
# =============================================================================


@main.command()
@click.option("--dense-only", is_flag=True, help="Skip sparse embeddings")
@click.option("--batch-size", type=int, default=50, help="Batch size for sparse embeddings")
@click.option("--project", "-p", help="Project path")
def rebuild(dense_only, batch_size, project):
    """Rebuild the search index from JSONL backup."""
    from knowlin_mcp.db import KnowledgeDB

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
# doctor
# =============================================================================


@main.command()
@click.option("--fix", is_flag=True, help="Auto-fix issues where possible")
@click.option("--project", "-p", help="Project path")
def doctor(fix, project):
    """Check knowledge system health."""
    root = _resolve_project(project)
    db_path = root / ".knowledge-db"
    issues = []
    ok_count = 0

    def ok(msg):
        nonlocal ok_count
        ok_count += 1
        console.print(f"  [green]OK[/green] {msg}")

    def warn(msg):
        issues.append(msg)
        console.print(f"  [yellow]WARN[/yellow] {msg}")

    def err(msg):
        issues.append(msg)
        console.print(f"  [red]FAIL[/red] {msg}")

    console.print("[bold]Knowledge DB Health Check[/bold]\n")

    # 1. DB directory exists
    if not db_path.exists():
        err("No .knowledge-db/ directory found")
        console.print(f"\n{ok_count} ok, {len(issues)} issues")
        raise SystemExit(1)
    ok(f".knowledge-db/ exists at {db_path}")

    # 2. Check each store
    import numpy as np

    for store_name, sub_store in [("kb", None), ("sessions", "sessions"), ("docs", "docs")]:
        store_path = db_path / sub_store if sub_store else db_path
        jsonl = store_path / "entries.jsonl"
        emb = store_path / "embeddings.npy"
        idx = store_path / "index.json"

        if not jsonl.exists():
            if sub_store:
                console.print(f"  [dim]--[/dim] {store_name}: not initialized")
            else:
                warn(f"{store_name}: no entries.jsonl")
            continue

        # Count JSONL entries
        jsonl_count = 0
        with open(jsonl) as f:
            for line in f:
                if line.strip():
                    jsonl_count += 1

        if not emb.exists():
            warn(f"{store_name}: entries.jsonl ({jsonl_count}) but no embeddings.npy")
            if fix:
                from knowlin_mcp.db import KnowledgeDB

                console.print(f"    [cyan]Rebuilding {store_name} index...[/cyan]")
                db = KnowledgeDB(str(root), sub_store=sub_store)
                db.rebuild_index()
                ok(f"{store_name}: rebuilt index")
            continue

        # Check embedding/JSONL alignment
        embeddings = np.load(str(emb))
        emb_count = len(embeddings)

        if not idx.exists():
            warn(f"{store_name}: embeddings but no index.json")
            continue

        with open(idx) as f:
            index_map = json.load(f)
        idx_count = len(index_map)

        if emb_count == idx_count == jsonl_count:
            ok(f"{store_name}: {jsonl_count} entries aligned")
        elif emb_count == idx_count and emb_count >= jsonl_count:
            # Zeroed embeddings from soft-deletes
            orphaned = emb_count - jsonl_count
            if orphaned > 0:
                warn(f"{store_name}: {orphaned} orphaned embeddings (run rebuild to compact)")
                if fix:
                    from knowlin_mcp.db import KnowledgeDB

                    console.print(f"    [cyan]Compacting {store_name}...[/cyan]")
                    db = KnowledgeDB(str(root), sub_store=sub_store)
                    db.rebuild_index()
                    ok(f"{store_name}: compacted")
            else:
                ok(f"{store_name}: {jsonl_count} entries aligned")
        else:
            err(
                f"{store_name}: misaligned - "
                f"JSONL={jsonl_count}, embeddings={emb_count}, index={idx_count}"
            )
            if fix:
                from knowlin_mcp.db import KnowledgeDB

                console.print(f"    [cyan]Rebuilding {store_name}...[/cyan]")
                db = KnowledgeDB(str(root), sub_store=sub_store)
                db.rebuild_index()
                ok(f"{store_name}: rebuilt")

    # 3. Check registries
    for name, reg_path in [
        ("session-registry", db_path / "session-registry.json"),
        ("doc-registry", db_path / "doc-registry.json"),
    ]:
        if reg_path.exists():
            try:
                reg = json.loads(reg_path.read_text())
                ok(f"{name}: {len(reg)} tracked files")
            except json.JSONDecodeError:
                err(f"{name}: corrupted JSON")
                if fix:
                    reg_path.unlink()
                    ok(f"{name}: removed corrupted file")

    # 4. Check sources config
    sources_path = db_path / "sources.yaml"
    if sources_path.exists():
        try:
            import yaml

            config = yaml.safe_load(sources_path.read_text())
            if config:
                ok(f"sources.yaml: valid ({len(config)} sections)")
                # Verify configured paths exist
                docs_config = config.get("docs", {})
                for p in docs_config.get("paths", []):
                    resolved = Path(p).expanduser()
                    if not resolved.is_absolute():
                        resolved = (root / resolved).resolve()
                    if not resolved.exists():
                        warn(f"configured doc path missing: {p}")
        except ImportError:
            warn("pyyaml not installed (sources.yaml not readable)")
        except Exception as e:
            err(f"sources.yaml: {e}")
    else:
        console.print("  [dim]--[/dim] No sources.yaml (using convention discovery)")

    # Summary
    console.print(f"\n{ok_count} ok, {len(issues)} issues")
    if issues:
        raise SystemExit(1)


# =============================================================================
# ingest
# =============================================================================


@main.group()
def ingest():
    """Ingest content from external sources."""
    from knowlin_mcp.models import models_cached

    if not models_cached():
        console.print(
            "[yellow]First run: embedding models (~200MB) will be downloaded. "
            "This may take a few minutes...[/yellow]"
        )


@ingest.command("sessions")
@click.option("--full", is_flag=True, help="Process all sessions (not just new)")
@click.option("--project", "-p", help="Project path")
def ingest_sessions(full, project):
    """Ingest Claude Code session transcripts."""
    root = _resolve_project(project)

    from knowlin_mcp.ingest_sessions import SessionIngester

    with console.status("[bold]Ingesting sessions...[/bold]"):
        ingester = SessionIngester(str(root))
        count = ingester.ingest(full=full)
    console.print(f"Ingested {count} entries from sessions")


@ingest.command("codex")
@click.option("--full", is_flag=True, help="Process all Codex sessions (not just new)")
@click.option("--project", "-p", help="Project path")
def ingest_codex(full, project):
    """Ingest Codex CLI session transcripts."""
    root = _resolve_project(project)

    from knowlin_mcp.ingest_codex import CodexIngester

    with console.status("[bold]Ingesting Codex sessions...[/bold]"):
        ingester = CodexIngester(str(root))
        count = ingester.ingest(full=full)
    console.print(f"Ingested {count} entries from Codex sessions")


@ingest.command("docs")
@click.option("--path", "docs_path", help="Path to docs directory")
@click.option("--full", is_flag=True, help="Re-process all docs")
@click.option("--project", "-p", help="Project path")
def ingest_docs(docs_path, full, project):
    """Ingest markdown/PDF documentation."""
    root = _resolve_project(project)

    from knowlin_mcp.ingest_docs import DocsIngester

    with console.status("[bold]Ingesting docs...[/bold]"):
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

    sources = [
        ("Sessions", "knowlin_mcp.ingest_sessions", "SessionIngester"),
        ("Codex", "knowlin_mcp.ingest_codex", "CodexIngester"),
        ("Docs", "knowlin_mcp.ingest_docs", "DocsIngester"),
    ]

    for label, module_name, class_name in sources:
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
            with console.status(f"[bold]Ingesting {label.lower()}...[/bold]"):
                ingester = cls(str(root))
                count = ingester.ingest(full=full)
            console.print(f"  {label}: {count} entries")
            total += count
        except Exception as e:
            console.print(f"  [yellow]{label} skipped: {e}[/yellow]")

    console.print(f"\nTotal: {total} entries ingested")


# =============================================================================
# sources
# =============================================================================


_SOURCES_TEMPLATE = """\
# KnowlinMCP - Source Configuration
# Declares which documents and sessions to index.
# Paths are relative to the project root unless absolute.

docs:
  paths:
    - docs/                    # relative to project root
    # - ~/Desktop/INFOS/       # absolute path (~ expanded)
    # - /shared/team-docs/     # another absolute path
  # include: ["*.md", "*.txt", "*.pdf", "*.rst"]  # default if omitted
  # exclude: ["drafts/**", "*.tmp"]

sessions:
  auto_discover: true          # scan ~/.claude/projects/ automatically
  # path: ~/custom/sessions/  # explicit override (disables auto-discover)
"""


@main.command()
@click.option("--init", "do_init", is_flag=True, help="Create a template sources.yaml")
@click.option("--project", "-p", help="Project path")
def sources(do_init, project):
    """Show or initialize source configuration."""
    root = _resolve_project(project)
    config_path = root / ".knowledge-db" / "sources.yaml"

    if do_init:
        if config_path.exists():
            console.print(f"[yellow]Already exists: {config_path}[/yellow]")
            console.print("Edit it directly to change sources.")
            return
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(_SOURCES_TEMPLATE)
        console.print(f"Created {config_path}")
        console.print("Edit it to configure your document sources.")
        return

    # Show current config
    if config_path.exists():
        console.print(f"[bold]Sources config:[/bold] {config_path}\n")
        click.echo(config_path.read_text())
    else:
        console.print("[dim]No sources.yaml found (using convention-based discovery)[/dim]")
        console.print("Run [bold]knowlin sources --init[/bold] to create one.")

        # Show what convention-based discovery finds
        from knowlin_mcp.ingest_docs import DocsIngester

        ingester = DocsIngester.__new__(DocsIngester)
        ingester.project_path = root
        ingester._sources_config = None
        ingester._include_globs = None
        ingester._exclude_globs = []
        dirs = ingester._find_docs_dirs()
        if dirs:
            console.print("\n[bold]Auto-discovered doc dirs:[/bold]")
            for d in dirs:
                console.print(f"  {d}")
        else:
            console.print(
                "\n[dim]No doc directories found (docs/, doc/, INFOS/, documentation/)[/dim]"
            )


# =============================================================================
# init
# =============================================================================


def _build_mcp_json() -> str:
    """Build .mcp.json with the absolute path to knowlin-mcp."""
    import shutil
    import sysconfig

    cmd = shutil.which("knowlin-mcp")
    if cmd is None:
        # Fallback: resolve from the Python environment's scripts dir
        scripts_dir = Path(sysconfig.get_path("scripts"))
        candidate = scripts_dir / "knowlin-mcp"
        cmd = str(candidate) if candidate.exists() else "knowlin-mcp"

    config = {"mcpServers": {"knowlin-mcp": {"command": cmd}}}
    return json.dumps(config, indent=2) + "\n"


@main.command()
@click.option("--mcp/--no-mcp", default=True, help="Configure MCP server for Claude Code")
@click.argument("path", required=False, default=".")
def init(mcp, path):
    """Initialize KnowlinMCP in a project directory.

    Creates .knowledge-db/ with sources.yaml template.
    Optionally writes .mcp.json for Claude Code integration.
    """
    root = Path(path).resolve()

    if not root.is_dir():
        console.print(f"[red]Not a directory: {root}[/red]")
        raise SystemExit(1)

    kb_dir = root / ".knowledge-db"
    kb_dir.mkdir(parents=True, exist_ok=True)

    # sources.yaml
    sources_path = kb_dir / "sources.yaml"
    if sources_path.exists():
        console.print("  sources.yaml already exists")
    else:
        sources_path.write_text(_SOURCES_TEMPLATE)
        console.print(f"  Created {sources_path.relative_to(root)}")

    # .mcp.json for Claude Code / MCP clients
    if mcp:
        mcp_path = root / ".mcp.json"
        if mcp_path.exists():
            console.print("  .mcp.json already exists")
        else:
            mcp_json = _build_mcp_json()
            mcp_path.write_text(mcp_json)
            console.print("  Created .mcp.json (MCP server config)")

    console.print()
    console.print("[bold]KnowlinMCP initialized.[/bold]")
    console.print()
    console.print("Next steps:")
    console.print("  1. Edit .knowledge-db/sources.yaml to add your doc paths")
    console.print("  2. Run [bold]knowlin ingest all[/bold] to index your docs")
    console.print('  3. Run [bold]knowlin search "your query"[/bold] to search')


if __name__ == "__main__":
    main()
