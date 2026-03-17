"""KnowlinMCP -- MCP server exposing hybrid knowledge search to any MCP client."""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from knowlin_mcp.platform import find_project_root
from knowlin_mcp.utils import logger

mcp = FastMCP(
    "knowlin-mcp",
    instructions=(
        "Per-project knowledge database with hybrid semantic search. "
        "Results are scoped to the current project's .knowledge-db/ directory.\n\n"
        "WHEN TO USE:\n"
        "- Before answering questions about the codebase or past decisions: "
        "call knowlin_search for relevant prior context\n"
        "- When you discover useful insights, solutions, or patterns: "
        "call knowlin_capture to persist them for future retrieval\n\n"
        "TIPS: Use specific queries over broad ones. Filter by source "
        "(kb, sessions, docs) when you know where the answer likely lives."
    ),
)

_project_root: str | None = None

_READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
_WRITE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
)


def _get_project_root() -> str:
    """Resolve and cache the project root path."""
    global _project_root
    if _project_root is None:
        root = find_project_root()
        if root is None:
            raise RuntimeError(
                "No project root found. Ensure .git or .knowledge-db exists "
                "in a parent directory, or set CLAUDE_PROJECT_DIR."
            )
        _project_root = str(root)
    return _project_root


def _parse_sources(sources: str) -> list[str]:
    """Parse comma-separated source string into list."""
    if not sources or sources.strip().lower() == "all":
        return ["kb", "sessions", "docs"]
    return [s.strip().lower() for s in sources.split(",") if s.strip()]


@mcp.tool(title="Search Knowledge Base", annotations=_READ_ONLY)
def knowlin_search(
    query: str,
    limit: int = 5,
    sources: str = "all",
    since: str = "",
    until: str = "",
    entry_type: str = "",
) -> str:
    """Search the knowledge database with hybrid semantic + keyword matching.

    Args:
        query: Natural language search query
        limit: Maximum results (1-20, default 5)
        sources: Comma-separated sources: kb, sessions, docs, or "all" (default)
        since: Filter results after this date (YYYY-MM-DD)
        until: Filter results before this date (YYYY-MM-DD)
        entry_type: Filter by type (finding, solution, pattern, warning, decision, discovery)
    """
    try:
        from knowlin_mcp.multi_search import MultiSourceSearch

        root = _get_project_root()
        limit = max(1, min(20, limit))
        source_list = _parse_sources(sources)
        ms = MultiSourceSearch(root)

        results = ms.search(
            query,
            sources=source_list,
            limit=limit,
            date_from=since or None,
            date_to=until or None,
            entry_type=entry_type or None,
        )

        if not results:
            return f"No results found for: {query}"

        lines = [f"Found {len(results)} result(s) from {', '.join(source_list)}:", ""]

        for r in results:
            title = r.get("title", "Untitled")
            score = r.get("score", 0)
            source = r.get("_source", "?")
            etype = r.get("type", "")
            date = (r.get("date") or r.get("found_date") or "")[:10]
            entry_id = r.get("id", "")
            insight = r.get("insight") or r.get("summary") or ""
            if len(insight) > 300:
                insight = insight[:297] + "..."

            lines.append(f"### {title} ({score:.0%})")
            meta = f"[{source}]"
            if etype:
                meta += f" {etype}"
            if date:
                meta += f" | {date}"
            lines.append(meta)
            if insight:
                lines.append(insight)
            if entry_id:
                lines.append(f"ID: {entry_id}")
            lines.append("")

        return "\n".join(lines)

    except Exception:
        logger.exception("[mcp] Search failed")
        raise ToolError("Operation failed")


@mcp.tool(title="Get Knowledge Entry", annotations=_READ_ONLY)
def knowlin_get(entry_id: str) -> str:
    """Retrieve the full details of a knowledge entry by its ID.

    Args:
        entry_id: The entry ID (returned by knowlin_search)
    """
    try:
        from knowlin_mcp.db import KnowledgeDB

        root = _get_project_root()

        # Try all sub-stores
        for sub in (None, "sessions", "docs"):
            db = KnowledgeDB(root, sub_store=sub)
            entry = db.get(entry_id)
            if entry:
                return _format_full_entry(entry, sub or "kb")

        return f"Entry not found: {entry_id}"

    except Exception:
        logger.exception("[mcp] Get failed")
        raise ToolError("Operation failed")


def _format_full_entry(entry: dict, source: str) -> str:
    """Format a full entry as Markdown."""
    lines = [f"# {entry.get('title', 'Untitled')}", ""]

    meta_parts = [f"Source: {source}"]
    if entry.get("type"):
        meta_parts.append(f"Type: {entry['type']}")
    date = (entry.get("date") or entry.get("found_date") or "")[:10]
    if date:
        meta_parts.append(f"Date: {date}")
    if entry.get("id"):
        meta_parts.append(f"ID: {entry['id']}")
    lines.append(" | ".join(meta_parts))
    lines.append("")

    for field, label in [
        ("insight", "Insight"),
        ("summary", "Summary"),
        ("problem_solved", "Problem"),
        ("what_worked", "Solution"),
        ("why_it_matters", "Why it matters"),
        ("context", "Context"),
        ("content", "Content"),
    ]:
        val = entry.get(field)
        if val:
            lines.append(f"**{label}:** {val}")
            lines.append("")

    kw = entry.get("keywords") or entry.get("tags")
    if kw:
        lines.append(f"**Keywords:** {', '.join(kw)}")

    if entry.get("url"):
        lines.append(f"**URL:** {entry['url']}")

    return "\n".join(lines)


@mcp.tool(title="Knowledge DB Statistics", annotations=_READ_ONLY)
def knowlin_stats() -> str:
    """Show knowledge database statistics (entry counts, sizes, health)."""
    try:
        from knowlin_mcp.multi_search import MultiSourceSearch

        root = _get_project_root()
        ms = MultiSourceSearch(root)
        all_stats = ms.stats()

        lines = ["# Knowledge DB Stats", ""]
        lines.append("| Source | Entries | Size | Last Updated | Status |")
        lines.append("|--------|---------|------|--------------|--------|")

        total = 0
        for source in ("kb", "sessions", "docs"):
            s = all_stats.get(source, {})
            count = s.get("count", 0)
            total += count
            size = s.get("size_human", "0 KB")
            updated = (s.get("last_updated") or "never")[:10]
            status = "ok" if s.get("available") else "empty"
            lines.append(f"| {source} | {count} | {size} | {updated} | {status} |")

        lines.append(f"\n**Total entries:** {total}")
        lines.append(f"**Project:** {root}")

        return "\n".join(lines)

    except Exception:
        logger.exception("[mcp] Stats failed")
        raise ToolError("Operation failed")


@mcp.tool(title="Ingest Documents & Sessions", annotations=_WRITE)
def knowlin_ingest(source: str = "all") -> str:
    """Ingest documents and/or session transcripts into the knowledge database.

    Args:
        source: What to ingest: "docs", "sessions", "codex", or "all" (default)
    """
    try:
        root = _get_project_root()
        source = source.strip().lower()
        if source not in ("docs", "sessions", "codex", "all"):
            return f"Invalid source: {source}. Use 'docs', 'sessions', 'codex', or 'all'."

        results = []

        if source in ("docs", "all"):
            from knowlin_mcp.ingest_docs import DocsIngester

            count = DocsIngester(root).ingest()
            results.append(f"Docs: {count} entries ingested")

        if source in ("sessions", "all"):
            from knowlin_mcp.ingest_sessions import SessionIngester

            count = SessionIngester(root).ingest()
            results.append(f"Sessions: {count} entries ingested")

        if source in ("codex", "all"):
            from knowlin_mcp.ingest_codex import CodexIngester

            count = CodexIngester(root).ingest()
            results.append(f"Codex: {count} entries ingested")

        return "\n".join(results)

    except Exception:
        logger.exception("[mcp] Ingest failed")
        raise ToolError("Operation failed")


@mcp.tool(title="Capture Knowledge Entry", annotations=_WRITE)
def knowlin_capture(
    title: str,
    insight: str,
    entry_type: str = "finding",
    keywords: str = "",
    priority: str = "medium",
) -> str:
    """Save a knowledge entry to the database.

    Use this to persist insights, solutions, patterns, or decisions discovered
    during a session so they can be retrieved later.

    Args:
        title: Short descriptive title (5+ chars, 2+ words)
        insight: The main content/insight to capture
        entry_type: finding, solution, pattern, warning, decision, or discovery
        keywords: Comma-separated keywords for search (e.g. "auth,jwt,security")
        priority: low, medium, high, or critical
    """
    try:
        from knowlin_mcp.capture import create_entry_from_json, save_entry

        root = _get_project_root()

        valid_types = {"finding", "solution", "pattern", "warning", "decision", "discovery"}
        if entry_type not in valid_types:
            return f"Invalid type: {entry_type}. Use one of: {', '.join(sorted(valid_types))}"

        valid_priorities = {"low", "medium", "high", "critical"}
        if priority not in valid_priorities:
            return f"Invalid priority: {priority}. Use: {', '.join(sorted(valid_priorities))}"

        kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []

        entry = create_entry_from_json(
            {
                "title": title,
                "insight": insight,
                "type": entry_type,
                "keywords": kw_list,
                "priority": priority,
            }
        )

        kb_dir = Path(find_project_root() or root) / ".knowledge-db"
        saved = save_entry(entry, kb_dir)

        if saved:
            return f"Saved: {entry.get('title', title)} (ID: {entry.get('id', '?')})"
        return "Failed to save entry. Check that .knowledge-db/ exists."

    except Exception:
        logger.exception("[mcp] Capture failed")
        raise ToolError("Operation failed")


def main():
    """Entry point for knowlin-mcp command."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
