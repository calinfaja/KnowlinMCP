"""Knowledge Search - Output formatters for search results."""

from __future__ import annotations

import json


def format_compact(results: list[dict]) -> str:
    """Compact format for quick overview."""
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        entry_type = r.get("type", "")
        source = r.get("_source", "")
        date = (r.get("date") or r.get("found_date", ""))[:10]
        entry_id = r.get("id", "")[:12]
        score = r.get("score", 0)
        insight = r.get("insight") or r.get("summary") or ""
        if len(insight) > 80:
            insight = insight[:77].rsplit(" ", 1)[0] + "..."
        tag = f"{entry_type}" if not source else f"{source}:{entry_type}"
        lines.append(f"{i}. [{tag}] {title} ({score:.0%}, {date}) [id:{entry_id}]")
        if insight:
            lines.append(f"   {insight}")
    return "\n".join(lines)


def format_detailed(results: list[dict]) -> str:
    """Detailed format with all metadata."""
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        title = r.get("title", "Untitled")
        entry_id = r.get("id", "")
        source_label = r.get("_source", "")
        lines.append(f"\n{'=' * 60}")
        header = f"[{i}] {title} ({score:.0%})"
        if source_label:
            header += f" [{source_label}]"
        lines.append(header)
        lines.append(f"{'=' * 60}")
        lines.append(f"ID: {entry_id}")

        if r.get("type"):
            lines.append(f"Type: {r['type']}")
        if r.get("priority"):
            lines.append(f"Priority: {r['priority']}")
        if r.get("date") or r.get("found_date"):
            lines.append(f"Date: {(r.get('date') or r.get('found_date', ''))[:10]}")
        if r.get("source"):
            lines.append(f"Source: {r['source']}")
        if r.get("branch"):
            lines.append(f"Branch: {r['branch']}")

        insight = r.get("insight") or r.get("summary")
        if insight:
            lines.append(f"\nInsight: {insight}")
        if r.get("keywords") or r.get("tags"):
            kw = r.get("keywords") or r.get("tags", [])
            if isinstance(kw, list):
                lines.append(f"Keywords: {', '.join(kw)}")
        if r.get("pinned"):
            lines.append("Pinned: yes")

    return "\n".join(lines)


def format_inject(results: list[dict]) -> str:
    """Format optimized for LLM prompt injection."""
    if not results:
        return "No relevant prior knowledge found."

    lines = ["RELEVANT PRIOR KNOWLEDGE:", ""]

    for r in results:
        title = r.get("title", "Untitled")
        score = r.get("score", 0)
        lines.append(f"### {title} (relevance: {score:.0%})")

        insight = r.get("insight") or r.get("summary") or ""
        if insight:
            lines.append(insight)

        etype = r.get("type", "")
        source = r.get("_source") or r.get("source", "")
        if etype or source:
            meta = " | ".join(filter(None, [etype, source]))
            lines.append(f"[{meta}]")

        lines.append("")

    return "\n".join(lines)


def format_json(results: list[dict]) -> str:
    """JSON format for programmatic use."""
    return json.dumps(results, indent=2)


def format_single_entry(entry: dict) -> str:
    """Format a single entry for detail view."""
    lines = [f"{'=' * 60}"]
    lines.append(f"Title: {entry.get('title', 'Untitled')}")
    lines.append(f"ID: {entry.get('id', '')}")
    lines.append(f"Type: {entry.get('type', 'finding')}")
    lines.append(f"Priority: {entry.get('priority', 'medium')}")
    if entry.get("date"):
        lines.append(f"Date: {entry['date'][:10]}")
    if entry.get("source"):
        lines.append(f"Source: {entry['source']}")
    if entry.get("branch"):
        lines.append(f"Branch: {entry['branch']}")
    if entry.get("pinned"):
        lines.append("Pinned: yes")

    insight = entry.get("insight") or entry.get("summary")
    if insight:
        lines.append(f"\nInsight:\n{insight}")

    kw = entry.get("keywords") or entry.get("tags", [])
    if isinstance(kw, list) and kw:
        lines.append(f"\nKeywords: {', '.join(kw)}")

    related = entry.get("related_to", [])
    if related:
        lines.append(f"Related: {', '.join(related)}")

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


FORMATTERS = {
    "compact": format_compact,
    "detailed": format_detailed,
    "inject": format_inject,
    "json": format_json,
}
