"""KnowlinMCP -- Hybrid semantic knowledge database with multi-source search."""

from __future__ import annotations

__version__ = "0.1.0"

# Lazy imports to avoid loading fastembed on import
def __getattr__(name):
    if name == "KnowledgeDB":
        from knowlin_mcp.db import KnowledgeDB
        return KnowledgeDB
    if name == "MultiSourceSearch":
        from knowlin_mcp.multi_search import MultiSourceSearch
        return MultiSourceSearch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
