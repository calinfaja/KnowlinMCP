"""Multi-source search with weighted RRF fusion.

Searches across curated KB, sessions, and docs stores with
intent-adjusted weighting and unified result ranking.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from knowlin_mcp.db import KnowledgeDB
from knowlin_mcp.query_utils import (
    classify_query,
    expand_query,
    get_source_weights,
)
from knowlin_mcp.utils import debug_log


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class MultiSourceSearch:
    """Search across multiple knowledge sources with weighted RRF fusion.

    Sources:
    - kb: curated knowledge entries (default store)
    - sessions: ingested Claude Code session transcripts
    - docs: ingested documentation chunks
    """

    RRF_K = 60

    def __init__(self, project_path: str):
        self.project_path = project_path
        self._stores: dict[str, KnowledgeDB | None] = {}

    def _get_store(self, source: str) -> KnowledgeDB | None:
        """Lazy-load a KnowledgeDB for a source."""
        if source not in self._stores:
            try:
                sub_store = None if source == "kb" else source
                db = KnowledgeDB(self.project_path, sub_store=sub_store)
                # Always cache the db object - entries may be added later
                self._stores[source] = db if db.count() > 0 else db
            except Exception as e:
                debug_log(f"Failed to load {source} store: {e}")
                self._stores[source] = None
        store = self._stores.get(source)
        # Return None if store exists but is empty (no results to search)
        if store is not None and store.count() == 0:
            return None
        return store

    def search(
        self,
        query: str,
        sources: list[str] | None = None,
        limit: int = 5,
        date_from: str | None = None,
        date_to: str | None = None,
        entry_type: str | None = None,
        branch: str | None = None,
        auto_expand: bool = True,
    ) -> list[dict[str, Any]]:
        """Search across multiple sources with weighted RRF fusion.

        Args:
            query: Search query string
            sources: Which sources to search (default: all available)
            limit: Maximum results
            date_from: Filter by start date
            date_to: Filter by end date
            entry_type: Filter by entry type
            branch: Filter by git branch
            auto_expand: Whether to expand query with synonyms

        Returns:
            Unified results sorted by weighted RRF score
        """
        if sources is None:
            sources = ["kb", "sessions", "docs"]

        # Classify intent and get weights
        intent = classify_query(query)
        weights = get_source_weights(intent)

        # Optionally expand query
        search_query = expand_query(query) if auto_expand else query

        # Oversample from each source
        per_source_limit = limit * 3

        # Collect results from each source with source tag
        all_results: list[dict[str, Any]] = []

        # Build list of (source_name, store) for sources that have data
        active_stores = []
        for source in sources:
            store = self._get_store(source)
            if store is not None:
                active_stores.append((source, store))

        def _search_source(source: str, store: KnowledgeDB) -> list[dict[str, Any]]:
            results = store.search(
                search_query,
                limit=per_source_limit,
                rerank=False,
                date_from=date_from,
                date_to=date_to,
                entry_type=entry_type,
                branch=branch,
            )
            source_weight = weights.get(source, 1.0)
            for rank, result in enumerate(results, 1):
                result["_source"] = source
                base_score = 1.0 / (self.RRF_K + rank)
                result["_weighted_score"] = base_score * source_weight
            return results

        # Parallel search across sub-stores (GIL released during NumPy/ONNX)
        if len(active_stores) > 1:
            with ThreadPoolExecutor(max_workers=len(active_stores)) as executor:
                futures = {
                    executor.submit(_search_source, src, st): src
                    for src, st in active_stores
                }
                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        all_results.extend(future.result())
                    except Exception as e:
                        debug_log(f"Search failed for {source}: {e}")
        elif active_stores:
            src, st = active_stores[0]
            try:
                all_results.extend(_search_source(src, st))
            except Exception as e:
                debug_log(f"Search failed for {src}: {e}")

        if not all_results:
            return []

        # Sort by weighted score descending
        all_results.sort(key=lambda x: x.get("_weighted_score", 0), reverse=True)

        # Deduplicate by fuzzy title similarity (Jaccard on tokens)
        deduped = []
        seen_titles: list[set[str]] = []  # Token sets of accepted titles
        seen_ids: set[str] = set()
        for r in all_results:
            eid = r.get("id", "")
            if eid and eid in seen_ids:
                continue
            title = r.get("title", "").lower().strip()
            if title:
                tokens = set(title.split())
                if tokens and any(_jaccard(tokens, s) > 0.7 for s in seen_titles):
                    continue
                seen_titles.append(tokens)
            if eid:
                seen_ids.add(eid)
            deduped.append(r)

        # Normalize scores to 0-1 range
        if deduped:
            max_score = deduped[0].get("_weighted_score", 1.0)
            if max_score > 0:
                for r in deduped:
                    r["score"] = r.get("_weighted_score", 0) / max_score

        results = deduped[:limit]

        # Add search metadata
        for r in results:
            r.setdefault("_search_meta", {})
            r["_search_meta"]["intent"] = intent.value
            r["_search_meta"]["expanded_query"] = search_query != query

        return results

    def available_sources(self) -> list[str]:
        """Return list of sources that have data."""
        available = []
        for source in ["kb", "sessions", "docs"]:
            store = self._get_store(source)
            if store is not None:
                available.append(source)
        return available

    def stats(self) -> dict[str, Any]:
        """Return stats for all sources."""
        result = {}
        for source in ["kb", "sessions", "docs"]:
            store = self._get_store(source)
            if store is not None:
                st = store.stats()
                st["available"] = True
                result[source] = st
            else:
                result[source] = {"count": 0, "available": False}
        return result
