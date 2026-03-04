"""Multi-source search with weighted RRF fusion.

Searches across curated KB, sessions, and docs stores with
intent-adjusted weighting and unified result ranking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kln_knowledge.db import KnowledgeDB
from kln_knowledge.query_utils import (
    QueryIntent,
    classify_query,
    expand_query,
    get_source_weights,
)
from kln_knowledge.utils import debug_log


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
                if db.count() > 0:
                    self._stores[source] = db
                else:
                    self._stores[source] = None
            except Exception as e:
                debug_log(f"Failed to load {source} store: {e}")
                self._stores[source] = None
        return self._stores[source]

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

        for source in sources:
            store = self._get_store(source)
            if store is None:
                continue

            try:
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
                    # Weighted RRF: apply source weight to the rank-based score
                    base_score = 1.0 / (self.RRF_K + rank)
                    result["_weighted_score"] = base_score * source_weight
                    all_results.append(result)

            except Exception as e:
                debug_log(f"Search failed for {source}: {e}")

        if not all_results:
            return []

        # Sort by weighted score descending
        all_results.sort(key=lambda x: x.get("_weighted_score", 0), reverse=True)

        # Deduplicate by title similarity (exact match for now)
        seen_titles = set()
        deduped = []
        for r in all_results:
            title = r.get("title", "").lower().strip()
            if title and title in seen_titles:
                continue
            seen_titles.add(title)
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
