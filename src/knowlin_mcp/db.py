"""Knowledge Database - Hybrid search with fastembed (dense + sparse + reranking).

Storage:
- entries.jsonl: Source of truth (human-readable)
- embeddings.npy: Dense vectors (384-dim BGE)
- sparse_index.json: Sparse vectors (SPLADE++ learned token weights)
- index.json: ID to row mapping

Search pipeline:
1. Dense search (semantic similarity via BGE)
2. Sparse search (keyword matching via SPLADE++)
3. RRF fusion of both result sets
4. Optional cross-encoder reranking
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from knowlin_mcp.platform import find_project_root
from knowlin_mcp.utils import debug_log, migrate_entry

# Fastembed imports
try:
    from fastembed import TextEmbedding
except ImportError:
    raise ImportError("fastembed not installed. Run: pip install fastembed")

# Optional sparse/rerank (lazy loaded)
SparseTextEmbedding = None
TextCrossEncoder = None

# Global model singletons
_dense_model: Optional[TextEmbedding] = None
_sparse_model = None
_reranker = None
_first_run_warned = False

_MODEL_NAMES = [
    "BAAI/bge-small-en-v1.5",
    "prithivida/Splade_PP_en_v1",
    "Xenova/ms-marco-MiniLM-L-6-v2",
]


def _warn_if_first_run() -> None:
    """Print a one-time warning if embedding models need to be downloaded."""
    global _first_run_warned
    if _first_run_warned:
        return
    _first_run_warned = True

    import os
    import sys

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "fastembed")
    if os.path.isdir(cache_dir) and os.listdir(cache_dir):
        return  # Models already cached

    print(
        "First run: downloading embedding models (~200MB). This may take a few minutes...",
        file=sys.stderr,
        flush=True,
    )


def get_dense_model() -> TextEmbedding:
    """Get or create singleton dense embedding model (BGE-small, 384-dim)."""
    global _dense_model
    if _dense_model is None:
        _warn_if_first_run()
        _dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _dense_model


def get_sparse_model():
    """Get or create singleton sparse embedding model (SPLADE++)."""
    global _sparse_model, SparseTextEmbedding
    if _sparse_model is None:
        if SparseTextEmbedding is None:
            try:
                from fastembed import SparseTextEmbedding as STE

                SparseTextEmbedding = STE
            except ImportError:
                debug_log("SparseTextEmbedding not available, falling back to dense-only")
                return None
        _sparse_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
    return _sparse_model


def get_reranker():
    """Get or create singleton cross-encoder reranker."""
    global _reranker, TextCrossEncoder
    if _reranker is None:
        if TextCrossEncoder is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder as TCE

                TextCrossEncoder = TCE
            except ImportError:
                debug_log("Cross-encoder reranker not available")
                return None
        _reranker = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    return _reranker


class KnowledgeDB:
    """Hybrid knowledge database using fastembed (dense + sparse + reranking).

    Supports a sub_store parameter to maintain separate JSONL + embeddings
    within the same .knowledge-db/ directory (e.g., for sessions, docs).
    """

    RRF_K = 60
    MAX_PINNED = 15

    def __init__(self, project_path: str | None = None, sub_store: str | None = None):
        """Initialize KnowledgeDB.

        Args:
            project_path: Path to project root. If None, auto-detects.
            sub_store: Optional sub-store name (e.g., "sessions", "docs").
                       Uses separate files within .knowledge-db/{sub_store}/.
        """
        root: Path | None
        if project_path:
            root = Path(project_path).resolve()
        else:
            root = find_project_root()

        if not root:
            raise ValueError(
                "Could not find project root. "
                "Make sure you're in a directory with .serena, .claude, or .knowledge-db"
            )
        self.project_root: Path = root

        self.sub_store = sub_store
        self.db_path = self.project_root / ".knowledge-db"

        if sub_store:
            store_path = (self.db_path / sub_store).resolve()
            if not str(store_path).startswith(str(self.db_path.resolve())):
                raise ValueError(f"sub_store '{sub_store}' escapes db_path")
            store_path.mkdir(parents=True, exist_ok=True)
            self.embeddings_path = store_path / "embeddings.npy"
            self.sparse_index_path = store_path / "sparse_index.json"
            self.index_path = store_path / "index.json"
            self.jsonl_path = store_path / "entries.jsonl"
        else:
            self.embeddings_path = self.db_path / "embeddings.npy"
            self.sparse_index_path = self.db_path / "sparse_index.json"
            self.index_path = self.db_path / "index.json"
            self.jsonl_path = self.db_path / "entries.jsonl"

        self.old_txtai_path = self.db_path / "index"

        # Create directory if needed
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Models (lazy load)
        self._dense_model: Optional[TextEmbedding] = None
        self._sparse_model = None
        self._reranker = None

        # In-memory index
        self._embeddings: Optional[np.ndarray] = None
        self._id_to_row: dict[str, int] = {}
        self._row_to_id: dict[int, str] = {}
        self._entries: list[dict[str, Any]] = []
        self._sparse_vectors: dict[int, dict[str, float]] = {}

        self._load_index()

        if not sub_store and self._needs_migration():
            debug_log("Detected old txtai index, migrating...")
            self.rebuild_index()

    @property
    def dense_model(self) -> TextEmbedding:
        if self._dense_model is None:
            self._dense_model = get_dense_model()
        return self._dense_model

    @property
    def sparse_model(self):
        if self._sparse_model is None:
            self._sparse_model = get_sparse_model()
        return self._sparse_model

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    def _needs_migration(self) -> bool:
        has_old = self.old_txtai_path.exists()
        has_new = self.embeddings_path.exists()
        has_entries = self.jsonl_path.exists()
        return has_old and not has_new and has_entries

    def _load_index(self) -> None:
        """Load embeddings, sparse vectors, index, and entries from disk."""
        if self.embeddings_path.exists() and self.index_path.exists():
            try:
                self._embeddings = np.load(str(self.embeddings_path))
                with open(self.index_path) as f:
                    self._id_to_row = json.load(f)
                self._row_to_id = {v: k for k, v in self._id_to_row.items()}

                self._sparse_vectors = {}
                if self.sparse_index_path.exists():
                    with open(self.sparse_index_path) as f:
                        sparse_data = json.load(f)
                        self._sparse_vectors = {int(k): v for k, v in sparse_data.items()}
                    debug_log(f"Loaded {len(self._sparse_vectors)} sparse vectors")

                all_entries = []
                if self.jsonl_path.exists():
                    with open(self.jsonl_path) as f:
                        for line in f:
                            if line.strip():
                                try:
                                    e = json.loads(line)
                                    if isinstance(e, dict):
                                        all_entries.append(migrate_entry(e))
                                except json.JSONDecodeError:
                                    pass

                indexed_ids = set(self._id_to_row.keys())
                entries_by_id = {}
                unindexed = []
                for e in all_entries:
                    eid = e.get("id", "")
                    if eid in indexed_ids:
                        entries_by_id[eid] = e
                    elif eid:
                        unindexed.append(e)

                self._entries = [None] * len(self._embeddings)  # type: ignore[list-item]
                for eid, row_idx in self._id_to_row.items():
                    if eid in entries_by_id and row_idx < len(self._entries):
                        self._entries[row_idx] = entries_by_id[eid]

                for i in range(len(self._entries)):
                    if self._entries[i] is None:
                        self._entries[i] = {"id": self._row_to_id.get(i, ""), "title": "?"}

                if unindexed:
                    debug_log(f"Indexing {len(unindexed)} unindexed entries from JSONL...")
                    for entry in unindexed:
                        searchable = self._build_searchable_text(entry)
                        embedding = list(self.dense_model.embed([searchable]))[0]
                        self._embeddings = np.vstack([self._embeddings, embedding])

                        row_idx = len(self._id_to_row)
                        entry_id = entry.get("id") or str(uuid.uuid4())
                        entry["id"] = entry_id
                        self._id_to_row[entry_id] = row_idx
                        self._row_to_id[row_idx] = entry_id
                        self._entries.append(entry)

                        sparse_vec = self._generate_sparse_embedding(searchable)
                        if sparse_vec:
                            self._sparse_vectors[row_idx] = sparse_vec

                    self._save_index()
                    debug_log(f"Repaired index: {len(self._id_to_row)} entries now indexed")
                else:
                    debug_log(f"Loaded {len(self._id_to_row)} embeddings from disk")
            except Exception as e:
                debug_log(f"Failed to load index: {e}")
                self._embeddings = None
                self._id_to_row = {}
                self._row_to_id = {}
                self._entries = []
                self._sparse_vectors = {}

    def _save_index(self) -> None:
        """Save embeddings, sparse vectors, and index to disk."""
        if self._embeddings is not None:
            np.save(str(self.embeddings_path), self._embeddings)
            with open(self.index_path, "w") as f:
                json.dump(self._id_to_row, f)

            if self._sparse_vectors:
                sparse_data = {str(k): v for k, v in self._sparse_vectors.items()}
                with open(self.sparse_index_path, "w") as f:
                    json.dump(sparse_data, f)

            debug_log(f"Saved {len(self._id_to_row)} embeddings to disk")

    def _build_searchable_text(self, entry: dict[str, Any]) -> str:
        """Build searchable text from entry fields (V3 + legacy)."""
        searchable_parts = [
            entry.get("context_prefix", ""),
            entry.get("title", ""),
            entry.get("insight", ""),
            " ".join(entry.get("keywords", [])),
            entry.get("summary", ""),
            entry.get("atomic_insight", ""),
            " ".join(entry.get("key_concepts", [])),
            " ".join(entry.get("tags", [])),
        ]
        return " ".join(filter(None, searchable_parts))

    @staticmethod
    def rrf_score(ranks: list[int], k: int = 60) -> float:
        """Calculate Reciprocal Rank Fusion score."""
        return sum(1.0 / (k + rank) for rank in ranks if rank > 0)

    def _generate_sparse_embedding(self, text: str) -> dict[str, float]:
        """Generate sparse embedding using SPLADE++. Returns empty dict if unavailable."""
        if self.sparse_model is None:
            return {}

        try:
            embeddings = list(self.sparse_model.embed([text]))
            if not embeddings:
                return {}

            sparse_emb = embeddings[0]
            result = {}
            for idx, val in zip(sparse_emb.indices, sparse_emb.values):
                if val > 0.01:
                    result[str(idx)] = float(val)
            return result
        except Exception as e:
            debug_log(f"Sparse embedding failed: {e}")
            return {}

    def _dense_search(self, query: str, limit: int) -> list[tuple]:
        """Dense (semantic) search using cosine similarity."""
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        query_embedding = list(self.dense_model.embed([query]))[0]
        scores = self._embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:limit]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def _sparse_search(self, query: str, limit: int) -> list[tuple]:
        """Sparse (keyword) search using SPLADE++."""
        if not self._sparse_vectors:
            return []

        query_sparse = self._generate_sparse_embedding(query)
        if not query_sparse:
            return []

        scores = []
        for row_idx, doc_sparse in self._sparse_vectors.items():
            score = 0.0
            for token, q_weight in query_sparse.items():
                if token in doc_sparse:
                    score += q_weight * doc_sparse[token]
            if score > 0:
                scores.append((row_idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]

    def _rerank_results(
        self, query: str, results: list[dict[str, Any]], limit: int
    ) -> list[dict[str, Any]]:
        """Rerank results using cross-encoder."""
        if self.reranker is None or not results:
            return results[:limit]

        try:
            documents = [self._build_searchable_text(r) for r in results]

            rerank_scores = list(self.reranker.rerank(query, documents))

            for r, score in zip(results, rerank_scores):
                r["rerank_score"] = float(score)

            results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return results[:limit]
        except Exception as e:
            debug_log(f"Reranking failed: {e}")
            return results[:limit]

    def add(self, entry: dict[str, Any], check_duplicates: bool = True) -> str:
        """Add a knowledge entry with semantic deduplication.

        Returns entry ID (new or existing if duplicate detected).
        """
        if "title" not in entry:
            raise ValueError("Entry must have 'title' field")
        if "insight" not in entry and "summary" not in entry:
            raise ValueError("Entry must have 'insight' field (or 'summary' for V2 compat)")

        # Reject low-quality entries
        title = entry.get("title", "").strip()
        if len(title) < 5 or len(title.split()) < 2:
            debug_log(f"Rejected entry with short title: '{title[:50]}'")
            return ""

        garbage_patterns = [
            r"^Session (?:started|ended|log)",
            r"^Docs?: .+ - ",
            r"^Documentation (?:reference|page)",
            r"^https?://",
            r"^Untitled",
        ]
        for pat in garbage_patterns:
            if re.match(pat, title, re.IGNORECASE):
                debug_log(f"Rejected garbage pattern '{pat}': '{title[:50]}'")
                return ""

        # Semantic deduplication (threshold 0.92)
        if check_duplicates and self._entries:
            query = f"{entry.get('title', '')} {entry.get('insight', entry.get('summary', ''))}"
            try:
                similar = self.search(query, limit=1, rerank=False)
                if similar and similar[0].get("score", 0) > 0.92:
                    existing_id = similar[0].get("id")
                    debug_log(
                        f"Duplicate detected (score={similar[0]['score']:.2f}): "
                        f"'{similar[0].get('title', '')[:50]}'"
                    )
                    return str(existing_id) if existing_id else ""
            except Exception as e:
                debug_log(f"Dedup check failed (proceeding with add): {e}")

        entry_id = entry.get("id") or str(uuid.uuid4())
        entry["id"] = entry_id

        if "date" not in entry:
            entry["date"] = (
                entry.get("found_date", "")[:10] or datetime.now().strftime("%Y-%m-%d")
            )

        if "insight" not in entry and "summary" in entry:
            entry["insight"] = entry["summary"]
        if "keywords" not in entry and "tags" in entry:
            entry["keywords"] = entry["tags"]

        entry.setdefault("type", "finding")
        entry.setdefault("priority", "medium")
        entry.setdefault("keywords", [])
        entry.setdefault("source", f"conv:{entry['date']}")
        entry.setdefault("timestamp", datetime.now().isoformat())
        entry.setdefault("branch", "")
        entry.setdefault("related_to", [])
        entry.setdefault("pinned", entry.get("priority") == "critical")

        searchable_text = self._build_searchable_text(entry)
        embedding = list(self.dense_model.embed([searchable_text]))[0]

        # Write JSONL first (source of truth), then update in-memory index
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        if self._embeddings is None:
            self._embeddings = embedding.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, embedding])

        row_idx = len(self._id_to_row)
        self._id_to_row[entry_id] = row_idx
        self._row_to_id[row_idx] = entry_id
        self._entries.append(entry)

        sparse_vec = self._generate_sparse_embedding(searchable_text)
        if sparse_vec:
            self._sparse_vectors[row_idx] = sparse_vec

        self._save_index()

        return entry_id

    def batch_add(
        self, entries: list[dict[str, Any]], check_duplicates: bool = False
    ) -> list[str]:
        """Add multiple entries with batch embedding (~275x faster than per-entry add).

        Args:
            entries: List of entry dicts
            check_duplicates: If True, check each entry for duplicates (slower)

        Returns:
            List of entry IDs
        """
        if not entries:
            return []

        # Validate and prepare entries
        valid_entries = []
        for entry in entries:
            if not entry.get("title") or (
                "insight" not in entry and "summary" not in entry
            ):
                continue

            title = entry.get("title", "").strip()
            if len(title) < 5 or len(title.split()) < 2:
                continue

            entry_id = entry.get("id") or str(uuid.uuid4())
            entry["id"] = entry_id
            if "date" not in entry:
                entry["date"] = datetime.now().strftime("%Y-%m-%d")
            if "insight" not in entry and "summary" in entry:
                entry["insight"] = entry["summary"]
            entry.setdefault("type", "finding")
            entry.setdefault("priority", "medium")
            entry.setdefault("keywords", [])
            entry.setdefault("source", f"conv:{entry['date']}")
            entry.setdefault("timestamp", datetime.now().isoformat())
            entry.setdefault("branch", "")
            entry.setdefault("related_to", [])
            entry.setdefault("pinned", False)

            valid_entries.append(entry)

        if not valid_entries:
            return []

        # Batch embed all texts at once
        texts = [self._build_searchable_text(e) for e in valid_entries]
        embeddings_list = list(self.dense_model.embed(texts))
        new_embeddings = np.array(embeddings_list)

        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        # Update index
        ids = []
        base_row = len(self._id_to_row)
        for i, entry in enumerate(valid_entries):
            row_idx = base_row + i
            entry_id = entry["id"]
            self._id_to_row[entry_id] = row_idx
            self._row_to_id[row_idx] = entry_id
            self._entries.append(entry)
            ids.append(entry_id)

        # Batch sparse embeddings
        if self.sparse_model is not None:
            try:
                sparse_list = list(self.sparse_model.embed(texts))
                for i, sparse_emb in enumerate(sparse_list):
                    row_idx = base_row + i
                    vec = {}
                    for idx, val in zip(sparse_emb.indices, sparse_emb.values):
                        if val > 0.01:
                            vec[str(idx)] = float(val)
                    if vec:
                        self._sparse_vectors[row_idx] = vec
            except Exception as e:
                debug_log(f"Batch sparse embedding failed: {e}")

        self._save_index()

        # Append to JSONL
        with open(self.jsonl_path, "a") as f:
            for entry in valid_entries:
                f.write(json.dumps(entry) + "\n")

        return ids

    def remove_entries(self, entry_ids: list[str]) -> int:
        """Soft-delete entries by zeroing their embeddings.

        Use rebuild() to compact after removal.
        """
        if not entry_ids:
            return 0

        id_set = set(entry_ids)
        removed = 0

        for eid in id_set:
            if eid in self._id_to_row:
                row_idx = self._id_to_row[eid]
                if self._embeddings is not None and row_idx < len(self._embeddings):
                    self._embeddings[row_idx] = 0.0
                if row_idx in self._sparse_vectors:
                    del self._sparse_vectors[row_idx]
                removed += 1

        if removed > 0:
            self._save_index()

            # Rewrite JSONL without removed entries
            if self.jsonl_path.exists():
                entries = []
                with open(self.jsonl_path) as f:
                    for line in f:
                        if line.strip():
                            try:
                                e = json.loads(line)
                                if isinstance(e, dict) and e.get("id") not in id_set:
                                    entries.append(e)
                            except json.JSONDecodeError:
                                pass
                with open(self.jsonl_path, "w") as f:
                    for e in entries:
                        f.write(json.dumps(e) + "\n")

        return removed

    def search(
        self,
        query: str,
        limit: int = 5,
        rerank: bool = True,
        date_from: str | None = None,
        date_to: str | None = None,
        entry_type: str | None = None,
        branch: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search with RRF fusion, filtering, and optional reranking."""
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        has_filters = any([date_from, date_to, entry_type, branch])
        candidate_limit = limit * (6 if has_filters else 4)

        dense_results = self._dense_search(query, candidate_limit)
        sparse_results = self._sparse_search(query, candidate_limit)

        dense_ranks = {row_idx: rank + 1 for rank, (row_idx, _) in enumerate(dense_results)}
        sparse_ranks = {row_idx: rank + 1 for rank, (row_idx, _) in enumerate(sparse_results)}

        all_rows = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        rrf_scores = []
        for row_idx in all_rows:
            ranks = [dense_ranks.get(row_idx, 0), sparse_ranks.get(row_idx, 0)]
            score = self.rrf_score(ranks, k=self.RRF_K)
            rrf_scores.append((row_idx, score))

        rrf_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for row_idx, rrf in rrf_scores[:candidate_limit]:
            if row_idx < len(self._entries):
                entry = self._entries[row_idx].copy()
                entry["score"] = rrf
                entry["_search_meta"] = {
                    "dense_rank": dense_ranks.get(row_idx, 0),
                    "sparse_rank": sparse_ranks.get(row_idx, 0),
                    "rrf_score": rrf,
                }
                results.append(entry)

        # Deduplicate by entry ID
        seen_ids: dict[str, dict[str, Any]] = {}
        for entry in results:
            eid = entry.get("id", "")
            if eid and (eid not in seen_ids or entry["score"] > seen_ids[eid]["score"]):
                seen_ids[eid] = entry
        if seen_ids:
            results = [e for e in results if seen_ids.get(e.get("id", "")) is e]

        # Boost pinned entries
        for r in results:
            if r.get("pinned"):
                r["score"] *= 1.3

        # Post-RRF filtering
        if date_from:
            results = [r for r in results if r.get("date", "") >= date_from]
        if date_to:
            results = [r for r in results if r.get("date", "") <= date_to]
        if entry_type:
            results = [r for r in results if r.get("type") == entry_type]
        if branch:
            results = [r for r in results if r.get("branch") == branch]

        if rerank:
            results = self._rerank_results(query, results, limit)
        else:
            results = results[:limit]

        return results

    def search_by_date(
        self, start_date: str, end_date: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Return entries in a date range, sorted by timestamp descending."""
        if end_date is None:
            end_date = start_date

        results = []
        for entry in self._entries:
            date = entry.get("date", "")
            if date and start_date <= date <= end_date:
                results.append(entry.copy())

        results.sort(key=lambda x: x.get("timestamp", "") or x.get("date", ""), reverse=True)
        return results[:limit]

    def get_timeline(self, date: str) -> list[dict[str, Any]]:
        """Return all entries for a specific day, sorted by timestamp ascending."""
        results = []
        for entry in self._entries:
            if entry.get("date", "") == date:
                results.append(entry.copy())

        results.sort(key=lambda x: x.get("timestamp", "") or "")
        return results

    def get_related(self, entry_id: str) -> list[dict[str, Any]]:
        """Return entries linked via related_to (bidirectional)."""
        related_ids = set()

        for entry in self._entries:
            if entry.get("id") == entry_id:
                for rid in entry.get("related_to", []):
                    related_ids.add(rid)
            elif entry_id in entry.get("related_to", []):
                related_ids.add(entry.get("id", ""))

        results = []
        for entry in self._entries:
            if entry.get("id") in related_ids:
                results.append(entry.copy())

        return results

    def get(self, entry_id: str) -> Optional[dict[str, Any]]:
        """Get a specific entry by ID (from in-memory index)."""
        row = self._id_to_row.get(entry_id)
        if row is not None and row < len(self._entries):
            entry = self._entries[row]
            if entry.get("title") != "?":  # Skip orphaned stubs
                return entry.copy()
        return None

    def stats(self) -> dict[str, Any]:
        """Get database statistics."""
        count = len(self._id_to_row) if self._id_to_row else 0
        size_bytes = 0

        for f in self.db_path.rglob("*"):
            if f.is_file():
                size_bytes += f.stat().st_size

        last_updated = None
        if self.embeddings_path.exists():
            last_updated = datetime.fromtimestamp(
                self.embeddings_path.stat().st_mtime
            ).isoformat()

        return {
            "count": count,
            "size_bytes": size_bytes,
            "size_human": f"{size_bytes / 1024:.1f} KB",
            "last_updated": last_updated,
            "db_path": str(self.db_path),
            "sub_store": self.sub_store,
            "backend": "fastembed-hybrid",
            "has_sparse_index": len(self._sparse_vectors) > 0,
            "sparse_entries": len(self._sparse_vectors),
        }

    def rebuild_index(self, dense_only: bool = False, batch_size: int = 50) -> int:
        """Rebuild the index from JSONL backup."""
        if not self.jsonl_path.exists():
            return 0

        entries = []
        needs_id_update = False
        with open(self.jsonl_path) as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        if isinstance(entry, dict):
                            entry = migrate_entry(entry)
                            if not entry.get("id"):
                                entry["id"] = str(uuid.uuid4())
                                needs_id_update = True
                            entries.append(entry)
                    except json.JSONDecodeError:
                        pass

        if not entries:
            return 0

        if needs_id_update:
            with open(self.jsonl_path, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

        texts = [self._build_searchable_text(e) for e in entries]

        debug_log(f"Generating dense embeddings for {len(texts)} entries...")
        embeddings_list = list(self.dense_model.embed(texts))
        self._embeddings = np.array(embeddings_list)

        self._sparse_vectors = {}
        if dense_only:
            debug_log("Skipping sparse embeddings (dense-only mode)")
        elif self.sparse_model is not None:
            debug_log(f"Generating sparse embeddings for {len(texts)} entries...")
            try:
                for batch_start in range(0, len(texts), batch_size):
                    batch_end = min(batch_start + batch_size, len(texts))
                    batch_texts = texts[batch_start:batch_end]

                    sparse_list = list(self.sparse_model.embed(batch_texts))
                    for local_idx, sparse_emb in enumerate(sparse_list):
                        global_idx = batch_start + local_idx
                        vec = {}
                        for token_idx, val in zip(sparse_emb.indices, sparse_emb.values):
                            if val > 0.01:
                                vec[str(token_idx)] = float(val)
                        if vec:
                            self._sparse_vectors[global_idx] = vec
            except Exception as e:
                debug_log(f"Sparse embedding failed (continuing with dense only): {e}")

        self._id_to_row = {}
        self._row_to_id = {}
        self._entries = entries
        for idx, entry in enumerate(entries):
            self._id_to_row[entry["id"]] = idx
            self._row_to_id[idx] = entry["id"]

        self._save_index()

        if self.old_txtai_path.exists():
            import shutil
            shutil.rmtree(self.old_txtai_path)

        debug_log(f"Rebuilt index with {len(entries)} entries")
        return len(entries)

    def list_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """List most recent entries (from in-memory index)."""
        entries = [
            e.copy() for e in self._entries
            if e.get("title") != "?"  # Skip orphaned stubs
        ]
        entries.sort(
            key=lambda x: (
                x.get("date", "") or x.get("found_date", ""),
                x.get("timestamp", ""),
            ),
            reverse=True,
        )
        return entries[:limit]

    def update_usage(self, entry_ids: list[str]) -> int:
        """Update usage stats. Increments usage_count, auto-pins at 3 uses."""
        if not entry_ids or not self.jsonl_path.exists():
            return 0

        now = datetime.now().isoformat()
        updated_count = 0
        entries = []
        id_set = set(entry_ids)

        with open(self.jsonl_path) as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        if isinstance(entry, dict):
                            if entry.get("id") in id_set:
                                entry["usage_count"] = entry.get("usage_count", 0) + 1
                                entry["last_used"] = now
                                if entry["usage_count"] >= 3 and not entry.get("pinned"):
                                    entry["pinned"] = True
                                updated_count += 1
                            entries.append(entry)
                    except json.JSONDecodeError:
                        pass

        if updated_count > 0:
            with open(self.jsonl_path, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

            # Sync in-memory cache from the JSONL-updated entries
            entries_by_id = {e["id"]: e for e in entries if e.get("id") in id_set}
            for i, cached in enumerate(self._entries):
                eid = cached.get("id")
                if eid in entries_by_id:
                    self._entries[i].update(entries_by_id[eid])

            self._enforce_pin_cap(entries)

        return updated_count

    def _enforce_pin_cap(self, entries: list[dict[str, Any]]) -> None:
        """Enforce max pinned entries. Unpin lowest-usage when cap exceeded."""
        pinned = [e for e in entries if e.get("pinned")]
        if len(pinned) <= self.MAX_PINNED:
            return

        pinned.sort(key=lambda e: e.get("usage_count", 0))
        for e in pinned[: len(pinned) - self.MAX_PINNED]:
            e["pinned"] = False

        with open(self.jsonl_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        unpinned_ids = {e.get("id") for e in entries if not e.get("pinned")}
        for i, cached in enumerate(self._entries):
            if cached.get("id") in unpinned_ids:
                self._entries[i]["pinned"] = False

    def get_recent_important(self, limit: int = 3) -> list[dict[str, Any]]:
        """Get recent/high-priority entries for context injection.

        Scores by type weight, recency (exponential decay), priority, and usage.
        """
        import math

        if not self._entries:
            return []

        type_weights = {
            "warning": 3.0, "solution": 2.5, "decision": 2.0, "pattern": 2.0,
            "finding": 1.5, "discovery": 1.5, "lesson": 1.5, "best-practice": 1.5,
            "commit": 0.5, "journal": 0.0, "session": 0.0,
        }
        priority_scores = {"critical": 100, "high": 50, "medium": 20, "low": 10}

        scored = []
        for entry in self._entries:
            entry_type = entry.get("type", "finding")
            weight = type_weights.get(entry_type, 1.0)
            if weight == 0.0:
                continue

            score = 0
            score += priority_scores.get(entry.get("priority", "medium"), 20)
            score += entry.get("usage_count", 0) * 5

            if entry.get("pinned"):
                score += 50
            else:
                try:
                    date_str = entry.get("date", "") or entry.get("found_date", "")
                    if date_str:
                        entry_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                        age_days = (datetime.now() - entry_date).days
                        half_life = 7 if entry_type == "warning" else 30
                        decay = math.exp(-age_days * 0.693 / half_life)
                        score += 50 * decay
                    else:
                        score += 10
                except (ValueError, TypeError):
                    score += 10

            score *= weight
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _, entry in scored[:limit]:
            results.append({
                "id": entry.get("id"),
                "title": entry.get("title", ""),
                "insight": entry.get("insight", entry.get("summary", ""))[:200],
                "type": entry.get("type", "finding"),
                "priority": entry.get("priority", "medium"),
                "keywords": entry.get("keywords", entry.get("tags", []))[:5],
                "pinned": entry.get("pinned", False),
            })

        return results

    def add_structured(self, data: dict) -> str:
        """Add a pre-structured entry (V3 schema, accepts V2 input)."""
        entry_type = data.get("type", "finding")
        if entry_type in ("lesson", "best-practice"):
            entry_type = "finding"

        insight = data.get("insight") or data.get("atomic_insight") or data.get("summary") or ""

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
            "id": data.get("id") or str(uuid.uuid4()),
            "title": data.get("title", ""),
            "insight": insight,
            "type": entry_type,
            "priority": data.get("priority", "medium"),
            "keywords": keywords[:10],
            "source": source,
            "date": data.get("date") or datetime.now().strftime("%Y-%m-%d"),
            "timestamp": data.get("timestamp") or datetime.now().isoformat(),
            "branch": data.get("branch", ""),
            "related_to": data.get("related_to", []),
        }

        if not entry["title"] and not entry["insight"]:
            raise ValueError("Entry must have 'title' or 'insight'")
        if not entry["title"]:
            entry["title"] = entry["insight"][:100]
        if not entry["insight"]:
            entry["insight"] = entry["title"]

        return self.add(entry)

    def migrate_all(self, rewrite: bool = False) -> dict:
        """Migrate all entries to V3 schema."""
        if not self.jsonl_path.exists():
            return {"status": "no_entries", "total": 0, "migrated": 0}

        entries = []
        migrated_count = 0
        skipped_lines = 0

        with open(self.jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if not isinstance(entry, dict):
                        skipped_lines += 1
                        continue
                    original_keys = set(entry.keys())
                    migrated = migrate_entry(entry)
                    if set(migrated.keys()) != original_keys:
                        migrated_count += 1
                    entries.append(migrated)
                except json.JSONDecodeError:
                    skipped_lines += 1

        result = {
            "status": "checked",
            "total": len(entries),
            "migrated": migrated_count,
            "needs_migration": migrated_count > 0,
            "skipped_lines": skipped_lines,
        }

        if rewrite and migrated_count > 0:
            backup_path = self.jsonl_path.with_suffix(".jsonl.bak")
            import shutil
            shutil.copy(self.jsonl_path, backup_path)

            with open(self.jsonl_path, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

            result["status"] = "migrated"
            result["backup"] = str(backup_path)

        return result

    def count(self) -> int:
        """Return number of entries."""
        return len(self._id_to_row) if self._id_to_row else 0
