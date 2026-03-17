"""Embedding model singletons -- lazy-loaded, shared across all KnowledgeDB instances.

Models:
- Dense: BAAI/bge-small-en-v1.5 (384-dim semantic embeddings)
- Sparse: prithivida/Splade_PP_en_v1 (learned sparse token weights)
- Reranker: Xenova/ms-marco-MiniLM-L-6-v2 (cross-encoder reranker)

All models are cached at ~/.cache/fastembed/ (~200MB total on first download).
"""

from __future__ import annotations

from typing import Optional

from knowlin_mcp.utils import debug_log

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

MODEL_NAMES = [
    "BAAI/bge-small-en-v1.5",
    "prithivida/Splade_PP_en_v1",
    "Xenova/ms-marco-MiniLM-L-6-v2",
]


def models_cached() -> bool:
    """Check if embedding models are already downloaded."""
    import os

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "fastembed")
    if not os.path.isdir(cache_dir):
        return False
    # Check for at least one model directory
    return any(d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d)))


def _warn_if_first_run() -> None:
    """Print a one-time warning if embedding models need to be downloaded."""
    global _first_run_warned
    if _first_run_warned:
        return
    _first_run_warned = True

    import sys

    if models_cached():
        return

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


def get_dense_embedding(text: str):
    """Generate a dense embedding for a single text input."""
    return list(get_dense_model().embed([text]))[0]


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
