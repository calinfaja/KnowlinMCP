# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KnowlinMCP -- hybrid semantic knowledge database with MCP server and multi-source search. Captures, indexes, and retrieves project knowledge using dense embeddings (BGE-small-en-v1.5), sparse matching (SPLADE++), and cross-encoder reranking.

## Commands

All tools are in the `.venv`. Use `.venv/bin/` prefix or activate the venv first.

```bash
# Install for development
.venv/bin/pip install -e ".[dev,mcp]"

# Run all unit tests
.venv/bin/pytest tests/ -v

# Run a single test file or specific test
.venv/bin/pytest tests/test_db.py -v
.venv/bin/pytest tests/test_db.py::test_function_name -v

# Integration tests (load real ML models, slow)
.venv/bin/pytest tests/ --integration

# Benchmarks
.venv/bin/pytest tests/ --benchmark-enable

# Coverage
.venv/bin/pytest tests/ --cov=knowlin_mcp

# Lint
.venv/bin/ruff check src/ tests/
.venv/bin/ruff check --fix src/ tests/

# Format
.venv/bin/black src/ tests/

# Type check
.venv/bin/mypy src/knowlin_mcp/ --ignore-missing-imports

# Security scan
.venv/bin/bandit -r src/knowlin_mcp/

# Dead code detection
.venv/bin/vulture src/knowlin_mcp/

# CLI entry point
.venv/bin/knowlin search "query"
.venv/bin/knowlin stats
.venv/bin/knowlin doctor --fix

# MCP server (stdio)
.venv/bin/knowlin-mcp
```

## Architecture

### Package layout: `src/knowlin_mcp/`

| Module | Responsibility |
|--------|---------------|
| `db.py` | Core `KnowledgeDB` class -- JSONL storage, dense/sparse search, RRF fusion, reranking |
| `models.py` | Embedding model singletons (BGE, SPLADE++, cross-encoder) -- lazy-loaded, shared |
| `multi_search.py` | `MultiSourceSearch` -- orchestrates search across 3 sub-stores with intent-aware weighting |
| `search.py` | Result formatters (compact, detailed, inject, json) |
| `capture.py` | Entry creation and save with fallback chain (server -> DB -> JSONL append) |
| `ingest_docs.py` | Markdown/PDF document ingestion with heading-based chunking |
| `ingest_sessions.py` | Claude Code session transcript extraction |
| `cli.py` | Click CLI (entry point: `knowlin`) |
| `mcp_server.py` | FastMCP server (entry point: `knowlin-mcp`) -- 4 tools for MCP clients |
| `server.py` | TCP daemon keeping embeddings in RAM for fast queries |
| `platform.py` | Cross-platform paths and process management (psutil + platformdirs) |
| `query_utils.py` | Query intent classification (DEBUG/HOWTO/RECALL/EXPLORE) and synonym expansion |
| `utils.py` | TCP helpers, schema constants, V2->V3 migration |

### Search pipeline

```
Query -> Intent classification -> Synonym expansion
  -> Per-source search (dense + sparse + RRF each)
  -> Weighted RRF across sources (weights vary by intent)
  -> Dedup by title -> Score normalize -> Cross-encoder rerank
```

Intent adjusts source weights: DEBUG favors sessions (2.0), HOWTO favors docs (2.0), RECALL favors sessions (2.5), EXPLORE is uniform (1.0).

### Sub-store pattern

Each project has `.knowledge-db/` with three isolated sub-stores:
- **kb** (root) -- curated entries
- **sessions/** -- ingested Claude Code transcripts
- **docs/** -- ingested markdown/PDF chunks

Each sub-store has its own `entries.jsonl`, `embeddings.npy`, `sparse_index.json`, `index.json`. `KnowledgeDB(path, sub_store="sessions")` targets a specific sub-store.

### Model singletons

Global lazy-loaded models in `models.py`:
- `_dense_model`: BAAI/bge-small-en-v1.5 (384-dim)
- `_sparse_model`: prithivida/Splade_PP_en_v1
- `_reranker`: Xenova/ms-marco-MiniLM-L-6-v2

First search/embed call is slow (model download); subsequent calls reuse singletons.

### Persistence fallback chain

`save_entry()` in `capture.py` tries: TCP server -> direct `KnowledgeDB.add()` -> raw JSONL append. This ensures writes succeed even when the server is down.

### Incremental ingestion

Ingestors use SHA-256 file hashing with registry files (`session-registry.json`, `doc-registry.json`) to skip unchanged files and clean up deleted ones.

### TCP server

Per-project daemon on port 14000+. Keeps models and embeddings in RAM for ~30ms queries. Auto-shuts down after 1 hour idle. PID/port files stored in runtime dir.

### MCP server

FastMCP server exposing 4 tools via stdio transport: `knowlin_search`, `knowlin_get`, `knowlin_stats`, `knowlin_ingest`. Compatible with Claude, Gemini, Codex, Cursor, VS Code, and other MCP clients. Installed as optional dep (`pip install "knowlin-mcp[mcp]"`).

## Key conventions

- **Python 3.9+** target. Uses `from __future__ import annotations` throughout.
- **Line length**: 100 (ruff + black).
- **Ruff rules**: E, F, W, I (errors, pyflakes, warnings, isort).
- **Lazy imports** in `__init__.py` to avoid loading fastembed on module import.
- **Entry schema V3** with backward-compatible V2 migration (summary->insight, tags->keywords, found_date->date). Migration happens transparently in `utils.py`.
- **Entry types**: finding, solution, pattern, warning, decision, discovery.
- **Dedup threshold**: 0.92 cosine similarity on add.
- **Test fixtures**: `temp_kb_dir`, `sample_entries`, `kb_with_entries`, `project_root` in `conftest.py`. Tests use `tmp_path` to avoid touching real data.
- **Markers**: `@pytest.mark.integration` (skipped without `--integration`), `@pytest.mark.benchmark`.
