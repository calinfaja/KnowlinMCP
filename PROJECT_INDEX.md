# Project Index: knowlin-mcp

Generated: 2026-03-14 | v0.1.0 | Python >=3.9 | 9,778 lines total

## Project Structure

```
src/knowlin_mcp/          # 15 files, 5,175 lines
  __init__.py             # Lazy imports (KnowledgeDB, MultiSourceSearch)
  db.py            1002L  # Core: KnowledgeDB class (JSONL, dense+sparse search, RRF, rerank)
  models.py          99L  # Singleton model loaders (BGE, SPLADE++, cross-encoder)
  multi_search.py   190L  # MultiSourceSearch: cross-source orchestrator with intent weighting
  query_utils.py    112L  # QueryIntent enum, classify_query, expand_query, source weights
  search.py         141L  # Output formatters (compact/detailed/inject/json)
  capture.py        227L  # Entry creation + save with fallback chain (server->DB->JSONL)
  server.py         476L  # TCP daemon (KnowledgeServer, 1hr idle TTL, ~30ms queries)
  mcp_server.py     262L  # FastMCP server (5 tools: search, get, stats, ingest, capture)
  cli.py            940L  # Click CLI (search, capture, list, get, delete, export, server, ingest, stats, doctor, init)
  ingest_docs.py    515L  # DocsIngester: markdown/PDF chunking into docs/ sub-store
  ingest_sessions.py 507L # SessionIngester: Claude Code JSONL extraction with scoring
  ingest_codex.py   307L  # CodexIngester: Codex CLI JSONL extraction
  platform.py       227L  # Cross-platform paths, process mgmt (psutil, platformdirs)
  utils.py          331L  # Schema V3, migration V2->V3, TCP helpers, type inference
tests/                    # 18 files, 4,603 lines
docs/INDEX.md             # Detailed architecture reference (667 lines)
```

## Entry Points

- CLI: `knowlin` -> `cli.py:main` (Click)
- MCP: `knowlin-mcp` -> `mcp_server.py:main` (FastMCP, stdio)
- Library: `from knowlin_mcp import KnowledgeDB, MultiSourceSearch`

## Architecture (5-line summary)

Hybrid semantic knowledge DB. Three sub-stores (kb, sessions, docs) each with JSONL + numpy embeddings + SPLADE++ sparse index with inverted index for sub-linear sparse search. Query pipeline: intent classification -> per-source dense+sparse+RRF search -> weighted cross-source RRF -> title dedup -> cross-encoder rerank. Writes use fallback chain (TCP server -> direct DB -> raw JSONL). Incremental ingestion via SHA-256 file hashing with registry files.

## Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `KnowledgeDB` | db.py | Core engine: add/search/remove on one sub-store |
| `MultiSourceSearch` | multi_search.py | Orchestrates search across kb+sessions+docs |
| `KnowledgeServer` | server.py | TCP daemon wrapping KnowledgeDB, keeps models in RAM |
| `DocsIngester` | ingest_docs.py | Markdown/PDF -> heading-based chunks -> docs/ store |
| `SessionIngester` | ingest_sessions.py | Claude Code JSONL -> scored entries -> sessions/ store |
| `CodexIngester` | ingest_codex.py | Codex CLI JSONL -> entries -> sessions/ store |

## Key Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `save_entry()` | capture.py | Write with fallback: server -> DB -> JSONL append |
| `classify_query()` | query_utils.py | DEBUG/HOWTO/RECALL/EXPLORE intent from keywords |
| `get_source_weights()` | query_utils.py | Intent -> per-source weight dict |
| `get_dense_model()` | models.py | Lazy-load BGE-small-en-v1.5 (384-dim) |
| `get_sparse_model()` | models.py | Lazy-load SPLADE++ |
| `migrate_entry()` | utils.py | V2->V3 schema migration |
| `knowlin_capture()` | mcp_server.py | MCP tool: save new knowledge entries via MCP clients |

## Dependencies

Core: fastembed, numpy, click, rich, psutil, platformdirs, pyyaml
Optional: mcp (MCP server), pymupdf4llm (PDF ingestion)
Dev: pytest, pytest-cov, pytest-benchmark, hypothesis, ranx, ruff, black

## Configuration

- `.knowledge-db/sources.yaml` -- docs paths, include/exclude globs, session auto-discover
- `KNOWLIN_DEBUG` env var -- enable stderr debug logging
- Sub-stores: `.knowledge-db/` (kb), `.knowledge-db/sessions/`, `.knowledge-db/docs/`

## Dependency Layers

```
L0: platform.py                    (no internal deps)
L1: utils.py, query_utils.py, models.py
L2: db.py, search.py
L3: multi_search.py, capture.py, server.py
L4: ingest_docs.py, ingest_sessions.py, ingest_codex.py
L5: cli.py, mcp_server.py          (top-level entry points)
```

## Testing

18 test files | 344 tests | Markers: `@integration` (real models), `@benchmark` (perf)
Fixtures: `temp_kb_dir`, `sample_entries`, `kb_with_entries`, `project_root`
Run: `.venv/bin/pytest tests/ -v` | Integration: `--integration` | Coverage: `--cov=knowlin_mcp`

## Quick Reference

```bash
.venv/bin/knowlin search "query"          # Search all sources
.venv/bin/knowlin capture "content"       # Add entry
.venv/bin/knowlin list                    # List recent entries
.venv/bin/knowlin get <id>                # Get full entry details
.venv/bin/knowlin delete <id>             # Remove an entry
.venv/bin/knowlin export                  # Export entries to JSONL/JSON
.venv/bin/knowlin ingest all              # Ingest docs + sessions + codex
.venv/bin/knowlin server start            # Start TCP daemon
.venv/bin/knowlin doctor --fix            # Health check + repair
.venv/bin/knowlin stats --json            # Database stats
.venv/bin/pytest tests/ -v                # Run tests
.venv/bin/ruff check src/ tests/          # Lint
```
