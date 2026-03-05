# KnowlinMCP -- Project Index

> Auto-generated project documentation. Comprehensive reference for architecture, API surface, data flows, and module relationships.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Module Reference](#module-reference)
- [Public API Surface](#public-api-surface)
- [Data Flow Diagrams](#data-flow-diagrams)
- [Schema Reference](#schema-reference)
- [Configuration Reference](#configuration-reference)
- [Test Architecture](#test-architecture)
- [Cross-Module Dependencies](#cross-module-dependencies)

---

## Architecture Overview

### System Design

```
                        +-----------+
                        |  CLI      |  cli.py (Click commands)
                        +-----+-----+
                              |
          +-------------------+-------------------+
          |                   |                   |
    +-----v-----+     +------v------+     +------v------+
    | capture.py |     | search.py   |     | ingest_*.py |
    | (write)    |     | (format)    |     | (bulk load) |
    +-----+------+     +------+------+     +------+------+
          |                   |                   |
          +-------------------+-------------------+
                              |
                   +----------v----------+
                   | MultiSourceSearch   |  multi_search.py
                   | (intent weighting)  |  (orchestrates 3 sub-stores)
                   +----------+----------+
                              |
               +--------------+--------------+
               |              |              |
          +----v----+   +----v----+   +-----v----+
          | KB (root)|  |sessions/|   | docs/    |
          +---------+   +---------+   +----------+
               |              |              |
               +-------+------+--------------+
                       |
              +--------v--------+
              |  KnowledgeDB    |  db.py (core engine)
              |  dense + sparse |
              |  RRF + rerank   |
              +--------+--------+
                       |
          +------------+------------+
          |            |            |
     +----v----+  +---v----+  +---v------+
     | fastembed|  | numpy  |  | JSONL    |
     | (models) |  | (vecs) |  | (storage)|
     +---------+  +--------+  +----------+

                   +----------------+
                   | KnowledgeServer|  server.py (TCP daemon)
                   | (wraps DB,     |  keeps models in RAM
                   |  1hr idle TTL) |
                   +----------------+
```

### Design Principles

1. **Sub-store isolation**: Each source (kb, sessions, docs) has independent JSONL + embeddings + index files
2. **Model singletons**: Dense/sparse/reranker models are module-level globals, lazy-loaded once
3. **Fallback chain**: Writes try server -> direct DB -> JSONL append (never fail silently)
4. **Incremental ingestion**: SHA-256 file hashing with registry files to skip unchanged content
5. **Intent-aware fusion**: Query classification adjusts per-source weights in cross-source RRF

---

## Module Reference

### `db.py` (1050 lines) -- Core Engine

The central class `KnowledgeDB` manages one sub-store's data. Handles JSONL persistence, dense/sparse embedding, RRF fusion, and cross-encoder reranking.

**Class: `KnowledgeDB`**
- Constructor: `KnowledgeDB(project_path=None, sub_store=None)`
- Locates project root via `find_project_root()` (searches up for `.knowledge-db`, `.serena`, `.claude`, `.git`)
- Sub-store parameter routes to `.knowledge-db/{sub_store}/`
- In-memory state: `_embeddings` (numpy), `_entries` (list), `_sparse_vectors` (dict), `_id_to_row`/`_row_to_id` (dicts)

**Key methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `add` | `(entry, check_duplicates=True) -> str` | Add entry with semantic dedup (0.92 threshold). Returns ID. |
| `batch_add` | `(entries) -> list[str]` | Bulk add, skips duplicates |
| `search` | `(query, limit=5, rerank=True, date_from, date_to, entry_type, branch) -> list[dict]` | Hybrid search: dense + sparse + RRF + optional rerank + post-filters |
| `remove_entries` | `(ids) -> None` | Remove by ID, rebuilds index |
| `get` | `(entry_id) -> dict \| None` | Fetch single entry by ID |
| `stats` | `() -> dict` | Count, file sizes, last updated |
| `rebuild_index` | `(dense_only=False) -> int` | Rebuild embeddings/sparse from JSONL |
| `count` | `() -> int` | Entry count |
| `migrate_all` | `() -> None` | Migrate all entries to V3 schema |
| `search_by_date` | `(date_from, date_to, limit) -> list` | Date-range search |
| `list_recent` | `(limit) -> list` | Most recent entries |
| `get_related` | `(entry_id, limit) -> list` | Find related entries by similarity |
| `add_structured` | `(data) -> str` | Add from structured dict (validates schema) |

**Module-level singletons:**
- `_dense_model`: `TextEmbedding("BAAI/bge-small-en-v1.5")` -- 384-dim dense vectors
- `_sparse_model`: `SparseTextEmbedding("prithivida/Splade_PP_en_v1")` -- learned sparse weights
- `_reranker`: `TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")` -- cross-encoder reranking
- Access via `get_dense_model()`, `get_sparse_model()`, `get_reranker()` (lazy init)

**Search internals:**
- `_dense_search(query, limit)`: Cosine similarity on numpy array
- `_sparse_search(query, limit)`: Dot product on sparse token weights
- `_rerank_results(query, results, limit)`: Cross-encoder re-scoring
- `rrf_score(ranks, k=60)`: Reciprocal Rank Fusion formula: `sum(1/(k+rank) for rank in ranks)`
- `_build_searchable_text(entry)`: Concatenates title + insight + keywords for embedding

---

### `multi_search.py` (177 lines) -- Cross-Source Orchestrator

**Class: `MultiSourceSearch`**
- Constructor: `MultiSourceSearch(project_path)`
- Lazily creates `KnowledgeDB` instances for each source via `_get_store(source_name)`
- Source names: `"kb"` (root), `"sessions"`, `"docs"`

**Key method: `search()`**
```python
search(query, sources=None, limit=5, date_from=None, date_to=None,
       entry_type=None, branch=None, auto_expand=True) -> list[dict]
```

Pipeline:
1. Classify intent via `classify_query(query)` -> `QueryIntent`
2. Get per-source weights via `get_source_weights(intent)`
3. Optionally expand query with synonyms
4. Search each source (oversample 3x), apply weighted RRF score
5. Deduplicate by title, normalize scores to 0-1
6. Tag results with `_source` and `_search_meta`

---

### `query_utils.py` (112 lines) -- Intent Classification

**Enum: `QueryIntent`**
- `DEBUG` -- error/bug investigation
- `HOWTO` -- how to accomplish something
- `RECALL` -- recall a specific past event/decision
- `EXPLORE` -- open-ended exploration

**Functions:**

| Function | Purpose |
|----------|---------|
| `classify_query(query) -> QueryIntent` | Keyword-based intent classification using `_INTENT_SIGNALS` |
| `expand_query(query) -> str` | Append synonym terms from `_SYNONYMS` dict |
| `get_source_weights(intent) -> dict` | Intent-to-weight mapping (see table below) |

**Source weight matrix:**

| Intent | kb | sessions | docs |
|--------|-----|----------|------|
| DEBUG | 1.5 | **2.0** | 0.5 |
| HOWTO | 1.5 | 0.8 | **2.0** |
| RECALL | 1.0 | **2.5** | 0.3 |
| EXPLORE | 1.0 | 1.0 | 1.0 |

---

### `capture.py` (227 lines) -- Entry Creation & Persistence

**Functions:**

| Function | Purpose |
|----------|---------|
| `create_entry(content, entry_type, tags, priority, url) -> dict` | Build V3 entry from content string |
| `create_entry_from_json(data) -> dict` | Build entry from structured JSON dict |
| `save_entry(entry, knowledge_dir) -> bool` | Persist with fallback chain: server -> DB -> JSONL |
| `send_entry_to_server(entry, project_path) -> bool` | Try TCP add command |
| `log_to_timeline(entry, knowledge_dir) -> None` | Append to timeline log |

**Fallback chain in `save_entry()`:**
1. TCP server `add` command (fast, in-memory sync)
2. Direct `KnowledgeDB(project_path).add(entry)` (loads models)
3. Raw JSONL file append (always works, needs `rebuild` later)

---

### `search.py` (135 lines) -- Output Formatting

**Formatters** (registered in `FORMATTERS` dict):

| Formatter | Key | Output |
|-----------|-----|--------|
| `format_compact` | `"compact"` | One-line per result: `[score] title` |
| `format_detailed` | `"detailed"` | Full entry details with metadata |
| `format_inject` | `"inject"` | Claude Code injection format (for prompt context) |
| `format_json` | `"json"` | Raw JSON array |
| `format_single_entry` | -- | Format a single entry by ID |

---

### `server.py` (476 lines) -- TCP Daemon

**Class: `KnowledgeServer`**
- Constructor: `KnowledgeServer(project_path=None)`
- TCP server on `127.0.0.1`, port auto-allocated (range 14000+)
- `IDLE_TIMEOUT = 3600` seconds (1 hour auto-shutdown)
- Keeps `KnowledgeDB` loaded in memory for ~30ms query response

**TCP commands** (JSON protocol over socket):
- `{"cmd": "search", "query": "...", "limit": 5}`
- `{"cmd": "add", "entry": {...}}`
- `{"cmd": "reload"}` -- reload index from disk
- `{"cmd": "ping"}` -- health check
- `{"cmd": "status"}` -- server stats

**Utility functions:**
- `find_available_port()` -- find free port
- `read_port_file()` / `write_port_file()` -- port persistence
- `send_command(project_path, cmd_dict)` -- send TCP command
- `list_running_servers()` -- enumerate active server processes

---

### `ingest_docs.py` (512 lines) -- Document Ingestion

**Class: `DocsIngester`**
- Constructor: `DocsIngester(project_path, docs_path=None)`
- Path resolution: CLI `--path` > `sources.yaml` > auto-discover (`docs/`, `doc/`, `INFOS/`, `documentation/`)
- Supports glob include/exclude from sources.yaml

**Chunking strategy:**
- `_chunk_by_headings(text, source_path)` -- split by `#`/`##`/`###` headings
- `_sub_split(text, max_chars)` -- split oversized chunks by paragraphs
- `_recursive_split(text, max_chars)` -- further split by sentences
- Constants: `MAX_CHUNK_CHARS = 2000`, `MIN_CHUNK_CHARS = 100`, `OVERLAP_CHARS = 200`

**Registry:** `doc-registry.json` maps `file_path -> {hash, chunks: [ids], mtime}`

**Key method:** `ingest(full=False) -> int`
- Scans for `.md`, `.txt`, `.pdf`, `.rst` files
- Hashes each file; skips unchanged
- Chunks content, creates entries, embeds into `docs/` sub-store
- Cleans up entries from deleted files
- Returns count of new entries

---

### `ingest_sessions.py` (422 lines) -- Session Transcript Ingestion

**Class: `SessionIngester`**
- Constructor: `SessionIngester(project_path, sessions_dir=None)`
- Auto-discovers sessions from `~/.claude/projects/`
- Extracts high-value messages from Claude Code JSONL transcripts

**Content scoring:**
- `_score_content(text) -> float` -- scores messages by `_VALUE_SIGNALS` (error, solution, architecture keywords)
- `_SKIP_PATTERNS` -- filters out low-value messages (greetings, status updates)
- `MIN_CONTENT_LENGTH = 100` -- minimum message length

**Key method:** `ingest(full=False) -> int`
- Scans JSONL files, extracts assistant messages
- Scores each message, keeps high-value content
- Creates entries with session metadata (date, source file)
- Registry: `session-registry.json`

---

### `cli.py` (673 lines) -- CLI Interface

Click-based CLI with `knowlin` entry point.

**Command groups:**
- `knowlin search` -- search with multi-source, filters, formatters
- `knowlin capture` -- create entries (text or JSON input)
- `knowlin server {start|stop|status}` -- daemon management
- `knowlin ingest {sessions|docs|all}` -- bulk ingestion
- `knowlin stats [--json]` -- database statistics
- `knowlin rebuild [--dense-only]` -- rebuild search index
- `knowlin doctor [--fix]` -- health check and repair
- `knowlin sources [--init]` -- show/create sources.yaml

**Internal:** `_resolve_project()` finds project root from CWD.

---

### `utils.py` (331 lines) -- Shared Utilities

**Schema constants:**
- `SCHEMA_V3_FIELDS`: `[id, title, insight, type, priority, keywords, source, date, timestamp, branch, related_to]`
- `SCHEMA_V3_DEFAULTS`: Default values for optional V3 fields
- `SCHEMA_V2_DEFAULTS`: Legacy V2 field defaults
- `TYPE_SIGNALS`: Keyword-to-type mapping for `infer_type()`

**Key functions:**
- `migrate_entry(entry) -> dict` -- V2->V3 migration (summary->insight, tags->keywords, found_date->date, quality->priority)
- `infer_type(title, insight) -> str` -- classify entry type from content
- `debug_log(msg, category)` -- conditional stderr logging (enabled by `KNOWLIN_DEBUG` env var)
- `get_server_port(project_path)` / `get_pid_path(project_path)` -- runtime file locations
- `is_server_running(project_path)` -- check server process
- `send_command(project_path, cmd)` / `search(project_path, query)` -- TCP client helpers

---

### `platform.py` (235 lines) -- Cross-Platform Utilities

**Constants:**
- `DEFAULT_KB_PORT = 14000`
- `KB_DIR_NAME = ".knowledge-db"`
- `PROJECT_MARKERS = [".knowledge-db", ".serena", ".claude", ".git"]`
- `HOST = "127.0.0.1"`

**Path functions:**
- `get_config_dir()` -- `~/.config/knowlin-mcp/` (Linux) via platformdirs
- `get_cache_dir()` -- `~/.cache/knowlin-mcp/` (Linux)
- `get_runtime_dir()` -- `/tmp/knowlin-{username}/`
- `get_project_hash(path)` -- SHA-256 hash of project path (for per-project PID/port files)
- `find_project_root(start_path)` -- walk up looking for `PROJECT_MARKERS`

**Process functions:**
- `find_process(pid)` / `is_process_running(pid)` -- psutil wrappers
- `spawn_background(cmd)` -- platform-aware background process launch
- `kill_process_tree(pid)` -- kill process and children
- `read_pid_file()` / `write_pid_file()` -- PID file management
- `cleanup_stale_files()` -- remove orphaned PID/port files

---

## Data Flow Diagrams

### Write Path (Capture)

```
User input
  -> create_entry() or create_entry_from_json()     [capture.py]
       -> Validates fields, sets defaults
       -> Infers type if missing                     [utils.py: infer_type()]
  -> save_entry(entry, knowledge_dir)                [capture.py]
       |
       +--[1] send_entry_to_server()  ----TCP----> KnowledgeServer.handle_client()
       |       (if server running)                    -> db.add(entry)
       |
       +--[2] KnowledgeDB.add(entry)
       |       -> Title validation (>=5 chars, >=2 words)
       |       -> Garbage pattern rejection
       |       -> Semantic dedup search (threshold 0.92)
       |       -> Generate dense embedding (fastembed)
       |       -> Append to JSONL
       |       -> Update in-memory numpy array
       |       -> Generate sparse embedding (SPLADE++)
       |       -> Save index
       |
       +--[3] Direct JSONL append (fallback)
              -> Needs `rebuild` to re-index
```

### Read Path (Search)

```
User query
  -> MultiSourceSearch.search()                      [multi_search.py]
       -> classify_query(query)                      [query_utils.py]
            -> Returns QueryIntent enum
       -> get_source_weights(intent)
       -> expand_query(query)                        [optional synonym expansion]
       |
       +-- For each source (kb, sessions, docs):
       |     -> KnowledgeDB.search()                 [db.py]
       |          -> _dense_search(): cosine similarity on numpy embeddings
       |          -> _sparse_search(): SPLADE++ sparse dot product
       |          -> RRF fusion (k=60)
       |          -> Apply filters (date, type, branch)
       |          -> Return ranked results
       |
       -> Weighted RRF across sources
       -> Deduplicate by title
       -> Normalize scores to [0, 1]
       -> Tag with _source and _search_meta
  -> format_*(results)                               [search.py]
       -> compact | detailed | inject | json
```

### Ingest Path (Docs/Sessions)

```
knowlin ingest all
  |
  +-- DocsIngester.ingest()                          [ingest_docs.py]
  |     -> Load doc-registry.json
  |     -> Scan docs dirs (sources.yaml or auto-discover)
  |     -> For each .md/.pdf/.txt/.rst:
  |          -> SHA-256 hash check vs registry
  |          -> Skip if unchanged
  |          -> Read file (_read_file or _pdf_to_markdown)
  |          -> _chunk_by_headings() -> _sub_split() -> _recursive_split()
  |          -> KnowledgeDB(sub_store="docs").batch_add(chunks)
  |     -> _cleanup_deleted_files() (remove entries from deleted files)
  |     -> Save updated registry
  |
  +-- SessionIngester.ingest()                       [ingest_sessions.py]
        -> Load session-registry.json
        -> Find JSONL files in sessions dir
        -> For each session file:
             -> SHA-256 hash check vs registry
             -> _extract_from_jsonl(): parse messages
             -> _score_content(): rate by value signals
             -> Keep high-scoring messages
             -> KnowledgeDB(sub_store="sessions").batch_add(entries)
        -> _cleanup_deleted_files()
        -> Save updated registry
```

---

## Schema Reference

### V3 Entry Schema (current)

```json
{
  "id": "uuid-string",              // Required. Unique identifier
  "title": "Entry Title",           // Required. >=5 chars, >=2 words
  "insight": "Main content text",   // Required. (or "summary" for V2 compat)
  "type": "finding",                // finding | solution | pattern | warning | decision | discovery
  "priority": "medium",             // low | medium | high | critical
  "keywords": ["tag1", "tag2"],     // Searchable terms
  "source": "conv:2026-03-04",      // Origin: conv:{date}, file:{path}, or URL
  "date": "2026-03-04",             // YYYY-MM-DD
  "timestamp": "2026-03-04T12:00:00",  // ISO 8601
  "branch": "main",                 // Git branch context
  "related_to": ["other-id"],       // Related entry IDs
  "pinned": false                   // Boosts relevance score (1.3x)
}
```

### V2 -> V3 Migration Map

| V2 Field | V3 Field | Transformation |
|----------|----------|----------------|
| `summary` | `insight` | Direct copy |
| `atomic_insight` | `insight` | Merged with summary |
| `tags` | `keywords` | Merged with key_concepts, deduped |
| `key_concepts` | `keywords` | Merged with tags |
| `found_date` | `date` | Truncated to YYYY-MM-DD |
| `url` | `source` | Direct copy if HTTP |
| `source_path` | `source` | Prefixed with `file:` |
| `confidence_score` | `priority` | Mapped: >=0.9 -> high, <0.5 -> low |
| `quality` | `priority` | Mapped: high/medium/low |

Migration is applied transparently by `migrate_entry()` in `utils.py`.

---

## Configuration Reference

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `KNOWLIN_DEBUG` | Enable debug logging to stderr (any truthy value) |

### sources.yaml

Located at `.knowledge-db/sources.yaml`. Optional -- convention-based discovery is used without it.

```yaml
docs:
  paths:                              # List of directories to scan
    - docs/                           # Relative to project root
    - ~/Desktop/INFOS/                # Absolute (~ expanded)
  include: ["*.md", "*.txt", "*.pdf", "*.rst"]  # File globs (default if omitted)
  exclude: ["drafts/**", "*.tmp"]     # Exclusion globs

sessions:
  auto_discover: true                 # Scan ~/.claude/projects/
  # path: ~/custom/sessions/         # Override auto-discovery
```

### Auto-Discovery Conventions

| Source | Directories scanned |
|--------|-------------------|
| Docs | `docs/`, `doc/`, `INFOS/`, `documentation/` (relative to project root) |
| Sessions | `~/.claude/projects/` (matching project name) |

---

## Test Architecture

### Test Files (15 files)

| Test File | Tests | Focus |
|-----------|-------|-------|
| `test_db.py` | KnowledgeDB core | CRUD, search, dedup, migration, index |
| `test_multi_search.py` | MultiSourceSearch | Cross-source fusion, intent weighting |
| `test_search_pipeline.py` | End-to-end search | Full pipeline from query to results |
| `test_cli.py` | CLI commands | Click runner integration tests |
| `test_capture.py` | Entry creation | save_entry fallback chain, validation |
| `test_formatters.py` | Output formatting | compact/detailed/inject/json formats |
| `test_query_utils.py` | Query processing | Intent classification, expansion |
| `test_ingest_docs.py` | Doc ingestion | Chunking, registry, incremental |
| `test_ingest_sessions.py` | Session ingestion | JSONL extraction, scoring |
| `test_server.py` | TCP server | Server lifecycle, commands |
| `test_integrity.py` | Data integrity | Schema validation, corruption recovery |
| `test_properties.py` | Property-based | Hypothesis fuzz tests |
| `test_benchmarks.py` | Performance | pytest-benchmark timing |
| `test_retrieval.py` | Retrieval quality | Search relevance metrics |
| `test_pdf_ingestion.py` | PDF support | pymupdf4llm integration |

### Fixtures (conftest.py)

| Fixture | Provides |
|---------|----------|
| `temp_kb_dir` | Temp `.knowledge-db/` with `.git` marker |
| `sample_entries` | 3 V3 sample entries |
| `kb_with_entries` | KB dir with entries in JSONL |
| `project_root` | Parent of `.knowledge-db/` |

### Test Markers

- `@pytest.mark.integration` -- loads real ML models, skipped without `--integration`
- `@pytest.mark.benchmark` -- performance tests, requires `--benchmark-enable`

---

## Cross-Module Dependencies

```
cli.py
  -> db.py (KnowledgeDB)
  -> multi_search.py (MultiSourceSearch)
  -> search.py (FORMATTERS)
  -> capture.py (create_entry, create_entry_from_json, save_entry)
  -> server.py (KnowledgeServer, list_running_servers)
  -> ingest_docs.py (DocsIngester)
  -> ingest_sessions.py (SessionIngester)
  -> platform.py (find_project_root)
  -> utils.py (debug_log, is_server_running)

multi_search.py
  -> db.py (KnowledgeDB)
  -> query_utils.py (classify_query, expand_query, get_source_weights)
  -> utils.py (debug_log)

capture.py
  -> utils.py (debug_log, send_command)
  -> platform.py (find_project_root, get_runtime_dir)

ingest_docs.py
  -> db.py (KnowledgeDB)
  -> utils.py (debug_log, migrate_entry)

ingest_sessions.py
  -> db.py (KnowledgeDB)
  -> ingest_docs.py (load_sources_config, _resolve_paths)
  -> utils.py (debug_log)

server.py
  -> db.py (KnowledgeDB)
  -> platform.py (find_project_root, get_runtime_dir, ...)
  -> utils.py (debug_log)

db.py
  -> utils.py (debug_log, migrate_entry, SCHEMA_V3_DEFAULTS)
  -> platform.py (find_project_root)

utils.py
  -> platform.py (HOST, KB_DIR_NAME, get_kb_pid_file, get_kb_port_file)
```

### Dependency Layers

```
Layer 0 (no internal deps):  platform.py
Layer 1:                      utils.py, query_utils.py
Layer 2:                      db.py, search.py
Layer 3:                      multi_search.py, capture.py, server.py
Layer 4:                      ingest_docs.py, ingest_sessions.py
Layer 5 (top):                cli.py
```
