# kln-knowledge-system

Hybrid semantic knowledge database with multi-source search. Captures, indexes, and retrieves project knowledge using dense embeddings (BGE-small-en-v1.5), sparse matching (BM42), and cross-encoder reranking.

## Features

- **Hybrid search**: Dense + sparse + RRF fusion + cross-encoder reranking (~30ms via TCP server)
- **Multi-source**: Search across curated KB, session transcripts, and documentation
- **Intent-aware**: Query classification adjusts source weights (debug queries favor sessions, howto queries favor docs)
- **Incremental ingestion**: SHA-256 registry tracks processed files, only re-processes changes
- **Per-project isolation**: Each project gets its own `.knowledge-db/` with separate sub-stores

## Installation

```bash
pip install kln-knowledge-system

# With PDF support
pip install "kln-knowledge-system[pdf]"
```

## Quick Start

```bash
# Capture knowledge
kln-kb capture "JWT tokens must be validated server-side" --type warning --tags "auth,security"

# Search
kln-kb search "authentication best practices"

# Search across all sources
kln-kb search "how to fix timeout" --source kb --source sessions --source docs

# Ingest documentation
kln-kb ingest docs --path ./docs/

# Ingest Claude Code sessions
kln-kb ingest sessions

# Start the TCP server for fast searches
kln-kb server start

# Show statistics
kln-kb stats
```

## Source Configuration

By default, the system auto-discovers docs from convention directories (`docs/`, `doc/`, `INFOS/`, `documentation/`) and sessions from `~/.claude/projects/`.

For explicit control, create a sources config:

```bash
kln-kb sources --init    # Creates .knowledge-db/sources.yaml template
kln-kb sources           # Show current configuration
```

```yaml
# .knowledge-db/sources.yaml
docs:
  paths:
    - docs/                    # relative to project root
    - ~/Desktop/INFOS/         # absolute path (~ expanded)
    - /shared/team-docs/       # another absolute path
  include: ["*.md", "*.txt", "*.pdf", "*.rst"]  # default if omitted
  exclude: ["drafts/**", "*.tmp"]

sessions:
  auto_discover: true          # scan ~/.claude/projects/ automatically
  # path: ~/custom/sessions/  # explicit override
```

Without a `sources.yaml`, convention-based discovery is used. The CLI `--path` flag always overrides config.

## Architecture

```
.knowledge-db/
├── sources.yaml             # Source configuration (optional)
├── entries.jsonl            # Curated KB entries
├── embeddings.npy           # Dense embeddings (384-dim BGE-small)
├── sparse_index.json        # BM42 sparse vectors
├── index.json               # ID -> row mapping
├── sessions/                # Ingested session transcripts
│   ├── entries.jsonl
│   ├── embeddings.npy
│   └── session-registry.json
└── docs/                    # Ingested documentation chunks
    ├── entries.jsonl
    ├── embeddings.npy
    └── doc-registry.json
```

### Search Pipeline

```
Query
  -> Intent classification (DEBUG/HOWTO/RECALL/EXPLORE)
  -> Synonym expansion
  -> Search each source (dense + sparse + RRF per source)
  -> Weighted RRF fusion across sources
  -> Cross-encoder reranking
  -> Results with source labels
```

## CLI Reference

### `kln-kb search`

```bash
kln-kb search "query"                          # Search curated KB
kln-kb search "query" -s kb -s sessions -s docs  # All sources
kln-kb search "query" -f compact               # Compact output
kln-kb search "query" -f json                  # JSON output
kln-kb search "query" -f inject                # Claude Code injection format
kln-kb search --id <entry-id>                  # Get specific entry
kln-kb search "query" --since 2026-01-01       # Date filter
kln-kb search "query" --type warning           # Type filter
kln-kb search "query" --branch main            # Branch filter
```

### `kln-kb capture`

```bash
kln-kb capture "insight text" --type finding
kln-kb capture "text" --type solution --tags "tag1,tag2" --priority high
kln-kb capture --json-input '{"title":"...", "insight":"...", "type":"warning"}'
```

Entry types: `finding`, `solution`, `pattern`, `warning`, `decision`, `discovery`

### `kln-kb ingest`

```bash
kln-kb ingest sessions              # Ingest Claude Code JSONL transcripts
kln-kb ingest docs --path ./docs/   # Ingest markdown/PDF files
kln-kb ingest all                   # Ingest from all sources
kln-kb ingest all --full            # Force re-processing everything
```

### `kln-kb server`

```bash
kln-kb server start     # Start TCP server (foreground)
kln-kb server stop      # Stop server
kln-kb server status    # Show running servers
```

### Other

```bash
kln-kb stats            # Database statistics
kln-kb stats --json     # JSON output
kln-kb rebuild          # Rebuild search index from JSONL
```

## Python API

```python
from kln_knowledge import KnowledgeDB, MultiSourceSearch

# Direct DB access
db = KnowledgeDB("/path/to/project")
results = db.search("query", limit=5)
entry_id = db.add({"title": "...", "insight": "...", "type": "finding"})

# Batch operations
ids = db.batch_add([entry1, entry2, entry3])
db.remove_entries(["id1", "id2"])

# Sub-stores
sessions_db = KnowledgeDB("/path/to/project", sub_store="sessions")

# Multi-source search with intent-aware weighting
ms = MultiSourceSearch("/path/to/project")
results = ms.search("how to configure auth", sources=["kb", "docs"])
```

## Development

```bash
git clone <repo>
cd kln-knowledge-system
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/ -v
```

## License

MIT
