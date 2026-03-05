# KnowlinMCP

Hybrid semantic knowledge database with MCP server and multi-source search. Captures, indexes, and retrieves project knowledge using dense embeddings (BGE-small-en-v1.5), sparse matching (BM42), and cross-encoder reranking.

## Features

- **MCP server**: Expose knowledge search to Claude, Gemini, Codex, Cursor, VS Code, and other MCP clients
- **Hybrid search**: Dense + sparse + RRF fusion + cross-encoder reranking (~30ms via TCP server)
- **Multi-source**: Search across curated KB, session transcripts, and documentation
- **Intent-aware**: Query classification adjusts source weights (debug queries favor sessions, howto queries favor docs)
- **Incremental ingestion**: SHA-256 registry tracks processed files, only re-processes changes
- **Per-project isolation**: Each project gets its own `.knowledge-db/` with separate sub-stores

## Installation

```bash
pip install knowlin-mcp

# With MCP server support
pip install "knowlin-mcp[mcp]"

# With PDF support
pip install "knowlin-mcp[pdf]"
```

## MCP Server

KnowlinMCP exposes 4 tools to any MCP client via stdio transport:

| Tool | Description |
|------|-------------|
| `knowlin_search` | Hybrid semantic + keyword search with source/date/type filtering |
| `knowlin_get` | Retrieve full entry details by ID |
| `knowlin_stats` | Database statistics (entry counts, sizes, health) |
| `knowlin_ingest` | Trigger docs/sessions ingestion |

### Client Configuration

**Claude Code** (`.mcp.json` at project root):
```json
{
  "mcpServers": {
    "knowlin-mcp": {
      "command": "knowlin-mcp"
    }
  }
}
```

**Gemini CLI** (`~/.gemini/settings.json`):
```json
{
  "mcpServers": {
    "knowlin-mcp": {
      "command": "knowlin-mcp"
    }
  }
}
```

**OpenAI Codex CLI** (`~/.codex/config.toml`):
```toml
[mcp_servers.knowlin-mcp]
command = "knowlin-mcp"
```

**Cursor** (`.cursor/mcp.json`):
```json
{
  "mcpServers": {
    "knowlin-mcp": {
      "command": "knowlin-mcp"
    }
  }
}
```

**VS Code / Copilot** (`.vscode/mcp.json`):
```json
{
  "servers": {
    "knowlin-mcp": {
      "type": "stdio",
      "command": "knowlin-mcp"
    }
  }
}
```

## Quick Start

```bash
# Capture knowledge
knowlin capture "JWT tokens must be validated server-side" --type warning --tags "auth,security"

# Search
knowlin search "authentication best practices"

# Search across all sources
knowlin search "how to fix timeout" --source kb --source sessions --source docs

# Ingest documentation
knowlin ingest docs --path ./docs/

# Ingest Claude Code sessions
knowlin ingest sessions

# Start the TCP server for fast searches
knowlin server start

# Show statistics
knowlin stats
```

## Source Configuration

By default, the system auto-discovers docs from convention directories (`docs/`, `doc/`, `INFOS/`, `documentation/`) and sessions from `~/.claude/projects/`.

For explicit control, create a sources config:

```bash
knowlin sources --init    # Creates .knowledge-db/sources.yaml template
knowlin sources           # Show current configuration
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

### `knowlin search`

```bash
knowlin search "query"                          # Search curated KB
knowlin search "query" -s kb -s sessions -s docs  # All sources
knowlin search "query" -f compact               # Compact output
knowlin search "query" -f json                  # JSON output
knowlin search "query" -f inject                # LLM injection format
knowlin search --id <entry-id>                  # Get specific entry
knowlin search "query" --since 2026-01-01       # Date filter
knowlin search "query" --type warning           # Type filter
knowlin search "query" --branch main            # Branch filter
```

### `knowlin capture`

```bash
knowlin capture "insight text" --type finding
knowlin capture "text" --type solution --tags "tag1,tag2" --priority high
knowlin capture --json-input '{"title":"...", "insight":"...", "type":"warning"}'
```

Entry types: `finding`, `solution`, `pattern`, `warning`, `decision`, `discovery`

### `knowlin ingest`

```bash
knowlin ingest sessions              # Ingest Claude Code JSONL transcripts
knowlin ingest docs --path ./docs/   # Ingest markdown/PDF files
knowlin ingest all                   # Ingest from all sources
knowlin ingest all --full            # Force re-processing everything
```

### `knowlin server`

```bash
knowlin server start     # Start TCP server (foreground)
knowlin server stop      # Stop server
knowlin server status    # Show running servers
```

### Other

```bash
knowlin stats            # Database statistics
knowlin stats --json     # JSON output
knowlin rebuild          # Rebuild search index from JSONL
```

## Python API

```python
from knowlin_mcp import KnowledgeDB, MultiSourceSearch

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
cd knowlin-mcp
python -m venv .venv
.venv/bin/pip install -e ".[dev,mcp]"
.venv/bin/pytest tests/ -v
```

## License

MIT
