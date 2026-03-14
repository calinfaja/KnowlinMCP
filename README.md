# KnowlinMCP

Per-project knowledge database with hybrid semantic search, exposed as an MCP server. Captures insights, indexes docs and session transcripts, retrieves them with dense + sparse + reranking search.

```
                          KnowlinMCP
  +-----------+     +-------------------+     +-----------+
  | Claude    |     |   MCP Server      |     | .knowledge-db/
  | Gemini    |<--->|   (stdio)         |<--->|   entries.jsonl
  | Codex     |     |  knowlin_search   |     |   embeddings.npy
  | Cursor    |     |  knowlin_get      |     |   sessions/
  | VS Code   |     |  knowlin_stats    |     |   docs/
  +-----------+     |  knowlin_ingest   |     +-----------+
                    +-------------------+
                            |
  +-------------------------+-------------------------+
  |                         |                         |
  v                         v                         v
  Dense Search          Sparse Search           Cross-encoder
  (BGE-small 384d)      (SPLADE++ sparse)        Reranker
  |                         |                         |
  +------------+------------+                         |
               |                                      |
               v                                      |
         RRF Fusion -------> Intent Weighting ------->+
         (per source)        DEBUG -> sessions
                             HOWTO -> docs
                             RECALL -> sessions
```

## Install

Requires **Python 3.9+**.

```bash
git clone https://github.com/calinfaja/KnowlinMCP.git && cd KnowlinMCP
./install.sh              # creates .venv, installs deps + MCP support
./install.sh --with-pdf   # also install PDF ingestion
```

Or manually:
```bash
pip install -e ".[mcp]"
```

## Quick Start (30 seconds)

```bash
# Activate the venv (or add .venv/bin to your PATH)
source .venv/bin/activate

# 1. Initialize in your project
cd /your/project
knowlin init                    # creates .knowledge-db/ + .mcp.json

# 2. Index your docs
knowlin ingest all              # indexes docs/ and Claude sessions
                                # First run downloads ~200MB of ML models

# 3. Search
knowlin search "authentication"
```

Example output:
```
1. [kb:warning] JWT tokens must be validated server-side (87%, 2026-01-10)
   Client-side JWT validation is bypassable; always verify on the server.
2. [docs:finding] OAuth2 PKCE flow for single-page apps (72%, 2026-02-15)
   Use PKCE instead of implicit grant for browser-based OAuth2 flows.
```

That's it. Claude Code (and other MCP clients) can now use `knowlin_search` automatically via the `.mcp.json` created by `init`.

## How It Works

**Three sources, one search:**

| Source | What | Auto-discovered from |
|--------|------|---------------------|
| `kb` | Manually captured insights | `knowlin capture "..."` |
| `docs` | Markdown, PDF, text files | `docs/`, `doc/`, or `sources.yaml` paths |
| `sessions` | Claude Code transcripts | `~/.claude/projects/` |

**Search pipeline:** Every query is classified by intent (debug? howto? recall?), then searched with dense embeddings + sparse keywords + RRF fusion per source, fused across sources with intent-aware weights, and reranked with a cross-encoder. ~30ms via TCP server.

**Incremental ingestion:** SHA-256 file hashing tracks what's been processed. Only new or changed files are re-indexed. Run `knowlin ingest all` anytime -- it's fast.

## CLI

```bash
# Search (default: compact format, all sources)
knowlin search "query"
knowlin search "query" -s kb -s docs       # specific sources
knowlin search "query" -f detailed         # verbose output
knowlin search "query" -f json             # machine-readable
knowlin search "query" --type warning      # filter by type
knowlin search "query" --since 2026-01-01  # date filter

# Capture knowledge
knowlin capture "JWT must be validated server-side" --type warning --tags "auth,jwt"

# Ingest
knowlin ingest all              # docs + sessions (incremental)
knowlin ingest docs             # docs only
knowlin ingest sessions         # sessions only
knowlin ingest all --full       # force re-process everything

# Browse & manage
knowlin list                    # recent entries across all sources
knowlin get <id>                # full details of an entry
knowlin delete <id>             # remove an entry

# Admin
knowlin init                    # set up project (.knowledge-db/ + .mcp.json)
knowlin stats                   # entry counts per source
knowlin doctor --fix            # health check and auto-repair
knowlin sources --init          # create sources.yaml template
knowlin server start            # TCP server for ~30ms queries (foreground)
```

Entry types: `finding`, `solution`, `pattern`, `warning`, `decision`, `discovery`

**Environment variables:**
- `CLAUDE_PROJECT_DIR` -- override project root detection (useful in CI or nested subdirs)
- `KNOWLIN_DEBUG` -- enable debug logging to stderr

## Source Configuration

Without config, KnowlinMCP auto-discovers `docs/`, `doc/`, `INFOS/` directories and Claude sessions from `~/.claude/projects/`.

For explicit control, edit `.knowledge-db/sources.yaml` (created by `knowlin init`):

```yaml
docs:
  paths:
    - docs/                       # relative to project root
    - ~/Desktop/INFOS/            # absolute path (~ expanded)
  # include: ["*.md", "*.txt", "*.pdf", "*.rst"]
  # exclude: ["drafts/**", "*.tmp"]

sessions:
  auto_discover: true             # scan ~/.claude/projects/
```

## MCP Server

`knowlin init` writes `.mcp.json` for Claude Code. For other clients:

<details>
<summary>Gemini CLI, Codex, Cursor, VS Code</summary>

**Gemini CLI** (`~/.gemini/settings.json`):
```json
{ "mcpServers": { "knowlin-mcp": { "command": "knowlin-mcp" } } }
```

**Codex CLI** (`~/.codex/config.toml`):
```toml
[mcp_servers.knowlin-mcp]
command = "knowlin-mcp"
```

**Cursor** (`.cursor/mcp.json`):
```json
{ "mcpServers": { "knowlin-mcp": { "command": "knowlin-mcp" } } }
```

**VS Code** (`.vscode/mcp.json`):
```json
{ "servers": { "knowlin-mcp": { "type": "stdio", "command": "knowlin-mcp" } } }
```

</details>

5 tools exposed: `knowlin_search`, `knowlin_get`, `knowlin_capture`, `knowlin_stats`, `knowlin_ingest`.

## Python API

```python
from knowlin_mcp import KnowledgeDB, MultiSourceSearch

db = KnowledgeDB("/path/to/project")
results = db.search("query", limit=5)

ms = MultiSourceSearch("/path/to/project")
results = ms.search("how to configure auth", sources=["kb", "docs"])
```

## Storage

```
.knowledge-db/
  sources.yaml              # source config (optional)
  entries.jsonl              # curated KB (source of truth)
  embeddings.npy             # dense vectors (384-dim)
  sparse_index.json          # SPLADE++ sparse vectors
  sessions/                  # ingested session transcripts
    entries.jsonl, embeddings.npy, session-registry.json
  docs/                      # ingested documentation chunks
    entries.jsonl, embeddings.npy, doc-registry.json
```

## Development

```bash
git clone https://github.com/calinfaja/KnowlinMCP.git && cd KnowlinMCP
./install.sh
.venv/bin/pytest tests/ -v           # unit tests
.venv/bin/ruff check src/ tests/     # lint
.venv/bin/black src/ tests/          # format
```

## License

MIT
