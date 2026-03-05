# KnowlinMCP -- MCP Server Plan

> Making knowlin-mcp available as an MCP server for Claude, Gemini, Codex, Cursor, VS Code/Copilot, Windsurf, Cline, Continue.dev, and JetBrains AI.

---

## Executive Summary

KLN already has the hard part (hybrid search engine with BGE-small + BM42 + RRF + cross-encoder reranking). The MCP layer is a **thin adapter** (~150-200 lines) wrapping `KnowledgeDB` and `MultiSourceSearch` with the official Python MCP SDK's FastMCP decorators. No architectural changes needed. The server exposes **4 read-only tools** -- agents consume knowledge, not write it.

**Effort estimate:** Single file addition + pyproject.toml changes. The existing TCP server (`server.py`) and CLI (`cli.py`) remain untouched.

---

## 1. Client Compatibility Matrix

All major AI agents now support MCP. stdio transport is universal.

| Client | Config Format | Root Key | Transports | Status |
|--------|-------------|----------|------------|--------|
| Claude Code CLI | `.mcp.json` | `mcpServers` | stdio, SSE, HTTP | Full |
| Claude Desktop | `claude_desktop_config.json` | `mcpServers` | stdio | Full |
| Gemini CLI | `~/.gemini/settings.json` | `mcpServers` | stdio, SSE, HTTP | Full |
| OpenAI Codex CLI | `~/.codex/config.toml` | `[mcp_servers.*]` | stdio, HTTP | Full |
| Cursor | `.cursor/mcp.json` | `mcpServers` | stdio, SSE, HTTP | Full |
| VS Code / Copilot | `.vscode/mcp.json` | `servers` (!) | stdio, SSE, HTTP | GA |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` | `mcpServers` | stdio, SSE, HTTP | Full |
| Cline | `cline_mcp_settings.json` | `mcpServers` | stdio, SSE | Full |
| Continue.dev | `.continue/mcpServers/*.yaml` | YAML | stdio, SSE, HTTP | Full |
| JetBrains AI | IDE settings | IDE-managed | stdio, SSE, HTTP | Full (2025.2+) |

**Decision:** Ship with **stdio transport** (universal). Add streamable-http later if needed for remote/cloud use.

---

## 2. Tool Surface Design (4 tools)

Research consensus: **5-15 tools per server**. More than 15 causes tool selection confusion and context bloat. Fewer workflow-shaped tools outperform many narrow API-mirror tools.

### The 4 tools

```
knowlin_search(query, limit?, sources?, since?, until?, type?)
  Read-only. The primary tool. Wraps MultiSourceSearch.search().

knowlin_get(entry_id)
  Read-only. Fetch a single entry by ID. Wraps KnowledgeDB.get().

knowlin_stats()
  Read-only. Returns entry counts per source, DB health. Wraps MultiSourceSearch.stats().

knowlin_ingest(source?)
  Write. Triggers ingestion of docs/sessions. Wraps DocsIngester/SessionIngester.ingest().
```

### Why these 4 and not more

| Rejected tool | Reason |
|--------------|--------|
| `knowlin_capture` | Write operation. Agents don't spontaneously capture knowledge -- it's a human-curated activity. Existing `/kln:learn` and `/kln:remember` hooks handle capture. Add later if cross-agent write demand emerges. |
| `knowlin_dense_search` / `knowlin_sparse_search` | Internal implementation detail. The LLM doesn't need to choose search algorithm. `knowlin_search` handles fusion internally. |
| `knowlin_related` | Can be achieved with `knowlin_search` using content from `knowlin_get` result. |
| `knowlin_recent` | Achievable via `knowlin_search` with `since` parameter. |
| `knowlin_rebuild` | Admin operation. Run via CLI, not agent. |
| `knowlin_doctor` | Admin operation. Run via CLI. |
| `knowlin_delete` | Destructive. Agents shouldn't delete knowledge autonomously. |

### Why these names

- Prefixed with `knowlin_` to avoid collisions when agents have multiple MCP servers connected.
- Verb-based: `search`, `get`, `stats`, `ingest`.
- Flat parameter schemas (no nested objects) -- LLMs handle these reliably.

---

## 3. Tool Specifications

### `knowlin_search`

```python
@mcp.tool()
def knowlin_search(
    query: str,
    limit: int = 5,
    sources: str = "all",     # "all", "kb", "sessions", "docs", or comma-separated
    since: str = "",          # YYYY-MM-DD
    until: str = "",          # YYYY-MM-DD
    type: str = "",           # finding|solution|pattern|warning|decision|discovery
) -> str:
    """Search the project knowledge base using hybrid semantic search.
    Searches across curated KB entries, session transcripts, and documentation.
    Returns ranked results with relevance scores. Use 'sources' to narrow scope.
    Example: knowlin_search("authentication timeout fix", sources="kb,sessions")"""
```

**Response format (Markdown, not JSON -- ~30% more token-efficient):**

```markdown
## 3 results for "authentication timeout fix" [kb, sessions]

### 1. JWT Token Refresh on Timeout (0.89) [kb]
**Type:** solution | **Date:** 2026-02-15
Implement token refresh interceptor that catches 401 responses and retries with a new token...

### 2. Auth Service Timeout Investigation (0.76) [sessions]
**Type:** finding | **Date:** 2026-01-20
The auth service timeout was caused by connection pool exhaustion under load...

### 3. OAuth2 Connection Pooling (0.71) [docs]
**Type:** pattern | **Date:** 2026-01-10
Configure max_connections=50 and timeout=30s for the OAuth2 HTTP client...
```

**Design rationale:**
- Markdown is readable by both humans and LLMs
- Scores help the agent assess confidence
- Source tags (`[kb]`, `[sessions]`, `[docs]`) give provenance
- Truncate insight to ~300 chars with `[...]` for long entries
- Include `entry_id` in response so the agent can call `knowlin_get` for full content

### `knowlin_get`

```python
@mcp.tool()
def knowlin_get(entry_id: str) -> str:
    """Retrieve the full content of a knowledge entry by its ID.
    Use after knowlin_search to get complete details of a specific result."""
```

**Response:** Full entry as formatted Markdown (all fields).

### `knowlin_stats`

```python
@mcp.tool()
def knowlin_stats() -> str:
    """Show knowledge base statistics: entry counts per source, last updated, DB health."""
```

**Response:**
```markdown
## Knowledge Base Stats
- **KB entries:** 142 (last updated: 2026-03-04)
- **Session entries:** 1,203 (last updated: 2026-03-03)
- **Doc chunks:** 567 (last updated: 2026-03-01)
- **Total:** 1,912 entries across 3 sources
```

### `knowlin_ingest`

```python
@mcp.tool()
def knowlin_ingest(source: str = "all") -> str:
    """Ingest documents or session transcripts into the knowledge base.
    Source: 'docs', 'sessions', or 'all'. Only processes new/changed files."""
```

**Response:** `"Ingested 23 new doc chunks, 5 new session entries (skipped 142 unchanged)"`

---

## 4. Implementation Plan

### File: `src/knowlin_mcp/mcp_server.py` (~150-200 lines)

```python
"""KnowlinMCP MCP Server.

Exposes hybrid semantic search over project knowledge via the Model Context Protocol.
Compatible with Claude, Gemini, Codex, Cursor, VS Code, and other MCP clients.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from knowlin_mcp.db import KnowledgeDB
from knowlin_mcp.multi_search import MultiSourceSearch
# ... etc

mcp = FastMCP(
    "knowlin-mcp",
    instructions="Knowledge base with hybrid semantic search. Use knowlin_search to find "
    "solutions, patterns, and decisions.",
)

@mcp.tool()
def knowlin_search(...) -> str:
    # Wraps MultiSourceSearch.search(), formats as Markdown
    ...

# ... remaining tools

def main():
    mcp.run()  # stdio transport (default)

if __name__ == "__main__":
    main()
```

### Changes to `pyproject.toml`

```toml
# Add to dependencies:
dependencies = [
    # ... existing deps ...
    "mcp>=1.2.0",          # MCP SDK (adds pydantic, anyio, starlette)
]

# Add to [project.optional-dependencies]:
[project.optional-dependencies]
mcp = ["mcp>=1.2.0"]       # Alternative: make it optional

# Add entry point:
[project.scripts]
knowlin = "knowlin_mcp.cli:main"
knowlin-mcp = "knowlin_mcp.mcp_server:main"    # NEW
```

### No other files change

The MCP server imports from existing modules. No refactoring needed. The existing TCP server (`server.py`) and CLI (`cli.py`) continue to work independently.

---

## 5. Client Configuration Examples

### Claude Code (`.mcp.json` at project root)

```json
{
  "mcpServers": {
    "knowlin-mcp": {
      "command": "knowlin-mcp",
      "env": {}
    }
  }
}
```

### Gemini CLI (`~/.gemini/settings.json`)

```json
{
  "mcpServers": {
    "knowlin-mcp": {
      "command": "knowlin-mcp"
    }
  }
}
```

### OpenAI Codex CLI (`~/.codex/config.toml`)

```toml
[mcp_servers.knowlin-mcp]
command = "knowlin-mcp"
```

### Cursor (`.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "knowlin-mcp": {
      "command": "knowlin-mcp"
    }
  }
}
```

### VS Code / Copilot (`.vscode/mcp.json`)

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

### uvx (without pre-installing)

After publishing to PyPI, all configs become:
```json
{ "command": "uvx", "args": ["knowlin-mcp"] }
```

---

## 6. Cross-Platform Considerations

### What the MCP Python SDK handles automatically (>=1.6.0)

- UTF-8 encoding on Windows (re-wraps stdout/stdin binary streams)
- CRLF line ending normalization
- Process lifecycle management

### What we must handle in our code

| Concern | Solution |
|---------|----------|
| Path separators | Use `pathlib.Path` throughout (already done in KLN) |
| Project root detection | `find_project_root()` in `platform.py` already uses `Path` (cross-platform) |
| Config/cache dirs | `platformdirs` already used in `platform.py` |
| No stdout pollution | MCP server file has no `print()` calls; all logging via `debug_log()` to stderr |
| Windows `cmd /c` spawn | Not our problem -- the client handles this. Our entry point is a proper `[project.scripts]` executable. |
| Python version | MCP SDK requires >=3.10. KLN currently targets >=3.9. **Bump to >=3.10.** |

### Python version bump: 3.9 -> 3.10

The MCP SDK requires Python 3.10+. Two options:

**Option A (recommended): Make MCP an optional dependency**
```toml
requires-python = ">=3.9"           # keep existing
[project.optional-dependencies]
mcp = ["mcp>=1.2.0"]               # opt-in
```
Users who only need CLI/TCP server stay on 3.9. MCP users need 3.10+.

**Option B: Bump minimum to 3.10**
```toml
requires-python = ">=3.10"
dependencies = [..., "mcp>=1.2.0"]
```
Simpler, but drops Python 3.9 support. 3.9 EOL was Oct 2025, so this is reasonable.

---

## 7. Context Efficiency Design

Research shows tool descriptions + schemas consume 50-1000 tokens per tool. Our 4 tools with concise descriptions will use ~400 tokens total -- well under the "danger zone" of 5,000+ tokens.

### Response size controls

| Control | Implementation |
|---------|---------------|
| Default limit | `knowlin_search` returns max 5 results (configurable via `limit`) |
| Truncation | Insight field truncated to 300 chars in search results |
| Full content | Available via `knowlin_get(id)` on demand |
| Markdown format | ~30% more token-efficient than JSON for prose |
| No metadata bloat | Strip internal fields (`_search_meta`, `_sparse_vectors`, embeddings) from responses |
| Source tags | Inline `[kb]`/`[sessions]`/`[docs]` tags instead of separate metadata objects |

### What NOT to expose to agents

- Raw embedding vectors
- Sparse index weights
- RRF fusion scores / internal rank metadata
- File system paths to JSONL/npy files
- Server PID/port information
- Registry hashes

---

## 8. Testing Strategy

### Unit tests (`tests/test_mcp_server.py`)

- Test each tool function directly (they're regular Python functions)
- Mock `KnowledgeDB` and `MultiSourceSearch` to avoid model loading
- Test Markdown formatting of responses
- Test error handling (empty DB, invalid entry_id, bad parameters)

### Integration test

- Use `mcp dev` inspector to verify tools/list and tools/call work over stdio
- Test with actual fastembed models loaded (mark as `@pytest.mark.integration`)

### Manual smoke test

```bash
# Start MCP inspector
npx -y @modelcontextprotocol/inspector
# Or:
mcp dev src/knowlin_mcp/mcp_server.py
```

---

## 9. Implementation Sequence

```
1. Add `mcp>=1.2.0` to pyproject.toml optional deps [mcp]
2. Create src/knowlin_mcp/mcp_server.py
   a. FastMCP instance with server instructions
   b. knowlin_search tool (wraps MultiSourceSearch)
   c. knowlin_get tool (wraps KnowledgeDB.get)
   d. knowlin_stats tool (wraps MultiSourceSearch.stats)
   e. knowlin_ingest tool (wraps DocsIngester + SessionIngester)
   f. main() entry point
3. Add knowlin-mcp entry point to pyproject.toml [project.scripts]
4. pip install -e ".[mcp]" and test with mcp dev
5. Add tests/test_mcp_server.py
6. Add .mcp.json example to repo root (gitignored or as .mcp.json.example)
7. Update README with MCP section
```

**Total new code: ~150-200 lines in mcp_server.py + ~100 lines in test_mcp_server.py**

---

## 10. What We Are NOT Doing (YAGNI)

| Deferred | Reason |
|----------|--------|
| `knowlin_capture` (v1) | Write operations deferred. Agents benefit from reading the KB, not writing to it autonomously. Existing CLI and hooks handle capture. Add if cross-agent write demand emerges. |
| Streamable HTTP transport | stdio covers all current clients. Add when there's a cloud deployment need. |
| MCP Resources | Tools are sufficient. Resources add complexity for marginal benefit here. |
| MCP Prompts | No pre-built prompt templates needed. Agents compose their own queries. |
| OAuth / auth | Local server via stdio. No auth needed for localhost. |
| Tool pagination cursors | 5-result default with configurable limit is sufficient. |
| Dynamic tool registration | Fixed 4-tool set. No need for runtime tool discovery. |
| WebSocket transport | Not widely adopted. stdio + HTTP cover everything. |
| Docker packaging | pip/uvx is simpler and sufficient for a Python library. Docker if demand arises. |
| Separate MCP config file | Use existing `.knowledge-db/` and `sources.yaml`. No new config. |

---

## Sources

- [MCP Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)
- [MCP Python SDK (mcp on PyPI, v1.26.0)](https://github.com/modelcontextprotocol/python-sdk)
- [Anthropic: Writing Tools for Agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- [Block Engineering: Playbook for Designing MCP Servers](https://engineering.block.xyz/blog/blocks-playbook-for-designing-mcp-servers)
- [Philipp Schmid: MCP Best Practices](https://www.philschmid.de/mcp-best-practices)
- [Layered.dev: MCP Tool Schema Bloat](https://layered.dev/mcp-tool-schema-bloat-the-hidden-token-tax-and-how-to-fix-it/)
- [Jenova AI: Tool Overload -- 5-7 Tools = 92% Accuracy](https://www.jenova.ai/en/resources/mcp-tool-scalability-problem)
- [Klavis.ai: Less is More -- MCP Design Patterns](https://www.klavis.ai/blog/less-is-more-mcp-design-patterns-for-ai-agents)
- [Gemini CLI MCP docs](https://google-gemini.github.io/gemini-cli/docs/tools/mcp-server.html)
- [OpenAI Codex MCP docs](https://developers.openai.com/codex/mcp/)
- [VS Code MCP GA announcement](https://github.blog/changelog/2025-07-14-model-context-protocol-mcp-support-in-vs-code-is-generally-available/)
- [Cursor MCP docs](https://cursor.com/docs/context/mcp)
