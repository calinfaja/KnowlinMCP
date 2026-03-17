# Changelog

## [Unreleased]

### Fixed
- MCP capture path type bug
- remove_entries row alignment after mid-list delete
- Semantic dedup using cosine similarity instead of RRF scores
- Docs incremental ingest preserving unchanged chunks
- Registry ID drift when batch_add filters entries
- Source routing for single-source TCP search and CLI kb mapping
- TCP server authentication with per-project token
- PID validation before process kill
- Secure /tmp runtime directory (0700, ownership, symlink protection)
- Structured logging via Python logging module
- sources.yaml schema validation
- Symlink traversal protection in doc ingestion
- Unbounded TCP limit clamping
- Error message sanitization (no path leaks)
- JSONL append durability (fsync)
- Silent exception swallowing in CLI commands

### Added
- Concurrent access safety (threading.RLock in KnowledgeDB)
- Atomic index writes (temp+fsync+rename)
- embeddings.npy size guard
- Auto-rebuild index from JSONL on corruption
- codex support in TCP ingest and sources.yaml template
- doctor validation for sparse_index.json and codex-registry
- AGPL license note for PDF extra
- py.typed marker (PEP 561)
- CONTRIBUTING.md and CHANGELOG.md
- CI matrix expanded (Python 3.9-3.13, Ubuntu + macOS)
- Install smoke test in CI
- PyPI classifiers and keywords

## [0.1.0] - 2026-01-15

- Initial release
