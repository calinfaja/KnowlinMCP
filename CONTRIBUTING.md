# Contributing to KnowlinMCP

## Setup

```bash
git clone https://github.com/calinfaja/KnowlinMCP.git && cd KnowlinMCP
./install.sh
```

## Development

```bash
.venv/bin/pytest tests/ -v          # run tests
.venv/bin/ruff check src/ tests/    # lint
.venv/bin/black src/ tests/         # format
```

## Pull Requests

- One concern per PR
- Include tests for bug fixes and new features
- Run lint + tests before submitting
- Use conventional commits: `type(scope): description`
