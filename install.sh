#!/usr/bin/env bash
set -euo pipefail

# KnowlinMCP installer
# Usage: ./install.sh [--global] [--with-pdf]
#
# Default: installs in .venv with MCP support
# --global: pip install into current Python environment
# --with-pdf: include PDF ingestion support (pymupdf4llm)

GLOBAL=false
PDF=false

for arg in "$@"; do
    case $arg in
        --global)  GLOBAL=true ;;
        --with-pdf) PDF=true ;;
        -h|--help)
            echo "Usage: ./install.sh [--global] [--with-pdf]"
            echo ""
            echo "  --global    Install into current Python (no venv)"
            echo "  --with-pdf  Include PDF support (pymupdf4llm)"
            exit 0
            ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

EXTRAS="mcp"
if $PDF; then
    EXTRAS="mcp,pdf"
fi

echo "Installing KnowlinMCP..."

if $GLOBAL; then
    pip install -e ".[$EXTRAS]"
else
    if [ ! -d .venv ]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
    fi
    echo "Installing dependencies..."
    .venv/bin/pip install -q -e ".[$EXTRAS]"

    # Ensure knowlin and knowlin-mcp are on PATH
    VENV_BIN="$(cd .venv/bin && pwd)"
    if ! echo "$PATH" | grep -q "$VENV_BIN"; then
        echo ""
        echo "Add to your shell profile:"
        echo "  export PATH=\"$VENV_BIN:\$PATH\""
        echo ""
        echo "Or activate the venv:"
        echo "  source .venv/bin/activate"
    fi
fi

echo ""
echo "Installed. To set up a project:"
echo ""
echo "  cd /your/project"
echo "  knowlin init"
echo ""
echo "This creates .knowledge-db/sources.yaml and .mcp.json."
echo "Then: knowlin ingest all && knowlin search \"your query\""
