#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
MCP_FILESYSTEM_HOST="${MCP_FILESYSTEM_HOST:-127.0.0.1}"
MCP_FILESYSTEM_PORT="${MCP_FILESYSTEM_PORT:-19090}"
MTAGENTRISK_SHARED_WORKSPACE="${MTAGENTRISK_SHARED_WORKSPACE:-$REPO_ROOT/outputs/shared_workspace}"

mkdir -p "$MTAGENTRISK_SHARED_WORKSPACE"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export MTAGENTRISK_SHARED_WORKSPACE

exec "$PYTHON_BIN" -m redir.eval.mtagentrisk.mcp_server.filesystem_server \
  --host "$MCP_FILESYSTEM_HOST" \
  --port "$MCP_FILESYSTEM_PORT" \
  --workspace "$MTAGENTRISK_SHARED_WORKSPACE"
