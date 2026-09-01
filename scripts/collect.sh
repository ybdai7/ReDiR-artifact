#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${REDIR_PYTHON:-}" ]]; then
  PYTHON_BIN="$REDIR_PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || true)"
fi

[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
  echo "Python was not found; activate the artifact environment or set REDIR_PYTHON" >&2
  exit 2
}

"$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 2)' || {
  echo "ReDiR collection requires Python 3.10 or newer" >&2
  exit 2
}

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m redir.collect "$@"
