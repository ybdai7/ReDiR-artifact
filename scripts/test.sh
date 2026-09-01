#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/test.sh <MT-AgentRisk-task-directory> [output-directory]

Required environment variables:
  REDIR_MODEL_PATH        Frozen base-model directory
  REDIR_CHECKPOINT_PATH   Trained ReDiR checkpoint

Optional environment variables:
  REDIR_PYTHON            Python executable
  REDIR_DEVICE            Reasoner device (default: cuda)
  REDIR_WEAVER_DEVICE     Weaver device (default: REDIR_DEVICE)
  REDIR_REUSE_SERVER      Use an existing endpoint on port 8010 when set to 1
  REDIR_SERVER_TIMEOUT    Health-check timeout in seconds (default: 180)
  MTAGENTRISK_CONFIG      Evaluation TOML config
  SERVER_HOST             Hostname passed to MT-AgentRisk (default: localhost)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TASK_PATH="${1:-}"
OUTPUT_DIR="${2:-$REPO_ROOT/outputs/test}"
SERVER_CONFIG="${REDIR_SERVER_CONFIG:-$REPO_ROOT/configs/redir_server.yaml}"
EVAL_CONFIG="${MTAGENTRISK_CONFIG:-$REPO_ROOT/configs/mtagentrisk_local.example.toml}"
SERVER_HOST="${SERVER_HOST:-localhost}"
SERVER_URL="http://127.0.0.1:8010"
SERVER_TIMEOUT="${REDIR_SERVER_TIMEOUT:-180}"
REUSE_SERVER="${REDIR_REUSE_SERVER:-0}"
VENDOR_PYTHONPATH="$REPO_ROOT/third_party/openhands:$REPO_ROOT/third_party/mcpmark"

[[ -n "$TASK_PATH" ]] || {
  usage >&2
  exit 2
}
[[ -d "$TASK_PATH" ]] || {
  echo "MT-AgentRisk task directory not found: $TASK_PATH" >&2
  exit 2
}
[[ -n "${REDIR_MODEL_PATH:-}" ]] || {
  echo "REDIR_MODEL_PATH is required" >&2
  exit 2
}
[[ -n "${REDIR_CHECKPOINT_PATH:-}" ]] || {
  echo "REDIR_CHECKPOINT_PATH is required" >&2
  exit 2
}
[[ -d "$REDIR_CHECKPOINT_PATH" ]] || {
  echo "REDIR_CHECKPOINT_PATH does not point to a directory: $REDIR_CHECKPOINT_PATH" >&2
  exit 2
}
[[ -f "$SERVER_CONFIG" ]] || {
  echo "Server config not found: $SERVER_CONFIG" >&2
  exit 2
}
[[ -f "$EVAL_CONFIG" ]] || {
  echo "Evaluation config not found: $EVAL_CONFIG" >&2
  exit 2
}

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

mkdir -p "$OUTPUT_DIR"
SERVER_PID=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$OUTPUT_DIR/server.pid"
  exit "$status"
}
trap cleanup EXIT INT TERM

if [[ "$REUSE_SERVER" != "1" ]]; then
  command -v curl >/dev/null 2>&1 || {
    echo "curl is required for the server health check" >&2
    exit 2
  }

  echo "Starting ReDiR endpoint at $SERVER_URL/v1"
  env PYTHONPATH="$REPO_ROOT/src:$VENDOR_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m latent_safety.server.openai_compatible \
      --backend latent \
      --config "$SERVER_CONFIG" \
      --host 127.0.0.1 \
      --port 8010 \
      --device "${REDIR_DEVICE:-cuda}" \
      --weaver-device "${REDIR_WEAVER_DEVICE:-${REDIR_DEVICE:-cuda}}" \
      >"$OUTPUT_DIR/server.log" 2>&1 &
  SERVER_PID=$!
  printf '%s\n' "$SERVER_PID" > "$OUTPUT_DIR/server.pid"

  deadline=$(( SECONDS + SERVER_TIMEOUT ))
  until curl --fail --silent "$SERVER_URL/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "ReDiR server exited before becoming healthy" >&2
      tail -n 80 "$OUTPUT_DIR/server.log" >&2 || true
      exit 3
    fi
    if (( SECONDS >= deadline )); then
      echo "ReDiR server did not become healthy within ${SERVER_TIMEOUT}s" >&2
      tail -n 80 "$OUTPUT_DIR/server.log" >&2 || true
      exit 3
    fi
    sleep 2
  done
else
  echo "Using existing ReDiR endpoint at $SERVER_URL/v1"
fi

echo "Running MT-AgentRisk task: $TASK_PATH"
mkdir -p "$OUTPUT_DIR/shared_workspace"
env \
  PYTHONPATH="$REPO_ROOT/src:$VENDOR_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
  MTAGENTRISK_SHARED_WORKSPACE="$OUTPUT_DIR/shared_workspace" \
  MCP_FS_DEST_DIR="$OUTPUT_DIR/shared_workspace" \
  NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0}" \
  no_proxy="${no_proxy:-${NO_PROXY:-localhost,127.0.0.1,0.0.0.0}}" \
  "$PYTHON_BIN" -m latent_safety.eval.mtagentrisk.run_eval \
    --task-path "$TASK_PATH" \
    --agent-llm-config agent \
    --agent-llm-config-file "$EVAL_CONFIG" \
    --env-llm-config env \
    --env-llm-config-file "$EVAL_CONFIG" \
    --outputs-path "$OUTPUT_DIR/evaluation" \
    --server-hostname "$SERVER_HOST"
