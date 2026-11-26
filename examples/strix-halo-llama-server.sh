#!/usr/bin/env bash
# Reference helper: run a small fast Qwen3-4B-Instruct llama-server on :8081
# with 4-slot continuous batching (~3.5 GB VRAM/GTT), safe to run alongside a
# larger resident model on :8080.
#
# Usage:
#   LLAMA_SERVER_BIN=/path/to/llama-server \
#   QWEN4B_MODEL=/path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
#   ./strix-halo-llama-server.sh {start|stop|status}

set -euo pipefail

BIN="${LLAMA_SERVER_BIN:-$(command -v llama-server || true)}"
MODEL="${QWEN4B_MODEL:-$HOME/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
PORT="${PORT:-8081}"
LOG="${LOG:-/tmp/llama-qwen3-4b-${PORT}.log}"

case "${1:-status}" in
  start)
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "qwen3-4b-instruct already running on :${PORT}"
      exit 0
    fi
    if [[ -z "$BIN" || ! -x "$BIN" ]]; then
      echo "Error: llama-server not found (set LLAMA_SERVER_BIN)" >&2
      exit 1
    fi
    nohup "$BIN" \
      -m "$MODEL" \
      --alias qwen3-4b-instruct \
      --host 127.0.0.1 \
      --port "$PORT" \
      -c 32768 \
      -ngl 999 \
      --flash-attn on \
      --parallel 4 \
      --cont-batching \
      --batch-size 2048 \
      --ubatch-size 512 \
      --cache-type-k f16 \
      --cache-type-v f16 \
      >"$LOG" 2>&1 </dev/null &
    for i in $(seq 1 30); do
      if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "qwen3-4b-instruct healthy on :${PORT} after ${i}s"
        exit 0
      fi
      sleep 1
    done
    echo "Error: failed to start on :${PORT}; see $LOG" >&2
    exit 1
    ;;
  stop)
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
    echo "Stopped server on :${PORT}"
    ;;
  status)
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "qwen3-4b-instruct running on :${PORT}"
    else
      echo "qwen3-4b-instruct not running on :${PORT}"
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
