#!/usr/bin/env bash

set -euo pipefail

# ============================================================
#  CONFIGURATION  —  edit these
# ============================================================
MODEL_NAME="Qwen2.5-1.5B-Instruct-GPTQ-Int4"
MODEL_DIR="$HOME/vllm-move/models/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
SERVED_MODEL_NAME="orin-vllm"
CONTAINER_NAME="vllm-qwen-it"
PORT=8000
IMAGE="ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04"

# Jetson Orin Nano 8 GB tuning (unified memory is tight; bias to one request).
GPU_MEMORY_UTILIZATION=0.15
MAX_MODEL_LEN=256
MAX_NUM_BATCHED_TOKENS=16
MAX_NUM_SEQS=1
SWAP_SPACE_GB=2
KV_CACHE_MEMORY_BYTES=67108864

# ============================================================
#  DO NOT EDIT BELOW THIS LINE
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

usage() {
  cat <<EOF
Usage: ./orin-vllm.sh <command>

Commands:
  start   Launch the vLLM backend (GPTQ) on Jetson Orin via docker.
  stop    Stop and remove the '$CONTAINER_NAME' container.

Backend only: OpenAI-compatible API on http://0.0.0.0:$PORT/v1
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "$1 is required but not found" >&2; exit 1; }
}

remove_container() {
  local name="$1"
  if docker inspect "$name" >/dev/null 2>&1; then
    if [ "$(docker inspect -f '{{.State.Status}}' "$name")" = "running" ]; then
      echo "Stopping container: $name"
      docker stop "$name" >/dev/null
    fi
    echo "Removing container: $name"
    docker rm "$name" >/dev/null
  fi
}

# Release GPU devices held by stale native Python processes (e.g. a crashed
# inference script). Docker-managed ports are freed by remove_container below.
release_gpu_devices() {
  local dev pids
  for dev in /dev/nvhost-gpu /dev/nvhost-ctrl-gpu /dev/nvhost-as-gpu; do
    [ -e "$dev" ] || continue
    pids="$(sudo fuser "$dev" 2>/dev/null | tr ' ' '\n' | grep -v '^$' || true)"
    if [ -n "$pids" ]; then
      echo "Killing processes holding $dev: $pids"
      echo "$pids" | xargs sudo kill -9 2>/dev/null || true
    fi
  done
}

cmd_start() {
  require_cmd docker

  if [ ! -d "$MODEL_DIR" ]; then
    echo "Model directory not found: $MODEL_DIR" >&2
    echo "Set MODEL_DIR (top of this script) to the local GPTQ model directory." >&2
    exit 1
  fi

  release_gpu_devices
  remove_container "$CONTAINER_NAME"
  sleep 5   # let IOMMU/NCCL release GPU mappings before docker re-acquires them

  mkdir -p "$LOG_DIR"
  local log_file="$LOG_DIR/vllm-gpu-$(date +%Y%m%d-%H%M%S).log"

  # PYTORCH_CUDA_ALLOC_CONF=cudaMallocAsync avoids the native allocator's NVML
  # assertion path during KV-cache allocation on Orin; NCCL P2P/IB disabled (no
  # NVLink/InfiniBand); GPTQ forced (Marlin weight path fails at startup here).
  docker run -d \
    --name "$CONTAINER_NAME" \
    --runtime=nvidia \
    --shm-size 1g \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
    -e NCCL_P2P_DISABLE=1 \
    -e NCCL_IB_DISABLE=1 \
    -p "$PORT:8000" \
    -v "$MODEL_DIR:/model:ro" \
    "$IMAGE" \
    python3 -m vllm.entrypoints.openai.api_server \
      --model /model \
      --served-model-name "$SERVED_MODEL_NAME" \
      --host 0.0.0.0 \
      --port 8000 \
      --dtype float16 \
      --quantization gptq \
      --enforce-eager \
      --disable-log-requests \
      --swap-space "$SWAP_SPACE_GB" \
      --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --max-num-seqs "$MAX_NUM_SEQS"

  docker logs -f "$CONTAINER_NAME" >"$log_file" 2>&1 &

  echo
  echo "Container started: $MODEL_NAME (GPU mode)"
  echo "  Model dir : $MODEL_DIR"
  echo "  GPU util  : $GPU_MEMORY_UTILIZATION"
  echo "  Context   : $MAX_MODEL_LEN tokens"
  echo "  Batched   : $MAX_NUM_BATCHED_TOKENS tokens, $MAX_NUM_SEQS seq(s)"
  echo "  Swap      : $SWAP_SPACE_GB GiB"
  echo "  KV cache  : $KV_CACHE_MEMORY_BYTES bytes"
  echo "  Logs      : $log_file"
  echo "  Models    : curl http://localhost:$PORT/v1/models"
  echo "  Chat API  : POST http://<host-ip>:$PORT/v1/chat/completions"
  echo
  echo "--- Tailing logs (Ctrl+C to detach, container keeps running) ---"
  tail -f "$log_file"
}

cmd_stop() {
  require_cmd docker
  if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    remove_container "$CONTAINER_NAME"
    echo "Stopped and removed: $CONTAINER_NAME"
  else
    echo "Container not present: $CONTAINER_NAME"
  fi
}

case "${1:-}" in
  start) cmd_start ;;
  stop)  cmd_stop ;;
  -h|--help|help) usage ;;
  *) usage; exit 1 ;;
esac
