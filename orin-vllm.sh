#!/usr/bin/env bash

set -euo pipefail

# ============================================================
#  CONFIGURATION  —  edit these values for your local setup
# ============================================================

# Human-readable model identifier used only in status output.
MODEL_NAME="Qwen3-1.7B-GPTQ-Int8"

# Local GPTQ model directory mounted read-only inside the container.
MODEL_DIR="$HOME/vllm-models-cache/Qwen3-1.7B-GPTQ-Int8"

# Model name exposed by the OpenAI-compatible /v1/models endpoint.
SERVED_MODEL_NAME="qwen3-1.7b"

# Docker container name used by the start/stop commands.
CONTAINER_NAME="vllm-orin"

# Host port mapped to the vLLM API server running on container port 8000.
PORT=8000

# vLLM container image built for Jetson/TeGra aarch64 with CUDA support.
IMAGE="ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin"

# Jetson Orin Nano 8 GB tuning (unified memory is tight; bias to one request).
# Fraction of GPU memory vLLM may reserve for model execution.
# Keep vLLM's cache target minimal on the 8 GB unified-memory Orin. Model
# weights and kernel workspaces are allocated separately during startup.
GPU_MEMORY_UTILIZATION=0.15

# Maximum prompt + generated context length accepted by the backend. The
# explicit 128 MiB KV cache provides enough room for a 1K local chat window.
MAX_MODEL_LEN=1024

# Upper bound for total tokens batched together by vLLM.
MAX_NUM_BATCHED_TOKENS=8

# Maximum concurrent sequences; keep at 1 for the constrained Orin Nano setup.
MAX_NUM_SEQS=1

# Explicit KV-cache budget in bytes to avoid over-allocating unified memory.
# Reserve 128 MiB for Qwen3's larger KV head count.
KV_CACHE_MEMORY_BYTES=134217728

# Some OpenAI-compatible clients send `tool_choice: "auto"` on every chat
# request. vLLM rejects that unless both options below are enabled. `hermes`
# is the function-call parser recommended for Qwen3.
ENABLE_AUTO_TOOL_CHOICE=true
TOOL_CALL_PARSER="hermes"

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

# NvMap requires reclaimable physical pages for CUDA allocations. Page cache
# can otherwise leave several GiB "available" to Linux but unusable by NvMap.
reclaim_page_cache() {
  echo "Reclaiming host page cache for Jetson unified memory"
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || {
    echo "Unable to reclaim page cache with passwordless sudo" >&2
    exit 1
  }
}

cmd_start() {
  require_cmd docker

  if [ ! -d "$MODEL_DIR" ]; then
    echo "Model directory not found: $MODEL_DIR" >&2
    echo "Set MODEL_DIR (top of this script) to the local GPTQ model directory." >&2
    exit 1
  fi

  # `latest-jetson-orin` is a mutable tag. Refresh it before stopping the
  # current container so a failed pull does not take an existing service down.
  echo "Pulling image: $IMAGE"
  docker pull "$IMAGE"

  release_gpu_devices
  remove_container "$CONTAINER_NAME"
  sleep 5   # let IOMMU/NCCL release GPU mappings before docker re-acquires them
  reclaim_page_cache

  mkdir -p "$LOG_DIR"
  local log_file="$LOG_DIR/vllm-gpu-$(date +%Y%m%d-%H%M%S).log"

  # cudaMallocAsync avoids Jetson's native CUDACachingAllocator NVML assertion.
  # NCCL P2P/IB are disabled (no NVLink/InfiniBand). Marlin avoids the legacy
  # GPTQ kernel's fixed 48 MiB contiguous workspace allocation on Jetson.
  docker run -d \
    --name "$CONTAINER_NAME" \
    --runtime=nvidia \
    --shm-size 1g \
    --ulimit memlock=-1:-1 \
    --ulimit stack=67108864:67108864 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
    -e NCCL_P2P_DISABLE=1 \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_CUMEM_ENABLE=0 \
    -e NCCL_SHM_DISABLE=1 \
    -p "$PORT:8000" \
    -v "$MODEL_DIR:/model:ro" \
    "$IMAGE" \
    python3 -m vllm.entrypoints.openai.api_server \
      --model /model \
      --served-model-name "$SERVED_MODEL_NAME" \
      --host 0.0.0.0 \
      --port 8000 \
      --dtype float16 \
      --quantization gptq_marlin \
      --enforce-eager \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      ${ENABLE_AUTO_TOOL_CHOICE:+--enable-auto-tool-choice} \
      ${TOOL_CALL_PARSER:+--tool-call-parser "$TOOL_CALL_PARSER"} \
      ${KV_CACHE_MEMORY_BYTES:+--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"}

  docker logs -f "$CONTAINER_NAME" >"$log_file" 2>&1 &

  echo
  echo "Container started: $MODEL_NAME (GPU mode)"
  echo "  Model dir : $MODEL_DIR"
  echo "  GPU util  : $GPU_MEMORY_UTILIZATION"
  echo "  Context   : $MAX_MODEL_LEN tokens"
  echo "  Batched   : $MAX_NUM_BATCHED_TOKENS tokens, $MAX_NUM_SEQS seq(s)"
  echo "  KV cache  : ${KV_CACHE_MEMORY_BYTES:-auto} bytes"
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
