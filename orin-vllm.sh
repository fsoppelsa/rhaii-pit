#!/usr/bin/env bash

set -euo pipefail

# ============================================================
#  CONFIGURATION  —  edit these
# ============================================================
# MODEL: a catalog key (qwen3_1_7b_gptq_int8, qwen3_1_7b_gptq_int4) or a
#        path to a local model directory (must contain a "/").
MODEL="qwen3_1_7b_gptq_int8"
BACKEND_PORT=8000                # vLLM OpenAI API port
SERVED_MODEL_NAME="orin2b"       # name exposed by /v1/models (fedyagpt family member)
CONTAINER_NAME="vllm-orin"       # docker container name
IMAGE="ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin"   # latest Jetson/tegra aarch64 build

# Quantization backend for the local GPTQ model dirs. gptq_marlin avoids the
# legacy GPTQ kernel's fixed 48 MiB contiguous workspace allocation on Jetson.
MODEL_QUANTIZATION="gptq_marlin"

# Jetson Orin 8 GB unified-memory tuning (bias to one short request).
GPU_MEMORY_UTILIZATION=0.15      # fraction of GPU memory vLLM may reserve
MAX_MODEL_LEN=4096               # prompt + generated context accepted
MAX_NUM_BATCHED_TOKENS=8         # upper bound for tokens batched together
MAX_NUM_SEQS=1                   # maximum concurrent sequences
KV_CACHE_MEMORY_BYTES=536870912  # explicit 512 MiB KV-cache budget (~4.6k tokens)

# Some OpenAI-compatible clients send `tool_choice: "auto"` on every chat
# request. vLLM rejects that unless both options below are enabled. `qwen3_xml`
# is the function-call parser for Qwen3 chat models.
ENABLE_AUTO_TOOL_CHOICE=true
TOOL_CALL_PARSER="qwen3_xml"

# ============================================================
#  DO NOT EDIT BELOW THIS LINE
# ============================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"

# Matching environment variables, when set, override the config above.
ORIN_MODEL="${ORIN_MODEL:-$MODEL}"
ORIN_PORT="${ORIN_PORT:-$BACKEND_PORT}"
ORIN_SERVED_MODEL_NAME="${ORIN_SERVED_MODEL_NAME:-$SERVED_MODEL_NAME}"
ORIN_CONTAINER_NAME="${ORIN_CONTAINER_NAME:-$CONTAINER_NAME}"
ORIN_IMAGE="${ORIN_IMAGE:-$IMAGE}"
ORIN_MODEL_QUANTIZATION="${ORIN_MODEL_QUANTIZATION:-$MODEL_QUANTIZATION}"
ORIN_API_KEY="${ORIN_API_KEY:-}"   # normally set with --bearer

# Logs
ORIN_BACKEND_BOOT_LOG_FILE="$LOG_DIR/orin-vllm-bootstrap.log"
ORIN_STARTUP_LOG_STREAM=1

# Backend readiness polling
ORIN_WAIT_TIMEOUT_SEC=240
ORIN_WAIT_INTERVAL_SEC=3

# ------------------------------
# Helpers
# ------------------------------
usage() {
  cat <<EOF
Usage: ./orin-vllm.sh <command> [options]

Upstream-only (vLLM) launcher for the Jetson Orin. Starts no RHAII
reasoning proxy and no WebUI: just the OpenAI-compatible vLLM API.

Commands:
  start                Start the vLLM backend (headless).
  stop                 Stop and remove the '$ORIN_CONTAINER_NAME' container.
  smoke-test           Probe the already-running backend and report whether the
                       model answers (non-zero exit if it does not).

Start options (combinable):
  --bearer <key>       Require this Bearer token on the API (enables vLLM auth).
                       Overrides the ORIN_API_KEY environment variable.
  --smoke-test         After startup, send a test prompt to the backend and
                       fail (non-zero exit) if the model does not answer.

Configuration is via environment variables (see the top of this script);
MODEL selects the served model, e.g. ORIN_MODEL=qwen3_1_7b_gptq_int4 ./orin-vllm.sh start
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "$1 is required but not found" >&2; exit 1; }
}

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "$1"
}

container_running() {
  docker ps --format '{{.Names}}' | grep -qx "$1"
}

resolve_model_dir() {
  case "$1" in
    qwen3_1_7b_gptq_int8) echo "$HOME/vllm-models-cache/Qwen3-1.7B-GPTQ-Int8" ;;
    qwen3_1_7b_gptq_int4) echo "$HOME/vllm-models-cache/Qwen3-1.7B-GPTQ-Int4" ;;
    *)
      if [[ "$1" == */* ]]; then
        echo "$1"
      else
        echo "Unknown MODEL key: $1" >&2
        echo "Available MODEL keys: qwen3_1_7b_gptq_int8, qwen3_1_7b_gptq_int4" >&2
        echo "Or set MODEL to a local model directory path." >&2
        exit 1
      fi
      ;;
  esac
}

stop_startup_log_stream() {
  if [ -n "${ORIN_LOG_STREAM_PID:-}" ] && kill -0 "$ORIN_LOG_STREAM_PID" >/dev/null 2>&1; then
    kill "$ORIN_LOG_STREAM_PID" >/dev/null 2>&1 || true
    wait "$ORIN_LOG_STREAM_PID" 2>/dev/null || true
  fi
  unset ORIN_LOG_STREAM_PID
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

# A single pre-start drop_caches goes stale while vLLM loads: reading the
# weights from disk refills page cache, and the KV-cache allocation then hits
# NvMap ENOMEM (Jetson unified memory needs reclaimable physical pages).
# Keep dropping until the API answers, then stop the loop.
start_page_cache_reclaimer() {
  reclaim_page_cache
  (
    while :; do
      sleep 10
      sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    done
  ) &
  ORIN_PAGE_CACHE_RECLAIM_PID=$!
}

stop_page_cache_reclaimer() {
  if [ -n "${ORIN_PAGE_CACHE_RECLAIM_PID:-}" ]; then
    kill "$ORIN_PAGE_CACHE_RECLAIM_PID" >/dev/null 2>&1 || true
    wait "$ORIN_PAGE_CACHE_RECLAIM_PID" 2>/dev/null || true
    unset ORIN_PAGE_CACHE_RECLAIM_PID
  fi
}

# Send a minimal chat completion to the backend and confirm a non-empty answer.
# Any extra arguments are passed through to curl (used for the auth header).
run_smoke_test() {
  local auth_args=("$@")
  local base="http://127.0.0.1:${ORIN_PORT}/v1"
  echo "Running smoke test against ${base} ..."

  local models_json model_name
  models_json="$(curl -fsS --max-time 10 "${auth_args[@]}" "${base}/models" 2>/dev/null || true)"
  model_name="$(printf '%s' "$models_json" | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]*"' | head -n1 | sed -E 's/.*"([^"]*)"$/\1/')" || true
  if [ -z "$model_name" ]; then
    echo "Smoke test FAILED: could not read served model from ${base}/models" >&2
    return 1
  fi

  local payload
  payload="$(printf '{"model":"%s","messages":[{"role":"user","content":"Reply with a single short word."}],"max_tokens":16,"temperature":0}' "$model_name")"

  local resp
  resp="$(curl -fsS --max-time 30 "${auth_args[@]}" -H 'Content-Type: application/json' -d "$payload" "${base}/chat/completions" 2>/dev/null || true)"
  if [ -z "$resp" ]; then
    echo "Smoke test FAILED: no response from ${base}/chat/completions" >&2
    return 1
  fi

  local answer
  answer="$(printf '%s' "$resp" | grep -oE '"content"[[:space:]]*:[[:space:]]*"([^"\\]|\\.)*"' | head -n1 | sed -E 's/^"content"[[:space:]]*:[[:space:]]*"(.*)"$/\1/')" || true
  if [ -z "$answer" ]; then
    echo "Smoke test FAILED: model '${model_name}' returned no answer." >&2
    echo "Response: $resp" >&2
    return 1
  fi

  echo "Smoke test PASSED. Model '${model_name}' answered: ${answer}"
}

# ------------------------------
# start
# ------------------------------
cmd_start() {
  local with_smoke="$1"

  # Model resolution
  local launch_model_dir
  launch_model_dir="$(resolve_model_dir "$ORIN_MODEL")"

  # Common preflight
  require_cmd docker
  require_cmd curl

  if [ ! -d "$launch_model_dir" ]; then
    echo "Model directory not found: $launch_model_dir" >&2
    echo "Set MODEL (top of this script) to a local model directory." >&2
    exit 1
  fi

  if container_exists "$ORIN_CONTAINER_NAME"; then
    echo "Container '$ORIN_CONTAINER_NAME' already exists. Run 'stop' first." >&2
    exit 1
  fi

  # Auth arguments (vLLM --api-key guards every endpoint, including /v1/models)
  local curl_auth_args=()
  [ -n "$ORIN_API_KEY" ] && curl_auth_args=(-H "Authorization: Bearer $ORIN_API_KEY")

  # `latest-jetson-orin` is a mutable tag. Refresh it before releasing GPU
  # devices so a failed pull does not leave the host in a torn-down state.
  echo "Pulling image: $ORIN_IMAGE"
  docker pull "$ORIN_IMAGE"

  release_gpu_devices
  remove_container "$ORIN_CONTAINER_NAME"
  sleep 5   # let IOMMU/NCCL release GPU mappings before docker re-acquires them
  start_page_cache_reclaimer

  mkdir -p "$LOG_DIR"
  local log_file="$LOG_DIR/vllm-gpu-$(date +%Y%m%d-%H%M%S).log"
  : >"$ORIN_BACKEND_BOOT_LOG_FILE"

  # cudaMallocAsync avoids Jetson's native CUDACachingAllocator NVML assertion.
  # NCCL P2P/IB are disabled (no NVLink/InfiniBand). Marlin avoids the legacy
  # GPTQ kernel's fixed 48 MiB contiguous workspace allocation on Jetson.
  local vllm_extra_args=()
  [ -n "$ORIN_API_KEY" ] && vllm_extra_args+=(--api-key "$ORIN_API_KEY")

  docker run -d \
    --name "$ORIN_CONTAINER_NAME" \
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
    -p "$ORIN_PORT:8000" \
    -v "$launch_model_dir:/model:ro" \
    "$ORIN_IMAGE" \
    python3 -m vllm.entrypoints.openai.api_server \
      --model /model \
      --served-model-name "$ORIN_SERVED_MODEL_NAME" \
      --host 0.0.0.0 \
      --port 8000 \
      --dtype float16 \
      --quantization "$ORIN_MODEL_QUANTIZATION" \
      --enforce-eager \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      ${ENABLE_AUTO_TOOL_CHOICE:+--enable-auto-tool-choice} \
      ${TOOL_CALL_PARSER:+--tool-call-parser "$TOOL_CALL_PARSER"} \
      ${KV_CACHE_MEMORY_BYTES:+--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"} \
      "${vllm_extra_args[@]}"

  echo "Waiting for vLLM API at http://127.0.0.1:$ORIN_PORT/v1/models ..."
  local deadline=$((SECONDS + ORIN_WAIT_TIMEOUT_SEC))
  local ready=0
  local startup_log_stream_started=0
  while [ "$SECONDS" -lt "$deadline" ]; do
    if [ "$startup_log_stream_started" -ne 1 ] && [ "$ORIN_STARTUP_LOG_STREAM" = "1" ] && container_running "$ORIN_CONTAINER_NAME"; then
      echo "Streaming vLLM startup logs..."
      docker logs -f "$ORIN_CONTAINER_NAME" &
      ORIN_LOG_STREAM_PID=$!
      startup_log_stream_started=1
    fi
    if curl -fsS --max-time 2 "${curl_auth_args[@]}" "http://127.0.0.1:${ORIN_PORT}/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! container_running "$ORIN_CONTAINER_NAME"; then
      break
    fi
    sleep "$ORIN_WAIT_INTERVAL_SEC"
  done

  stop_startup_log_stream

  if [ "$ready" -ne 1 ]; then
    stop_page_cache_reclaimer
    if ! container_running "$ORIN_CONTAINER_NAME"; then
      echo "vLLM container exited during startup." >&2
    else
      echo "vLLM did not become ready within ${ORIN_WAIT_TIMEOUT_SEC}s." >&2
    fi
    if container_exists "$ORIN_CONTAINER_NAME"; then
      echo "Recent vLLM logs:" >&2
      docker logs --tail 80 "$ORIN_CONTAINER_NAME" >&2 || true
    fi
    exit 1
  fi

  stop_page_cache_reclaimer

  if ! container_running "$ORIN_CONTAINER_NAME"; then
    echo "vLLM container is not running after startup." >&2
    exit 1
  fi

  echo
  echo "Stack is up."
  echo "vLLM API: http://127.0.0.1:${ORIN_PORT}/v1"
  echo "Served model: ${ORIN_SERVED_MODEL_NAME} (model dir: ${launch_model_dir})"
  echo "Logs: $log_file"

  if [ "$with_smoke" -eq 1 ]; then
    echo
    run_smoke_test "${curl_auth_args[@]}" || exit 1
  fi
}

# ------------------------------
# smoke-test
# ------------------------------
cmd_smoke() {
  require_cmd curl
  local auth_args=()
  [ -n "$ORIN_API_KEY" ] && auth_args=(-H "Authorization: Bearer $ORIN_API_KEY")
  run_smoke_test "${auth_args[@]}"
}

# ------------------------------
# stop
# ------------------------------
cmd_stop() {
  require_cmd docker
  if container_exists "$ORIN_CONTAINER_NAME"; then
    remove_container "$ORIN_CONTAINER_NAME"
    echo "Stopped and removed: $ORIN_CONTAINER_NAME"
  else
    echo "Container not present: $ORIN_CONTAINER_NAME"
  fi
  echo "Shutdown complete."
}

# ------------------------------
# Dispatch
# ------------------------------
cmd="${1:-}"
shift || true
case "$cmd" in
  start)
    with_smoke=0
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --bearer)
          if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
            echo "--bearer requires a non-empty value" >&2; usage; exit 1
          fi
          ORIN_API_KEY="$2"
          shift
          ;;
        --smoke-test) with_smoke=1 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
      esac
      shift
    done
    cmd_start "$with_smoke"
    ;;
  stop)
    cmd_stop
    ;;
  smoke-test)
    cmd_smoke
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
