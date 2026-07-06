#!/usr/bin/env bash

set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ./rhaii-universal.sh [--api-key <key>]

Starts the RHAII vLLM backend. Without --api-key the API is open; with it,
all OpenAI-compatible endpoints (including /v1/models) require:
  Authorization: Bearer <key>
Select the model via MODEL_KEY or MODEL (see below).
EOF
}

API_KEY=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --api-key)
      if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
        echo "--api-key requires a non-empty value" >&2
        usage >&2
        exit 1
      fi
      API_KEY="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# ------------------------------
# Model selection
# ------------------------------
# Pick one key via MODEL_KEY, or set MODEL directly to override everything.
# Default: a locally cached Qwen3-14B quantized model that fits on a Tesla T4.
# Example: MODEL_KEY=qwen3_14b ./rhaii-universal.sh
# Example: MODEL=meta-llama/Llama-3.1-8B-Instruct ./rhaii-universal.sh

MODEL_CHOICES_KEYS=(
  deepseek_r1_qwen_14b_awq
  qwen3_4b
  qwen3_14b
  granite_8b
  llama31_8b
  whiterabbit_7b_awq
)

MODEL_CHOICES_VALUES=(
  "casperhansen/deepseek-r1-distill-qwen-14b-awq"
  "Qwen/Qwen3-4B-Instruct-2507"
  "RedHatAI/Qwen3-14B-quantized.w4a16"
  "ibm-granite/granite-3.3-8b-instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "solidrust/WhiteRabbitNeo-7B-v1.5a-AWQ"
)

MODEL_KEY="${MODEL_KEY:-qwen3_14b}"
MODEL="${MODEL:-}"

if [ -z "$MODEL" ]; then
  found=0
  for i in "${!MODEL_CHOICES_KEYS[@]}"; do
    if [ "${MODEL_CHOICES_KEYS[$i]}" = "$MODEL_KEY" ]; then
      MODEL="${MODEL_CHOICES_VALUES[$i]}"
      found=1
      break
    fi
  done

  if [ "$found" -ne 1 ]; then
    echo "Invalid MODEL_KEY: $MODEL_KEY" >&2
    echo "Available MODEL_KEY values:" >&2
    for i in "${!MODEL_CHOICES_KEYS[@]}"; do
      echo "  - ${MODEL_CHOICES_KEYS[$i]} => ${MODEL_CHOICES_VALUES[$i]}" >&2
    done
    exit 1
  fi
fi

IMAGE="${RHAII_IMAGE:-${RHAII_IMAGE:-registry.redhat.io/rhaii-early-access/vllm-cuda-rhel9:3.4.0-ea.2}}"
CACHE_DIR="${RHAII_CACHE_DIR:-${RHAII_CACHE_DIR:-$HOME/rhaii-cache}}"
LOG_DIR="${RHAII_LOG_DIR:-${RHAII_LOG_DIR:-$PWD/logs}}"
CONTAINER_NAME="${RHAII_CONTAINER_NAME:-${RHAII_CONTAINER_NAME:-rhaii}}"
PORT="${RHAII_PORT:-${RHAII_PORT:-8000}}"
SHM_SIZE="${RHAII_SHM_SIZE:-${RHAII_SHM_SIZE:-4g}}"
USERNS_MODE="${RHAII_USERNS_MODE:-${RHAII_USERNS_MODE:-keep-id}}"
MODEL_DOWNLOADER_SCRIPT="${RHAII_MODEL_DOWNLOADER_SCRIPT:-${RHAII_MODEL_DOWNLOADER_SCRIPT:-$ROOT_DIR/model-downloader.sh}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${VLLM_DTYPE:-half}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_KEY}}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
RHAII_FOLLOW_LOGS="${RHAII_FOLLOW_LOGS:-${RHAII_FOLLOW_LOGS:-1}}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/${CONTAINER_NAME}-${MODEL_KEY}-${RUN_ID}.log"

MODEL_QUANTIZATION="${MODEL_QUANTIZATION:-}"
if [ -z "$MODEL_QUANTIZATION" ]; then
  case "$MODEL_KEY" in
    deepseek_r1_qwen_14b_awq)
      MODEL_QUANTIZATION="awq"
      ;;
    qwen3_4b|granite_8b|llama31_8b)
      MODEL_QUANTIZATION=""
      ;;
    qwen3_14b)
      MODEL_QUANTIZATION="compressed-tensors"
      ;;
    whiterabbit_7b_awq)
      MODEL_QUANTIZATION="awq"
      ;;
    *)
      MODEL_QUANTIZATION=""
      ;;
  esac
fi

if [ -z "$GPU_MEMORY_UTILIZATION" ] || [ -z "$MAX_MODEL_LEN" ] || [ -z "$MAX_NUM_SEQS" ]; then
  case "$MODEL_KEY" in
    deepseek_r1_qwen_14b_awq)
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
      ;;
    qwen3_14b)
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
      ;;
    qwen3_4b)
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
      ;;
    whiterabbit_7b_awq)
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
      ;;
    *)
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
      ;;
  esac
fi

VLLM_EXTRA_ARGS=()
if [ -n "$API_KEY" ]; then
  VLLM_EXTRA_ARGS+=(--api-key "$API_KEY")
fi
if [ -n "$MODEL_QUANTIZATION" ]; then
  VLLM_EXTRA_ARGS+=(--quantization "$MODEL_QUANTIZATION")
fi

VLLM_CHAT_TEMPLATE_ARGS=()
CHAT_TEMPLATE_HOST_PATH="$CACHE_DIR/chat_template.jinja"
if [ -f "$CHAT_TEMPLATE_HOST_PATH" ]; then
  VLLM_CHAT_TEMPLATE_ARGS=(
    --chat-template "/opt/app-root/src/.cache/chat_template.jinja"
  )
fi

case "$MODEL_KEY" in
  qwen3_14b)
    VLLM_EXTRA_ARGS+=(
      --enable-auto-tool-choice
      --tool-call-parser qwen3_xml
      --reasoning-parser qwen3
    )
    VLLM_CHAT_TEMPLATE_ARGS=()
    ;;
esac

if ! command -v podman >/dev/null 2>&1; then
  echo "podman is required but not found" >&2
  exit 1
fi

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/HF_TOKEN}"
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
  HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
fi

if [ "${HF_HUB_OFFLINE}" != "1" ] && [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is required when HF_HUB_OFFLINE!=1 and $HF_TOKEN_FILE was not readable or empty" >&2
  exit 1
fi

mkdir -p "$CACHE_DIR" "$LOG_DIR"
touch "$LOG_FILE"

if [ ! -x "$MODEL_DOWNLOADER_SCRIPT" ]; then
  echo "Model downloader script not found or not executable: $MODEL_DOWNLOADER_SCRIPT" >&2
  exit 1
fi

echo "Ensuring model is cached locally..."
MODEL_KEY="$MODEL_KEY" \
MODEL="$MODEL" \
RHAII_CACHE_DIR="$CACHE_DIR" \
HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
HF_TOKEN="${HF_TOKEN:-}" \
"$MODEL_DOWNLOADER_SCRIPT"

RUNTIME_HF_HUB_OFFLINE=1

if podman ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Container '$CONTAINER_NAME' already exists. Remove/rename it or set RHAII_CONTAINER_NAME." >&2
  exit 1
fi

echo "Starting container: $CONTAINER_NAME"
echo "Model: $MODEL"
echo "Image: $IMAGE"
echo "Endpoint (IPv4): http://0.0.0.0:$PORT"
echo "Endpoint (IPv6): http://[::]:$PORT"
if [ -n "$API_KEY" ]; then
  echo "Auth: enabled with required Bearer token"
fi
echo "Logging to: $LOG_FILE"

PUBLISH_ARGS=(
  -p "$PORT:8000"
)

# Optional: publish an explicit IPv6 bind in addition to IPv4.
# Disabled by default because some hosts treat the IPv6 publish as dual-stack,
# which can make the second publish fail with a false "port already occupied".
if [ "${RHAII_EXPLICIT_IPV6_BIND:-${RHAII_EXPLICIT_IPV6_BIND:-0}}" = "1" ]; then
  PUBLISH_ARGS+=(
    -p "[::]:$PORT:8000"
  )
fi

podman run -d --rm \
  --name "$CONTAINER_NAME" \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --shm-size="$SHM_SIZE" \
  "${PUBLISH_ARGS[@]}" \
  --userns="$USERNS_MODE" \
  --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}" \
  --env "HF_HUB_OFFLINE=$RUNTIME_HF_HUB_OFFLINE" \
  --env "VLLM_NO_USAGE_STATS=1" \
  --env "PYTORCH_ALLOC_CONF=expandable_segments:True" \
  -v "$CACHE_DIR:/opt/app-root/src/.cache:Z" \
  "$IMAGE" \
  --model "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype "$DTYPE" \
  "${VLLM_EXTRA_ARGS[@]}" \
  "${VLLM_CHAT_TEMPLATE_ARGS[@]}" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enforce-eager >/dev/null

if [ "$RHAII_FOLLOW_LOGS" = "1" ]; then
  echo "Container is running. Press Ctrl+C to stop following logs; container will keep running."
  trap 'echo; echo "Stopped log follow. Container still running: $CONTAINER_NAME"; exit 0' INT TERM
  podman logs -f "$CONTAINER_NAME" 2>&1 | tee -a "$LOG_FILE"
else
  echo "Container is running in detached mode (log follow disabled)."
  podman logs -f "$CONTAINER_NAME" >>"$LOG_FILE" 2>&1 &
  LOGGER_PID=$!
  disown "$LOGGER_PID" 2>/dev/null || true
  echo "Streaming container logs to: $LOG_FILE"
  echo "Use: tail -f $LOG_FILE"
fi
