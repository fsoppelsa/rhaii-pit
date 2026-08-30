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
  qwen38_27b_int4
  granite_8b
  llama31_8b
  whiterabbit_7b_awq
)

MODEL_CHOICES_VALUES=(
  "casperhansen/deepseek-r1-distill-qwen-14b-awq"
  "Qwen/Qwen3-4B-Instruct-2507"
  "RedHatAI/Qwen3-14B-quantized.w4a16"
  "RedHatAI/Qwen3.8-27B-INT4"
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

if [ "${RHAII_UPSTREAM:-0}" = "1" ]; then
  case "$MODEL_KEY" in
    qwen38_27b|qwen38_27b_int4)
      IMAGE="${RHAII_IMAGE:-docker.io/vllm/vllm-openai:v0.27.1}" ;;
    *)
      IMAGE="${RHAII_IMAGE:-docker.io/vllm/vllm-openai:latest}" ;;
  esac
  CONTAINER_CACHE_PATH="/root/.cache"
else
  IMAGE="${RHAII_IMAGE:-registry.redhat.io/rhaii-early-access/vllm-cuda-rhel9:3.4.0-ea.2}"
  CONTAINER_CACHE_PATH="/opt/app-root/src/.cache"
fi
CACHE_DIR="${RHAII_CACHE_DIR:-${RHAII_CACHE_DIR:-$HOME/rhaii-cache}}"
LOG_DIR="${RHAII_LOG_DIR:-${RHAII_LOG_DIR:-$PWD/logs}}"
CONTAINER_NAME="${RHAII_CONTAINER_NAME:-${RHAII_CONTAINER_NAME:-rhaii}}"
PORT="${RHAII_PORT:-${RHAII_PORT:-8000}}"
SHM_SIZE="${RHAII_SHM_SIZE:-${RHAII_SHM_SIZE:-4g}}"
# Upstream vLLM runs as root (no userns needed). RHAII runs as non-root (needs keep-id).
if [ "${RHAII_UPSTREAM:-0}" = "1" ]; then
  USERNS_MODE="${RHAII_USERNS_MODE:-}"
else
  USERNS_MODE="${RHAII_USERNS_MODE:-keep-id}"
fi
MODEL_DOWNLOADER_SCRIPT="${RHAII_MODEL_DOWNLOADER_SCRIPT:-${RHAII_MODEL_DOWNLOADER_SCRIPT:-$ROOT_DIR/model-downloader.sh}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${VLLM_DTYPE:-half}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-}"
RHAII_ENFORCE_EAGER="${RHAII_ENFORCE_EAGER:-}"
RHAII_V2_RUNNER="${RHAII_V2_RUNNER:-}"
RHAII_LANGUAGE_MODEL_ONLY="${RHAII_LANGUAGE_MODEL_ONLY:-}"
RHAII_ENABLE_PREFIX_CACHING="${RHAII_ENABLE_PREFIX_CACHING:-}"
RHAII_PREFIX_MATCH_UNIT="${RHAII_PREFIX_MATCH_UNIT:-}"
RHAII_OFFLOAD_GROUP_SIZE="${RHAII_OFFLOAD_GROUP_SIZE:-}"
RHAII_OFFLOAD_NUM_IN_GROUP="${RHAII_OFFLOAD_NUM_IN_GROUP:-1}"
RHAII_OFFLOAD_PREFETCH_STEP="${RHAII_OFFLOAD_PREFETCH_STEP:-1}"
RHAII_SPEC_CONFIG="${RHAII_SPEC_CONFIG:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
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
    qwen3_4b|granite_8b|llama31_8b|qwen38_27b)
      MODEL_QUANTIZATION=""
      ;;
    qwen3_14b|qwen38_27b_int4)
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

  case "$MODEL_KEY" in
    deepseek_r1_qwen_14b_awq)
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
      SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-fedyagpt_14b}"
      ;;
    qwen3_14b)
      # Pi's repository instructions exceed an 8k context. A single 16k request
      # fits on the T4 when KV cache is prioritized over concurrent sequences.
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
      ;;
    qwen38_27b)
      # 51.75 GiB FP16 on a 15 GiB T4. 42 GiB UVA offload leaves ~9 GiB weights
      # on GPU; CUDA graphs stay on (eager off). Context kept short for KV.
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
      CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-42}"
      DTYPE="${VLLM_DTYPE:-float16}"
      SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen38_27b}"
      RHAII_V2_RUNNER="${RHAII_V2_RUNNER:-0}"
      ;;
    qwen38_27b_int4)
      # T4 profile: text-only W4A16 plus native MTP. Keeping 18 language
      # modules resident and prefetching 46 leaves room for a 16k KV cache.
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.99}"
      DTYPE="${VLLM_DTYPE:-float16}"
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
      MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
      CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
      SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen_27b}"
      RHAII_V2_RUNNER="${RHAII_V2_RUNNER:-0}"
      RHAII_LANGUAGE_MODEL_ONLY="${RHAII_LANGUAGE_MODEL_ONLY:-1}"
      RHAII_ENABLE_PREFIX_CACHING="${RHAII_ENABLE_PREFIX_CACHING:-1}"
      RHAII_PREFIX_MATCH_UNIT="${RHAII_PREFIX_MATCH_UNIT:-16}"
      if [ -z "$RHAII_OFFLOAD_GROUP_SIZE" ]; then
        RHAII_OFFLOAD_GROUP_SIZE=64
        RHAII_OFFLOAD_NUM_IN_GROUP=46
      fi
      if [ -z "$RHAII_SPEC_CONFIG" ]; then
        RHAII_SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":4}'
      fi
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

SERVED_MODEL_ARGS=()
if [ -n "$SERVED_MODEL_NAME" ]; then
  SERVED_MODEL_ARGS=(--served-model-name "$SERVED_MODEL_NAME")
fi

VLLM_EXTRA_ARGS=()
if [ -n "$API_KEY" ]; then
  VLLM_EXTRA_ARGS+=(--api-key "$API_KEY")
fi
if [ -n "$MODEL_QUANTIZATION" ]; then
  VLLM_EXTRA_ARGS+=(--quantization "$MODEL_QUANTIZATION")
fi
if [ -n "$CPU_OFFLOAD_GB" ]; then
  VLLM_EXTRA_ARGS+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
fi
if [ -n "$MAX_NUM_BATCHED_TOKENS" ]; then
  VLLM_EXTRA_ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi
if [ "$RHAII_LANGUAGE_MODEL_ONLY" = "1" ]; then
  VLLM_EXTRA_ARGS+=(--language-model-only)
fi
if [ "$RHAII_ENABLE_PREFIX_CACHING" = "1" ]; then
  VLLM_EXTRA_ARGS+=(--enable-prefix-caching)
fi
if [ -n "$RHAII_PREFIX_MATCH_UNIT" ]; then
  VLLM_EXTRA_ARGS+=(--prefix-match-unit "$RHAII_PREFIX_MATCH_UNIT")
fi
if [ -n "$RHAII_OFFLOAD_GROUP_SIZE" ]; then
  VLLM_EXTRA_ARGS+=(
    --offload-backend prefetch
    --offload-group-size "$RHAII_OFFLOAD_GROUP_SIZE"
    --offload-num-in-group "$RHAII_OFFLOAD_NUM_IN_GROUP"
    --offload-prefetch-step "$RHAII_OFFLOAD_PREFETCH_STEP"
  )
fi

# Optional speculative decoding, e.g.
#   RHAII_SPEC_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":2}'
if [ -n "${RHAII_SPEC_CONFIG:-}" ]; then
  VLLM_EXTRA_ARGS+=(--speculative-config "$RHAII_SPEC_CONFIG")
fi
VLLM_CHAT_TEMPLATE_ARGS=()
CHAT_TEMPLATE_HOST_PATH="$CACHE_DIR/chat_template.jinja"
if [ -f "$CHAT_TEMPLATE_HOST_PATH" ]; then
  VLLM_CHAT_TEMPLATE_ARGS=(
    --chat-template "$CONTAINER_CACHE_PATH/chat_template.jinja"
  )
fi

case "$MODEL_KEY" in
  deepseek_r1_qwen_14b_awq)
    VLLM_EXTRA_ARGS+=(
      --enable-auto-tool-choice
      --tool-call-parser xlam
    )
    ;;
  qwen3_14b)
    VLLM_EXTRA_ARGS+=(
      --enable-auto-tool-choice
      --tool-call-parser hermes
      --reasoning-parser qwen3
    )
    VLLM_CHAT_TEMPLATE_ARGS=()
    ;;
  qwen38_27b)
    VLLM_EXTRA_ARGS+=(
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
      --reasoning-parser qwen3
      --trust-remote-code
    )
    VLLM_CHAT_TEMPLATE_ARGS=()
    ;;
  qwen38_27b_int4)
    VLLM_EXTRA_ARGS+=(
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
      --reasoning-parser qwen3
      --trust-remote-code
    )
    VLLM_CHAT_TEMPLATE_ARGS=()
    ;;
esac

ENFORCE_EAGER_ARGS=()
if [ "${RHAII_ENFORCE_EAGER:-}" = "1" ]; then
  ENFORCE_EAGER_ARGS+=(--enforce-eager)
fi

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
echo "Backend: $([ "${RHAII_UPSTREAM:-0}" = "1" ] && echo "Upstream vLLM" || echo "Red Hat AI Inference")"
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

# vLLM 0.26+ uses positional model arg; older/RHAII uses --model flag.
if [ "${RHAII_UPSTREAM:-0}" = "1" ]; then
  MODEL_ARGS=("$MODEL")
else
  MODEL_ARGS=(--model "$MODEL")
fi

podman run -d --rm \
  --name "$CONTAINER_NAME" \
  ${USERNS_MODE:+--userns="$USERNS_MODE"} \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --shm-size="$SHM_SIZE" \
  "${PUBLISH_ARGS[@]}" \
  --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}" \
  --env "HF_HUB_OFFLINE=$RUNTIME_HF_HUB_OFFLINE" \
  --env "VLLM_NO_USAGE_STATS=1" \
  --env "PYTORCH_ALLOC_CONF=expandable_segments:True" \
  --env "VLLM_USE_V2_MODEL_RUNNER=${RHAII_V2_RUNNER:-1}" \
  -v "$CACHE_DIR:$CONTAINER_CACHE_PATH:Z" \
  "$IMAGE" \
  "${MODEL_ARGS[@]}" \
  "${SERVED_MODEL_ARGS[@]}" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype "$DTYPE" \
  "${VLLM_EXTRA_ARGS[@]}" \
  "${VLLM_CHAT_TEMPLATE_ARGS[@]}" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  ${ENFORCE_EAGER_ARGS[@]+"${ENFORCE_EAGER_ARGS[@]}"} >/dev/null

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
