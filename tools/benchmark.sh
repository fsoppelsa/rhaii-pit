#!/usr/bin/env bash

set -euo pipefail

# ============================================================
#  CONFIGURATION  —  edit these
# ============================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PODMAN="${PODMAN:-podman}"
GUIDELLM_IMAGE="${GUIDELLM_IMAGE:-ghcr.io/vllm-project/guidellm:stable}"
LM_EVAL_IMAGE="${LM_EVAL_IMAGE:-localhost/rhaii-lm-eval:latest}"
LM_EVAL_BASE_IMAGE="${LM_EVAL_BASE_IMAGE:-registry.access.redhat.com/ubi9/python-312}"
LM_EVAL_VERSION="${LM_EVAL_VERSION:-0.4.8}"
BENCHMARK_RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-$ROOT_DIR/../logs/benchmarks}"
BENCHMARK_CACHE_DIR="${BENCHMARK_CACHE_DIR:-$HOME/rhaii-cache}"

# ============================================================
#  DO NOT EDIT BELOW THIS LINE
# ============================================================

usage() {
  cat <<EOF
Usage: tools/benchmark.sh <command> [options]

Commands:
  install       Pull/build the container images for GuideLLM and LM Evaluation Harness.
  benchmark     Run GuideLLM performance benchmark (same as performance).
  performance   Run GuideLLM performance benchmark.
  quality       Run LM Evaluation Harness quality benchmark.

Benchmark options:
  --url URL     OpenAI API server origin or IP address. A bare IP defaults to
                http://IP:8000 (default: http://127.0.0.1:8000).
  --model NAME  Served model name (default: RedHatAI/Qwen3-14B-quantized.w4a16).
  -h, --help    Show this help text.

Configuration variables:
  PODMAN                 Podman executable (default: podman)
  GUIDELLM_IMAGE         GuideLLM image (default: ghcr.io/vllm-project/guidellm:stable)
  LM_EVAL_IMAGE          Locally built TrustyAI-compatible LM Eval image
                          (default: localhost/rhaii-lm-eval:latest)
  LM_EVAL_BASE_IMAGE     Red Hat UBI Python base image for TrustyAI LM Eval
                          (default: registry.access.redhat.com/ubi9/python-312)
  LM_EVAL_VERSION        LM Evaluation Harness version used by TrustyAI LM-Eval
                          (default: 0.4.8)
  BENCHMARK_ENDPOINT     OpenAI API server origin, without /v1
                          (default: http://127.0.0.1:8000)
  BENCHMARK_MODEL        Served model name
                          (default: RedHatAI/Qwen3-14B-quantized.w4a16)
  BENCHMARK_TOKENIZER    Hugging Face tokenizer ID for GuideLLM and LM Eval
                          (default: RedHatAI/Qwen3-14B-quantized.w4a16)
  BENCHMARK_QUALITY_TASKS LM Eval tasks for quality benchmarks
                          (default: hellaswag,arc_challenge,gsm8k,mmlu)
  BENCHMARK_QUALITY_LIMIT Optional maximum examples per task; use for smoke tests
  BENCHMARK_API_KEY      Optional Bearer token for the remote OpenAI-compatible server
  BENCHMARK_RESULTS_DIR  Host directory mounted at /results
                          (default: $ROOT_DIR/../logs/benchmarks)
  BENCHMARK_CACHE_DIR    Writable host cache mounted for Hugging Face downloads
                          (default: $HOME/rhaii-cache)

All benchmark software runs in containers. This script never creates a host
Python environment or installs Python packages on the host.
EOF
}

require_podman() {
  command -v "$PODMAN" >/dev/null 2>&1 || { echo "$PODMAN is required but not found" >&2; exit 1; }
}

install_tools() {
  require_podman

  echo "Pulling GuideLLM image: $GUIDELLM_IMAGE"
  "$PODMAN" pull "$GUIDELLM_IMAGE"

  echo "Building TrustyAI-compatible LM Eval image: $LM_EVAL_IMAGE"
  "$PODMAN" build \
    --pull=always \
    --build-arg "LM_EVAL_BASE_IMAGE=$LM_EVAL_BASE_IMAGE" \
    --build-arg "LM_EVAL_VERSION=$LM_EVAL_VERSION" \
    --tag "$LM_EVAL_IMAGE" \
    --file "$ROOT_DIR/Containerfile.lm-eval" \
    "$ROOT_DIR"

  "$PODMAN" run --rm "$GUIDELLM_IMAGE" --version
  "$PODMAN" run --rm "$LM_EVAL_IMAGE" --help >/dev/null

  echo
  echo "Container images ready:"
  echo "  GuideLLM: $GUIDELLM_IMAGE"
  echo "  LM Eval:  $LM_EVAL_IMAGE"
  echo "Use '$0 benchmark --url <IP>' to run a performance benchmark."
}

normalize_endpoint() {
  local endpoint="$1"

  endpoint="${endpoint%/}"
  case "$endpoint" in
    http://*|https://*) ;;
    *:*) endpoint="http://$endpoint" ;;
    *) endpoint="http://$endpoint:8000" ;;
  esac

  endpoint="${endpoint%/v1}"
  printf '%s\n' "$endpoint"
}

parse_benchmark_options() {
  BENCHMARK_ENDPOINT="${BENCHMARK_ENDPOINT:-http://127.0.0.1:8000}"
  BENCHMARK_MODEL="${BENCHMARK_MODEL:-RedHatAI/Qwen3-14B-quantized.w4a16}"

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --url)
        [ "$#" -ge 2 ] && [ -n "$2" ] || { echo "--url requires a value" >&2; exit 1; }
        BENCHMARK_ENDPOINT="$2"
        shift 2
        ;;
      --model)
        [ "$#" -ge 2 ] && [ -n "$2" ] || { echo "--model requires a value" >&2; exit 1; }
        BENCHMARK_MODEL="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown benchmark option: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
  done

  BENCHMARK_ENDPOINT="$(normalize_endpoint "$BENCHMARK_ENDPOINT")"
}

run_performance() {
  parse_benchmark_options "$@"
  require_podman

  local tokenizer="${BENCHMARK_TOKENIZER:-RedHatAI/Qwen3-14B-quantized.w4a16}"
  local run_timestamp
  run_timestamp="$(date '+%Y%m%d-%H%M%S')"
  mkdir -p "$BENCHMARK_CACHE_DIR"
  [ -w "$BENCHMARK_CACHE_DIR" ] || {
    echo "Benchmark cache is not writable: $BENCHMARK_CACHE_DIR" >&2
    exit 1
  }
  local cache_args=(
    --volume "$BENCHMARK_CACHE_DIR:/cache:Z"
    --env HOME=/cache
    --env HF_HOME=/cache/huggingface
    --env HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub
  )

  mkdir -p "$BENCHMARK_RESULTS_DIR"

  echo "Benchmarking $BENCHMARK_MODEL at $BENCHMARK_ENDPOINT"
  echo "Writing results to $BENCHMARK_RESULTS_DIR/benchmarks-${run_timestamp}.{csv,json}"
  "$PODMAN" run --rm --network host \
    --userns=keep-id \
    --user "$(id -u):$(id -g)" \
    "${cache_args[@]}" \
    --volume "$BENCHMARK_RESULTS_DIR:/results:Z" \
    "$GUIDELLM_IMAGE" run \
    --backend "kind=openai_http,target=$BENCHMARK_ENDPOINT,model=$BENCHMARK_MODEL${BENCHMARK_API_KEY:+,api_key=$BENCHMARK_API_KEY}" \
    --tokenizer "kind=huggingface_auto,model=$tokenizer" \
    --constraint kind=max_duration,seconds=30 \
    --output "kind=csv,path=/results/benchmarks-${run_timestamp}.csv" \
    --output "kind=json,path=/results/benchmarks-${run_timestamp}.json" \
    --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
}

run_quality() {
  parse_benchmark_options "$@"
  require_podman

  local tokenizer="${BENCHMARK_TOKENIZER:-RedHatAI/Qwen3-14B-quantized.w4a16}"
  local quality_tasks="${BENCHMARK_QUALITY_TASKS:-hellaswag,arc_challenge,gsm8k,mmlu}"
  local quality_limit="${BENCHMARK_QUALITY_LIMIT:-}"
  local run_timestamp
  run_timestamp="$(date '+%Y%m%d-%H%M%S')"
  local quality_output_dir="/results/quality/quality-${run_timestamp}"
  local model_args="model=$BENCHMARK_MODEL,base_url=$BENCHMARK_ENDPOINT/v1/completions,tokenizer=$tokenizer"
  if [ -n "${BENCHMARK_API_KEY:-}" ]; then
    model_args+=",auth_token=$BENCHMARK_API_KEY"
  fi
  local quality_args=(
    --model local-completions
    --model_args "${model_args[@]}"
    --tasks "$quality_tasks"
    --apply_chat_template
    --log_samples
    --output_path "$quality_output_dir"
  )
  if [ -n "$quality_limit" ]; then
    quality_args+=(--limit "$quality_limit")
  fi

  mkdir -p "$BENCHMARK_CACHE_DIR"
  [ -w "$BENCHMARK_CACHE_DIR" ] || {
    echo "Benchmark cache is not writable: $BENCHMARK_CACHE_DIR" >&2
    exit 1
  }
  local cache_args=(
    --volume "$BENCHMARK_CACHE_DIR:/cache:Z"
    --env HOME=/cache
    --env HF_HOME=/cache/huggingface
    --env HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub
  )

  mkdir -p "$BENCHMARK_RESULTS_DIR/quality"
  echo "Evaluating $BENCHMARK_MODEL at $BENCHMARK_ENDPOINT"
  echo "Tasks: $quality_tasks"
  echo "Writing results to $BENCHMARK_RESULTS_DIR/quality/quality-${run_timestamp}"
  "$PODMAN" run --rm --network host \
    --userns=keep-id \
    --user "$(id -u):$(id -g)" \
    "${cache_args[@]}" \
    --volume "$BENCHMARK_RESULTS_DIR:/results:Z" \
    "$LM_EVAL_IMAGE" \
    "${quality_args[@]}"
}

case "${1:-}" in
  install)
    [ "$#" -eq 1 ] || { usage >&2; exit 1; }
    install_tools
    ;;
  benchmark|performance)
    shift
    run_performance "$@"
    ;;
  quality)
    shift
    run_quality "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
