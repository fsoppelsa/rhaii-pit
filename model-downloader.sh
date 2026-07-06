#!/usr/bin/env bash

set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

# ============================================================
#  CONFIGURATION  —  edit these
# ============================================================
DEFAULT_MODEL_KEY="deepseek_r1_qwen_14b_awq"   # default model key (env MODEL_KEY/MODEL override)
CACHE_DIR="$HOME/rhaii-cache"                  # Hugging Face model cache
OFFLINE=1                                      # 1 = require local model, 0 = allow download
TOKEN_FILE="$HOME/HF_TOKEN"                    # token file used when downloading

# ============================================================
#  DO NOT EDIT BELOW THIS LINE
# ============================================================
# The launcher scripts call this downloader with MODEL_KEY, MODEL,
# RHAII_CACHE_DIR, HF_HUB_OFFLINE and HF_TOKEN set in the environment; those
# still take precedence over the defaults above.

usage() {
  cat <<'EOF'
Usage: ./model-downloader.sh

Downloads or verifies a model in the local Hugging Face cache used by RHAII.

Environment:
  MODEL_KEY            Known local model key, for example: deepseek_r1_qwen_14b_awq
  MODEL                Raw Hugging Face model id, for example: casperhansen/deepseek-r1-distill-qwen-14b-awq
  RHAII_CACHE_DIR      Cache root. Default: $HOME/rhaii-cache
  HF_HUB_OFFLINE       1 = require model to already exist locally, 0 = allow download
  HF_TOKEN             Hugging Face token. Required when HF_HUB_OFFLINE=0
  HF_TOKEN_FILE        Token file fallback. Default: $HOME/HF_TOKEN
  MODEL_DOWNLOAD_FORCE 1 = force snapshot_download even if already cached
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

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

MODEL_KEY="${MODEL_KEY:-$DEFAULT_MODEL_KEY}"
MODEL="${MODEL:-}"
CACHE_DIR="${RHAII_CACHE_DIR:-$CACHE_DIR}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-$OFFLINE}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$TOKEN_FILE}"
MODEL_DOWNLOAD_FORCE="${MODEL_DOWNLOAD_FORCE:-0}"
PYTHON_BIN="${RHAII_MODEL_DOWNLOADER_PYTHON:-python3}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

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

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3 with huggingface_hub is required but not found" >&2
  exit 1
fi

if "$PYTHON_BIN" -c 'import huggingface_hub' >/dev/null 2>&1; then
  :
else
  echo "python3 does not have huggingface_hub installed" >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
  HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
fi

CACHE_HUB_DIR="$CACHE_DIR/huggingface/hub"
CACHE_KEY="${MODEL//\//--}"
SNAPSHOT_DIR="$CACHE_HUB_DIR/models--${CACHE_KEY}/snapshots"

model_is_cached() {
  [ -d "$SNAPSHOT_DIR" ] && find "$SNAPSHOT_DIR" -mindepth 1 -maxdepth 1 -type d | grep -q .
}

mkdir -p "$CACHE_HUB_DIR"
WRITE_PROBE="$CACHE_HUB_DIR/.rhaii-write-test-$$"
: >"$WRITE_PROBE"
rm -f "$WRITE_PROBE"

echo "Model downloader"
echo "  model id       : $MODEL"
echo "  model key      : $MODEL_KEY"
echo "  cache dir      : $CACHE_DIR"
echo "  HF_HUB_OFFLINE : $HF_HUB_OFFLINE"

if model_is_cached && [ "$MODEL_DOWNLOAD_FORCE" != "1" ]; then
  echo "Model already cached."
  exit 0
fi

if [ "$HF_HUB_OFFLINE" = "1" ]; then
  echo "Model is not cached locally and HF_HUB_OFFLINE=1." >&2
  echo "Set HF_HUB_OFFLINE=0 to allow downloading." >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is required when downloading a model." >&2
  exit 1
fi

echo "Downloading model into local cache..."
RHAII_MODEL_ID="$MODEL" \
RHAII_CACHE_HUB_DIR="$CACHE_HUB_DIR" \
HF_TOKEN="$HF_TOKEN" \
HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
"$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download
import os

repo_id = os.environ["RHAII_MODEL_ID"]
cache_dir = os.environ["RHAII_CACHE_HUB_DIR"]
token = os.environ["HF_TOKEN"]

snapshot_path = snapshot_download(
    repo_id=repo_id,
    cache_dir=cache_dir,
    token=token,
)

print(f"Downloaded snapshot: {snapshot_path}")
PY

if model_is_cached; then
  echo "Model download completed."
else
  echo "Model download did not produce a cached snapshot." >&2
  exit 1
fi
