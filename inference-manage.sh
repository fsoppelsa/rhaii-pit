#!/usr/bin/env bash

set -euo pipefail

# ============================================================
#  CONFIGURATION  —  edit these
# ============================================================
# MODEL: a catalog key (deepseek_r1_qwen_14b_awq, qwen3_4b, qwen3_14b,
#        granite_8b, llama31_8b, whiterabbit_7b_awq) or a raw Hugging Face id.
MODEL="deepseek_r1_qwen_14b_awq"
UPSTREAM=0                       # 0 = Red Hat AI Inference, 1 = upstream vLLM
BACKEND_PORT=8000                # vLLM OpenAI API port
PROXY_PORT=8001                  # reasoning-hiding proxy port
CACHE_DIR="$HOME/rhaii-cache"    # Hugging Face model cache
WEBUI_HOST="fedyagpt.local"      # WebUI hostname (also the TLS certificate CN)
WEBUI_PORT=443                   # WebUI HTTPS port

# ============================================================
#  DO NOT EDIT BELOW THIS LINE
# ============================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Matching environment variables, when set, override the config above.
RHAII_PORT="${RHAII_PORT:-$BACKEND_PORT}"
RHAII_PROXY_PORT="${RHAII_PROXY_PORT:-$PROXY_PORT}"
RHAII_CACHE_DIR="${RHAII_CACHE_DIR:-$CACHE_DIR}"
WEBUI_PUBLIC_HOST="${WEBUI_PUBLIC_HOST:-$WEBUI_HOST}"
RHAII_API_KEY="${RHAII_API_KEY:-}"   # normally set with --bearer
RHAII_UPSTREAM="${RHAII_UPSTREAM:-$UPSTREAM}"   # 0 = Red Hat, 1 = upstream vLLM

# Component scripts
RHAII_SCRIPT="$ROOT_DIR/rhaii-universal.sh"
WEBUI_SCRIPT="$ROOT_DIR/start-webui.sh"
RHAII_PROXY_SCRIPT="$ROOT_DIR/reasoning_proxy.py"

# Container names
RHAII_CONTAINER_NAME="rhaii"
WEBUI_CONTAINER_NAME="open-webui"
WEBUI_PROXY_CONTAINER_NAME="open-webui-tls-proxy"

# Logs and proxy pid file
RHAII_PROXY_PID_FILE="$ROOT_DIR/logs/rhaii-proxy.pid"
RHAII_PROXY_LOG_FILE="$ROOT_DIR/logs/rhaii-proxy.log"
RHAII_BACKEND_BOOT_LOG_FILE="$ROOT_DIR/logs/rhaii-backend-bootstrap.log"
RHAII_STARTUP_LOG_STREAM=1

# Backend readiness polling
RHAII_WAIT_TIMEOUT_SEC=180
RHAII_WAIT_INTERVAL_SEC=3

# ------------------------------
# Helpers
# ------------------------------
usage() {
  cat <<'EOF'
Usage: inference-manage.sh <command> [options]

Commands:
  start rhaii          Start the Red Hat AI Inference backend (headless).
  start upstream       Start the upstream vLLM backend (headless).
  stop                 Stop every component that is running (backend, proxy, WebUI).
  smoke-test           Probe the already-running backend and report whether the
                       model answers (non-zero exit if it does not).

Start options (combinable with 'start rhaii' or 'start upstream'):
  --bearer <key>       Require this Bearer token on the API (enables vLLM auth).
                       Overrides the RHAII_API_KEY environment variable.
  --with-proxy         Also start the reasoning-hiding proxy.
  --with-ui            Also start Open WebUI. It talks to the proxy when
                       --with-proxy is given, otherwise directly to the backend.
  --smoke-test         After startup, send a test prompt to the backend and
                       fail (non-zero exit) if the model does not answer.
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "$1 is required but not found" >&2; exit 1; }
}

container_exists() {
  podman ps -a --format '{{.Names}}' | grep -qx "$1"
}

container_running() {
  podman ps --format '{{.Names}}' | grep -qx "$1"
}

resolve_model_id() {
  case "$1" in
    deepseek_r1_qwen_14b_awq) echo "casperhansen/deepseek-r1-distill-qwen-14b-awq" ;;
    qwen3_4b)                 echo "Qwen/Qwen3-4B-Instruct-2507" ;;
    qwen3_14b)                echo "RedHatAI/Qwen3-14B-quantized.w4a16" ;;
    granite_8b)               echo "ibm-granite/granite-3.3-8b-instruct" ;;
    llama31_8b)               echo "meta-llama/Llama-3.1-8B-Instruct" ;;
    whiterabbit_7b_awq)       echo "solidrust/WhiteRabbitNeo-7B-v1.5a-AWQ" ;;
    *)                        echo "$1" ;;
  esac
}

model_is_cached() {
  local model_id="$1"
  local cache_key="${model_id//\//--}"
  local snapshot_dir="${RHAII_CACHE_DIR}/huggingface/hub/models--${cache_key}/snapshots"
  [ -d "$snapshot_dir" ] && find "$snapshot_dir" -mindepth 1 -maxdepth 1 -type d | grep -q .
}

stop_startup_log_stream() {
  if [ -n "${RHAII_LOG_STREAM_PID:-}" ] && kill -0 "$RHAII_LOG_STREAM_PID" >/dev/null 2>&1; then
    kill "$RHAII_LOG_STREAM_PID" >/dev/null 2>&1 || true
    wait "$RHAII_LOG_STREAM_PID" 2>/dev/null || true
  fi
  unset RHAII_LOG_STREAM_PID
}

stop_container_if_running() {
  local name="$1"
  if container_running "$name"; then
    echo "Stopping container: $name"
    podman stop "$name" >/dev/null
    echo "Stopped: $name"
  else
    echo "Container not running: $name"
  fi
}

stop_proxy_if_running() {
  local pid
  local stopped=0

  if [ -f "$RHAII_PROXY_PID_FILE" ]; then
    pid="$(cat "$RHAII_PROXY_PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "Stopping reasoning proxy from pid file: pid=$pid"
      kill "$pid" >/dev/null 2>&1 || true
      sleep 1
      kill -0 "$pid" >/dev/null 2>&1 && kill -9 "$pid" >/dev/null 2>&1 || true
      stopped=1
    fi
  fi

  if pgrep -af "python3 .*${RHAII_PROXY_SCRIPT}" >/dev/null 2>&1; then
    echo "Stopping reasoning proxy by command match: $RHAII_PROXY_SCRIPT"
    pkill -f "python3 .*${RHAII_PROXY_SCRIPT}" >/dev/null 2>&1 || true
    sleep 1
    pgrep -af "python3 .*${RHAII_PROXY_SCRIPT}" >/dev/null 2>&1 && pkill -9 -f "python3 .*${RHAII_PROXY_SCRIPT}" >/dev/null 2>&1 || true
    stopped=1
  fi

  [ "$stopped" -eq 0 ] && echo "Reasoning proxy not running." || echo "Stopped reasoning proxy."
  rm -f "$RHAII_PROXY_PID_FILE"
}

# Send a minimal chat completion to the backend and confirm a non-empty answer.
# Any extra arguments are passed through to curl (used for the auth header).
run_smoke_test() {
  local auth_args=("$@")
  local base="http://127.0.0.1:${RHAII_PORT}/v1"
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
  local with_proxy="$1"
  local with_ui="$2"
  local with_smoke="$3"

  # Model resolution
  local launch_model_key="$MODEL"
  local launch_model=""
  if [[ "$MODEL" == */* ]]; then
    launch_model_key="custom"
    launch_model="$MODEL"
  fi
  local resolved_model_id
  resolved_model_id="$(resolve_model_id "$MODEL")"

  if [ -z "${HF_HUB_OFFLINE:-}" ]; then
    if model_is_cached "$resolved_model_id"; then HF_HUB_OFFLINE=1; else HF_HUB_OFFLINE=0; fi
  fi

  # ----------------------------------------------------------
  #  Settings summary
  # ----------------------------------------------------------
  local backend_label
  if [ "${RHAII_UPSTREAM:-0}" = "1" ]; then
    backend_label="Upstream vLLM"
  else
    backend_label="Red Hat AI Inference"
  fi
  local proxy_label
  if [ "$with_proxy" -eq 1 ]; then proxy_label="enabled  (port $RHAII_PROXY_PORT)"; else proxy_label="disabled"; fi
  local ui_label
  if [ "$with_ui" -eq 1 ]; then ui_label="enabled  (https://${WEBUI_PUBLIC_HOST}:${WEBUI_PORT})"; else ui_label="disabled"; fi
  local auth_label
  if [ -n "$RHAII_API_KEY" ]; then auth_label="enabled"; else auth_label="open"; fi

  echo "================================================================"
  echo "  Inference Stack Settings"
  echo "================================================================"
  printf "  %-18s %s\n" "Backend:" "$backend_label"
  printf "  %-18s %s (%s)\n" "Model:" "$MODEL" "$resolved_model_id"
  printf "  %-18s %s\n" "Backend port:" "$RHAII_PORT"
  printf "  %-18s %s\n" "Proxy:" "$proxy_label"
  printf "  %-18s %s\n" "WebUI:" "$ui_label"
  printf "  %-18s %s\n" "Cache dir:" "$RHAII_CACHE_DIR"
  printf "  %-18s %s\n" "HF offline:" "$HF_HUB_OFFLINE"
  printf "  %-18s %s\n" "Auth:" "$auth_label"
  echo "================================================================"
  echo

  # Common preflight
  require_cmd podman
  require_cmd curl

  if [ ! -x "$RHAII_SCRIPT" ]; then
    echo "RHAII script not found or not executable: $RHAII_SCRIPT" >&2
    exit 1
  fi

  if container_exists "$RHAII_CONTAINER_NAME"; then
    echo "Container '$RHAII_CONTAINER_NAME' already exists. Run 'stop' first." >&2
    exit 1
  fi

  # Proxy preflight
  if [ "$with_proxy" -eq 1 ]; then
    require_cmd python3
    if [ ! -f "$RHAII_PROXY_SCRIPT" ]; then
      echo "RHAII proxy script not found: $RHAII_PROXY_SCRIPT" >&2
      exit 1
    fi
    if [ -f "$RHAII_PROXY_PID_FILE" ]; then
      local proxy_pid
      proxy_pid="$(cat "$RHAII_PROXY_PID_FILE" 2>/dev/null || true)"
      if [ -n "$proxy_pid" ] && kill -0 "$proxy_pid" >/dev/null 2>&1; then
        echo "Reasoning proxy already running (pid=$proxy_pid). Run 'stop' first." >&2
        exit 1
      fi
      rm -f "$RHAII_PROXY_PID_FILE"
    fi
    if pgrep -af "python3 .*${RHAII_PROXY_SCRIPT}" >/dev/null 2>&1; then
      echo "Reasoning proxy already running for script $RHAII_PROXY_SCRIPT. Run 'stop' first." >&2
      exit 1
    fi
  fi

  # UI preflight
  if [ "$with_ui" -eq 1 ]; then
    require_cmd sudo
    if [ ! -x "$WEBUI_SCRIPT" ]; then
      echo "WebUI script not found or not executable: $WEBUI_SCRIPT" >&2
      exit 1
    fi
    if container_exists "$WEBUI_CONTAINER_NAME"; then
      echo "Container '$WEBUI_CONTAINER_NAME' already exists. Run 'stop' first." >&2
      exit 1
    fi
    if container_exists "$WEBUI_PROXY_CONTAINER_NAME"; then
      echo "Container '$WEBUI_PROXY_CONTAINER_NAME' already exists. Run 'stop' first." >&2
      exit 1
    fi
  fi

  # Kill stale VLLM::EngineCore processes that survived previous container teardown.
  if command -v nvidia-smi >/dev/null 2>&1; then
    local stale_pids
    stale_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    if [ -n "$stale_pids" ]; then
      for pid in $stale_pids; do
        if ps -p "$pid" -o comm= 2>/dev/null | grep -q "VLLM::EngineCore"; then
          echo "Killing stale VLLM::EngineCore (PID $pid) holding GPU memory..."
          kill -9 "$pid" 2>/dev/null || sudo kill -9 "$pid" 2>/dev/null || true
        fi
      done
      sleep 1
    fi
  fi

  # Auth arguments
  local curl_auth_args=()
  local rhaii_start_args=()
  if [ -n "$RHAII_API_KEY" ]; then
    curl_auth_args=(-H "Authorization: Bearer $RHAII_API_KEY")
    rhaii_start_args=(--api-key "$RHAII_API_KEY")
  fi

  # Privileged port binding is only needed for the WebUI TLS proxy on :443.
  if [ "$with_ui" -eq 1 ]; then
    echo "Requesting sudo to enable privileged port binding for rootless Podman..."
    sudo -v
    sudo sysctl -w net.ipv4.ip_unprivileged_port_start=443 >/dev/null
  fi

  mkdir -p "$ROOT_DIR/logs"
  : >"$RHAII_BACKEND_BOOT_LOG_FILE"

  echo "Starting $backend_label backend (MODEL=$MODEL)..."
  echo "Bootstrap log: $RHAII_BACKEND_BOOT_LOG_FILE"
  RHAII_FOLLOW_LOGS=0 \
  RHAII_UPSTREAM="$RHAII_UPSTREAM" \
  RHAII_CACHE_DIR="$RHAII_CACHE_DIR" \
  HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  MODEL_KEY="$launch_model_key" \
  MODEL="$launch_model" \
  "$RHAII_SCRIPT" "${rhaii_start_args[@]}" >"$RHAII_BACKEND_BOOT_LOG_FILE" 2>&1 </dev/null &
  local rhaii_start_pid=$!

  echo "Waiting for $backend_label API at http://127.0.0.1:$RHAII_PORT/v1/models ..."
  local deadline=$((SECONDS + RHAII_WAIT_TIMEOUT_SEC))
  local ready=0
  local launcher_exited_early=0
  local startup_log_stream_started=0
  while [ "$SECONDS" -lt "$deadline" ]; do
    if [ "$startup_log_stream_started" -ne 1 ] && [ "$RHAII_STARTUP_LOG_STREAM" = "1" ] && container_running "$RHAII_CONTAINER_NAME"; then
      echo "Streaming RHAII startup logs..."
      podman logs -f "$RHAII_CONTAINER_NAME" &
      RHAII_LOG_STREAM_PID=$!
      startup_log_stream_started=1
    fi
    if curl -fsS --max-time 2 "${curl_auth_args[@]}" "http://127.0.0.1:${RHAII_PORT}/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$rhaii_start_pid" >/dev/null 2>&1; then
      launcher_exited_early=1
    fi
    sleep "$RHAII_WAIT_INTERVAL_SEC"
  done

  stop_startup_log_stream

  if [ "$ready" -ne 1 ]; then
    echo "$backend_label did not become ready within ${RHAII_WAIT_TIMEOUT_SEC}s." >&2
    if kill -0 "$rhaii_start_pid" >/dev/null 2>&1; then
      echo "Launcher process is still running (pid=$rhaii_start_pid)." >&2
    elif [ "$launcher_exited_early" -eq 1 ]; then
      wait "$rhaii_start_pid" || true
      echo "Launcher process exited before the API became ready." >&2
      if [ -s "$RHAII_BACKEND_BOOT_LOG_FILE" ]; then
        echo "Backend bootstrap output:" >&2
        tail -n 80 "$RHAII_BACKEND_BOOT_LOG_FILE" >&2 || true
      fi
    fi
    echo "Recent RHAII logs:" >&2
    podman logs --tail 80 "$RHAII_CONTAINER_NAME" >&2 || true
    exit 1
  fi

  echo "$backend_label is ready."

  # Upstream the WebUI will talk to: the proxy if present, else the backend directly.
  local webui_upstream_port="$RHAII_PORT"

  if [ "$with_proxy" -eq 1 ]; then
    echo "Starting reasoning-hiding proxy on http://127.0.0.1:${RHAII_PROXY_PORT}/v1 ..."
    UPSTREAM_BASE="http://127.0.0.1:${RHAII_PORT}" \
    LISTEN_HOST="0.0.0.0" \
    LISTEN_PORT="$RHAII_PROXY_PORT" \
    python3 "$RHAII_PROXY_SCRIPT" >"$RHAII_PROXY_LOG_FILE" 2>&1 &
    local rhaii_proxy_pid=$!
    echo "$rhaii_proxy_pid" >"$RHAII_PROXY_PID_FILE"

    local proxy_deadline=$((SECONDS + RHAII_WAIT_TIMEOUT_SEC))
    local proxy_ready=0
    while [ "$SECONDS" -lt "$proxy_deadline" ]; do
      if curl -fsS --max-time 2 "${curl_auth_args[@]}" "http://127.0.0.1:${RHAII_PROXY_PORT}/v1/models" >/dev/null 2>&1; then
        proxy_ready=1
        break
      fi
      if ! kill -0 "$rhaii_proxy_pid" >/dev/null 2>&1; then
        break
      fi
      sleep "$RHAII_WAIT_INTERVAL_SEC"
    done

    if [ "$proxy_ready" -ne 1 ]; then
      echo "Reasoning proxy did not become ready within ${RHAII_WAIT_TIMEOUT_SEC}s." >&2
      if kill -0 "$rhaii_proxy_pid" >/dev/null 2>&1; then
        kill "$rhaii_proxy_pid" >/dev/null 2>&1 || true
        sleep 1
        kill -0 "$rhaii_proxy_pid" >/dev/null 2>&1 && kill -9 "$rhaii_proxy_pid" >/dev/null 2>&1 || true
      fi
      rm -f "$RHAII_PROXY_PID_FILE"
      if [ -f "$RHAII_PROXY_LOG_FILE" ]; then
        echo "Recent reasoning proxy logs:" >&2
        tail -n 80 "$RHAII_PROXY_LOG_FILE" >&2 || true
      fi
      exit 1
    fi

    webui_upstream_port="$RHAII_PROXY_PORT"
  fi

  if [ "$with_ui" -eq 1 ]; then
    echo "Starting Open WebUI + TLS reverse proxy..."
    OPENAI_API_BASE_URL="http://host.containers.internal:${webui_upstream_port}/v1" \
    OPENAI_API_KEY="${RHAII_API_KEY:-EMPTY}" \
    "$WEBUI_SCRIPT"

    if ! container_running "$WEBUI_CONTAINER_NAME"; then
      echo "Open WebUI container is not running after startup." >&2
      exit 1
    fi
    if ! container_running "$WEBUI_PROXY_CONTAINER_NAME"; then
      echo "Open WebUI TLS proxy container is not running after startup." >&2
      exit 1
    fi
  fi

  if ! container_running "$RHAII_CONTAINER_NAME"; then
    echo "RHAII container is not running after startup." >&2
    exit 1
  fi

  echo
  echo "Stack is up."
  echo "$backend_label API: http://127.0.0.1:${RHAII_PORT}/v1"
  [ "$with_proxy" -eq 1 ] && echo "Filtered API: http://127.0.0.1:${RHAII_PROXY_PORT}/v1"
  [ "$with_ui" -eq 1 ] && echo "WebUI URL: https://${WEBUI_PUBLIC_HOST}:${WEBUI_PORT}"
  echo "Model: ${MODEL}"
  if [ "$with_ui" -eq 1 ]; then
    echo
    echo "Note: self-signed TLS cert is used; browser warning is expected until trusted."
  fi

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
  [ -n "$RHAII_API_KEY" ] && auth_args=(-H "Authorization: Bearer $RHAII_API_KEY")
  run_smoke_test "${auth_args[@]}"
}

# ------------------------------
# stop
# ------------------------------
cmd_stop() {
  require_cmd podman
  # Tear down in reverse order: TLS proxy, WebUI, reasoning proxy, backend.
  stop_container_if_running "$WEBUI_PROXY_CONTAINER_NAME"
  stop_container_if_running "$WEBUI_CONTAINER_NAME"
  stop_proxy_if_running
  stop_container_if_running "$RHAII_CONTAINER_NAME"
  echo "Shutdown complete."
}

# ------------------------------
# Dispatch
# ------------------------------
cmd="${1:-}"
shift || true
case "$cmd" in
  start)
    sub="${1:-}"
    shift || true
    case "$sub" in
      rhaii)   RHAII_UPSTREAM=0 ;;
      upstream) RHAII_UPSTREAM=1 ;;
      *)
        echo "'start' requires a backend argument: 'rhaii' or 'upstream'" >&2
        echo "Try: inference-manage.sh start rhaii [options]" >&2
        usage
        exit 1
        ;;
    esac
    with_proxy=0
    with_ui=0
    with_smoke=0
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --bearer)
          if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
            echo "--bearer requires a non-empty value" >&2; usage; exit 1
          fi
          RHAII_API_KEY="$2"
          shift
          ;;
        --with-proxy) with_proxy=1 ;;
        --with-ui) with_ui=1 ;;
        --smoke-test) with_smoke=1 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
      esac
      shift
    done
    cmd_start "$with_proxy" "$with_ui" "$with_smoke"
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
