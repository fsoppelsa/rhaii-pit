#!/usr/bin/env bash

set -euo pipefail

# ============================================================
#  CONFIGURATION  —  edit these
# ============================================================
RAG_IMAGE="docker.io/opensearchproject/opensearch:2.19.3"
RAG_CONTAINER_NAME="rhaii-opensearch"
RAG_BIND_HOST="127.0.0.1"              # Keep loopback-only unless TLS trust is configured for remote clients.
RAG_PORT=9200
RAG_DATA_DIR="$HOME/rhaii-opensearch-data"
RAG_STATE_DIR="$HOME/.config/rhaii/rag"
RAG_HEAP_SIZE="1g"
RAG_FIXTURE_INDEX="rhaii-rag-fixtures-v1"

# ============================================================
#  DO NOT EDIT BELOW THIS LINE
# ============================================================
RAG_PASSWORD_FILE="$RAG_STATE_DIR/opensearch-admin-password"
RAG_MIN_VM_MAX_MAP_COUNT=262144

usage() {
  cat <<'EOF'
Usage: ./rag.sh <command>

Commands:
  setup       Create local state, start OpenSearch, and load deterministic fixtures.
  start       Start the independent local OpenSearch service.
  stop        Stop OpenSearch; persistent data and fixtures remain intact.
  status      Report authenticated cluster health.
  logs        Follow OpenSearch container logs.

The service listens only on 127.0.0.1 by default. It uses the OpenSearch
security demo configuration with a generated, local admin password. This is
suitable for local development only; do not expose it to a LAN without
replacing the demo TLS configuration and creating a restricted Pi credential.

Pi owns embedding, vector indexing, retrieval, and RHAII context/tool calls.
The fixtures are text documents for connectivity and lexical-search validation,
not a substitute for a production embedding pipeline.
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "$1 is required but not found" >&2; exit 1; }
}

container_exists() {
  podman ps -a --format '{{.Names}}' | grep -qx "$RAG_CONTAINER_NAME"
}

container_running() {
  podman ps --format '{{.Names}}' | grep -qx "$RAG_CONTAINER_NAME"
}

check_vm_max_map_count() {
  local current
  current="$(sysctl -n vm.max_map_count 2>/dev/null || true)"
  if [ -z "$current" ] || [ "$current" -lt "$RAG_MIN_VM_MAX_MAP_COUNT" ]; then
    echo "vm.max_map_count must be at least $RAG_MIN_VM_MAX_MAP_COUNT for OpenSearch." >&2
    echo "Run: sudo sysctl -w vm.max_map_count=$RAG_MIN_VM_MAX_MAP_COUNT" >&2
    exit 1
  fi
}

ensure_password() {
  if [ -s "$RAG_PASSWORD_FILE" ]; then
    return
  fi

  require_cmd openssl
  mkdir -p "$RAG_STATE_DIR"
  umask 077
  openssl rand -base64 36 >"$RAG_PASSWORD_FILE"
  chmod 600 "$RAG_PASSWORD_FILE"
  echo "Created local OpenSearch admin password: $RAG_PASSWORD_FILE"
}

password() {
  tr -d '\r\n' <"$RAG_PASSWORD_FILE"
}

api_url() {
  printf 'https://%s:%s' "$RAG_BIND_HOST" "$RAG_PORT"
}

api_curl() {
  curl --fail --silent --show-error --insecure \
    --user "admin:$(password)" "$@"
}

wait_until_ready() {
  local deadline=$((SECONDS + 120))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if api_curl "$(api_url)/_cluster/health?wait_for_status=yellow&timeout=2s" >/dev/null 2>&1; then
      return
    fi
    if ! container_running; then
      echo "OpenSearch container exited before becoming ready." >&2
      podman logs --tail 80 "$RAG_CONTAINER_NAME" >&2 || true
      exit 1
    fi
    sleep 2
  done

  echo "OpenSearch did not become ready within 120s." >&2
  podman logs --tail 80 "$RAG_CONTAINER_NAME" >&2 || true
  exit 1
}

cmd_start() {
  require_cmd podman
  require_cmd curl
  check_vm_max_map_count
  ensure_password

  if container_running; then
    echo "OpenSearch is already running at $(api_url)."
    return
  fi
  if container_exists; then
    echo "OpenSearch container exists but is not running. Run './rag.sh stop' first." >&2
    exit 1
  fi

  mkdir -p "$RAG_DATA_DIR"
  echo "Starting OpenSearch at $(api_url) ..."
  podman run -d \
    --name "$RAG_CONTAINER_NAME" \
    --userns=keep-id \
    -p "$RAG_BIND_HOST:$RAG_PORT:9200" \
    -e "discovery.type=single-node" \
    -e "OPENSEARCH_JAVA_OPTS=-Xms$RAG_HEAP_SIZE -Xmx$RAG_HEAP_SIZE" \
    -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=$(password)" \
    -v "$RAG_DATA_DIR:/usr/share/opensearch/data:Z" \
    "$RAG_IMAGE" >/dev/null

  wait_until_ready
  echo "OpenSearch is ready at $(api_url)."
}
cmd_stop() {
  require_cmd podman
  if ! container_exists; then
    echo "OpenSearch is not running."
    return
  fi
  podman rm -f "$RAG_CONTAINER_NAME" >/dev/null
  echo "OpenSearch stopped. Persistent data remains at $RAG_DATA_DIR."
}

cmd_status() {
  require_cmd podman
  require_cmd curl
  if ! container_running; then
    echo "OpenSearch is not running." >&2
    exit 1
  fi
  api_curl "$(api_url)/_cluster/health?pretty"
}

load_fixtures() {
  api_curl -X PUT "$(api_url)/$RAG_FIXTURE_INDEX" \
    -H 'Content-Type: application/json' \
    -d '{"settings":{"number_of_shards":1,"number_of_replicas":0},"mappings":{"properties":{"title":{"type":"text"},"content":{"type":"text"},"category":{"type":"keyword"},"source":{"type":"keyword"}}}}' \
    >/dev/null 2>&1 || true

  api_curl -X POST "$(api_url)/_bulk" \
    -H 'Content-Type: application/x-ndjson' \
    --data-binary @- <<EOF >/dev/null
{"index":{"_index":"$RAG_FIXTURE_INDEX","_id":"platform-overview"}}
{"title":"RHAII platform overview","category":"platform","source":"fixture:platform-overview","content":"RHAII exposes an OpenAI-compatible vLLM API on /v1. Pi is the client and performs retrieval before sending grounded context to RHAII."}
{"index":{"_index":"$RAG_FIXTURE_INDEX","_id":"rag-conventions"}}
{"title":"RAG conventions","category":"rag","source":"fixture:rag-conventions","content":"OpenSearch stores indexed documents. Pi owns embeddings, vector queries, reranking, and context assembly. RHAII does not query OpenSearch directly."}
{"index":{"_index":"$RAG_FIXTURE_INDEX","_id":"sql-safety"}}
{"title":"SQL safety conventions","category":"sql","source":"fixture:sql-safety","content":"Generate read-only SQL only. Validate every statement with EXPLAIN before execution. Retrieve authoritative schema before referring to a table or column. Restrict output with a LIMIT."}
EOF

  api_curl -X POST "$(api_url)/$RAG_FIXTURE_INDEX/_refresh" >/dev/null
  echo "Loaded deterministic fixtures into index: $RAG_FIXTURE_INDEX"
}

cmd_setup() {
  cmd_start
  load_fixtures
  echo "Fixture verification:"
  api_curl "$(api_url)/$RAG_FIXTURE_INDEX/_search?pretty" \
    -H 'Content-Type: application/json' \
    -d '{"query":{"match":{"content":"authoritative schema"}},"size":1}'
}

cmd="${1:-}"
case "$cmd" in
  setup) cmd_setup ;;
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  logs)
    require_cmd podman
    podman logs -f "$RAG_CONTAINER_NAME"
    ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 1 ;;
esac
