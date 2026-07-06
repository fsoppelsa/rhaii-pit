#!/usr/bin/env bash

set -euo pipefail

# ============================================================
#  CONFIGURATION  —  edit these
# ============================================================
WEBUI_IMAGE="ghcr.io/open-webui/open-webui:v0.9.4"
WEBUI_NAME="FedyaGPT"               # branding shown in the UI
WEBUI_PUBLIC_HOST="fedyagpt.local"  # hostname and TLS certificate CN
WEBUI_PORT=443                      # HTTPS port
WEBUI_DATA_DIR="$HOME/open-webui-data"   # persistent WebUI data
WEBUI_CERT_DIR="$HOME/open-webui-certs"  # TLS cert + key location
DEFAULT_MODELS="qwen3_14b"          # model preselected in the UI
ENABLE_SIGNUP="True"                # allow new user registration
ENABLE_IMAGE_GENERATION="True"
ENABLE_TITLE_GENERATION="False"

# ============================================================
#  DO NOT EDIT BELOW THIS LINE
# ============================================================
WEBUI_CONTAINER_NAME="open-webui"
WEBUI_PROXY_CONTAINER_NAME="open-webui-tls-proxy"
WEBUI_NETWORK_NAME="open-webui-net"
WEBUI_BIND_HOST="0.0.0.0"
WEBUI_CONTAINER_PORT="8080"
WEBUI_MAX_UPLOAD_SIZE="100m"
WEBUI_HOST_ALIAS="host.containers.internal"
WEBUI_CERT_FILE="$WEBUI_CERT_DIR/open-webui.crt"
WEBUI_KEY_FILE="$WEBUI_CERT_DIR/open-webui.key"
WEBUI_AUTH="True"
WEBUI_URL="https://$WEBUI_PUBLIC_HOST:$WEBUI_PORT"

# Upstream model API — inference-manage.sh passes these in; env overrides the defaults.
OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL:-http://$WEBUI_HOST_ALIAS:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

if ! command -v podman >/dev/null 2>&1; then
  echo "podman is required but not found" >&2
  exit 1
fi

mkdir -p "$WEBUI_DATA_DIR"
mkdir -p "$WEBUI_CERT_DIR"

if [ ! -s "$WEBUI_CERT_FILE" ] || [ ! -s "$WEBUI_KEY_FILE" ]; then
  echo "Creating self-signed TLS certificate in $WEBUI_CERT_DIR"
  WEBUI_OPENSSL_CONF="$(mktemp)"
  cat >"$WEBUI_OPENSSL_CONF" <<EOF
[req]
default_bits = 4096
prompt = no
default_md = sha256
x509_extensions = v3_req
distinguished_name = dn

[dn]
CN = ${WEBUI_PUBLIC_HOST}

[v3_req]
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = ${WEBUI_PUBLIC_HOST}
DNS.2 = localhost
IP.1 = 127.0.0.1
EOF
  openssl req -x509 -newkey rsa:4096 -sha256 -nodes \
    -keyout "$WEBUI_KEY_FILE" \
    -out "$WEBUI_CERT_FILE" \
    -days 365 \
    -config "$WEBUI_OPENSSL_CONF" >/dev/null 2>&1
  rm -f "$WEBUI_OPENSSL_CONF"
  chmod 600 "$WEBUI_KEY_FILE"
fi

if podman ps -a --format '{{.Names}}' | grep -qx "$WEBUI_CONTAINER_NAME"; then
  echo "Container '$WEBUI_CONTAINER_NAME' already exists. Remove/rename it or set WEBUI_CONTAINER_NAME." >&2
  exit 1
fi

if podman ps -a --format '{{.Names}}' | grep -qx "$WEBUI_PROXY_CONTAINER_NAME"; then
  echo "Container '$WEBUI_PROXY_CONTAINER_NAME' already exists. Remove/rename it or set WEBUI_PROXY_CONTAINER_NAME." >&2
  exit 1
fi

if ! podman network exists "$WEBUI_NETWORK_NAME"; then
  podman network create "$WEBUI_NETWORK_NAME" >/dev/null
fi

WEBUI_NGINX_CONF="$(mktemp)"
cat >"$WEBUI_NGINX_CONF" <<EOF
events {}
http {
  server {
    listen 443 ssl;
    server_name ${WEBUI_PUBLIC_HOST};
    client_max_body_size ${WEBUI_MAX_UPLOAD_SIZE};
    ssl_certificate /etc/nginx/certs/open-webui.crt;
    ssl_certificate_key /etc/nginx/certs/open-webui.key;

    location / {
      proxy_pass http://open-webui-upstream:${WEBUI_CONTAINER_PORT};
      proxy_http_version 1.1;
      proxy_set_header Host \$host;
      proxy_set_header X-Real-IP \$remote_addr;
      proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
      proxy_set_header X-Forwarded-Proto https;
      proxy_set_header Upgrade \$http_upgrade;
      proxy_set_header Connection "upgrade";
      proxy_read_timeout 300s;
      proxy_send_timeout 300s;
      proxy_buffering off;
    }
  }
}
EOF

echo "Starting Open WebUI container: $WEBUI_CONTAINER_NAME"
echo "WebUI URL: https://$WEBUI_PUBLIC_HOST:$WEBUI_PORT"
echo "Model API URL: $OPENAI_API_BASE_URL"

podman run -d --rm \
  --name "$WEBUI_CONTAINER_NAME" \
  --network "$WEBUI_NETWORK_NAME" \
  --network-alias open-webui-upstream \
  --add-host "$WEBUI_HOST_ALIAS:host-gateway" \
  -e "HOST=$WEBUI_BIND_HOST" \
  -e "PORT=$WEBUI_CONTAINER_PORT" \
  -e "WEBUI_URL=$WEBUI_URL" \
  -e "WEBUI_NAME=$WEBUI_NAME" \
  -e "ENABLE_TITLE_GENERATION=$ENABLE_TITLE_GENERATION" \
  -e "ENABLE_SIGNUP=$ENABLE_SIGNUP" \
  -e "ENABLE_IMAGE_GENERATION=$ENABLE_IMAGE_GENERATION" \
  -e "DEFAULT_MODELS=$DEFAULT_MODELS" \
  -e "OPENAI_API_BASE_URL=$OPENAI_API_BASE_URL" \
  -e "OPENAI_API_KEY=$OPENAI_API_KEY" \
  -e "WEBUI_AUTH=$WEBUI_AUTH" \
  -v "$WEBUI_DATA_DIR:/app/backend/data:Z" \
  "$WEBUI_IMAGE" >/dev/null

podman run -d --rm \
  --name "$WEBUI_PROXY_CONTAINER_NAME" \
  --network "$WEBUI_NETWORK_NAME" \
  -p "$WEBUI_BIND_HOST:$WEBUI_PORT:443" \
  -v "$WEBUI_NGINX_CONF:/etc/nginx/nginx.conf:ro,Z" \
  -v "$WEBUI_CERT_FILE:/etc/nginx/certs/open-webui.crt:ro,Z" \
  -v "$WEBUI_KEY_FILE:/etc/nginx/certs/open-webui.key:ro,Z" \
  docker.io/library/nginx:alpine >/dev/null

rm -f "$WEBUI_NGINX_CONF"

echo "Open WebUI is running."
echo "User registration/auth enabled: WEBUI_AUTH=$WEBUI_AUTH, ENABLE_SIGNUP=$ENABLE_SIGNUP"
echo "Per-user accounts/sessions and document uploads are enabled through Open WebUI auth."
echo "Image generation toggle requested: ENABLE_IMAGE_GENERATION=$ENABLE_IMAGE_GENERATION (requires configured image backend in Admin > Settings > Images)."
