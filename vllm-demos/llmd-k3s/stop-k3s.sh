#!/bin/bash
# Stop the llm-d simulator and shut down K3s (minimal — no Prometheus RBAC).
# Usage: bash llmd-k3s/stop-k3s.sh

set -euo pipefail

NAMESPACE="llm-d-demo"

echo "========================================="
echo "  llm-d Demo — Shutdown K3s"
echo "========================================="
echo ""

# ── 1. Remove the namespace (simulator) ─────────────────────────────────────
if kubectl get namespace "${NAMESPACE}" &>/dev/null; then
  echo "→ Removing resources in ${NAMESPACE}..."
  kubectl delete namespace "${NAMESPACE}" --timeout=30s || true
  echo "  Namespace removed."
else
  echo "→ Namespace ${NAMESPACE} does not exist, no cleanup needed."
fi

echo ""

# ── 2. Stop K3s ─────────────────────────────────────────────────────────────
if systemctl is-active --quiet k3s 2>/dev/null; then
  echo "→ Stopping K3s..."
  sudo -n systemctl stop k3s
  echo "  K3s stopped."
else
  echo "→ K3s was not running."
fi

echo ""
echo "========================================="
echo "  Shutdown complete."
echo "========================================="
