#!/bin/bash
# Start K3s and prepare the environment for the llm-d demo.
#
# Deploys the 3-replica simulator (deterministic: echo mode, 1 worker,
# 200 ms/token) plus Prometheus + Grafana for the closing dashboard view.
# The demo scripts themselves read per-pod metrics directly with curl from
# the node to the pod IPs — no Prometheus dependency in the narrative.
#
# Usage: bash llmd-k3s/start-k3s.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="llm-d-demo"

echo "========================================="
echo "  llm-d Demo — Setup K3s"
echo "========================================="
echo ""

# ── 1. Start K3s if it is not already running ───────────────────────────────
if ! systemctl is-active --quiet k3s 2>/dev/null; then
  echo "→ Starting K3s..."
  sudo -n systemctl start k3s
  echo "  K3s started."
else
  echo "→ K3s already running."
fi

# ── 2. Make the kubeconfig readable ─────────────────────────────────────────
# kubectl here is a symlink to k3s and reads /etc/rancher/k3s/k3s.yaml (root-only
# by default); without this step `kubectl` fails with "permission denied".
echo "→ Making the kubeconfig readable..."
sudo -n chmod 644 /etc/rancher/k3s/k3s.yaml

# ── 3. Wait for the node to be Ready ────────────────────────────────────────
echo "→ Waiting for the node to be Ready..."
until kubectl get nodes 2>/dev/null | grep -q " Ready"; do
  sleep 2
  printf "."
done
echo ""
echo "  Node ready."

# ── 4. Apply the simulator (3 replicas) ─────────────────────────────────────
echo "→ Applying llmd-simulator.yaml..."
kubectl apply -f "${SCRIPT_DIR}/llmd-simulator.yaml"

echo "→ Waiting for the rollout (max 90s)..."
kubectl -n "${NAMESPACE}" rollout status deployment/vllm-simulator --timeout=90s

# ── 5. Apply the monitoring stack (closing visual) ─────────────────────────
echo "→ Applying prometheus.yaml..."
kubectl apply -f "${SCRIPT_DIR}/prometheus.yaml"
kubectl -n "${NAMESPACE}" rollout status deployment/prometheus --timeout=90s

echo "→ Applying grafana.yaml..."
kubectl apply -f "${SCRIPT_DIR}/grafana.yaml"
kubectl -n "${NAMESPACE}" rollout status deployment/grafana --timeout=90s

echo ""
echo "→ Running pods:"
kubectl -n "${NAMESPACE}" get pods -o wide
echo ""

# ── 6. Verify the NodePort (31800) responds ─────────────────────────────────
SIM_URL="http://localhost:31800"
echo "→ Verifying endpoint ${SIM_URL}/v1/models..."
READY=0
for _ in $(seq 1 15); do
  if curl -sf --max-time 3 "${SIM_URL}/v1/models" >/dev/null; then READY=1; break; fi
  sleep 2
done

if [ "${READY}" = "1" ]; then
  echo ""
  echo "========================================="
  echo ""
  echo "    bash llmd-k3s/demo-routing.sh"
  echo "    bash llmd-k3s/demo-load.sh"
  echo ""
  echo "  Grafana dashboard: http://localhost:31901"
  echo "  Prometheus UI:     http://localhost:31900"
  echo ""
  echo "  Per-pod metrics (from the node):"
  echo "========================================="
else
  echo "  WARNING: the simulator does not respond on ${SIM_URL}."
  echo "  Check:"
  echo "    kubectl -n ${NAMESPACE} logs -l app=vllm-sim --tail=20"
  exit 1
fi
