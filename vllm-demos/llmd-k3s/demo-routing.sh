#!/bin/bash
# Demo B: llm-d Intelligent Routing (on minimal K3s)
# 3 simulator replicas behind a Service; shows why round-robin is
# not optimal for LLMs and introduces llm-d's intelligent routing.
#
# NB: per-pod metrics are read from the NODE with curl to the pod IPs
# (the simulator image has no curl, so `kubectl exec ... curl` would fail).
# No Prometheus/Grafana dependency.

set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────────────
BOLD="\e[1m"
RESET="\e[0m"
CYAN="\e[1;36m"
YELLOW="\e[1;33m"
GREEN="\e[1;32m"
RED="\e[1;31m"
MAGENTA="\e[1;35m"
DIM="\e[2m"

NAMESPACE="llm-d-demo"
SIM_URL="${SIM_URL:-http://localhost:31800}"
MODEL="meta-llama/Llama-3.1-8B-Instruct"

echo -e "${CYAN}=========================================${RESET}"
echo -e "${CYAN}  Demo B: llm-d — Intelligent Inference${RESET}"
echo -e "${CYAN}  Routing on Kubernetes${RESET}"
echo -e "${CYAN}=========================================${RESET}"
echo ""
sleep 1

echo -e "${YELLOW}→ Model server replicas (simulated):${RESET}"
kubectl -n "${NAMESPACE}" get pods -o wide
echo ""
sleep 0.5

echo -e "${YELLOW}→ Service endpoints (round-robin across 3 pods):${RESET}"
kubectl -n "${NAMESPACE}" get endpoints vllm-sim-service
echo ""
sleep 1

echo -e "${CYAN}=========================================${RESET}"
echo -e "${CYAN}  Test 1: Round-Robin ${DIM}(K8s default)${RESET}"
echo -e "${CYAN}=========================================${RESET}"
echo ""
sleep 0.5

RR_PROMPTS=(
  "What is Kubernetes and why is it important for deploying LLMs?"
  "Explain the difference between a Pod and a Deployment in Kubernetes"
  "How does load balancing work in a K8s cluster?"
  "What is a NodePort Service in Kubernetes?"
  "Why is round-robin not optimal for serving LLM models?"
)

for i in 1 2 3 4 5; do
  PROMPT="${RR_PROMPTS[$((i-1))]}"
  echo -e "${YELLOW}--- Request $i ---${RESET}"
  echo -e "${DIM}  prompt: \"${PROMPT}\"${RESET}"
  curl -s --max-time 10 "${SIM_URL}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":20}" \
    &>/dev/null && echo -e "${GREEN}  → OK${RESET}" || echo -e "${RED}  → [connection error]${RESET}"
  echo ""
  sleep 0.5
done

# Ready replicas (Service endpoints are the source of truth)
PODS=($(kubectl -n "${NAMESPACE}" get endpoints vllm-sim-service -o jsonpath='{.subsets[*].addresses[*].targetRef.name}' 2>/dev/null))
IPS=($(kubectl -n "${NAMESPACE}" get endpoints vllm-sim-service -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null))

sleep 1
echo -e "${CYAN}=========================================${RESET}"
echo -e "${CYAN}  Metrics per pod — one replica saturated${RESET}"
echo -e "${CYAN}=========================================${RESET}"
echo ""
sleep 0.5

# Saturate the first replica with background requests
LOAD_PIDS=()
SAT_POD_SHORT="${PODS[0]##*-}"
echo -e "${YELLOW}→ Saturating replica ${SAT_POD_SHORT}: 4 background generators hit its pod IP directly (bypassing the Service)...${RESET}"
for _ in $(seq 1 4); do
  ( while true; do
      curl -s --max-time 30 "http://${IPS[0]}:8000/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"Background workload: a long-running generation that keeps this replica busy\",\"max_tokens\":200}" \
        &>/dev/null
    done
  ) &
  LOAD_PIDS+=($!)
done
sleep 2  # let the queue build

echo -e "${YELLOW}→ KV cache / queue depth for each replica:${RESET}"
echo ""

BOX_WIDTH=62
BOX_LINE=$(printf '═%.0s' $(seq 1 "$BOX_WIDTH"))
echo -e "${MAGENTA}╔${BOX_LINE}╗${RESET}"
printf "${MAGENTA}║${RESET}  ${BOLD}%-${BOX_WIDTH}s${MAGENTA}║${RESET}\n" "vLLM METRICS — KV CACHE & QUEUE DEPTH"
echo -e "${MAGENTA}╠${BOX_LINE}╣${RESET}"

for pod in $(kubectl -n "${NAMESPACE}" get endpoints vllm-sim-service -o jsonpath='{.subsets[*].addresses[*].targetRef.name}' 2>/dev/null); do
  POD_IP=$(kubectl -n "${NAMESPACE}" get pod "${pod}" -o jsonpath='{.status.podIP}' 2>/dev/null) || continue
  [ -n "${POD_IP}" ] || continue
  printf "${MAGENTA}║${RESET}  ${CYAN}%-${BOX_WIDTH}s${MAGENTA}║${RESET}\n" "Pod: ${pod} (${POD_IP})"
  if [ "${POD_IP}" = "${IPS[0]}" ]; then
    printf "${MAGENTA}║${RESET}  ${YELLOW}%-${BOX_WIDTH}s${MAGENTA}║${RESET}\n" "  ← saturated: background load hits this pod directly"
  fi
  printf "${MAGENTA}║${RESET}  ${DIM}%-${BOX_WIDTH}s${MAGENTA}║${RESET}\n" "──────────────────────────────────────────────────────────"

  METRICS=$(curl -s --max-time 3 "http://${POD_IP}:8000/metrics" 2>/dev/null | \
    grep -E "^(vllm:num_requests_running|vllm:num_requests_waiting)" | head -4)

  if [ -n "${METRICS}" ]; then
    while IFS= read -r line; do
      printf "${MAGENTA}║${RESET}  ${GREEN}%-${BOX_WIDTH}s${MAGENTA}║${RESET}\n" "${line}"
    done <<< "${METRICS}"
  else
    printf "${MAGENTA}║${RESET}  ${DIM}%-${BOX_WIDTH}s${MAGENTA}║${RESET}\n" "(metrics not available)"
  fi

  printf "${MAGENTA}║${RESET}  %-${BOX_WIDTH}s${MAGENTA}║${RESET}\n" ""
  sleep 0.5
done

echo -e "${MAGENTA}╚${BOX_LINE}╝${RESET}"
echo ""

# ── Test 2a: round-robin while one replica is saturated ────────────────────
echo -e "${CYAN}=========================================${RESET}"
echo -e "${CYAN}  Test 2a: Round-Robin ${DIM}(K8s default)${RESET}"
echo -e "${CYAN}=========================================${RESET}"
echo ""
echo -e "${BOLD}  #   Pod       Latency   Status${RESET}"
echo -e "${DIM}  ────────────────────────────${RESET}"

pod_log_count() {
  local pod="$1" n
  n=$(kubectl -n "${NAMESPACE}" logs "${pod}" --tail=200 2>/dev/null | grep -c "completion request received") || true
  [ -n "${n}" ] || n=0
  echo "${n}"
}

RR_DURS=()
RR_OK=0
for i in $(seq 1 12); do
  for idx in "${!PODS[@]}"; do
    BEFORE[$idx]=$(pod_log_count "${PODS[$idx]}")
  done

  PROMPT_IDX=$(( (i - 1) % ${#RR_PROMPTS[@]} ))
  PROMPT="${RR_PROMPTS[$PROMPT_IDX]}"
  START_NS=$(date +%s%N)
  RESULT=$(curl -s --max-time 90 "${SIM_URL}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":20}" 2>/dev/null) || true
  END_NS=$(date +%s%N)
  DURATION_MS=$(( (END_NS - START_NS) / 1000000 ))
  RR_DURS+=("${DURATION_MS}")

  # Attribute the request: idle pods only serve our requests, so the pod
  # whose log grew by exactly 1 served it. The saturated pod is excluded
  # from the scan — its background churn also produces +1 by coincidence —
  # and is only blamed when no idle pod incremented.
  SERVED_SHORT="${SAT_POD_SHORT}"
  for idx in "${!PODS[@]}"; do
    [ "${idx}" -eq 0 ] && continue
    AFTER[$idx]=$(pod_log_count "${PODS[$idx]}")
    DELTA=$(( AFTER[$idx] - BEFORE[$idx] ))
    if [ "${DELTA}" -eq 1 ]; then
      SERVED_SHORT="${PODS[$idx]##*-}"
      break
    fi
  done

  if echo "${RESULT}" | python3 -c "import sys,json; json.load(sys.stdin)" &>/dev/null; then
    STATUS_C="${GREEN}OK${RESET}"
    RR_OK=$((RR_OK+1))
  else
    STATUS_C="${RED}ERR${RESET}"
  fi
  printf "  %-3s %-10s %6d ms   " "${i}" "${SERVED_SHORT}" "${DURATION_MS}"
  echo -e "${STATUS_C}"
done

# ── Test 2b: load-aware — pick the least-loaded replica ────────────────────
echo ""
echo -e "${CYAN}=========================================${RESET}"
echo -e "${CYAN}  Test 2b: Load-aware ${DIM}(llm-d style)${RESET}"
echo -e "${CYAN}=========================================${RESET}"
echo ""
echo -e "${BOLD}  #   Pod       Latency   Status${RESET}"
echo -e "${DIM}  ────────────────────────────${RESET}"

LA_DURS=()
LA_OK=0
for i in $(seq 1 12); do
  BEST_IDX=0
  BEST_LOAD=999999
  for idx in "${!IPS[@]}"; do
    LOAD=$(curl -s --max-time 2 "http://${IPS[$idx]}:8000/metrics" 2>/dev/null \
      | grep -E "^vllm:num_requests_(running|waiting)" \
      | grep -oE '[0-9]+$' \
      | awk '{s+=$1} END {print s+0}') || true
    [ -n "${LOAD}" ] || LOAD=999
    if [ "${LOAD}" -lt "${BEST_LOAD}" ]; then
      BEST_LOAD="${LOAD}"
      BEST_IDX="${idx}"
    fi
  done

  POD_SHORT="${PODS[$BEST_IDX]##*-}"
  PROMPT_IDX=$(( (i - 1) % ${#RR_PROMPTS[@]} ))
  PROMPT="${RR_PROMPTS[$PROMPT_IDX]}"
  START_NS=$(date +%s%N)
  RESULT=$(curl -s --max-time 90 "http://${IPS[$BEST_IDX]}:8000/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":20}" 2>/dev/null) || true
  END_NS=$(date +%s%N)
  DURATION_MS=$(( (END_NS - START_NS) / 1000000 ))
  LA_DURS+=("${DURATION_MS}")
  if echo "${RESULT}" | python3 -c "import sys,json; json.load(sys.stdin)" &>/dev/null; then
    STATUS_C="${GREEN}OK${RESET}"
    LA_OK=$((LA_OK+1))
  else
    STATUS_C="${RED}ERR${RESET}"
  fi
  printf "  %-3s %-10s %6d ms   " "${i}" "${POD_SHORT}" "${DURATION_MS}"
  echo -e "${STATUS_C}"
done

# ── Comparison ──────────────────────────────────────────────────────────────
RR_MIN=999999; RR_SUM=0; RR_MAX=0
for d in "${RR_DURS[@]}"; do
  [ "${d}" -lt "${RR_MIN}" ] && RR_MIN="${d}"
  [ "${d}" -gt "${RR_MAX}" ] && RR_MAX="${d}"
  RR_SUM=$((RR_SUM + d))
done
RR_AVG=$((RR_SUM / ${#RR_DURS[@]}))

LA_MIN=999999; LA_SUM=0; LA_MAX=0
for d in "${LA_DURS[@]}"; do
  [ "${d}" -lt "${LA_MIN}" ] && LA_MIN="${d}"
  [ "${d}" -gt "${LA_MAX}" ] && LA_MAX="${d}"
  LA_SUM=$((LA_SUM + d))
done
LA_AVG=$((LA_SUM / ${#LA_DURS[@]}))

echo ""
echo -e "${CYAN}=========================================${RESET}"
echo -e "${CYAN}  Pass comparison${RESET}"
echo -e "${CYAN}=========================================${RESET}"
echo ""
echo -e "  ${RED}round-robin${RESET} (K8s default):   min ${RR_MIN} ms   avg ${RR_AVG} ms   max ${RR_MAX} ms   OK ${RR_OK}/12"
echo -e "  ${GREEN}load-aware${RESET} (llm-d style):   min ${LA_MIN} ms   avg ${LA_AVG} ms   max ${LA_MAX} ms   OK ${LA_OK}/12"
echo ""

# Stop the background load
for pid in "${LOAD_PIDS[@]}"; do
  kill "${pid}" 2>/dev/null || true
done
wait "${LOAD_PIDS[@]}" 2>/dev/null || true

echo ""
cat "$(dirname "$0")/../llmd-gateway-showcase.yaml"
