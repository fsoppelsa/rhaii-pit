#!/bin/bash
# Demo C: Parallel load test on the llm-d simulator (minimal K3s)
# Shows how latencies vary with the Service's round-robin and explains
# how llm-d would optimize the load distribution.
#
# Usage: bash llmd-k3s/demo-load.sh [N_REQUESTS]
#      N_REQUESTS=20 bash llmd-k3s/demo-load.sh

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

SIM_URL="${SIM_URL:-http://localhost:31800}"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
N_REQUESTS="${N_REQUESTS:-10}"

echo -e "${CYAN}=========================================${RESET}"
echo -e "${CYAN}  Demo C: Load Test — ${BOLD}${N_REQUESTS} parallel requests${RESET}"
echo -e "${DIM}  Server: ${SIM_URL}${RESET}"
echo -e "${CYAN}=========================================${RESET}"
echo ""
sleep 1
echo -e "${YELLOW}→ Sending ${N_REQUESTS} requests in parallel...${RESET}"
echo ""
sleep 0.5

TMPDIR_RESULTS=$(mktemp -d)
trap 'rm -rf "${TMPDIR_RESULTS}"' EXIT

for i in $(seq 1 "${N_REQUESTS}"); do
  (
    PROMPT="Explain PagedAttention in one sentence: Request ${i}"
    START_NS=$(date +%s%N)
    RESULT=$(curl -s --max-time 90 "${SIM_URL}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":30}" 2>/dev/null) || true
    END_NS=$(date +%s%N)
    DURATION_MS=$(( (END_NS - START_NS) / 1000000 ))

    if echo "${RESULT}" | python3 -c "import sys,json; json.load(sys.stdin)" &>/dev/null; then
      STATUS="OK"
    else
      STATUS="ERR"
    fi

    echo "${i} ${DURATION_MS} ${STATUS}" > "${TMPDIR_RESULTS}/req_${i}.txt"
  ) &
done

wait

echo -e "${BOLD}  #   Latency   Status${RESET}"
echo -e "${DIM}  ──────────────────${RESET}"

TOTAL_MS=0
OK_COUNT=0
MAX_MS=0
MIN_MS=999999

for i in $(seq 1 "${N_REQUESTS}"); do
  if [ -f "${TMPDIR_RESULTS}/req_${i}.txt" ]; then
    read -r REQ_ID DURATION_MS STATUS < "${TMPDIR_RESULTS}/req_${i}.txt"
    if [ "${STATUS}" = "OK" ]; then
      STATUS_COLOR="${GREEN}${STATUS}${RESET}"
    else
      STATUS_COLOR="${RED}${STATUS}${RESET}"
    fi
    printf "  %-3s %6s ms   " "${REQ_ID}" "${DURATION_MS}"
    echo -e "${STATUS_COLOR}"
    TOTAL_MS=$(( TOTAL_MS + DURATION_MS ))
    [ "${STATUS}" = "OK" ] && OK_COUNT=$(( OK_COUNT + 1 ))
    [ "${DURATION_MS}" -gt "${MAX_MS}" ] && MAX_MS="${DURATION_MS}"
    [ "${DURATION_MS}" -lt "${MIN_MS}" ] && MIN_MS="${DURATION_MS}"
    sleep 0.05
  fi
done

AVG_MS=0
[ "${N_REQUESTS}" -gt 0 ] && AVG_MS=$(( TOTAL_MS / N_REQUESTS ))

echo ""
sleep 0.5
echo -e "${DIM}  ──────────────────────────────────────${RESET}"
printf "  Successful requests: ${GREEN}%d${RESET} / %d\n" "${OK_COUNT}" "${N_REQUESTS}"
printf "  Min latency:         ${GREEN}%d ms${RESET}\n" "${MIN_MS}"
printf "  Max latency:         ${RED}%d ms${RESET}\n" "${MAX_MS}"
printf "  Average latency:     ${YELLOW}%d ms${RESET}\n" "${AVG_MS}"
echo -e "${DIM}  ──────────────────────────────────────${RESET}"
echo ""
echo -e "${CYAN}=========================================${RESET}"
