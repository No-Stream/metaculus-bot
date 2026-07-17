#!/bin/bash
# Sequential driver-replay arms — sequential to avoid AskNews rate-limit
# contention between processes (each process has its own rate-gate semaphore).
# GOOGLE_API_KEY is prefixed by the tmux launcher, not written to any file.
set -uo pipefail
cd /Users/flatljan/personal/metaculus-bot || exit 1

run_arm() {
  local model="$1" effort="$2" arm="$3"
  echo "=== $(date '+%H:%M:%S') START ${arm} (${model} effort=${effort})"
  uv run python scratch/driver_replay_2026-07-17/replay.py "${model}" "${effort}" "${arm}"
  echo "=== $(date '+%H:%M:%S') EXIT=$? ${arm}"
}

run_arm "openai/gpt-5.6-terra" "low" "arm_terra_low"
run_arm "openai/gpt-5.6-terra" "medium" "arm_terra_medium"
run_arm "openai/gpt-5.6-sol" "low" "arm_sol_low"
run_arm "anthropic/claude-sonnet-5" "medium" "arm_sonnet5_medium"
run_arm "openai/gpt-5.6-luna" "medium" "arm_luna_medium"
echo "=== ALL ARMS DONE ==="
