#!/usr/bin/env bash
#
# Wrapper for the scheduled research-archive sync (launchd job
# com.metaculusbot.research-sync). Cd's to the repo, runs `make sync_research`,
# and appends a timestamped, dated logfile.
#
# WHY: GHA uploads each bot run's research_outputs/ artifact with retention-days: 90.
# After 90 days the artifact is gone forever and backtests/research_archive/ is the
# only durable copy. This wrapper is what the launchd job invokes weekly so the pull
# happens automatically, well inside the retention window. See README.md for install.
#
# launchd runs jobs with a minimal PATH (typically /usr/bin:/bin:/usr/sbin:/sbin),
# so `uv` and `gh` are NOT on PATH by default. We prepend their known locations.

set -euo pipefail

REPO_DIR="/Users/flatljan/personal/metaculus-bot"
LOG_DIR="${REPO_DIR}/scripts/research_sync/logs"

# Prepend the dirs holding uv (~/.local/bin) and gh (Homebrew) so launchd can find them.
export PATH="/Users/flatljan/.local/bin:/opt/homebrew/bin:/usr/local/bin:${PATH}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/sync_$(date +%Y-%m-%d).log"

cd "${REPO_DIR}"

{
  echo "=========================================================="
  echo "research-sync starting at $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "PATH=${PATH}"
  echo "=========================================================="
  make sync_research
  echo "research-sync finished OK at $(date '+%Y-%m-%d %H:%M:%S %z')"
} >>"${LOG_FILE}" 2>&1
