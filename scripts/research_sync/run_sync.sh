#!/usr/bin/env bash
#
# Wrapper for the scheduled archive sync (launchd job
# com.metaculusbot.research-sync). Cd's to the repo, runs `make sync_all`, and
# appends a timestamped, dated logfile.
#
# WHY: GHA uploads each bot run's research_outputs/ AND run_logs/ artifacts with
# retention-days: 90. After 90 days the artifacts are gone forever and the local
# archives (backtests/research_archive/ and backtests/telemetry_archive/) are the
# only durable copies. `make sync_all` pulls BOTH — research + run-log telemetry — so
# neither silently goes stale. This wrapper is what the launchd job invokes weekly so
# the pull happens automatically, well inside the retention window. See README.md.
#
# launchd runs jobs with a minimal PATH (typically /usr/bin:/bin:/usr/sbin:/sbin),
# so `uv` and `gh` are NOT on PATH by default. We prepend their known locations.
#
# WAKE RACE: launchd runs a job missed while asleep at the next wake, and at that
# moment Wi-Fi has typically not re-associated yet. All five scheduled runs from
# 2026-06-28 through 2026-07-26 died this way — a ConnectionError ("Network is
# unreachable" / NameResolutionError) inside the first network call, with nothing to
# notice it. So this wrapper now (a) waits for the network before invoking make and
# (b) makes a failure impossible to miss, since a silent weekly failure is what turned
# one bug into six weeks of staleness against a 90-day retention window.

set -euo pipefail

# Resolve the repo root relative to this script (scripts/research_sync/run_sync.sh)
# so the wrapper works regardless of where the repo is cloned. launchd invokes this
# by absolute path, so BASH_SOURCE[0] is absolute and the cd's resolve correctly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

# Prepend the dirs holding uv (~/.local/bin) and gh (Homebrew) so launchd can find them.
export PATH="${HOME}/.local/bin:/opt/homebrew/bin:/usr/local/bin:${PATH}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/sync_$(date +%Y-%m-%d).log"
FAILURE_SENTINEL="${LOG_DIR}/LAST_SYNC_FAILED"

# Network-readiness preflight. Polls a cheap unauthenticated endpoint on the host we
# actually need (api.github.com — the GHA artifact source) until it answers or we give
# up after NETWORK_WAIT_TRIES * NETWORK_WAIT_SLEEP_S seconds. Both hosts we hit come up
# together, so one probe is enough; probing GitHub rather than Metaculus checks the half
# whose data expires.
NETWORK_WAIT_TRIES=30
NETWORK_WAIT_SLEEP_S=10

wait_for_network() {
  local attempt
  for ((attempt = 1; attempt <= NETWORK_WAIT_TRIES; attempt++)); do
    if curl -sf -m 5 https://api.github.com/ >/dev/null 2>&1; then
      echo "network ready after ${attempt} probe(s)"
      return 0
    fi
    echo "network not ready (probe ${attempt}/${NETWORK_WAIT_TRIES}); sleeping ${NETWORK_WAIT_SLEEP_S}s"
    sleep "${NETWORK_WAIT_SLEEP_S}"
  done
  echo "network still unreachable after $((NETWORK_WAIT_TRIES * NETWORK_WAIT_SLEEP_S))s; aborting"
  return 1
}

# Failure is announced three ways so it cannot pass unnoticed like the 2026-06/07 run of
# silent failures did: a sentinel file (greppable, and the thing to check when the
# archives look stale), a macOS notification, and a non-zero exit for launchd.
report_failure() {
  local stage="$1"
  {
    echo "STAGE=${stage}"
    echo "FAILED_AT=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "LOG=${LOG_FILE}"
  } >"${FAILURE_SENTINEL}"
  osascript -e "display notification \"research-sync failed at ${stage}. See ${LOG_FILE}\" with title \"Metaculus bot archive sync FAILED\"" >/dev/null 2>&1 || true
}

cd "${REPO_DIR}"

{
  echo "=========================================================="
  echo "research-sync starting at $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "PATH=${PATH}"
  echo "=========================================================="
  if ! wait_for_network; then
    report_failure "network-preflight"
    exit 1
  fi
  if ! make sync_all; then
    report_failure "make sync_all"
    exit 1
  fi
  # Clear the sentinel only on a fully green run, so a stale one from a previous
  # failure never masks recovery — or claims failure after one.
  rm -f "${FAILURE_SENTINEL}"
  echo "research-sync finished OK at $(date '+%Y-%m-%d %H:%M:%S %z')"
} >>"${LOG_FILE}" 2>&1
