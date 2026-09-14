#!/usr/bin/env bash
# Link this checkout's gitignored artifact trees to the private artifacts repo.
#
# The bot's local artifacts (research archives, telemetry, residual-round scratch) are ~3 GB of
# logs and data. They cannot live in this repo: it is a PUBLIC fork, and 3 GB of blobs in a
# public history is both permanent and full of run logs. They also cannot be regenerated, because
# GitHub Actions deletes the source artifacts at 90 days and backtests/gha_artifact_store/ is the
# only surviving copy (see the Makefile's resync_from_store target).
#
# So they live in a separate PRIVATE repo, No-Stream/metaculus-bot-artifacts, cloned as a sibling
# directory, and this script symlinks each tree back into place. Every artifact path in the code
# is built relative to the repo root and no .resolve() is applied to the artifact dirs themselves,
# so the symlinks are transparent to the pipeline, the analysis modules and the test suite.
#
# Raw files rather than compressed archives, deliberately: the residual rounds hold ten
# near-duplicate copies of perf_all_tagged.json, and git's delta compression packs four of them
# from 308 MB to 27 MB (11.4x), where zstd -19 on the same bytes manages only 5.4x. Tarballs
# would defeat the delta chain that makes this cheap.
#
# Usage:
#   scripts/artifacts_link.sh              # create or verify the symlinks
#   scripts/artifacts_link.sh --check      # report only, change nothing, non-zero if unlinked
#   scripts/artifacts_link.sh --migrate    # FIRST RUN on the machine that holds the data:
#                                          # move each real directory into the artifacts repo,
#                                          # then symlink it back
#
# Override the artifacts repo location with METACULUS_BOT_ARTIFACTS_DIR.

set -o errexit
set -o nounset
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="${METACULUS_BOT_ARTIFACTS_DIR:-$(dirname "$REPO_ROOT")/metaculus-bot-artifacts}"
CLONE_URL="https://github.com/No-Stream/metaculus-bot-artifacts.git"

# Each entry is a gitignored tree that must survive a machine move. scratch_docs_and_planning/ is
# absent on purpose: it mixes tracked design docs with untracked ones, so it cannot be a symlink.
UNITS=(
  backtests
  benchmarks
  research_outputs
  run_logs
  scratch
)

MODE="link"
case "${1:-}" in
  --check) MODE="check" ;;
  --migrate) MODE="migrate" ;;
  "") ;;
  *)
    echo "unknown argument: $1 (expected --check, --migrate, or nothing)" >&2
    exit 2
    ;;
esac

if [[ ! -d $ARTIFACTS_DIR/.git ]]; then
  echo "No artifacts repo at $ARTIFACTS_DIR" >&2
  echo "Clone it first:  git clone $CLONE_URL $ARTIFACTS_DIR" >&2
  echo "Or point METACULUS_BOT_ARTIFACTS_DIR at an existing checkout." >&2
  exit 1
fi

exit_code=0

# A relative link keeps the pair valid when both checkouts move together, but it only resolves
# when the artifacts repo really is a sibling. Anywhere else (METACULUS_BOT_ARTIFACTS_DIR pointing
# at an external disk, say) an absolute link is the only correct one.
ARTIFACTS_DIR="$(cd "$ARTIFACTS_DIR" && pwd)"
if [[ "$(dirname "$ARTIFACTS_DIR")" == "$(dirname "$REPO_ROOT")" ]]; then
  LINK_PREFIX="../$(basename "$ARTIFACTS_DIR")"
else
  LINK_PREFIX="$ARTIFACTS_DIR"
fi

for unit in "${UNITS[@]}"; do
  local_path="$REPO_ROOT/$unit"
  target="$ARTIFACTS_DIR/$unit"
  relative_target="$LINK_PREFIX/$unit"

  if [[ -L $local_path ]]; then
    if [[ "$(readlink "$local_path")" == "$relative_target" ]]; then
      printf '  ok       %-18s -> %s\n' "$unit" "$relative_target"
    else
      printf '  WRONG    %-18s -> %s (expected %s)\n' "$unit" "$(readlink "$local_path")" "$relative_target"
      exit_code=1
    fi
    continue
  fi

  if [[ -d $local_path ]]; then
    if [[ $MODE != "migrate" ]]; then
      printf '  UNLINKED %-18s is a real directory; run --migrate to move it into the artifacts repo\n' "$unit"
      exit_code=1
      continue
    fi
    if [[ -e $target ]]; then
      printf '  CONFLICT %-18s exists in both checkouts; resolve by hand\n' "$unit"
      exit_code=1
      continue
    fi
    mv "$local_path" "$target"
    ln -s "$relative_target" "$local_path"
    printf '  migrated %-18s -> %s\n' "$unit" "$relative_target"
    continue
  fi

  if [[ $MODE == "check" ]]; then
    printf '  MISSING  %-18s not linked\n' "$unit"
    exit_code=1
    continue
  fi

  if [[ ! -d $target ]]; then
    mkdir -p "$target"
  fi
  ln -s "$relative_target" "$local_path"
  printf '  linked   %-18s -> %s\n' "$unit" "$relative_target"
done

exit "$exit_code"
