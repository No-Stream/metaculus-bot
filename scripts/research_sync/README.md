# Scheduled archive sync (launchd)

Keeps `backtests/research_archive/` (incl. `raw/`) AND `backtests/telemetry_archive/`
fresh by running `make sync_all` on a weekly schedule.

## Why this exists

Every bot run uploads its `research_outputs/` AND `run_logs/` to GitHub Actions with
`retention-days: 90` (see `.github/workflows/run_bot_on_{tournament,metaculus_cup,minibench}.yaml`
and `test_bot.yaml`). **After 90 days GitHub deletes the artifacts forever, and 90 is a
ceiling we cannot raise** — the API reports `{"days":90,"maximum_allowed_days":90}` for
this repo. So GHA is a staging area, not storage, and local disk is the source of truth
from the moment an artifact is grabbed.

That makes the **persisted artifact store** the first line of defense, ahead of the three
parsed archives below:

- `backtests/gha_artifact_store/<artifact-name>/` — the raw extracted artifact, exactly as
  `gh run download` unzipped it, plus a `_meta.json` carrying `artifact_id` / `name` /
  `created_at` / `run_id` (so an artifact's age is knowable without another API call). One
  dir per artifact NAME, not per run, because a single run can upload more than one (the
  pre-rename test workflows uploaded both `research-*` and `logs-*`). 859 dirs / 38 MB as
  of 2026-08-03, growing ~17 MB/month — no compression or pruning is warranted, and
  nothing is ever pruned on purpose. Downloads land here and STAY; the three archives
  below are all parsed FROM here, so a parse bug is re-runnable for free (see "Offline
  re-parse"). Re-grabbing an artifact already in the store is a no-op — uploads are
  immutable, so only absent or half-extracted dirs are fetched.

The three parsed archives (all gitignored) are the queryable views built from that store:

- `backtests/research_archive/` — per-question research text; feeds the backtest replay
  (`make backtest_with_cache`) and residual / per-provider research attribution.
- `backtests/research_archive/raw/` — the raw research-provider payloads
  (`raw_research_<run_id>.jsonl`, one file per run) that `metaculus_bot.research.raw_log`
  appends to `run_logs/`: each provider's RAW return before formatting (AskNews article
  dicts per phase, native/Gemini raw responses + grounding, market contracts, resolution
  fetches, gap-fill results). This is the durable raw evidence behind every forecast,
  independent of published comments — it makes the AskNews summarizer relevance gate
  auditable after the fact.
- `backtests/telemetry_archive/` — run-log telemetry markers (`EXTRACTION_RUNG`,
  `GAP_FILL_V2`, `GHOST_PRE[_JSON]`, `GHOST_FORECAST[_JSON]`, `OPEN_BOUND_PILING`,
  `CREDIT_*`) harvested from the
  same artifacts; feeds parser-drift watch, gap-fill v2 diagnostics, credit burn-rate,
  and the ghost-vs-published scoring gate.

The pullers are manual (`make sync_research` / `make sync_telemetry`, both wrapped by
`make sync_all`), so without a scheduler the archives silently go stale and old data is
lost. This launchd job runs the pull twice weekly — well inside the 90-day window, with
margin for a missed run.

### The 2026-06/07 silent-failure run (why the hardening below exists)

Every scheduled run from 2026-06-28 through 2026-07-26 failed, and nothing said so. Three
bugs compounded:

1. **Wake race.** launchd runs a job missed while asleep at the next wake, and Wi-Fi has
   not re-associated at that instant. The first network call died with `ConnectionError`
   ("Network is unreachable" / `NameResolutionError`).
2. **Wrong-order abort.** That first call was the Metaculus-comment backfill, and under
   `set -euo pipefail` its failure aborted the recipe before a single GHA artifact
   downloaded — a failure in the half whose data lives forever killed the half that
   expires at 90 days.
3. **No signal.** The failure existed only in a dated logfile nobody reads, so six weeks
   of staleness accumulated silently.

Each is now fixed independently: `run_sync.sh` waits for the network before invoking
make, the backfill is non-fatal in the Makefile recipe, a failed run writes
`logs/LAST_SYNC_FAILED` and fires a macOS notification, and the plist wakes twice a week.

## What it does

`com.metaculusbot.research-sync.plist` invokes `run_sync.sh`, which:

1. `cd`s to the repo,
2. prepends the dirs holding `uv` and `gh` to `PATH` (launchd jobs get a minimal PATH),
3. waits for the network — polls `https://api.github.com/` up to 30 times, 10s apart, and
   aborts with a reported failure if it never answers (the wake race above),
4. runs `make sync_all`, which:
   - backfills from Metaculus comments (`backfill_research_from_comments.py`) FIRST — it
     hits Metaculus, not GHA, and writes `comments_backfill.jsonl` for the research build
     to load. **Non-fatal** (the recipe line is `-`-prefixed): comments stay on Metaculus
     indefinitely, so a backfill failure must never block the expiring GHA pull. On a
     failed backfill the research build just reads the previous run's
     `comments_backfill.jsonl`;
   - then runs the **single-pass** driver `scripts/sync_all.py`, which enumerates EVERY
     `research-*` AND `logs-*` artifact ONCE via the complete, paginated artifacts REST
     endpoint (no 1000-result `gh run list` cap, so nothing in the 90-day window is
     missed), persists each unique artifact ONCE into `backtests/gha_artifact_store/`
     (skipping whatever is already there), and runs all three harvests over those
     persisted run dirs:
     - research JSONL (research-* dirs only) → rebuilds the research archive (download
       records + comment backfill, dedup, build);
     - run-log telemetry markers → merged into the telemetry archive (replace-by-run);
     - `raw_research_<run_id>.jsonl` raw-payload logs → one file per run under
       `backtests/research_archive/raw/` (replace-by-run);
5. appends a dated logfile under `scripts/research_sync/logs/`, and on failure writes
   `logs/LAST_SYNC_FAILED` (naming the stage + logfile), fires a macOS notification, and
   exits non-zero. The sentinel is removed only on a fully green run, so it never masks a
   recovery or outlives one.

Running the driver in one pass avoids the three-pass waste of the old chain (each of the
three standalone `sync_*` targets re-enumerated every artifact and re-downloaded the
overlapping `research-*`/`logs-*` families into its own temp dir). The three standalone
targets (`sync_research` / `sync_telemetry` / `sync_raw_research`) still exist for a
single-archive refresh.

## Offline re-parse (`make resync_from_store`)

```bash
make resync_from_store          # all three archives, rebuilt from local disk, zero network
```

This is what persisting downloads buys. After fixing an ingest or parse bug, the artifacts'
bytes are already on local disk, so the corrected harvest re-runs for free — and keeps
working on artifacts GitHub has since deleted. It rebuilds all three archives from
`backtests/gha_artifact_store/` plus the archive data already on disk, and makes **no
network call at all** (guarded by a test that fails if anything shells out). Every archive
build is replace-by-run, so running it repeatedly is safe.

Each script also takes `--from-store` (and `--store-dir`) directly. Note the distinction in
`download_research.py`: `--rebuild-only` re-merges the records already in `by_qid/`, while
`--from-store` re-reads the persisted JSONL and so can RECOVER records a past ingest bug
dropped, which a rebuild cannot.

One limitation worth knowing: the offline path can't ask GitHub which workflow a run
belonged to, so it recovers that from the telemetry archive's own `runs.jsonl`. Runs already
harvested keep their attribution; a run appearing in the store for the first time during an
offline re-parse reads `workflow: unknown` until the next online sync.

`sync_all` hits only the **read-only, free** GitHub + Metaculus APIs — no paid
LLM/research calls and no publishing.

## Install

`run_sync.sh` self-locates the repo root relative to its own path, so it needs no
edits. The **plist** still hardcodes the absolute path to `run_sync.sh` (launchd
requires an absolute `ProgramArguments` path) and the log locations — **if your repo
path differs, edit the paths in the plist first.** `run_sync.sh` is already executable
(`chmod +x`).

Copy the plist into your per-user `LaunchAgents` directory and bootstrap it:

```bash
# 1. Install the plist into the per-user LaunchAgents dir.
cp /Users/flatljan/personal/metaculus-bot/scripts/research_sync/com.metaculusbot.research-sync.plist \
   ~/Library/LaunchAgents/com.metaculusbot.research-sync.plist

# 2. Bootstrap (load) it into your GUI login session. `gui/$(id -u)` is the
#    per-user domain; `id -u` resolves to your numeric uid.
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.metaculusbot.research-sync.plist
```

On older macOS where `bootstrap` is unavailable, use the legacy command instead:

```bash
launchctl load ~/Library/LaunchAgents/com.metaculusbot.research-sync.plist
```

### Run once immediately (optional sanity check)

```bash
launchctl kickstart -k gui/$(id -u)/com.metaculusbot.research-sync
```

Then watch the dated logfile (below). This is the fastest way to confirm the job
runs end-to-end without waiting for the next scheduled wake.

## Verify it's installed and ran

```bash
# Is the job registered? (prints the label with its last exit status / PID)
launchctl list | grep com.metaculusbot.research-sync

# Full job state (next run time, last exit code, etc.)
launchctl print gui/$(id -u)/com.metaculusbot.research-sync

# Did the last run succeed? Tail the most recent dated logfile.
ls -t /Users/flatljan/personal/metaculus-bot/scripts/research_sync/logs/sync_*.log | head -1 | xargs tail -n 40

# launchd's own stdout/stderr (job-start failures land here, before run_sync.sh runs):
tail -n 40 /Users/flatljan/personal/metaculus-bot/scripts/research_sync/logs/launchd.err.log
```

A healthy run ends with `research-sync finished OK at ...` and the manifest under
`backtests/research_archive/manifest.json` gains an entry per newly-seen question. Don't
read `latest_timestamp` as a freshness signal: it is the timestamp of whichever record won
precedence, not the newest record's, so promoting an artifact over a later comment moves it
BACKWARD (that happened on 255 questions in the 2026-08-03 fix). If
the archive looks stale, **check `logs/LAST_SYNC_FAILED` first** — its presence names the
failing stage and the logfile to read. Then check the logfile itself: the download phase
logs "Artifacts endpoint returned N total, M research-* artifacts", how many downloaded,
records added, and (loudly) any EXPIRED artifact by name + created_at so a short pull or
any data loss is visible.

## Verifying maximal completeness

After a sync, prove the archive captured every live artifact:

```bash
uv run python -m scripts.research_sync.verify_completeness
```

It re-enumerates every live `research-*` artifact via the same paginated endpoint, loads
the rebuilt archive, and prints PASS / FAIL with the exact count of live artifacts vs.
those persisted in the store and represented in the archive — flagging any genuine gap and
any expired (lost-forever) artifact. Read-only and free (GitHub API only).

**Two independent signals, and the store one is the sharper of the two.** A live artifact
absent from `backtests/gha_artifact_store/` FAILS the check: its only copy is on the 90-day
clock. Being absent from the *archive* is far weaker evidence, because most artifacts hold
no research at all — 632 of the 859 on disk carry only `run_logs/`, which is why the archive
holds artifact records from 227 runs rather than 859. The check parses each flagged
artifact's JSONL to tell a genuine gap from a legitimately empty one, reading the persisted
copy where there is one, so on a healthy store that classification needs no downloads.

Run it right after a sync — a couple of unpersisted artifacts simply means a bot run fired
since the last one.

## Logs

- `scripts/research_sync/logs/sync_<YYYY-MM-DD>.log` — full `make sync_all` output, one file per run-day.
- `scripts/research_sync/logs/launchd.out.log` / `launchd.err.log` — launchd's own capture (job-start issues).
- `scripts/research_sync/logs/LAST_SYNC_FAILED` — present only while the most recent run
  failed; names the stage, timestamp, and logfile. Absence means the last run was green.

The `logs/` directory is created on first run.

## Uninstall

```bash
launchctl bootout gui/$(id -u)/com.metaculusbot.research-sync   # modern
# or, legacy: launchctl unload ~/Library/LaunchAgents/com.metaculusbot.research-sync.plist
rm ~/Library/LaunchAgents/com.metaculusbot.research-sync.plist
```
