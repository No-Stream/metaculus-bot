# Scheduled archive sync (launchd)

Keeps `backtests/research_archive/` (incl. `raw/`) AND `backtests/telemetry_archive/`
fresh by running `make sync_all` on a weekly schedule.

## Why this exists

Every bot run uploads its `research_outputs/` AND `run_logs/` to GitHub Actions with
`retention-days: 90` (see `.github/workflows/run_bot_on_{tournament,metaculus_cup,minibench}.yaml`
and `test_bot.yaml`). **After 90 days GitHub deletes the artifacts forever.** Two local
archives (both gitignored) are the only durable copies:

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
  `GAP_FILL_V2`, `GHOST_FORECAST`, `OPEN_BOUND_PILING`, `CREDIT_*`) harvested from the
  same artifacts; feeds parser-drift watch, gap-fill v2 diagnostics, credit burn-rate,
  and the ghost-vs-published scoring gate.

The pullers are manual (`make sync_research` / `make sync_telemetry`, both wrapped by
`make sync_all`), so without a scheduler the archives silently go stale and old data is
lost. This launchd job runs the pull weekly — well inside the 90-day window, with margin
for a missed week.

## What it does

`com.metaculusbot.research-sync.plist` invokes `run_sync.sh`, which:

1. `cd`s to the repo,
2. prepends the dirs holding `uv` and `gh` to `PATH` (launchd jobs get a minimal PATH),
3. runs `make sync_all`, which:
   - backfills from Metaculus comments (`backfill_research_from_comments.py`) FIRST — it
     hits Metaculus, not GHA, and writes `comments_backfill.jsonl` for the research build
     to load;
   - then runs the **single-pass** driver `scripts/sync_all.py`, which enumerates EVERY
     `research-*` AND `logs-*` artifact ONCE via the complete, paginated artifacts REST
     endpoint (no 1000-result `gh run list` cap, so nothing in the 90-day window is
     missed), downloads each unique artifact ONCE into a shared temp dir, and runs all
     three harvests over the same downloaded run dirs:
     - research JSONL (research-* dirs only) → rebuilds the research archive (download
       records + comment backfill, dedup, build);
     - run-log telemetry markers → merged into the telemetry archive (replace-by-run);
     - `raw_research_<run_id>.jsonl` raw-payload logs → one file per run under
       `backtests/research_archive/raw/` (replace-by-run);
4. appends a dated logfile under `scripts/research_sync/logs/`.

Running the driver in one pass avoids the three-pass waste of the old chain (each of the
three standalone `sync_*` targets re-enumerated every artifact and re-downloaded the
overlapping `research-*`/`logs-*` families into its own temp dir). The three standalone
targets (`sync_research` / `sync_telemetry` / `sync_raw_research`) still exist for a
single-archive refresh.

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
runs end-to-end without waiting for Sunday 03:00.

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
`backtests/research_archive/manifest.json` updates its `latest_timestamp` values. If
the archive looks stale, check the logfile — the download phase logs "Artifacts
endpoint returned N total, M research-* artifacts", how many downloaded, records added,
and (loudly) any EXPIRED artifact by name + created_at so a short pull or any data loss
is visible.

## Verifying maximal completeness

After a sync, prove the archive captured every live artifact:

```bash
uv run python -m scripts.research_sync.verify_completeness
```

It re-enumerates every live `research-*` artifact via the same paginated endpoint, loads
the rebuilt archive, and prints PASS / FAIL with the exact count of live artifacts vs.
those represented in the archive — flagging any genuine gap and any expired (lost-forever)
artifact. Read-only and free (GitHub API only).

## Logs

- `scripts/research_sync/logs/sync_<YYYY-MM-DD>.log` — full `make sync_all` output, one file per run-day.
- `scripts/research_sync/logs/launchd.out.log` / `launchd.err.log` — launchd's own capture (job-start issues).

The `logs/` directory is created on first run.

## Uninstall

```bash
launchctl bootout gui/$(id -u)/com.metaculusbot.research-sync   # modern
# or, legacy: launchctl unload ~/Library/LaunchAgents/com.metaculusbot.research-sync.plist
rm ~/Library/LaunchAgents/com.metaculusbot.research-sync.plist
```
