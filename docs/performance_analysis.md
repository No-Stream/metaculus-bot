# Residual / performance analysis: methodology and conventions

This is the reference half of the bot's residual-analysis work: the conventions that
make a number trustworthy, and the receipts behind each one. It covers the round pull
and the `--prior` rescoring diff, what the archives hold and which of their fields are
historically unreadable, era bucketing and the merge-to-main dating rule, the standing
exclusion cohorts, the scoring conventions (spot peer, and `spot_peer_delta` for
counterfactuals), the clip-threshold sweep, per-model forecast recovery, the PIT
convention for out-of-range resolutions, starved outer tails, and question-supply /
forfeit accounting.

Two sibling documents own the other halves, and this doc cross-links rather than
restates them. `docs/operations.md` § "Performance analysis and the width monitor" is
the **runbook** — the exact commands, the `--exclude-qids` mechanics, and what each
report prints. `scratch_docs_and_planning/residual_analysis_playbook.md` owns the
per-round **procedure** (Pre-pull → Recon → Pull → automated dims → per-question trace
dossiers with adversarial verification → Synthesize). Read this doc for *why a number
means what it means*; read those two for *what to type* and *what order to do it in*.

Every constant, function, module path and CLI flag here was verified against the code
on 2026-09-03. Measured figures are dated and carry their receipt path, because most of
them are a snapshot of one round's archive rather than a repo constant.

## The round pull, and why `--prior` is mandatory

`metaculus_bot/performance_analysis/` evaluates the live bot's calibration against
actual resolutions. Entry point:

```bash
uv run python -m metaculus_bot.performance_analysis --tournament <slug> --output <path>
```

The `--tournament` default (`DEFAULT_TOURNAMENT`, `performance_analysis/cli.py`) lags
the live season, so pass the current slug explicitly. The **pull is read-only and
free** — it hits only the Metaculus API (resolved questions plus the bot's own
comments, user id 275109, auth via `METACULUS_TOKEN`), makes no LLM or research calls
and publishes nothing, so it is **not subject to the repo's cost gate** (unlike
`make backtest_*` and live runs).

**Pass `--prior <previous round's dataset>` on every round pull.** Metaculus
re-resolves questions IN PLACE without moving any timestamp we store. It edited q44798
(Halo: Campaign Evolved Metascore) from 80 to 82 with `resolution_set_time` left at a
stamp that PRECEDES the pull which still read 80, flipping that record's spot peer from
+5.41 to −5.42 between two rounds while the earlier round's published tables went
silently stale.

`--prior` diffs the resolution VALUE and every `metaculus_scores` field against the
prior pull (`performance_analysis/rescore_diff.py`), tags each moved record
`platform_rescored=True` with `platform_rescored_fields` / `prior_resolution` /
`prior_metaculus_scores`, and emits one `PLATFORM_RESCORED` WARN per changed field plus
a printed summary. Three details:

- The tag reads as a **TERNARY**: None means "no prior record, never compared", and
  only False means "compared, nothing moved".
- The field is `platform_rescored_fields` and **not** `rescored_fields`, which a round
  script already spends on the bot-side scores `collector.rescore_records` healed.
  Two different facts.
- Records store `metadata.resolution_set_time`, useful for bounding when an edit
  happened but never as the detector.

## What `make sync_all` pulls

"Residual analysis" implies `make sync_all` first, always (read-only and free). It
pulls **everything** sync-shaped in one command, and that matters because GHA artifacts
expire at 90 days, so a single-source pull silently and permanently drops whatever it
did not fetch. Three archives:

- **The research archive** (`backtests/research_archive/latest/<qid>.json`) —
  per-question post-summarizer research. Precedence rules below.
- **The run-log telemetry archive** (`backtests/telemetry_archive/`) — the
  `EXTRACTION_RUNG` / `GAP_FILL_V2` / `GHOST_FORECAST` / `OPEN_BOUND_PILING` /
  `CREDIT_*` markers, plus the 2026-08-25 honesty set
  (`NUMERIC_DEGENERATE_DECLARATION`, `NUMERIC_AGGREGATE_GRID_MISMATCH`,
  `SPREAD_UNDEFINED`, `MARKET_RANKING_DEGRADED`, `CDF_MAXSTEP_CLIP`), the 2026-09-01
  bundle's set (`EXTREME_CALL`, `THIN_PUBLISH_FLOOR`, `RESOLUTION_SOURCE_FETCH`,
  `CREDIT_ROLE_SPEND`, `GEMINI_GROUNDING_DENSITY`, `GEMINI_UNSUPPORTED_ATTRIBUTION`,
  `FINANCIAL_NOISE_FLAG`, `MARKET_TIER_CAPPED`, `FRED_UNKNOWN_SERIES`), the 2026-09-02
  additions (`AGENTIC_FETCH_THROTTLED`, `MEMBER_FORECAST`), plus `GEMINI_USAGE` (the
  google-genai token and grounded-query accounting for all three Gemini surfaces, which bill
  outside OpenRouter and so appear in no `CREDIT_*` marker; the `role` field partitions them
  into `grounded_search`, gap-fill v2's `read_document` and the resolution-source ladder's
  `resolution_source`),
  `RESOLUTION_SOURCE_ESCALATION` (one line per escalated fetch rung, with what
  triggered it and what it cost), `AGENTIC_FETCH_LOCAL_DOC` (one line per gap-fill v2
  document read served from the host's own bytes instead of the paid reader — fires only
  where text was actually served, so its absence measures nothing) and
  `AGENTIC_URLCONTEXT_ROBOTS_SKIP` (one line per paid read skipped because the host's
  robots.txt disallows `Google-Extended`). `scripts/telemetry/markers.py` is the registry.
- **The raw research-provider payload archive**
  (`backtests/research_archive/raw/<run_id>.jsonl`, one file per run) — each provider's
  RAW return before formatting: AskNews article dicts per HOT/HISTORICAL phase,
  native-search and Gemini raw responses with grounding, prediction-market contracts,
  resolution-source per-URL fetches, gap-fill v1 search results. Written by
  `metaculus_bot.research.raw_log` when `RAW_RESEARCH_LOG_ENABLED` is set, so the raw
  evidence behind every forecast is auditable without depending on published comments.
  `financial_data` is deliberately not captured — its raw series live only inside
  `to_thread` workers.

`scripts/research_sync/` holds the launchd job, wired to `sync_all` for the same
reason. The telemetry archive also feeds `make score_ghosts` (the gap-fill v2
ghost-vs-published log-score gate, ~0 scoreable until v2-era questions resolve). Dated
round outputs land under `scratch/residual_<date>/` (gitignored).

Per-question tracing is a first-class phase, not optional — operator directive
2026-08-24: it is often the most valuable part of a residual round, and every miss
dossier gets an adversarial verification pass (the 2026-08-02 round revised 6 of 6
verified dossiers). The playbook supersedes the older `residual_rerun_workflow.js`.

## The research archive's `latest/` records come from three writers, and the difference matters

**GHA run artifacts** are the source of truth for every question since 2026-05-29 — the
exact research text the forecasters saw, plus `provider_results` / `gap_fill_v2` /
`asknews_raw` on schema-v2 records. Most older artifact records predate those fields,
so their absence means "old record", not "degraded run".

**Metaculus comments** are a lossy fallback for older questions only: middle-trimmed
(`RESEARCH_SECTION_CHAR_LIMIT` / `COMMENT_CHAR_LIMIT`), sections re-headed one level
deeper (so `^## ` presence probes read near-zero), and missing `resolution_criteria`.

A third writer, `scripts/backfill_research_from_logs.py`, parses run logs and keys its
`qid` on the **POST id** while every other writer keys on the **QUESTION id**. The two
share one integer space, so a single `by_qid/<N>.jsonl` can legitimately hold two
different questions' research.

Precedence for `latest/<qid>.json` is therefore `artifact` > `comment_backfill` >
`log_backfill` (`record_precedence_key` in `scripts/download_research.py`), then
newest-by-parsed-timestamp within a class. Log-backfill text is untrimmed but
post-keyed, and `latest/` is read question-id-first, so promoting it serves the wrong
question — measured: it made `latest/43592` return question 43591's research verbatim.

Reading rules that follow from this:

- Read the record's `source` field (`"artifact"` | `"comment_backfill"` |
  `"log_backfill"`, mirrored as `latest_source` in `manifest.json`). **Never infer the
  class from `run_id` alone** — log-backfill run_ids are plain GHA run ids,
  indistinguishable from artifacts.
- **Never pool the classes** for a presence, provider-mix, or length claim.
  `providers_used` on a comment record is reconstructed from trimmed text:
  `financial_data` reads 31 where the artifacts say 253.
- `latest_timestamp` is the winning record's timestamp, not a freshness signal.
- `scripts/research_sync/verify_completeness.py` gates the merge stage — a question
  holding an artifact record must be served by one in `latest/`.
- `make backtest_with_cache` logs the source split it replays, so pre- and
  post-2026-08-03 cached-backtest numbers are not comparable.

## Two treatment tags read as TERNARY, and two archived fields are historically unreadable

`research_tags.gfv2_loop_ran` is None on any record whose WRITER could not carry the
`gap_fill_v2` payload — only a schema-v2 `artifact` record can, and that writer omits
the key when the loop did not run. So on a carryable record its absence is a
*measurement*, and everywhere else it is *silence*. Reading it as a bool put 880
archived can't-carry records into the untreated arm against 77 measured ones, which
would have poisoned the v2 treated/untreated calibration split outright.

The companion `gfv2_confidence` grades a False `gfv2_present` the way
`anchor_confidence` grades the anchor read, with five values
(`performance_analysis/research_tags.py`):

| value | meaning |
|---|---|
| `header` | the section header itself was found |
| `payload_ran_no_section` | the loop ran and contributed nothing (a soft-fail) |
| `payload_confirms_absent` | a carryable record with neither header nor payload |
| `ambiguous_trimmed_no_payload` | a trimmed comment record — the section may have been trimmed away |
| `absent_no_payload` | an untrimmed record from a writer that cannot carry the payload |

Separately, `metadata.nr_forecasters` (the Metaculus CROWD size) reads **0 in all 2196
records pulled before 2026-08-25**, because the collector read it off the question
dict, where it does not exist — it lives on the POST. Nothing rewrites the archive, so
treat a 0 on an older record as UNKNOWN, never as an empty crowd. Fresh pulls carry
real counts (typically 100-250 on tournament questions) or None when the post omits the
field, and `audit.py` renders None as `n/a`.

## Era bucketing is mandatory for calibration claims

Any calibration, aggregation, or bias claim computed on pooled resolved data is suspect
until split by config/roster era (proxy: `source_tournament`, or
`bot_comment_created_at` versus config-flip dates).

Three separate conclusions have flipped under era-bucketing:

- the numeric "too wide" verdict (2026-06, computed on pre-flip-only data);
- the "current pipeline too narrow" verdict (2026-07, softened and then reversed as
  post-flip n grew);
- the YES-side overconfidence finding (2026-07-08), which turned out to be
  spring-2026-era-local — fall was well-calibrated, and a pooled fit would have
  degraded fall out-of-sample.

Bucket by **major** config/roster changes: model swaps, aggregation changes, widening
flips, research-stage changes. **NOT** by every git hash — a small prompt tweak does not
start a new era; a forecaster-roster or pipeline-behavior change does. The judgment
call is "would this change plausibly shift the forecast distribution?" If unsure, run
the analysis both ways.

Read the merge-date rule below before fixing any boundary. In particular the whole
july15 bundle — everything authored 2026-07-15 through 07-20 — is a SINGLE boundary at
**2026-07-21T17:07:37Z (`b4e9df0`)**. It carried gap-fill v2 on, the native-search and
crux-analyzer sol→terra swaps, both same-day forecaster-roster changes (the fable-5 →
opus-4.7 forecaster plus opus-4.8 stacker swap, then the drop from 6 to the 3-member
latest-per-vendor triple gpt-5.6-sol / opus-4.8 / gemini-3.1-pro-preview), the sol
forecaster's xhigh→high effort drop, and `MIN_FORECASTERS_TO_PUBLISH` 3→1. None of them
is separately datable, and no shift across the boundary can be attributed to any one of
them.

**Fitted calibration layers (shrinks, clamps, haircuts) require a decisive
out-of-sample era test before shipping** — fit on eras 1..k-1, must improve era k — else
they are drift bombs.

## Era boundaries are merge-to-main timestamps, never authoring dates

Prod runs from `main`, so a config change is live only from the moment its merge commit
lands there. Get the boundary from the first-parent log of `main`
(`git log --first-parent --format='%H %cI %s' main`) and then read the merge commit's
committer date (`TZ=UTC git log -1 --date=iso-local --format='%h %cd' <merge-sha>`).

A branch can sit for days. Two dated examples:

- the july15 bundle was authored 2026-07-15..07-20 and landed
  **2026-07-21T17:07:37Z (`b4e9df0`)**;
- the `base_rate_anchor` / `criteria_clauses` telemetry was authored 2026-07-08
  (`30bca2f`) and landed **2026-07-11T16:37:17Z (`642b027`)**.

Keying on the authoring date files every run in the gap under the wrong config, and
this has already cost real analysis twice.

First, it manufactured a phantom one-record `ts_anchor` era in `width_monitor.py` out of
a question whose own comment names the retired six-model roster (`grok-4.5`, `gpt-5.5`,
`opus-4.6`) — all dropped by the same merge that landed the anchor, so the combination
is impossible post-merge. That phantom is gone. **Today the `ts_anchor` row is absent
from the width-monitor table for a different and correct reason:** empty eras are
omitted, and no post-july15-bundle numeric has resolved yet. The two causes are
sequential, not competing — a phantom row that was wrong, then a legitimately empty row
that is omitted.

Second, it made the guard-telemetry presence check read an "intermittent emission" rate
instead of a clean 100% — **58%** on the receipt's per-slot cohort
(`scratch/residual_2026-08-02/dim_ghosts-and-guards.md:118`) and 78.9% on a per-comment
recount. The spread between those two figures is itself a second reason not to lean on
the authoring-date number. The mechanism is exact regardless of cohort: all 8 binary
comments in the authored-but-unmerged gap window carry no anchor, and the first one that
does is 2026-07-12, after the merge — exactly as "prod runs from main" predicts.

**Corollary: several authoring dates often collapse into one boundary.** Nothing on
`main` changed between the 2026-07-12 merge (`f084bf7`) and `b4e9df0` — a
`git diff --stat f084bf7 b4e9df0^1 -- metaculus_bot/ .github/workflows/` comes back
empty — so 2026-07-15 / 07-17 / 07-18 / 07-20 are **one** era boundary, not four. That
merge landed the TS anchor, gap-fill v2, the six-models-to-triple roster drop,
`MIN_FORECASTERS_TO_PUBLISH` 3→1 and the sol→terra role swaps together, which also means
no width or score shift across it can be attributed to any one of them. Treating those
dates as separable slices a period of constant prod config and reads noise as a config
effect. When a doc gives an authoring date, say so, and put the landing date next to it.

The boundary constants live in `performance_analysis/analysis.py`:
`WIDENING_FLIP_MERGED_AT` (`0e85e1b`, 2026-05-18T17:21:19Z), `FT_0292_MERGED_AT`
(`325b1b0`, 2026-07-24T19:16:26Z) and `B4E9DF0_MERGED_AT` (2026-07-21T17:07:37Z). See
"Vocabulary that collides" at the end for the width monitor's aliases of the same
instants.

## The known-pipeline-bug cohort

**One canonical home: `KNOWN_BUG_QIDS` in `performance_analysis/cohorts.py`.** These are
questions whose published forecast came out of a since-fixed pipeline defect rather than
judgment, so pooling them into a calibration or miss-ranking row measures the retired
bug. It currently holds five ids:

- **43746 / 43747** — the pre-2026-07-07 open-bound arithmetic bug.
- **43913**, added 2026-08-25 — the pre-`9f1175c` discrete max-step cap. All six
  forecasters stated 79.5-83% on the outcome that resolved; the published CDF carried
  20.00% with its first bin pinned at exactly 0.200000 on an 11-point grid. Receipts:
  `scratch/residual_2026-08-24/dossiers/43913_dossier.md`.
- **43147 / 41798**, added 2026-09-01 — the same defect family on pre_flip discrete
  records: published mass at the resolving value pinned at exactly 0.200000 by the
  retired flat cap, while even the least concentrated member wanted 0.525 / 0.635 there.
  Peers −34.75 / −35.50. Identified by the shipped `max_step_clamp_screen`. Receipts:
  `scratch/residual_2026-08-31/dim_numeric-width.md`.

Import the constant instead of re-hardcoding the ids; every private copy in a round's
analysis scripts has drifted from it at least once. Nothing excludes the cohort by
default — a caller passes it explicitly (`--exclude-qids known_bug`) and the excluded
count is rendered per row, so an exclusion is a visible choice rather than a silent
filter. See `docs/operations.md` for the `--exclude-qids` mechanics.

## The degraded-run cohort (dry-donated-key incident, 2026-07-26 → 07-28)

The pre-fix dry-key window published eleven triple-era questions on a thinned ensemble.
Exclude them from headline aggregates and report them separately. Since 2026-08-31 they
live beside `KNOWN_BUG_QIDS` in `performance_analysis/cohorts.py`:

- `DEGRADED_RUN_QIDS` — full 1-of-3 publishes, gemini only (the personal-key-pinned
  slot): question ids **44870-44877**.
- `PARTIAL_DEGRADED_QIDS` — partial 2-of-3: **44841, 44856, 44912**.

Both are reachable as `from metaculus_bot.performance_analysis import DEGRADED_RUN_QIDS`
and both are wired into `--exclude-qids` under the shorthands `degraded_run` /
`partial_degraded`. Import them; three separate rounds hardcoded private copies before
the constants existed.

**These are QUESTION ids.** The same eight questions carry post ids **44721-44728**, and
minibench POST ids 44873-44877 land inside the question-id range, so a join that matches
"either id" admits five unrelated questions. Translate through
`performance_analysis/id_mapping`, never raw integers.

On the research side, 44841 / 44856 are degraded identically to the full cohort (native
search errored, both gap-fill passes dead), so a research-conditioned cut must exclude
both sets together even though the forecaster-count tagging separates them.

The first two resolved in 2026-08, both favorably — 44870 spot peer **+20.11**
(published on gemini alone; coverage-scaled peer +14.38), 44841 spot peer **+24.52**
(peer +21.54). That is a two-question favorable draw, not evidence that degraded
publishes are fine. Receipts: `scratch/residual_2026-08-24/degraded_cohort.json`.

## The tournament ranks on SPOT PEER

**Never rank or aggregate on the coverage-scaled `peer_score`.** Verified against the
live API on 2026-08-31: the project carries `score_type=spot_peer_tournament`, every
question's `default_score_type` is `spot_peer`, and `spot_scoring_time` equals
`actual_close_time` on all 158 posts pulled.

`peer_score` on the same record is `spot_peer_score × coverage`. Measured on the
2026-08-31 round's 30 new records, that identity reproduces the platform's own
`peer_score` to a **median residual of 0.69 points (max 13.05)**, and the residual is
crowd movement in the 1.5-3h window between our submit and the close rather than
anything the bot did. Those are that round's numbers, not a repo constant — re-derive
with `scratch/residual_2026-08-31/dossiers/44798_peer_vs_spot.py`.

Because the bot submits exactly once and never revises (forecast history length 1 on
157 of 158), its coverage is mostly a function of how early it submitted. So coverage
scaling FLATTERS misses and dulls hits: q44872 scored peer −15.0 against spot peer
−38.8.

`performance_analysis/platform_scores.py` is the one place that encodes the preference:

- use `spot_peer_score()` / `ranking_score()` rather than indexing `metaculus_scores`;
- report peer beside it as a labelled secondary;
- never sort a mixed set on whichever field happens to be present — `RankingScore.tier`
  keeps spot-scored and peer-only records in separate sort tiers.

Bot-side scores are a different quantity entirely and are unaffected: Brier and log
score in `performance_analysis/scoring.py`, `expected_baseline_score` in
`scoring_patches.py` (a log score against the community prediction, not a platform peer
score), and `backtest.py`'s own scoring, which never reads platform peer at all.

## Price a counterfactual with `spot_peer_delta`, never by hand

The halving gets applied twice or not at all. Metaculus computes spot peer as
`100·(N/(N−1))·ln(p/gmp)` and then HALVES it for a continuous question (numeric,
discrete, date). The crowd's geometric mean includes us, so a counterfactual that moves
only OUR mass on the resolving outcome is worth `100·ln(new/old)`, halved for
continuous, with no crowd term left. Read from Metaculus's `scoring/score_math.py` on
2026-09-02; fetched copy at
`scratch/residual_2026-09-01/dossiers/44798_verify_metaculus_score_math.py`.

Both conversions have already been got wrong, and both mistakes INFLATE the figure:

- `numeric_log_score` ALREADY carries the halving (it returns `50·ln(...)`), so a
  difference of two of its values is already in spot-peer points. The 2026-08-31 q45065
  cap-smear replay doubled exactly that and priced the near-miss counterfactual at up to
  +404 when the truth is +202.
- Thirteen 2026-09-01 dossier scripts quoted `binary_log_score` deltas (log base 2) as
  peer points, which OVER-states each one by 1/ln2 ≈ 1.44. Correcting an archived binary
  figure therefore means **multiplying** it by ln 2 ≈ 0.693, not dividing.

`spot_peer_delta` (`metaculus_bot/scoring_common.py`, re-exported from
`performance_analysis.scoring`) is the one implementation. It raises on an unrecognized
question type rather than silently taking the un-halved branch, and
`tests/test_peer_delta_convention.py` pins both conversions. Corrected q45065 figures
and the full per-script sweep: `scratch/residual_2026-09-01/DOSSIER_SYNTHESIS.md` §7.2.

## The clip-threshold sweep

**The clip floors are priced by a standing sweep, and a looser clip is censored, never
measured.** Entry point `metaculus_bot/performance_analysis/clip_threshold.py`:

```bash
uv run python -m metaculus_bot.performance_analysis.clip_threshold --cached <dataset> \
  --exclude-qids known_bug,degraded_run,partial_degraded
```

It reprices every resolved binary and MC publish under a grid of candidate floors `c`
(binary 0.005 to 0.10, MC 0.005 to 0.10 — module constants `BINARY_FLOOR_GRID` /
`MC_FLOOR_GRID` in `clip_threshold_sweep.py`) and reports each in spot-peer points via
`spot_peer_delta`, floor-only / ceiling-only / symmetric.

**Windows.** The NESTED windows are `all` / `last_300` / `last_200` / `last_100` (MC
adds `last_50`, because its whole archive is under 100) / `last_90d` /
`current_clamp_regime` / `triple_era`. The DISJOINT config-era slices are
`era_pre_flip` / `era_post_flip`, with `triple_era` the third. The distinction is
load-bearing: nested windows re-count one set of records at different sizes and so
cannot disagree, while the era slices partition the dated records, so agreement between
them is real evidence.

**Censoring.** A candidate at least as tight as the clamp that was in force is exact
from the published value. A candidate LOOSER than it cannot be priced on a record that
sat at that clamp, because the per-member clamp erased the raw value. Those records are
counted and bounded, never estimated:

- `cen` keys on the published value.
- `cen_m` keys on a clamped MEMBER in a median position, and that is the rule that
  actually bounds what could have moved: an even roster averages two middle members, so
  a 0.02 member publishes 0.025.
- Bounds are labelled `at_floor` (nothing moved) and `at_c` (every censored value was at
  or below `c`), plus the identified bracket.

**The in-force clamp is looked up per record** from `bot_comment_created_at` against
`WIDENING_FLIP_MERGED_AT` (binary, `0e85e1b`, 2026-05-18T17:21:19Z) and
`FT_0292_MERGED_AT` (MC, `325b1b0`, 2026-07-24T19:16:26Z) — both merge-to-main committer
dates, living beside `B4E9DF0_MERGED_AT` in `analysis.py`, and `width_monitor.WIDENING_FLIP`
aliases the first.

**Each window carries an insurance view**: the break-even clipped-side rate, a Jeffreys
interval on the observed rate, the expected loss under the bot's OWN prices (spot peer is
proper, so a clip costs a calibrated forecaster that much regardless), and the best case.

**Selection discipline.** The report prints an out-of-bag value of the fitted argmax,
because the argmax is a choice over the grid and its own row's CI ignores that
selection. The out-of-sample rule is that a floor fitted on the records older than a
window ships only if it carries into the window, and a fit that moves nothing in its own
complement is flagged `moves nothing` — its carry of 0 is vacuous rather than a pass. A
row that moves no record renders `identity` rather than a CI.

**Result on 2026-09-02** (`scratch/residual_2026-09-01/clip_threshold/dim_clip-threshold.md`):

- The live clamp has bound NO binary publish since it went live — 70 strict post-flip
  binaries span 0.034 to 0.925.
- Raising the binary floor loses in every window and era. At c = 0.05 the pooled figure
  is **−217.48 over 447** records, of which −214.76 is the retired pre-flip regime and
  −2.72 the 70 live-regime records. 0 of 81 moved records resolved on the clipped side,
  against a break-even rate of 3.08%, and the properness cost alone is −91.19.
- An MC floor is a tax on every question: **−3.53 per question** at c = 0.05,
  era-stable.
- Loosening is bounded at **+10.91 over 447 binaries** and **+0.50 over 97 MC**, all
  pre-flip.
- Every out-of-sample fit is the do-nothing candidate.
- The only pro-tightening row in either cohort is the single-survivor degraded publish
  q44874, whose shape the thin publish floor prices at **+51.08 over the 4 genuine k=1
  publishes with zero cost to the other three**.

## Recovering per-model forecasts

The bot's published Metaculus comments are the durable per-model record: on non-stacked
questions the summary carries one `*Forecaster N (model)*: value` bullet per ensemble
member (post-clamp values). `performance_analysis/collector.py`
`build_performance_dataset` already parses these into `per_model_forecasts` /
`per_model_mc_option_probs` / `per_model_numeric_percentiles` /
`per_base_model_forecasts`. Consumers import the bullet regex and the `Model:`-prefix
attribution from `performance_analysis/parsing.py`, which re-exports the mechanics from
`performance_analysis/comment_sections.py`.

Gotchas:

- Comments longer than `COMMENT_CHAR_LIMIT` are middle-trimmed (`comment/trimming.py`)
  — summary bullets survive, but rationale-body percentile detail may not.
- Stacked-era questions publish only the stacker's aggregate bullet; base values are
  recoverable only from self-declared rationale text (the `## Base Model Reasoning`
  sub-blocks).
- Soft-deadline drops mean some questions have fewer than N bullets.
- Old-era (May-June 2026) blocks carry retired tier-2 fields (`mixture_components`,
  `tails`, `distribution_family_hint`) that the strict `parse_structured_block` schemas
  reject wholesale. A tolerant raw-JSON fallback rung recovers the declared values from
  block-only rationales that would otherwise vanish — strict block → prose regex →
  tolerant salvage, added 2026-07-15, imported from `parsing.py` and implemented in
  `performance_analysis/declared_value_recovery.py`. That rung explains the false
  "gemini missed 5/45" screening artifact. The other historical offender, an edge-value
  `concentration: 0.0`, no longer needs the salvage: since 2026-09-02 the strict MC
  schema reads an unusable `concentration` / `other_mass` as absent instead of rejecting
  the block, because both fields were retired from the prompt and a dormant field must
  never cost a ballot.
- Roster drift makes era-conditioning mandatory — see the era-bucketing section above.

### Per-model cuts run on a filtered cohort; aggregates don't

When no `Model:` line identifies a bullet, the parser keys it by position instead
(`anonymous_model_key` → `Forecaster N`), and on a stacker-fired question that
positional bucket holds the stacker's aggregate. Pooling it across questions therefore
produces a stacker-vs-base-model mixture posing as one model — measured: 50 such
forecasts in the 2026-04 data.

Every per-model cut in `analysis.py` (`per_model_binary_scores`,
`stacking_effectiveness`, `disagreement_predicts_error`) therefore goes through
`per_model_cohort`, which drops anonymous keys and drops records whose stacker is
*confirmed* fired, logging both counts at INFO under `PER_MODEL_COHORT`. Only the
confirmed verdict excludes: `likely_stacker` is a high-spread-plus-large-delta heuristic
that also matches an ordinary MEAN-era aggregate, so honoring it would delete the
high-disagreement questions those cuts exist to measure.

The audit's per-question rankings and the synthesis tally inherit the same guards via
`ranking_cohort.per_model_ranking_cohort`, which calls `per_model_cohort` rather than
restating it, so the rule cannot drift between the aggregate cuts and the dossiers.
Numeric rankings additionally drop declared percentile curves under
`MIN_SCOREABLE_ANCHORS` (9) distinct anchors — unless EVERY member on the record is
equally sparse, which is sparse-ERA output rather than a partial recovery and still
compares equals. Otherwise a sparse recovery gets PCHIP'd into a full CDF and
log-scored beside 11-anchor siblings, worth ~96 points either direction.

`max_step_clamp_screen` gates on that same shared floor (in its `_member_bin_masses`
helper), because the screen's verdict turns on the MINIMUM member bin mass, so one
sparse recovery can decide it. The floor lives in `parsing.py` precisely so those two
consumers cannot drift; `stacker_detection.py` and `audit.py` read it too.
**`declared_percentile_pit` deliberately does NOT gate on it** — it only linearly
interpolates the declared pairs in percentile space for a single quantile, where a
3-anchor curve is coarse but not a fabricated distribution, and gating there would
delete the uniformly-sparse-era records (fall-2025 comments declare 8-percentile sets)
whose PITs are valid. It does still exclude anonymous keys.

Aggregate and overall calibration paths are deliberately untouched by all of this — they
still count every record.

### The attribution parsers are guarded on two cohorts, and only one runs in CI

`tests/data/performance_comments_mini.jsonl` is a checked-in miniature: one real comment
per distinct SHAPE (attributable vs not, trimmed vs intact, with vs without the
`### Research Summary` boundary marker, named vs anonymized, all four question types),
redacted down to the structural skeleton the parsers key on. It is the deterministic CI
floor — `TestMiniFixtureAttribution` (`tests/test_performance_analysis_attribution.py`)
and `TestAgainstCheckedInMiniComments` (`tests/test_comment_trimming.py`) are not
skip-gated, so a parse or trim regression reddens every PR.

The broad sweep over `scratch/performance_data.json` (283 records, every era) still runs
locally and catches shapes the miniature has not been taught, but that file is gitignored
and rewritten by each collector run, so it can never be the only guard — a parse
regression hid behind exactly that gap until 2026-07-27.

Regenerate the miniature with `uv run python scripts/derive_mini_comment_fixture.py`
when a pull introduces a genuinely new shape. The derivation only admits a record whose
miniature parses IDENTICALLY to its full-size source, and the shape-coverage test fails
loudly if the set ever narrows.

## An out-of-range resolution gives a SET-valued PIT reading, not a forced 1.0 / 0.0

Metaculus reports a resolution past the displayed range as the bare string
`above_upper_bound` / `below_lower_bound`, so the resolution VALUE is unknown and
`F(resolution)` is pinned only to an interval: `[cdf[-1], 1]` above the ceiling,
`[0, cdf[0]]` below the floor. The old forced convention counted q44842 as a coverage
miss even though that forecast deliberately put 13% of its mass above the displayed
ceiling and won spot peer +24.4.

The convention has exactly one home — `PitReading` and `out_of_range_pit_reading` in
`performance_analysis/analysis.py` — and the two conventions riding on it differ
deliberately. **Coverage** counts an interval as covered when it INTERSECTS the band, a
miss only when the whole interval lies outside. **Point** statistics (`pit_std`,
`mean_pit`, the histogram) EXCLUDE intervals and disclose the excluded count as
`n_oob_interval`, because imputing a midpoint would manufacture a reading nobody
measured. `docs/operations.md` describes what the width monitor prints, where the same
quantity appears as the `set-valued (pt n)` column.

When a bound is closed, or open with no out-of-range mass, the interval's endpoints
coincide and the reading degenerates to exactly the old value, so nothing moves on those
records. The measured effect on the archive is triple-era cov80 0.727 → 0.818, which at
n=11 is one record (q44842) going from miss to covered rather than a distributional
shift.

Two API notes for any round script:

- `compute_pit_details` is now `compute_pit_reading` (`width_monitor.py`), renamed with
  no compatibility wrapper on purpose, so a stale round script fails with an ImportError
  rather than silently getting the retired convention.
- `EraWidthMetrics.pit_std` / `mean_pit` are `float | None` behind a
  `point_metrics_underpowered` gate and render as `n/a`, so JSON consumers must expect
  nulls.

## A starved outer tail is a different defect from the max-step smear, and it is systematic

On an open bound the declared outer tail can end up routed past the displayed range
entirely, leaving every in-range bin above the members' declared p99 pinned at the
platform's per-bin minimum step. Every resolution in that band then earns the same floor
score (~−219 at any grid size), so it is a CLIFF at a fixed location rather than a band
of the wrong width — which is why widening does not fix it, and why shipping the
detector is not in tension with the standing `k_tail` hold.

`scan_outer_tails` (`performance_analysis/outer_tail.py`, printed by the width monitor's
CLI) triggers on the band's MEAN per-bin mass expressed as a multiple of the platform
minimum step (`STARVED_OUTER_TAIL_FLOOR_MULTIPLE = 2.0`). It deliberately does **not**
trigger on the plan's proposed `tail_mass = 1 − F(p99)`, which carries no signal at all:
with the canonical anchors that quantity is ≈0.01 on every record by construction, so
q45218 reads 0.0142 and would not fire at any threshold that does not fire on nearly
everything. The multiple is scale-free, which is what tells a 2-bin band holding 0.003
(real density, harmless) from a 27-bin band holding 0.004 (the cliff).

```bash
python -m metaculus_bot.performance_analysis.width_monitor --cached <path>
```

prints a per-question "Starved outer tails" section after the era table;
`--output-starved-json` writes every scanned side, and `--exclude-qids` cohorts apply.
`docs/operations.md` describes the per-row fields and the member census.

**The first result is itself the finding**: it fires on 68 of the 417 measurable
open-bound sides across 49 questions, 19 of them starved on both sides, with 44 sides
sitting essentially exactly at the pipeline's own applied floor. So read a fire as "this
question carries a cliff", not "something broke here".

There is deliberately NO publish-time `STARVED_OUTER_TAIL` WARN. The reason, and what a
no-plumbing alternative would have to measure instead, are in the code comment above
`STARVED_OUTER_TAIL_FLOOR_MULTIPLE` and tracked in `FUTURE.md`.

## Question-supply counts need post status `closed`

`scripts/supply_probe.py` (`make supply_probe`, read-only and free — the Metaculus posts
list plus post detail) is the tracked replacement for two rounds' worth of scratch probes
that each queried only `statuses=open` and `statuses=resolved`, and so missed the 178
summer-tournament posts sitting at `closed` (closed to forecasting, not yet resolved),
26 of them the frozen-triple checkpoint cohort.

It also reports the backlog of unresolved questions past their own
`scheduled_resolve_time`, which is what separates "Metaculus is late resolving" from
"our pull is missing questions". Resolution is read per QUESTION, not per post, since a
group post's members resolve on their own schedules.

Since 2026-09-02 it also sweeps **FORFEITS** — every question on a `closed` or
`resolved` post that the bot never forecast at all, with its open/close window. A
forfeited question never enters the performance dataset and so is invisible to any sweep
that starts from questions the bot intook. The 2026-09-01 round found the triple era had
lost SIX questions to delivery where the prior sweep saw one: q44801 to a cron gap,
q45085 to a late submit, q45093 / q45374 / q45375 to cancelled runs, and q45216 to a
retroactive close.

The sweep costs one extra read-only detail GET per closed/resolved post whose list page
did not already carry `my_forecasts` (`--no-forfeits` opts out), and a question whose
state stays unreadable is reported as `unknown` rather than filed as a forfeit. Default
slugs come from the repo's constants; see `docs/operations.md` "Season-start checklist".

## Vocabulary that collides

Four pairs of names describe one thing, or two different things under similar names.
Keeping them straight is the same discipline the era rule asks for.

**Era-boundary constants have two vocabularies for one instant.** The width monitor
names its boundaries `WIDENING_FLIP` and `TS_ANCHOR_ENABLE`; both are aliases defined in
`width_monitor.py` of `analysis.py`'s `WIDENING_FLIP_MERGED_AT` and
`B4E9DF0_MERGED_AT`. So the width monitor's `TS_ANCHOR_ENABLE` and the clip sweep's
`B4E9DF0_MERGED_AT` are the same 2026-07-21T17:07:37Z instant under two names. Prefer
the `*_MERGED_AT` names in new code, and never introduce a third spelling.

**Two anchor-count thresholds mean different things.** `MIN_SCOREABLE_ANCHORS` (9,
`parsing.py`) is the distinct-anchor floor for per-model RANKING and the max-step clamp
screen — below it a curve gets PCHIP-rebuilt into a full CDF and log-scored, worth ~96
points either direction. The outer-tail scan's own rule is separate and far lower: it
drops a member curve carrying fewer than two distinct percentile labels, because a
single recovered pair interpolates to a constant PIT at every resolution. They are not
in conflict; they gate different computations at different costs.

**`n_oob_interval` and the rendered column `set-valued (pt n)` are the same quantity** —
the count of out-of-range-interval PIT readings excluded from the point statistics. The
field name is what a script reads; the column label is what the width monitor prints.

**The `ts_anchor` era's absence has two true causes, in sequence.** A phantom
one-record `ts_anchor` era once existed because the boundary was keyed on an authoring
date; its tell was that the record's own comment named the retired `grok-4.5` /
`gpt-5.5` / `opus-4.6` roster, dropped by the same merge that landed the anchor. That is
fixed. The row is absent *today* because empty eras are omitted and no post-july15
numeric has resolved. Neither statement is stale — read them in that order.
