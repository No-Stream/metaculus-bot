# Residual analysis playbook (standing methodology)

Codified 2026-08-24 from the practice that evolved across `scratch/residual_2026-05-29/` →
`06-15` → `06-26` → `07-08` → `07-18` → `08-02`. This supersedes
`residual_rerun_workflow.js` (the 2026-06 workflow script, kept for history) as the
methodology reference AGENTS.md points at. Everything here is free and offline: Metaculus
API reads, GitHub artifact reads, local files. Zero paid provider calls, zero publishing,
no commits from analysis agents.

Outputs land in a fresh dated dir `scratch/residual_<date>/` (gitignored — grep prior
rounds with `rg --no-ignore`). Never modify a prior round's dir; it is the diff baseline.

## Phase 0 — Pre-pull (always)

`make sync_all` before anything reads the archives. GHA artifacts expire at 90 days; a
stale archive silently drops recent questions and receipts. Non-negotiable first step.

## Phase 1 — Recon

1. **Era map.** `git log --first-parent --format='%H %cI %s' main` since the last round's
   boundary. For each merge, diff `metaculus_bot/ .github/workflows/` and classify:
   forecast-distribution-shifting (new sub-era) vs neutral. **Era boundaries are
   merge-to-main committer timestamps, never authoring dates** — this has manufactured
   phantom eras twice (AGENTS.md era-bucketing rule). Verify the roster actually stayed
   frozen (`git log -p -- metaculus_bot/llm_configs.py`).
2. **Follow-up ledger.** Read the prior round's SYNTHESIS (watch-item ledger + recommended
   next steps) and FUTURE.md. For each item: shipped since (commit evidence)? date/n-gate
   due now? still open? The ledger tells the dimension and dossier agents what to check.

## Phase 2 — Pull

- Fresh `resilient_pull.py` on the active tournament(s); probe for new slugs via the
  workflow yamls + a cheap API probe. Reuse verified-complete baselines from prior rounds
  (each round's README records which pulls are safe to reuse and why).
- Era-tag every record on `bot_comment_created_at` (submission time) against the era map.
  Tag exclusion cohorts by IMPORTING the constants, never by retyping the ids:
  `KNOWN_BUG_QIDS`, `DEGRADED_RUN_QIDS` (dry-key 1-of-3), `PARTIAL_DEGRADED_QIDS` (2-of-3)
  from `metaculus_bot.performance_analysis` — the same sets the `known_bug` /
  `degraded_run` / `partial_degraded` `--exclude-qids` shorthands expand to. Three rounds
  hardcoded private copies of the degraded ids before the constants existed, and the
  known-bug copies have drifted from the canonical set at least once. Excluded from
  headline aggregates, reported separately — never silently dropped.
- Diff vs prior round (`new_since_prior.json`): the new cohort is what the round is about.
- Spot-check re-pull stability (a handful of prior spot-peer scores must reproduce).

## Phase 3 — Automated dimensions (era-bucketed, parallel)

The standing set, each reconciling explicitly with the prior round's same-named dim doc
(agree / disagree / refine + why):

- numeric width + PIT (width_monitor eras, cov80/cov50/cov@10, PIT std, band-miss lo/hi)
- binary + MC calibration (Beta-Binomial CIs per bucket, slope/intercept by era)
- per-model (through `per_model_cohort`; **ex-floor means + declared-band miss rates** —
  the −219.97 log-score floor censoring once flipped a worst-member conclusion)
- aggregation faithfulness + stacker state confirmation
- bot health (extraction rungs, gap-fill v2 telemetry, credit, close-margin latency,
  provider diagnostics)
- market snapshot informativeness
- ghosts + guards (score_ghosts, structured-JSON presence)
- cross-tournament / category
- consensus-miss mode statistics (see Phase 4 rules)

## Phase 4 — Per-question tracing (the centerpiece; often the highest-value output)

**Mandatory every round, resourced at least as generously as the automated dims.** The
operator's standing directive (2026-08-24): human-style question-by-question tracing is
frequently the most valuable part of a residual round — treat it as a first-class phase,
not an optional garnish.

1. **Rank.** Rolling-window miss ranking (SPOT peer score, all types) → `MISS_RANKING.md`.
   `audit.select_cohort` already ranks on spot and logs `PLATFORM_RANKING_SOURCE`; a WARN
   there means some record fell back to coverage-scaled peer.
2. **Select.** All material new misses, plus **good-call controls** (3–5). Controls are
   load-bearing, not decorative: the 2026-08-02 publish-vs-own-anchor metric only died
   because a hit-side baseline exposed it as outcome-tracking (63% miss vs 33% hit — and
   the worse-than-own-LR version inverted). Any miss-side process metric quoted without
   its hit-side rate is presumptively survivorship bias.
3. **Trace.** One dossier agent per question, full pipeline walk: research bundle
   (`research_archive/latest/` + the RAW per-provider payloads in
   `research_archive/raw/`), per-model forecasts and rationales, aggregation path,
   telemetry markers (`marker_records_for_question` via
   `performance_analysis.id_mapping` — post-id vs question-id spaces differ per marker
   and hand-rolled joins produce false matches), resolution mechanics, and score
   counterfactuals. Classify: research-miss / consensus-judgment-miss / weighting failure
   (corrective was computed or in the bundle and published past) / aggregation loss /
   pipeline bug / defensible-loss. Name the cheapest change that would have saved it.
   **A dossier's per-model ranking table can legitimately be EMPTY or short, with the
   reason printed beneath it — read the caveat lines rather than treating a missing table
   as a parse failure.** `ranking_cohort.per_model_ranking_cohort` drops the whole record
   when the stacker fired (every per-model slot then holds the stacker's aggregate), drops
   anonymous `Forecaster N` keys (positional buckets, not models — `Forecaster 1` was the
   third most frequent "best model" in a previous synthesis tally), and on numeric drops
   declared curves under 9 distinct anchors unless every member on the record is equally
   sparse. A row labelled sparse-era is comparable WITHIN its question and not across
   questions: don't quote its absolute log score beside a denser question's. Any round
   script that tallies `ranked[0]["model"]` / `ranked[-1]["model"]` inherits these
   exclusions automatically, so read a shrinking tally as the fix landing rather than as
   lost data.
4. **Adversarially verify every miss dossier.** In 2026-08-02, 6 of 6 verified dossiers
   came back REVISED — classifications held but load-bearing numbers and headline
   counterfactuals were corrected or refuted. The verifier re-derives key numbers from
   primary sources, checks counterfactual dates (was the proposed source even published
   before the run?), and recomputes score math. Unverified dossiers are quotable only as
   "plausible but unaudited".
5. **Cross-dossier statistics.** Consensus vs dissent (a material dissenter must be
   right-sided; magnitude above the ensemble noise floor), failure-class shares, and the
   dossier↔breadth-statistics reconciliation. Declared-percentile PIT is censored at the
   outermost label — unanimous censoring is itself maximal consensus signal.

## Phase 5 — Synthesis

Headline first (the era scoreboard and whatever the round's load-bearing question was),
then: failure-mode verdict, watch-item ledger with per-item verdicts
(fired / clear / still-unmeasurable + what changes), bot-health verdict, recommended next
steps **split free-offline vs paid-operator-decides** (nothing paid runs during a residual
round — surface command + rough cost), curiosities. Caveat n honestly; state effective n
after clustering (same-day resolutions share a world state).

## Standing gotchas (each has burned a round at least once)

- **The tournament ranks on SPOT PEER; `peer_score` is `spot_peer_score × coverage`.**
  Never rank, aggregate, or headline on the coverage-scaled figure — the bot submits once
  and never revises, so its coverage is mostly submission timing, and the scaling flatters
  misses (q44872: peer −15.0 vs spot peer −38.8). Read platform scores through
  `performance_analysis/platform_scores.py`, which prefers spot and keeps peer-only records
  in their own sort tier; report peer beside spot as a labelled secondary only.
- Merge dates, not authoring dates, for every era boundary.
- `performance_analysis.id_mapping` for any marker↔record join; never "match either id".
- Never pool research-archive record classes (`artifact` / `comment_backfill` /
  `log_backfill`) for presence, provider-mix, or length claims; comment-backfill re-heads
  sections (`##`→`###`) and trimming eats leading sections — the trim-immune instrument is
  the provider-diagnostics block.
- `per_model_cohort` is a scoring filter; using it for aggregate faithfulness once
  manufactured 43 phantom drifters.
- Numeric PIT on `zero_point` questions needs the geometric value grid
  (`build_cdf_value_grid`), not linear interpolation.
- Numeric per-model means must be quoted ex-floor alongside raw.
- Miss-side process metrics require hit-side controls.
- Small-n honesty: quote cluster structure and effective n; two catastrophes can carry a
  cohort mean.
- The Metaculus comments API only returns our own comments; competitor analysis needs
  other channels.
