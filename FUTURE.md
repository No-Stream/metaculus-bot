# Future Ideas

Ideas for improving the forecasting bot, roughly ordered by expected impact and feasibility.

> **Status as of 2026-05-10** (closing residual on spring-aib-2026, n=189; receipts
> `scratch/analysis_2026-05/{analysis_synthesis,extended_hits_misses_postmortem,NEXT_SESSION_QUEUE}.md`).
> Two findings then reshaped priorities: (1) 17/20 worst misses were high-spread (>0.15) with a base
> model closer than the ensemble — models disagree on reference class and median pulls away from the
> closer minority (though "right model" attribution is post-hoc); (2) stacking treatment effect
> directionally measurable at +89.8% bootstrap (n=8, `analysis_stacking_historical_treatment.md`).
> Both are now largely superseded — stacking was later rejected/disabled, and the 2026-07-18 residual
> found the modal worst-miss has moved to consensus-with-zero-dissenters (see the top High-priority entry).

> **Status as of 2026-07-18 (july15 branch — shipped state).** This branch flipped several
> long-gestating items live in prod (all four workflow yamls). This block is just the index;
> detailed status lives in the per-item entries below.
>
> - **Agentic gap-fill v2** is ON (`GAP_FILL_V2_ENABLED: 'true'`) since 2026-07-17, running
>   concurrently with v1 during the overlap window.
> - **Time-series anchor (text)** is ON (`TS_ANCHOR_ENABLED: 'true'`) since 2026-07-17; the
>   chart-image side-channel (`TS_ANCHOR_CHART_ENABLED`) stays OFF pending its A/B.
> - **Summarizer, native-search, and crux-analyzer models migrated sol→terra** (native-search +
>   crux 2026-07-17; summarizer 2026-07-18) — see the per-role entries below.
> - **Supporting infrastructure landed**: raw pre-summarization AskNews text archived as
>   `asknews_raw` in the research-persistence record (`research/persistence.py` +
>   `research/orchestrator.py`), OpenRouter credit telemetry (`credit_telemetry.py`, wired in
>   `cli.py`), and the era-bucketed numeric width monitor
>   (`performance_analysis/width_monitor.py`).

## High-priority

### Spread-triggered second forecast round (re-forecast, NOT stacker) — top-priority design item (added 2026-07-19)

Status: design sketch only. Operator decision 2026-07-19 — this is the top-priority design
item, but explicitly NOT for right now.

**Honest caveat (load-bearing, read first).** Per the 2026-07-18 residual refresh, the modal
worst-miss has MOVED to consensus-with-zero-dissenters under a shared briefing — spread-gating is
structurally blind to that mode. This lever addresses the *disagreement* subset only; do not
expect it to fix the current worst misses. The consensus-miss counters are separate items
(gap-fill v2 verify / DISCREPANCY channel; cross-question coherence / resolution-metric
verification / publish-vs-own-anchor checks from the residual round).

**What.** On questions where forecaster spread exceeds the existing CONDITIONAL_STACKING
thresholds (~30% of questions per the one recorded estimate): run the EXISTING crux extractor →
EXISTING targeted search → append `## Targeted Research` to the bundle → re-fan-out all 6 base
forecasters → MEDIAN over round-2. The stacker stays off entirely — this is a re-forecast, not a
judge.

**Evidence.** BTF-2 (arXiv:2604.26106): 8 independent Opus rollouts straddle 50% on 38% of 200
difficult questions (mean per-q σ=0.08), and the strongest research agent (Opus-class) scored
*worse* on fixed evidence than on its own research (0.131 → 0.153; the effect was model-dependent,
with Gemini slightly *better* on fixed shared research at 0.143 → 0.141) — so the mechanism is
disagreement-triggered EXTRA
RESEARCH, not a smarter judge. AIA Forecaster (arXiv:2511.07678) independently found
disagreement → targeted-search its biggest aggregation lever. Crucially: ALL of this repo's
stacking rejections (n=88 ablation, stack_aug arms, trio-50q) tested "stacker LLM rewrites the
forecast"; a targeted-research-fed SECOND BASE-MODEL ROUND was never built or tested here, so the
rejection evidence does not apply to it.

**Wiring sketch.** `compute_spread` already runs on every prod question (`forecaster.py:676`);
the crux extractor + targeted search are importable functions currently welded to the stacker
path only by code arrangement (`forecaster.py:698-775`); reuse the fan-out helper
(`forecaster.py:593-604`) for round 2; re-aggregate via the existing `_base_combine` MEDIAN.
Flag-gated (e.g. `SECOND_ROUND_ENABLED`) + a telemetry marker. Open design decisions:
round-2-only vs pooled (round-1+round-2) median; a tighter round-2 soft deadline (~5 min) to fit
the 58:30 per-question wall clock; interaction with `WALL_CLOCK_STACKING_MIN_BUDGET`.

**Cost.** ~8-14 extra LLM calls (1 crux + 1 targeted search + 6 re-forecasts + parsers) on the
~30% triggered subset → roughly +30-40% forecaster spend.

**Eval.** Ablation harness `--stages forecast` on frozen research, paired on the triggered
subset; effects <0.02 Brier are undetectable at our n, so this is a big-lever ship-and-watch bet
with era-bucketing.

### Score the archived gap-fill v2 ghost forecasts (added 2026-07-18, HIGH, cheap)

Both approach reviews on this branch (`scratch/branch_review_july15/reviews/`) flagged the
gap-fill v2 ghost forecast as a latent asset that nothing scores. The loop already privately
dry-runs a forecast per question and archives it — the transcript + telemetry go through
`archive_sink=_capture_gap_fill_v2` (`research/orchestrator.py`) and a `GHOST_FORECAST` marker
lands in the run logs — but it is only ever logged for telemetry, never scored against
resolution. Build an offline harness that CRPS/log-scores three things per resolved question:
ghost-with-findings, resolution, and the panel's published forecast. **Interpretation guardrail:**
the ghost is a same-model (terra-low driver) counterfactual, NOT a panel proxy — it measures
whether the v2 findings alone, forecast by one cheap model, land near truth, not whether v2 would
improve the 6-model ensemble. This is the most decision-relevant single number for the
gap-fill-v1-retirement call (v1-retirement gate in the "Bundle content-audit findings" entry):
v1 currently carries the decisive single-source fact in most sampled questions, and a ghost score
is the cheapest read on whether v2 findings stand on their own before v1 goes off.

**Addendum 2026-07-19 (promoted to HIGH; scoreability work started).** (a) Shipped same day: the
archive payload now carries the full structured ghost (`gap_fill_v2.ghost`, pydantic
`model_dump`) alongside the transcript + telemetry. (b) In flight: a `GHOST_FORECAST_JSON`
single-line structured marker (full 13 numeric percentiles + full MC option probs) + a
telemetry-archive harvest + a `score_ghosts.py` upgrade to prefer JSON records and score numerics
via the existing CDF machinery, with a regex fallback for the pre-upgrade marker era. Rationale:
the legacy `GHOST_FORECAST` marker exposes only a numeric median, so numeric ghosts were
countable but not scoreable — the operator wants gap-free ghost-vs-published analysis.

### Re-evaluate the grok drop (6→5) once grok-4.5 has ~20-30 resolved questions (added 2026-07-19, HIGH)

Status: decision DEFERRED 2026-07-19 pending grok-4.5 evidence — do NOT act yet.

A paired leave-one-out replay on 2026-07-19 found that dropping grok from the 6-model roster
IMPROVES binary accuracy: Δlog-score **+1.83 [+0.74, +3.00]** favoring the drop (n=184). But
that signal is entirely **grok-4.3-lineage** — grok-4.5 (the current slot) has too few resolved
questions to score. Grok has long read as deadweight-but-harmless in the ensemble-composition
screening (the 8-era LOO replay found no subset beats the full roster, grok the least
load-bearing — see the "Ensemble composition screening 2026-07" memory); this is the first data
pointing at mild HARM rather than neutrality, but on the retired lineage.

**Gate:** re-run once grok-4.5 has ~20-30 resolved questions. The replay script
`scratch/residual_2026-07-18/followups/grok_loo_replay.py` is parameterized and free to re-run
(offline, no API). If grok-4.5 reproduces the drop-helps sign with a CI excluding zero, drop to
n=5 and rebalance the provider mix; if it's neutral/positive, keep grok. Era-bucket the read (a
roster swap starts a new era) — do NOT pool 4.3 and 4.5 evidence.

### Reconsider claude-fable-5 for the forecaster slot (added 2026-07-20, HIGH)

Status: pulled 2026-07-20; reconsideration tracked, do NOT re-add until the trigger below fires.

Fable-5 held a forecaster slot from 2026-07-15 (when it replaced opus-4.6) until 2026-07-20, when
it was pulled after returning `message.content=None` on 4/4 attempts for Q14333's numeric forecast
(the question was dropped, published 5/6) and a truncated no-JSON-block output on Q578 that needed
rung-3 LLM-parser salvage — both in the 2026-07-19 test_bot run (receipts:
`scratch/gha_test_bot_2026_07_19.md`). Suspected cause: fable-5 content classifiers refusing certain
question content, surfacing as fast deterministic empty completions (NOT timeouts). opus-4.7 took the
slot (xhigh, mirroring opus-4.8). **Fable-5's forecast quality was never the issue** — this is a
reliability/refusal problem, not a capability one.

**Revisit when:** root cause is confirmed (e.g. replay fable-5 against the exact Q14333/Q578 prompts
via a cheap manual call to see whether the empty completions reproduce and isolate the content
trigger), or provider behavior changes. If the refusals turn out to be narrow/rare, fable-5 is the
strongest available Anthropic tier and worth restoring.

### Reconsider fable-5 as the stacker (currently opus-4.8) (added 2026-07-20, HIGH)

Status: pulled 2026-07-20 alongside the forecaster slot; same trigger as the entry above.

The primary stacker was `claude-fable-5` from 2026-07-07 to 2026-07-20; it moved to
`claude-opus-4.8` in the same 2026-07-20 change (both roles pulled together after the content=None
failures). Note stacking is **prod-disabled** (all workflow yamls pin `*_STACKING_ENABLED=false`), so
this is backtest/ablation-only exposure today — lower urgency than the forecaster slot, but track it
so both roles are reconsidered together when the root cause lands. The `gpt-5.6-sol` cross-provider
fallback is unchanged.

## Research-triage round 2026-07-16 (lit + repo survey + codebase verification)

> Provenance: eight parallel research agents surveyed the last ~year of LLM-forecasting
> literature (evaluation pitfalls, prompting, ensembling, calibration, research harnesses,
> market-lifecycle) and the open-source forecasting-agent repos (Gnosis, MiroFlow,
> predictionprophet, vox, TimeCopilot, ForecastBench, futuresim, Bench-to-the-Future),
> plus two codebase verifications and a rigorous era-bucketed calibration audit. Entries
> below are grouped by priority within this block; thread them into the topical sections
> above if they survive discussion. Every "why it helps" from a repo is a mechanism argument
> unless a paper number is cited. The killed items are logged at the bottom so they aren't
> re-litigated.

### Calibration is probably NOT a realistic improvement lever for us (added 2026-07-16, decision) — but promote the audit to a monitoring module

Rigorous era-bucketed Bayesian calibration audit (2026-07-16; scripts
`scratch/calibration_audit_2026-07-16/`, data `scratch/coherence_2026-07-15/perf_all_tagged.json`,
694 records). **Conclusions:**

- **Not cleanly over- or under-confident — it flips by era.** Bayesian logistic slope:
  fall-aib-2025 **1.70 [1.28, 2.20]** (under-confident), spring-aib-2026 **0.83 [0.58, 1.10]**
  (over-confident), summer n=15 uninformative. Opposite-signed eras cancel to a ~calibrated pooled
  slope (1.17) — why past pooled reads were contradictory. Even within spring the early/late split
  flips 0.64 → 1.25 (roster churn inside one tournament).
- **"More NO right than YES right" is a base-rate artifact, NOT directional miscalibration.**
  YES-rate 24%/33%/60% by era; the decisive per-side test (observed vs expected-if-calibrated,
  Beta-Binomial CIs) is consistent with calibration on both sides in every era. Puzzle resolved.
- **The killed spring YES-overconfidence finding reproduces as spring-local** (intercept −0.40
  [−0.80, 0.00], P(a>0)=0.017) and is untestable on the current post-flip roster (n=15); fall shows
  the opposite sign. Nothing to fit.
- **Chronically data-starved for the miscalibrations a fitted layer would target.** Power sim:
  detecting mild overconfidence (slope 0.85) has power ~0.25 even at N=400; a roster era yields only
  ~150–210 resolved binaries before it churns. We catch only *large* miscalibrations, and any fit
  spans opposite-signed eras — a drift bomb by construction.

**Decision (operator, 2026-07-16): calibration is not a realistic source of improvement given our
budget. Prefer SOTA forecasters over fitted calibration.** Do NOT ship isotonic / Platt /
directional-shrink layers (see the killed-calibration entries; the Beta-Bernoulli "calibrator"
from arXiv 2605.27668 degrades OOD in its own tables — cite it as anti-adoption evidence).

**DO promote the audit to a standing monitoring module (free, zero scar risk).** Complementary to
`performance_analysis/analysis.py` (point-estimate buckets, a single `bias_pp` scalar, PIT
coverage): the audit adds multi-era comparison, Beta-Binomial CIs per bucket, the slope/intercept
decomposition (separates confidence from directional bias, which `bias_pp` conflates), the YES/NO
base-rate-artifact test, the power sim, and partial pooling. Reuse the existing `_interpolate_pit`
(log-grid fix) + bucket plumbing. Cadence: on each roster change print era-bucketed slope/intercept
CIs + reliability table + base-rate check; **act only if a CI excludes the null AND reproduces
across ≥2 eras** — "inconclusive" is the honest default.

### ~~Geometric-mean-of-odds base-combine vs MEDIAN~~ — RUN 2026-07-16, DECISIVE NULL (keep MEDIAN)

The one open aggregation variant the prior rejection never covered (binary geo-mean-of-odds as
the scored base-combine; MEDIAN is logit-invariant so only a mean-type pool can differ). Offline
replay, zero API (`scratch/geo_odds_2026-07-16/`, `geo_odds` arm + 5 tests in
`ablation/offline_replay.py`): NULL everywhere — no era's bootstrap CI excludes zero on log-score
or Brier (fall +0.32 [−1.34, +2.04]; spring +2.14 [−0.52, +5.34], Wilcoxon p=0.16; pooled +1.07
[−0.40, +2.68]), Brier opposite-signed to log-score (wash tell), and on the full-ensemble n≥5
filter fall FLIPS negative — confirming the documented "sharpening pools wash out on the diffuse
6-model ensemble" headwind. **Verdict: keep MEDIAN; geo-odds joins the settled dead paths (with
mean, stacking-as-default, coherence-weighting). Do NOT re-run** absent a materially different
ensemble regime (a confident small-N roster) or post-flip data with a different disagreement
structure. Arm + tests stay in the harness for cheap re-runs.

### Backtest statistical hardening + leak-aware replay (added 2026-07-16)

Two independent findings (papers-skeptic + eval-repos) plus a codebase check converge here.

**(a) Statistical rigor — AUDITED 2026-07-16, now a scoped consolidation program.** A methodology
audit found the recent scratch analyses strong-to-gold-standard (coherence phase2: two-way FE +
cluster-robust SEs; `ensemble_screen`: paired deltas + era-bucketing; calibration audit: Bayesian
posteriors + power sim) but the **standing pipeline is the thinnest machinery in the repo**, weak
in the two ways the house rubric names mandatory (pairing, era-bucketing). **Operator directive:
promote the scratch machinery into the standing pipeline.** Ranked gaps:

1. **`backtest.py` never pairs** (highest leverage): runs multiple bots on identical questions,
   persists per-qid scores (`backtest/analysis.py:205`), then reports pooled means + population SD
   with zero inferential content (`analysis.py:43-58,104-118`). Fix: join arms on qid, per-qid
   deltas, import the in-repo primitives `ablation/scoring.py`
   `bootstrap_mean_ci`/`bootstrap_median_ci`/`sign_test`/`wilcoxon_signed_rank`.
2. **`performance_analysis/analysis.py` emits pooled, CI-free, era-blind calibration — the
   actively-misleading item** (the pooled numbers that flipped three times). Fix: Beta-Binomial CIs
   on every bucket (`calibration_audit/binary_calibration.py:53-56`, a 4-line drop-in) + a
   first-class era key. This is where the calibration-monitor promotion lands.
3. **Per-model comparisons unpaired** in both standing surfaces — same fix as (1).
4. **Percentile bootstrap under-covers on heavy-tailed log-score deltas** — `ensemble_screen` needs
   the median/robust (or Bayesian) hedge ablation Path A already has.
5. **No clustering/effective-N outside coherence phase2** — mirror phase2's `cluster_bootstrap_mean`
   where a cheap cluster key exists.
6. **No multiplicity correction** in ablation (~15) or ensemble_screen (~48) — BH q-values or
   partial-pooling. Low priority.

**Explicitly NOT broken (don't make work):** `backtest/scoring.py` CRPS/log-score (correct),
`audit.py` (non-inferential by design), ablation Path B fold-std (stability diagnostic), plain
means on large-n bounded proportions. The problem is never "a mean was used" — it's "used as the
*comparison* with no pairing, CI, or era split." Reusable-machinery map: primitives
`ablation/scoring.py`; era-bucketing `ensemble_screen.py:109-140`; Bayesian CI drop-in
`calibration_audit/binary_calibration.py:53-56`; cluster-robust
`coherence_2026-07-15/phase2_lib.py:176-345`.

**(b) Leak-aware replay — plumbing exists, archive quality is the gap.** Default
`backtest_{smoke,small,medium,large}` targets **re-run live research at replay time**
(leakage-exposed; the leakage detector is a filter, not a fix). Frozen-replay already exists
(`--research-dir` → `make backtest_with_cache` → orchestrator skips providers), so code distance
to leak-free ≈ zero. The gap is archive quality: of 921 records in
`backtests/research_archive/latest/`, only **19 are genuine GHA-captured payloads; 902 are
reconstructed from published comments** (trimmed, empty `providers_used`), and uncached qids
silently fall back to live fetch. So: (1) flip default targets to `--research-dir`, (2) grow
genuine GHA-captured coverage from prod runs. Cheap near-term RetroSearch (below).

**Framing takeaway (papers-skeptic, arXiv 2506.00723 + Vaticinus preprint):** the re-run backtest
is an optimistic **upper bound**; calibration-on-own-published-forecasts (elicited at forecast
time, leak-free) is the trustworthy axis. And **stop treating "community Brier − our Brier" as
edge-over-market** — it's an affine shift of Brier on a balanced panel (ρ=1.000 across 25 rounds),
zero edge.

### Forecaster-prompt audit — verified clean, no action (added 2026-07-16)

The preregistered Schoenegger/Tetlock result (arXiv 2506.01578) found two prompt scaffolds
reliably *hurt* accuracy via over-decisiveness: step-by-step "Bayesian reasoning" and
"propose-evaluate-select". Audit of `prompts.py`: **neither harmful form present** — the closest
(binary:394 anchor-and-adjust) is the benign outside-view move the study carves out. The three
base prompts (~11.7k/7.8k/12.8k chars) are focused, not bloated. **No action.** Follow-up
**RESOLVED 2026-07-16:** the `base_rate_anchor` / `criteria_clauses` binary-schema fields are NOT
dead — live-elicited since `30bca2f` (2026-07-08); the "0/2203 rows" finding was a data-window
artifact (archive ended 2026-07-01). Wrong-mechanism claims in `comment/markers.py` +
`scratch/coherence_2026-07-15/synthesis.md` fixed same-day.

### Market-deference: time-to-close term MEASURED DEAD; liquidity fixes survive, downsized (updated 2026-07-16)

Applicability gate (offline archive mining) killed the headline half:

- **Time-to-close term: structural null — drop to bottom-of-low.** Of 285 matches with real close
  dates (n=64 questions, ~July 2026), **0.0% within 30 days of close** (median 185d), liquid∧near-close
  intersection **zero**. Structural, not small-n: a near-identical market closes ≈ when the question
  resolves, we forecast near open (`skip_previously_forecasted_questions=True`, `cli.py:119-148`), and
  the `as_of` filter drops markets closing before resolution. Revisit only if we add late re-forecasting.
- **Surviving — three small liquidity/matching fixes (top-of-low / bottom-of-medium):** (1)
  **Fallback-chain bug**: 39 real-money matches render `no-liquidity-data` because volume/OI fields are
  dropped — fix the `total_volume`/`open_interest` fallback (`prediction_market.py:498-506`). (2)
  **Fuzzy floor (40) so loose "match" ≈ topical-adjacent**: 100% "match" but confidence never exceeds
  0.77; ≥0.7-confidence (the defer trigger) is only ~8% of questions, ≥decent-liquidity ~12% — raise
  the floor or add a confidence tier. (3) Optional mild deference nudge for deep+high-confidence matches
  (plumbing already exists; prompt/render-level).
- Mechanics: `## Prediction Market Snapshot` header renders only on ≥1 match (`orchestrator.py:401-403`),
  so header-absence ≠ provider-off — use `providers_attempted`; close dates captured for
  Polymarket/Kalshi/Manifold (PredictIt lacks them).

**Rides along (low-value, operator 2026-07-16):** a bias-corrected **∆LL-over-matched-market diagnostic**
on the ~8% near-identical subset — do we add signal beyond the price or defer harder. NOT worth building
against the Metaculus CP (bot-only questions, CP is mostly-poor bots + null-hidden for our account).
Small subset → accumulates signal slowly.

### ~~JS-divergence diversity lens on ensemble screening~~ — BUILT 2026-07-16, verdict: NOT a selection signal (paper's heuristic INVERTS under median)

The "Diversity is the Strength of the AI Crowd" paper (arXiv 2606.29661) argues decorrelation
matters more than solo accuracy for roster decisions. Prototyped a JS-divergence metric on top of
`ensemble_screen`'s loaders (`scratch/js_divergence_2026-07-16/`, JS over
Bernoulli/option-simplex/202-bin PMFs). Findings: the metric is internally valid (Spearman(JS,
per-question score-corr) pooled −0.67, p≈1e-9 — high-JS pairs make independent errors), **but under
MEDIAN, decorrelation is INVERSELY related to marginal contribution** — Spearman(decorr,
removal-drop) = **+0.83** in the only rankable era (fall_6m): the most-decorrelated members (kimi,
grok) are the *least* load-bearing. Grok's independence is *incompetent* (off in a different
direction, not right-when-others-wrong) and MEDIAN discards outliers rather than harvesting them
(the paper's combiner was learned). **Verdict: do NOT promote JS as a roster-selection signal; the
marginal-contribution benchmark stays the decision instrument.** Keep the scratch code for two
uses: redundancy/near-clone detection (spring `opus-4.5|opus-4.6=0.017`, summer
`gpt-5.4|gpt-5.5=0.018` bits) and the error-complementarity Spearman as a validity check. Caveat:
the inversion is median-specific — a learned combiner/stacker could exploit decorrelation (relevant
only if the stacker-revisit lands).

### Revisit the conditional stacker with the AIA supervisor evidence (added 2026-07-16) — SUPERSEDED 2026-07-19

Folded into the top-priority entry "Spread-triggered second forecast round (re-forecast, NOT
stacker)" at the top of this file, which reframes the AIA (arXiv:2511.07678, supervisor 0.1125 vs
no-supervisor 0.1199 ≈ 0.0074 Brier) + BTF-2 evidence as motivating a targeted-research-fed
second BASE-MODEL round rather than a stacker revisit. The stacker-as-judge rejection still stands
(disabled in prod; our own benchmark found it counterproductive on the current ensemble).

### Time-series anchor for numeric questions — Phase A verdict IN; Phase B text anchor SHIPPED ON in prod (2026-07-17), chart still OFF

For numeric questions resolving on a fetchable series, render an empirical P10/P50/P90 band from
the series' own history into the briefing — **TS-as-anchor, not TS-as-answer**; a principled
version of the "status-quo / last-print anchor" finding. The shipped provider is deterministic
naive/empirical (no statsforecast dependency, no model selection).

**Phase A offline-replay verdict** (`scratch/ts_anchor_replay_2026-07-16/synthesis.md`). On 105
mapped class-A resolved numerics the empirical band beats our published CDF: paired relative skill
−1.17 (sign-flip permutation p=0.005 over 22 clusters; conservative CR2-t p=0.15 — cluster-fragile
but directionally robust). Decisive result is **coverage**: our published low tail is badly
miscalibrated (~3% below published P10 vs 10% target) while the anchor's P10 coverage is 0.18 and
P90 on target — sharper AND better-tail-calibrated. **CV model selection rejected**: the
must-beat-naive gate passed 76% but those picks beat naive OOS only 43% (worse than a coin flip),
so we render the naive/empirical band directly.

**Phase B text anchor SHIPPED ON 2026-07-17** (`TS_ANCHOR_ENABLED: 'true'` in all four yamls; chart
side-channel `TS_ANCHOR_CHART_ENABLED` stays OFF, next entry). Provider
`research/timeseries_anchor.py` + `ts_fetch.py` (deterministic routing + point-in-time/ALFRED-vintage
fetch + empirical bands); prompt clause `_ts_anchor_evidence_clause` (gated on section header).
Cleared its validation ladder (paid 3-arm smoke → `test_bot.yaml` eyeball → `make backtest_medium`,
leakage-safe because the provider date-ceilings the fetch to `open_time` under `is_benchmarking`) —
the FIRST prod research provider also measurable in backtest. The `TS_ANCHOR_ENABLE` config era
(2026-07-17) is tracked by the width monitor. Seed doc:
`scratch_docs_and_planning/ts_anchor_plan_seed_2026-07-16.md`.

**Applicability gate** (`scratch/ts_anchor_gate_2026-07-16/ts_labeled.json`): **53.2% (123/231) map to
a standard fetchable series** (fall 57% / spring 41% / summer 75%; strictest level-only reading 27%)
vs the ~10% skip bar. Class A is recurring templates (10Y yield, HY OAS, VIX, index/commodity returns,
gasoline, unemployment, CPI, payrolls, poll averages, TSA volume). Design notes:

- **Strong-value core = 63 level anchors** (macro prints, rates, spreads, poll averages, TSA).
  Spot-check: q43611/q43591 (poll averages) off in exactly the direction a live anchor corrects;
  q43647 (HY OAS) center-correct, would sharpen. (n=3, anecdotal.)
- **47 relative-return spread questions need the anchor fit on the SPREAD series** (center≈0 +
  historical-vol band), not a naive level forecast. The 13 max-functional questions (VIX/commodity
  highs) need window logic.
- **Net-new vs `financial_data`:** its allowlist misses HY OAS, gasoline, Brent, VIX, poll averages,
  TSA (4/5 spot-checked FRED series absent), and emits a raw last-6-obs table, not a fitted band.
- **Class B later-add:** Mauna Loa CO2 + Norwegian EV-share (held out of A on ingestion only; fold in
  once statsforecast wiring exists). Backtest measurement uses the gap-fill v2 eval-ladder pattern.

### TS anchor chart image — enable + A/B (HIGH, added 2026-07-17)

Beyond the text anchor's prose band, the chart side-channel passes each base model a rendered
800×400 PNG of the series + projected band as a vision message. **Skeleton shipped env-gated OFF**
(`TS_ANCHOR_CHART_ENABLED`, `'false'` in all four yamls): `research/ts_chart.py`
(`render_anchor_chart`, deterministic base64 PNG), a provider hook stashing the chart per-qid for
single LEVEL questions only (max-window/spread deferred), and forecaster plumbing threading the b64
→ `self._research_images[qid]` → the three runners via
`VisionMessageData(..., image_resolution="low")`. All 6 roster models are vision-capable via
OpenRouter (verified); stacker/summarizer/gap-fill/parser never see the image.

**Cost:** ~$0.02/question at low resolution — cheap enough to A/B but not free, hence gated
separately so the validated text anchor ships first.

**A/B needs:** arms bare / stats (text) / stats+chart on numeric-anchored questions, real read is
era-bucketed residuals on the level/spread cohort. **Prerequisite for archived-replay: research-sink
schema v3** — the archive stores only the text bundle; to replay chart arms offline the sink must
persist the chart b64 (or the band + series slice to re-render deterministically). Until v3, chart-arm
measurement is live-only. **Cheapest first signal — 3-arm smoke:** one hard resolved level-series
question, run a cheap model 3× (bare / stats / stats+chart), eyeball whether the image moves the
distribution beyond the text. Paid — gate behind operator sign-off.

### Necessary-condition / scenario decomposition scaffold (added 2026-07-16, bottom-of-medium)

Gnosis ThinkThoroughly (gnosis-repos): generate ~3 necessary preconditions + ~5 hypothetical
scenarios (incl. the negation) as sub-questions, research/score each, synthesize. Highest
ceiling of the "experiment" bucket, weakest evidence (no in-repo ablation; the PROPHET benchmark
found agentic-RAG scaffolding gave only marginal Brier improvement and naive RAG sometimes
*hurt*). Overlaps gap-fill v2's private dry-run (a lighter cousin). **Operator prior:** prompting
tricks mostly don't work — bots check the boxes you give them then put down what they want. File
it; gate hard on beating a flat forecaster in our own backtest before shipping.

### Lower-priority / logged (added 2026-07-16)

- **Embedding-NN market matcher + LLM resolution-equivalence check (low).** Replace/augment our
  fuzzy-string market matching (rapidfuzz) with embedding nearest-neighbor + similarity
  threshold (Gnosis pattern), plus an LLM "is this really the same resolution criterion" gate on
  top — catches paraphrase matches we miss, serves defer-to-market. Low-risk recall win, but low
  priority.
- **Consistency-check as an offline diagnostic (bottom of low).** The 0.85 consistency↔Brier
  correlation (arXiv 2412.18544) is a cheap no-resolutions-needed roster-health signal — probe
  members with negation/paraphrase variants, flag incoherent ones. **Diagnostic only** — the
  paper's own experiments show enforcing consistency does NOT reliably improve accuracy, and we
  killed coherence-*weighting* on 2026-07-15. Never in the prediction path.
- **Deterministic coherence projection (bottom of low).** KL/Brier projection onto monotone
  deadline/threshold ladders + MC-sum-to-1. Survives the era-test trivially (it's algebra, not a
  fit) but **rarely binds** — most AIB questions are standalone; sibling A/¬A/A⇒B ladders are
  rare (unlike a market book). We already enforce the two that apply (MC renormalize, CDF
  monotonicity). Low-yield; build a lightweight same-tournament ladder detector only if the
  ladder frequency turns out non-trivial.
- **ForecastBench external submission (bottom of low).** MIT code, open submission,
  contamination-free scoreboard vs superforecasters + other bots (arXiv 2409.19839). Genuinely
  useful for knowing where we stand leak-free — but **not free to forecast** (spends API on the
  submission question set), so it sits at the bottom.

### Study-only / longer-term (added 2026-07-16)

- **RetroSearch-style frozen point-in-time corpus** (FutureSearch, Bench-to-the-Future): live
  Google for ranking quality, filtered so the agent only ever receives pages already in a frozen,
  date-bounded per-question snapshot ("no tool returns a document from after the simulated now").
  The rigorous fix for backtest leakage. **Reproduce the design only** — RetroSearch code+corpus
  are proprietary; futuresim (openforecaster.github.io/futuresim) has NO license (study, don't
  copy). Cheaper reproducible variant: date-gated CommonCrawl News + LanceDB with query
  date-range control (both open). The near-term substitute is the frozen-artifact-replay in the
  backtest-hardening entry above — do that first.
- **Longitudinal trajectory scoring** (futuresim: sum of Brier-skill-score over timesteps;
  rewards correctness + calibration + timeliness). Suggestive that *timing* matters (TimeSeek),
  but no paper shows re-forecasting the same question over calendar time improves a bot's score,
  and it's an uncertain fit for single-shot Metaculus tournament scoring. Longer-term R&D.

### Answer to "should we go end-to-end agentic research?" — NO (shared, not integrated) (added 2026-07-16)

The strongest evidence (BTF-2, arXiv 2604.26106) argues against per-forecaster integrated
research+forecast pipelines: the single most accurate forecast was a **strong prompt on good
shared/fixed research** (0.129 Brier), edging the best self-directed integrated agent (0.131), and
integrated-vs-shared was **model-dependent** (only the Opus-class model clearly benefited from its
own search; Gemini was slightly *better* on fixed shared research). Our architecture (multi-provider
shared briefing + gap-fill + 6-model ensemble + median) already **is** BTF-2's winning recipe, at ~7
calls vs ~5N. The lever is **shared-research quality + a strong prompt**, not integration topology.
So: make the shared research more agentic (gap-fill v2,
`scratch_docs_and_planning/agentic_gap_fill_v2_plan.md`, SHIPPED + ON since 2026-07-17 — that half is
done); do NOT rebuild into per-forecaster agents; if ever tried, give it only to the Opus-class slot.
Calibration caveat: these agentic-research wins optimize pass@1, and edge-over-consensus has a
*flat-to-negative* trend across model generations — "more search → more decisive" trades calibration
for sharpness, so track era-bucketed calibration alongside Brier. See "Longer-term → Agentic deep
research" below (superseded by this).

**Addendum 2026-07-19 (BTF-2 re-read).** BTF-2 (arXiv:2604.26106) strengthens the case that
per-predictor research is the literal ideal for the strongest research agent (Opus-class), which
scored worse on fixed evidence than on its own research (0.131 → 0.153; the effect was
model-dependent, with Gemini slightly better on fixed shared research at 0.143 → 0.141). But at
~$3-10/question for 6 frontier agents each running
live search, it stays not-planned. Read the design as a continuum — shared bundle → shared bundle
+ one agentic researcher (gap-fill v2, current) → per-predictor research loops — where the
affordable middle is more v2 rounds / parallelism, not a rebuild. The NO verdict stands.

## Near-term (worth exploring soon)

### Agentic gap-fill v2: SHIPPED, ON in prod since 2026-07-17 (added 2026-07-16)

**FLAG STATUS: `GAP_FILL_V2_ENABLED: 'true'` in all four yamls** — flipped ON 2026-07-17 after
the paid smoke, the blind driver eval (winner: gpt-5.6-terra effort=low, now the prod default),
and the Exa-alive confirmation replay (`scratch/driver_replay_2026-07-17/arm_terra_low_exa_alive/`).
Pending: turn v1 gap-fill OFF after the overlap window (operator must remember — **now gated on
quality, not just time, per the 2026-07-18 content-audit entry**); the 3.5-flash researcher switch
is undecided. Full design (source of truth): `scratch_docs_and_planning/agentic_gap_fill_v2_plan.md`
(rev 4).

Summary: a bounded agentic tool loop is the second-pass research stage — a driver LLM briefed with
the forecaster prompt template privately dry-runs the forecast to find fill/verify targets, then
iterates over four tools: `search_news` (AskNews rate-limit machinery), `search_web` (Exa direct,
key `~/.keys/exa_key` / GHA secret `exa_key`), `fetch` (ladder plain → headless Chromium → Gemini
url_context), `read_document` (Gemini flash url_context). Output: a detached citation-only findings
artifact appended to the bundle; a ghost forecast logged for telemetry only. DIY litellm-direct,
append-only message array for prompt-cache discipline.

**Rollout: BOTH v1 and v2 run in prod during an overlap window** (independent flags
`GAP_FILL_ENABLED` / `GAP_FILL_V2_ENABLED`, distinct headers); artifact diffs + resolution scoring
harvested from the both-on era. Turning v1 OFF is a deliberate pending step — nothing does it
automatically.

Evidence base (plan doc §7): Metaculus Fall 2025 bot survey — research breadth ~ accuracy r=0.42;
FutureSearch #1 with agentic research ~$1/question; Bridgewater ablation agrees; date-filtered live
search leaks resolution info on 41–55% of resolved questions, so **`make backtest_*` is
uninterpretable for research-stage changes** — eval is test_bot QA then early prod flip.

Post-review (plan rev 4): dup-query semantic stuck-detector deferred (v1 ships only a per-run
`dup_tool_calls` counter); traditional-researcher tier-tagging is a follow-up to ride the same era
boundary; era-bucketed calibration is in the v2 eval ladder.

**DeepNews (AskNews's agentic research product; we call only basic HOT+HISTORICAL today).** Two
options vs the v2 driver: (a) expose as an optional heavy tool `search_news_deep(query, max_depth)`
the driver escalates to when basic search comes up thin; (b) upgrade the `search_news` backend to
DeepNews with a depth param. **Blocked on operator checking DeepNews limits/pricing** (separate
quota pool, possibly subsidized for tournament participants — if cheap, solid case for (a)).

### Bundle content-audit findings: v1 retirement gate, AskNews reform, market header (added 2026-07-18)

Three findings from the 2026-07-18 bundle content audit
(`scratch/bundle_content_audit_2026-07-17/RESULTS.md`, Fable-judged per-section value/redundancy).

1. **Gap-fill v1 retirement risk — do NOT flip v1 off on the calendar alone.** v1 is the MOST
   load-bearing section per token (59% unique content; carried the decisive single-source fact in
   the majority of sampled questions). The ~$190/quarter retirement savings holds only if quality
   is preserved. **Gate:** compare v1 vs v2 findings on the first ~20 prod questions where both are
   present; flip only if v2 consistently surfaces the same decisive facts.
2. **AskNews reform — shipped 2026-07-18.** Audit: 44% of the bundle by tokens, 57% padding, AND
   stale/directionally-wrong in 5/10 sampled questions while smaller sections had the right answer.
   "Longer is better" nudge removed (`5cfb6cd`); quality audit
   `scratch/asknews_quality_audit_2026-07-18/RESULTS.md` (R1-R5 + per-failure prevention matrix).
   Shipped in `asknews_summarizer_prompt`: **R1** recency-first ordering, **R2** supersession +
   quote-the-deadline-inputs arithmetic transparency, **R3** hard per-article relevance gate
   (off-topic DROPPED to a screened-out list), **R4** evidence-age disclosure opening the briefing,
   and an operator **proportionality rule** (length tracks decision-relevant content). The
   `asknews_raw` archiving hygiene rec is a separate in-flight change (orchestrator + persistence).
   R5 (fetch query enrichment) remains optional. Do NOT blindly halve the section — prioritization
   change + eyeball, not a cap.
3. **Prediction-market header revision (low effort, medium impact).** The `STRONG EVIDENCE -- weight
   these markets heavily` header fires unconditionally (`research/prediction_market.py:1184`) even at
   low fuzzy-match relevance — ~56% of sampled questions had off-topic/loosely-matched markets and
   forecasters anchor on them. Fix: qualify the research-side header by match confidence; the
   forecaster-prompt clause (`prompts.py:372`) already discounts on mismatch — the problem is the
   research-side header priming the model first. **Caution:** the `[PRE-WINDOW]` apparatus in the
   summarizer output is load-bearing (prevents pre-open events read as resolutions — has saved
   multiple questions); do NOT remove, at most abbreviate after first occurrence.

### Confirm Gemini `url_context` actually fires in prod (added 2026-06-28)

The 2026-06-28 research-quality audit found **zero positive evidence** that Gemini's `url_context`
tool (built to directly read criteria-named resolution URLs) ever fires in prod: across 17 Period-B
records every Gemini section cites only `grounding-api-redirect/` links — 0/17 a direct
`.gov`/`fred`/`cboe` URL. Live resolving values came from the **gap-fill native-search pass** or the
**financial-data API**, not url_context (damning case q43650: Gemini's snippet was wrong 4.44–4.46%
while gap-fill returned the exact 4.48% that resolved). The fetch gap is *masked by gap-fill*, not
*closed by url_context*.

**Telemetry added 2026-06-28** (`gemini_search.py` `_extract_url_context_telemetry`): each grounded
call logs `N/M url_context fetches` and writes a greppable marker — `### URL Context Fetches` (reads +
URLs) or `_url_context: none_`. **Action (free):** after the next prod run, grep
`backtests/research_archive/latest/*.json` for those markers to settle whether url_context fires and
reads the named URL. If it reliably direct-reads named sources, the deterministic-fetch question
dissolves; if never, that justifies the narrow named-URL fetcher.

**Related (deferred, needs a small paid re-bench — clear cost):** the audit could NOT test the gap's
worst case — obscure non-API official counters/registries/dashboards (state policy trackers, CBP
tables, WHO-style dashboards, mesonet tables). Period B had zero; the only two clean research-side
fetch failures in the 40-tracker corpus (q43046 WHO extranet, q43139 IEM mesonet) were both this type,
both pre-current-stack. A ~10–15-question adversarial re-bench enriched for that archetype is the
highest-leverage missing evidence. Narrow design (parse criteria for a resolver URL → force-fetch, or
make the gap-fill analyzer treat a criteria-named URL as mandatory) sketched in
`scratch/research_audit_2026-06-27/SYNTHESIS_62.md` §4. FRED/Yahoo URL-extraction (shipped 2026-06-28)
already covers the API-backed financial subset.

### Resolution-source fetcher: Tier-2 LLM fetch + oversized-source summarization (added 2026-07-09)

Tier-1 deterministic fetcher shipped `66e31c0` (`research/resolution_source.py` +
`research/http_fetch.py`, `RESOLUTION_SOURCE_ENABLED`). Smoke-validated on 40 cached questions
(`scratch/resolution_source_smoke_2026-07-09/REPORT.md`): 24/40 (60%) get a non-empty snapshot vs
the 62.5% Tier-1 target, 30/45 URL success, 0 SSRF false positives, first-cited URL was the primary
grading source in all 12 multi-URL questions; remaining misses are known Tier-2 hosts (JS walls /
fingerprinting). **Flipped ON in all three prod yamls 2026-07-10** after a `test_bot.yaml` eyeball
(same commit added the per-URL truncation marker + dropped-section note).

**Truncation-cap study (2026-07-09, don't re-derive):** uncapped re-fetch of the 29 smoke URLs:
p25=697 / p50=2,201 / p75=5,206 / p90=67,041 / max=438,049 chars. Elbow at 6,000/URL (3,000 cap
truncates 48%, 6,000 truncates 21%, past 6,000 only whales that need summarization). **Shipped:**
`RESOLUTION_SOURCE_PER_URL_MAX_CHARS` 3,000→6,000, `RESOLUTION_SOURCE_TOTAL_MAX_CHARS` 12,000→18,000.

Follow-ups:

1. **MEDIUM — conditional summarization for oversized sources.** First-cited URL stays verbatim
   (provenance); URLs 2+ / whales (≥~10k chars) go through the cheap summarizer (`gpt-5.4-mini`, temp
   0, ~$0.01/call). ~5 whale sources per 40 questions no cap captures.
2. **MEDIUM — Tier-2 LLM fetch** for the js_wall/blocked slice (~15%; Masters.com, childmortality.org,
   UNICEF, Tesla IR, sagaftra.org). The per-URL `FetchStatus` (blocked/js_wall) is the seam.
   **Precondition:** the Gemini `url_context` probe above. *Note 2026-07-16:* the gap-fill v2 fetch
   ladder gives the driver this capability inside the loop, so the js_wall slice may get covered
   agentically first — re-assess after the v2 overlap window.
3. **LOW (deferred):** module split of `resolution_source.py` (~670 LoC; extract `ssrf_guard.py`).

### Parser hardening + forecasting-tools upgrade path (added 2026-07-07)

Full plan `scratch_docs_and_planning/parser_hardening_and_ft_upgrade_plan.md`. Decision: do NOT
migrate forecaster calls to native `response_format` structured outputs (OpenRouter
silent-degradation footguns, load-bearing rationale channel, zero competitive precedent).

- **Workstream A — DONE (superseded).** Shadow-divergence logging shipped, served its purpose,
  deleted 2026-07-10 when the block became authoritative (`EXTRACTION_RUNG` telemetry replaced it).
  Strict json_schema on the *parser call* (`structured_parse.py`) shipped and is now the ladder's
  rung-3 salvage.
- **JSON-block-as-authoritative for ALL question types + stacker — DONE 2026-07-10.** Value
  extraction runs the deterministic four-rung ladder in `value_extraction.py` (block parse →
  json-repair → LLM-parser salvage → `ValueExtractionError`). The old "wait for ~50 questions of
  shadow-divergence data" trigger was operator-waived for `EXTRACTION_RUNG` telemetry + a gated
  `test_bot` eyeball.
- **Workstream B (active, ~1 focused day):** unfreeze `forecasting-tools` 0.2.54 → 0.2.92+. Two
  verified breaks: our PCHIP subclasses override `.cdf` but HEAD moved internals to `get_cdf()`
  (silent bypass of our CDF machinery); `fetch_hardening` patch target moved to `MetaculusClient`
  (silent no-op). Plus a validator audit — HEAD's `_check_too_far_from_bounds` (25% wiggle) may
  conflict with our beyond-range open-bound design. Unlocks the litellm/cryptography CVE fixes below.

### Percent-form block labels vanish silently in comment recovery (added 2026-07-15)

A numeric STRUCTURED FORECAST block whose `declared_percentiles` keys are percent-form ("2.5" …
"97.5") not fraction-form (0.025 …) is dropped by BOTH recovery rungs in
`performance_analysis/parsing.py` (strict `parse_structured_block` rejects the schema; the tolerant
salvage rung drops the keys on its `0 < pct < 1` guard). Historically harmless (prose "Percentile
2.5: X" lines rescued these, e.g. qid 43684 / grok-4.3), but post-2026-07 block-last-no-prose prompts
leave no fallback — that model's percentiles vanish silently from residual analysis. Fix: teach the
tolerant rung to detect a canonical-set×100 key match and rescale (validator + canonical sets already
exist: `_validate_percentile_labels`, `_CANONICAL_PERCENT_LABEL_SETS`, parsing.py:600-650). Watch
signal: a model whose per-question percentile coverage drops to zero in a post-flip pull while its
`EXTRACTION_RUNG` prod telemetry stays healthy.

### Dependency CVEs gated by the frozen `forecasting-tools` pin

`make audit` (osv-scanner over `uv.lock`) flags CVEs we can't patch while
`forecasting-tools==0.2.54` is frozen. As of 2026-07-19 the gated set is down to two packages:

- **litellm 1.80.0** — four high-severity CVEs (GHSA-4xpc-pv4p-pm3w 9.5, GHSA-jjhc-v7c2-5hh6 9.4,
  GHSA-53mr/69x8 8.6–8.7), fixed in 1.83.x–1.84.0; forecasting-tools pins litellm to exactly 1.80.0
  (our `<2.0.0` cap is not binding), unreachable without bumping forecasting-tools.
- **cryptography 45.0.4** — incl. one 9.8 (PYSEC-2026-36), transitive via asknews/google-auth/mcp.

Resolved 2026-07-19: pillow and transformers came off the gated list via `[tool.uv]` settings in
pyproject.toml (pillow forced past the never-importing forecasting-tools `<12` cap to 12.3.0 via
`override-dependencies`; transformers — declared-but-unused, with an unreachable-fix critical RCE —
removed from resolution entirely via `exclude-dependencies`), and pydantic-settings via an in-range
bump to 2.14.2. Re-evaluate (and ideally delete) both settings at the next forecasting-tools bump.

Accepted consequence of freezing forecasting-tools. Revisit on the next forecasting-tools upgrade
(re-run `make audit`); if a CVE becomes actively exploited first, evaluate a `[tool.uv]
override-dependencies` + re-validate the numeric/stacking pipeline. CI runs the same scan on every PR.

### Promote the core pipeline to `basedpyright` strict

The 2026-06 uv migration wired `basedpyright` at **standard** mode repo-wide, zero errors. The
intent was **strict on the core pipeline**; deferred because against an unclean base strict surfaced
~900 findings (~450 low-value "partially unknown" noise at the untyped `forecasting-tools` boundary).
Now the base is clean the promotion is much smaller. Spec:

1. Add the `strict` path list to `[tool.basedpyright]`: `metaculus_bot/{numeric, research,
   probabilistic_tools, forecaster.py, aggregation_pipeline.py, stacking.py}`.
2. Resolve the resulting `reportUnknown*` findings (~447 last measured) by **annotating our own
   functions** — ~95% of the unknowns are our own under-typed signatures/locals, not the library.
3. `forecasting-tools` ships **no `py.typed`** despite inline annotations, so strict re-raises
   `reportMissingTypeStubs`. `--createstub` is NOT a clean fix (drops Pydantic attrs, 892→1011).
   Options: (a) thin hand-maintained `typings/forecasting_tools/` stub for imported symbols only;
   (b) a `reportMissingTypeStubs` override scoped to the strict execution environments (the global
   `= false` does not survive the strict promotion); (c) request `py.typed` upstream.
4. Re-add the local `basedpyright` pre-commit hook (removed during migration) + gate CI.
5. Do as a heavily-parallel workflow (one agent per core module).

### ~~Supervisor agent for high-disagreement questions~~ DONE

Implemented as conditional stacking (`AggregationStrategy.CONDITIONAL_STACKING`).

### ~~Financial data tool access (yFinance, FRED)~~ DONE

Implemented as `financial_data_provider.py`.

### Run crux extraction on every question + always-on stacker (added 2026-05-17) — largely superseded

Original idea: always run forecaster fan-out → crux extract → targeted re-research → stacker (vs the
current ~30% high-spread trigger). **Cost** at gpt-5.5 high effort, ~250 Qs/tournament: crux $14 +
targeted search $19 + stacker $75 ≈ **+$80/tournament** vs disagreement-only. Open question: do cruxes
move predictions on uncontroversial questions, or just add latency? Paired — the always-on stacker half
is now benchmark-rejected (stacker disabled in prod). The live version of "spend the crux research on a
real re-forecast" is the top-priority "Spread-triggered second forecast round" entry, which keeps the
trigger and drops the stacker.

### Re-run native-search model evaluation each quarter (added 2026-05-17)

**Current model (2026-07-17): `openai/gpt-5.6-terra` at `effort=low` + `verbosity=low`, 360s**
(`NATIVE_SEARCH_DEFAULT_MODEL` / `NATIVE_SEARCH_REASONING_EFFORT_DEFAULT` in `constants.py`). Slot
history: grok→gpt-5.5 (2026-05-17) → gpt-5.6-sol (2026-07-09) → gpt-5.6-terra (2026-07-17, blind
research-role audit `scratch/research_role_audit_2026-07-17/` — terra 1st, "MARGINAL EDGE", −42%
cost); effort low since 2026-05-20 for latency. Baseline
`scratch/native_search_bench_2026-05-17/comparison_v3.md`. As cheaper/better OpenAI search models
ship (or Anthropic/Google add native search to OpenRouter), re-run `python
scratch/native_search_bench_2026-05-17/run.py --question-id <new open Q>` and update
`NATIVE_SEARCH_DEFAULT_MODEL`. Bake into the quarterly review cadence.

### Gemini on the donated OpenRouter key: pro-preview still blocked by free-tier BYOK (updated 2026-06-16)

The donated key (`OAI_ANTH_OPENROUTER_KEY`) now serves most Gemini (verified live:
`gemini-3.5-flash`, `gemini-3.1-flash-lite` both SUCCEED). **Remaining blocker —
`gemini-3.1-pro-preview` only (our forecaster slot):** routed through a free-tier Google AI Studio
**BYOK** key on the donated account (`is_byok:true`), and Pro-preview has no Google free tier, so
quota is structurally 0 — every donated call 429s `RESOURCE_EXHAUSTED`.

**Resolution — SURGICAL pin:** `GEMINI_USE_DONATED_OPENROUTER_KEY=true` in all four yamls (flash uses
donated), but `gemini-3.1-pro-preview` is **pinned to the personal key** via
`DONATED_KEY_BLOCKED_GOOGLE_MODELS` in `fallback_openrouter.py` (`should_route_via_donated_key` →
`False`; no donated attempt, no 429, no fallback-counter bump). Without the pin, each donated→429→personal
fallback bumps the counter → `cli` `sys.exit(1)` → **red CI every run**. OpenAI/Anthropic on the donated
key are unaffected.

**⚠️ TEMPORARY WORKAROUND, tagged `TODO(gemini-3.1-pro-donated)`.** Remove the blocklist entry once
Metaculus fixes the BYOK routing (pick one, account-side): (1) enable Cloud billing on the BYOK GCP
project → Tier 1 quota; (2) remove the Google AI Studio BYOK integration so `google/*` uses native
OpenRouter Google credits; (3) disable "Always use for this provider" on that BYOK key. Does NOT help:
raising OpenRouter-side native limits (the 429 is Google-side). Pinged Ben; after a fix, re-verify with
one live call and delete the entry.

### ✅ RESOLVED 2026-05-29 — `OAI_ANTH_OPENROUTER_KEY` data-policy block for OpenAI native search

Metaculus enabled OpenAI on the donated key. `build_native_search_llm` routes through
`build_llm_with_openrouter_fallback` (donated primary, personal fallback); verified end-to-end on
`openai/gpt-5-mini` (grounded result, 404 fallback count = 0). The original block was a 404 "no
endpoints matching your guardrail restrictions and data policy"; the guardrail/data-policy fallback
matcher in `fallback_openrouter.py` stays as a safety net for the next provider migration.

### Second-pass web search + scrape pipeline

**SUPERSEDED 2026-07-16** by the agentic gap-fill v2 plan
(`scratch_docs_and_planning/agentic_gap_fill_v2_plan.md`). The three use cases (gap-filling,
resolution-source reading, reopening inaccessible PDF/JS/paywalled sources) are covered by the v2
tool loop; Firecrawl/Olostep were rejected in favor of a DIY fetch ladder (plain → headless
Chromium → Gemini url_context).

### Separate outside/inside view stages

Split the single-prompt outside+inside view into separate LLM calls so the inside view genuinely
adjusts FROM an explicit base rate — could help the arithmetic-override problem (models computing
correct probabilities then ignoring them). Smingers does this with cross-pollination (outside view
from Model A → inside view for Model B, adding diversity). Prototype: first pass produces base rate +
reference class, fed explicitly to the second. Moderate-to-high effort.

### ~~Post-hoc isotonic calibration on binary predictions~~ — DROPPED 2026-05-10

The May closing analysis (n=109 binary) did NOT replicate the [0.20,0.30]-band NO-bias concentration
that the April-new n=27 cohort showed — the April finding was a small-N artifact, and the 20 worst May
misses span many failure modes. The original proposal: fit `sklearn.isotonic.IsotonicRegression` on
`(our_prob_yes, resolution)` behind a `USE_BINARY_CALIBRATION` flag, refit quarterly, log
raw+calibrated. **Killed**, and independently re-confirmed dead by the 2026-07-16 calibration audit
(don't ship isotonic/Platt/directional-shrink — see the top calibration entry). Defer any global
isotonic until a >50pp residual is observed in a predicted-prob band on N≥50.

### Probabilistic tooling for base forecasters — DORMANT / on the settled dead-paths list

Infra built (`probabilistic_tools/`, `tool_runner.py`, `structured_output_schema.py` — Beta-binomial
updaters, survival/hazard, log-pooling + Satopää, distribution fits with out-of-bounds mass,
Dirichlet-with-Other, NegBinom/Beta-binomial-ceiling for counts, consistency checks), 261 tests green,
gated behind `PROBABILISTIC_TOOLS_ENABLED`. **Currently DISABLED in prod** (all prod yamls `'false'`;
tier-2 scaffold removed from prompts via Workstream C2) and on the settled dead-paths list (benchmarked
+ rejected — don't re-recommend without new evidence). Activation guide (exact edits, parser-ordering
gotcha, A/B sequence): `scratch_docs_and_planning/probabilistic_tools_activation.md`. Original motivation
was two numeric representation failures the percentile elicitation can't express: NM1 (DOJ antitrust —
model wrote "~92%" of 0 in prose but only ~55% mass at-or-below 0; needs Beta-binomial-ceiling/NegBinom)
and NM3 (MSFT EPS — knew the GAAP-vs-adjusted tail risk, couldn't widen enough, resolved past P97.5;
needs out-of-bounds mass reporting).

### Status-quo / last-print anchor for slow-moving numeric trackers (added 2026-06-28, NEEDS BACKTEST) — partly shipped as TS anchor

**Finding (2026-06-28 period-split audit, Period-B n=17, ~5 numeric trackers — directional only):** on
numeric/discrete questions resolving on a slow-moving/mean-reverting tracker, research surfaces the exact
current value and the ensemble then *degrades* it with directional drift or asymmetric widening. The three
worst Period-B trackers all had the resolving value in research (all scored positively — "left points on
the table," not lost):

- **q43647 HY-OAS** (peer +52.4): research handed over 2.71; medians skewed UP to 2.73–2.80; truth below p40.
- **q43611 generic ballot** (peer +31.8): research surfaced Silver +6.8 + a mean-reversion base rate; ensemble
  extrapolated to ~6.7–7.2; reverted to 6.4.
- **q43591 Trump approval** (peer +14.3): research surfaced 38.5; ensemble drifted down to 37.8; ticked up to 38.6.

**Proposed change (prompt-only):** in `numeric_prompt`/discrete handling, when research surfaces a current
authoritative value for a slow-moving/mean-reverting tracker, default p50 + the bulk of mass to that last
print and require an EXPLICIT named justification before applying drift — a *rebuttable* default, not a hard
constraint. **Note:** the shipped TS anchor (see the TS-anchor entry) covers the fetchable-series subset of
this with an empirical band; this prompt lever is the broader, non-fetchable version.

**Why NOT shipped:** n=3; over-correction risk on genuinely-trending / count-toward-deadline numerics (the
model must classify slow/mean-reverting vs trending); prompt-behavior value is unprovable without a paid
backtest. **Backtest gate (clear cost first):** A/B on a numeric/discrete-heavy slice — improvement on
tracker questions AND no regression on trending/event-driven ones. Full evidence
`scratch/research_audit_2026-06-27/SYNTHESIS_62.md` §3.

### LLM-based forecast self-evaluation

After each forecast, run a cheap model to assess: research relevance, factual accuracy,
reasoning soundness, date/chronology correctness, resolution criteria interpretation.
Flag potential issues before submission.

Smingers found this invaluable for catching date confusion, hallucinated sources, and
reasoning failures. Implementation: easy (structured eval prompt + cheap model call).

### Hits-side reasoning prompt-test ideas (added 2026-05-10) — LOW PRIORITY, defer

Three prompt edits from the May analysis (`scratch/analysis_2026-05/analysis_hits_reasoning_patterns.md`),
hypothesis-generating not shipping recs (need N≥30 prompt-vs-prompt backtest):

1. **"State your Poisson lambda explicitly"** — 5/10 top hits used `P(≥1)=1−exp(−λT)` with stated λ, T.
2. **"Required-vs-observed pace section"** — 3/10 top hits used threshold-by-deadline arithmetic. Its
   Wikipedia hit/miss base is N=1 not N=2 (miss 42238 applied the math correctly, landed in a 16% tail).
3. **"Distrust briefing claims that contradict the question's open status"** — inverse of the April
   Klimt-sale hallucination (miss 42243, 4/5 models pulled by a fake datapoint). N=1, weakest.

Deferred: #1/#2 overlap the (now-dormant) probabilistic_tools Poisson/pace math — don't A/B in parallel.
Risk: prompt-length growth degrades simple questions; gate is "mean Brier improves, no per-cell regression
on easy/middle tier."

### Stacker prompt: tell it which models are reliable dissenters (added 2026-05-10)

May C5+C7 analysis: gpt-5.2 is a **contrarian signal source** (8/20 best on worst-misses, 0/20 on hits,
mid-pack Brier 0.150) — the high-disagreement signal the conditional stacker should up-weight;
claude-4.6-opus has the inverse profile. The stacker prompt strips model IDs (self-agreement bias) but
the *historical* "reliable dissenter on high-spread questions" pattern is signal we throw away.
Hypothesis: a small "historical dissenter quality" hint could improve stacked outputs on the high-spread
cohort. Blocked on: STACKER_OUTCOME marker fix + ≥30 stacked records under it (and the stacker is
disabled in prod). Defer.

### Per-forecaster critic/revision pass (added 2026-07-08, medium priority)

An unconditional adversarial critic reviews each forecaster's draft against a resolution-criteria
checklist BEFORE aggregation (window discipline, already-resolved-events-don't-count, listing/instrument
bar, blind-spot pricing); the forecaster answers point-by-point and re-issues, revision capped ±20%.
**Evidence:** Laertes (summer futureeval-2026 #4) runs this on all forecasters — qid 42024: its Forecaster
1 drafted 97% (our published number) and the critic reversed it to 4% after flagging the "resolving" event
fell outside the open window. GreeneiBot2 (spring #1) runs capped critique rounds. Two top bots converging
is a notable signal.

**Caveat:** the demonstrated evidence is on **degenerate** failures (pre-open-window traps, criteria
misreads); generalizing to non-degenerate misses is unproven. Cost ~1 extra LLM call/forecaster/question.
**Distinct from the stacker** (benchmarked + rejected): this is per-member BEFORE aggregation with a bounded
±20% revision, not a post-hoc aggregate rewrite. **Gate:** `make backtest_medium` mixed cohort, primary peer
score — (a) improvement/no-regression on the degenerate subset, (b) no regression on non-degenerate. If (a)
lands but (b) regresses, ship as a **conditional** critic (fires only on pre-open-event/listing-bar/mismatch
flags).

**Update 2026-07-08 (acid test + guard counterfactuals):** down-scoped. On the two non-degenerate consensus
misses (41800 / 42855), advisory critique captured ~0 points even when it diagnosed the exact defect — only
BINDING corrections (bounded numeric adjustment / floor / cap, mechanically applied) moved numbers. The
"sequence after free deterministic guards" premise is falsified: all three guards were buried on offline
replay (era sign-flips / top-5 concentration / fall-hostile fire rates — see the three guard entries in the
"Killed by July 2026-07-08" section). So the critic carries the full burden on its own paid backtest, and
its gate must include a fall-like era-stability check (harvesting the spring miss cluster is exactly what
damages the largest, best-calibrated era). Receipts: `scratch/residual_2026-07-08/ACID_TEST_VERDICT.md` §3,
`scratch/residual_2026-07-08/experiments/GUARDS_SYNTHESIS.md`.

### Telemetry-first guard revival program (added 2026-07-08, passive)

Shipped `30bca2f` telemetry (`base_rate_anchor {low, high}` + `criteria_clauses` on `BinaryStructured`)
plus `PREDICTION_MARKETS_ENABLED: 'true'` make future guard replays exact rather than parser-based. No
code on the roadmap; passive. Note: the computed `ANCHOR_OVERSHOOT_PP` / `CLAUSE_PRODUCT_DIVERGENCE_PP`
markers emit only from `tool_runner` (gated behind `PROBABILISTIC_TOOLS_ENABLED`, off in prod) so they're
DORMANT; the raw `base_rate_anchor` / `criteria_clauses` JSON lands unconditionally and the
overshoot/divergence math is trivially replayable offline from it.

Two free checks next residual session: (1) **structured-JSON presence rate per forecaster** — grep the
archive for the raw JSON keys, confirm every slot emits them; (2) **does the spring overshoot pattern
reproduce on the current roster?** — if the confident-overshoot cluster (42024 / 42304 / 41800 analogues)
doesn't appear post-`30bca2f`, the prompt fixes sufficed and all three guard-revival conditions become
moot. `clause_product_divergence_pp` (published vs the model's own priced clause product) is the first
trigger keying on divergence-from-own-math — the conditionality the three tested guards failed to achieve.
Watch, don't act.

Also watch (MC, 2026-07-09): whether the low-bucket over-payment closes under the merged MC calibration
bullet (`ceab2df`). Baseline: [0-5%) options assigned mean 2.4%, resolve at 1.0% (n=96, both eras —
"courtesy mass" on named-dead longshots; MC_CONFIDENCE_FINDINGS.md). If the gap persists, add one prompt
line: price clearly-dead NAMED options near the 1% floor (residual/"Other" keep honest mass). The 1% floor
stays (operator 2026-07-09: sub-1% headroom ~+0.01 nats/question vs parser/clamp regression risk — not worth it).

### File splits + shared fetch-primitive promotion (added 2026-07-18, low, standalone PRs)

Structure findings from the branch-review forge + structure reviewers
(`scratch/branch_review_july15/reviews/`). Each is a clean behavior-neutral refactor — keep
them OUT of feature work, land as their own PRs.

- **Three files over the monolith threshold** (measured 2026-07-18):
  `research/timeseries_anchor.py` (986 LoC — split the routing registry out into
  `ts_routing.py` per the structure reviews), `research/agentic/tools.py` (784 — split the
  search vs fetch subsystems), `tests/test_agentic_tools.py` (1055 after this branch's added
  tests).
- **Promote the shared SSRF/fetch primitives into `http_fetch.py`.** `agentic/tools.py`
  currently reaches into `resolution_source.py`'s private functions — it calls
  `resolution_source._get_session`, `_sem_for_host`, and `_extract_main_text` directly
  (`agentic/tools.py:217,448,529,625`). That private-function coupling across modules is the
  smell; hoist those three into the shared `research/http_fetch.py` as public primitives and
  have both call sites use them.
- **Give the anchor-chart `_session_charts` global a public accessor.** It's a module-level
  dict in `research/timeseries_anchor.py:919` mutated/read by qid; expose a small
  get/set/clear surface instead of touching the global directly.

## Medium-term (requires more exploration)

### Consider migrating scheduled runs off GitHub Actions cron (added 2026-07-19, MEDIUM)

The 2026-07-18 latency/completeness audit
(`scratch/residual_2026-07-18/followups/latency_completeness.md`) traced ~80% of the
submission-latency p90 creep (24 → 58 min) to GitHub Actions cron STARVATION: the scheduled
workflows fire ~36 times/day against a nominal 72 (GHA silently drops scheduled runs under
load). Queue-delay p90 is already 82 min against 90-minute question windows, so worst-case
queue (82) + pipeline (12 p90) breaches the close deadline — and a missed close forfeits the
whole spot score. Gap-fill v2 (up to 540s wall) stacks more pipeline time on top once the
july15 branch merges.

Time-sensitive because the failure mode is a hard forfeit, not a soft regression. No clean easy
migration exists (hence MEDIUM, not HIGH) — alternatives to scope: a self-hosted GHA runner
(kills queue starvation, adds ops burden), EventBridge Scheduler → `workflow_dispatch`, or a
small VM / fly.io cron firing `workflow_dispatch`. The new CLOSE_MARGIN watch
(`make close_margin_watch`) is the instrument to confirm the problem persists and to measure any
migration's effect.

### Split `forecaster.py` (1066 LoC, past the ~1000 ceiling) (added 2026-07-20, MEDIUM)

Status: deferred refactor from the 2026-07-20 forge review of the run-QA commits (finding F3);
deliberately NOT fixed in the july15 branch.

`metaculus_bot/forecaster.py` is 1066 lines, past our ~1000-LoC file ceiling, and keeps growing as
research stages and post-processing accrete. Natural extraction seams: (1) the gather /
wall-clock / soft-deadline concurrency machinery (`_forecaster_with_soft_deadline` and the
parallel fan-out plumbing), and (2) the stacking-finalization helpers, which belong in the
existing `stacking` module rather than the forecaster. Large-blast-radius (touches the hottest
file in the pipeline), so it warrants its own PR **after the july15 branch merges** — tracked here
so the file doesn't keep accreting in the meantime.

### Price the high→xhigh reasoning-effort premium via backtest A/B (added 2026-07-20, MEDIUM)

Status: operator explicitly deferred 2026-07-20 — worth doing, but the backtest budget is
contended; revisit when budget frees up or before the next major effort-config decision.

**Motivation.** The 2026-07-20 reasoning-effort audit
(`scratch/reasoning_effort_audit_2026-07-20/synthesis.md`, built from the AIB spring-2026 metac
baseline-bot leaderboard) found default→high effort is clearly worth it: 8/8 within-model
contrasts positive, sign test p=0.0078, median +2.9 spot-peer points/question, and thinking
OFF→ON is the single largest knob (opus-4-5 flipped −1.9 → +2.9/Q). BUT the board has zero xhigh
variants, so the high→xhigh premium — what we actually pay on 4 of 6 forecaster slots, ~64% of
donated per-run spend per the 2026-07-19 credit audit — is unmeasured. The audit's
diminishing-returns hint is a weak single-seed prior, not evidence.

**Proposed test.** A paired A/B backtest, high vs xhigh arms on the same questions, on
gpt-5.6-sol + gpt-5.5 (the two xhigh OpenAI slots); ~`backtest_medium` scale (2 arms × 32
questions), rough cost $60-90 at recent per-question rates.

**Decision rule when run.** If xhigh ≈ high within noise, drop those slots to high (saves a large
fraction of the ~64%); if xhigh wins meaningfully, keep and consider extending xhigh to more slots.

### ~~Harden `BoundSafeNumericDistribution.cdf` fallback for coarse grids~~ — DONE 2026-07-20 (added 2026-07-20)

Landed with the 2026-07-20 discrete-hardening pass. `BoundSafeNumericDistribution.cdf`
(`numeric/pchip_processing.py`) now computes `grid_step_constraints(len(base))` and threads the
grid-scaled min/max step into `safe_cdf_bounds`, so the fallback matches the pipeline's resample
path on a coarse discrete grid instead of clipping every bin to the 201-grid `max_step=0.2`.
Regression test: `tests/test_thirteen_percentile_e2e.py::TestFallbackCdfRespectsOpenBounds::test_fallback_coarse_grid_uses_grid_scaled_constraints`.

### ~~Bundle section-content audit before any content cuts~~ — DONE 2026-07-18 (added 2026-07-17)

Operator directive: no willy-nilly trimming; a Fable-judged per-section value/redundancy audit is the
prerequisite for any cut. Token measurements `scratch/bundle_token_audit_2026-07-17/`; audit ran
`scratch/bundle_content_audit_2026-07-17/RESULTS.md`. Findings + follow-ups in the Near-term entry
"Bundle content-audit findings" (v1 retirement gate, AskNews reform, market header).

### Research-output audit: temporal/provenance error sweep (added 2026-07-08, low priority)

Motivated by the qid 42304 INES miss: the then-native-search provider (`x-ai/grok-4.1-fast`, retired)
cited an undated NucNet archive article from 1 Feb 1999 (the 1998–99 Istanbul INES-3 accident) as a
"February 2026" Turkish event with a fabricated "reported March 1, 2026" date. All five forecasters
anchored on it (81% published; resolved NO; peer −115.9); the same phantom reached ≥2 top-competitor
stacks — a field-wide hazard of undated archive URLs, not a one-off. Idea: a free offline audit over
`backtests/research_archive/latest/` — sample research per provider, spot-check high-leverage claims
(dates, event existence, numbers) against cited URLs, classify error modes (temporal displacement,
fabricated dates, summarizer certainty inflation). Low priority (offending provider gone; prompt-side
date-stamping + single-source flagging are the nearer lever) but worth doing before any new provider swap.

Related dependency follow-up (from the 2026-06-01 desloppify pass): **raise version floors to
current-installed** (`litellm ^1.80` vs `^1.59.1`, `openai` to latest, evaluate moving `forecasting-tools`
off the pinned `0.2.54`). Forecast-affecting — **gate on a `make backtest_medium` before/after, don't bump
blind**. (The poetry→uv migration follow-up shipped 2026-06.)

### Gemini grounding via OpenRouter — currently NOT supported (added 2026-05-17)

Goal would be: route Gemini Google-Search-grounded calls (currently in `metaculus_bot/gemini_search_provider.py` via direct `google-genai` SDK + `GOOGLE_API_KEY`) through OpenRouter so the donated Metaculus credits cover them, freeing up personal Google API budget.

**Status as of 2026-05-17**: NOT supported. OpenRouter's web plugin and `:online` suffix expose native search ONLY for Anthropic / OpenAI / Perplexity / xAI. Gemini falls back to **Exa** (verified HIGH confidence: <https://openrouter.ai/docs/guides/features/plugins/web-search>). Migrating today would silently swap Google's grounded retrieval for Exa text-search — quality regression, not just cost optimization.

**Recheck periodically**: <https://openrouter.ai/changes> — if/when OpenRouter announces native Google grounding (or a passthrough for `tools=[{"google_search":{}}]`), revisit this migration. Until then, no action.

### Update analysis-CLI defaults to summer-futureeval-2026 (added 2026-05-17)

Tournament rolled spring→summer 2026-05-17; live `TOURNAMENT_ID` updated but **three CLI defaults stay
pinned to spring** intentionally (`ablation/cli.py:95`, `performance_analysis/collector.py:30`,
`performance_analysis/cli.py:17`) so analysis defaults to the resolved dataset, not the freshly-opened one.
**Flip when** summer has ~30+ resolved Qs (mid-July 2026); also update the stale slug example in
`tests/test_tournament_dates.py:127,131`.

### Mixture model parameterization for numeric questions — largely rejected

Ask LLMs to parameterize a mixture (2-3 components: mean, std, weight) instead of percentiles, for
smoother CDFs (Mantic reports good results). **Note:** a mixture path
(`NumericStructured.mixture_components` + router branch) was built and REMOVED 2026-07-08 after zero
prod fires — percentiles+PCHIP beat it in every benchmark (`mixtures.py` library preserved but dormant).
Re-proposing the LLM-parameterized-mixture form has to clear that bar.

### Aggregation strategy improvements

**Status 2026-05-29: STACKER DISABLED ON ALL TYPES (default off in code).** Numeric was already off
(ablation median > stack CRPS, p=0.042); binary + MC now off too (binary was a *tie*, p=0.496 — low-risk
default given compute cost, not a measured harm; binary/MC treatment effect unmeasured on the current
stack). Revisit when post-2026-04-27 marker-era resolved questions exist.

Item status (from analysis; prompt changes address the bigger issues):

- **Trimmed mean** (drop hi+lo, mean of middle 4 of 6) — untested, on backlog.
- ~~Post-aggregation shrinkage toward 50%~~ — **KILLED 2026-05-10** (May data didn't replicate the NO-bias
  at n=109; shrinkage costs well-calibrated extremes).
- **Spread-aware aggregation** — **SHIPPED as CONDITIONAL_STACKING** (April 2026; prob-range trigger
  justified by May ρ=0.616 disagreement-error).
- **Per-type weighting by historical performance** — LOW-PRIORITY, deferred to Q3+ (only gemini-3.1-pro fit
  the binary-vs-numeric asymmetry; revisit only when ≥2 active models show it on ≥100 binary AND ≥30 numeric).

### Per-model peer ranking: GPT-strong / Claude-weak on binary (2026-05-29, NEEDS BENCHMARK before acting)

Peer-equivalent recompute (`scratch/residual_2026-05-29/dim_peer_recompute.md`, validated exact vs
`spot_baseline_score`), binary spring-aib-2026 n≈150:

- **GPT carries the binary ensemble** (gpt-5.1 +19, gpt-5.2 +17 peer, CIs exclude 0); the **Claude pair
  is the binary drag** (opus-4.6 −9, opus-4.5 +2.2; confound runs the wrong way, so genuine).
- **INVERTS on numeric** — Claude strong (opus-4.6 +24), gpt-5.1 weakest — so binary-specific, not
  "Claude is bad"; a blanket cut hurts numeric.
- Counterfactuals (paired, in-sample): drop Claude pair → +5.94 binary peer [+1.0, +11.7]; drop opus-4.6
  alone → +3.66 (survives jackknife, most robust). GPT-only +10.6 hinges on 1-2 questions — don't quote clean.
- gpt-5.2 is NOT high-variance (lowest binary sd); wildcards are gemini-3.1-pro (sd 96), opus-4.6 (sd 80).

**Do NOT act without an intense OOS benchmark** — in-sample (n=62 paired), multiple-comparisons exposed,
epoch-confounded, roster already rotated. Credible reading: on binary the Claude pair is a measurable drag
and GPTs carry the ensemble. Next step: a prospective per-type model-inclusion benchmark (GPT-heavy binary,
retain Claude numeric), gated on OOS peer. (The stacker's own model opus-4.5 being a weak base binary
forecaster is part of why disabling the binary stacker is low-risk.)

### Domain-aware CDF spread tuning

**Status 2026-05-29: HOLD — measured on a STALE pipeline; re-measure on current version before ANY narrowing.**
Residual analysis (`scratch/residual_2026-05-29/dim_numericpit.md`, `numeric_width_version_confound.md`)
re-confirmed across two rosters that the *analyzed* CDFs are too wide (PIT std 0.24–0.26 vs ideal 0.289; 90%
coverage 92–98%) and a uniform k≈1.2–1.3 contraction would hit the ideal. BUT all analyzed forecasts (≤2026-04-13)
ran tail-widening at full strength (k_tail=1.25); prod flipped to k_tail=1.0 on 2026-05-12 (`b8d730f`) *after* the
data, so current prod is already narrower. **Applying k≈1.3 now would overcorrect** — over-width is in the *body*
while deep tails are already too thin (1.5–6% mass vs 10% ideal).

Before anything here: (1) PIT log-grid measurement-bug fix in `analysis.py::_interpolate_pit` (mis-scores
log/`zero_point` questions by up to 0.86 PIT) — since shipped; (2) add PIT std as a first-class `backtest.py`
metric; (3) `make backtest_large` for a current-version PIT measurement; (4) only then, if still over-wide, a
smaller narrowing factor with a per-side tail-mass floor. Direction (mild body over-width) may survive; magnitude
k=1.3 won't. Note: the 2026 finance carve-out **inverts** the old advice — financial questions were the *most*
over-wide (cov90=100%), so "exclude finance/markets" is wrong on current data. (Retire the old "PIT std 0.143"
figure — it was the April n=11 ~2nd-percentile draw; population ~0.24–0.26.) The older forecastability-conditional
framing is superseded by this version-confound note.

### Numeric-width history — receipts + ongoing monitor (added 2026-07-17)

Consolidated, git-verified history of numeric-width config, so future sessions never re-litigate from memory.
Short version: tails deliberately widened 2025-09 → 2026-05, turned off 2026-05-12, and the 2026-07-17 TS-anchor
clause now pushes back toward sharpening.

| Date | Change | Value before → after | Commit | Source |
|---|---|---|---|---|
| 2025-08-21 | Numeric prompt: earliest "widen" language added | (none) → "aim to produce somewhat wider and less confident predictions" | `4bd8685` | `prompts.py` |
| 2025-09-06 | Numeric prompt: widen language strengthened | "somewhat wider" → "wider and less confident … penalties for narrow intervals are severe" | `0437f3e` | `prompts.py` |
| 2025-09-07 | **Tail-widening machinery introduced** — `widen_declared_percentiles` + constants born | function default `k_tail=1.25`, `span_floor_gamma=1.0`; `TAIL_WIDEN_K_TAIL=1.25`, `TAIL_WIDEN_SPAN_FLOOR_GAMMA=1.0` | `4c6481b` | `tail_widening.py` + `numeric_config.py` |
| 2026-03-30 | Numeric prompt: blanket-widen replaced by forecastability-conditional wording | "aim to produce wider … penalties for narrow intervals severe" → "produce wide, diffuse [for volatile] … anchor tightly [for stable] … penalties for overly wide intervals on predictable quantities also accumulate" | `3217aab` | `prompts.py` |
| 2026-04-27 | Numeric prompt: added "Hedge audit" anti-over-widening clause | (none) → "Only widen tails when you can name specific evidence creating that uncertainty, not because it feels safer" | `95c4fff` | `prompts.py` |
| **2026-05-12** | **`k_tail` 1.25 → 1.0 and `span_floor_gamma` 1.0 → 0.0** (function defaults AND both constants), plus `ValueError` guard on `k_tail<1` / negative params | `k_tail: 1.25 → 1.0`; `span_floor_gamma: 1.0 → 0.0` | `b8d730f` | `tail_widening.py` + `numeric_config.py` + `constants.py` |
| 2026-06-01 | Subpackage move (no value change): `tail_widening.py`/`numeric_config.py` → `numeric/` | rename only | `78c5182` | subpackage extraction |
| 2026-07-09 | Numeric prompt de-overfit prune; **kept** hedge-audit | large prune | `5c6640a` | `prompts.py` |
| 2026-07-17 | Numeric prompt: **TS-anchor clause** reframes widening as the failure mode ("SHARPEN, not another license to widen"; "cov@10 ≈ 0.03 vs 0.10 target") | (none) → anchor clause | `3a7ba7d` | `prompts.py` |
| 2026-07-17 | Numeric prompt: fixed the surviving blanket-widen contradiction — Phase-8 "keep tails far apart for unknown unknowns" scoped to *nameable* unknowns, not padded from generic caution or beyond a calibrated anchor's band | unconditional → conditional | (this branch) | `prompts.py` |

Notes on the story:

- **`k_tail`/`span_floor_gamma` have a two-point history, no intermediate values**: born 1.25/1.0 (`4c6481b`,
  2025-09-07), flipped 1.0/0.0 (`b8d730f`, 2026-05-12); function default and constant always changed together.
  Current: `tail_widening.py` 1.0/0.0, `numeric/config.py` `TAIL_WIDEN_K_TAIL=1.0`, `TAIL_WIDEN_SPAN_FLOOR_GAMMA=0.0`.
- **No constant literally named "PIT."** The "PIT widening 1.25 → 1.0" memory = the `TAIL_WIDEN_K_TAIL` flip; PIT is
  the calibration metric (`performance_analysis/analysis.py`) that drove it, not a separate knob.
- **The 2026-05-12 study** (`scratch_docs_and_planning/tail_widening_empirical_calibration.md` +
  `scratch/tail_widening_calibration_2026-05-12/`, 43 resolved numerics Feb–May): drop `k_tail` to 1.0 (every
  segment minimized |PIT std − 0.289| there; 1.25 moved PIT *away* from ideal); `span_floor_gamma` a no-op (→0.0);
  no per-category `k_tail` (CIs overlap); narrowing (`k_tail<1`) is a silent no-op (motivated the `ValueError`
  guard). Caveat: "revisit after ~150 resolved numerics."
- Prompt-language and `k_tail`-code changes are **decoupled** (separate commits).

**Ongoing monitor: `metaculus_bot/performance_analysis/width_monitor.py`** (read-only, free — not cost-gated).
Run: `uv run python -m metaculus_bot.performance_analysis.width_monitor --cached scratch/coherence_2026-07-15/perf_all_tagged.json`
(or `--tournament <slug>` for a live read-only pull; `--output-json <path>` to persist). Per config era it reports
central-80% / central-50% coverage with Jeffreys CIs, cov@10/50/90, PIT std (uniform ideal 0.289 — below ⇒ too
wide, above ⇒ too narrow), and median relative band width `(P90−P10)/|P50|` as the raw sharpness metric. Eras are
the two width-relevant flips: `WIDENING_FLIP` (2026-05-12, k_tail 1.25→1.0) and `TS_ANCHOR_ENABLE` (2026-07-17); the
`ts_anchor` bucket is intentionally empty until the anchor provider is flipped on in prod, so post-enable records
land in their own era instead of contaminating the widening-off baseline.

**Measured 2026-07-17 on the 231 recovered numeric+discrete questions** (`scratch/coherence_2026-07-15/perf_all_tagged.json`):

| era | n | cov80 [95% CI] | cov50 | cov@10 | PIT std | med rel width |
|---|---|---|---|---|---|---|
| widening_on (k_tail=1.25) | 197 | 0.851 [0.798, 0.897] | 0.558 | 0.096 | 0.267 | 0.674 |
| widening_off (k_tail=1.0) | 24 | 0.740 [0.555, 0.888] | 0.580 | 0.083 | 0.286 | 0.561 |
| all | 231 | 0.847 [0.798, 0.890] | 0.567 | 0.091 | 0.267 | 0.647 |

Reading: widening-off moved PIT std 0.267 → 0.286 (toward the 0.289 ideal), cov80 0.851 → 0.740, median rel width
0.674 → 0.561 — consistent with the 2026-05-12 study; no longer over-wide in the body. `widening_off` n is only 24
(loose CIs); baseline to watch as the TS-anchor clause lands (forward risk is over-sharpening). `cov@10` gap: even
widening-off, only ~8% of resolutions fall below published P10 (vs 10% target), so the low tail runs slightly wide —
what the anchor's better-calibrated P10 pulls in.

### Width post-ship watch + monitor attribution tagging (added 2026-07-18, medium)

Two follow-ups from the 2026-07-18 full-branch width audit (`scratch/width_audit_2026-07-18/synthesis.md`).
The audit found no era-stable width bias and motivated Option B, shipped `f4f7984`: delete the Step-7
hedge-audit narrowing push and Step-9b's LOW→wide IQR prescription from the numeric prompt.

1. **Post-ship over-sharpening watch (load-bearing gate).** Option B removed a one-directional narrowing
   instruction, so forward risk flips over-wide → over-sharp. Over the next ~15 numeric resolutions on the
   `width_monitor`: if `cov80` climbs back toward ~0.88+ WITH PIT std below ~0.25 (the pre-flip over-wide
   signature), re-add the hedge audit — but as a SYMMETRIC clause ("match width to reasoning; don't pad OR
   sharpen from disposition"), never the one-directional form just cut. Standing gate: n≥25 with theme
   diversity before fitting any width knob (the "~150 numerics" caveat governs a fitted change). Do NOT cite
   `cov@10 ≈ 0.03` as a too-wide signal here — that's a pooling artifact; the current-era value is 0.107 (on
   target). The ~0.03 the TS-anchor prompt clause quotes is the live-prod low tail, a different cohort.
2. **Tag monitor records with `anchor_present` / `gap_fill_v2_present` (low effort).** The `ts_anchor` era
   bucket is confounded (TS anchor + gap-fill v2 + native-search/crux terra swaps all flipped 2026-07-17) AND
   pools treated with untreated (only ~53% of numerics route to a fetchable series). Fix: at collection time
   grep research text for the `## Time Series Anchor` and `## Agentic Research Findings` headers, thread the
   booleans into the record, split era rows by presence. Cheap; unblocks any real anchor-effect read.

### Ideas reverse-engineered from high-scoring competitor bots (added 2026-06-26)

Source: dissection of 12 high-scoring outputs from GreeneiBot2 / Preseen-Atlas / SynapseSeer
(`/Users/flatljan/Documents/prompts/metac-examples-strong-bots-june-2026.md`; report + grounding-critic
verdict `scratch/competitor_analysis_2026-06-26/REPORT.md`). **Caveat: the corpus has NO resolution
outcomes**, so every "why it helps" is a mechanism argument, not outcome-validated. Only the
source-provenance trust ladder shipped (`prompts.py:_SOURCE_PROVENANCE_LADDER`); the rest is gated on a
benchmark.

**1. Stacker deviation cap from the base-model median (experiment; needs benchmark).** Bound stacker
output to within K of the base median (binary in prob points — ABSOLUTE, not their unstable-near-0/1
multiplicative "±20% of average"; numeric as a percentile/location shift), in
`aggregation_pipeline.py::_stacking_aggregate` just before `_apply_platt_calibration`, new caps in
`constants.py`. *Grounding:* GreeneiBot2 caps consolidated updates to ±20% of the panel average (line 1352).
*Why:* a cap might let us safely re-enable the numeric stacker MEDIAN beat (CRPS p=0.042). *Caution:* too
tight defeats the point of stacking — ablate, don't assume. **Gate:** `make backtest_medium` capped-stacker
vs MEDIAN-default.

**2. Shared-reliance / consensus-fragility audit (experiment; higher-risk, needs benchmark).** The hole:
`compute_spread` takes MEDIAN when the N agree and never asks WHY — if all 6 swallowed the same unverified
shared-research fact, that consensus is falsely confident and invisible to spread. The missing *mirror* of
the disagreement branch. Three operationalizations, cheap→expensive: (A) **prompt-only self-report** — add a
`load_bearing_claims: [{claim, verified, source}]` field to the structured JSON; surfaces shared reliance,
zero aggregation change, low-risk (could ship in a prompt session); (B) **deterministic cross-forecaster
flag** — one unverified claim in ≥K of N → mark fragile, FORCE the crux/stacker path even at low spread (the
stacker prompt `prompts.py:725` already has the language, just never runs on agreement; hard part is
clustering free-text claims); (C) **dedicated consensus-auditor LLM** on the low-spread branch. *Grounding:*
OUR idea, seeded not copied — competitors discount overlapping-rationale agreement (SpaceX line 2028, Iran
line 1330) and notice shared unverified figures (Metaculus-predictions lines 36/43) but respond by
downweighting+widening, NEVER by gating aggregation. *Caution:* (B)/(C) add cost to the COMMON cheap MEDIAN
path — the main reason they're deferred. **Gate:** benchmark (B)/(C); (A) is the low-risk start.

**3. Numeric "unverified-conflict → variance" rider (low priority; tension with our calibration).** When
the trust ladder can't adjudicate two candidate values for a load-bearing quantity, place mass across both
(widen the relevant percentiles) with a materiality gate. *Grounding:* Metaculus-predictions fact-checkers
facing 3,856,697 vs 3,895,701 widened rather than picked ("sigma 95000 to cover the discrepancy", lines
36/43). *Tension:* our CDFs are ALREADY too wide (`TAIL_WIDEN_K_TAIL=1.0`) and the hedge-audit penalizes
caution-widening; the defense is this widens for a NAMED, quantified reason (satisfies the hedge-audit
carve-out) but risks over-application. Decided 2026-06-26 NOT to ship standalone. Revisit only as a
tightly-scoped numeric-only A/B; **gate** on P10/P90 coverage not regressing.

**Also observed, NOT pursued (don't re-litigate):** binary-complement coherence guard (Yes+No=100%,
candidate for a future prompt session, report rec #4); interrogate-resolution-source-quality clause
(Preseen-Atlas line 2679 — partly covered by the trust ladder); numeric partial-resolution incorporation +
reporting-vs-outcome-uncertainty (report rec #5). **Explicitly rejected as conflicting with verified data
(do NOT re-recommend):** GreeneiBot2's one-sided anti-overprediction shave (our binary slope flips 0.83→1.66
across rounds, so a fixed shave helps one and hurts the next); blanket sigma-widening (CDFs already too
wide); parametric mean/sigma numeric representation (our percentile→PCHIP→CDF-space pipeline subsumes it);
the open-tail "spike" grid-compliance trick (we solve grid validity deterministically in `pchip_cdf.py`).

### ~~Summarizer model: bench sol-low vs terra-low~~ — DECIDED 2026-07-18 (switched to terra-low)

The 2026-07-17 role audit kept sol (best synthesis/provenance, terra 2nd with one attribution blur, gap
"MARGINAL EDGE"), but an operator value-call 2026-07-18 **switched to terra-low**: AskNews is auxiliary
(16% unique content vs native-search 54% / gap-fill 59%) so the frontier tier isn't warranted, and the
AskNews quality audit (`scratch/asknews_quality_audit_2026-07-18/`) blamed 4/5 briefing failures on
prompt-era issues not model tier. Terra: −43% cost, ~50s vs ~118s. Packets:
`scratch/research_role_audit_2026-07-17/`.

## Longer-term (significant R&D)

### Agentic deep research (ReAct loop) — SHIPPED 2026-07-17

The gap-fill v2 plan (near-term entry; `scratch_docs_and_planning/agentic_gap_fill_v2_plan.md`) IS this —
a bounded tool-loop second pass, ON in prod (`GAP_FILL_V2_ENABLED: 'true'`). The old cost blocker is
resolved by budget caps (~$0.50/q) + early-stop; selective activation via the template dry-run. Direction
confirmed by the 2026-07-16 lit survey: keep it a **shared** stage (one loop → detached artifact all
forecasters read), NOT per-forecaster integrated pipelines (BTF-2 arXiv 2604.26106 — strong prompt on
shared research edged the best integrated agent; integrated benefit was Opus-class-only). Watch
era-bucketed calibration alongside Brier (agentic research trades calibration for over-decisiveness).

### Prediction market integration — SHIPPED (strong evidence, criteria/date-matched)

Structured Polymarket/Kalshi/Manifold/PredictIt access shipped as the prediction-market snapshot
(`PREDICTION_MARKETS_ENABLED: 'true'` in prod). Framing: markets are STRONG EVIDENCE, not a footnote. The
forecaster prompts anchor on a market whose criteria + date MATCH, discount proportionally to a specific
mismatch, and when only the DATE differs, extrapolate the market's probability to our date with a simple
model (constant-hazard / base-rate-over-time) rather than a vague haircut. Superseded the old "not beholden
to them" language after a referendum miss where the bot dismissed a market sitting at the correct answer.

## Killed by May 2026-05 closing analysis

These were investigated and either failed to replicate or were superseded. Listed
here so future sessions don't re-recommend them without new data.

- **Drop or down-weight gemini-3.1-pro (binary axis)** — April rec #6 did not
  replicate at May n=62 (+0.024 vs ensemble, below threshold; net contributor on
  hits cohort 5 best vs 4 worst). Numeric weakness exists (rank 7/8) but is solved
  by per-type weights (deferred), not removal.
- **AI-capability "private preview is a leading indicator" prompt edit** —
  April rec #7 was N=1 (Anthropic Opus miss 43131); May still N=1 (same question).
  Don't ship a prompt edit on a single observation.
- **Time-of-tournament Brier rolling-avg analysis** — confounded by mid-tournament
  roster swaps (gpt-5.2 → 5.4 → 5.5; opus-4.5 → 4.6 → 4.7; gemini-3-pro → 3.1-pro).
  Cannot isolate bot drift from roster changes.
- **`nr_forecasters` difficulty stratification** — peer score already normalizes for
  difficulty better than `nr_forecasters` (which is more correlated with question
  attention/age than difficulty).
- **3D calibration grid (predicted-bucket × type × stage)** — ~132 cells with
  ~0.7 questions average. Noise dominates. Stick with the 1D predicted-bucket cut.
- **MC per-model audit** — per-model MC predictions don't survive in stored
  comments. Would require collector changes for n=24/round. Not worth it.

## Killed by July 2026-07-08 residual + competitor analysis

- **YES-side shrink / any fitted directional calibration layer** — the YES-overconfidence finding is
  real but era-local (spring-only; fall confident-YES ~100% accurate). All three stability criteria failed;
  a two-era fit degraded held-out fall +4.3e-3 Brier. Surviving hard rule: symmetric shrink was strictly
  worse in every era — never touch the NO side. `scratch/residual_2026-07-08/experiments/ARM_B_FINDINGS.md`.
- **Anchor-guard as a clamp/gate** — models publish outside their own base-rate anchor on ~88% of forecasts
  (normal outside→inside update); "flag it" gate precision ≤29% and sign flips across eras. Surviving
  fragment: overshoot MAGNITUDE >15pp degrades Brier monotonically in both well-powered eras → ship as
  telemetry + prompt nudge only (telemetry shipped 2026-07-08). `.../ARM_A_FINDINGS.md`.
- **Advisory (non-binding) critic passes** — GreeneiBot2's critique diagnosed our exact 41800 defect and the
  number never moved; laertes's correct 42855 recomputation was half-overridden. BINDING corrections
  (formula/floor/structural) captured 30–130 spot-peer points, advisory ~0. The critic-pass entry stays but
  binding-output only. `scratch/residual_2026-07-08/ACID_TEST_VERDICT.md`.
- **Fixed-direction haircuts** — GreeneiBot2's always-downward haircut helped 42855 but harmed 41800
  (extremized 12→10 on a YES). Any damping must push toward 0.5, not a fixed direction (mirrors the killed
  one-sided anti-overprediction shave).
- **"Median drowns the correct dissenter"** (re-confirmed dead) — 7/9 washouts framing is survivorship bias;
  median stays at the non-oracle frontier across two more pulls; even laertes's 41800 "win" discarded its own
  best member and was saved by its floor, not aggregation.
- **Anchor-floor guard on cheap tails** — median-band variant sign-flips (fall +1.68 / spring −2.19 total
  Brier at 10pp) because 76% of parsed anchor bands are degenerate points; union-band variant is
  sign-consistent but 100% top-5-concentrated (1–2 Qs/era) and catches 1/5 of known misses; same-side tail
  clamping hurts both well-powered eras. Revival: ≥50 current-roster binaries with `base_rate_anchor`
  telemetry (30bca2f) AND an era-stable, top-5<50% replay. `.../GUARD1_FINDINGS.md`.
- **No-market-no-extremize cap** — feasibility kill: market-presence signal exists in ~one era (fall 89%
  NO_SIGNAL, `## Prediction Market Snapshot` in exactly 1 archived binary), so era-stability un-testable;
  marketless fallback = Arm A's point-anchor drift bomb; strict form's gain is 3 spring questions and the
  gate itself adds ~nothing. Revival: ≥~50 binaries/era carrying the structured snapshot AND the market gate
  earning the delta. `.../GUARD2_FINDINGS.md`.
- **Signed deadzone haircut toward 0.5** (closed permanently) — extends the Arm B symmetric-shrink kill to
  thresholded/high-t forms: 0 of 24 (λ,t) cells help in both well-powered eras; fire rate ≥30% even at t=0.4
  (confident bot, median |p−0.5|≈0.32–0.38); LOTO on spring+summer degrades fall +1.12 Brier. Keeper: the
  signed-toward-0.5 direction constraint is a design rule. Revival: calibration profile qualitatively changes
  AND leave-one-era-out passes everywhere; don't re-grid λ/t absent that. `.../GUARD3_FINDINGS.md`.
- **Cross-cutting** — at mid-grid all three guards fire on the same spring miss cluster (42024 / 42304 /
  41800; often 42018 / 42577 / 42855 / 41644) — one lever measured three ways. The shipped `30bca2f` prompt
  changes target that cluster, so any future guard replay must never fit on pre-`30bca2f` eras (the
  roster-drift bomb with a prompt-change fuse; config-era bucketing already handles this).

## Killed / evaluated 2026-07-11 (blind-forecaster review + crowd-signal audit)

Context: blind judge pass over the 2026-07-11 `test_bot` run (`scratch/test_bot_july_11_2026 .md`) + a
live-API audit of surfacing crowd-signal informativeness in research packets.

- **Metaculus `similar-posts` endpoint as a research provider** — KILLED. Live audit, 23 source posts / 160
  rows (`GET /api/posts/{id}/similar-posts/`). Two dealbreakers: (1) community-prediction VALUE is `null` for
  our bot account on 160/160 rows; (2) returns ONLY open questions (160/160 unresolved) so the base-rate
  payload is absent. What remains (title + `nr_forecasters` + un-followable link) is decorative; match quality
  bimodal (good AI/US-politics/geo, garbage weather/sports/non-US — a Boston-Marathon source returned "World
  Bog Snorkelling" + a Zambian election), pads to 8 with no relevance score, rate-limits ~10 calls → 429.
- **Resolved-sibling base-rate lookup** (`?search=<kw>&statuses=resolved`) — general version KILLED (~110-call
  audit): **Metaculus null-outs `question.resolution` (+ description/criteria/CP) on every post this bot didn't
  itself forecast** (view-level AIB anti-cheating; no bypass via `with_cp`/`minimize`), so the Metaculus-wide
  resolved corpus is unreadable to our token. Endpoint mechanics work (`statuses=` plural; `actual_resolve_time__lt`
  + `order_by=-scheduled_resolve_time` silently broken). **Salvage (optional, LOW):** a
  `forecaster_id=275109&statuses=resolved` self-history lookup over the bot's own ~770 resolved posts —
  AIB recycles templates so sibling quality is excellent for recurring indicators / event-window binaries,
  naive title-as-query works, and it's backtest-MEASURABLE (date-filter `actual_resolve_time < open_time`, not
  a hard `is_benchmarking` disable). Ranked low — the indicator series are already pulled by the
  resolution-source + financial-data providers; unique value is structured prior-outcomes for repeated
  event-window binaries.
- **Crowd-signal informativeness surfacing** — WORTH DOING; market-liquidity half SHIPPED (snapshot now
  renders thin/decent/deep labels from total volume / OI / `uniqueBettorCount`, which the fetchers previously
  discarded — the old `vol`=24h-volume number was ~always ≈0 for long-horizon questions, misleading).
  `nr_forecasters` is free on the fetched object (`MetaculusQuestion.num_forecasters`, zero extra HTTP).
  Proposed labels: Metaculus <30 thin / 30–49 decent / 50–99 good / ≥100 high; real-money <$5k / $5k–50k /
  >$50k; Manifold bettors <20/20–100/>100. Aggregators SKIPPED: Metaforecast shut down; Adjacent News
  redundant (Kalshi+Polymarket only); PredictIt an optional politics-only price-only add; PMXT only if venue
  breadth becomes binding.
- **Open-bound tail-cramming on discrete/open-upper numeric** — FIXED (see the `OPEN_BOUND_PILING` telemetry
  + nominal-bounds prompt render, CLAUDE.md). gpt-5.6-sol / gpt-5.5 crammed ~20% / ~12.6% of mass onto the top
  displayed bin of Q38195 because "Respect the explicit bounds" read as a hard cap; both OpenAI models failed,
  Claude/Gemini/Grok handled it. Fix: prompt clarification + a WARN-only boundary-piling detector into GHA
  artifacts.

## Instrumentation bugs

**All three from the May 2026 closing analysis are FIXED (2026-05-10; 1187 tests pass).**

1. ~~STACKER_OUTCOME tri-state marker~~ FIXED — producer sets `_stacker_outcome[qid]` (primary /
   fallback_llm / fallback_median / skipped) at the END of each path; `_create_unified_explanation` emits
   the tri-state `STACKER_OUTCOME=` + legacy `STACKED=` markers (median-fallback previously silently emitted
   `STACKED=true`).
2. ~~Targeted-research header missing from comments~~ FIXED — `main.py:839` returned
   `research_report=research` not `combined_research` on the conditional-stacking branch; one-line fix.
3. ~~`audit.py::emit_synthesis` KeyError on numeric-mixed cohorts~~ FIXED — type-aware skip for non-binary
   entries in the spread section (`_rank_numeric` produces no `prob` key).

### New parser feature shipped — historical-header-aware detection

`parse_inferred_stacker_outcome` detects three older stacker-output body signatures (`## Stacker
Meta-Analysis`, `## Meta-Analysis`, `# Meta-Analysis and Synthesis`) plus the tri-state + legacy markers.
Unlocked the May 2026 stacker treatment-effect estimate (`analysis_stacking_historical_treatment.md`) —
first measurable signal (n=8 stacker-ran, −0.090 Brier vs counterfactual, P(helped)=89.8%).
