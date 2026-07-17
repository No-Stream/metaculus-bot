# Future Ideas

Ideas for improving the forecasting bot, roughly ordered by expected impact and feasibility.

> **Status as of 2026-05-10** (closing residual analysis on spring-aib-2026, n=189). See
> `scratch/analysis_2026-05/analysis_synthesis.md` for full per-rec discussion,
> `scratch/analysis_2026-05/extended_hits_misses_postmortem.md` for deeper per-question
> diagnostic, and `scratch/analysis_2026-05/NEXT_SESSION_QUEUE.md` for the prioritized
> implementation backlog. Per-rec status notes are inlined below; some recs were retired
> this round, and the priority order shifted materially based on the deeper post-mortem
> read of 102 per-question audit files.
>
> **Two findings reshaped the priority list:**
>
> 1. **17/20 worst misses are high-spread** (>0.15 disagreement) — at least one base model was
>    closer to truth than the ensemble. The post-mortem labeled this as "good input wrongly
>    weighted" in 14/20, but reading those misses critically, calling the closer model "right
>    for the right reason" and the others "wrongly weighted" is post-hoc — the closer model
>    often just made a different reasonable reference-class choice that happened to win.
>    What's robustly true: models genuinely disagree on which reference class to weight, and
>    the ensemble's averaging often pulls away from the closer-to-truth minority. Whether the
>    stacker can systematically pick the better minority is what the n=8 treatment-effect
>    signal hints at, but at n=8 it's a hint, not proof.
> 2. Stacking treatment effect is now **directionally measurable** at +89.8% bootstrap
>    confidence (n=8 stacker-ran vs n=57 triggered-counterfactual on May binary;
>    `analysis_stacking_historical_treatment.md`). First measurable signal in the project's
>    history; needs the marker fix for definitive measurement.

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

Rigorous era-bucketed Bayesian calibration audit (2026-07-16; scripts in
`scratch/calibration_audit_2026-07-16/`, data = `scratch/coherence_2026-07-15/perf_all_tagged.json`,
694 records). **Conclusions:**

- **The bot is not cleanly over- or under-confident — it flips by era.** Bayesian logistic
  fit (2-D grid quadrature, no MCMC needed): fall-aib-2025 slope **1.70 [1.28, 2.20]**
  (under-confident), spring-aib-2026 slope **0.83 [0.58, 1.10]** (over-confident), summer n=15
  uninformative. The opposite-signed eras cancel to a ~calibrated-looking pooled slope
  (1.17 [0.95, 1.40]) — this is exactly why past pooled calibration reads were contradictory.
  Even *within* spring the early/late temporal split flips slope 0.64 → 1.25 (roster churn
  inside one tournament).
- **The "we get more NO right than YES right" pattern is a base-rate artifact, NOT a
  directional miscalibration.** Tournament YES-rate is 24% (fall) / 33% (spring) / 60%
  (summer n=15); a calibrated forecaster naturally "gets more NO questions right" because more
  resolve NO. The decisive test (observed hit-rate vs expected-if-calibrated, per side,
  Beta-Binomial CIs) is **consistent with calibration on both sides in every era**. Puzzle
  resolved.
- **The killed spring YES-overconfidence finding reproduces as spring-local** (intercept
  −0.40 [−0.80, 0.00], P(a>0)=0.017; high-confidence YES bins under-resolve) **and is
  untestable on the current post-flip roster** (n=15, intercept CI [−1.85, +1.40]). Fall shows
  the opposite sign. Consistent with the scar tissue; nothing to fit.
- **We are chronically data-starved for the miscalibrations a fitted layer would target.**
  Bayesian power sim (type-I correctly ~0.05): detecting mild overconfidence (slope 0.85) has
  power ~0.25 even at **N=400**; moderate (slope 0.80) reaches only ~0.49 at N=400; a
  spring-like directional bias (intercept −0.40) reaches ~0.67 at N=400. A roster era yields
  only ~150–210 resolved binaries before the roster churns (fall 210/96d, spring 167/92d,
  summer 15 so far). So we can catch only *large* miscalibrations, never the subtle ones — and
  any fit spans eras with opposite-signed calibration, i.e. a drift bomb by construction.

**Decision (operator, 2026-07-16): calibration is not a realistic source of improvement given
our budget.** Well-funded entrants can run multi-hundred-to-multi-thousand-question paid
backtests to fit and validate a calibration layer; we cannot, and the power analysis says even
that volume barely detects moderate effects. **Prefer SOTA forecasters over fitted
calibration** — now the evidenced call, not just the instinct. Do NOT ship isotonic / Platt /
directional-shrink layers (see the killed-calibration entries elsewhere in this file; the
Beta-Bernoulli "calibrator" from arXiv 2605.27668 is a mini-forecaster on the wrong shift axis
for our roster-drift trap, and its own tables show Platt/isotonic degrade OOD — cite it as
anti-adoption evidence).

**DO promote the audit to a standing monitoring module (free, high-insight, zero scar risk).**
It is complementary to, not duplicative of, `performance_analysis/analysis.py`: the existing
module gives per-tournament point-estimate calibration buckets, a single `no_bias_check`
`bias_pp` scalar, and PIT coverage. The audit adds (1) automatic multi-era comparison in one
view, (2) Beta-Binomial credible intervals per bucket, (3) the slope/intercept logistic
decomposition (separates confidence from directional bias — the `bias_pp` scalar conflates
them), (4) the YES/NO base-rate-artifact test, (5) the power/data-adequacy sim, (6) partial
pooling across eras. Build it to **reuse** the existing `_interpolate_pit` (has the log-grid
fix) and bucket plumbing, layering the Bayesian/era/power machinery on top. Monitoring cadence:
on each roster change, print era-bucketed slope/intercept credible intervals + reliability
table + the base-rate-artifact check; **act only if a CI excludes the null AND reproduces
across ≥2 eras** — expect "inconclusive" to be the honest default (that's correct, not a
failure). This is the practical payoff and it's free (read-only `performance_analysis` pull).

### ~~Geometric-mean-of-odds base-combine vs MEDIAN~~ — RUN 2026-07-16, DECISIVE NULL (keep MEDIAN)

Background: verified 2026-07-16 that the prior aggregation rejection covered **arithmetic
mean** and geometric pooling for **MC**/**numeric**, but never binary geometric-mean-of-odds
as the scored base-combine (the `log_pool` fn existed but only printed a stacker-prompt display
line). MEDIAN is logit-invariant, so only a mean-type pool can differ from it — so this was a
genuinely open (if headwind-facing) experiment.

**Run same day (offline replay, zero API): NULL everywhere.** A `geo_odds` arm
(`sigmoid(mean(logit(p)))`, reusing `log_pool`, output clamped [0.02,0.98] for parity) was
added to `ablation/offline_replay.py::build_binary_configs` (~:502/:624, +5 tests) and
era-bucketed via the validated comment-recovery dataset (the ablation cache is single-era;
driver in `scratch/geo_odds_2026-07-16/run_geo_odds.py`). Paired per-question deltas, 5000-rep
bootstrap CIs, sign + Wilcoxon tests: **no era's CI excludes zero on either metric** — fall
n=208 logΔ +0.32 [−1.34, +2.04]; spring n=149 +2.14 [−0.52, +5.34] (the only faintly
suggestive cell, Wilcoxon p=0.16); summer n=12 uninformative; pooled +1.07 [−0.40, +2.68].
Brier deltas tiny and *opposite-signed* to log-score (wash tell). The operators genuinely
diverge (~95% of questions differ) — the divergence just doesn't reliably help. On the
full-ensemble (n≥5) filter, fall FLIPS to −0.61 [−2.16, +0.83]: **confirms the documented
"sharpening pools wash out on the diffuse 6-model ensemble" headwind** (the trio-win half was
untestable — no confident-trio binary cohort exists). Cross-check on the harness's own
spring-only cache: +0.66 [−9.33, +11.27], consistent.

**Verdict: keep MEDIAN; geo-odds joins the settled dead paths** (with mean, stacking-as-default,
coherence-weighting). Do not re-run absent a materially different ensemble regime (e.g. a
confident small-N roster) or post-flip data showing a qualitatively different disagreement
structure. The arm + tests stay in the harness for cheap future re-runs. Lint/typecheck/422
tests green. Advisory: the +20 lines tipped `offline_replay.py` over the 1000-line
monolithic-file threshold — split is a separate refactor if it bothers anyone.

### Backtest statistical hardening + leak-aware replay (added 2026-07-16)

Two independent findings (papers-skeptic + eval-repos) plus a codebase check converge here.

**(a) Statistical rigor — AUDITED 2026-07-16, now a scoped consolidation program.** A full
methodology audit (per-surface scorecard, same-day) confirmed the split: the recent scratch
analyses are strong-to-gold-standard (coherence phase2: two-way FE + cluster-robust SEs by qid;
`ensemble_screen`: paired-vs-replica deltas + era-bucketing; calibration audit: full Bayesian
posteriors + power sim), but the **standing, always-on pipeline is the thinnest machinery in
the repo** — and it is weak in exactly the two ways the house rubric names as mandatory
(pairing, era-bucketing). **Operator directive: promote the scratch machinery into the standing
pipeline — no point building elegant scratch stats and relying on lower-quality code for the
repeated analyses.** Ranked gaps from the audit:

1. **`backtest.py` never pairs** (highest leverage): it runs multiple bots on identical
   questions, persists per-qid scores (`backtest/analysis.py:205`), then reports pooled means +
   population SD with zero inferential content (`analysis.py:43-58,104-118`). Fix: join arms on
   qid, per-qid deltas, and **import the inference primitives that already exist in-repo** —
   `ablation/scoring.py` `bootstrap_mean_ci`/`bootstrap_median_ci`/`sign_test`/
   `wilcoxon_signed_rank`.
2. **`performance_analysis/analysis.py` emits pooled, CI-free, era-blind calibration — the
   actively-misleading item**: it still produces exactly the pooled numbers that flipped three
   times (pooled slope ~1 masking opposite-signed eras; base-rate-confounded bias_pp). Fix:
   Beta-Binomial CIs on every bucket (`calibration_audit/binary_calibration.py:53-56` is a
   4-line drop-in) + a first-class era key so the standing report *cannot* emit pooled-only
   calibration. This is where the calibration-monitor promotion (see the calibration entry
   above) lands.
3. **Per-model comparisons unpaired** in both standing surfaces — same fix as (1), free power.
4. **Percentile bootstrap on heavy-tailed log-score deltas under-covers** — ablation Path A
   already hedges (median bootstrap + NoSat panels); `ensemble_screen` doesn't — add a
   median/robust hedge (or Bayesian bootstrap, per operator preference; naive percentile
   bootstrap is NOT one-size-fits-all on skewed CRPS/log-loss).
5. **No clustering/effective-N outside coherence phase2** — correlated question families make
   CIs too narrow everywhere else; mirror phase2's `cluster_bootstrap_mean` where a cheap
   cluster key exists.
6. **No multiplicity correction** in ablation (~15 comparisons) or ensemble_screen (~48) — BH
   q-values or partial-pooling shrinkage (the audit's normal-normal template). Low priority.

**Explicitly NOT broken (don't make work):** the scoring/metric layer
(`backtest/scoring.py` CRPS/log-score implementations — correct, leave alone), `audit.py`
(appropriately non-inferential), ablation Path B's fold-std (self-labeled stability
diagnostic), and plain means on large-n bounded proportions. The problem is never "a mean was
used" — it's "a mean was used as the *comparison* with no pairing, CI, or era split."
Reusable-machinery map: inference primitives `ablation/scoring.py`; era-bucketing template
`ensemble_screen.py:109-140`; Bayesian CI drop-in `calibration_audit/binary_calibration.py:53-56`;
cluster-robust template `coherence_2026-07-15/phase2_lib.py:176-345`.

**(b) Leak-aware replay — the plumbing already exists, the archive quality is the gap.** Verified
(2026-07-16): our default `backtest_{smoke,small,medium,large}` targets **re-run live research
providers at replay time** (leakage-exposed — live search returns post-resolution info; a
leakage *detector* drops contaminated questions but that's a filter, not a fix). A frozen-replay
mode **already exists** (`--research-dir` flag → `make backtest_with_cache` → orchestrator
short-circuits to cached research, skips providers). Code distance to leak-free replay ≈ zero.
**The real gap is archive quality:** of 921 records in `backtests/research_archive/latest/`, only
**19 are genuine GHA-captured provider payloads**; **902 are reconstructed from published
Metaculus comments** (trimmed/summarized, empty `providers_used`), and uncached qids silently
fall back to live fetch. So: (1) flip the default backtest targets to `--research-dir`, and
(2) grow genuine GHA-captured coverage from prod runs so replay isn't mostly comment
reconstructions. This is the cheap, near-term version of RetroSearch (below).

**Framing takeaway (papers-skeptic, arXiv 2506.00723 + the Vaticinus preprint):** the re-run
backtest is an optimistic **upper bound**; the calibration-on-own-published-forecasts pipeline
(genuinely elicited at forecast time, leak-free) is the trustworthy axis. Also: **stop treating
"community Brier − our Brier" as edge-over-market** — it's provably an affine shift of Brier on
a balanced panel (ρ=1.000 across 25 rounds) and encodes zero edge. If any comment/analysis
marker frames "we beat the community by X Brier" that way, it's not an edge claim.

### Forecaster-prompt audit — verified clean, no action on harmful-scaffold grounds (added 2026-07-16)

The preregistered Schoenegger/Tetlock result (arXiv 2506.01578) found two prompt scaffolds
reliably *hurt* forecast accuracy via miscalibrated over-decisiveness: an explicit step-by-step
"Bayesian reasoning" (state prior → sequential likelihood updates → running posterior) and
"propose-evaluate-select". Audit (2026-07-16) of `prompts.py`: **neither harmful form is
present.** The closest thing (binary:394 "My base rate was X%… moving to Y% because…") is
single-narrative anchor-and-adjust — the benign Tetlock outside-view move the study explicitly
carves out, further guarded by the "anchor on your math" clauses that forbid vibe-hedging both
directions. The three base prompts (~11.7k / 7.8k / 12.8k chars) are **focused, not bloated** —
nearly all load-bearing gotchas (the `_forecasting_window_str` "events before open don't count"
guard, status-quo derivation, bait-and-switch check, conjunctive-clause pricing,
open-vs-closed-bound handling, units gotcha, MC must-assign-every-option, stacker dissent
clause). **No action.** One minor follow-up — **RESOLVED 2026-07-16:** the `base_rate_anchor` /
`criteria_clauses` optional JSON fields (binary schema) are NOT dead — they are live-elicited in
the binary prompt (added `30bca2f`, 2026-07-08) and land in every prod comment. The "0/2203
archived rows" finding was a data-window artifact (the archive ended 2026-07-01, one week before
the fields shipped). Two wrong-mechanism claims in `comment/markers.py` and
`scratch/coherence_2026-07-15/synthesis.md` were fixed same-day. The guard-revival program's
presence-rate check works once post-07-08 comments are pulled.

### Market-deference: time-to-close term MEASURED DEAD; liquidity fixes survive, downsized (updated 2026-07-16 same-day)

The applicability gate was run same-day (offline, archive mining; scripts referenced in the
2026-07-16 audit notes) and it kills the headline half of this entry:

- **Time-to-close term: structural null — drop to bottom-of-low.** Of 285 matches with real
  close dates in the provider-ran archive window (n=64 questions, essentially July 2026),
  **0.0% are within 30 days of close** (median 185d, min 45d), and the liquid∧near-close
  intersection is exactly **zero**. This is structural, not small-n: a near-identical market
  closes ≈ when the question resolves, we forecast near open (verified:
  `skip_previously_forecasted_questions=True` in all prod modes, `cli.py:119-148` — the bot
  never re-forecasts late), and the `as_of` leakage filter drops markets closing before
  resolution. TimeSeek's "models lose near close" dynamic is real but never binds on our
  question stream. Revisit only if we ever add late re-forecasting.
- **What survives — three small liquidity/matching fixes (top-of-low / bottom-of-medium):**
  (1) **Fallback-chain bug**: 39 real-money matches render `no-liquidity-data` because
  volume/OI fields are dropped rather than absent — fix the `total_volume`/`open_interest`
  fallback chain (`prediction_market.py:498-506`). (2) **The fuzzy floor (40) is so loose that
  "match" ≈ topical-adjacent**: 100% of provider-ran questions "match" but match confidence
  never exceeds 0.77, and ≥0.7-confidence (the actual near-identical/defer trigger) is only
  ~8% of questions; ≥decent-liquidity matches are ~12%. Consider raising the floor or adding a
  confidence tier so "near-identical" means what the defer policy needs it to mean.
  (3) Optionally a mild deference nudge for deep+high-confidence matches — the plumbing
  (liquidity labels, `_strong_evidence_market_clause` prompt weighting, close dates) already
  exists; this is prompt/render-level, not new fetching.
- Useful mechanics documented by the gate: the `## Prediction Market Snapshot` header only
  renders on ≥1 match (`orchestrator.py:401-403`), so header-absence ≠ provider-off — use
  `providers_attempted` to disambiguate; market close dates ARE already captured for
  Polymarket/Kalshi/Manifold (only PredictIt lacks them).

**Rides along here (low-value general case, per operator 2026-07-16):** a bias-corrected
**∆LL-over-matched-market diagnostic** on the ~8% near-identical-match subset — tells us
whether, on those questions, we add signal beyond the price or should defer harder. NOT worth
building against the Metaculus community prediction: these are bot-only tournament questions
(CP is a pool of mostly-poor bots, a bar we already clear; CP is also null-hidden for our
account). Note the subset is small (~8% × question stream), so this diagnostic accumulates
signal slowly — set expectations accordingly.

### ~~JS-divergence diversity lens on ensemble screening~~ — BUILT 2026-07-16, verdict: NOT a selection signal (paper's heuristic INVERTS under median)

Background: the "Diversity is the Strength of the AI Crowd" paper (arXiv 2606.29661) argues
decorrelation matters more than solo accuracy for roster decisions (their Grok was
least-replaceable despite ranking 3rd solo). We already had the marginal-contribution half
(`ensemble_screen.py`: LOO + leave-one-question-out, era-bucketed, bootstrap CIs); the gap was
a distributional-diversity metric. Prototyped same day:
`scratch/js_divergence_2026-07-16/js_divergence.py` (imports ensemble_screen's loaders /
era-bucketing / `member_cdf` wholesale; JS in bits over Bernoulli / option-simplex / 202-bin
PMFs incl. out-of-bounds tail mass; results in `js_results.json`).

**Findings:**

- **The metric is internally valid** — it measures real error-decorrelation: Spearman(pairwise
  JS, pairwise per-question score-correlation) is negative and significant wherever n is real
  (pooled −0.67, p≈1e-9, n=65 pairs). High-JS pairs genuinely make independent errors, and JS
  is not redundant with solo score.
- **But under our MEDIAN aggregation, decorrelation is INVERSELY related to marginal
  contribution** — Spearman(decorr, removal-drop) = **+0.83** in the only era with enough
  families to rank (fall_6m): the most-decorrelated members (kimi, grok) are the *least*
  load-bearing; the consensus-hugging accurate models (gpt-5, o3, anthropic) carry the
  ensemble. **Opposite of the paper's operational conclusion.**
- **Grok puzzle reconciled mechanistically:** our grok has exactly the paper's profile (worst
  solo, most decorrelated) but its independence is *incompetent* independence — off in a
  different direction, not right-when-others-wrong — and MEDIAN discards outliers rather than
  harvesting them (the paper's combiner was learned, not median). The "keep the decorrelated
  underdog" heuristic is an artifact of their aggregator; importing it under median would
  protect exactly the deadweight slot the marginal benchmark correctly flags as replaceable.
- **One genuinely additive use — redundancy detection:** the pairwise JS matrix cleanly
  surfaces near-clone slots the LOO benchmark only sees indirectly: spring
  `opus-4.5|opus-4.6 = 0.017`, summer `gpt-5.4|gpt-5.5 = 0.018` bits. An "are we double-paying
  for two clones" check when composing rosters.

**Verdict: do NOT promote JS as a roster-selection signal; the marginal-contribution benchmark
stays the decision instrument.** Keep the scratch code for two minor uses: (a) redundancy/
near-clone detection on candidate rosters, (b) the error-complementarity Spearman as a validity
sanity check. Caveats: screens predecessor lineages (current slots have zero resolved; fable
absent entirely); the inversion is median-specific (a learned combiner or stacker could exploit
decorrelation — relevant only if the stacker-revisit ever lands); family n is 3–6/era so only
fall_6m has ranking resolution.

### Revisit the conditional stacker with the AIA supervisor evidence (added 2026-07-16, medium — larger item)

papers-ensemble found our disagreement → targeted-search → stacker path **is** the AIA
Forecaster's single biggest aggregation lever (arXiv 2511.07678: agentic supervisor 0.1125 vs
no-supervisor 0.1199 ≈ 0.0074 Brier, *larger* than single→mean-of-10 at ~0.0042), and AIA warns
hard against "best-of / pick-the-best-model" selection (structurally can't beat its best input;
picks among the worst 7.2% of the time). We run our stacker **disabled in prod** because our own
benchmark found it counterproductive on the current ensemble. This is a reason to eventually
revisit an **era-bucketed median-vs-conditional-stacker head-to-head on the disagreement subset
specifically** — distinct from the learned stacking we rejected. **Not viable now** (our
benchmark evidence says it hurts on the current ensemble; needs post-flip marker-era resolved
data to re-measure the real treatment effect). Larger item to revisit, not a near-term flip.
(Operator note: the stacker is what originally motivated joining the tournament; empirically it
just hasn't worked yet.)

### Time-series anchor for numeric questions — GATE PASSED 2026-07-16 (53% applicability, ~5x the bar): promote to a real build item

For numeric questions that resolve on a fetchable series, fit a cheap probabilistic TS model
(`statsforecast` AutoARIMA/ETS/Theta — light dep, no GPU) and render a model-implied
P10/P50/P90 quantile block in the briefing — same shape as the prediction-market snapshot.
**TS-as-anchor, not TS-as-answer.** A principled version of the "status-quo / last-print
anchor" finding. Do NOT take TimeCopilot as a dependency; only `statsforecast`.

**Applicability gate RUN same-day (offline classification of all 231 numeric+discrete recovered
questions; auditable per-question labels in `scratch/ts_anchor_gate_2026-07-16/ts_labeled.json`):
53.2% (123/231) map to a standard fetchable series** — fall 57%, spring 41%, summer 75% —
vs the ~10% skip bar; even the strictest level-anchors-only reading is 27%. Class A is
dominated by recurring templates (10Y yield, HY OAS, VIX, index/commodity returns, gasoline,
unemployment, CPI, payrolls, approval/generic-ballot averages, TSA volume), so it's
representative of forward mix. **Applicability is no longer the question; design is:**

- **Strong-value core = the 63 level anchors** (macro prints, rates, spreads, poll averages,
  TSA) — where a fitted quantile band directly disciplines the documented over-reasoning
  failure. Spot-check of the three canonical cases: q43611/q43591 (poll averages) were off in
  exactly the direction a live poll-average anchor corrects; q43647 (HY OAS) was
  center-correct and would be sharpened. (n=3, anecdotal.)
- **The 47 relative-return spread questions need the anchor fit on the SPREAD series**
  (center≈0 + historical-vol band), not a naive level forecast — otherwise they add noise.
  The 13 max-functional questions (VIX/commodity highs) need window logic on top of a level
  model.
- **Net-new vs `financial_data`:** the provider's curated allowlist misses HY OAS, gasoline,
  Brent, **VIX**, poll averages, and TSA (4/5 spot-checked FRED series not in the allowlist;
  all exist with decades of history) — and for series it does cover it emits a raw last-6-
  observations table, not a fitted band. Historical "already-served" counts are era-confounded
  (provider effectively off before summer: 0/73 fall, 1/32 spring, 14/18 summer).
- **Class B later-add:** Mauna Loa CO2 + Norwegian EV-share are the most deterministically
  fittable series in the corpus — held out of A only on ingestion; cheap to fold in once
  statsforecast wiring exists.

Was bottom-of-medium pending the gate; **with a 53% hit rate it merits promotion — solid
medium, arguably higher** (operator to confirm). Validation note: prompt-visible research
changes can't be measured by the leakage-exposed backtest — use the gap-fill v2 eval-ladder
pattern (artifact QA + both-on overlap + era-bucketed residuals incl. calibration).

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
research+forecast pipelines: the single most accurate forecast in that study was a **strong
prompt on good shared/fixed research** (0.129 Brier), edging the best self-directed integrated
agent (0.131), and integrated-vs-shared was **model-dependent** — only the Opus-class model
clearly benefited from running its own search; Gemini was slightly *better* on fixed shared
research. Our architecture (multi-provider shared briefing + gap-fill + 6-model ensemble +
median) already **is** BTF-2's winning recipe, at ~7 calls vs ~5N for full pipelines. The lever
is **shared-research quality + a strong prompt**, not integration topology. So: make the shared
research more agentic and higher-quality (the gap-fill v2 plan in
`scratch_docs_and_planning/agentic_gap_fill_v2_plan.md`), do NOT rebuild into per-forecaster
agents. If per-forecaster search is ever tried, give it only to the Opus-class slot. Calibration
caveat (research-repos + papers-skeptic): all these agentic-research wins optimize pass@1
accuracy, and edge-over-consensus has a *flat-to-negative* trend across a dozen model
generations — "more search → more decisive" can trade calibration for sharpness, so track
era-bucketed calibration slope/intercept alongside Brier when validating any research change.
See the deeper stub under "Longer-term → Agentic deep research" below (superseded by this).

## Near-term (worth exploring soon)

### Agentic gap-fill v2: plan agreed, implementation starting (added 2026-07-16)

Full design in `scratch_docs_and_planning/agentic_gap_fill_v2_plan.md` (rev 4, self-contained —
that doc is the source of truth; this entry is a pointer). Summary: a bounded agentic tool loop
becomes the second-pass research stage — a driver LLM (dev on gpt-5.6-luna, then vibe-eval
luna / terra / sol-low / sonnet-5.0) is briefed with the actual forecaster prompt template,
privately dry-runs the forecast to find fill/verify targets, then iterates over four tools:
`search_news` (existing AskNews rate-limit machinery), `search_web` (Exa direct on the
operator's new key — `~/.keys/exa_key` locally, GHA secret `exa_key`), `fetch`
(auto-escalating ladder plain → headless Chromium → Gemini url_context), and `read_document`
(Gemini flash url_context). Output is a detached, citation-only findings artifact appended to
the bundle; a ghost forecast is logged for telemetry only (never published). The loop is DIY
litellm-direct, no framework — append-only message array for prompt-cache discipline is the
reason. Codex agents implement; Fable orchestrates and writes the driver prompt + tool
descriptions with operator review.

**Rollout: BOTH v1 and v2 gap-fill run in prod during an overlap window** — two independent
flags (`GAP_FILL_ENABLED`, `GAP_FILL_V2_ENABLED`), both `'true'`, distinct research-section
headers, artifact diffs + resolution scoring harvested from the both-on era. **Turning v1 OFF
afterwards is a deliberate pending step the operator must remember** ("i just have to remember
to turn it off") — nothing does it automatically.

Evidence base (details + citations in the plan doc §7): Metaculus Fall 2025 bot survey —
research breadth correlates with accuracy at r=0.42; FutureSearch sits #1 with agentic
research at ~$1/question; Bridgewater's ablation points the same direction; and date-filtered
live search leaks resolution info on 41–55% of resolved questions (prompt-side "ignore
post-cutoff knowledge" fails), so **`make backtest_*` is uninterpretable for research-stage
changes** — eval is test_bot QA then an early prod flip, not backtests.

**Post-review additions (2026-07-16, external design review folded into plan rev 4):**

- Duplicate-query semantic stuck-detector + dedup-vs-primary-provider queries: deferred
  enhancement — v1 ships only a per-run dup-counter (`dup_tool_calls=N` on the marker line
  plus a gentle warning on the duplicate's tool result).
- Traditional-researcher tier-tagging (`web_research_prompt` + AskNews summarizer don't carry
  the provenance ladder): follow-up, ship AT the v2 prod flip so it rides the same era boundary.
- Era-bucketed calibration slope/intercept is now part of the v2 eval ladder alongside Brier
  (plan §7) — guards against agentic-research over-decisiveness.

DeepNews is AskNews's agentic iterative research product (AskNews KG + Google/Wiki/X/Reddit,
OpenAI-SDK-compatible endpoint). Our current integration only calls the basic HOT+HISTORICAL
news endpoints. Two options, relative to the gap-fill v2 driver above:

- **(a)** expose DeepNews as an optional heavy tool `search_news_deep(query, max_depth)` the
  v2 driver can escalate to when basic `search_news` comes up thin;
- **(b)** upgrade the basic `search_news` backend to DeepNews with a depth param.

**Blocked on: operator checking DeepNews limits/pricing.** It's a separate quota pool from the
OpenRouter donated key, possibly subsidized for tournament participants — if cheap/free there's
a solid case for (a).

### Confirm Gemini `url_context` actually fires in prod (added 2026-06-28)

The 2026-06-28 research-quality audit found **zero positive evidence** that Gemini's `url_context`
tool (purpose-built to directly read resolution-source URLs named in question fine print) ever fires
in production. Across all 17 post-2026-05-17 Period-B research records, every Gemini section cites only
`vertexaisearch.cloud.google.com/grounding-api-redirect/` links — 0/17 contain a direct
`.gov`/`fred`/`cboe` resolution URL. In every observed tracker case the live resolving value was
surfaced by the **gap-fill OpenAI native-search pass** or the **financial-data API**, NOT by
url_context. Most damning single data point: on q43650 Gemini's grounded snippet was *wrong*
(4.44–4.46%) while gap-fill returned the exact 4.48% that resolved. The fetch gap is *masked by
gap-fill*, not *closed by url_context*. (Inference from research text only — at audit time there was
no telemetry.)

**Telemetry was added 2026-06-28** (`gemini_search.py` `_extract_url_context_telemetry`): each
grounded call now logs `N/M url_context fetches` and writes a greppable marker into the persisted
research blob — `### URL Context Fetches` (successful reads, with the retrieved URLs) or
`_url_context: none_` (tool fired but fetched nothing). So this is now *observable for free* on the
next scheduled tournament/backtest run.

**Action (free, rides an already-scheduled run):** after the next prod run with Gemini enabled, grep
`backtests/research_archive/latest/*.json` for `### URL Context Fetches` vs `_url_context: none_` vs
absence, to settle definitively whether url_context fires and, if so, whether it reads the
criteria-named resolution URL. If it reliably direct-reads named sources, the deterministic-fetch
question (below) largely dissolves. If it never fires or never reads the *named* URL, that's the
evidence that would justify the narrow deterministic named-URL fetcher.

**Related (deferred, needs a small paid re-bench — clear cost first):** the audit could NOT test the
gap's worst case — obscure, low-news, *non-API* official counters/registries/dashboards (state policy
trackers, CBP encounters tables, WHO-style dashboards, mesonet data tables, "infants enrolled"
homepage counters). Period B contained zero of that archetype; the only two clean research-side
fetch failures in the entire 40-tracker corpus (q43046 WHO extranet dashboard, q43139 IEM mesonet
precip table) were both this type and both pre-current-stack. A deliberately adversarial ~10–15-question
re-bench enriched for that archetype is the single highest-leverage missing evidence — it either kills
the deterministic-fetcher project or scopes it precisely. The narrow design (parse criteria for a
resolver URL → force-fetch+parse; or make the gap-fill analyzer treat a criteria-named URL as a
*mandatory* gap) is sketched in `scratch/research_audit_2026-06-27/SYNTHESIS_62.md` §4. The FRED/Yahoo
URL-extraction shipped 2026-06-28 already covers the API-backed financial subset of this class.

### Resolution-source fetcher: flip prod workflows + Tier-2 LLM fetch (added 2026-07-09)

The Tier-1 deterministic resolution-source fetcher shipped in `66e31c0`
(`research/resolution_source.py` + shared `research/http_fetch.py`, gated by
`RESOLUTION_SOURCE_ENABLED`, currently ON in `test_bot.yaml` only). Smoke-validated on 40 cached
real questions (`scratch/resolution_source_smoke_2026-07-09/REPORT.md`): 24/40 questions (60%)
get a non-empty `## Resolution Source Snapshot` vs the probe's 62.5% Tier-1 target, 30/45 URL
success, 0 SSRF false positives, and the first-cited URL was the primary grading source in all
12 multi-URL questions. Remaining misses are all known Tier-2 hosts (JS walls / bot
fingerprinting).

**Truncation-cap distribution study (2026-07-09, don't re-derive):** uncapped re-fetch of the
29 real successful URLs from the smoke run, full trafilatura extraction: p25=697 / p50=2,201 /
p75=5,206 / p90=67,041 / max=438,049 chars. Elbow at 6,000 chars/URL — a 3,000 cap truncates
14/29 URLs (48%), 6,000 truncates 6/29 (21%), and past 6,000 only whale pages (67k+) remain,
which need summarization, not bigger caps. Mean prompt cost across all 40 smoke questions:
380→578 tokens/question at 6k. **Shipped:** `RESOLUTION_SOURCE_PER_URL_MAX_CHARS` 3,000→6,000
and `RESOLUTION_SOURCE_TOTAL_MAX_CHARS` 12,000→18,000 (headroom so the per-URL cap is the
binding constraint; max observed section simulates to ~11.1k at 6k/URL).

Follow-ups:

1. **Flip prod workflows — DONE 2026-07-10.** Live `test_bot.yaml` run confirmed real rubric/fact
   content, visible forecaster uptake, and clean diagnostics; `RESOLUTION_SOURCE_ENABLED=true`
   now set in all three `run_bot_on_*.yaml` prod workflows (tournament / metaculus_cup / minibench).
   Same commit added a per-URL truncation marker (`[truncated at N chars — full source at URL]`)
   and a dropped-section note so forecasters can tell when the snapshot is partial.
2. **MEDIUM — Conditional summarization for oversized sources.** The first-cited URL's content
   stays verbatim (provenance for the primary grading source); URLs 2+ and/or whale pages
   (full extraction ≥ ~10k chars; p90 of the real distribution is 67k) go through the existing
   cheap summarizer path (`gpt-5.4-mini`, temp 0, ~$0.01/call). Rationale: the distribution
   study found ~5 whale sources per 40 questions that no reasonable cap captures; raising caps
   past 6k has near-zero marginal rescue.
3. **MEDIUM — Expand fetching for other site types (Tier-2 LLM fetch)** for the
   js_wall/blocked slice (~15% of questions; e.g. Masters.com, childmortality.org, UNICEF,
   Tesla IR, sagaftra.org). The per-URL `FetchStatus` (blocked / js_wall retained
   deliberately) is the seam — a follow-on pass feeds those URLs to an LLM-mediated reader
   (Gemini `url_context` or OpenAI native-search URL-read).
   **Precondition:** the "Confirm Gemini `url_context` actually fires in prod" probe above
   (added 2026-06-28) — no point building on url_context until we know it fires.
   *Note 2026-07-16:* the gap-fill v2 fetch ladder (plain → headless Chromium → Gemini
   url_context; see the agentic gap-fill v2 entry above) gives the driver this capability
   inside the loop — the js_wall slice may get covered agentically before a dedicated
   Tier-2 resolution-source pass is built. Re-assess after the v2 overlap window.
4. **LOW — minor follow-up from review, explicitly deferred:** module split of
   `resolution_source.py` (~670 LoC; extract `ssrf_guard.py`).

### Parser hardening + forecasting-tools upgrade path (added 2026-07-07)

Full plan in `scratch_docs_and_planning/parser_hardening_and_ft_upgrade_plan.md` (written
after an 8-agent structured-outputs exploration). Decisions: do NOT migrate forecaster calls
to native `response_format` structured outputs (OpenRouter silent-degradation footguns,
load-bearing rationale channel, zero competitive precedent — no upstream layer uses it
either). Instead:

- **Workstream A — DONE and superseded.** Shadow divergence logging shipped, served its
  purpose, and was deleted 2026-07-10 when the block became authoritative (see the DONE
  entry below; `EXTRACTION_RUNG` telemetry replaced it). The strict json_schema on the
  *parser call* via `GeneralLlm(response_format=...)` also shipped (`structured_parse.py`,
  constrained primary + framework `structure_output` fallback) and now serves as the
  ladder's rung-3 salvage parser.
- **Workstream B (between rounds, ~1 focused day):** unfreeze `forecasting-tools` 0.2.54 →
  0.2.92+. Two verified breaks: our PCHIP subclasses override `.cdf` but HEAD internals moved
  to `get_cdf()` (silent bypass of our CDF machinery); `fetch_hardening` patch target moved
  to the new `MetaculusClient` (silent no-op). Plus a validator audit — HEAD's new
  `_check_too_far_from_bounds` (25% wiggle) may conflict with our beyond-range open-bound
  percentile design. Unlocks the litellm/cryptography CVE fixes below.
- **DONE (2026-07-10): JSON-block-as-authoritative for ALL question types
  (binary, MC, numeric) AND the stacker.** Value extraction runs through the
  deterministic four-rung ladder in `metaculus_bot/value_extraction.py`:
  fenced ```json block parse → json-repair salvage → LLM-parser salvage
  (`parse_structured`) → `ValueExtractionError`. The old Workstream A trigger
  ("wait for ~50 questions of shadow-divergence agreement data") was
  **consciously waived by the operator** in favor of two lighter-weight
  verification channels: (a) the `EXTRACTION_RUNG` INFO telemetry emitted on
  every extraction — `rung=llm` salvages and `block_present=false` are the
  drift signals to watch in prod logs; and (b) a user-gated live `test_bot`
  rerun that eyeballs those rungs before the first tournament run. The
  shadow-divergence logging module and its tests were deleted (superseded
  by rung telemetry); the F5 block-lift fallback was absorbed into rung 1
  of the numeric ladder.

### Percent-form block labels vanish silently in comment recovery (added 2026-07-15)

Found during the coherence-study cleanup. A numeric STRUCTURED FORECAST block whose
`declared_percentiles` keys are percent-form ("2.5", "5", …, "97.5") instead of fraction-form
(0.025, …) is dropped by BOTH recovery rungs in `performance_analysis/parsing.py`: the strict
`parse_structured_block` rejects the schema, and the tolerant salvage rung (added in `f530968`)
drops the keys on its own `0 < pct < 1` guard. Historically harmless — the prose
"Percentile 2.5: X" lines rescued these (that's exactly how qid 43684 / grok-4.3 survived) —
but post-2026-07 prompts are block-last-with-NO-prose-value-lines, so a future percent-form
block has no fallback and that model's percentiles vanish silently from residual analysis.
Fix is small and localized: teach the tolerant rung to detect an exact canonical-set×100 key
match and rescale deterministically — the validator + canonical sets already exist
(`_validate_percentile_labels`, `_CANONICAL_PERCENT_LABEL_SETS`, parsing.py:600-650, shipped
in the same commit); the rung just needs to route through them instead of hard-dropping.
Watch signal until fixed: a model whose per-question percentile coverage drops to zero in a
post-flip pull while its `EXTRACTION_RUNG` prod telemetry stays healthy.

### Dependency CVEs gated by the frozen `forecasting-tools` pin

`make audit` (osv-scanner over `uv.lock`, added in the 2026-06 uv migration)
flags known CVEs we currently can't patch because the fixed versions are
unreachable while `forecasting-tools==0.2.54` is frozen. As of 2026-06:

- **litellm 1.80.0** — four high-severity CVEs (GHSA-4xpc-pv4p-pm3w 9.5,
  GHSA-jjhc-v7c2-5hh6 9.4, GHSA-53mr/69x8 8.6–8.7), fixed in 1.83.x–1.84.0.
  `forecasting-tools 0.2.54` resolves litellm to exactly 1.80.0; our own
  `<2.0.0` cap is not the binding constraint. `uv tree --invert` confirms the
  pin chain. Unreachable without bumping forecasting-tools.
- **cryptography 45.0.4** — incl. one 9.8 (PYSEC-2026-36), pulled transitively
  via asknews / google-auth / mcp, all ultimately under forecasting-tools.
- **pillow 11.3.0, pydantic-settings 2.14.1, transformers 4.57.6** — lower
  severity, also transitive.

These are an accepted consequence of freezing forecasting-tools for behavioral
stability. Revisit when forecasting-tools is next upgraded (re-run `make audit`
after any bump); if a litellm/cryptography CVE becomes actively exploited before
then, evaluate an out-of-tree override (`[tool.uv] override-dependencies`) and
re-validate the numeric/stacking pipeline against the bumped litellm. CI runs
the same scan via `google/osv-scanner-action`, so new CVEs surface on every PR.

### Promote the core pipeline to `basedpyright` strict

The 2026-06 Poetry→uv migration wired `basedpyright` at **standard** mode across
the whole repo and drove it to zero errors. The original intent was **strict on
the core forecasting pipeline** (`forecaster.py`, `aggregation_pipeline.py`,
`stacking.py`, `numeric/`, `research/`, `probabilistic_tools/`); that was
deferred because, against an unclean base, strict surfaced ~900 findings of which
~450 were low-value "type is partially unknown" noise at the untyped
`forecasting-tools` boundary, and the cleaner standard-everywhere target caught
every real type bug with no suppression hacks.

Now that the base is clean, the strict promotion is much smaller. Spec:

1. Add the `strict` path list back to `[tool.basedpyright]` in `pyproject.toml`:
   ```toml
   strict = [
       "metaculus_bot/numeric",
       "metaculus_bot/research",
       "metaculus_bot/probabilistic_tools",
       "metaculus_bot/forecaster.py",
       "metaculus_bot/aggregation_pipeline.py",
       "metaculus_bot/stacking.py",
   ]
   ```
2. Resolve the resulting `reportUnknown*` findings (~447 last measured) by
   **annotating our own functions** — attribution showed ~95% of the unknowns are
   our own under-typed signatures/locals, not the library boundary. This is the
   real "make it pretty" work and should add genuine type coverage.
3. `forecasting-tools` is frozen and ships **no `py.typed`** marker despite having
   inline annotations, so strict re-raises `reportMissingTypeStubs` for it.
   `basedpyright --createstub forecasting_tools` is NOT a clean fix — it drops the
   Pydantic-generated attributes and made things worse (892→1011). Options, in
   order of preference: (a) hand-maintain a thin `typings/forecasting_tools/` stub
   covering only the symbols we import; (b) a one-line `reportMissingTypeStubs`
   override scoped to the strict execution environments (note: the global
   `reportMissingTypeStubs = false` does not survive the `strict` promotion —
   needs an `executionEnvironments` entry or per-file directive); (c) request a
   `py.typed` marker upstream.
4. Re-add the local `basedpyright` pre-commit hook in `.pre-commit-config.yaml`
   (removed during the migration so commits weren't blocked while the codebase was
   still dirty) and gate CI on it (the `typecheck` step already runs `basedpyright`).
5. Do this as a heavily-parallel workflow (one agent per core module), same shape
   as the standard-mode cleanup.

### ~~Supervisor agent for high-disagreement questions~~ DONE

Implemented as conditional stacking (`AggregationStrategy.CONDITIONAL_STACKING`).

### ~~Financial data tool access (yFinance, FRED)~~ DONE

Implemented as `financial_data_provider.py`.

### Run crux extraction on every question + always-on stacker (added 2026-05-17)

Today crux-extract + targeted-search + stacker only fire on the ~30% high-spread subset (CONDITIONAL_STACKING + `spread > threshold`). User raised whether the pipeline would be cleaner if it always ran: forecaster fan-out → crux extract → targeted re-research → final stacker.

**Cost** at gpt-5.5 high effort, ~250 Qs/tournament:

- Crux extract: 250 × $0.055 ≈ $14
- Targeted search (gpt-5.4-mini): 250 × $0.076 ≈ $19
- Stacker (claude-opus-4.5): 250 × $0.30 ≈ $75
- Total uplift vs disagreement-only: ~$80/tournament

**Open question**: do cruxes on uncontroversial questions actually move predictions, or just add latency + cost? User's framing — "cruxes only make sense if we stack all the time, otherwise the crux research is wasted on a one-pass prediction." So this is paired: either both flip on together, or neither.

**Test before flipping**: pick 50 historical resolved Qs from past tournaments where base models AGREED, run both pipelines offline, compare Brier / log-loss. ~Half-day of eng work, but the math win is ambiguous, hence test before flip.

Priority: HIGH (clean architecture + the user views the 0.15 disagreement bar as somewhat arbitrary).

### Re-run native-search model evaluation each quarter (added 2026-05-17)

After migrating from `x-ai/grok-4.1-fast` (deprecated) to OpenAI's native-search models 2026-05-17, baseline data lives at `scratch/native_search_bench_2026-05-17/comparison_v3.md` (v3 supersedes the earlier v2 verdict; the final landing config is `gpt-5.5` with `reasoning={"effort":"medium"}` + `verbosity=low` under a 360s cap). As cheaper / better OpenAI search models ship (gpt-5.6, mini variants), or if Anthropic/Google add native search to OpenRouter, re-run the harness:

1. `python scratch/native_search_bench_2026-05-17/run.py --question-id <new open Q>` (or refactor into a make target if it gets reused).
2. Update `metaculus_bot/constants.py:NATIVE_SEARCH_DEFAULT_MODEL` based on results.

Bake into the quarterly review cadence.

### Gemini on the donated OpenRouter key: pro-preview still blocked by free-tier BYOK (updated 2026-06-16)

**Update (2026-06-16, fresh live calls):** Metaculus raised the Google rate limits (per Ben's Discord announcement). The donated OpenRouter key (`OAI_ANTH_OPENROUTER_KEY`) now serves **most** Gemini models — verified by live call:

- `openrouter/google/gemini-3.5-flash` → **SUCCESS** on the donated key.
- `openrouter/google/gemini-3.1-flash-lite` → **SUCCESS** on the donated key.

**Remaining blocker — `gemini-3.1-pro-preview` only (our forecaster slot):** that specific model is still routed through the free-tier Google AI Studio **BYOK** key on the donated account (`is_byok:true`), and Pro-preview has no Google free tier, so the BYOK quota is structurally 0. Every donated-key `gemini-3.1-pro-preview` call still 429s with `RESOURCE_EXHAUSTED` (`is_byok:true` + free-tier `limit: 0`) and `FallbackOpenRouterLlm` gracefully falls back to the personal `OPENROUTER_API_KEY`.

**Resolution (this session) — SURGICAL pin, not all-or-nothing:** default flipped back to `GEMINI_USE_DONATED_OPENROUTER_KEY=true` and all four prod YAMLs (`run_bot_on_{tournament,metaculus_cup,minibench}.yaml`, `test_bot.yaml`) flipped on, so flash Gemini models (`gemini-3.5-flash`, `gemini-3.1-flash-lite`) use the donated key. But `gemini-3.1-pro-preview` is **pinned to the personal key** via the `DONATED_KEY_BLOCKED_GOOGLE_MODELS` blocklist in `metaculus_bot/fallback_openrouter.py` — `should_route_via_donated_key` returns `False` for it, so `build_llm_with_openrouter_fallback` builds a plain `GeneralLlm` on the personal key. No donated attempt, no 429, no fallback-counter bump.

This avoids the **CI-red-every-run** problem: `gemini-3.1-pro-preview` is a core forecaster on every question; without the pin, each donated→429→personal fallback bumps the personal-key-fallback counter → `cli` `sys.exit(1)` → red CI on every prod run. The surgical pin removes the donated attempt entirely for Pro while keeping the donated subsidy for the flash models that actually work on it.

**⚠️ TEMPORARY WORKAROUND — remove the pin once Metaculus fixes the BYOK routing.** `gemini-3.1-pro` *should* work on the donated key; the **only** blocker is Metaculus's free-tier Google BYOK routing. The pin is tagged in code as `TODO(gemini-3.1-pro-donated)` on the `DONATED_KEY_BLOCKED_GOOGLE_MODELS` constant — delete the matching entry there (the doc-and-code source of truth for the workaround) once any one of the Metaculus-side fixes below lands, then re-verify with one live call.

OpenAI / Anthropic on the donated key are **unaffected**: there's no broken BYOK key for those providers, so they route on the donated subsidy normally.

**Fix (Metaculus-account-side, for pro-preview specifically, pick one):**

1. Enable Cloud billing on the BYOK key's GCP project so it reaches Tier 1 (gemini-3.x-pro-preview gets a non-zero quota).
2. Remove the Google AI Studio BYOK integration so `google/*` uses native OpenRouter Google credits instead of the BYOK key.
3. Disable "Always use for this provider" on that BYOK key.

**Does NOT help:** raising OpenRouter-side native limits — the pro-preview 429 is Google-side on the BYOK key, not an OpenRouter throttle.

**Action:** pinged Ben. Once Metaculus applies one of the fixes above, re-verify pro-preview with one live `openrouter/google/gemini-3.1-pro-preview` call and confirm the error no longer carries `is_byok:true` + free-tier `limit: 0`; then remove the `gemini-3.1-pro` entry from `DONATED_KEY_BLOCKED_GOOGLE_MODELS` (see `TODO(gemini-3.1-pro-donated)` in `metaculus_bot/fallback_openrouter.py`) so Pro rejoins the donated subsidy.

### ✅ RESOLVED 2026-05-29 — `OAI_ANTH_OPENROUTER_KEY` data-policy block for OpenAI native search

Metaculus enabled OpenAI on the donated key. `build_native_search_llm` now routes through
`build_llm_with_openrouter_fallback` (donated key primary, personal key fallback). Verified
end-to-end on `openai/gpt-5-mini`: grounded result returned, donated-key 404 fallback count = 0,
i.e. the call succeeded on the donated subsidy. The guardrail/data-policy fallback matcher stays
in place as a safety net. Original note retained below for context.

When migrating native search from `x-ai/grok-4.1-fast` (deprecated) to OpenAI native search on 2026-05-17 (final landing config: `openai/gpt-5.5` medium-effort + verbosity=low, see W-C v2), the donated Metaculus OpenRouter key (`OAI_ANTH_OPENROUTER_KEY`) returned a 404 with:

> No endpoints available matching your guardrail restrictions and data policy.
> Configure: <https://openrouter.ai/settings/privacy>

This means OpenAI native-search calls fall back to the personal `OPENROUTER_API_KEY` instead of the donated subsidy. At ~$0.15/call × 250 Qs/tournament that's ~$40/tournament (size grew vs the original mini-only estimate of ~$3-5 because we landed on gpt-5.5 medium-effort, not mini) — still small enough to defer, but worth reclaiming.

**Investigation paths**:

1. **Email Metaculus** (<ben@metaculus.com>) — ask whether the data-policy guardrail on the shared OpenRouter account can be relaxed for OpenAI native search, or whether they need to whitelist a specific endpoint.
2. **OpenRouter request preferences** — try `provider: {data_collection: "deny"}` or `provider: {require_parameters: true}` in the chat completions request to see if a compliant OpenAI endpoint is available; OpenRouter's privacy doc at <https://openrouter.ai/docs/features/privacy> describes the routing knobs. The metaculus-bot's `build_native_search_llm` already uses `extra_body` for plugins, so adding `provider` is one line.
3. **Accept personal-key spend** — if (1) and (2) both fail, the migration still solved the Grok deprecation; the cost is small enough to live with.

Today's status: code falls back automatically via `FallbackOpenRouterLlm` (added pattern matcher for "guardrail" / "data policy" in `fallback_openrouter.py`). No incidents expected.

Priority: HIGH — it's free money to reclaim, and the same guardrail may bite us when the next OpenAI/Anthropic/Google migration comes up.

### Second-pass web search + scrape pipeline

> **SUPERSEDED 2026-07-16** by the agentic gap-fill v2 plan
> (`scratch_docs_and_planning/agentic_gap_fill_v2_plan.md`; near-term entry above). The three
> use cases below are covered by the v2 tool loop. Firecrawl/Olostep were rejected in favor of
> a DIY fetch ladder (plain → headless Chromium → Gemini url_context). Kept for historical
> context.

Our first-pass research (AskNews API dump, Grok native search) is a black box — we can't
control what gets fetched, can't parse PDFs or JS-heavy pages, and can't follow up on gaps.
A second pass with full-control scraping would address this.

**Three use cases for the second pass:**

1. **Gap-filling**: After initial research + forecasting, identify information gaps or
   unanswered questions from the first pass and run targeted searches to fill them.
2. **Resolution source reading**: Many questions include specific resolution source URLs
   in the fine print. Directly scraping the authoritative source (e.g., a government
   dataset, a specific report) gives ground-truth current state instead of secondhand
   summaries.
3. **Reopening inaccessible sources**: The first-pass research often surfaces URLs that
   the bot can't open (PDFs, paywalled content, JS-rendered pages). The forecasting
   prompt should instruct models to flag interesting sources they couldn't access, so
   the second pass can scrape them with more sophisticated tools.

**Tool candidates**: Olostep (cheaper, PAYG) or Firecrawl (pricier but the industry norm).
Both handle PDFs, JS rendering, and give full control over what gets fetched.

**Architecture**: Runs after the initial research phase, before (or as input to) forecasting.
Could also feed into the stacking pass for high-disagreement questions. Moderate effort.

### Separate outside/inside view stages

Currently both phases happen in a single prompt. Making them separate LLM calls means the
inside view genuinely adjusts FROM an explicit base rate rather than constructing both
together. This could help with the arithmetic-override problem (models computing correct
probabilities then ignoring them).

Smingers uses this architecture with cross-pollination: outside view from Model A feeds
into inside view for Model B, introducing diversity.

We could prototype this by having a first pass produce only a base rate + reference class,
then feeding that explicitly to the second pass. Moderate-to-high effort.

### Post-hoc isotonic calibration on binary predictions

> **Status 2026-05-10: NOT REPLICATED at the larger N — DROP.** The May closing
> analysis (n=109 binary, full cohort) showed the [0.20, 0.30] band did not show
> the 5/6-of-worst-misses concentration that April-new (n=27) did. The April
> finding was small-N artifact. The 20 worst May misses span a much wider range of
> failure modes (high-spread contested questions like 42243 Christie's, 42923
> Senate, 42926 CDC). Defer global isotonic until a >50pp residual is observed in
> any predicted-prob band on N≥50. — Section preserved below for historical context.

Our performance analyses (2026-Q1 and 2026-Q2) have repeatedly found systematic
NO-bias in the 10–30% predicted-probability band: Q1 showed −7.3pp overall bias;
Q2's new cohort showed −22.7pp overall and −59pp in the 0.10–0.30 bucket. Six of
the eight worst binary misses in Q2 sat in that band. The top 3 worst
spot_peer misses (post_ids 43131, 41835, 42116) are all failures of this same
pattern plus correlated LLM priors — ensembling doesn't help when every model
shares the prior.

Calibration is the statistical fix where prompting is least likely to work,
because the failure pattern is statistically real across N≥100 questions, not
vibes from N=3.

**Proposal**: fit a monotonic (isotonic) regression on the combined Q1+Q2
resolved-binary dataset, mapping the ensemble's aggregate `prob_yes` to a
calibrated output. Apply the mapping to binary predictions before
submission.

**Why isotonic and not shrink-toward-50%** (which is the existing bullet in
"Aggregation strategy improvements"): shrinkage is a single-parameter global
pull that costs correctly-confident predictions to fix overconfident ones.
Isotonic is non-parametric and monotonic — it pulls 25% → 40% if that's
what the data says while leaving 5% and 95% largely alone if those buckets
are well-calibrated. Much less risk of overcorrection.

**Implementation sketch**:

- Use `sklearn.isotonic.IsotonicRegression` fit on
  `(our_prob_yes, resolution_as_float)` pairs from
  `scratch/analysis_2026-04/performance_data.json`.
- Hold out 20% as validation, report pre/post Brier + log score + PIT
  calibration.
- Wrap as a pure transform: `calibrated = iso.transform([raw_prob])[0]`.
- Store the fitted model in `metaculus_bot/calibration/binary_isotonic.pkl`
  with a training-date stamp; refit quarterly.
- Ship behind a feature flag (e.g. `USE_BINARY_CALIBRATION = True` in
  `metaculus_bot/constants.py`) so we can A/B via the bench runner.

**Risks**:

- **Overfitting to historical cohort**. Isotonic with N=100-200 is noisy;
  use 5-fold CV to pick breakpoints. If CV Brier is not robustly better
  than raw, don't ship.
- **Distribution shift**: next round's question mix may differ. Mitigate
  by refitting each round, keeping the training set rolling.
- **Trust erosion**: if the calibration ever silently flips a confident
  prediction, it'll look like a bug. Log raw + calibrated side-by-side
  in the bot comment for the first cohort.

**Out of scope**: numeric/MC calibration. Binary only — other question
types don't have enough resolved data to fit a reliable mapping yet.

Easy-to-moderate effort. The biggest risk is shipping it without proper
held-out CV.

### Probabilistic tooling for base forecasters (DORMANT — activation guide written)

> **Status 2026-05-10: PROMOTED — activate after instrumentation bugs ship.** The
> strongest case is two numeric misses where the failure is in *representation*,
> not reasoning, and prompt edits demonstrably didn't help even when the model
> knew it should:
>
> - **NM1 (DOJ antitrust=0):** model wrote *"Probability: ~92%"* of 0 in prose
>   but its 11-percentile output represented only ~55% mass at-or-below 0. The
>   percentile elicitation can't represent a point mass at a discrete endpoint;
>   `Beta-binomial-ceiling` / `NegBinom` utilities fix this directly.
> - **NM3 (MSFT EPS):** model identified the GAAP-vs-adjusted risk and wrote
>   *"I deliberately widen the 90-95% interval relative to the analyst range"*
>   — and still couldn't widen enough; resolution at $5.16 was past P97.5.
>   Prompt-driven tail-widening failed against a known risk; structural
>   `out-of-bounds mass reporting` handles this case explicitly.
>
> A softer secondary case: 9/15 binary hits' load-bearing reasoning is explicit
> Poisson math; the survival/hazard calculator would standardize this across
> base LLMs. Less defensible because we can't tell whether the math caused the
> wins or just marks questions that fit a clean reference class.
>
> Infrastructure is built, 261 tests green, gated behind `PROBABILISTIC_TOOLS_ENABLED`
> — A/B-able immediately. Backtest gate: improvement on numeric (especially
> count-distribution and discrete-mode questions) with no regression on simple
> binary cohort.

Base-forecaster failure mode identified in Q2 2026 analysis: models state
base rates, percentiles, and priors in prose but don't compute on them
("arithmetic override"). Ships `metaculus_bot/probabilistic_tools/`,
`metaculus_bot/structured_output_schema.py`, and
`metaculus_bot/tool_runner.py` — pure-function Beta-binomial updaters,
survival / hazard calculators, log-pooling + Satopää extremization,
distribution fitting (normal / lognormal / Student-t) with out-of-bounds
mass reporting, Dirichlet-with-Other for MC, NegBinom / Beta-binomial-ceiling
for counts, and prior-posterior + percentile-family consistency checks.

**Status:** tools + tool_runner + 261 unit/E2E tests all green. NOT wired
into prompts, `_make_prediction`, or the stacker. Activation is a
single-session prompt + main.py + stacking.py edit behind a
`PROBABILISTIC_TOOLS_ENABLED` env flag.

**Activation plan:** `scratch_docs_and_planning/probabilistic_tools_activation.md`
— exact file-and-line edits, parser-ordering gotcha (JSON block before
`Probability: ZZ%`), A/B backtest verification sequence, known landmines.
A fresh session can flip it live with minimal context loss.

### Status-quo / last-print anchor for slow-moving numeric trackers (added 2026-06-28, NEEDS BACKTEST before acting)

**Finding (2026-06-28 period-split research audit, current-stack "Period B" cohort, n=17, of which
~5 numeric trackers — small sample, directional only):** on numeric/discrete questions that resolve on
a slow-moving / mean-reverting tracker series, the research pipeline now reliably surfaces the **exact
current value**, and then the forecaster ensemble *degrades* it by layering directional drift or
asymmetric-widening tails on top. The three worst Period-B numeric trackers all had the resolving value
sitting in the research:

- **q43647 HY-OAS spread** (peer +52.4): research handed over 2.71 exactly; per-model medians skewed UP
  to 2.73–2.80 anticipating widening that never came over a calm ~two-week window; truth landed below p40.
- **q43611 generic ballot** (peer +31.8): research surfaced Silver net +6.8 *plus* an explicit
  mean-reversion base rate; ensemble extrapolated the late-May uptrend to ~6.7–7.2; series reverted to 6.4.
- **q43591 Trump approval** (peer +14.3): research surfaced 38.5; ensemble applied a damped *downtrend* to
  37.8; the tracker ticked *up* to 38.6.

NOTE these three scored **positively** — the bot did fine; the finding is "left points on the table by
over-reasoning on a value research already nailed," not "lost." A flat "status-quo = last surfaced print"
central anchor would have materially improved all three. This is a forecaster-**reasoning** lever, fully
independent of the research/fetch work shipped 2026-06-28.

**Proposed change (prompt-only, numeric + discrete paths):** add a conditional step to
`numeric_prompt` / discrete handling: *when the research surfaces a current authoritative value for a
slow-moving or mean-reverting tracker (rates, spreads, approval/poll averages, index levels), default the
central estimate (p50, and the bulk of the mass) to that last print, and require an EXPLICIT,
named justification before applying directional drift.* Frame it as a rebuttable default, not a hard
constraint — the model must still be free to move when it has a concrete catalyst (a scheduled release,
a structural break, a genuine trend with a stated mechanism). Likely lives alongside the existing
conditional-hazard step in the numeric prompt; reuse that "compute the number, then state the assumption"
pattern.

**Why this is NOT shipped yet (the trap to avoid):**

1. **n=3.** A real pattern but a thin sample; could be noise dressed as signal.
2. **Over-correction risk.** A blanket anti-drift instruction will *hurt* genuinely-trending or
   event-driven numerics (a series mid-breakout, a count accumulating toward a deadline). The rebuttable
   framing mitigates this but doesn't eliminate it — the model has to correctly classify "slow/mean-
   reverting" vs "trending," which is itself a judgment call the prompt is now asking it to make.
3. **Prompt-behavior changes can only be validated on a real run** — unlike the observability/routing
   work this session (all unit-testable offline), this one's value is unprovable without a paid backtest.

**Backtest gate (clear cost with the operator first):** A/B on a **numeric/discrete-heavy slice** —
improvement (or no regression) on slow-moving-tracker questions (rates/spreads/poll-averages/index-levels)
AND, critically, **no regression on trending/event-driven numerics and count-toward-deadline questions**
(the over-correction failure mode). A `make backtest_small`/`medium` seeded with both archetypes is the
cheapest discriminating test. Do NOT ship on the strength of the n=3 hits alone. Full per-question
evidence in `scratch/research_audit_2026-06-27/SYNTHESIS_62.md` §3.

### LLM-based forecast self-evaluation

After each forecast, run a cheap model to assess: research relevance, factual accuracy,
reasoning soundness, date/chronology correctness, resolution criteria interpretation.
Flag potential issues before submission.

Smingers found this invaluable for catching date confusion, hallucinated sources, and
reasoning failures. Implementation: easy (structured eval prompt + cheap model call).

### Hits-side reasoning prompt-test ideas (added 2026-05-10)

> **Status 2026-05-10: LOW PRIORITY — defer pending `probabilistic_tools` backtest.**
> The dormant `probabilistic_tools` infrastructure provides a stronger version
> of what these prompt edits try to elicit informally — #1's Poisson math via the
> survival/hazard calculator, #2's pace arithmetic via Beta-binomial updaters.
> Backtesting both classes of intervention in parallel would confound A/B
> attribution, so activate `probabilistic_tools` first; revisit these ideas only
> if the dormant tools don't move the needle on similar question shapes.
>
> Additionally, prompt-test #2's evidentiary base is smaller than implied — the
> Wikipedia hit/miss N=2 pair is actually N=1: miss 42238 *did* apply the
> required-vs-observed math correctly and just landed in a 16% tail by chance.
> Only hit 42235 fits "qualitative-hedging-overrides-arithmetic". Defer #2.

Three prompt edits identified by the May closing analysis (`scratch/analysis_2026-05/analysis_hits_reasoning_patterns.md`)
from reasoning shapes that preceded the top-10 binary hits. **Hypothesis-generating, not
shipping recs** — needs N≥30 prompt-vs-prompt backtest before any change ships.

1. **"State your Poisson lambda explicitly when applicable"** — 5/10 top hits used
   `P(≥1) = 1 - exp(-λ·T)` arithmetic with stated λ and T. Misses-side reasoning
   often skipped this. Add ~3 lines to binary system prompt.
2. **"Required-vs-observed pace section"** — 3/10 top hits used arithmetic on
   threshold-by-deadline questions ("539/day observed vs 636/day required").
   **Note:** the Wikipedia hit/miss pair is N=1, not N=2 (see status above).
3. **"Distrust briefing claims that contradict the question's open status"** — the
   April Klimt-sale hallucination (miss 42243, 4/5 models pulled by a fake research
   datapoint) is the inverse failure mode. Explicit prompt clause: "If a fact in the
   briefing would, if true, definitively resolve the question YES or NO, but the
   question is still open, treat that fact as suspect rather than authoritative."
   **N=1 in May binary cohort; weakest of the three.**

Risk: prompt-length growth degrades simple-question performance. Backtest gate is
"mean Brier improves with no per-cell regression on the easy/middle tier."

### Stacker prompt: tell it which models are reliable dissenters (added 2026-05-10)

May C5+C7 analysis showed gpt-5.2 is a **contrarian signal source** (8/20 best on
worst-misses, 0/20 best on hits, mid-pack full-cohort Brier 0.150) — exactly the
high-disagreement signal source the conditional stacker is supposed to up-weight.
claude-4.6-opus has the inverse profile (best on the random-middle cohort, worst
on hard questions). The current stacker prompt strips model IDs (LLM-as-judge
self-agreement bias) but the *historical pattern* of "this model is a reliable
dissenter on high-spread questions" is signal we're throwing away.

Hypothesis: a stacker prompt that includes a small "historical dissenter quality"
hint (e.g. "Forecaster X has historically been the closest model on high-disagreement
questions Y% of the time") could produce better stacked outputs on the high-spread
cohort.

Blocked on: STACKER_OUTCOME marker fix (Priority 1A in NEXT_SESSION_QUEUE.md), then
≥30 stacked records under the new marker, before this can be tested. Defer.

### Per-forecaster critic/revision pass (added 2026-07-08, medium priority)

An unconditional adversarial critic reviews each forecaster's draft against a resolution-criteria
checklist BEFORE aggregation — window discipline (does the resolution window actually cover the
event?), already-resolved-events-don't-count (an event before the window opened cannot resolve
the question), listing/instrument bar (does the specific instrument or ticker named in the
question exist and clear the criteria?), blind-spot pricing (did the model implicitly assume
a fact it should have priced). The forecaster then answers the critic point-by-point and
re-issues its forecast, with the revision capped at ±20% from the original draft to prevent
over-swing on a single critic pass.

**Evidence.** Laertes (summer futureeval-2026 #4 slot) runs this critic pass on all forecasters.
The most striking single data point is qid 42024: Laertes's Forecaster 1 initially drafted 97%
(the exact number we published) and the critic reversed it to 4% pre-publication after flagging
that the "resolving" event fell outside the question's open window. GreeneiBot2 (spring-aib-2026
#1 slot) runs capped critique rounds with similar structure. Two independent top bots
converging on this pattern is a notable signal.

**Caveat (explicit).** The demonstrated evidence is on **degenerate** failures — pre-open-window
event traps and similar resolution-criteria misreads where a critic pass mechanically catches a
model applying the wrong reference class. Generalizing to non-degenerate misses (routine
mid-range calibration errors, close-call disagreements) is **unproven** and would require a
proper paid backtest (`make backtest_medium`-class ablation of critic-on vs. critic-off on a
mixed cohort) before shipping. Cost is ~1 extra LLM call per forecaster per question, so at
6 forecasters × ~250 Qs/tournament this is a non-trivial but not-huge line item.

**Distinct from the stacker (which was benchmarked and rejected).** The stacker is a *post-hoc
aggregate rewrite* — one meta-model looks at the N base forecasts and produces a single
consolidated number, which the ablation showed ties MEDIAN on binary and loses on numeric. The
critic pass here acts **per-member BEFORE aggregation** with a **bounded revision** (±20%),
which is a structurally different lever — it hardens each base forecast against a checklist of
known reasoning traps rather than trying to arbitrate a disagreement after the fact.

**Gate before shipping:** `make backtest_medium` on a mixed-question cohort, primary metric peer
score, secondary metric Brier / CRPS. Look for: (a) improvement or no regression on the
"degenerate-failure" subset (window-trap, listing-bar, already-resolved cases — the demonstrated
class); (b) no regression on non-degenerate misses (the unproven-generalization class). If (a)
lands but (b) shows regression, ship it as a **conditional** critic (fires only when the
question exhibits pre-open-event / listing-bar / criteria-mismatch flags) rather than
unconditionally.

**Update 2026-07-08 (pt2 acid test):** down-scoped. On the two non-degenerate consensus misses
(41800 / 42855), advisory critique captured ~0 points even when it diagnosed the exact defect;
only BINDING corrections moved numbers. If built, the critic must emit a bounded numeric
adjustment / floor / cap that is mechanically applied in aggregation — not prose. Sequence AFTER
the free offline counterfactuals of the deterministic guards (anchor-floor, no-market cap,
signed haircut), which capture most of the demonstrated value at zero cost. Honest ceiling:
~30–60 spot-peer points of damage limitation on consensus misses, no flips. See
`scratch/residual_2026-07-08/ACID_TEST_VERDICT.md` §3.

**Update 2026-07-08 (guard counterfactuals):** the sequencing premise is falsified — all
three deterministic guards buried on offline replay (era sign-flips / top-5 concentration /
fall-hostile fire rates; see `scratch/residual_2026-07-08/experiments/GUARDS_SYNTHESIS.md`
and the three guard entries in the "Killed by July 2026-07-08" section below). The critic
pass now carries the full burden of demonstrating era-stable conditional firing on its own
paid backtest; there is no free deterministic fallback capturing the same value. Its gate
must include a fall-like era-stability check, not just capture of the spring miss cluster —
the guards showed that harvesting that cluster is exactly what damages the largest and
best-calibrated era.

### Telemetry-first guard revival program (added 2026-07-08, passive)

The shipped `30bca2f` telemetry (`base_rate_anchor {low, high}` and `criteria_clauses` on
`BinaryStructured`) plus `PREDICTION_MARKETS_ENABLED: 'true'` across all four prod workflows
make future guard replays exact rather than parser-based — Arm A's regex parser at 84.9% text
coverage is now a structured field, and the market snapshot section starts populating the
archive going forward. No code is on the roadmap here; the whole program is passive.

Important gating note on the telemetry channel: the computed
`ANCHOR_OVERSHOOT_PP` / `CLAUSE_PRODUCT_DIVERGENCE_PP` HTML-comment markers emit only from
`tool_runner.run_tools_for_forecaster`, which is gated behind `PROBABILISTIC_TOOLS_ENABLED`
(all three prod workflows pin it to `'false'`), so those computed markers are currently
DORMANT in published prod comments. What DOES land unconditionally in every prod R1
rationale is the raw `base_rate_anchor` and `criteria_clauses` JSON the forecaster writes
into its own STRUCTURED FORECAST block — the primary telemetry channel today. The computed
markers become the primary channel only if the flag is flipped on; until then the
overshoot / divergence math (which lives in `tool_runner`) is trivially replayable offline
from the raw JSON.

First checks in the next residual session, both free:

1. **Structured-JSON presence rate per forecaster** in published comments — the whole replay
   program depends on the telemetry actually landing. Grep the archive for the raw
   `base_rate_anchor` and `criteria_clauses` JSON keys inside each forecaster's STRUCTURED
   FORECAST block and confirm every slot is emitting them. If `PROBABILISTIC_TOOLS_ENABLED`
   is ever flipped on, additionally grep for the computed `ANCHOR_OVERSHOOT_PP` /
   `CLAUSE_PRODUCT_DIVERGENCE_PP` HTML-comment markers as a cross-check; today those markers
   are absent and their absence carries no signal.
2. **Does the spring overshoot pattern reproduce on the current roster at all?** If the
   confident-overshoot cluster (analogues of 42024 / 42304 / 41800) does not appear in
   post-`30bca2f` resolutions, the prompt fixes were sufficient and all three guard revival
   conditions (Guard 1 anchors, Guard 2 markets, Guard 3 confidence deadzone) become moot.
   Compute overshoot / divergence offline from the raw JSON (same math as `tool_runner`)
   until the flag lands.

One novel candidate trigger becomes analyzable for free once current-roster binaries
resolve: `clause_product_divergence_pp` (published forecast vs. the model's own priced
clause product). It is the first trigger that keys on divergence-from-own-math rather than
confidence, anchor band, or market presence — the exact conditionality the three tested
guards failed to achieve. Watch, don't act.

Also watch (MC, added 2026-07-09): whether the low-bucket over-payment closes under the new
merged MC calibration bullet (`ceab2df`). Baseline: [0-5%) options assigned mean 2.4%,
resolve at 1.0% (n=96 pairs, both eras — "courtesy mass" on named-dead longshots leaking
from under-committed favorites; see MC_CONFIDENCE_FINDINGS.md). If the gap persists at the
next residual pull, add one prompt line: price clearly-dead NAMED options at/near the 1%
floor (residual/"Other" options keep honest mass — asymmetric by option type). The 1% floor
itself stays (operator decision 2026-07-09: sub-1% headroom is ~+0.01 nats/question ideal
case vs. parser/clamp regression risk — not worth it).

## Medium-term (requires more exploration)

### Research-output audit: temporal/provenance error sweep (added 2026-07-08, low priority)

Motivated by the qid 42304 INES miss: the then-native-search provider (`x-ai/grok-4.1-fast`,
since retired) cited a real but undated-URL NucNet archive article from 1 Feb 1999 (the
1998–99 Istanbul INES-3 accident) and presented it as a "February 2026" Turkish event with a
fabricated "reported March 1, 2026" date — likely cross-contaminated from an adjacent 2026
search result. All five forecasters anchored on it (81% published; resolved NO; peer −115.9).
The same phantom claim independently reached at least two top-competitor research stacks, so
this is a field-wide hazard of undated archive URLs, not a one-off provider quirk.

Idea: a free, offline audit pass over `backtests/research_archive/latest/` — sample research
texts per provider, spot-check high-leverage factual claims (dates, event existence, numbers)
against their cited URLs, and classify error modes (temporal displacement, fabricated dates,
certainty inflation by the summarizer). Output: per-provider error-rate estimates + a list of
recurring hazard patterns to feed prompt/summarizer hardening. Low priority — the offending
provider is gone and prompt-side mitigations (date-stamping relative to question open date,
single-source claim flagging) are the nearer-term lever — but worth doing before trusting any
new research provider swap.

Deferred from the 2026-06-01 desloppify code-health pass (which did only behavior-neutral pyproject hygiene: dropped unused `python-decouple`, removed the unused `litellm[proxy]` extra, declared the directly-imported `scipy`/`pandas`/`pydantic` that were previously only transitive via `forecasting-tools`).

Two follow-ups intentionally left for a separate, gated PR:

1. **Raise version floors to current-installed** (e.g. `litellm ^1.80` vs the current `^1.59.1` floor, `openai` to latest, and evaluate moving `forecasting-tools` off the hard-pinned `0.2.54`). This is **forecast-affecting**: a litellm/openai/forecasting-tools behavior change can subtly shift model outputs and therefore predictions. **Gate:** run a medium backtest (`make backtest_medium`) before and after the bump and confirm scores don't regress before shipping. Do NOT bump blind.
2. **Migrate dependency management from poetry to `uv`.** Larger refactor (pyproject `[tool.poetry]` → PEP 621 `[project]`, regenerate lockfile, update Makefile `install`/`run`/`test` targets and CI). An orphaned 136-byte `uv.lock` stub + `[tool.uv]` block (declaring a contradictory `requires-python >=3.12`) were removed in the 2026-06-01 pass so the repo has one source of truth (poetry); a real uv migration would regenerate the lockfile from scratch. Worth doing for speed + the team's broader uv standardization — poetry is dated and the team is standardizing on uv, so this is the intended direction; it's pure tooling churn with no forecast impact, so schedule when there's appetite for a no-functional-change infra PR.

   **Update 2026-06-16:** the contentless `uv.lock` stub re-appeared (an incidental `uv` invocation in this environment keeps regenerating it). Rather than delete-and-forget again, added `uv.lock` to `.gitignore` so it can't sneak back into a commit before the real migration lands. When the migration happens, remove that gitignore line and check in the real resolved lockfile.

### Gemini grounding via OpenRouter — currently NOT supported (added 2026-05-17)

Goal would be: route Gemini Google-Search-grounded calls (currently in `metaculus_bot/gemini_search_provider.py` via direct `google-genai` SDK + `GOOGLE_API_KEY`) through OpenRouter so the donated Metaculus credits cover them, freeing up personal Google API budget.

**Status as of 2026-05-17**: NOT supported. OpenRouter's web plugin and `:online` suffix expose native search ONLY for Anthropic / OpenAI / Perplexity / xAI. Gemini falls back to **Exa** (verified HIGH confidence: <https://openrouter.ai/docs/guides/features/plugins/web-search>). Migrating today would silently swap Google's grounded retrieval for Exa text-search — quality regression, not just cost optimization.

**Recheck periodically**: <https://openrouter.ai/changes> — if/when OpenRouter announces native Google grounding (or a passthrough for `tools=[{"google_search":{}}]`), revisit this migration. Until then, no action.

### Update analysis-CLI defaults to summer-futureeval-2026 (added 2026-05-17)

Tournament rolled over from `spring-aib-2026` to `summer-futureeval-2026` on 2026-05-17. The bot's live tournament target (`metaculus_bot/constants.py:TOURNAMENT_ID`) updated immediately, but **three CLI default constants stayed pinned to spring** intentionally:

- `metaculus_bot/ablation/cli.py:95` — `DEFAULT_TOURNAMENTS = ["spring-aib-2026"]`
- `metaculus_bot/performance_analysis/collector.py:30` — `DEFAULT_TOURNAMENT = "spring-aib-2026"`
- `metaculus_bot/performance_analysis/cli.py:17` — same constant

**Rationale for not updating yet**: ablation + perf-analysis are run against *resolved* questions. Summer just opened (zero resolved Qs); spring just closed (n=189, ongoing residual analysis). Updating the defaults now would force `--tournament spring-aib-2026` on every analysis command for what's still the active dataset.

**Flip when**: summer accumulates ~30+ resolved Qs (probably 6-8 weeks in, mid-July 2026), enough to ablate against. At that point also update the slug example in `tests/test_tournament_dates.py:127,131` (currently still references spring as the example slug in error-path messaging — harmless but stale).

### Mixture model parameterization for numeric questions

Instead of asking LLMs for 11 percentiles (which they find unnatural), ask them to
parameterize a mixture of distributions: specify 2-3 components with means, stds, and
weights. This naturally produces smoother, better-shaped CDFs.

Mantic uses this approach and reports good results. The LLM selects components capturing
different scenarios, and the final prediction is a weighted combination.

Would require changes to the numeric prompt, parsing, and CDF construction pipeline.

### Aggregation strategy improvements

> **Status 2026-05-29: STACKER NOW DISABLED ON ALL TYPES (default off in code).** Numeric was
> already off (ablation: median > stack CRPS, p=0.042). Binary + MC now off too — the ablation
> binary result was a *tie* (p=0.496), so this is a low-risk default (tie-at-best + compute cost),
> NOT a measured harm; the binary/MC treatment effect is unmeasured on the current stack. The
> code default flipped to `default=False` so the stacker only runs on explicit opt-in. **Revisit
> when post-2026-04-27 (marker-era) resolved questions exist** to measure the real treatment effect.
>
> **Status 2026-05-10:** Spread-aware aggregation (item 3) is **SHIPPED** as
> CONDITIONAL_STACKING (April 2026); the prob-range trigger metric is durably
> justified by May ρ=0.616 disagreement-error. Post-aggregation shrinkage toward
> 50% (item 2) is **explicitly killed** — costs correctly-confident predictions to
> fix overconfident ones, and the May data did not replicate the NO-bias at the
> larger N. Per-type weighting (item 4) is **LOW-PRIORITY, deferred to Q3+** —
> May data showed gemini-3.1-pro's binary-vs-numeric asymmetry, but only one
> model fits the pattern, infra doesn't exist, and the next-tournament roster
> will likely refresh this model anyway; revisit only when ≥2 active models show
> the asymmetry on ≥100 binary AND ≥30 numeric records each. Trimmed mean (item
>
> 1) remains untested — keep on backlog.

Ideas from analysis (lower priority since prompt changes address the bigger issues):

- Trimmed mean (drop highest + lowest, mean of middle): robustness of median with
  better signal preservation. With 6 models, could drop top and bottom, mean of 4.
- ~~Post-aggregation shrinkage toward 50% (~15-20%)~~ — **KILLED 2026-05-10.** May
  data did not replicate the NO-bias finding at n=109. Shrinkage costs well-
  calibrated extremes to fix a problem that didn't recur.
- Spread-aware aggregation: widen uncertainty when models disagree rather than just
  picking the middle. **SHIPPED as CONDITIONAL_STACKING.**
- Weighted aggregation by historical model performance (per question type).
  **Deferred — see status note above.**

Need more data (more resolved questions) to confidently evaluate these.

### Per-model peer ranking: GPT-strong / Claude-weak on binary (2026-05-29, NEEDS BENCHMARK before acting)

> A peer-score recompute (`scratch/residual_2026-05-29/dim_peer_recompute.md`, method validated
> exact against `spot_baseline_score`; reviewed in `review_peer_analysis.md`) ranked base models on
> **peer-equivalent** (the metric that matters), not Brier. On binary (spring-aib-2026, n≈150):

- **GPT models carry the binary ensemble** (gpt-5.1 +19, gpt-5.2 +17 peer; CIs exclude 0). The
  **Claude pair is the binary drag** (claude-opus-4.6 −9, claude-opus-4.5 +2.2). The confound was
  checked and runs the *wrong way* (Claude saw slightly easier questions), so the drag is genuine.
- **The story INVERTS on numeric** — Claude is *strong* there (opus-4.6 +24), gpt-5.1 weakest. So
  this is a binary-specific effect, NOT "Claude is bad." A blanket roster cut would hurt numeric.
- **Counterfactual ensembles (paired, in-sample):** dropping the Claude pair → **+5.94 binary peer
  [+1.0, +11.7]**; dropping claude-opus-4.6 alone → +3.66 (survives jackknife, the most robust
  sub-claim). GPT-only → +10.6 but that point estimate hinges on 1-2 questions — don't quote it clean.
- Corrected an earlier wrong claim: gpt-5.2 is **not** high-variance (lowest binary sd); the
  wildcards are gemini-3.1-pro (sd 96) and claude-opus-4.6 (sd 80).

> **Do NOT act on this without an intense out-of-sample benchmark.** It's in-sample (n=62 paired),
> multiple-comparisons exposed, epoch-confounded, and the roster has already rotated (current Claude
> slot is opus-4.5 + opus-4.6; new gpt-5.4 / grok-4.1-fast have n=3-12, uninformative). The credible
> reading is narrow: *on binary, the Claude pair is a measurable drag and the GPTs carry the
> ensemble.* The natural next step is a **prospective per-type model-inclusion benchmark** (GPT-heavy
> on binary, retain Claude for numeric), gated on out-of-sample peer — not a reweight off this slice.
> Relatedly: the stacker's own model (claude-opus-4.5) being a weak base binary forecaster is part of
> why disabling the binary stacker is low-risk (see Aggregation status above).

### Domain-aware CDF spread tuning

> **Status 2026-05-29: HOLD — measured on a STALE pipeline; re-measure on current version before ANY narrowing.**
> The residual analysis (`scratch/residual_2026-05-29/dim_numericpit.md`, `numeric_width_version_confound.md`)
> re-confirmed across two independent rosters that the *analyzed* CDFs are too wide (PIT std 0.24–0.26 vs ideal
> 0.289; 90% coverage 92–98%), and a uniform contraction k≈1.2–1.3 toward the median would hit the ideal. **BUT all
> analyzed forecasts (≤ 2026-04-13) ran with tail-widening ON at full strength (k_tail=1.25).** Prod flipped to
> k_tail=1.0 (identity) on **2026-05-12 (`b8d730f`)**, *after* the data — plus the mixture-distribution router went
> live and numeric stacking was disabled. So current prod is **already narrower** than what we measured; the bot's
> own calibration study showed widening-off moved PIT std 0.238 → 0.245. **Applying k≈1.3 on top of the current
> pipeline would overcorrect** — the over-width is in the *body* while deep tails are *already too thin* (1.5–6% mass
> vs 10% ideal), and the widening-off flip thinned tails further.
>
> **Before doing anything here:** (1) ship the PIT log-grid measurement-bug fix in `analysis.py::_interpolate_pit`
> (it mis-scores log-scaled / `zero_point` questions by up to 0.86 PIT — version-independent, do regardless);
> (2) add PIT std as a first-class `backtest.py` metric (it emits CRPS/log but not PIT, and doesn't persist the CDF);
> (3) run `make backtest_large` to get a **current-version** PIT measurement on fresh predictions; (4) only then,
> if still over-wide, choose a (smaller) narrowing factor with a per-side tail-mass floor. The *direction* (mild body
> over-width) may survive; the *magnitude* k=1.3 will not.
>
> Also note the 2026 finance carve-out **inverts** the old advice: financial questions were the *most* over-wide
> (cov90=100%), not the least — so the old "exclude finance/markets" guidance is wrong on current data.
>
> Reconciled along the way: the long-standing "PIT std 0.143" figure was just the April **n=11** subsample under the
> same metric (a ~2nd-percentile draw) — retire it; population is ~0.24–0.26.

Older (2026-05-10) framing, now superseded by the version-confound note above: PIT analysis found CDFs too wide; the
pipeline could use the forecastability classification (output by the prompt) to apply different tail-widening
parameters, or use the FORECASTABILITY tag to adjust smoothing / tail-mass allocation / post-hoc CDF scaling.

### Ideas reverse-engineered from high-scoring competitor bots (added 2026-06-26)

Source: a systematic dissection of 12 high-scoring forecast outputs from three competitor bots
(GreeneiBot2, Preseen-Atlas, SynapseSeer) captured in `/Users/flatljan/Documents/prompts/metac-examples-strong-bots-june-2026.md`,
analyzed via an ultracode workflow on 2026-06-26. Full report + grounding-critic verdict at
`scratch/competitor_analysis_2026-06-26/REPORT.md`. **Important caveat for all of these: the corpus
has NO resolution outcomes**, so every "why it helps" is a *mechanism* argument, never an
outcome-validated one — none of these is evidence that the competitor scores better. The one
genuinely shipped this session was the source-provenance trust ladder (prompt edit, in
`prompts.py:_SOURCE_PROVENANCE_LADDER`); everything below was deliberately deferred as
higher-risk-than-this-session and is gated on a benchmark before acting.

**1. Stacker deviation cap from the base-model median (experiment; needs benchmark).**
Bound the stacker's output to within K of the base-forecaster median — binary in probability
points, numeric as a percentile/location shift — mirroring the existing Platt deviation-cap
pattern, with K a new constant near the `CONDITIONAL_STACKING_*_THRESHOLD`s. Land it in
`aggregation_pipeline.py::_stacking_aggregate` (clamp just before `_apply_platt_calibration`),
new caps in `constants.py`.
*Grounding:* GreeneiBot2's fact-checker layer caps any consolidated update to ±20% of the
forecaster panel average (source line 1352: "any consolidated forecast should remain between
roughly 42% and 62%"). *Why it could matter for us:* our own ablation disabled the numeric
stacker because MEDIAN beat it (CRPS p=0.042, see Aggregation status above) — a reviewer
drifting too far from a well-calibrated central statistic is exactly the failure a cap prevents,
so a cap might let us safely re-enable numeric stacking. *Cautions:* (a) the whole point of
conditional stacking is to let the reviewer MOVE the number when one model is right — a cap that
is too tight defeats it, so ablate before shipping, do not assume benefit; (b) their "±20% of the
average" is multiplicative and unstable near 0/1 — use an ABSOLUTE points cap for binary, not a
percentage. **Gate:** `make backtest_medium` (or large) ablation of capped-stacker vs current
MEDIAN-default before shipping.

**2. Shared-reliance / consensus-fragility audit (experiment; higher-risk, needs benchmark).**
The hole: our spread-gate (`spread_metrics.compute_spread`) takes MEDIAN whenever the N
forecasters agree and **never asks WHY they agree**. If all 6 agree because they swallowed the
same unverified fact from shared research, that consensus is falsely confident and invisible to a
pure numeric-spread metric. This is the missing *mirror* of our existing disagreement branch
(`spread > threshold` → crux → targeted search → stack); the agreement branch has no counterpart.
Three operationalizations, cheap→expensive:
  - (A) **Prompt-only self-report** — add a `load_bearing_claims: [{claim, verified, source}]`
    field to the structured JSON each forecaster already emits (consumed by `tool_runner`); each
    forecaster names the 1-3 facts its forecast most depends on and whether it could verify them.
    Surfaces shared reliance, improves rationales, zero aggregation change. Cheapest; could be done
    in a future prompt session without the benchmark gate (strict rationale-quality improvement).
  - (B) **Deterministic cross-forecaster flag** — if one unverified load-bearing claim appears in
    ≥K of N forecasters' self-reports, mark consensus fragile and FORCE the stacker/crux path to
    fire even at low spread. Reuses existing machinery; the stacker prompt already has the right
    language (`prompts.py:725`: "hedged consensus can reflect shared priors more than shared
    evidence") — it just never runs on agreement today. Hard part: clustering free-text claims
    across models is fuzzy (may itself need an LLM).
  - (C) **Dedicated consensus-auditor LLM** — one cheap pass over the N rationales on the
    low-spread branch: "do they agree from independent reasoning or shared reliance on claim X? is
    X verified?" Symmetric to the disagreement-crux extractor.
*Grounding (honest provenance):* this is **OUR idea, seeded by — not copied from — the competitors.**
Their fact-checkers do two things SEPARATELY that this fuses: (a) discount agreement as
non-independent when rationales overlap (SpaceX-stock line 2028: "their agreement should not be
treated as two independent signals because the rationales and inputs are highly overlapping";
Iran line 1330: "their rationales are very similar, so I would not treat them as fully
independent evidence"), and (b) notice both forecasters leaned on the same unverified figure
(Metaculus-predictions lines 36, 43: "both lean somewhat on an unverified GPT search count") —
but they respond to (b) by downweighting+widening, NEVER by gating aggregation. No competitor
passage fuses "shared reliance on one unverified fact" with "therefore discount the consensus."
*Caution:* (B)/(C) add cost to the COMMON cheap MEDIAN path (most questions agree), inverting our
"cheap path for agreement, expensive only for disagreement" design — that per-question cost bump
is the main reason it's deferred. **Gate:** benchmark (B)/(C) before shipping; (A) is the
low-risk starting point.

**3. Numeric "unverified-conflict → variance" rider (low priority; tension with our calibration).**
When the source-provenance ladder genuinely CANNOT adjudicate between two candidate values for a
load-bearing quantity, place mass across both rather than committing to one (widen the relevant
percentiles), with a materiality gate so it only fires when the gap exceeds plausible
measurement/timing noise.
*Grounding:* the Metaculus-predictions fact-checkers, facing a background count (3,856,697) vs an
unverified live count (3,895,701), widened rather than picking: "this discrepancy should widen
uncertainty and slightly reduce confidence … not secure enough to fully adopt" (line 36); "treat
the current count as uncertain, centered between these values but with extra variance … sigma
95000 to cover the current-count discrepancy" (line 43); and a quantified materiality test ("a
39004 same-day discrepancy is too large to dismiss as normal intraday growth").
*Why deferred / tension:* our verified PIT analysis says our numeric CDFs are ALREADY too wide
(why `TAIL_WIDEN_K_TAIL=1.0`, see Domain-aware CDF entry above), and our hedge-audit deliberately
*penalizes* widening-out-of-caution. The narrow defense is that this widens for a NAMED, specific,
quantified reason — which satisfies the hedge-audit's own "name specific evidence to widen"
carve-out rather than violating it — but it risks an LLM over-applying it (every question has
*some* source tension). Decided in design discussion (2026-06-26) NOT to ship as a standalone
widening license; the trust ladder + the existing gap-fill tiebreaker-search already adjudicate
most conflicts by authority. Revisit only as a tightly-scoped numeric-only A/B, never blanket;
**gate** on P10/P90 coverage of resolved numerics not regressing.

**Also observed but NOT pursued (logged so future sessions don't re-litigate):** binary-complement
coherence guard ("state implied No; confirm Yes+No=100%", cheap defect protection — candidate for a
future prompt session, see report rec #4); interrogate-the-resolution-source-quality prompt clause
(Preseen-Atlas "Nasdaq calendar dates are 'not official'", line 2679 — partially covered by our
trust ladder now placing "the question's own named resolution source" in tier A); numeric
partial-resolution incorporation ("fix the realized portion, put variance only on the remainder")
and reporting-vs-outcome-uncertainty clauses (report rec #5). **Explicitly rejected as conflicting
with verified data** (do NOT re-recommend): GreeneiBot2's one-sided anti-overprediction shave
`X=min(10−certainty, 0.2×estimate)` (our binary calibration slope flips 0.83→1.66 across rounds, so
a fixed shave helps one round and hurts the next); blanket sigma-widening (CDFs already too wide);
switching numeric to parametric mean/sigma representation (our percentile→PCHIP→CDF-space pipeline
strictly subsumes a single normal and additionally supports mixtures via OPTION B); the open-tail
"spike" grid-compliance trick (we solve grid validity deterministically in `pchip_cdf.py` and our
prompt already tells forecasters spiky tricks don't pay).

## Longer-term (significant R&D)

### Agentic deep research (ReAct loop)

> **In progress as of 2026-07-16**: the gap-fill v2 plan (near-term entry above;
> `scratch_docs_and_planning/agentic_gap_fill_v2_plan.md`) is exactly this — a bounded
> tool-loop second pass. The cost blocker below is resolved by budget caps (~$0.50/q target)
> and encouraged early-stop; selective activation happens via the template dry-run.
>
> **Direction confirmed by the 2026-07-16 lit survey (see the triage block near the top):**
> keep this a **shared** agentic research stage (one loop → detached artifact all forecasters
> read), NOT per-forecaster integrated research+forecast pipelines. BTF-2 (arXiv 2604.26106)
> found a strong prompt on good shared research edged the best self-directed integrated agent,
> and the integrated benefit was model-dependent (only the Opus-class model gained). Watch
> era-bucketed calibration alongside Brier when validating — agentic-research wins optimize
> pass@1 accuracy and can trade calibration for over-decisiveness.

Move from one-shot research to an iterative research agent that can: execute search queries,
evaluate results, identify gaps, execute follow-up queries, run code for analysis, and
synthesize findings. Smingers is moving this direction.

Main blocker: API costs. The canned Grok native search works adequately for most questions.
Agentic research would be most valuable for complex questions where a single search
doesn't surface the right information.

Could prototype with selective activation (only for questions where initial research
scores poorly on a relevance check).

### Prediction market integration (strong evidence, criteria/date-matched)

Direct API access to Polymarket, Kalshi, Metaculus community predictions (where available)
as one research input. Currently markets show up in web search results inconsistently.
Structured access would be more reliable.

Framing: markets are STRONG EVIDENCE, not a footnote. The forecaster prompts now instruct
models to anchor on a market whose resolution criteria and date MATCH the question, discount
proportionally to any specific mismatch, and — when the only difference is the resolution
DATE — explicitly extrapolate the market's probability to our date with a simple model
(constant-hazard / base-rate-over-time) rather than applying a vague haircut. Superseded the
old "not beholden to them" language after a referendum miss where a market sat at the correct
answer and the bot dismissed it.

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

- **YES-side shrink / any fitted directional calibration layer** — the P6 YES-overconfidence
  finding is real on pooled data but era-local (spring-2026 only; fall confident-YES buckets
  are ~100% accurate). All three pre-registered stability criteria failed; a fit-on-two-eras
  layer degraded held-out fall performance (+4.3e-3 mean Brier). Receipts:
  `scratch/residual_2026-07-08/experiments/ARM_B_FINDINGS.md`. One sub-finding survives as a
  hard rule: symmetric shrink was strictly worse in every era — never touch the NO side.
- **Anchor-guard as a clamp/gate** — models publish outside their own stated base-rate anchor
  on ~88% of forecasts (that is the normal outside → inside update, not a bug); precision on
  a binary "flag it" gate is ≤29% and the sign of the effect flips across eras. Surviving
  fragment: overshoot MAGNITUDE >15pp degrades Brier monotonically in both well-powered eras,
  so ship it as telemetry plus a prompt nudge only (telemetry shipped 2026-07-08). Receipts:
  `scratch/residual_2026-07-08/experiments/ARM_A_FINDINGS.md`.
- **Advisory (non-binding) critic passes** — pt2 natural experiment: GreeneiBot2's critique
  diagnosed our exact 41800 defect verbatim and the number never moved; laertes's correct
  42855 reference-class recomputation was half-overridden by its own forecasters. Corrections
  that BIND (formula/floor/structural) captured 30–130 spot-peer points; advisory captured
  ~0. Receipts: `scratch/residual_2026-07-08/ACID_TEST_VERDICT.md`. The critic-pass Near-term
  entry stays, but in binding-output form only.
- **Fixed-direction haircuts** — GreeneiBot2's always-downward haircut was best-of-group on
  42855 AND actively harmful on 41800 (it extremized a 12 down to 10 on a YES resolution).
  Any damping mechanism must push toward 0.5, not in a fixed direction. Consistent with the
  already-killed one-sided anti-overprediction shave.
- **"Median drowns the correct dissenter"** (re-confirmed dead) — the 7/9 washouts framing is
  survivorship bias; median remains at the non-oracle frontier per dim_peer-calibration across
  two additional pulls; and even laertes's 41800 "win" discarded its own best member and was
  saved by its floor, not by aggregation.
- **Anchor-floor guard on cheap tails** — the deterministic follow-up to the Arm A anchor-guard
  kill. The median-band variant sign-flips across eras (fall +1.68 / spring −2.19 total Brier
  at the 10pp margin) because 76% of parsed anchor bands are degenerate points, collapsing the
  clamp back into Arm A's buried point-anchor clamp. The union-band variant is sign-consistent
  but 100% top-5-concentrated (1–2 questions per era) and catches only 1/5 of the known misses.
  Decomposing fires shows same-side tail clamping — the mechanism the guard is named for —
  hurts BOTH well-powered eras. Revival condition: ≥50 current-roster binaries carrying the
  new structured `base_rate_anchor` telemetry (shipped 30bca2f) AND an era-stable,
  non-concentrated (top-5 < 50% of gross) replay. Receipts:
  `scratch/residual_2026-07-08/experiments/GUARD1_FINDINGS.md`.
- **No-market-no-extremize cap** — feasibility kill: the market-presence signal exists in
  essentially one era (fall is 89% NO_SIGNAL because the prediction-market provider was
  benchmark-suppressed; the canonical `## Prediction Market Snapshot` header appears in
  exactly 1 archived binary), so era-stability is un-testable for the market-conditioned form.
  Marketless fallback form = Arm A's point-anchor clamp drift bomb; strict form's gain is 3
  spring questions (74–98% of gross improvement) and the market gate itself adds ~nothing
  (with-cond vs. no-cond within ~0.2 Brier on spring). Revival condition: ≥~50 resolved
  binaries per era carrying the structured `## Prediction Market Snapshot` AND the market
  gate itself earning the delta the marketless form does not. Receipts:
  `scratch/residual_2026-07-08/experiments/GUARD2_FINDINGS.md`.
- **Signed deadzone haircut toward 0.5** (question closed permanently) — extends the Arm B
  symmetric-shrink kill to all thresholded / high-t forms. 0 of 24 (λ, t) grid cells help or
  are neutral in both well-powered eras; fall has no improving cell anywhere. Fire rate ≥30%
  at even the loosest deadzone (t=0.4) because the bot is a confident forecaster
  (median |p−0.5| ≈ 0.32–0.38) so confidence-keyed deadzones fire on the bulk, not a tail.
  LOTO fit on spring+summer degrades held-out fall +1.12 total Brier. Keeper: the
  signed-toward-0.5 direction constraint stands as a design rule (mirrors the fixed-direction
  haircut kill above). Revival condition: the ensemble's calibration profile qualitatively
  changes (a fall-like era where confident calls are wrong at scale) AND leave-one-era-out
  passes on every held-out era; do not re-grid λ/t on new data absent that. Receipts:
  `scratch/residual_2026-07-08/experiments/GUARD3_FINDINGS.md`.
- **Cross-cutting** — at mid-grid configs all three guards fire on the same spring miss
  cluster (42024 / 42304 / 41800 in particular; often 42018 / 42577 / 42855 / 41644 too), so
  they are one lever measured three ways rather than three independent findings. The shipped
  `30bca2f` prompt changes (status-quo derivation, conjunctive clause pricing, anchor/clause
  telemetry) target that exact cluster, so any future guard replay on post-`30bca2f` data
  must never fit on pre-`30bca2f` eras — that is the roster-drift bomb with a prompt-change
  fuse. Config-era bucketing (keyed on submission time) already handles this; the new prompt
  era starts at `30bca2f`.

## Killed / evaluated 2026-07-11 (blind-forecaster review + crowd-signal audit)

Context: blind (model-IDs-stripped) judge pass over the 2026-07-11 `test_bot` run
(`scratch/test_bot_july_11_2026 .md`), plus a live-API audit of surfacing crowd-signal
informativeness (forecaster counts / market liquidity) in research packets.

- **Metaculus `similar-posts` endpoint as a research provider** — KILLED. Audited live
  across 23 source posts / 160 related rows (`GET /api/posts/{id}/similar-posts/`). Two
  dealbreakers, both confirmed at scale: (1) the community-prediction VALUE is `null` for our
  bot account on 160/160 rows (`aggregations.recency_weighted.latest`), and (2) it returns
  ONLY open questions — 160/160 unresolved, even when the source has obvious resolved
  monthly/weekly siblings (probed TSA-weekly, cabinet-departure, WatchCharts) — so the
  base-rate payload we wanted is absent. What remains is title + `nr_forecasters` +
  un-followable link = decorative (tells a forecaster a related question *exists*, not what
  the crowd thinks). Match quality is also sharply bimodal: good for AI-frontier / US-politics
  / geopolitics / broad-finance, garbage for weather / sports / non-US elections / niche
  product events (a Boston-Marathon source returned the "2026 World Bog Snorkelling
  Championships" and a Zambian election), and the endpoint pads to exactly 8 with no relevance
  score so a provider couldn't filter the junk. Also rate-limits at ~10 rapid calls → 429.
  NOTE: exact-match Metaculus community-prediction surfacing is separately pointless in prod —
  tournament questions are bot-specific so there's no meaningful CP, on top of the bot-account
  API hiding (which itself is Metaculus defending against systematic CP scraping).

- **Resolved-sibling base-rate lookup** (`GET /api/posts/?search=<kw>&statuses=resolved`) —
  general version KILLED; a narrow self-history salvage is a LOW-PRIORITY optional experiment.
  Audited live 2026-07-11 (~110 calls). The hypothesis — "full-text search over resolved
  Metaculus questions = a leakage-safe base-rate library (past Fed cycles, Bitcoin, ACX
  classics)" — is dead on a dealbreaker: **Metaculus null-outs `question.resolution` (plus
  description/resolution_criteria/fine_print/CP) on every post this bot account did NOT itself
  forecast** (view-level, presumably AIB anti-cheating; verified ~15 detail fetches, zero
  counterexamples, no bypass via `with_cp`/`minimize`/`include_descriptions`). So the
  Metaculus-wide resolved corpus is unreadable to our token. Endpoint mechanics all work
  (`statuses=` plural; `resolution` null on list rows, populated on detail for
  bot-forecast posts; formats: binary `yes`/`no`, MC option string, numeric stringified-float
  or `above_upper_bound`/`below_lower_bound`, plus `annulled` rows to filter; server-side
  `scheduled_resolve_time__lt` / `open_time__lt` work, `actual_resolve_time__lt` and
  `order_by=-scheduled_resolve_time` are silently broken).
  **Salvage (optional, low priority):** a `forecaster_id=275109&statuses=resolved` self-history
  lookup over the bot's own ~770 resolved posts (~9mo of fall-aib + spring-aib + cup). AIB
  recycles templates heavily, so sibling quality is excellent for recurring indicators (TSA
  weekly volume, Fed funds bound, ISM PMIs) and repeated event-window binaries (cabinet
  departures, city-rain, AI-benchmark thresholds); naive title-as-query works (no LLM keyword
  step). Backtest-MEASURABLE (unlike prediction_market) via a `actual_resolve_time < question.
  open_time` date-filter rather than a hard `is_benchmarking` disable. Design if pursued:
  title-as-query, self-history filter, K=5, resolve-date guard, drop `annulled`. Ranked below
  recent pipeline work on impact/complexity — the indicator series it surfaces are already
  pulled by the resolution-source fetcher + financial-data provider (FRED/TSA.gov); its only
  UNIQUE value is structured prior-outcomes for repeated event-window binaries not otherwise
  in the briefing.

- **Crowd-signal informativeness surfacing (forecaster count + market liquidity)** — WORTH
  DOING, plan pending. Findings: `nr_forecasters` is already on the fetched question object
  (forecasting-tools `MetaculusQuestion.num_forecasters`, populated from the post payload the
  bot already fetches — zero extra HTTP). Market fetchers (`research/prediction_market.py`)
  already RECEIVE but DISCARD total volume / open-interest / liquidity / `uniqueBettorCount`,
  and the one liquidity number they DO print (`vol` = 24h volume) is ~always ≈0 for the
  long-horizon questions we forecast — systematically misleading. Proposed labels (raw shown
  alongside): Metaculus <30 thin / 30–49 decent / 50–99 good / ≥100 high-confidence;
  real-money <$5k thin / $5k–50k decent / >$50k deep (sub-$10k is bot-dominated, may raise the
  thin cutoff); Manifold on `uniqueBettorCount` <20/20–100/>100. Aggregators evaluated and
  SKIPPED: Metaforecast is shut down (offline July 2026, repo archived, search API erroring);
  Adjacent News covers only Kalshi+Polymarket (redundant) with unpriced API. PredictIt is an
  optional cheap politics-only add (free/no-auth, but price-only, no volume/liquidity). PMXT
  ("CCXT for prediction markets") is the thing to evaluate only if venue breadth becomes a
  binding constraint.

- **Open-bound tail-cramming on discrete/open-upper numeric** — FIX IN PROGRESS (see the
  dedicated review). gpt-5.6-sol and gpt-5.5 crammed ~20% / ~12.6% of probability mass onto
  the top displayed bin of Q38195 (open upper bound) instead of placing percentiles above the
  ceiling, because a prompt line ("Allowed range … Respect the explicit bounds") is read by
  literal-minded models as a hard cap that overrides the open-bound carve-out. Fix = prompt
  clarification (pending redundancy check vs the existing open-bound guidance series) + a
  WARN-only boundary-piling detector routed into GHA artifacts (EXTRACTION_RUNG-style
  telemetry). Model-family split was clean: both OpenAI models failed, all of
  Claude/Gemini/Grok handled it correctly.

## Instrumentation bugs

> **All three identified in May 2026 closing analysis are FIXED.** See commits in
> the working tree as of 2026-05-10. Verification: 1187 tests pass, lint clean.

1. ~~STACKER_OUTCOME tri-state marker~~ **FIXED 2026-05-10.** Producer now sets
   `_stacker_outcome[qid] = "primary" | "fallback_llm" | "fallback_median" | "skipped"`
   at the END of each path (after success), not at entry. `_create_unified_explanation`
   emits both the new tri-state `STACKER_OUTCOME=...` marker AND the legacy
   `STACKED=true|false` marker for one round of back-compat. Median-fallback path
   (which previously silently emitted `STACKED=true`) now correctly emits
   `STACKER_OUTCOME=fallback_median` + `STACKED=false`.
2. ~~Targeted-research header missing from comments~~ **FIXED 2026-05-10.** Root
   cause: `main.py:839` returned `research_report=research` instead of
   `research_report=combined_research` on the conditional-stacking-triggered branch.
   The `## Targeted Research (addressing model disagreement)` header lives in
   `combined_research` but never reached the published comment. One-line fix.
3. ~~`audit.py::emit_synthesis` KeyError on numeric-mixed cohorts~~ **FIXED 2026-05-10.**
   Type-aware skip for non-binary entries in the spread section (the previous code
   assumed all `ranked` entries had a `prob` key, which numeric ranking via
   `_rank_numeric` doesn't produce).

### New parser feature shipped this session — historical-header-aware detection

Three older code variants emitted recoverable stacker-output body signatures
that earlier residual analyses missed: `## Stacker Meta-Analysis` (current),
`## Meta-Analysis` (April-2 stacker ship era), and `# Meta-Analysis and
Synthesis` (earliest H1 variant). The new
`metaculus_bot.performance_analysis.parsing.parse_inferred_stacker_outcome`
detects all three plus the new tri-state marker plus the legacy marker.
This unlocked the May 2026 stacker treatment-effect estimate
(`analysis_stacking_historical_treatment.md`) — first measurable signal in
the project's history (n=8 stacker-ran, point estimate −0.090 Brier vs
counterfactual, P(stacker helped) = 89.8%).
