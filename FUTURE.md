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

## Near-term (worth exploring soon)

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

## Medium-term (requires more exploration)

### Dependency hygiene: version-floor bumps + uv migration (added 2026-06-01)

Deferred from the 2026-06-01 desloppify code-health pass (which did only behavior-neutral pyproject hygiene: dropped unused `python-decouple`, removed the unused `litellm[proxy]` extra, declared the directly-imported `scipy`/`pandas`/`pydantic` that were previously only transitive via `forecasting-tools`).

Two follow-ups intentionally left for a separate, gated PR:

1. **Raise version floors to current-installed** (e.g. `litellm ^1.80` vs the current `^1.59.1` floor, `openai` to latest, and evaluate moving `forecasting-tools` off the hard-pinned `0.2.54`). This is **forecast-affecting**: a litellm/openai/forecasting-tools behavior change can subtly shift model outputs and therefore predictions. **Gate:** run a medium backtest (`make backtest_medium`) before and after the bump and confirm scores don't regress before shipping. Do NOT bump blind.
2. **Migrate dependency management from poetry to `uv`.** Larger refactor (pyproject `[tool.poetry]` → PEP 621 `[project]`, regenerate lockfile, update Makefile `install`/`run`/`test` targets and CI). An orphaned 136-byte `uv.lock` stub + `[tool.uv]` block (declaring a contradictory `requires-python >=3.12`) were removed in the 2026-06-01 pass so the repo has one source of truth (poetry); a real uv migration would regenerate the lockfile from scratch. Worth doing for speed + the team's broader uv standardization, but it's pure tooling churn with no forecast impact — schedule when there's appetite for a no-functional-change infra PR.

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

## Longer-term (significant R&D)

### Agentic deep research (ReAct loop)

Move from one-shot research to an iterative research agent that can: execute search queries,
evaluate results, identify gaps, execute follow-up queries, run code for analysis, and
synthesize findings. Smingers is moving this direction.

Main blocker: API costs. The canned Grok native search works adequately for most questions.
Agentic research would be most valuable for complex questions where a single search
doesn't surface the right information.

Could prototype with selective activation (only for questions where initial research
scores poorly on a relevance check).

### Prediction market integration (read, not anchor)

Direct API access to Polymarket, Kalshi, Metaculus community predictions (where available)
as one research input. Currently markets show up in web search results inconsistently.
Structured access would be more reliable.

Important: should be presented as one noisy input, not an anchor. The prompt already says
"not beholden to them" and our analysis found this works reasonably well.

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
