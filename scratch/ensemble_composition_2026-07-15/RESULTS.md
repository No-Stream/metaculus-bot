# Retrospective ensemble-composition screening — 2026-07-15

**Question:** using per-forecaster predictions recovered from the bot's own published
Metaculus comments on resolved questions, would the published ensemble aggregate have
scored better or worse under alternative compositions (leave-one-family-out, drop-grok,
drop-gemini, drop-both, top-3 families)? This is screening, not optimization — the
deliverable is "is any model clearly helping or hurting", with honest uncertainty.

**Reproduce:** `uv run python scratch/ensemble_composition_2026-07-15/ensemble_screen.py`
(reads the tagged collector dataset at `scratch/coherence_2026-07-15/perf_all_tagged.json`;
100% offline — no API, LLM, or paid calls). Full generated tables in `tables.md`,
machine-readable results in `results.json`.

## TL;DR

- **openai-gpt5 is the one clearly load-bearing family.** Removing it hurts in 6 of 8
  eras, with per-era CIs excluding zero in three of them and a pooled delta of
  −2.55 log points [95% CI −3.87, −1.27]. Keep.
- **gemini is not deadweight.** Pooled drop-gemini = −2.16 [−3.67, −0.58]; it hurts to
  remove in both large spring eras, and in the current summer era gemini-3.1-pro-preview
  is the *best individual member* (+51.6 mean log) with a binary-Brier degradation on
  removal whose CI excludes zero (+0.0127 [+0.0038, +0.0237]). The suspicion that
  gemini-3.1-pro is deadweight is not supported.
- **grok is the closest thing to deadweight, but it isn't hurting.** It is the worst
  individual member in every era it appears (grok-4-fast +45.7 vs peers ~55-60;
  grok-4.1-fast +18.6, worst; grok-4.3 +27.3, worst). Yet removal deltas straddle zero
  everywhere (pooled +0.24 [−1.12, +1.49]) — the median aggregate mostly ignores it.
  Evidence supports "contributes nothing measurable", not "actively harmful".
- **anthropic is era-dependent.** In spring-aib-2026 the opus-4.5/4.6 pair actively
  hurt: drop-anthropic in spring_5m_b = **+3.29** [+0.26, +6.47] with binary Brier
  improving −0.0164 [−0.0294, −0.0048], and the adjacent transitional era agrees. In
  the current summer era (opus-4.6/4.7/4.8) the sign flips back to helpful
  (−1.70 [−3.84, +0.66], opus-4.8 second-best individual member). No action indicated
  today, but the spring result is a reminder that a stale anthropic version can drag.
- **Nothing supports shrinking the ensemble.** drop-grok+gemini (4-model ensemble) in
  summer = −0.16 [−2.18, +1.86]; top-3-families (leave-one-question-out selection)
  never beats the full ensemble in any era where it's computable.

**The biggest caveat:** the *current* roster slots (grok-4.5 since 2026-07-08,
gpt-5.6-sol since 2026-07-09) have **zero resolved questions**. Everything above is
about the predecessor lineages (grok-4.3, gpt-5.4/5.5, etc.).

## Method

### Data

- Source: `scratch/coherence_2026-07-15/perf_all_tagged.json` — the collector dataset
  over fall-aib-2025 + spring-aib-2026 (reused from the 2026-07-08 pull of those closed
  tournaments) + summer-futureeval-2026 (fresh 2026-07-15 read-only pull). 694 resolved
  question records with per-model values recovered from the bot's published comments.
- Per-model values are **post-clamp, pre-aggregation** — exactly what the prod
  aggregation consumed. Binary: probability per model; MC: full option-probability dict
  per model; numeric: the declared percentile list per model (block-parsed post-2026-07,
  prose-parsed before).

### Baseline: replica, not published

The baseline for every comparison is the **replica** aggregate — recomputed from
recovered per-model values with the era-appropriate combine — rather than the published
number. Subset-vs-replica paired deltas then share every recovery artifact (parse noise,
missing members, no prod tail-widening), so the deltas isolate the composition effect.
Published-vs-replica agreement is reported separately as validation (below).

### Aggregation rules (mirrors prod)

- Binary: median of member probabilities, rounded to 3 decimals (mean for the earliest
  3-model era — verified exact on all 21 of its binary records). Matches
  `aggregation_strategies.aggregate_binary_median` / `aggregate_binary_mean`.
- MC: per-option median (mean in the mean era), renormalized — matches
  `_aggregate_mc_options`.
- Numeric/discrete: each member's 13/11-point percentile list → 201-point PCHIP CDF via
  `metaculus_bot.numeric.pchip_cdf.generate_pchip_cdf` (same machinery prod and the
  audit module use), then pointwise **median in CDF space** across members, then the
  prod ensemble postprocess (clip, monotone, open/closed endpoint pinning, min-step
  ramp) — mirrors `numeric/utils.aggregate_numeric` + `_postprocess_ensemble_cdf`.
- Platt calibration is identity in every era (checked-in params), so omitting it is
  exact, not an approximation.

### Scoring

Metaculus-style log-score family from `metaculus_bot.scoring_common`, all higher-better:
`binary_log_score`, `mc_log_score` (log score of the resolved option), and
`numeric_log_score` (PMF-bucket log score at the resolved value / out-of-bounds bucket).
Binary comparisons additionally report ΔBrier as a second lens. Discrete questions are
scored on the 201-point grid for **both** arms (prod resamples to the native bin count
before submitting; using a consistent grid keeps the paired deltas meaningful even
though the absolute levels differ slightly from Metaculus's own).

### Eras (roster changes = boundaries, submit-timestamp driven)

Derived from `git log -p -- metaculus_bot/llm_configs.py` cross-checked against the
model names actually observed in each comment. Prompt tweaks did not start eras;
roster/aggregation changes did.

| era | window | roster (families) | combine | n scored |
|---|---|---|---|---|
| fall_mean3 | ≤2025-09-11 (+3-model stragglers to 09-20) | gpt-5, o3, sonnet-4 | mean | 41 |
| fall_5m | 2025-09-12..09-20 | + kimi, qwen | median | 36 |
| fall_6m | 2025-09-21..12-31 | + grok-4-fast(→4.1); sonnet-4→4.5, gpt-5→5.1 in-era | median | 277 |
| spring_5m_a | 2026-01-01..02-07 | opus-4.5, 2×gemini-3, gpt-5 + 5.2 | median | 69 |
| spring_trans | 2026-02-08..02-20 | transitional churn | median | 37 |
| spring_5m_b | 2026-02-21..04-03 | opus-4.5+4.6, gemini-3.1-pro, gpt-5.1+5.2 | median | 138 |
| spring_6m | 2026-04-04..05-18 | + grok-4.1-fast, gpt-5.4 | median | 13 |
| summer_6m | 2026-05-19..07-14 | gpt-5.4+5.5, opus-4.6+4.7(→4.8), gemini-3.1-pro, grok-4.3 | median | 45 |

In-era version swaps within the same vendor slot (sonnet-4→4.5, opus-4.7→4.8,
grok-4-fast→4.1-fast) were deliberately NOT made era boundaries — the family-level LOO
absorbs them, and splitting further would shred the n.

### Model families (judgment call, stated)

grok-4-fast/4.1-fast/4.3 → **grok**; gpt-5/5.1/5.2/5.4/5.5 → **openai-gpt5**; o3 kept
separate as **openai-o3** (different lineage, coexisted with gpt-5 in fall);
claude sonnet-4/4.5 + opus-4.5/4.6/4.7/4.8 → **anthropic** (vendor-slot lineage; some
eras run two anthropic slots, so LOO then removes both — the per-era tables list exact
members); gemini-3-flash/3-pro/3.1-pro → **gemini**; kimi, qwen as themselves.

### Comparisons and inference

Per era: leave-one-family-out for every family (only on questions where that family
actually forecast, and only when ≥2 members remain), drop-grok+gemini where both exist,
and a top-3-family subset with **leave-one-question-out selection** (family ranking
recomputed per question excluding that question's own scores — no double-dipping; the
comparison is skipped in eras with ≤3 families where it degenerates to the full
ensemble). Paired per-question deltas, bootstrap CIs over questions (4000 reps, seed
fixed). **48 CI-bearing comparisons are reported; at 95% coverage ~2-3 would be
expected to exclude zero by chance alone.** The findings called out in the TL;DR are
the ones that replicate across adjacent eras / pooled cuts / both scoring lenses, not
single-cell hits.

## Validation (recovery fidelity)

- **Binary:** replica == published on 365/369 non-stacked questions (mean |Δp| =
  0.0003). The 4 misses (3 ≥0.005) are spring records where a soft-deadline-dropped
  forecaster's bullet survived in the comment or vice versa.
- **MC:** replica score vs the published `mc_log_score` correlates at 1.000 (n=68),
  mean diff +0.04.
- **Numeric:** replica-vs-published score diff has median ≈ 0 in every era, but a
  right-skewed mean (+8.9 pooled): the replica omits prod post-processing that was
  active in pre-2026-05-12 eras (tail widening) and scores discrete on the 201-grid.
  In the current summer era, where prod post-processing is minimal, the replica is
  near-exact (mean −0.19, median −0.00). Because both arms of every comparison go
  through the identical replica pipeline, these level offsets cancel in the paired
  deltas.
- The 19 pre-2025-09-12 binary records confirmed the MEAN-era combine exactly (21/21
  mean-exact).

## Coverage and losses

Of 694 records: **656 usable**, 38 dropped:

| reason | n |
|---|---|
| no members recoverable (12 fall group-posts with no comment at all, 18 stacked-collapsed, misc parse failures) | 31 |
| <2 attributed members after anon filtering | 7 |

Member-level and stacked-question losses (from `drop_counts` in results.json):

- **Stacked-published questions: 22 total, 18 unrecoverable, 4 recovered.** Spring-era
  stacked comments (12 binary, 2 MC, 4 numeric) collapse everything — summary bullet
  AND rationale attribution — to "Forecaster 1", so base values are gone. The 4
  summer-era stacked records (3 binary, 1 MC) recover base values from
  `per_base_model_forecasts`; those are self-declared pre-parse values (small deltas
  possible) and are flagged with an unstacked-only sensitivity split in the summer
  table — the split doesn't change any conclusion.
- **Anonymous "Forecaster N" labels:** 37 attributed by elimination against the era's
  modal roster (single anon member + single missing modal model). 18 of those (fall_6m)
  fell outside the candidate version's date window and were kept at family level only
  (`unattributed:openai-gpt5`) — right family for LOO, version unknowable.
- 2 member-level PCHIP failures; a handful of unparseable member strings.
- 12 fall numeric records have no comment at all (group posts whose comment landed
  elsewhere); they are in the "no members" bucket.
- Comment middle-trimming (>150k chars) is why some fall numeric members carry 8
  instead of 11 percentiles (33 lists); members with ≥5 distinct percentiles were kept
  — PCHIP handles sparse percentile sets, and dropping them would have biased against
  verbose models.

## Per-era results (abridged — full tables in `tables.md`)

Δlog = mean paired (subset − replica) log score; negative = removal hurts (family was
helping). ΔBrier on binary subset: positive = removal hurts.

### openai-gpt5 (drop it and...)

| era | n | Δlog [95% CI] | ΔBrier |
|---|---|---|---|
| fall_mean3 | 41 | −2.09 [−6.62, +2.08] | +0.005 |
| fall_5m | 36 | **−6.42 [−13.25, −1.00]** | **+0.012 [+0.001, +0.026]** |
| fall_6m | 275 | **−1.75 [−3.08, −0.53]** | **+0.006 [+0.002, +0.011]** |
| spring_5m_a | 69 | +1.55 [−3.12, +6.08] | −0.002 |
| spring_trans | 37 | −8.32 [−19.24, +0.50] | +0.026 |
| spring_5m_b | 137 | **−5.18 [−9.13, −1.90]** | **+0.023 [+0.011, +0.036]** |
| spring_6m | 13 | −0.18 [−6.57, +4.91] | +0.002 |
| summer_6m | 45 | +1.01 [−2.31, +3.78] | −0.003 |
| **pooled** | 653 | **−2.55 [−3.87, −1.27]** | |

Consistently load-bearing; the summer point estimate is mildly positive but well inside
noise (n=45, and gpt-5.4 was the weaker of the two openai slots there).

### gemini

| era | n | Δlog [95% CI] | ΔBrier |
|---|---|---|---|
| spring_5m_a | 69 | −2.13 [−6.50, +2.88] | −0.006 |
| spring_trans | 37 | **−4.12 [−9.00, −0.75]** | +0.006 |
| spring_5m_b | 138 | **−2.05 [−3.72, −0.41]** | +0.006 |
| spring_6m | 13 | −0.05 [−1.65, +2.08] | −0.000 |
| summer_6m | 40 | −1.45 [−4.03, +1.32] | **+0.013 [+0.004, +0.024]** |
| **pooled** | 297 | **−2.16 [−3.67, −0.58]** | |

Helping in every era with meaningful n. In summer, gemini-3.1-pro-preview is the top
individual member (+51.6 mean log over 40 questions; it missed 5 questions to
soft-deadline drops — that unreliability is the honest knock on it, not forecast
quality).

### grok

| era | n | Δlog [95% CI] | ΔBrier |
|---|---|---|---|
| fall_6m | 270 | +0.36 [−1.17, +1.71] | −0.004 [−0.008, −0.000] |
| spring_6m | 12 | +1.07 [−0.73, +3.29] | −0.000 |
| summer_6m | 45 | −0.65 [−5.32, +2.78] | −0.007 |
| **pooled** | 327 | +0.24 [−1.12, +1.49] | |

Never load-bearing anywhere; the one CI that grazes zero (fall_6m binary Brier
*improving* on removal) is the sole hint of active harm, and it doesn't replicate in
later eras. Meanwhile grok is the worst-scoring individual member in all three eras it
appears. Diversity value to the median: not detectable at these n.

### anthropic

| era | n | Δlog [95% CI] | ΔBrier |
|---|---|---|---|
| fall_mean3 | 41 | −1.68 [−9.77, +4.14] | +0.010 |
| fall_5m | 36 | +1.03 [−1.86, +4.06] | +0.003 |
| fall_6m | 274 | −1.34 [−3.01, +0.07] | +0.001 |
| spring_5m_a | 66 | +1.53 [−1.83, +5.14] | +0.001 |
| spring_trans | 37 | +4.18 [−1.10, +10.36] | **−0.018 [−0.040, −0.001]** |
| spring_5m_b | 137 | **+3.29 [+0.25, +6.47]** | **−0.016 [−0.029, −0.005]** |
| spring_6m | 13 | +0.29 [−5.36, +5.64] | +0.003 |
| summer_6m | 45 | −1.70 [−3.84, +0.66] | +0.005 |
| pooled | 649 | +0.36 [−0.81, +1.52] | |

The pooled null hides real era structure (this is why era-bucketing is mandatory):
helpful in fall (sonnet-4.5) and summer (opus-4.8), actively harmful in spring
(opus-4.5/4.6 — both individual members ~12 log points below the era's gpt slots).
Note spring LOO removes BOTH anthropic slots at once (2 of 5 members), so it's a
bigger surgery than other families' LOO.

### Hypothesis subsets and top-3

- **drop_grok+gemini (summer):** −0.16 [−2.18, +1.86]. A 4-model ensemble would have
  been statistically indistinguishable over this sample — but the point estimate is
  negative and the gemini-side evidence above argues against it.
- **top3_families_LOQO:** +1.60 (fall_5m), +1.13 (fall_6m), +0.98 (spring_6m),
  −0.65 (summer). Never a significant improvement; ensembling breadth is not obviously
  costing anything, and honest per-question selection doesn't beat just keeping
  everyone.
- **o3 (historical):** pooled −1.33 [−2.69, −0.08] — it was quietly load-bearing in
  fall alongside gpt-5. kimi (+0.47) and qwen (−0.39): nulls.

## Conclusions (plain prose)

1. Keep the openai flagship slot(s); it has been the most consistently load-bearing
   family across three tournaments.
2. Keep gemini-3.1-pro on forecast quality — the deadweight suspicion is not supported;
   it is currently the best individual scorer. Its real cost is reliability (missed 5
   of 45 summer questions), which is an ops issue, not an ensemble-math one.
3. Grok is where the evidence points if a slot must be freed: worst individual member
   in every era, no detectable contribution to (or drag on) the median aggregate.
   Swapping it for a candidate with better solo scores has upside and, on these data,
   ~zero downside. But note this is grok-4.3 evidence; grok-4.5 has no resolutions yet.
4. Watch the anthropic slots for version staleness — the spring opus-4.5/4.6 drag was
   large and CI-solid; the current opus-4.8 looks healthy.
5. Don't shrink the ensemble on this evidence; no subset reliably beats the full
   roster.
6. With 48 comparisons at 95%, expect 2-3 false positives among the isolated
   single-cell hits; only the patterns that replicate across eras (points 1, 2, and
   the spring anthropic drag) deserve weight.

## Known limitations

- **No data on the current roster.** grok-4.5 and gpt-5.6-sol (both swapped in
  ~2026-07-08/09) have zero resolved questions. This analysis screens lineages, not
  today's exact models.
- Counterfactual, not causal: removing a member could have changed stacking triggers,
  min-forecaster guards, and (for stacked questions) the stacker's output. We replay
  the median path only; stacking is disabled in prod now, which makes the median
  counterfactual the right one going forward.
- 18 of 22 stacked-published questions unrecoverable (spring format collapsed base
  values); the recovered 4 use self-declared pre-parse values.
- Numeric replica omits era-specific prod post-processing (tail widening pre-2026-05-12,
  discrete native-grid resample) — cancels in paired deltas, but absolute per-era log
  levels are not comparable to Metaculus's own scores in early eras.
- Small eras (spring_6m n=13, fall_5m/spring_trans n≈36) are directional at best.
- Soft-deadline drops mean LOO for family m runs only on questions where m actually
  forecast; families with many drops (gemini in summer) are screened on a slightly
  easier/different question mix than always-present families.
