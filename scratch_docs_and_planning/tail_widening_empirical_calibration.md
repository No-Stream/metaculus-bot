# Tail Widening Empirical Calibration

Date: 2026-05-12
Author: data-analysis agent
Source notebook: `scratch/tail_widening_calibration_2026-05-12/tail_calibration.ipynb`
Data: `scratch/analysis_2026-05/performance_data.json` (43 resolved numeric questions, Feb–May 2026)

## TL;DR

- **Drop `k_tail` to 1.0 (disable widening).** Under this dataset the current `k_tail=1.25` pushes PIT std slightly further from the uniform ideal (0.289), not closer. The effect is small in aggregate but directionally consistent, and it's strongest on non-financial questions.
- **Do not roll out per-category `k_tail`.** Per-category CIs overlap; segment-specific tuning isn't supported by this sample size (n≤26 per segment).
- **`span_floor_gamma` is a no-op in this dataset.** Every value from 0 to 2 produces identical PIT stats; the declared 5–10 and 90–95 spans from the ensemble are already wide enough that the floor never binds. Keep it at 1.0 or drop it to 0 — either way, no impact today.
- **Flag: the `tail_widening.py:116` short-circuit makes `k_tail<1.0` a no-op** (no narrowing path). If we ever want to sharpen, we'd need to implement it. Not needed right now based on this data, but the docstring overstates what `k_tail<1` does.

## Data pull

- 43 resolved numeric questions, pulled from `scratch/analysis_2026-05/performance_data.json` (already collected by `performance_analysis.cli`). Resolution dates span 2026-02-04 through 2026-05-06. All 43 have parsed resolutions, 201-point submitted CDFs, and per-model declared percentiles (11 points).
- Category distribution: Economy & Business (26), Sports & Entertainment (4), Unknown (4), Politics (3), Environment & Climate (3), Geopolitics/Social Sciences/Health (1 each).
- 6 questions were stacked (stacker produces one combined 11-point curve, labeled "Forecaster 1"); 37 were not stacked (per-base-model percentiles averaged into an ensemble pre-widening curve).
- Financial flag (title keyword match on price/stock/futures/yield/index/VIX/oil/gold/etc.): 26 financial, 17 non-financial.

## Method

1. **Recover pre-widening curve.** For each question, reconstruct the ensemble-average 11-point declared curve from `per_model_numeric_percentiles`. For stacked questions this is the stacker's declared 11-point output. Both are emitted in the comment body before `sanitize_percentiles` (the widening step runs after that).
2. **Simulate a `k_tail` sweep.** Apply `metaculus_bot.tail_widening.widen_declared_percentiles` directly to each pre-widening curve at `k_tail ∈ {0.5, 0.55, …, 2.0}` (31 values) × `span_floor_gamma ∈ {0, 0.5, 1.0, 1.5, 2.0}`.
3. **Compute PIT.** Interpolate a piecewise-linear CDF through the widened 11 percentiles plus range endpoints (closed: add (L, 0) / (U, 1); open: implicit 2.5%/97.5% mass toward bounds). PIT = CDF value at the resolution.
4. **Validate.** Compare the simulated k=1.25 PIT to the PIT computed directly from the actually-submitted 201-point CDF. Mean difference: 0.003, std of difference: 0.018 — the simulation tracks the real pipeline well (the PCHIP smoothing in the real pipeline adds small noise, but the tail geometry is preserved).

Ideal PIT statistics under perfect calibration and n=43: mean 0.5, std 0.289, 50% coverage 0.5, 90% coverage 0.9.

## Baseline (current production, k_tail=1.25)

| Segment | n | mean PIT | std [95% CI] | 50% cov | 90% cov |
|---|---|---|---|---|---|
| All | 43 | 0.524 | 0.241 [0.192, 0.271] | 0.60 | 0.98 |
| Financial | 26 | 0.528 | 0.261 [0.195, 0.293] | 0.54 | 0.96 |
| Non-financial | 17 | 0.519 | 0.212 [0.118, 0.261] | 0.71 | 1.00 |

Financial is near-ideal on all three metrics. Non-financial shows the classic overwide signature: PIT std noticeably below 0.289, too many PITs in the 25–75% band (cov50 = 0.71 vs. ideal 0.5), 90% cov saturated at 1.0 (vs. ideal 0.9). This is consistent with the FUTURE.md narrative that non-financial is overwidened.

Note: these numbers differ from the FUTURE.md values (std 0.143, 50% cov 57.1%, 90% cov 98.2%). That analysis appears to have used a different question pool or scaling convention — the 90% cov lines up, but std and 50% cov don't. I ran the calibration against the current dataset; if the FUTURE.md pool is reconstructible we should reconcile before acting on aggressive changes.

## k_tail sweep (span_floor_gamma held at 1.0 — no effect)

| Segment | k=1.0 | k=1.10 | k=1.25 (current) | k=1.5 | k=2.0 |
|---|---|---|---|---|---|
| All — std | **0.245** | 0.242 | 0.238 | 0.234 | 0.228 |
| All — \|std − 0.289\| | **0.044** | 0.047 | 0.051 | 0.055 | 0.061 |
| Financial — std | **0.266** | 0.262 | 0.258 | 0.254 | 0.247 |
| Financial — \|std − 0.289\| | **0.023** | 0.027 | 0.031 | 0.035 | 0.042 |
| Non-fin — std | **0.213** | 0.210 | 0.206 | 0.202 | 0.197 |
| Non-fin — \|std − 0.289\| | **0.076** | 0.079 | 0.083 | 0.087 | 0.092 |

**Every segment prefers `k_tail=1.0`** by the "minimize |std − 0.289|" criterion. Widening makes the calibration worse on the std target in every cut.

### Does the tail look better at higher k?

On the 13 tail-resolved questions (pre-widening PIT < 0.2 or > 0.8), higher `k_tail` does pull PITs toward the center — mean |PIT − 0.5| drops from 0.386 (k=1.0) to 0.352 (k=2.0). So widening does what it's designed to do at the tails. The problem is that **only 13 of 43 questions have tail-resolved outcomes**, and for the other 30 widening does nothing (resolution in [20%, 80%] range, outside the tail-weight ramp). In aggregate, widening mildly compresses std without improving PIT uniformity.

The counterargument: a larger dataset with a higher tail-hit rate might swing this toward "widening helps." With 43 questions and the current hit distribution it doesn't. **We should revisit after ~150 resolved numerics.**

## span_floor_gamma

In this dataset, `span_floor_gamma ∈ [0, 2]` produces identical PIT stats at every k_tail value. The reason: the ensemble-averaged declared 5-10 and 90-95 percentile spans are already wide enough that the floor constraint `(p05 − p025) >= gamma * (p10 − p05)` doesn't bind. The floor would only kick in if a forecaster gave very sharp tails (p025 ≈ p05), which this ensemble doesn't produce.

Recommendation: **drop `span_floor_gamma` to 0**, or leave it at 1.0 — no observable downstream effect either way. If we ever see models that give sharp tails, the floor would matter; for now it's dead code.

## k_tail < 1.0 (narrowing) is a no-op

`tail_widening.py:116`:

```python
if not percentile_list or (k_tail <= 1.0 and span_floor_gamma <= 0.0):
    return percentile_list
```

And at line 150:

```python
if k_tail > 1.0:
    widened_x = np.array([inv(y) for y in widened_y_arr], dtype=float)
else:
    widened_x = x_vals.copy()
```

The ramp-weight logic in `_tail_weight` returns `>= 0` always (no narrowing branch). So `k_tail < 1.0` is equivalent to `k_tail = 1.0`: just the span_floor floor. To actually sharpen tails you'd need to let `k_eff = 1.0 + k_delta * w` go below 1.0 (currently `k_delta = max(0.0, k_tail - 1.0)`) AND let the inverse-transform branch run for any k_tail != 1.0. **This is worth a comment in the docstring** — "values < 1.0 are clamped to the identity transform" — or a refactor if narrowing ever becomes desired.

## Category-level breakdown at k_tail=1.25

| Category | n | mean PIT | std | cov50 | cov90 |
|---|---|---|---|---|---|
| Economy & Business | 26 | 0.497 | 0.218 | 0.65 | 1.00 |
| Sports & Entertainment | 4 | 0.440 | 0.337 | 0.25 | 1.00 |
| Unknown | 4 | 0.597 | 0.252 | 0.75 | 0.75 |
| Politics | 3 | 0.716 | 0.177 | 0.33 | 1.00 |
| Environment & Climate | 3 | 0.420 | 0.169 | 0.67 | 1.00 |
| Geopolitics / Social / Health | 1 each | — | — | — | — |

n per category is too small for meaningful per-category tuning. Economy & Business (n=26) is the only cell with any statistical weight, and it's already decent (std 0.218, slightly overwide).

## Recommendation

**Ship: `TAIL_WIDEN_K_TAIL = 1.0` (disable widening). Keep `span_floor_gamma = 1.0` (no-op today, cheap insurance against sharp-tail declarations in future models).**

Rationale:

- Point estimate of std is closer to 0.289 at k=1.0 than k=1.25 in every segment.
- The 50% and 90% coverage stats also move toward ideal at k=1.0 (cov90 becomes 0.91 for the full sample, vs. 0.98 at k=1.25 — closer to the 0.90 target).
- No observed benefit from widening in aggregate.
- Individual models' declared percentiles, after ensemble averaging, already embed sufficient tail uncertainty.

**Do NOT ship (at least not yet):**

- (b) Per-category `k_tail`. Segments are too noisy (n≤4 for most categories outside Economy).
- (c) Narrowing (`k_tail < 1.0`). The code doesn't support it, and the data doesn't demand it — the issue is "slightly too wide" not "dangerously overwide". Would need a code change AND a larger dataset.

## Caveats

1. **Small sample per segment.** 17 non-financial questions is not a lot; bootstrap 95% CI on PIT std for non-financial at k=1.0 is [0.118, 0.275] — the ideal 0.289 is just outside the upper bound, not definitively ruled out.
2. **Category imbalance.** 26 of 43 questions are Economy & Business (mostly financial-market-like). The calibration conclusions weight that segment heavily.
3. **Simulation vs. reality gap.** My PIT-from-widened-declared simulation approximates the real pipeline (which widens per-model, then aggregates via PCHIP). Spot-checked against the actually-submitted 201-point CDF: mean |PIT_sim − PIT_submitted| = 0.018. Good enough for directional conclusions, not good enough for precise tuning.
4. **Temporal drift.** Questions span Feb–May 2026; the bot's stacking config changed during that window (aed2670 reminder scheduling, 3546dce gemini grounding, etc.). Calibration could differ for questions resolved under different configs.
5. **Tail-weight interaction.** The `tail_start=0.2` parameter means widening only affects p ∉ [0.3, 0.7]. For questions where the resolution is near the median, widening has zero effect on PIT — this inflates the noise floor of any k_tail signal.

## Artifacts

- `scratch/tail_widening_calibration_2026-05-12/tail_calibration.ipynb` — full analysis notebook
- `scratch/tail_widening_calibration_2026-05-12/pit_by_question.csv` — per-question PIT at k ∈ {1.0, 1.1, 1.25, 1.5, 2.0}
- `scratch/tail_widening_calibration_2026-05-12/pit_std_by_ktail.png` — PIT std vs. k_tail by segment
- `scratch/tail_widening_calibration_2026-05-12/pit_hist_by_ktail.png` — PIT histograms at k ∈ {1.0, 1.25, 1.5, 2.0}
- `scratch/tail_widening_calibration_2026-05-12/pit_hist_final.png` — comparison of k=1.0 vs k=1.25 histograms
