# Prod-ish Ablation Re-Run Plan

**Goal:** re-run the 5-arm aggregation ablation on a paid 3-model prod-ish ensemble to settle whether the binary stack-aug signal (P=0.954 on free-tier) replicates, and to characterize numeric/MC behavior with realistic models. Reuse the existing 88-question manifest + cached research; only forecast→score re-runs.

Key design constraints (from user):
- **Fail-fast.** No fallbacks anywhere. We use the operator's personal `OPENROUTER_API_KEY` directly via plain `GeneralLlm`, no donated-key wrapper. Failures abort the question; we resume manually.
- **Save outputs as we go.** Per-question, per-arm cache writes after each successful step so a mid-run failure loses at most the in-flight question.
- **Median-basic baseline. Paired comparisons throughout.** Forest plot is paired-delta vs `median_basic`; Bayesian + frequentist tests for the headline pairs.
- **Marimo notebook with strong narrative.** Not a kitchen-sink dump. Top-level figure → headline conclusion → per-type breakdowns → caveats. Confidence intervals everywhere.
- **Concurrency: 3-4 simultaneous forecasters.** OpenRouter handles paid models comfortably at this rate.

---

## Part 1: Per-type stacking flags (LAND FIRST)

Independent of the ablation. Small commit, lets us flip behavior per question type once the ablation answers the question.

Naming is `_ENABLED` (positive polarity) for all three — consistent, more readable than mixed-polarity. Renames the existing `NUMERIC_STACKING_DISABLED_ENV` constant in the same commit.

**Files:**
- `metaculus_bot/constants.py`:
  - **Rename** `NUMERIC_STACKING_DISABLED_ENV` (currently `"NUMERIC_STACKING_DISABLED"`) → `NUMERIC_STACKING_ENABLED_ENV` (`"NUMERIC_STACKING_ENABLED"`).
  - Add two new env vars:
    ```python
    BINARY_STACKING_ENABLED_ENV: str = "BINARY_STACKING_ENABLED"
    MC_STACKING_ENABLED_ENV: str = "MC_STACKING_ENABLED"
    ```
  - Defaults: all three default to **enabled** (matching current prod behavior for binary/MC; the operator currently sets `NUMERIC_STACKING_DISABLED=true` in deploy env to bypass numeric stacking, so post-rename the deploy env must be updated to `NUMERIC_STACKING_ENABLED=false` to preserve current prod behavior).
  - Add `# TODO: revisit defaults after prod-ish ablation results (see scratch_docs_and_planning/prod_ish_ablation_plan.md)` comment near the constants.
- `main.py:962-973` — extend the per-question-type stacker gate:
  ```python
  # Per-question-type stacking gates. All three default to enabled (preserves
  # historical prod behavior). Set <TYPE>_STACKING_ENABLED=false to bypass.
  type_enabled_envs = {
      BinaryQuestion: BINARY_STACKING_ENABLED_ENV,
      MultipleChoiceQuestion: MC_STACKING_ENABLED_ENV,
      NumericQuestion: NUMERIC_STACKING_ENABLED_ENV,
  }
  for q_type, env_name in type_enabled_envs.items():
      if isinstance(question, q_type) and not env_flag_enabled(env_name, default=True):
          spread_exceeds_threshold = False
          break
  ```
  This requires `env_flag_enabled` to accept a `default` kwarg, OR we add a sibling `env_flag_disabled` helper. Check `metaculus_bot/env_utils.py` (or wherever `env_flag_enabled` lives) — extend if needed, mirror existing test coverage.
- Tests: `tests/test_conditional_stacking.py` already covers the numeric path at line 1065/1111. Mirror those tests for binary + MC. Update the existing numeric tests to use the new flag name + flipped polarity (`NUMERIC_STACKING_ENABLED=false` instead of `NUMERIC_STACKING_DISABLED=true`).
- **Document the deploy-env migration**: add a one-liner to `scratch_docs_and_planning/ablation_branch_state_2026-05-23.md` flagging that the prod deploy env needs `NUMERIC_STACKING_ENABLED=false` set after this lands (replacing `NUMERIC_STACKING_DISABLED=true`).

**Verification:** `make test` clean, `make lint` clean. No production behavior change unless the env vars are flipped on the deploy side simultaneously.

**Effort:** ~80 LOC + test updates. ~20 min subagent work.

---

## Part 2: Prod-ish ablation — pre-run code changes

### 2.1 Add `PROD_FORECASTER_MODELS` lineup

**File:** `metaculus_bot/ablation/forecaster_lineup.py`

Add a new constant + builder. Use plain `GeneralLlm` (no donated-key wrapper) to match the fail-fast stance. Personal `OPENROUTER_API_KEY` only.

```python
# Prod-ish ensemble for the 2026-05-24 ablation. Three models at "medium" effort:
# we want forecaster behavior representative of prod without paying high-effort
# token costs across 88 questions × N arms. Plain GeneralLlm — no donated-key
# wrapper, no fallbacks. This is benchmark-mode (single key, fail-fast); see
# the prod_ish_ablation_plan doc for routing rationale.
PROD_FORECASTER_SPECS: list[tuple[str, dict]] = [
    ("openrouter/google/gemini-3.1-pro-preview", {}),  # auto-reasons; no explicit knob
    ("openrouter/anthropic/claude-opus-4.5", {"reasoning": {"max_tokens": 16_000}}),
    ("openrouter/openai/gpt-5.5", {"reasoning": {"effort": "medium"}}),
]
PROD_FORECASTER_MODELS: list[str] = [m for m, _ in PROD_FORECASTER_SPECS]


def build_prod_forecaster_llms() -> list[GeneralLlm]:
    """Construct plain GeneralLlm instances for the prod-ish forecaster ensemble.

    Fail-fast: no donated-key wrapper, no fallbacks. litellm reads
    OPENROUTER_API_KEY (operator's personal key) at invoke time. Failures
    propagate up; the caller resumes manually after fixing the underlying
    issue. See scratch_docs_and_planning/prod_ish_ablation_plan.md.
    """
    return [
        GeneralLlm(model=model, **{**REASONING_MODEL_CONFIG, **kwargs})
        for model, kwargs in PROD_FORECASTER_SPECS
    ]
```

`REASONING_MODEL_CONFIG` is already imported by transitive use; we'll add a direct import from `llm_configs.py`.

Update the module docstring to note "two lineups now: free-tier (`build_free_*`) and prod-ish (`build_prod_*`); selected via `--lineup` CLI flag in `cli.py`."

### 2.2 Lineup selector in CLI + forecasters.py

**File:** `metaculus_bot/ablation/cli.py`

Add `--lineup {free,prod}` flag (default `free` to preserve backward compatibility). Threads through to:
- `metaculus_bot/ablation/forecasters.py:54` — current hardcoded `from forecaster_lineup import build_free_forecaster_llms, FREE_FORECASTER_MODELS`. Replace with a `get_forecaster_lineup(name: str) -> tuple[list[GeneralLlm], list[str]]` helper.
- `cli.py:1124` — `lineup_filter=list(FREE_FORECASTER_MODELS)` becomes `lineup_filter=list(active_lineup_models)`.

### 2.3 Add `mean` arm (basic outputs)

**New file:** `metaculus_bot/ablation/run_mean.py`

Mirror `run_median.py` exactly, swap `statistics.median` → `statistics.mean` for binary/MC option probabilities and percentile values. Numeric uses `numeric_utils.aggregate_numeric` (already supports both — pass `aggregation="mean"` instead of `"median"`).

Cache key: `stacker_outputs/<qid>/arm_mean.json`.

### 2.4 Add mean variant of PDF arm

**File:** `metaculus_bot/ablation/run_pdf.py`

The current arm aggregates per-model probability dists with **pointwise median** (per the synthesis doc, section 2). Add a parameter `aggregation: Literal["mean", "median"]` and split into two arms used in this run:

| Arm | Aggregation | Min forecasters |
|---|---|---|
| `pdf_min1_median` | median | 1 |
| `pdf_min1_mean` | mean | 1 |

Existing `pdf_min1` alias preserved (= `pdf_min1_median`) for back-compat with the cached free-tier ablation.

**Decision (user-confirmed):** prod-ish run uses `min_forecasters=1` only. The free-tier ablation showed `min1` had a noisy single-forecaster tail with the unreliable free models, but we only have 3 forecasters in the prod-ish ensemble (frontier models, recent), so dropping to 1-of-3 is reasonable for coverage and the tail-noise concern is much weaker.

**Five arms total:** `median`, `mean`, `stack_aug`, `pdf_min1_median`, `pdf_min1_mean`.

**Skip:** `stack` (plain stack without tools). User explicitly excluded ("no need to have stack w/o proba tools — costs more to run stacker 2x"). We assume `stack_aug ≥ stack` based on free-tier finding (P=0.954) and similar token cost.

### 2.5 Drop fallbacks for stacker (and use plain GeneralLlm)

**File:** `metaculus_bot/ablation/run_stacker.py`

Two changes for fail-fast + maximum observability:

1. **Plain `GeneralLlm` for the stacker** (no donated-key wrapper). Currently `_default_stacker_llm()` uses `build_llm_with_openrouter_fallback`. Add a `--plain-llm` CLI flag (or just gate on `--lineup prod`) that swaps to plain `GeneralLlm(model=..., **kwargs)` — same pattern as `forecaster_lineup.py:build_free_forecaster_llms` already does. litellm reads `OPENROUTER_API_KEY` from env. No automatic donated→personal fallback at the wrapper level.

2. **Drop both fallback chains**: add a `--no-stacker-fallback` CLI flag. Threads to `run_stacker_for_arm`:
   - `fallback_stacker_llm=None`
   - Skip the median fallback block (lines 606-649 per the survey)
   - On stacker failure: write a marker JSON (`{"qid": ..., "arm": "stack_aug", "error": "..."}`), log at WARNING, propagate the exception so the run aborts.

Together these mean: a stacker failure on any question aborts the run with the actual underlying error (auth, network, model-side issue). Resume after manual fix: re-run with `--stages stack_aug,score` on the failed qids — the cache layer hydrates the rest.

Default flag values: both `False` (keep current behavior for free-tier compatibility). Smoke + full prod-ish runs pass both flags.

### 2.6 Cache hardening (already present — verify)

The existing cache layer writes per-(qid, arm) atomically. Per the survey, `forecaster_outputs/<qid>/<model_slug>.json` is written after each forecaster completes. Verify by reading `cache.py` and confirming write-after-success ordering. If a partial write is possible, fix.

This is the "save outputs as we go" requirement. If a forecaster succeeds but the next one fails, the first one is on disk; resume re-uses it.

---

## Part 3: Smoke run

**Set:** 3 question IDs from the existing manifest, one of each type. I'll pick by reading `backtests/ablation/qids.json` + the cached `forecaster_outputs/` to find one binary, one MC, one numeric with full cached upstream stages.

**Command:**
```bash
~/miniconda3/envs/metaculus-bot/bin/python -m metaculus_bot.ablation.cli \
    --qids <smoke_csv> \
    --lineup prod \
    --stages forecast,stack_aug,pdf,median,mean,score \
    --rate-limit-mode fast \
    --concurrency 4 \
    --plain-llm \
    --no-stacker-fallback
```

**Wall clock:** 3 questions × (forecaster ~5 min for 3 models in parallel + stacker ~3 min) = ~25 min total in tmux. Verify all five arms produce valid scores.

**Gate:** if any arm crashes, fix before the full run. If scores look sane (paired deltas in plausible ranges), proceed.

---

## Part 4: Full run

**Set:** all 88 question IDs from `backtests/ablation/qids.json`.

**Command:**
```bash
tmux new-session -d -s prod_ablation \
    "~/miniconda3/envs/metaculus-bot/bin/python -m metaculus_bot.ablation.cli \
        --qids <all_88_csv> \
        --lineup prod \
        --stages forecast,stack_aug,pdf,median,mean,score \
        --rate-limit-mode fast \
        --concurrency 4 \
        --plain-llm \
        --no-stacker-fallback \
        2>&1 | tee /tmp/prod_ablation_$(date +%Y%m%d_%H%M).log; echo EXITCODE=\$?"
```

**Wall clock:** ~5-10 min per question × 88 = 7-15 hours. Local on MBP; tmux session detached. Tail the log to monitor.

**Recovery:** any failure aborts the run. Caches are atomic; resume by:
1. Inspect the log to identify the failed qid.
2. Fix the underlying issue (auth, transient network, etc).
3. Re-run with `--qids <remaining>` (drop the completed ones from the CSV) — script auto-skips cached arms via cache.py.

---

## Part 5: Analysis notebook

**File:** `scratch/analysis_2026-05/prod_ish_ablation.py` (marimo, NOT ipynb).

Marimo conventions per CLAUDE.md: reactive DAG, real PR diffs, `code_mode`. Use both `marimo` and `marimo-pair` skill conventions.

**Structure** (intentional narrative, not kitchen sink):

1. **Title cell + 2-3 sentence framing.** What we ran, why, what the headline finding is. Reader should know the conclusion before they look at any chart.

2. **Headline figure: paired forest plot vs `median_basic` baseline.** One chart, four arms (`mean`, `stack_aug`, `pdf_min2_median`, `pdf_min2_mean`), three question types (binary, MC, numeric). Y-axis: arm × type. X-axis: paired delta vs `median_basic` (signed so positive = arm better). 95% bootstrap CIs. Color = significance level. This is the chart that tells the whole story.

3. **Conclusion cell** (markdown). 4-6 sentences interpreting the forest plot. State the recommendation up front.

4. **Per-type drill-downs** (collapsed by default if marimo supports it):
   - **Binary**: did stack_aug beat median? Replicates free-tier P=0.954? Side-by-side table of paired deltas + Bayesian P + Wilcoxon p. Sub-figure: paired-delta scatter (one dot per question) so the reader sees the variance, not just the mean.
   - **MC**: same shape. With paid models, does the directional "stacker hurts" signal solidify or evaporate?
   - **Numeric**: same shape. Does PDF still dominate? Does mean-over-PDF beat median-over-PDF?

5. **Cross-comparison with free-tier ablation.** Side-by-side table of free-tier vs prod-ish paired deltas for the same comparisons. Comment on direction agreement and effect-size shifts. This is the load-bearing comparison — explicitly references `scratch/analysis_2026-05/ablation_synthesis.md`.

6. **Caveats.** n=88 still. MC underpowered. Single ensemble draw, not a randomized A/B. Same-day judgment, etc.

7. **Recommendation.** 3-5 sentences: based on this evidence, what should prod's binary/MC stacking flags be set to. Cite specific Bayesian P values + paired Wilcoxon p values.

**Visualization rules** (per `visualization` skill):
- Tufte-clean. No chartjunk.
- Forest plot: dot + horizontal CI line. Vertical zero line dashed.
- Color: green for "arm better than median", red for "worse", gray for null. Saturation = significance.
- Small multiples for per-type drill-downs (3-up grid).
- Paired-delta scatters use Gelman zero-axis rule (don't suppress zero on the delta axis — it's the load-bearing reference).

**Notebook hygiene** (per `notebook-best-practices`):
- Thin orchestration. All scoring/aggregation logic stays in importable `.py` modules (`metaculus_bot/ablation/scoring.py` already has it).
- `display()` over `print()` for DataFrames.
- Markdown cells interpret each output, no naked tables.
- Polars-first for any large-data manipulation; pandas for small summary tables that pair with seaborn/matplotlib.

**Companion synthesis doc:** `scratch/analysis_2026-05/prod_ish_ablation_synthesis.md`. Same skeleton as `ablation_synthesis.md` for direct comparability. Lives alongside the notebook so a reader can grep findings without opening marimo.

---

## File-touch summary

**New:**
- `metaculus_bot/ablation/run_mean.py`
- `scratch/analysis_2026-05/prod_ish_ablation.py` (marimo)
- `scratch/analysis_2026-05/prod_ish_ablation_synthesis.md`

**Modified:**
- `metaculus_bot/constants.py` (Part 1: add 2 env vars)
- `main.py` (Part 1: per-type stacker disable for binary + MC)
- `metaculus_bot/ablation/forecaster_lineup.py` (add prod lineup)
- `metaculus_bot/ablation/forecasters.py` (lineup selector)
- `metaculus_bot/ablation/cli.py` (--lineup, --no-stacker-fallback, --stages includes mean, route mean arm)
- `metaculus_bot/ablation/run_pdf.py` (parameterize aggregation: mean | median)
- `metaculus_bot/ablation/run_stacker.py` (--no-stacker-fallback support)
- `tests/test_ablation_cli.py`, `tests/test_ablation_forecaster_lineup.py`, `tests/test_ablation_run_*.py`, `tests/test_conditional_stacking.py` (extend coverage for new flags + arms)

---

## Execution order

1. **Land Part 1** (per-type stacking flags). Subagent. Independent of everything else.
2. **Land Part 2** (ablation code changes). Single subagent — files are tightly coupled, parallelizing would race on `cli.py`.
3. **Smoke run.** Pick 3 qids. Run locally (no tmux needed for 25 min).
4. **Full run.** All 88 qids. tmux. ~10h overnight.
5. **Analysis.** Marimo notebook + synthesis doc. Subagent for the structure, manual review for narrative quality.

---

## Open issues / risks

- **Gemini 3.1 Pro reasoning param.** Per the survey, it's set with no explicit reasoning kwarg in `llm_configs.py:82`. We mirror that. If "medium effort" Gemini is supposed to be a specific tunable, flag at smoke-test time.
- **`make lint` for marimo .py files.** Ruff handles them. Confirm no special-case lint rules.
- **Free-tier vs prod-ish wall-clock variance.** The free-tier ablation took ~? hours; prod-ish models are slower per-call but we have only 3 forecasters vs 4. Net effect uncertain. Smoke run will tell us.
- **Stacker effort.** Plan calls for Opus 4.5 with `max_tokens=32_000` (already the default in `run_stacker.py`). User said "Opus 4.5 high"; 32k thinking is what prod uses for the stacker, so this matches.

---

## Acceptance criteria

- All 88 questions processed end-to-end across 5 arms.
- Marimo notebook renders cleanly, narrative reads top-to-bottom, conclusion is supported by the headline forest plot.
- `prod_ish_ablation_synthesis.md` exists with the same skeleton as `ablation_synthesis.md`.
- No bare error stacktraces in the analysis output. Failures (if any) are documented with the affected qids.
- Recommendation states explicit env-var settings for the per-type stacker flags landed in Part 1.
