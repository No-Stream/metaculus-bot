<!-- markdownlint-disable MD040 -->

# Probabilistic tools — activation guide

**Status as of this doc:** tools exist and are unit-tested. The prompt edits,
runtime wiring, and stacker-prompt injection are NOT yet done. This doc
describes exactly what to change to turn the feature on, what to verify,
and what to watch out for.

**Why it's dormant:** we wanted thorough unit tests on the math and
extraction before flipping anything live. Prompt changes are the riskiest
piece (they can degrade free-text reasoning quality) and need a controlled
rollout with a feature flag and backtest validation.

## Companion artifacts already in the tree

- `metaculus_bot/structured_output_schema.py` — Pydantic v2 models for the
  per-question-type JSON block (Binary, Numeric, MultipleChoice,
  DiscreteCount) + `extract_json_block()` and `parse_structured_block()`.
- `metaculus_bot/probabilistic_tools/` — pure-function tools:
  `base_rate.py` (Beta-binomial, Laplace, base-rate blend, implied LR,
  explicit Bayes), `survival.py` (constant-hazard, Weibull, Poisson-≥1,
  base-rate→hazard), `aggregation.py` (linear / log / Satopää /
  inverse-variance pools, Dirichlet pool), `distributions.py` (normal /
  lognormal / Student-t fits from percentiles, out-of-bounds mass,
  CDF-at-threshold, Metaculus-201-point CDF wrapper around
  `pchip_cdf.generate_pchip_cdf`), `mc_discrete.py` (Dirichlet with
  "Other", NegBinom / Poisson / Beta-binomial-ceiling percentile
  generators), `consistency.py` (percentile-family SSE comparison,
  stated-base-rate vs. evidence-strength flagger, percentile monotonicity).
- `metaculus_bot/tool_runner.py` — takes a list of
  `(forecaster_id, rationale_text, question)` tuples, extracts JSON blocks,
  dispatches the right tools per question type, and returns
  `(per_forecaster_blocks: dict[forecaster_id, str], aggregated_block: str)`
  as markdown-ready strings.
- `tests/test_probabilistic_tools.py`,
  `tests/test_structured_output_schema.py`, `tests/test_tool_runner.py` —
  unit + integration coverage.

## Activation — three changes

### 1. Append a structured-block instruction to each forecaster prompt

File: `metaculus_bot/prompts.py`

Current line-ranges (verify before editing — file counts drift with minor
edits):

- `binary_prompt` body ends at a block like `[The last thing you write MUST BE your final answer as an INTEGER percentage. "Probability: ZZ%"]` — currently ~line 204.
- `multiple_choice_prompt` ends after the `Option_N: NN%` example — currently ~line 298.
- `numeric_prompt` ends after the `Percentile 97.5: 123.4` example — currently ~line 457.

**Critical ordering constraint:** the parser path
(`main.py:1111-1115 / 1138-1146 / 1219-1227`) reads the *trailing* answer
lines. If the JSON block is emitted AFTER the trailing answer, the parser
may grab the wrong text. Two options, pick one:

**Option A (preferred):** emit JSON block FIRST, then the final answer
line(s). Example for binary — append after the existing analysis template,
before the trailing `Probability: ZZ%` line:

> ── STRUCTURED FORECAST (machine-readable; required) ──
> Emit a fenced JSON block on its own. Schema:

```json
{
  "question_type": "binary",
  "prior": {"prob": 0.15, "source": "annual incidence 2015-2024"},
  "base_rate": {"k": 3, "n": 12, "ref_class": "years matching condition"},
  "hazard": {"rate_per_unit": 0.25, "unit": "year", "elapsed_fraction": 0.33, "remaining_fraction": 0.67},
  "evidence": [{"summary": "Q1 policy shift", "direction": "up", "strength": "moderate"}],
  "scenarios": [],
  "posterior_prob": 0.28
}
```

All fields other than `posterior_prob` are optional — omit if not applicable.
The final `Probability: ZZ%` line is written *after* the JSON block.

`extract_json_block()` already picks the LAST fenced block on purpose, so
if a model re-dumps its JSON block on a revision, we get the latest. The
final `Probability: ZZ%` line is after the block and the existing regex
parser finds it correctly.

**Option B:** emit JSON block as the VERY LAST thing, after the answer
line. Requires changing `extract_json_block()` to also handle the case
where the JSON block follows the answer, and carefully verifying the
answer parser doesn't accidentally walk into the JSON. More fragile —
prefer A.

Per-question-type schema variations:

- Numeric: `question_type: "numeric"`, `declared_percentiles` (dict p→value
  for {0.1, 0.5, 0.9} minimum), `distribution_family_hint`, `student_t_df`,
  `tails: {below_min_expected, above_max_expected}`, optional `scenarios`,
  optional `mixture_components`.
- MC: `question_type: "multiple_choice"`, `option_probs: dict[option_name,
  prob]` (sums to 1.0), optional `other_mass`, optional `concentration`.
- Discrete/count: `question_type: "discrete_count"`, `mean_estimate`,
  `dispersion ∈ {poisson, negbinom, beta_binom_ceiling}`, `ceiling` (for
  beta_binom_ceiling), `overdispersion_factor` (for negbinom).

### 2. Run tool_runner post-prediction in `_make_prediction`

File: `main.py`, around line 819-852.

After line 849 (where `prediction.reasoning = f"Model: {actual_llm.model}\n\n{prediction.reasoning}"`)
— but *before* returning — call the tool runner on this single forecaster's
rationale and append a computed-quantities section. The feature flag is
checked *inside* `run_tools_for_forecaster` (it returns `""` when disabled),
so callers do not need to gate the call themselves:

```python
from metaculus_bot.tool_runner import run_tools_for_forecaster

computed_md = run_tools_for_forecaster(
    question=question,
    rationale=prediction.reasoning,
    forecaster_id=actual_llm.model,
)
if computed_md:
    prediction.reasoning = f"{prediction.reasoning}\n\n## Computed quantities\n{computed_md}"
```

The cross-model aggregated section is built *later*, when we have all
predictions together — see step 3.

### 3. Prepend aggregated cross-model section to stacker prompts

Cross-model aggregation (log-pool, Satopää, consistency summary) runs once
per question and belongs at the top of the stacker prompt. File:
`metaculus_bot/stacking.py`.

Two integration points:

- `run_stacking_binary`, `run_stacking_mc`, `run_stacking_numeric` (lines
  67 / 96 / 140) take `base_texts: Sequence[str]`. Upstream callers can
  prepend an `## Cross-model aggregation\n...` block to `base_texts` as a
  synthetic "Model 0" analysis. Cleanest: take a new optional arg
  `aggregated_tool_output: str | None = None` on each `run_stacking_*`
  function and string-concat it before `predictions_text` in the
  corresponding prompt (`metaculus_bot.prompts.stacking_binary_prompt`
  etc., lines 462 / 532 / 610).

- The caller lives in `main.py:626-732` (the STACKING and
  CONDITIONAL_STACKING branches of `_research_and_make_predictions`).
  After `valid_predictions` is populated but before
  `_aggregate_predictions` is called, invoke
  `build_cross_model_aggregation` (which internally no-ops when the
  feature flag is off). For strongly typed callers, the three
  `aggregate_{binary,numeric,mc}_values` entry points are exposed as
  well — pick whichever fits the call site:

```python
from metaculus_bot.tool_runner import build_cross_model_aggregation

aggregated_tool_output = build_cross_model_aggregation(
    question=question,
    rationales=[p.reasoning for p in valid_predictions],
    prediction_values=prediction_values,
)  # returns "" when the flag is off
if not aggregated_tool_output:
    aggregated_tool_output = None
```

Then pass `aggregated_tool_output` into `_aggregate_predictions` → down
into `run_stacking_*`. The `_aggregate_predictions` signature
(`main.py:854`) may need a new keyword arg; plumb it through carefully
(base-class compatibility — see how `reasoned_predictions` is threaded).

### 4. Feature flag

The flag is the env var `PROBABILISTIC_TOOLS_ENABLED` (read via
`env_flag_enabled` per `constants.py:95`). It is checked *inside* the
public entry points in `metaculus_bot/tool_runner.py`
(`run_tools_for_forecaster`, `build_cross_model_aggregation`, the typed
`aggregate_*_values` helpers, and `cdf_at_threshold_for_forecaster`). Each
returns an empty string (or `None`) when the flag is unset, so activation
sites do not need a separate conditional.

Flip to True via `PROBABILISTIC_TOOLS_ENABLED=1` in `.env` for backtests
only, leaving prod/tournament OFF until we have a positive signal.

## Comment marker

Add a `TOOLS_USED=true/false` marker similar to `STACKED_MARKER_TRUE`
(`metaculus_bot/comment_markers.py`) so residual analysis can distinguish
tool-augmented from vanilla runs. Fold it alongside the STACKED marker
in `_create_unified_explanation` (`main.py:792-817`).

## Verification sequence

1. **Unit tests still green:**

   ```
   make test
   ```

   (or `conda run -n metaculus-bot poetry run pytest tests/test_probabilistic_tools.py tests/test_structured_output_schema.py tests/test_tool_runner.py -v`)

2. **Manually hand-craft a rationale** with a valid JSON block and run the
   tool_runner on it from a Python REPL to eyeball the markdown output.
   No LLM call needed — verifies the extraction/formatting path.

3. **Smoke backtest with the flag ON:**

   ```
   PROBABILISTIC_TOOLS_ENABLED=1 make backtest_smoke_test
   ```

   (4 questions.) Confirm: tool output blocks appear in per-model
   rationales; aggregated block appears at top of stacker prompt on
   triggered questions; no runtime errors; comments show
   `TOOLS_USED=true`.

4. **Small backtest A/B:**

   ```
   make backtest_small                              # tools OFF control
   PROBABILISTIC_TOOLS_ENABLED=1 make backtest_small # tools ON
   ```

   12 questions each. Compare stacker final probabilities where they
   differ; look for cases where tool output appears to have shifted the
   stacker's number toward a more reasonable value.

5. **Medium backtest:**

   ```
   PROBABILISTIC_TOOLS_ENABLED=1 make backtest_medium
   ```

   32 questions. Metrics to track, per April 2026 analysis:
   - **PIT std on numerics** — target moving toward 0.289 (currently 0.143
     per `project_performance_analysis_2026q2.md`).
   - **0.20-0.30 binary bucket** — hit rate; currently ~0.24 predicted vs
     ~0.75 actual.
   - **Stacker Brier delta** vs. tools-off control.
   - **Log-loss delta** across question types.

6. **Cost check:** per-question LLM call count should be unchanged (no
   new model calls); token count per call can grow by up to ~5% from the
   structured block in the prompt. Confirm via log scraping or
   `LitellmCostTracker`.

## Known landmines

- **Parser ordering.** If the JSON block is placed after the final answer
  line, the existing structure_output parser
  (`main.py:1111 / 1138 / 1219`) may parse JSON-block text as the answer.
  Use Option A ordering (JSON before answer).

- **Percentile-set rigidity.** The numeric parser asserts exactly the 11
  canonical percentiles [2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5]
  (see `main.py:1228` and `numeric_validation.validate_percentile_count_and_values`).
  The JSON block's `declared_percentiles` dict need not match these exactly
  (it's used for tool input, not for the official forecast), but if the
  model re-emits percentiles in the JSON that differ from the trailing
  `Percentile X: …` lines, tools operate on the JSON version. Document
  this in the prompt: "the JSON percentiles should match your final
  Percentile lines."

- **Comment trim.** `REPORT_SECTION_CHAR_LIMIT = 49_999`,
  `COMMENT_CHAR_LIMIT = 149_999` (`constants.py:87-88`). Stacker reads
  `reasoned_predictions[i].reasoning` directly before trim (safe), but
  the published Metaculus comment gets nibbled from the middle — long
  computed-quantity sections on every base model × N models may push
  readable content out. Compact output format: 2-4 bullet lines per
  tool, not paragraphs.

- **Malformed JSON is visible, not fatal.** `parse_structured_block`
  returns None and logs a WARNING on malformed input. tool_runner skips
  tools that need the structured block for that forecaster and emits
  no "Computed quantities" section. Aggregated math still runs over
  whatever DID parse. Confirmed fail-visible, not fail-silent.

- **Benchmark cache.** `research_cache` in `main.py:129, 437` keys by
  question ID to keep backtests reproducible. Tool outputs are
  deterministic given the same structured input, so backtest
  reproducibility is preserved without changes.

- **`drop_params=True` in LiteLLM.** Only relevant if we ever switch to
  actual native tool-calling (not this design). Noted for future.

- **Prompt length growth.** Each forecaster's prompt grows by ~15 lines of
  schema instruction; each rationale may grow by ~10-30 lines (tool output).
  Stacker prompt grows by N × 10-30 lines + aggregated section. Budget
  roughly +3-7% tokens at most.

## Rollback

Set `PROBABILISTIC_TOOLS_ENABLED=0` (or unset). Prompts with the JSON
block instruction remain but forecasters can ignore it (we already log
"block not found" at DEBUG, not WARNING, for the no-block case). To fully
roll back, revert the prompts.py changes.

## Files to edit at activation time (summary)

- `metaculus_bot/prompts.py` — append schema instruction to `binary_prompt`
  (~line 204), `multiple_choice_prompt` (~line 298), `numeric_prompt`
  (~line 457). Option A ordering: JSON block before trailing answer.
- `main.py:819-852` (`_make_prediction`) — call `run_tools_for_forecaster`
  after `prediction.reasoning` is tagged.
- `main.py:626-756` (STACKING + CONDITIONAL_STACKING branches) — call
  `build_cross_model_aggregation` before stacking invocation; plumb the
  result into `_aggregate_predictions` → `run_stacking_*`.
- `metaculus_bot/stacking.py:67, 96, 140` — accept
  `aggregated_tool_output: str | None` and inject into prompts.
- `metaculus_bot/prompts.py:462, 532, 610` (`stacking_*_prompt`) — accept
  and prepend `aggregated_tool_output`.
- `metaculus_bot/comment_markers.py` — add `TOOLS_USED_MARKER_*`.
- `main.py:792-817` (`_create_unified_explanation`) — append TOOLS_USED
  marker.
- `metaculus_bot/tool_runner.py` — already wires feature-flag checks into
  `run_tools_for_forecaster`, `build_cross_model_aggregation`, the typed
  `aggregate_*_values` helpers, and `cdf_at_threshold_for_forecaster`. No
  activation-time change required beyond setting the env var.

## Future extensions (not v1)

- Phase 3 tools (scenario EV, Weibull with hazard shape, piecewise hazard,
  mixture fitting from percentiles, AR(1) series extrapolator). Plug into
  `probabilistic_tools/` as new modules; extend `tool_runner` dispatch.
- Expose tools to the stacker itself via a second structured-output
  convention — only worth it if v1 shows the stacker ignoring per-base
  tool outputs (i.e., it'd rather run its own math).
- Native LiteLLM tool-calling loop (would require overriding
  `GeneralLlm._mockable_direct_call_to_model` to handle tool-call
  responses — see `forecasting_tools/ai_models/general_llm.py:244-252`).
  Only attempt if post-hoc design proves insufficient.
