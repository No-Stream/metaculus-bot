<!-- markdownlint-disable MD040 -->

# Atlas-inspired improvements — plan

Date: 2026-05-12
Companion evidence: `scratch_docs_and_planning/tail_widening_empirical_calibration.md` (2026-05-12), `scratch_docs_and_planning/probabilistic_tools_activation.md` (pre-existing), `~/.claude/projects/-Users-flatljan-personal-metaculus-bot/memory/project_performance_analysis_2026q2.md`, `FUTURE.md`.

## Why these changes

Manual inspection of 6 long-form Preseen-Atlas comments (spring-AIB-2026 leader, 1st of 214 bots) revealed five recurring patterns:

1. **Post-hoc calibration formula** printed in every comment: `submitted = 0.96 * model_estimate + 0.02`. Caps predictions in `[0.02, 0.98]` plus 4% linear shrink toward 50.
2. **Explicit quantitative base-rate math** — gamma/exponential waiting-time for recurring events, Poisson record-break hazard for "will any model hit X by date Y", noisy-OR decomposition for rare events (`1 - ∏(1-pᵢ)`), all computed in Python (not in the LLM's head).
3. **Mixture-of-normals for numeric CDFs** — LLM emits `(weight, mean, sd)` scenarios; code builds the CDF and enforces Metaculus constraints mechanically.
4. **Primary-source citations** — every comment cites `.gov`, `docs.anthropic.com`, SEC IR pages, `tsa.gov` CSVs. Wire services are secondary.
5. **Polymarket as a cross-check**, not an anchor — one TSA example used Polymarket consensus to sanity-check a recent weekly total.

Our repo already contains partial answers to (2) — `metaculus_bot/probabilistic_tools/` plus `tool_runner.py` — but activation edits are pending (see `probabilistic_tools_activation.md`). Gaps against Atlas's math: noisy-OR, mixture-of-normals CDF, Gamma-waiting-time conditional-on-survival. The other four patterns are net-new.

A sixth, repo-specific concern surfaced during planning: tail-widening. Our PIT analysis says CDFs are already too wide; an empirical calibration on 43 resolved numerics (report linked above) says `k_tail=1.0` beats the current `k_tail=1.25` in every segment. This isn't an Atlas import — it's a hyperparameter flip that the existing data supports independently.

## Principles

- **No empirical guessing.** Calibration decisions use resolved-question data. Pipeline changes run through `make backtest_medium` (and `backtest_large` before merge) before flipping defaults.
- **TDD.** Every new tool, provider, and pipeline stage has tests written before implementation.
- **Feature-flagged rollout.** Anything that can shift forecaster behaviour ships behind an env flag and a backtest A/B before flipping defaults.
- **Subagent-driven implementation.** Each independent workstream goes to a subagent with a scoped context packet. The main session handles planning, review, and integration.
- **Unified code review pass at the end.** `/review --fresh` before merge, addressing every finding.
- **Rollback is free.** Every change is rollback-able by flipping a single env var or reverting one commit. No migrations, no state.

## Workstreams

### Workstream A — Tail-widening: flip default to k_tail=1.0

**Evidence.** `scratch_docs_and_planning/tail_widening_empirical_calibration.md`. On 43 resolved numerics (Feb–May 2026), `k_tail=1.0` produces PIT std closest to the uniform ideal (0.289) in every segment. Current `k_tail=1.25` moves away from ideal in every segment. Difference is small but directionally consistent.

**Decision.** Flip default to 1.0. Fix the three real bugs the empirical study surfaced rather than just patching the surface.

**The three real bugs in `tail_widening.py`:**

1. **`k_tail < 1.0` is silently a no-op.** Line 116 short-circuits when `k_tail <= 1.0 and span_floor_gamma <= 0.0`. Line 150 only applies the inverse transform `if k_tail > 1.0`. So configuring `k_tail=0.8` (narrowing) does nothing visible — no error, no warning, no effect. This is a fail-silent configuration footgun per our global "fail fast" rule.
2. **`span_floor_gamma` enforcement never fires on current data.** The code path is wired — `if span_floor_gamma > 0 and None not in (i025, i05, i10):` at `tail_widening.py:171` and the upper mirror at L178 — it's just that at our current ensemble's declared spans, the floor constraint `(p05 − p025) ≥ γ·(p10 − p05)` never binds, per the empirical study's finding at §3. The `> 0` gate is explicit, not silent: setting the default to `0.0` turns the check off cleanly, and any future user who sets it back to `1.0` (or higher) re-enables the existing floor enforcement with no code change. So "dead code" was the wrong framing — it's correctly-gated dormant code.
3. **Docstring overstates the behavior.** Args: `k_tail: maximum stretch factor at deepest tails in transformed space (>=1.0)` — the `(>=1.0)` parenthetical is advisory, not enforced; nothing raises if you pass `k_tail=0.5`. Docstring should either describe the real no-op behavior OR the code should raise.

**Full fix.**

- `metaculus_bot/numeric_config.py:122` — change `TAIL_WIDEN_K_TAIL` from 1.25 to 1.0. Update the existing explanatory comment to cite the empirical report.
- `metaculus_bot/numeric_config.py` — change `TAIL_WIDEN_SPAN_FLOOR_GAMMA` default from 1.0 to 0.0. Comment: "Floor enforcement (tail_widening.py:171/178) is gated on `> 0`. Default disabled because in all 2026 data the floor never bound (see tail_widening_empirical_calibration.md §3). Setting this to any positive value re-enables the existing floor enforcement — kept configurable for future models with unusually sharp declared tails."
- `metaculus_bot/tail_widening.py:95-118` (public function):
  - **Raise `ValueError` on `k_tail < 1.0`** with a clear message pointing to the fact that narrowing is not implemented. Fail-fast per repo convention. If future work needs narrowing, adding the branch is its own PR.
  - **Raise `ValueError` on `k_tail < 0.0` or `span_floor_gamma < 0.0`** — currently these "work" but produce nonsense. The current `k_tail <= 1.0 and span_floor_gamma <= 0.0` short-circuit on L116 stays, but only after the validation raises on negative values.
  - Update the docstring: replace `(>=1.0)` advisory with "`k_tail=1.0` disables widening (identity pass). Values < 1.0 raise `ValueError` — narrowing is not implemented. See `tail_widening_empirical_calibration.md` for the empirical rationale."
  - Update `span_floor_gamma` docstring: "Enforces `(p05 − p02.5) >= gamma*(p10 − p05)` and the upper mirror. `0.0` disables. In all 2026 data this floor never binds; kept configurable for forecasters with unusually sharp declared tails."

**Tests.**

- `tests/test_tail_widening.py` — extend:
  - Assert `widen_declared_percentiles(..., k_tail=1.0)` returns the input percentile list unchanged (identity). Currently untested — verify and lock in.
  - Assert `widen_declared_percentiles(..., k_tail=0.8)` raises `ValueError` with a message mentioning "narrowing is not implemented."
  - Assert `widen_declared_percentiles(..., k_tail=-1.0)` raises `ValueError`.
  - Assert `widen_declared_percentiles(..., span_floor_gamma=-0.5)` raises `ValueError`.
  - Assert `widen_declared_percentiles(..., span_floor_gamma=0.0)` is equivalent to the current gamma=1.0 behavior on our actual declared-percentile shapes (proving the empirical no-op claim in regression form — use a handful of real declared-percentile fixtures from the calibration study's CSV).
- `tests/test_numeric_config.py` (new, or add to existing config test file) — assert `TAIL_WIDEN_K_TAIL == 1.0` and `TAIL_WIDEN_SPAN_FLOOR_GAMMA == 0.0`, with a test docstring linking the empirical report.

**Verification.**

- `make test` green.
- `make backtest_medium` A/B with `TAIL_WIDEN_K_TAIL=1.25` (override via env) vs the new 1.0 default, on the same seed/question set. Confirm PIT-delta matches the empirical study within noise.
- Residual check: scan post-merge bot comments for `k_tail` or `span_floor_gamma` hardcodes in tests or scripts that might regress the config.

**Dependencies.** None. Ship first.

---

### Workstream B — Binary clip to [0.02, 0.98]

**Evidence.** Atlas comments all show `submitted = 0.96*p + 0.02`. We're going minimal: clip-only, no linear shrink. One constant edit covers both per-model and stacker output because both call sites read the same constants.

**Files.**

- `metaculus_bot/constants.py:149-150` — `BINARY_PROB_MIN: 0.01 → 0.02`, `BINARY_PROB_MAX: 0.99 → 0.98`.
- `tests/test_constants.py` (or wherever clamp behaviour is covered) — update expected values.

**Tests.**

- Check `main.py:1381` clamp site uses the constants (verified: it does).
- Check `stacking.py:92` clamp site uses the constants (verified: it does).
- Check MC clamp is separate (`MC_PROB_MIN=0.005` stays unchanged).
- Regression: confirm no test hardcodes 0.01/0.99 as the binary bound. If any do, update with a code-comment justifying the Atlas reference.

**Verification.**

- `make test` green.
- `make backtest_medium` A/B. Expect near-zero Brier delta on well-calibrated bins, small improvement on 0%/100%-clamped tail questions.

**Dependencies.** None. Ship in the same PR as Workstream A or separate — both are one-line constant changes.

---

### Workstream C — probabilistic_tools activation (3-edit plan)

**Evidence.** `scratch_docs_and_planning/probabilistic_tools_activation.md`. All library code and unit tests exist; three wiring edits are pending.

**Scope.** Implement the activation plan as written, adapted to current line numbers:

1. Append fenced-JSON structured-block instruction to `binary_prompt` (`prompts.py:144-238` body, before trailing `Probability:` line), `multiple_choice_prompt` (L241-331), `numeric_prompt` (L334-490). Per-question-type schema variations per the activation doc.
2. In `_make_prediction` (`main.py:1002`), call `run_tools_for_forecaster` on each finished rationale and append a `## Computed quantities` section before returning.
3. Plumb `build_cross_model_aggregation` output into `run_stacking_{binary,mc,numeric}` via a new optional `aggregated_tool_output: str | None = None` parameter. Call site is in the STACKING / CONDITIONAL_STACKING branches at `main.py:750 / 781`. Inject at the top of each `stacking_*_prompt` in `prompts.py:493 / 563 / 641` as a synthetic "Cross-model aggregation" header.

**Feature flag.** `PROBABILISTIC_TOOLS_ENABLED` (already wired inside `tool_runner.py`, unset by default). Flip on in `.env` for backtests; leave off for prod/tournament until backtest A/B shows a positive signal.

**Comment marker.** `TOOLS_USED=true/false` marker added to `_create_unified_explanation` (`main.py:925-960`), alongside the existing `STACKER_OUTCOME` and `STACKED` markers.

**Files.** Per the activation doc's "Files to edit" summary (§297-317). Line numbers since the doc was written:

- `metaculus_bot/prompts.py`: `binary_prompt` ends at L238, `multiple_choice_prompt` at L331, `numeric_prompt` at L490. Insert the JSON-block schema before the trailing answer line (Option A ordering).
- `main.py:1002-1035` (`_make_prediction` full body) — add the tool_runner call before returning.
- `main.py:750-820 / 781+` (STACKING + CONDITIONAL_STACKING branches) — call `build_cross_model_aggregation` before `_aggregate_predictions`. Plumb through `_aggregate_predictions` (L1037).
- `metaculus_bot/stacking.py:67, 96, 140` — accept `aggregated_tool_output`.
- `metaculus_bot/prompts.py:493, 563, 641` (stacking prompts) — accept and prepend.
- `metaculus_bot/comment_markers.py` — add `TOOLS_USED_MARKER_*` constants.
- `main.py:925-960` (`_create_unified_explanation`) — emit the new marker.

**Tests (TDD — write first).**

- `tests/test_tool_runner_activation.py` (new): assert that with `PROBABILISTIC_TOOLS_ENABLED=1`, a fake rationale with a valid JSON block produces `## Computed quantities` output; with flag off, no output.
- `tests/test_structured_output_in_prompts.py` (new): golden-file check that binary/MC/numeric prompts contain the structured-block schema instruction.
- `tests/test_stacker_aggregated_input.py` (new): when `aggregated_tool_output` is provided, it appears at the top of the stacker prompt.
- `tests/test_comment_markers.py` — extend to cover `TOOLS_USED_MARKER_*`.
- Existing `tests/test_probabilistic_tools.py`, `test_structured_output_schema.py`, `test_tool_runner.py` stay green (regression gate).

**Verification.** Activation doc §192-245 — smoke (n=4) → small A/B (n=12 × 2) → medium A/B (n=32 × 2). Confirm:

- Tool output blocks appear in per-model rationales when flag is on.
- Aggregated block appears at top of stacker prompt on triggered questions.
- Token count per call grows ≤ 7% (activation doc §286-289).
- Comments show `TOOLS_USED=true`.

**Dependencies.** Adds fields to the structured-output schema in Workstream D (mixture) and uses the new tools in Workstream E (noisy-OR, mixture CDF, Gamma). Workstream C can land first with existing tools; D and E can extend it.

**Rollback.** Set `PROBABILISTIC_TOOLS_ENABLED=0`. Prompts still contain the JSON schema instruction but forecasters can ignore it (the "block not found" path is fail-visible not fail-silent, per activation doc §271-275).

---

### Workstream D — New probability tools (fills Atlas coverage gaps)

**Evidence.** My previous inventory of `probabilistic_tools/` showed:

- Exponential waiting-time: ✅ `survival.prob_event_before`
- Poisson ≥1 event: ✅ `survival.poisson_at_least_one`
- Weibull unconditional: ⚠️ `survival.weibull_prob_event_before` (no conditional-on-survival)
- Gamma waiting-time fit: ❌ missing
- Noisy-OR (independent union): ❌ missing
- Mixture-of-normals CDF + percentile fit: ❌ missing (schema slot referenced in activation doc but `NumericStructured` doesn't have `mixture_components` yet, and `extra="forbid"` rejects it)

**Scope — three new tools, each in the natural module:**

D1. **`noisy_or`** in `metaculus_bot/probabilistic_tools/aggregation.py`.

```python
def noisy_or(probs: Sequence[float], *, weights: Sequence[float] | None = None) -> float:
    """P(≥1 of N independent events) = 1 - ∏(1 - p_i).

    Used for 'will any of X, Y, Z happen' decomposition. Caller is
    responsible for the independence assumption; tool does the math.
    """
```

D2. **`fit_gamma_from_gaps`** and **`gamma_prob_event_before`** in `metaculus_bot/probabilistic_tools/survival.py`.

```python
@dataclass
class GammaFit:
    shape: float  # k
    scale: float  # θ
    mean: float
    variance: float

def fit_gamma_from_gaps(observed_gaps: Sequence[float], *, method: Literal["mom","mle"]="mom") -> GammaFit:
    """Fit Gamma to historical inter-arrival gaps (method-of-moments by default)."""

def gamma_prob_event_before(fit: GammaFit, *, elapsed: float, remaining: float) -> SurvivalResult:
    """P(next event by elapsed+remaining | no event in [0, elapsed])."""
```

D3. **Mixture-of-normals** in a new `metaculus_bot/probabilistic_tools/mixtures.py`.

```python
@dataclass
class MixtureComponent:
    weight: float  # >= 0
    mean: float
    sd: float      # > 0

@dataclass
class MixtureOfNormals:
    components: tuple[MixtureComponent, ...]
    # ...normalized weights in __post_init__

def mixture_cdf(mix: MixtureOfNormals, grid: np.ndarray) -> np.ndarray:
    """Analytic mixture CDF evaluated on a sorted grid."""

def fit_mixture_from_percentiles(
    percentiles: Mapping[float, float],
    *,
    n_components: Literal[2, 3, 4] = 3,
    seed: int = 0,
) -> MixtureOfNormals:
    """Fit 2-4 component mixture to declared percentiles via constrained LSQ.
    Fallback to single-normal if convergence fails."""

def percentiles_to_metaculus_cdf_via_mixture(
    mix: MixtureOfNormals,
    question: NumericQuestion,
) -> list[Percentile]:
    """Build a 201-point CDF from the mixture, enforcing Metaculus constraints
    (min-step, max-step, open/closed bounds) via the existing pchip_cdf helpers.
    Returns same format as distributions.percentiles_to_metaculus_cdf."""
```

**Schema extension.** `metaculus_bot/structured_output_schema.py:NumericStructured` gets a new optional field:

```python
mixture_components: list[MixtureComponentDeclaration] | None = None
```

with its own Pydantic model + validators (weights sum to 1.0 within tolerance, sd > 0, ≥ 2 components).

**Tool dispatcher extension.** `metaculus_bot/tool_runner.py:_run_numeric_tools` learns to call `fit_mixture_from_percentiles` and `mixture_cdf` when `mixture_components` is present in the structured block; output goes into the "Computed quantities" markdown.

**Tests (TDD).**

- `tests/test_probabilistic_tools.py` — extend with noisy_or, Gamma, mixture tests:
  - `noisy_or([0.1]*10)` ≈ 0.651 (verified against `1 - 0.9**10`).
  - `noisy_or([1e-9, 0.5, 1e-9])` ≈ 0.5 (numerical stability).
  - Gamma MOM against known synthetic gaps.
  - Gamma conditional survival: `P(T ≤ 200 | T > 100)` via `(F(200) - F(100)) / (1 - F(100))`.
  - Mixture CDF matches `sum_i w_i * Φ((x-μᵢ)/σᵢ)` on a grid.
  - Mixture fit recovers a known 3-component mixture from its own percentiles within `rmse < 0.01` on [0.025, 0.975].
- `tests/test_structured_output_schema.py` — extend for `NumericStructured.mixture_components`.
- `tests/test_tool_runner.py` — extend to assert the numeric tool path handles both percentiles-only and mixture-only blocks.

**Verification.** Unit tests green. No runtime wiring of mixture-path in Workstream D itself; that's Workstream E. D is library-only.

**Dependencies.** C's activation plan lands first (otherwise the new tools are unreachable from production). D is purely additive to the library.

**Rollback.** Trivially revertible: library code + schema field + tests. Nothing in production depends on it until Workstream E.

---

### Workstream E — Numeric format: EITHER percentiles OR mixture

**User steer.** "Give the LLM the choice and parse whichever format it prefers to give, at least for a start." No consistency check; if the LLM emits both, we prefer one deterministically and log a warning so we can audit frequency.

**Design.**

- `numeric_prompt` (`prompts.py:334-490`) grows a second optional output format:

```
OUTPUT FORMAT OPTION A — PERCENTILES (default, what most models will use):
Percentile 2.5: ...
[...]
Percentile 97.5: ...

OUTPUT FORMAT OPTION B — MIXTURE OF NORMALS (use when you prefer to describe
the distribution as a mixture of scenarios, e.g. bimodal questions, or
questions where you naturally reason in 'underperform / baseline / breakout'
terms). Emit a JSON block like:

```json
{
  "components": [
    {"weight": 0.2, "mean": 68.0, "sd": 10.0},
    {"weight": 0.55, "mean": 85.0, "sd": 7.0},
    {"weight": 0.25, "mean": 105.0, "sd": 12.0}
  ]
}
```

Emit ONE of the two formats. If you emit both, we will use the mixture and ignore percentiles.

```

- Parser (new: `metaculus_bot/numeric_format_router.py`):
  - Detect mixture block first (fenced JSON with `components` key). If present and valid: build CDF via `fit_mixture_from_percentiles` → `percentiles_to_metaculus_cdf_via_mixture`.
  - Otherwise fall back to the existing percentile parse path.
  - Emit a `numeric_format` attribute on the returned prediction (values: `"percentiles" | "mixture" | "both"`), logged for analysis.
- Existing pipeline stages continue to apply to mixture-derived percentiles:
  - `sanitize_percentiles` (maybe — mixture-derived percentiles are already monotonic, but cluster spreading / bound clamping is still safe).
  - `widen_declared_percentiles` respects the Workstream A default (1.0 → no-op).
  - Discrete-integer snap, unit-mismatch guard, and ensemble aggregation in CDF space all work unchanged (they operate on the 201-point CDF, not the percentiles).

**Tests.**
- `tests/test_numeric_format_router.py`: mixture-only, percentiles-only, both, malformed-mixture-falls-back, missing-everything-raises.
- `tests/test_numeric_pipeline_mixture.py` (extension): assert that a mixture-only forecaster produces a valid 201-point CDF end-to-end.
- `tests/test_numeric_prompt.py`: golden-file check that the prompt contains both format descriptions.

**Verification.**
- `make test`.
- Smoke backtest (n=4) and small backtest (n=12) — verify at least one LLM actually emits a mixture, both paths work, CDF validation passes.
- Medium backtest A/B — track log-loss delta and per-format frequency.

**Dependencies.** Workstream D (mixture math). Workstream C (activation — structured blocks live in the numeric prompt alongside the mixture option).

**Rollback.** Remove the "OPTION B" block from the prompt. Parser continues to work (falls back to percentile path when mixture absent). Tool still available for manual use.

---

### Workstream F — Primary-source prompt steer

**Evidence.** `metaculus_bot/prompts.py:86-141` (`web_research_prompt`) FOCUS AREAS is generic. The `targeted_search_prompt:779` and `gap_fill_search_prompt:888` have some primary-source steering; first-pass does not. Atlas's comments lean heavily on `.gov`, `docs.anthropic.com`, SEC IR, etc.

**Scope.** Small prompt edits only.

- `web_research_prompt:131-136` — add a `PRIMARY SOURCES (preferred)` bullet that enumerates:
  - Government statistics (`.gov`, `.gouv.fr`, `ec.europa.eu`, `*.go.jp`, etc.)
  - SEC filings and investor-relations (`sec.gov`, `q4cdn.com`, `*/investor-relations/`)
  - Official company/product docs (`platform.*.com`, `docs.*.com`, `*.company.com/press/`)
  - Scientific registries (`who.int`, `cdc.gov`, `ecdc.europa.eu`, `pubmed.ncbi.nlm.nih.gov`)
  - Central banks and macro agencies (`federalreserve.gov`, `ecb.europa.eu`, `imf.org`, `worldbank.org`, `bls.gov`, `bts.gov`, `census.gov`, `tsa.gov` DOE, etc.)
  - Wire services (AP, Reuters) are acceptable secondary.

- Also add this steer to the benchmarking-safe version (no prediction markets during benchmarks) so the primary-source nudge is active even when the market nudge is silenced.

**Tests.**
- `tests/test_prompts.py` — golden-file: `web_research_prompt(...)` contains the primary-sources section.
- Confirm `_benchmarking_warning` coexists with the new bullet.

**Verification.**
- `make test`.
- Smoke backtest — eyeball 2-3 rationales for `.gov` / `docs.*.com` citations appearing more frequently.

**Dependencies.** None. Independent.

---

### Workstream G — Prediction-market provider (Polymarket + Kalshi + Manifold as peers)

**Evidence.** Research agent report (2026-05-12, agent `a7dfce9170c8e5737`). Key endpoints:

- Polymarket Gamma search: `GET https://gamma-api.polymarket.com/public-search?q=<q>` (4000 req/10s, unauth).
- Polymarket market probability: `outcomePrices` field on market object (JSON-encoded string array). Or CLOB `/midpoint?token_id=<id>` when `enableOrderBook=true`.
- Kalshi markets list: `GET https://external-api.kalshi.com/trade-api/v2/markets?status=open&series_ticker=<t>` (unauth). No keyword search — paginate + client-side fuzzy match on `title` / `rules_primary`.
- Kalshi probability: `yes_bid_dollars`, `yes_ask_dollars`, `last_price_dollars` (dollar-denominated, binary = probability).
- Manifold search: `GET https://api.manifold.markets/v0/search-markets?term=<q>&contractType=BINARY` (500 req/min, unauth). `probability` field is direct.

**G0 — keyword-extraction experiment (precursor).** Before building the provider, dispatch a subagent with `isolation: worktree` to:

1. Select ~15 diverse open or recently-closed Metaculus questions from our real tournament set (`spring-aib-2026`, `metaculus-cup-spring-2026`, `market-pulse-26q2`) — mix of AI, politics, financial, science, sports.
2. For each question, try N keyword-extraction strategies against all three platforms' search endpoints:
   - S1: first 60 chars of `question_text`.
   - S2: `question_text` truncated at the first `?` or first comma.
   - S3: first sentence of `question.title` + first sentence of `question.resolution_criteria`.
   - S4: LLM-extracted 3-5 noun phrases via `gpt-5-mini` with a focused prompt.
   - S5: LLM-extracted keywords using the question's primary entity + event + deadline.
3. For each (question, platform, strategy), record: number of candidate matches returned, best match's fuzzy-similarity score (via `rapidfuzz`), best match's resolution-criteria alignment judgment (manual eyeball or LLM-judged).
4. Report: per-strategy hit rate (fraction of questions with ≥1 good match per platform), match precision (fraction of matches that are genuinely the same question), cost per question, and a recommended default strategy (or hybrid).
5. Write findings to `scratch_docs_and_planning/prediction_market_keyword_extraction_experiment.md` with raw CSV artifacts.

This experiment drives the G design — no guessing. Expected output: a recommended default + a feature flag for switching strategies.

**Scope.** One provider module, `metaculus_bot/prediction_market_provider.py`. All three platforms. Async via `aiohttp` (already a repo dep; no new HTTP client).

- Public function: `async def fetch_market_snapshot(question: MetaculusQuestion, *, platforms=("polymarket","kalshi","manifold"), max_matches_per_platform=3, timeout=5.0, as_of=None) -> MarketSnapshot`.
- Title extraction: driven by G0 findings (probably hybrid, but TBD until the experiment lands).
- Match confidence: fuzzy match via `rapidfuzz` (pure-Python dep, lightweight) on title + description. Drop matches below a configurable threshold.
- Return dataclass:
```python
@dataclass
class MarketMatch:
    platform: Literal["polymarket","kalshi","manifold"]
    market_title: str
    market_url: str
    implied_prob_yes: float
    bid: float | None
    ask: float | None
    spread: float | None
    volume_24h: float | None
    close_time: datetime | None
    is_resolved: bool
    match_confidence: float
    raw_rules: str  # for the LLM to judge resolution-criteria alignment

@dataclass
class MarketSnapshot:
    matches: list[MarketMatch]
```

- Formatter: one fenced block with columns `platform | title | prob | vol | close | conf`, plus a one-line "NOT AN ANCHOR — verify the resolution criteria match" caveat, plus the raw rules for each match.

**Backtest leakage gate (critical, per the research report's §6).** Resolved markets on these platforms keep their last-trade price *after resolution*; pulling them during a resolved-question backtest is leakage. Two-layer defense:

- `as_of` parameter filters out matches where `close_time <= as_of`. Required in backtest; optional in prod.
- `backtest.py` sets `PREDICTION_MARKETS_ENABLED=false` by default. User must opt in explicitly.

**Env flag.** `PREDICTION_MARKETS_ENABLED` env var (`constants.py`). Default OFF until smoke + medium backtest pass; **then flipped ON in all 4 production workflows** per user decision 2026-05-12. Backtest leakage defense (the `as_of` filter) is the load-bearing guardrail in prod, not the flag itself.

**Tests (TDD, with mocked HTTP).**

- `tests/test_prediction_market_provider.py`:
  - Polymarket match + probability extraction from fixture JSON.
  - Kalshi title-match and probability extraction from fixture JSON.
  - Manifold match from fixture JSON.
  - `as_of` filter drops post-as-of matches.
  - Malformed response returns empty snapshot (fail-visible, log WARNING).
  - Rate-limit backoff: on 429, retry with exponential backoff (bounded).
  - Timeout returns empty snapshot (soft-fail).
- `tests/test_prediction_market_integration.py` (opt-in, real HTTP — skipped in CI): one search against each platform's real API, assert non-empty response, assert probability parses.

**Verification.**

- `make test`.
- Run with flag ON against 3-5 hand-picked open questions where we expect matches; eyeball the snapshots.
- Medium backtest — confirm `as_of` leakage defense actually filters resolved markets.

**Dependencies.** None, but touches `research_providers.py:_select_research_providers` to add the new opt-in provider (parallel fan-out alongside AskNews, Gemini, native-search, financial-data). Plumbs a `rapidfuzz` dep into `pyproject.toml`.

**Rollback.** Flip `PREDICTION_MARKETS_ENABLED=0`.

---

### Workstream H — AGENTS.md corrections

My earlier rewrite had two errors. Fix now, independently.

- Fix Providers section: Perplexity + Exa are **priority-ordered fallbacks to AskNews**, not parallel providers. Clarify that in production (AskNews creds set) Exa/Perplexity/OpenRouter do NOT run. The parallel fan-out is AskNews (primary) + native Grok + Gemini grounded + financial-data.
- Fix research-flag discussion: all four production workflows set `NATIVE_SEARCH_ENABLED=true`, `GEMINI_SEARCH_ENABLED=true`, `FINANCIAL_DATA_ENABLED=true`, `GAP_FILL_ENABLED=true`. Confirm in AGENTS.md rather than implying opt-in means "probably off in prod."

**Dependencies.** None.

---

## Cross-cutting concerns

### Tests before code, always

For each workstream, the subagent's packet specifies: (1) which tests must be written and fail first, (2) which files to edit, (3) what the green state looks like. TDD agent (subagent type: `tdd-coder`) is the right choice for the implementation subagents.

### Backtest budget

Each workstream requires a medium backtest (~32 questions, ~15 min per run). With ON/OFF A/B that's ~30 min per workstream. Budget ~4 hours of backtest wall time across all workstreams. Run smokes first to catch parser errors before burning backtest budget.

### Dependency graph

```
A (tail-widening)   ──┐
B (binary clip)     ──┤
H (AGENTS fix)      ──┤── [ship independently, any order]
F (primary sources) ──┘

C (tool activation) ──┐
                      ├── E (numeric format) ── [depends on D's mixture math]
D (new tools)       ──┘

G (market provider) ──── [independent, but best after A/B/F land for baseline]
```

Phase 1 (parallel): A, B, F, H. One subagent per workstream, all in separate worktrees.
Phase 2: C (alone first, land it clean), then D (library-only extension), then E (consumes D).
Phase 3: G.
Final: `/review --fresh` full repo review, then merge each phase to main.

### Worktrees

Every implementation subagent runs with `isolation: "worktree"` so they don't stomp on each other's edits. The auto-merge hook (see user's global CLAUDE.md) handles rolling their changes into the parent tree on completion.

### Model choice for subagents

- `tdd-coder` (inherited, i.e. Opus 4.7) for workstreams C, D, E, G (non-trivial multi-file).
- `tdd-coder` (Sonnet) for workstreams A, B, F, H (mostly mechanical).

### Unified code review

When every workstream merges to main, run `/review --fresh` on the full diff vs. `origin/main`. Address every finding before flipping production flags.

## Verification plan

1. **Per-workstream unit tests** via `make test`. Green gate.
2. **Smoke backtest** (`make backtest_smoke_test`, n=4) per workstream after implementation. Confirms pipeline doesn't blow up.
3. **Small backtest A/B** (`make backtest_small`, n=12) on workstreams that change forecaster behavior: A, B, C, E. Compare Brier / log-loss / PIT std per cohort.
4. **Medium backtest A/B** (`make backtest_medium`, n=32) on the combined Phase 1 + Phase 2 stack. Budget: flag-on vs. flag-off delta on:
   - Binary: Brier, log-loss, clamped-tail rate.
   - Numeric: PIT std toward 0.289, 50%/90% coverage toward 0.5/0.9.
   - MC: log-loss.
5. **Full-fat backtest** (`make backtest_large`, n=100) before flipping defaults to ON in production workflows.
6. **Production flip**: one workstream per week, with residual-analysis follow-up after ~30 new resolved questions.

## Explicitly out of scope

- **Learned calibration models.** User ruled out fitted calibration (isotonic, Platt, etc.). Workstream A's tail-widening study is *evidence-gathering* for a hyperparameter flip, not a trained model. No serialized models, no holdout splits.
- **Atlas's 0.96*p + 0.02 linear shrink.** Rejected in favor of clip-only (Workstream B).
- **Replacing the percentile path with mixture.** Rejected in favor of either-or (Workstream E).
- **Consistency checks between percentiles and mixture.** User rejected the "emit both, we judge" framing.
- **Agentic ReAct research loop.** Separately tracked in FUTURE.md; not part of this plan.
- **Per-category tail_widening.** Empirical study says sample size per category is too small to justify per-segment tuning. Single global `k_tail=1.0` is the call.

## Open questions for the user (before I start)

1. **Phasing vs. batching.** Phase 1 (A, B, F, H) is low-risk. Ship them as one stacked PR to main after one round of backtest confirmation, or individual PRs? I'd suggest one stacked PR unless you prefer to isolate Brier deltas per change.
2. **Prediction-market default in GH Actions.** Workstream G defaults OFF. Do we want it ON in production workflows after validation, or keep it backtest-only for another round? My recommendation: backtest-only for the first 2-3 weeks of resolved data, then flip prod on if Brier delta is positive.
3. **Mixture math location.** D's mixture tools could live in `distributions.py` instead of a new `mixtures.py`. Preference? I lean toward a new file because `distributions.py` is already ~250 lines and mixtures + their Metaculus-CDF adapter are a self-contained 150-200 lines.
4. **LLM keyword extraction for G.** Do we use a tiny `gpt-5-mini` call for keyword extraction (+$ cost) or just truncate `question_text` to 60 chars (+noise)? Start simple, upgrade later — or start right?

Once these are settled I'll start dispatching subagents. Default assumptions if no reply: (1) one stacked PR per phase, (2) backtest-only for G initially, (3) new `mixtures.py`, (4) `question_text` truncation, upgrade later.
