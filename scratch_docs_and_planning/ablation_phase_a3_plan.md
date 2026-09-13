# Probabilistic-Tools Ablation — Phase A.3 Plan

**Last updated**: 2026-05-14, after Phase A.2 (n=8 paired comparisons) + outlier diagnosis.
**Owner**: User. Picked up by a fresh-context Claude session via this doc + the artifacts on disk.

This doc replaces the (stale) `scratch_docs_and_planning/probabilistic_tools_activation.md`
status section for the ablation work. It is intentionally complete enough that a
zero-context session can act on it without re-reading the conversation transcript.

---

## TL;DR

We have a working benchmark that produces real paired-Δ statistics on tools-on
vs tools-off stacker arms, but n is too small (n=8) and the directional signal
is dominated by one outlier (qid 43171, Δ_log_score=-227). Before scaling up,
we are doing four things in this phase:

1. **Diagnose how outliers happen** — done, see `## Outlier diagnosis` below
2. **Make the summary outlier-robust** — design below, implement here
3. **Add a manual-review QA stage between screen and forecast** — design + implement
4. **Re-fire Phase A.2 cleanly + push to Phase A.3 (~50q)** once 1-3 land

---

## Where we are (state of the world)

### Pipeline status — all stages working end-to-end

`metaculus_bot/ablation/` package:

| Stage | Status |
|---|---|
| `fetch` | Stratified fetch from Spring 2026 tournament; respects manifest delta semantics |
| `research` | Gemini 2.5 Flash grounded, no gap-fill (gap-fill amplifies leakage on resolved qids) |
| `prune` | Headless `claude -p` redactor; ~42% prune-validation failure on non-screened qids |
| `screen` | GLM-4.5-air structured-JSON detector; ~30-58% catch rate depending on type mix |
| `forecast` | 5 free OpenRouter models, plain GeneralLlm (paid key only), gentle/patient retry modes |
| `stack_a` (tools-off) + `stack_b` (tools-on) | Claude Opus 4.5 primary via donated key, gpt-5.5 fallback |
| `score` | Paired-Δ with bootstrap CI, sign test, Wilcoxon |

Test count: 350+ ablation tests passing, full repo 1700+ passing, lint clean.

### Phase A.2 result (latest, 2026-05-14)

- 19 qids in manifest (12 fetched this round + 7 carried over)
- 8 paired comparisons in summary (6 numeric + 2 MC, 0 binary — all leaked)
- All 16 stacker calls (8 qids × 2 arms) succeeded on primary opus-4.5
- Total spend ~$3 (mostly Bedrock for redactor)
- Summary file: `backtests/ablation/scores/summary_20260514_223311.md`

Headline (n=8):

| Metric | n | Mean Δ (B−A) | 95% CI | Read |
|---|---|---|---|---|
| CRPS | 6 | +0.0049 | [-0.006, +0.018] | tools-OFF wins (small) |
| Numeric log-score | 6 | -37.0 | [-113.5, +3.1] | tools-OFF wins (large, outlier-driven) |
| MC log-score | 2 | -0.50 | — | tools-OFF wins (small) |

Mean Δ_log = -37 is misleading: dominated by qid 43171 Δ=-227. Median Δ would
be ≈ 0.

### Leakage rates (Phase A.2 round)

By question type, of the 12 newly fetched:

- Binary: 0 of 5 surviving the screen (100% leakage rate) — questions like "did event X happen by date Y" trivially answered post-resolution
- MC: 1 of 5 surviving (~80% leakage) — "which option was top in [event]" similarly
- Numeric: most surviving (~30% leakage) — specific values are harder for the LLM to recall verbatim

**Implication**: any future fetch should over-allocate binary and MC to net the
target type-balance. For 50q with target 10/15/25 final, fetch ~40/60/35.

### Cache state

- `backtests/ablation/qids.json` — 19-entry manifest (binary 4, MC 5, numeric 10)
- `backtests/ablation/research/` — 13 raw blobs
- `backtests/ablation/research_pruned/` — 8 sanitized (5 missing = prune validation failed)
- `backtests/ablation/leakage_screens/` — 13 verdicts (6 leaked, 7 clean)
- `backtests/ablation/forecaster_outputs/` — outputs for 7-12 qids depending on cell
- `backtests/ablation/stacker_outputs/` — 8 qids × 2 arms (success), rest had insufficient_forecasters
- `backtests/ablation/scores/summary_20260514_223311.md` — most recent

---

## Outlier diagnosis (qid 43171)

### Question

"What will US airline passenger volume be for these weeks in April 2026,
according to the TSA? (Apr 13–19, 2026)"

- Bounds: 13M – 21M passengers
- Resolution: 17,212,999 (above the median of all five forecasters)

### Research blob — almost empty

The Gemini 2.5 Flash grounded search returned only 3 lines explaining
"TSA does not provide future projections... requested information is not
available." So both arms are forecasting from prior knowledge alone, with
no actual data on TSA volumes for the resolution week.

### Forecaster outputs (per-model P10/P50/P90)

| Forecaster | P10 (M) | P50 (M) | P90 (M) | Width | Captures resolution? |
|---|---|---|---|---|---|
| gemma-4-26b | 17.30 | 18.60 | 20.30 | 3.0M | ✓ resolution at ~P5 |
| minimax-m2.5 | 15.01 | 16.80 | 18.59 | 3.6M | ✓ resolution at ~P73 |
| nemotron-3-super | 15.42 | 16.04 | 16.78 | 1.4M | ✗ resolution above P90 |
| qwen-3-next | (rate-limited 4×, dropped) | | | | |
| **glm-4.5-air** | **15.64** | **15.66** | **15.68** | **0.04M** | ✗ resolution far above P95+ |

### GLM-4.5-air hallucinated partial-week data

GLM's reasoning (verbatim):
> "Total week = 15.661M if ratio holds (high confidence). First 3 days (7.2M)
> exceed 2025 (6.9M) by 4.35%, supporting continued growth. Stable historical
> ratios (1.174-1.176)."

**None of that data was in the research blob.** GLM made up TSA daily volumes
that "happened" through April 16 and then locked in σ=13,000 (0.08% of mean).

### Tool runner output (arm B only)

```text
- **Forecaster medians**: min 1.57e+07, max 1.86e+07, n=4
- **Declared distribution families**: lognormal, normal (4 forecasters)
```

Plus per-forecaster:

```text
glm-4.5-air: claimed 'normal', best-fit 'normal' (parametric family validates)
```

The family-check ratifies GLM's parametric claim WITHOUT flagging the
implausibly tight σ. Arm B's stacker reads "claimed normal, best-fit normal
— looks fine!" and over-weights GLM.

### Why the arms diverged

- **Arm A** (tools-off): stacker reasoned "Model 4's σ ≈ 13K is unrealistic
  for HIGH forecastability" and added a 15% mixture component at 17.2M to
  hedge. Got log-score -3.82 (small loss).
- **Arm B** (tools-on): stacker read "GLM family validates → trust it" and
  rejected the higher-median forecasters. Got log-score -219.98 (saturated
  at floor).

### The -219.9756 number

Decoded: it's `50 * log(pmf / baseline)` where `pmf` is the minimum-step CDF
floor (~5.5e-5 with two open bounds) and `baseline = 0.0045`. So:

```text
50 * log(5.5e-5 / 0.0045) ≈ -220
```

**Not a hard floor** — it's the score for "max-confident wrong at the
schema's min-step minimum spread." Different qids saturate at slightly
different values depending on bounds + zero_point, but cluster near -220.

### Generalization

n=1 outlier; can't conclude tools-on systematically tightens bad
distributions. But the **mechanism** is real: tool-runner output validates
forecaster shapes without flagging implausible spreads, and that omission can
boost a confidently-wrong forecaster.

**Hypothesis to test at scale**: tools may help when forecasters disagree
informatively, hurt when one forecaster is confidently-and-narrowly wrong.

---

## Other saturated cases (qids 43129 + 42747)

Both arms hit -219.9756 identically (Δ=0). Spent < 30 sec inspecting:

- 43129: TSA week 27 (May 3 2026) — research blob refused to provide future data → all forecasters guessing from prior knowledge → both arms equally confident-wrong
- 42747: another future-volume question with similar structure

**These are non-informative draws** (Δ=0 contributes nothing to mean), not
outliers. Worth flagging in summary as "saturated" so the operator
understands they're not real comparisons.

---

## Plan: four work packages

The phases are ordered by dependency. Package 1 is non-load-bearing
diagnostics; package 2 is summary-output cleanup; package 3 is the QA
harness; package 4 is the next live run. Packages 1-2 can land in parallel.

### Package 1: outlier-robust summary

**Goal**: surface saturation events and use median statistics that aren't
dominated by single saturated outliers.

**Where to edit**: `metaculus_bot/ablation/scoring.py` (the `aggregate_paired`
function and `render_summary_markdown` function).

**Concrete changes**:

1. **Detect saturation per (qid, arm, metric)**. A score is "saturated" if it
   sits within ε of the schema's max-confident-wrong floor:
   - For numeric_log_score: `score <= -200 + ε` (where ε ≈ 25 to catch
     variations across questions with different bounds)
   - For mc_log_score: `score <= -log2(K) * 100 + ε` where K is the number of
     options (the floor for max-confident-wrong on K-way MC is also bounded)
   - For binary_log_score / brier: similar floor logic; binary clip to [0.02,
     0.98] sets a known floor at log(0.02) ≈ -50 in the Metaculus 100-scale
   - For CRPS: no comparable floor (CRPS is bounded by the resolution range
     × max-spread, which doesn't compress to a single number)

   Implementation: extend `score_arm_for_qid` to also return a `saturated:
   bool` flag per metric. Thread it through `PairedScore` and the summary
   table.

2. **Add median Δ + bootstrap CI on median to overall + per-type rows**.
   Mean stays in the table for continuity; median is a new column. Bootstrap
   CI on median is straightforward — same `np.random.choice` resampling, just
   compute median rather than mean per resample.

3. **Add a "saturation" column to the per-question diagnostic table**.
   Marks each row with one of:
   - `"clean"` — neither arm saturated
   - `"a_sat"` — only arm A saturated (informative, B did better than the floor)
   - `"b_sat"` — only arm B saturated (informative the other way)
   - `"both"` — both arms saturated (Δ=0 by construction, non-informative draw)

4. **Add a "non-saturated mean Δ" row alongside the headline mean**, so the
   operator can see at a glance whether the mean is robustness-of-floor or
   a real signal.

5. **Caveats section**: add a paragraph explaining what "saturated" means and
   why both-saturated rows are draws.

**Tests** (TDD):

- `test_score_with_resolution_below_p10_floor_marks_saturated` — synthesize a
  prediction where the resolution falls in a bucket with min-step PMF mass;
  assert the saturation flag fires for numeric_log_score.
- `test_median_delta_robust_to_single_outlier` — synthesize 5 paired Δs
  including one extreme outlier; assert mean diverges, median doesn't.
- `test_summary_marks_both_saturated_rows_as_draws` — render summary
  markdown, assert the "saturation" column is present and "both" rows are
  flagged.

**Estimated diff**: ~150 LOC including tests. Scoring code changes are
mechanical; the saturation detection is the hardest bit (it's per-metric).

**Sign-off needed**: how aggressive should ε be for numeric_log_score? My
recommendation is ε=25 (catches anything ≤ -195) so we don't miss
near-saturated cases that are equivalent in practice. Could go tighter (ε=5,
catches only exact saturations) if we want sharper definitions.

---

### Package 2: pre-forecast QA harness

**Goal**: between `screen` and `forecast`, give the operator (and a subagent)
a chance to review surviving qids and reject ones that look problematic
before spending forecaster + stacker budget on them.

**Motivation**: qid 43171 is a "research blob is empty, forecaster
hallucinated" failure mode. The screen stage couldn't catch it because the
problem isn't leakage — the problem is that the question is **unforecastable
from the available research**. A human/subagent review pass between screen
and forecast catches these.

**New stage**: `qa_review` (slots between `screen` and `forecast`).

**Behavior**:

1. After `screen` produces clean qids, `qa_review`:
   - Reads each surviving qid's question text + sanitized research blob +
     screen verdict + ground truth from disk
   - Skips qids already in `backtests/ablation/manual_rejects.json`
   - Runs a Claude subagent (sonnet model, no LLM budget impact since this is
     local Claude Code spend) to score each qid on:
     - **Forecastability**: does the research blob contain enough signal to
       make a non-trivial forecast? Or is it "empty" (e.g., 43171's case where
       Gemini explicitly refused)?
     - **Subtle leakage**: a second-pass leakage check at a different angle
       than the LLM-based screen (e.g., does the question text + research
       implicitly reveal the answer through phrasing?)
     - **Question quality**: is the resolution criteria well-defined? Are the
       bounds reasonable for the question's content?
     - **Hallucination risk**: does the research blob's emptiness invite
       forecasters to fill in fake data? (Like GLM did on 43171.)
   - Writes a structured QA report per qid: `backtests/ablation/qa_reports/<qid>.json`
   - Aggregates into `backtests/ablation/qa_summary_<timestamp>.md`

2. Each qid gets a recommendation: `accept | reject | needs_human_review`.
3. Auto-rejects (with operator-set thresholds) update `manual_rejects.json`;
   `needs_human_review` are surfaced in the summary for the operator.

4. The operator reviews `qa_summary_<timestamp>.md`, edits
   `manual_rejects.json` as needed, and re-runs.
5. If `--qa-review` flag is set on the CLI, the run halts here. If unset, the
   stage runs in advisory mode (logs to summary but doesn't halt) so existing
   no-QA invocations work unchanged. Default: run-but-not-halt.

**Concrete change set**:

- New module `metaculus_bot/ablation/qa_review.py`:
  - `run_qa_for_qid(qid, question, research_blob, screen_verdict, ground_truth) -> QAReport`
  - `run_qa_batch(qids, working_set) -> dict[int, QAReport]`
  - `QAReport` dataclass with `forecastability_score: float`, `subtle_leakage_risk: float`, `question_quality_score: float`, `hallucination_risk_score: float`, `recommendation: Literal[...]`, `notes: str`

- Subagent prompt: scopes the subagent to read 1-3 cached files (question
  metadata, research blob, screen verdict) and emit a structured JSON
  recommendation. Subagent uses sonnet model. No external LLM budget.

- Integration in `metaculus_bot/ablation/cli.py`:
  - Add `qa_review` to the `STAGES` list
  - Add `--qa-review-mode {advisory, halt, skip}` flag
  - In `run_ablation`, when `qa_review` is in `requested` stages, run
    `_stage_qa_review` between `_stage_screen` and `_stage_forecast`
  - The stage drops manually-rejected qids from `working.research_blobs` so
    forecast/stack/score don't see them

- New file: `backtests/ablation/manual_rejects.json` — schema:

  ```json
  {
    "rejects": {
      "43171": {
        "rejected_at": "2026-05-15T10:00:00",
        "reason": "GLM hallucinated TSA partial-week data, research blob empty",
        "operator": "manual"
      }
    },
    "version": 1
  }
  ```

- Tests: `tests/test_ablation_qa_review.py` — mock the subagent's response
  via patching the subagent invocation; assert the harness reads the right
  files, writes the QA report correctly, drops manually-rejected qids.

**Estimated diff**: ~400 LOC (new module + cli integration + tests).
Significant but contained. Most of the work is the subagent prompt design +
testing the harness.

**Cost discussion**: subagent runs use Claude Code's parent process (not API
billing). At 50q it's ~50 subagent dispatches, each ~5-10 tool calls (read
files, output JSON). Wall-clock ~5-10 minutes per run.

**Sign-off needed**:

- (a) Is "advisory by default" the right semantics? Or should every run halt
  by default and require `--no-qa-halt` to proceed without operator review?
- (b) Should the manual-rejects list be repo-checked-in (committed) or
  operator-local? Repo-checked-in means the team's reject decisions are
  shared; operator-local means each operator curates their own.
- (c) What's the right model for the subagent (sonnet good enough, or opus)?

---

### Package 3: forecaster reliability hardening

**Goal**: reduce forecaster-side failures (rate limits, timeouts) and
forecaster-side hallucinations.

**Three sub-problems**:

#### 3a. Free-tier rate limits at scale

At 50q × 5 forecasters = 250 calls, we'll exceed Venice/OpenInference
per-minute throttles even with `gentle` mode. Two paths:

(i) **Reduce concurrency further** to `patient` mode (concurrency=1,
max_retries=8). Wall-clock impact: forecaster stage takes ~25-30 minutes per
50q run instead of ~10. Cheap, no code changes.

(ii) **Switch to BYOK paid models** for the forecaster ensemble. e.g., use
gpt-5-mini ($0.05/M input, $0.40/M output) instead of free OpenRouter
variants. Predicted cost: 5K input × 5 models × 50q = 1.25M tokens × $0.05/M
= ~$6 per 50q run. Adds reliability and removes the "GLM hallucinates"
problem since gpt-5-mini is more reliable.

(iii) **Mixed**: keep 3 free forecasters (for diversity / cost) + 2 paid
(for reliability anchor). Compromise; ~$3-4 per 50q.

#### 3b. GLM-4.5-air hallucinations

GLM made up partial-week TSA data on 43171. Two options:

(i) **Drop GLM from the lineup**. Replace with another free or paid model.
Reduces lineup to 4 free models, which is fine.

(ii) **Add a reasonability check post-forecast**: if forecaster's σ is < 1%
of the bounds range, flag for review. Wouldn't fix 43171 (we'd still spend
the call) but would surface it before the stacker reads it.

(iii) **Add an anti-hallucination instruction to the forecaster prompt**:
"If the research blob says no data is available for this period, do NOT
fabricate data; say so explicitly and widen your distribution accordingly."
Cheap; might help; might not.

#### 3c. Tool runner blind spot (variance/spread sanity check)

The tool runner validates parametric families but not implausible spreads.
On qid 43171 it ratified GLM's "claimed normal, best-fit normal" without
flagging σ=13K vs ensemble spread of 1-3M.

Fix: add a **spread plausibility check** to the tool runner's per-forecaster
output. If a forecaster's σ is < 10% of the median ensemble σ, emit a
WARNING line: `⚠ Spread anomaly: σ=13K is 1.3% of ensemble median σ=965K`.
This gives the stacker a signal to discount the suspect forecaster.

Touches `metaculus_bot/tool_runner.py` (production code, shared with main.py
ensemble run). Should be a cheap addition. Affects production.

**Estimated diff for 3a-3c combined**: ~100-150 LOC + tests.

**Sign-off needed**:

- (a) Free vs paid forecasters? My recommendation: option (iii) — mixed
  lineup, 3 free (gemma-4, nemotron, minimax) + 2 paid (gpt-5-mini + claude
  haiku-ish). Total cost ~$5/50q. Keeps free-tier cost discipline while
  anchoring reliability.
- (b) Drop GLM from lineup? Recommendation: yes — its 43171 behavior was
  wild. Replace with a different free model or one of the paid anchors.
- (c) Tool runner spread check? Recommendation: yes — defense-in-depth, very
  small diff, helps both ablation and production.

---

### Package 4: Phase A.3 (50q intermediate)

After packages 1-3 land:

**Fetch plan**:

- Target: 50 final-clean qids in summary
- Allowing for ~50% leak rate + ~10% manual-QA reject + ~20% prune fail =
  fetch ~125 qids with type-balanced split
- Type breakdown: aim for 15 binary, 15 MC, 20 numeric in final summary
  → fetch ~60 binary, ~50 MC, ~30 numeric (binary leaks heavily)

**Predicted spend**:

- Gemini: ~125 free calls (well under daily quota)
- Redactor: ~13 batches × ~$1 = ~$13 Bedrock
- Forecasters: ~50 surviving × 5 (or 4 if GLM dropped) = ~250 calls
  - All-free: $0 + retry overhead
  - Mixed: ~$5-6
- Stacker: ~50 × 2 arms × opus-4.5 ≈ 100 calls × ~$0.06 = ~$6 (donated
  absorbs)
- **Total: $13-25** depending on forecaster lineup choice

**Run command shape** (assuming default `--qa-review-mode advisory`):

```bash
make ablation_phase_a3  # new Makefile target
```

which expands to:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python -m metaculus_bot.ablation.cli \
  --num-binary 60 --num-multiple-choice 50 --num-numeric 30 \
  --no-gap-fill --gemini-model gemini-2.5-flash \
  --rate-limit-mode patient \
  --qa-review-mode advisory
```

**Sign-off**:

- Sign-off on packages 1-3 first (in any order)
- Then sign-off on Phase A.3 spend ($13-25 estimated)
- Run completes, summary written, operator reviews
- If signal looks robust (n=50, multiple metrics consistent), can scale to
  ~100q for medium-power statistics; otherwise iterate

---

## Open questions / known issues (carry-overs)

1. **MC and binary leakage rates are very high** (75-100% on this
   tournament's window). Worth investigating: is the issue (a) the screen
   detector being too aggressive, (b) Gemini grounded actually returning
   leaked content, or (c) the question types are inherently leakier? At 50q
   we'll have more data to triage.

2. **The redactor ~42% prune-validation failure rate** is high. Worth
   investigating: is the redactor failing on truly un-redactable blobs (the
   answer is too pervasive to remove), or is the redactor just imperfect? A
   subagent running in advisory mode on a sample of 5 prune-failures could
   tell us.

3. **The May 12 gpt-5-mini spike (422 calls)** never got investigated. Low
   priority but worth a 5-min look: was it from a backtest run, a manual
   `--mode test_questions` invocation, or something else?

4. **`stacking.py` MC-prompt fix is in production code** (touched by Bucket
   2.5). Production runs have been emitting the new "real option names"
   format since 2026-05-14. Worth a smoke-check on the next minibench run
   to confirm production parser still extracts correctly with the new
   format. If gpt-5-mini handles it cleanly, no action needed.

---

## Critical files for fresh-context pickup

If a session opens this doc and needs to navigate:

- `metaculus_bot/ablation/cli.py:_build_parser` — argparse surface
- `metaculus_bot/ablation/cli.py:run_ablation` — orchestrator
- `metaculus_bot/ablation/cli.py:STAGES` — pipeline order (after package 2:
  `["fetch", "research", "prune", "screen", "qa_review", "forecast", "stack_a", "stack_b", "score"]`)
- `metaculus_bot/ablation/scoring.py:aggregate_paired` — paired-Δ stats (touched in package 1)
- `metaculus_bot/ablation/scoring.py:render_summary_markdown` — summary rendering (touched in package 1)
- `metaculus_bot/ablation/qa_review.py` — NEW in package 2
- `metaculus_bot/ablation/forecaster_lineup.py:FREE_FORECASTER_MODELS` — touched in package 3 if lineup changes
- `metaculus_bot/tool_runner.py` — touched in package 3 (spread sanity check)
- `backtests/ablation/manual_rejects.json` — NEW in package 2
- `backtests/ablation/qa_reports/<qid>.json` — NEW per-qid in package 2
- `backtests/ablation/qa_summary_<timestamp>.md` — NEW per-run in package 2
- `tests/test_ablation_*.py` — extend per package

## Run cheat sheet

```bash
# Verify state matches this doc:
~/miniconda3/envs/metaculus-bot/bin/python -m pytest tests/test_ablation_*.py -q
make lint
ls backtests/ablation/scores/  # most recent: summary_20260514_223311.md

# Required env (in addition to what .env already provides):
export GOOGLE_API_KEY=$(cat ~/.keys/gemini_key)
export OPENROUTER_API_KEY=$(grep -E "^OPENROUTER_API_KEY=" .env | head -1 | cut -d= -f2-)
unset PROBABILISTIC_TOOLS_ENABLED GAP_FILL_ENABLED GEMINI_SEARCH_MODEL

# After package 1 lands, re-render summary on existing cached results:
python -m metaculus_bot.ablation.cli --stages score
# (Should produce a new summary file with median Δ + saturation flagging.)

# After package 2 lands, retroactively QA the existing 19 qids:
python -m metaculus_bot.ablation.cli --stages qa_review --qa-review-mode halt
# (Operator reviews the summary, populates manual_rejects.json, then:)
python -m metaculus_bot.ablation.cli --stages forecast,stack_a,stack_b,score --force-stages forecast,stack_a,stack_b
```

## Sign-off discipline (carries forward)

Any live run remains sign-off-gated. State predicted spend, wait for "go",
then run. Same as before.

## Rough ordering of next session

1. Read this doc top to bottom (~5 min).
2. Verify state via the cheat-sheet's pytest + lint + ls commands.
3. Confirm package ordering with the user. My recommendation:
   1. Package 1 (summary robustness) — fastest to land, immediate value
   2. Package 3 (forecaster + tool-runner hardening) — small, pre-fire
   3. Package 2 (QA harness) — biggest, most design, lands last before A.3
   4. Package 4 (Phase A.3 fire) — sign-off + run
4. Ask sign-off questions for whichever package starts first.
5. Implement via subagents per CLAUDE.md guidance (delegate non-trivial
   edits, especially the QA-harness module). Use TDD.
