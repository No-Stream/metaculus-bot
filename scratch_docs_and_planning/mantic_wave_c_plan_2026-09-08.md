# Mantic Wave C: per-bin elicitation, the supply probe's Mantic mode, and the live-data items (plan, 2026-09-08, revised after review)

Branch `mantic-competition`, tip e37b8ae. This is the design and execution plan for "Wave C" of
the Mantic (Crucible) tournament support. It is written for implementation agents with no other
context: every file, function and constant named below was read on this branch on 2026-09-08,
and every claim about Mantic's API was checked with free, read-only GETs the same day. Where
something is believed rather than verified, the text says "assumed".

**Revision note.** This is the second version, revised the same night against the adversarial
review at `scratch_docs_and_planning/mantic_wave_c_plan_review_2026-09-08.md` (verdict: sound
with changes). Every item on the review's "Changes required before implementation" checklist is
resolved in the text below; the four largest are: the aggregation rule for per-bin members is now
decided inside C1 with the arithmetic (section 2.6), the platform gate defaults to Mantic-only
with the Metaculus switch as a one-line constant (section 2.2), the 0.59 cap question is closed
by the corpus receipt (section 2.5), and the design is rebased on the aggregate tail floor that
landed on the working tree during the review (`numeric/out_of_range_floor.py`; sections 2.8 and
2.9).

Line numbers are as of commit e37b8ae; a forge review's fix agents were editing the working tree
while this plan was written (about fifty files at review time, including `numeric/pipeline.py`,
`member_forecast.py`, `constants.py`, `prompts.py` and `value_extraction.py`), so treat every
`file:line` as a locator to re-verify and every function, constant and test name as the stable
reference. The fan-out in section 5 starts only once that fix wave has landed and the tree is
quiet.

Nothing in this document spends money. `make test`, `make all`, reading files and unauthenticated
or token-authenticated GETs against the Mantic API are all free. Any run of `main.py` in a live
mode, any backtest or ablation, and `make test_live` go through the operator (the cost gate in
`AGENTS.md`).

## 1. Context

### What the repo is

`/Users/flatljan/personal/metaculus-bot` is a forecasting bot built on the forecasting-tools
0.2.92 framework. For each open question it runs research providers, fans out three frontier
large language models (LLMs) as independent forecasters, aggregates their forecasts (the pointwise
MEDIAN in production; numeric aggregation is done pointwise in cumulative distribution function
(CDF) space) and publishes a prediction plus a comment. Numeric, discrete and date questions are
all forecast the same way today: each LLM declares 13 standard percentiles
(`STANDARD_PERCENTILES` in `metaculus_bot/numeric/config.py`), the pipeline turns them into a
CDF by monotone PCHIP interpolation (Piecewise Cubic Hermite Interpolating Polynomial,
`metaculus_bot/numeric/pchip_cdf.py`) on the question's own value grid, and the members' CDFs are
medianed pointwise (`aggregate_numeric` in `metaculus_bot/numeric/utils.py`).

### What Mantic is

Mantic runs a bot-only forecasting competition ("Crucible") at competitions.mantic.com on a fork
of the open-source Metaculus platform. The API has the same shape as Metaculus. `--mode mantic`
runs the whole pipeline against it through `ManticClient` (`metaculus_bot/mantic.py`) on the
operator's personal keys only. Mantic pays $3 per forecast; a question costs the bot about $2.60.
The bot's Mantic user is `nostreambot-bot`, user id 81 (verified 2026-09-08 with a public GET of
`https://competitions.mantic.com/api/users/81/`: `"username":"nostreambot-bot"`, `"is_bot":true`).

Mantic scores quantitative questions by the bin the outcome lands in. A submitted CDF of
`inbound_outcome_count + 1` values defines a probability mass function (PMF) over the question's
bins plus one bucket below the range and one above it; the score is the log of the mass in the
resolved bucket. Series 1 used `50 * ln(p / baseline)` (baseline `1/N` inside the range, a fixed
0.05 for an out-of-range bucket); the Series 2 rules document says the unadjusted categorical
form `100 * (log_N(p) + 1)` will be used for every type (rules doc section 4, "A small
modification"). Either way, mass placed on a bin that cannot resolve is simply lost, and a bin
that resolves with near-zero mass is a cliff.

### Why Wave C

Phases 1 and 2 are built, gated green and committed (handoff:
`scratch_docs_and_planning/handoff-2026-09-08-mantic-phase2.md`). They made date questions first
class, added the Mantic-optimized prompt clauses and numeric fixes, and left six items for Wave C
(Phase 2 plan `scratch_docs_and_planning/mantic_phase2_plan_2026-09-08.md`, section "Wave C";
`FUTURE.md` "Mantic Crucible: Phases 1 and 2 shipped"):

- **C1** Per-bin PMF elicitation for enumerable grids (about 31 bins or fewer). Operator-APPROVED.
- **C2** A Mantic mode for the free supply probe (`scripts/supply_probe.py`), to measure forfeits
  per release hour and settle the cadence decision on evidence.
- **C3** Fast-path alertability in mantic mode, once the Series 2 window length is known.
- **C4** Median versus mean CDF aggregation replayed under Mantic's baseline formula.
- **C5** After 2026-09-20 12:00 UTC: read question 651's resolution to close the on-edge bin
  question, and one resolved Series 2 score to learn the coefficient Mantic actually uses.
- **C6** The starved outer tail stays a documented watch item.

The motivation for C1, made concrete. Preseason question 651 ("On which date will the S&P 500 post
its largest single-day percentage move between 8 and 18 September 2026?") has 12 one-day bins,
2026-09-08 through 2026-09-19, both bounds closed. Its criteria name nine eligible trading days
(8, 9, 10, 11, 14, 15, 16, 17 and 18 September). Bins for 12, 13 and 19 September are weekend days
that cannot resolve. A percentile declaration cannot say "zero on those three bins": PCHIP
interpolation spreads mass smoothly across neighbouring days, so a quarter of the mass can land
on impossible bins. Under the Series 1 formula that costs `50 * ln(0.75) = -14.4` baseline points
with zero information content (the 14.4 figure quoted in the Phase 2 plan); under the Series 2
categorical form it is `100 * log_12(0.75) = -11.6`. A per-bin declaration says it directly.

Corpus facts behind the threshold (all 556 public Mantic posts, recorded 2026-09-08 in
`scratch_docs_and_planning/mantic_research_2026-09-08/mantic_all_posts_2026-09-08.json`, gitignored):
discrete questions 159, median 71 bins, 46 of them at 31 bins or fewer (29%), 63 at 50 or fewer;
numeric questions 156, of which 148 sit on the legacy 200-bin grid and 2 have 29 or 30 bins; date
questions 209, of which 208 are legacy 200-bin grids and one (651) has 12 daily bins. Series 2
changes the defaults: date questions default to daily bins with a one-day minimum, numeric
questions default to a power-of-ten step producing 100 to 1,000 bins, and question writers may set
a step size (rules doc sections 5 and 6). So coarse grids under Series 2 come from date questions
with short windows or week granularity, and from discrete count questions with writer-chosen
steps, which was already 29% of Series 1 discrete questions.

### The simpler alternative, and why per-bin elicitation wins

The review asked for the alternative to be named. It is: keep percentile elicitation and add one
optional `excluded_bins` list to the numeric and date blocks, zero those bins after the PCHIP
build, re-run the floor blend. One schema field, one post-processing step, no new template, no new
ladder; it keeps `sanitize_percentiles` and the unit-mismatch guard and solves the weekend case
exactly. It loses on the two shapes that are most of Mantic's coarse-grid population: (1) a count
question on 13 bins or fewer, where 13 strictly increasing percentile values cannot exist on the
grid, so the model repeats integers and the pipeline reads a degenerate declaration (the
`NUMERIC_DEGENERATE_DECLARATION` failure mode; 18 of the 46 coarse discrete grids have 13 bins or
fewer), and (2) any genuinely multimodal per-bin view (a vote count that lands on one of two
thresholds, a launch count that is 0 or 3), which a monotone interpolation through 13 anchors
cannot express at all. Per-bin elicitation is the natural declaration on a grid that IS the
outcome space, which is also why the multiple-choice path already elicits exactly that. Honest
size, taken from the review and accepted: three new modules plus about twelve touched source
files (about 1,000 to 1,200 lines), about twelve touched or new test files (about 1,000 lines),
six docs; roughly 2,200 lines for a per-bin loss of 11 to 14 points per affected question plus the
aggregation cliff in section 2.6, which the percentile path was only masking.

## 2. C1: per-bin PMF elicitation for enumerable grids

### 2.1 The one-paragraph design

On a Mantic question whose published bins are its outcome space and whose bin count is at most
`PMF_ELICITATION_MAX_BINS` (recommended 31), each forecaster is asked for one probability per bin,
keyed by a human-readable bin label, plus a `below_range` key when the lower bound is open and an
`above_range` key when the upper bound is open. The block is read by a new `pmf` rung family in
the extraction ladder (mirroring multiple choice), normalized, blended with the platform's minimum
per-bin floor, run through the existing `safe_cdf_bounds` enforcement, pinned, checked against the
server rules (fail shut), and wrapped in the same `create_pchip_numeric_distribution` object the
discrete path already produces. Per-bin members are aggregated by the linear opinion pool (the
pointwise MEAN of their CDFs, which `aggregate_numeric` already computes) instead of the pointwise
median, because a log score in the resolved bin punishes near-zero mass on a bin any member
believed (section 2.6). The publish path, the comment, the spread metric, the out-of-range
telemetry and the Mantic aggregate tail floor are untouched. The 201-point Metaculus continuous
path, and every Metaculus question, is byte-identical because the gate is false for them by
construction.

### 2.2 Gate: where "enumerable grid" is decided, and on which platform

**Decision.** Two constants and one predicate in `metaculus_bot/numeric/config.py`, next to
`grid_is_outcome_space`:

```python
# Bins at or below this count are elicited per bin (a probability per bin) instead of as
# percentiles: on a coarse grid the bins are the outcome space, 13 percentile anchors cannot
# express "zero on this bin", and PCHIP spreads mass onto bins the criteria exclude (question
# 651: three weekend days in a 12-day trading-day window, a quarter of the mass, -14.4 baseline
# points). A month of daily bins is the natural Series 2 date shape; 29% of Series 1 discrete
# questions sit at or below it. Above it the per-bin ask grows long and noisy and the 13-anchor
# curve is the better instrument.
PMF_ELICITATION_MAX_BINS: int = 31

# The platforms whose coarse grids are elicited per bin. Mantic only for the first landing: the
# ask is unproven live, and on Metaculus it would move about half of all discrete questions
# (141 of 300 sampled have 31 bins or fewer; the modal grids are 41, 31, 101 and 11 bins), a
# config-era change of its own. Adding PLATFORM_METACULUS here is that change, once one Mantic
# season shows the per-bin declaration is faithful.
PMF_ELICITATION_PLATFORMS: frozenset[str] = frozenset({PLATFORM_MANTIC})


def elicit_per_bin(question: NumericQuestion) -> bool:
    """True when ``question`` is forecast as one probability per bin instead of as percentiles."""
    return (
        question_platform(question) in PMF_ELICITATION_PLATFORMS
        and grid_is_outcome_space(question)
        and (question.cdf_size - 1) <= PMF_ELICITATION_MAX_BINS
    )
```

Rationale for the shape: `grid_is_outcome_space` (verified, `numeric/config.py:103-121`) is already
the repo's predicate for "the published bins are the outcome space and nothing downstream reshapes
them": true for every `DiscreteQuestion` (every Metaculus discrete question and every Mantic
quantitative question, whose wire type the client rewrites to `discrete`) and for every non-201
grid. It is false for a `NumericQuestion` on the 201-point grid, which is what keeps the Metaculus
continuous path untouched. `cdf_size` is `inbound_outcome_count + 1` on every real question
(framework `_get_cdf_size_from_json`, default 201), so `cdf_size - 1` is the bin count. A 200-bin
Mantic discrete question (`cdf_size == 201`, `DiscreteQuestion`) is an outcome-space grid but has
200 bins, so it stays on percentiles. `question_platform` (`metaculus_bot/question_platform.py`, a
leaf that reads the host off `page_url` and imports only `constants`) is the gate the Mantic prompt
clauses already use (`prompts.py:1223-1235`), so this is not a new mechanism; `numeric/config.py`
may import it under the import-linter contracts (numeric may not import research or the forecaster
stages; a leaf is fine). The epoch adapter carries `page_url` (`as_epoch_question` copies every
field it does not override), so the predicate works on the view the date runner holds.

**Platform scope, stated with the real numbers.** The review pulled 300 Metaculus discrete
questions read-only: 141 (47%) have 31 bins or fewer, 179 (60%) have 50 or fewer, and the modal
grid sizes are 41 bins (24 questions), 31 (22), 101 (19) and 11 (13). A platform-agnostic gate
would therefore change elicitation on about half of all Metaculus discrete questions, with no live
evidence yet that the per-bin ask is faithful, and it would have to land before the fall
FutureEval's first question on 2026-09-28. The grid argument (bins are the outcome space;
percentiles cannot express a per-bin view; PCHIP leaks mass onto excluded bins; Metaculus scores
discrete questions by the same bin log score) is genuine and is why the Metaculus follow-up is
recorded as a one-line change (`PMF_ELICITATION_PLATFORMS = frozenset({PLATFORM_MANTIC,
PLATFORM_METACULUS})`) rather than as never. Recorded default: Mantic-only for this landing.
Either answer is that one constant. Operator decision 2.

**Threshold value.** 31 is the operator's "about 31" and the Mantic corpus supports it (section 1).
Two things to know: 31 sits exactly on a Metaculus modal grid size (22 of 300 sampled Metaculus
discrete questions have 31 bins), so it is not a neutral cut there, which matters only if the
Metaculus switch is flipped; and the nearest alternative, 51, would also take the nine 51-bin
Series 1 discrete questions. Raise it only after live per-bin data shows the ask stays faithful
at 31. Operator decision 1.

Test stubs: `tests/pipeline_test_helpers.make_real_date_question` already carries a Mantic
`page_url` (`https://competitions.mantic.com/questions/<qid>/`, `:300`), while
`make_real_numeric_question` carries a Metaculus one (`:245`), so a predicate test for a coarse
quantity grid builds its question with the Mantic URL explicitly; a `page_url` of `None` reads as
Metaculus by design (`question_platform.py` module docstring) and keeps every legacy stub on
percentiles.

### 2.3 Bin labels: how the model names a bin

Bins are the platform's right-closed intervals `(edge_k, edge_{k+1}]`, with the first bin also
owning the left edge; `cdf[0]` is the mass below the range and `1 - cdf[N]` the mass above it
(verified by the Phase 2 research against Mantic's own scores on 192 of 192 date questions and
146 of 146 out-of-range resolutions; handoff, "Key decisions"). The edges are
`build_cdf_value_grid(lower, upper, zero_point, cdf_size)` (`numeric/pchip_cdf.py:263-293`),
which reproduces the platform's `continuous_range` exactly (test-pinned in Phase 2).

**Decision.** A new module `metaculus_bot/numeric/pmf_grid.py` owns labelling:

```python
@dataclass(frozen=True)
class PmfGrid:
    labels: tuple[str, ...]        # one per bin, in grid order (len == cdf_size - 1)
    edges: tuple[float, ...]       # cdf_size edges on the question's value axis
    open_lower_bound: bool
    open_upper_bound: bool
    style: Literal["center", "interval", "day", "week", "timestamp"]

    @property
    def keys(self) -> tuple[str, ...]:
        """The block's keys in prompt order: ``below_range`` (open lower) + labels + ``above_range`` (open upper)."""


def pmf_grid(view: NumericQuestion) -> PmfGrid: ...
def format_bin_value(value: float) -> str: ...
def fold_bin_label(label: str) -> str: ...
```

Labelling rules, one style per grid, chosen by `pmf_grid`:

- **`day` / `week`** (`EpochDateQuestion` with `date_granularity` `day` or `week`): label k is
  `format_epoch(edge_k, granularity)` (`numeric/date_axis.py:100-110`, renders `YYYY-MM-DD`), the
  first UTC calendar day the bin covers. On question 651 this yields `2026-09-08` through
  `2026-09-19`, and the last label equals the API's `nominal_max` (verified: `nominal_max`
  1789776000 = 2026-09-19T00:00Z). For `week` the prompt says each key names the first day of the
  seven-day bin.
- **`timestamp`** (`EpochDateQuestion` with granularity `""`, edges at arbitrary times of day):
  label k is `f"{format_epoch(edge_k, '')} to {format_epoch(edge_{k+1}, '')}"`. No such grid has 31
  or fewer bins in the corpus; the rule exists so the module has no unsupported input, not because
  the case is expected.
- **`center`** (linear quantity grid whose displayed LOWER bound is the first bin's centre: Mantic's
  and Metaculus's discrete convention, `nominal_bounds` in `numeric/utils.py:234-259`): label k is
  `format_bin_value(lower + (k + 0.5) * step)` with `step = grid_bin_width(lower, upper, cdf_size)`.
  On an integer-count question the labels are `"0"`, `"1"`, ..., on post 650's shape `"55000"`,
  `"55100"`, ..., `"99900"`. Detection: `zero_point is None` (the resolved one, so a geometric grid
  always falls to `interval`) and
  `abs((nominal_lower - lower) - step / 2) <= 1e-9 * max(1.0, upper - lower)`. **Step 0 deviation
  from the first draft, which also tested the top offset:** the corpus (159 linear Mantic discrete
  grids, measured 2026-09-08) has 158 with both offsets at half a step and one, post 650, whose
  `nominal_max` (100000) sits one step ABOVE its last bin centre (99900; `range_max` 99950), so the
  writer's declared maximum is not a bin centre and, taken literally, resolves `above_range`. All 46
  coarse discrete grids have both offsets at half a step, so the two rules agree on every grid the
  gate admits today; the lower-only rule is kept because a Series 2 count grid created the way 650
  was would otherwise be labelled `"-0.5 to 0.5"`, `"0.5 to 1.5"`, ... instead of `"0"`, `"1"`, ....
  Consequence for the prompt (Agent C): on a `center` grid the last label is NOT guaranteed to equal
  `nominal_max`; render the bin list from `grid.labels`, never from the bound message. Pinned in
  `tests/test_numeric_pmf_grid.py` (`test_post_650_labels_the_450_hundred_dollar_bins_by_centre`,
  `test_a_coarse_count_grid_with_650s_top_still_labels_its_counts`).
- **`interval`** (everything else, including the two 29/30-bin Series 1 numeric questions whose
  `nominal_min == range_min`): label k is `f"{format_bin_value(edge_k)} to {format_bin_value(edge_{k+1})}"`,
  and the prompt states the intervals are right-closed with the first bin including its left edge.

`format_bin_value(x)` is `f"{round(x, 9):.9f}".rstrip("0").rstrip(".")` with `"-0"` mapped to `"0"`:
`55000.0` renders `55000`, `55.1` renders `55.1`, `0.25` renders `0.25`, post 619's step-0.1 edges
(`77.35000000000001`) render `77.35`, and a six-significant-digit `:g` rendering (which would turn
123456.7 into 123457) is avoided.

`fold_bin_label(label)` is `mc_processing.fold_option_label(label).replace(",", "")`; if the
result parses as a float it is re-rendered through `format_bin_value`, so `"7.0"`, `" 7 "` and
`"55,000"` all match the canonical `"7"` and `"55000"`. Dates fold as text (a non-zero-padded
`2026-9-16`, a `Sep 16`, or a full timestamp does not match, fails the block rung and falls
through to the repair and LLM rungs, exactly as an unmatched multiple-choice key does today; the
prompt's "spelled exactly as listed" and the verbatim keys in the parse notes are the mitigation,
and `EXTRACTION_RUNG rung=llm` already monitors the salvage rate).

**Out-of-range mass** is two reserved keys inside the same `bin_probs` object, present only when
the corresponding bound is open: `below_range` and `above_range`. Constants
`PMF_BELOW_RANGE_KEY = "below_range"` and `PMF_ABOVE_RANGE_KEY = "above_range"` live in
`metaculus_bot/constants.py` next to `MC_PROB_MIN` (they are prompt and schema tokens read by
`prompts.py`, `value_extraction.py` and the new module; `constants.py` is the import leaf all
three already read). One object that sums to 1 is what the model sees, and the block rung and the
LLM rung then share a single conversion from label/probability pairs. Collision with a bin label is
impossible: labels are numbers, dates or intervals.

### 2.4 The structured block and the extraction rung

**Schema** (`metaculus_bot/structured_output_schema.py`, modelled on `MultipleChoiceStructured`
at `:446-489`):

```python
class PmfStructured(BaseModel):
    """Per-bin declaration on an enumerable grid: one probability per bin label, plus the
    reserved ``below_range`` / ``above_range`` keys where the question's bound is open."""

    model_config = ConfigDict(extra="forbid")

    question_type: Literal["pmf"]
    bin_probs: dict[str, float]

    @field_validator("bin_probs")
    @classmethod
    def _check_bin_probs(cls, v: dict[str, float]) -> dict[str, float]:
        # non-empty; every key a non-empty string; every value in [0, 1];
        # sum within _PMF_PROB_SUM_TOLERANCE (= 0.02, the multiple-choice ballot's tolerance) of 1.0
```

Registration. The forge fix F32 (on the working tree after e37b8ae, reported by the
`prompts-fixes` agent and verified on disk) deleted the `StructuredQuestionType` alias: the three
parsers (`parse_structured_payload`, `_decode_structured_payload`, `parse_structured_block`),
`_run_ladder` and `_try_candidate` are annotated with `question_types.QuestionType`, and
`tests/test_structured_output_schema.py::TestQuestionTypeVocabulary` pins that
`set(_QUESTION_TYPE_TO_MODEL) == set(get_args(QuestionType))` and that the name
`StructuredQuestionType` is absent ("the restated copy is back"). `"pmf"` is a block type, not a
question type, so `question_types.QuestionType` is NOT widened (it types the `qtype` of
`MEMBER_FORECAST`, `question_type_of`, the gap-fill ghost types and the tool runner). Instead
`structured_output_schema.py` defines the block-type vocabulary as a widening of the question-type
one, never a restated copy:

```python
# The ``question_type`` a STRUCTURED FORECAST block may declare: every question type plus the
# per-bin block, which is an elicitation of a numeric or date question rather than a question type.
BlockType = QuestionType | Literal["pmf"]
```

and re-annotates the three parsers, `_run_ladder` and `_try_candidate` with `BlockType`;
`StructuredBlock` and `_QUESTION_TYPE_TO_MODEL` gain `PmfStructured`. (`DiscreteCountStructured`
is the precedent for a model that exists but is not mapped, and that path is dead at runtime;
`PmfStructured` must be mapped.) The vocabulary pin changes to "the model map's keys are exactly
the question types plus `pmf`" (`set(_QUESTION_TYPE_TO_MODEL) == set(get_args(QuestionType)) |
{"pmf"}`) and "the parsers' `question_type` annotation equals `BlockType`"; the absence pin on
`StructuredQuestionType` stays. The four existing strings are unchanged, so no behaviour moves.

**Extraction** (`metaculus_bot/value_extraction.py`, mirroring `extract_mc` at `:652-699`):

```python
@dataclass
class PmfForecast:
    """A per-bin extraction in the platform's own PMF shape: ``declared`` is
    ``[below, p_0, ..., p_{N-1}, above]`` (length ``N + 2``), as the block or the parser declared
    it and BEFORE the floor blend; a closed bound's tail is 0.0. This is the shape Mantic itself
    exposes for every resolved forecast (``disagreement_forecasts.forecasts[].pmf``)."""

    declared: list[float]


async def extract_pmf(text, grid: PmfGrid, parser_llm, *, prompt_notes="", question_id=None,
                      model_name="") -> ExtractionOutcome[PmfForecast]: ...
```

- Block rung conversion `_make_pmf_from_block(grid)`: fold every block key with `fold_bin_label`
  and map it onto `grid.keys` (labels plus the reserved keys the grid admits); an unmatched key
  fails the rung ("block key 'x' matches no bin of this grid"); a reserved key on a CLOSED bound
  fails the rung; a duplicate fold onto one label is summed (the MC rule); EVERY key of
  `grid.keys` must be present (the MC set-equality rule, so a block truncated before its last bins
  fails and falls through rather than publishing a partial declaration); total must be positive.
  Output: the `N + 2` vector with 0.0 in a closed bound's tail slot.
- Validate `_make_validate_pmf(grid)`: length `N + 2`, every value finite in `[0, 1]`, sum within
  `_PMF_PROB_SUM_TOLERANCE` of 1.0.
- Repair rung: unchanged machinery (`_repair_infidelity_reason` refuses a truncated numeric
  literal or an introduced number).
- LLM rung: `parse_structured(text, list[BinProbability], parser_llm, prompt_notes=...)` with a new
  `BinProbability(BaseModel)` (`label: str`, `probability: float`) and a `BinProbabilityListWrapper`
  registered in `structured_parse._get_wrapper_type` (`:75-88`; without the branch the constrained
  schema falls back to a bare list), then the same pair conversion as the block rung. The parse
  notes (`build_pmf_parse_notes(grid)` in `forecaster_runners.py`, a sibling of
  `build_parse_notes` / `build_date_parse_notes`) list the exact keys verbatim.
- `EXTRACTION_RUNG` logs `qtype=pmf` for this ladder: an additive token value in a `\S+` field, the
  same kind of addition `qtype=date` was on 2026-09-08. The line's `qtype` has always been the block
  type the ladder parsed; on a per-bin date question it reads `pmf` while the `MEMBER_FORECAST`
  line beside it reads `qtype=date`, so the QUESTION type is joined from the member line. That
  sentence goes into the `EXTRACTION_RUNG` docstring bullet in `scripts/telemetry/markers.py`.

### 2.5 PMF to CDF, and how the server rules are met

New module `metaculus_bot/numeric/pmf_cdf.py`:

```python
def build_pmf_distribution(declared: Sequence[float], view: NumericQuestion, *, model_name: str = "") -> NumericDistribution
def published_pmf(prediction: NumericDistribution) -> list[float]   # [cdf[0], diff(cdf)..., 1 - cdf[-1]]
def validate_grid_cdf(cdf: Sequence[float], *, cdf_size: int, open_lower: bool, open_upper: bool) -> None
```

`build_pmf_distribution`, in order:

0. **Refuse mass in a closed tail** (the review's item 1). If the lower bound is closed and
   `declared[0] > 0`, or the upper bound is closed and `declared[-1] > 0`, raise `ValueError`.
   The block and LLM rungs reject reserved keys on closed bounds, so no runner input arrives this
   way, but the builder is a public module boundary and a guard fails SHUT; the first version of
   this plan pinned `cdf[0] = 0.0` after the cumulative sum, which would have silently moved a
   closed tail's mass into the first in-range bin (the review reproduced 0.297 landing in bin 1).
1. **Normalize**: divide the `N + 2` vector by its sum (the ladder already bounded the sum to
   `1 ± 0.02`).
2. **Floor blend**: the server requires every in-range bin step to be at least
   `min_step = round(0.01 / N, 9)` and an open bound's tail to be at least `0.001`
   (`grid_step_constraints` in `numeric/config.py:56-87`; `assert_server_accepts_cdf` in
   `tests/pipeline_test_helpers.py:390-415` is the test-side replica). Let `f` be the floor vector
   (`min_step` per bin, `0.001` per OPEN tail, `0` per closed tail), each floor raised by
   `PMF_FLOOR_MARGIN = 1e-9` so the server's 9-decimal rounding of the PMF can never land a bin one
   unit under its floor. Blend toward the floor distribution `t = f / sum(f)` with the smallest
   `alpha` that lifts every deficient cell to its floor: `alpha = max over cells with p_i < f_i of
   (f_i - p_i) / (t_i - p_i)`, `p' = (1 - alpha) p + alpha t`. This is the same idea as
   `_blend_with_uniform` in `pchip_cdf.py:433-444` (the repo's primary min-step mechanism), applied
   to the cell floors exactly instead of to a uniform mixture, so it moves the least mass possible
   and a bin the model set to 0 ends up at exactly the platform minimum. The alpha always exists
   and is at most `sum(f)` (about 0.012), because `t_i > f_i > p_i` for every deficient cell. A
   certain member keeps 0.988 to 0.992 on its bin (the review reproduced this on 63 grid and input
   combinations, all accepted by the server oracle; this plan's own rerun with the repo's functions
   gives 0.9908 on the 12-bin closed grid).
3. **Assemble the CDF**: `cdf[0] = p'[0]` (below-range mass), `cdf[k] = cdf[k-1] + p'[k]` for
   `k = 1..N`; the last value is `1 - p'[N+1]`. Pin a CLOSED lower bound to exactly `0.0` and a
   closed upper bound to exactly `1.0` (step 0 guarantees this pin moves only float residue, which
   the last bin absorbs).
4. **`safe_cdf_bounds`** (`pchip_cdf.py:138-213`) with `grid_step_constraints(cdf_size)`: the one
   implementation of the open-bound pins, the max-step packing (`_pack_excess_nearest_first`, with
   its `CDF_MAXSTEP_CLIP` marker) and the min-step sweep, which every published CDF path reaches
   (`docs/numeric_pipeline.md` "Step 5"). At 31 bins or fewer the max step is `1.0`
   (`min(1.0, floor9(40 / N))` is 1.0 for every `N <= 40`), so the packing never fires today; it is
   there so raising `PMF_ELICITATION_MAX_BINS` past 40 needs no new code. Re-pin closed bounds
   afterwards (`safe_cdf_bounds` does not pin closed bounds; its callers do).
5. **`validate_grid_cdf`**: length `== cdf_size`, monotone non-decreasing, every 9-decimal-rounded
   step in `[min_step, max_step]`, `cdf[0] == 0.0` or `>= 0.001`, `cdf[-1] == 1.0` or `<= 0.999`,
   no NaN. Raises `RuntimeError` on any violation. A guard fails SHUT (`AGENTS.md`); this is the
   production twin of the test helper and the last thing before publish on a new path.
6. **Wrap** exactly as `_build_discrete_distribution` does (`numeric/pipeline.py:116-164`):
   `zero_point = resolve_zero_point(view)`; `value_grid = build_cdf_value_grid(lower, upper,
   zero_point, cdf_size)`; `declared_percentiles = [Percentile(percentile=cdf_k, value=edge_k)]`;
   `create_pchip_numeric_distribution(pchip_cdf=cdf, percentile_list=declared_percentiles,
   question=view, zero_point=zero_point)`. That constructor (`pchip_processing.py:221-283`)
   stores the CDF, overrides `get_cdf()` to return it on `build_cdf_value_grid`, copies
   `cdf_size` (which `publish_report_to_metaculus` needs, or it rebuilds from percentiles) and
   sets `is_date` from `isinstance(question, EpochDateQuestion)`, which is what makes the comment
   render dates.

What is bypassed, and why each bypass is right:

- `sanitize_percentiles` (`numeric/pipeline.py:51-72`): requires exactly the 13 standard
  percentiles (`filter_to_standard_percentiles`, `validate_percentile_count_and_values`). There
  are none. Nothing it does (jitter, clamp, tail widening, cluster spreading) has a per-bin
  analogue worth building: the floor blend is the whole repair.
- `build_numeric_distribution` and the PCHIP repair tiers: replaced by steps 0 to 5 above.
  `validate_cdf_construction` (`numeric/diagnostics.py:53-60`) goes with them at no loss: it
  returns immediately for any distribution carrying `_pchip_cdf_values`, which
  `create_pchip_numeric_distribution` sets.
- `detect_unit_mismatch` (`numeric/validation.py:80-158`): the guard catches values declared in
  the wrong units. A per-bin declaration states no values, only mass on labelled bins, so the
  failure mode cannot occur; there is nothing to guard. Do not route the grid-shaped
  `declared_percentiles` into it: on grid input all three ratios pass trivially and the guard
  would be fail-open. The residual risk on this path is label MISREADING (a `center` label
  `55000` meaning the interval (54950, 55050]), which the prompt's grid sentence addresses and no
  numeric guard can.
- `_resolve_discrete_vote` (`forecaster_runners.py:357-400`): a PMF block has no `outcome_type`,
  and the fallback is a PAID parser call, so the per-bin runner never calls it; the vote is `None`
  (`forecaster.py:1086` already tolerates it) and the discrete snap is already skipped on every
  outcome-space grid (`discrete_snap.py:230` via `grid_is_outcome_space`).
- `log_open_bound_piling_diagnostics` (`numeric/diagnostics.py:80-142`): takes model-declared
  percentiles; on the per-bin path piling on an open edge is the `above_range` value itself, which
  the `oor_high` field records. Skipped. `log_final_prediction` is kept.

**The flat 0.59 max step is not a server rule.** Mantic's OpenAPI schema text
(`mantic_openapi.yml:798-802`) says "No two adjacent values of the CDF can differ by more than
0.59, which is the largest number obtainable via the sliders"; that sentence describes the slider
user interface, not the validator. Receipt, verified against the recorded corpus this session and
independently by the review: of 4,318 stored competitor forecasts under
`disagreement_forecasts.forecasts[].pmf`, 34 carry a single in-range bin above 0.59 (maximum
0.8952 on post 317, a 15-bin grid; 0.8468 by the `Mantic` account itself on post 113, 13 bins;
0.8412 on the 5-bin post 507), and 0 exceed `0.2 * 200 / N`. The review fetched the upstream
serializer (`questions/serializers/common.py`, `continuous_validation`), which enforces only
`max_diff = 0.2 * DEFAULT_INBOUND_OUTCOME_COUNT / inbound_outcome_count`, rounds the CDF to 10
decimals and the PMF to 9, and has no 0.59 constant. So a certain per-bin forecast (0.99 on one
bin) is legal on every grid this design covers, no ordering constraint on the smokes exists, and
no operator decision is needed. Outside this plan, the "unresolved until the 651 smoke" wording
survives in `scratch_docs_and_planning/handoff-2026-09-08-mantic-phase2.md` (lines 134 and 192)
and in the research reader notes under `mantic_research_2026-09-08/`; the committed and
working-tree `docs/numeric_pipeline.md` and `FUTURE.md` do not carry it (checked with `rg` at
HEAD and in the tree), so the orchestrator's follow-up is the handoff file only.

### 2.6 Aggregating per-bin members: the linear opinion pool

**The problem the review found (F1).** Production aggregates numeric members by the pointwise
MEDIAN of their CDF heights (`aggregate_numeric`, `numeric/utils.py:185-231`, called with
`"median"` from `base_combine` under `CONDITIONAL_STACKING`). Percentile members are smooth, so
the median of their CDFs is a sensible compromise curve. Per-bin members can be sharp (0.99 on one
bin), and the pointwise median of three sharp CDFs that disagree is the MIDDLE member's CDF
outright: the published forecast then carries 0.99 on that member's bin and the platform floor on
the bins the other two believed. That is the cliff a log score punishes hardest, and the
percentile path was only masking it because members were never that sharp.

**Three rules, scored with the repo's own functions.** Computed this session with the plan's
floor blend, `grid_step_constraints`, `safe_cdf_bounds` and, for the open-bound case, the tree's
`floor_out_of_range_tails`, on question 651's 12-bin grid. `S1` is the Series 1 form
`50 * ln(p / baseline)` (baseline `1/12` inside, `0.05` for the out-of-range bucket); `S2` is the
Series 2 categorical form `100 * (log_K(p) + 1)` with `K` the number of outcomes (12, or 13 with
one open bound; how Series 2 references the out-of-range bucket is not published, so the `S2`
out-of-range column is the plan's assumption).

Three members certain of bins 3, 5 and 7, both bounds closed:

| Rule | p on bin 3 | S1 / S2 if bin 3 resolves | p on bin 5 | S1 / S2 if bin 5 resolves |
|---|---|---|---|---|
| pointwise CDF median (today) | 0.0008 | −230 / −185 | 0.9908 | +124 / +100 |
| per-bin median, then renormalize (the MC rule) | 0.0833 | 0 / 0 | 0.0833 | 0 / 0 |
| linear pool (mean of the N+2 PMFs, then the CDF) | 0.3308 | +69 / +56 | 0.3308 | +69 / +56 |

If the three bins are equally likely to be the truth, the expected Series 1 scores are −112
(median), 0 (per-bin median) and +69 (pool). Two members certain of bin 5 and one of bin 3: the
median and the per-bin median both publish 0.9908 on bin 5 (+124 if right, −230 if bin 3), the
pool publishes 0.66 on bin 5 and 0.33 on bin 3 (+104 / +69), so the pool's expected Series 1
score is +92 against +6 for the median. Open upper bound, members certain of bin 3, bin 5 and
`above_range`: before the aggregate tail floor the median gives the out-of-range bucket 0.001
(−196 if it resolves there) and the pool 0.331 (+95); after the Mantic 0.05 tail floor
(`numeric/out_of_range_floor.py`, section 2.9) the median's bucket is lifted to 0.05 (score 0)
while the pool is above the floor and untouched. When all three members agree, the three rules
coincide. The pool never gives a bin less than a third of the mass any member put there, which is
exactly what a log score in the resolved bin rewards; the per-bin median discards the members'
confidence entirely under disagreement; the CDF median discards two members.

**Decision.** Per-bin members are aggregated by the linear opinion pool. It is the pointwise
MEAN of the members' CDFs, because the mean of CDFs at each grid index is the CDF of the mixture
PMF (linearity), and `aggregate_numeric(predictions, question, "mean")` already computes it and
re-applies the grid's step floors and pins in `_postprocess_ensemble_cdf`. No new combiner. This
revives, for per-bin members only, a decision the repo recorded as benchmarked and rejected
(median over mean for percentile members under Metaculus scoring); percentile members keep the
MEDIAN everywhere, on both platforms. Operator decision 3 asks for confirmation because it changes
what Mantic publishes on per-bin questions; the arithmetic is the recommendation.

**Where it plugs in, without touching the percentile path.** `aggregation_pipeline.py`:

- `base_combine` (`:246-345`) computes `base_combine_strategy = MEDIAN if self.strategy ==
  CONDITIONAL_STACKING else MEAN`. It gains one call: `base_combine_strategy =
  self._numeric_combine_strategy(question, base_combine_strategy)`, where the new helper returns
  `AggregationStrategy.MEAN` when `isinstance(question, (NumericQuestion, DateQuestion)) and
  elicit_per_bin(numeric_view(question))` and `strategy` unchanged otherwise. The log line
  `"STACKING base combine: numeric %s aggregation | CDF points=%d"` already prints the effective
  strategy name.
- `simple_combine` (the non-stacking strategies; the `effective_strategy` it logs) and
  `_median_fallback` (the stacker-failure fallback) route through the same helper, so every
  numeric combine of per-bin members is a pool. Both are unreachable in production
  (`CONDITIONAL_STACKING`, stacking off) but the rule must not depend on that.
- `_combine_by_type` and `numeric/utils.aggregate_numeric` are untouched. A question whose
  members are percentile-elicited reaches `_combine_by_type` with the strategy it has today, so
  the percentile path is byte-identical; `aggregation_pipeline` importing `elicit_per_bin` from
  `numeric.config` is allowed by the import contracts (numeric may not import the pipeline; the
  reverse edge already exists for `numeric_view`).
- The stacker's numeric path (`_run_stacking_numeric`) re-elicits percentiles and is off in
  production; when the stacker publishes, its distribution is the whole ensemble and no pooling
  question arises. Section 2.9.

**How `NUMERIC_AGGREGATE` records which rule ran.** The pipeline records the combine rule at the
point of choice, the way it already records `outcomes[qid]` and `skip_reasons[qid]`: a new
`numeric_combine_methods: dict[int, str]` on the pipeline, written `"mean"` or `"median"` in
`_combine_by_type`'s numeric branch (the shared dispatch core of `base_combine`,
`_median_fallback` and `simple_combine`), `"stacked"` where `stack_predictions` adopts the
stacker's distribution, and `"single"` on the lone-survivor return in `base_combine`.
`forecaster._aggregate_predictions` pops it (`self._pipeline.numeric_combine_methods.pop(qid,
"unrecorded")`) and hands it to `format_numeric_aggregate_marker` as a new keyword `method`,
appended as the LAST field: `... tail_floor=0.050000 method=mean`. `"unrecorded"` is a bug signal,
never an expected value; a test asserts every numeric path writes the dict. Registry: one
appended optional group `r"(?:\s+method=(?P<method>\S+))?"` after the tail-floor group in the
`numeric_aggregate` spec (`scripts/telemetry/markers.py:1149-1178` on the tree), so every earlier
line harvests with `method` `None`. The docstring bullet gains one sentence.

**What the end-to-end case asserts** (post 643 in section 6, three members certain of `"1"`,
`"3"` and `above_range`, each declaring 1.0 on its key and 0 everywhere else so the floor blend,
not the model, fills the other cells): the published 22-value CDF is accepted by
`assert_server_accepts_cdf`; bins `1` and `3` and the above-range bucket each carry between 0.30
and 0.36 after the pool and the floor blend; every other bin carries at most
`server_min_step(21) + 2e-9` (the mean of three floors is the floor); the aggregate's
above-range mass is above `MANTIC_OUT_OF_RANGE_TAIL_FLOOR` so `tail_floor=0.000000`; the
`NUMERIC_AGGREGATE` line ends `method=mean`; and the same run's percentile question (post 650,
450 bins) still logs `method=median`.

### 2.7 The forecaster runner

`metaculus_bot/forecaster_runners.py` gains one runner and two branches:

```python
async def _run_pmf_forecast(question, view: NumericQuestion, research, forecaster_llm, parser_llm,
                            *, chart_b64, label: str) -> ReasonedPrediction[NumericDistribution]:
    grid = pmf_grid(view)
    upper_bound_message, lower_bound_message = pmf_bound_messages(view)
    prompt = pmf_prompt(question, view, research, lower_bound_message, upper_bound_message)
    reasoning = await invoke_with_broad_retry(lambda: forecaster_llm.invoke(_forecaster_input(prompt, chart_b64)),
                                              wall_timeout=FORECASTER_SOFT_DEADLINE, label=label)
    _log_llm_output(forecaster_llm.model, question.id_of_question, reasoning)
    outcome = await extract_pmf(reasoning, grid, parser_llm, prompt_notes=build_pmf_parse_notes(grid),
                                question_id=question.id_of_question, model_name=forecaster_llm.model)
    prediction = build_pmf_distribution(outcome.value.declared, view, model_name=forecaster_llm.model)
    logger.info(format_member_forecast_marker(question_id=..., model=forecaster_llm.model,
                role=MEMBER_FORECAST_ROLE_MEMBER, qtype=numeric_qtype(view),
                raw=outcome.value.declared, published=published_pmf(prediction),
                out_of_range=out_of_range_mass(prediction), elicitation=ELICITATION_PMF))
    log_final_prediction(prediction, view)
    return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)
```

`run_numeric_forecast` (`:277-315`): after the docstring, `if elicit_per_bin(question): return
await _run_pmf_forecast(question, question, ..., label="forecaster_numeric"), None`.
`run_date_forecast` (`:318-354`): after `epoch_question = as_epoch_question(question)`,
`if elicit_per_bin(epoch_question): return await _run_pmf_forecast(question, epoch_question, ...,
label="forecaster_date")`. The soft deadline, the broad retry and the drop classification
(`ValueExtractionError` still maps to `parse_extraction` in `drop_telemetry.py`) are unchanged.
`forecaster.py` dispatch is untouched.

### 2.8 The prompt: a per-bin elicitation of the one continuous template

The continuous template (`_continuous_prompt`, `prompts.py:1468-1605`) is shared by the numeric
and date prompts through the `_ContinuousAxis` dataclass (`:1153-1169`), whose eight slots are
"the text that differs between the numeric and date renderings". Per-bin elicitation is a second
axis of variation (percentiles versus bins) crossed with the first (quantity versus date). To
avoid four axis builders or a duplicated template, split the slots by what they vary with:

- `_ContinuousAxis` keeps the QUESTION-KIND slots (`status_quo_question`,
  `reference_class_rules`, `tail_scenarios`, `forecastability_bullet`, `final_check_lead`).
- A new frozen dataclass `_Elicitation` takes the ELICITATION slots: `axis_block`,
  `schema_block`, `outcome_type_step`, `spread_noun` and `spread_short` (the preamble's calibration
  sentence, below), `scoring_rule`, `out_of_range_clause`, `tail_instruction` (step 8, third
  bullet, below), `consistency_line` (final checks). `_continuous_prompt(question, *, view,
  research, lower_bound_message, upper_bound_message, axis, elicitation)` reads both.
  `_percentile_elicitation(question, view)` carries today's text verbatim (moving lines, not
  rewording them) and `_pmf_elicitation(question, view, grid)` the per-bin text below.
  `numeric_prompt` and `date_prompt` keep their signatures and behaviour; `pmf_prompt(question,
  view, research, lower_bound_message, upper_bound_message)` is the new public builder, listed
  in `__all__`.

Two rules that differ between elicitations only by a noun are ONE constant each with a slot, per
the review's proportion finding and the repo's own state-once rule, rather than a percentile
constant and a per-bin twin:

- **`_CALIBRATION_CLAUSE`** (preamble): "Accuracy **and** calibration (especially {spread_noun})
  are critical; how to set {spread_short} is step (8) of the template below." Percentile fill:
  "the width of your prediction interval" / "that width" (today's text, byte for byte); per-bin
  fill: "how far your probability spreads across the bins" / "that spread".
- **`_UNKNOWN_UNKNOWNS_BULLET`** (step 8, third bullet): "{tail_instruction} to cover unknown
  unknowns you can actually name, but not padded out of generic caution." Percentile fill: "Keep
  your extreme tails (P1 and P99) wide enough" (today's sentence, with its dash rendered as the
  template's comma); per-bin fill: "Keep enough probability on the outer bins (and on
  `below_range` / `above_range` where they exist)".

Three shared lines mention percentiles where the meaning is elicitation-neutral; reword them
once, neutrally, on both platforms (a config-era change already covered by the Wave C merge):
step (3) "shift percentiles" becomes "shift your distribution"; step (7) "Small delta check: would
+/- 10 percent on key percentiles still fit the reasoning?" becomes "would shifting your central
estimate by +/- 10 percent still fit the reasoning?", and "your percentiles should stay close to
it" becomes "your distribution should stay close to it"; step (8) second bullet "Match your
interval width" becomes "Match the spread of your distribution", with "a narrow interval" and "a
wide interval" becoming "a narrow distribution" and "a wide distribution". The multi-resolution
constant `_MULTI_RESOLUTION_CONTINUOUS_RULE` ends "report the mixture's percentiles"; change to
"report that mixture" (same rule, neutral). The review confirmed exactly three pins sit on these
phrases (`tests/prompts/test_date_prompt.py:46`, `test_base_prompt_rules.py:828`,
`test_platform_and_mantic_clauses.py:277`); they move with the text.

The per-bin elicitation text, each a named constant with its one-clause reason in a comment:

- **`_PER_BIN_SCORING_RULE`** (scoring-rule slot, replacing `_CONTINUOUS_SCORING_RULE`; reason:
  on a bin-scored grid the model must know that mass on an excluded bin is lost, which is the
  loss this elicitation exists to remove):
  "This question is scored on the bin the outcome falls in: the score is the logarithm of the
  probability you gave that bin (or the `below_range` / `above_range` key when the outcome falls
  beyond an open bound, scored as its own outcome against a reference of a few percent, so
  starving it is heavily punished). This is a proper scoring rule: to maximize expected score,
  report your true probability for every bin. Probability on a bin the resolution criteria
  exclude (a weekend on a trading-day question, a count the rules rule out) is simply lost, so
  give such a bin 0."
- **`_PER_BIN_OUTPUT_RULE`** (axis-block bullet; reason: the parser maps keys onto the grid and
  every key must be present): "Give one probability for EVERY key listed in the schema below,
  spelled exactly as listed and in that order, so that the probabilities sum to 1.0. Use 0 for a
  bin you are certain cannot occur; the platform's per-bin minimum (about {min_step} on this grid)
  is added to every bin for you." The interpolated `min_step` is `round(0.01 / N, 9)` rendered
  `:g`, so the sentence names the per-bin floor and cannot be read as describing the 5% tail floor
  the aggregate carries on Mantic (section 2.9), which the model is not told about because it is
  a property of the published ensemble, not of the member's honest declaration.
- **`_PMF_CONSISTENCY_LINE`** (final checks; reason: the percentile line asks which percentile is
  the status quo): "Consistency line: which bin holds the status quo or trend value, how much
  probability did you give it, and is that sensible?"
- **Axis block** (`_pmf_axis_block(view, grid)`): header "── Bins & Bounds ──"; bullets: the base
  unit (quantity) or "Every key is a UTC calendar date; a key names the whole day it covers (for a
  week grid, the seven days beginning on it)" (date); "Scoring grid: {N} bins of width {step}
  {unit}" or "{N} bins of one calendar {granularity} each"; for the `interval` style, "Each key
  is a right-closed interval: the bin `a to b` contains values above a up to and including b, and
  the first bin also contains its lower edge"; `_PER_BIN_OUTPUT_RULE`. The percentile
  `_scoring_grid_clause` is not rendered here (the bin list is the grid).
- **Schema block**: the `MultipleChoiceStructured` shape with real keys:

  ```json
  {
    "question_type": "pmf",
    "bin_probs": {"below_range": 0.03, "0": 0.03, "1": 0.03, ..., "20": 0.03, "above_range": 0.03}
  }
  ```

  Example probabilities from `_build_example_probs(len(grid.keys))` (`prompts.py:57-73`, the
  even split MC already uses), rendered by a `_bin_probs_example(grid)` sibling of
  `_option_probs_example`. Then: "The `bin_probs` object must contain every key above, spelled
  exactly, and sum to 1.0. The LAST thing you write MUST be this fenced ```json block. Write
  nothing after it." No `outcome_type`, so `outcome_type_step` is `""` and the final step is (9).
- **Bound messages**: `pmf_bound_messages(view) -> tuple[str, str]` in `numeric/utils.py` beside
  `bound_messages` (same `(upper, lower)` return order, same date rendering through
  `format_epoch` on an `EpochDateQuestion`; reason: `bound_messages` is percentile-worded, "your
  percentiles are the ONLY way you express probability mass"). Open upper: "The upper bound is
  open: {upper} is the top of the displayed range, not a hard limit. `above_range` is the
  probability that the outcome resolves above {upper}; it is scored as its own outcome, so give
  it your honest probability, however large." Closed upper: "The upper bound is closed: the
  outcome cannot be higher than {upper}, and there is no `above_range` key." Lower bound
  likewise with `below_range`.
- **Mantic out-of-range base rates, per-bin wording**: `_MANTIC_OUT_OF_RANGE_RATE_DATE_PMF` and
  `_MANTIC_OUT_OF_RANGE_RATE_QUANTITY_PMF`, the same measured facts as the percentile constants
  (`prompts.py:1209-1220`) ending "...so a forecast that gives `above_range` only a token
  probability asserts a near-zero chance of an out-of-range outcome" instead of the percentile
  sentence. Rendered through the same gate as today (`question_platform(question) ==
  PLATFORM_MANTIC`, date only when the upper bound is open, quantity when either is open), from
  the elicitation's `out_of_range_clause` slot. These two are genuinely different rules from their
  percentile twins (they instruct a different action), which is why they are not slot fills.

### 2.9 Telemetry, and the rebase on the aggregate tail floor

The working tree now carries, from the fix wave that ran during the review, a Mantic-only floor on
the PUBLISHED aggregate's open tails: `numeric/out_of_range_floor.py` (`floor_published_tails`,
applied at `forecaster._aggregate_predictions`, the one seam every aggregation path returns
through; `MANTIC_OUT_OF_RANGE_TAIL_FLOOR = 0.05` in `constants.py:84`, operator-approved
2026-09-08), and the `NUMERIC_AGGREGATE` marker gained three fields: `oor_low_raw`,
`oor_high_raw` and `tail_floor`, in one optional trailing group. This design sits on top of it:

- Per-bin MEMBERS are never touched by the floor (it acts on the aggregate only; the module
  docstring says so and the e2e `test_the_members_keep_their_own_tails` pins it). Their
  `MEMBER_FORECAST` tails keep measuring what the models declared.
- Under the linear pool, a member who put mass on `above_range` lifts the aggregate's tail to at
  least a third of that mass, usually past 0.05, so the floor is inert exactly when the members
  already believe in the escape; when every member gives the tail about 0, the floor raises the
  published tail to 0.05, which is the operator-approved behaviour and the same thing it does to
  a percentile aggregate today. No interaction needs code.
- **`MEMBER_FORECAST`** gains ONE additive trailing field, `elicitation=pmf`, appended after the
  `oor_low` / `oor_high` pair by `format_member_forecast_marker` when its new keyword
  `elicitation: str | None = None` is given (`ELICITATION_PMF = "pmf"` in `member_forecast.py`). On
  such a line `raw` and `published` are `[below, p_0, ..., p_{N-1}, above]`, the platform's PMF
  shape with `N + 2` entries: `raw` as the ladder read it (before the floor blend), `published`
  from `published_pmf(prediction)` (so `oor_low == published[0]` and `oor_high == published[-1]`).
  Lines without the field keep today's meaning exactly (percentile pairs on numeric and date,
  the option vector on multiple choice, a float on binary), so a consumer reads `elicitation`
  before interpreting `raw`, and an absent field means "percentiles". This is the doctrine's
  "ADD a field" path (the `oor_*` append on 2026-09-08 is the worked example), and it keeps the
  invariant that this marker carries a member's value on every question. The review verified no
  reader of `member_forecast.jsonl` exists outside the harvester. Registry change
  (`scripts/telemetry/markers.py`, the `member_forecast` spec): append
  `r"(?:\s+elicitation=(?P<elicitation>\S+))?"` as a third regex line; the field coerces to a
  string and reads `None` on every older line. The module docstring bullet and the spec comment
  gain one sentence each. `tests/test_telemetry_markers.py` gets a verbatim example line copied
  from the format string and a back-compat pin (an old line harvests with `elicitation` `None`);
  `tests/test_member_forecast_marker.py` gets a parametrized `pmf` case.
- **`NUMERIC_AGGREGATE`** gains the additive `method` field of section 2.6, after `tail_floor`.
  The tree's e2e assertions on this marker (`tests/test_mantic_e2e.py:800-849`, exact lines with
  the three tail fields) are extended with ` method=median` (post 650) and ` method=mean` (per-bin
  questions).
- **`EXTRACTION_RUNG`**: `qtype=pmf` is an additive token value (section 2.4). Docstring bullet.
- **`FORECASTERS_SURVIVED`**, **`TIME_BUDGET`**, **`MANTIC_QUESTION`**: unchanged. `oor_low` /
  `oor_high` are computed from `get_cdf()` on both member and aggregate lines and need no change.
- No new marker. No renamed or re-spelled field.

### 2.10 What is unchanged downstream, verified

- **Publish**: `publish_report_to_metaculus` reads `get_cdf()` heights into `continuous_cdf`
  (framework `numeric_report.py:684-714`); the publish gate and hardening are type-agnostic.
- **Comment**: forecasting-tools' `make_readable_prediction` (`numeric_report.py:666-682`) renders
  six rows interpolated from `declared_percentiles` at heights 0.1/0.2/0.4/0.6/0.8/0.9 and uses
  `is_date` for the date rendering. A grid-shaped member renders the same six rows as today's
  discrete members do, with values interpolated between bin edges. Rendering the PMF per bin
  would mean overriding a framework classmethod on the report type; the comment is private on
  Mantic and the per-bin declaration survives in the rationale's fenced block, which the trimmer
  pins, and on the `MEMBER_FORECAST` line. Decided: unchanged.
- **Spread metric and stacking gate**: `compute_spread` reads `declared_percentiles`, takes the
  grid branch (`spread_metrics.py:50-99`, `np.interp` over cumulative labels, passes with 5 or more
  points) exactly as for today's discrete members; on Mantic's mostly closed-bound coarse grids the
  denominator is the full range and `spread_undefined` is unreachable. The numeric and date
  stacking gates default off (`stacking_route._STACKING_ENV_BY_QUESTION_TYPE`). The `_MIN_GRID_POINTS`
  guard can fire only on a 3-bin grid (4 CDF points) with an out-of-range label at or above 0.10;
  the corpus has one 3-bin grid (post 253, closed both, cannot trip), and an undefined spread
  routes to MEDIAN with skip reason `spread_undefined` rather than a forfeit. Pre-existing on the
  discrete path, not widened here.
- **Stacker**: `_run_stacking_numeric` re-elicits 13 percentiles and builds through
  `build_numeric_distribution`, which on a coarse grid already takes `_build_discrete_distribution`.
  It stays percentile-based; a FUTURE.md note records that re-enabling numeric stacking on a
  coarse grid would mix a percentile stacker with per-bin members, which the aggregation handles
  but the stacker prompt does not teach. Decided: unchanged.
- **Gap-fill v2's template skeleton** (`research/agentic/driver_prompt.py`) dry-runs the percentile
  prompt; research needs are identical, so it stays. Decided: unchanged, with a FUTURE.md note.
- **Ablation, backtest, artifact persistence**: members with grid-shaped `declared_percentiles`
  already exist (the discrete branch), so `serialize_prediction_value` and the replay dataset
  handle the shape. `ablation_score` is unaffected; the ablation and backtest harnesses replay
  Metaculus questions, which the platform gate keeps on percentiles.
- **Bounds clamp buffer**, **cluster spread cap**, **discrete snap skip**: all live on the
  percentile path and are not reached.

## 3. C2: the supply probe's Mantic mode

### 3.1 What exists

`scripts/supply_probe.py` (740 lines, Metaculus-only) pages `GET /api/posts/` per slug and status
(`{"tournaments": slug, "statuses": status, "limit": 100, "offset": ...}`, stopping on the first
short page), classifies each question as backlog or forfeit, and identifies the bot's forecasts
ONLY through the token's own `my_forecasts` block (`bot_forecast_state`, `:244-264`), fetching a
detail page per forfeit-eligible post because Metaculus list pages do not carry the block. Its
`POSTS_URL` derives from `MetaculusClient().base_url`, its token is `METACULUS_TOKEN`, its
preflight is `verify_metaculus_api_identity()`, and its default slugs are the Metaculus ones.
`make supply_probe` runs it (`Makefile:320-321`). Tests (`tests/test_supply_probe.py`, 880 lines)
are pure dict builders with `_get_json` monkeypatched.

### 3.2 What the Mantic API offers (all verified 2026-09-08 with curl, read-only)

| Fact | Verified how |
|---|---|
| Public list paging works unauthenticated; `next` is populated even on the last page; `count` is `null`; the Series 1 closed+resolved walk returned 100, 100, 100, 100, 100, 20, then 0 (520 posts) | seven GETs of `/api/posts/?tournaments=series-1&statuses=closed&statuses=resolved&limit=100&offset=<k>` |
| Every public post carries `nr_forecasters`, `forecasts_count`, `open_time`, `scheduled_close_time`, `status` | list and detail GETs of posts 500, 645, 650 |
| A RESOLVED post exposes every competitor's spot-time forecast unauthenticated and without `with_cp`: `question.aggregations.recency_weighted.score_data.disagreement_forecasts.forecasts[]`, each `{author_id, author_username, start_time, end_time, values, pmf}` (`values` has `N + 1` entries, `pmf` has `N + 2`) | `GET /api/posts/500/` (8 entries: users 13, 22, 26, 31, 45, 56, 57, 58); corpus check: a non-empty list on 520 of 520 resolved posts |
| On CLOSED-but-unresolved posts that snapshot is rare: 4 of the 32 closed practice posts carry a non-empty `forecasts` list (the first version of this plan said 13, which counted a non-empty `score_data` dict) | corpus scan by `(status, bool(forecasts list))` |
| An OPEN post has `score_data: {}` | `GET /api/posts/650/?with_cp=true` |
| `forecaster_id=<uid>` and `not_forecaster_id=<uid>` on the list are self-only and need a token: unauthenticated answers HTTP 403 `"You do not have permission to perform this action."`; authenticated for our own id 81 they return post 650 and posts 648/649/651 respectively; authenticated for another user's id (13) answers 403 | four GETs on `preseason-2` and `series-1` |
| With a token, list pages carry `my_forecasts` when `with_cp=true` is passed, and omit it otherwise; on Series 1 pages the block is the empty shape `{"history": [], "latest": null, "score_data": {}}` | authenticated list GETs with and without `with_cp=true` |
| The bot is user 81 `nostreambot-bot`, `is_bot: true` | `GET /api/users/81/` |
| The Series 1 project leaderboard is public at `/api/leaderboards/project/3/` (entries carry `user.id`, `score`, `coverage`, `contribution_count`, `rank`); `/api/projects/3/leaderboard/` is 404 | two GETs |
| The recorded fixture `tests/data/mantic_series1_date_post_500_2026_09_08.json` does NOT carry the `disagreement_forecasts` snapshot (its `score_data` has six keys and no `forecasts`); the live post does today | fixture read versus live GET |

Under Series 2's 60-minute windows a question closes within the hour and resolves when the
outcome is known, so most recent posts sit at `closed` for days, where only the token path can
see the bot's forecast. The token path is therefore the PRIMARY instrument and the public
snapshot the fallback that makes Series 1 and every eventually-resolved post measurable without a
secret.

### 3.3 Design

A `--platform {metaculus,mantic}` flag, default `metaculus`, that forks exactly the seams the
Metaculus assumptions live in:

- `POSTS_URL` becomes a function `posts_url(platform)` returning `MetaculusClient().base_url +
  "/posts/"` or `f"{MANTIC_API_BASE_URL}/posts/"`; the URL-shares-the-preflight-host pin
  (`test_probe_url_shares_the_host_the_preflight_vets`, `tests/test_supply_probe.py:856`) is
  extended to both platforms.
- Preflight: `verify_api_identity(MANTIC_API_BASE_URL)` (`api_preflight.py:147`) in mantic mode.
- Default slugs: `(MANTIC_TOURNAMENT_ID,)` in mantic mode.
- Token: `MANTIC_TOKEN_ENV` in mantic mode, primary but OPTIONAL there (the Metaculus mode keeps
  its `parser.error` when `METACULUS_TOKEN` is unset, because Metaculus reads are authenticated).
  With no token the probe runs public-only and reports the rows it cannot resolve as `unknown`.
- `with_cp=true` is added to the list params in mantic mode WHEN a token is present, so
  `my_forecasts` arrives on the list page and the per-post detail GETs (`resolve_bot_forecasts`)
  are skipped entirely. Never send `forecaster_id` (403 unauthenticated; redundant with
  `my_forecasts` when authenticated).
- Identification order for each forfeit-eligible question, in a new
  `mantic_forecast_state(question, *, bot_user_id, has_token) -> str`:
  1. `my_forecasts` present (token) → today's `bot_forecast_state` verdict.
  2. else the public snapshot: `forecasts` under
     `aggregations.recency_weighted.score_data.disagreement_forecasts`; present and non-empty →
     `forecast` if any entry has `author_id == bot_user_id`, else `no_forecast`.
  3. else `unknown`.
  The public snapshot is the platform's own spot-time record, which is exactly what is scored;
  the caveat (post 500 shows 8 snapshot entries against `nr_forecasters` 9, so a forecast
  withdrawn before spot time reads as `no_forecast`) is documented in the report header, not
  modelled.
- `MANTIC_BOT_USER_ID: int = 81` in `constants.py` next to `MANTIC_TOKEN_ENV`, with the
  verification receipt in its comment. `nostreambot-bot` already appears in the docs and the
  workflow as prose; the id is what the snapshot carries.
- **The new instrument: miss rate per release hour.** A table over the forfeit-eligible questions
  (closed and resolved posts, deduplicated per question), grouped by the UTC hour of day of the
  question's `open_time`: columns `hour_utc`, `questions`, `forecast`, `no_forecast`, `unknown`,
  `miss_rate` (`no_forecast / (forecast + no_forecast)`, blank when the denominator is 0), plus a
  `total` row and one line with the window length distribution (`scheduled_close_time -
  open_time` in minutes: min, median, max) so the 60-minute assumption is checked by the same
  run. Rendered after the existing forfeit block; in the JSON output as `by_release_hour` and
  `window_minutes`, with `platform` and `bot_user_id` at the top level.
- Paging: unchanged (`PAGE_SIZE` 100, stop on a short page, `MAX_PAGES` 40); the seven-page walk
  above shows it terminates on Mantic. `count` is never read.
- Everything else (backlog, status partition, group posts, rendering) is reused as is.

### 3.4 Tests

Pure dict builders, as today, plus one recorded fixture:

- Re-record post 500 with `curl -s https://competitions.mantic.com/api/posts/500/ >
  tests/data/mantic_series1_resolved_post_500_public_2026_09_09.json` (free, unauthenticated) and
  assert it carries `score_data.disagreement_forecasts.forecasts` with 8 entries before writing
  the test; the existing 2026-09-08 fixture predates the snapshot and must not be edited.
- `mantic_forecast_state`: token block wins; public snapshot with and without author 81;
  no snapshot and no token → `unknown`; a `forecasts: []` snapshot → `unknown` (an empty list is
  the pre-scoring state, not evidence of absence).
- `--platform mantic` main wiring: `POSTS_URL` is the Mantic host; `verify_api_identity` is
  called with `MANTIC_API_BASE_URL` and the Metaculus preflight is not; `MANTIC_TOKEN` absent does
  not `parser.error`; `with_cp=true` is in the list params only with a token; `forecaster_id` is
  never sent; default slugs are `(MANTIC_TOURNAMENT_ID,)`; `resolve_bot_forecasts` is not called
  in mantic mode.
- Per-hour table: three questions opening at 14:00, 14:00 and 15:00 UTC with states
  forecast / no_forecast / unknown render `14 | 2 | 1 | 1 | 0 | 50.0%` and `15 | 1 | 0 | 0 | 1 | -`;
  the window line reads 60/60/60 on 60-minute fixtures; JSON carries the same numbers.
- Paging on Mantic's always-populated `next`: a fake serving 100, 20, 0 stops after the second page.
- Metaculus mode is byte-identical: the existing `TestMain` and `TestDefaults` classes pass
  unchanged.

### 3.5 Wiring and docs

`Makefile`: `supply_probe_mantic:` → `uv run python scripts/supply_probe.py --platform mantic
$(ARGS)`, in `.PHONY`, under the same "read-only and free" comment. `docs/operations.md`
"Scheduling reliability" gets the command and what the per-hour table answers, with the token path
described as primary; `AGENTS.md`'s free list gains `make supply_probe_mantic`. After the first
Series 2 week, the read-only command that settles the cadence decision is:

```
make supply_probe_mantic ARGS="--slugs <series-2-slug> --output scratch/mantic_supply_$(date -u +%Y%m%d).json"
```

## 4. C3 to C6

**C3, fast-path alertability.** `time_budget_fast_path` is a summand of `alertable_total`
(`degradation_counters.py:52-78`) and reddens the run through `cli._report_degradation_and_exit`
(`cli.py:576`, `alertable = bot_alertable + generic_fallback - suppressed_credit_fallback +
mantic_post_drops`). Under 60-minute windows any pickup after about :28:45 is fast path
(`TIME_BUDGET_FAST_PATH_THRESHOLD` 1815 s plus `PUBLISH_RESERVE_SECONDS` 60 s), so roughly a fifth
of Mantic runs would exit non-zero for a designed outcome (edge review rank 13). Blocked on the
Series 2 window length, which no API field announces before the first question opens. Decided:
nothing changes now; if windows are 60 minutes, the fast-path count is subtracted from
`bot_alertable` in mantic mode only, keeping the counter, the WARNING and the `TIME_BUDGET` marker,
with a pin that Metaculus modes are unchanged. Read-only command once Series 2 has run a day:
`make sync_all` then

```
uv run python -c "import json; q={json.loads(l)['qid'] for l in open('backtests/telemetry_archive/mantic_question.jsonl')}; r=[json.loads(l) for l in open('backtests/telemetry_archive/time_budget.jsonl')]; m=[x for x in r if x['qid'] in q]; print(len(m), 'mantic budgets;', sum(1 for x in m if x['fast_path']), 'fast path;', sorted(x['budget_s'] for x in m)[:5])"
```

(the `budget_s` values reveal the window length directly, since every question is picked up
inside its own window).

**C4, median versus mean under the baseline formula, reframed.** The review is right that this
item cannot answer the question C1 raises: archived members are percentile-elicited, and the
sharp-member cliff of section 2.6 is decided there by construction. What C4 can still measure is
whether the pointwise MEAN also beats the MEDIAN for PERCENTILE members under a bin log score,
which is what Mantic scores the bot's percentile-elicited questions (every grid above 31 bins) by.
Nothing is blocked: `_score_member_curve` (`performance_analysis/audit.py:399-453`) already
rebuilds each member's CDF on the record's own grid from the comment-recovered percentiles
(`per_model_ranking_cohort`) and scores it with `scoring_common.numeric_log_score`, the `50 *
ln(p / baseline)` form Mantic used in Series 1; `_postprocess_ensemble_cdf` takes `"mean"` or
`"median"`. Prepare now (free, read-only): a small analysis module
`metaculus_bot/performance_analysis/aggregation_replay.py` that, per resolved numeric or discrete
record, rebuilds the members, aggregates both ways with `aggregate_numeric`, scores both under the
Series 1 form AND the Series 2 form `100 * (log_N(p) + 1)` (a second scorer beside
`numeric_log_score`, because the two weight questions differently even though they rank the two
aggregates identically within a question), era-buckets on the merge dates, and prints the mean
delta with a bootstrap interval. Command, after `make sync_all` and the dataset build
`uv run python -m metaculus_bot.performance_analysis --tournament <slug> --output <path>`:
`uv run python -m metaculus_bot.performance_analysis.aggregation_replay --dataset <path>`. The
Mantic-specific re-run waits for about thirty resolved Mantic percentile questions with
`MEMBER_FORECAST` lines. This revisits a decision the repo recorded as benchmarked and rejected
(median versus mean), under a different scoring rule and question mix; it is a re-measurement, and
a flip would extend the section 2.6 helper to percentile members on Mantic, one branch, not a new
mechanism.

**C5, after 2026-09-20 12:00 UTC.** Question 651 closes, resolves and is spot-scored at that
instant (`scheduled_close_time`, `scheduled_resolve_time`, `cp_reveal_time` and
`spot_scoring_time` are all `2026-09-20T12:00:00Z`). Two read-only reads, both free. Public:
`curl -s https://competitions.mantic.com/api/posts/651/ | jq '.question.resolution,
.question.aggregations.recency_weighted.score_data.disagreement_forecasts.forecasts[] |
select(.author_id == 81) | .pmf'` gives the resolution string and our own spot-time PMF, and
`metaculus_bot.scoring_common.resolution_to_bucket_index` applied to the resolution against the
fixture's `continuous_range` says which bin the platform charged, closing the on-edge question
(a resolution stored at midnight UTC of day D must land in bin `D - 2026-09-08`, not the bin
before it). Authenticated: `curl -s -H "Authorization: Token $MANTIC_TOKEN"
"https://competitions.mantic.com/api/posts/651/?with_cp=true" | jq
'.question.my_forecasts.score_data'` reads our reported spot baseline score; comparing it with
`50 * ln(p / (1/12))` and `100 * (log_12(p) + 1)` on our published bin mass tells which
coefficient Mantic actually applies. Repeat the second read on the first resolved Series 2
question. If the reported score matches neither form, that is the C5 finding and the
`_PER_BIN_SCORING_RULE` wording ("the logarithm of the probability") still holds.

**C6, the starved outer tail.** Documented at `docs/performance_analysis.md` (about -219 points
wherever the resolution lands inside the range but outside the members' declared interval; 68 of
417 measurable open-bound sides on the Metaculus archive). Per-bin elicitation removes the
mechanism on every grid it covers, because the model states the outer bins' probability directly,
the floor blend keeps every bin at least at the platform minimum, and the linear pool keeps every
bin any member believed; on percentile grids the cliff stays as it is, deliberately unbundled from
the min-step tolerance fix (edge review rank 19). Blocked on a Mantic residual dataset, which
Phase 2 left out of scope (`performance_analysis` is date-free and Metaculus-only). Prepare
nothing. Read-only command when data exists: the per-round residual procedure's
`scan_outer_tails` (`metaculus_bot/performance_analysis/outer_tail.py`) over the Mantic records,
plus, until then, `oor_high` on the `MEMBER_FORECAST` lines of open-upper Mantic questions
(`jq 'select(.oor_high != null)' backtests/telemetry_archive/member_forecast.jsonl`).

## 5. Execution split

Step 0 is sequential and small; A to F run in parallel on disjoint files. Every agent works TDD
(`test-driven-development` skill: failing test first), commits nothing (the orchestrator gates
and commits), runs `make test_fast` and its own test files while working and `make all` before
reporting, and reads `AGENTS.md` first. Model: `opus`. No agent runs anything paid. **The fan-out
starts only after the forge fix wave has landed and the tree is quiet**: `member_forecast.py`,
`constants.py`, `prompts.py`, `value_extraction.py` and `numeric/pipeline.py` were all being
edited during the review, and two agents editing one file is how a fix gets lost.

**Step 0 (orchestrator or one agent, about 20 minutes): the grid module.**
Files: `metaculus_bot/numeric/config.py` (`PMF_ELICITATION_MAX_BINS`, `PMF_ELICITATION_PLATFORMS`,
`elicit_per_bin`), `metaculus_bot/constants.py` (`PMF_BELOW_RANGE_KEY`, `PMF_ABOVE_RANGE_KEY`,
`MANTIC_BOT_USER_ID`), NEW `metaculus_bot/numeric/pmf_grid.py`, NEW `tests/test_numeric_pmf_grid.py`,
predicate pins in `tests/test_numeric_config.py`. Everything downstream imports `PmfGrid`,
`pmf_grid`, `fold_bin_label`, `format_bin_value` and `elicit_per_bin` from here, so it lands first
and its interface (sections 2.2 and 2.3) is frozen for A to F.

**Agent A, "pmf-cdf" (numeric build).**
Owns: NEW `metaculus_bot/numeric/pmf_cdf.py`; `metaculus_bot/numeric/utils.py`
(`pmf_bound_messages`, added to `__all__`); NEW `tests/test_numeric_pmf_cdf.py`;
`tests/test_numeric_pipeline.py` if a seam pin is needed; `docs/numeric_pipeline.md`.
TDD notes: start from the server-rule oracle `assert_server_accepts_cdf` and the five grid sizes
(3, 4, 12, 21, 31) × (closed/closed, closed/open, open/open); then the floor-blend arithmetic
(zeros land exactly at `min_step + 1e-9`, a certain bin keeps `>= 0.988`); then the closed-tail
refusal (step 0 raises on `[0.3, ...]` with a closed lower bound; the review's reproduction is the
test); then the wrap (`is_date`, `cdf_size`, grid-shaped `declared_percentiles`, `published_pmf`
round trip); then the `validate_grid_cdf` raise paths (a length-off CDF, a step under the floor, a
closed bound not pinned). Do not touch `pchip_cdf.py`, `pipeline.py` or `pchip_processing.py`.

**Agent B, "pmf-block" (schema, extraction, parser wrapper).**
Owns: `metaculus_bot/structured_output_schema.py`, `metaculus_bot/value_extraction.py`,
`metaculus_bot/structured_parse.py`, NEW `tests/test_pmf_extraction.py`, additions to the
existing `tests/test_structured_output_schema.py` and `tests/test_structured_parse.py`,
`docs/value_extraction.md`.
TDD notes: schema first (`PmfStructured` accepts the example, rejects a 1.05 sum, rejects an
unknown top-level key; introduce `BlockType = QuestionType | Literal["pmf"]`, re-annotate the three
parsers, `_run_ladder` and `_try_candidate` with it, and move `TestQuestionTypeVocabulary` to
"question types plus `pmf`" while keeping its absence pin on `StructuredQuestionType`); then the
block rung against a `PmfGrid` built directly from the frozen dataclass (folded keys `"7.0"`,
`" 2026-09-16 "`, `"55,000"` match; a missing bin fails; an unknown key fails; `below_range` on a
closed bound fails; duplicates sum); then the repair rung refuses a block cut mid-number; then the
LLM rung's wrapper unwraps `list[BinProbability]` and runs the same conversion. Pin that
`_run_ladder` logs `EXTRACTION_RUNG ... qtype=pmf rung=block`.

**Agent C, "pmf-prompt" (the elicitation split and the per-bin text).**
Owns: `metaculus_bot/prompts.py`, `tests/prompts/*` (NEW `test_pmf_prompt.py`; pin moves in
`test_base_prompt_rules.py`, `test_platform_and_mantic_clauses.py`, `test_structured_block.py`,
`test_date_prompt.py`), `tests/prompt_builders.py` (a `_pmf_prompt_text` builder and a 21-bin
open-upper discrete stub with a Mantic `page_url`), `docs/prompts.md`.
TDD notes: first the refactor with zero text change (every existing prompt test green with
`_Elicitation` introduced; add a byte-equality pin of `numeric_prompt`/`date_prompt` output
before and after on the fixture stubs, then delete it once green); then the two slot templates
(`_CALIBRATION_CLAUSE`, `_UNKNOWN_UNKNOWNS_BULLET`) with the percentile fills reproducing today's
text byte for byte; then the three neutral rewordings with their pins moved; then `pmf_prompt`
with presence pins on every new constant (`_flat(CONSTANT) in _flat(prompt)` plus two distinctive
sub-phrases each and a relative `prompt.index` position where order matters), the example-block
parse pin (`parse_structured_payload(body, "pmf")` returns a block whose keys equal `grid.keys`),
the absence pins (`"percentile" not in _flat(pmf_prompt_text)`, no `outcome_type`, no PMF constant
in the three stacking prompts or in the percentile numeric and date prompts), and the Mantic gate
pins for the two `_PMF` base-rate constants (rendered on a Mantic open-bound stub, absent on the
Metaculus stub and on closed bounds). Add `pmf_prompt` to `_EXAMPLE_BLOCK_BUILDERS`
(`test_structured_block.py:268`) and to `test_example_block_parses`' accepted set.

**Agent D, "pmf-runner" (runners, end to end, ops docs). The long pole.**
Owns: `metaculus_bot/forecaster_runners.py`, `tests/test_forecaster_runners.py`,
`tests/test_mantic_e2e.py`, `tests/mantic_fakes.py`, NEW
`tests/data/mantic_series1_discrete_post_643_2026_09_08.json` (from the corpus dump, reshaped:
status `open`, `resolved` false, empty `my_forecasts`, `projects.tournament` = preseason-2; the
e2e re-dates it), `tests/test_date_question_pipeline.py` (the date question now takes the per-bin
path), `docs/operations.md` ("Mantic" section and "Mantic-optimized forecasting", NOT the
"Scheduling reliability" subsection), `docs/architecture.md`, `AGENTS.md` (layout table row),
`FUTURE.md`.
TDD notes: the runner branch first with a stubbed `extract_pmf` and `build_pmf_distribution`
(the branch is taken on a 12-bin Mantic date and an 11-bin Mantic discrete question, NOT on a
201-point numeric question, a 200-bin discrete one, or a Metaculus 11-bin discrete one;
`_resolve_discrete_vote` and `detect_unit_mismatch` are not called; the `MEMBER_FORECAST` line
carries `elicitation=pmf` and a 14-entry vector, using the `format_member_forecast_marker`
keyword Agent F adds); then the e2e once A, B, C and F have landed (until then the e2e tests may
be written and marked `xfail(strict=True)` with the reason "awaits pmf-cdf/pmf-block/pmf-prompt/
pmf-aggregate"; remove the marks in the same session). D must NOT edit `prompts.py`,
`value_extraction.py`, `member_forecast.py`, `aggregation_pipeline.py`, `forecaster.py`,
`scripts/telemetry/markers.py` or anything under `numeric/`.

**Agent F, "pmf-aggregate" (the pool, the telemetry fields, the marker registry).**
Owns: `metaculus_bot/aggregation_pipeline.py` (`_numeric_combine_strategy`, the
`numeric_combine_methods` dict and its three write sites), `metaculus_bot/forecaster.py` (the
`_aggregate_predictions` seam: pop the method, pass it to the marker), `metaculus_bot/member_forecast.py`
(`elicitation` keyword and `ELICITATION_PMF`; `method` keyword on
`format_numeric_aggregate_marker`), `scripts/telemetry/markers.py` (both regex appends, the
docstring bullets for `MEMBER_FORECAST`, `NUMERIC_AGGREGATE` and `EXTRACTION_RUNG`),
`tests/test_aggregation_pipeline.py`, `tests/test_aggregation.py`, `tests/test_telemetry_markers.py`,
`tests/test_member_forecast_marker.py`, `tests/test_template_forecaster.py` if the seam has pins
there.
TDD notes: the marker formatters and registry first (verbatim example lines with `elicitation=pmf`
and `method=mean`; back-compat pins that lines without either field harvest with `None`); then
`_numeric_combine_strategy` (MEAN for a per-bin Mantic question, `strategy` unchanged for a
percentile question on either platform and for binary and MC); then the recording dict (every
numeric path writes it: base-combine median, base-combine mean, single survivor, stacked; the
seam pops it and the marker ends `method=<token>`); then the pool arithmetic pin from section 2.6
(three sharp members on a 12-bin grid through `aggregate_numeric(..., "mean")` give 0.33 each and
pass the server oracle, and through `"median"` give 0.99 on one bin, which is the regression this
rule prevents). F must NOT edit `forecaster_runners.py` or anything under `numeric/`.

**Agent E, "supply-probe-mantic" (C2), independent of C1.**
Owns: `scripts/supply_probe.py`, `tests/test_supply_probe.py`, NEW
`tests/data/mantic_series1_resolved_post_500_public_2026_09_09.json` (recorded with the curl in
section 3.4), `Makefile` (`supply_probe_mantic`), the "Scheduling reliability" subsection of
`docs/operations.md` (E appends one paragraph at the end of that subsection and touches nothing
else in the file), `AGENTS.md` free-list line for `make supply_probe_mantic` (E owns this one
line; D owns the layout table row).
TDD notes: `mantic_forecast_state` first, then the per-hour table as a pure function over
`QuestionRow`s, then the CLI wiring with `requests.get` monkeypatched, then the recorded-fixture
test. `MANTIC_BOT_USER_ID` comes from Step 0.

**Orchestrator after the fan-out**: `make all`; `/forge` over `main...HEAD`; fix findings by the
same ownership; re-gate; commit. Then the paid steps are the operator's: after Wave C lands, ONE
per-bin smoke on 651 (`make run_mantic_one POST=651`, about $3, publishes), verified with the
authenticated `with_cp=true` read (13-value CDF accepted, `cdf[0] == 0.0`, `cdf[12] == 1.0`,
weekend bins 12/13/19 September at the platform minimum, `MEMBER_FORECAST ... elicitation=pmf`,
`NUMERIC_AGGREGATE ... method=mean`). The Phase 2 date smoke on 651 that the handoff lists is no
longer a prerequisite for anything in this plan (section 2.5); whether to run it at all is the
operator's call on its own merits.

## 6. Test plan (what green means)

Unit:

- `tests/test_numeric_pmf_grid.py`: `elicit_per_bin` on a 201-point Mantic `NumericQuestion`
  (False), a 200-bin Mantic `DiscreteQuestion` (False), the recorded 12-bin date post 651 through
  `as_epoch_question` (True), an 11-bin Mantic-shaped `DiscreteQuestion` (True), a 30-bin Mantic
  non-discrete `NumericQuestion` with `nominal_min == range_min` (True; the review's missing pin),
  31 bins (True), 32 bins (False), and an 11-bin Metaculus `DiscreteQuestion` (False under the
  recorded default; the same stub with `PLATFORM_METACULUS` added to `PMF_ELICITATION_PLATFORMS`
  reads True, which pins that the switch is one constant). Labels: 651 gives `2026-09-08` ...
  `2026-09-19` and the last label equals `format_epoch(nominal_max)`; post 253's shape (3 bins,
  counts 0 to 2) gives `"0","1","2"`; post 650's shape gives `"55000"`, `"55100"`, ... (450 bins,
  labels still well-defined); post 619's step-0.1 shape gives `77.3`, `77.4`, ...; a
  week-granularity synthetic gives seven-day-spaced dates; post 560's shape (30 bins, `nominal_min
  == range_min`) gives `interval` labels; `grid.keys` carries `below_range` first and `above_range`
  last exactly when the bound is open; `fold_bin_label` equivalences.
- `tests/test_numeric_pmf_cdf.py`: section 5, Agent A. Include the oracle: a member certain of
  2026-09-16 on 651 puts `>= 0.98` in bin 8 (Phase 2's PCHIP pin was `> 0.9`), and the weekend
  bins carry exactly the floor. Aggregation: the pointwise MEAN of three per-bin members passes
  `assert_server_accepts_cdf` on every grid size tested.
- `tests/test_pmf_extraction.py`: section 5, Agent B.
- `tests/prompts/test_pmf_prompt.py` and the pin moves: section 5, Agent C.
- `tests/test_forecaster_runners.py`: section 5, Agent D.
- `tests/test_aggregation_pipeline.py`, `tests/test_aggregation.py`, `tests/test_telemetry_markers.py`,
  `tests/test_member_forecast_marker.py`: section 5, Agent F.
- `tests/test_supply_probe.py`: section 3.4.

End to end (`tests/test_mantic_e2e.py`, mocked HTTP and mocked LLMs as today):

- Question 651 (12-bin date, closed both): canned per-bin blocks for three members who AGREE (for
  example 0.90/0.92/0.88 on `2026-09-16`, the remainder spread over the other eight trading days,
  0 on the three weekend days). Asserts: the forecast POST is a 13-value CDF that
  `assert_server_accepts_cdf` accepts; `cdf[0] == 0.0`, `cdf[12] == 1.0`; bin 8 mass `> 0.85`
  after the pool; bins 4, 5 and 11 (12, 13 and 19 September) each carry at most
  `server_min_step(12) + 2e-9`; three `MEMBER_FORECAST` lines with `qtype=date`,
  `elicitation=pmf`, 14-entry `raw` and `published` vectors, `oor_low=0.000000
  oor_high=0.000000`; the `NUMERIC_AGGREGATE` line is the tree's exact line plus ` method=mean`;
  `EXTRACTION_RUNG ... qtype=pmf rung=block block_present=true`; no parser or stacker LLM call
  (the existing `test_no_parser_or_stacker_call_was_needed` keeps its teeth); the comment renders
  `2026-09-16` and no epoch second.
- Post 643 (21-bin discrete count, closed lower, OPEN upper, "How many public releases will
  U.S. Central Command publish ..."): canned blocks for three members who DISAGREE sharply,
  certain of `"1"`, `"3"` and `above_range` respectively (each 1.0 on its key and 0 on every other
  key, so the floor blend alone fills the other cells). Asserts: section 2.6's list (a 22-value
  CDF the server accepts with
  `cdf[21] <= 0.999`; the three believed cells each between 0.30 and 0.36; every other bin at
  most `server_min_step(21) + 2e-9`; `oor_high` above `MANTIC_OUT_OF_RANGE_TAIL_FLOOR` and
  `tail_floor=0.000000`; `method=mean`); `MANTIC_QUESTION` line with `cdf_size=22`; comment
  targets its own post. Update `_FORECAST_QUESTION_COUNT` (4 to 5), `_STILL_FRESH_POST_IDS`,
  `_canned_responses`, the parametrize list at `test_each_comment_targets_its_own_post`, and
  confirm `_MANTIC_FILTER_TYPE` already maps `discrete` (it does, `:290-296`).
- The bitcoin question 650 (450 bins) and the legacy 200-bin date post 500 stay on percentiles:
  their existing assertions are unchanged except that 650's `NUMERIC_AGGREGATE` line gains
  ` method=median`, which pins that the pool did not widen to percentile members.

Gates: `make all` green (format, lint, deptry, import-linter, basedpyright at 0 errors, full
pytest). The import edges Wave C adds were checked against the `[tool.importlinter]` contracts in
`pyproject.toml` on 2026-09-08: `structured_output_schema.py`, `value_extraction.py` and
`structured_parse.py` already import from `metaculus_bot.numeric.date_axis`, so importing
`numeric.pmf_grid` from them is the same edge; `numeric/pmf_grid.py` importing
`mc_processing.fold_option_label` is the edge `numeric/utils.py` already has
(`clamp_and_renormalize_probs`), and `mc_processing.py` itself imports only `constants` and
`simple_types`; `numeric/config.py` importing `question_platform` is a leaf edge (that module
imports only `constants` and `urllib`), and `numeric/out_of_range_floor.py` already does it;
`aggregation_pipeline.py` importing `numeric.config` is the direction the contracts allow.
Nothing new reaches a forbidden module. Run `make lint_imports` anyway.

## 7. Docs to update

- `docs/numeric_pipeline.md` (Agent A): a new section "Per-bin elicitation on enumerable grids"
  between Step 4 and Step 5: the gate (Mantic-only, the one-line switch), the labels, the floor
  blend, the closed-tail refusal, the reuse of `safe_cdf_bounds`, `validate_grid_cdf`, what is
  bypassed and why (the list in section 2.5); in "Step 9" a paragraph that per-bin members are
  pooled by the pointwise MEAN with the section 2.6 arithmetic and the `method` field, while
  percentile members keep the MEDIAN. In "Server-side constraints", replace any mention of the
  0.59 sentence with the receipt: it is slider text, 34 of 4,318 stored competitor forecasts exceed
  it, none exceed `0.2 * 200 / N`, and the upstream validator has no such constant.
- `docs/prompts.md` (Agent C): under "Platform-aware and Mantic clauses", a new subsection "Per-bin
  elicitation (Wave C)" listing every constant with its reason and its pin file, the two slot
  templates and their fills, the `_Elicitation` split, the three neutral rewordings, and the rule
  that no PMF constant appears in the stacking prompts; "Test pins" names `test_pmf_prompt.py`.
- `docs/value_extraction.md` (Agent B): the `pmf` ladder, `PmfForecast`'s `N + 2` shape, the
  fold-matching rule, the "every key present" rule and why (truncation), `qtype=pmf` on
  `EXTRACTION_RUNG` with the join-from-`MEMBER_FORECAST` caveat, and the `elicitation=pmf` field
  on `MEMBER_FORECAST` with the reading rule (absent means percentiles).
- `docs/operations.md` (Agent D, and Agent E for one paragraph): "Mantic-optimized forecasting"
  gains a bullet for per-bin elicitation (motivation, threshold, the pool, the smoke verification
  list); "Scheduling reliability" gains the `make supply_probe_mantic` paragraph with the token
  path as primary (Agent E); the season-start checklist mentions the probe's Mantic mode beside
  the Metaculus one.
- `docs/architecture.md` (Agent D): one sentence in "3. Forecaster fan-out" that a coarse-grid
  Mantic numeric or date question is elicited per bin (`elicit_per_bin`) and pooled by the mean,
  with the pointer to `docs/numeric_pipeline.md`.
- `AGENTS.md` (Agent D for the table, Agent E for the free-list line): layout table row "Per-bin
  PMF elicitation on enumerable grids | `numeric/pmf_grid.py`, `numeric/pmf_cdf.py`"; the free and
  safe list gains `make supply_probe_mantic`.
- `FUTURE.md` (Agent D): in the Mantic section, mark C1 and C2 shipped with the date, record the
  decisions (threshold, Mantic-only gate with the one-line Metaculus switch and its 141-of-300
  blast radius, the linear pool for per-bin members with the section 2.6 numbers, the
  `elicitation` and `method` fields, the comment left as is, the stacker and gap-fill skeleton
  left on percentiles), and keep C3 to C6 as written with the read-only commands from section 4.
- `scripts/telemetry/markers.py` docstring (Agent F): the `MEMBER_FORECAST`, `NUMERIC_AGGREGATE`
  and `EXTRACTION_RUNG` bullets.
- Outside this plan, the orchestrator's follow-up after the fix wave: the "0.59 unresolved"
  sentences in `scratch_docs_and_planning/handoff-2026-09-08-mantic-phase2.md` (lines 134 and
  192). `docs/numeric_pipeline.md` and `FUTURE.md` do not carry the sentence at HEAD or in the
  tree.

## 8. Operator decisions needed

Only the decisions that remain genuinely open. Each item stands alone; the recommendation is
first. Everything else the first version listed (comment rendering, the `MEMBER_FORECAST` field,
the probe's token handling, C3's deferral, the stacker and gap-fill skeleton staying on
percentiles) is decided in the plan and no longer needs a call.

1. **Per-bin threshold: 31 bins.** Recommend `PMF_ELICITATION_MAX_BINS = 31` (a month of daily bins;
   29% of Series 1 discrete questions at or below it; the operator's "about 31"). Alternative: 51,
   which also takes the nine 51-bin Series 1 discrete questions; raise only after live per-bin
   data shows the ask stays faithful at 31. Note for later: 31 is also a Metaculus modal grid size
   (22 of 300 sampled discrete questions), which matters only if decision 2's switch is flipped.
2. **Platform scope: Mantic-only for this landing, Metaculus as a one-line follow-up.** Recommend
   `PMF_ELICITATION_PLATFORMS = frozenset({PLATFORM_MANTIC})` now. The platform-agnostic
   alternative would change elicitation on about half of all Metaculus discrete questions (141 of
   300 sampled have 31 bins or fewer; modal grids 41, 31, 101 and 11 bins) with no live evidence
   yet that the per-bin ask is faithful, and it would have to land before the fall FutureEval's
   first question on 2026-09-28. Flipping later is adding `PLATFORM_METACULUS` to that one
   constant, as its own config-era change once a Mantic season has shown the declaration holds.
   Either answer is that one line.
3. **Aggregation of per-bin members: the linear opinion pool (pointwise MEAN of the members'
   CDFs), percentile members unchanged on MEDIAN.** Recommend the pool. On question 651's 12-bin
   grid with three members certain of bins 3, 5 and 7, today's pointwise median publishes 0.99 on
   bin 5 and the platform floor on bins 3 and 7 (about −230 Series 1 / −185 Series 2 points if
   bin 3 resolves; expected −112 over the three bins), the multiple-choice-style per-bin median
   goes uniform (0), and the pool gives each believed bin 0.33 (+69 / +56; expected +69). With two
   members on bin 5 and one on bin 3 the pool's expected score is +92 against +6 for the median.
   This revives, for per-bin members only, a decision the repo recorded as benchmarked and
   rejected for percentile members; it is one branch in `aggregation_pipeline.py` that leaves the
   percentile path byte-identical, and the `NUMERIC_AGGREGATE` marker records `method=mean` or
   `method=median` on every question so the choice stays auditable. Alternative: keep the median
   and accept the cliff.
