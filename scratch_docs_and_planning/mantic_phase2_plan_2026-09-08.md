# Mantic Phase 2: first-class date questions and Mantic-optimized forecasting (plan, 2026-09-08)

Branch `mantic-competition`. Phase 1 (the `--mode mantic` platform seam) is committed through
b404470 and smoke-tested live on 2026-09-08 (post 650, 451-point CDF accepted, private comment
accepted, exit 0). This plan is the spec for Phase 2. A fresh agent should read, in order:
`scratch_docs_and_planning/mantic_integration_plan_2026-09-08.md` (Phase 1 design and verified API
facts), `docs/operations.md` "Mantic (Crucible) tournament", and the research dossier in
`scratch_docs_and_planning/mantic_research_2026-09-08/` (four reader reports, the ranked edge-case
review, the Series 1 and Series 2 rules docs, Mantic's OpenAPI spec, and the recorded corpus of all
556 public Mantic posts). Every number below has a receipt in that dossier.

## Operator direction (2026-09-08)

Full, first-class support for every Mantic question type is required before this is "done";
no hacks; the implementation should be robust and optimized for Mantic's implementation.
Publishing on Mantic right away is fine; volume is not a concern (paid $3 per forecast).
One paid smoke per approval; the next approved smoke is the date question (post 651) once
date support lands, run with `--only-posts 651`.

## What the research established (load-bearing facts)

- forecasting-tools and the Metaculus backend (which Mantic forked) both treat a date question
  as a numeric question on an epoch-seconds axis. `NumericDistribution.from_question` converts
  the datetime bounds with `.timestamp()` and sets `is_date=True`; `DateReport` inherits
  `NumericReport.publish_report_to_metaculus` verbatim; the server validates a date CDF with the
  same type-agnostic rules (length inbound+1, min step round(0.01/N, 9), max step 0.2*200/N,
  bound pins). The publish path needs nothing date-specific.
- Bin edges are uniform in unscaled location by construction (`linspace(0, 1, N+1)` mapped
  through the linear or geometric scaling). `numeric/pchip_cdf.build_cdf_value_grid` reproduces
  the platform's `continuous_range` to 0.0 on the live 12-bin date question and to 2e-15 on every
  log-scaled numeric question in the corpus. Calendar-month bins are NOT expressible in the
  platform's scaling math and `month` is absent from the API enum (`date_granularity` is
  `day | week | ""`), so non-uniform date grids cannot happen today. Do not build for them; assert
  on the enum.
- CDF index semantics, confirmed by reproducing Mantic's own baseline scores on 192/192 resolved
  date questions and 146/146 out-of-range resolutions: cdf[0] = P(x < range_min); cdf[i] =
  P(x <= edge_i); bin k (0-based) carries cdf[k+1] - cdf[k] over (edge_k, edge_{k+1}], with the
  first bin also owning the left edge; 1 - cdf[N] is the above-upper mass. For a day-granularity
  question, calendar day D = nominal_min + k days is bin k. Mantic sets `nominal_max` to the LAST
  bin's left edge (range_max = nominal_max + one bin) for dates; for discrete questions the
  convention is center-aligned (half-step), and the live quantitative question 650 even violates
  the documented discrete convention. Read `nominal_min/max` from `api_json` and never re-derive.
- Mantic's question mix (Series 1, 520 resolved): date 200, discrete 157, numeric 137, MC 26,
  binary 0. 51% of resolved date questions resolved ABOVE the upper bound; 22% of discrete and 7%
  of numeric resolved out of range (Metaculus archive: 2 to 3%). The out-of-range bucket is scored
  against a fixed 0.05 reference: a 1% tail scores -80.5, 5% scores 0, 50% scores +115. Our
  pipeline publishes exactly 1% out of range whenever all 13 percentiles sit inside the range.
- Baseline scoring is strictly proper: the optimal submission is the honest distribution; only the
  incentive to answer every question changes (a miss scores the field's 25th percentile, measured
  27 points below a median answer). Clamps: loosening binary 0.98 is worth at most +2.7 points and
  Series 1 had zero binary questions; the MC 0.01 floor is forecasting-tools' own. Refused.
- Multi-resolution (`multi_resolution: true`, live on post 650): one distribution scored against N
  resolution values, averaged. Optimal = the expected empirical distribution of the resolution set
  (mixture). Priced on 650: a single-day distribution loses 38.7 points. For MC/binary the optimal
  is the expected FREQUENCY over options / expected fraction of Yes. The count N is not in the
  API while open (`resolutions` is null); it lives in the criteria prose.
- GitHub Actions currently delivers about 22% of this repo's scheduled cron firings (7 to 23 of
  72 per day since 2026-08-27, measured via the GitHub API). Every Series 1 question opened on the
  hour with an exactly 60-minute window. The Mantic workflow has two crons (:17/:47).

## Design

### Wave A: date questions, first class (epoch-seconds adapter; decision: option A)

The question object stays a `DateQuestion` end to end (so the framework builds a `DateReport`,
telemetry says `qtype=date`, persistence and markers see a date). Numeric math runs on an adapter.

1. NEW `metaculus_bot/numeric/date_axis.py`: `as_epoch_question(q: DateQuestion) -> NumericQuestion`
   (lower/upper = `.timestamp()` of the tz-aware bounds; open flags, zero_point, cdf_size copied;
   `nominal_lower_bound/nominal_upper_bound` read from `q.api_json["question"]["scaling"]
   ["nominal_min"/"nominal_max"]`, falling back to range only when absent; id_of_question,
   id_of_post, page_url, unit_of_measure carried); `numeric_view(q)` (identity for NumericQuestion,
   adapter for DateQuestion; raises for anything else); `parse_iso_utc(s) -> datetime`: strict
   ISO-8601, tz-aware UTC, a naive timestamp is UTC (never host-local: forecasting-tools' own
   template has this bug), a DATE-ONLY value maps to 12:00:00 UTC of that day so mass lands
   inside that day's bin under the right-closed convention; `format_epoch(x, granularity)`
   renders `YYYY-MM-DD` for day/week granularity and `YYYY-MM-DDTHH:MM:SSZ` otherwise.
2. `forecaster.py`: `supported_types` gains `DateQuestion` (fix the comment); `_make_prediction`
   dispatches `DateQuestion` to a new `_run_forecast_on_date` (overrides the framework's stub;
   `ConditionalQuestion` stays NotImplemented); fix the stale comment above the EXTREME_CALL block.
3. `forecaster_runners.py`: `run_date_forecast` as a thin wrapper of the numeric runner: date
   prompt, same `invoke_with_broad_retry`/soft deadline, `extract_date`, then the SAME
   `_build_guarded_numeric_distribution` on `as_epoch_question(q)` with `is_date=True`; no
   discrete-integer vote. `build_parse_notes` gets a date sibling (ISO format rules, forbid bare
   year or month, open-upper semantics).
4. `prompts.py`: `date_prompt` FACTORED from the numeric template (shared rules stated once as
   named constants used by both; docs/prompts.md rule), replacing only the date-sensitive blocks:
   Units/Bounds (answer-date range from nominal bounds; bin granularity from
   `date_granularity`; "a date means that calendar day's bin"; UTC), the STRUCTURED FORECAST
   example (ISO strings, `question_type: "date"`), no `outcome_type` step. `bound_messages`
   (numeric/utils.py) takes a value formatter so open-bound text reads dates, not epoch floats.
5. `structured_output_schema.py`: `DateStructured` (`declared_percentiles: dict[float, datetime]`,
   same monotonic/key validators, `extra="forbid"`); registered in the union, the type map and
   the three `question_type` Literals. `value_extraction.py`: `extract_date` mirroring the numeric
   trio (fidelity checks on the epoch values). `structured_parse.py`: a `DatePercentile` list
   wrapper for rung 3 (reuse forecasting-tools' `DatePercentile`).
6. `question_types.py`: `QuestionType` gains `"date"`; `question_type_of(DateQuestion) == "date"`
   (additive marker token; the regexes read `\S+`). `tool_runner._VALID_TYPES` stays without date.
7. Routing sites made date-aware through `numeric_view`: `aggregation_pipeline._combine_by_type`,
   `get_threshold_for_question` (date uses the numeric threshold), `run_stacking`;
   `spread_metrics.compute_spread`; `stacking_route._STACKING_ENV_BY_QUESTION_TYPE` gains
   `DateQuestion -> NUMERIC_STACKING_ENABLED` (closes the fail-open default). Pin: a date question
   skips stacking under the default env.
8. `is_date` threaded through the three distribution constructors (`pchip_processing.py` x2,
   `numeric/utils.py`), so the published comment renders ISO dates via the framework formatter.
9. Bounds clamp tolerance (`numeric/bounds_clamping.calculate_bounds_buffer`): for a bin-defined
   grid the buffer is `max(current formula, one bin width)`; a date one day outside a closed
   bound clamps instead of dropping the member; a scale error still raises. Constant-backed in
   `numeric/config.py`; log the clamp with the original value. (Also edge-case item 11.)
10. `nominal_bounds` (numeric/utils.py): never derive half-step bounds for a DateQuestion (nominal
    == range unless the API says otherwise); the adapter supplies them.
11. Gap-fill v2: `research/agentic/driver_prompt.py` `SupportedQuestion` gains `DateQuestion`
    with `_question_header`/`_template_skeleton` branches rendering dates; `loop.py` `_GHOST_QTYPES`
    and `_declared_qtype` gain date.
12. Publish hardening: NO change to `_PATCHED_REPORT_TYPES` (adding DateReport would raise);
    extend the pin in `tests/test_publish_hardening_concurrency.py` to assert `DateReport`
    resolves through the MRO to the patched publish.
13. Tests: `build_cdf_value_grid` equals the recorded `continuous_range` (12-bin date, 201-bin
    date, 451-bin discrete fixtures); a second date fixture (legacy 201-bin, closed-lower/open-upper
    "when will X" question from the corpus dump) in `tests/mantic_fakes.py`; UTC parse tests
    (naive date-only -> noon UTC, 'Z' timestamps honoured, local-timezone independence);
    closed-bound clamp; `_type_gate_enabled` for dates; e2e pins inverted (the date question is
    forecast and published: 13 values, cdf[0]==0.0, cdf[12]==1.0, comment renders ISO dates,
    `MEMBER_FORECAST qtype=date`); an oracle that a forecaster certain of 2026-09-16 puts the
    maximum mass in bin 8 of question 651.
14. Out of scope, documented in FUTURE.md and at the code's own asserts: backtest, ablation and
    performance_analysis for date questions; Mantic residual analysis.

### Wave B: Mantic-optimized forecasting and robustness (edge review + scoring reader)

Prompt edits (one merge; every clause a named constant with its reason, one interpolation site,
presence and absence pins under tests/prompts/, none in the stacking prompts):

- B1 Out-of-range base rate (edge #1): Mantic-gated sentence in `bound_messages` for open bounds:
  on this platform roughly one in five resolved discrete questions and half of date questions
  resolved outside the displayed range, so keeping every percentile inside asserts a 1% chance
  of an out-of-range outcome. Platform read via a leaf `question_platform(question)` on
  `page_url`'s host against `QUESTION_PLATFORM_HOSTS` (default Metaculus when None).
- B2 Scoring paragraph (edge #6): delete the Metaculus-specific "uniform 0.01 PDF floor" and
  "sharpness above ~35" sentences and the "Your Metaculus question" attribution; keep the
  proper-scoring sentence; add one sentence: mass beyond an open bound is scored as its own
  outcome against a reference of a few percent, so starving it is heavily punished.
- B3 Platform-aware scoring sentence (F8): Metaculus text names the SPOT peer log score (what
  those tournaments use); Mantic text names Crucible's spot baseline log score, "compared to a
  uniform distribution rather than to other forecasters, so nothing is gained by disagreeing with
  the obvious answer and nothing is lost by giving it".
- B4 Series-variant clause (edge #9, prompts.py ~789-793): keep the correction, drop the false
  premise: the displayed range is weak evidence about WHICH series variant resolves and no
  evidence about the magnitude of the outcome.
- B5 Multi-resolution clause: gated on `api_json["question"].get("multi_resolution") is True`
  (identity test, not truthiness: MagicMock stubs are truthy), type-aware: quantitative/date ->
  forecast each resolution instance, pool into one mixture, report the mixture's percentiles;
  MC -> the expected share of resolutions per option; binary -> the expected fraction of Yes.
  Never interpolate N. (The edge review preferred to watch; the scoring reader priced the omission
  at 38.7 points on the live question; the operator wants Mantic-optimized, so ship it.)
- B6 Coarse-grid sentence (edge #7 part 1): templated on `precision`/`date_granularity`: name the
  bin width and bin count; a percentile's value selects the bin it falls in; mass finer than a bin
  is wasted.

Numeric fixes:

- B7 `pchip_cdf.py`: give the min-step rebuild trigger (~664) and the raise (~529) the 1e-10
  tolerance the file already uses at 552/575, so a near-total out-of-range forecast builds instead
  of dropping the member (edge #3). Reproductions at 15, 451, 2001 points as tests.
- B8 Count-like cluster spread capped by the grid: spread = min(1.0, bin width), total plateau
  spread <= one bin width (edge #7 part 2), so a concentrated forecast on a 3 to 21-bin grid does
  not publish flattened. Config-era boundary for Metaculus discrete too; land before the season.
- B9 Discrete-snap guard keyed on the grid step being 1.0, not `cdf_size == 201` (edge #14).
- B10 Telemetry: additive `oor_low=` and `oor_high=` fields (out-of-range mass) on the per-member
  and aggregate numeric markers, appended at line end; registry + verbatim example lines (edge #8).
  HOLD the mechanical 5% tail floor until these fields show whether models already place mass
  beyond bounds (break-even escape rate 5%; Mantic measures 15% and 51%).

Fetch/publish/ops robustness:

- B11 URL extraction: balanced-bracket atom and trailing-backtick strip in
  `research/resolution_url_scan.py` (edge #4; six truncated Federal Register queries in the
  corpus, the truncated form answers 200 with an unfiltered count). Tests from posts 434 and 598.
- B12 MC label normalization: NFKC plus curly-quote/dash/nbsp folding in `mc_processing._normalize_name`
  and the canonical map in `value_extraction` (edge #15).
- B13 Parse-drop counter: `ManticClient` bumps a module counter and logs `MANTIC_POST_DROPPED:
  post=<id> type=<wire type> error=<class>` before re-raising; cli reads it into the alertable
  arithmetic (pattern: generic_fallback); `.get` reads for telemetry-only fields (edge #12).
- B14 Ten-question cap truncation at WARNING with the dropped post ids (edge #22).
- B15 Stale slug goes red: mantic mode turns "past MANTIC_TOURNAMENT_END_DATE" into a non-zero
  exit after publishing (pattern: check_fall_cup_reminder), keeping the shared hard stop (edge #5);
  plus a `MANTIC_TOURNAMENTS` startup marker listing ongoing bots-only tournaments on the API and
  warning when one is not the configured slug (Series 2 discovery).
- B16 Forecaster-permission preflight in mantic mode: one authenticated GET of
  `/api/projects/tournaments/<slug>/` after the identity preflight; raise `ApiIdentityError`
  unless `user_permission` grants forecasting (edge #20). Fails shut before spend.
- B17 Pagination: pass a ceiling `num_questions` with `error_if_question_target_missed=False` so
  the framework walks pages instead of trusting Mantic's unreliable `next` (edge #21).
- B18 Cron cadence: six entries :05/:15/:25/:35/:45/:55 (free minutes; distinct-minute pin holds);
  document the 22% delivery finding and the Series 2 decision (in-run re-poll vs an always-on
  `gh workflow run` dispatcher, since dispatch events are not dropped).
- B19 `mc_processing.py` docstring: Mantic runs 50-option ballots (refused loosening recorded).

### Wave C: before Series 2, with live data

- C1 Per-bin PMF elicitation for enumerable grids (<= ~31 bins): forecasters emit a probability
  per bin (as MC does), the CDF is built from the PMF, aggregation stays in CDF space. Motivation:
  impossible bins (weekends on a trading-day date question) cost 14.4 points with zero information
  under percentile-to-PCHIP; Series 2 makes coarse grids the default. Own design and e2e.
- C2 Supply probe Mantic mode (free, unauthenticated) to measure forfeits per release hour
  (edge #10); then the cadence/dispatcher decision (edge #2).
- C3 Fast-path alertability in mantic mode once the window length is known (edge #13).
- C4 Replay archived per-member CDFs under the Mantic baseline formula, median vs mean
  aggregation (edge #18; re-measurement of a recorded decision, not a re-litigation).
- C5 After 2026-09-20 12:00 UTC: read question 651's resolution string and reported baseline
  score to close the on-edge bin-mapping question; read one resolved Series 2 question's score to
  learn the continuous coefficient Mantic actually uses.
- C6 Starved outer tail (edge #19) stays a documented watch item; not bundled with B7.

### Accepted with no change (do not re-raise)

Binary clamp loosening (max +2.7 points; zero binary questions in Series 1); MC floor (forecasting-
tools' own); never withdraw a forecast (a withdrawn forecast scores 0 instead of the 25th
percentile fallback); early-close min() (0 of 520 Series 1 questions closed early); publish thread
pool sizing; clock/timezone handling; upcoming/group/conditional/notebook posts (Mantic produces
none); 400/404/405/429 publish handling; resolution-time bin aggregation (no adjacent-bin hedging);
the 70-minute step cap; high-cardinality MC ceiling; Mantic comment backfill.

## Execution

Wave A and B run as parallel implementation agents on disjoint files (A1-A13 split by module
family; B grouped as prompts, numeric, fetch/ops, telemetry), TDD, opus, no paid runs, no
commits by agents; orchestrator runs `make all`, commits, `/forge`, fixes, re-gates. Then the
approved date smoke (`--only-posts 651`), verified on the API (13-value CDF accepted, comment
renders dates, `my_forecasts.history` populated). Then merge (operator), which auto-enables the
six-cron schedule.
