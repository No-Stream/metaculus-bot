# Next-season bundle (2026-09): everything for the fall 2026 season, in one merge

Branch `next-season-bundle`, forked from `resids-sept1` at 3e3b950. **`resids-sept1` has not
merged to `main`** (10 commits ahead, 0 behind at fork time), so this PR carries both. The plan's
fallback rule for that case was followed; if you would rather merge `resids-sept1` first, this
branch rebases cleanly onto it (it IS a superset).

Plan: `scratch_docs_and_planning/next_season_bundle_2026-09_plan.md`. Every item below was
implemented, tested with the free gates only, and merged in ONE branch so next season's records
carry one era boundary. No paid run was made while these items were built; no live mode, backtest,
ablation, or GitHub Actions dispatch. (The three dispatches made since, Test Bot run 67 and the
zero-spend cup QA run on 2026-09-03 and Test Bot run 33907102246 on 2026-09-04, plus the two
operator-authorized cents-level live probes of 2026-09-04, are recorded in the last two sections.)
Validation artifacts (corpus
scripts, counts, hand-review write-ups) are under `scratch/next_season_bundle_2026-09/<item>/` in
the main checkout (gitignored).

## Gates

HEAD f6eae2b, the end of the eighteen-item block (all 18 items + codex fixes + forge fixes merged):
`make format` no changes,
`make lint` clean, `make typecheck` 0 errors, `make lint_imports` 6/6 contracts, `make deps` clean,
`make test` 6562 passed / 14 skipped / 5 deselected in 139 s. Forge review and its fix pass: see
the section at the end. Every later block re-ran the same gates; the last, at the tip 001b6f9, is in
the final section (7,655 passed), and PR CI is green on the pushed 1f2b504; the commits after it
carry the Codex review triage, the browser-transport closure and the closure's review fix wave.

## Tier A: publish path

**Item 1: k=1 thin-publish floor for binary questions.** When exactly one forecaster survives a
binary question, the published probability is clamped into `[0.05, 0.95]`. Trigger is
`skip_reasons[qid] == "single_forecaster"` AND `BinaryQuestion` inside
`AggregationPipeline._base_combine`'s single-prediction branch; the stacker's pre-stacked single
output shares that branch and is never floored (no skip reason). Helper
`apply_thin_publish_floor(value, survivors)` in `post_processing.py`; constants
`THIN_PUBLISH_BINARY_FLOOR` / `CEIL` are aliases of `EXTREME_CALL_LOW` / `HIGH` so the band has
one definition. WARN `THIN_PUBLISH_FLOOR: question= raw= clamped= survivors=1` only when the value
moved; harvested as `thin_publish_floor`. The per-model comment bullet keeps the raw value. Offline
replay over the 2026-08-31 archive reproduces the receipt: only q44874 moves (0.03 to 0.05, +51.08
spot peer); q44870/44871/44873 and all seven k=3 binaries are untouched. Accepted gap, stated in
the code: a lone survivor under plain MEAN/MEDIAN is not floored (prod and the default run
CONDITIONAL_STACKING). Under STACKING in the benchmark factory the floor does fire, deliberately.

Codex (GPT-5.x) read-only review of this diff, per the plan: one important finding and two nits,
all applied. The implementing agent had added a defensive pop of a stale skip reason when a stack
fires; codex showed that with `research_reports_per_question > 1` a sibling report's failed stack
attempt would pop the lone survivor's reason and publish it un-floored with its
`STACKER_SKIP_REASON` marker lost. The pop is reverted; the pre-existing asymmetry (only skip
paths write the reason, only the comment builder clears it) is documented at the trigger site, and
a regression test covers the multi-report case. The constants were aliased rather than duplicated,
and "widens the admissible range" was corrected to "narrows" (relative to the per-model
`[0.02, 0.98]` clamp).

## Tier B: gemini search provider

**Item 2: vintage / as-of clause** in `web_research_prompt` GUIDELINES (both consumers: native
search and gemini): every dated or forward-looking claim carries its publication date; schedules
and plans say when and where they were announced; no undated recollection presented as current.

**Item 3: strip model-authored hierarchical citation indices** (`[2.4.1]`, `[1.1.1, 1.1.2]`,
`[A: NASA, 1.1.2]` to `[A: NASA]`) in `gemini_search._strip_model_citation_indices`, run AFTER
`_splice_inline_citations` (byte offsets) and never on the `### Sources` block. Prompt half: the
auto-annotated citation clause now bans model-written indices (gemini only; native's markdown
branch untouched). Corpus validation over all 323 archived sections: 171 touched, 2,609 bracket
groups holding a dotted token, 4,307 tokens removed, 0 false positives, 0 idempotency failures.
The plan's named risk, two-component tokens like `[1.5]`, was resolved by reading all 165 distinct
cases: every one is a citation index, so the `>=3-component` fallback was NOT adopted (it would
have left 318 fake markers standing). Rule as shipped: citation-delimited dotted runs whose
components are at most two digits; `[3.8%]`, `[$1.5]`, `[v2.1.3]`, `[2026.08]`, `[192.168.1.1]`
each have a preservation test.

**Item 4: unsupported-attribution check** in new `research/gemini_attribution.py`, called after
the strip on the grounded path. Outlet-named tier tags whose outlet is not in the response's own
grounded labels are rewritten to `[unverified attribution]`; the tier grade goes with the outlet;
supported outlets in a mixed bracket survive verbatim; nothing outside a bracket is ever changed
(proved structurally in the harness). Six keep-biased matching rules (concatenation, all identity
tokens, token intersection, sub-brand core, single-token abbreviation, domain-core abbreviation)
plus the generic-tier-word skip. Corpus validation: 478 of 681 outlet-named tags (70%) rewritten
across 48 sections; reconciles to cutB's 87% (86% under the audit's looser rule through the same
harness); all 11 fully-unsupported sections hand-read as true positives; residual arguable 2 of
681 (NewsRadio WFLA vs iheart.com); false-keep exposure 20 occurrences / 10 names (2.9%). Skipped
when no chunk carries a renderable label (q44802). Marker
`GEMINI_UNSUPPORTED_ATTRIBUTION: question= tagged= unsupported= groups= labels=`; a new
`details["counts"]` diagnostics convention carries the count to the Provider Diagnostics line
(zeros omitted) and the schema-v2 archive.

**Item 5: market-odds bullet narrowed** to the operator-confirmed wording (sources OTHER than
Polymarket/Kalshi/Manifold/PredictIt; name market and date; do not report covered-venue prices
from search). Verified character-exact. Benchmarking still renders nothing.

**Item 7: `GEMINI_GROUNDING_DENSITY: question= chunks= supports= chars=`** on every grounded
response (chars = raw model text). Not a gate. Not emitted on the suppressed or url_context-only
paths.

## Tier C: research / provider fixes

**Item 8: null-result reading clause** (`_NULL_RESULT_READING`) in the three base prompts beside
the evidence-weighting rubric and, auditor-phrased, in the gap-fill analyzer. Stacking prompts
untouched (tested).

**Item 9: present-tense instrument-gap rule** in the gap-fill analyzer (a live-data-source
question must carry at least one "what does it read now" gap; future-tense unanswerable gaps are
rewritten or dropped) and a matching bullet in the v2 driver's question-type defaults.

**Item 10: financial_data bundle.** (a) `HARD_PEG_ANCHORS`: eleven pegged currencies, each entry
sourced; a pegged cross renders its own block plus the liquid anchor's block, labeled, never
substituted. (b) Variance-ratio noise flag: Lo-MacKinlay `variance_ratio` on the full held
history in `ts_estimators.py`; `FINANCIAL_VARIANCE_RATIO_FLOOR = 0.6` calibrated on 20 seeds
(clean random walk mean 1.005, min 0.636; quote-noise series mean 0.465, max 0.617: 0/20 false
flags, 18/20 caught). Divergence from the plan, stated: promoting the long-window vol does not
reach the case (both horizons use one-day returns, equally inflated), so the flagged headline is
the multi-period (5-day) vol, which equals one-day vol times sqrt(VR) and recovered 10.8% where
truth was 9.4% and the 30-row print said 14.6%. (c) Long-horizon vol line beside the 30-row one,
row count and step unit labeled. (d) FRED `.4g` replaced by fixed-point up to six decimals
(331.893 stays 331.893; the Fed balance sheet no longer prints `6.7e+06`). (e) ALFRED
first-release vs current-vintage table for FRED series named in the resolution criteria, with the
mandatory one-adjustment caveat; skipped for `FRED_NON_REVISING_SERIES`; live path only (the
keyless benchmarking CSV cannot pin first releases). Sibling fix in `ts_render._realized_vol_lines`
(same noise screen, separate droppable commit 2c2c8a5). New `FINANCIAL_NOISE_FLAG` line, harvested.

**Item 11: `no_resolving_content` FetchStatus** for HTTP-200 pages whose extracted text is an
embed shell (Infogram/Flourish/Tableau; Datawrapper excluded because it has a fetch route).
Divergence from the plan, stated: the receipt page (q44554, racetothewh.com) extracts ~2,900
characters of real prose, so no floor the plan's false-positive test allows can catch it. Shipped
the honest split: below a 400-character floor (calibrated on 116 archived per-URL results; every
archived success below 400 is site chrome; the shortest content-bearing page is 401) the page is
withheld under the new status; above it the prose renders and one bracketed line discloses that
figures inside the embed are NOT in the text. That disclosure is forecaster-visible text beyond
the item's literal ask.

**Item 12: market staleness.** Close cell renders `2026-02-27 (162d ago)` when the close predates
the forecast (`MarketSnapshot.forecast_time`, from `as_of` or now, so archived snapshots replay
identically); ranker prompt gains a recency tiebreaker within a relation tier;
`cap_stale_top_tier` demotes `same_quantity_same_date` one rung when the close predates the
question's open by more than `MARKET_STALENESS_TIER_CAP_DAYS = 60`, annotated in the `why` cell
via a new `tier_cap_note` field; Kalshi close-vs-settle sentence in the legend. Measured on 102
archived snapshots: disclosure fires on 62/711 parent rows (7 still `open`); the tier cap fires on
ZERO archived rows, and would not have touched q45163's offender (graded
`same_quantity_other_cut`, one rung below the cap's target). Widening the cap to that rung is an
operator decision (it changes how the forecaster prompt weights the row). Section character
budgets re-derived (maxed 11,050 vs 11,249 ceiling). `MARKET_TIER_CAPPED` line, harvested.

**Item 13: fetch escalation ladder design doc** at
`scratch_docs_and_planning/fetch_escalation_ladder_design.md` (626 lines, no code). Recommends the
meta-refresh rider and a registry-driven derived-API rung first (deterministic, no spend), and the
url_context rung behind a flag after the operator-gated probe. Seven open questions listed at the
end for separate approval. Superseded on 2026-09-03: the operator answered those questions inline,
`scratch_docs_and_planning/fetch_ladder_plan_2026-09-03.md` replaced the routing half of the
design, and the ladder itself shipped on this branch (see "The 2026-09-03 work" below).

## Tier D: measurement and telemetry

**Item 14: FORECASTERS_USED guards.** Verified already emitted on every path; added presence
tests across all three published question classes and both strategy branches, a trim-survival
test at the non-stacking call site, and a test pinning the multi-report accumulation invariant.
Mutation-checked (three temporary production mutations, each caught).

**Item 15: `EXTREME_CALL: question= model= p= side= lone= survivors=`** per extreme binary member
(p <= 0.05 or >= 0.95, inclusive), same-side lone rule, constants beside `BINARY_PROB_MIN/MAX`.
Replayed over 436 archived binaries: 570 lines emitted, 570 parsed back; reproduces the memo's
gemini row (7 extreme, 3 lone). Caveat: the memo's scripts used an either-side lone rule; they
differ on 4 of 570 pre_flip member-calls (52 vs 48 lone), agree exactly post_flip and triple era.

**Item 16: per-role dollar attribution, IMPLEMENTED** (not the spec fallback). litellm 1.92
already requests OpenRouter usage accounting; `build_llm_with_openrouter_fallback(role=...)`
stamps a litellm `metadata` role tag; `credit_telemetry.RoleSpendTracker` (a CustomLogger
installed once in `cli.main`) sums `usage.cost` + `usage.cost_details.upstream_inference_cost`
per (role, key); end-of-run
`CREDIT_ROLE_SPEND: role= key= usd= calls= costed_calls= byok_usd=` beside `CREDIT_SPEND`;
`usd=n/a` when no call carried cost, never a fabricated zero. `cli.py` drains litellm's callback
worker before `asyncio.run` returns (`_forecast_with_callback_drain`, 10 s bound).
`scripts/reconcile_credit_spend.py --roles` compares the ledger to settled spend. One fact only a
live run could settle, whether the donated key's BYOK routes populate `upstream_inference_cost` as
OpenRouter's docs state, was settled yes by the 2026-09-04 smoke run (see "Smoke evidence so far"). Roles outside OpenRouter (Gemini grounded search, v2 read_document, AskNews,
Exa) are documented as absent.

**Item 17: FUTURE.md** now quotes the measured $0.38 to $0.41/question (OpenRouter-only lower
bound, 29 triple-era runs / 33 questions) and marks the ~$1.65 figure superseded and never
measured.

**Item 18:** `docs/operations.md` "Season-start checklist" (live OpenRouter model-list read
command, what to check before a roster change, the era-boundary rule, `make supply_probe`);
pointer comment above `FORECASTER_LLMS`.

**Item 19a: set-valued PIT.** Out-of-range resolutions are intervals (`[cdf[-1], 1]` above,
`[0, cdf[0]]` below); coverage counts intersection; point statistics exclude intervals and report
`n_oob_interval`; one home (`PitReading` / `out_of_range_pit_reading` in analysis.py), consumed by
width_monitor. `compute_pit_details` renamed `compute_pit_reading` (no compat shim, by design);
`pit_std` / `mean_pit` are `float | None`. Archive: exactly one record changes verdict (q44842);
triple-era cov80 0.727 to 0.818 (n=11).

**Item 19b: starved-outer-tail detector** (`scan_outer_tails` in width_monitor.py; new
"Starved outer tails" report section; `--output-starved-json`). Divergence from the plan, stated:
`1 - F(declared_p99)` is ~0.01 on every record by construction and carries no signal (q45218
reads 0.0142); the shipped trigger is the in-range band's mean per-bin mass as a multiple of the
platform minimum step, `STARVED_OUTER_TAIL_FLOOR_MULTIPLE = 2.0`. Calibration: q45218 fires both
sides (band 0.004242 over 27 bins, flat-zone score -219.53, reproducing its dossier), q44182 fires
(-219.02 = its actual published score), q44842 cannot fire (`declared_beyond_bound`). Finding: the
shape is systematic, 68 of 417 measurable open-bound sides across 49 questions. The optional
publish-time WARN was deliberately NOT shipped: on discrete questions (where both motivating
records live) the resample overwrites `declared_percentiles` with a bound-pinned grid, so a
publish-time reader would silently never fire on its own cohort; firing correctly needs new
publish-path plumbing. Recorded in the code comment, docs, and FUTURE.md.

**Item 19c: `scripts/supply_probe.py`** + `make supply_probe`: per-slug counts at open / closed /
resolved plus the past-`scheduled_resolve_time` backlog. Validated against the 2026-08-31 round's
own census (26 closed, 37 total, 16 overdue, worst 17.1 d, exact match). Default slugs from
constants; the `metaculus-cup-fall-2026` row going non-zero is the cheapest fall-cup opening
signal.

**Item 19d: `RESOLUTION_SOURCE_FETCH: question= url= status= http= embeds=`** per fetched URL
(Datawrapper hops share the line, distinguishable by url), replacing the free-text
`resolution_source fetched ...` lines (no double logging; reason lines that carry more than the
triple remain).

## Telemetry archive

Eight new `MarkerSpec`s, each with parse tests and an archive round-trip test: `extreme_call`,
`thin_publish_floor`, `resolution_source_fetch`, `credit_role_spend`, `gemini_grounding_density`,
`gemini_unsupported_attribution`, `financial_noise_flag`, `market_tier_capped`.

## Docs

AGENTS.md (CLAUDE.md symlink) described every change above until the 2026-09-03 de-bloat moved
that narrative into `docs/` (see "The 2026-09-03 work" below); docs/operations.md gained the
season-start checklist, per-role spend, and one "Reading run logs" bullet per new marker;
docs/research.md gained the gemini strip / density / attribution paragraphs; FUTURE.md carries
the follow-ups (unshipped publish-time WARN and its no-plumbing alternative, starved-tail
prevalence watch, wider tier-cap option, `ts_render._fmt` precision sibling, `no_resolving_content`
seam, "unreachable" notice wording, `metaculus_get` helper consolidation, Manifold fixture
`outcomeType` gap).

## Operator decisions still open (from plan §5, plus new ones)

1. **Paid smoke test:** done twice. The operator dispatched `test_bot.yaml` (Test Bot run 67,
   2026-09-03) on 39877f1, which covers everything up to and including the 2026-09-02 prompt work
   and the clip-threshold commit, and again on cbc26bf (Test Bot run 33907102246, 2026-09-04),
   which covers the fetch ladder, the paid rung turned on, the cup, credit and model changes. The
   Kalshi single-flight fix landed after the second run and was not smoked separately: its
   exposure is a rate limiter, and the free gate covers it. Details in "The 2026-09-03 work" and
   "The 2026-09-04 work" below.
2. **Merge timing:** one merge before the fall cup publishes its first questions (the project has
   existed since 2026-08-28 and still had zero questions on 2026-09-03), and in any case before
   2026-09-20, when the summer tournament's hard stop starts reddening the tournament runs.
3. **Item 13 design doc:** superseded; its routing half was replaced by
   `scratch_docs_and_planning/fetch_ladder_plan_2026-09-03.md` and implemented on this branch.
4. **Item 12:** widen the tier cap to `same_quantity_other_cut` as the 45163 dossier wanted?
5. **Item 11:** the above-floor embed disclosure line is forecaster-visible text the plan did not
   ask for; keep or drop.
6. **Branch base:** merge `resids-sept1` first, or take this PR as the superset.

## Forge review

`/forge` ran over the whole branch diff (base 3e3b950): 17 reviewers (8 always-on, dataframe and
structure conditionals, four stated-concern reviewers, three codex/GPT-5.x lenses), a 3-lens
adversarial panel on every Important finding, and one triage agent; 75 agents in total. Full report:
`scratch/next_season_bundle_2026-09/forge_report.md` (plan JSON beside it).

**Verdict: needs-work, 0 critical, 10 important, 31 report-only, 1 dropped.** Triage's own framing:
no design errors; one run-status bug, two forecaster-facing text errors, one half-fixed precision
regression, two oversized modules, four test/telemetry gaps.

All ten FIX findings were applied, plus 28 of the 31 report-only items under the operator's rule
(fix if real and the fix is cleaner than living with it; skip unreachable corners). Seven worktree
agents did the work; every branch passed the full free gate set before merge.

| id | what was wrong | what shipped |
|---|---|---|
| F1 | litellm callback drain in `cli.py`'s `finally` could raise on its 10 s bound and turn a published run into a crash with no end-of-run summary | `TimeoutError` caught inside `drain_litellm_callbacks`, WARN `LITELLM_CALLBACK_DRAIN_TIMEOUT`, stale comment corrected, drain-level test |
| F2 | euro-pegged anchors used `EURUSD=X` (moves opposite to `USDDKK=X`); regime sentence put the per-euro level under the USD cross | anchors are `EUR=X` (per-USD, like every other row), gloss on the anchor line, regime sentence re-subjected to the currency, table-invariant test |
| F3 | `ts_render._fmt` still printed 100-10000 levels at one decimal | fixed-point to three decimals above 100, `:.4g` below |
| F4 | `width_monitor.py` 1192 lines | starved-tail detector moved to `outer_tail.py`; `cdf_and_grid` / `NUMERIC_TYPES` to `scaling.py`; width_monitor 716 lines |
| F5 | `financial_data.py` 1365 lines | `currency_pegs.py` (167) + `fred_rendering.py` (311, incl. both FRED fetchers so `Fred` stays patchable); financial_data 949; tests split with shared `tests/financial_fakes.py` |
| F6 | embed disclosure trailed the page text, so the aggregate section trim deleted it | disclosure leads, wording "NOT in the page text below", aggregate-trim regression test |
| F7 | `FINANCIAL_NOISE_FLAG` had no series identifier | `symbol=` on both emitters, required regex group |
| F8 | three markers had no emitter-vs-regex pin | real caplog line parsed through `MARKER_SPECS` in each emitter test |
| F9 | estimators had no exact-value tests | hand-derived VR(2)=1.5375 / vol=3214.34 on `exp([0,1,2,4,7,11,16])`, lag-sensitive VR(2)=0 / VR(3)=0.35 |
| F10 | gemini citation ban contradicted the tier-tag mandate 26 lines later | carve-out sentence naming the tier block; `tier_tags` count recorded beside `unsupported_attributions` |

Report-only items applied: R1, R2, R4, R5, R6, R8 (the two sites guarding the new tier cap), R9,
R10, R11, R12, R13, R14, R16, R17, R18 (kept to a pure `asyncio.run` relocation), R19, R20, R21
(documentation only), R22, R23, R25, R26, R27, R28, R29, R30, R31, R7 (financial test split).

Report-only items skipped, each with a FUTURE.md note: **R3** (per-URL markers lost on the
resolution-source wall-clock timeout; operator decision: timing and fallback paths are not touched
for minor findings, and the fix would also start rendering partial snapshots on a timing path),
**R24** (the market-odds bullet's "snapshot provided separately" is false only when the provider
fails; the cheap gate cures the never-happens disabled case), **R15** (same peg anchor rendered
twice when two crosses share it; corner), **R8 sweep** (~30-50 pre-existing `getattr` defaults
across the research package; own PR), **R7** (`tests/test_telemetry_markers.py` split).

Three things the fix agents found beyond the findings: an out-of-order FRED response cost the
WHOLE series block, not just a scrambled table (the year-over-year `.loc` slice raised on a
non-monotonic index and the soft-fail swallowed it); a `.env.local` `METACULUS_API_BASE_URL`
override made the supply probe send the token to a host the identity preflight never vetted
(preflight URL now resolves lazily); the R16 comment was backwards in the direction the reviewer
said (a median anchor reads 61.5x the floor on the q45218 geometry, so lowering the constant would
convert a disclosed verdict into a clean bill of health).

Forecaster-facing text changed by the review pass, verbatim in the final report to the operator:
the tier-cap `why` note ("demoted from same-date: closed 163d before the question opened"), one
legend sentence defining it, the gap-fill analyzer "ALREADY ANSWERED COUNTS AS ANSWERED" carve-out,
the `_SOURCE_PROVENANCE_LADDER` bullet defining `[unverified attribution]`, the gemini citation
clause's tier-tag carve-out, and the narrowed market-odds policy now shared by the two Perplexity
prompts.

Market-snapshot character budgets were re-derived (maxed 11,150 / realistic 8,300 / fixed 2,850;
99 characters of headroom remain to the `RESEARCH_SECTION_CHAR_LIMIT / 4` ceiling, so the next
widening of that section has to cut prose).

Codex (GPT-5.x) second-opinion review of item 1, separately, per the plan: one important
finding (the stacked-path skip-reason pop; reverted) and two nits (constants aliased; "widens" corrected to
"narrows"); all applied before forge ran.

## After f6eae2b: the 2026-09-01 residual round and the 2026-09-02 prompt and schema work

Everything above describes the branch at f6eae2b. Two more blocks of work landed on top of it before
the merge, and both belong in this one commit for the same reason the eighteen items do: prompt text
and structured-block shape move the forecast distribution, so the fall cup's records have to fall on
one side of a single config-era boundary. That block ended at 8cd2295. The plan for the second block,
with the operator's item-by-item decisions, is
`scratch_docs_and_planning/announced_unscheduled_fix_plan_2026-09-02.md`.

Gates at 8cd2295, superseding the f6eae2b figures near the top: `make lint` clean, `make typecheck`
0 errors, `make lint_imports` 6/6 contracts, `make deps` clean, `make test` 6787 passed / 22 skipped
/ 5 deselected in 142 s. No paid call was made in either block.

## The 2026-09-01 residual round (six fixes, all merged here)

- **A hallucinated FRED series no longer erases the financial block** (6bf7641, ce06e6f). The
  classifier invented `DEXBOUS` for q45363's Boliviano cross, a series FRED does not have, so a
  currency question reached the forecasters with no level and no realized volatility. The classifier
  now routes every exchange rate to a Yahoo cross and may not invent a FRED id, a nonexistent id
  raises `UnknownFredSeries` and reaches diagnostics as `unknown_series` with a `FRED_UNKNOWN_SERIES`
  WARN, and a question whose exchange-rate identifiers all come back empty renders no section at all,
  with the gap carried by `counts.fx_identifiers_empty` rather than by prose (any prose would flip the
  orchestrator's provider status from `empty` to `ok` and defeat every downstream empty guard).
- **A host throttle interstitial is no longer a gap-fill v2 fetch success** (367dc69, f4f331e).
  ogimet.com answered two of three parallel same-host fetches with a 304-character "Limit for old
  data queries exceeded" page under HTTP 200, the driver read those as fetches, and the exact-date
  reference class it published came to 4 years instead of 6 (q45191). A short body carrying a throttle
  phrase is now `status=throttled`, never cached, never stamped `fetched` by provenance, logged as
  `AGENTIC_FETCH_THROTTLED`, and the driver's own retry of the URL no longer counts as a duplicate
  tool call.
- **Four outside-view prompt rules and a declared-window telemetry field** (6229f90, ca65cc5):
  `_COUNT_IN_PERIOD_REFERENCE_CLASS`, `_REMAINING_EXPOSURE_RULE`, `_ANCHOR_CONSISTENCY_RULE`, the
  analyzer's `_LAST_REAL_USE_GAP_RULE`, and `remaining_window_days` with its `WINDOW_DECLARED` line
  measured against scheduled resolution rather than close. Three of those four rules and the field are
  superseded by the 2026-09-02 work below, so they are listed here as branch history rather than as
  what ships.
- **Content-free HTTP 200s are withheld, and a page's own chart config is read** (19c4e27, 2b3fd1b).
  The `no_resolving_content` verdict lost its named-provider gate, because the round found five
  content-free `success` renders and not one of them named an embed provider (q45088's 127-character
  single-page-app tab list, q45215's 385 characters of Kazakh region names), so the gate was
  withholding one shape of chrome and publishing the other. Separately,
  `resolution_chart_data.render_inline_chart_data` now reads a Highcharts `data-chart` attribute or
  `Highcharts.chart(...)` call straight out of the HTML we already hold, with no second request and no
  LLM call, because q43949's resolving IOM page extracted about 80,000 characters of prose carrying
  none of the resolving figures while its annual series sat in that attribute. The scan is bounded by
  sites examined and states what the cap omitted.
- **`spot_peer_delta` is now the one place a counterfactual is priced** (96b9756, 3851f44, 90e73ea).
  Metaculus halves spot peer for continuous questions, and both conversions had been got wrong in the
  direction that inflates the figure: a q45065 replay doubled an already-halved `numeric_log_score`
  difference and priced a near-miss at +404 when the truth is +202, and thirteen dossier scripts
  quoted log-base-2 binary deltas as peer points, overstating each by 1/ln2, about 1.44. The helper
  raises on an unrecognised question type rather than silently taking the unhalved branch,
  `DISCRETE_QUESTION_TYPES` became `UNHALVED_QUESTION_TYPES` (binary and multiple choice) so the name
  says what the set is for, and `tests/test_peer_delta_convention.py` pins both conversions.
- **In-place re-resolutions and forfeited questions are now detectable** (a5f4b8d, d98b0d6, d45aedc,
  b690abb). Metaculus edited q44798 from 80 to 82 with `resolution_set_time` left at a stamp that
  precedes the pull which still read 80, flipping that record's spot peer from +5.41 to -5.42 while
  the earlier round's published tables went silently stale. So `--prior` diffs the resolution value
  and every `metaculus_scores` field against the previous pull, tags moved records
  `platform_rescored` as a ternary (None means never compared), and emits `PLATFORM_RESCORED`; the
  summary no longer claims stability when nothing was compared. `make supply_probe` additionally
  sweeps forfeits, questions the bot never forecast at all, which is how the round found that the
  triple era had lost six questions to delivery rather than one.

## The 2026-09-02 prompt and schema work (plan items A to E and section 10)

**Item D: post-hoc fields out of the structured block, lenient reads for what stays** (e74f20f,
c7926c2, cabb62d). Why: the block is written last, after the forecast is fixed, so a slot there
cannot scaffold reasoning. It can only be the forecast, transmit a decision the pipeline needs, or
echo prose for our post-hoc convenience, and the third kind was costing real forecasts.
`remaining_window_days` is gone entirely, field, validator, `_log_window_declared`, marker spec and
docs, having shipped the same day and never appeared in a prod block. `base_rate_anchor` and
`criteria_clauses` left the binary prompt and its example block: their only reader was
`tool_runner._anchor_and_clause_telemetry_lines`, which runs behind `PROBABILISTIC_TOOLS_ENABLED`,
pinned `'false'` in every prod workflow, hence zero harvested rows for both markers, so the helpers,
the comment markers and the marker specs went with them, while the Pydantic fields stay tolerant
because 49 and 12 published comments carry the keys and `performance_analysis` strict-parses them.
MC's `concentration` and `other_mass` left the prompt and now read leniently, which closes a live
bug: q45189's gemini wrote `"concentration": 0.0` beside a valid three-option ballot, the `> 0`
validator rejected the block, `json_repair` cannot alter valid JSON, and MC has no strip-and-retry,
so a good ballot fell all the way to the LLM salvage rung. `outcome_type` left the stacking numeric
prompt only (nothing reads the stacker's vote; the base numeric prompt keeps it), an unrecognised
spelling now reads as absent with a WARNING naming the raw value instead of taking the whole numeric
block down with the percentiles, and `_check_percentiles` accepts ties the way the extraction ladder
and `sanitize_percentiles` already did. Receipt: `scratch/schema_bloat_audit_2026-09-02.md` sections
2, 3, 4, 8 and 9.

**Item E: de-bloat the three base prompts and the gap-fill analyzer** (eb2ae6b, 558997a, db2748b,
ba6a7c2, 13b678f, 9403c18, 5695b3c). Why: forecast quality first. A rule is a shackle, so it stays
only if the pipeline requires it, it scaffolds the model's reasoning, or it corrects a measured
failure with no shorter form; each survivor is stated once, with its one-clause reason attached, as a
named constant or a named template step. The market clause renders only when the research carries
`## Prediction Market Snapshot`, so a backtest or a soft-failed provider no longer gets 1.5k
characters about a table it does not have, and what remains is policy (`_MARKET_READING_RULES`)
because the rendered legend already owns the notation. `_SOURCE_PROVENANCE_LADDER` names the source
tier tags instead of redefining all four tiers, since every artifact record since the tagging landed
carries them. The pre-open footgun is stated twice on purpose and lost only its third statement. The
trailing checklists are gone, with their two non-duplicate items moved into the template as a "Final
checks" step, because instructions that shape the structure of the answer belong in the numbered
template rather than in a post-hoc reminder. Numeric step 9b's `FORECASTABILITY: HIGH/MEDIUM/LOW`
output line, which nothing parsed, was redesigned rather than deleted, into a "Forecastability and
width" template step that also absorbs the calibration paragraph, keeping the even-handed wording the
2026-07 width audit settled on. The gap-fill analyzer lost its "0-2 real gaps, a few have 3-5" counts
(the analyzer fills every slot regardless, 55 to 77% of records sit at the cap, and "3-5" is stale
against `GAP_FILL_MAX_GAPS = 4`) and keeps "DO NOT invent gaps" with its reason, each gap being a
paid search. Every removal was traced through `git log -S` before it was made: the per-removal
receipts are `scratch/prompt_debloat_2026-09-02/receipts.md`, the audit behind them is
`scratch/prompt_bloat_audit_2026-09-02.md`, and the wording pass over the result is
`scratch/prompt_debloat_2026-09-02/wording_review.md`.

**Item B: two of the four 2026-09-01 rules reversed, keeping their corrections** (1af5ee5). Why: the
corrections were right and the constants were the third statement of them.
`_REMAINING_EXPOSURE_RULE` became the one-sentence `_REMAINING_EXPOSURE_SENTENCE`, which opens the
binary conditional-hazard bullet its first bullet had restated, and stands alone in MC, which has
neither a hazard bullet nor a union. `_ANCHOR_CONSISTENCY_RULE` is gone, with its 15-point size
carried into the single "Anchor on your math" bullet, because its first bullet was the fourth request
to state the outside-view number and its second priced at about zero across the whole archive while
suppressing good moves toward the resolution. `_LAST_REAL_USE_GAP_RULE` folded into the analyzer's
gap type 6 as a candidate clause, since a second "one gap MUST ask" mandate does not add spend, it
displaces other gaps. `_COUNT_IN_PERIOD_REFERENCE_CLASS` is kept verbatim.

**Item A: `_SOFT_CLOCK_RULE`, the miss class the plan is named for** (f07e3af, 966847f, 76f5eb1,
25059bc). A target date the responsible actor has not bound itself to, meaning no statute, no
contract, and no published schedule it has a measured record of meeting, is evidence that a target
exists, not that it will hold; price the probability that the target lands inside the question window
as its own number from that actor's record of slips and scrubs; and where a binding clock does exist,
compute the date from it and say which clock. It sits at the end of the binary and MC reference-class
step and carries its receipt in its own text. Why: the 2026-09-02 whole-archive failure-mode audit
found the shape on 52 of 815 strict records (6.4%, 8.3% of binaries, coder kappa 0.74); on the 37
flagged binaries the bot published a mean 0.44 for events that happened 3 times, 13 records above 0.5
resolved NO and none went the other way, and flagged records score 18.7 spot-peer points worse than
unflagged ones (95% CI 5.9 to 33.4). Soft targets without the decomposition score fine (+13.6) and
deadline questions are calibrated in general, so the rule names the move rather than the question
shape, and the "measured record of meeting" carve-out is load-bearing (on q42305 a weekly bulletin
with a measured publication lag was a binding clock in practice, and a near-1 timing term was right).
Receipts q43837, q44424 and q44557, against the contrast q45217, where a statutory clock existed, the
members computed the date and scored +45; audit at
`scratch/failure_mode_audit_2026-09-02/AUDIT_SYNTHESIS.md` (lens A) with `STATS.md` beside it. There
is deliberately no structured-block field for it: the schema audit rejected a
`target_holds_probability` slot for the same reason Item D removed the others, so the number lives in
the rationale.

**Item C: `_HISTORY_DISCHARGED_RULE`, shipped on the plan's recommendation with the operator's final
say pending** (f07e3af, 966847f). If your own analysis names a reason the historical cadence has been
discharged, its driver was met, the deadline passed, the rule changed, then that cadence is a bound
on your estimate rather than its center, and the post-change estimate has to be stated with what it
rests on. Why: the same audit's lens C found the pattern on 12.1% of coded rationales, worth about 7
spot-peer points per flagged record (95% CI 2.7 to 12.2), with the old cadence failing in 83% of
fires and in 13 of 13 on the live triple. Those are upper bounds: coder agreement was 0.59 and the
label is partly hindsight-contaminated, which is why the plan left this one open (its section 6). The
rule is conditional on the member's own written acknowledgment so it cannot fire where nothing
changed, and it only became consistent once `_ANCHOR_CONSISTENCY_RULE`'s "do not move off your number
when history counsels caution" was gone, since the two pulled opposite ways. It is one constant and
one test class, so declining it means deleting both.

**Section 10 housekeeping** (63d96bc, 24d7e9d, 58bd537, 9b21a1b, 473659a, f5a89a1, 40be0d4,
8490f05). Eight pieces of FUTURE.md bookkeeping that the last two residual rounds recommended and
nobody applied, one of them surviving three rounds, are now written with their receipt paths: the
triple-era peer-gap flag recorded as retired at n=20 and re-confirmed at n=30, the spread gate's
"numeric-only lever" corrected to numeric-dominant with the 2026-09-01 recount, the mixture branch's
"removed after zero prod fires" justification corrected for the third time, and the anchor-overshoot
self-consistency screen recorded as measured and rejected. Then the forge report-only items worth
doing while those files were open: `tests/test_resolution_source_provider.py`, at 2,054 lines the
largest test file in the repo, split into a `tests/resolution_source/` package along the three layers
its own docstring already named; the third copy of the fixed-point number formatter given one home in
`research/number_format.py`; three near-identical binary-post fixture builders collapsed into one;
`RESCORE_ATOL` de-duplicated between `rescore_diff.py` and `collector.py`; and the supply probe's
forfeit sweep now saying where it is during its detail GETs.

**Sizes landed above the plan's targets, deliberately.** Measured by rendering each prompt on one
fixture question whose research carries the market header, then collapsing all whitespace: the true
pre-bundle baseline at 7e7d449 is binary 19,396, MC 14,184, numeric 18,983 and analyzer 4,693, and
the branch tip is 16,095 / 11,895 / 14,805 / 4,023, a net cut of 17.0% / 16.1% / 22.0% / 14.3% (the
binary and MC figures moved by 35 and 29 characters when the forge pass reworded the
history-discharged receipt). The
plan's targets were about 14,300 / 10,700 / 13,700 / 3,800, so what shipped is 12% / 11% / 8% / 6%
over them. The reason is the operator's own decisions in plan section 7.2, which kept the step-5b
"three valid moves" reconciliation paragraph and the meta-justification sentences that make the
prompt explain itself, together with principle 3's requirement that every surviving rule carry its
one-clause reason, plus the plan's own verbatim Item A wording measuring 648 whitespace-collapsed
characters rather than the 450 it estimated (`_HISTORY_DISCHARGED_RULE` is 366). `docs/prompts.md`
carries the same numbers and the measurement method, including one correction: an earlier
draft of that paragraph quoted 18,640 / 13,820 / 18,446 as the baseline, which came from an
intermediate post-Item-D commit and understated it by roughly the Item D schema-field text.

**One era boundary.** Prompt text and structured-block shape both shift the forecast distribution,
and prod runs from `main`, so the boundary is this merge commit's timestamp and everything in the
branch crosses it together. That is the accepted cost: nothing in these two blocks is separately
datable afterwards, so no score or width shift across the boundary can be attributed to any single
item in it.

**Effect measurement, and when.** None of this is measurable offline. The first read is the residual
round after the fall cup's first resolutions: re-code rationales with the audit's lens definitions
(the bullet list at the top of `AUDIT_SYNTHESIS.md`), then compare the announced-but-unbound shape's
incidence and spot peer on live-triple records against the 6.4% and -18.7 baseline, and the
history-repeats rate against 12.1%. Zero of the 30 resolved live-triple records carry the announced
shape yet, so Item A is pre-emptive on the current roster and the first fall-cup cohort is the
earliest honest test of it.

**Operator decisions this block adds** to the list above: whether to keep Item C
(`_HISTORY_DISCHARGED_RULE`), reversible by deleting the constant, its two interpolation sites and
only the history-discharged cases in the test class it shares with the approved `_SOFT_CLOCK_RULE`
(`docs/prompts.md` carries the recipe); and the same merge-timing decision, which now covers the prompt and
schema changes as well.

**`/forge` ran on the full 7e7d449..8cd2295 diff and its findings are applied.** It returned six FIX
findings and twenty report-only ones, no criticals, and nothing that would have published a wrong
forecast. The six: F1, the multiple-choice prompt still asked for a probability "(1-99%)" and "at
least 1%" four lines above the JSON block that demands decimals summing to 1.0, which is a receipted
production failure because a percent-scale `option_probs` is hard-rejected, cannot be repaired, and
drops the whole ballot to the paid LLM salvage rung (q44558); F2, the history-discharged rule handed
forecasters "held in 0 of 13 recent cases" as a flat fact when the label behind it had coder
agreement 0.59 and 0 of 13 is an ordinary draw at the 17-18% held rate the older eras show, so the
count now names its own sample; F3, FUTURE.md described Item C as live with no open-decision marker
and AGENTS.md's revert recipe would have deleted coverage for the approved soft-clock rule; F4, the
numeric tie tolerance had no test on the block rung it exists to serve, and re-tightening that rung
left the whole suite green while every tie-carrying block silently started paying for salvage; F5,
the gap-fill v2 driver template's comment claimed prod emits the market header on every question
when the measured rate is 59 of the 60 newest archived records; F6, four test classes each carried
their own copy of the same prompt-render helpers. Sixteen of the twenty report-only items were also
taken, including a dedicated test module for the shared number formatters and their four stale
references, three lenient-validator contract fixes in `structured_output_schema.py` (a WARNING that
ignored the `log_failures` suppression, an `OverflowError` that escaped the "unusable reads as
absent" contract, and an accepted-value set restated below the `Literal` it duplicates), the
prompt-size waiver now recorded in `docs/prompts.md` (it sat in AGENTS.md until the 2026-09-03
de-bloat), and two test-file splits. R5, R6, R15 and R17 were
left as they are with the reason recorded in the code: two are prompt-wording calls that shift the
forecast distribution and so are the operator's, one would add a parameter for a single caller, and
one would add a second gating condition whose false negative would silently drop the market policy
from every prompt that does have a table.

**Free gates at the end of that block (d38980b).** `make lint` clean, `make typecheck` 0 errors,
`make lint_imports` 6 of 6 contracts, `make deps` clean, `make test` 6815 passed / 22 skipped / 5
deselected in 145 s. No paid call was made at any point. The commit range is 7e7d449..d38980b.

## The 2026-09-03 work: fetch ladder, docs de-bloat, fall cup, credit alerting, Gemini models

Sixty-three more commits landed after d38980b, all on 2026-09-03. The last code merge of the day
was dd1074b; the two commits above it (9e1a66a, 16ca9ab) only update the tracked handoff document
`scratch_docs_and_planning/HANDOFF_2026-09-03_fetch_ladder_wrapup.md`, so the branch tip was 16ca9ab
when this section was first written. The evening's review and fix wave and the 2026-09-04 work (both
below) carried it to 1f2b504, and the Codex review triage carried it to c07d7cf. Against d38980b the day
changed 104 files (+21,604 / -1,676 lines); against `main` the whole PR at 001b6f9 is 366 commits
and 247 files (+58,570 / -6,362).
`resids-sept1` is still unmerged (10 ahead of `main`, 0 behind) and is an ancestor of this branch, so
the branch-base decision above stands. The pushed head on GitHub is 1f2b504, where PR CI is green
(run 33917127083).

The blocks below are in landing order, except the smoke evidence, which comes first because
everything after it is what the second smoke run (in "The 2026-09-04 work") had to cover.

### Smoke evidence so far: Test Bot run 67

The operator dispatched `test_bot.yaml` (workflow "Test Bot") at 15:58 UTC on 2026-09-03 as run
33775800806, on 39877f1. It forecast four questions in `--mode test_questions`, concluded success,
and was checked on three axes: operational markers, forecast content and research content. It is the
smoke evidence for everything on the branch up to and including the clip-threshold commit below; the
fetch ladder, the cup, the credit and the model changes all landed after it.

Two spend facts from its log. The donated key (`OAI_ANTH_OPENROUTER_KEY`) reported
`run_delta_usd=0.00 remaining=0.00`: it was empty at the time (the $1,500 grant arrived later that
day), so the whole run billed the personal key. The personal key's `CREDIT_SPEND` read
`run_delta_usd=1.83` tagged `usage_delta_unsettled`, which the marker itself says is a lower bound.
The new per-role ledger (item 16) did populate: eleven `CREDIT_ROLE_SPEND` lines, every one
`key=personal`, whose `usd=` column sums to $10.50 and whose `byok_usd=` column sums to $8.67. Those
figures do not reconcile yet. For `forecaster:openai` the two columns are equal (1.2016 each) while
for `forecaster:anthropic` `usd=` is exactly twice `byok_usd=` (2.3673 against 1.1837), so OpenRouter
reports the routes differently per provider, and the run's true personal-key cost sits somewhere
between $1.83 and $10.50 until `scripts/reconcile_credit_spend.py --roles` is run against a settled
figure. Item 16's donated-key question was settled by the 2026-09-04 smoke run: its ten donated-key `CREDIT_ROLE_SPEND` lines carry `byok_usd=` equal to `usd=` (for example `gap_fill_resolver`, 3.7737 in both columns), so the donated key's BYOK routes do populate `upstream_inference_cost`. The personal-key doubling persists there for `forecaster:google` (1.1433 against 0.5716) and stays unreconciled.

That further `test_bot.yaml` dispatch was made on 2026-09-04, after the push, as Test Bot run
33907102246 on cbc26bf, and was checked the way run 67 was plus the new markers. It is written up in
"The 2026-09-04 work" below; run 67's figures stay here as the baseline that run was compared
against.

### The clip-threshold sweep and the MEMBER_FORECAST marker (39877f1)

Committed the morning of 2026-09-03 and inside run 67's head, this is the last piece of the
2026-09-01 residual round. `performance_analysis/clip_threshold*.py` is a standing residual
dimension: it reprices every resolved binary and multiple-choice publish under a grid of candidate
floors (floor-only, ceiling-only, symmetric) over nested lookback windows and disjoint era slices,
priced in spot peer through `spot_peer_delta`. A candidate looser than the clamp in force is censored
(the raw member value is gone) and is reported as a bracket, never a point estimate; each window
carries an out-of-sample carry test and an out-of-bag argmax; an infeasible MC floor (n times c above
1) is disclosed rather than rendered. Result on the strict cohort (447 binary, 97 MC): the live clamp
has bound no binary since 2026-05-18, raising the floor loses in every window and era, an MC floor is
a tax on every question, and loosening is bounded at a few pre-flip points, so neither floor moves.
FUTURE.md carries the dated result and the two open operator decisions.

The censoring finding is why the same commit adds
`MEMBER_FORECAST: question= model= role=member|stacker qtype= raw=<json> published=<json>`: raw
pre-clamp member values had been recoverable only from the middle-trimmed published comment (74 of
451 resolved binaries), because the runners logged the clamped value with no model name. Every runner
and stacker path now logs one line per forecast value and `scripts/telemetry/markers.py` harvests it
verbatim. `extract_mc` now returns the declared vector beside the option list so the raw MC ballot is
recoverable too, the normalize-match-sum logic lives once in
`mc_processing.accumulate_declared_option_probs`, and `clamp_and_renormalize_probs` gains
keyword-only bounds with a strict degenerate guard (ten options at 0.10 is a valid in-bounds answer;
the old test sent it down the sub-floor fallback). Fixing the test doubles exposed that the MC
stacker end-to-end test had been falling through to MEDIAN and passing; all four stacking end-to-end
tests now pin the primary stacker path. Shared helpers spun out along the way:
`performance_analysis/markdown.py`, `metaculus_bot/bootstrap.py` (one seeded resampler) and
`analysis.jeffreys_ci`.

### Fetch ladder: what the resolution-source fetcher and gap-fill v2 now do

Two fetch paths were reworked. "Tier-1" is the resolution-source fetcher
(`metaculus_bot/research/resolution_source.py`), which fetches the URL a question names as its
resolution source and renders it to the forecasters as primary grading evidence. "v2" is the gap-fill
agentic research loop (`metaculus_bot/research/agentic/`), whose driver model has `fetch` and
`read_document` tools. Plan, evidence table and the operator's inline decisions:
`scratch_docs_and_planning/fetch_ladder_plan_2026-09-03.md`.

The finding that reshaped the design: the archived 403s from Akamai-fronted federal hosts (bls.gov,
cdc.gov, fsis.usda.gov) do not reproduce from the laptop or an EC2 box with the bot's own client;
only a GitHub Actions runner gets them. So the ladder is deterministic and free first, with the one
paid rung last and behind its own flag, and the TLS-impersonation rung the earlier design led with is
not built until a free diagnostic run from the runner says it would help (below).

**Phase 1** (four merges, the last 3a541ee). `resolution_url_scan.py` keeps parentheses that belong to
a cited URL. New `metaculus_bot/research/document_text.py` reads PDFs locally with pypdf under page
and time caps (`DOCUMENT_TEXT_MAX_PAGES` 400, `DOCUMENT_TEXT_MAX_SECONDS` 20,
`DOCUMENT_TEXT_PDF_MAX_BYTES` 40 MB), walks the outline iteratively so a pathological bookmark tree
cannot raise, and selects passages by BM25 into a digest with a truncation note. The measurement
behind it is a 6.7 MB, 220-page PDF that pypdf read in 5.3 s (833,450 characters, the wanted passage
inside) where the paid Gemini `url_context` read of the same file returned nothing. Both native
google-genai clients (`gemini_client_config.py`) get `HttpRetryOptions` and an explicit thinking
level, and every call logs `GEMINI_USAGE: role= model= ... search_queries=` (`gemini_usage.py`), the
first per-call accounting of the Google AI Studio side of a run.

**Phase 2** (99d3438). Tier-1 gains a meta-refresh hop, an ARIA `role="table"` rewrite into real
tables, a local PDF digest (`route=pdf_local`) and the `unreadable_document` status; every
`RESOLUTION_SOURCE_FETCH` line carries `route=` when a rung other than `direct` produced the outcome,
each rung that fired adds one `RESOLUTION_SOURCE_ESCALATION` line, and the per-host politeness gate in
`http_fetch.py` is loop-scoped and shared. v2's `read_document` becomes acquisition-first: a document
the loop already holds is digested locally (`method="digest_local"`, marker
`AGENTIC_FETCH_LOCAL_DOC`) and a PDF is read local-first with pagination; a held document estimated
above `URL_CONTEXT_SIZE_GATE_TOKENS` (100,000) is never sent to the paid reader (the nine archived
documents above that bound carried 67% of all reader tokens); the Chromium rung waits for
`domcontentloaded` and salvages the DOM on a navigation timeout.

**Review and live QA fix wave** (06ded11, a98a2b2, 15d6c04, c450a86). A review of the Phase 1 and 2
diff produced 36 findings marked fix, and an independent live QA pass over 62 archived URLs at
99d3438 (`/tmp/fetchprobe/qa_report.md`, laptop-local, no paid call, every provider key blanked in the
process environment) found 12 defects, the top one HIGH. All were worked through in those four
merges. The largest single change is dropping `favor_precision=True` from the trafilatura extractor:
re-fetching 149 archived HTML URLs live, default recall gained text on 85 pages and lost it on none
(the two shorter outputs were a 404 page and 122 characters of whitespace), and 10 pages crossed the
400-character chrome floor upward, 9 of them carrying the resolving content; the floor itself is
unmoved. The rest: the aiohttp session pins its CA store to certifi's bundle (trade.gov failed
verification against the default store) and raises its header caps to 65,536 bytes (two hosts serve a
CSP header larger than aiohttp's 8,190-byte default); a shared two-slot PDF-parse gate
(`http_fetch.pdf_parse_semaphore()`) bounds concurrent parses across both paths, with the parse moved
out of the per-host gate hold; hop timeouts are clamped to the remaining wall; three new reason
tokens (`no_matching_passage`, `budget_skipped`, `parse_contention`) keep `success` meaning content;
pypdf's decoded-stream cap is 8,000,000 bytes per stream; BM25 runs with `b=0` (length
normalisation off, so a one-line stamp cannot outrank the paragraph that carries the fact) over a
query stopword list; v2 falls through digest-first, refuses to render nothing, and hides its
`ladder_exhausted` argument from the driver-facing schema. The QA pass's headline at 99d3438: of the
47 archived failure URLs, 20 now return `success`; across all 62 URLs, 23 improved, 37 unchanged, 2
regressed; largest single-URL wall time 6.63 s (the 220-page PDF).

The same wave ships the robots pre-check for the paid reader. Gemini's `url_context` honours the
`Google-Extended` robots token, and `scripts/probes/gemini_verify.py` proved it live (three billed
calls on the operator's key: the robots-allowed control retrieved, `internationalaisafetyreport.org`
returned `URL_RETRIEVAL_STATUS_ERROR`). So before every paid `url_context` read, `robots_policy.py`
fetches the host's robots.txt through the SSRF-guarded plain fetch (cached per host), and a
disallowed URL is skipped with status `robots_disallowed` and the `AGENTIC_URLCONTEXT_ROBOTS_SKIP`
marker instead of spending on a read that cannot retrieve.

**Phase 3** (dd1074b, eight commits). Headless Chromium moves into shared
`metaculus_bot/research/rendered_fetch.py` (v2's `tools._try_rendered_fetch` is now a thin wrapper),
and Tier-1 gains four rungs. The rendered rung (`route=rendered`, floor
`RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S` 12 s, new reason `renderer_unavailable`) renders a `js_wall`
page. The derived-API rung (`research/derived_api.py`, `route=derived_api`, floor 3 s) serves the JSON
feed a rendered dashboard fetched for itself over XHR; measured on six dashboards, grepping the HTML
found one candidate and it was a Maps key, three of four hand-guessed endpoints were wrong, and
recording the page's own XHR found a working unauthenticated endpoint for all six. The endpoint is
remembered per host for the run, and a reuse on a different page says so in a coverage disclosure.
The Wayback rung (`research/wayback.py`, `route=wayback`, floor 8 s) fetches
`web.archive.org/web/2026id_/<url>` when the host refuses our address, reads the 14-digit capture
stamp off the redirect, and is admissible only clearly marked stale: an as-of line always, withheld
past `RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS` (30) under the existing `stale_data` token, at most
`RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS` (2) per question, never for `js_wall` (the archive stores the
unrendered shell), with the Metaculus and SSRF checks re-run on the unwrapped target; a capture the
archive never served declines and leaves the direct status standing. The url_context rung
(`research/url_context_reader.py`, `route=url_context`, one attempt, 15 s floor) is the only paid rung
and sits behind `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED`, which defaults off in code and, since 06b3fd9
on 2026-09-04, is set to `'true'` in every bot workflow (see "The 2026-09-04 work" below); it adds the
terminal status `ungrounded` (the reader answered without retrieving) and a third `GEMINI_USAGE`
role, `resolution_source`. The per-question cap on paid reads,
`RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS` (2), landed in the telemetry follow-up that evening
(below). Every non-direct route present in a question's
snapshot renders one caveat sentence under the primary-grading-evidence caption, from `ROUTE_CAVEATS`
in `resolution_fetch_result.py`; a question served entirely by the direct route renders
byte-identically to before, pinned by test. The route vocabulary is `direct`, `meta_refresh`,
`impersonate`, `pdf_local`, `derived_api`, `rendered`, `wayback`, `url_context`; `impersonate` is
reserved in the `Literal` and the caveat map and nothing emits it. All new tokens harvest with no
marker-regex change.

**What the flag being on means.** `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED=true` makes the
resolution-source provider a paid Gemini surface: a url_context call is a model round-trip on the
operator's personal Google AI Studio key, and what comes back is an answer about the page rather than
the page itself. At dd1074b the flag was set in no workflow, the cost-gate list in AGENTS.md counted
the provider as free, and the greppable lines on that path were deliberately left unregistered until
the flag was ever turned on. Since 06b3fd9 (2026-09-04) the flag is on in every bot workflow, AGENTS.md
and `.env.template` say the provider is paid, and all three lines on that path
(`RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP`, `RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED`,
`RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED`) are registered marker specs. The spend is bounded by the
trigger population (only a page no free rung could read reaches the paid rung), the free
`Google-Extended` robots pre-check, the 15 s budget floor and the per-question cap of 2 paid reads.

**Deliberately not built.** TLS impersonation (curl_cffi) is not in the ladder: the 403s do not
reproduce off the runner, so the free `fetch_diagnostic.yaml` run after merge decides it (below);
`curl-cffi` is a dev-group dependency since f02672e (so the test suite's egress guard can import it
unconditionally) and is not a runtime dependency of the bot; deptry carries it under `DEP004` beside
matplotlib. An egress change (an HTTP proxy on the operator's EC2 box, or a
self-hosted runner) is parked in FUTURE.md at low priority by operator decision. The 45 s Tier-1 wall
is not raised (FUTURE.md item 5, skipped by operator decision); every rung self-bounds inside it on
the Datawrapper pattern, wall minus elapsed minus a margin, skipped below its own floor. No
residential IPs. The two DataDome / Cloudflare-challenge hosts among the 47 archived failures
(sagaftra.org, trueup.io) try url_context when the flag is on and are otherwise accepted as lost.

**Gemini models** (c670015). Both native google-genai surfaces run `gemini-3.8-flash`:
`GEMINI_SEARCH_DEFAULT_MODEL` (grounded search, thinking `medium`) and `GAP_FILL_V2_READER_MODEL`
(the v2 reader, thinking `low`), each overridable by its env var. `scripts/probes/gemini_verify.py`
verified the id live on grounding, `thinking_level` and url_context. The thinking levels are pinned
explicitly so a model swap never changes reasoning effort silently, and the lite tiers were rejected
for the reader because its output enters the `fetched` provenance tier, the only tier that reaches
the SUPERSEDE block. The OpenRouter forecaster roster in `metaculus_bot/llm_configs.py` is untouched
by this work (last changed 2026-09-01).

**A containment follow-up, real and open as of dd1074b.** The test suite's egress guard
(`tests/conftest.py`, which patches `socket.socket.connect`) covers neither the Chromium subprocess
nor libcurl, and three pre-existing marker tests launched a real browser when the Tier-1 rendered rung
landed. `tests/resolution_source/conftest.py` now declines the browser and the Wayback rungs by
default through autouse fixtures; widening the global guard is the proper fix, and bd10996 did it the
same evening (see "Review and live re-QA of the Phase 3 diff" below).

**Docs for Phase 3 are also in progress as of 16ca9ab.** The Phase 1 and 2 modules, statuses and
markers are documented; the Phase 3 modules (`rendered_fetch`, `derived_api`, `wayback`,
`url_context_reader`, `robots_policy` moved out of `agentic/`), the `ungrounded` and
`renderer_unavailable` tokens, the new `RESOLUTION_SOURCE_*` budget constants, the flag and its
cost-gate consequence still needed their lines in AGENTS.md and `docs/research.md` at that point. They
landed the same evening in a1ff0f7 (merged 19dd8e1, with small fixes c2f8812 and 7830d10), and 06b3fd9
revised them again on 2026-09-04 when the flag went on.

### AGENTS.md de-bloat (6716c53, loss-checked in c87a5be)

AGENTS.md (which `CLAUDE.md` symlinks) went from 178,864 bytes to 26,814 and is now a terse starting
point: the cost gate, the repo overrides, the layout, the pipeline outline, the standing rules whose
violation is silent and expensive, and an index of the docs. The narrative moved into `docs/`: four
new guides, `docs/performance_analysis.md`, `docs/prompts.md`, `docs/roster_history.md` and
`docs/value_extraction.md`, with `docs/numeric_pipeline.md`, `docs/research.md`,
`docs/architecture.md`, `docs/operations.md` and `docs/agentic_gap_fill.md` extended. An independent
loss check against the pre-rewrite file (token, sentence and section coverage passes over AGENTS.md
plus `docs/`) found seven facts with no home anywhere, and c87a5be restored each in the doc that owns
the subject: the primary stacker's `STACKER_SOFT_DEADLINE` and entry points, the crux and
targeted-search settings, `provenance._METHOD_TO_TIER`, the DEP004 matplotlib exemption, the pre-push
hook's exact command and the precommit targets, and the osv-scanner action with the
repo-relative-path idiom. The report is
`scratch/agents_md_debloat_2026-09-03/loss_check_independent.md` (gitignored, laptop-local).

### Fall Metaculus Cup configured (ea1d558)

Metaculus now rejects the undated `metaculus-cup` slug (the posts list answers HTTP 400 for
`tournaments=metaculus-cup`, verified 2026-09-03), so `METACULUS_CUP_ID` carries the season's dated
slug, `metaculus-cup-fall-2026`: project 33108, "Metaculus Cup Fall 2026", start 2026-08-28,
`forecasting_end_date` 2027-01-01, `score_type` `peer_tournament`, and zero questions on 2026-09-03.
`run_bot_on_metaculus_cup.yaml` is at parity with the tournament workflow (the diff is the name, the
cron minutes and `--mode metaculus_cup`) and runs hourly at :13, :33 and :53, staggered off the
tournament's :03/:23/:43; a run that finds no new question makes no LLM call. `cli.persisted_tournament_id`
labels archive records by run mode so cup and tournament records stay separable, and the fall-cup time
bomb is discharged (`FALL_CUP_CONFIGURED = True`; the companion test now pins that the cup stays
configured). For residual analysis: `peer_tournament` scoring means cup records carry a coverage-scaled
peer score and no spot peer; `platform_scores.RankingScore.tier` already keeps the two apart.

The QA dispatch of that workflow on the branch (run 33815141451, 22:53 UTC on ea1d558) retrieved 0
questions from `metaculus-cup-fall-2026`, made no model call, logged
`CREDIT_SPEND: key=donated run_delta_usd=0.00 remaining=1449.19` and
`key=personal run_delta_usd=0.00`, and exited clean. The workflow is `disabled_manually` on GitHub
again and must stay so until this merges: its schedule runs from `main`, whose `constants.py` still
holds the dead `metaculus-cup` slug, so enabling it now would produce hourly red runs. Minibench stays
disabled by design.

### Credit alerting back on (0cd5e36)

Metaculus granted $1,500 of donated OpenRouter credits on 2026-09-03, so `CREDIT_ALERT_RESUME_DATE`
moves up from 2026-09-10 to 2026-09-03 and a credit shortfall reddens CI again.
`OPENROUTER_CREDIT_FLOOR_USD` goes from $1 to $100: only Metaculus can refill this key, so the warning
has to arrive with runway left to ask for a top-up (roughly 250 questions at the measured $0.38 to
$0.41 each), where $1 fired only once the key was already dry. `CREDIT_FLOOR_BREACH` now reads as that
early warning. The suppression mechanism is unchanged and re-armable by pushing the resume date
forward, in `constants.py` or through `OPENROUTER_CREDIT_ALERT_RESUME_DATE`; tests cover both branches
with injected dates and pin that the shipped default plus the real clock exit non-zero on a breach.

### Operator-run diagnostics (18bc926) and the tornado bump (98f5876)

`.github/workflows/fetch_diagnostic.yaml` is `workflow_dispatch`-only and passes no secret at all, so
it is structurally incapable of spending or publishing. It runs `scripts/probes/fetch_diagnostic.py`
from a GitHub runner: three probes per URL (the bot's real client, the same GET under curl_cffi Chrome
impersonation, and the Wayback copy with its age) and one table plus a verdict. From the laptop only
congress.gov splits, the four federal hosts serve 200 on both rungs, and Wayback answers for 8 of 10.
The job passes `uv run --with curl_cffi` because it syncs `--no-dev` and `curl-cffi` is a dev-group
dependency (f02672e).
`scripts/probes/gemini_verify.py` makes three billed calls on the operator's key, prints a cost
estimate, and refuses without `--i-accept-spend`; a test pins both the refusal and the exact call
count.

tornado 6.5.7 to 6.5.8 clears GHSA-8423-8fgw-73vq, GHSA-mpf4-983q-p7j4 and GHSA-wwv5-g3v4-889x, which
had turned the PR's audit job red; `make audit` reports no issues and the branch CI is green at the
pushed ea1d558.

### Telemetry

`scripts/telemetry/markers.py` went from 51 specs at d38980b to 56: `member_forecast`,
`gemini_usage`, `resolution_source_escalation`, `agentic_fetch_local_doc` and
`agentic_urlcontext_robots_skip`, plus the optional `route=` group on the existing
`resolution_source_fetch`. Adding a token is fine and renaming one is a breaking telemetry change;
`direct` is the one route value that does not ride the marker, so every archived line stays
byte-identical. The 2026-09-04 flag flip registered three more (the `resolution_source_urlcontext_*`
specs, below), so the registry stood at 59 at 1f2b504 and stands at 60 at c07d7cf, after the Codex triage
registered `litellm_callback_drain_timeout`.

### Gates at dd1074b

`make lint` clean, `make typecheck` 0 errors, `make lint_imports` 6 of 6 contracts, `make deps`
clean, `make test_fast` 7,291 passed / 14 skipped / 33 deselected in 150.93 s, exit 0. That gate ran
on the merged code; the two handoff-doc commits above it change no code.

### Post-merge operator steps

1. Enable the cup workflow: `gh workflow enable "Forecast on Metaculus Cup" --repo No-Stream/metaculus-bot`.
   This turns on the hourly :13/:33/:53 crons against `metaculus-cup-fall-2026`; do it only after
   `main` carries this merge.
2. Run the free egress diagnostic: `gh workflow run fetch_diagnostic.yaml --repo No-Stream/metaculus-bot`
   (the yaml has to be on `main` first). Read the table in the job log: rows 1 to 4 are the Akamai
   federal hosts. If the bot-client column says 403 and the impersonated column 200, the runner's TLS
   fingerprint is being scored and the TLS-impersonation rung is worth building; if both say 403, the
   runner's egress IP is what is blocked and the rung is dropped for good.
3. When Metaculus publishes the fall bot tournament (no object existed on 2026-09-03: the tournaments
   list had 193 projects with none among them, project ids 33100 to 33140 hold only the cup and an
   unrelated series, and four plausible slugs 404), set `TOURNAMENT_ID` and `TOURNAMENT_END_DATE` in
   `metaculus_bot/constants.py` from the object's slug and its `forecasting_end_date` (not
   `close_date`), per the season-start checklist in `docs/operations.md`. From 2026-09-20 (the summer
   end date 2026-09-06 plus the two-week hard stop) `check_tournament_dates` raises
   `TournamentExpiredError`, which reddens the tournament crons and `tests/test_tournament_dates.py` on
   purpose as the reminder; the cup mode never calls that check. `make supply_probe` (free) is the
   watch.
4. Clean up the local worktree branches (blocked for agents):
   `git branch -d $(git branch --list 'worktree-agent-*' --merged)` removes the 65 merged ones, and
   `git branch -D worktree-agent-aed63df23d441c8e4` removes the one unmerged leftover, whose single
   commit is an older revision of a plan-status section the bundle already carries in d38980b.

### Operator decisions this block adds

Whether to ever turn on `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED`: settled on 2026-09-04 by the
operator's standing rule that a flagged feature ships with its flag on, so it is on in every bot
workflow (06b3fd9) and the resolution-source provider is a paid surface; whether to build the
TLS-impersonation rung, still decided by the diagnostic above; and the reconciliation of run 67's
personal-key spend, superseded by the second smoke run's $10.00 in role lines (below).

### Review and live re-QA of the Phase 3 diff

Phase 3 (the rendered, derived-API, Wayback and url_context rungs, merged at dd1074b) then went
through a live QA pass and a code review pass on the evening of 2026-09-03, and the fixes from both
landed in the commits from 9c6071e to 55e02f5. No paid call was made at any point: every provider key
was blanked in the process environment for the QA pass, and the paid rung's flag stayed off.

**The live QA pass** re-ran `fetch_resolution_sources` over 106 URLs (all 97 distinct resolution-source
URLs in the research archive plus 9 from the 2026-09-03 prod run). 93 of the 106 succeed; the archived
corpus goes from 59 of 97 successes to 85 of 97, and 35 of the 47 archived failures now return content
(20 did after Phases 1 and 2). Routes over the 106 were direct 76, wayback 13, rendered 13, pdf_local
3 and meta_refresh 1; the telemetry invariants held on every line, and all 48 questions served
entirely by the direct route rendered byte-identically to before. Two defects were rated high. The
rendered rung had no bound of its own on a page that keeps navigating (one page took 76 s against the
provider's 45 s wall), so 9811ca3 caps the DOM read, bounds the whole render at the remaining wall
budget, and records a cut-off render as `render_timeout` rather than as the renderer being
unavailable. And the earlier switch to recall-first extraction, which had gained text on 85 pages,
turned out to publish navigation chrome on chrome-heavy pages and on congress.gov to replace the
bill-status card. A calibration study (`scratch/fetch_ladder_2026-09-03/chrome_calibration.md`,
gitignored) re-fetched 118 bodies, ran five extractor variants on identical bytes, labelled the
results by hand and scored five publish policies. The winner, in 162294e, keeps recall-first
extraction and scores the extracted text by line shape: the share of characters in table rows or in
non-table lines of at least `RESOLUTION_SOURCE_CONTENT_LINE_MIN_CHARS` (60) characters must reach
`RESOLUTION_SOURCE_CONTENT_SHARE_MIN` (0.38); below that the page is re-extracted precision-first and
published only if that clears both the floor and the metric, and otherwise withheld as
`no_resolving_content` with reason `thin_page`. On the labelled corpus this publishes 46 of 46
content bodies and 2 chrome bodies and withholds no content, against 43, 11 and 0 for the code as
shipped and 40, 4 and 5 for a plain revert. It still publishes prose-shaped boilerplate (a
cookie-consent wall, a glossary), it withholds one Korean agency homepage whose news ticker carried
its question's resolving fact, and the threshold has about 0.05 of margin on each side on a one-day
corpus. The two decisions ride `details["counts"]` as `chrome_metric_withholds` and
`precision_fallback_rescues` so a later re-calibration has data. The medium defects were fixed in
292d306 (Wayback replays arrive brotli- or zstd-encoded, so `brotli` and `backports-zstd` are now
declared dependencies, and a zero-passage PDF digest is `no_resolving_content` rather than `success`)
or are limitations of the rungs as designed (the derived-API rung cannot reach a dashboard whose feed
is cross-origin, by the same-origin rule).

**The code review pass** over the Phase 3 diff returned no critical findings, 24 findings marked fix
(17 important, 7 minor) and 29 more surfaced for the record. The fixes landed in four groups. The
Wayback request URL had hardcoded the year 2026, which from 2027-01-01 would have returned only
end-of-2026 captures that the 30-day freshness bound then withholds; f98f7b8 derives the year from
the fetch's own clock (the same fix is in `scripts/probes/fetch_diagnostic.py`). The ladder core
(twelve commits ending fbb6aa1) gives the auxiliary fetches their own bookkeeping so an archived PDF
capture no longer hijacks the route stamp, lets the paid rung stay reachable past a stale Wayback
capture, gates reuse of a remembered JSON feed on a real JSON content type, unwraps nested Wayback
captures and re-checks the innermost URL against the Metaculus self-reference guard, re-reads the paid
rung's budget after a robots.txt pre-check that is now bounded at 5 s, serialises a question's
same-host browser escalations so the second URL reuses the harvested feed, threads the time-budget
fast path into the provider so a thin-window question declines the render and the paid read (counted
as `fast_path_skips`), computes the route caveats over the sections that actually render, fills the
shared robots.txt cache single-flight, and breaks the budget skips out per rung. The browser
transport (ea6d11d) recomputes the navigation budget after both gate acquires so a render admitted
late navigates on what is actually left or declines before a launch, scopes the rendered-to-nothing
memo per caller, owns and drains the harvest handlers, harvests sibling subdomains through the
public-suffix rule, caps the DOM size, blocks service workers, fails closed on a non-ASCII host, and
carries the main frame's HTTP status so a browser-targeted 403 leaves the direct result standing
instead of publishing as a success. The test pins (78c7e6d) cover the paid rung's timeout and attempt
bounds, the SSRF exclusion on both third-party rungs, and the reader prompt's load-bearing
instructions. Separately, bd10996 widens the test suite's egress guard beyond Python sockets: an
autouse fixture now refuses Playwright browser launches and curl_cffi requests and fails the test at
teardown if a refusal was swallowed, which a proof run showed nine resolution-source tests would
otherwise have needed.

Two review findings were left as they are on purpose: cross-page reuse of a remembered JSON feed is
what the design specified and is disclosed twice, and the Wayback attempt cap being per question while
the gate it contends for is loop-wide is the open FUTURE.md item 5. Four are deferred to their own
pull request: the 3,600-line `tests/test_agentic_tools.py`, the 1,400-line `constants.py`, a shared
fake Playwright graph for the tests, and the fact that the Tier-1 paid rung is configured by the
`GAP_FILL_V2_READER_*` constants. A telemetry and structure follow-up merged as aca5cd8 (twelve
commits, the last the split of the 1,484-line escalation test module): per-rung `outcome=` and
`wall_s=` on `RESOLUTION_SOURCE_ESCALATION` instead of whole-ladder values credited to every rung,
`failure_class=` / `exc=` / `server=` fields on the fetch marker, one provider-level test per rung,
`FetchContext.claim_rung_budget` replacing six copies of the wall-budget preamble, a `RungSkipReason`
Literal that also took over `renderer_unavailable` and `render_timeout`, and the optional per-question
cap on paid url_context reads (`RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS`, default 2, skip reason
`url_context_cap`, count `url_context_cap_skips`). The full free gate at aca5cd8: 7,440 passed, 14
skipped, exit 0.

Gates at 55e02f5: `make lint` clean, `make typecheck` 0 errors, `make lint_imports` 6 of 6 contracts,
`make deps` clean, `make test_fast` 7,428 passed, 14 skipped, 33 deselected, exit 0.

One correction to the smoke-test figures above. Test Bot run 67's "$1.83" was the personal key's
`CREDIT_SPEND` delta, which the marker tags as unsettled and is a lower bound; the eleven per-role
`CREDIT_ROLE_SPEND` lines in the same log sum to $10.50 (`usd=`) and $8.67 (`byok_usd=`), all on the
personal key because the donated key was empty at the time. The donated key held $1,449.19 of its
$2,300 limit going into the second `test_bot.yaml` dispatch, which was therefore priced at about $10
for four questions, billed mostly to the donated key, plus Google AI Studio cents. It came in at
$10.00 in role lines (below).

## The 2026-09-04 work: second review pass, paid rung on, live probes, the smoke run, the Kalshi fix, the browser-transport SSRF closure

Everything after the aca5cd8 merge landed on 2026-09-04. The operator pushed in steps (e267d66,
ad9fec3, cbc26bf, 1f2b504); PR CI is green on cbc26bf (run 33905943087) and on 1f2b504 (run
33917127083). After the last push came the Codex review triage through c07d7cf, and then the
browser-transport SSRF closure, which the operator asked for the same afternoon rather than in a
follow-up PR. Receipts for the day sit under `scratch/fetch_ladder_2026-09-03/` (gitignored,
laptop-local): `forge_fixwave_*.md`, `fable_panel_*.md`, `paid_rung_probe_2026-09-04.log`,
`ogimet_live_check_2026-09-04.log` and `qa_smoke_2026-09-04.md`.

### Second review pass over the fix wave (3900084 to ad9fec3)

Before the smoke, the operator asked whether a final review should precede it, and two review
pipelines ran over the fix-wave diff 16ca9ab..aca5cd8. A three-lens panel (timing arithmetic,
cross-branch interactions, forecaster-facing output) returned six findings, five of them fixes: the
render harvest drain is clamped to the caller's deadline (3900084); rung verdicts are glossed for
forecasters in the failure notice, and a rescued section renders only above a 300-character floor,
`RESOLUTION_SOURCE_MIN_SECTION_CHARS`; the direct fetch's diagnostics (`failure_class`, `exc`,
`server`) stay on the fetch marker when a rung verdict replaces it; a paid url_context read whose
reply opens with the `NOT_ADDRESSED` sentinel the shared reader prompt now demands is withheld as
`no_resolving_content` / `not_addressed` instead of publishing under the primary-grading-evidence
caption, which gap-fill v2's `read_document` tool also sees since it shares the prompt; and when
policy D withholds the page text, only the chart block publishes (89fe55f, four commits).

The forge run (14 lenses, three of them GPT-5.6-sol, five stated concerns, batched verifiers and a
triage step) returned 18 fixes (12 important, 6 minor), 26 report-only items and 2 refuted. All 18
landed in three worktree merges. Ladder code (6650f3f): the precision re-extraction is bounded by the
remaining wall budget (`RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S`, 5 s), one `_run_rung` closes
every rung the same way, the budget-gated rung set is derived rather than listed, a returned Wayback
verdict keeps its own route and outcome, every transport failure bucket is pinned and
`malformed_response` is a new one, and `chrome_metric_withholds` follows the URL's ladder with a
`chrome_metric_withholds_rescued` twin. Tests and docs (db64969): the `_aux_ctx` attribution and the
marker-field helpers are pinned with mutation proofs, `curl-cffi` becomes a dev dependency so the
egress guard's import can stay hard (f02672e), the Wayback clock is pinned with a 2025 fixture date,
the probe reuses the rung's own `parse_snapshot_url`, and the docs record the paid per-question cap
and that policy D widens the paid rung's population. Browser transport (acf6b0c): five reviewers
independently found that the rendered rung's new wall bound did not bound the rung, because
`asyncio.wait_for` returns at its deadline while three separately bounded teardown steps and an
unbounded driver stop run afterwards; the reconciled remedy subtracts `RENDER_EXIT_RESERVE_MS` (3 s,
the 2 s teardown budget plus 1 s) from the deadline handed to the transport, shares one lazily started
teardown budget, and leaves the driver stop unbounded as the named residual; the outer cut records
`wall_budget` and only the transport's own cut records `render_timeout`; `render_non_200` and
`render_dom_too_large` are new skip tokens; the public-suffix helper moves to its own leaf module,
`research/public_suffix.py`; harvest responses are screened in the listener and read one body at a
time; and the render tests share one fake Playwright graph (`tests/playwright_fakes.py`). The docs
were aligned in 3e5697f and ad9fec3. One PR CI failure on the pushed e267d66 was fixed along the way:
the yfinance egress-guard test asserted yfinance's own error handling, which differs on a cold runner
(56555a8). Gate at acf6b0c: 7,557 passed, 14 skipped, exit 0.

### The paid url_context rung turned on (06b3fd9)

`RESOLUTION_SOURCE_URL_CONTEXT_ENABLED` had shipped default-off and unset in every workflow. The
operator's rule is that a feature built behind a flag ships with the flag on unless stated otherwise,
so leaving it off was a miss on the lead's part rather than a decision, and 06b3fd9 sets
`RESOLUTION_SOURCE_URL_CONTEXT_ENABLED: 'true'` on the bot step of all five bot workflows (tournament,
cup, minibench, test_bot, test_bot_basic), beside the `GOOGLE_API_KEY` it bills to. Nothing else in
those yamls changed, and the code default stays off. The resolution-source provider is therefore a
paid surface, bounded by the trigger population (only a page no free rung could read reaches the paid
rung), the free `Google-Extended` robots pre-check, the 15 s budget floor and
`RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS` (2 paid reads per question). The same commit registers
the rung's three log lines as marker specs, since they can now fire in production
(`resolution_source_urlcontext_robots_skip`, `resolution_source_urlcontext_ungrounded_suppressed`,
`resolution_source_urlcontext_not_addressed`, each with fixture lines, parse tests and collision tests
against their gap-fill v2 twins); adds a test in `tests/test_workflow_reliability.py` that pins the
flag on and a `GOOGLE_API_KEY` secret wired on the bot step of every bot workflow, so a flag-on run can
never be the `no_api_key` misconfiguration; and makes every record say the flag is on: the AGENTS.md
cost gate, the env-flag table and run-log marker list in `docs/operations.md`, the ladder section in
`docs/research.md`, the marker inventory in `docs/performance_analysis.md`, `.env.template`, and the
"off in every workflow" comments in `constants.py`, `resolution_source.py`, `markers.py` and the
rung's tests. It also fixes the two stale curl_cffi comments (`fetch_diagnostic.yaml` still said the
package was undeclared, and the AGENTS.md probe command carried a redundant `--with curl_cffi`).

### Live probes of the paid and rendered rungs

Two operator-authorized live checks closed the review's two open verification gates. The paid rung
(cents, receipt `paid_rung_probe_2026-09-04.log`): four blocked URLs through
`fetch_resolution_sources` with the real Google key and the flag on, 4.5 s total. trueup.io was
skipped free by the robots pre-check (`Google-Extended` disallow, `RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP`).
imf.org: Wayback served nothing, the paid read fired, Gemini reported `URL_RETRIEVAL_STATUS_ERROR`, and
the result was withheld as `ungrounded` with `GEMINI_USAGE role=resolution_source` logged (542 tokens).
sagaftra.org (DataDome): the paid read fired, the model opened with `NOT_ADDRESSED`, withheld as
`no_resolving_content` / `not_addressed` (3,100 tokens). congress.gov answered 200 that day, and the
extractor policy's precision fallback rescued the bill-status card (`precision_fallback_rescues: 1`).
Two paid reads in total, so the per-question cap was binding, and nothing from a challenge page
reached forecaster text. The rendered rung (free, every provider key blanked, receipt
`ogimet_live_check_2026-09-04.log`): the ogimet page that overran to 76 s in the Phase 3 QA sweep now
returns in 42.3 s, inside the 45 s wall, recorded as `render_timeout` (the transport's own DOM-read cut)
with the direct `js_wall` result left standing, and memoised so it is not rendered twice.

### The authorized smoke: Test Bot run 33907102246

The ONE authorized dispatch of `test_bot.yaml` on the branch ran on 2026-09-04 as GitHub run
33907102246 (9 m 26 s; the "Run bot" step 8 m 37 s) on cbc26bf, and a fresh-context QA pass checked it
against Test Bot run 67 as the baseline (report `qa_smoke_2026-09-04.md`; 153 commits separate the two
heads). All four test questions were researched, forecast by 3 of 3 members, aggregated and
published; the Metaculus API shows a new forecast on each post with values identical to the log and a
new private comment with three per-model bullets. Every `TIME_BUDGET` line read `fast_path=false`, as
in run 67.

The run exited 1 by design, from six alertable degradation events with one external cause: Kalshi's
events-catalogue API answered HTTP 429 to three of four concurrent full paginations (at pages 62 to 64
of 75; the fourth pull finished alone), and each lost pull bumps `prediction_market_degraded` and
`prediction_market_source_losses` once, the documented double count. The Kalshi code and constants are
unchanged since run 67, which paged the same shape cleanly the day before; the pre-existing exposure
was that the 6-hour catalogue cache had no single-flight guard, so four questions starting together
each opened a whole pagination. No forecast was affected (every market pool still carried 100 Kalshi
rows, and the ranker chose only Manifold rows in both runs). The fix is the next subsection.

The new ladder telemetry is complete and parses: 4 `RESOLUTION_SOURCE_FETCH`, 5
`RESOLUTION_SOURCE_ESCALATION` (three Wayback, two url_context), 2
`RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED`, 2 `GEMINI_USAGE role=resolution_source` (13
`GEMINI_USAGE` in total, all `gemini-3.8-flash`, 32 grounded search queries), 15
`AGENTIC_FETCH_LOCAL_DOC`; `route=` appears only on rung-produced outcomes, and the provider counts
match the lines. One real rescue: the BLS Work Stoppages page (`bls.gov/wsp/`, 403 to our client in
both runs) for question 38195 via a 6-day-old Wayback capture, with its as-of line and the
archived-copy caveat rendered once, carrying the resolution metric's own base rate (30 major work
stoppages in 2025). Both paid reads were withheld as `not_addressed` (a sagaftra.org press release
behind DataDome, and the Scott Aaronson five-worlds post, which answered our client with an nginx 429
that day); the per-question cap was not reached and no robots skip fired. Policy D withheld and
rescued nothing (all three counts 0). The rendered rung did not fire in the provider, because every
unreadable direct fetch was `blocked`, which routes to Wayback and url_context by design; Chromium
installed cleanly and gap-fill v2 rendered one page, so the browser works on the runner. Spend: $10.00
in `CREDIT_ROLE_SPEND` lines, $8.86 on the donated key across ten roles and $1.14 on the personal key
for the Google forecaster (the donated key covers OpenAI and Anthropic only); the donated key went
from $1,449.19 to $1,441.54. Google AI Studio side: cents.

Forecast content: question 38195's published median moved from about 1 in run 67 to about 3, because
three research surfaces (native search, Gemini grounded search and gap-fill v2) surfaced the Metaculus
community forecast for the exact question (6.64 from 10 forecasters) and two of three models anchored
on it; the ladder's only contribution to that question, the BLS base rate, argues for a lower count.
That is an observation for the operator about community-forecast anchoring, not a defect of the
branch. The other three questions moved modestly or not at all.

QA verdict: fit to merge for the fall season, no defect attributable to the branch. Low items it
raised: the Wayback 30-day bound plus a `NOT_ADDRESSED` reply cost question 20683 a static
definitional page that run 67 had served from a direct 200 (gap-fill v2 quoted the same post minutes
later, so the definitions reached the forecasters anyway); the all-failed notice still makes
`resolution_source` `status=ok` with `sources=0/1` (pre-existing design, and the shape the
prose-never-stands-in rule warns about); personal-key `CREDIT_ROLE_SPEND` rows show `usd` exactly
twice `byok_usd` in both runs (pre-existing, unverified); the free-text summary line said the ladder
"rescued none of them" on the question where it had rescued bls.gov (fixed below); and the text of a
withheld url_context reply was persisted nowhere, so the sagaftra.org verdict cannot be told apart
from a summarised bot-check page after the fact (fixed below).

### The Kalshi single-flight fix and two smoke follow-ups (2ceb481)

Three commits from one worktree, reviewed with zero defects, merged as 2ceb481 with a doc reflow
(d70a36f). None of the three changes forecaster-facing text, and the smoke was not re-run for them.

**Each market catalogue is pulled single-flight** (056152e). Four questions start concurrently with
both catalogue caches cold, and the TTL check cannot see a pull that has started and not finished, so
each question opened its own whole-catalogue pagination against the same venue: 60 to 75 pages apiece
for Kalshi, four at once, which Kalshi rate-limited three of on the smoke run, and three questions
then reported their own lost catalogue for one outage. Both prefetches (Kalshi and PredictIt) now go
through an in-flight future per cache key in `market_retrieval/session_state.py`, mirroring the
robots.txt guard: the first caller paginates and fills the cache, and callers arriving while that
pull is in flight await the same future and share its outcome, a venue failure included. Sharing the
failure is the point, because the failure this guards against is a rate limiter, and three more
questions re-asking it would be a second violation rather than a retry (the page fetcher already
refuses to retry a 429). A leader that produces no result (its pull raised, or its caller's deadline
cancelled it) resolves its waiters to nothing and sends them back through the guard, where the first
to wake leads one fresh pull. The cache write and the catalogue-failure counter move inside the pull,
so one lost pull counts once; per-question recording is untouched (every caller still records its
catalogue-size observation, and a question whose snapshot went out without Kalshi rows still records
that source loss). `docs/research.md` describes the guard.

**The resolution-source summary states rescue counts** (6be176f). The question-level summary line
closed every unfetched count with "js_wall/blocked, the escalation ladder rescued none of them",
unconditionally, and on the smoke run printed that for the question where Wayback had just served
bls.gov. It now states how many cited URLs went unfetched and with which statuses, and how many a
later rung rescued (a success whose route is not the direct fetch, the reading the rung counts already
use). Free text, not a marker.

**The head of a withheld url_context reply is kept in the log** (78b2e7b). Both url_context withholds
discard a read that was paid for and keep its text nowhere, which left the smoke QA unable to say what
sagaftra.org's `NOT_ADDRESSED` reply meant: a page that genuinely does not discuss the ask and a
DataDome challenge page the model dutifully summarised reach that branch identically, and the reply is
the only thing that separates them. Each withhold now logs the head of the discarded reply on its own
INFO line, whitespace-collapsed and capped at `RESOLUTION_SOURCE_WITHHELD_REPLY_LOG_CHARS` (300). The
lines are deliberately unregistered and sit after the markers rather than inside them, so the two
marker line shapes are untouched; the `ungrounded` line fires only when the reader said something,
since that branch also covers an empty reply. `docs/operations.md` documents both lines.

### The browser transport closure

Two commits, `6646a0b` and `8ced8a5`, close the two P1 findings from the Codex review below, plus a
third channel the verification turned up, instead of deferring them to a follow-up PR; the operator
asked for the work the same afternoon. A forge review, an eight-commit fix wave and two rounds of
free live QA followed, all recorded below. Three terms, because the rest of this leans on them: the
PINNED HOST is the one hostname the `--host-resolver-rules` launch argument holds Chromium to, the
ROUTE HANDLER is the `context.route` callback that re-runs `is_public_http_url` on each request the
browser is about to make, and the LANDING HOST is the hostname `page.url` carries once the
navigation has settled.

**An off-host landing is refused unread, or discarded unpublished.** The route handler never sees a
server-side redirect hop, because Playwright auto-continues a request that carries a
`redirectedFrom`, so a page that walled the plain aiohttp GET with a 200 and answered the browser
with a 302 to a private address was rendered and handed back attached to the cited URL.
`rendered_fetch` now reads `page.url` after the settle on the normal path and on the salvage path
where the goto raised, and raises `RenderOffHost` before `page.content()` is called when the landing
host is not the pinned one, an IP literal included. The same comparison runs a second time on
`page.url` immediately after the DOM read, because `page.url` is a client-side cache the driver
updates on its `navigated` event while `page.content()` is a driver round trip evaluated in whatever
document is current, so a navigation that commits inside that window would otherwise hand back the
other host's DOM with the pre-read check already passed. The second read costs no await and a DOM
that fails it is discarded unpublished, which is why the guarantee is "refused unread, or discarded
unpublished" rather than "never read". The predicate, `_landed_off_host`, fails shut: only an empty
landing URL and the `about:` scheme are no-document landings, and every other scheme is a stranger,
so Chromium's own `chrome-error://chromewebdata/` document after a failed navigation is refused too
and `render_off_host_skips` is an upper bound on hostile landings that also counts failed
navigations, told apart by the marker's `landed_host`. A navigation that never committed leaves
`about:blank`, which is nobody's host and falls through to the empty-DOM read it always produced.
`RenderedPage` gained `final_url`, taken from that second read, and `RenderedPage.document_url`, the
landing URL when a navigation committed and the requested URL otherwise; `RenderedPage.url` still
carries the requested URL, because that is the key for both render memos.

**Page WebSockets are blocked.** A WebSocket handshake is invisible to `context.route` by
construction, since HTTP interception is the CDP `Fetch` domain while WebSockets surface only on the
report-only `Network.webSocket*` events. `context.route_web_socket("**/*", _block_web_socket)` is
registered before the page exists, because only sockets created after the registration are routed,
and the handler logs one INFO line, the level `cli.py` configures the root logger at, and never
calls `connect_to_server`, which is the whole block. It is written to be provably raise-free, because
`unroute_all` clears HTTP routes only and Playwright has no `unroute_web_socket`, so the handler
stays registered through the context close by design. The block is an in-page mitigation rather than
a network boundary, and `_block_web_socket`'s docstring now says so and records the limits: the
driver injects an init script at registration that replaces `globalThis.WebSocket` in every frame,
leaves the enumerable `__pwWebSocketBinding` and `__pwWebSocketDispatch` globals behind, and does not
reach a dedicated Worker's global scope.

**The browser is handed the URL the direct fetch landed on.** `FetchResult.url` is already the last
hop of the direct fetch's re-guarded redirect loop, so the rendered rung passes that rather than the
cited URL. The pin and the landing check then hold Chromium to the host that serves the content, and
a page whose canonical form is one ordinary hop away (`example.com` to `www.example.com`) is not
refused for taking it. When the two URLs differ the landing is re-vetted with the same
self-reference and public-URL checks every derived hop owes, and those two checks now have one home,
`_hop_refusal`, which `_vetted_hop_target`, this render-landing re-vet and the Wayback
innermost-URL check all call, publicness first so a URL that is both non-public and a self-reference
still reports `ssrf_blocked`. The HTML classifier's base URL is `RenderedPage.document_url`, and
gap-fill v2 resolves the links it harvests against the same document URL, so a same-host redirect
from `/senate` to `/senate/2026/` resolves a relative `href` against the document the browser
actually landed on; both render memos stay keyed on the requested URL. One accounting consequence,
recorded where the archive reader looks: on a `route=rendered` rescue the `FetchResult.url` is the
landing URL, which is what the fetch line's `url=` and the published section heading carry, while the
`RESOLUTION_SOURCE_ESCALATION` line's `url=` is the cited URL, so a per-URL join between the two
lines keys on the escalation line.

**The names the decline gets.** Tier-1 records it as its own skip token `render_off_host`, added to
`RungSkipReason`, counted as `render_off_host_skips` in `details["counts"]`; the direct result
stands and nothing from the render is published. Gap-fill v2 folds it into the None outcome it
already returns for its sibling declines. A skipped attempt emits no `RESOLUTION_SOURCE_ESCALATION`
line, so the per-event record is the transport's WARNING, registered in `aa37f11` as the marker
`RENDERED_FETCH_OFF_HOST: scope=<resolution_source|gap_fill_v2> pinned_host=<host> landed_host=<host>
same_publisher=<true|false>` (spec `rendered_fetch_off_host`), which names hostnames only because a
landing URL can carry a session token; `same_publisher` is true for a benign hop inside the pinned
host's registrable domain and false otherwise, including every landing with no hostname.

**The probe that priced strict host equality, free and local.** Playwright 1.61 was driven directly
with the transport's navigation shape and no pin, over the 106-URL Phase 3 QA sweep united with the
47-URL replay, one page at a time. 103 of the 106 navigations committed, and 0 of the 103 landed on
a different host. Of the 22 render targets in that corpus, 22 committed, 0 landed off host, and 0
opened a WebSocket. 3 base-rate pages opened one: two Yahoo Finance history pages to
`streamer.finance.yahoo.com` and manifold.markets to `api.manifold.markets`, all three public hosts
inside their own registrable domain. 6 pages ended on a different path or query on the same host, so
a host-only comparison is what the corpus supports and a path-level one would have refused all 6.
The only main-frame hops were same-host: trueup.io answering a 307 to itself and then a 403,
ballotpedia.org and fts.unocha.org reloading their own bot-challenge documents, and a cdc.gov meta
refresh. 3 URLs did not commit, a PDF download and two HTTP/2 protocol errors, and 6 hosts answered
headless Chromium with a 403, which is a fingerprinting result rather than a host change. So strict
pinned-host refusal costs no recall on this corpus, and a re-render hop to the landing host is not
worth building for a case the corpus does not contain.

**Review and fix wave.** A 22-agent forge over the closure (11 reviewers, 4 batched verifiers and a
triage pass) returned needs-work: 8 fix findings, 13 report-only items, 1 dropped, and 14
verification gates, almost all of them the same gate under different names, that real Chromium
behaviour is unobservable in a suite which refuses every browser launch. Three of the eight fix
findings were important. The landing check was taken once before the DOM read and never re-taken
after it, so a navigation committing inside the `page.content()` round trip handed back the other
host's DOM with the check already passed. The two-check hop policy had a third hand-rolled copy, on
the one surface where a missed check means Chromium dials the host. And nothing in the suite pinned
that gap-fill v2 renders the plain fetch's post-redirect URL: two reviewers independently rewrote
both call sites to the pre-redirect URL and the whole suite stayed green. The five minor findings
were the tri-state landing helper failing open on every non-http(s) scheme, the shared Playwright
fake pinning a fixed host whatever URL a test rendered, a launch-cap test green only because the
fake's settle happens not to suspend, the WebSocket block logging at a level the root logger drops,
and the in-page shim's limits recorded nowhere. One reviewer could not read line coverage because
`pytest --cov` died in its own checkout; `make cov` on the main checkout ran clean at 7,649 passed
and 92 percent total branch coverage. Of the report-only items the lead fixed R1 (the `final_url`
production wrote and nothing read), R4, R5, R6, R9, R10, R12 and R13 anyway, and left four alone: R2
and R3, R7 (two file sizes, their own PR), R8 (a harvested data endpoint remembered under the cited
host on a cross-host redirect, measured at 0 of 47 replay URLs) and R11, already fixed by then. The
forge report and its execution plan are in the session's receipts.

The fix wave is 8 commits, `c4b8a57` to `001b6f9`, each written test-first: fail the landing-host
guard shut and re-check it after the DOM read; pin that gap-fill v2 renders the plain rung's
post-redirect URL; give the two-check hop policy one home; pin the requested host by default in the
shared Playwright fake, which is also what fixed the launch-cap test; log the WebSocket block at INFO
and record the shim's limits; fold in the small review items; classify and resolve links against the
document the browser landed on; and drop the em dashes from the re-wrapped Wayback docstring. Both
new tripwires were verified by re-applying the defect they exist for. Rewriting both gap-fill v2 call
sites to the pre-redirect URL turns three new tests red, and an `await asyncio.sleep(0)` in the
fake's settle leaves the semaphore test green. Both mutations were reverted.

Live QA against real Chromium ran twice, free, with every provider key blanked and no LLM or paid
call. Round one, against `8ced8a5` and `91b6845`, found the branch fit to merge from the transport's
point of view: a live loopback canary landing and a public off-host landing were both refused with
the marker line, a same-host redirect, a bot-challenge reload and a meta refresh were let through,
the WebSocket block fired on manifold.markets where a control run without the transport exchanged
frames on the same socket, and the render rung handed the transport the post-redirect https URL end
to end for two `http://` inputs. It found one gap, the one the fail-shut predicate then closed: a
redirect to a private target that was not listening landed on `chrome-error://chromewebdata/` and
came back as an empty page rather than a refusal. Round two ran the same launcher against `001b6f9`
and all 8 cases passed in 32 s. The two chrome-error landings now refuse with
`landed_host=chromewebdata`, the loopback canary and the example.org landing refuse with the marker,
the same-host redirect and the meta refresh render with `final_url` recorded, the WebSocket block
line appears at INFO under the default root logger, and the dcas.dmdc.osd.mil casualty page renders
through the provider as `status=ok route=rendered` with `render_off_host_skips=0`.

**What is left.** A cross-host SUBRESOURCE is still resolved by Chromium with no pin, so the route
handler's own `getaddrinfo` and Chromium's connect resolve independently and a rebinding host with
TTL 0 can win that race. Chromium 149's Local Network Access enforcement is believed to gate a
public page's subresource requests to loopback, RFC1918 and link-local addresses, but that is
inferred from the feature lists and vendor docs and has never been observed in this headless build,
because the test suite refuses every real browser launch. The main-frame channel is closed.
FUTURE.md item 8 is now a record of what shipped, carrying that residual and the single observation
that would settle it.

**Tests and the gate.** `tests/test_rendered_fetch.py` gains `TestTheLandingHost`, fifteen cases
after the fix wave, covering the refusal before the DOM read, a navigation that commits during the
read being discarded unpublished, `final_url` coming from the post-read landing, `document_url` being
the landing when one committed, an IP-literal landing, a same-host landing on another path, a
case-only difference, an uncommitted navigation, Chromium's own error document, a landing on some
other scheme, the no-document allowlist, the salvage path, a salvaged same-host `final_url`, and the
Tier-1 rung's own skip; and `TestTheWebSocketChannel`, four covering registration before the page,
the handler never connecting, the handler not raising on an odd URL, and a Playwright error at the
registration taking the pre-page path.
`tests/resolution_source/test_resolution_source_rendered_rung.py` gains
`TestRenderedRungLandedOffHost` and `TestRenderedRungRendersTheFinalUrl`, which pin the skip
claiming no route, the count reaching the provider's details, the final URL being what the browser
is handed, the memos keyed on it, and a landing the ladder would not fetch never being rendered.
`tests/test_agentic_tools.py` gains `TestGapFillV2RendersThePlainRungsFinalUrl`, three cases: one per
v2 call site with a recording browser rung, and one end to end through `fetch` with a scripted 302.
The shared Playwright fakes gained `page.url`, `land_on`, `content_reads`, `route_web_socket` and a
`WebSocketRoute` double whose `close` takes Playwright 1.61's keyword-only arguments, they now pin
the requested host by default, and the OS-timeout test pins the classification it exists for instead
of accepting any logged exception.

### Gates and CI at the tip

The full free gate on 001b6f9, the branch tip (`make lint`, `make typecheck` at 0 errors,
`make lint_imports` with 6 contracts kept, `make deps`, `make test_fast`): 7,655 passed, 14 skipped,
33 deselected, exit 0 (`~/logs/gate18.log`). The commit after 001b6f9 is this documentation update.
The 63 tests above the earlier gate's 7,592 on c07d7cf are the browser-transport closure's and its
fix wave's; the five above the gate before that were the Codex triage's, three for the drain-timeout
marker and two for model-id locations. Line coverage was read separately, because one reviewer's
`pytest --cov` died in its own checkout: `make cov` on the main checkout ran clean at 7,649 passed
and 92 percent total branch coverage.
A post-fix verification pass (a forge verify-mode agent plus three adversarial reviewers, one each on the
landing-host guard, the render path and the hop policy) judged all 18 targeted findings resolved, each
with a test that goes red under the original defect, and found no way through the guard against any
scheme, hostname form or mid-read navigation it could reach. It surfaced nine comment and doc sites that
still carried the pre-fix absolute "refused unread" (swept in e6c0f91) and five small, strictly-safer
improvements that landed as a final fix wave: the `about:` allow-arm keys on the landing having no host
rather than on the scheme; a plain Playwright error from `page.content()`, which is what a page
navigating mid-read raises, is classified as `render_timeout` instead of `renderer_unavailable` and no
longer burns the once-per-run install warning; a failed-navigation landing writes the timed-out memo so
a run does not relaunch Chromium for the same dead URL; the marker gained `same_publisher=true|false`
so a benign hop inside the publisher's registrable domain is distinguishable from a hostile landing;
and the fake browser context accepts only real keyword names. The full free gate on the merged tree, with these documentation edits in the working tree, is green: `make lint`, `make typecheck` at 0 errors, `make lint_imports` with 6 contracts kept, `make deps`, and `make test_fast` at 7,673 passed, 14 skipped, 33 deselected, exit 0 (`~/logs/gate19.log`). The code tip is df05c8c; the commit above it carries only this documentation and one docstring.
PR CI on the pushed 1f2b504 (run 33917127083: lint, test, secret scan, audit) completed green at
20:43 UTC on 2026-09-04; the commits after it carry the Codex triage's documentation, comments and
tests,
this description, the browser-transport closure with its documentation, and the closure's review fix
wave. The marker registry stands at 61 specs at 001b6f9, the last of them `rendered_fetch_off_host`.
Against `main` the whole PR at 001b6f9 is 366 commits and 247 files (+58,570 / -6,362).

### Codex review triage

The GitHub-side Codex reviewer left five comments on the PR, four against ad9fec3 and one against ea1d558. Each was checked against HEAD 1f2b504 by one verifier and then by two adversarial reviewers, one arguing the defect does not exist and one arguing the proposed fix is unsafe. Where a library claim carried weight they read the pinned Playwright 1.61.0 driver and pypdf 6.16.2 source rather than the docs. Three findings were wholly or mostly out of date, and two are real but belong in a follow-up. What landed here is documentation truth, one offline marker spec and one test.

**Move Gemini model defaults into llm_configs.py (P2): does not hold.** Both constants were already in `constants.py` on `main`; the branch changed their values only. `constants.py` has carried the support-model id strings for the native-SDK callers since 2025-12, because it is a leaf module that `llm_configs.py` imports and so cannot import back, and a test already fails if the two Gemini surfaces ever run different ids. The rule text was what was wrong. AGENTS.md and `docs/roster_history.md` now state the real split (`llm_configs.py` owns the roster and the support-model `GeneralLlm` objects it builds once at import time; `constants.py` owns a bare id string for each support role whose consuming module builds its own client at call time, beside that role's env override and price note), and a new test inventories every model-id literal under `metaculus_bot/` and pins the files allowed to carry one.

**Register the new operational marker prefixes (P2): three quarters stale.** The three `RESOLUTION_SOURCE_URLCONTEXT_*` markers were registered in 06b3fd9, after the commit Codex read. The fourth, `LITELLM_CALLBACK_DRAIN_TIMEOUT`, was added on this branch without a spec and recorded in FUTURE.md as a deferral. It is now a registered spec with a fixture test and a collision test against the `CREDIT_*` lines. Harvester code only; nothing in the bot's runtime path changed.

**Enforce the PDF wall-clock bound (P2): real, low severity, deferred.** The page count, the outline walk and the digest run before or after the parse clock, a started thread worker cannot be cancelled, and both routes hand the shared parse permit back while the worker keeps running. The magnitude is finite rather than unbounded: pypdf's own caps hold the un-clocked prologue to about 16 s on an adversarial 40 MB file (a nested outline of 100,000 entries alone costs 13.7 s), so an abandoned worker lives about 36 to 40 s at worst. Each remedy either drops the outline in a corner case, can wedge the two-slot parse gate, or is a subprocess build, so under the standing rule that timing code takes only strictly-safer changes the code fix is deferred. FUTURE.md carries the design and the measurements, and the comments that described the worker as bounded were corrected.

**Block WebSocket requests during browser renders (P1) and Pin DNS for every redirected browser host (P1): both real, both built in this PR.** The transport's own comment already named both as accepted residuals. Verification found a third channel nobody had recorded: Playwright never runs a route handler on a server-side redirect hop, so a cited page that answers the bot's plain fetch with a JavaScript wall and answers Chromium with a 302 to a private address rendered that response into the main frame with no check at all. Chromium 149's Local Network Access enforcement gates the subresource and WebSocket channels (inferred from the feature lists; no browser was launched to observe it), and it does not cover a main-frame navigation, so the redirect was the exploitable channel. Neither fix is strictly safer on its face, which is why the plan had been to defer them: the WebSocket block turns a socket-fed page into a run-long no-text memo, and strict host pinning refuses cross-host redirects at a recall cost the replay never measured. The operator asked for the work the same afternoon, so a free local render probe measured the recall cost first, at zero cross-host landings and zero WebSockets across 22 real render targets, and the transport then gained a landing-host refusal, a WebSocket route that never connects, and the direct fetch's landing URL as what the browser is handed. The closure was then reviewed on its own: a 22-agent forge returned 8 fix findings, a fix wave of 8 commits applied all of them, and two rounds of free live QA against real Chromium observed the landing refusal and the WebSocket block working. Both P1s are therefore built, reviewed and QA'd. The subsection "The browser transport closure" above carries the commits, the probe numbers, the review and the fix wave, both live QA rounds, the skip token and count key, the marker, the tests and the one residual left. The route-guard comment, `docs/agentic_gap_fill.md`, `docs/research.md`, `docs/operations.md`, AGENTS.md and FUTURE.md all state what the guard now covers.
