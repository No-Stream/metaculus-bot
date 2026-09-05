# Next-season bundle — implementation plan (2026-09)

Written 2026-09-01 for a fresh implementing session with zero prior context. Everything you
need is in this document plus the receipt files it cites. Do not start until you have read §1
(constraints) in full — two of them (the cost gate and the one-merge rule) are hard rules that
override any instinct to "just verify with a quick run" or "split this into smaller PRs".

## §0. Context and provenance

The tournament season is effectively over: question supply for `summer-futureeval-2026`
exhausted 2026-08-20 and the tournament closes 2026-09-06. The fall Metaculus Cup (project
33108, slug `metaculus-cup-fall-2026`) exists and is empty; it is expected to start opening
questions around 2026-09-20, and the `FALL_CUP_REMINDER` machinery already shipped (`f836242`).
This bundle is the set of improvements the operator approved on 2026-09-01 out of the
2026-08-31 residual round and its three follow-up investigations, to be live before the cup's
first question. The receipts live in `scratch/residual_2026-08-31/` (gitignored — read with
direct paths, grep with `rg --no-ignore`): the round synthesis (`SYNTHESIS.md`, esp. §6), the
cross-dossier read (`DOSSIER_SYNTHESIS.md`, esp. §7), the gemini forecaster-slot decision memo
(`gemini_review/RECOMMENDATION.md`), the gemini search-provider audit
(`gemini_search_audit/VERDICT.md` + `cutB_pattern.md` + `cutC_harm_benefit.md`), the
market-odds prompt-bullet study (`market_odds_coverage.md`), and the branch forge review
(`forge_plan_resids-sept1.json`, report-only items R1/R2). Every item below was individually
approved by the operator; the verdicts are final. Where an item says "pending operator
confirmation" that is the one exception, and the plan marks it.

Item numbering below preserves the operator's list (1–19; there is no item 6). The items fall
into four tiers: **A** — the publish-path change (the k-conditional binary floor, the one item
the operator flagged "let's be careful to get it right"); **B** — gemini search-provider fixes;
**C** — other research/provider fixes; **D** — measurement and telemetry. The implementation
ORDER is D → C → B → A (see §2), so the riskiest item lands last, after the session is warmed
up on the codebase, and gets an adversarial review before merge.

## §1. Cross-cutting constraints (read before touching anything)

1. **One merge, one era.** Residual analysis buckets by config era, and era boundaries are
   merge-to-main COMMITTER timestamps (see the era-bucketing section of the repo AGENTS/CLAUDE
   instructions). Every prompt-affecting or forecast-distribution-shifting change in this
   bundle must therefore land on `main` in a SINGLE merge, so next season's records carry one
   boundary, not five. The telemetry/tooling items (Tier D, items 14–19) do not shift the
   forecast distribution and may ride the same merge; do not split them out into a separate
   earlier merge of prompt-adjacent code "to be safe" — one PR, one merge.
2. **Free gates only. The cost gate is absolute.** During implementation you may run only the
   free gates: `make test`, `make lint`, `make typecheck`, `make lint_imports`, `make deps`
   (plus `make format`, `make cov`, `make test_fast` as needed). NO paid runs: no
   `main.py` live modes, no `make backtest_*`, no `make ablation_*` (except the free
   `ablation_score`), no GHA bot-workflow dispatches, no one-off scripts that hit research or
   LLM providers. If at any point verification seems to want a paid run, the correct move is to
   write down the exact command and rough cost and STOP — the operator decides (§5). This holds
   even when a paid run is the only way to verify something; say so and stop.
3. **TDD, per repo convention.** Write the failing test first for every behavioral change.
   basedpyright must stay at 0 errors (`make typecheck`). Never weaken, bypass, or delete a
   test to make it pass. All work on a feature branch — suggested name **`next-season-bundle`**
   — branched from `main` after the `resids-sept1` branch merges (or from `resids-sept1` if it
   has not; check `git log --first-parent main` first and ask via the PR description rather
   than guessing silently).
4. **No AI attribution in commit messages.** No Co-authored-by trailers, no tool references.
5. **Timing.** The bundle should be merged before the fall cup opens questions (~2026-09-20)
   so the new prompts are live from question one. That leaves slack; do not rush Tier A into
   the merge without its adversarial review.
6. **Corpus validations are gates, not garnishes.** Items 3 and 4 each carry a mandatory
   offline validation over the archived gemini sections. Those validations are free (local
   files only) and their pass criteria are stated per item; the items do not ship without them.

## §2. Sequencing

Single PR, commits grouped by tier, in this order:

1. **Tier D first** (items 14–19, plus 17's doc edit): telemetry markers, analysis-tooling
   conventions, doc corrections. Risk-free, no prompt surface, and several later items reuse
   the marker-registration pattern established here.
2. **Tier C** (items 8–13): research/provider fixes. Item 13 is a design DOC only — no code.
3. **Tier B** (items 2, 3, 4, 5, 7): the gemini provider batch, including the two corpus
   validations. Item 5's wording needs operator confirmation before the commit that touches
   `prompts.py:266` — batch the question with the Tier A review request if unanswered by then.
4. **Tier A last** (item 1): the k-conditional publish floor, followed by an adversarial
   review of that diff specifically (run `/forge` over the branch; additionally hand item 1's
   diff to a codex second-opinion review — the repo's standing reviewer-pairing practice).
   Triage findings, fix, re-run gates.
5. Full gates green → PR → operator reviews → merge (one merge; see §1.1).

Suggested checkpoint discipline: commit per item or per small item-group with the item number
in the subject (e.g. "item 11: no_resolving_content status for embed-shaped 200s"), so the PR
review can walk the operator's list.

---

## §3. Per-item specifications

### Tier D — measurement and telemetry (build first)

#### Item 14 — `forecasters_used` stamped on every published record

**What/why.** "How often does the bot publish thinned" is unanswerable outside the triple era
because only 21 of 799 archived records carry the ensemble-size disclosure; it is precisely the
exposure the Tier A floor addresses, so next season's records must all carry it. Receipt:
`scratch/residual_2026-08-31/gemini_review/RECOMMENDATION.md` §4 item 2.

**State of the code (verified 2026-09-01 — this item is mostly already shipped).** The
comment-side marker exists and is emitted unconditionally on the fan-out path:
`metaculus_bot/comment/markers.py:168-178` (`FORECASTERS_USED` formatter + regex),
`metaculus_bot/comment/formatting.py:66-78` (`_forecasters_used_suffix`, appended in
`build_unified_explanation` for BOTH stacking and non-stacking strategies), and
`metaculus_bot/forecaster.py:855-863` + `:982-996` (count recorded at fan-out, threaded into
every comment; delegated-path fallback included). The collector already reads it:
`performance_analysis/collector.py:236` and `:428-429`. The 21/799 is archive history — the
marker landed late in the season and nothing rewrites old records — so season-forward coverage
is automatic.

**Deliverable (verification + guard, not a rebuild):**
- A presence-guard test asserting every published comment carries the `<!-- FORECASTERS_USED=n/N -->`
  marker across: all four question types, the single-forecaster skip path, the stacked path,
  and the non-stacking-strategy path. Home: `tests/test_comment_formatting.py` /
  `tests/test_main_comment_output.py` (both already reference the marker — extend, don't
  duplicate).
- A trim-survival assertion: the marker rides the comment TAIL and `trim_comment` middle-trims,
  so it should survive an oversized comment — pin that with a test if none exists (check
  `tests/test_comment_formatting.py` first).
- Confirm the collector surfaces `forecasters_used`/`forecasters_configured` on fresh records
  (a unit test against a fixture comment carrying the marker; likely already exists in
  `tests/test_performance_analysis_parsing.py` — verify, extend if not).

If all three guards already exist, this item closes as verified with a note in the PR
description; do not manufacture work.

#### Item 15 — standing per-member extreme-call counter with lone/accompanied flag

**What/why.** The cleanest per-model signal three residual rounds have produced is gemini's
extreme-call rate (7/43 post-flip vs 10/219 pooled others, Fisher p=0.0107), and the actual
finding is the lone-vs-accompanied split (lone extremes 4/9 right at stated 0.972; accompanied
21/23) — currently reconstructed by hand from parsed comments every round. Receipt:
`gemini_review/RECOMMENDATION.md` §2 ("The mechanism") and §4 item 3.

**Design.**
- Site: `metaculus_bot/forecaster.py`, immediately after the `FORECASTERS_SURVIVED` log
  (`forecaster.py:885-891`) — the one point where the surviving `valid_predictions` (binary
  values are floats in `prediction_value`) and per-model display names
  (`extract_model_display_name_from_reasoning`) are both in hand.
- Scope: **binary questions only** (the finding is binary; MC concentration is a separate
  standing cut the RECOMMENDATION names but the operator did not include).
- Emission: one INFO line per member whose p ≤ 0.05 or ≥ 0.95:
  `EXTREME_CALL: question=<id_of_question> model=<name> p=<value> side=low|high lone=true|false survivors=<k>`.
  `lone` = no OTHER surviving member is in the extreme band on the SAME side. Carry
  `survivors=` because at k=1 "lone" is vacuous (the RECOMMENDATION excludes degraded-window
  records from the lone test for exactly this reason) — analysis needs to be able to condition
  on it.
- Thresholds: module constants `EXTREME_CALL_LOW = 0.05` / `EXTREME_CALL_HIGH = 0.95` in
  `metaculus_bot/constants.py`, placed beside (and deliberately equal to) the Tier A floor
  constants, with a comment noting the two share one definition of "extreme band" and must move
  together if ever retuned.
- Register a `MarkerSpec` in `scripts/telemetry/markers.py` (`MARKER_SPECS`, `qid_kind=QID_KIND_QUESTION_ID`
  — same id space as `forecasters_survived`, since the emitter uses `id_of_question`). Adding
  the spec is what makes `make sync_telemetry` harvest it; follow the existing spec-comment
  convention naming the emitting module.
- The denominator for rate claims (all binary member-forecasts) is recoverable from
  `FORECASTERS_SURVIVED` + question type; do not emit a marker for non-extreme members.

**Tests.** Marker fires for a member at 0.03 (side=low) and 0.97 (side=high); no marker at
0.10/0.90; `lone=false` when two members are extreme same-side, `lone=true` when the second
extreme member is on the opposite side; `survivors=` correct at k=1; marker-regex parse test in
`tests/test_telemetry_markers.py` (mirror the existing per-marker parse tests).

#### Item 16 — per-role dollar attribution on credit telemetry

**What/why.** Every cost argument the gemini review needed was blocked on decomposition:
measured $0.38–0.41/question total OpenRouter spend cannot be split into forecaster vs research
vs ranker, so "a 4th member costs +33%" stays an assertion. Receipt:
`gemini_review/RECOMMENDATION.md` §3 (option 4 cost paragraph) and §4 item 4.

**Deliverable shape: investigate-and-spec is acceptable.** The operator accepts either a
working implementation or a written spec naming the blocking facts. Sequence:

1. **Investigation (offline; read code + docs, no paid calls).** Establish which per-call cost
   source works on BOTH keys: (a) litellm's computed `response_cost` hidden param, (b)
   OpenRouter's usage accounting (request `usage: {include: true}` → response carries
   `usage.cost`), (c) the `GET /api/v1/generation?id=<gen_id>` endpoint (authoritative, but one
   extra HTTP call per LLM call and needs the generation id surfaced through litellm). Also
   establish the routing coverage fact: does ALL OpenRouter traffic flow through
   `FallbackOpenRouterLlm` / `build_llm_with_openrouter_fallback`
  (`metaculus_bot/fallback_openrouter.py`), or do some roles construct `GeneralLlm` directly
  (check `llm_configs.py`, `research/providers.py`, `research/targeted.py`,
  `market_retrieval/ranking.py`, `research/agentic/`)?
2. **If a clean path exists, implement:** thread a `role=` tag at LLM construction time
   (roles: `forecaster:<slot>`, `stacker`, `stacker_fallback`, `parser`, `summarizer`,
   `crux_analyzer`, `native_search`, `gap_fill_analyzer`, `gap_fill_resolver`, `market_ranker`,
   `gap_fill_v2_driver`, `gap_fill_v2_reader`, `financial_classifier` — use the descriptive
   names, no opaque labels); accumulate per-role cost + call count in `credit_telemetry.py`;
   emit end-of-run `CREDIT_ROLE_SPEND: role=<role> usd=<x> calls=<n>` lines beside the existing
   `CREDIT_SPEND` marker; register a `MarkerSpec` (`qid_kind=None`, like the other credit
   markers). **Hard constraint:** the accounting block in
   `fallback_openrouter.record_donated_key_fallback` must stay free of `await`s after the
   threaded probe (documented race — module-global `+=` is interruptible between bytecodes);
   any per-role accumulation there must be synchronous.
3. **If blocked** (e.g. cost not returned on the donated key, litellm swallows the generation
   id): write the spec to `scratch_docs_and_planning/per_role_cost_attribution_spec.md` stating
   what was tried, what the blocking fact is, and the recommended path (likely the
   `/generation` endpoint with sampled reconciliation), and reference it from item 17's
   FUTURE.md edit.

**Tests (implementation path).** Mocked responses carrying cost fields → per-role totals
correct; roles missing cost data degrade to `usd=n/a calls=n` rather than fabricating zero;
marker parse test. Note `scripts/reconcile_credit_spend.py` and
`tests/test_credit_telemetry.py` exist — extend rather than duplicate.

#### Item 17 — FUTURE.md cost correction

**What/why.** FUTURE.md still prices the 6→3 roster drop as "~$3.05 to ~$1.65" per question;
the measured figure is $0.4082/question (triple era) and $0.3836 (ranked-markets window), an
order of magnitude lower, and the review explicitly says the $1.65 figure should not appear in
any re-add decision. Receipt: `gemini_review/RECOMMENDATION.md` §3, "Cost, and a standing
figure that should stop being quoted".

**Edit.** `FUTURE.md:144-148` (the "Cost context for the re-add decision" paragraph): replace
the standing ~$1.65/question figure with the measured **$0.38–0.41/question**, LABELED
verbatim-close to: "OpenRouter-only lower bound; excludes Google AI Studio prepaid, AskNews
subscription, and Exa; measured over 29 triple-era runs / 33 questions". Keep the historical
~$3.05→~$1.65 estimate visible as a superseded estimate if the sentence still needs it for
context, but the quotable figure must be the measured one. Point at item 16 (per-role
attribution — or its spec doc) as the real fix. No test; doc-only.

#### Item 18 — live model-list read at season start (documented manual step)

**What/why.** The roster design is latest-per-vendor, and nothing in the repo can establish
what "latest Google/OpenAI/Anthropic" currently resolves to — the gemini review flagged that a
roster decision needs one live model-list read, which is a check, not a finding. Receipt:
`gemini_review/RECOMMENDATION.md` §3 (option 5, last sentence) and §4 item 6.

**Deliverable (documentation only; the read itself is an operator step, not run now).**
- Add a short comment in `metaculus_bot/llm_configs.py` adjacent to the `FORECASTER_LLMS`
  roster comment (around line 62): at season start, resolve latest-per-vendor with a live
  model-list read, never from memory, and name the command.
- Document the command itself in `docs/operations.md` (a "season-start checklist" stanza is a
  natural home if one doesn't exist). Suggested command (free metadata read, no inference
  spend, but it IS a network call — it belongs to the operator's season-start ritual, not to
  this implementation session):
  `curl -s https://openrouter.ai/api/v1/models | jq -r '.data[] | [.id, .created] | @tsv' | sort` filtered per vendor prefix (`openai/`, `anthropic/`, `google/`, `x-ai/`).
- No code change, no test.

#### Item 19 — analysis-tooling set

Four sub-items, all in analysis/tooling code paths (no prompt surface, no publish surface
except 19b's optional WARN).

**(a) Set-valued PIT for out-of-range resolutions.**
- Why: `numeric_pit_analysis` (`performance_analysis/analysis.py:274-277`) forces an
  `above_upper_bound` resolution to PIT 1.0 and `below_lower_bound` to 0.0. On q44842 the bot
  deliberately placed 13% of its mass above the displayed ceiling, won peer +24.4 — and the
  convention scored it a band miss. The receipt (`SYNTHESIS.md` §3, width-watch row) prices the
  fix at 8 points of triple-era cov80 (0.700 → 0.778). Convention: an out-of-range resolution's
  PIT is only known to lie in the SET `[cdf[-1], 1]` (above) or `[0, cdf[0]]` (below) — the
  platform says "beyond the range", and our own CDF says how much mass we put out there.
- Design: represent the OOB PIT as an interval. Coverage metrics (cov80/cov50/cov@10/cov@90)
  count an interval record as covered iff the interval INTERSECTS the band (equivalently: a
  band miss only when the entire interval lies outside). Point statistics (pit_std, histogram,
  mean) exclude interval records and disclose the excluded count (`n_oob_interval`) rather than
  imputing a midpoint — an imputed value is exactly the kind of manufactured reading the repo's
  measurement conventions exist to prevent. Apply the same convention in BOTH consumers:
  `performance_analysis/analysis.py` (`numeric_pit_analysis`) and
  `performance_analysis/width_monitor.py` (its `compute_pit_details` path — width_monitor
  already imports PIT helpers from analysis.py; keep the rule in ONE home, analysis.py, and
  have width_monitor consume it, mirroring how `pit_on_grid` is shared today).
- Tests: q44842-shaped fixture (above_upper_bound, cdf[-1]=0.87) counts as covered at cov80 and
  is excluded from pit_std with the disclosure count; a starved-tail fixture (cdf[-1]=0.999)
  still counts as a high-side band miss; below_lower_bound mirror case.

**(b) STARVED_OUTER_TAIL detector (open-bound p99 cliff).**
- Why: distinct from the already-shipped `CDF_MAXSTEP_CLIP` smear fix. 45218's WINNING rig-count
  forecast carries a flat −219.5 log-score zone starting 16 rigs above its declared p99, and the
  same shape made 44182 (−161.6 spot) the standing board #1. On an open bound, mass beyond the
  declared p99 can starve to (near) the 0.001 open-bound floor, so every resolution out there
  scores at the platform's floor — a cliff nobody declared. Receipts:
  `scratch/residual_2026-08-31/dossiers/45218_dossier.md` + its verification (D1 cliff, flat
  −219.5 on 604–630), `SYNTHESIS.md` §6.7, `DOSSIER_SYNTHESIS.md` §7 open items.
- Design, detector half (REQUIRED): in `performance_analysis` (width_monitor.py is the natural
  home — it already walks published CDFs per era), flag published open-bound numeric/discrete
  records where the CDF's mass beyond the declared p99 value is starved: compute
  `tail_mass = 1 − F(declared_p99)` on the open side (mirror for open lower bounds) and flag
  when the bound is open and `tail_mass` is below a named threshold constant (suggested
  `STARVED_OUTER_TAIL_MIN_MASS = 0.005`, i.e. within 5x of the 0.001 structural floor — pick
  the constant, name it, and calibrate it against 45218/44182 (must fire) and 44842 (13% above
  ceiling; must NOT fire) from the archive). Report per-record: tail_mass, declared p99, grid
  ceiling, and the log score a resolution in the flat zone would earn.
- Design, pre-publish half (OPTIONAL — spec here, implement if time permits): a
  `STARVED_OUTER_TAIL` WARN beside `OPEN_BOUND_PILING` in `metaculus_bot/numeric/diagnostics.py`
  (threshold constant in `numeric/config.py`, matching that file's pattern), emitted on the
  published aggregate CDF at publish time, plus a `MarkerSpec`
  (`qid_kind=QID_KIND_QUESTION_ID`). It is a DETECTOR only — any width response stays gated on
  the standing `k_tail` hold (fifth consecutive round; do not touch widening knobs).
- Tests: detector fires on a synthetic starved-tail CDF and not on an honest-tail one;
  archive-shaped fixture reproducing the 45218 geometry; if the WARN half ships, marker parse
  test + a phantom-fire guard (learn from the `CDF_MAXSTEP_CLIP` F1 lesson: the marker must fire
  only on the CDF actually published, not on discarded provisional builds).

**(c) `statuses=closed` in supply probes.**
- Why: two consecutive residual rounds' supply projections missed because the round's probe
  counted only `statuses=resolved/open` while 178 questions sat at Metaculus post status
  `closed` (26 of them past their own `scheduled_resolve_time`). The probe was a scratch script
  (`scratch/residual_2026-08-31/probe_slugs.py`), so the fix keeps getting re-lost. Receipts:
  `SYNTHESIS.md` §3 (frozen-triple checkpoint row) and §7.
- Design: a small TRACKED utility — suggested `scripts/supply_probe.py` — that, given
  tournament slug(s), reports question counts by status INCLUDING `closed` (open / closed /
  resolved, plus a "past scheduled_resolve_time" backlog count with the worst overdue days).
  Read-only Metaculus API, free, no cost-gate concerns. Reuse the API-access patterns already
  in `performance_analysis/collector.py` (note its `statuses: "resolved"` at collector.py:106
  is CORRECT for its purpose — scoring needs resolved only; do not "fix" it).
- Tests: response-fixture test covering the status partition and the backlog computation (no
  live API in CI; the network-egress-blocking conftest fixture will enforce that anyway).

**(d) MarkerSpec for resolution-source per-URL statuses.**
- Why: the per-URL fetch outcomes (`success` / `blocked` / `js_wall` / `stale_data` /
  `empty_body` / ...) currently live in free-text log lines
  (`resolution_source fetched <netloc> (<status> ...)`, e.g. resolution_source.py:541, 584,
  593, 644, 681) and in the comment's provider-diagnostics block — neither is harvested, so
  cuts like "cdc.gov is 0 successes in 1,069 fetch records" require re-scraping run logs that
  expire at 90 days. Receipt: `DOSSIER_SYNTHESIS.md` §7 residual-tooling bullet
  ("resolution-source per-URL statuses need a MarkerSpec", from dim_bot-health rec 3).
- Design: normalize the emission into ONE greppable line per fetched URL —
  `RESOLUTION_SOURCE_FETCH: question=<id> url=<url> status=<FetchStatus|ok> http=<code|n/a>` —
  emitted where the current per-URL info logs fire in `research/resolution_source.py` (the
  question id must be threaded to the fetch layer if it isn't already — verify; if threading
  the qid is invasive, emit one line per URL from the per-question aggregation point instead,
  where `_fetch_result_sources` already assembles the outcome map). Register the `MarkerSpec`
  (`qid_kind` per whichever id the emitter actually has — follow the markers.py comment
  convention and the `tests/test_id_mapping.py` pin). Keep the existing human-readable lines or
  replace them, but do not double-log the same fetch under two formats.
- Tests: marker emitted for success and non-success statuses; parse test; if item 11 ships
  first, `no_resolving_content` appears as a status value in the parse test's fixtures.

### Tier C — research/provider fixes

#### Item 8 — null-result reading clause (forecaster prompts + gap-fill analyzer)

**What/why.** On 44799, 4 of 6 members read "no record found" as "did not happen" — a failed
search licenses only "could not find evidence", and the strength of that evidence depends on
how well-covered the topic is. Receipts: `scratch/residual_2026-08-31/dossiers/44799_dossier.md`,
`DOSSIER_SYNTHESIS.md` §7 (gap-fill null-result bullet: "weaker still where the actor
demonstrated the capability in-jurisdiction").

**Design.** One shared module-level clause constant in `metaculus_bot/prompts.py` (follow the
existing `_SOURCE_PROVENANCE_LADDER` pattern), interpolated into all three base forecaster
prompts — `binary_prompt` (natural home: the PHASE 2 "Evidence weighting" step 4 rubric area,
prompts.py:~750), `multiple_choice_prompt`, `numeric_prompt` — and, phrased for an auditor
rather than a forecaster, into `gap_fill_analyzer_prompt` (prompts.py:1493; a natural
extension of its gap-type list). Substance the clause must carry: (i) a search that found
nothing licenses "could not find evidence of X", never "X does not exist / did not happen";
(ii) weight the absence by coverage — absence from a comprehensive, well-indexed source the
question domain reliably hits is real (weak-to-moderate) evidence; absence from general web
search on a poorly-covered topic is nearly no evidence; (iii) absence is weaker still when the
actor has already demonstrated the capability/behavior in question. Scope: base prompts + the
analyzer only — NOT the stacking prompts (stacking is prod-disabled; keep the diff minimal).

**Tests.** Prompt-content assertions in `tests/test_prompts.py` for all four surfaces
(assert a distinctive phrase from the clause appears; keep the assertion loose enough to
survive rewording of neighboring text).

#### Item 9 — present-tense instrument-query clause (gap-fill analyzer + v2 driver)

**What/why.** The Nebraska/Texas natural experiment (44554 miss / 44556 control — same
template, same day, same roster): Texas's gap-fill asked the present-tense question and got the
tracker's live value; Nebraska burned all three analyzer slots on future-tense, unanswerable
gaps. Second instance on 44799; 44839 wasted 4 of 5 calls the same way. Gap-fill v1 is ~44% of
research spend. Receipts: `DOSSIER_SYNTHESIS.md` §4 (Nebraska pair) and §7;
`dossiers/44554_verification.md`, `dossiers/44556_dossier.md`.

**Design.** SCOPED clause — it applies only when the question resolves off a live data source
(a tracker, index, average, counter, or dashboard):
- `gap_fill_analyzer_prompt` (`prompts.py:1493`): add to the instructions (and/or as a gap
  type): when the question resolves off a live data source, at least ONE gap must ask for the
  CURRENT value of that source, phrased in the present tense ("what does <tracker> read
  today"); and forbid future-tense gaps no search can answer ("what will X be on <date>") —
  rephrase such a gap as its present-tense observable or drop it.
- v2 driver, if applicable: `metaculus_bot/research/agentic/driver_prompt.py` already directs
  market verification (lines ~123-125); add the matching present-tense discipline line where
  the driver plans its searches/fills. Read the driver prompt first — if its existing
  fill/verify framing already forces present-tense targets, add nothing and note that in the PR
  (the item says "if applicable").

**Tests.** Prompt-content assertions (`tests/test_prompts.py`, plus the agentic prompt test
file if one exists — check `tests/` for driver_prompt coverage).

#### Item 10 — financial_data bundle (five fixes in `research/financial_data.py`)

Receipts: `dossiers/44797_verification.md` (pegged-cross vol; variance ratio),
`dossiers/44944_dossier.md` (`.4g` precision; ALFRED vintage lever +66.6 spot ceiling; the
double-count guard), `DOSSIER_SYNTHESIS.md` §7 financial_data bullet, `SYNTHESIS.md` §6.4.

**(a) Pegged/illiquid-cross anchor substitution.** 44797's 17.8% "volatility" on `USDSZL=X`
was 79% vendor noise; the honest like-for-like number off the liquid anchor `ZAR=X` was ~10.6%.
Design: a small static table of hard currency pegs (module constant, descriptive name, e.g.
`HARD_PEG_ANCHORS: dict[str, str]` mapping pegged Yahoo FX tickers to their liquid anchor —
SZL/LSL/NAD→ZAR, HKD→USD, DKK→EUR, XOF/XAF→EUR, AED/SAR/QAR→USD, BND→SGD; source each entry
with a comment). When a fetched ticker matches, ALSO fetch and render the anchor cross's block,
with one line naming the peg relationship and stating that the anchor's series is the reliable
read of the pair's true dynamics. Do NOT silently substitute — render both, labeled. A dynamic
correlation-based peg detector is out of scope (over-engineering; the static table covers the
realized failure class).

**(b) Variance-ratio noise flag.** Detect vendor-noise-dominated series: compute a
variance-ratio statistic over the FULL held history (the 44797 verification §11 shows it cannot
run on the 30-row window — it discriminates 0.47 (noisy) against 0.74 (clean), so the window
must be long enough to separate those). Design: implement the VR estimator in
`research/ts_estimators.py` (beside `annualized_realized_vol_pct`); flag when VR falls below a
named threshold constant calibrated so the noisy series (USDSZL-shaped, ~0.47) fires and clean
series (~0.7+) do not — suggested `FINANCIAL_VARIANCE_RATIO_FLOOR = 0.6`, but calibrate against
fixture series in tests rather than trusting the suggestion. On a flag: print an explicit
noise-flag line in the rendered block and make the LONG-window vol the headline number (the
short-window number stays visible, labeled as noise-suspect).

**(c) Long-horizon vol beside the 30-day print.** In `_yfinance_stats_lines`
(financial_data.py:398-418): add an annualized vol line over the full fetched window (or
1-year, whichever the fetch actually holds) beside the existing
`FINANCIAL_YFINANCE_RECENT_DAYS` line, so a forecaster sees both horizons; keep the existing
estimator (`annualized_realized_vol_pct`) and its sample-size labeling discipline (the file's
comment about "a vol wearing the window's label without its sample size" applies to the new
line too).

**(d) `.4g` precision fix.** `_render_fred_series` (financial_data.py:618-654) prints FRED
levels/changes at `:.4g`, which destroys index levels (Case-Shiller 331.893 → "331.9"; a
question with a [328, 332] displayed range turns on the digits `.4g` throws away). Change the
level lines (latest/previous at :629/:633) and the change lines (:640/:654) to a format that
preserves the series' native precision — `.6g` is the minimal fix; value-aware formatting
(more decimals for large index levels) is acceptable if kept simple. Update any tests pinning
the old rendering.

**(e) ALFRED first-release/vintage lines for FRED-resolved questions.** 44944 resolved on the
FIRST-release Case-Shiller print, and a revision-adjusted anchor was worth +66.6 spot ceiling;
the bundle knew the revision direction but not the first-release convention. Design: for
questions whose resolution source is a FRED series (the deterministic extraction in
`extract_financial_identifiers_from_criteria` already identifies these), fetch the ALFRED
(vintage) view of the recent observations — first-release value vs current-vintage value for
the last few prints — and render a short "first release vs revised" table with the observed
revision direction. **Mandatory double-count guard (stated by the 44944 dossier, "The two
levers still on the table"):** the rendered block must carry an explicit caveat that
revision-direction adjustments and same-source leading indicators (e.g. ICE HPI for
Case-Shiller) partly measure the same underlying data — apply ONE adjustment, not both;
stacking them overshot by −15 spot on the dossier's own arithmetic. Implementation notes:
ALFRED is the same Fred API with `realtime_start`/`realtime_end` parameters (verify against
`research/ts_fetch.py`'s existing FRED client; `FRED_NON_REVISING_SERIES` already exists there
— skip the vintage table for series on that list, they cannot revise).

**Tests.** Fixture-driven, no network (the conftest egress block enforces this): peg table
triggers the anchor fetch + labeled render; VR flag fires on a synthetic noisy series and not
on a clean one, and flips the headline vol; long-horizon line present; `.4g` regression test
pinning full precision on a Case-Shiller-scale level; ALFRED table rendered with the
double-count caveat, skipped for `FRED_NON_REVISING_SERIES`; existing
`tests/test_financial_data_provider.py` / `tests/test_financial_data_integration.py` stay
green.

#### Item 11 — `no_resolving_content` status for embed-shaped 200s

**What/why.** 44554/44556's polling tracker pages returned HTTP 200 whose extractable content
was an Infogram embed shell — the resolving numbers live inside an iframe trafilatura drops —
and the fetch reported `success`, making a tracker-family failure invisible. A dedicated status
converts it into a query. Receipts: `dossiers/44554_verification.md` /
`dossiers/44556_dossier.md` (Infogram), `DOSSIER_SYNTHESIS.md` §7.

**Design.**
- Add `"no_resolving_content"` to the `FetchStatus` literal
  (`metaculus_bot/research/resolution_fetch_result.py:35-45`), with a comment defining it: an
  HTTP-200 page whose extracted text is embed scaffolding — the content exists but lives inside
  a third-party embed (iframe) our extractor cannot read. Distinct from `empty_body` (nothing
  there at all) and from `js_wall` (page assembled by JS); like `blocked`/`js_wall` it IS a
  Tier-2 escalation seam (the item 13 design doc's ladder should list it).
- Detection at extraction time in `research/resolution_source.py`: when the RAW HTML contains
  known embed iframes (infogram, datawrapper, flourish, tableau — reuse/extend the existing
  Datawrapper chart-ref scan, which already parses embeds out of raw HTML) AND the
  trafilatura-extracted text is vacuous-or-near-vacuous outside the embed scaffolding (short
  text floor and/or embed-boilerplate patterns), assign `no_resolving_content` instead of
  `success`. Tune the floor against the archived 44554/44556 pages if recoverable from
  `backtests/research_archive/raw/`; otherwise construct the fixture from Infogram's standard
  embed shell.
- Propagate: `_fetch_result_sources` keeps the status verbatim as a loss token (it does this
  for every non-success automatically — verify the docstring list at
  resolution_fetch_result.py:143-159 gets the new status added); the section formatter's
  "resolving page was unreachable" notice must treat it as a non-success; provider diagnostics
  carry it; item 19d's marker inherits it.
- A page carrying a Datawrapper embed that the Tier-2 hop successfully serves should NOT
  regress: the Tier-1 page result may be `no_resolving_content` while the Tier-2 dataset result
  is `success` — that combination is correct and must render the dataset.

**Tests.** Infogram-shell fixture → `no_resolving_content`, not rendered as grading evidence,
counted in the lost tokens; ordinary article 200 → `success` (no false positives on pages that
merely CONTAIN an embed beside real prose — a fixture with substantial prose + one iframe stays
`success`); Datawrapper page keeps its Tier-2 dataset path; the `FetchResult.__post_init__`
blank-success guard still holds.

#### Item 12 — ranker staleness disclosure (market_retrieval)

**What/why.** 45163's rank-0 market closed FIVE MONTHS before the forecast and still graded
`same_quantity_same_date`; across the resolved ranked-era six, Kalshi's `close` field trails
actual settlement by a median +317 days, so close dates read as resolution dates when they are
not. Operator verdict: DISCLOSURE, not drop. Receipts: `SYNTHESIS.md` §6.8,
`DOSSIER_SYNTHESIS.md` §5 (market bullets), `dim_market-informativeness.md`.

**Design — four small pieces:**
1. **Row-level staleness disclosure (rendering.py).** Any rendered row (parent or child) whose
   `close_time` precedes the forecast time shows it plainly — e.g. the close cell reads
   `closed 152d ago` instead of a bare date, or the date plus `(N d before forecast)`. Site:
   `market_retrieval/rendering.py` `_priced_cells` (close cell construction, ~line 346-373).
2. **Within-tier recency tiebreaker (ranker prompt, ranking.py).** The render-verbatim contract
   (renderer shows the ranker's order; no deterministic re-sorting — that was a measured design
   decision) means the tiebreaker belongs in the RANKER PROMPT: add an instruction that between
   candidates of equal relation tier, prefer open/recently-active markets over long-closed
   ones. The candidate segments already carry `closes: <date>` (ranking.py:253), so the model
   has the input.
3. **Hard top-tier exclusion for egregious staleness only (ranking.py, deterministic).** After
   the LLM ranking, a kept row whose `close_time` predates the QUESTION's open time by more
   than a named constant (suggested `MARKET_STALENESS_TIER_CAP_DAYS = 60`; 45163's offender was
   ~5 months) cannot keep the `same_quantity_same_date` tier: cap its relation tier one rung
   down and annotate the why/relation cell so the demotion is visible (e.g.
   `same_quantity_other_cut (stale: closed 152d before question opened; ranker said
   same_quantity_same_date)`). Keep the row and its rank position — disclosure, not drop.
4. **Kalshi close-vs-settle caveat (rendering.py legend).** One line in the table legend text
   (the block around rendering.py:214-255): Kalshi close dates are the venue's trading close,
   which is routinely long before settlement (median observed gap ~10 months) — do not read a
   close date as a resolution date. Note `venues/kalshi.py`'s own header already documents that
   `close_time` is a max-over-strikes derivation; the caveat is for the forecaster, who never
   sees that code comment.

**Tests.** Rendering fixture with a stale row shows the disclosure; tier-cap fires on an
egregious fixture (close 5 months before question open) and NOT on a fresh/open market; the
annotation preserves the ranker's original tier in text; legend contains the Kalshi caveat;
ranker prompt contains the tiebreaker line; existing rendering/ranking test files
(`tests/test_market_retrieval_rendering.py`, `tests/test_market_retrieval_ranking.py`) stay
green — extend them rather than adding parallel files.

#### Item 13 — fetch escalation ladder: DESIGN DOC ONLY

**What/why.** cdc.gov is 0 successes in 1,069 archived fetch records; Baker Hughes was readable
ONLY via Gemini url_context (every direct egress timed out); 44872's resolving count sat one
unauthenticated JSON GET away behind the page's own Mapotic JS config. A Tier-2 escalation
design has enough moving parts (and enough spend/latency implications) that the operator wants
a doc for separate approval, NOT code in this bundle. Receipts: `DOSSIER_SYNTHESIS.md` §6
pattern 11 and §7 (Tier-2 escalation bullet), `SYNTHESIS.md` §6.2,
`dossiers/44872_verification.md` (Mapotic), `dossiers/45218_verification.md` (Baker Hughes /
egress-vs-fetcher distinction), `dossiers/44873_verification.md` (meta-refresh label/value
pairing).

**Deliverable.** `scratch_docs_and_planning/fetch_escalation_ladder_design.md`, self-contained,
covering at minimum:
- The ladder: direct fetch → page-JS-config API discovery (read the fetched page's inline JS
  config for an API base and issue the derived unauthenticated GET — the Mapotic pattern, and
  the same shape as the shipped Datawrapper CSV hop in `resolution_source.py`) →
  url_context/read_document escalation (runs on the personal Google key, so it survives
  donated-key outages).
- Which `FetchStatus` values are escalation seams (`blocked`, `js_wall`, `error`, and item 11's
  `no_resolving_content`) and which are terminal (`empty_body`, `unsupported_type`,
  `not_found`, `ssrf_blocked`, `stale_data`).
- Failure-reason recording that distinguishes "our fetcher failed" vs "our egress failed" vs
  "host down" (the 45218 lesson: all four direct fetches timed out while Google's fetch
  worked — an egress property, not a fetcher bug).
- The meta-refresh rider: following a meta-refresh must preserve label/value pairing (44873's
  fix works; done naively, 44874's page yields a bare "2" with no label).
- Cost/latency envelope per rung, the SSRF posture for derived-API GETs (the existing
  `is_public_http_url` + `FilteringResolver` boundary must guard derived URLs too), and the
  url_context positive-control probe as a listed prerequisite
  (`scratch/urlctx_probe_2026-08-03/probe.py` — a paid two-call probe, operator-gated).
- Explicitly: what evidence would justify each rung, and a validation plan that respects the
  cost gate.

No code, no tests. The doc ships in the same PR (it is inert).

### Tier B — gemini search provider (receipts: `scratch/residual_2026-08-31/gemini_search_audit/`)

Background for this tier, in one paragraph: the provider audit confirmed both q44872
"expeditions" were false — a stale-source year swap, real 2021/2022 OCEARCH material restamped
2026, with the invented window landing exactly inside the resolution period — while the overall
verdict was KEEP the provider and KEEP the zero-chunk suppression floor (fabrication 1 of 11
suppressed payloads; embellishment, not invention, is the channel; the section is the bundle's
densest unique-quantity source and forecasters absorb its numbers at 3x native's rate). The
fixes below are the audit's shipping list, operator-approved. Read `VERDICT.md` §§1–3 before
touching `gemini_search.py`.

#### Item 2 — vintage/as-of clause in `web_research_prompt`

**What/why.** The 44872 mechanism was parametric recall with a broken clock: the model
searched correctly, Google attached nothing, and it answered from memory of 2021/2022 press
releases restamped as 2026 plans. The prompt has "say so explicitly" and "DO NOT hallucinate
sources" and no date discipline at all (verified against the file by the audit). Receipt:
`gemini_search_audit/VERDICT.md` §1 ("The mechanism") and §3 ship item 2.

**Design.** Add to the GUIDELINES block of `web_research_prompt`
(`metaculus_bot/prompts.py`, the block around lines 278-283): every dated or forward-looking
claim must carry its source's publication date ("announced <date>", "published <date>");
forward-looking schedules/plans must state WHEN and WHERE they were announced; never present an
undated recollection as current fact. This clause is shared by BOTH consumers of the prompt
(native search via `research/providers.py:524`, gemini via `research/gemini_search.py:331`) —
that is intended; native's two visibly-stale market emissions would have been self-labelling
under it too (market_odds_coverage.md §5). Note the same fix family already exists precedent:
the AskNews summarizer's supersession rules at `prompts.py:375-376`.

**Tests.** Prompt-content assertion in `tests/test_prompts.py` (both citation styles).

#### Item 3 — strip model-authored hierarchical citation markers

**What/why.** Half of all archived gemini sections carry TWO citation systems, one fake: our
formatter splices real `[N]` markers from grounding metadata, and gemini also writes its own
hierarchical `[x.y.z]` indices resolving to nothing we hold — 173 of 323 archived sections,
2,504 markers, and a forecaster cannot tell which brackets are checkable. Receipts:
`gemini_search_audit/VERDICT.md` §2 ("One new undocumented defect"),
`cutB_pattern.md` §3.1.

**Design — two halves:**
1. **Formatter strip** in `metaculus_bot/research/gemini_search.py`. Placement is load-bearing:
   `_splice_inline_citations` (gemini_search.py:126) indexes into the ORIGINAL response text by
   grounding-support byte offsets, so the strip must run AFTER splicing (and after
   `_render_sources_section`), never before — stripping first would shift every offset. Our
   spliced markers are plain `[N]` (single integers, no dots), so a dotted-token pattern cannot
   collide with them. Strip rules: remove dot-separated integer index tokens
   (`\d+(?:\.\d+)+`-shaped) from within square-bracket groups; bracket groups may hold several
   comma-separated tokens and may MIX tokens with tier tags (`[A: NASA, 1.1.2]` → `[A: NASA]`);
   a bracket group left empty after token removal is removed entirely; normalize leftover
   separators/whitespace. Apply on the grounded (passed-floor) path; the suppressed path never
   renders, so it needs nothing.
2. **Prompt line** telling the model not to emit its own citation indices. Home: the
   auto-annotate branch of `citation_clause` in `web_research_prompt` (prompts.py:~250-254) —
   the gemini-specific branch, so native's markdown-citation instruction is untouched.

**MANDATORY validation step (a gate, free/offline).** Before shipping, run the strip over ALL
archived gemini sections (the audit counted 323 sections carrying the
`## Web Research (Google Search via Gemini)` header across `backtests/research_archive/latest/`
artifact-class records; `gemini_search_audit/attribution_gap.py` shows how the corpus was
enumerated) and hand-review every bracketed match that is NOT a citation index. The named
false-positive risk: two-component numerics like `[1.5]` (a legitimate bracketed number matches
"two dot-separated integers"). Measure the false-positive rate; it must be ~0 before shipping.
If two-component tokens produce ANY false positives, restrict the pattern to ≥3 components
(`[x.y.z]`, which covers every example the audit quotes) and/or to comma-grouped/tier-tag
contexts, re-run, and document the final rule + counts in the PR and in a validation output
directory (suggested `scratch/next_season_bundle_2026-09/citation_strip_validation/`). The
tests below must encode whatever the validation decided.

**Tests.** Strips `[2.4.1]`; strips multi-token groups (`[1.1.1, 1.1.2]`); mixed group
`[A: NASA, 1.1.2]` → `[A: NASA]`; preserves spliced `[1]` / `[12]`; preserves legitimate
bracketed numerics per the validation findings (e.g. `[1.5]` if the ≥3-component rule was
adopted); idempotent (stripping twice = stripping once); splice-then-strip integration test on
a fixture with grounding supports.

#### Item 4 — unsupported-attribution check (spec carefully — touches bundle content)

**What/why.** 87% of the outlets gemini names in its self-invented source-tier tags (276 of
318, across 51 of 311 archived sections) are absent from the same response's own
grounded-domain list — e.g. q44953 claims `[A: NASA]` while its Sources list is perlan.is /
guidetoiceland.is / timeanddate.com. The floor cannot touch this (it fires only on zero
chunks); a deterministic format-time check can. Receipts: `gemini_search_audit/cutB_pattern.md`
§3.2, `VERDICT.md` §2 (embellishment channel) — the 87% figure is cutB's measurement,
reproduced exactly by the VERDICT's cross-check.

**Design.** Deterministic, offline, at format time in `gemini_search.py`, running on the
grounded path after item 3's strip:
- **Extract asserted attributions.** Primary surface (the measured one): outlet names inside
  tier tags — `[A: <Outlet>]`, `[B: <Outlet>]`, `[C: <Outlet>, ...]`. Secondary surface (prose
  patterns like "per <Outlet>", "according to <Outlet>"): include ONLY if the corpus validation
  shows the extraction is precise; otherwise ship tier-tags-only and record the prose surface
  as out of scope. Generic tier words (`official`, `wire service`, `government`, ...) are
  skipped, as in the audit's measurement.
- **Compare against the response's own grounded-domain list** (the domains behind
  `metadata.grounding_chunks`, i.e. what `_render_sources_section` renders). Matching is
  CONSERVATIVE and token-loose: normalize case/punctuation/whitespace; an outlet matches if its
  significant tokens appear in any grounded domain ("Golf Channel" → golfchannel.com credits as
  supported). When in doubt, KEEP the attribution — false strips are worse than false keeps
  (an absent outlet does not prove the fact wrong; Google's chunk attribution can name an
  aggregator domain while the text names the original outlet — cutB says this explicitly).
- **Action on unsupported:** rewrite the tier tag's outlet to `[unverified attribution]`
  (keeping any surviving supported outlets in the same bracket), or strip the tag when nothing
  survives. Never touch the factual sentence itself — only the attribution decoration.
- **Diagnostics:** emit the per-section count into the provider diagnostics
  (`record_provider_detail(qid, "gemini_search", ...)` already exists on the suppression path —
  add e.g. `{"unsupported_attributions": N}` on the grounded path), so the incidence is a query.

**MANDATORY validation step (a gate, free/offline).** Run the check over the same 323-section
archive corpus as item 3 (reuse `attribution_gap.py`'s extraction as the starting harness — it
already parses tier tags and grounded domains per section; the plan expects you to adapt it,
not rewrite it). Report in the PR: sections touched, attributions rewritten/stripped/kept,
and a hand-review of a sample (minimum: every section where ALL attributions would be
rewritten, expected ~30 per cutB) checking for false strips — i.e. cases where the outlet IS
plausibly the true source despite being absent from the grounded domains. The check's matching
rules get loosened until the hand-review shows false strips at ~0; the kept/rewritten counts at
the final rule go in the PR and the validation directory.

**Tests.** Supported outlet kept (exact and token-loose: "Golf Channel"/golfchannel.com);
unsupported tier tag rewritten to `[unverified attribution]`; mixed bracket keeps the supported
outlet and rewrites the unsupported one; generic tier words untouched; prose sentences never
modified; diagnostics count emitted; interaction test with item 3 (dotted indices stripped
first, attribution check still sees the tier tag).

#### Item 5 — market-odds bullet narrowing (`prompts.py:266`) — wording CONFIRMED by operator 2026-09-01

**What/why.** In 42 ranked-era bundles, the bullet's covered-venue half produced exactly one
content-redundant retrieval and THREE stale covered-venue prices that contradicted correct live
snapshot rows (the only measured harm mode), while every realized instance of the operator's
fear scenario — rare decisive market evidence — came from venues OUTSIDE
Polymarket/Kalshi/Manifold/PredictIt (GJO q44869, CME FedWatch q45401, Metaculus q20683).
Narrow the bullet away from covered venues; do not remove it. Receipt:
`scratch/residual_2026-08-31/market_odds_coverage.md` (whole doc; §5 for the recommendation and
why it beats both keep-as-is and full removal).

**Design.** Replace `prompts.py:265-267`'s

```
- Prediction market odds and forecasts (if available)
```

with the text below — **wording confirmed verbatim by the operator on 2026-09-01; commit as
written**:

```
- Market-implied or crowd odds from sources OTHER than Polymarket, Kalshi, Manifold, or
  PredictIt (e.g. Metaculus, Good Judgment Open, CME FedWatch, bookmakers) — always name the
  market and the date you observed the price. Do NOT report Polymarket/Kalshi/Manifold/
  PredictIt prices from search results: a dedicated live snapshot of those venues is provided
  separately, and search-indexed copies of their prices are usually days stale.
```

Blast radius (verified by the receipt's §0): the bullet reaches exactly two consumers — native
search and gemini. Gap-fill v1/v2 never see it; benchmarking mode already suppresses it
(`is_benchmarking` branch) and must continue to.

**Tests.** Prompt test: new bullet text present when not benchmarking; absent (as today) when
benchmarking; the date-stamp requirement phrase present (it is the item-2 vintage clause
landing on this surface too — the receipt calls this out).

#### Item 7 — grounding-density telemetry (NOT a gate)

**What/why.** The floor answers "did anything ground?" while most sentences carry no
attribution: post-floor, 41% of passing responses carry ≤3 grounding supports and the median
response has one support per ~872 chars — which is where the ~33% embellishment rate lives. A
density MARKER makes "did embellishment move" measurable on a defined cohort without a hand
audit. It is explicitly NOT a gate: q44944's decisive, true, gap-fill-verified ICE figure came
out of a 1-support response, so a density gate would have suppressed the round's best find.
Receipts: `gemini_search_audit/VERDICT.md` §2 (floor-immune surface) and §3 ship item 3.

**Design.** In `_format_grounded_response`'s success path (`gemini_search.py:~254`), emit one
INFO line per grounded response:
`GEMINI_GROUNDING_DENSITY: question=<qid> chunks=<n_chunks> supports=<n_supports> chars=<len(text)>`.
Emit unconditionally on the grounded path (not only below a threshold — a threshold is a gate's
first half; the archive can compute any threshold later). Register the `MarkerSpec`
(`qid_kind` per what the formatter's `qid` actually is — check the caller in
`invoke_gemini_grounded`; follow the markers.py comment convention). No behavior change of any
kind keyed on the values.

**Tests.** Marker emitted with correct fields on a grounded fixture; NOT emitted on the
suppressed path; parse test in `tests/test_telemetry_markers.py`;
`tests/test_gemini_search_provider.py` stays green.

### Tier A — publish path (build LAST; adversarial review before merge)

#### Item 1 — k-conditional publish floor for thinned binary publishes

**What/why.** When the ensemble thins to ONE surviving forecaster, the median that absorbs a
member's extreme tail call does not exist — q44874 published gemini's lone 3% on a question
that resolved YES and took −105.27 spot peer. A floor of `[0.05, 0.95]` on the PUBLISHED binary
probability, applied ONLY when the survivor count is exactly 1, is worth +51.08 spot peer on
q44874 with ZERO measured cost on any other archived record (at k≥2 the median self-clamps; on
the 4 archived solo binaries it is one win and three exact zeros). Always-on and global
variants were priced and REJECTED (always-on never improves the published pre-flip ensemble;
global `[0.05,0.95]` over 408 binaries is −52.02, 50 losses to 1 win) — do not widen the
trigger. The floor is structural, not fitted: median-of-1 supplies no variance reduction, so
widening the admissible range in exactly that state prices the missing aggregation. Receipt:
`scratch/residual_2026-08-31/gemini_review/RECOMMENDATION.md` §2 (clamp-variant table + "A
synthesis correction the individual cuts miss") and §3 option 1=. Operator: "let's be careful
to get it right."

**Honest caveat to carry into the code comment and PR:** the entire measured benefit is one
question (n=4 solo binaries, one non-zero row), and the specific value 0.05 is informed by that
question; the neighboring values 0.07/0.10 buy more on 44874 but start taxing 44870/44873,
which were right. Bounded downside: the maximum cost per instance is publishing 5% where a
correct sub-5% call would have scored, ≈ −3.11 spot peer, against a −105 tail.

**Site (verified during planning — this corrects the brief's suggested site).** The brief
suggested the single-forecaster short-circuit in `stacking_route.py:382-392`. That branch
handles the right EVENT but the wrong OBJECT: it returns the lone `ReasonedPrediction` inside
the predictions collection, and that same object feeds the comment's per-model summary bullet —
clamping there would rewrite the per-model record, which the spec forbids ("NEVER per-model":
the published aggregate is clamped; the member's declared value stays on the record). The
published value actually materializes in **`AggregationPipeline._base_combine`'s
`len(predictions) == 1` branch (`metaculus_bot/aggregation_pipeline.py:264-275`)** — implement
there. Gating: that same branch ALSO serves the pre-stacked STACKING single-output path (the
in-code comment says so), so `len == 1` alone is NOT the trigger. The correct trigger is
`self.skip_reasons.get(qkey) == "single_forecaster"` — that key is written by
`route_after_forecasts` exactly when `len(valid_predictions) == 1`
(`stacking_route.py:382-392`), which is the SAME count `FORECASTERS_SURVIVED` logs
(`forecaster.py:843/886`), satisfying the "read the actual survivor count from the same source"
requirement; and it is never written for a stacked output, so the stacker path cannot be
clamped by accident. The pipeline and the bot share these dicts by construction
(`forecaster.py:336-346` — `_stacker_skip_reason` IS `self._pipeline.skip_reasons`).

**Ordering fact to verify with a test, not by reading:** `skip_reasons[qid]` is written in
`route_after_forecasts` (during `_research_and_make_predictions`) and popped at
comment-building time (`forecaster.py:963`); the framework's aggregation (which calls
`_base_combine`) runs between the two. Write an integration-style test that drives the real
sequence (the existing single-survivor tests in `tests/test_conditional_stacking.py` /
`tests/test_template_forecaster.py` show how) and asserts the floor fired — if the ordering
assumption is wrong, this test is what catches it.

**Design details.**
- **Binary only.** The branch must check `isinstance(question, BinaryQuestion)` (the len==1
  branch also carries lone numeric/MC survivors — snap-to-integers already handles discrete
  there; leave MC/numeric untouched, explicitly do NOT extend).
- **Constants:** `THIN_PUBLISH_BINARY_FLOOR: float = 0.05` and
  `THIN_PUBLISH_BINARY_CEIL: float = 0.95` in `metaculus_bot/constants.py`, directly beside
  `BINARY_PROB_MIN`/`BINARY_PROB_MAX` (constants.py:366-367), with a comment naming the
  mechanism (median-of-1 has no variance reduction; fires only on single-survivor publishes;
  evidence = one question, q44874, +51.08 spot with zero measured cost; always-on rejected —
  cite the receipt path) and noting the values deliberately equal the item-15
  `EXTREME_CALL_LOW/HIGH` band.
- **Write the clamp as a tiny helper taking the survivor count** (e.g.
  `apply_thin_publish_floor(value: float, survivors: int) -> float`), so the k≤2
  generalisation the receipt discusses is a one-line change later — but wire it at k==1 only,
  and do not tune anything without data (a k=2 publish has never happened; there is nothing to
  fit).
- **Marker:** on an actual clamp (value moved), emit a WARN:
  `THIN_PUBLISH_FLOOR: question=<id_of_question> raw=<raw> clamped=<clamped> survivors=1`.
  No line when the lone value is already inside the floor (silence = nothing moved; the
  single-survivor EVENT is already observable via `FORECASTERS_SURVIVED` + skip_reason).
  Register the `MarkerSpec` in `scripts/telemetry/markers.py`
  (`qid_kind=QID_KIND_QUESTION_ID`, matching `forecasters_survived`).
- **Interaction notes for the implementer:** the existing per-model clamp to
  `[BINARY_PROB_MIN, BINARY_PROB_MAX]` = [0.02, 0.98] happens upstream in
  `forecaster_runners.py:155` — a 3% call passes it untouched, which is why the floor is a new
  mechanism, not a retune of the old one. The strategy gate: `skip_reasons` is only written
  under STACKING/CONDITIONAL_STACKING (`_STACKING_STRATEGIES`); a single survivor under a plain
  MEAN/MEDIAN strategy routes through `_simple_aggregate` and will NOT be floored — prod and
  the code default both run CONDITIONAL_STACKING, so this gap is accepted; state it in the code
  comment rather than papering over it with a second wiring.

**Tests (all four required by the operator's spec, plus the guards above):**
1. Fires at k=1 outside bounds, both sides: lone binary 0.03 → published 0.05 (marker WARN with
   raw/clamped), lone 0.97 → 0.95.
2. Inert at k=1 inside bounds: lone 0.30 → 0.30, no marker line.
3. Inert at k≥2 even with extreme members: e.g. members [0.03, 0.10, 0.15] → median 0.10
   untouched; AND the sharper case — members [0.02, 0.03, 0.04] → median 0.03 published AS IS
   (the floor must never touch a multi-member publish, even one below 0.05).
4. Marker parse test (`tests/test_telemetry_markers.py`).
5. Stacked-output guard: under STACKING (or a fired CONDITIONAL stack), the single pre-stacked
   output passing through `_base_combine` len==1 is NOT clamped (no `single_forecaster` skip
   reason present).
6. Per-model record integrity: on a clamped k=1 publish, the comment's per-model summary bullet
   still carries the RAW member value (0.03), while the published prediction is 0.05.
7. Non-binary lone survivor untouched (numeric lone survivor keeps its snap path; MC lone
   survivor unchanged).

**Adversarial review (required before merge).** After item 1 is green: run `/forge` over the
branch; separately hand item 1's diff to a codex (GPT-5.x) review with a tight spec ("find
cases where this clamp fires when it must not, or fails to fire when it must — trace every
caller of `_base_combine` and every writer of `skip_reasons`"). Triage findings per the repo's
never-auto-apply rule. This is the "let's be careful" item; budget real time for it.

---

## §4. Verification

**Free gates, run at every commit boundary and before the PR:** `make test`, `make lint`,
`make typecheck` (basedpyright, 0 errors), `make lint_imports`, `make deps`. `make cov` before
the PR. The suite is self-contained (network egress is blocked by an autouse fixture), so
nothing here can spend.

**Corpus validations (items 3 and 4)** are shipping gates, free and offline, defined in their
item specs: the citation-strip false-positive review and the attribution-check false-strip
review, both over the ~323 archived gemini sections. Their final rules, counts, and sample
reviews go in the PR description and a validation output directory.

**What a paid run would add, and who decides.** One `test_bot_basic.yaml` dispatch
(~$2.60, one numeric question, publishes to Metaculus) after the bundle lands would (a)
live-exercise the new prompt surfaces and markers end-to-end, and (b) settle **forge R2** — the
open question of whether clipped CDF bins can difference to 1 ulp ABOVE the platform's max-step
cap on a REAL submission (`forge_plan_resids-sept1.json` reportOnly R2: measured
0.20000000000000007 locally; q45065 published at exactly 0.2, so equality is tolerated;
equality-plus-one-ulp is unknowable offline; do NOT ship a speculative 1e-12 haircut). This is
an OPTION for the operator, not a step in this plan — surface it, with the command
(`gh workflow run test_bot_basic.yaml --repo No-Stream/metaculus-bot`) and cost, and stop. Also
report-only **R1** (closed-bound fallback skips `safe_cdf_bounds`; compound-rare, stacking
prod-disabled) stays accepted-with-tracking per the 2026-08-31 operator decision — do not fix
it in this bundle.

**Instruction-drift check before declaring done:** re-read the operator's item list (§3
headers) and confirm each item is either implemented+tested, delivered as its specified doc
(13, possibly 16), or explicitly closed as already-shipped-with-guards (14). Confirm nothing
from §6 crept in.

## §5. What the operator still decides

1. **Item 5 wording.** The replacement market-odds bullet text (quoted in item 5) is the
   recommendation, marked pending; confirm or amend before the `prompts.py:266` edit is
   committed.
2. **The optional paid smoke test** (`test_bot_basic`, ~$2.60, publishes): fires only on the
   operator's explicit go, after merge or on the branch as they prefer. It would also settle
   forge R2 (the 1-ulp-over-cap question). Nothing in this plan runs it.
3. **Merge timing.** One merge, before the fall cup opens questions (~2026-09-20). The
   operator owns the merge (and the `cr`/PR-approval step); the implementing session prepares
   the branch and asks.
4. **Item 13's design doc** is a deliverable for SEPARATE approval — the operator decides
   whether/when the ladder gets built; nothing in this bundle implements it.
5. **Item 16's shape** if the investigation hits a wall: accept the spec doc, or direct further
   work.

## §6. Explicitly OUT of scope (do not "helpfully" add)

Each of these was considered this round and the operator excluded it from the bundle. Do not
implement, and do not partially implement under another item's umbrella:

- **rate×exposure prompt clause** (44874's #1 cheapest fix — excluded from this bundle).
- **cadence-vs-ascertainment-lag prompt clause** (44874's #2 — excluded).
- **Disagreement-triggered re-forecast** (the 45244 drowned-dissenter design question — a
  design discussion, not a bundle item).
- **Zero-chunk gemini retry** (VERDICT ship item 4 — excluded here).
- **A 4th roster member** (arity is real and small; blocked on candidates and the era clock —
  `gemini_review/RECOMMENDATION.md` §3 option 4).
- **Always-on gemini clamp** in any form (rejected on the numbers; the k==1 floor is the whole
  ship — RECOMMENDATION §3 options 6/7).
- **Gap-fill v1 recency rule** (skipped — v1 retirement is pending its own decision).
- **Span-level grounding filter** on gemini output (telemetry first — item 7 — escalate only
  on evidence).

Also out by standing decision: any width/calibration knob (`k_tail` hold, fifth round), any
`MIN_FORECASTERS_TO_PUBLISH` change (both directions refused on receipts), forge R1's
restructuring, and everything in `DOSSIER_SYNTHESIS.md` §7's "Withdrawn / refuted by
verification" list.

## §7. Receipts index

All under `scratch/residual_2026-08-31/` unless noted (gitignored; read by direct path):

| receipt | backs items |
|---|---|
| `SYNTHESIS.md` §6 (+§3 width row, §5) | 8, 9, 10, 12, 19a–d, sequencing context |
| `DOSSIER_SYNTHESIS.md` §7 (+§4, §5, §6) | 8, 9, 10, 11, 12, 13, 19b/d |
| `gemini_review/RECOMMENDATION.md` | 1, 14, 15, 16, 17, 18 |
| `gemini_search_audit/VERDICT.md` | 2, 3, 4, 7 (tier B background) |
| `gemini_search_audit/cutB_pattern.md` (+`attribution_gap.py/.json`) | 3, 4 |
| `gemini_search_audit/cutC_harm_benefit.md` | 5 (gemini channel), 7 |
| `market_odds_coverage.md` | 5 |
| `forge_plan_resids-sept1.json` reportOnly R1/R2 | §4 (smoke-test option; R1 accepted) |
| `dossiers/44797_verification.md`, `dossiers/44944_dossier.md` | 10 |
| `dossiers/44554_verification.md`, `dossiers/44556_dossier.md` | 9, 11 |
| `dossiers/44799_dossier.md` | 8 |
| `dossiers/45218_dossier.md` + verification, `dossiers/44872_verification.md`, `dossiers/44873_verification.md` | 13, 19b |
| `dim_market-informativeness.md` | 12 |
| `q45065_capbug.md` | context for 19b (smear vs starved-tail distinction) |
