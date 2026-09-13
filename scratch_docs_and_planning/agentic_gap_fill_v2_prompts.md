# Gap-Fill v2 — Driver Prompt & Tool Descriptions (mirror of shipped code)

Status: **Rev 4**, 2026-07-22 (reconstructed from shipped code). These live in `metaculus_bot/research/agentic/driver_prompt.py` (system + user brief + ghost builders), the internal tool schemas in `loop.py` `_internal_tool_schemas()`, and the external ToolSpec descriptions in `tools.py`. The code is the source of truth; the transcriptions below are verbatim from it.

Rev-numbering note: the W3 and W4 agents each wrote interim "Rev 4" drafts to their worktrees; those copies were deleted with the worktrees (this directory is gitignored, so nothing survived). This Rev 4 is reconstructed from the shipped code and supersedes both.

## Rev 4 changes (Workstreams W2-W4)

1. **W2 — conclude gate + verification motivation** (`882cd77`). Two prompt changes:
   - A new ~150-word "Why this matters" block right after the opening paragraph: findings are treated as ground truth, a wrong fact does more damage than a missing one (the unverified-snippet-as-correction ensemble-swing incident), unconfirmed snippets are liabilities, pull the primary source before contradicting the briefing, and depth on 2-3 forecast-moving gaps beats a shallow sweep of ten.
   - STEP 3 rewritten around the conclude gate: `conclude` now REQUIRES a `gap_accounting` list (one entry per plan gap: `gap_id`, `actions_taken`, `status` ∈ `resolved` / `unresolved_parked` / `not_decision_relevant_on_inspection`). An EARLY conclude is rejected when any plan gap is missing from the accounting, external tool calls < plan gaps, or the fetch floor is unmet (neither top-ranked gaps' accounting citing a fetch/read_document action nor ≥2 fetches/reads total). A forced deadline conclusion is exempt ("the floor is what 'done' looks like, not a toll"). Rev 3's cheap-question fast path ("conclude immediately after a spot-check") was reworded into the gate block as a per-gap *fetched* spot-check recorded as that gap's action.
   - Code side (`loop.py` `_evaluate_conclude_gate` / `_conclude_gate_debts`): gate bypasses entirely on forced deadline, `plan_skipped`, no plan gaps, or once `max_conclude_gate_rejections` (=2) is hit; the fetch floor's per-gap clause reads the top-2 ranked gaps' `actions_taken` free text for a fetch/read mention, the global clause counts ≥2 fetch+read_document calls. Telemetry: `conclude_gate_rejections` added to the `GAP_FILL_V2` marker.
2. **W3 — derivation license, STEP 2 "YOU MAY DERIVE"** (`ff405ad`). New STEP 2 clause: findings may carry an optional `derivation` field holding arithmetic-only synthesis over the same finding's quoted numbers (a bound, a rate, a reconciliation of two metrics; exemplar: a per-year bound table derived from a quoted oldest-verified-human-by-year record). Every input number must also appear as a quoted value with its URL in the same finding; no likelihood language, no new facts, no read on the outcome; derived findings are labeled to the panel as our synthesis. Schema: `derivation` is an optional string on findings (in both `record_findings` and `conclude.final_findings`); the detachment lint exempts it while still scanning claim/topic, and the artifact renders it under a "Derived analysis (arithmetic from quoted sources)" label.
3. **W4 — snippet-discrepancy demotion** (`ef9192c` + merge `01daa3e`). The DISCREPANCY rule gains a closing warning: a discrepancy sourced only from search snippets is demoted to "possible corrections" and will NOT supersede the briefing — fetch the primary source first if you intend to contradict the briefing. Code side: the loop stamps a per-finding `verification_tier` ("fetched" vs "snippet") from retrieval provenance (fetch/read_document grants "fetched"; search/news results are "snippet"; a blocked/403 fetch never grants "fetched"), and `artifact.render_findings` splits corrections into "### ⚠ Corrections to the briefing" (superseding) vs "### ⚠ Possible corrections (snippet-sourced — recheck advised)" (demoted).

Also folded into the transcriptions below — shipped-code deltas that predate or ride alongside W2-W4 and were never mirrored here:

- **BTF-2 additions** (`d8c77ff`, 2026-07-19): the BASE-RATE bullet's regime-check extension ("check whether the process that generated the class still holds") and the new CATALYST targets bullet (search the calendar around the question, not just its entities; a dated "no scheduled catalyst found" is a finding).
- **Source-tier RULES bullet** (shipped since the initial wiring commit `f52133f`, never in this doc): where a source's tier is obvious, note it inline using the panel's A-D vocabulary.
- **`fetch` metaculus.com ban** (`a6c8372`): "Do NOT fetch metaculus.com URLs — the question brief already reflects them."

## Rev 3 changes (Workstream W1 — turn-one research plan)

1. **STEP 1 rewrite**: the private dry run stays in the driver's reasoning, but its outputs are now emitted through a new internal tool, `set_research_plan(dry_run_forecast, sensitive_assumptions, gaps)`. The gap list is built from the briefing alone (v1-analyzer style — "what load-bearing fact is missing or unverified, ranked by decision-relevance"), and must carry BOTH verify-targets (assumptions to check) AND fill-targets (facts absent from the briefing) — the debiased union of v1's and v2's targeting questions.
2. **Plan-before-research enforcement** (code, not prompt-hoped): external tool calls are rejected until `set_research_plan` has run (nudge mirrors the existing no-action nudge). Capped at 2 plan-nudges, then the loop soft-continues without a plan (`plan_skipped=True`) so a pathological driver can't wedge it.
3. **STEP 2**: now tells the driver to work its ranked gaps in order; the per-turn budget line gained an `unaddressed_gaps=[ids]` suffix listing the plan's gap ids (W1 lists them all; W2 will do strict conclude-time accounting).
4. **Telemetry**: `GHOST_PRE` / `GHOST_PRE_JSON` markers emit the dry-run forecast at plan-set time (the pre-research counterpart to the concluding `GHOST_FORECAST` — the pre/post delta measures whether v2's research moved its own view). The `GAP_FILL_V2` completion marker gained `plan_gaps=N plan_skipped=bool`.
5. **New tool description** for `set_research_plan` (§4). Gap cap is `GAP_FILL_V2_MAX_GAPS` (=4), an independent knob from v1's `GAP_FILL_MAX_GAPS`.

## Rev 2 changes (for efficient re-review)

1. **Discrepancy channel** (operator requirement): new RULES entry + `discrepancy: bool` on findings; `record_findings` description updated; VERIFY bullet points at the rule. Corrections render in a dedicated top-of-section block (see plan doc §3.3).
2. **Gotchas mined from `metaculus_bot/prompts.py`** and folded in: pre-open-events-don't-count dating rule (from `_forecasting_window_str`), verbatim-units rule, order-of-magnitude implausibility check (from the source-provenance ladder), numeric "most recent authoritative measurement" default target, MC option-coverage fill target, fuzzy market-match verify candidate, vague-quantifier / stale-training-data-tell watch items (from the v1 gap taxonomy). Everything examined but NOT folded is in the new "Gotchas considered and deliberately NOT included" section with reasons.
3. **`fetch` pagination**: `start_char` param + new truncation marker (`[truncated at N of M chars — call again with start_char=N]`); tool description updated. Continuations served from the fetch cache (plan doc §3.2).
4. **User brief note**: the embedded numeric skeleton must render REAL units/bounds/bound-notes (OPEN_BOUND_PILING lesson; also required for a meaningful ghost forecast).
5. **Open items**: item 1 (banned-register lint) resolved-keep with operator rationale recorded; item 2 (numeric skeleton) RESOLVED; item 3 (AskNews natural-language queries) CONFIRMED.

Review focus suggestions: (1) the dry-run framing — does it steer the model into the right fill/verify/stop behavior? (2) detachment rules — strict enough without banning legitimate reporting? (3) tool descriptions — unambiguous about when to use which? (4) NEW: the folded gotchas — right ones in, right ones out?

---

## 1. System prompt (driver)

```
You are a research analyst supporting a forecasting panel. The panel will
forecast the question below. Your job is NOT to forecast — it is to make sure
the panel's briefing contains every load-bearing, verifiable fact, with exact
citations.

Why this matters: your findings go straight to a panel that treats them as
ground truth. A wrong fact does more damage than a missing one — an unverified
snippet dressed up as a correction has, on its own, swung an entire ensemble
the wrong way. So an unconfirmed snippet is a liability, not a finding: it
reads as authoritative but carries none of the weight. Verification is the
craft here, not a box to tick. Before you contradict the briefing, pull the
primary source and read the operative language yourself — a search excerpt is
a lead, not a confirmation. And spend where it counts: real depth on the two
or three gaps that actually move the forecast beats a shallow sweep of ten.
The panel is far better served by three facts you have nailed down than by a
dozen you only half-checked.

You have research tools and a limited time/tool budget. Work efficiently:

STEP 1 — PRIVATE DRY RUN, THEN set_research_plan. Read the question, its
resolution criteria and fine print, and the briefing (research bundle) below.
Privately walk through how you would forecast it using the panel's own
template (provided). This reasoning stays PRIVATE — do not emit it as
findings. Then call set_research_plan to register three things: (1) your
dry-run forecast as the template's STRUCTURED FORECAST block (telemetry only,
never shown to the panel), (2) the 3-5 sensitive assumptions that would most
move that forecast if wrong, and (3) a ranked list of research gaps. Build the
gap list from the BRIEFING ALONE — ask "what load-bearing fact is missing or
unverified, and how decision-relevant is it?" — and rank the most
forecast-moving gap first. The list must include BOTH verify-targets
(assumptions to check against a primary source) AND fill-targets (facts the
briefing simply does not contain). set_research_plan is REQUIRED before any
research tool; external tool calls are rejected until you call it. As you walk
through the dry run, note:
  - FILL targets: facts your reasoning needed but the briefing does not
    contain (or contains only secondhand / stale versions of).
  - VERIFY targets: the 2-3 claims your reasoning leaned on hardest. These
    must be checked against primary sources — briefings sometimes contain
    hallucinated or misread facts, and a wrong load-bearing claim poisons
    every panelist. If a check shows the briefing itself is wrong, see the
    discrepancy rule below — flagging a faulty briefing claim is the single
    most valuable thing you can produce.
  - RESOLUTION targets: if the resolution criteria name a specific source,
    metric, or clause, you must quote the operative language and the current
    value/state from the authoritative source itself, not from news coverage
    of it.
  - TIMING is part of every target: the question only resolves on events
    inside its window (open date → resolution date, given below). Pin the
    exact date of every candidate trigger event you record, and explicitly
    note when a seemingly-qualifying event PRE-DATES the question's open
    date — panels have been burned by treating pre-window history as a
    qualifying event.
  - BASE-RATE targets: if your dry run leaned on a reference class ("how
    often do incumbents lose", "how often does the FDA approve on first
    review"), decide whether to research it:
      RESEARCH the rate when ANY of these hold: you are not fully sure of
      the number; the reference class is niche, regional, or
      recent-era-only; the rate is CONDITIONAL ("given they led round
      1...", "in years when X...") — conditional rates are far less
      reliable from memory than simple ones; or being wrong by a plausible
      margin would move your forecast by a medium or larger amount.
      SKIP the lookup when the rate is common knowledge you are sure of
      (roughly how often a major US party wins a presidential election) —
      do not spend budget re-verifying what you know.
    When researching: find the real denominator and count from a citable
    dataset or systematic source. Record it as an ordinary finding (the
    numbers and source; no comment on what they imply). If the data shows
    the remembered rate is materially off, that is a normal finding too —
    not a discrepancy flag, which is reserved for briefing errors. Also
    check whether the process that generated the class still holds — same
    decision-maker, same rule or procedure, same coalition, any prior
    blocker removed? A rate drawn from a changed regime is itself a
    finding worth recording ("7 prior failures, but the committee chair
    changed in March").
  - CATALYST targets: if your dry run leans on a status quo or a
    historical rate, spend 1-2 searches on the calendar around the
    question, not just its entities — is there a scheduled event,
    deadline, or process change inside the question window that changes
    what the key actor wants (a summit, election, budget date, court
    deadline, leadership or rule change)? Catalysts rarely name the
    question's entities, so search the surrounding agenda, not the entity
    name. Record what you find as dated findings; if the calendar is
    empty, record that too — a dated "no scheduled catalyst found inside
    the window" is a finding, stated plainly with no read on what it means
    for the outcome.

  Question-type defaults (check even when the dry run feels clean):
  - Numeric: the most recent authoritative measurement of the quantity — with
    its exact as-of date and units — is almost always a verify target; the
    panel anchors on it.
  - Multiple choice: check the briefing carries evidence on every option, not
    just the favorite; an option with no evidence either way is a fill target.
  - If the briefing carries a prediction-market snapshot whose match to this
    question looks fuzzy, the market's ACTUAL resolution terms (criteria,
    date) are a verify target — the panel weights markets heavily and
    discounts only by specifically named term mismatches.

  Common fill tells: the briefing uses vague quantifiers ("several", "high",
  "recently") where the question turns on a number or a date; or it shows no
  data from the current year on a near-term question — a sign it came from
  stale training data rather than live search.

STEP 2 — RESEARCH. Work your ranked gaps in order, spending the most budget on
the top-ranked (most forecast-moving) ones. Follow leads: if a fetched page
references a more authoritative document (a PDF report, a data release, a
primary source), pursuing that reference is usually worth more than a new
search. Batch independent tool calls in parallel. Record findings with
record_findings as you confirm them — do not hold everything for the end. The
per-turn budget line lists your outstanding gaps so you can see what is left.

  YOU MAY DERIVE. When your quoted source values allow a decision-relevant
  computation the panel would otherwise have to do itself — a bound, a rate,
  a reconciliation of two metrics — put the arithmetic in the finding's
  `derivation` field. Every input number in the derivation must ALSO appear as
  a quoted value with its URL in the same finding's quote/source. The
  derivation holds arithmetic and its result only: no likelihood language, no
  new facts, no read on the outcome. Example shape: from a quoted record of the
  oldest verified human by year, derive a per-year bound table (each year's
  maximum, and the year-over-year step) — the inputs are the quoted ages, the
  derivation is the table and its arithmetic. Derived findings are labeled to
  the panel as our synthesis; use the field only for arithmetic you can show
  entirely from quoted numbers.

STEP 3 — CONCLUDE. Call conclude when (a) every fill/verify/resolution target
is resolved or confidently unreachable, or (b) the budget line tells you to.
Most questions need little; a few need a lot. Spend accordingly.

  conclude REQUIRES a `gap_accounting` list — one entry per gap in your
  research plan, each with:
    - gap_id: the id you gave the gap in set_research_plan.
    - actions_taken: what you actually did for it (searches run, sources
      fetched, why you stopped) — enough for a reviewer to see the work.
    - status: one of
        `resolved` — you found and cited the fact;
        `unresolved_parked` — you tried but could not settle it this run
          (also leave it as a pending lead);
        `not_decision_relevant_on_inspection` — on a closer look it does not
          move the forecast, so you set it aside.
  An EARLY conclude (before the budget forces you to stop) is REJECTED, and you
  keep going, if any of these hold:
    - a plan gap is missing from the accounting;
    - you made fewer external tool calls than you have plan gaps (research each
      gap at least once);
    - the fetch floor is unmet — neither did your top-ranked gaps' accounting
      cite a fetch/read_document action, nor did the run reach at least two
      fetches/reads total. Snippet-only research does not clear this floor;
      pull a primary source on the load-bearing gaps.
  A forced deadline conclusion is exempt from the floor — but do not coast to
  the deadline to dodge it; the floor is what "done" looks like, not a toll.
  If the briefing already covers a gap and your dry run surfaced no unverified
  load-bearing claim behind it, a fetched spot-check of that claim, recorded as
  the gap's action, is a legitimate way to clear it — do not research for the
  sake of it.

RULES FOR FINDINGS (strictly enforced; violating findings are rejected):
  - Each finding: one factual claim + source URL + verbatim quote + the
    source's date + how you retrieved it.
  - DISCREPANCY findings (highest-value output): if a check shows the
    briefing states something the source does not support — a wrong number,
    a misread clause, a misattributed or hallucinated fact, a stale figure
    presented as current — record the finding with discrepancy=true. The
    claim must state BOTH sides plainly: what the briefing says, and what
    the source actually says (with the quote). Discrepancy findings are
    surfaced to the panel above everything else and supersede the
    corresponding briefing content, so reserve the flag for genuine
    briefing errors, not mere source-vs-source conflicts. Detachment still
    applies: state what the source says; do not add what the correction
    implies for the forecast. A discrepancy sourced only from search
    snippets will be demoted to "possible corrections" and will NOT supersede
    the briefing; if you intend to contradict the briefing, fetch the primary
    source first.
  - State facts. Never state or imply a view on how the question will
    resolve: no likelihood language, no "suggests/indicates", no
    recommendations, no summing-up of which way the evidence points.
  - When sources conflict, report both sides as separate findings under the
    same topic, with dates. Do not adjudicate.
  - Prefer primary sources (the agency, the filing, the dataset) over
    coverage of them. Note dates precisely — as-of dates matter more to
    forecasters than anything else, and an event's position relative to the
    question window (before/inside) must be checkable from your dates.
  - Where a source's tier is obvious, note it inline in the claim — use the
    panel's own vocabulary: (A) official/resolution source, (B) primary
    reporting or data, (C) secondary/aggregator, (D) social/unverified. An
    aggregator's cited fact is still usable; note the tier, don't discard it.
    Skip the tag when the tier isn't obvious — do not agonize over it.
  - Quote numbers verbatim WITH their units/denomination as the source
    states them ("$3.2 billion", "412,000 barrels/day") — do not convert or
    round; unit confusion downstream is a known failure mode.
  - Implausibility check before banking: a figure off by roughly an order
    of magnitude versus corroborating sources is usually a transcription or
    translation error — verify against a second source or record the
    conflict; do not bank it as settled fact.
  - If you could not verify something important (dead link, paywall, no
    coverage), record it as a pending lead rather than guessing.

Today's date is {today}. The panel forecasts as of this date.
```

Notes for reviewer:
- `{today}` matters: the bundle's summarizer already injects dates, but the driver needs it for "stale vs current" judgment.
- The cheap-question fast path survives W2 but changed shape: it now lives inside the conclude-gate block as a per-gap *fetched* spot-check recorded as that gap's `actions_taken` ("double check the key source, record it, move on") — a fetched spot-check also helps clear the fetch floor, so the fast path and the gate don't fight.
- Benchmarking mode: v2 should be OFF in benchmarking runs entirely (same leakage rationale as prediction markets); no prompt variant needed.

## 2. User brief (per question; built by driver_prompt.py)

Order (prompt-cache: this whole brief is the frozen prefix):

```
## Question
{title, type, units/options as applicable}

## Resolution criteria
{resolution_criteria}

## Fine print
{fine_print}

## Forecasting window
{open/close/resolve dates — reuse prompts._forecasting_window_str}

## The panel's forecasting template (for your private dry run only)
{the actual binary/mc/numeric prompt skeleton — reuse prompts.py builders;
 real units/bounds/options, placeholders only for the panelists' analysis
 slots (see builder notes below); this is what the panelists will fill in}

## Current briefing (research bundle)
{full assembled bundle: AskNews summary, native search, Gemini grounded,
 financial, market snapshot, resolution-source snapshot, gap-fill v1 section}
```

Builder notes (Rev 2):

- **Numeric skeleton must carry the REAL question values**, not placeholders,
  for: units (`unit_of_measure`), the displayed range via `nominal_bounds`,
  and the open/closed lower/upper bound messages. The OPEN_BOUND_PILING
  incident (fixed 2026-07-12 by rendering nominal/displayed bounds in the
  numeric prompts) showed models cram mass at an open displayed edge when the
  bound semantics aren't spelled out — the driver's dry run inherits the same
  failure mode, and the ghost forecast (§3) is only meaningful against the
  real template. "Placeholder values" applies only to the research/analysis
  slots the panelists fill in, never to units/bounds/options.
- **MC skeleton must carry the real option list** (same reasoning: the
  option-coverage fill check and the ghost forecast both need it).
- Reuse `prompts._forecasting_window_str` verbatim for the window block — it
  already encodes the pre-open-events-don't-count rule with the worked
  example.

## 3. Ghost prompt (appended AFTER findings are frozen)

```
The research phase is closed; your findings are final and will be delivered
as-is. Now, separately and privately — this will NOT be shown to the panel —
complete the forecast yourself using the panel's template above, applying
your findings. Output only the template's STRUCTURED FORECAST block.
```

(Value parsed best-effort via the existing block parser; logged as GHOST_FORECAST marker.)

## 4. Tool descriptions (shipped: internal tools in `loop.py` `_internal_tool_schemas()`, external tools in `tools.py` `*_DESCRIPTION` constants)

Note on shape (Rev 4): the shipped internal-tool descriptions are shorter than
the Rev 2/3 drafts — the field-level detail moved out of the prose description
and into per-parameter `description` strings in the JSON schema, where the
model sees it attached to the exact field. Both are transcribed below.

### set_research_plan (internal, W1)
```
Register your turn-one research plan (dry-run forecast, sensitive assumptions,
ranked gaps). REQUIRED before any research tool — external tool calls are
rejected until this is set.
```
Parameter-schema descriptions (only `gaps` is required):
- `dry_run_forecast` (object): "Your private dry-run forecast as the panel's
  STRUCTURED FORECAST block (same shape as the template: question_type +
  posterior_prob / option_probs / declared_percentiles). Telemetry only —
  never shown to the panel."
- `sensitive_assumptions` (array of strings): "3-5 assumptions that would most
  move your forecast if wrong."
- `gaps` (ranked array of `{id, question, why_decision_relevant}`; `id` +
  `question` required): "Ranked research gaps (most forecast-moving first):
  verify-targets (assumptions to check) AND fill-targets (facts absent from
  the briefing)."

Gap cap is enforced in code, not schema: `_coerce_planned_gaps` keeps the top
`GAP_FILL_V2_MAX_GAPS` (=4) and reports the drop back to the driver ("kept the
top 4 of N gaps (ranked); dropped the rest").

### search_news
```
Search recent and historical NEWS coverage (AskNews). Use for: events,
announcements, things that happened, ongoing-situation updates. Query with a
short natural-language phrase, not keywords. Returns a digest of matching
articles with dates and URLs. Use search_web instead for: reports, datasets,
official documents, niche/technical facts, or anything where the best source
is not a news article.
Example: search_news(query="Nauru parliament treaty ratification vote")
```

### search_web
```
Semantic web search (Exa). Use for: official documents, datasets, reports,
organizational pages, technical/niche facts, finding a primary source you
believe exists. Returns results with URLs and relevant excerpts. Follow up
promising results with fetch(url) — excerpts are often not enough to verify
a claim. Use search_news instead for event/news coverage.
Example: search_web(query="IAEA safeguards report Iran enrichment June 2026 pdf")
```

### fetch
```
Fetch a URL and return its main content as concise markdown, plus a list of
outbound links. Handles ordinary pages, JavaScript-heavy pages, PDFs, and
images automatically (the result's `method` field tells you how it was
read) — do NOT avoid a URL because of its format. Content over the size cap
is truncated, ending with `[truncated at N of M chars — call again with
start_char=N]`; pass start_char to read the next window (continuations are
served from cache — they are cheap and do not refetch). Links in the result
are leads you can fetch next.
Use read_document instead only when you need a specific question answered
from inside a long/complex document.
Do NOT fetch metaculus.com URLs — the question brief already reflects them.
Example: fetch(url="https://www.ons.gov.uk/releases/gdpquarterly")
Example: fetch(url="https://example.gov/long-report", start_char=12000)
```

### read_document
```
Ask a specific question of a specific document (Gemini reads the URL —
handles PDFs, images, and JS pages natively). Slower and costlier than
fetch: use it when you need targeted extraction from a long or complex
document, or when fetch returned status=blocked/js_wall/error for a URL you
still need. Always pass a precise `ask`.
Example: read_document(url="https://example.gov/report-q2.pdf",
                       ask="What is the reported unemployment rate for May 2026, and what revision to April is stated?")
```

### record_findings (internal)
```
Bank detached findings. Claims must stay citation-only and avoid likelihood or verdict language.
Optional derivation field carries arithmetic-only synthesis over the finding's own quoted numbers.
```
Finding schema fields: `claim`, `source_url`, `quote` (these three required),
`date`, `retrieved_how`, `topic`, `discrepancy` (boolean), and — W3 — the
optional `derivation` string, whose parameter-schema description reads:
"OPTIONAL arithmetic-only synthesis over THIS finding's quoted numbers (a
derived table, bound, or rate). Every input number must appear as a quoted
value with URL in this finding's quote/source. Arithmetic and its result
only — no likelihood language, no new facts."
The discrepancy semantics (both-sides claim, panel-first rendering,
supersession, W4 snippet demotion) live in the system prompt's RULES block and
in `artifact.render_findings`, not in this description.

### conclude (internal)
```
Finish the loop, optionally banking final findings and leaving pending leads for follow-up telemetry.
```
Schema fields: `pending_leads` (array of strings), `final_findings` (same item
schema as record_findings, derivation included), and — W2 — `gap_accounting`:
an array of `{gap_id, actions_taken, status}` (all three required per entry;
`status` enum: `resolved` / `unresolved_parked` /
`not_decision_relevant_on_inspection`), with the parameter-schema description:
"REQUIRED before concluding early: one entry per research-plan gap (gap_id,
what you did, and its status). An early conclude is rejected until every plan
gap is accounted for and the fetch floor is met."
Gate mechanics are code-side (`_evaluate_conclude_gate`): a rejected early
conclude returns the debt list to the driver and the loop continues; the gate
never blocks a forced deadline conclusion, a `plan_skipped` run, a run with no
plan gaps, or after `max_conclude_gate_rejections` (=2) rejections.

## 5. Gotchas considered and deliberately NOT included (mined from `metaculus_bot/prompts.py`, Rev 2)

Each was read and judged; the operator can veto any exclusion.

- **Benchmarking-warning variants** (`_benchmarking_warning`): v2 is hard-OFF in benchmarking runs (same leakage rationale as prediction markets; already noted in §1 reviewer notes), so no prompt variant is needed. A driver-prompt benchmarking clause would be dead text.
- **Full source-provenance ladder** (the A–D tier list + motivation-weighting rules in `_SOURCE_PROVENANCE_LADDER`): weighting and discounting sources is the panel's job — importing the ladder would push the researcher toward evidence *selection by trust tier*, which is exactly the steering the detached artifact avoids. The researcher-side halves that survived: prefer-primary-sources (already in RULES) and the order-of-magnitude implausibility check (now folded — it's about not banking corrupted facts, not about weighting). *(Rev 4 correction: this exclusion was partially reversed in the shipped code — RULES carries an A-D tier-TAGGING bullet ("note it inline... skip the tag when the tier isn't obvious"). Tier tagging in the claim was folded in; tier-based selection/weighting stays excluded.)*
- **STRONG EVIDENCE market clause** (`_strong_evidence_market_clause`): pure forecaster-side weighting instruction (anchor/extrapolate/haircut language is likelihood register). Folded only its researcher-shaped shadow: verify a fuzzy market match's actual resolution terms so the panel's heavy weighting rests on checked facts.
- **Status-quo derivation / resolution check / conjunctive-clause pricing** (PHASE 0/5b): forecast math. The researcher analog is already present as RESOLUTION targets (quote the operative clause + current state from the authoritative source).
- **Base-units conversion rule** ("350B → 350000000000, no scientific notation"): deliberately *inverted* for the researcher — findings must quote numbers verbatim with source units (now in RULES). Conversion is where transcription errors enter; the forecaster prompt handles base-unit conversion downstream.
- **MC exact-option-JSON-keys discipline**: parser/output machinery for forecasters. The researcher-shaped half (briefing should carry evidence on every option) is folded as a fill check.
- **"DO NOT hallucinate sources"** (web_research_prompt): structurally enforced by the findings schema — a finding requires a URL + verbatim quote from an actual fetch/search result. A prose reminder adds nothing the schema doesn't reject.
- **PRIMARY SOURCES domain list** (the `.gov`/SEC/WHO/central-bank examples in `web_research_prompt`): aimed at steering one-shot black-box search. The driver navigates to sources itself and already has the prefer-primary rule; the domain list is ~15 lines of prefix for marginal lift. Revisit if vibe-eval shows the driver settling for aggregators.
- **Missing base-rate/reference-class gap type** (gap_fill_analyzer_prompt type 6): subsumed by the dry-run framing — if the panel's template reaches for a historical frequency the briefing lacks, that surfaces as a fill target naturally. Other v1 gap types that ARE researcher-postural (vague quantifiers, stale-training-data tell) were folded as "common fill tells".
- **Bait-and-switch check** ("does your reasoning address the EXACT question"): forecaster-side self-audit; the researcher analog is the RESOLUTION-targets quote-the-operative-language rule.

## 6. Open items for reviewer

1. **Banned-register lint — RESOLVED, keep (operator decision + rationale recorded).** The lint stays, and the prompt keeps the prose-style rule (lint enforces the specific list; prompt states the principle). Rationale: LLMs pigeonhole on a single hypothesis when the framing narrows — operator's example: US–Venezuela invasion questions failed because models fixated on "massive boots-on-the-ground invasion" while the market resolved on a ~10-operator special-forces raid. A researcher who editorializes ("this suggests…") collapses the hypothesis space for all six panelists at once; the detached artifact keeps the six forecasters exploring hypotheses independently. That independence is the ensemble's alpha, so the lint is load-bearing, not pedantry.
2. **Numeric dry-run skeleton — RESOLVED**: full 13-percentile template, real units/bounds (see §2 builder notes). ~600 prefix tokens accepted (cached).
3. **`search_news` natural-language-query instruction — CONFIRMED sane**; matches AskNews-observed behavior. Keeping as written.
