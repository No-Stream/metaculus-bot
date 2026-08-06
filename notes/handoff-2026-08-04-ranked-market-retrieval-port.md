# Ship ranked market retrieval to prod (port + fix + enrichment)

**Date:** 2026-08-04
**Branch:** `resids-aug-2`
**Repo:** `/Users/flatljan/personal/metaculus-bot`
**Status:** In Progress — recon stage; implementation not started

## What we accomplished this session

Committed on `resids-aug-2` (all gates green at each commit; tree currently clean):
- `a721c7c` — research-archive precedence fix: artifact records now beat comment
  reconstructions in `latest/` (25→271 artifact-sourced); three-class source model
  (`artifact` > `comment_backfill` > `log_backfill`); non-destructive `--rebuild-only`;
  truncation guard; flush-in-finally; test workflows now persist research.
- `d911ca8` — AGENTS.md note documenting the three-class source model.
- `1c75871` — persisted GHA artifact store (`backtests/gha_artifact_store/`, local disk =
  source of truth); all harvests read from the store; `make resync_from_store` = zero-network
  re-parse; `verify_completeness` fails on store gaps.
- `1d15bdf` — store growth-rate comment correction (~18 MB/month measured).
- `9861b5b` — test-suite cleanup: 23 unawaited-coroutine warnings fixed at cause, 7 vacuous
  pchip tests deleted + replaced with a mutation-checked parametrized test, dead assertions
  tightened. Verdict from the git-blame retrospective: NO reward hacking. Suite now
  4,485 passed / 12 skipped / 1 warning.

Bake-off (all artifacts in `scratch/bakeoff_run_2026-08-03/`, gitignored):
- Ran the 5-arm market-retrieval bake-off paid run ($0.85 upstream — see gotcha below),
  then a ground-up re-analysis (free), then built + measured the operator-designed
  **ranked arm** ($0.25 upstream). Total paid spend this session: **$1.10**.
- Ranked arm results (`results/RANKED_ARM_RESULTS.md`; seven-arm comparison in
  `results/ranked_arm_scores.json` — NOT the per-replicate runner files, they were
  restamped single-arm): **12/16 question-level (both replicates)** vs 9-10 for the
  shipped arms and 15/16 same-pool oracle; **12/17 near-identical (r1), 10/17 (r2)** vs
  13/17 oracle and 0/17 for prod today; lowest TN noise of any LLM arm; ~$0.007/question.
- Key mechanism finding (`results/GROUND_UP_SYNTHESIS.md`): the earlier audit arms failed
  because a **round-robin-by-venue render step** evicted 43 of 58 LLM-kept near-identical
  rows. The audit LLM itself kept 58/58. Selection/rendering was the whole problem.

## Current state

- Working tree: clean. Nothing uncommitted. No tmux sessions running.
- **Two recon agents were in flight when this session ended** (background agents die with
  the session — the next session must check their output on disk, not wait for them):
  - `port-spec` — writing `scratch/market_port_2026-08-04/PORT_SPEC.md` (seam inventory of
    prod `metaculus_bot/research/prediction_market.py`, deletion list, module plan, LLM
    wiring, ranker+query-author prompt drafts, test plan). NOT yet on disk at handoff time.
  - `enrich-recon` — `scratch/market_port_2026-08-04/ENRICHMENT_SPEC.md` **IS on disk**
    (26.8KB, written 12:49): Manifold enrichment spec + PredictIt verdict. Its structured
    summary (manifold_verdict / predictit_verdict one-liners) was never received — read the
    doc itself.
- A monitoring loop (session-only cron) was driving the port pipeline; it died with this
  session. Recreate via `/loop` if desired.

## What's left to do

1. **Read both specs** in `scratch/market_port_2026-08-04/`. If `PORT_SPEC.md` is missing
   or partial, re-dispatch the port-spec brief (its full text is reproducible from this
   doc's "The validated design" + "Seams" sections below). **Review both critically** —
   verify seam claims carry file:line receipts, check the two specs don't conflict at the
   Manifold/PredictIt extension points, resolve open questions where evidence decides them.
2. **TDD implementation** from the specs (2-3 scoped agents or a workflow; worktree
   isolation if parallel writers share files). Replace prod retrieval in
   `metaculus_bot/research/prediction_market.py` with:
   - GENERATION (recall-maximal union): settlement-source join (reference
     `scratch/bakeoff_run_2026-08-03/arms/arm_join.py`; reuses
     `research/resolution_source.py`'s URL extractor) + venue-native short-query search
     (Manifold relaxation ladder already in prod; Polymarket public-search) + full
     Kalshi/PredictIt prefetch (complete pull measured FASTER than today's truncated one)
     + LLM query-author with ADDITIVE queries (reference `arms/arm_two_pass.py`).
   - SELECTION: **one adaptive-width LLM ranking call per question** — up to 8 markets,
     MAY return zero, inclusion-biased (anything somewhat relevant included; only
     irrelevant excluded), each row carries a one-phrase relevance label. Relation
     hierarchy explicit in prompt: same-quantity-same-date > same quantity diff
     date/threshold > related drivers > no-bearing = exclude. Settlement provenance shown
     as a signal.
   - RENDER: ranked order verbatim. **NO round-robin, no venue fairness** (the measured
     defect). Fail-open: ranker failure → deterministic pool-order top-8, marked degraded.
   - THE 44941 PROMPT FIX: the old auditor prompt named "a different OFFICE's or RACE's
     election result, even same state/cycle" as canonically irrelevant — this cost an
     entire question (4 wanted markets in pool, ranker returned zero). The new ranker
     prompt must carry the operator's relevance definition and MUST NOT inherit that
     example.
   - Fold in ENRICHMENT_SPEC.md's Manifold enrichment and PredictIt verdict.
   - Kill list: S4/S5 keyword prompts, fuzzy floors, `_RELEVANCE_STOPWORDS`,
     `max_matches_per_platform`, entity-series matching, prod `as_of` derivation
     (set None on provider path; `is_benchmarking` is the real leakage guard), and the
     `nested[0].close_time` bug (event close = max over nested markets).
   - New LLM slots in `llm_configs.py` ONLY (ranker + query author,
     `openrouter/openai/gpt-5.6-luna` low effort via `build_llm_with_openrouter_fallback`).
3. **/forge adversarial review** on the diff; drive the fix round.
4. **Full gates**: `make test` / `make lint` / `make typecheck` green; commit with a real
   message. (Pre-push hook runs the full suite; CI needs `gh run list --repo
   No-Stream/metaculus-bot --branch resids-aug-2` to confirm.)
5. **OPERATOR triggers `test_bot`** as final live validation — they fire it themselves,
   explicitly agreed. Do NOT dispatch it or any paid call.

Steps 1-2 sequential; inside step 2 the module split may parallelize per PORT_SPEC.

## Key decisions made (don't re-litigate)

- **Ranked design ships** (operator: "let's ship a great version of this"): adaptive width
  0..8, inclusion-biased, relevance labels per row. Rationale: measured 12/16 vs 9-10
  shipped, near-oracle NI recall, cheapest paid arm.
- **TN disqualification is NOT the operator's utility** — recall >> precision, 3-4 false
  positives per false negative is a good trade. TN rows are a cost column only. The
  original plan's disqualifier gate is retired for design purposes.
- **Prod incumbent retires** (0/17 NI, most TN noise, keeps labeled no-bearing rows).
- **Fix-and-port in one motion, no re-measure** of the 44941 prompt fix — operator chose
  test_bot as the validation instead (~$0.25 saved).
- **Input interleaving of venues in the ranker prompt is fine; OUTPUT round-robin is the
  defect.** Don't confuse the two.
- **Query author stays** (adds vocabulary-gap rows the question's surface tokens can't
  reach — measured real; its gains were eaten by the old render step, now fixed).
- **Paid-spend gate discipline**: every paid run priced first, canary-gated, operator
  sign-off per run. Operator approved up to ~$1 for the ranked-arm measurement (spent
  $0.25); NO standing approval exists now.

## Gotchas & context the next session needs

- **BYOK cost multiplier**: OpenRouter calls on the personal key route through a BYOK
  integration — OpenRouter credits stay ~flat and `check_credits` shows nothing; the REAL
  charge lands upstream on the OpenAI account at ~2.12x the harness's computed rate
  (measured $0.2383/$1.20 per 1M vs the promotional $0.10/$0.60). Budget/reconcile against
  `usage.cost_details.upstream_inference_cost` in responses, never the computed ledger.
- **Agents go idle without delivering reports** — every one this session did. Their plain
  text does NOT reach the orchestrator; only SendMessage does. Ping idle agents via
  SendMessage; check disk for their outputs first.
- **Context exhaustion kills agents on this codebase** (~1.2M tokens): never let an agent
  read the bake-off plan doc (`scratch_docs_and_planning/market_retrieval_bakeoff_plan_2026-08-03.md`,
  935 lines) AND whole large sources. Scoped briefs with explicit reading lists; the
  workflow 3-min watchdog compounds this (build-heavy stages did better as direct
  background agents).
- **Bake-off harness caches everything**: all 432+ paid LLM responses are in
  `scratch/bakeoff_run_2026-08-03/cache/llm/`; re-analysis is $0 (pattern:
  `analyze_bakeoff.py` runs with OPENROUTER_API_KEY unset so cache misses raise).
- The prod section header `## Prediction Market Snapshot`, provider name
  `prediction_market`, `is_benchmarking` suppress, `PREDICTION_MARKETS_ENABLED` gate, and
  liquidity field names (commit `b96d64d`) are load-bearing seams — telemetry and residual
  analysis parse them.
- `prompts.py` `_strong_evidence_market_clause` must be updated to reference the new
  relevance labels (it currently only references liquidity).
- One selftest in the bake-off scratch harness (`test_a_paid_body_reaches_disk_before...`)
  fails since the first paid run (asserts the real LLM cache is empty; it now has 432
  entries). Scratch-only, not a repo test; harmless.
- `arm_audit_only.py` (scratch) is 1,024 lines and was deliberately not split pre-spend;
  irrelevant to the port unless code is lifted from it.
- Known open cleanups (non-blocking): `tests/test_research_sync_job.py:202` skip has wrong
  polarity (should fail, not skip, if a workflow drops RAW_RESEARCH_LOG_ENABLED);
  FUTURE.md follow-ups untouched.

## Enrichment recon verdicts (arrived at handoff time — headline conclusions; full receipts in ENRICHMENT_SPEC.md)

- **Manifold**: add `textDescription` truncated at 300 chars via `GET /v0/market/{id}` —
  the ONLY field the search endpoint omits — ~1s wall for 60 rows at concurrency 10,
  +3.1k prompt tokens. Close time and liquidity need NO extra fetch: prod's search parse
  already reads them off every search row (30/30 non-null measured; the bake-off's blank
  fields were an artifact of reading the frozen snapshot instead of the search row — the
  port gets that fix free by reading search rows). Do NOT fund this by narrowing width:
  width 60 retains 22/22 in-pool labeled rows, width 30 only 16/22.
- **Prod bug found**: `_manifold_rules_text` prefers `textDescription`/`description`,
  but the search response carries neither, so its title fallback ALWAYS wins — raw_rules
  is a duplicate of the title on every Manifold row today.
- **PredictIt**: KEEP, fix = delete the width=20 fuzzy pre-filter and render all 197
  markets (+6.8k tokens; 6/6 labeled-row pool recall by construction). 5 of the 6 labeled
  rows were cut by the fuzzy pre-filter at true ranks 26-107, and the fuzzy scorer is
  ~noise on short political headlines (rank-1 for an Ethereum question was "Will the Pope
  visit Cuba in 2026?"). PredictIt lines are BETTER formatted than Manifold's (contract
  rules on 360/360), contrary to the brief's suspicion.
- **44941 confirmed as the DROP-clause, not attention decay**: the ranker returned a
  literal [] for the whole question in both replicates; the "different office/race is
  irrelevant" clause matches a Florida down-ballot primary exactly. Attention decay is
  UNTESTED — input-order venue interleaving is cheap insurance, not a fix for a measured
  mechanism.
- **OPEN QUESTION 1 (bigger Manifold lever, decide scope)**: `contractType=BINARY` in
  `_manifold_search` is a hard ceiling — 27 of 89 labeled-wanted Manifold markets are
  MULTIPLE_CHOICE/MULTI_NUMERIC/NUMBER/DATE and unreachable at any width. Fixing it needs
  a rendering decision for multi-outcome markets (`probability` is null on all of them).
  Recommend: flag to operator as a follow-up rather than folding into this port.
- **OPEN QUESTION 2 (token budget)**: ranker input is ALREADY 24.3k tokens median (the
  brief's 10-15k target was wrong); both fixes take it to ~31k. Recommendation: accept —
  at upstream rates that is still <$0.01/question input, and the alternative (trimming
  Kalshi/Polymarket line format, 84% of input chars) risks the rows that carry the value.

## References

- Port working dir: `scratch/market_port_2026-08-04/` (ENRICHMENT_SPEC.md on disk;
  PORT_SPEC.md expected from the port-spec agent).
- Bake-off evidence: `scratch/bakeoff_run_2026-08-03/results/` — `GROUND_UP_SYNTHESIS.md`
  (design rationale + measured facts), `RANKED_ARM_RESULTS.md` (the winning arm),
  `ranked_arm_scores.json` (seven-arm comparison), `RESULTS.md` (original bake-off,
  superseded in interpretation by the synthesis).
- Reference implementations: `scratch/bakeoff_run_2026-08-03/arms/arm_ranked.py` (the
  validated arm), `arm_join.py`, `arm_two_pass.py`.
- Ground truth: `scratch/market_search_design_2026-08-03/ground_truth_markets.json`
  (26 questions, 363 labeled rows; near_identical/diff-date/diff-threshold/adjacent are
  WANTED, no_bearing is noise; TN questions 44969+44916).
- Archive-fix background: `scratch_docs_and_planning/archive_precedence_fix_2026-08-03.md`.
