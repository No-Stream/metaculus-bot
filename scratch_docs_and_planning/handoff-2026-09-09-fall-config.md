# Fall 2026 configuration: the cost pass, the residual round and the fall merge plan

**Date:** 2026-09-09 (late evening, Pacific)
**Branch:** `mantic-competition` (off `main` at 660fd35; tip 0f51815; 56 commits carry today's
committer date, four of them from the operator's peer session)
**Repo:** `/Users/flatljan/personal/metaculus-bot` (operator's laptop; Python 3.12, uv)
**Status:** Tree clean at 0f51815. Nothing is pushed; the branch is the operator's to merge. One paid
bench is running in tmux and one teammate is still working (see "Current state"). The fall merge
work in "What's left to do" has NOT started. It must ship in ONE merge before the first fall
question closes on 2026-09-28.

## What this work is

The repo is a Metaculus forecasting bot. For each question it runs research providers in parallel,
adds two gap-fill passes (v1 sends targeted searches to a resolver model; v2 runs a bounded agentic
tool loop), asks three frontier LLM forecasters, publishes the median, and is scored on spot peer.
The summer season (`summer-futureeval-2026`) closed on 2026-09-06 and the fall season opens on
2026-09-28, so today was the between-seasons round: a residual analysis of the summer, a cost pass
over the gap-fill passes, and the tooling and plan for the fall configuration.

## What we accomplished this session

**Season and money.** The summer provisional leaderboard puts the bot at rank 15 of 277
(`scratch/residual_2026-09-09/leaderboard_summer.json`). The fall has two tournaments:
`fall-futureeval-2026` (project 33121; questions open 2026-09-28; forecasting ends 2027-01-06; one
question posted so far) and `metaculus-cup-fall-2026` (five posted so far; the summer cup had 58
questions and the spring cup 43). Plan for about 330 tournament questions plus 60 cup questions.

| Item | Value |
|---|---|
| Summer spot peer (rank 15 of 277) | 3,239.84 (about $1,584 prize) |
| Summer leader, futuresearch | 4,822.89 |
| Cost per question, booked | $2.07 to $2.21 |
| Cost per question after the ledger double-count fix | about $2.00 |
| Donated OpenRouter share of that | about $1.86 |
| Donated OpenRouter balance | $1,429 |
| Metaculus pays per question | about $1.50 |
| Mantic stipend | $3 |
| Paid spend today (resolver probe) | $5.97, plus the running bench (cap $10) |

The operator's targets were $1 a question with a $1.50 ceiling. They decided tonight to keep both
gap-fill passes for now and accept the budget rather than trade performance for cost.

**Residual round 2026-09-09** (`scratch/residual_2026-09-09/`: `SYNTHESIS.md`,
`SYNTHESIS_CRITIQUE.md`, `README.md`, the `dim_*.md` files, `dossiers/`). The archive holds 905
tagged records, 77 of them new. The live three-model roster beats the six-model era on the strict
read, but the horizon-matched interval includes zero, so we make no roster-improvement claim. Zero
records have resolved under the live configuration (newest bot comment 2026-08-17). The numeric
width gate was met; the bot leans slightly wide, which the operator prefers, and the k_tail widening
holds. Five misses and two controls were traced and verified. The frozen-triple checkpoint is
closed.

| Read | Value |
|---|---|
| Live roster, strict spot peer (n 63) | +21.99 |
| Six-model era, strict spot peer | +12.70 |
| Type-adjusted gap | +10.71 [+1.72, +19.69] |
| Horizon-matched gap | +8.19 [-3.21, +19.90] |
| Numeric PIT standard deviation (n 27; uniform is 0.2887) | 0.2646 |
| Numeric 80% coverage | 0.889 |

**Standing tooling landed on the branch.**

- Era gap reader, `metaculus_bot/performance_analysis/era_gap.py` (e5f23a4, 405f878, eff72aa):
  horizon matching as a standing part of the read, a curated cluster map via `--clusters`, and
  `--era-field` so the fall read runs once the tagging pass writes `fall_config`. The watch rule is
  two-sided: a concern only below -5 with the cluster interval excluding zero; a favourable read is
  reported and never flagged.
- Fall preregistration, `scratch_docs_and_planning/fall_2026_preregistration.md` (aa27e7c): six
  reads, n floors of 15 and 30, the season's commitments.
- Cost telemetry (e00ff44, c553b88, f4fa773, 292b340): a `CREDIT_RUN_SUMMARY` line per run,
  `make cost_report`, a `PROMPT_SIZE_ALERT` at 150k tokens, and `charged_usd`, `byok_calls` and token
  fields on `CREDIT_ROLE_SPEND`. The ledger had double-counted the personal key's bring-your-own-key
  (BYOK) calls; the new field separates them.
- Ghost instruments (0cf14f6; `scripts/score_ghosts.py` prints both reads). The published-versus-
  ghost "loop_moved -3.82" is NOT a read on gap-fill v2, because both arms carry v2. The same-driver
  pre-versus-post read is +7.18 log points on n 36 [+1.26, +13.99]. A v1 ghost
  (`GHOST_FORECAST_V1`) now measures gap-fill v1 the same way.
- Gap-fill v1's resolver now sees the resolution criteria and fine print (9c49378; the receipt is
  question 44267 at -95.66).
- Dispatcher: cron-job.org fires the GitHub Actions workflows because GitHub delivers about 22% of
  this repo's scheduled runs. Jobs: 8417341 tournament at :02 and :32 UTC, 8417342 cup at :12 and
  :42, 8417343 Mantic at :01 and :16, DISABLED until `run_bot_on_mantic.yaml` is on `main`. The
  GitHub personal access token expires around September 2027. More than 35 dispatched runs
  succeeded today. `make dispatch_watch` shows delivery per day. The skip-previously-forecasted
  guard now fails shut (`SKIP_GUARD_UNREADABLE`).
- Fetch reach in the gap-fill v2 loop: four rungs ported from the resolution-source ladder (21f3122
  HTML extraction steps, 8eba338 Wayback with the capture date, 246fe68 the harvested JSON feed on
  an empty render, 924cfc2 a driver note to stop scraping FRED, Kalshi and Yahoo Finance).
- Shared-ladder unification plan, `scratch_docs_and_planning/fetch_ladder_unification_plan_2026-09-09.md`
  (f120e23 through 0f51815): the implementer's plan, the rung-port constraints, and the three
  resolved decisions.
- SEC EDGAR client, `metaculus_bot/research/sec_edgar.py` (f692c0c, 820b1a7): standalone, not yet
  wired. It refuses to dial without `SEC_EDGAR_CONTACT_EMAIL` (the SEC fair-access policy wants a
  User-Agent naming a contact; no account needed). The operator set that variable in `.env` and as
  a GitHub Actions repository secret on 2026-09-09; the workflows do not read it yet.
- Public data APIs ranked, `scratch/cost_pass_2026-09-09/public_apis/PUBLIC_APIS.md`: FRED and Yahoo
  Finance top the list of blocked hosts and already have clients; FSIS recalls, BLS, OCHA, mesonet,
  ClinicalTrials.gov, congress.gov, EIA and PortWatch come next.
- Docs: README rewritten for outside readers (66ea12f); FUTURE.md (a120297) and AGENTS.md (5d8943c)
  cleaned; the exemption-marker rule corrected (b714c18; deleting a marker by fixing code is
  welcome, adding an ignore is the wrong direction); 17 function-level imports hoisted (8b77859).
- Blog material for the summer write-up, `scratch/blog_summer_2026/` (`IMAGES.md`, `images/*.png`
  light and `*_dark.png`, `contact_sheet_dark.png`, `STORIES.md`, `PROBABILISTIC_TOOLS_SUMMARY.md`,
  `PROBTOOLS_PROD_TRACE.md`, `TFIT_REPLAY.md`).

**Cost pass and gap-fill evidence** (`scratch/cost_pass_2026-09-09/`: `COST_PASS.md`,
`cost_anatomy.md`, `v2_cost_anatomy.md`, `ghost_instrument.md`, `v1_vs_v2/V1_VS_V2_GATE.md` and
`TRACES_SYNTHESIS.md` with `traces/`, `v1_gap_redundancy/REDUNDANCY.md`,
`fetch_gap_inventory/INVENTORY.md` and `FUTURE_ENTRIES.md`, `section_token_cost/SECTION_TOKENS.md`,
`public_apis/`, `api_tools/` in progress; the resolver probe outputs are
`scratch/probes/gap_fill_resolver_probe_*.md`). The v1 resolver is the largest cost line. The blind
v1-versus-v2 comparison over 99 bundles with both passes on came out mixed and complementary: v1
builds reference classes, v2 reads the resolving instrument. About a third of v1's calls buy nothing
(future-dated gaps are 18% of them, and re-fetched dated readings the rest; about $0.21 a question).
Paraphrase repeats appear on one question in three. A positional cap of 2 drops the useful gap on 4
of 6 traced questions, so any lean on v1 must filter by grade, never by position. The resolver probe
(64 calls, $5.97) showed `NATIVE_SEARCH_CONTEXT_SIZE` is inert (1.02x and 0.96x), and gpt-5.6-luna
costs 0.21x terra per call but repeats the definitional error and fabricated a citation at medium
effort, so the resolver stays on terra at low effort. The whole forecaster prompt is median 17.5k
tokens (max 24k); carrying the bundle to three forecaster slots costs about $0.13 of input a question
and the slots are 60% to 85% output and reasoning. The operator: "17k is extremely not-concerning".

| Line | Cost per question |
|---|---|
| Gap-fill v1 resolver | $0.76 |
| Gap-fill v2 loop | $0.36 |
| Three forecasters together | $0.76 |
| v1 calls that buy nothing | about $0.21 |

| Blind v1 vs v2 lens (v1 better / v2 better / tie) | Counts |
|---|---|
| Forecaster lens | 59 / 32 / 8 |
| Resolution lens | 28 / 24 / 11 |

## Current state

**Deployed where.** Production runs from `main` through the four `run_bot_on_*.yaml` GitHub Actions
workflows, fired by the cron-job.org dispatcher. Nothing from this branch is live yet. The Mantic
workflow is on the branch only, so its dispatcher job stays disabled. `run_bot_on_minibench.yaml` is
disabled by design and never was enabled; the cup workflow is enabled for the fall.

**What is running right now** (verified at 22:36 PT with `tmux ls` and the log tails).

- tmux session `strip_bench3`, log `~/logs/strip_bench3.log`, output directory
  `scratch/probes/section_strip_bench_20260910T052824Z/` (`bench.log`, `calls.jsonl`, `run.json`;
  `SUMMARY.md` and `results.json` land at the end). This is the section-strip bench:
  `make strip_bench ARGS="--i-accept-spend --concurrency 10"`, 756 calls (63 resolved both-passes-on
  questions, 4 arms, 3 replicates) on the cheap `meta/muse-spark-1.3-contributor` model over the
  operator's personal OpenRouter key. Estimate $1.23, cap $10. It started at 22:28 PT after the
  operator cleared two OpenRouter gates: the 18+ attestation, then the default workspace guardrail
  on paid training at https://openrouter.ai/workspaces/default/guardrails (the account privacy
  toggles alone were not enough). Two earlier attempts (`..._20260910T004916Z`, `..._20260910T051516Z`)
  failed at $0 on those gates. At 22:36 PT 49 of 756 calls had logged. Rescore offline with
  `make strip_bench ARGS="--rescore <dir>"`.
- Teammate `fred-kalshi-yahoo-check` (running). Output goes to
  `scratch/cost_pass_2026-09-09/api_tools/API_TOOLS.md`; its result JSON files are already there
  (`live_smoke_results.json`, `archive_financial_market_results.json`, `telemetry_markers_results.json`,
  `url_shapes_results.json`; logs `~/logs/api_tools_*.log`). It is checking that the deterministic
  FRED, Yahoo Finance and Kalshi providers fire in production (the residual round read zero rows on
  the `financial_data` markers, and a FRED-id hallucination defect was recorded on 2026-09-01),
  running a live smoke, checking Kalshi host drift (the repo dials `api.elections.kalshi.com`; the
  docs name `external-api.kalshi.com`), and designing three deterministic loop tools (`fred_series`,
  `market_snapshot`, `yahoo_history`) with URL-to-tool translation at the ladder's rung 0. Its report
  goes to the lead session's transcript; a successor reads `API_TOOLS.md`.
- Every other teammate from today is finished and idle; none owns anything on disk. No session crons.

**What passes.** The free gate run for this handoff at the tip (22:37 to 22:41 PT, log
`~/logs/handoff_gate_2026-09-09.log`): Ruff clean, deptry clean, import-linter 6 contracts kept,
basedpyright 0 errors, and `make test` at 9,767 passed, 33 skipped, 5 deselected, exit 0.

## What's left to do

Steps 1 and 2 are independent of each other and of step 3. Step 3's chunks (a) through (f) are
independent of each other; the ladder chunk (b) should land before (c) and (d) because they build on
it. Step 4 depends on the merge. Step 5 is independent housekeeping.

1. Read the bench results when `strip_bench3` exits (the log ends with an `EXITCODE=` line). Relay
   the per-arm scores and the paired full-minus-arm deltas with bootstrap intervals from
   `SUMMARY.md`; the per-question table is in `results.json`. Caveat for the reader: a Sonnet-tier
   model may lean on spoon-fed facts more than the frontier trio, so the deltas are an upper bound.
   If replicate spread is nonzero and the cap has room, a 5-replicate rerun is allowed (the operator
   approved about $10 total). [independent]
2. Read `scratch/cost_pass_2026-09-09/api_tools/API_TOOLS.md` when the FRED/Kalshi/Yahoo teammate
   finishes. If the deterministic providers are not firing in production, that is a bug to fix
   before anything else in step 3. [independent]
3. Fall merge implementation, test-first, ON `mantic-competition`. The operator ruled out a
   separate branch or pull request ("fun side project, throw it in here"). Run `/forge` on each
   chunk; free gates only. The pre-merge smoke, one `test_bot_basic.yaml` dispatch at about $2,
   fires only on the operator's explicit sign-off at that time; they have more changes inbound
   before the merge, so do not propose it early.
   (a) Grade-based gap-fill v1 filter and dedupe in code: the analyzer emits per-gap structured
   fields `answerable_now`, `already_in_first_pass` and `same_need_as`; gaps failing them are
   dropped; no second LLM pass; the cap stays 4. Files: `metaculus_bot/research/targeted.py`
   (analyzer schema), `metaculus_bot/prompts.py` (`gap_fill_analyzer_prompt`), pins in
   `tests/prompts/` with a presence pin per the `docs/prompts.md` rules.
   (b) The shared fetch ladder per `fetch_ladder_unification_plan_2026-09-09.md`: six migration
   commits with golden replay tests. The 86 pre-existing comment findings in `tools.py`,
   `fetch_outcomes.py`, `provenance.py`, `derived_api.py` and `test_agentic_tools.py` move to docs in
   the extraction commit; the `tools.py` split is absorbed by the package layout.
   (c) The three deterministic API tools (`fred_series`, `market_snapshot`, `yahoo_history`) plus
   rung-0 URL translation, and the SEC EDGAR client wired behind `SEC_EDGAR_CONTACT_EMAIL`. The
   operator has already set that variable in `.env` and as the GitHub Actions repository secret of
   the same name (both on 2026-09-09 at 22:45 PT). What remains is surfacing the secret into the
   env blocks of all four `run_bot_on_*.yaml` workflows when the client is wired in. The value is
   the operator's personal address: never commit it, and keep the commented placeholder in
   `.env.template` as it is.
   (d) The `page_digest_extractor` role: constants `PAGE_DIGEST_EXTRACTOR_MODEL` and
   `PAGE_DIGEST_EXTRACTOR_EFFORT` in `constants.py`, `CREDIT_ROLE_SPEND` billing under that role
   name, marker fields `passages_returned`, `passages_grounded` and `fallback_used`.
   (e) RESOLVED 2026-09-09 (operator): no fire-rate flag and no extra logging in the forecaster prompts ("let them focus on what matters"). Read the two new prompt rules by hand-coding rationales after the fact, per the preregistration, with a small subagent workflow when the sample is large enough.
   (f) The tagging pass writes `fall_config` into `triple_subera_fine`
   (`scratch/residual_2026-09-09/bucket_by_era.py` is where the field is written today), so the era
   gap fall read works:
   `--era-field triple_subera_fine --treated-era fall_config --comparison-era ranked_markets`.
4. After the merge: enable the Mantic dispatcher job with
   `make cronjob_dispatch_setup ARGS="--apply --enable-mantic"` once `run_bot_on_mantic.yaml` is on
   `main` (paid dispatches; ask first). Watch `make dispatch_watch` and `make cost_report`. Run the
   first fall residual read at 15 resolved records, per the preregistration.
5. Housekeeping the operator flagged. [independent]
   - Replace the two `HARNESS-SCAN-EXEMPT-broad-except` markers added to
     `metaculus_bot/ablation/research.py` today (lines 184 and 237) by catching the expected
     exception types instead. The operator wants this fixed eventually; the next session should
     also give it a one-line FUTURE.md entry so it is not lost.
   - Split `tests/test_cli.py` (1,927 lines; not urgent).
   - Reconcile FUTURE.md with `docs/operations.md` on the post-651 Mantic per-bin smoke: FUTURE.md
     treats it as run and reviewed, `docs/operations.md` still lists firing it as operator step 2.
     The peer session ran it on 2026-09-09 (see the 2026-09-08 handoff, "Session 4 checkpoint").
   - The Kalshi host drift (above).
   - FRED's terms-of-use clause on LLM use: the operator's licensing call.

## Key decisions made (don't re-litigate)

- The fall merge is built on `mantic-competition`, in one merge, with no separate branch or pull
  request. The operator's words: "fun side project, don't want to split off a separate PR; throw it
  in here."
- Keep both gap-fill passes for the fall. The operator does not want to trade performance for cost,
  and the blind comparison says the passes are complementary.
- Lean v1 by grade-based filtering in code, never by a positional cap. A cap of 2 dropped the useful
  gap on 4 of 6 traced questions. No second LLM pass; the analyzer emits the fields and code drops
  the failing gaps; the cap stays 4.
- The resolver stays `openai/gpt-5.6-terra` at low effort. Luna is 0.21x the cost but repeats the
  definitional error and fabricated a citation at medium effort. No per-gap model routing on
  current evidence. `NATIVE_SEARCH_CONTEXT_SIZE` stays as it is because the probe showed it inert.
- Build the shared fetch ladder per the plan. Its three open decisions are resolved yes: share the
  throttle-phrase check across both callers; digest long pages by LLM-extractive quotes with a
  literal-substring grounding check (role `page_digest_extractor`, `openai/gpt-5.6-luna` at MEDIUM
  effort, BM25 only as a pre-filter and fallback, the opening passage kept); surface the Wayback
  capture date to the driver (the 30-day bound stays for cited sources).
- Agents may call FRED, Kalshi and Yahoo Finance by API as deterministic tools (no LLM cost) with
  URL-to-tool translation. OpenAI native search and the paid Google page read are LATER tiers inside
  the loop, capped, and the prompt does not push them.
- Heavy tools (screenshot plus vision, OCR, video) stay in FUTURE.md at the bottom of high priority.
- Deleting a scan-exemption marker by fixing the code is always welcome. Adding an ignore to silence
  a linter is the wrong direction.
- The v1 ghost and the same-driver pre-versus-post read are the instruments for v2 and v1. The
  "loop_moved versus published" read is retired because both arms carry v2.
- Roster fixed for the fall; k_tail held; no stacking, no mean, no probabilistic tools (all
  benchmarked and rejected in earlier rounds, see memory `project_settled_dead_paths`).
- Charts for the blog: dark Swiss idiom by default, signed by Claude, with a writing pass before
  handover.
- Fan-outs of 5 to 10 agents per workflow.
- Constants, `numeric/config.py` and `llm_configs.py` keep their receipt comments; the ladder files'
  86 findings move with the ladder refactor.

## Gotchas & context

- Everything under `scratch/` is gitignored (read it with `cat` or `rg --no-ignore`);
  `scratch_docs_and_planning/` is gitignored too but its files are tracked, so add them by path
  with `git add -f`.
- `CLAUDE.md` is a symlink to `AGENTS.md`; edit `AGENTS.md`.
- Paid runs: every command that spends goes through the operator, and approvals are per run. The
  bench and the resolver probe were each approved once.
- The personal OpenRouter account carries an OpenAI bring-your-own-key (BYOK) key, so those calls
  bill the operator's OpenAI account. Known and fine.
- `SEC_EDGAR_CONTACT_EMAIL` holds the operator's personal email address. It lives in `.env` and in
  the GitHub Actions secret only; never write the value into a tracked file.
- Shared-index collisions happened twice today when two agents staged at the same time (hunks rode
  into another agent's commit). Commit by path, one agent per file, and retry on
  `.git/index.lock: File exists`.
- Era boundaries are merge-to-main committer timestamps, never authoring dates. Rank on spot peer,
  never `peer_score`. Import cohorts from `performance_analysis/cohorts.py`. Markers: add fields,
  never rename.
- The smell stop-gate makes pre-existing findings in any file an agent edits that agent's to fix.
  Today's rulings: `constants.py`, `numeric/config.py` and `llm_configs.py` receipt comments stay;
  the ladder files' 86 findings move with the ladder refactor.
- Pyright LSP diagnostics are stale or wrong-interpreter noise; `make typecheck` (basedpyright) is
  the authority.
- `--mode mantic` refuses the donated OpenRouter key and requires
  `DONATED_OPENROUTER_KEY_ENABLED=false`, so on a Mantic run every OpenRouter auth error is the
  personal key.
- `gh` needs `--repo No-Stream/metaculus-bot`; `origin` is the fork and `upstream` is the template.
- Memory: `~/.claude/projects/-Users-flatljan-personal-metaculus-bot/memory/` carries the durable
  facts (`project_mantic_integration_2026_09_08.md`, `project_current_config.md`); update it when
  the state changes.

## References

- Previous handoff (Mantic build, the per-bin smoke, merge readiness):
  `scratch_docs_and_planning/handoff-2026-09-08-mantic-phase2.md`
- Fall preregistration: `scratch_docs_and_planning/fall_2026_preregistration.md`
- Shared fetch ladder plan: `scratch_docs_and_planning/fetch_ladder_unification_plan_2026-09-09.md`
- Residual round: `scratch/residual_2026-09-09/` (`SYNTHESIS.md` first)
- Cost pass: `scratch/cost_pass_2026-09-09/COST_PASS.md`
- Resolver probe outputs: `scratch/probes/gap_fill_resolver_probe_*.md`
- Bench: `~/logs/strip_bench3.log`, `scratch/probes/section_strip_bench_20260910T052824Z/`
- Dispatcher script: `scripts/cronjob_dispatch_setup.py`; delivery view: `make dispatch_watch`
- Era gap reader: `metaculus_bot/performance_analysis/era_gap.py`; conventions:
  `docs/performance_analysis.md`
- Cost telemetry and the shared-versus-personal key model: `docs/operations.md`
- Repo guide: `AGENTS.md`; design log: `FUTURE.md`
