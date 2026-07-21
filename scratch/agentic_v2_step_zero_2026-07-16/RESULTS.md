# Step zero for gap-fill v2: what fraction of our worst misses would better research have fixed?

**Date**: 2026-07-16. Free/read-only work per the v2 plan §7 evaluation ladder step 0
(`scratch_docs_and_planning/agentic_gap_fill_v2_plan.md`).

## Headline

**5 of 32 worst recent misses (16% by count, 20% by peer-score loss) are
research-comprehension failures (buckets A–D) that the v2 agentic research loop plausibly
fixes. The other 84% are judgment, calibration, pipeline, or genuine-surprise failures
(E–F) that better research cannot touch.**

A large sub-story: 5 of the 27 E/F misses (and ~30% of all peer-loss in the sample) are the
**numeric open-bound tail-truncation pipeline pathology** — models correctly declared large
above/below-bound mass, the submitted CDF shipped only 2.5% beyond the bound. That is the
single biggest addressable loss source in this sample and it is a *pipeline* fix, not a
research fix (partially addressed by the 2026-07-12 nominal-bounds prompt rendering +
OPEN_BOUND_PILING telemetry — all five instances predate that fix).

## Method

- **Selection**: worst 32 misses by Metaculus peer score among recent-era resolved
  questions: all of summer-futureeval-2026 plus spring-aib-2026 forecasts submitted on/after
  2026-03-01 (155 + 45 = ~200-question pool; the cutoff keeps the roster close to
  current — claude-4.6-opus onward — and excludes the January/February lineups).
  Type mix of the worst-32: 14 binary, 9 numeric, 7 MC, 2 discrete — roughly proportional
  to the pool.
- **Control**: 10 randomly sampled (seed 20260716) good outcomes (peer > +20) from the same
  pool, read with the same critical lens to check the buckets aren't hindsight artifacts.
- **Data**: `scratch/coherence_2026-07-15/perf_all_tagged.json` (per-model forecasts,
  resolutions, scores, full comment text) + fresh question detail (description, resolution
  criteria, fine print) pulled read-only from the Metaculus API + the research bundle each
  question's forecasters actually saw (`backtests/research_archive/latest/<qid>.json`,
  re-synced via `make sync_research` this session; **coverage was 42/42 — no question
  lacked a research artifact**).
- **Process**: `build_dossiers.py` assembled one dossier per question (question + criteria +
  resolution + published forecast + per-model forecasts + research bundle + full forecaster
  rationales). Ten parallel Opus classifiers each read 4–5 dossiers in full against
  `RUBRIC.md` (taxonomy A–F, decision procedure, strict output format, no-web-lookup rule)
  and wrote per-question classifications to `classifications/batch_*.md`. I (session lead)
  spot-checked the five worst dossiers independently and reviewed every classification;
  disagreements noted below.

### Taxonomy

A missing fact / B stale fact / C misread resolution mechanics / D hallucinated-misattributed
research (all v2-addressable) vs E judgment-or-pipeline / F genuinely surprising resolution
(not v2-addressable). One primary bucket per question; E covers both "weighed adequate
research badly" and "pipeline distorted an adequate forecast" (sub-tagged below).

## Tallies

### By primary bucket (n=32 misses)

| bucket | n | share | peer-loss | loss share |
|---|---|---|---|---|
| A missing fact | 1 | 3% | −19.6 | 2% |
| B stale fact | 0 | 0% | 0 | 0% |
| C misread mechanics | 2 | 6% | −66.4 | 6% |
| D hallucinated research | 2 | 6% | −125.3 | 12% |
| E judgment/pipeline | 25 | 78% | −746.6 | 72% |
| F genuine surprise | 2 | 6% | −78.0 | 8% |
| **A–D (v2-addressable)** | **5** | **16%** | **−211.3** | **20%** |

Sub-split of E (from classifier justifications):

- **E-pipeline (open-bound tail truncation)**: 43746, 42112, 43747, 43137, 43147 — n=5,
  peer-loss −314.1 (30% of the sample's total loss). Models declared 22–78% out-of-bound
  mass (`above_max_expected`/`below_min_expected`, percentiles piled at the bound); the
  submitted CDF shipped ~2.5%. All five predate the 2026-07-12 nominal-bounds fix.
- **E-judgment (weighing failures)**: the other 20 — market herding (42687, 42688, 43728,
  43727), base-rate neglect/overriding own arithmetic (42577, 42864, 43141, 42312*, 42861),
  overconfidence/narrative anchoring (42313, 42298, 42299, 42751, 42740, 42865, 42591,
  43656, 43048), tail width (42869 too wide, 43077 too thin, 43652 too wide).

*42312 classified A primary (see per-question line) — listed here because it also has a
strong E component.

### By era

| era | n misses | A–D | E–F | A–D share |
|---|---|---|---|---|
| spring-aib-2026 (Mar 1 – Apr 13 submissions; 5-model claude-4.6-opus roster) | 26 | 5 | 21 | 19% |
| summer-futureeval-2026 (May 19 – Jul 1 submissions; current 6-model roster) | 6 | 0 | 6 | 0% |

All five A–D cases are spring-era. The summer worst-6 are two open-bound pipeline cases
(now fixed), two market-herding cases (Monsanto), one MC bucketing judgment (Armenia), one
width judgment (crude-vs-S&P spread). Small n (6) — but the direction is consistent with
the research stack having improved (resolution-source fetcher and prediction-market
snapshot shipped after spring) and with the pool: summer's worst peer scores are much
shallower than spring's (−78 min vs −116 min).

## Per-question one-liners (misses, worst first)

| qid | peer | bucket | one-liner |
|---|---|---|---|
| 42304 | −116 | **D** | 81% YES on an INES-3 event built on a single native-search claim (Turkish Feb-2026 "Level 3" incident, hallucination-pattern NucNet URL, absent from AskNews, never IAEA-listed); resolved NO. |
| 43746 | −78 | E-pipe | Minions & Monsters: research correctly said $53–62M vs $75M floor; models set below_min mass 0.25–0.78; published CDF shipped 2.5% below bound. Resolved $37M. |
| 42112 | −76 | E-pipe | ATP points Apr 27: two models correctly called below-floor resolution, "forced to pile up at the lower bound"; aggregate shipped 2.5% below. Resolved below_lower_bound. |
| 42855 | −67 | **F** | UCL QF first legs all ≤3 goals — a ~15–22% tail with real precedent; whole field missed (baseline −188); our 89% vs base-rate ~78–82% is mild E on top. |
| 43747 | −66 | E-pipe | Toy Story 5: research had "$150–175M" (truth $159.7M); models declared 0.22–0.52 above-bound mass; CDF shipped 2.5%. |
| 43048 | −64 | E | Copernicus wording: bundle carried the March-2025 "lowest on record" precedent + 2026 record-low reporting, but 3/5 models claimed "no 2026 data provided" and used stale priors (truncated summarizer output a plausible contributor — see side findings). |
| 42861 | −62 | E | Senate judges: 4/5 over-weighted nominee "supply" vs floor-time bottleneck; gemini got 23% from the same bundle; resolved NO exactly via its mechanism. |
| 43137 | −59 | E-pipe | Cloud Next count: forecast made 12 days before the event; models flagged 10–15% overflow >250 but bound capped expression at 2.5%; resolved above_upper_bound. |
| 42864 | −58 | E | CDC novel flu: bundle had the 67→3 YoY collapse + 6 dry FluView weeks; models Poisson-extrapolated 3 early cases to 65–84%. |
| 43145 | −53 | **C** | Anthropic Opus-class: models narrowed "available to any external users" to "generally available," excluding the in-bundle Mythos private preview (11–100 external companies); resolved YES. |
| 42577 | −47 | E | Faroe: 5/5-election base rate ("no party ≥10 seats") in bundle and cited, then overridden by naive seat math on one pre-campaign poll the bundle itself flagged stale. |
| 42313 | −41 | E | Apple exploited-language: bundle contained the counterexample (Apple patching exploited CVEs without the language); Claudes went 88–97% on the Coruna narrative. |
| 43147 | −35 | E-pipe | NATO Article 4: research said 0 requests (truth 0); every model put P50 ≤0.2; discrete CDF build smeared the atom-at-0 across 0–32; catastrophic vs baseline +77. |
| 43727 | −28 | E | Monsanto vote count: herded down to 4–5 justices off oral-argument vibes + thin markets against the in-bundle SG-support ~75–80% base rate; 7-2 Monsanto win. |
| 42298 | −20 | E | VIX: anchored on the fresh spike + futures level, discounting in-bundle mean-reversion evidence; resolved 15–20 bucket. |
| 42312 | −20 | **A** | Android bulletin: bundle lacked the long-run base rate (~75–85% of monthly bulletins have a Critical Framework/System entry) and likely miscounted Dec-2025; models anchored on a mistabulated "2/5 recent months" → 52% on a clear YES. |
| 43077 | −16 | E | "Michael" gross: bundle had "presales killing it / potentially $100M"; ensemble centered $82M with thin right tail; resolved $97.2M. |
| 43059 | −14 | **C** | Brent high: briefing pointed forecasters at the S&P physical dated-Brent print ($141.36) as if it were the EIA spot series the question resolves on (in-bundle FRED cite showed ~$109–122); median $146 vs resolution $116.63. |
| 42751 | −12 | E | UK reshuffle: best-sourced leak said changes depend on the May elections (after the window); two Claudes at 55–65% dragged the median to 44%; resolved NO. |
| 42236 | −11 | **F** | Duke: ~18% on the realized Elite Eight bin ≈ the #1-seed base rate; ordinary tournament variance, mild deep-run tilt. |
| 42299 | −11 | E | S&P 500: bundle's strategist targets (~7,269) bracketed the truth (7,209); models pushed medians below spot on the March shock. |
| 42687 | −11 | E | Iran invasion: herded on a Polymarket line that measured a looser event and was wrong; 30% vs strict criteria + 45-year restraint base rate. |
| 42591 | −10 | E | UNMISS: 2025 same-mission precedent (unanimous technical rollover) in bundle, under-weighted; 63% YES resolved NO. |
| 42107 | −9 | **D** | ATP points Mar 30: synthesis asserted Alcaraz defends "410 points" in March, contradicting the in-bundle Virgilio "1,010 points expiring (IW + Miami)"; all five anchored on 410, ~700 points high. |
| 43141 | −8 | E | FrontierMath: own Poisson said ~15% for a new solve; ensemble put 45% on ≥1; resolved "1 or fewer". |
| 43728 | −7 | E | Monsanto binary: models computed Beta-binomial posteriors ~0.70–0.75, declared 0.58–0.63 after herding to a $272-volume Polymarket; resolved YES. |
| 42740 | −7 | E | TSMC month: models' own outside-view anchors favored March; blended down to January on guidance-midpoint arithmetic; resolved March. (Secondary A: historical monthly series absent.) |
| 42869 | −7 | E | Goods trade balance: median dead-center (−82 to −92 vs −87.9); deliberately over-widened tails to −150/−170 on a series that moves ~$10–15B. |
| 42865 | −7 | E | OFAC Belarus: bundle said no planned actions, priorities elsewhere; two models invented a "follow-on cleanup" hypothesis at 72–85%. |
| 42688 | −6 | E | Hormuz: directionally right (28% YES, resolved NO) but pulled up off a market measuring a harder threshold; disciplined models said 7–12%. |
| 43656 | −6 | E | Armenia seats: gap-fill memo pointed at ~64% seat share (resolving bucket); 3/6 models anchored on the 52% floor bucket instead. |
| 43652 | −5 | E | Crude-vs-S&P spread: median essentially exact; distribution wider than peers (took option-implied vol at face value, no variance-premium discount). |

## Control sample (10 good outcomes, same critical reading)

9/10 **clean** — bundle adequate, mechanics understood, forecast well-founded. Notables:

- 43745 (UMich sentiment): the bundle *caught and corrected* a stale-FRED-49.8 trap
  ("Use 44.8 as authoritative") — the pipeline actively doing what a B-bucket miss would
  look like when it fails. Direct evidence the B bucket isn't hindsight (we found zero B
  in misses, consistent).
- 43131 (TSA volume): one latent-B observation — a provider slice was staler (Apr 2) than
  siblings (Apr 9), but the forecaster demonstrably used the fresher figures; non-biting.
- 42116 (Gemini #1 on LM Arena): clean of A–D, but a **latent E-pipeline** case — the
  decisive current-standings fact was in the bundle while 3/5 forecasters said they had no
  briefing and the Research Summary was a one-line stub; right answer from generic priors.
  Same truncated-summary signature as miss 43048.

Verdict: the same critical reading applied to good outcomes produces almost no A–D flags,
so the A–D labels on misses are unlikely to be hindsight artifacts. If anything the control
read surfaced the *same* two systemic non-research issues the misses did (summarizer
truncation, staleness-with-recovery).

## Side findings (not the question asked, but load-bearing)

1. **Open-bound tail truncation was the #1 fixable loss source in this sample** (5 misses,
   −314 peer). The 2026-07-12 nominal-bounds prompt fix + OPEN_BOUND_PILING telemetry
   target exactly this; all five cases predate it. Worth watching the telemetry to confirm
   the fix holds; 43147 (discrete atom-at-0 smearing) may be a distinct sub-pathology worth
   its own look — the models' declared percentiles were sub-integer and tightly clustered
   near 0, and the 33-bucket discrete grid turned that into a smear.
2. **Truncated/stub Research Summary sections** appear in miss 43048 and control 42116
   (verified in both dossiers: summary cut off mid-heading or one line long), and in both
   cases forecasters behaved as if they had no briefing even though the full research was
   embedded below the summary in the prompt. Cheap fix candidates: guard against
   summarizer truncation, or assert non-empty section bodies. (The framework embeds the
   full research under `# RESEARCH` regardless, so the failure suggests models sometimes
   over-index on the summary section.)
3. **Median-vs-best-model**: in at least 5 of the E-judgment misses (42861, 42688, 42687,
   42751, 42865) one forecaster read the same bundle correctly and was drowned out by the
   median — consistent with the long-standing "ensemble never beats the best individual"
   finding in `scratch/analysis_reasoning_failures.md` (Finding 3).
4. **Market herding remains the most repeated E-judgment signature** (4+ cases), matching
   the 2026-Q1 analysis (Finding 2 / Faulty Model 1). The "STRONG EVIDENCE" market clause +
   liquidity labels shipped since then; the Monsanto cases (May 29) post-date it and still
   herded — including on a $272-volume Polymarket the bundle itself flagged as thin.

## What this means for v2's expected value

- **The prize is real but modest on this evidence: ~16–20% of worst-miss loss.** The three
  strongest cases are exactly the failure classes v2 was designed for: 42304 (verify a
  load-bearing single-source claim → fetch the IAEA page; the research summary even told
  forecasters to verify it and nobody could), 43059 (open the resolution source / FRED
  series and quote the operative number → wrong-series conflation dies), 42107 (chase the
  "Miami defenses unspecified" lead → the 410-vs-1,010 contradiction resolves). All three
  are fill/verify targets a driver with fetch hands should hit.
- **The verify half of v2 matters more than the fill half here.** 3 of 5 A–D cases (both
  D's and the C-Brent) are "the bundle contained wrongness," not "the bundle lacked a
  fact." Only 42312 is a clean missing-fact A. The v1 gap-fill can only add facts; it
  cannot check them — that's the structural gap v2's dry-run + verify targeting closes.
- **Don't expect v2 to move headline peer score much by itself.** 72% of loss is E, and the
  largest single E cluster (open-bound piling) is already fixed by other work. The
  E-judgment signatures (market herding, base-rate override, arithmetic distrust) are
  prompt/aggregation problems documented since Q1 — v2 won't touch them.
- **Caveat on the era split**: zero A–D in summer's worst-6 hints the current stack
  (resolution-source fetcher, market snapshot, gap-fill) already closed part of the
  research gap — but n=6 and the summer pool is younger (fewer resolved). Re-run this
  audit on summer-only data once ~30 summer misses have resolved before treating 16–20%
  as the current-config prize.

## Caveats

- **Hindsight bias**: partially controlled (10 good outcomes read the same way found ~0
  A–D), but the miss classifiers knew the resolution when reading; bucket boundaries
  (especially E-vs-F and C-vs-E) are judgment calls. Six classifications are
  medium-or-lower confidence (42304, 43145, 42107, 43059, 42312, 42236); the A–D headline
  spans 3 high-confidence + 3 medium cases — a reasonable range is **3–7 of 32 (9–22%)**.
- **Single-classifier subjectivity**: each dossier was read by one Opus classifier +
  spot-check review; no independent double-classification. The rubric and quoted evidence
  make each call auditable (`classifications/batch_*.md`).
- **n=32 from a ~200-question pool**; worst-miss selection over-samples tail outcomes
  (which inflates F and pipeline-blowup representation relative to "typical" misses).
- **Classifiers could not verify world facts** (no web access by design); D-vs-C-vs-F for
  42304 and 43145 rests on dossier-internal evidence.
- **Peer-loss weighting** treats peer score as the loss currency; tournament scoring is
  nonlinear and the worst questions dominate.

## Files

- `build_dossiers.py` — dossier assembly (read-only Metaculus API + local archives)
- `RUBRIC.md` — the binding taxonomy + procedure given to classifiers
- `index.json` — selection manifest (42 questions, scores, artifact coverage)
- `dossiers/` — 42 per-question dossiers (~3.5MB total; gitignored data)
- `classifications/batch_{M1..M8,C1,C2}.md` — per-question classifications with quoted
  evidence
