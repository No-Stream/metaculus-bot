# Base-rate audit — sizing the prize for sourced base rates (2026-07-16)

Operator hypothesis under test: forecaster models invoke base rates from memory, unsourced;
if those remembered rates are systematically off even modestly, fixing them via research
(the v2 gap-fill BASE-RATE targets) is a broad, compounding gain invisible to worst-miss audits.

## Method

- **Sample**: 40 resolved questions from `scratch/coherence_2026-07-15/perf_all_tagged.json`
  (20 binary, 10 MC, 10 numeric incl. discrete), spring-aib-2026 (31) + summer-futureeval-2026 (9),
  stratified by peer-score tercile within type (`sample_questions.py`, seed 20260716).
- **Extraction**: 2-3 per-model reasoning sections per question (rotating which models),
  resolution and scores stripped from the rationale files to blunt hindsight bias.
  Five reader passes (subagents + one direct pass for a stalled batch) recorded every explicit
  reference-class rate: verbatim quote, stated rate, class, basis (memory / research-attributed /
  computed-from-briefing-data), source citation, load-bearingness (anchor / adjustment / passing).
  Outputs in `extracted/claims_batch*.json` (batches 2b/4b are fallback re-runs of stalled
  originals; `tally.py`/`correlate.py` use the original when both exist — batch 2 original landed
  late and supersedes 2b in the tallies; batch 4 uses the 4b fallback).
- **Verification**: the 22 most load-bearing distinct claim clusters (~45 of the extracted
  claims; several clusters bundle 2-4 models' versions of the same class) checked against citable
  data via free web lookups (Wikipedia, ABS, Transparency International, TSMC IR, icj-cij.org,
  OPCW/UN records, sports references, election archives). Outputs in `verified/*.json`.
- **Correlation**: descriptive join of claim basis against peer score (`correlate.py`). n is small;
  no significance claims.

## Headline tallies

**Base-rate reliance is near-universal.**

- 111/114 (97%) of (question, model) rationales lean on ≥1 explicit reference-class rate;
  37/40 questions have at least one leaning rationale. (The 3 without are comment-trimming
  artifacts where only a stacker meta-analysis section survived, not evidence of non-use.)
- 343 claims total, ~3.0 per rationale. 288 numeric, 55 qualitative.
- **Basis: 226/343 (66%) pure memory, 73 (21%) computed from briefing data, 44 (13%)
  research-attributed.** Only 64/333 load-bearing claims (19%) cite any source.
- The prompt template's PHASE-1 "outside view / reference class" scaffold is why reliance is
  near-100%: models are instructed to produce a base rate, so they always produce one —
  sourced or not.

**Verified accuracy: remembered rates are mostly right; the wrong tail is real but modest.**

- Of 22 verified clusters: **14 accurate (64%), 5 materially off (23%), 3 uncertain** (materially
  off = ≥1.5x ratio or ≥10pp).
- Restricted to the 17 pure-memory clusters, only **2-3 are materially wrong remembered numbers**
  (~12-18%): the NCAA #1-seed outcome table (C1), the Baden-Württemberg CDU-plurality blend (C4),
  and arguably the personalist-leader exit hazard (G8, uncertain). Several memories were *exactly*
  right: US CPI 75(2017)→65(2024) to the point, Africa coup rate 1.83/yr to the decimal (research),
  Australian unemployment ranges within 0.1-0.2pp, 2022 generic-ballot level within 0.1pt,
  TSMC March-wins-Q1 ~55-60% vs actual 54.5-58%.
- Of 3 research-attributed clusters, 2 were accurate to the decimal (Wisconsin margins, coup rate)
  and 1 was stale (Anthropic release cadence, G6) — research beats memory here but n=3.

**Direction: when rates were wrong, fixing them would have helped.**

- Direction of correction across the 22 clusters: 8 toward the resolved outcome, 12 neutral,
  2 away. **All 5 materially-off clusters point toward** — correcting each would have moved the
  forecast toward the resolution. (Caveat below: "toward" is a coin-flip null, so 8-vs-2 is
  suggestive, not proof.)

**Score association: base-rate wrongness is not what separates good from bad questions.**

- Mean peer score, questions whose load-bearing rates are memory-only: **+12.9** (n=10) vs
  any-sourced: **+3.8** (n=27). Spearman(memory share, peer) = **+0.19** (n=37). If anything the
  association runs *opposite* the hypothesis — but it's confounded: computed/research rates show up
  on data-rich briefing questions, and the sampling deliberately over-weighted bad outcomes.
- The verified set tells the sharper story. The three catastrophic misses in it
  (shutdown −55.9, Anthropic release −52.8, novel-flu −58.2) each involved a base-rate *failure* —
  but only one was a wrong remembered number. The other two were a stale research rate (G6) and a
  seasonality-blind extrapolation (G7). Meanwhile the strongest scores (+25 to +42) all sat on
  accurate rates.

## The real failure taxonomy

The verification pass surfaced three distinct failure modes, and "misremembered number" is the
smallest of them:

1. **Override failures — the model's own base rate was right and the published forecast ignored
   it (3 clear cases, including 2 of the sample's big misses).**
   - 43982 AfD runoff: grok-4.3 correctly recalled "3 wins in ~10 comparable cases, ~30%"; the
     ensemble published AfD 55-59%; AfD lost. The accurate base rate was in the rationale and got
     argued away by inside-view momentum reasoning.
   - 42800 TSMC: two models correctly recalled March wins Q1 ~55-60% (actual 54.5-58%); the
     ensemble's top pick was January at 56%; March won.
   - 41835 shutdown: gpt-5.2 anchored 15-25% per deadline (defensible; per-fiscal-cycle it's
     ~30%), then published 11% — below its own anchor — on a question that resolved YES.
2. **Reference-class construction/application errors (3 cases).** Wrong blending or wrong model,
   not wrong facts: 42116 BW election (blended pre-2011 CDU dominance into a 55-65% CDU-plurality
   prior when the recent era is 0-for-3 — Greens won 2016 and 2021; bot published 88% CDU,
   resolved NO); 42926 novel flu (constant-rate extrapolation from a Q1 case rate, ignoring the
   well-documented summer/fall fair-season clustering of variant swine flu; bot 65% YES, resolved
   NO); 41838 Alphabet (remembered as-reported seasonal uplift applied to a constant-currency
   question — basis mismatch, though it accidentally helped).
3. **Genuinely wrong remembered numbers (2-3 cases).** NCAA #1-overall-seed Final-Four rate
   remembered as 54-58% vs ~40-50% actual (three models gave three conflicting tables for the same
   class — a tell that all were guessing); the BW blend above; possibly the personalist exit rate.

A recurring tell: **when multiple models state the same reference class, the numbers disagree**
(NCAA exit tables; AfD conversion 30% vs 40% vs 45%; TSMC March 60% vs 55-60%). Cross-model
disagreement on a *checkable historical frequency* is exactly the signature a targeted lookup
would resolve.

## 10 most instructive examples

| # | Q | Claim (class) | Remembered | Actual (source) | Would fixing have helped? |
|---|---|---|---|---|---|
| 1 | 42116 BW CDU (peer −19.6, resolved NO on 88% YES) | CDU largest party in BW | blend → 55-65% | Recent era 0/3 — Greens largest 2016 & 2021 (Wikipedia BW Landtag results) | Yes, strongly — recency-correct class flips the forecast |
| 2 | 43982 AfD runoff (peer +1.7, resolved CDU) | AfD leads-R1 → wins runoff, east Germany | grok: 30% (right); ensemble published 57% | ~30-40% (Sonneberg, Raguhn-Jeßnitz wins vs Nordhausen, Saale-Orla, Pirna etc. losses) | Trusting it, not fetching it, was the gap |
| 3 | 42800 TSMC month (peer −7.0, resolved March) | March wins Q1 | 55-60% (right) | 54.5-58% (TSMC IR monthly revenue, 2015-2025) | Same — rate was right, ensemble picked January |
| 4 | 43131 Anthropic release (peer −52.8, resolved YES on 6.5%) | flagship release cadence | 90-120d (research-attributed, stale) | recent gaps ~42-111d, mean ~74d (anthropic.com/news) | Yes — faster cadence makes NO near-impossible |
| 5 | 41835 shutdown (peer −55.9, resolved YES on 11%) | shutdown per funding deadline, post-2010 | 15-25% | ~10% per deadline but ~30% per fiscal cycle; era exceptionally shutdown-prone | Yes — and published number sat below even the remembered anchor |
| 6 | 42438 Duke (peer −11.1, resolved Elite Eight) | #1 overall seed outcomes | F4 54-58% (claude-4.6) | ~40-50% F4 for overall #1s; 3 models, 3 conflicting tables | Yes — E8-exit bucket was underweighted |
| 7 | 42926 novel flu (peer −58.2, resolved NO on 65%) | novel-flu case arrival rate | 0.27/wk constant-rate extrapolation | variant-flu reports cluster in summer/fall fair season (CDC); spring window ≠ YTD rate | Yes — seasonality-aware rate cuts the forecast sharply |
| 8 | 41841 US CPI (peer −6.0, resolved 64) | US CPI trajectory | 75(2017)→65(2024); ±1-3/yr | Exact (transparency.org) | No — rate was perfect; miss was distribution shape |
| 9 | 41846 coups (peer +25.0, resolved NO on 12%) | Africa coups/yr 2020-2025 | 1.8-1.9/yr (research) | 1.83/yr — 11 coups/6yrs | Control: accurate sourced rate → strong score |
| 10 | 42110 Wisconsin (peer +28.2, resolved liberal) | WI SC margins | ~11pp 2023, ~10pp 2025 (research) | 11.04pp, 10.10pp | Control: accurate research rate → strong score |

## Caveats

- **Hindsight bias**: extraction was resolution-blind (rationale files withheld resolutions), but
  the batch-3 extraction was done by the orchestrator, who had already seen some resolutions —
  those load-bearing labels mostly follow the template's mechanical "my base rate was X, moving
  to Y" statements, limiting the leak. Verification necessarily saw resolutions (direction analysis
  requires them). Under the null, ~50% of corrections would score "toward"; observed 8/10 non-neutral.
- **Small n everywhere**: 22 verified clusters, 40 questions. The 23%-materially-off figure has a
  wide interval (roughly 10-45%).
- **Confounded score comparison**: memory-share vs peer-score association mixes question
  difficulty, briefing richness, and deliberate oversampling of bad outcomes. Treat +0.19 as
  "no evidence memory rates hurt", not "memory rates help".
- **Verification quality varies**: financial-vol clusters (M5, M6) were graded generously against
  approximate public data; two clusters could not be pinned (marked uncertain, excluded from the
  materially-off numerator).
- **Era note**: 31/40 questions are spring-2026 era, 9 summer; all use the modern PHASE-1
  outside-view prompt scaffold, which is why base-rate invocation is ~100%. Older eras (fall-2025)
  were not sampled; conclusions are about the current prompt regime. Per repo convention, pooled
  claims across the two sampled eras are low-risk here because the measurement (rate accuracy) is
  about model memory, not calibration of the pipeline — but the score-association numbers should
  not be projected backward.
- Comment trimming cost 3 of 40 questions their per-model rationales (only stacker meta survived)
  and stripped `Model:` attribution from 8 first sections ("unknown" model labels).

## Bottom line on the hypothesis

**Partially supported, and smaller than hoped in its original form.** Remembered base rates are
mostly decent — roughly 4 in 5 verified clusters were within tolerance, and several were exact.
The materially-wrong tail (~20% of load-bearing clusters) is real, and every wrong rate we checked
would have moved the forecast toward the resolution if corrected — so sourcing base rates is
worth positive expected peer points, spread thinly across many questions. The v2 gap-fill
BASE-RATE targets are the right mechanism for that tail: every materially-wrong rate we found
(NCAA seed tables, BW election history, release cadence, shutdown frequency, fair-season flu
timing) is trivially findable by a targeted search.

But the score damage in this sample was dominated by two failure modes that sourcing alone
doesn't fix: **models overriding their own correct base rates with inside-view narratives**
(AfD, TSMC, shutdown — including two of the worst misses), and **reference-class
construction errors** (recency-blind blending, seasonality-blind extrapolation). If the goal is
peer points, the highest-leverage complement to the v2 line is making verified outside-view
anchors *sticky* — e.g., when a rate is confirmed by research, instruct forecasters (and the
stacker) to treat deviations beyond ~2x odds from it as requiring explicit, itemized evidence —
plus a prompt nudge to check the reference class for recency breaks and seasonality before
extrapolating. Cross-model disagreement on the same named reference class (three conflicting NCAA
tables, three conflicting AfD conversion rates) is a cheap detectable trigger for exactly which
base rates to send to the gap-fill pass.
