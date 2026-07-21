# Research-bundle CONTENT audit — 2026-07-17

**Question:** the token audit (`scratch/bundle_token_audit_2026-07-17/RESULTS.md`) found
40-47% numeric-fact redundancy across the news-ish sections and proposed cuts. The
operator's condition for acting: a content-level audit on real questions — per section,
what portion is (a) uniquely load-bearing, (b) corroborative duplication (independent
confirmation, has value), (c) pure repetition/padding (safe to cut) — judged carefully,
with receipts.

**Data:** 10 judging packets built from the same 64 schema-2 prod artifacts the token
audit measured (`backtests/research_archive/by_qid/`, 2026-06-29 → 2026-07-16), each
pairing the full bundle (split per section) with the question's live resolution criteria
+ fine print (Metaculus API, cached in `meta_cache/`). Sample spans 4 binary / 3 MC-
discrete / 3 numeric questions, includes 3 of the >p90 fat bundles (44773, 44558, 44555)
and both token-audit redundancy exemplars (44255, 44563). Builder: `build_packets.py`.

**Judges:** three parallel Claude Fable 5 subagents (`model="fable"`), 3-4 packets each,
shared instructions (`JUDGING_INSTRUCTIONS.md`), independent write-ups in `judgments/`.
Deterministic corpus-wide screens over all 64 bundles (`boilerplate_scan.py` + inline
analyses): boilerplate density, verbatim 12-gram duplication, numeric-novelty shares,
URL overhead, prediction-market topical-match rate.

**Everything here was free:** archived artifacts, the read-only Metaculus API, and
subagent judging. No research-provider or forecast LLM calls.

## 1. Deterministic corpus-wide screens (all 64 bundles)

These sharpen the token audit's fingerprint-overlap numbers before any judgment call:

- **The 40-47% "redundancy" is fact-level, NOT copy-paste.** Verbatim 12-gram overlap
  between AskNews/native/Gemini is ~0.000 (max 0.007). The providers restate the same
  facts in fully independent prose. Any dedup must therefore be semantic (prompt-side
  prioritization), not mechanical (string dedup won't find anything).
- **Numeric novelty per secondary section:** ~37-40% of native-search's and Gemini's
  distinct numbers appear nowhere else in the bundle (median 38%/41%). vs. AskNews
  alone they're ~55% novel. So the secondary searchers are roughly ⅓ unique, ⅔ overlap
  at the fact level — consistent with the token audit, but the unique third is real.
- **Gap-fill v1 numeric novelty: mean 37%** (range 0-77%). On some questions it's pure
  re-search (44216: 0% new numbers) and on others it's the bundle's main value add
  (44409: 76%, 44221: 77%).
- **AskNews boilerplate apparatus = 6.0% of section chars** (mean 6.4 `Credibility:` +
  8.1 `Source:` tagged lines per question). Native/Gemini/gap-fill ≤0.6%.
- **AskNews head-loading:** the first half of the briefing already contains 77% of the
  section's distinct numbers (first quarter: 44%) — the tail is mostly elaboration,
  matching the token audit's "p90→max tail is padding" hunch.
- **URL overhead:** raw URLs are **31.6% of Gemini section chars** (p90 51%; nearly all
  ~250-char `vertexaisearch.cloud.google.com/grounding-api-redirect/...` blobs that
  forecasters cannot resolve) and **22.4% of native-search chars**. AskNews: ~0%.
- **Prediction-market topical match rate: on 36/64 questions (56%), NO listed market
  shares ≥2 content words with the question.** The fuzzy matcher fills the table with
  keyword-adjacent but resolution-irrelevant markets (SOFR markets for a jobs question,
  Dept-of-Education markets for a travel-advisory count). The `conf` column only weakly
  separates: a 0.55 threshold keeps 47% of on-topic rows while still passing 15% of
  off-topic ones.
- **Chatbot-offer leakage:** "If you want, I can..." style assistant offers appear in
  **32/64 AskNews** sections and **30/58 gap-fill** sections — the summarizer/searcher
  prompts leak conversational framing into a document six forecasters read as evidence.
- **Freshness is fine:** 48% of AskNews-cited dates are within 30 days of the run;
  <2% are older than 180 days. Staleness is not the padding mechanism — topical drift is.

## 2. Judged taxonomy (10 questions, 3 judges)

Full per-question tables, verbatim padding exhibits, cut-first verdicts, and
halving tests live in `judgments/judge_B.md` (q44453/44558/44563),
`judgments/judge_C_opus.md` (q44255/44512/44551), and `raw_judgments.md`
(q44219/44225/44555/44773 — the recovery run after the Fable judges died on
inference flakiness). All three judges used the same rubric and independently
converged on the same section ranking.

### Aggregate content-share by section (mean % across all 10 questions)

| Section | Present | a: unique | b: corrob | c: padding | Verdict |
|---|---:|---:|---:|---:|---|
| **gap_fill_v1** | 10/10 | **59** | 25 | 16 | Highest unique share — the workhorse. Protect. |
| **native_search** | 10/10 | **54** | 32 | 15 | Best value-per-token. Protect. |
| financial_data | 2/10 | 48 | 48 | 5 | Tiny, primary-source anchor when present. Protect. |
| gemini_search | 10/10 | 38 | 39 | 24 | Real net-new facts; ¼ mechanical URL/marker overhead. |
| resolution_source | 1/10 | 25 | 25 | 50 | Rare; when it's the generic site text not the graded value, half is padding. |
| **asknews** | 10/10 | **16** | 27 | **57** | Largest section (44% of bundle), most padding. Prime trim. |
| **prediction_market** | 10/10 | **6** | 6 | **88** | Near-total padding via fuzzy-match failure. Gate. |
| diagnostics | 10/10 | 0 | 0 | 100 | Operator telemetry, not forecaster content. Drop from bundle. |

Sorted by unique share. The ordering is stable across all three judges: gap_fill
and native_search top every judge's table, asknews and prediction_market bottom
every one. This **inverts the token-cost ranking** — the two heaviest sections by
tokens (asknews 44%, gap_fill 22%) sit at opposite ends of the value ranking, so
"cut the biggest" and "cut the least valuable" point in different directions.

### Findings, mapped to the five token-audit hypotheses

1. **AskNews tail restates its head — CONFIRMED, and it's worse than that.**
   Every judge saw asknews's closing "Key facts / Bottom line / Contradictions"
   subsections re-list content already stated above (q44453 literally re-lists its
   own numbers; q44219 flags a deliberate repeat of the April datapoint). But the
   larger waste is **topical drift**: content that shares the question's keywords
   but not its resolution criteria (off-topic Australian advisories in q44558,
   arXiv finances + a crypto-mining incident in q44225, an obsolete Platner
   chronology in q44555, ~15 stale intraday oil quotes in q44773). This matches the
   deterministic screen (§1: first half of the briefing holds 77% of distinct
   numbers; topical drift, not staleness, is the padding mechanism).

2. **Native/Gemini merely restate AskNews headline numbers — REFUTED as a cut
   rationale.** The headline numbers ARE heavily duplicated (§1: ~⅔ fact-overlap),
   but native/gemini's unique third is real and repeatedly decision-critical. In
   **five separate questions the smaller live-search sections carried the one fact
   AskNews missed or got backwards**: the current-July Anthropic leaderboard lead
   (q44219, asknews leaned OpenAI off stale April data), the July-10 auto-enactment
   math (q44255, asknews concluded "Yes" wrongly), the correct Australia total-gold
   series (q44512, asknews mislabeled swimming-only golds), Death Valley as the hot
   station (q44551), and the replacement-candidate polls (q44555). **AskNews length
   correlates with worse conclusions, not better ones** — the single most important
   cross-judge finding.

3. **gap_fill re-searches things the bundle already answers — REFUTED.** gap_fill
   carried the decisive single-source fact in the large majority of questions: the
   official AA leaderboard + score reconciliation (q44219), the exact-phrase debunk
   of Gemini's inflated count (q44225), the actual RaceToTheWH Maine averages — the
   literal resolution source (q44555), the BZ=F→October-contract rollover mechanic
   (q44773), the August base-rate + official date table (q44558), the 2018 travel
   base rate + venue check (q44563), live payroll market prices + print history
   (q44453), rival-field strength (q44512), the Ogimet/Death-Valley mechanic
   (q44551), and official-status verification (q44255). Its only waste is
   "If you want, I can..." offers and honest-null hedging (~14-16%). The §1 "37%
   mean numeric novelty, range 0-77%" is real variance, but on THESE 10 questions
   it was the most load-bearing section per token.

4. **Prediction market is high-value-per-token — REFUTED in 9/10, with one telling
   exception.** The fuzzy matcher returned zero on-topic markets in 9 of 10
   questions (pizza/HLE for an AI-leaderboard question; MLB/rapper-streams for Brent
   crude; SOFR/mortgage rates for a jobs number; Senate-control for a Maine
   polling-lead), each under a "STRONG EVIDENCE — weight these markets heavily"
   header. This matches the §1 screen: 56% of questions have NO market sharing ≥2
   content words. The lone exception (**q44255**: SAVE Act markets at 0.07-0.08
   priced Trump's signing pre-condition) is genuinely high-value — which is exactly
   why the fix is match-quality **gating**, not deletion. The prior audit's
   "most-cited section" finding (§3 of the token audit) reflects the questions where
   it DID match; the header makes it actively harmful when it doesn't.

5. **The Source/Credibility apparatus is load-bearing — MIXED, mostly padding.**
   The deterministic screen puts it at 6.0% of asknews chars. It's mildly useful on
   procedural/contested questions (q44255: trusting E&E News over BeInCrypto's
   unverified CBDC claim), pure padding on quantitative ones (a gold count or oil
   price doesn't hinge on outlet authority). The standalone "Source-by-source
   credibility notes" *subsection* is always padding — it duplicates the inline
   tags. The `[PRE-WINDOW]/[SINGLE-SOURCE]` bracket pair repeated 20-40× per
   asknews section (worst in q44555, q44773) is load-bearing once, overhead the
   rest.

### Cross-judge cut-first tally

Cut-first verdict (drop ONE whole section): **prediction_market in 9 of 10
questions** (all three judges), the sole exception being judge C's q44255 where
prediction_market was the keeper and gemini_search the cut (its unique day-count
math was fully corroborated by native + gap_fill). No judge ever nominated
gap_fill, native_search, or financial_data for a cut-first drop.

### Risk notes

- **Corroboration is real and mechanical dedup won't find it (§1).** Cross-section
  verbatim 12-gram overlap is ~0.000 — the providers restate shared facts in fully
  independent prose. Any dedup must be semantic (prompt-side prioritization), never
  string-matching. The 40-47% fact overlap is genuine independent-retrieval
  corroboration on the *easy* facts; sections diverge on the *decisive* fact (q44255
  is the clean example — 0.75 number-overlap, but only 3 sections did the deadline
  math).
- **AskNews is not safe to delete wholesale even though it's mostly padding**, because
  on several questions its trimmed core (2020 precedent in q44555, sector context in
  q44453) is only lightly duplicated. The move is prioritized halving, not a whole
  cut — and a before/after eyeball on live questions, since it's the highest-content
  section.
- **The single largest divergence between this audit and the token audit:** the token
  audit's #1 cut is "retire gap_fill v1 (~$190/qtr)"; this content audit ranks
  gap_fill the MOST load-bearing section per token. They are reconcilable only if
  gap-fill v2 demonstrably reproduces v1's gap-resolution quality — that
  side-by-side validation is a hard prerequisite, not a formality.

## 3. Recommendations

Ranked by content-backed safety, with $/quarter cross-referenced from the sibling
token audit (`../bundle_token_audit_2026-07-17/RESULTS.md`: **$10.50/quarter per 1k
bundle tokens** input across the ~7 readers at 300 q/quarter; AskNews additionally
carries its own summarizer output bill). All measurement only — nothing implemented.

1. **Halve AskNews by prioritization (~$60/quarter, moderate risk — the biggest
   content-backed lever).** It's 44% of the bundle, ~57% padding by content share,
   and its length tracks *worse* conclusions (q44219/44255/44512). Instruct the
   summarizer to prioritize resolution-relevant facts and drop cross-article
   repetition and topical-drift content (the summarizer prompt currently *asks* for
   maximal length). Saves ~2,350 bundle tokens/q (~$25/qtr input across readers)
   **plus** roughly halves the summarizer's own ~8k-token output bill (~$36/qtr).
   Risk: highest-content section — cut by prioritization, not blind truncation, and
   watch a before/after on a few live questions. Ties to the token audit's cut #2.

2. **Gate prediction_market on fuzzy-match relevance (~$3-5/quarter, near-zero
   risk, removes an actively-harmful instruction).** On the ~56% of questions where
   no market shares ≥2 content words (§1), suppress the section AND its "STRONG
   EVIDENCE — weight these markets heavily" header, which currently points
   forecasters at pizza/MLB/EU-referendum markets. Token savings are small (518
   tok/q, and only on non-matching questions), but the value is correctness: the
   header is misleading exactly when the match fails. Keep the section verbatim when
   it matches (q44255 SAVE Act shows its real value). Implement via the existing
   `conf` column threshold, tightened — §1 notes 0.55 still passes 15% of off-topic
   rows, so gate higher or add a content-word-overlap check.

3. **Bundle-hygiene pack (~$6-8/quarter, near-zero risk, purely mechanical).**
   (a) Drop the Provider Diagnostics block from the forecaster-facing bundle (keep
   it in the comment/archive — it's operator telemetry; 63 tok/q, 0% forecaster
   value in all 10). (b) Strip the ~250-char `vertexaisearch.cloud.google.com`
   grounding-redirect URLs from gemini_search — they're 31.6% of gemini chars (§1),
   ~441 tok/q, and a forecaster cannot resolve them; also strip the garbled mid-word
   citation markers ("O[1]n", "G[1]emini"). (c) Collapse the 20-40× repeated
   `[PRE-WINDOW]/[SINGLE-SOURCE]` apparatus and the standalone source-credibility
   subsection in asknews to a single statement of the rule. None of these touch a
   single load-bearing fact.

**Do NOT cut on content grounds:** gap_fill_v1 (most load-bearing per token — 59%
unique, decisive single-source fact in the large majority of questions),
native_search (best value-per-token — 54% unique), financial_data (the
primary-source resolution anchor when present, 518 chars). The token audit's
proposed gap_fill-v1 retirement (~$190/qtr) is the one cut this content audit
actively cautions against: it is a token-cost decision that would remove the
highest-value content, defensible ONLY after v2 is proven to reproduce v1's
gap-resolution quality on side-by-side live runs.

**Order-of-magnitude honesty (carried from the token audit):** the whole bundle is
only ~$0.39-0.60 of the $3.5-4.5/question (~10-13% of cost); the reasoning/output
side dominates. These cuts are worth taking — they also plausibly shave reasoning
time and remove a misleading market header — but bundle trimming is not the primary
cost lever.
