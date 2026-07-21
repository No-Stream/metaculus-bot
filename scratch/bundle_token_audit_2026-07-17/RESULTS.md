# Research-bundle token budget audit — 2026-07-17

**Question:** how many tokens does each research source contribute to the bundle every
forecaster reads, what's redundant, and what could be cut?

**Data:** the 64 schema-2 (untrimmed) prod artifacts in
`backtests/research_archive/by_qid/` (2026-06-29 → 2026-07-16, all tournament mode), plus
the 2026-07-17 Zambia smoke bundle from `/tmp/v2-smoke.log` (copied to
`zambia_bundle.txt`). Token estimate = chars/4 (crude, ±15% for markdown prose).
Script: `measure_bundle.py` (re-runnable, free). Raw stats: `section_stats.json`.

**Important scope note:** the 64-question sample predates two flags flipped on
2026-07-17: `GAP_FILL_V2_ENABLED` (83c4f10) and `TS_ANCHOR_ENABLED` (e3d0c92). The
measured distribution is the *pre-v2, pre-anchor* bundle; the Zambia smoke shows what
v2 adds (+~1.2k tokens), and TS anchor can add up to another 1.5k tokens
(`TS_ANCHOR_SECTION_MAX_CHARS=6000`). The bundle is about to get ~2.5k tokens/question
heavier than the table below.

## 1. Per-source token counts (64 prod questions)

| Section | Present | Mean (chars) | Median | p90 | Max | Mean tokens | % of bundle |
|---|---:|---:|---:|---:|---:|---:|---:|
| AskNews briefing | 100% | 19,397 | 18,815 | 26,304 | 32,666 | **4,849** | **44%** |
| Gap-fill v1 addendum | 100% | 9,539 | 9,066 | 13,016 | 19,824 | **2,384** | **22%** |
| Native search (sol-low) | 98% | 6,630 | 6,384 | 9,196 | 10,357 | 1,631 | 15% |
| Gemini grounded | 100% | 5,585 | 5,372 | 7,536 | 9,158 | 1,396 | 13% |
| Prediction market | 100% | 2,075 | 1,697 | 3,383 | 3,843 | 518 | 5% |
| Resolution source | 16% | 4,338 | 3,447 | 6,194 | 10,870 | 169 (avg over all Qs) | 2% |
| Financial data | 6% | 509 | 509 | 532 | 532 | 7 | <1% |
| Provider diagnostics | 100% | 254 | 243 | 290 | 311 | 63 | <1% |
| **TOTAL bundle** | | **44,089** | 43,160 | 53,664 | 61,371 | **~11,022** | 100% |

Zambia smoke (2026-07-17, v1+v2 both on): **68,424 chars ≈ 17,106 tokens** — AskNews
6,251 tok, gap-fill v1 4,900 tok (an outlier at 2x the sample mean), native 2,203,
gemini 1,622, **gap-fill v2 findings 1,214**, market 670, resolution source 170,
diagnostics 73.

Base prompt overhead on top of the bundle (measured via `prompts.py` builders +
`tests/pipeline_test_helpers.py` fixtures; includes question fields, criteria, source
ladder, structured-block schema): binary **13,263 chars (~3.3k tok)**, MC 9,437
(~2.4k), numeric **15,065 (~3.8k tok)**. Each forecaster reads bundle + overhead, so a
numeric question is ~14.8k input tokens per forecaster at the mean bundle.

Trim fact-check: 15/64 bundles (23%) exceeded `REPORT_SECTION_CHAR_LIMIT=49,999`. The
trim applies only to the published comment; forecasters read the full text — consistent
with the operator's framing.

### Existing caps (constants.py)

- `RESOLUTION_SOURCE_PER_URL_MAX_CHARS=6000`, `RESOLUTION_SOURCE_TOTAL_MAX_CHARS=18000`, `RESOLUTION_SOURCE_MAX_URLS=5` — caps working; section is small.
- `TS_ANCHOR_SECTION_MAX_CHARS=6000` — self-budgeted, just enabled.
- AskNews: `n_articles=6` hot + `10` historical, then summarizer — **no output cap**; the summarizer prompt (`asknews_summarizer_prompt`) explicitly says "Be COMPREHENSIVE — a longer, thorough summary is better than a short one."
- Native search: `NATIVE_SEARCH_MAX_RESULTS=20`, `NATIVE_SEARCH_CONTEXT_SIZE="high"`, verbosity already "low" — no output cap, but observed mean 1.6k tok is reasonable.
- Gap-fill v1: `GAP_FILL_MAX_GAPS=5`, **no per-gap or per-addendum output cap** (Zambia's addendum hit 19.6k chars).
- Repo policy comment at constants.py:307: raw passthrough is capped, "LLM-emitted research is never truncated" — the two biggest sections (AskNews, gap-fill v1) are both LLM-emitted and hence uncapped by design.

## 2. Redundancy analysis

Overlap coefficient (|A∩B| / min(|A|,|B|)) on extracted fingerprints, mean over the 64
questions:

| Pair | Numbers | Entities | Source domains |
|---|---:|---:|---:|
| AskNews vs native search | **0.45** (p90 0.69) | 0.24 | 0.02 |
| AskNews vs Gemini | **0.47** (p90 0.71) | 0.16 | 0.00 |
| Native vs Gemini | 0.40 (p90 0.60) | 0.30 | 0.00 |

Reading: the three news-ish providers converge on **the same quantitative facts** about
half the time, while citing **almost entirely disjoint URLs/domains** — they're
independent retrieval paths to the same story, which is corroboration (useful) but also
means roughly 40-50% of the numeric content in native+gemini (~3k tok combined) is a
re-statement of what the AskNews briefing already carries.

Concrete examples:

- **Q44255 (H.R.6644):** vote counts 358, 85, 32, the bill number, and 10+ dates appear in both AskNews and native search sections; shared entities "French Hill", "President Trump", "The House/Senate". Overlap coefficient 0.75.
- **Q44559 (Trump southern states):** 0.80 number overlap between AskNews and native.
- **Q44563 (Trump midwest visits):** "Jon Husted", "Sherrod Brown", "Mike Rogers" named in AskNews, native, and Gemini sections alike.

Gap-fill v1 vs v2 (now both ON in prod): both are "find factual gaps in the bundle,
search, append". In the Zambia smoke they produced 4,900 + 1,214 tokens respectively,
and v2's findings (ZERN poll crosstabs, election-date confirmation) overlap v1's gaps
1-2 (post-nomination polls, official schedule). Running both is paying twice for the
same job — v1's four parallel sol web searches were ~$0.55 of the $3.34 smoke spend.

## 3. Value-vs-cost screen (rationale citations)

Scanned the `# FORECASTS` part (rationales only, not the research echo) of the 45
comments since 2026-05-01 in `scratch/coherence_2026-07-15/perf_all_tagged.json` for
source-distinctive markers:

| Source | Mentions | Questions w/ ≥1 mention | Token cost (mean) | Verdict |
|---|---:|---:|---:|---|
| Prediction market (Polymarket/Kalshi/Manifold/PredictIt) | 191 | **37/45** | 518 | **Highest value density in the bundle** — keep |
| Financial data (FRED/yfinance) | 143 | 5/45 | 7 | Cheap, heavily used when present — keep |
| Generic "the research/briefing" | 125 | 34/45 | — | AskNews is the substrate of these |
| Gemini (named) | 14 | 4/45 | 1,396 | Rarely named |
| Gap-fill | 13 | 8/45 | 2,384 | Rarely named |
| AskNews (named) | 4 | 4/45 | 4,849 | Content used via "the research", not named |
| Native search (named) | 1 | 1/45 | 1,631 | Rarely named |
| Resolution source | 0 | 0/45 | 169 | Never named, but only present on 16% of Qs and cheap |

Caveat: name-mention is a weak proxy — forecasters consume facts without attributing
the section (the 125 generic "the research" mentions are mostly AskNews-derived
content). What it does show cleanly: **the prediction-market snapshot is the one
section forecasters explicitly anchor on**, at 5% of bundle cost, and no section is
both heavy and demonstrably ignored. The cut case rests on redundancy (section 2), not
on any section being dead weight.

## 4. Input-cost model

Roster input prices (pulled live from `openrouter.ai/api/v1/models`, 2026-07-17):
gpt-5.6-sol $5/M, gpt-5.5 $5/M, claude-fable-5 $10/M, claude-opus-4.8 $5/M,
gemini-3.1-pro-preview $2/M, grok-4.5 $2/M → **Σ forecaster input rates = $29/M
tokens**. Bundle-readers beyond the six: gap-fill v1 analyzer (terra, $2.5/M — reads
the pre-gap-fill bundle, ~78% of total) and the v2 driver (**terra $2.5/M input, not
the $0.40/M in the task brief** — that figure is stale; terra is $2.5/$15, and the
driver re-reads accumulating context over ~7 steps, ≈1.6x effective with OpenAI cache
discount). Effective total ≈ **$35/M bundle tokens ≈ $0.035 per 1k bundle tokens per
question**, i.e. **$10.50/quarter per 1k bundle tokens** at 300 q/quarter.

| Scenario | Bundle tokens | Bundle-input $/question | $/quarter (300 q) |
|---|---:|---:|---:|
| Mean (measured, pre-v2) | 11,022 | $0.39 | $116 |
| p90 (measured) | 13,416 | $0.47 | $141 |
| Post v2+TS-anchor (projected) | ~14,700-17,100 | $0.51-0.60 | $154-180 |

**Honest headline:** bundle input is ~$0.39-0.60 of the $3.5-4.5/question — **~10-13%
of question cost**. The xhigh reasoning/output side dominates. Bundle cuts are worth
taking (they also plausibly shave reasoning time), but the order-of-magnitude lever on
cost remains forecaster effort tiering, as the smoke-cost doc already noted.

## 5. Cut list (ranked; measurement only, nothing implemented)

1. **Retire gap-fill v1 once v2 is validated (~$190/quarter).** Both are ON as of
   today and do the same job. Savings: v1's own research spend ~$0.55/q (four parallel
   sol web searches + terra analyzer ≈ $165/qtr) plus 2,384 bundle tokens/q
   (~$25/qtr input). Risk: v2 must demonstrably cover v1's gap-resolution quality —
   needs a few side-by-side prod runs first (the Zambia smoke is one datapoint where
   v2 found the same core gaps in 66s vs v1's 4-search fan-out). Interim option: cap
   the v1 addendum (it has no cap; Zambia hit 19.6k chars).

2. **Cap the AskNews briefing at ~10k chars (~$60/quarter).** It's 44% of the bundle
   because the summarizer prompt *asks* for maximal length ("longer... is better").
   A "prioritize; ~2,500 words max, drop repetition across articles" instruction saves
   ~2,350 bundle tokens/q (~$25/qtr input across 7 readers) **plus** halves the
   summarizer's own output bill (~8k tok out at $30/M ≈ $0.24/q → ~$36/qtr). Risk:
   this is the highest-content section — cut by prioritization, not truncation, and
   watch a before/after on a few live questions. The p90→max range (26k→33k chars)
   suggests the tail is padding, not signal.

3. **Bundle-hygiene pair (~$20/quarter, near-zero risk).**
   (a) Drop the Provider Diagnostics block from the forecaster-facing bundle (keep it
   in the comment/archive — it's operator telemetry; 63 tok/q).
   (b) Given 40-47% numeric redundancy, tell the *summarizer* or native-search prompt
   to skip restating market/statistical figures verbatim when they'll arrive from
   other sections — or accept the corroboration value and skip this. Dropping Gemini
   outright would save 1,396 tok/q (~$14/qtr input; the search itself is free-tier)
   but sacrifices the only first-party Google index — not recommended.

Not recommended for cutting: prediction market (highest citation rate per token),
resolution source (rare, capped, cheap), financial data (near-free), native-search
verbosity (already "low").

Adjacent flag (outside the bundle, same 6-7x multiplier): numeric base-prompt overhead
is 15.1k chars — ~3.8k tokens × 6 forecasters ≈ $0.11/q ($33/qtr). A prompt-tightening
pass there is comparable in value to cut #2's input side.
