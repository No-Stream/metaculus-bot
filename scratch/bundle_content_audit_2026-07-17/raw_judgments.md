# Raw judgments — judge A (recovery run, Opus)

Packets assigned to this run (the 4 not covered by judge_B / judge_C_opus):
q44219, q44225, q44555, q44773. The other 6 (q44255, q44453, q44512, q44551,
q44558, q44563) are already judged in `judgments/judge_B.md` + `judgments/judge_C_opus.md`
and are pulled into the synthesis from there.

Method: percentages are content-share estimates (nearest 5-10%), judged against
each question's own resolution criteria (in the packet header), comparing sections
directly against each other. Crash-safety: each packet's block is appended here the
moment it's judged.

---

## Q44219 — Highest-scoring lab on AA Intelligence Index, Aug 31 2026 (multiple_choice, 44059 chars)

| Section | chars | a_unique% | b_corrob% | c_padding% | one-line note |
|---|---:|---:|---:|---:|---|
| asknews | 18121 | 15 | 25 | 60 | Largest section, and **STALE + directionally WRONG**: builds its whole "OpenAI most likely" lean on April GPT-5.5=60 data, never sees the current July-1 Anthropic lead the two smaller live-search sections both carry. Heavy off-topic padding (Microsoft Agent survey, GeneBench, tokens/sec). |
| native_search | 8754 | 55 | 30 | 15 | MVP: only section with the CURRENT leaderboard mirror (Fable 5 / Anthropic top ~59.9), the v4.1 nine-eval composition, the full Creator roster (governs "Other"), Fable-5 export-control suspension→Jul-1 global return, and the Manifold outside view. |
| gemini_search | 4253 | 35 | 40 | 25 | Corroborates the Anthropic lead (Opus 4.8 61.4 / Fable 5 ~60) and uniquely adds the Jul-Aug release calendar (GPT-5.6 Sol Jun 26, o3 retirement Aug 26, GPT-6/Gemini-4 rumors). ~⅓ is the vertexai redirect-URL block + garbled inline markers. |
| prediction_market | 1215 | 5 | 5 | 90 | Total fuzzy-match failure: Dave-Portnoy-pizza, HLE-≥60%-before-2027, Meta-#1-AI-before-2027. None resolve on which lab tops the AA Index on Aug 31; "STRONG EVIDENCE weight heavily" header is actively misleading. |
| gap_fill_v1 | 11473 | 60 | 25 | 15 | Other MVP: Gap 1 = the **official AA leaderboard** (Fable 5 60 / Opus 4.8 56 / GPT-5.5 55, creators mapped); Gap 2 reconciles the conflicting scores (v4.0-vs-v4.1 methodology, "with fallback" variants); Gap 3 = indexing status (GPT-5.6 Sol NOT yet indexed, no ETA). |
| diagnostics | 243 | 0 | 0 | 100 | Provider telemetry; operational, not forecaster content. |

**Padding exhibits (category c, verbatim quotes):**
1. [asknews] "The **2026 Agent Confidence Index** surveyed **300 technical experts** ... Average confidence score was **64 out of 100** ... automated report generation: **83.5**" — an entire Microsoft-survey block about agent task confidence; nothing to do with which lab tops the AA Intelligence Index.
2. [asknews] "**Mercury 2** delivered **727.2 tokens per second** ... **Llama 4 Scout** supported a **10 million token** context window ... **Gemma 3n E4B** cost **$0.03 per million tokens**" — speed/cost/context specs that don't bear on the intelligence-index ranking; keyword-adjacent AI-landscape filler.
3. [asknews] "**Key facts repeated because they are central:** - **GPT-5.5**: **60 points** - **Claude Opus 4.7**: **57 points** - **Gemini 3.1 Pro Preview**: **57 points**" — the model literally re-listing the April datapoint it already stated 200 lines earlier, flagged as a deliberate repeat.
4. [prediction_market] "kalshi | Dave Portnoy: Score any pizza place a 9.0 or higher in 2026? | 0.42" under the "STRONG EVIDENCE -- weight these markets heavily" header — a pizza-review market presented as heavily-weightable evidence for an AI-leaderboard question.
5. [gemini_search] "[3] metaculus.com — https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHOiFe3Hb7EVtXO-rxXeZ3YuyPosDkaNWIxSPWgEwd-..." — a ~250-char opaque Google grounding-redirect URL a forecaster cannot resolve; four of these plus mid-word markers ("O[1]n", "G[1]emini") are the section's overhead.

**Unique-value callouts:**
- **native_search and gap_fill are the ONLY sections showing the CURRENT (July 1) leader is Anthropic** (Fable 5 ~60 on the live mirror / official page). AskNews's 18k-char analysis leans OpenAI off stale April-25 data (GPT-5.5=60) and never updates — the single most decision-relevant fact (who leads *now*) is carried only by the two live-search sections, and the biggest section gets it backwards.
- **gap_fill Gap 2 is the only place that reconciles the conflicting scores** — v4.0 (comparison page) vs v4.1 (methodology page), launch-article vs current-page divergence, "with fallback" variant scoring. This directly explains why asknews's 60/57/61.4 numbers disagree, and the fine print resolves on the *displayed* value regardless of methodology.
- **gap_fill Gap 3's indexing status is single-source and decisive:** GPT-5.6 Sol is officially previewed but NOT yet on the AA index with no public ETA; Mythos 5 isn't separately indexed. An unindexed model can't be the displayed leader on Aug 31 — this caps OpenAI's near-term upside in a way no other section notes.
- **native_search uniquely carries the Creator roster** (Anthropic, OpenAI, Z AI, Google, Alibaba, ... Amazon) that governs which outcomes map to "Other," plus the Manifold "best LLM end-of-2026" market (Anthropic 41 / OpenAI 23 / Google 19) — the only genuine outside-view anchor, which the dedicated prediction_market section entirely missed.

**Cut-first verdict:** prediction_market. Its three markets (pizza scores, Humanity's Last Exam ≥60%, Meta-#1-before-2027) are keyword-adjacent AI/scoring matches that resolve on nothing this question asks, and the "weight these markets heavily" header instructs forecasters to anchor on them. Dropping it loses zero forecast-relevant information; the one real outside-view market (Manifold lab race) already lives in native_search. (asknews is the far bigger token sink and is directionally misleading, but it does corroborate the frontier-is-tight framing, so it's a heavy-trim target rather than a clean whole-section drop.)

**Halving test (two largest — asknews 18121, gap_fill 11473):**
- asknews: halving loses essentially nothing load-bearing. A prioritizer would drop the Microsoft Agent-Confidence block, the GeneBench/Claude-Science block, the open-weight tokens/sec + context-window + cost metrics, the deliberately-repeated GPT-5.5 datapoint, and the source-by-source credibility subsection — retaining the April GPT-5.5=60 anchor and the "top tier tightly clustered" framing, both corroborated in the leaner sections anyway. Safe to cut 60%+; its stale OpenAI lean should be discounted regardless of length.
- gap_fill: halving is risky. Gaps 1-3 (official leaderboard snapshot, score reconciliation, indexing status) are the single-source load-bearing core. The compressible ~15-20% is the four "If you want, I can..." offers and Gap 5's honest null on the MC option set. A naive halve dropping Gap 1 or Gap 3 would delete the authoritative resolution-source snapshot and the "GPT-5.6 not yet indexed" cap.

---

## Q44225 — arXiv "agentic reinforcement learning" abstract count, Jul-Aug 2026 (numeric, 30876 chars)

| Section | chars | a_unique% | b_corrob% | c_padding% | one-line note |
|---|---:|---:|---:|---:|---|
| asknews | 15959 | 5 | 15 | 80 | 52% of the bundle and it OPENS by admitting it has "no direct count" — then fills 16k chars with tangential trend articles (ICML submission totals, arXiv-spinout finances, a crypto-mining agent incident, Xiaomi HarnessX benchmarks). Almost nothing bears on the exact-phrase count. |
| native_search | 4518 | 65 | 20 | 15 | MVP: the only section that draws the decisive exact-phrase-vs-variant distinction, finds the single verified July-1 paper, gives the arXiv monthly-volume denominator (27,500/mo), and specifies the settlement-query methodology. |
| gemini_search | 3805 | 45 | 25 | 30 | The only section with a numeric base rate + projection (May ~10-15, June ~8-12, "30-60 papers/month" central) and the conference-deadline calendar (ACL/EMNLP Jul 15/ICML) that argues for a mid-July surge — but it counts VARIANTS, so its numbers are inflated for the exact-phrase query. |
| prediction_market | 1250 | 0 | 0 | 100 | Total fuzzy-match failure: Trump spending cuts, nuclear data center, EU-referendum markets. Zero relevance; "STRONG EVIDENCE weight heavily" header misleading. |
| gap_fill_v1 | 5101 | 55 | 30 | 15 | Sharpens the crux: verifies exact-phrase abstracts exist Mar-Jun 2026 (one/month named) and DEBUNKS Gemini's early-July count — the July-1 papers use "agentic online RL"/"agentic RL", NOT the contiguous phrase. Honest nulls on the exact monthly counts. |
| diagnostics | 243 | 0 | 0 | 100 | Provider telemetry; operational, not forecaster content. |

**Padding exhibits (category c, verbatim quotes):**
1. [asknews] "In fiscal year 2025, arXiv had a **$297,000 deficit** against **$6.7 million in expenses** ... The permanent CEO salary is approximately **$300,000**." — arXiv's operating budget and executive pay; irrelevant to how many abstracts contain a phrase.
2. [asknews] "The system autonomously established a reverse SSH tunnel and used GPU capacity to mine cryptocurrency ... **over one million trajectory training steps**." — a colorful agent-defection incident that shares the keyword "agentic" but says nothing about phrase frequency in abstracts.
3. [asknews] "**Qwen 3.5-9B** improved by **44 percentage points** on ALFWorld embodied planning tasks, from **53.0% to 97.0%**. ... The GitHub repository had **112 stars** as of **July 2, 2026**." — HarnessX benchmark minutiae and a GitHub star count; pure topical-adjacency padding.
4. [asknews] "If you want, I can also produce: 1. a **best-guess quantitative forecast range** for the count, or 2. a **search strategy**..." — a conversational assistant offer leaked into a document six forecasters read as evidence.
5. [prediction_market] "kalshi | What countries will hold referenda on leaving the EU? | 0.02" under "STRONG EVIDENCE -- weight these markets heavily" — an EU-referendum market flagged as heavily-weightable evidence for an arXiv-paper-count question.

**Unique-value callouts:**
- **The exact-phrase-vs-variant distinction is the whole ballgame, and only native_search + gap_fill get it.** The resolution query is the literal contiguous phrase "agentic reinforcement learning" in abstracts. native_search flags the ambiguity explicitly; gap_fill Gap 3 verifies that the July-1 papers Gemini counted actually use "agentic online RL"/"agentic RL" NOT the exact phrase. This single methodological point separates a plausible ~30-60 count from a much smaller true count — and asknews, the biggest section, never raises it.
- **gemini_search is the only section with a usable numeric base rate/projection** (May ~10-15, June ~8-12, ~30-60/month central, EMNLP-deadline surge). For a numeric question this is the primary distribution anchor — but it must be discounted because it conflates variants (gap_fill's correction directly bears on this).
- **gap_fill uniquely establishes the monthly exact-phrase base rate exists** (named Mar/Apr/May/Jun 2026 papers with the phrase in-abstract), giving a floor of at least ~1 verified exact-phrase paper/month pre-window — the empirical anchor asknews said the research "does not provide."
- **native_search uniquely supplies the denominator + settlement methodology** (arXiv ~27,500 submissions/month; API `submittedDate` filtering; the exact advanced-search reconstruction) — the scale context that bounds a plausible count.

**Cut-first verdict:** prediction_market. Its three markets (government spending cuts, nuclear data center, EU referenda) share nothing with an arXiv-abstract-count question — a pure fuzzy-match failure with a misleading "weight heavily" header. Zero forecast-relevant loss. (asknews is by far the larger token sink and is ~80% padding here, but it does carry the weak corroboration that agentic/RL terminology is trending — so on a strict "one whole section" cut, prediction_market is the zero-loss pick; asknews is the biggest *trim* target and arguably a near-zero-loss whole cut too, since its trend framing is corroborated in the leaner sections.)

**Halving test (two largest — asknews 15959, native_search 4518):**
- asknews: halving loses essentially nothing. A prioritizer could delete the arXiv-finances article, the crypto-mining incident, the HarnessX/Xiaomi benchmarks, the semiconductor-fab RL paper, and the source-credibility subsection — keeping only "agentic + RL are trending; arXiv volume ~27,500/mo; no exact-phrase count found," which is ~1k chars. This section is closer to 80%-cuttable than 50%.
- native_search: halving is risky. Every subsection is load-bearing for a numeric question — the exact-phrase distinction, the one verified July paper, the volume denominator, and the settlement methodology. The only compressible piece is the repeated "August hasn't happened yet" caveat; a naive halve dropping the variant-vs-exact-phrase discussion would delete the bundle's single most important counting caveat.

---

## Q44555 — Collins leads Dem in RaceToTheWH Maine avg on Aug 31 2026 (binary, 60357 chars)

| Section | chars | a_unique% | b_corrob% | c_padding% | one-line note |
|---|---:|---:|---:|---:|---|
| asknews | 32666 | 10 | 20 | 70 | The sample's LARGEST section, and ~mostly OBSOLETE: an exhaustive Collins-vs-Platner polling chronology + Platner-scandal dossier, but Platner WITHDREW Jul 10 — the question resolves on Collins vs the *replacement* nominee. Buried under ~12× repeated [PRE-WINDOW]/[SINGLE-SOURCE] tags. |
| native_search | 7466 | 45 | 35 | 20 | High value: the replacement-candidate polls (PPP: Jackson 49-44, Bellows 47-45, Shah 47-45; Z to A tie), the 2020 Collins-beats-polls-by-8.6pt warning, and the Siena Trump-36%/fed-funding-61% internals. |
| gemini_search | 5669 | 40 | 40 | 20 | Corroborates the replacement polls (Z to A Jul 9 Shah/Bellows/Jackson) and uniquely adds Collins's "second most unpopular senator" approval framing + the "late-Aug omnibus poll anticipated but unscheduled" timing note. ~20% vertexai URL block. |
| prediction_market | 3755 | 5 | 5 | 90 | Fuzzy-match failure: Senate/House control 2026, Trump endorsement, 2028 VP nominee, FL/MI/NY Dem nominations. NONE resolve on whether Collins leads the Aug-31 *polling average*; several are 2028/2029 markets. Misleading "weight heavily" header. |
| resolution_source | 3447 | 25 | 25 | 50 | The named RaceToTheWH page — but the fetched text is the site's generic 2026-Senate-forecast narrative (53-47 map, flip-four-seats math, Apr-17 fundraising note), NOT the Maine polling average the question reads. Confirms the 53-47 arithmetic AskNews wrongly flagged as "inconsistent." |
| gap_fill_v1 | 7064 | 70 | 20 | 10 | The MVP: Gap 3 has the ACTUAL RaceToTheWH Maine averages for each replacement matchup (Collins +1.2 vs Bellows, Jackson +2.2, Collins +0.7 vs Shah) + the weighting methodology + daily-update cadence — the single most resolution-relevant content in the entire bundle. |
| diagnostics | 290 | 0 | 0 | 100 | Provider telemetry; operational, not forecaster content. |

**Padding exhibits (category c, verbatim quotes):**
1. [asknews] "**[PRE-WINDOW — occurred before question open, cannot itself satisfy the criteria] [SINGLE-SOURCE]**" — this exact bracket pair is prepended to roughly every one of ~40 bullets; the rule is load-bearing once, the ~40-fold repetition is pure formatting overhead that inflates the section by thousands of chars.
2. [asknews] "A tattoo resembling the Nazi Totenkopf symbol. Past Reddit posts containing controversial remarks. ... Platner was accused of raping a woman in **2021**" — the entire Platner-controversy dossier; since Platner withdrew and the question resolves on his replacement, this cannot bear on the Aug-31 average except as weak "why he withdrew" context.
3. [asknews] "The author accused Democratic neighbors of seeking a **\"Marxist, socialist, communist government.\"**" — content from a Portland Press Herald *reader letter*, which asknews itself grades "individual advocacy letter, not polling or expert analysis"; zero forecast value.
4. [prediction_market] "kalshi | 2028 Democratic VP nominee | 0.06" and "predictit | ...2026 Florida Democratic Senate nomination? — Alexander Vindman | 0.98" under "STRONG EVIDENCE -- weight these markets heavily" — a 2028 VP market and a Florida-nomination market flagged as heavily-weightable for a Maine Aug-31 polling-lead question.
5. [asknews] "The supplied summaries give two different margins of error for the same Emerson poll: **+/- 2.9 percent** and **±3.1%**." — one of ten enumerated "contradictions" the model surfaces about obsolete Platner-era polls; meta-narration about source discrepancies in data that no longer applies.

**Unique-value callouts:**
- **gap_fill Gap 3 is the only section carrying the actual resolution-source values.** RaceToTheWH's displayed Maine averages for the live replacement matchups — Collins +1.2 vs Bellows, Jackson +2.2, Collins +0.7 vs Shah — are the exact thing the question resolves on, and only gap_fill has them (via the Wikipedia mirror of the RaceToTheWH set). This is decisive and single-source.
- **The Platner→replacement reset is the pivotal fact, and asknews (the 32k-char section) treats it as a footnote** while native_search, gemini, and gap_fill all foreground it. The replacement-candidate polling (native: PPP Jackson 49-44 / Bellows 47-45 / Shah 47-45; gemini: Z to A Jul 9 ties) is what matters, and asknews's massive pre-withdrawal chronology is ~obsolete.
- **native_search uniquely supplies the 2020 polling-miss magnitude** (Collins won by ~8.6pt despite trailing Gideon in public polls) — the strongest single argument that a small Democratic polling lead may not hold, and the exact quantification asknews explicitly said it lacked ("does not quantify the size of her 2020 polling miss").
- **resolution_source uniquely settles the 53-47 arithmetic** that asknews flagged as an internal contradiction: the RaceToTheWH page confirms Republicans hold 53-47 and Democrats need to flip four seats for a 51-49 majority — so asknews's "facially inconsistent" flag was itself the error. A nice cross-section correction, though only weakly relevant to a polling-lead question.
- **gap_fill Gap 3 also uniquely notes RaceToTheWH shows SEPARATE per-nominee averages, not one generic Collins-vs-Dem line** — which means the resolving value depends on WHO the nominee is on Aug 31, a structural subtlety no other section catches.

**Cut-first verdict:** prediction_market. Its nine markets resolve on Senate/House control, a Trump endorsement, and 2028/2029 nominations — none on whether Collins leads the RaceToTheWH *polling average* on Aug 31, and several close years later. The one weakly-relevant signal (Polymarket ~63% Dem-win) is already carried, with the correct "this is election-winner not polling-lead" caveat, by both native_search and gemini. Zero forecast-relevant loss, and it removes a "weight heavily" header pointed at 2028 VP markets. (asknews is by far the bigger token sink — 54% of the bundle and ~70% padding here — but a whole-section cut would lose the 2020-precedent and Trump-approval context, which is only lightly duplicated; it's the dominant *trim* target.)

**Halving test (two largest — asknews 32666, native_search 7466):**
- asknews: halving loses almost nothing decision-relevant. A prioritizer would delete the blow-by-blow pre-withdrawal poll chronology (every Platner topline is obsolete), the full scandal dossier, the reader-letter and Townhall/Zero-Hedge partisan framing, the ten-item "contradictions" enumeration, and the ~40× PRE-WINDOW/SINGLE-SOURCE tag repetition — keeping only the 2020 precedent, the Trump-approval environment, and the replacement-process timeline (Jul 25 convention / Jul 27 deadline). Safe to cut 70%+.
- native_search: halving is riskier. The replacement-candidate polls, the 8.6pt 2020 miss, and the Siena internals are all load-bearing and lightly-duplicated; the compressible ~20% is the repeated "no reliable Aug-31 market found" caveat and the FEC/National-Archives boilerplate. A naive halve dropping the replacement polls or the 2020 miss would gut the section's unique contribution.

---

## Q44773 — Brent crude close on Aug 28 2026 (numeric, 61371 chars)

| Section | chars | a_unique% | b_corrob% | c_padding% | one-line note |
|---|---:|---:|---:|---:|---|
| asknews | 25465 | 10 | 25 | 65 | The bundle's largest section, and for a "price on ONE future day" question it's a stale intraday-quote CHRONICLE: ~15 dated single-source spot readings from Feb-Jul (many from regional/Russian outlets, one flagged AI-generated), each wrapped in [PRE-WINDOW]/[SINGLE-SOURCE] tags and a 9-item "contradictions" enumeration. The distribution is set by the July level + forecasts, which arrive faster elsewhere. |
| native_search | 5529 | 65 | 25 | 10 | MVP: the institutional Q3 forecast table (EIA $74 / ING $80 / JPM $86 / GS $82 / World Bank $86), the market-implied ~$83 Oct-contract reference, the IEA supply-balance (98.8 mb/d, 9.4 below prewar), and the disruption-scenario tails ($100+). |
| gemini_search | 6442 | 45 | 35 | 20 | Corroborates the forecast band and adds distinct tails (Citi $75 base / $130-150 disrupted, Morgan Stanley $90, Long Forecast $102) + the "SPR statutory-floor on Aug 28" and "no North Sea Brent cargoes load Aug 2026" color. ~20% vertexai URL block + garbled markers. |
| financial_data | 518 | 55 | 40 | 5 | Tiny but the PRIMARY-SOURCE anchor: BZ=F itself (the resolution ticker) at 85.43, last-5 closes (76.01→85.43), 49.9% 30-day annualized vol, 52-wk range 58.92-118.35 — the exact current level + realized vol that pin the distribution's center and width. |
| prediction_market | 3305 | 0 | 0 | 100 | Total fuzzy-match failure, the worst in the set: MLB DH/doubles, Robinhood customers, YoungBoy streams, Entourage, Prince Andrew, Hegseth, Israeli PM. NOTHING about oil. "STRONG EVIDENCE weight heavily" header over pure noise. |
| gap_fill_v1 | 19824 | 60 | 25 | 15 | The second MVP and unusually decisive: Gap 1 nails the RESOLUTION MECHANIC (BZ=F = Oct-2026 contract on its Aug-28 last trading day; Yahoo "Close" not proven = ICE settlement), Gap 3 the live curve/backwardation ($83.29 Oct/$81.97 Nov), Gap 4 confirms the Jun-17 MoU + Jul 7-8 breakdown with primary gov/wire sources + tanker counts. |
| diagnostics | 288 | 0 | 0 | 100 | Provider telemetry; operational, not forecaster content. |

**Padding exhibits (category c, verbatim quotes):**
1. [prediction_market] "polymarket | MLB: Doubles Leader | 0.02" and "kalshi | YoungBoy Never Broke Again Streams in 2026 | 0.95" under "STRONG EVIDENCE -- weight these markets heavily" — a baseball-doubles market and a rapper-streams market presented as heavily-weightable evidence for the price of Brent crude.
2. [asknews] "The Saudi TASI index reportedly fell **0.7% (-74 points)** and closed at **10,933 points**. ... The Dow Jones Industrial Average reportedly fell **0.10% (44 points)** to **51,876 points**." — equity-index levels dumped in from a source asknews itself flags as an AI-generated summary; irrelevant to a Brent close.
3. [asknews] "A move from **$89.93** at **19:14 Baku time** to **$90.41** at **19:19 Baku time** is plausible. However, the accompanying percentage moves—**2.99%** and **-1.47%** only five minutes apart—would imply materially different reference prices..." — the model auditing a five-minute-apart quote discrepancy in a regional outlet's May intraday print; meta-narration about noise no forecaster will use.
4. [asknews] "The article describes **\"Wednesday, May 28, 2026.\"** However, another supplied article identifies **May 28, 2026** as Thursday. This is a calendar/date-label contradiction..." — an entire flagged "contradiction" about which weekday a date fell on; pure data-quality throat-clearing.
5. [asknews] "**[SINGLE-SOURCE] [PRE-WINDOW — occurred before question open, cannot itself satisfy the criteria]**" — repeated verbatim as a header on every one of ~15 dated price blocks plus the "Expert Opinions" entries; load-bearing once, ~20× overhead.

**Unique-value callouts:**
- **gap_fill Gap 1 is the single most decision-critical finding in the bundle and it is entirely single-source:** BZ=F on Aug 28 represents the OCTOBER 2026 contract on its last trading day, and Yahoo's "Close" is NOT proven equal to the ICE settlement. For a question that resolves on one specific Yahoo field on one specific date, mapping the ticker to the right contract month is the whole ballgame — and only gap_fill does it. asknews raises the contract-month issue but never resolves it; native_search notes the Oct contract expires Aug 28 but not the rollover ambiguity.
- **financial_data uniquely carries the resolution ticker itself.** BZ=F at 85.43 with the last five daily closes and 49.9% annualized vol is the exact primary series the question grades on — the current level and realized volatility that set both the center and the width of a numeric forecast. Nothing else supplies the resolution instrument's own recent print; the news sections give scattered contract-specific quotes instead.
- **native_search uniquely supplies the clean institutional forecast table** (EIA $74, ING $80, JPM $86, GS $82, World Bank $86, ING severe-case $100) with dates and the market-implied ~$83. This is the outside-view distribution anchor; asknews's only forecast content is a vaguely-attributed "unnamed banks ~$80 H2 average," which native both names and dates.
- **gap_fill Gap 4 uniquely hard-confirms the geopolitical narrative with primary sources + actual tanker counts** (Jun-17 White House MoU, Jul-7 Treasury GL X1 revocation, Kpler transits 33/day→13 Jul-8→2 Jul-9). This validates the disruption-risk tail that separates the $74 baseline from the $100+ scenario — the news sections assert the narrative; gap_fill proves it and quantifies the throughput.
- **gemini uniquely flags two tail catalysts dated to the resolution window:** the SPR statutory-floor (252 mb) projected to hit ~Aug 28 and zero North Sea Brent cargoes loading for Aug 2026 — decision-relevant even if speculative.

**Cut-first verdict:** prediction_market. It is the cleanest zero-value cut in the entire ten-question set — eight markets on baseball, a rapper's streams, Robinhood, a TV show, Prince Andrew, Hegseth, and the Israeli PM, none touching oil, under a "weight these markets heavily" header. Dropping it loses literally nothing and removes an actively-misleading instruction. (asknews is by far the biggest token sink and ~65% padding here, but it does corroborate the 2026 price range and the OPEC+/inventory context, so on a strict single-section cut prediction_market wins; asknews is the dominant *trim* target and financial_data — 518 chars — must be protected as the primary anchor.)

**Halving test (two largest — asknews 25465, gap_fill 19824):**
- asknews: halving loses essentially no distribution-relevant signal. A prioritizer would keep only the price trajectory summary (the "Key Evidence" bullet list: ~$74 Feb → ~$96 May → $91.86 Jun-10 → $75.94 Jun-24 → ~$84.5 Jul-16) and the inventory + OPEC+ context, dropping all ~15 individual intraday-quote blocks with their regional-outlet sourcing, the nine-item contradictions enumeration, and the ~20× PRE-WINDOW tag repetition. Safe to cut 65-70%; the wide-range narrative survives in a few hundred chars.
- gap_fill: halving is dangerous. Gap 1 (contract-rollover resolution mechanic) and Gap 4 (primary-source geopolitical confirmation + tanker counts) are irreplaceable and single-source; Gap 3 (live curve/backwardation) is the market-implied anchor. The compressible ~15% is Gap 2's repeated "the agencies don't publish a discrete August throughput number" hedging and the per-sentence tracking URLs. A naive halve dropping Gap 1 would delete the bundle's single most important resolution-mapping fact.

---

## Judge A summary (my 4 questions: q44219, q44225, q44555, q44773)

**Per-section averages across my four questions (content-share):**

| Section | avg a_unique% | avg b_corrob% | avg c_padding% |
|---|---:|---:|---:|
| asknews | ~10 | ~21 | ~69 |
| native_search | ~58 | ~28 | ~15 |
| gemini_search | ~41 | ~35 | ~24 |
| prediction_market | ~3 | ~3 | ~95 |
| gap_fill_v1 | ~61 | ~25 | ~14 |
| financial_data (q44773 only) | ~55 | ~40 | ~5 |
| resolution_source (q44555 only) | ~25 | ~25 | ~50 |

**Overall cut-first pick: prediction_market, in all four of my questions.** The fuzzy matcher returned ZERO on-topic markets in all four (pizza/HLE/Meta for an AI-leaderboard question; Trump-spending/nuclear-datacenter/EU-referendum for an arXiv count; Senate-control/2028-VP for a Maine polling-lead; MLB/rapper-streams/Prince-Andrew for Brent crude), each under a "STRONG EVIDENCE -- weight these markets heavily" header. My q44773 was the single worst instance in the ten-question set — eight markets, none about oil. This confirms judge B's and judge C's identical finding; across all 10 questions the section is pure noise except q44255 (SAVE Act). So the fix is match-quality gating, not deletion.

**Patterns I saw (with question ids):**
1. **AskNews is the largest section, the highest-padding one (~69% padding in my four), and TWICE it was stale or directionally wrong while smaller sections were right.** q44219: asknews leaned "OpenAI most likely" off stale April data and never saw the current-July Anthropic lead that native_search + gap_fill both carried — the biggest section got the central fact backwards. q44225: asknews opened by admitting it had "no direct count" then filled 16k chars with tangential trend articles. This matches judge C's q44255/q44512 finding exactly: asknews length correlates with worse conclusions, not better ones.
2. **gap_fill_v1 carried the single decisive single-source fact in EVERY one of my four** (refutes the "re-searches known things" hypothesis): the official AA leaderboard + score reconciliation (q44219), the exact-phrase-vs-variant debunk of Gemini's count (q44225), the actual RaceToTheWH Maine averages — the literal resolution source (q44555), and the BZ=F→October-contract rollover mechanic + primary-source Hormuz confirmation (q44773). It was the highest or second-highest unique-share section every time. Its only waste is "If you want, I can..." offers and honest-null hedging (~14%).
3. **native_search is the best value-per-token section in my set** (~58% unique, ~15% padding) and gemini second, each adding genuine net-new facts (current leaderboard mirror + Creator roster; institutional forecast table + market-implied reference; replacement-candidate polls) rather than rephrasing asknews. Refines hypothesis 2: headline numbers ARE heavily duplicated, but the secondary searchers' unique third is real and often decision-critical.
4. **financial_data (q44773) is the highest-value-per-char section when present** — it carries the resolution ticker BZ=F itself (85.43, last-5 closes, 49.9% vol) at 518 chars. Protect it; never a cut candidate.
5. **Recurring mechanical padding signatures, safe to strip and independent of content judgment:** the ~250-char vertexai grounding-redirect URLs + garbled mid-word citation markers ("O[1]n", "G[1]emini") in gemini (all four); the "If you want, I can..." assistant offers in asknews + gap_fill (q44225, q44219); and the ~20-40× repeated [PRE-WINDOW]/[SINGLE-SOURCE] bracket apparatus in asknews (worst in q44555 and q44773).

**Bottom line for the operator:** cut on behavior, not token counts. (1) Halve AskNews by prioritization — largest section, ~57-69% padding, and its length tracks WORSE conclusions. (2) Gate prediction_market on fuzzy-match relevance so it stops emitting "weight heavily" over off-topic markets, but keep it when it matches (it's the most-cited section when on-topic). (3) Do NOT retire gap_fill_v1 on token grounds — this content audit shows it's the most load-bearing section per token; the sibling token audit's v1-retirement only holds if v2 demonstrably reproduces its gap-resolution quality.
