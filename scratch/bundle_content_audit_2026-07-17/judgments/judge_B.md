# Judge B — content audit judgments

Packets: q44558 (Level-4 advisories count, MC), q44563 (Trump midwest visit, binary), q44453 (July 2026 jobs, discrete).
Method note: percentages are content-share estimates (nearest 5-10%), judged against each question's actual resolution criteria as pulled from the Metaculus API today, not general topical interest. I compared sections against each other directly (country rosters, candidate names, the June-57k figure, market lists) rather than trusting the token audit's overlap numbers.

---

## Q44558 — Level 4 "Do Not Travel" advisory count, Aug 2026 (multiple_choice, 61115 chars)

| Section | chars | a_unique% | b_corrob% | c_padding% | one-line note |
|---|---:|---:|---:|---:|---|
| asknews | 24992 | 25 | 35 | 40 | Carries the unique "issued vs. remains-in-effect" counting analysis, but ~40% is off-topic Australia advisories, superseded secondary reporting, and repeated PRE-WINDOW/Credibility tags. |
| native_search | 8705 | 45 | 40 | 15 | Dense; the only source with Kalshi resolution semantics and independent tracker counts (23/24). |
| gemini_search | 5451 | 30 | 35 | 35 | Adds the "Feb/Mar advisories due for Aug review" projection and an April-2025 rate datapoint, but ~1/3 is speculative crisis-labels and opaque source URLs. |
| prediction_market | 3708 | 10 | 5 | 85 | Near-total fuzzy-match failure — 7 of 8 markets are economy/elections/student-loans; nothing resolves on the August count. |
| gap_fill_v1 | 17968 | 60 | 25 | 15 | The MVP: official 23-country Date-Issued table, the Aug-2024=0 / Aug-2025=0 base rate, and the "review republishes a new date" finding. |
| diagnostics | 291 | 0 | 0 | 100 | Provider telemetry, not forecaster-facing content (operational, not "padding" in the news sense). |

**Padding exhibits (category c, verbatim quotes):**
1. [prediction_market] "kalshi | Will Trump abolish the Department of Education? | 0.14" and "kalshi | US student loan debt at end of 2026 | 0.96" — under a header reading "STRONG EVIDENCE -- weight these markets heavily," yet neither market bears on how many Level 4 advisories issue in August; the instruction to weight them is actively misleading.
2. [asknews] "**No prediction-market prices, odds, polls, or explicit forecasts of the August count appear in the research.**" — a filler sentence asserting an absence; adds nothing a forecaster can use.
3. [asknews] "**[PRE-WINDOW — occurred before question open, cannot itself satisfy the criteria] [SINGLE-SOURCE]**" — this exact block is repeated on ~12 bullets; the rule is load-bearing once, the twelvefold repetition is formatting overhead.
4. [gemini_search] "Projections for August 2026 include an ongoing **\"Iran war fuel crisis\"** and a **\"Strait of Hormuz crisis\"**" — vague scare-quoted trend labels with no fact, date, or source behind them.
5. [gemini_search] "A[1, 2, 3, 4, 5, 6] change in **U.S. government staffing**" — a garbled inline-citation artifact left in the prose; pure formatting noise.

**Unique-value callouts:**
- The historical **August base rate** (zero Level 4 entries dated August 2024, zero dated August 2025, plus a full monthly 2024-2026 distribution) is ONLY in gap_fill (Gap 2). Both asknews and native explicitly gave up on it ("no reliable August-specific historical average can be calculated"). This is the single most decision-relevant fact in the bundle and it is single-source.
- The **official "Date Issued" for all 23 Level 4 countries** is ONLY in gap_fill (Gap 1) — it lets a forecaster compute which countries are due for six-month reissuance in August (e.g. Afghanistan Feb 20 → due ~Aug). native has only a partial list.
- **Kalshi's resolution semantics** ("issues, updates, or reaffirms" broad reading vs. a narrower "not already at Level 4" contract that resolved No in Jan 2026) is ONLY in native_search — directly governs how "issue" is counted.
- The finding that a **routine six-month review republishes a new displayed Date Issued** even for minor edits (Libya, North Korea, PNG examples) is ONLY in gap_fill (Gap 4) — it validates the entire "reissuance → new count" inference the answer rests on.

**Cut-first verdict:** prediction_market. Seven of its eight markets are wholly unrelated (US economy, Senate/House races, Dept of Education, student loans), and the one adjacent market (Kalshi advisory *downgrade* at 0.04) concerns downgrades, not the issuance count. native_search already notes no relevant market exists. Dropping the section loses essentially zero forecast-relevant information and removes a misleading "weight these heavily" instruction. (asknews is the far bigger token sink, but it does carry the unique counting-distinction analysis, so it is a trim target, not a full cut.)

**Halving test (two largest — asknews 24992, gap_fill 17968):**
- asknews: halving loses almost nothing load-bearing. An intelligent prioritizer would drop the entire Australia block (Qatar/UAE/Kemish/Long — a different country's advisory system), the repeated PRE-WINDOW/Credibility apparatus, and the closing "Evidence most directly relevant" summary that restates the head; the counting-distinction analysis and the six-month/21-country facts fit comfortably in half.
- gap_fill: halving is risky if done naively — Gap 1 (date table) and Gap 2 (August=0 base rate) are irreplaceable and single-source. A smart halve keeps Gap 1+2+4 and trims Gap 3 (verification, overlaps asknews), Gap 5 (post-cutoff null), and the per-sentence "([travel.state.gov](...))" citation URLs; a dumb halve that dropped Gap 2 would gut the bundle's best fact.

---

## Q44563 — Trump visits OH/WI/MI in Aug 2026 (binary, 48404 chars)

| Section | chars | a_unique% | b_corrob% | c_padding% | one-line note |
|---|---:|---:|---:|---:|---|
| asknews | 24617 | 15 | 30 | 55 | ~51% of the bundle, but for a "did Trump physically show up" question most of it is down-ballot Democratic-primary horse-race detail; its genuinely unique+relevant share is thin. |
| native_search | 5387 | 55 | 35 | 10 | Tight and high-value: "schedule lighter than 2018 pace," One Nation ad spend, Dallas Sept convention, and the only mention of the relevant "states Trump visits in 2026" market. |
| gemini_search | 5256 | 35 | 40 | 25 | Adds concrete August competing commitments (Freedom 250 in DC, Aug 21-23) and the "rallies precede primaries" reasoning; padded with source URLs and garbled markers. |
| prediction_market | 3843 | 5 | 5 | 90 | All nine markets resolve on election *outcomes* or Iran talks — none on whether Trump sets foot in the state. |
| gap_fill_v1 | 9013 | 65 | 25 | 10 | MVP: venue-level check (Ramaswamy/Rogers/Tiffany/WI-GOP calendars, no Trump) and the August-2018 travel base rate. |
| diagnostics | 288 | 0 | 0 | 100 | Provider telemetry, not forecaster-facing content. |

**Padding exhibits (category c, verbatim quotes):**
1. [prediction_market] "polymarket | Which party will win the House in 2026? | 0.84" — a $4.7M-volume, "deep"-liquidity market flagged as STRONG EVIDENCE, but it resolves on House control, not on whether Trump visits OH/WI/MI.
2. [asknews] "> \"Mike Lindell in Minnesota, Tom Tiffany in Wisconsin; election-denying gubernatorial candidates in the Midwest. The only difference is the mustache.\"" — asknews itself immediately says "This is political commentary, not evidence of planned Trump travel"; it shares keywords but bears nothing on the resolution.
3. [asknews] "Progressive Change Campaign Committee co-founder Adam Green characterized these races as a: > \"last chance\"" — Democratic-primary strategy framing that cannot move a forecast about a Trump visit.
4. [asknews] "Kalshi traders assigned a **79% chance** of a Senate vote before the August recess. ... Polymarket traders priced a **39% probability** that the bill would be signed into law in 2026." — CLARITY Act legislative market prices, quoted at length under a section on Trump's schedule; no travel content.
5. [asknews] "The supplied research contains **no announced or reported Trump visit to Ohio, Wisconsin, or Michigan during August 2026.**" — the same negative is restated in the executive summary, "Factual Claims vs. Opinions," "Contradictions," and "Key Dates"; one statement is load-bearing, four are repetition.

**Unique-value callouts:**
- The **August 2018 comparable-cycle base rate** (nine road shows by July 7; eight August trips ≈ two per week; Ohio specifically got TWO August visits — Lewis Center Aug 4 and Columbus Aug 24; WV/Indiana as competitive-Senate destinations) is ONLY in gap_fill (Gap 3). This is the single strongest forecasting input in the bundle and it is single-source.
- The **venue/campaign-level negative check** (Ramaswamy's Gallia/Allen County fairs list no Trump, Rogers's and Tiffany's calendars show no August Trump event, WI-GOP's Aug 4 Grothman event has no Trump) is ONLY in gap_fill (Gap 2) — far more granular than the other sections' "no announcement" claims.
- **Freedom 250 Grand Prix in DC, Aug 21-23** — a concrete competing August commitment — is ONLY in gemini (unverified, but decision-relevant).
- **"Schedule lighter than the 2018 midterm pace" + One Nation's $28M-Ohio/$11M-Iowa/$100M-Michigan ad reservations + the Dallas September convention + the Polymarket "which states will Trump visit in 2026" market** are ONLY in native_search. Note the relevant travel market lives here, not in the prediction_market section.
- The candidate-identity facts the token audit flagged (Husted/Brown/Rogers) are genuinely triplicated across asknews+native+gemini — I confirmed this — but for *this* resolution criteria they are context ("why he might visit"), not resolution-bearing; one statement suffices and the other two are corroboration at best.

**Cut-first verdict:** prediction_market. Every market resolves on an election outcome (Senate/House control, individual seat winners, a Democratic nomination) or on Iran peace talks — none on Trump's physical presence in the three states. The one genuinely relevant market ("which states will Trump visit in 2026") is already surfaced by native_search. Dropping the section loses nothing and removes a misleading heavy-weight instruction. Strong secondary finding for the operator: asknews is 51% of the bundle yet its unique, resolution-relevant contribution is small — a full asknews cut would lose only the Rove GOP-environment polling (weakly relevant) because the primary dates, June-Wisconsin precedent, candidate identities, and Trump travel pattern are all corroborated in the leaner sections.

**Halving test (two largest — asknews 24617, gap_fill 9013):**
- asknews: halving loses almost nothing for this question. A prioritizer would cut the Democratic-primary ideology fight (El-Sayed/Stevens/McMorrow, Schumer/Sanders), the Minnesota candidate quotes, the Sunshine/CLARITY Act digressions, the Iran casualty figures, and the fourfold "no August visit announced" restatement — keeping the Aug 4/Aug 11 primary dates, the June precedent, and the Rove context, which is essentially all the resolution-relevant signal it holds.
- gap_fill: halving is dangerous. Gap 2 (venue check) and Gap 3 (2018 base rate) are the load-bearing single-source core; only Gap 1 (an honest "August is still future" null) is compressible. Cutting to half by dropping Gap 3 would remove the bundle's best quantitative anchor.

---

## Q44453 — U.S. jobs added, July 2026 (discrete/numeric, 37096 chars)

| Section | chars | a_unique% | b_corrob% | c_padding% | one-line note |
|---|---:|---:|---:|---:|---|
| asknews | 12932 | 15 | 35 | 50 | Anchors on June's 57k (corroborated everywhere) but ~half is source-reconciliation narration, a duplicated "Key facts" list, a doubled Goldman block, and unemployment detail that doesn't bear on the payroll count. |
| native_search | 5901 | 45 | 45 | 10 | Dense; the only source with JOLTS, First Trust trailing averages, and the June market-over-prediction calibration lesson. |
| gemini_search | 4955 | 35 | 40 | 25 | Adds the AI-displacement drag and the federal-workforce cut; padded with opaque URLs and garbled inline markers. |
| financial_data | 501 | 40 | 55 | 5 | Tiny but the only primary-source (FRED PAYEMS) anchor — the exact level series the question resolves on. |
| prediction_market | 1708 | 5 | 5 | 90 | Mortgage-rate/economy-state/productivity/SOFR markets — none resolve on the July payroll change. |
| gap_fill_v1 | 10857 | 60 | 25 | 15 | MVP: live July payroll market prices, recent-July print history + revision magnitudes, and the SPF disambiguation. |
| diagnostics | 242 | 0 | 0 | 100 | Provider telemetry, not forecaster-facing content. |

**Padding exhibits (category c, verbatim quotes):**
1. [asknews] "If you want, I can also convert this into a forecaster-friendly format with: 1. a probability distribution for July 2026 payroll outcomes, or 2. a short \"base case / bullish / bearish\" scenario memo." — an AI-assistant offer with zero research content, embedded in a briefing forecasters read as fact.
2. [asknews] "Combined revisions for April and May were reported as: - **74,000 jobs lower** in one source - **74,000 jobs lower** in another source - One source says **74,000**; another says **74,000** through the combination of **31,000** and **43,000**." — the model narrating its own source-reconciliation, repeating "74,000" four times to convey one number.
3. [prediction_market] "kalshi | SOFR at end of Q4 2026 | 0.96" — a Q4 SOFR-level market presented as STRONG EVIDENCE for a July jobs-added question it cannot inform.
4. [asknews] the entire "#### Key quantitative facts collected" list — ~24 figures (June 57k, May 172k, claims 215k, Goldman 1.8%/30%/4.6%, etc.) that were all already stated earlier in the same section; a within-section restatement.
5. [gemini_search] "*[5]   **Sector Outlook:**" and "[6]### **Key Influencing Factors**" — stray citation markers fused into headings; formatting artifacts.

**Unique-value callouts:**
- **Live prediction-market prices for the July release** — Coinbase "above 80,000" at 52% (a coin-flip right at 80k) and Manifold "≥150,000" at 25% — are ONLY in gap_fill (Gap 1). native and asknews only note the markets *exist*. For a numeric point-estimate question this is the strongest single outside-view anchor.
- **Recent-July print history and revision magnitudes** (July 2025 +73k, 2024 +114k, 2023 +187k, 2022 +528k; mean absolute revision 41k second-vs-first, 56k third-vs-first) are ONLY in gap_fill (Gap 4) — they set both the location and the width of the forecast distribution, which nothing else supplies.
- **Trailing averages (3-mo +111k, 6-mo +92k, 12-mo +42k) and JOLTS May (7.6M openings, 5.2M hires, 1.7M layoffs)** are ONLY in native_search; the **June market over-prediction** (IBKR/Kalshi ~111.7k coin-flip vs. 57k actual) is also native-only and is a direct warning against over-anchoring on the July markets.
- **AI displacement removing ~10-15k jobs/month (Goldman) and the federal workforce down 324,000 since Jan 2025** are ONLY in gemini — structural drags that argue for a lower central estimate.
- **financial_data** is the only non-news, primary-source anchor: PAYEMS level series (Jan 158.6M → June 159.0M, +57k, YoY +506k). High value per its 501 chars.

**Cut-first verdict:** prediction_market. Its four markets (30-yr mortgage rate, economy-state combo, nonfarm productivity, Q4 SOFR) are macro-flavored keyword matches; none resolves on the July payroll change. The genuinely relevant payroll markets (Coinbase, Manifold) are captured by native_search and gap_fill. Cutting it loses nothing and removes the misleading "weight heavily" header. (financial_data is even smaller but is the primary-source anchor — do not cut it. asknews is the biggest sink and the prime trim target, but a full cut sacrifices its corroboration of June's 57k and the sector detail.)

**Halving test (two largest — asknews 12932, gap_fill 10857):**
- asknews: halving loses essentially no payroll-relevant signal. A prioritizer would delete the revisions-confusion narration, the duplicated "Key quantitative facts collected" list, the doubled Goldman block, the demographic/long-term-unemployment detail (bears on the unemployment rate, not the payroll change), and the "If you want, I can convert this..." closing — leaving June 57k, the May→June deceleration, and claims 215k intact.
- gap_fill: halving is risky. Gap 1 (market prices), Gap 3 (SPF 34,600 annual vs. 68,900 Q2 disambiguation), and Gap 4 (July-print history + revisions) are the single-source load-bearing core; Gap 2 (June consensus, overlaps native's LSEG 110k) and Gap 5 (indicator calendar) plus the two "If you want, I can keep checking..." offers are the compressible half. A naive halve that dropped Gap 4 would remove the only content shaping the distribution's width.

---

## Judge summary

**Per-section averages across my three questions (content-share):**

| Section | avg a_unique% | avg b_corrob% | avg c_padding% |
|---|---:|---:|---:|
| asknews | ~18 | ~33 | ~48 |
| native_search | ~48 | ~40 | ~12 |
| gemini_search | ~33 | ~38 | ~28 |
| prediction_market | ~7 | ~5 | ~88 |
| gap_fill_v1 | ~62 | ~25 | ~13 |
| financial_data (q44453 only) | ~40 | ~55 | ~5 |

**Overall cut-first pick: prediction_market, in all three questions.** This was the clearest and most consistent finding, and it directly refutes the audit's hypothesis 4. In every packet the section's fuzzy matcher pulled keyword-adjacent markets that do NOT resolve on the question's criteria (student loans / Dept of Education for a travel-advisory count; Senate/House control for a "did Trump visit" question; SOFR / mortgage rates for a jobs-added number), while the genuinely relevant markets — the Kalshi Level-4 semantics, the Polymarket "which states will Trump visit in 2026," the Coinbase/Manifold July-payroll thresholds — were surfaced instead by native_search and gap_fill. The section's "STRONG EVIDENCE — weight these markets heavily" header is therefore not just low-value but actively harmful here, instructing forecasters to anchor on irrelevant prices. If the operator wants a token cut backed by content, prediction_market is the safe first cut on questions where the fuzzy match misses.

**Patterns seen repeatedly:**
- **gap_fill_v1 is the workhorse, not a redundant re-search (refutes hypothesis 3).** In all three it filled the decision-critical gap nothing else had: the August-issuance base rate (q44558 Gap 2), the 2018 travel base rate and venue check (q44563 Gaps 2-3), the live market prices + July-print history (q44453 Gaps 1,4). Its "Why it matters" framing reads as meta but is cheap and it earns its tokens. Do not treat it as a cut candidate.
- **AskNews is the largest section and the largest waste, but for a subtle reason (partially confirms hypotheses 1 and 5).** Its bloat is less about tail-restates-head (though that happens — the closing summaries in all three restate their heads, and q44453 literally re-lists its own numbers) and more about **content that shares the topic's keywords but not the resolution criteria**: off-topic Australian advisories (q44558), down-ballot Democratic-primary horse-race (q44563), and unemployment-rate detail on a payroll-change question (q44453). Its per-claim "[PRE-WINDOW]/[SINGLE-SOURCE]/Credibility: Medium-low" apparatus is load-bearing exactly once (it correctly flags that asknews leans on weak secondary outlets later superseded by gap_fill's official data) and pure repetition the other dozen times. AskNews consistently halves with near-zero loss of resolution-relevant signal.
- **native_search is the best value-per-token section** (~48% unique, ~12% padding) and gemini second, each adding real net-new facts (JOLTS/trailing-averages/market-calibration; AI-displacement/Freedom-250/base-rate reasoning) rather than merely rephrasing AskNews. This refines hypothesis 2: the *headline numbers* are heavily triplicated (I confirmed the Level-4 country roster is quadruplicated and the June-57k figure is in five sections), but native/gemini are net-positive on unique content and are the wrong place to cut.
- **The recurring padding signatures across providers:** AI-assistant "If you want, I can..." offers (asknews + gap_fill in q44453), garbled inline citation markers and opaque `vertexaisearch.cloud.google.com` redirect URLs (gemini in all three), and per-sentence tracking-URL citations (gap_fill). These are mechanical, safe to strip, and independent of the harder content-relevance judgment.
