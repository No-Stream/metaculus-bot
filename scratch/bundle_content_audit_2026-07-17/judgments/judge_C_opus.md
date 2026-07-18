# Judge C (Opus) — content audit judgments

Packets judged: q44512 (Australia CWG golds, discrete), q44551 (hottest US state Jul 31, MC),
q44255 (H.R.6644 becomes law before Jul 4, binary, RESOLVED). All three bundles compared
section-by-section against each question's own resolution criteria.

---

## Q44512 — Australia gold medals at 2026 Commonwealth Games (discrete, 47425 chars)

| Section | chars | a_unique% | b_corrob% | c_padding% | one-line note |
|---|---:|---:|---:|---:|---|
| asknews | 14947 | 30 | 35 | 35 | McKeown withdrawal + Nicholson opinion are unique; but the **Bottom line mislabels 25/28 (swimming-only) as Australia's total golds**, and 3 tail subsections re-list the head. |
| native_search | 9231 | 55 | 30 | 15 | Highest unique share: correct total-gold series AND the CGA "52 gold in the 196 overlapping Birmingham disciplines" reference class — the single best-scoped anchor for the shrunk program. |
| gemini_search | 6713 | 40 | 45 | 20 | Unique expert projection "60-plus golds" + per-sport breakdown (swimming 30+, athletics 12); projection looks inflated vs. the reduced program but is a real anchor. URL block is overhead. |
| prediction_market | 1187 | 0 | 0 | 100 | Fuzzy-match failure: Alaska salmon, Carvana, WNBA. Zero relevance; "STRONG EVIDENCE" header makes it actively misleading. |
| gap_fill_v1 | 15104 | 50 | 30 | 20 | High value: Gap 4 (rival field strength) + Gap 5 (current form) are unique and decision-relevant; Gap 3 corroborates squad sizes. Refutes the "gap_fill re-searches known things" hypothesis here. |
| diagnostics | 243 | 0 | 0 | 100 | Provider telemetry; not forecaster content (fine as-is). |

**Padding exhibits (category c, verbatim quotes):**
1. [asknews] "If you want, I can also turn this into a **forecasting memo with a probability-weighted gold-medal range** for Australia." — an offer to do more work; carries no fact.
2. [asknews] "Kaylee McKeown is described as: - **24-year-old** in CNA and The Globe and Mail - **25-year-old** in easternriverinachronicle.com.au" — an age discrepancy that cannot move a gold-count forecast.
3. [asknews] "Perth Now refers to **"Kaylee McKeon"**, clearly intended to be **Kaylee McKeown**" — a spelling-typo note with zero forecast value.
4. [prediction_market] "STRONG EVIDENCE -- weight these markets heavily." atop the salmon/Carvana/WNBA table — a heavy-weighting instruction attached to markets that share nothing but the year 2026.
5. [gap_fill_v1] "If you want, I can do a second pass and build a **full event-by-event mapping table** for the remaining sports from official Birmingham 2022 results pages." — filler offer, repeated in spirit at the end of every gap.

**Unique-value callouts:**
- **native_search is the only section that gets Australia's total-gold history right** (67/80/48/74/84/81/80). AskNews's own Bottom line says "25 at Birmingham 2022 and 28 at Gold Coast 2018" — those are *swimming-only* golds mislabeled as team totals; native_search corrects it. Single most important correction in the bundle.
- **native_search alone carries the correctly-scoped reference class:** CGA's statement that in the 196 Birmingham-2022 disciplines also on the Glasgow program, Australia won "105 medals including 52 gold." For a discrete count on a shrunk 215-event program, that 52 is the strongest anchor anywhere in the bundle.
- **gap_fill Gap 4 uniquely supplies rival-field strength** (England 43-swimmer squad led by Adam Peaty; Matthew Richardson now riding for England; NZ 19 cyclists; Canada only 6 swimmers with Summer McIntosh absent from the roster) — the one section addressing gold *conversion* vs. rivals, which offsets McKeown's loss.
- **asknews uniquely carries the McKeown withdrawal** (glandular fever, was a 4-gold winner in 2022) — genuinely load-bearing late news, though it moves the count only a few golds.

**Cut-first verdict:** prediction_market. It is 100% irrelevant (salmon/Carvana/WNBA) and its "weight heavily" header is a net negative. Dropping it loses zero forecast-relevant information.

**Halving test (two largest: gap_fill_v1 15104, asknews 14947):**
- gap_fill_v1: an intelligent halving that keeps Gaps 4-5 (rival strength, current form) and compresses the Gap 2/3 "I couldn't fully resolve… if you want I can dig more" hedging would lose little. A *naive* halving that dropped Gap 4 or 5 would delete the only rival-field-strength and current-form signals in the whole bundle — real loss.
- asknews: halving is safe. Keep the McKeown withdrawal, the historical anchors, and the Nicholson opinion; drop the "Expert Opinions and Attributions," "Key Quantitative Data Points," and "Contradictions/Inconsistencies" subsections, which restate the head. Minimal load-bearing loss (and you'd want to fix the 25/67 mislabel regardless).

---

## Q44551 — hottest US state on July 31, 2026 (multiple_choice, 44028 chars)

| Section | chars | a_unique% | b_corrob% | c_padding% | one-line note |
|---|---:|---:|---:|---:|---|
| asknews | 23864 | 15 | 25 | 60 | 54% of the bundle, worst padding density: seasonal-forecast safety tips, a weather column's non-weather local news, Palm-Springs-117°F repeated 4×, and 3 restated bottom-line subsections. Its Arizona lean ignores the actual Ogimet/Death-Valley resolution mechanic. |
| native_search | 4711 | 60 | 20 | 20 | Carries THE crux the others miss: Death Valley/Furnace Creek as the dominant US hot station, July 31 normal high 118°F / record 125°F → California climatological favorite. |
| gemini_search | 5647 | 35 | 45 | 20 | Corroborates Death Valley (125–130°F) and adds a date-specific "July 28–31 blistering finale" for the Southern Plains. Garbled inline citations + URL block are overhead. |
| prediction_market | 1338 | 0 | 0 | 100 | Fuzzy-match failure again: H5N1 / mpox / screwworm case-count markets. Zero relevance; misleading "weight heavily" header. |
| gap_fill_v1 | 8225 | 55 | 15 | 30 | On-crux throughout: Gap 2 investigates whether Ogimet includes Death Valley (the decisive mechanic), Gap 1 the Ogimet base rate, Gap 4 the updated late-July CPC guidance. Verbose "couldn't resolve" hedging is the padding. |
| diagnostics | 243 | 0 | 0 | 100 | Telemetry. |

**Padding exhibits (category c, verbatim quotes):**
1. [asknews] "Safety recommendations included: staying hydrated / preparing emergency kits / updating first-aid supplies / keeping pets indoors / improving home cooling measures such as **upgrading windows**, **weather stripping**, and **keeping blinds closed**" — heat-safety advice with no bearing on which state records the national high.
2. [asknews] "opioid and drug overdose deaths in South Carolina dropped by **one-third from 2023 to 2024**" … "the state is set for a tax-free weekend from **August 7 to August 9**" … "**$23.5 million** in federal funding for **18 bridge replacements in nine counties**" — a local column's non-weather content dumped verbatim into a weather briefing.
3. [asknews] "Brack described the weather and climate context sarcastically, saying the severity should convince deniers that **"something kooky is going on."**" — opinion, explicitly flagged as "not a meteorological forecast."
4. [prediction_market] "STRONG EVIDENCE -- weight these markets heavily." atop H5N1/mpox/screwworm markets — instruction to heavily weight three wholly off-topic disease-count markets.
5. [gemini_search] "Cli[1]matological records show that July is the hottest month for the region" and "T[1]his pattern is contributing to a prolonged period" — citation markers spliced mid-word, formatting corruption.

**Unique-value callouts:**
- **native_search is the only section that identifies the actual resolution mechanic's favorite:** Death Valley/Furnace Creek (California), July 31 normal high 118°F, record 125°F. AskNews's 23.9k-char pile of general-heat anecdotes never names Death Valley and leans Arizona on "appears most often in extreme-heat reports" — a subtly wrong anchor for a station-level Ogimet question.
- **gap_fill Gap 2 uniquely targets the true crux** — does Ogimet's US ranking include Death Valley/Furnace Creek? — and surfaces the candidate station names (GREENLAND RANCH / DEATH VALLEY CA). Even unresolved, this is the highest-leverage question anyone in the bundle asked.
- **gap_fill Gap 4 uniquely updates the outlook past the July-10 first pass:** CPC shows a stronger above-normal signal over Texas/Southern Plains than the Southwest, where monsoon cloud cover cuts confidence — a real tilt among the named options.

**Cut-first verdict:** prediction_market. 100% irrelevant (H5N1/mpox/screwworm), zero loss. (The section most in need of *trimming* is asknews, but it can't be dropped whole without losing the recent-heat regional context; see halving.)

**Halving test (two largest: asknews 23864, gap_fill_v1 8225):**
- asknews: halving to ~12k would lose essentially nothing load-bearing — you could delete the Almanac safety tips, the Charleston non-weather local news, the triplicate bottom-line subsections, and the 4× Palm-Springs-117°F repetition, and still retain every decision-relevant fact (regional heat pattern; Arizona/California prominence). You could safely cut it ~70%, not just 50%.
- gap_fill_v1: halving is riskier — all four gaps are on-crux. The compressible ~25–30% is the repeated "I couldn't reliably resolve… if you want I can keep digging" hedging. Keep the four findings, cut the hedging, and loss is low; a naive halving that dropped Gap 2 (Ogimet/Death Valley) would delete the bundle's most decision-relevant investigation.

---

## Q44255 — H.R.6644 becomes law before July 4, 2026 (binary, RESOLVED, 40630 chars)

| Section | chars | a_unique% | b_corrob% | c_padding% | one-line note |
|---|---:|---:|---:|---:|---|
| asknews | 18296 | 20 | 25 | 55 | Unique Trump-refusal quotes + Speaker Johnson "won't veto." But ~half is bill-policy detail orthogonal to a *timing* question, plus a source-grading subsection — and the **Bottom line concludes "Yes," which is wrong for a before-July-4 question** (window expires ~July 10). |
| native_search | 5793 | 50 | 35 | 15 | Gets it right: presented June 29 → automatic route "would not appear to mature before July 4… would require an affirmative signature." Unique official-status check (GovInfo ENR, no public-law number) + full legislative history. |
| gemini_search | 7264 | 35 | 45 | 20 | Uniquely spells out the day-by-day 10-day count (June 30–July 4 = days 1–5, July 5 Sunday excluded, → ~July 10) + veto-override math. But that conclusion is corroborated by native + gap_fill; heavy URL/citation overhead. |
| prediction_market | 1621 | 35 | 30 | 35 | The one question where this section earns its tokens: SAVE Act markets at 0.07–0.08 directly price Trump's stated pre-condition for signing. Midterms balance-of-power market is irrelevant. |
| gap_fill_v1 | 7413 | 55 | 30 | 15 | Decisive verification: no public-law number (GovInfo), no White House signing/veto June 29–Jul 4, Congress not adjourned (no pocket veto) — the official-source checks that actually resolve toward No. |
| diagnostics | 243 | 0 | 0 | 100 | Telemetry. |

**Padding exhibits (category c, verbatim quotes):**
1. [asknews] "Civil penalties can reach **$1 million per violation** or **three times the purchase price**." — bill substance; cannot bear on *when* the bill becomes law. (Representative of the entire "Bill content and policy details" section: institutional-investor threshold, BTR/ROFR, Innovation Fund, HOME limits — all orthogonal to the timing question.)
2. [asknews] "BeInCrypto says the bill includes a provision banning the Federal Reserve from issuing a central bank digital currency (CBDC) through **December 31, 2030**." — orthogonal to timing and single-source-unverified; the section itself flags it "not repeated elsewhere."
3. [asknews] "Based on the research provided, the most likely outcome is **Yes**… he is refusing to sign it but will allow the **10-day window excluding Sundays** to expire, which would make it law without his signature." — worse than padding: **actively wrong for this question**, because letting the window expire yields enactment ~July 10, *after* the July 4 cutoff. AskNews never does the date math against the deadline.
4. [asknews] the entire "### Source-by-source credibility notes" subsection re-grading E&E News / National Law Review / JD Supra / BeInCrypto — redundant with the per-claim "Credibility:" tags already inline above it.
5. [gemini_search] the ~13-entry "### Sources" block of `vertexaisearch.cloud.google.com/grounding-api-redirect/…` URLs plus mid-word markers like "Presidential Passage:**[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]" — formatting overhead a forecaster cannot use.

**Unique-value callouts:**
- **native_search and gemini_search get the directional answer RIGHT; AskNews gets it WRONG.** The decisive fact is calendar math: presented June 29 → auto-enactment ~July 10 → the only path to Yes before July 4 is an affirmative signature, which Trump is refusing. native states it in prose; gemini uniquely lays out the day count; AskNews concludes "Yes" by conflating "becomes law eventually" with "becomes law before July 4."
- **gap_fill uniquely does the official-source verification that resolves the question:** GovInfo shows ENR with no public-law number, White House shows no H.R.6644 signing/veto in the window, and Congress was in pro-forma (not adjourned, so no pocket veto). These are the checks that actually pin the answer to No.
- **prediction_market is uniquely valuable here:** the SAVE Act at 0.07–0.08 quantifies the pivotal pre-condition Trump attached to signing — no other section prices it. This confirms the "markets are high-value-per-token" hypothesis, but only when the fuzzy matcher finds a genuinely related market.
- On the token-audit duplication (358-32, 85-5, June 29, 10-day rule shared across asknews/native/gemini): this is legitimate **(b) corroboration** — independent retrieval paths confirming the passage margins and presentment date. But the shared numbers are *not* the decisive facts; the decisive fact is the deadline math, which only native/gemini/gap_fill carry. So the 0.75 overlap overstates redundancy: the sections agree on the easy facts and diverge on the fact that decides the question.

**Cut-first verdict:** gemini_search. Its unique contribution — the explicit day-by-day 10-day count landing on ~July 10 — is fully corroborated by native_search's prose conclusion and gap_fill Gap 2, its bill-provision summary duplicates asknews, and it carries the heaviest formatting overhead. Dropping it loses the tidiest presentation of the math but not the conclusion. (asknews has unique Trump-refusal quotes; native has the cleanest timing + official status; prediction_market has the unique SAVE Act price — all costlier to lose.)

**Halving test (two largest: asknews 18296, gemini_search 7264):**
- asknews: halving to ~9k loses little decision-relevant content — cut the entire "Bill content and policy details" section (orthogonal to timing), the "Source-by-source credibility notes" subsection, and the "Key facts and quantitative data extracted" re-list. Keep the Trump-refusal narrative, Speaker Johnson "won't veto," and the 10-day rule. Safe to cut ~60% (and the wrong "Yes" conclusion should go regardless).
- gemini_search: an intelligent halving drops the Sources URL block and the duplicated bill provisions while keeping the step-by-step day count — low loss. A naive halving that cut the day count would remove the clearest statement of the decisive math (though native/gap_fill still carry the conclusion).

---

## Judge summary

**Per-section averages across my three questions (rough):**

| Section | mean a_unique% | mean b_corrob% | mean c_padding% |
|---|---:|---:|---:|
| asknews | ~22 | ~28 | ~50 |
| native_search | ~55 | ~28 | ~17 |
| gemini_search | ~37 | ~45 | ~20 |
| prediction_market | ~12 | ~10 | ~78 |
| gap_fill_v1 | ~53 | ~25 | ~22 |

**Overall cut-first pick:** prediction_market — it was pure irrelevant noise in 2 of my 3 questions (q44512 salmon/Carvana/WNBA; q44551 H5N1/mpox/screwworm), each carrying a misleading "STRONG EVIDENCE -- weight these markets heavily" header over off-topic fuzzy matches. Critical caveat: in q44255 the same section was genuinely high-value (SAVE Act markets at 0.07–0.08 priced Trump's signing pre-condition). So the fix is not to delete the section but to **gate it on match quality** — suppress it (and especially the "weight heavily" header) when the fuzzy matcher returns markets that don't share the question's actual subject. If the operator's real goal is token reduction with minimal forecast loss, the single highest-leverage move is not dropping a section at all — it's **halving AskNews**, which was safe (often cut-70%-safe) in all three questions.

**Patterns seen repeatedly (with question ids):**
1. **AskNews is the biggest section and the highest-padding one, and twice it reached or leaned toward the WRONG conclusion while smaller sections got it right.** q44255: AskNews concluded "Yes" (ignoring that auto-enactment matures ~July 10, after the July 4 deadline) while native/gemini/gap_fill correctly reasoned No. q44512: AskNews's Bottom line mislabeled swimming-only golds (25/28) as Australia's team totals, while native_search carried the correct 67/80 series and the 52-gold reference class. This *refutes* the "native/gemini merely restate AskNews's headline numbers" hypothesis — in every question the smaller sections carried the one decision-critical fact AskNews missed or botched. That is my single most important finding.
2. **AskNews's own tail restates its head** (hypothesis 1, confirmed): q44512's "Expert Opinions" + "Key Quantitative Data Points" subsections re-list content already stated above; q44551's "Bottom-Line / Evidence-Quality / Practical-implication" subsections restate the Arizona-vs-California call three times; q44255's "Key facts extracted" re-lists the bill-detail numbers. Plus AskNews imports large slabs of source-native junk that survived summarization — heat-safety tips and a column's non-weather local news in q44551, orthogonal bill provisions in q44255.
3. **gap_fill_v1 filled real gaps, not answered ones** (hypothesis 3, refuted): rival-field strength + current form in q44512 (Gaps 4-5), the Ogimet/Death-Valley resolution mechanic in q44551 (Gap 2), and official-status + adjournment verification in q44255 (Gaps 1/3/4). It was among the two highest-unique-share sections every time; its only waste is verbose "couldn't resolve / if you want I can dig more" hedging (~15-30%).
4. **The per-claim Source/Credibility apparatus is marginally useful only for procedural/contested questions** (hypothesis 5, mixed): mildly load-bearing in q44255 (trusting E&E News over BeInCrypto's unverified CBDC claim helps), padding in q44512/q44551 (a gold-count or hottest-state forecast doesn't hinge on outlet authority). The standalone "Source-by-source credibility notes" *subsection* is always padding — it duplicates the inline tags.

**Bottom line for the operator:** don't cut on token counts — cut on the two behaviors this audit exposes. (1) Halve AskNews: it is the largest section, ~50% padding, and its length correlates with *worse* conclusions, not better ones. (2) Gate prediction_market on fuzzy-match relevance so it stops emitting "weight heavily" over salmon/mpox markets — but keep it when it finds a real market. native_search and gap_fill_v1 are the highest-value-per-token sections in my three questions and should be protected.
