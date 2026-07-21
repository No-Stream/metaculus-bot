# Batch M2 classifications — gap-fill v2 step-zero miss audit

Classifier: batch M2 (qids 43747, 43048, 42861, 43137). One primary bucket per question per RUBRIC.md.

**Batch-level headline (read first):** None of the four M2 misses is a clean A/B/C/D research-comprehension failure. Two (43747, 43137) are the *same* numeric/discrete pipeline pathology — the truth landed above an open upper bound the models could not express past, so their declared above-bound mass never reached the submitted CDF. The other two (43048, 42861) are judgment failures (base-rate neglect / herding) where the decisive material was already in the bundle. A better agentic research stage fixes none of these as the *primary* driver. Only 43048 carries a plausible research-synthesis secondary (a truncated briefing summary).

---

### qid 43747 — Toy Story 5 domestic opening-weekend gross (numeric)
- **Cohort**: miss
- **Miss summary**: Bot's distribution put its 97.5th percentile at $150.0M (only ~2.5% above $150M); the film opened to $159.68M — above the question's displayed upper bound. Peer −66.2.
- **Decisive consideration**: The submitted distribution's upper tail was truncated at the $150M range ceiling even though every forecaster believed substantial mass sat above it.
- **Bundle check**: Research was excellent and pointed *at* the answer: Boxoffice Pro (updated June 5, forecast day) gave a "$150 million to $175 million" range whose midpoint (~$162M) brackets the $159.68M truth; native search added presales "pacing ahead of Inside Out 2" ($154.2M) and a documented Inside Out 2 tracking miss of "over 70%." Nothing decisive was missing.
- **Rationale check**: All six models declared large above-bound mass in the JSON `above_max_expected` field — gpt-5.4 0.22, gpt-5.5 0.45, claude-opus-4.7 0.42, claude-opus-4.6 0.52, gemini 0.4999. gpt-5.4: "the prompt's upper bound of 150000000 truncates the visible upper tail. Therefore, my high percentiles bunch just under 150000000." gemini: "roughly 50% of this distribution correctly falls above the maximum allowed boundary." grok confirms the range: "respect explicit [75M,150M] bounds." The published forecast nonetheless shows "97.50% chance of value below 150000000.0" — the declared above-bound tail was dropped by the pipeline.
- **PRIMARY bucket**: E
- **Secondary bucket**: none
- **Justification**: Pipeline/formatting failure (rubric's explicit E case). The models correctly read the research (they even nailed the direction — truth above $150M) but expressed above-bound mass via `above_max_expected`/`mixture_components`, which the percentile→CDF path did not consume, so only ~2.5% above-bound mass survived. This predates the repo's own 2026-07-12 OPEN_BOUND_PILING fix.
- **Confidence**: high — the rationales unanimously and explicitly declare the above-bound mass the published CDF lacks.
- **v2-addressable**: no (E) — better research would change nothing; the research already contained the answer.

---

### qid 43048 — Copernicus explicit wording, March 2026 Arctic sea ice (multiple choice)
- **Cohort**: miss
- **Miss summary**: Resolved "Lowest on record"; bot put only 15.53% there and 70.87% on "Other explicit rank or classification." Peer −64.0.
- **Decisive consideration**: Whether the models used the in-bundle 2026 record-low signals (plus Copernicus's own March-2025 "lowest on record" precedent) or fell back on a stale pre-2026 base rate.
- **Bundle check**: The bundle's native-search section explicitly carried the decisive material: Copernicus "describes 2025 Arctic sea ice as 'lowest on record for March'" (direct precedent), and the March 2026 maximum was "'lowest winter level on record'" / "ties record low" (Scientific American, NASA, NSIDC), with the 2026 peak 5.52M sq mi below 2025's 5.53M. (Confound: the top-of-report Research Summary is truncated — empty after "### **1. Critical Data Points**".)
- **Rationale check**: Three of five models asserted the 2026 data was absent and reverted to training-era base rates. gpt-5.1: "Given my knowledge cutoff (2024‑10), I do not know the actual March 2026 sea-ice data or bulletin text. I can only update based on structural/physical reasoning, not real 2026 news." gemini: "no real-time raw satellite data for March 2026 was provided in the intelligence briefing." gpt-5.2: "the prompt provides no concrete March 2026 sea-ice extent/rank numbers." These three put the lowest weights on the truth (5%, 16%, 8%); the two Anthropic models engaged more with warming and reached 25–32%.
- **PRIMARY bucket**: E
- **Secondary bucket**: A
- **Justification**: Base-rate neglect (rubric E). Per the dossier's ground rule the decisive fact was IN the bundle (Copernicus March-2025 precedent + 2026 record-low reporting), so this is not A by the letter; the models failed to use it, leaning on the stale "March record last set 2017, ~8–15%/yr" prior. Secondary A because the demonstrably truncated Research Summary is a real research-synthesis confound.
- **Confidence**: medium — turns on whether the forecaster prompt carried the full bundle (→ E, weighting failure) or only the truncated summary (→ research-synthesis, closer to A/yes). Three independent models claiming "no 2026 data" is suggestive of the latter.
- **v2-addressable**: no as primary (E); flag that a reliable-synthesis fix plausibly helps if the truncated summary is what reached the forecasters.

---

### qid 42861 — Senate confirms ≥5 Article III judicial nominees before May 1, 2026 (binary)
- **Cohort**: miss
- **Miss summary**: Resolved NO (<5 confirmed in the Mar 10–Apr 30 window); bot published 78% YES. Peer −62.4.
- **Decisive consideration**: Whether ≥3 more confirmations could clear a recess-compressed window given only 2 nominees were actually floor-ready and judicial nominees still require individual cloture.
- **Bundle check**: Adequate and two-sided. The bundle showed exactly 2 in-window confirmations, 2 on the Executive Calendar, 6 in committee, a Mar 30–Apr 10 recess, "No Senate confirmations reported since March 27," and that the Sept-2025 nuclear option/en-bloc batching covered "non-Cabinet, non-judicial nominees" only.
- **Rationale check**: Gemini used the same bundle to reach the correct 23%: "Even if the Senate confirms the 2 nominees on the calendar immediately upon returning on April 13, they will be stranded at 4 total... Any single scheduling snag pushes the 5th confirmation into May" — the minority "can certainly [delay] for a week or two—exactly enough to run out the April clock." The four bullish models (68–88%) over-weighted the 8-nominee "supply" and the non-representative "6 in a week" February burst; claude-opus-4.6 even noted "The pre-recess pace (only 2 in 19 days)" but discounted it.
- **PRIMARY bucket**: E
- **Secondary bucket**: none
- **Justification**: Judgment (rubric E). Research was adequate; the majority mis-weighted a procedural bottleneck — treating nominee supply as the binding constraint when floor time / committee-to-floor lag in a recess-shortened window was. Gemini's correct read from the identical bundle, and the resolution matching its bear case, confirm the info was there.
- **Confidence**: high — a same-bundle correct forecast exists (gemini 23%) and the NO outcome tracked its stated mechanism.
- **v2-addressable**: no (E).

---

### qid 43137 — Numbered items in Google Cloud Next '26 wrap-up post (discrete)
- **Cohort**: miss
- **Miss summary**: Resolved `above_upper_bound` (>250 numbered items, up from 231 in 2025); the bot capped its distribution at ~250 with only ~2.5% above. Peer −59.3.
- **Decisive consideration**: The submitted upper tail was jammed against the 250 range ceiling despite forecasters flagging a real "overflow above 250" scenario.
- **Bundle check**: Adequate for the task; the decisive fact (the actual count) did not yet exist (event April 22–24, forecast April 10 — a future event, so not A). Research gave the three historical anchors (161 / 218 / 231, increments +57 then +13) and an announcement-rich 2026 backdrop.
- **Rationale check**: Every forecaster pinned the 97.5th percentile at ~250.3, and several named the bound as the binding constraint. gpt-5.4: "The right tail is compressed by the practical upper range near 250." claude-opus-4.6: "The upper bound constraint at 250.5 compresses the right tail." claude-opus-4.5 flagged "High Tail (>248)... ~10-15% (constrained by practical cap)." gemini was explicit: "because the allowed scoring range caps at 250.5, any legitimate probability of the count exceeding 250 must be captured by compressing the upper percentiles against the boundary... the 97.5th percentile jammed near the 250.5 ceiling (250.3) to capture high-growth and overflow scenarios."
- **PRIMARY bucket**: E
- **Secondary bucket**: F
- **Justification**: Pipeline + tail-width (rubric E, "pipeline"). Structurally identical to 43747: models believed real mass sat above the upper bound (10–15% "overflow") but the range capped expression at ~2.5%, so the `above_upper_bound` truth was badly under-weighted. Point estimates (~238) were defensible, hence the F secondary — the count re-accelerating past 250 after a decelerating +57→+13 trend carries a genuine surprise element.
- **Confidence**: high that it is not a research failure (future event, adequate research); medium on the E-vs-F split for the primary driver.
- **v2-addressable**: no (E/F) — better research cannot forecast an unpublished editorial count, and the tail-truncation is a pipeline/numeric issue.

---

## Batch M2 summary

| qid | primary | secondary | confidence | one-line justification |
|---|---|---|---|---|
| 43747 | E | none | high | Numeric open-bound pipeline drop: models declared 0.22–0.52 above-$150M mass (gemini "~50% above the boundary"), CDF shipped only 2.5%; truth $159.7M above bound; research already had the $150–175M answer. |
| 43048 | E | A | medium | Base-rate neglect: 3/5 models claimed "no 2026 data" and used stale pre-2026 priors, though the bundle held Copernicus's March-2025 "lowest on record" precedent + 2026 record-low reporting; truncated Research Summary is a research-synthesis confound. |
| 42861 | E | none | high | Judgment/herding: 4/5 over-weighted the 8-nominee supply + non-representative Feb "6-in-a-week"; gemini used the same bundle to correctly get 23% via the floor-time/cloture bottleneck; resolved NO per its bear case. |
| 43137 | E | F | high | Same open-bound pipeline pathology as 43747: all 6 jammed the 97.5th at ~250.3; several flagged 10–15% "overflow >250" but only 2.5% shipped; truth above_upper_bound; point estimate ~238 defensible (F flavor). |
