# Batch M3 classifications — gap-fill v2 step-zero miss audit

Classifier: classifier-M3. Cohort: all four are misses.

---

### qid 42864 — CDC novel influenza A human infection before May 1, 2026
- **Cohort**: miss
- **Miss summary**: Bot published 65% YES that CDC would newly report ≥1 novel influenza A human infection in the Mar 10–Apr 30 window; it resolved NO (peer −58.2).
- **Decisive consideration**: The 2025→2026 collapse in spillover reporting (67 H5N1 cases in 2025 vs 3 in 2026 YTD, plus multiple recent FluView weeks of zero new novel cases) meant the effective hazard was far below the naive early-2026 extrapolation — the forecasters over-extrapolated a tiny, already-declining sample.
- **Bundle check**: The bundle HAD every relevant signal: "67 in 2025 and 3 in 2026," and "No novel influenza mentions in CDC FluView summaries for weeks 6, 7, 9, or 11 (February-March 2026)." The declining trend and the dry weeks were explicit.
- **Rationale check**: gpt-5.2 (Forecaster 1) read the same bundle correctly and anchored at 32%, calling the elevated-H5N1 items "internally inconsistent" and "less like a direct CDC FluView extract." The three that dragged the median up over-extrapolated: gpt-5.1 built a Poisson off "3 cases in ~80 days → λ ≈ 1.1 … ≈ 67%," and gemini-3.1-pro went to 84% off "3 cases in 2.5 months … Poisson probability of >0 cases with a mean of 2.04 is approximately 87%." Both acknowledged the dry-weeks bear signal, then discounted it.
- **PRIMARY bucket**: E
- **Secondary bucket**: none
- **Justification**: Research was adequate (declining rate + dry weeks both in-bundle and explicitly discussed); the median was dragged up by base-rate neglect — over-fitting a Poisson to 3 early-2026 cases while under-weighting the sharp YoY decline and 6–8 quiet weeks. One forecaster demonstrated the correct read was achievable from the same bundle.
- **Confidence**: high — would change only if a fresher pre-submission FluView week had reported a qualifying case (it hadn't; the bundle's own search found none through week 11).
- **v2-addressable**: no (E)

---

### qid 43145 — Anthropic Opus-class model newer than Opus 4.6 before May 1, 2026
- **Cohort**: miss
- **Miss summary**: Bot published 6.5% YES that Anthropic would announce a newer-than-4.6 Opus-class model available to external users before May 1; it resolved YES (peer −52.8, baseline −187.7 — a catastrophic miss).
- **Decisive consideration**: The resolution bar was "available to **any external users**," and Claude Mythos was announced Apr 7 in a private preview to external companies — before the Apr 12 forecast; the forecasters narrowed "any external users" to "general public availability," excluding a preview that literally satisfied the phrase.
- **Bundle check**: The externally-available frontier model was IN the bundle: "Private preview via 'Project Glasswing' limited to 11 select companies (and approximately 100 companies in total)." The bundle also noted the tension the resolver had to weigh — title says "Opus-**class**," criteria says "Opus-**branded**."
- **Rationale check**: gemini-3.1-pro quoted the criterion phrase and then narrowed it: Mythos "has explicitly not been made 'available to any external users' **in the public, general availability sense**." opus-4.6 (Forecaster 4) did the same: "Mythos exists but isn't publicly available." gpt-5.4 leaned on an unverifiable claim that "the admins explicitly say the criteria 'have not yet been satisfied'" (not present in the bundle) to exclude Mythos.
- **PRIMARY bucket**: C
- **Secondary bucket**: A
- **Justification**: Misread resolution mechanics — the models imposed a "public/general availability" threshold that isn't in the criterion; a private preview to 11–100 external companies (in-bundle, pre-submission) plausibly already cleared "available to any external users." Specific misreading: treating "any external users" as "generally available." Secondary A because I cannot rule out that YES was instead driven by a clean Opus-branded public release (e.g., an Opus 4.7) shipped in the Apr 13–30 window that the Apr 12 bundle couldn't contain.
- **Confidence**: medium — I cannot establish from the dossier whether the YES came from Mythos being counted (→ C confirmed) or from a separate public Opus-branded release after the research cutoff (→ A, or F if genuinely unforeseeable). The "Opus-branded" (vs "Opus-class") wording is a real ambiguity that could make the strict Mythos-excludes reading correct. What would change the call: the actual resolution note / whether an Opus 4.7-type public release landed before May 1.
- **v2-addressable**: yes (C; and A if the alt path holds — either way A–D)

---

### qid 42577 — Faroe Islands 2026: every party under 10 seats
- **Cohort**: miss
- **Miss summary**: Bot published 22% YES that every party would win <10 seats; it resolved YES (peer −46.6, baseline −96.6). The bot bet ~78% that some party would break 10 seats; none did.
- **Decisive consideration**: A single, stale (Feb 3, pre-campaign, n=500) Spyr.fo poll putting the People's Party at 37.3% was mechanically converted to "0.373 × 33 ≈ 12.3 seats," overriding a 5/5 historical record of no party reaching 10 seats under the 33-seat system.
- **Bundle check**: Bundle had the strong base rate — "In all five elections [2008–2022], no single party reached the 10-seat threshold" — AND the staleness caveats: "The 37.3% figure … reflects the sentiment *before* the formal campaign began" and "No polls found after February 25 announcement," plus "No seat projections available." The correct anchor and the reasons to discount the poll were both present.
- **Rationale check**: Every forecaster acknowledged the ~85% fragmentation base rate, then abandoned it. claude-4.6-opus: "base rate was ~25% … moving to 15%" driven by "37.3% × 33 ≈ 12.3 seats." gemini-3.1-pro moved from an "85%" base to 22% on the "D'Hondt … only about 27.5%–29%" seat math. gpt-5.2 stayed closest to sane (35%) but still inverted the base rate.
- **PRIMARY bucket**: E
- **Secondary bucket**: none
- **Justification**: Textbook base-rate neglect — the single most predictive fact (5/5 elections, max 9 seats) was in the bundle and explicitly cited by every model before being overridden by naive proportional seat math applied to one stale, pre-campaign poll whose staleness the bundle flagged.
- **Confidence**: high — the historical regularity and the poll caveats are both in-bundle. Would change only if a decisive fresher pre-submission poll/seat-projection existed (the bundle's own search found none) or if a resolution-source technicality drove YES (no sign of that; forecasters read the mechanics correctly).
- **v2-addressable**: no (E)

---

### qid 42313 — Apple iOS/iPadOS security note citing active exploitation, Mar 15–May 1, 2026
- **Cohort**: miss
- **Miss summary**: Bot published 76% YES that Apple would publish an iOS/iPadOS note with "actively exploited" (or equivalent) language in the window; it resolved NO (peer −41.3, baseline −86.8).
- **Decisive consideration**: The window happened to be quiet, and the forecasters conflated "exploitation is confirmed by CISA/Google (Coruna, 19 unpatched)" with "Apple will publish an iOS/iPadOS note using the explicit exploitation language" — over-weighting a scary threat narrative and over-estimating the base rate for the specific criterion.
- **Bundle check**: The bundle contained BOTH the base-rate ingredients (~9 zero-days across Apple in 2025) AND the exact counter-signal: SecurityWeek's note that for the Mar 11 Coruna legacy patches, "Apple's advisory does **not** explicitly state awareness of in-the-wild attacks." So the bundle explicitly showed Apple patches exploited vulns without the qualifying language.
- **Rationale check**: gpt-5.1 (Forecaster 2) did the careful conversion — "~5 iOS/iPadOS exploited advisories/year … P(≥1) ≈ 47–48%" → final 51% — and flagged the mapping gap. The Claude models over-weighted Coruna: claude-4.6-opus went to 97% ("19 unpatched exploits creates near-certain demand"), opus-4.5 to 88%. gemini-3.1-pro (the 76% median) leaned on a speculative mechanic — Apple would "retroactively update … changing the page's formal publication date" into the window — which didn't happen.
- **PRIMARY bucket**: E
- **Secondary bucket**: none
- **Justification**: Research was adequate — the base rate and the decisive "confirmed exploitation ≠ Apple's explicit language" nuance were both in-bundle. The large miss came from overconfidence: over-weighting the Coruna "19 unpatched" narrative and a speculative retroactive-date mechanic, while under-weighting the in-bundle counterexample. A calibrated read (~50%, per Forecaster 2) resolving NO would be unremarkable; the 76% median (and 88–97% tail) is the miscalibration.
- **Confidence**: high — the counter-evidence Apple sometimes omits exploited language is verbatim in the bundle and was cited by the models, who then discounted it.
- **v2-addressable**: no (E)

---

## Summary

| qid | primary | secondary | confidence | one-line justification |
|-----|---------|-----------|------------|------------------------|
| 42864 | E | none | high | Over-extrapolated 3 early-2026 flu cases into a Poisson; ignored the in-bundle 67→3 YoY collapse + 6–8 dry weeks (one model read it right at 32%). |
| 43145 | C | A | medium | Narrowed "available to any external users" to "public availability," excluding the Mythos private preview (in-bundle, pre-submission); alt path A if a clean Opus release drove YES. |
| 42577 | E | none | high | Base-rate neglect: overrode 5/5 "no party ≥10 seats" with naive 37.3%×33 seat math off one stale pre-campaign poll the bundle flagged as stale. |
| 42313 | E | none | high | Overconfidence: conflated CISA/Coruna-confirmed exploitation with Apple's explicit iOS/iPadOS language, ignoring the in-bundle Mar 11 counterexample. |
