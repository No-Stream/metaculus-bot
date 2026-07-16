# Batch M4 classifications — gap-fill v2 step-zero miss audit

Classifier: classifier-M4. Cohort: all four are misses.

---

### qid 43147 — NATO Article 4 consultation requests citing 2026 Iran conflict (discrete count)
- **Cohort**: miss
- **Miss summary**: Truth was 0 distinct states; the published distribution put only 53.4% below 7.5, 68% below 15.5, and pinned the 97.5th percentile at the upper bound 32.5 — i.e. it smeared roughly half its mass above 7 states on a question where every forecaster's own numbers said the answer was ~0 (peer −34.7 vs baseline 77.2).
- **Decisive consideration**: The per-model percentile→CDF construction for a discrete count with a huge atom at 0 destroyed the concentration at 0 and spread the mass across the whole 0–32 range; the reasoning and research were correct, the submitted distribution was not.
- **Bundle check**: Research was excellent and dead-on: native search states "**As of April 12, 2026, zero distinct NATO member states have formally requested Article 4 consultations**," AskNews shows Turkey "did not mention Article 4" after four intercepts, Rutte "absolutely no plans," and "I could not find information on any planned before May 1." Nothing decisive was missing or stale — the truth (0) matched the bundle.
- **Rationale check**: Every forecaster concentrated mass tightly near 0. gpt-5.4 (F1) gave P50=0.2, P90=1.08, **P97.5=3.8**; gemini-3.1-pro (F5) gave "**P50 = 0.00**", P90=0.45, P97.5=3.50; opus-4.6 (F4) P90=0.6, P97.5=2.8. Yet the "Forecaster 1" pipeline summary reports only **55.11% below 7.5** and 97.5% below 32.5 — i.e. the pipeline turned F1's ~99% below 7.5 into 55% below 7.5. The distortion is present at the per-model level, before ensemble aggregation.
- **PRIMARY bucket**: E
- **Secondary bucket**: none
- **Justification**: E-**pipeline** (the rubric's explicit "pipeline/formatting failure distorted the submitted forecast" case). The models put >97% of mass below ~3–4 states (correct; truth=0), but the discrete-count CDF builder — bounds [-0.5, 32.5], 33 integer outcomes, most declared percentiles sub-integer and clustered near 0 — spread the atom-at-0 mass across integers up to the bound (the 97.5th pct pinned at exactly 32.5 across all six models is the tell). A better research stage cannot fix this; the research was already right.
- **Confidence**: high — the per-model raw percentiles vs the per-model displayed CDF are flatly inconsistent, and a tight-near-0 submission would have scored well (truth=0), so the large negative peer score confirms a smeared distribution was actually submitted, not just displayed.
- **v2-addressable**: no (E)

---

### qid 43727 — # SCOTUS Justices voting for Monsanto in Monsanto v. Durnell (MC)
- **Cohort**: miss
- **Miss summary**: Truth was 7 (a 7-2 Monsanto win); the bot peaked at 4 (25.5%) and 5 (28.1%) and put only 6.1% on 7, leaning toward a narrow Monsanto loss (peer −27.8, baseline −16.4).
- **Decisive consideration**: The forecasters discounted a strong structural pro-Monsanto prior (Solicitor General support + 6-3 conservative Court + ~70% markets, a reference class that wins ~70-80% often by 6-3/7-2) in favor of noisy oral-argument "skepticism" reporting, herding the mode down to 4-5 and starving the comfortable-win region (6-7) that the base rate supported.
- **Bundle check**: The decision had not issued (expected late June/July), so there was no missing/stale outcome fact. The bundle carried both the structural prior — "The U.S. filed an amicus brief supporting Monsanto … SG position prevails ~70-80% of the time" — and the mixed oral-argument reads ("divided," "tossup," one outlier "**7-2 in favor of Durnell**"). It even flagged the trap: "Justices frequently ask tough questions of the side they ultimately support."
- **Rationale check**: gpt-5.4 (F1) set mode=4 on "the simplest coalition … most plausibly Barrett … joining Monsanto, while the other two [Roberts/Gorsuch] join the liberals … yields a 5-4 decision against Monsanto." grok-4.3 (F6) likewise: "Most-likely = 4 votes (narrow Monsanto loss) … US support raises floor but does not overcome reported argument signals." opus-4.7 (F3) was the widest and least wrong, keeping 7 at 13% and 8 at 7% off its "peak at 5-6" prior — and the aggregation pulled toward the narrower gpt/gemini/grok models that peaked hard at 4-5.
- **PRIMARY bucket**: E
- **Secondary bucket**: F
- **Justification**: E-judgment — over-updating on oral-argument vibes against a strong SG/preemption base rate, and collective herding on the "divided court" narrative (rubric: herding on signals that were wrong = E). A 7-2 win for an SG-backed petitioner before a 6-3 Court is a mainline outcome, not a tail; the ensemble talked itself out of it and over-concentrated on 4-5. Secondary F because the whole field lost (baseline also −16.4), so there was a genuine-surprise component — but the bot's extra ~11-point loss vs baseline is the over-concentration error.
- **Confidence**: medium-high — would shift toward F only if the SG/base-rate prior were weaker than the bundle presents; would shift to A/C only if a decisive pre-decision fact (leak/opinion) existed pre-submission, which it did not (case undecided as of May 29).
- **v2-addressable**: no (E/F)

---

### qid 42298 — VIX close range on April 30, 2026 (MC)
- **Cohort**: miss
- **Miss summary**: Truth was 15.0–<20.0; the bot put only 23.7% there and centered on 20-25 (37.1%) and 25+ (30.9%) — i.e. ~68% on ≥20 (peer −19.6 vs baseline −3.7).
- **Decisive consideration**: Forecasting on March 7 with spot VIX freshly spiked to ~30 (Iran war + weak jobs print), the models over-anchored on the current spike and on April/May futures (~22-25) and under-weighted the strongly-documented rapid mean-reversion, starving the 15-20 bucket that actually resolved.
- **Bundle check**: The bundle was balanced and complete for a future-date question — it had the spike ("VIX … close at **29.49** … highest since April 2022"), the futures ("April 2026 futures … **22.37** … alternative … 25.00"), AND the mean-reversion history ("surged from 16 to 22 on Oct 10, briefly approached 30, then fell back to the teens by Oct 21"; April 2025 52→~20 in weeks; "such spikes are often short-lived"). It even warned futures "reflect … mid-April … not the spot close on April 30 specifically."
- **Rationale check**: gpt-5.2 (F1) centered 20-25 because "the term structure implies mean reversion … futures ~22–24," discounting its own unconditional anchor that had 15-20 at 30%. gemini-3.1-pro (F5) noted "Futures suggest an expected spot price centered around **20–22** (accounting for the … risk premium)" yet still put 34% on 20-25 and only 30% on 15-20. All five leaned on the elevated regime + futures and shaded away from the mean-reversion base case.
- **PRIMARY bucket**: E
- **Secondary bucket**: none
- **Justification**: E-judgment (anchoring / over-persistence). The research already contained the mean-reversion evidence and the risk-premium caveat that pointed spot toward ~20-21; a calibrated forecast would have kept more mass on 15-20 (the modal region under the base rate). The models herded to the futures level — which sits systematically above realized spot due to the vol risk premium — and over-weighted the fresh spike. 15-20 was not a tail (their own unconditional anchors put it at 30-40%), so this is judgment, not a surprising world.
- **Confidence**: medium — there is a defensible F reading (the futures anchor was legitimate and late-April was genuinely uncertain), but the heavy ~68% weight on ≥20 against strong in-bundle mean-reversion evidence, plus the underperformance vs a much-better baseline, tips it to E.
- **v2-addressable**: no (E)

---

### qid 42312 — ≥1 "Critical" in Android Framework/System, April 2026 bulletin (binary)
- **Cohort**: miss
- **Miss summary**: Resolved YES; the bot published 52% (near coin-flip) while peers were much more confident YES (peer −19.6 vs baseline +4.0). The bot was under-confident on a fairly clear YES.
- **Decisive consideration**: Every forecaster anchored on a "last 5 months = 2/5 = 40%" base rate that was both too-small a window and likely mis-tabulated; the true rate of Critical Framework/System entries is materially higher, and a correct base rate would have pushed the forecast to a confident YES.
- **Bundle check**: The bundle's own recent-window count is questionable and it explicitly lacked long-run data. It counted **December 2025 as NO**, yet its own source articles describe "**a critical vulnerability in the Android Framework that enables remote denial-of-service (DoS) attacks**" (Dataconomy/Android Authority) — and that CVE (CVE-2025-48631) later appears as "Critical" in March 2026. February was a hedged NO ("insufficient details on Criticals in those specific components"). Native search admits "**aggregate monthly statistics or long-term frequency data … were not found in reliable sources**." The long-run rate (which one model knew to be ~75-85%) was absent.
- **Rationale check**: gpt-5.2 (F1): "I anchor on **40%** as the cleanest documented base rate … moving to 52%." gemini-3.1-pro (F5) went *below* the anchor to 38% on "the record-breaking sweep in March likely cleared the … backlog." Only opus-4.7 (F3) up-weighted history — "**Longer historical (2020–2024): … ~75-85% of monthly bulletins**" — and reached 72%, the closest model; the aggregate median was dragged down by the four that anchored on 40%.
- **PRIMARY bucket**: A
- **Secondary bucket**: C
- **Justification**: A — the decisive input (an accurate, longer-run base rate showing Critical Framework/System entries are common, well above the bundle's 40%) was absent; the bundle even states it could not find long-term frequency data. Secondary C — the recent-window tabulation the models anchored on was itself likely miscounted (Dec 2025's Critical Framework DoS counted as NO; Feb a hedged NO), biasing the anchor down. A better agentic stage that pulled and correctly tallied the primary source.android.com tables over 12-24 months would very plausibly have produced a >50% base rate and flipped this to a hit.
- **Confidence**: medium — there is a real E/judgment component (the ensemble over-anchored on a tiny noisy sample when one model reached the right answer by up-weighting the long-run rate from its own knowledge), and I cannot verify from the dossier alone whether Dec 2025's official table labeled the Framework DoS "Critical," so the C component is probabilistic. What would change the call: the actual Dec-2025 source.android.com severity table and a verified long-run frequency.
- **v2-addressable**: yes (A/C)
