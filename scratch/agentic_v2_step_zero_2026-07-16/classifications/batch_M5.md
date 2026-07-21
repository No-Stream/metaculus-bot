# Batch M5 classifications — gap-fill v2 step-zero miss audit

Classifier: classifier-M5. Four misses (43077, 43059, 42751, 42236). Judged from
the dossiers only; no external lookups. Where a decisive fact's pre-submission
public existence can't be established from the dossier, confidence is lowered and
noted explicitly.

---

### qid 43077 — "Michael" opening-weekend domestic gross (numeric)
- **Cohort**: miss
- **Miss summary**: Bot centered its distribution at a $82M median (90% below $96M, 97.5% below $100M); the film opened to $97.2M, landing at roughly the 92nd percentile of the bot's own distribution (peer −15.7).
- **Decisive consideration**: The submitted distribution's center was ~$15M low and its right tail too thin relative to the bullish momentum signals already present in the bundle; had the center/upper tail been pushed up, $97.2M would sit near the middle instead of the far tail.
- **Bundle check**: The bundle carried the full projection range and the bullish momentum — BoxOffice Pro "$80 million to $90 million," Deadline "north of $60M," and multiple higher signals: "early ticket sales are 'absolutely killing it,'" k.sina "could potentially reach $100 million," fan/MJ Beats "$100 million or more," and a native-search note of "a $30M jump" in tracking. The central estimate was near the truth; the distribution wasn't.
- **Rationale check**: Forecaster 1's meta-analysis notes all six models "cluster tightly around a median of $80.8–$84.5M" and even flags that "the intelligence briefing's own targeted research conclusion of '$60-70M' is notably more conservative," yet the ensemble still centered at $82M with p90 ≈ $96M — under-weighting the surging-presales upside it acknowledged. The right tail was capped just at the $100M question ceiling.
- **PRIMARY bucket**: E
- **Secondary bucket**: F
- **Justification**: Research was adequate and correctly summarized; the models weighed it slightly too conservatively (center low, right tail thin) despite explicit "presales killing it / potentially $100M" signals in the bundle — a tail-width/central-tendency judgment miss, not a research-comprehension gap. F is secondary because $97.2M did exceed every professional projection (max $90M), so part of the miss is a genuinely strong opening.
- **Confidence**: medium — a future opening-weekend gross is inherently unknowable at forecast time, so the only lever was calibration; would shift toward F if the community also missed high (peer −15.7 suggests they didn't, supporting the judgment call).
- **v2-addressable**: no (E–F)

---

### qid 43059 — highest Europe Brent daily spot price, 2nd half April 2026 (numeric)
- **Cohort**: miss
- **Miss summary**: Bot forecast a $146 median (p10 ≈ $100, p90 ≈ $189); the highest EIA Europe Brent daily spot April 15–30 was $116.63, near the bot's ~18th–20th percentile (peer −13.5, baseline +4.6).
- **Decisive consideration**: The models anchored the whole distribution on the $141.36 "spot" print, which is the S&P Global physical/dated-Brent assessment — a *different series* from the resolution's EIA Europe Brent Spot Price FOB (mirrored on FRED DCOILBRENTEU), which the bundle showed trading ~$30 lower.
- **Bundle check**: Two conflicting series were both in the bundle. AskNews/S&P Global: spot "surged to $141.36 per barrel on April 2 ... $32.33 higher than the June Brent futures contract" amid "a historic 'decoupling' between physical spot prices and futures." But the native search cited the actual resolution source directly — "recent historical highs of $121.88 on March 30, 2026" and "Brent spot traded around $108.51–$111.25" — with citation to `fred.stlouisfed.org/series/DCOILBRENTEU`. The resolution value ($116.63) is consistent with the FRED/EIA range, not the S&P $141 physical premium.
- **Rationale check**: Forecaster 1's synthesis states "All models anchor on the $141.36 April 2 spot print" and derives "The median ~146 reflects: $141 already achieved + maximum statistic uplift." The briefing itself drove this conflation: "Forecasters should prioritize the spot price trends, as the resolution criteria specifically uses the EIA Daily Spot Price" — pointing them at the $141.36 physical figure as if it were the EIA daily spot, when FRED (the candidate source) showed ~$109–122.
- **PRIMARY bucket**: C
- **Secondary bucket**: D
- **Justification**: Wrong-source/wrong-metric misread — the specific error is treating the S&P Global physical dated-Brent spot ($141.36) as the EIA Europe Brent Spot Price FOB the question resolves on (~$109–122 per the bundle's own FRED cite), inflating the median ~$30. D secondary because the research synthesis affirmatively misattributed the $141.36 figure to "the EIA Daily Spot Price."
- **Confidence**: medium — internal evidence is strong (bundle's FRED cite of $121.88 max + the $116.63 resolution both refute a ~$141 EIA print, and peer −13.5 vs baseline +4.6 implies the community used the lower/correct series). Would flip toward F/E only if the EIA Europe Brent Spot Price FOB itself actually printed ~$141 in early April, which the dossier contradicts.
- **v2-addressable**: yes (A–D)

---

### qid 42751 — new cabinet-level UK Secretary of State before May 1, 2026 (binary)
- **Cohort**: miss
- **Miss summary**: Bot published 44% YES (per-model 15/40/44/55/65); it resolved NO — no Secretary of State was newly appointed in the window (peer −12.4).
- **Decisive consideration**: The window (Mar 17–May 1) ends six days *before* the May 7 local elections, and the most credible leak in the bundle said any reshuffle's timing/scope depended on the election result — strongly implying action *after*, not before, May 1.
- **Bundle check**: The decisive delay signal was present and explicit. Guido Fawkes (the best-sourced item): the reshuffle "scope will depend on the outcome of the May elections" and "No immediate ministerial changes are expected." The bundle's own contradictions section flagged: reshuffle "contingent on post-May political conditions." Also present: zero SoS changes in 2026, last reshuffle only ~6.5 months prior, purdah around late March. Nothing missing or stale.
- **Rationale check**: Gemini (Forecaster 5, 15%) weighed this correctly — "the most credible operational leak explicitly states changes depend on the election outcome ... strong gravitational pull to delay ... until the May 1 deadline has passed." But claude-opus-4.5 (65%) and claude-4.6-opus (55%) over-weighted "civil service turnover ... historically predictive of ministerial change" and multi-outlet reshuffle chatter, dragging the ensemble median to 44%. All models passed the bait-and-switch/mechanics check.
- **PRIMARY bucket**: E
- **Secondary bucket**: none
- **Justification**: Research was adequate and mechanics understood; the ensemble weighed it badly — over-weighting speculative reshuffle chatter and a loosely-coupled civil-service-turnover reference class while under-weighting the strong, in-bundle structural reason to expect NO (pre-election delay / election-contingent scope). A base-rate/over-confidence judgment failure, not a research gap.
- **Confidence**: high — NO is the base-rate default here, the best-reasoned model (Gemini) called it at 15%, and no decisive missing/stale/misattributed fact exists.
- **v2-addressable**: no (E–F)

---

### qid 42236 — how far Duke advances in 2026 March Madness (multiple choice)
- **Cohort**: miss
- **Miss summary**: Bot put ~57% on Final Four-or-better (incl. 21.9% on winning the title) and only 17.7% on Elite Eight; Duke lost in the Elite Eight (peer −11.1).
- **Decisive consideration**: This is a single-elimination outcome for a genuine co-favorite; the realized bin (Elite Eight, ~18% assigned) is a normal ~20% result for a #1 overall seed, so the miss is driven by which bin resolved rather than by any information failure.
- **Bundle check**: Fully adequate and current — Duke consensus #1 (AP/NET/KenPom), +315 title co-favorite, −155 Final Four (~61% implied), 28-2, top-4 offense/#1 defense. Nothing missing, stale, or misattributed; the bundle even flagged 2026 "significant parity" and 16-seed upset precedent.
- **Rationale check**: The models correctly cited #1-overall-seed base rates (gpt-5.1: "Elite Eight ... single most common exit point ~30%"; claude-4.6 base: E8 ~20%, champ ~24%) then shifted mass toward Final Four+/title for Duke's dominance. The ensemble landed close to a defensible #1-overall-seed base rate (E8 17.7% vs ~20% base; champ 21.9% vs market ~22%) — only mildly light on the realized bin.
- **PRIMARY bucket**: F
- **Secondary bucket**: E
- **Justification**: The distribution closely tracked a reasonable #1-overall-seed base rate and the betting market; a co-favorite losing in the Elite Eight (~18% assigned) is ordinary tournament variance, defensible ex ante. E is secondary because the ensemble did shave the base-rate-modal Elite Eight bin down (~20%→17.7%) and made "win the title" the modal bin, a mild deep-run over-concentration that cost the peer points.
- **Confidence**: medium — the E/F line is genuinely close; would tip to E if the community's distribution was much more Elite-Eight-weighted (the modest −11.1 peer suggests only a small calibration gap, supporting F).
- **v2-addressable**: no (E–F)
