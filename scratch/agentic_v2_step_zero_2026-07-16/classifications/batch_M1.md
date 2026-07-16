# Batch M1 classifications — gap-fill v2 step-zero miss audit

Classifier notes: judged from dossier contents only, no external lookups. Where a
call hinges on a world-fact I cannot verify (whether an asserted event was real),
confidence is lowered and the alternative bucket is named.

---

### qid 42304 — INES Level 3+ event listed by May 1, 2026
- **Cohort**: miss
- **Miss summary**: We said 81% YES (median of 65/85/80/82/81); it resolved NO — no 2026 INES Level 3+ event was listed on the IAEA database by May 1. Peer −115.9.
- **Decisive consideration**: The entire high forecast rested on one asserted "fact" — a Turkish Feb-2026 radiological over-exposure "rated INES Level 3." If that claim was fabricated, misattributed, or never made the IAEA listing, the base-rate answer (~8–25% per the forecasters' own priors) points to NO.
- **Bundle check**: The claim appears ONLY in the native-search section: *"As of March 9, 2026, there is one reported INES Level 3 event in 2026: a radiological over-exposure incident in Turkey ... rated INES Level 3 by authorities"* with a slug-shaped URL (`nucnet.org/news/turkish-radiation-incident-rated-ines-level-3`). The research summary escalated it to *"the condition for a 'Yes' resolution appears to have been met by the Turkish radiation incident (Level 3)."* Crucially, the AskNews articles — the primary provider, which surfaced numerous trivial Level 0/1 events (Golfech, Yangjiang, Paluel) — contain NO Turkish event at all, despite a Level 3 exposure affecting 10 people being far more newsworthy. Native search itself hedged: *"no direct IAEA confirmation was found in public sources."*
- **Rationale check**: All five anchored on the Turkish claim. gpt-5.1: *"≈0.95 × 0.85 ≈ 0.81 (81%)"* built entirely on the event being real and accepted. claude-4.6-opus dismissed the fabrication risk: *"the specificity (10 people, Turkey, February 2026) makes fabrication unlikely"* (exactly the wrong instinct for LLM hallucinations). gemini even named the failure mode as its blind spot: *"NucNet confused an incident involving a 'Category 3 source' with an 'INES Level 3 rating.'"*
- **PRIMARY bucket**: D
- **Secondary bucket**: E
- **Justification**: The specific unsupported/uncorroborated claim ("Turkish Feb-2026 INES Level 3 event") drove 65–85% forecasts; it surfaced in a single low-effort web-search pass with a hallucination-pattern URL and was contradicted by omission in the more reliable primary provider. Secondary E: models over-trusted a single uncorroborated source and over-bet the narrow fallback clause. (If the event was genuinely real but simply never IAEA-listed, this shifts toward C/E on resolution-mechanics — but the AskNews omission and the NO resolution favor the research-fabrication read.)
- **Confidence**: medium — I cannot verify whether a real Turkish Feb-2026 INES-3 event occurred. Confirming it was fabricated/misattributed (e.g., a Category-3 source over-exposure mislabeled "Level 3") would lock D; confirming it was real-but-unlisted would move it toward C/E.
- **v2-addressable**: yes (A–D)

---

### qid 43746 — Minions & Monsters opening weekend gross
- **Cohort**: miss
- **Miss summary**: Published distribution centered ~$88M with only 2.50% mass below the $75M floor; it resolved at $37.0M — far below the displayed range. Peer −77.8 vs baseline −20.5.
- **Decisive consideration**: The research already carried a central estimate below the Metaculus floor ($53–62M tracking; truth $37M is lower still, so the tracking was directionally correct in pointing below the range). The failure is that the submitted distribution did not place its mass where the evidence and the forecasters themselves said it belonged — below the open lower bound.
- **Bundle check**: The bundle repeatedly and correctly flagged the sub-floor tracking: bottom line *"$53 million–$62 million domestically over its ... 3-day opening weekend,"* explicitly *"below the Metaculus displayed lower bound of 75000000"* (gpt-5.5's own words). Research was adequate/good; this is not a research-comprehension gap.
- **Rationale check**: Forecasters KNEW the mass belonged below the floor. gpt-5.5 set `"below_min_expected": 0.78`; gemini wrote *"a very high probability (~65%) that the resolution value will fall below the $75M minimum bound"* and piled percentiles at `75000001`. claude-opus-4.6 set `below_min_expected: 0.55`. Yet the published aggregate reports *"2.50% chance of value below 75000000.0"* — the below-bound conviction (median ~0.55 across models) collapsed to ~0.025 in the submitted CDF.
- **PRIMARY bucket**: E
- **Secondary bucket**: none (both sub-drivers are E)
- **Justification**: Pipeline/formatting failure — mass the models expressed for the open lower bound (via `below_min_expected` and boundary-piled percentiles) was not propagated into the aggregate, exactly the "mass piled at an open bound the model couldn't express beyond" pattern. Rubric numeric rule also applies: bundle central estimate near/below truth, submitted distribution wasn't → E, not research. Co-driver (also E): some models, notably grok-4.3, lifted the median "to respect bounds" (median $90M) rather than trusting the sub-floor tracking.
- **Confidence**: high — the SUMMARY's 2.5%-below-floor vs the models' 0.25–0.78 `below_min_expected` is a direct, in-dossier contradiction pinning the distortion to aggregation/expression, not research.
- **v2-addressable**: no (E–F)

---

### qid 42112 — ATP Singles #1 ranking points on April 27, 2026
- **Cohort**: miss
- **Miss summary**: Published distribution centered ~$14,300 points with only 2.50% below the 13,550 floor; it resolved below_lower_bound (#1's points fell under 13,550). Peer −76.1 vs baseline −24.4.
- **Decisive consideration**: 13,550 was a fresh peak (Alcaraz just added the 2,000-pt AO title) heading into a heavy spring title-defense window under the 52-week rolling system — so the expected move was DOWN, likely below the floor. The bundle supported this; the submitted distribution instead centered above the floor with a rising median.
- **Bundle check**: Adequate. The bundle carried the defense-burden dynamic explicitly — *"Alcaraz must defend a significant number of points earned during the 2025 spring Masters 1000"* and the Tiscali *"vulnerable moment"* framing (Feb–Apr 2026). All the raw material to conclude "points likely drop below 13,550" was present.
- **Rationale check**: gemini derived the right answer from the bundle — *"the true resolution value is heavily favored to be out of bounds (below 13550)"* and noted the author's *"common error ... to set the lower bound at a player's current points"* — but its displayed percentiles were clamped to a pile at 13,550.1–13,552.5. gpt-5.1 said *"~50% of actual mass would sit below 13,550, which we're forced to 'pile up' at the lower bound."* Meanwhile gpt-5.2 and claude-opus-4.5 got the DIRECTION wrong, both centering at 14,650 on "modest upward drift." Published aggregate: *"2.50% chance of value below 13550.0."*
- **PRIMARY bucket**: E
- **Secondary bucket**: none (dual driver, both E)
- **Justification**: Same open-bound expression failure as 43746 — the two models that correctly read a below-floor resolution were "forced to pile up at the lower bound" and their conviction was squashed to 2.5% in the aggregate. Compounded by a genuine judgment split: half the ensemble (gpt-5.2, claude-opus-4.5) treated the current peak as a floor for further gains rather than recognizing the roll-off/defense drag. Research was adequate (gemini solved it from the bundle) so this is not v2-addressable.
- **Confidence**: high — "below_lower_bound" confirms the bound was open (out-of-range resolution was expressible), and the dossier shows both the correct in-bundle reasoning (gemini) and the pipeline squash (2.5% below floor).
- **v2-addressable**: no (E–F)

---

### qid 42855 — ≥1 UCL QF first leg with 4+ goals
- **Cohort**: miss
- **Miss summary**: We said 89% YES (median of 82/89/96/95/87); it resolved NO — all four April first legs finished with ≤3 goals. Peer −66.9, baseline −187.7.
- **Decisive consideration**: The correct base-rate direction was high-YES (all-four-low is a ~15–22% tail with clear historical precedent), so any reasonable forecast misses here; a genuinely uncommon outcome occurred and beat a defensible forecast. The excess loss vs peers traces to modest overconfidence above the base rate.
- **Bundle check**: Fully adequate — no missing/stale/hallucinated facts. Solid base-rate material: *"7 YES seasons / 9 total ≈ 77.8%,"* season average 3.52 goals/match, record R16 (68 goals), and the acknowledged counter-precedent *"In seasons like 2022–23 and 2018–19, all four QF first legs finished with 3 or fewer goals."*
- **Rationale check**: Reasoning was careful and mostly well-founded. gpt-5.2 stayed near base rate at 82% (*"1−(1−0.33)^4 ≈ 79%"*). But claude-4.6-opus and claude-opus-4.5 pushed to 96%/95% by extrapolating the record R16 tally they themselves flagged as mismatch-inflated — claude-4.6-opus: *"96% = 24:1 against"* (vs a true historical ~8:1). The whole field forecast high-YES (baseline −187.7 hammered everyone; peer gap implies peers clustered ~82% YES).
- **PRIMARY bucket**: F
- **Secondary bucket**: E
- **Justification**: A ~15–22% tail (all four first legs ≤3 goals, with real precedent in 2/9 seasons) resolved against a forecast that was reasonable and field-consistent ex ante; nothing in the bundle should have singled out this round as low-scoring. Secondary E: the median 89% (and the two Claude models at 95–96%) over-extrapolated the mismatch-inflated R16 scoring above the ~78–82% base rate, making us ~7 pts hotter than the calibrated peer consensus and amplifying the loss.
- **Confidence**: medium — E-primary is defensible if one weights the peer-gap/overconfidence more heavily; I land on F because the correct base-rate direction was high-YES, the outcome is a genuine tail, and the entire field missed together.
- **v2-addressable**: no (E–F)
