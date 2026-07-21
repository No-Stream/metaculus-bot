# Batch M8 — miss classifications

Classifier: batch M8. All four are misses (no control questions in this batch).

---

### qid 42865 — OFAC Belarus-related sanctions action before May 1, 2026 (binary)
- **Cohort**: miss
- **Miss summary**: Bot said 58% YES (median of 26/35/58/72/85); question resolved NO — OFAC published no Belarus-tagged action dated April 1–30, 2026. Peer -6.5.
- **Decisive consideration**: Whether OFAC would issue *any* Belarus-tagged action (FAQ/GL/delisting) in the specific one-month window; the modal outcome, given a ~25–30% monthly base rate and a just-completed late-March package, was "no action."
- **Bundle check**: The bundle carried the decisive negative evidence, not a gap. It states "No US Treasury announcements of planned Belarus sanctions before May 1, 2026," that "OFAC's 2026 priorities [are] on Venezuela, Iran, Russia—not new Belarus actions," and "zero Belarus SDN additions in 2025-2026; only removals and GLs." The big package (delistings + GL 14) was dated March 26, before the window.
- **Rationale check**: The two high forecasters over-weighted a "cleanup follow-on" hypothesis — claude-4.6-opus (85%): "Major actions almost always generate follow-up FAQs/guidance within weeks"; claude-opus-4.5 (72%) leaned on the same. The two disciplined forecasters read the *same* bundle correctly — gemini (26%): the "'desk-clearing' at the end of Q1 (March 26, March 31) strongly suggests the relevant OFAC desk has completed its major tasks," and gpt-5.1 (35%) anchored on the ~27% empirical monthly rate. The median got dragged up to 58% by the cleanup-follow-on overweight.
- **PRIMARY bucket**: E
- **Secondary bucket**: none
- **Justification**: Research was adequate (it explicitly documented the easing trend, priorities-elsewhere, and no planned action); the miss was over-weighting the "post-package administrative follow-on" hypothesis over the base rate. A correct read of the same bundle (26–35%) was available and would have flipped this toward the resolution.
- **Confidence**: high — would only change if a specific scheduled/announced April Belarus action had existed pre-submission and been omitted (that would be A), but the bundle affirmatively says none was planned.
- **v2-addressable**: no (E)

---

### qid 42688 — Strait of Hormuz shipping back to normal before April 27, 2026 (binary)
- **Cohort**: miss
- **Miss summary**: Bot said 28% YES (median of 7/12/28/30/40); question resolved NO — no single day reached ≥45 transits. Peer -6.5. The bot was directionally right (72% on the correct outcome) but far less confident than peers.
- **Decisive consideration**: Whether a single day would hit ≥45 transits before April 27 while an active war continued; with traffic flat at ~6/day, insurers withdrawn, and no ceasefire, the strong modal outcome was "stays well below 45."
- **Bundle check**: Fully adequate. It reports traffic "averages ~6 ships/day; only one day hit 9–11," "insurance re-entry is a prerequisite," "No reliable data... indicates traffic will return to normal... before April 27," and gives the Polymarket line at "26% Yes (74% No)" for a *harder* threshold (7-day avg ≥60).
- **Rationale check**: The three higher forecasters anchored upward off the prediction market — gpt-5.1 (40%): "market data implies our event should be notably more likely than 26%, plausibly in the high-30s or low-40s," and gemini (28%) leaned on the "single-day spike... backlog-clearing" story. The two disciplined forecasters read the same facts correctly — claude-4.6-opus (7%): "No historical chokepoint disruption of this severity... has recovered... within 5 weeks while hostilities continued," and claude-opus-4.5 (12%) similarly. The median settled at 28%, pulled up by the market-anchoring and single-day-spike overweight.
- **PRIMARY bucket**: E
- **Secondary bucket**: none
- **Justification**: Herding upward on a prediction market (that itself resolved NO) plus over-weighting the one-day-spike scenario, against a bundle that argued for ~7–12%. Per the rubric, market-herding when the info was present is E; and the well-centered forecasters prove the correct read was available.
- **Confidence**: high — the research was complete and the disciplined forecasters demonstrate the miss was calibration, not information.
- **v2-addressable**: no (E)

---

### qid 43656 — Armenia Civil Contract seat share, June 2026 (multiple choice)
- **Cohort**: miss
- **Miss summary**: Bot's modal bucket was "52–58%" at 33%, with only 23% on the resolving bucket "58–64%"; question resolved "At least 58% but less than 64%." Peer -5.9. The bot centered one bucket too low.
- **Decisive consideration**: How far Armenia's wasted-vote/threshold amplification would push Civil Contract's *seat* share above its low-30s *vote* share — the mechanic that produced 66% seats from 54% votes in 2021.
- **Bundle check**: Adequate, arguably good — the outcome was a future election (June 7) so no result could be in the bundle, but the gap-fill/targeted-research memo pointed almost exactly at the answer: "I would forecast Civil Contract (CC) at roughly 60–67% of National Assembly seats, with a point estimate around 64%," and flagged the pivotal 8% alliance threshold. Polling (IRI 32%, EVN 32.5%/40.7–51.1%, Gallup 26.7%), the 52% floor, the two-thirds cap, and the 2021 benchmark were all present.
- **Rationale check**: The base models split — claude-opus-4.6 correctly put its mode on "58–64%" (28%) and grok on it (25%), matching the research memo, but gpt-5.4 (37% on 52–58%), gpt-5.5 (32% on 52–58%), and claude-opus-4.7 (36% on 52–58%) anchored on the 52% "stable-majority floor" as the natural bunching point. gpt-5.4: "largest party + threshold waste + seat-allocation bias... = final share most often in the 52%-58% bucket." The stacker averaged to a 52–58% mode, under-weighting the amplification the research (and 2021) supported.
- **PRIMARY bucket**: E
- **Secondary bucket**: F (the exact threshold-clearing outcome was genuinely knife-edge and hard to pin from divergent polls)
- **Justification**: Research was adequate and its own memo pointed at ~64%; the ensemble under-weighted the wasted-vote amplification and over-anchored on the 52% floor, placing its mode one bucket below the resolution. This is bad weighing of adequate (even correct-pointing) research, not a research gap.
- **Confidence**: medium-high — would shift toward F if the outcome hinged on a genuinely unpredictable late shock rather than the amplification the research already flagged; from the dossier it tracks the memo's estimate, so E.
- **v2-addressable**: no (E)

---

### qid 43652 — Crude Oil vs S&P 500 futures relative return, Jun 1–Jun 12 (numeric)
- **Cohort**: miss
- **Miss summary**: Bot's distribution was centered at ~0 pp (51.47% below 0) with an 80% interval roughly [-10.5, +11] pp; question resolved at -0.7225 pp — essentially dead-center. Peer -5.0, baseline +20.0. The miss vs peers is width, not location.
- **Decisive consideration**: How wide the spread distribution should be — the realized value landed almost exactly on the bot's median, so the score was determined entirely by how much mass the bot concentrated near zero.
- **Bundle check**: Excellent and directly on point — it supplied the resolution-relevant option-implied vols (CL ~59% → ±10.7% 12-day move; ES ~13.11% → ±2.38%), correct contract roll (CL=F→CLN26, ES=F→ESM26), the P0 closes, and a historical spread base-rate table. Nothing decisive was missing or stale.
- **Rationale check**: Every forecaster centered correctly near zero and flagged FORECASTABILITY: LOW, then set width off implied vol — gpt-5.4: "spread sigma near 11 pp... a wide distribution is warranted"; claude-opus-4.6: "Spread σ (assuming ρ ≈ 0): √(12² + 2.6²) ≈ 12.3 pp." The center was right (~0, actual -0.72); the distribution was simply wider than peers who concentrated near zero and scored better.
- **PRIMARY bucket**: E
- **Secondary bucket**: F
- **Justification**: Per the rubric's numeric rule — the bundle's central estimate was near the truth but the submitted distribution wasn't (too wide) — this is E (judgment/calibration: it didn't discount the variance-risk-premium in implied vol). Secondary F because an ~11pp width off genuine 59% implied vol was defensible ex ante and was beaten by a low-variance central realization.
- **Confidence**: medium — the E/F line is genuinely close; it turns on whether "too wide given implied vol" is a calibration error or a reasonable ex-ante choice beaten by one draw. The rubric's numeric guidance and the exact-center realization push to E.
- **v2-addressable**: no (E)

---

## Summary

| qid | primary | secondary | confidence | one-line justification |
|---|---|---|---|---|
| 42865 | E | none | high | Bundle had the negative evidence (no planned action, priorities elsewhere); models over-weighted a "post-package FAQ/GL cleanup" hypothesis over the base rate. |
| 42688 | E | none | high | Directionally right but under-confident; herded up off a prediction market (harder threshold, itself NO) + over-weighted a one-day backlog spike vs an adequate bundle arguing 7–12%. |
| 43656 | E | F | medium-high | Research (and its own memo) pointed at ~64% seats; ensemble anchored on the 52% floor bucket, under-weighting the wasted-vote amplification, landing one bucket below the resolution. |
| 43652 | E | F | medium | Center was near-exact (median ~0, actual -0.72); distribution too wide vs peers — didn't discount the variance-risk-premium in implied vol. |

All four M8 misses are judgment/calibration failures (E), not research-comprehension failures — none is addressable by a better agentic research stage. Notably, in 43656 and 43652 the research bundle actually pointed at (or centered on) the correct answer, and the models under-used it.
