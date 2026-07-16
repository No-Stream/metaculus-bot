# Batch M6 classifications — gap-fill v2 step-zero miss audit

Classifier: classifier-M6. Judged from dossier only; no web lookups. Where a decisive
fact could not be verified from the dossier, confidence is lowered and the gap is stated.

---

### qid 42687 — U.S. ground invasion of Iran before May 2026
- **Cohort**: miss
- **Miss summary**: Bot published 30% YES; question resolved NO (no ≥100-troop, >5-day
  ground presence by May 1). Peer −11.1, so peers were far lower on YES.
- **Decisive consideration**: Whether to weight the loose Polymarket "US forces enter Iran
  by Apr 30" line (59%) and the escalation narrative, vs. the strict 100-troop/5-day
  threshold plus 45 years of U.S. restraint toward invading Iran.
- **Bundle check**: Research was adequate and even self-correcting. It carried the strict
  criteria, the strong base-rate case ("The US has never conducted a ground invasion of
  Iran"), Trump's denial ("I'm not putting troops anywhere"), 74% public opposition, AND
  the caveat that Polymarket "resolves Yes if active US personnel deliberately enter
  Iranian land" — i.e., a looser event than the question. Nothing decisive was missing;
  the outcome (no invasion) was a future event research could not have known.
- **Rationale check**: The two high forecasters herded on the market. Forecaster 4
  (claude-opus-4.5, 47%): "Polymarket odds (59%) reflect informed aggregated judgment …
  I'm moving to 47%." Forecaster 1 (43%) leaned on "prediction-market pricing high → crowd
  belief in near-term entry." The lower models correctly discounted it — Forecaster 5
  (gemini, 13%): "Prediction markets often conflate minor incursions … with sustained
  operations." The median (30%) was dragged up by the market-herders.
- **PRIMARY bucket**: E
- **Secondary bucket**: none
- **Justification**: Judgment failure — herding on a prediction market that was (a) wrong
  and (b) measuring a looser event, plus over-weighting the escalation narrative against a
  strict threshold and a strong restraint base rate. Rubric is explicit: "Herding on
  prediction markets that were themselves wrong = E."
- **Confidence**: high — the outcome was a future non-event, research was adequate, and the
  peer score shows a much lower forecast was both available and correct.
- **v2-addressable**: no (E)

---

### qid 42299 — S&P 500 close on April 30, 2026 (numeric)
- **Cohort**: miss
- **Miss summary**: Aggregated median landed near ~14,300… (equity: median ~6,560 with a
  negative skew); index resolved at **7209.01**, near the ~80th percentile of the bot's
  distribution. Peer −11.0.
- **Decisive consideration**: Whether to keep the median at/above spot (~6,740, matching
  equity drift + Wall Street targets) or push it below spot with a heavy negative skew in
  reaction to the March 6 volatility shock (VIX ~29.5, Hormuz/oil, weak payrolls).
- **Bundle check**: The bundle contained the anchor that bracketed the truth — "Wall Street
  strategists generally forecast S&P 500 year-end 2026 targets in the 7,200-8,100 range …
  average ~7,269." Spot was ~6,740. The right central tendency was in the bundle; the
  models discounted it for the transient shock.
- **Rationale check**: Models explicitly skewed down. Forecaster 3 (claude-4.6-opus):
  "I'm moving the median down to ~6,400 (about −5%)." Forecaster 5 (gemini): "decisively
  moving my median downward to ~6,580 and dramatically fattening the left tail." Forecaster
  1 (gpt-5.2) even noted the reversal risk — "markets historically recover from
  geopolitical shocks faster than expected" — then still set the median at 6,700 below
  spot. The market did exactly that: recovered to 7,209.
- **PRIMARY bucket**: E
- **Secondary bucket**: none
- **Justification**: Judgment failure — over-reacting to a salient transient shock and
  applying a below-spot median with negative skew, neglecting positive equity drift and the
  bundle's own Wall Street targets (~7,269) that bracketed the outcome. Rubric: "if the
  bundle's central estimate was near the truth but the submitted distribution wasn't, that
  is E (judgment or pipeline)." Genuine judgment, not a pipeline artifact.
- **Confidence**: high — bundle had the near-truth anchor; the models reasoned their way to
  a lower, negatively-skewed distribution.
- **v2-addressable**: no (E)

---

### qid 42591 — UNMISS mandate renewal with ≥1 abstention before Apr 30, 2026 (binary)
- **Cohort**: miss
- **Miss summary**: Bot published 63% YES; resolved NO. Peer −9.6, so peers were notably
  lower on YES.
- **Decisive consideration**: The weight on the "on-time substantive renewal draws an
  abstention" path (2/2 recent renewals had abstentions) vs. the "next resolution before
  the deadline is a unanimous technical rollover, or the substantive vote slips past Apr
  30" path — both of which resolve NO.
- **Bundle check**: The single closest precedent was in the bundle and pointed hard to NO —
  "In 2025, a technical rollover (Resolution 2778) was required on April 30 to extend the
  mandate by nine days … Adopted unanimously (15-0-0)," with the substantive Res. 2779
  slipping to May 8. Same mission, same April 30 expiry, prior year → the pre-deadline
  resolution was a unanimous rollover. Research was adequate; the vote itself was a future
  event.
- **Rationale check**: Models understood the mechanics but under-weighted the precedent.
  Forecaster 5 (gemini, 63%): "there is a substantial (~35%) risk that the Council requires
  a unanimous technical rollover or pushes the vote to the last minute." Forecaster 4
  (claude-opus-4.5, 82%): "Technical rollovers are rare; substantive renewal is
  overwhelmingly more likely" — the outlier that pulled the median up. Even the bearish
  reasoners set only ~35–47% on the NO paths that actually obtained.
- **PRIMARY bucket**: E
- **Secondary bucket**: none
- **Justification**: Judgment failure — leaning YES (~63%) against the dominant same-mission
  2025 precedent (unanimous rollover before the deadline + substantive vote in May, both
  → NO). Mechanics were read correctly (not C); the outcome was future (not A). Peer −9.6
  shows a lower, better-calibrated forecast was available.
- **Confidence**: medium — clearly not a research-comprehension failure (research adequate,
  mechanics understood), but E-vs-F is a judgment call; the strong 2025 precedent plus the
  peer score tilt it to E over F (I would not defend 63% ex ante).
- **v2-addressable**: no (E)

---

### qid 42107 — ATP Singles #1 ranking points on March 30, 2026 (numeric)
- **Cohort**: miss
- **Miss summary**: Aggregated median ~14,300 (all five forecasters clustered ~14,190–
  14,400); resolved at **13,590**, roughly the 3rd–5th percentile of the bot's distribution
  and only ~40 above the 13,550 annulment floor. Peer −9.4.
- **Decisive consideration**: Alcaraz's true March points-to-defend. Every forecaster used
  ~410; the whole distribution's height rests on the arithmetic "13,550 − 410 + up to
  2,000." If the real defense was materially higher, the distribution was shifted up by
  that amount — and the ~700-point overshoot matches the discrepancy below.
- **Bundle check**: The bundle contains a direct, unreconciled contradiction on the defense
  figure. The synthesis asserts "Carlos Alcaraz: Defends **410 points** in March
  (specifically 400 from a 2025 Indian Wells semifinal)," and the native search adds "Miami
  defenses **unspecified in sources**." But the Virgilio Sport source states Alcaraz has
  "**1,010 points in expiration** … at the Masters 1000 tournaments in Indian Wells and
  Miami." The summary committed to 410 (implicitly zeroing Miami) — a total-defense claim
  the sources do not support and one source contradicts.
- **Rationale check**: All five leaned on 410 and never engaged the 1,010 figure or the
  flagged Miami gap. Forecaster 2 (gpt-5.1): "Alcaraz max around 15,140 … (13,550 − ~410 +
  2,000)." Forecaster 5 (gemini): "his absolute maximum score … is 13,550 − 410 + 2,000 =
  15,140," and treated the floor as trimming all downside: "the annulment rules trim the
  entire downside, leaving an asymmetric upside." That reasoning collapses if the defense
  was ~1,010 (floor-crossing then requires a strong Sunshine Double, not a weak one).
- **PRIMARY bucket**: D
- **Secondary bucket**: E
- **Justification**: The research synthesis asserted an unsupported total March
  points-to-defend (~410) that contradicted an in-bundle source (Virgilio's 1,010) and the
  explicit "Miami defenses unspecified" flag; all forecasters anchored on 410 and set
  distributions ~600–700 points too high — matching the miss (median ~14,300 vs. outcome
  13,590). Under the ~1,010-defense reading, 13,590 implies Alcaraz earned ~1,050 in 2026
  (a normal-to-good result), not a collapse. Secondary E: the distributions were narrow /
  the left tail thin regardless, and if the defense truly was ~410 the miss is instead a
  judgment/sports-surprise (Alcaraz underperforming the Sunshine Double).
- **Confidence**: medium — I cannot verify from the dossier whether the true defense was
  ~410 or ~1,010 (Alcaraz's actual 2025 Miami result and 2026 IW/Miami finishes are not in
  the dossier). What would change the call: if the real defense was genuinely ~410 and
  Alcaraz simply lost early at both Masters, this flips to E (thin-tail/overconfidence) or
  F (genuine upset). The magnitude match between the 410-vs-1,010 gap and the miss, plus
  the annulment floor being set exactly at the March-2 level, tilt me to D.
- **v2-addressable**: yes (D) — a better research stage that resolved Alcaraz's exact
  2025 Miami result / total March defense would have shifted the distribution toward the
  outcome.
