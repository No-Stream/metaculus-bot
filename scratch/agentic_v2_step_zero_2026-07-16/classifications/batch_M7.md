# Batch M7 classifications — gap-fill v2 step-zero miss audit

Classifier: classifier-M7. Judged from dossier contents only; no external lookups.

---

### qid 43141 — FrontierMath Open Problems solved by AI, index count on May 1 2026 (MC)
- **Cohort**: miss
- **Miss summary**: Bot put 55% on "1 or fewer" (the correct outcome) but hedged 45% onto "2" (32%) and "greater than 2" (13%); the community was far more confident in "1 or fewer", so the bot scored peer -7.7 for over-hedging the modal, correct answer.
- **Decisive consideration**: Whether any *new* AI solve would post on the index in the ~20-day window (Apr 11 close → May 1 resolution); the base rate strongly said no.
- **Bundle check**: The bundle had the decisive facts. Native search: "No reliable sources report additional AI solves on this index as of April 11, 2026" and "The page's changelog last updated on March 5, 2026 ... with no further changes noted." The only pipeline item was partial Small Diophantine progress ("2–3 of 9 required equations", far from a full solve).
- **Rationale check**: The meta-analysis notes Model 6 "contributes a useful Poisson rate calculation (λ≈0.25/month → ~15% chance of ≥1 new solve in 20 days)" and that Models 2/5 were "carefully distinguishing 'displayed index count' from underlying capability." Yet the ensemble put 45% on ≥1 new solve — triple its own ~15% arithmetic — and landed 55/32/13, "push[ing] slightly higher on A" but keeping "meaningful mass on B" on the strength of the acceleration narrative.
- **PRIMARY bucket**: E
- **Secondary bucket**: D
- **Justification**: Research was adequate and even contained the base-rate math (Poisson → ~15% for a new solve). The models overweighted the "high-velocity AI math" narrative and the partial Small Diophantine progress, refusing to trust their own ~15% arithmetic — overconfidence / base-rate under-trust. Secondary D: the "Summary for Forecaster" conflated Erdős-problem velocity ("11 AI-driven solves ... suggests a high-velocity environment ... immediately preceding the May 1 deadline") with the distinct 15-problem Open Problems index, nudging forecasters toward more solves even though the raw search cleanly separated the two.
- **Confidence**: high (bundle plainly had the "stuck at 1, nothing in the pipeline" facts and the base-rate calc; the miss is over-hedging).
- **v2-addressable**: no (E–F)

---

### qid 43728 — SCOTUS rule in favor of Monsanto in Monsanto v. Durnell (binary)
- **Cohort**: miss
- **Miss summary**: Bot said 59.5% Yes (six models 53–63%); the case resolved Yes; the community was much more confident, so the bot scored peer -7.5 for being underconfident on the correct side.
- **Decisive consideration**: The strong structural prior — cert granted on the petitioner's split-resolving question + Solicitor General supporting the petitioner → reverse/vacate ~75–80%, with vacate/remand *also* counting as Yes.
- **Bundle check**: The bundle stated the base rate explicitly ("When U.S. Solicitor General files amicus supporting petitioner: petitioner wins ~75-80% of the time"), flagged the broad Yes resolution ("The Metaculus resolution counts reversal/vacatur/remand as Yes"), and carried the thin market signals ("Polymarket ... about 55% ... only $272 volume", "FantasySCOTUS ... 51% for reversal").
- **Rationale check**: The Computed-quantities blocks show the models discarding their own arithmetic: Forecaster 3's Beta-binomial from the SG-support reference class gave posterior mean 0.747 [0.691, 0.800] but it declared 0.63 (Δ = -0.117); Forecaster 5's gave 0.694 but declared 0.58 (Δ = -0.114). Every model updated *down* from a ~0.67–0.75 prior toward the ~55% crowd, citing "oral argument appeared genuinely mixed" and the "$272 volume" Polymarket.
- **PRIMARY bucket**: E
- **Secondary bucket**: none
- **Justification**: The models herded on a thin ($272) prediction market plus noisy oral-argument "divided" reads and abandoned a strong, ultimately-correct base rate (cert + SG support + broad Yes criterion). This is the rubric's canonical "herding on prediction markets that were themselves wrong = E," compounded by refusing to trust their own base-rate/Beta-binomial arithmetic.
- **Confidence**: high (the correct base rate was in the bundle and in the models' own computations; they knocked it down on argument vibes and a near-empty market).
- **v2-addressable**: no (E–F)

---

### qid 42740 — Highest TSMC monthly net revenue among Jan/Feb/Mar 2026 (MC)
- **Cohort**: miss
- **Miss summary**: Bot said January 62% / February 1% / March 37%; the question resolved March 2026; the bot put only 37% on the correct answer and scored peer -7.0.
- **Decisive consideration**: Whether March revenue would exceed January's record NT$401.255B; seasonality (post-Lunar-New-Year rebound, more working days, quarter-end shipment concentration), the analyst/MOEA signals, and TSMC's guidance-beat tendency all pointed to March.
- **Bundle check**: The bundle carried the March-favoring signals: UDN analysts projecting March "surge above 400 billion NTD" (+26% MoM, +40% YoY), MOEA March export orders "+38.4% to +42% YoY", and the fewer-working-days-in-February / March-rebound seasonality. It lacked TSMC's actual historical monthly revenue series (2021–2025), so the models argued Jan-vs-March seasonality from memory and contradicted each other.
- **Rationale check**: Forecaster 1's own outside-view anchor was "March 55% / January 40%"; Forecaster 2 anchored "Mar highest ~60%"; Forecaster 3 landed March 52% (the correct side). But the ensemble median flipped to January because most models over-anchored on the guidance *midpoint* (US$35.2B → March ~NT$393B < January). Forecaster 2 even computed "MOEA-only framing → ~44% chance March beats January" and "macro/analyst → 40–45%," then blended it down to 35% by heavily weighting the midpoint.
- **PRIMARY bucket**: E
- **Secondary bucket**: A
- **Justification**: The models discounted their own March-favoring seasonality anchor and the present analyst/MOEA signals, treating the conservative guidance midpoint as March's expected value on a quarter that was ramping (end-of-quarter concentration + guidance-beat history point above midpoint). Secondary A: TSMC's historical monthly revenue series was absent from the bundle, so the Jan-vs-March seasonality — a decisive input — was resolved by contradictory model guesses rather than a pulled fact; supplying it could plausibly have shifted the forecast.
- **Confidence**: medium (genuine ex-ante uncertainty and the missing historical series both cut against a clean call; but the present bundle signals were enough to lean March, and 3/5 models' own anchors did).
- **v2-addressable**: no (E–F). Note: the secondary-A missing-series gap *is* the one v2-relevant thread here — a research stage that pulled TSMC's 2021–2025 monthly figures would have removed the seasonality guesswork.

---

### qid 42869 — March 2026 advance goods trade balance, signed $B (numeric)
- **Cohort**: miss
- **Miss summary**: The five forecasters' medians clustered at -82 to -92 (resolution -87.9, essentially dead-center), but their 95% intervals spanned roughly -150 to -45; the aggregate diffuse distribution scored peer -6.7 despite a near-perfect central estimate.
- **Decisive consideration**: The *tightness* of the distribution, not its center — the advance goods balance is a predictable monthly series and -87.9 sat squarely between the recent prints.
- **Bundle check**: The bundle had tight anchors that bracket -87.9 closely: "January 2026 advance goods balance: -81.8" and "December 2025 ... goods deficit ... -98.5 billion" (initial -86). Native search confirmed no March-specific preview existed, so recent-months persistence was the right frame.
- **Rationale check**: Every forecaster nailed the center (P50 ≈ -82 to -92) then deliberately fattened the tails: Forecaster 1 "intentionally using a wide 95% interval to avoid the common overconfidence failure mode on noisy monthly macro releases" (P2.5 = -170); Forecaster 2 placed "2.5th percentile near -125B ... substantially wider than a naïve historical interval, to reduce overconfidence"; Forecaster 5 "maintaining a wide distribution to reflect extreme geopolitical volatility." A -150 to -170 goods deficit would be roughly double any value in the series.
- **PRIMARY bucket**: E
- **Secondary bucket**: none
- **Justification**: The bundle's central estimate was near-exact; the miss is bad tail width — deliberate over-widening justified by an "avoid overconfidence" heuristic and the Iran/Hormuz/oil/DXY volatility narrative, applied to a series that rarely moves more than ~$10–15B month to month. This is the rubric's explicit "bad tail width on numerics" / "central estimate near the truth but the submitted distribution wasn't → E (judgment)."
- **Confidence**: high (median dead-center, so this cannot be a research or missing-fact failure; it is purely calibration of spread). Not a pipeline artifact — no open-bound piling; the width was chosen in the rationales.
- **v2-addressable**: no (E–F)
