# Batch C1 — Control cohort classifications

Control-cohort audit: five questions the bot scored WELL on. For each, I applied the
same critical reading as the miss classifiers and asked whether a latent A/B/C/D
research failure is hiding inside a good outcome (i.e. whether our miss buckets are
partly hindsight). Verdict up front: four are cleanly clean; one (43131) is clean with
a genuine-but-non-biting stale-data observation in its raw research block.

---

### qid 43130 — TSA US airline passenger volume, Apr 6–12, 2026 (numeric)
- **Cohort**: control
- **Miss summary**: Bot median 17,700,000; resolution 17,662,948 — essentially exact (~0.2% high). Peer +53.8.
- **Decisive consideration**: The week was half-observed at forecast time; the 4-day partial total was the load-bearing anchor.
- **Bundle check**: Bundle contained the running actuals — *"Cumulative (4 days): 9,882,898"* (Apr 6–9), plus TSA's Fri–Sun not yet posted. The most decisive fact was present and fresh.
- **Rationale check**: The forecaster anchored correctly — *"All models correctly anchor on the 9,882,898 known 4-day total. The critical remaining question is Fri-Sun volume"* — and reasoned about day-of-week ratios. Sound use of the bundle.
- **PRIMARY bucket**: clean
- **Secondary bucket**: none
- **Justification**: Decisive fact (4-day partial actuals) present and fresh; mechanics (rolling 7-day total) understood; median landed on the number. No gap, staleness, misread, or hallucination.
- **Confidence**: high (would only change if the partial-total figures in the bundle were themselves wrong, which the near-exact hit rules out).
- **v2-addressable**: no (clean)

---

### qid 43729 — Bayer (BAYRY) day-over-day % change after *Monsanto v. Durnell* decision (numeric)
- **Cohort**: control
- **Miss summary**: Aggregate bimodal forecast (favorable mode centered ~+11–14%, adverse mode ~-13–15%); resolution +16.98%, i.e. a Bayer-favorable ruling landed in/just above the upper mode. Peer +50.4.
- **Decisive consideration**: A genuinely forward-looking binary event — the SCOTUS merits decision had not been issued at forecast time (May 29, 2026; decision expected late June/July).
- **Bundle check**: Exceptionally thorough — historical event-study analogs (*"cert grant … +6.08%"*, *"Solicitor General support … +11.97%"*, *"oral argument … -4.46%"*), analyst win/loss scenario targets, thin Polymarket (~57% win), and June options IV (~29.5%). The one fact that would flip the miss (the actual ruling/outcome) did not exist pre-submission.
- **Rationale check**: gpt-5.4 built an explicit state-price model (*"the full-resolution jump is approximately: Bayer win: … ≈ +13%"*); claude-opus-4.7 and others built bimodal mixtures with wide, justified tails. All ran the BAYRY-close-to-close bait-and-switch check. Reasoning was disciplined.
- **PRIMARY bucket**: clean
- **Secondary bucket**: none
- **Justification**: The outcome was unknowable ex ante (future SCOTUS ruling → F-territory, not a research gap); the bundle was adequate-to-excellent and correctly used, and the actual +17% fell inside the models' favorable mode. Placing the favorable-mode center a couple points low is calibration, not a research failure.
- **Confidence**: high (would change only if a decision or leaked outcome had been publicly available before May 29, for which the dossier shows the opposite — *"No decision has been released in this case since May 29, 2026, or at any point prior"*).
- **v2-addressable**: no (clean)

---

### qid 43745 — June 2026 UMich Consumer Sentiment > May's 44.8? (binary)
- **Cohort**: control
- **Miss summary**: Bot 69.5% Yes; resolution Yes. Correctly leaned Yes with room to spare. Peer +42.1.
- **Decisive consideration**: Whether a record-low May (44.8) would tick up in June given easing gas prices — the threshold and its direction.
- **Bundle check**: Bundle nailed the threshold — *"May 2026 final UMich Consumer Sentiment = 44.8"* — and, crucially, **caught the stale-data trap**: *"Some early/secondary items mistakenly list 49.8 as the latest value … This appears to be a stale-data issue from an earlier April reading. Use 44.8 as authoritative."* Also carried consensus forecasts (46.0/46.6), gas-price easing, and a directly-checked resolution PDF. The FRED block did show a stale 49.8 (April), but it was explicitly overridden.
- **Rationale check**: gpt-5.4 anchored on mean-reversion base rates and correctly noted preliminary-print timing (*"June UMich sentiment value, with preliminary sufficient, must be greater than 44.8"*); claude-opus-4.7 weighed the reversing gas-price driver. Threshold and preliminary-resolution mechanics understood.
- **PRIMARY bucket**: clean
- **Secondary bucket**: none
- **Justification**: This is the strongest evidence *against* hindsight bias: the stale FRED 49.8 is exactly the kind of staleness a B-miss flags, and the pipeline **caught and corrected it** rather than being bitten. Threshold correct, mechanics correct, forecast well-founded and directionally right.
- **Confidence**: high.
- **v2-addressable**: no (clean)

---

### qid 43129 — TSA US airline passenger volume, Apr 27–May 3, 2026 (numeric)
- **Cohort**: control
- **Miss summary**: Bot median 17,200,000; resolution 17,000,271 (~1.2% high). Peer +22.3.
- **Decisive consideration**: A fully-future week (forecast Apr 11, target Apr 27–May 3) with no partial data — the run-rate and seasonal-taper judgment carried it.
- **Bundle check**: Fresh recent actuals present (*"Latest available TSA daily totals (as of April 9, 2026): April 9: 2,691,308 …"*), plus A4A 2.8M/day and day-of-week trends. Native search correctly noted the target week *"is in the future … TSA … does not issue public forecasts"* — no forecastable fact was missing (the actual didn't exist yet).
- **Rationale check**: Forecaster used *"recent TSA daily average ~2.47M/day (→~17.3M/week)"* with *"mild downward adjustment … for post-Easter normalization,"* landing at 17.2M. Grounded, appropriately-tailed.
- **PRIMARY bucket**: clean
- **Secondary bucket**: none
- **Justification**: Target-week actual was future-dated (not an A-miss), the bundle's run-rate anchor was fresh (through Apr 9), and the median landed within ~1% of truth. No research failure.
- **Confidence**: high.
- **v2-addressable**: no (clean)

---

### qid 43131 — TSA US airline passenger volume, Apr 20–26, 2026 (numeric)
- **Cohort**: control
- **Miss summary**: Bot median 17,500,000; resolution 17,237,442 (~1.5% high). Peer +25.1.
- **Decisive consideration**: A fully-future week; choice of reference class (2024 analog vs A4A pre-crisis forecast) was the crux.
- **Bundle check**: Two things. (1) The reference-class reasoning was excellent and well-supported. (2) But the **displayed native-search block is staler than its siblings**: it cites TSA actuals only through *"April 2: 2,710,611; April 1: 2,360,739; March 31: 2,154,213,"* whereas 43130/43129 — submitted ~1.5h *earlier* the same day — carried actuals through April 9. That is a genuine present-but-stale slice of the kind a B-miss flags.
- **Rationale check**: The forecaster nonetheless reasoned with the *fresher* data — *"Post-Easter daily TSA data (Apr 7–9): averaging ~2.39M/day, with Apr 9 YoY at -6.64%"* — and made a sharp calendar catch: *"2025 Easter was April 20, making direct YoY comparison for this week impossible … The 2024 comparison is more reliable"* (2024 analog ~17.24M, which is almost exactly the resolution).
- **PRIMARY bucket**: clean
- **Secondary bucket**: B (latent, non-biting)
- **Justification**: The raw native-search block was stale (through Apr 2 vs Apr 9 available), a real latent-B-flavored gap — but it did **not** bite: the forecaster demonstrably used Apr 7–9 figures and a well-chosen 2024 reference class, and the median landed within ~1.5% of truth. Effective research was adequate; the staleness lived only in one displayed provider slice.
- **Confidence**: medium on "clean" (the raw-block staleness is real; I judge it non-biting because the reasoning shows fresher data was in hand). Would flip toward latent-B-that-mattered only if the sub-models had in fact been confined to the stale Apr-2 slice — the meta-forecaster's Apr 7–9 citation is evidence they were not.
- **v2-addressable**: no as-scored (clean outcome); the stale-slice observation is the kind of thing a fresher-fetch agentic stage would tidy up, but it changed nothing here.

---

## Control-cohort takeaway
Four of five are unambiguously clean, and one of those (43745) shows the pipeline *actively
catching* a stale-data trap rather than being fooled by it — direct evidence against
hindsight bias in the B bucket. The only latent flag is 43131's staler raw research slice,
and it was non-biting because the forecaster had the fresher figures anyway. Net: on this
control sample the A/B/C/D buckets look largely real, not hindsight artifacts — good
outcomes here mostly had adequate, fresh, correctly-used bundles.
