# Batch C2 — Control cohort classifications

Cohort = control (questions the bot scored WELL on). For each, the reading asks:
would the same critical lens that we apply to misses flag a research gap (A/B/C/D) here
anyway? PRIMARY bucket is "clean" or "latent <letter>".

Headline: 4 of 5 are clean with high confidence. The 5th (42116) is clean of any
research-bundle defect, but critical reading surfaces a genuine E-flavored
PIPELINE/JUDGMENT issue — the decisive fact was in the bundle but the forecasters
did not use it (3 of 5 explicitly said they had no briefing). That is exactly the
distinction that matters for the hindsight-bias check: applying the miss lens to good
outcomes does NOT manufacture latent A/B/C/D gaps here; where something is off, it is
an E-type process issue, not a research-comprehension gap.

---

### qid 43730 — SCOTUS: will Clarence Thomas vote for Monsanto in *Durnell*?
- **Cohort**: control
- **Miss summary**: Bot said 82% Yes; resolved Yes (Thomas voted to set aside the Missouri judgment). Peer +25.2 — a clean win.
- **Decisive consideration**: Whether Thomas's *Bates* express-preemption dissent plus the SG's pro-Monsanto position outweigh his *Hencely* anti-broad-preemption signal — a forward-looking judgment about a not-yet-issued vote.
- **Bundle check**: Adequate and deep. Had oral-argument posture, the on-point *Bates* dissent ("state-law failure-to-warn claims can count as 'requirements'... preempted by FIFRA"), the countervailing *Hencely* opinion, SG amicus, and FantasySCOTUS "Thomas 64% Reverse". The targeted-research pass even correctly hedged to 55–65%.
- **Rationale check**: Models reasoned soundly and independently (gpt-5.4 72%, gpt-5.5 76%, opus-4.7 78%, opus-4.6 90%, gemini 88%, grok 70%); all distinguished express (FIFRA/*Bates*) from implied (*Hencely*) preemption correctly. No fact asserted beyond the bundle.
- **PRIMARY bucket**: clean
- **Secondary bucket**: none
- **Justification**: Genuine future event (decision pending as of the 2026-05-29 submission); the vote could not have been known. Bundle was comprehensive, mechanics (incl. the concurrence/dissent-counts-as-Yes nuance) were understood, and the forecast was well-founded. No decisive fact was missing, stale, misread, or fabricated.
- **Confidence**: high (would change only if the decision had actually issued pre-submission, which the dossier explicitly says it had not — "As of May 28, 2026, the Court has not issued a decision").
- **v2-addressable**: no

---

### qid 42862 — Lake Mead end-of-April 2026 elevation strictly above 1062.50 ft?
- **Cohort**: control
- **Miss summary**: Bot said 3% Yes; resolved No (end-of-April came in below 1062.50). Peer +30.0 — clean win.
- **Decisive consideration**: The USBR 24-Month Study "Most Probable" end-April projection of 1,056.88 ft (5.62 ft below threshold) plus the observed ~1.34 ft/week decline, against a razor-thin 0.32 ft margin.
- **Bundle check**: Adequate. Had the current gauge reading (1,062.82 ft on Mar 29), the USBR projection, the weekly decline, and the low-inflow / snow-drought context. One internal inconsistency — the native-search block lists an "End-of-March 2026 elevation recorded at 1,066.43 feet" alongside a "projected March 2026 end-of-month of 1,062.99 feet" and the 1,062.82 current reading — but forecasters used the correct fresh value.
- **Rationale check**: All five (gpt-5.2, gpt-5.1, claude-4.6-opus, claude-opus-4.5, gemini) correctly targeted the end-of-month April value, explicitly ran the bait-and-switch check ("not intramonth highs or projections"), and noted that Yes would require a ~5.6 ft positive forecast error. gpt-5.1's explicit Bayes calc landed at ~2.7%.
- **PRIMARY bucket**: clean
- **Secondary bucket**: none
- **Justification**: The one decisive quantity (USBR projection) was present and correctly applied; the actual April value was a future observation, well-forecast by the projection direction. The minor end-of-March data inconsistency in the bundle did not bite and was not used.
- **Confidence**: high (the near-certainty of the No side is robust to the small data-quality wrinkle).
- **v2-addressable**: no

---

### qid 42094 — Will Cloudflare experience another critical incident before May 2026?
- **Cohort**: control
- **Miss summary**: Bot said 68% Yes; resolved Yes (a Critical (red) incident occurred in the Feb 25–May 1 window). Peer +35.7 — clean win.
- **Decisive consideration**: An elevated incident-hazard regime (Nov/Dec 2025 + Feb 4 critical + Feb 20 6h outage) over a ~61-day window, cross-checked against Polymarket at 72% by Apr 30 for the same "Critical (red)" criterion.
- **Bundle check**: Adequate and well-matched to the resolution mechanic. Explicitly flagged that the resolver is Cloudflare's own "Critical (red)" status-page label, and that recent big outages "appear lower severity... not explicitly 'Critical (red)' in summaries" — i.e., it surfaced the exact ambiguity forecasters needed. Confirmed Polymarket's definition matched the question ("classified as Critical (red)... as of the time it is marked 'Resolved'").
- **Rationale check**: Forecasters (60–82%, median 68%) all ran clean bait-and-switch checks on "Critical (red)" vs "any outage," built Poisson base rates (~40%) and updated upward for the cluster + market. gpt-5.1 and gemini both explicitly red-teamed the "recent big outages may have been Major (orange), not Critical" scenario rather than assuming.
- **PRIMARY bucket**: clean
- **Secondary bucket**: none
- **Justification**: Bundle comprehensive, resolution mechanics correctly understood (and the label-severity ambiguity was surfaced, not glossed), forecast well-founded on base rate + market. No missing/stale/misread/fabricated decisive fact.
- **Confidence**: high
- **v2-addressable**: no

---

### qid 42116 — Will any Gemini model rank #1 on LM Arena on May 1, 2026?
- **Cohort**: control
- **Miss summary**: Bot said 21% Yes; resolved No (Gemini not #1). Peer +24.9 — a scoring win, but see below on *why* it was right.
- **Decisive consideration**: As of early March, Gemini was NOT #1 — Claude Opus 4.6 held #1 (1504 Elo) and Claude-opus-4-6-thinking #2 (1502), with gemini-3.1-pro-preview at #3 (1500), and Google I/O (typical Gemini launch window) falling *after* May 1.
- **Bundle check**: Bundle was **adequate** — the native-search block clearly stated the current standings ("ranks Anthropic's claude-opus-4-6 as #1 with 1504 Elo... Google's gemini-3.1-pro-preview is #3 at 1500 Elo... last updated about 8 hours prior") plus the I/O-timing and multi-market context. So there is NO A/B/C/D research-bundle gap here.
- **Rationale check**: Here is the wrinkle. The front-matter "Research Summary" is a one-line stub (unlike the full summaries on the other four questions), and 3 of 5 forecasters explicitly disclaimed having the briefing: gpt-5.2 — "I do not have live access here to the current LM Arena standings"; gpt-5.1 — "Because I can't see live 2025–2026 data... I'll base this on... pre-2024 data"; gemini — "does not provide a detailed intelligence briefing." None of the five referenced the decisive fact that Gemini was currently #3 behind two Claude models. They reached ~11–28% from generic base rates ("Gemini #1 on a random date ~20–25%") that happened to align with the outcome.
- **PRIMARY bucket**: clean
- **Secondary bucket**: E (pipeline/judgment)
- **Justification**: No research-comprehension failure — the bundle contained the decisive current-standings fact, so this is not latent A/B/C/D. But the forecasters did not use it (empty summarizer stub + explicit "no briefing" disclaimers), which is an E-type pipeline/judgment issue: right answer, largely wrong path. Worth surfacing because it means a "good outcome" here masks a process gap, not a research gap.
- **Confidence**: medium. The "clean of A/B/C/D" call is high-confidence (the fact was demonstrably in the bundle). The uncertainty is whether the E-issue is a genuine research-delivery bug vs. LLM boilerplate disclaimers; either way it stays E, not A–D. What would sharpen it: confirming whether the forecaster prompt actually received the full bundle or only the truncated summary.
- **v2-addressable**: no (E — a better research stage wouldn't help; the research was already good enough)

---

### qid 42578 — BLS seasonally adjusted LFPR for women 25–54, March 2026 (MC)
- **Cohort**: control
- **Miss summary**: Bot put 66.67% on "Greater than 78.1"; resolved "Greater than 78.1". Peer +112.3 — the batch's strongest win.
- **Decisive consideration**: The series had printed 78.3 (Feb), 78.3 (Jan), 78.4 (Dec) — a stable plateau — so remaining ≥78.2 was the modal outcome; a ≥0.2 pp one-month drop would be needed to miss it.
- **Bundle check**: Adequate. Had the exact recent monthly path, the March-3 release-date/mechanics note, the "no data yet for March" native-search confirmation, and softening signals (Feb payrolls −92k, unemployment 4.4%) to temper the upper bin.
- **Rationale check**: All five correctly identified the series, ran bait-and-switch checks (women 25–54 SA, March 2026, first release, rounded to tenth), anchored on 78.3, and appropriately kept tail mass. Spread (56–80% on the top bin) reflects honest disagreement on the size of the downside drift, not a shared error.
- **PRIMARY bucket**: clean
- **Secondary bucket**: none
- **Justification**: Bundle had the decisive current level + history; mechanics were understood precisely; the distribution was well-founded and correctly centered. The March value was a genuine future observation. No missing/stale/misread/fabricated fact.
- **Confidence**: high
- **v2-addressable**: no
