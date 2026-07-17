# Anchor-vs-final: does deviating from your own base rate help or hurt? (2026-07-16)

Follow-on to the base-rate audit (`../base_rate_audit_2026-07-16/RESULTS.md`). That audit
found remembered base rates are mostly accurate but surfaced vivid cases of models
abandoning their own correct anchors (AfD runoff, TSMC month). The operator flagged the
obvious confirmation-bias hole: nobody counted the cases where deviating from the anchor
*helped*. This analysis measures deviation symmetrically, with no thumb on the scale.

Two live hypotheses:
- **H1** — models over-correct toward the inside view, so anchor stickiness would help.
- **H2** — models deviate appropriately (or under-deviate), so stickiness would hurt or be neutral.

## Headline

**Deviation direction is a coin flip; the net cost is a magnitude story, and that
magnitude lives entirely on the base rates that were *already wrong*.** Across 63
(question, model) pairs where the model moved off its own outside-view anchor: **helped 31,
hurt 32** (two-sided sign test p = 1.00). Mean Δlog = **−0.064** [bootstrap 95% CI
−0.132, −0.001], i.e. deviating cost a small amount of log score on average. But the
decomposition by anchor accuracy is the whole story:

| anchor (audit-verified) | n | helped | hurt | mean Δlog | anchorLL → finalLL |
|---|---|---|---|---|---|
| **accurate** | 12 | 7 | 5 | **+0.004** | −1.008 → −1.004 |
| **wrong** | 13 | 2 | 11 | **−0.326** | −1.305 → −1.631 |
| uncertain | 4 | 3 | 1 | +0.003 | −0.142 → −0.140 |
| unverified | 34 | 19 | 15 | +0.004 | −0.526 → −0.522 |

The 13 verified-wrong-anchor pairs contribute **−4.24** of the total **−4.06** log-points
of net damage. Every other stratum nets ~zero. So: where the anchor was good, deviating
neither helped nor hurt (H1 refuted for the good-anchor case); where the anchor was bad,
deviating usually made a bad forecast worse — but so did staying (both anchorLL and finalLL
are deeply negative, ≈ −1.3 to −1.6). Staying anchored was **not** a safe harbor on the
wrong anchors, because the reference class itself was the defect.

## Method

### Mapping-judgment protocol (the load-bearing subjectivity)

Analysis 1 needs, per rationale, an outside-view rate that maps DIRECTLY onto the
question's probability. I used the model's own **"My base rate was X. After considering
current evidence, I'm moving to Y"** declaration wherever present — it is resolution-blind,
lives in the question's own probability space, and is the model's own mapping, so it
minimizes analyst discretion. Where no such line existed I mapped the committed
outside-view anchor ("I anchor at X%", "Outside-view base rate: X%") that the model
carried into PHASE 2. Sources: `../base_rate_audit_2026-07-16/rationales/<pid>.md` (finals
and scores were stripped from these before extraction) + per-model finals from
`../coherence_2026-07-15/perf_all_tagged.json`. The curated map is `anchor_map.py`; every
`anchor_quote` was verbatim-checked against the rationale text this session.

**Mapping calls.** 69 pairs mapped (45 binary, 24 MC), of which **60 direct, 9 borderline**.
Borderline (flagged individually in `anchor_map.py`):
- **41846 coups** (3 models): the model states 36% for the *full* 3-month window but
  time-discounts to ~11% for the remaining 23 days and carries the ~11% into its final. I
  used the horizon-matched ~11%, not the 36%, so this is NOT scored as a huge deviation.
- **42110 WI SC / 42509 Brent**: anchor is a component the model blends/updates within the
  outside-view stage.
- **42438 Duke NCAA**: resolved option (Elite Eight) is a *middle* bucket where both anchor
  and final sit at ~17%; the audit's inflated-F4 miss lives in the tail shape, not the E8
  prob, so this pair barely moves and is honestly a weak anchor-map.

**MC scoring convention.** For MC I score P(the resolved option): anchor's P(option) vs
final's P(option), graded by whether the option happened. AfD-runoff anchors stated as
"AfD converts 30%" map to CDU = 1 − 0.30 = 0.70 (CDU resolved).

**Excluded (not a direct map, listed in `anchor_map.NOT_DIRECT`):** 41848 (no per-model
finals survived comment-trimming), 43652 Armenia (vote→seat conversion feeds a 5-bucket
distribution, no single mapped anchor), and 43058/43076/43135 (only stacker-meta section
survived trimming).

### Confirmation-bias cross-check (independent blind adjudicator)

Because the mapping is judgment-heavy, an independent Opus adjudicator re-extracted anchors
from the same rationales **resolution-blind** (`_adj_B.json`, 82 sections). On the 45
overlapping binary pairs, **mean |my_anchor − blind_anchor| = 0.006** (44/45 within 0.05).
Re-running the headline on the adjudicator's anchors reproduces the coin flip: **helped 21,
hurt 20, mean Δlog −0.056** (mine on the same subset: 20/20, −0.066). The adjudicator
independently flagged the same hard borderline calls (41846 horizon, 41754 Putin-horizon,
42119 chem-weapons war-lifetime, 42438 Duke cumulative). The conclusion is not an artifact
of my anchor choices. (`interrater.py`; a second adjudicator `_adj_A.json` was dispatched
and will fold in automatically if it lands — not required for the result.)

### Scoring

Log score = log(p on realized outcome) and negated Brier (higher = better for both);
Δlog = finalLL − anchorLL (>0 = deviation HELPED). Deviation magnitude = symmetric odds
ratio between anchor and final. Probabilities clamped to [1e-6, 1−1e-6] for finite logs.
Era per manifest (`config_era`); sample is 31 pre-flip / 9 post-flip, spring+summer 2026.
Bootstrap CI = 20k resamples, seed 20260716. Scripts: `score_anchor.py`, `robustness.py`.

## Analysis 1 — symmetric anchor-vs-final scoring

### Overall (moved pairs only; n=63 of 69, 6 pairs didn't move)

| slice | n | helped | hurt | mean Δlog [95% CI] | mean ΔBrier |
|---|---|---|---|---|---|
| all mapped | 63 | 31 | 32 | −0.064 [−0.132, −0.001] | −0.026 |
| direct-only (drop borderline) | 57 | 28 | 29 | −0.065 [−0.137, +0.003] | −0.026 |
| binary | 40 | 20 | 20 | −0.074 [−0.159, +0.001] | −0.022 |
| MC | 23 | 11 | 12 | −0.048 [−0.160, +0.064] | −0.032 |

Sign test on 31 vs 32: **p = 1.00**. The negative mean survives dropping borderline pairs.

### By deviation magnitude (odds ratio anchor↔final)

| magnitude | n | helped | hurt | mean Δlog | sd |
|---|---|---|---|---|---|
| small (<1.5x odds) | 35 | 18 (avg +0.072) | 17 (avg −0.113) | −0.018 | 0.118 |
| medium (1.5–3x) | 22 | 10 (avg +0.197) | 12 (avg −0.285) | −0.066 | 0.313 |
| large (>3x odds) | 6 | 3 (avg +0.114) | 3 (avg −0.773) | −0.330 | 0.468 |

Large deviations are **balanced in direction (3 helped / 3 hurt)** but the standard
deviation explodes (0.47) — magnitude predicts *variance*, not a systematic wrong-way bias.
This is a **soft-leash** signature, not hard stickiness: small moves are ~free, big moves
are high-variance bets that occasionally cost a lot (the two BW-CDU pairs, each −0.92).
There is no magnitude threshold beyond which deviations flip reliably net-negative — even
the >3x bucket is a coin flip on direction.

### By verified anchor accuracy — see the Headline table. This is the decisive split.

### By tercile (sample deliberately oversamples "worst"; strata reported separately)

| tercile | n | helped | hurt | mean Δlog |
|---|---|---|---|---|
| worst | 26 | 7 | 19 | **−0.196** |
| middling | 23 | 15 | 8 | **+0.054** |
| good | 14 | 9 | 5 | −0.014 |

Equal-weight-across-terciles mean Δlog = **−0.052** (vs −0.064 raw), so the oversampling of
bad outcomes inflates the apparent cost only modestly. Critically, in the **middling**
tercile deviation was net *positive* (helped 15 / hurt 8). The "deviation hurts" signal is
concentrated in the worst tercile — which is exactly where the audit's verified-wrong
anchors cluster (BW-CDU, novel-flu, Anthropic-release, TSMC, AfD are all worst/low-tercile).

### By era

pre_flip (n=49): −0.085 [−0.162, −0.016]; post_flip (n=14): +0.007 [−0.128, +0.148]. The
negative signal is a pre-flip-era artifact; post-flip is a coin flip. n post-flip is tiny —
don't over-read, but it means the cost is not clearly a property of the current roster.

### Leave-one-out

Full mean Δlog −0.064 → drop the single most-hurt pair (42116 BW/gpt-5.1) → −0.051 → drop 2
→ −0.037 → drop 5 → −0.010. The headline "deviation costs log score" is carried by a
literal handful of large wrong-anchor bets; trim 5 of 63 and it's ~zero.

## Analysis 2 — cross-model reference-class dispersion: bug or feature?

`dispersion.py` (2a, question-level anchor dispersion), `citability.py` (2b, class tally).

### 2a. Does anchor dispersion predict anything? (n=19 questions with ≥3 mapped anchors)

Dispersion = odds-spread across models' anchors (log-scaled for correlation). Spearman
(descriptive, n=19, no significance claimed):

- **anchor dispersion vs question peer score = −0.33.** Weakly consistent with the
  guessing-signature hypothesis: questions where models disagree more on the base rate
  tend to score a bit worse. Confounded with difficulty (hard questions invite both
  dispersion and bad scores). Not significant at this n.
- **anchor dispersion vs final ensemble spread = −0.05.** Essentially zero — models that
  disagree on the *base rate* do NOT end up with more dispersed *final* forecasts. The
  inside-view stage and aggregation wash out anchor disagreement. (E.g. 42120 Frontier:
  22x anchor odds-spread → only 0.06 final spread.)
- **anchor dispersion vs median-advantage-over-mean-individual = +0.13.** Weakly positive.

### The diversity-as-feature test: does the MEDIAN absorb diverse anchors?

Median log-score advantage over the mean individual model, by anchor-dispersion tercile:

| dispersion tercile | n | mean odds-spread | mean median-advantage |
|---|---|---|---|
| low | 6 | 1.22 | +0.018 |
| mid | 6 | 1.97 | −0.015 |
| high | 7 | 8.62 | **+0.047** |
| all | 19 | — | +0.018 |

The median's advantage is **positive and largest in the high-dispersion bucket**. When
models bring genuinely different base rates, the median absorbs them *better* than it does
when they agree — exactly the diversity-as-feature signature. Combined with the near-zero
dispersion→final-spread link, this says cross-model base-rate diversity is closer to a
**feature than a bug**: it is not propagating into wild final forecasts, and the ensemble
median is net-helped by it. Homogenizing base rates across models would remove a source of
robustness the aggregator is already exploiting. (n=19, terciles of 6–7; directional, not
significant.)

### 2b. How many high-dispersion classes have a citable dataset? (n=27 named classes)

Descriptive tally of whether a public dataset with the numerator/denominator plausibly
exists (this sizes the "route these to gap-fill base-rate research" idea):

- **yes = 19** (public dataset, audit-confirmed or directly countable: election archives,
  sports-reference seed histories, FDA/SEC/CDC/EIA/FRED-style series, release histories)
- **partial = 7** (a series exists but the exact class-slice needs assembly/judgment:
  UNMISS abstention-rate by resolution type, ULCC-with-liquidity bankruptcy, ICJ orders/yr)
- **no = 1** (genuinely bespoke: 41672 "community aggregate beats THIS named individual over
  a season" — no dataset answers that)

**26/27 (96%) are routable** to a base-rate lookup. And the classes where models already
*agreed* AND were *right* are overwhelmingly the citable ones the audit verified (TSMC month,
coups/yr, generic-ballot drift, US CPI, Armenia vote→seat) — dispersion is low precisely
because a real number exists and models converge on it. Where a citable number exists but
models diverge (NCAA seed tables 50% vs 54-58%, AfD conversion 30/40/45%, BW recency vs
long-run), that divergence is the exact "three conflicting tables for one checkable
frequency" tell the base-rate audit named — and it maps 1:1 onto a routable lookup.

## Verdict

**(a) Is anchor stickiness supported? — Refuted in the form "make models stick to their
anchors," and unresolved in the sharper form "make *verified* anchors sticky."** Deviation
direction is a coin flip (31/32, p=1.00). Where the anchor was verified-accurate, deviating
was net-zero (+0.004, 7 helped / 5 hurt) — so a blanket "trust your base rate" rule would
NOT have helped even on the good anchors; models mostly deviate sensibly off good anchors.
The net cost of deviation (−0.064 log) is real but small and comes almost entirely from
large moves off anchors that were *already wrong* (−4.24 of −4.06 total log-points on 13
wrong-anchor pairs). On those, staying anchored was no refuge either — both the anchor and
the final scored terribly because the reference class was broken. The observed pattern is a
**soft leash** (small moves free, big moves high-variance) rather than hard stickiness, and
there's no magnitude threshold where deviation reliably flips net-negative. H1 (over-
correction toward inside view) gets weak support only in the worst tercile / wrong-anchor
cases; H2 (deviation is roughly appropriate) fits the accurate-anchor and middling-tercile
data. Net: the lever isn't "deviate less," it's "**start from a correct anchor**."

**(b) Is base-rate diversity a bug or a feature? — Closer to a feature; do not homogenize.**
Cross-model anchor dispersion does not propagate to final-forecast spread (Spearman −0.05),
and the ensemble median's advantage over the mean individual is *positive and largest* on
high-dispersion questions (+0.047 vs +0.018 overall). The aggregator already absorbs diverse
anchors as robustness. The genuine bug isn't dispersion per se — it's dispersion **on a
checkable historical frequency** (three conflicting NCAA tables, 30/40/45% AfD), which is a
cheap detectable trigger, not something to average away.

**(c) What, if anything, to change.** The evidence points at one lever, consistent with the
base-rate audit's bottom line: **source the anchor, don't leash the deviation.**
1. **Route high-dispersion-on-a-checkable-frequency classes to the gap-fill base-rate pass.**
   26/27 dispersed classes here are citable; cross-model disagreement on a named historical
   frequency is a free, computable trigger (odds-spread across per-model anchors) for exactly
   which base rates to send. This is where the peer points are, and it fixes the −4.24
   wrong-anchor tail at its root.
2. **Do NOT ship a blanket anchor-stickiness / deviation-penalty rule.** On accurate anchors
   it's net-zero, and it would suppress the middling-tercile pairs where deviation helped
   (+0.054). A "treat >2x-odds deviations as requiring itemized evidence" rule would only pay
   off if gated on the anchor being *verified* first — which is just item (1) with an extra
   step. Any such fitted leash needs a decisive out-of-sample era test before shipping
   (fit on eras 1..k−1, must improve era k), per repo convention, or it's a drift bomb.
3. **Do NOT homogenize base rates across the ensemble.** Diversity is feeding the median.

## Confidence and caveats

- **n is small and the result is honest about it.** 63 scored pairs, 22 verified clusters,
  40 questions. No significance is claimed anywhere; the sign test (p=1.00) is the one place
  a test is clean, and it says "coin flip." The wrong-anchor Δlog (−0.326) has n=13.
- **The mapping is judgment-heavy but cross-validated.** 60/69 direct, 9 borderline (each
  flagged). An independent blind Opus adjudicator reproduced the binary anchors to within
  0.006 mean and reproduced the coin-flip headline — the single strongest guard against the
  confirmation-bias worry that motivated this analysis.
- **Verified-accuracy labels come from the base-rate audit's 22 clusters** (its own caveats
  apply: financial-vol clusters graded generously, 3 clusters uncertain). The accurate/wrong
  split is the load-bearing axis, so its ~20% error interval propagates here.
- **Oversampling of "worst" is handled** by reporting terciles separately and reweighting
  (equal-weight-across-terciles −0.052 vs raw −0.064); it inflates the apparent cost only
  modestly and the middling-tercile positive result is not an artifact.
- **MC anchor→option mapping is the weakest link** (esp. 42438 Duke, a middle-bucket
  resolution); binary-only numbers (−0.074) tell the same story as the pooled ones, so MC
  fragility doesn't drive the verdict.
- **Era**: pre-flip carries the negative signal; post-flip (n=14) is a coin flip. Don't
  project the cost onto the current roster without more post-flip data.

Bottom line, if the result were "no clear signal" that would be the result — and on the
central question (does deviating from your anchor systematically help or hurt?) the answer
genuinely IS "no clear signal, it's a coin flip." The only clean, non-null finding is
conditional: deviating hurts specifically when the anchor was already wrong, and there the
fix is a better anchor, not a shorter leash.
