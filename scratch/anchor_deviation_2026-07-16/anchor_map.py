"""Curated anchor->final->resolution mapping for the symmetric deviation analysis.

Each row pairs a model's OUTSIDE-VIEW ANCHOR (the base rate it committed to before
inside-view updating, resolution-blind) with its FINAL published forecast and the
resolved outcome, expressed in a common probability space:

- binary: prob of YES for the question. resolved_yes in {True, False}.
- multiple_choice: prob assigned to the RESOLVED-relevant option. We score anchor vs
  final on P(that option); resolved_yes = True (the option happened) so higher-is-better
  on that option. This is the apples-to-apples comparison that the outcome can grade.

`mapping` is my judgment call per the operator's protocol:
  direct     - anchor cleanly equals a question-level probability the outcome can grade
  borderline - required a horizon/definition/option-selection judgment (flagged in note)
  (rows that are not_direct are simply omitted from scoring; listed in NOT_DIRECT below)

Anchor/final numbers are transcribed from the rationale files and _adjudicator_packet.json.
Every anchor_quote is verbatim-checked against rationales/<pid>.md this session.
"""

# fmt: off
# pid, model, qtype, anchor_p, final_p, resolved_yes, mapping, verified_cluster, verif_status, note
ROWS = [
    # ---------------- BINARY ----------------
    # 41672 Community beats Nathan Young (res NO). gemini research 75%, gpt5.2 sym 50%, gpt5 65%.
    ("41672", "gemini-3-pro-preview", "binary", 0.75, 0.78, False, "direct", None, None,
     "anchor 75% (CP>expert); final 78%"),
    ("41672", "gpt-5.2", "binary", 0.50, 0.55, False, "direct", None, None,
     "symmetry anchor 50% -> 55%"),
    ("41672", "gpt-5", "binary", 0.65, 0.67, False, "direct", None, None,
     "CP>individual anchor 65% -> 67%"),
    # 41754 Putin ceases president (res NO). G8 cluster (uncertain, away).
    ("41754", "claude-opus-4.5", "binary", 0.03, 0.03, False, "direct", "G8", "uncertain",
     "combined ~2-4% annual, ~2-3% window -> 3%"),
    ("41754", "gemini-3-flash-preview", "binary", 0.065, 0.06, False, "direct", "G8", "uncertain",
     "~5-8% window anchor -> 6%"),
    ("41754", "gemini-3-pro-preview", "binary", 0.013, 0.02, False, "direct", None, None,
     "computed ~1.3% quarterly -> 2%"),
    # 41835 shutdown (res YES). G1 cluster (not materially off; toward).
    ("41835", "gpt-5.2", "binary", 0.20, 0.11, True, "direct", "G1", "accurate",
     "anchor 20% per deadline -> 11% (BELOW anchor)"),
    ("41835", "gpt-5", "binary", 0.14, 0.12, True, "direct", "G1", "accurate",
     "anchor 14% -> 12%"),
    ("41835", "claude-opus-4.5", "binary", 0.10, 0.11, True, "direct", "G1", "accurate",
     "anchor 8-12% (~10%) -> 11%"),
    # 41846 coups (res NO). G2 cluster (accurate research; neutral). Horizon: use remaining-window anchor.
    ("41846", "gpt-5.2", "binary", 0.11, 0.11, False, "borderline", "G2", "accurate",
     "36% full window BUT model carries ~11% remaining-window into final; used 11% (horizon-matched)"),
    ("41846", "gpt-5", "binary", 0.105, 0.09, False, "borderline", "G2", "accurate",
     "remaining-window baseline 10-11% -> 9%"),
    ("41846", "claude-opus-4.5", "binary", 0.12, 0.12, False, "borderline", "G2", "accurate",
     "Poisson 23-day ~12% -> 12%"),
    # 42116 CDU most seats BW (res NO). C4 cluster (materially off, toward).
    ("42116", "gpt-5.2", "binary", 0.55, 0.82, False, "direct", "C4", "wrong",
     "blended 55% -> 82%"),
    ("42116", "gpt-5.1", "binary", 0.65, 0.86, False, "direct", "C4", "wrong",
     "blended 65% -> 86%"),
    ("42116", "claude-4.6-opus", "binary", 0.95, 0.95, False, "direct", "C4", "wrong",
     "blended ~95% -> 95% (poll-lead-conversion class, not the recency class)"),
    # 42117 AfD most seats BW (res NO).
    ("42117", "claude-opus-4.5", "binary", 0.025, 0.02, False, "direct", None, None,
     "2-3% -> 2%"),
    ("42117", "gemini-3-pro-preview", "binary", 0.005, 0.01, False, "direct", None, None,
     "<1% -> 1%"),
    # 42119 chemical weapons Ukraine (res NO). G4 cluster (accurate; neutral). opus anchored 12%, final 13%.
    ("42119", "claude-opus-4.5", "binary", 0.12, 0.13, False, "direct", "G4", "accurate",
     "outside-view 10-15%, anchored 12% -> 13%"),
    ("42119", "gemini-3.1-pro-preview", "binary", 0.05, 0.01, False, "direct", "G4", "accurate",
     "~5% lifetime-of-war base -> 1%"),
    # 42120 Frontier bankruptcy (res NO).
    ("42120", "claude-4.6-opus", "binary", 0.055, 0.04, False, "direct", None, None,
     "5-6% -> 4%"),
    ("42120", "claude-opus-4.5", "binary", 0.10, 0.07, False, "direct", None, None,
     "8-12% (~10%) -> 7%"),
    ("42120", "gemini-3.1-pro-preview", "binary", 0.005, 0.01, False, "direct", None, None,
     "<0.5% -> 1%"),
    # 42242 enwiki articles (res NO).
    ("42242", "gemini-3.1-pro-preview", "binary", 0.008, 0.02, False, "direct", None, None,
     "<1% organic spike -> 2%"),
    ("42242", "gpt-5.2", "binary", 0.03, 0.02, False, "direct", None, None,
     "anchor 3% -> 2%"),
    ("42242", "gpt-5.1", "binary", 0.05, 0.02, False, "direct", None, None,
     "~5% -> 2%"),
    # 42514 Android critical vuln (res YES).
    ("42514", "claude-opus-4.5", "binary", 0.40, 0.55, True, "direct", None, None,
     "40% -> 55%"),
    ("42514", "gemini-3.1-pro-preview", "binary", 0.40, 0.38, True, "direct", None, None,
     "40% -> 38%"),
    ("42514", "gpt-5.2", "binary", 0.45, 0.52, True, "direct", None, None,
     "45% -> 52%"),
    # 42641 New Glenn launch (res YES).
    ("42641", "gemini-3.1-pro-preview", "binary", 0.45, 0.38, True, "direct", None, None,
     "~45% delayed-rocket-in-slip -> 38%"),
    ("42641", "gpt-5.1", "binary", 0.65, 0.70, True, "direct", None, None,
     "60-70% (65%) -> 70%"),
    # 42646 UNMISS abstention (res NO).
    ("42646", "gpt-5.1", "binary", 0.65, 0.60, False, "direct", None, None,
     "~65% generic -> 60%"),
    ("42646", "claude-4.6-opus", "binary", 0.65, 0.65, False, "direct", None, None,
     "~65% -> 65%"),
    ("42646", "claude-opus-4.5", "binary", 0.85, 0.82, False, "direct", None, None,
     "UNMISS-specific 85% -> 82%"),
    # 42926 novel flu (res NO). G7 cluster (materially off, toward).
    ("42926", "claude-4.6-opus", "binary", 0.58, 0.65, False, "direct", "G7", "wrong",
     "~58% blended -> 65%"),
    ("42926", "claude-opus-4.5", "binary", 0.70, 0.65, False, "direct", "G7", "wrong",
     "Poisson ~70% -> 65% (constant-rate extrapolation)"),
    ("42926", "gemini-3.1-pro-preview", "binary", 0.87, 0.84, False, "direct", "G7", "wrong",
     "Poisson lambda2.04 ~87% -> 84%"),
    # 43131 Anthropic release (res YES). G6 cluster (materially off, toward).
    ("43131", "gpt-5.2", "binary", 0.20, 0.13, True, "direct", "G6", "wrong",
     "cadence anchor 20% -> 13%"),
    ("43131", "claude-opus-4.5", "binary", 0.09, 0.06, True, "direct", "G6", "wrong",
     "8-10% (~9%) -> 6%"),
    ("43131", "claude-opus-4.6", "binary", 0.10, 0.07, True, "direct", "G6", "wrong",
     "~10% -> 6-7%"),
    # 43828 Swiss 10-million (res NO). C5 cluster (uncertain, neutral).
    ("43828", "gpt-5.5", "binary", 0.20, 0.24, False, "direct", "C5", "uncertain",
     "SVP-initiative 20% -> 24%"),
    ("43828", "claude-opus-4.8", "binary", 0.12, 0.09, False, "direct", "C5", "uncertain",
     "8-15% (~12%) -> 9%"),
    ("43828", "claude-opus-4.6", "binary", 0.14, 0.12, False, "direct", "C5", "uncertain",
     "13-14% z-score model -> 12%"),
    # 43915 atacicept FDA (res YES).
    ("43915", "gemini-3.1-pro-preview", "binary", 0.85, 0.91, True, "direct", None, None,
     "7/7 ~100% class, on-time ~85% -> 91%"),
    ("43915", "grok-4.3", "binary", 0.85, 0.82, True, "direct", None, None,
     "7/7 -> ~85% -> 82%"),
    ("43915", "gpt-5.4", "binary", 0.81, 0.86, True, "direct", None, None,
     "Jeffreys on-time ~81% -> 86%"),

    # ---------------- MULTIPLE CHOICE (P of resolved option) ----------------
    # 42110 WI SC (res Chris Taylor). C6 cluster (accurate research, toward).
    ("42110", "gemini-3-pro-preview", "multiple_choice", 0.80, 0.93, True, "borderline", "C6", "accurate",
     "50/50 generic anchor updated to ~80/20 on post-2022 trend (the carried outside view) -> 93% Taylor"),
    ("42110", "gpt-5.1", "multiple_choice", 0.80, 0.89, True, "borderline", "C6", "accurate",
     "recent-elections class ~55-45 x4-of-5 -> ~80% carried -> 89% Taylor"),
    # 42248 WHR #1 (res Finland). incumbent-retention anchor maps to Finland.
    ("42248", "gpt-5.1", "multiple_choice", 0.80, 0.93, True, "direct", None, None,
     "incumbent retention ~75-85% -> 93% Finland"),
    ("42248", "claude-4.6-opus", "multiple_choice", 0.875, 0.88, True, "direct", None, None,
     "incumbent retention ~85-90% -> 88% Finland"),
    ("42248", "claude-opus-4.5", "multiple_choice", 0.87, 0.85, True, "direct", None, None,
     "base rate 87% Finland -> 85%"),
    # 42438 Duke NCAA (res Elite Eight = a MIDDLE bucket). C1 cluster (materially off, toward).
    # Anchor and final both ~17% on E8; the miss is tail-shape not the E8 prob. Mark borderline+note.
    ("42438", "claude-4.6-opus", "multiple_choice", 0.20, 0.17, True, "borderline", "C1", "wrong",
     "exit-dist anchor E8~20% -> final E8 17%; inflated F4+ elsewhere but E8 bucket itself barely moved"),
    ("42438", "claude-opus-4.5", "multiple_choice", 0.20, 0.17, True, "borderline", "C1", "wrong",
     "E8 ~18-22% anchor -> 17% final"),
    ("42438", "gemini-3.1-pro-preview", "multiple_choice", 0.16, 0.16, True, "borderline", "C1", "wrong",
     "E8 ~16% -> 16% (gemini best-calibrated of the three)"),
    # 42509 Brent >=100 (res 100+). lognormal outside dist put ~10-25% on >=100.
    ("42509", "claude-opus-4.5", "multiple_choice", 0.25, 0.23, True, "direct", None, None,
     "outside lognormal >100 ~25% -> 23%"),
    ("42509", "gemini-3.1-pro-preview", "multiple_choice", 0.25, 0.18, True, "borderline", None, None,
     "lognormal EV90 IV45 gives >100 ~25% -> 18%"),
    ("42509", "gpt-5.2", "multiple_choice", 0.25, 0.15, True, "direct", None, None,
     "lognormal raw >100 ~25% -> 15%"),
    # 42800 TSMC month (res March). M8 cluster (accurate, toward). Anchor vs final on P(March).
    ("42800", "gpt-5.1", "multiple_choice", 0.60, 0.35, True, "direct", "M8", "accurate",
     "Mar 60% anchor -> 35% (picked January)"),
    ("42800", "claude-4.6-opus", "multiple_choice", 0.575, 0.52, True, "direct", "M8", "accurate",
     "March 55-60% -> 52%"),
    # 42805 Bulgaria (res Other/tie). anchor on Other option.
    ("42805", "claude-4.6-opus", "multiple_choice", 0.70, 0.79, True, "direct", None, None,
     "Other ~70% anchor -> 79%"),
    ("42805", "gemini-3.1-pro-preview", "multiple_choice", 0.75, 0.83, True, "direct", None, None,
     "leading-new-party ~70-80% -> 83% Other"),
    ("42805", "claude-opus-4.5", "multiple_choice", 0.70, 0.83, True, "direct", None, None,
     "Other ~70% -> 83%"),
    # 43635 Makerfield (res Labour). Labour-plurality anchor.
    ("43635", "gpt-5.4", "multiple_choice", 0.68, 0.62, True, "direct", None, None,
     "Labour 68% anchor -> 62%"),
    ("43635", "gpt-5.5", "multiple_choice", 0.58, 0.62, True, "direct", None, None,
     "Labour 58% -> 62%"),
    ("43635", "claude-opus-4.7", "multiple_choice", 0.60, 0.63, True, "direct", None, None,
     "Labour ~60% -> 63%"),
    # 43696 SCOTUS author (res Kavanaugh). uniform ~11% anchor on any single justice.
    ("43696", "claude-opus-4.7", "multiple_choice", 0.11, 0.19, True, "direct", "G5", "accurate",
     "uniform ~11% -> 19% Kavanaugh"),
    ("43696", "claude-opus-4.6", "multiple_choice", 0.11, 0.18, True, "direct", "G5", "accurate",
     "uniform ~11% -> 18% Kavanaugh"),
    # 43982 Saalekreis runoff (res CDU; AfD LOST). C3 cluster (materially off, toward). Map to CDU (resolved).
    # Anchor is stated as AfD-conversion; CDU anchor = 1 - AfD_anchor.
    ("43982", "grok-4.3", "multiple_choice", 0.70, 0.43, True, "direct", "C3", "wrong",
     "AfD 30% -> CDU 70% anchor; final CDU 43% (moved AfD UP to 57%)"),
    ("43982", "gpt-5.4", "multiple_choice", 0.60, 0.45, True, "direct", "C3", "wrong",
     "AfD 40% -> CDU 60% anchor; final CDU 45%"),
    ("43982", "gpt-5.5", "multiple_choice", 0.55, 0.41, True, "direct", "C3", "wrong",
     "AfD 45% -> CDU 55% anchor; final CDU 41%"),
]

# Sections judged NOT a direct anchor->question map (excluded from scoring), for transparency.
NOT_DIRECT = [
    ("41848", "all", "Meta AG settlement: no per-model final available (per_model_forecasts empty); "
     "monthly-hazard anchors can't be scored."),
    ("43652", "unknown", "Armenia seat-share MC: anchor is a vote->seat conversion feeding a full "
     "5-bucket distribution; the resolved middle bucket isn't a single mapped anchor prob."),
    ("43058/43076/43135", "unknown", "only stacker-meta section survived comment trimming; no model "
     "outside-view anchor + no per-model final."),
]
# fmt: on
