"""Per-member EXTREME_CALL telemetry for binary questions.

One line per surviving forecaster whose probability sits at or past either edge of
the extreme band (``EXTREME_CALL_LOW`` / ``EXTREME_CALL_HIGH`` in ``constants.py``),
carrying whether ANY OTHER survivor was extreme on the same side::

    EXTREME_CALL: question=<id> model=<name> p=<value> side=low|high lone=<bool> survivors=<k>

Why this is a standing marker rather than a per-round script: the lone-versus-
accompanied split is the cleanest per-model signal three residual rounds have
produced (``scratch/residual_2026-08-31/gemini_review/RECOMMENDATION.md`` §2 — 9 lone
extreme binary calls, 4 right, at a mean stated confidence of 0.972, against 21 of 23
right where another member agreed), and it was reconstructed by hand from parsed
Metaculus comments every time. Comments are a lossy source: they are middle-trimmed
past a char limit, a stacked question publishes only the aggregate bullet, and nothing
in them says which members were extreme on the SAME SIDE. Logging it at the fan-out
makes the next season's read a query.

Scope notes that keep the numerator honest:

* Binary only. MC concentration is a different measurement (top-option probability,
  normalised entropy) and a separate standing cut that was not adopted; a dominant MC
  option is not an extreme binary call.
* Nothing is emitted for a member inside the band. The denominator for a rate claim is
  the survivor list, which ``FORECASTERS_SURVIVED`` already states for the same
  question in the same run log.
* ``lone`` is vacuous at one survivor, so ``survivors`` rides along and an analysis can
  condition on it (the receipt above excludes the degraded-window records from the lone
  test for exactly that reason) rather than joining to another marker to find out.

One caveat before pooling these lines with the receipt's own counts. ``lone`` here means
"no other member extreme on the SAME SIDE", which is what the memo's prose says and what
the bundle plan specified; the memo's own scripts (``cut1_binary.py`` ``band_of`` +
``all()``, ``cut1_supp.py`` S2 ``len(ex_models) == 1``) implement the looser "no other
member extreme at all". The two disagree only when members straddle, and a replay of this
emitter over the 570 archived extreme member-calls found 4 such rows — all pre_flip, all
of them cases where the looser rule reads plainly wrong (q42314: gemini alone at 0.99 with
three peers at 0.02-0.05 counted as accompanied). Pre_flip lone counts are therefore 52
under this marker against the memo's 48; post_flip and triple_era agree exactly, including
the memo's headline gemini row (7 extreme, 3 lone). Receipts:
``scratch/next_season_bundle_2026-09/item15/``.

The band is a MEASUREMENT boundary here: this module reads probabilities and returns
strings, and never changes a forecast.
"""

from __future__ import annotations

from collections.abc import Sequence

from metaculus_bot.constants import EXTREME_CALL_HIGH, EXTREME_CALL_LOW

EXTREME_SIDE_LOW = "low"
EXTREME_SIDE_HIGH = "high"

# Same spelling FORECASTERS_SURVIVED uses when no prediction carried a ``Model:``
# prefix, so the two lines stay joinable on the model field. Deliberately not a
# marker-parser None sentinel ("none"/"n/a"/"null"), so it harvests as a readable
# string rather than an absent value.
UNKNOWN_MODEL_NAME = "unknown"


def extreme_call_side(probability: float) -> str | None:
    """Which edge of the extreme band ``probability`` sits at, or None if neither.

    Membership is INCLUSIVE at both edges (``p <= LOW`` or ``p >= HIGH``), so a call
    exactly on an edge counts as extreme.
    """
    if probability <= EXTREME_CALL_LOW:
        return EXTREME_SIDE_LOW
    if probability >= EXTREME_CALL_HIGH:
        return EXTREME_SIDE_HIGH
    return None


def format_extreme_call_markers(question_id: int, members: Sequence[tuple[str | None, float]]) -> list[str]:
    """Build the EXTREME_CALL lines for one binary question's surviving members.

    ``members`` is ``(model display name or None, probability)`` per SURVIVING
    forecaster, in fan-out order; the returned list holds one line per extreme member
    and is empty when none is. A None or blank name renders ``UNKNOWN_MODEL_NAME``.

    ``lone`` is true when no OTHER member is extreme on the same side. Same side is the
    load-bearing half: two members at 0.03 and 0.97 are maximally disagreeing, so each
    is lone despite both being extreme.
    """
    sides = [extreme_call_side(probability) for _, probability in members]
    survivors = len(members)
    lines: list[str] = []
    for index, ((model, probability), side) in enumerate(zip(members, sides, strict=True)):
        if side is None:
            continue
        accompanied = any(other == side for position, other in enumerate(sides) if position != index)
        lines.append(
            f"EXTREME_CALL: question={question_id} model={model or UNKNOWN_MODEL_NAME} "
            f"p={probability:.4f} side={side} lone={str(not accompanied).lower()} survivors={survivors}"
        )
    return lines
