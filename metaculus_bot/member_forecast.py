"""Forecast-value telemetry: every member's value raw and as published, and the aggregate's tails.

One INFO line per forecast VALUE a runner hands onward, for all four question types,
for every ensemble member and for the stacker::

    MEMBER_FORECAST: question=<id> model=<name> role=member|stacker qtype=<type> raw=<json> published=<json>

A numeric or date line carries two additive trailing fields, ``oor_low=<f> oor_high=<f>``:
the OUT-OF-RANGE mass of the CDF the runner built from ``published``, i.e. ``cdf[0]`` (below
the lower bound) and ``1 - cdf[-1]`` (above the upper). The per-question aggregate marker
carries the same pair for the PUBLISHED distribution, then the pair as it stood before the
platform tail floor and the floor that moved it::

    NUMERIC_AGGREGATE: question=<id> qtype=numeric|date cdf_size=<n> oor_low=<f> oor_high=<f>
        oor_low_raw=<f> oor_high_raw=<f> tail_floor=<f>

Why the tails: the platform scores an out-of-range resolution against a fixed 0.05
reference, and on Mantic half of all resolved date questions and a quarter of discrete ones
resolved outside the displayed range, while this pipeline publishes exactly 1% there
whenever every percentile sits inside. On a Mantic question the published aggregate's open
tails are raised to ``MANTIC_OUT_OF_RANGE_TAIL_FLOOR`` (``numeric/out_of_range_floor.py``);
``oor_low_raw`` / ``oor_high_raw`` keep the aggregate's own tails and ``tail_floor`` is the
floor that moved an endpoint (``0`` on Metaculus, on a closed-bound question, or when the
tails were already at or above it), so the floor can be benchmarked on this bot's own
forecasts from the archive, and the member lines keep measuring what the models declared.

``raw`` is the value the extraction ladder read off the model's rationale, before any
clamp, renormalise or sanitise touched it. ``published`` is what the runner returns.
Both are compact JSON literals with NO whitespace, so the harvester regex takes each
with ``\\S+`` and a consumer always ``json.loads`` them, whatever the type:

* ``binary`` — a probability each: ``raw=0.005 published=0.02``
* ``multiple_choice`` — the option-probability vector in ``question.options`` order,
  the same order on both sides: ``raw=[0.9,0.005,0.095] published=[0.891,0.01,0.099]``.
  ``raw`` is ``ExtractionOutcome[McForecast].value.declared_probs``, NOT a read of the
  extracted ``PredictedOptionList`` (``McForecast.option_list``): that list is clamped on
  construction (our pre-construction clamp exists so ft's validator is a no-op), so the
  runner never holds the raw vector on the list itself. On the ladder's rarely-taken LLM rung the strict sub-path parses
  straight into an ft model, so there ``raw`` is the parser's output after that clamp.
* ``numeric`` — the declared ``[percentile, value]`` pairs with the percentile as the
  decimal in (0, 1) the block declares (``0.025``, not ``2.5``), ``published`` being
  the post-``sanitize_percentiles`` list: ``raw=[[0.025,9.2],[0.05,9.6],...]``

Why this exists (2026-09-02). Before it, no run-log marker carried a member's forecast
value: ``EXTRACTION_RUNG`` says which rung read the value, ``FORECASTERS_SURVIVED`` who
survived, ``EXTREME_CALL`` a probability only for a member already past the extreme band,
and the runner's own ``Forecasted URL`` line the CLAMPED value with no model name. The
only writer of a raw value was the published Metaculus comment, where each rationale's
fenced block carries it — and that comment is middle-trimmed at ``COMMENT_CHAR_LIMIT``,
only carries the block since 2026-05, and publishes a stacked question's members as
sub-blocks. The 2026-09-01 clip-threshold re-read (what a looser ``BINARY_PROB_MIN`` would
have done) could recover a raw binary probability for 74 of 451 resolved binary questions,
and nothing at all for the value a member declared on any question the comment trimmed
past. Logging the pair at the clamp point makes that a query over
``backtests/telemetry_archive/member_forecast.jsonl`` (harvested by ``make sync_telemetry``,
spec ``member_forecast`` in ``scripts/telemetry/markers.py``).

Scope notes:

* The line is emitted where the runner's post-processing happens, so on numeric and date
  questions it follows the CDF build (whose tails it reports) and precedes the unit-mismatch
  guard: a member that guard then withholds STILL leaves its line (its raw declaration is
  exactly what an audit of the guard needs), and the drop shows in ``FORECASTER_DROPS``. A
  member whose CDF build itself raised leaves no line, only the drop. Join against
  ``FORECASTERS_SURVIVED``'s ``models`` to restrict to members that reached the aggregate.
* The stacker's numeric line is emitted by ``aggregation_pipeline._run_stacking_numeric``
  rather than ``stacking.run_stacking_numeric``, because that is where its declared
  percentiles are sanitised; its binary and MC lines come from ``stacking.py``.
* Values are ``json.dumps``'d with the default shortest round-trip float repr and
  ``allow_nan=False`` — the ladder already guarantees finite values, so a NaN here is a
  bug upstream and should crash rather than write ``NaN`` into the archive.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from forecasting_tools import NumericDistribution, Percentile, PredictedOptionList

from metaculus_bot.question_types import QuestionType

MEMBER_FORECAST_ROLE_MEMBER = "member"
MEMBER_FORECAST_ROLE_STACKER = "stacker"

MemberValue = float | Sequence[float] | Sequence[Sequence[float]]
OutOfRangeMass = tuple[float, float]


def option_vector(option_list: PredictedOptionList) -> list[float]:
    """The option probabilities in the list's own order (``question.options`` order).

    A published-side read: the list is clamped on construction, so the declared vector
    comes from ``McForecast.declared_probs`` instead (module docstring).
    """
    return [float(option.probability) for option in option_list.predicted_options]


def percentile_pairs(percentiles: Sequence[Percentile]) -> list[list[float]]:
    """``[[percentile, value], ...]`` with the percentile as the declared decimal."""
    return [[float(p.percentile), float(p.value)] for p in percentiles]


def _compact_json(value: MemberValue) -> str:
    text = json.dumps(value, separators=(",", ":"), allow_nan=False)
    assert " " not in text, text  # whitespace would split the field under the harvester's \S+ capture
    return text


def out_of_range_mass(prediction: NumericDistribution) -> OutOfRangeMass:
    """``(below the lower bound, above the upper bound)`` mass of a built CDF: ``cdf[0]``, ``1 - cdf[-1]``."""
    heights = prediction.get_cdf()
    return float(heights[0].percentile), 1.0 - float(heights[-1].percentile)


def _out_of_range_suffix(out_of_range: OutOfRangeMass) -> str:
    oor_low, oor_high = out_of_range
    return f" oor_low={oor_low:.6f} oor_high={oor_high:.6f}"


def format_member_forecast_marker(
    *,
    question_id: int | None,
    model: str,
    role: str,
    qtype: QuestionType,
    raw: MemberValue,
    published: MemberValue,
    out_of_range: OutOfRangeMass | None = None,
) -> str:
    """Build one MEMBER_FORECAST line. Pure: reads values, returns the string.

    ``out_of_range`` is the built CDF's tail mass on a numeric or date question (module
    docstring); binary and MC lines have no CDF and carry no such fields.
    """
    line = (
        f"MEMBER_FORECAST: question={question_id} model={model} role={role} qtype={qtype} "
        f"raw={_compact_json(raw)} published={_compact_json(published)}"
    )
    if out_of_range is not None:
        line += _out_of_range_suffix(out_of_range)
    return line


def format_numeric_aggregate_marker(
    *,
    question_id: int | None,
    qtype: QuestionType,
    cdf_size: int,
    out_of_range: OutOfRangeMass,
    out_of_range_raw: OutOfRangeMass,
    tail_floor: float,
) -> str:
    """Build the per-question NUMERIC_AGGREGATE line for the published distribution.

    ``out_of_range`` is the PUBLISHED distribution's tail mass, ``out_of_range_raw`` the same pair
    before the platform tail floor, ``tail_floor`` the floor that moved an endpoint (module docstring).
    """
    raw_low, raw_high = out_of_range_raw
    return (
        f"NUMERIC_AGGREGATE: question={question_id} qtype={qtype} cdf_size={cdf_size}"
        f"{_out_of_range_suffix(out_of_range)}"
        f" oor_low_raw={raw_low:.6f} oor_high_raw={raw_high:.6f} tail_floor={tail_floor:.6f}"
    )
