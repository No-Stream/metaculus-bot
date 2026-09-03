"""Per-member MEMBER_FORECAST telemetry: every forecast value, raw and as published.

One INFO line per forecast VALUE a runner hands onward, for all three question types,
for every ensemble member and for the stacker::

    MEMBER_FORECAST: question=<id> model=<name> role=member|stacker qtype=<type> raw=<json> published=<json>

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

* The line is emitted where the runner's post-processing happens, so on numeric questions
  it precedes the unit-mismatch guard: a member that guard then withholds STILL leaves its
  line (its raw declaration is exactly what an audit of the guard needs), and the drop
  shows in ``FORECASTER_DROPS``. Join against ``FORECASTERS_SURVIVED``'s ``models`` to
  restrict to members that reached the aggregate.
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

from forecasting_tools import Percentile, PredictedOptionList

from metaculus_bot.question_types import QuestionType

MEMBER_FORECAST_ROLE_MEMBER = "member"
MEMBER_FORECAST_ROLE_STACKER = "stacker"

MemberValue = float | Sequence[float] | Sequence[Sequence[float]]


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


def format_member_forecast_marker(
    *,
    question_id: int | None,
    model: str,
    role: str,
    qtype: QuestionType,
    raw: MemberValue,
    published: MemberValue,
) -> str:
    """Build one MEMBER_FORECAST line. Pure: reads values, returns the string."""
    return (
        f"MEMBER_FORECAST: question={question_id} model={model} role={role} qtype={qtype} "
        f"raw={_compact_json(raw)} published={_compact_json(published)}"
    )
