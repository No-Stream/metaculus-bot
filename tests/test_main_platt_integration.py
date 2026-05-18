"""Integration tests for the Platt-scaling hook on ``_aggregate_predictions``.

The hook itself is ``TemplateForecaster._apply_platt_calibration`` (defined in
``main.py``). It is wired into every fresh-aggregation return point of
``_aggregate_predictions`` (binary, MC, numeric, stacker primary, stacker
fallback LLM, stacker MEDIAN-fallback) and DELIBERATELY NOT into the
STACKING base-combine re-entry block (``main.py:1145-1214``) where the inputs
were already calibrated by an upstream call to the hook.

These tests pin five behaviors:

1. With the env flag unset, the hook is a no-op (regression guard).
2. With the flag on but identity params, the hook is a no-op even with a cap.
3. With the flag on and non-identity params, the binary aggregation path
   returns ``apply_binary_platt(raw_median, params, max_abs_deviation=...)``.
4. With the flag on and non-identity params, the MC aggregation path returns
   per-option probabilities matching ``apply_mc_platt(combined, params, ...)``
   on the raw aggregated PredictedOptionList.
5. The stacker MEDIAN-fallback path (both stacker LLM attempts fail) STILL
   applies calibration to the resulting MEDIAN.
6. The STACKING base-combine re-entry (reasoned_predictions=None,
   research=None, single pre-stacked input) does NOT apply calibration.
"""

from __future__ import annotations

import math
from copy import deepcopy
from unittest.mock import MagicMock, Mock, patch

import pytest
from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
)
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOption,
    PredictedOptionList,
)

import main
from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import (
    AggregationStrategy,
    combine_binary_predictions,
    combine_multiple_choice_predictions,
)
from metaculus_bot.calibration import (
    PlattParams,
    apply_binary_platt,
    apply_mc_platt,
)
from metaculus_bot.constants import (
    BINARY_PROB_MIN,
    MC_PROB_MAX,
    MC_PROB_MIN,
    PLATT_BINARY_MAX_ABS_DEVIATION,
    PLATT_MC_MAX_ABS_DEVIATION,
)

# ---------------------------------------------------------------------------
# Bot / question fixtures
# ---------------------------------------------------------------------------


def _make_median_bot() -> TemplateForecaster:
    """Plain MEDIAN-strategy bot — exercises the non-stacking aggregation path."""
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    return TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        aggregation_strategy=AggregationStrategy.MEDIAN,
        llms={
            "forecasters": [test_llm],
            "default": test_llm,
            "parser": test_llm,
            "researcher": test_llm,
            "summarizer": test_llm,
        },
        is_benchmarking=True,
        min_forecasters_to_publish=1,
    )


def _make_stacking_bot(
    *,
    aggregation_strategy: AggregationStrategy = AggregationStrategy.STACKING,
    stacking_fallback_on_failure: bool = True,
) -> TemplateForecaster:
    """Bot configured for the STACKING (or CONDITIONAL_STACKING) aggregation path."""
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    return TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        aggregation_strategy=aggregation_strategy,
        llms={
            "forecasters": [test_llm, test_llm],
            "stacker": test_llm,
            "analyzer": test_llm,
            "default": test_llm,
            "parser": test_llm,
            "researcher": test_llm,
            "summarizer": test_llm,
        },
        is_benchmarking=True,
        stacking_fallback_on_failure=stacking_fallback_on_failure,
        min_forecasters_to_publish=1,
    )


def _make_binary_question(qid: int = 12345) -> MagicMock:
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = qid
    q.question_text = "Will it happen?"
    q.page_url = f"https://metaculus.com/questions/{qid}/"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    return q


def _make_mc_question(qid: int = 22222, n_options: int = 4) -> MagicMock:
    q = MagicMock(spec=MultipleChoiceQuestion)
    q.id_of_question = qid
    q.question_text = "Which option occurs?"
    q.page_url = f"https://metaculus.com/questions/{qid}/"
    q.options = [f"opt_{i}" for i in range(n_options)] if n_options > 4 else ["A", "B", "C", "D"][:n_options]
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    return q


def _make_mc_pred(probs: list[float], names: list[str] | None = None) -> PredictedOptionList:
    """Build a PredictedOptionList for the given probabilities (no normalization).

    Used both as input to ``_aggregate_predictions`` and to construct the
    expected reference output by combining + applying ``apply_mc_platt`` to a
    deep copy.
    """
    if names is None:
        names = ["A", "B", "C", "D"][: len(probs)] if len(probs) <= 4 else [f"opt_{i}" for i in range(len(probs))]
    if len(names) != len(probs):
        raise ValueError("names/probs length mismatch")
    return PredictedOptionList(
        predicted_options=[PredictedOption(option_name=n, probability=p) for n, p in zip(names, probs)]
    )


# ---------------------------------------------------------------------------
# 1. Flag unset -> no-op (regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calibration_off_no_change_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the env flag unset the hook returns the prediction unchanged.

    Even if the params are non-identity, the flag check at the top of
    ``_apply_platt_calibration`` short-circuits before the math runs. This is
    the regression guard that the recalibration is OFF by default.
    """
    monkeypatch.delenv("PLATT_CALIBRATION_ENABLED", raising=False)
    # Non-identity params, deliberately set, to prove the flag (not the
    # params) is what gates the no-op.
    monkeypatch.setattr(main, "BINARY_PLATT_PARAMS", PlattParams(bias=0.5, slope=1.5))

    bot = _make_median_bot()
    question = _make_binary_question()

    raw_expected = combine_binary_predictions([0.6, 0.7], AggregationStrategy.MEDIAN)
    result = await bot._aggregate_predictions([0.6, 0.7], question)

    assert result == raw_expected, f"flag unset should be a bytewise no-op; got {result!r}, expected {raw_expected!r}"


# ---------------------------------------------------------------------------
# 2. Flag on, identity params -> no-op (the cap can't move identity output)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calibration_on_identity_no_change_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag on + identity params: result still equals the raw aggregation.

    Identity params (bias=0, slope=1) produce ``p_adj == p_raw``; the cap is
    therefore a no-op too (deviation is exactly 0). The Platt apply does run
    a final ``[BINARY_PROB_MIN, BINARY_PROB_MAX]`` clamp, but raw values in
    [0.02, 0.98] survive that unchanged.
    """
    monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
    # main.BINARY_PLATT_PARAMS is identity by default in params.py, but pin it
    # explicitly so the test doesn't silently regress if defaults change.
    monkeypatch.setattr(main, "BINARY_PLATT_PARAMS", PlattParams.identity())

    bot = _make_median_bot()
    question = _make_binary_question()

    raw_expected = combine_binary_predictions([0.6, 0.7], AggregationStrategy.MEDIAN)
    result = await bot._aggregate_predictions([0.6, 0.7], question)

    assert result == raw_expected, (
        f"identity params should be a no-op even with flag on; got {result!r}, expected {raw_expected!r}"
    )


# ---------------------------------------------------------------------------
# 3. Flag on, non-identity params -> binary calibration applied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calibration_on_nonidentity_applies_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Binary aggregation returns the calibrated value of the raw median.

    Reference value is computed by calling ``apply_binary_platt`` directly
    (not mocked) on the raw median — the Platt apply is a pure function so
    the integration test is meaningful as an equality check.
    """
    monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
    params = PlattParams(bias=0.5, slope=1.5)
    monkeypatch.setattr(main, "BINARY_PLATT_PARAMS", params)

    bot = _make_median_bot()
    question = _make_binary_question()

    preds = [0.6, 0.7]
    raw_median = combine_binary_predictions(preds, AggregationStrategy.MEDIAN)
    expected = apply_binary_platt(raw_median, params, max_abs_deviation=PLATT_BINARY_MAX_ABS_DEVIATION)
    result = await bot._aggregate_predictions(preds, question)

    assert result == expected, (
        f"binary aggregation should equal apply_binary_platt(raw_median, params, cap); "
        f"got {result!r}, expected {expected!r} (raw_median={raw_median!r})"
    )
    # And explicitly: the calibration must actually move the result (else the
    # test is vacuous; raw_median=0.65 + bias/slope above produces ~0.806
    # uncapped, capped at raw+0.10 = 0.75).
    assert result != raw_median, (
        "non-identity params with non-zero deviation should change the output; "
        "if this fires, the cap or the params are wrong."
    )


# ---------------------------------------------------------------------------
# 4. Flag on, non-identity MC params -> per-option calibration applied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calibration_on_nonidentity_applies_mc(monkeypatch: pytest.MonkeyPatch) -> None:
    """4-option MC aggregation returns the per-option Platt-calibrated list.

    The expected output is built by combining the raw MC inputs (deep-copied
    so the bot's in-place option mutation doesn't pollute the reference) and
    then running ``apply_mc_platt`` directly with the patched params.
    """
    monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
    params = PlattParams(bias=0.3, slope=1.4)
    monkeypatch.setattr(main, "MC_PLATT_PARAMS", params)

    bot = _make_median_bot()
    question = _make_mc_question()

    pred_a = _make_mc_pred([0.50, 0.30, 0.15, 0.05])
    pred_b = _make_mc_pred([0.40, 0.35, 0.20, 0.05])

    # Build the expected reference. _aggregate_predictions mutates options in
    # place (via apply_mc_platt → option.probability = ...), so deep-copy the
    # inputs separately for the reference build.
    ref_inputs = [deepcopy(pred_a), deepcopy(pred_b)]
    combined = combine_multiple_choice_predictions(ref_inputs, AggregationStrategy.MEDIAN)
    expected_pol = apply_mc_platt(combined, params, max_abs_deviation=PLATT_MC_MAX_ABS_DEVIATION)
    expected_probs = {o.option_name: o.probability for o in expected_pol.predicted_options}

    result = await bot._aggregate_predictions([pred_a, pred_b], question)

    assert isinstance(result, PredictedOptionList)
    result_probs = {o.option_name: o.probability for o in result.predicted_options}

    assert set(result_probs) == set(expected_probs)
    for name, expected_p in expected_probs.items():
        assert result_probs[name] == pytest.approx(expected_p), (
            f"option {name!r}: got {result_probs[name]!r}, expected {expected_p!r}"
        )

    # And calibration should actually move the result vs. the raw combination
    # (else the test is vacuous).
    raw_combined = combine_multiple_choice_predictions([deepcopy(pred_a), deepcopy(pred_b)], AggregationStrategy.MEDIAN)
    raw_probs = {o.option_name: o.probability for o in raw_combined.predicted_options}
    assert any(abs(result_probs[n] - raw_probs[n]) > 1e-9 for n in result_probs), (
        "non-identity MC params should change at least one option probability."
    )


# ---------------------------------------------------------------------------
# 4b. Many-option MC, low-prob options can fall below BINARY_PROB_MIN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mc_many_options_can_fall_below_binary_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: MC calibration must not inherit the binary 0.02 floor.

    Builds a 10-option question with three options below the binary floor
    (0.005, 0.010, 0.015). The pre-fix path routed every option through
    ``apply_binary_platt``, which clamped each to >= 0.02 BEFORE MC
    renormalization — making the effective MC floor 0.02 instead of the
    documented 0.005, and inflating tiny options 4-10x. The fix routes
    options through the bound-free Platt helper and lets
    ``clamp_and_renormalize_mc`` apply the MC-correct ``[0.005, 0.995]``
    bounds at the tail.

    Reference is hand-computed (Platt math + MC-bounds clamp + renormalize)
    rather than calling ``apply_mc_platt`` itself, to avoid circularity.
    """
    monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
    params = PlattParams(bias=0.0, slope=1.2)
    monkeypatch.setattr(main, "MC_PLATT_PARAMS", params)
    # Keep the cap loose so the per-option Platt math, not the cap, is what
    # produces sub-0.02 outputs. The shipping cap is small enough that it
    # would otherwise pin tiny options to within 0.02-ish of their input.
    # 1.0 is loose enough that the cap never binds for any p in [0, 1].
    loose_cap = 1.0
    monkeypatch.setattr(main, "PLATT_MC_MAX_ABS_DEVIATION", loose_cap)

    bot = _make_median_bot()
    question = _make_mc_question(n_options=10)

    # Three options below the binary floor + seven absorbing the rest.
    small = [0.005, 0.010, 0.015]  # sums to 0.030
    large = [0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.07]  # sums to 0.97
    raw = small + large
    assert sum(raw) == pytest.approx(1.0, abs=1e-12)
    names = [f"opt_{i}" for i in range(len(raw))]

    pred_a = _make_mc_pred(raw, names=names)
    pred_b = _make_mc_pred(raw, names=names)

    # Reference: combine MEDIAN (identical inputs → equals raw), then apply
    # Platt math by hand, apply the loose cap (no-op for p in [0, 1]), then
    # clamp to MC bounds and renormalize. Hand-rolled rather than calling
    # apply_mc_platt to avoid circularity with the code under test.
    combined = combine_multiple_choice_predictions([deepcopy(pred_a), deepcopy(pred_b)], AggregationStrategy.MEDIAN)
    raw_combined = [o.probability for o in combined.predicted_options]
    per_option_unclipped = [
        1.0 / (1.0 + math.exp(-(params.bias + params.slope * math.log(p / (1.0 - p))))) for p in raw_combined
    ]
    per_option = [max(p - loose_cap, min(p + loose_cap, q)) for p, q in zip(raw_combined, per_option_unclipped)]
    clamped = [max(MC_PROB_MIN, min(MC_PROB_MAX, q)) for q in per_option]
    total = sum(clamped)
    expected_probs = {n: q / total for n, q in zip(names, clamped)}

    result = await bot._aggregate_predictions([pred_a, pred_b], question)

    assert isinstance(result, PredictedOptionList)
    result_probs = {o.option_name: o.probability for o in result.predicted_options}

    for name, exp in expected_probs.items():
        assert result_probs[name] == pytest.approx(exp, abs=1e-9), (
            f"option {name!r}: got {result_probs[name]!r}, expected {exp!r}"
        )
    assert sum(result_probs.values()) == pytest.approx(1.0, abs=1e-9)

    # Load-bearing regression assertion: at least one of the three small-input
    # options must end up below the binary floor (0.02). If the binary clamp
    # leaks into the MC path, every option gets pinned at >= 0.02 and this
    # fires.
    low_probs = [result_probs[names[i]] for i in range(3)]
    assert any(p < BINARY_PROB_MIN for p in low_probs), (
        f"At least one small-input option should fall below the binary floor "
        f"({BINARY_PROB_MIN}); got {low_probs!r}. If this fails, the binary "
        f"clamp leaked into the MC path."
    )


# ---------------------------------------------------------------------------
# 5. Stacker MEDIAN-fallback path still applies calibration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stacker_median_fallback_applies_calibration_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both stacker LLM attempts fail → MEDIAN aggregation → calibration applied.

    This exercises the failure path at ``main.py:1304-1323`` (the
    ``STACKER_FALLBACK_FAILED`` branch). The MEDIAN value of [0.4, 0.6] is
    0.5; with bias=0.5/slope=1.5 the unconstrained Platt would push to ~0.622,
    but |0.622 - 0.5| = 0.122 exceeds the binary cap of 0.10, so the cap
    binds and the result is exactly 0.6000.
    """
    monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
    params = PlattParams(bias=0.5, slope=1.5)
    monkeypatch.setattr(main, "BINARY_PLATT_PARAMS", params)

    bot = _make_stacking_bot()
    question = _make_binary_question()

    preds = [0.4, 0.6]
    raw_median = combine_binary_predictions(preds, AggregationStrategy.MEDIAN)
    expected = apply_binary_platt(raw_median, params, max_abs_deviation=PLATT_BINARY_MAX_ABS_DEVIATION)

    # Force both the primary and the fallback _run_stacking calls to fail so
    # the MEDIAN-fallback branch is taken. RuntimeError is below
    # asyncio.CancelledError (BaseException in 3.11+) and matches what
    # test_stacking.py uses in the equivalent failure-path test.
    with patch.object(bot, "_run_stacking", side_effect=RuntimeError("stacking failed")):
        result = await bot._aggregate_predictions(
            predictions=preds,
            question=question,
            research="test research",
            reasoned_predictions=[Mock(), Mock()],
        )

    assert result == expected, (
        f"stacker MEDIAN-fallback should apply Platt calibration; "
        f"got {result!r}, expected apply_binary_platt(raw_median={raw_median!r}, params, cap)={expected!r}"
    )
    # The calibration must have actually moved the value (else the assertion
    # above is vacuous — MEDIAN already equals raw_median).
    assert result != raw_median, (
        "stacker MEDIAN-fallback with non-identity params should change the "
        "result; if this fires, calibration is being skipped on the "
        "MEDIAN-fallback path."
    )
    # And the fallback path was actually exercised (sanity).
    assert bot._stacker_fallback_failed_count == 1


# ---------------------------------------------------------------------------
# 6. STACKING base-combine re-entry does NOT apply calibration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stacking_base_combine_reentry_does_not_double_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-entry from the base framework with one pre-stacked input is bare passthrough.

    When ``_aggregate_predictions`` is called with ``research=None`` AND
    ``reasoned_predictions=None`` AND a non-empty ``predictions`` list, the
    framework is asking us to combine an already-stacked output. Those inputs
    were calibrated by the FRESH-aggregation call that produced them, so
    re-applying calibration here would double-apply. The correct behavior
    for a single-input re-entry is to return the input as-is.
    """
    monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
    params = PlattParams(bias=0.5, slope=1.5)
    monkeypatch.setattr(main, "BINARY_PLATT_PARAMS", params)

    bot = _make_stacking_bot(aggregation_strategy=AggregationStrategy.STACKING)
    question = _make_binary_question()

    # The re-entry signature: research=None, reasoned_predictions=None.
    result = await bot._aggregate_predictions(
        predictions=[0.42],
        question=question,
        research=None,
        reasoned_predictions=None,
    )

    assert result == 0.42, (
        f"STACKING base-combine re-entry must return the pre-stacked value as-is; "
        f"got {result!r}, expected 0.42 (Platt would have shifted to "
        f"{apply_binary_platt(0.42, params, max_abs_deviation=PLATT_BINARY_MAX_ABS_DEVIATION)!r})"
    )


@pytest.mark.asyncio
async def test_calibration_on_primary_stacker_success_applies_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primary stacker succeeds → calibration applies to its output (line 1262).

    This is the most common production path under STACKING /
    CONDITIONAL_STACKING-with-disagreement. We mock ``_run_stacking`` to
    return a raw stacker float and assert the bot returns the calibrated
    value. ``side_effect=[value]`` works on async methods (same pattern as
    ``test_stacking.py:481``).
    """
    monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
    params = PlattParams(bias=0.5, slope=1.5)
    monkeypatch.setattr(main, "BINARY_PLATT_PARAMS", params)

    bot = _make_stacking_bot()
    question = _make_binary_question()

    stacker_raw = 0.42
    expected = apply_binary_platt(stacker_raw, params, max_abs_deviation=PLATT_BINARY_MAX_ABS_DEVIATION)

    with patch.object(bot, "_run_stacking", side_effect=[stacker_raw]):
        result = await bot._aggregate_predictions(
            predictions=[0.4, 0.6],
            question=question,
            research="research",
            reasoned_predictions=[Mock(), Mock()],
        )

    assert result == expected, f"primary stacker success should apply Platt; got {result!r}, expected {expected!r}"
    # Sanity: primary path was used (no fallback counters bumped).
    assert bot._stacker_outcome[question.id_of_question] == "primary"
    assert bot._stacker_primary_failed_count == 0
    assert bot._stacker_fallback_used_count == 0


@pytest.mark.asyncio
async def test_calibration_on_fallback_llm_success_applies_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primary stacker fails, fallback LLM succeeds → calibration applies (line 1295)."""
    monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
    params = PlattParams(bias=0.5, slope=1.5)
    monkeypatch.setattr(main, "BINARY_PLATT_PARAMS", params)

    bot = _make_stacking_bot()
    question = _make_binary_question()

    stacker_raw = 0.42
    expected = apply_binary_platt(stacker_raw, params, max_abs_deviation=PLATT_BINARY_MAX_ABS_DEVIATION)

    # side_effect sequence: primary raises, fallback returns the raw stacker
    # value. Same pattern as test_stacking.py:481.
    with patch.object(
        bot,
        "_run_stacking",
        side_effect=[RuntimeError("primary failed"), stacker_raw],
    ):
        result = await bot._aggregate_predictions(
            predictions=[0.4, 0.6],
            question=question,
            research="research",
            reasoned_predictions=[Mock(), Mock()],
        )

    assert result == expected
    assert bot._stacker_primary_failed_count == 1
    assert bot._stacker_fallback_used_count == 1
    assert bot._stacker_fallback_failed_count == 0
    assert bot._stacker_outcome[question.id_of_question] == "fallback_llm"


@pytest.mark.asyncio
async def test_stacker_median_fallback_applies_calibration_mc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both stacker LLM attempts fail → MEDIAN MC aggregation → calibration applied (line 1331).

    Mirrors ``test_stacker_median_fallback_applies_calibration_binary`` but
    for the MC arm at main.py:1331. Reference output is built by combining
    the inputs with MEDIAN and applying ``apply_mc_platt`` directly.
    """
    monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
    params = PlattParams(bias=0.3, slope=1.4)
    monkeypatch.setattr(main, "MC_PLATT_PARAMS", params)

    bot = _make_stacking_bot()
    question = _make_mc_question()

    pred_a = _make_mc_pred([0.50, 0.30, 0.15, 0.05])
    pred_b = _make_mc_pred([0.40, 0.35, 0.20, 0.05])

    # Reference: combine + apply MC Platt with the cap. Deep-copy inputs so
    # the bot's in-place mutation doesn't pollute the reference build.
    ref_inputs = [deepcopy(pred_a), deepcopy(pred_b)]
    combined = combine_multiple_choice_predictions(ref_inputs, AggregationStrategy.MEDIAN)
    expected_pol = apply_mc_platt(combined, params, max_abs_deviation=PLATT_MC_MAX_ABS_DEVIATION)
    expected_probs = {o.option_name: o.probability for o in expected_pol.predicted_options}

    with patch.object(bot, "_run_stacking", side_effect=RuntimeError("stacking failed")):
        result = await bot._aggregate_predictions(
            predictions=[pred_a, pred_b],
            question=question,
            research="research",
            reasoned_predictions=[Mock(), Mock()],
        )

    assert isinstance(result, PredictedOptionList)
    result_probs = {o.option_name: o.probability for o in result.predicted_options}
    for name, exp in expected_probs.items():
        assert result_probs[name] == pytest.approx(exp), (
            f"option {name!r}: got {result_probs[name]!r}, expected {exp!r}"
        )
    assert bot._stacker_fallback_failed_count == 1
    assert bot._stacker_outcome[question.id_of_question] == "fallback_median"


@pytest.mark.asyncio
async def test_conditional_stacking_base_combine_reentry_does_not_double_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same as above for CONDITIONAL_STACKING — both strategies share the re-entry guard."""
    monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
    params = PlattParams(bias=0.5, slope=1.5)
    monkeypatch.setattr(main, "BINARY_PLATT_PARAMS", params)

    bot = _make_stacking_bot(aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING)
    question = _make_binary_question()

    result = await bot._aggregate_predictions(
        predictions=[0.42],
        question=question,
        research=None,
        reasoned_predictions=None,
    )

    assert result == 0.42, (
        f"CONDITIONAL_STACKING base-combine re-entry must return the pre-stacked "
        f"value as-is; got {result!r}, expected 0.42."
    )


@pytest.mark.asyncio
async def test_conditional_stacking_skip_path_applies_platt_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F16: CONDITIONAL_STACKING low-spread (skip) path must apply Platt calibration.

    The CS-skip branch returns raw per-forecaster predictions; the parent class
    then re-enters ``_aggregate_predictions`` with ``research=None`` and
    ``reasoned_predictions=None``, hitting the multi-input base-combine block.
    Pre-fix that block produced an UN-calibrated MEDIAN — meaning low-disagreement
    questions silently bypassed Platt while high-disagreement questions (which
    flow through the stacker fresh-aggregation path) had it applied. That
    asymmetry contaminates any treatment-effect cut.

    After the fix, the base-combine block applies Platt when the strategy is
    CONDITIONAL_STACKING and the input list is the multi-element CS-skip case.
    The single-input case (high-spread stacker output) remains a no-op
    passthrough since the stacker fresh-aggregation already calibrated it.
    """
    monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
    params = PlattParams(bias=0.5, slope=1.2)
    monkeypatch.setattr(main, "BINARY_PLATT_PARAMS", params)

    bot = _make_stacking_bot(aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING)
    question = _make_binary_question()
    bot._register_expected_base_combine(question)

    preds = [0.45, 0.50, 0.55]
    raw_median = combine_binary_predictions(preds, AggregationStrategy.MEDIAN)
    expected = apply_binary_platt(raw_median, params, max_abs_deviation=PLATT_BINARY_MAX_ABS_DEVIATION)

    result = await bot._aggregate_predictions(
        predictions=preds,
        question=question,
        research=None,
        reasoned_predictions=None,
    )

    assert result == expected, (
        f"CONDITIONAL_STACKING skip-path multi-input re-entry must apply Platt to "
        f"the median of raw forecasters; got {result!r}, expected {expected!r} "
        f"(raw_median={raw_median!r})."
    )
    assert result != raw_median, (
        "Non-identity params should move the result; if this fires, calibration "
        "is being skipped on the CS low-spread path."
    )


@pytest.mark.asyncio
async def test_conditional_stacking_skip_path_applies_platt_mc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F16 MC arm: CS low-spread re-entry must apply MC Platt to the combined options.

    Mirrors the binary case for the MC arm of the base-combine block.
    """
    monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
    params = PlattParams(bias=0.3, slope=1.4)
    monkeypatch.setattr(main, "MC_PLATT_PARAMS", params)

    bot = _make_stacking_bot(aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING)
    question = _make_mc_question()
    bot._register_expected_base_combine(question)

    pred_a = _make_mc_pred([0.50, 0.30, 0.15, 0.05])
    pred_b = _make_mc_pred([0.40, 0.35, 0.20, 0.05])

    ref_inputs = [deepcopy(pred_a), deepcopy(pred_b)]
    combined = combine_multiple_choice_predictions(ref_inputs, AggregationStrategy.MEDIAN)
    expected_pol = apply_mc_platt(combined, params, max_abs_deviation=PLATT_MC_MAX_ABS_DEVIATION)
    expected_probs = {o.option_name: o.probability for o in expected_pol.predicted_options}

    result = await bot._aggregate_predictions(
        predictions=[pred_a, pred_b],
        question=question,
        research=None,
        reasoned_predictions=None,
    )

    assert isinstance(result, PredictedOptionList)
    result_probs = {o.option_name: o.probability for o in result.predicted_options}
    for name, exp in expected_probs.items():
        assert result_probs[name] == pytest.approx(exp), (
            f"option {name!r}: got {result_probs[name]!r}, expected {exp!r}"
        )


@pytest.mark.asyncio
async def test_stacking_base_combine_reentry_multi_input_does_not_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STACKING control: multi-input re-entry under STACKING is the per-research-report
    re-aggregation of already-calibrated stacker outputs. Re-applying Platt would
    double-apply, so it remains a no-op even with the F16 fix (which is
    CS-only). Pin this so the F16 fix doesn't accidentally regress STACKING.
    """
    monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
    params = PlattParams(bias=0.5, slope=1.5)
    monkeypatch.setattr(main, "BINARY_PLATT_PARAMS", params)

    bot = _make_stacking_bot(aggregation_strategy=AggregationStrategy.STACKING)
    question = _make_binary_question()
    bot._register_expected_base_combine(question)

    preds = [0.42, 0.48]
    raw_mean = combine_binary_predictions(preds, AggregationStrategy.MEAN)

    result = await bot._aggregate_predictions(
        predictions=preds,
        question=question,
        research=None,
        reasoned_predictions=None,
    )

    assert result == raw_mean, (
        f"STACKING multi-input re-entry must NOT apply Platt (inputs are already "
        f"calibrated stacker outputs); got {result!r}, expected raw MEAN={raw_mean!r}."
    )
