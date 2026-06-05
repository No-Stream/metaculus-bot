"""Tests for the ablation forecaster runner (single-pass per question).

Mocks ``TemplateForecaster._make_prediction`` so no live API calls happen.
The runner instantiates a per-(question, model) bot with research_cache
populated, then calls ``_make_prediction`` directly under the window patch.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile

from main import TemplateForecaster
from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from metaculus_bot.aggregation_strategies import AggregationStrategy

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


_OPEN = datetime(2026, 1, 1)
_RESOLVE = datetime(2026, 6, 1)


def _make_binary_question(qid: int = 1234) -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will X happen?",
        id_of_post=qid,
        id_of_question=qid,
        page_url=f"https://example.com/q/{qid}",
        background_info="Background",
        resolution_criteria="Resolves YES if X.",
        fine_print="",
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


def _make_mc_question(qid: int = 2345) -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Which option?",
        id_of_post=qid,
        id_of_question=qid,
        page_url=f"https://example.com/q/{qid}",
        background_info="Background",
        resolution_criteria="Resolves to one of the listed options.",
        fine_print="",
        options=["A", "B", "C"],
        option_is_ordered=False,
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


def _make_numeric_question(qid: int = 3456) -> NumericQuestion:
    return NumericQuestion(
        question_text="What value?",
        id_of_post=qid,
        id_of_question=qid,
        page_url=f"https://example.com/q/{qid}",
        background_info="Background",
        resolution_criteria="Numeric value resolves.",
        fine_print="",
        unit_of_measure="people",
        open_upper_bound=False,
        open_lower_bound=False,
        upper_bound=100.0,
        lower_bound=0.0,
        zero_point=None,
        cdf_size=201,
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


@pytest.fixture
def cache(tmp_path: Path) -> AblationCache:
    return AblationCache(tmp_path / "abl")


def _make_forecaster_llms(count: int | None = None) -> list[GeneralLlm]:
    """Build a list of mock GeneralLlm instances with distinct model slugs.

    Uses the actual ``FREE_FORECASTER_MODELS`` slugs so cache keys match the
    production lineup; test scenarios never touch the real LLMs (we mock
    ``_make_prediction``, never call ``.invoke``). With ``count=None`` (the
    default), returns one mock per model in the live lineup so tests don't
    have to track lineup-size changes.
    """
    from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS

    slugs = FREE_FORECASTER_MODELS if count is None else FREE_FORECASTER_MODELS[:count]
    llms: list[GeneralLlm] = []
    for slug in slugs:
        llm = MagicMock(spec=GeneralLlm)
        llm.model = slug
        llms.append(llm)
    return llms


@pytest.fixture
def parser_llm() -> GeneralLlm:
    """Mock parser pinned to whatever the live ``FREE_PARSER_MODEL`` is.

    Avoids brittleness when the parser model is swapped in the lineup
    (e.g., gpt-oss-120b → gemma-4 in task #16).
    """
    from metaculus_bot.ablation.forecaster_lineup import FREE_PARSER_MODEL

    parser = MagicMock(spec=GeneralLlm)
    parser.model = FREE_PARSER_MODEL
    return parser


@pytest.fixture
def six_forecaster_llms() -> list[GeneralLlm]:
    """Historical fixture name; returns one mock per live FREE_FORECASTER_MODELS entry.

    Despite the "six" name, this is now the *full lineup size* — kept under
    the original name to minimize test-touch churn. The actual count is
    pinned by ``test_ablation_forecaster_lineup.test_free_forecaster_models_count_is_four``;
    consumers here just iterate the returned list rather than asserting a
    specific count.
    """
    return _make_forecaster_llms()


# ---------------------------------------------------------------------------
# Serialize / deserialize round-trips
# ---------------------------------------------------------------------------


def test_serialize_binary_prediction_value() -> None:
    from metaculus_bot.ablation.forecasters import deserialize_prediction_value, serialize_prediction_value

    q = _make_binary_question()
    payload = serialize_prediction_value(0.42, "binary")
    assert payload == {"type": "binary", "prob": 0.42}

    restored = deserialize_prediction_value(payload, q)
    assert restored == 0.42


def test_serialize_mc_prediction_value() -> None:
    from metaculus_bot.ablation.forecasters import deserialize_prediction_value, serialize_prediction_value

    options = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="A", probability=0.5),
            PredictedOption(option_name="B", probability=0.3),
            PredictedOption(option_name="C", probability=0.2),
        ]
    )
    q = _make_mc_question()
    payload = serialize_prediction_value(options, "multiple_choice")
    assert payload["type"] == "multiple_choice"
    assert {o["option_name"] for o in payload["options"]} == {"A", "B", "C"}
    assert sum(o["probability"] for o in payload["options"]) == pytest.approx(1.0)

    restored = deserialize_prediction_value(payload, q)
    assert isinstance(restored, PredictedOptionList)
    restored_dict = {o.option_name: o.probability for o in restored.predicted_options}
    assert restored_dict == {"A": 0.5, "B": 0.3, "C": 0.2}


def test_serialize_numeric_prediction_value() -> None:
    """Serialize a NumericDistribution to the full-payload schema and round-trip back.

    The serialize function takes a ``NumericDistribution`` (Pydantic model) — passing
    the bare ``list[Percentile]`` was the original tuple-bug shape. The payload must
    carry both ``declared_percentiles`` (11 inputs) AND ``cdf_probabilities`` (the
    computed 201-point CDF), so deserialize can reconstruct a PchipNumericDistribution
    without re-running ``build_numeric_distribution``.
    """
    from forecasting_tools import NumericDistribution

    from metaculus_bot.ablation.forecasters import deserialize_prediction_value, serialize_prediction_value
    from metaculus_bot.numeric.pchip_processing import create_pchip_numeric_distribution

    percentiles = [
        Percentile(value=v, percentile=p)
        for v, p in zip(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 99.0],
            [0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.975],
        )
    ]
    q = _make_numeric_question()
    # Build a real PchipNumericDistribution with a deterministic 201-point CDF so
    # we can compare round-tripped CDFs exactly.
    pchip_cdf = [i / 200 for i in range(201)]
    distribution = create_pchip_numeric_distribution(
        pchip_cdf=pchip_cdf,
        percentile_list=percentiles,
        question=q,
        zero_point=q.zero_point,
    )

    payload = serialize_prediction_value(distribution, "numeric")
    assert payload["type"] == "numeric"
    assert len(payload["declared_percentiles"]) == 11
    assert len(payload["cdf_probabilities"]) == 201
    assert payload["lower_bound"] == 0.0
    assert payload["upper_bound"] == 100.0
    assert payload["open_lower_bound"] is False
    assert payload["open_upper_bound"] is False
    assert payload["zero_point"] is None
    assert payload["cdf_size"] == 201

    restored = deserialize_prediction_value(payload, q)
    assert isinstance(restored, NumericDistribution)
    # Round-trip equivalence: cdf probabilities preserved exactly.
    restored_probs = [p.percentile for p in restored.cdf]
    assert restored_probs == pytest.approx(pchip_cdf)
    # Declared percentiles preserved.
    assert len(restored.declared_percentiles) == 11
    assert restored.declared_percentiles[0].value == 10.0
    assert restored.declared_percentiles[0].percentile == 0.025


def test_deserialize_rejects_old_payload_shape() -> None:
    """Old payloads (``percentiles`` key, no ``cdf_probabilities``) are rejected loudly.

    Cached payloads predating the full-CDF round-trip must not silently flow into
    the stacker — they would force re-derivation of the CDF via build_numeric_distribution,
    coupling cached artifacts to whatever pipeline code happens to be loaded.
    """
    from metaculus_bot.ablation.forecasters import deserialize_prediction_value

    q = _make_numeric_question()
    old_payload = {
        "type": "numeric",
        "percentiles": [{"percentile": 0.5, "value": 50.0}],
    }
    with pytest.raises(ValueError, match="--force-stages forecast"):
        deserialize_prediction_value(old_payload, q)


# ---------------------------------------------------------------------------
# run_forecasters_for_question
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_forecasters_returns_payload_per_model(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
) -> None:
    """Mocked _make_prediction returns canned ReasonedPrediction; runner returns 6 entries."""
    from metaculus_bot.ablation.forecasters import run_forecasters_for_question

    q = _make_binary_question(qid=1001)
    canned = ReasonedPrediction(prediction_value=0.42, reasoning="rationale")

    with patch.object(TemplateForecaster, "_make_prediction", new=AsyncMock(return_value=canned)):
        result = await run_forecasters_for_question(
            q,
            "research blob",
            cache,
            forecaster_llms=six_forecaster_llms,
            parser_llm=parser_llm,
        )

    assert len(result) == len(six_forecaster_llms)
    expected_keys = {model_slug_to_filename(llm.model) for llm in six_forecaster_llms}
    assert set(result.keys()) == expected_keys


@pytest.mark.asyncio
async def test_payload_has_required_fields(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
) -> None:
    from metaculus_bot.ablation.forecasters import run_forecasters_for_question

    q = _make_binary_question(qid=1002)
    canned = ReasonedPrediction(prediction_value=0.42, reasoning="rationale")

    with patch.object(TemplateForecaster, "_make_prediction", new=AsyncMock(return_value=canned)):
        result = await run_forecasters_for_question(
            q,
            "research blob",
            cache,
            forecaster_llms=six_forecaster_llms,
            parser_llm=parser_llm,
        )

    for slug, payload in result.items():
        assert "model" in payload
        assert "prediction_value" in payload
        assert "reasoning" in payload
        assert "errors" in payload
        assert "ran_at" in payload
        assert "duration_seconds" in payload
        assert payload["reasoning"] == "rationale"
        # ran_at should be ISO-parseable
        datetime.fromisoformat(payload["ran_at"])
        assert isinstance(payload["duration_seconds"], float)


@pytest.mark.asyncio
async def test_cache_hit_skips_make_prediction(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
) -> None:
    """One model pre-cached → _make_prediction called for the OTHER 5."""
    from metaculus_bot.ablation.forecasters import run_forecasters_for_question

    q = _make_binary_question(qid=1003)
    canned = ReasonedPrediction(prediction_value=0.42, reasoning="rationale")

    cached_model = six_forecaster_llms[0].model
    cached_slug = model_slug_to_filename(cached_model)
    cache.write_forecaster_output(
        qid=q.id_of_question,
        model_slug=cached_slug,
        payload={
            "model": cached_model,
            "prediction_value": {"type": "binary", "prob": 0.7},
            "reasoning": "cached rationale",
            "errors": [],
            "ran_at": "2026-01-01T00:00:00",
            "duration_seconds": 1.5,
        },
    )

    mock_make = AsyncMock(return_value=canned)
    with patch.object(TemplateForecaster, "_make_prediction", new=mock_make):
        result = await run_forecasters_for_question(
            q,
            "research blob",
            cache,
            forecaster_llms=six_forecaster_llms,
            parser_llm=parser_llm,
        )

    # _make_prediction was called once for each non-cached forecaster.
    assert mock_make.await_count == len(six_forecaster_llms) - 1
    # Cached payload returned for the cached model.
    assert result[cached_slug]["reasoning"] == "cached rationale"
    assert result[cached_slug]["prediction_value"]["prob"] == 0.7


@pytest.mark.asyncio
async def test_force_true_re_runs_all_forecasters(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
) -> None:
    """force=True ignores cache; _make_prediction called for all 6."""
    from metaculus_bot.ablation.forecasters import run_forecasters_for_question

    q = _make_binary_question(qid=1004)
    canned = ReasonedPrediction(prediction_value=0.42, reasoning="fresh rationale")

    # Pre-populate cache for ALL forecasters.
    for llm in six_forecaster_llms:
        cache.write_forecaster_output(
            qid=q.id_of_question,
            model_slug=model_slug_to_filename(llm.model),
            payload={
                "model": llm.model,
                "prediction_value": {"type": "binary", "prob": 0.99},
                "reasoning": "stale",
                "errors": [],
                "ran_at": "2026-01-01T00:00:00",
                "duration_seconds": 1.0,
            },
        )

    mock_make = AsyncMock(return_value=canned)
    with patch.object(TemplateForecaster, "_make_prediction", new=mock_make):
        result = await run_forecasters_for_question(
            q,
            "research blob",
            cache,
            forecaster_llms=six_forecaster_llms,
            parser_llm=parser_llm,
            force=True,
        )

    assert mock_make.await_count == len(six_forecaster_llms)
    # All payloads have the fresh reasoning, not the stale cached one.
    for payload in result.values():
        assert payload["reasoning"] == "fresh rationale"


@pytest.mark.asyncio
async def test_window_patch_active_during_make_prediction(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
) -> None:
    """During _make_prediction the window patch context manager is active."""
    from metaculus_bot.ablation import window_patch as wp_module
    from metaculus_bot.ablation.forecasters import run_forecasters_for_question

    q = _make_binary_question(qid=1005)
    flags_during_call: list[bool] = []

    async def observer(self, question, research, llm) -> ReasonedPrediction:  # type: ignore[no-untyped-def]  # noqa: ASYNC124
        flags_during_call.append(wp_module._window_patch_active)
        return ReasonedPrediction(prediction_value=0.42, reasoning="rationale")  # noqa: ASYNC910

    with patch.object(TemplateForecaster, "_make_prediction", new=observer):
        await run_forecasters_for_question(
            q,
            "research blob",
            cache,
            forecaster_llms=six_forecaster_llms,
            parser_llm=parser_llm,
        )

    # Patch must be active during every _make_prediction call.
    assert flags_during_call, "no _make_prediction calls observed"
    assert all(flags_during_call), "window patch was not active during _make_prediction"
    # And inactive after the runner returns.
    assert wp_module._window_patch_active is False


@pytest.mark.asyncio
async def test_per_forecaster_failure_caches_error_and_continues(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
) -> None:
    """First forecaster raises; the other 5 succeed; error cached, batch continues."""
    from metaculus_bot.ablation.forecasters import run_forecasters_for_question

    q = _make_binary_question(qid=1006)
    failing_model = six_forecaster_llms[0].model
    canned = ReasonedPrediction(prediction_value=0.42, reasoning="rationale")

    async def selective(self, question, research, llm) -> ReasonedPrediction:  # type: ignore[no-untyped-def]  # noqa: ASYNC124
        if llm.model == failing_model:
            raise RuntimeError("model exploded")
        return canned  # noqa: ASYNC910

    with patch.object(TemplateForecaster, "_make_prediction", new=selective):
        result = await run_forecasters_for_question(
            q,
            "research blob",
            cache,
            forecaster_llms=six_forecaster_llms,
            parser_llm=parser_llm,
        )

    failing_slug = model_slug_to_filename(failing_model)
    assert failing_slug in result
    failed_payload = result[failing_slug]
    assert failed_payload["prediction_value"] is None
    assert failed_payload["errors"]
    assert "RuntimeError" in failed_payload["errors"][0]
    assert "model exploded" in failed_payload["errors"][0]

    # Other forecasters succeeded.
    other_slugs = {model_slug_to_filename(llm.model) for llm in six_forecaster_llms[1:]}
    for slug in other_slugs:
        assert result[slug]["prediction_value"] is not None
        assert result[slug]["errors"] == []

    # Failed payload was persisted to cache too.
    on_disk = cache.read_forecaster_output(qid=q.id_of_question, model_slug=failing_slug)
    assert on_disk is not None
    assert on_disk["prediction_value"] is None


@pytest.mark.asyncio
async def test_research_cache_populated_in_bot(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
) -> None:
    """TemplateForecaster.__init__ receives research_cache={qid: blob}; run_research returns blob without providers."""
    from metaculus_bot.ablation.forecasters import run_forecasters_for_question

    q = _make_binary_question(qid=1007)
    captured_kwargs: list[dict] = []

    real_init = TemplateForecaster.__init__

    def init_spy(self, **kwargs):  # type: ignore[no-untyped-def]
        captured_kwargs.append({"research_cache": kwargs.get("research_cache")})
        return real_init(self, **kwargs)

    canned = ReasonedPrediction(prediction_value=0.42, reasoning="rationale")
    with (
        patch.object(TemplateForecaster, "__init__", new=init_spy),
        patch.object(TemplateForecaster, "_make_prediction", new=AsyncMock(return_value=canned)),
    ):
        await run_forecasters_for_question(
            q,
            "research blob",
            cache,
            forecaster_llms=six_forecaster_llms,
            parser_llm=parser_llm,
        )

    # Every bot instantiated for this question got research_cache={qid: blob}.
    assert captured_kwargs
    for kwargs in captured_kwargs:
        rc = kwargs["research_cache"]
        assert rc == {q.id_of_question: "research blob"}


@pytest.mark.asyncio
async def test_run_research_short_circuits_on_cached_blob(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
) -> None:
    """With research_cache populated, calling bot.run_research returns the cached blob without invoking providers."""
    from metaculus_bot.ablation.forecasters import _build_bot

    q = _make_binary_question(qid=1008)
    bot = _build_bot(
        question=q,
        research_blob="cached research",
        forecaster_llm=six_forecaster_llms[0],
        parser_llm=parser_llm,
    )

    # Sentinel: if providers WOULD run, this would fail.
    bot._select_research_providers = MagicMock(  # ty: ignore[invalid-assignment]
        side_effect=AssertionError("providers must not run")
    )  # type: ignore[method-assign]

    research = await bot.run_research(q)
    assert research == "cached research"


@pytest.mark.asyncio
async def test_aggregation_strategy_is_mean_so_stacker_doesnt_fire(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
) -> None:
    from metaculus_bot.ablation.forecasters import run_forecasters_for_question

    q = _make_binary_question(qid=1009)
    captured: list[object] = []

    real_init = TemplateForecaster.__init__

    def init_spy(self, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(kwargs.get("aggregation_strategy"))
        return real_init(self, **kwargs)

    canned = ReasonedPrediction(prediction_value=0.42, reasoning="r")
    with (
        patch.object(TemplateForecaster, "__init__", new=init_spy),
        patch.object(TemplateForecaster, "_make_prediction", new=AsyncMock(return_value=canned)),
    ):
        await run_forecasters_for_question(
            q,
            "research blob",
            cache,
            forecaster_llms=six_forecaster_llms,
            parser_llm=parser_llm,
        )

    assert captured
    assert all(strategy == AggregationStrategy.MEAN for strategy in captured)


@pytest.mark.asyncio
async def test_is_benchmarking_true_in_constructor(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
) -> None:
    from metaculus_bot.ablation.forecasters import run_forecasters_for_question

    q = _make_binary_question(qid=1010)
    captured: list[object] = []

    real_init = TemplateForecaster.__init__

    def init_spy(self, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(kwargs.get("is_benchmarking"))
        return real_init(self, **kwargs)

    canned = ReasonedPrediction(prediction_value=0.42, reasoning="r")
    with (
        patch.object(TemplateForecaster, "__init__", new=init_spy),
        patch.object(TemplateForecaster, "_make_prediction", new=AsyncMock(return_value=canned)),
    ):
        await run_forecasters_for_question(
            q,
            "research blob",
            cache,
            forecaster_llms=six_forecaster_llms,
            parser_llm=parser_llm,
        )

    assert captured
    assert all(value is True for value in captured)


@pytest.mark.asyncio
async def test_runner_serializes_binary_prediction_value(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
) -> None:
    """The cached payload's prediction_value matches the binary serialization format."""
    from metaculus_bot.ablation.forecasters import run_forecasters_for_question

    q = _make_binary_question(qid=1011)
    canned = ReasonedPrediction(prediction_value=0.42, reasoning="r")

    with patch.object(TemplateForecaster, "_make_prediction", new=AsyncMock(return_value=canned)):
        result = await run_forecasters_for_question(
            q,
            "research blob",
            cache,
            forecaster_llms=six_forecaster_llms[:1],  # one is enough for serialize check
            parser_llm=parser_llm,
        )

    payload = next(iter(result.values()))
    assert payload["prediction_value"] == {"type": "binary", "prob": 0.42}


@pytest.mark.asyncio
async def test_runner_serializes_numeric_prediction_value(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
) -> None:
    """End-to-end regression: a NumericDistribution from ``_make_prediction`` round-trips
    through the runner's cache write + deserialize without losing the CDF.

    This is the regression for the "tuple object has no attribute percentile" bug:
    the previous serializer treated ``value`` as ``list[Percentile]``, which crashed
    on the real ``NumericDistribution`` (a Pydantic model — iterating yields
    ``(field_name, value)`` tuples). The runner must now serialize the full payload
    AND deserialize back to a ``PchipNumericDistribution`` whose ``.cdf`` matches
    the original probabilities exactly.
    """
    from forecasting_tools import NumericDistribution

    from metaculus_bot.ablation.forecasters import deserialize_prediction_value, run_forecasters_for_question
    from metaculus_bot.numeric.pchip_processing import create_pchip_numeric_distribution

    q = _make_numeric_question(qid=1012)
    percentiles = [
        Percentile(value=v, percentile=p)
        for v, p in zip(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 99.0],
            [0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.975],
        )
    ]
    pchip_cdf = [i / 200 for i in range(201)]
    distribution = create_pchip_numeric_distribution(
        pchip_cdf=pchip_cdf,
        percentile_list=percentiles,
        question=q,
        zero_point=q.zero_point,
    )
    canned = ReasonedPrediction(prediction_value=distribution, reasoning="r")

    with patch.object(TemplateForecaster, "_make_prediction", new=AsyncMock(return_value=canned)):
        result = await run_forecasters_for_question(
            q,
            "research blob",
            cache,
            forecaster_llms=six_forecaster_llms[:1],
            parser_llm=parser_llm,
        )

    assert len(result) == 1
    payload = next(iter(result.values()))
    pv = payload["prediction_value"]
    assert pv is not None, "tuple bug regressed: prediction_value should not be None"
    assert pv["type"] == "numeric"
    assert len(pv["cdf_probabilities"]) == 201
    assert len(pv["declared_percentiles"]) == 11
    assert pv["declared_percentiles"][0] == {"percentile": 0.025, "value": 10.0}
    assert payload["errors"] == [], f"unexpected errors (tuple bug regressed?): {payload['errors']}"

    # Round-trip the cached payload back to a NumericDistribution. The result is a
    # ``PchipNumericDistribution`` (subclass), and its .cdf must match the original.
    restored = deserialize_prediction_value(pv, q)
    assert isinstance(restored, NumericDistribution)
    assert type(restored).__name__ == "PchipNumericDistribution"
    restored_probs = [p.percentile for p in restored.cdf]
    assert restored_probs == pytest.approx(pchip_cdf)


@pytest.mark.asyncio
async def test_one_serialize_failure_does_not_drop_other_forecaster_payloads(
    cache: AblationCache,
    parser_llm: GeneralLlm,
) -> None:
    """Per-forecaster serialize failure must not cascade to sibling forecasters.

    Before the inner try/except wrapper around serialize+cache-write, a single
    forecaster emitting a malformed prediction_value raised through
    ``asyncio.gather`` (default ``return_exceptions=False``), the outer batch
    runner caught it as "qid=X failed entirely", and EVERY other forecaster's
    cache write for that qid was lost. This test pins that contract: the broken
    forecaster gets ``prediction_value=None`` and a non-empty errors list; the
    healthy forecaster's payload still lands.
    """
    from metaculus_bot.ablation.forecasters import run_forecasters_for_question
    from metaculus_bot.numeric.pchip_processing import create_pchip_numeric_distribution

    q = _make_numeric_question(qid=1014)
    two_llms = _make_forecaster_llms(count=2)
    healthy_model = two_llms[0].model
    broken_model = two_llms[1].model

    # Healthy forecaster returns a real NumericDistribution; broken returns a
    # raw string (the exact failure mode of the tuple bug — bad shape that the
    # serializer rejects).
    percentiles = [
        Percentile(value=v, percentile=p)
        for v, p in zip(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 99.0],
            [0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.975],
        )
    ]
    pchip_cdf = [i / 200 for i in range(201)]
    distribution = create_pchip_numeric_distribution(
        pchip_cdf=pchip_cdf,
        percentile_list=percentiles,
        question=q,
        zero_point=q.zero_point,
    )
    healthy_pred = ReasonedPrediction(prediction_value=distribution, reasoning="ok")
    broken_pred = ReasonedPrediction(prediction_value="not-a-distribution", reasoning="bad")  # type: ignore[arg-type]

    async def selective(self, question, research, llm) -> ReasonedPrediction:  # type: ignore[no-untyped-def]  # noqa: ASYNC124
        if llm.model == healthy_model:
            return healthy_pred  # noqa: ASYNC910
        return broken_pred  # noqa: ASYNC910

    with patch.object(TemplateForecaster, "_make_prediction", new=selective):
        result = await run_forecasters_for_question(
            q,
            "research blob",
            cache,
            forecaster_llms=two_llms,
            parser_llm=parser_llm,
        )

    healthy_slug = model_slug_to_filename(healthy_model)
    broken_slug = model_slug_to_filename(broken_model)
    assert set(result.keys()) == {healthy_slug, broken_slug}

    # Healthy forecaster's payload landed intact (no cascade loss).
    healthy_payload = result[healthy_slug]
    assert healthy_payload["prediction_value"] is not None
    assert healthy_payload["prediction_value"]["type"] == "numeric"
    assert healthy_payload["errors"] == []

    # Broken forecaster's payload has prediction_value=None and a populated errors list.
    broken_payload = result[broken_slug]
    assert broken_payload["prediction_value"] is None
    assert broken_payload["errors"], "broken forecaster's serialize error must be captured"
    assert "TypeError" in broken_payload["errors"][0] or "Expected NumericDistribution" in broken_payload["errors"][0]

    # And the broken payload was still written to cache for diagnostics.
    on_disk = cache.read_forecaster_output(qid=q.id_of_question, model_slug=broken_slug)
    assert on_disk is not None
    assert on_disk["prediction_value"] is None
    assert on_disk["errors"]


@pytest.mark.asyncio
async def test_runner_serializes_mc_prediction_value(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
) -> None:
    from metaculus_bot.ablation.forecasters import run_forecasters_for_question

    q = _make_mc_question(qid=1013)
    options = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="A", probability=0.5),
            PredictedOption(option_name="B", probability=0.3),
            PredictedOption(option_name="C", probability=0.2),
        ]
    )
    canned = ReasonedPrediction(prediction_value=options, reasoning="r")

    with patch.object(TemplateForecaster, "_make_prediction", new=AsyncMock(return_value=canned)):
        result = await run_forecasters_for_question(
            q,
            "research blob",
            cache,
            forecaster_llms=six_forecaster_llms[:1],
            parser_llm=parser_llm,
        )

    payload = next(iter(result.values()))
    pv = payload["prediction_value"]
    assert pv["type"] == "multiple_choice"
    names = {o["option_name"] for o in pv["options"]}
    assert names == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# run_forecasters_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_returns_dict_keyed_by_qid(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
) -> None:
    from metaculus_bot.ablation.forecasters import run_forecasters_batch

    q1 = _make_binary_question(qid=2001)
    q2 = _make_binary_question(qid=2002)
    canned = ReasonedPrediction(prediction_value=0.42, reasoning="r")

    with patch.object(TemplateForecaster, "_make_prediction", new=AsyncMock(return_value=canned)):
        result = await run_forecasters_batch(
            [(q1, "blob 1"), (q2, "blob 2")],
            cache,
            forecaster_llms=six_forecaster_llms,
            parser_llm=parser_llm,
        )

    assert set(result.keys()) == {2001, 2002}
    for per_qid in result.values():
        assert len(per_qid) == len(six_forecaster_llms)


@pytest.mark.asyncio
async def test_batch_per_question_failure_does_not_kill_batch(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
) -> None:
    """If run_forecasters_for_question raises for Q1, Q2 still completes."""
    from metaculus_bot.ablation import forecasters as forecasters_module
    from metaculus_bot.ablation.forecasters import run_forecasters_batch

    q1 = _make_binary_question(qid=2003)
    q2 = _make_binary_question(qid=2004)

    canned = ReasonedPrediction(prediction_value=0.42, reasoning="r")
    real_runner = forecasters_module.run_forecasters_for_question

    async def selective_runner(question, *args, **kwargs):  # type: ignore[no-untyped-def]
        if question.id_of_question == 2003:
            raise RuntimeError("question 1 exploded")
        return await real_runner(question, *args, **kwargs)

    with (
        patch.object(forecasters_module, "run_forecasters_for_question", new=selective_runner),
        patch.object(TemplateForecaster, "_make_prediction", new=AsyncMock(return_value=canned)),
    ):
        result = await run_forecasters_batch(
            [(q1, "blob 1"), (q2, "blob 2")],
            cache,
            forecaster_llms=six_forecaster_llms,
            parser_llm=parser_llm,
        )

    assert 2003 in result
    assert 2004 in result
    # Q1 failed entirely → empty per-model dict.
    assert result[2003] == {}
    # Q2 succeeded.
    assert len(result[2004]) == len(six_forecaster_llms)


@pytest.mark.asyncio
async def test_batch_concurrency_respected(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
) -> None:
    """per_question_concurrency caps in-flight runner invocations."""
    from metaculus_bot.ablation import forecasters as forecasters_module
    from metaculus_bot.ablation.forecasters import run_forecasters_batch

    questions = [_make_binary_question(qid=3000 + i) for i in range(4)]

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def slow_runner(question, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return {}

    with patch.object(forecasters_module, "run_forecasters_for_question", new=slow_runner):
        await run_forecasters_batch(
            [(q, f"blob {q.id_of_question}") for q in questions],
            cache,
            forecaster_llms=six_forecaster_llms,
            parser_llm=parser_llm,
            per_question_concurrency=2,
        )

    assert max_in_flight <= 2
    # Sanity: > 1 means concurrency actually happened (otherwise semaphore over-restrictive).
    assert max_in_flight == 2


# ---------------------------------------------------------------------------
# Bug-1 regression: PROBABILISTIC_TOOLS_ENABLED env var must NOT leak into
# the forecast stage. If the operator's shell has the var set, every
# forecaster rationale would otherwise get baked-in tool output before
# caching, contaminating BOTH ablation arms with the treatment.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forecast_stage_disables_tools_env_var_during_make_prediction(
    cache: AblationCache,
    six_forecaster_llms: list[GeneralLlm],
    parser_llm: GeneralLlm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator-shell ``PROBABILISTIC_TOOLS_ENABLED=1`` must not contaminate cached rationales.

    The ablation pairs cached forecaster rationales across both arms; if a stray
    env var causes ``_make_prediction`` to append "## Computed quantities" before
    caching, BOTH arms see the same contaminated rationale and the A/B comparison
    becomes meaningless. The forecast stage must explicitly disable the env var
    while running, then restore the operator's original setting on exit.
    """
    from metaculus_bot.ablation.forecasters import run_forecasters_for_question

    monkeypatch.setenv("PROBABILISTIC_TOOLS_ENABLED", "1")

    q = _make_binary_question(qid=9001)
    canned = ReasonedPrediction(prediction_value=0.42, reasoning="rationale")
    observed_env_values: list[str | None] = []

    async def env_observer(self, question, research, llm) -> ReasonedPrediction:  # type: ignore[no-untyped-def]  # noqa: ASYNC124
        import os  # noqa: PLC0415  - intentional in-fixture observation

        observed_env_values.append(os.environ.get("PROBABILISTIC_TOOLS_ENABLED"))
        return canned  # noqa: ASYNC910

    with patch.object(TemplateForecaster, "_make_prediction", new=env_observer):
        await run_forecasters_for_question(
            q,
            "research blob",
            cache,
            forecaster_llms=six_forecaster_llms,
            parser_llm=parser_llm,
        )

    # Every _make_prediction call saw the env var disabled (None or absent).
    assert observed_env_values, "no _make_prediction calls observed"
    assert all(value is None for value in observed_env_values), (
        f"_make_prediction saw PROBABILISTIC_TOOLS_ENABLED set during forecast stage: {observed_env_values}"
    )

    # After the runner returns, the operator's original "1" must be restored.
    import os  # noqa: PLC0415

    assert os.environ.get("PROBABILISTIC_TOOLS_ENABLED") == "1"


# ---------------------------------------------------------------------------
# Regression: notepad lifecycle around _make_prediction
#
# Every other test in this file mocks ``TemplateForecaster._make_prediction``
# directly, which structurally bypasses the framework's ``_get_notepad``
# lookup. That lookup is what failed in the first live ablation run: the
# framework raises ``ValueError("No notepad found...")`` if the bot hasn't
# registered a notepad for the question, and the runner never registered one.
# Mocking lower (at the forecast-method level) lets the framework's
# ``_make_prediction`` actually run and exercises the lifecycle our runner
# now owns.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_one_forecaster_initializes_notepad_so_framework_get_notepad_works(
    cache: AblationCache,
    parser_llm: GeneralLlm,
) -> None:
    """``_run_one_forecaster`` must register a notepad before ``_make_prediction``.

    The framework's ``_make_prediction`` (main.py:1085 / forecast_bot.py:473)
    starts with ``await self._get_notepad(question)``, which raises if no
    notepad has been registered. The ablation runner bypasses
    ``_run_individual_question`` (the framework's normal entry that calls
    ``_initialize_notepad`` + appends to ``_note_pads``), so the runner must
    own that lifecycle itself.

    This test exercises the real ``_make_prediction`` path by mocking the
    LLM-calling forecast method (``_run_forecast_on_binary``) one level down,
    not ``_make_prediction`` itself.
    """
    from metaculus_bot.ablation.forecasters import run_forecasters_for_question

    q = _make_binary_question(qid=9101)
    canned = ReasonedPrediction(prediction_value=0.42, reasoning="rationale")
    one_llm = _make_forecaster_llms(count=1)

    with patch.object(TemplateForecaster, "_run_forecast_on_binary", new=AsyncMock(return_value=canned)):
        result = await run_forecasters_for_question(
            q,
            "research blob",
            cache,
            forecaster_llms=one_llm,
            parser_llm=parser_llm,
        )

    assert len(result) == 1
    payload = next(iter(result.values()))
    assert payload["errors"] == [], f"unexpected errors (notepad bug regressed?): {payload['errors']}"
    assert payload["prediction_value"] is not None
    assert payload["prediction_value"]["type"] == "binary"
    assert payload["prediction_value"]["prob"] == pytest.approx(0.42)


@pytest.mark.asyncio
async def test_run_one_forecaster_removes_notepad_after_make_prediction_raises(
    cache: AblationCache,
    parser_llm: GeneralLlm,
) -> None:
    """Notepads must not leak across calls when the forecast method raises.

    The runner registers a notepad on entry and removes it on exit (in a
    ``finally`` block). Without removal, repeated calls on the same bot would
    accumulate notepads — not catastrophic in our per-(qid, model) bot lifetime
    but a correctness invariant that should not regress.
    """
    from metaculus_bot.ablation.forecasters import _build_bot, _run_one_forecaster

    q = _make_binary_question(qid=9102)
    one_llm = _make_forecaster_llms(count=1)[0]

    # Build the bot manually so we can inspect ``_note_pads`` after the call.
    # The runner builds its own bot internally; here we patch ``_build_bot``
    # to return our inspectable instance.
    bot = _build_bot(
        question=q,
        research_blob="research",
        forecaster_llm=one_llm,
        parser_llm=parser_llm,
    )

    async def boom(self, question, research, llm) -> ReasonedPrediction:  # type: ignore[no-untyped-def]  # noqa: ASYNC124
        raise RuntimeError("simulated forecaster failure")  # noqa: ASYNC910

    semaphore = asyncio.Semaphore(1)
    with (
        patch("metaculus_bot.ablation.forecasters._build_bot", return_value=bot),
        patch.object(TemplateForecaster, "_run_forecast_on_binary", new=boom),
    ):
        slug, payload = await _run_one_forecaster(q, "research", one_llm, parser_llm, cache, semaphore=semaphore)

    # The forecast raised, so prediction_value is None and errors is populated.
    assert payload["prediction_value"] is None
    assert any("RuntimeError" in e for e in payload["errors"]), payload["errors"]
    # The notepad must have been removed even though _make_prediction raised.
    assert bot._note_pads == [], f"notepad leaked after exception: {bot._note_pads}"


# ---------------------------------------------------------------------------
# Rate-limit retry behavior
#
# OpenRouter ``:free`` model variants are rate-limited at the upstream provider
# level (Venice, OpenInference, etc.) — separate from the per-key quota.
# When 429s fire, OpenRouter exposes the upstream provider's ``Retry-After``
# value in the exception payload's ``retry_after_seconds`` field. The runner
# must honor that hint and retry up to ``max_retries`` times before giving up.
# ---------------------------------------------------------------------------


def _build_rate_limit_exc(retry_after: int | float | None = 13) -> Exception:
    """Build a litellm.RateLimitError that mirrors the live OpenRouter 429 shape.

    Matches a real exception from /tmp/ablation_phase_a1_v3.log so the parser
    is exercised against the actual JSON shape, not a synthetic one.
    """
    import litellm  # noqa: PLC0415  - import at use to avoid module-load cost in fixture-heavy tests

    if retry_after is None:
        # Omit the retry_after_seconds field so the parser falls back to exponential backoff.
        msg = (
            'RateLimitError: OpenrouterException - {"error":{"message":"Provider returned error",'
            '"code":429,"metadata":{"raw":"qwen/qwen3-next-80b-a3b-instruct:free is temporarily '
            'rate-limited upstream.","provider_name":"Venice","is_byok":false,'
            '"headers":{"Retry-After":"13"}}}}'
        )
    else:
        msg = (
            f'RateLimitError: OpenrouterException - {{"error":{{"message":"Provider returned error",'
            f'"code":429,"metadata":{{"raw":"qwen/qwen3-next-80b-a3b-instruct:free is temporarily '
            f'rate-limited upstream.","provider_name":"Venice","is_byok":false,'
            f'"retry_after_seconds":{retry_after},"retry_after_seconds_raw":12.315,'
            f'"headers":{{"Retry-After":"13"}}}}}}}}'
        )
    return litellm.RateLimitError(
        message=msg,
        model="openrouter/qwen/qwen3-next-80b-a3b-instruct:free",
        llm_provider="openrouter",
    )


@pytest.mark.asyncio
async def test_run_one_forecaster_retries_on_rate_limit_then_succeeds(
    cache: AblationCache,
    parser_llm: GeneralLlm,
) -> None:
    """``RateLimitError`` on first two attempts, success on third → final payload OK, await_count=3."""
    from metaculus_bot.ablation.forecasters import _run_one_forecaster

    q = _make_binary_question(qid=9201)
    one_llm = _make_forecaster_llms(count=1)[0]
    canned = ReasonedPrediction(prediction_value=0.42, reasoning="rationale")
    rate_limit_exc = _build_rate_limit_exc(retry_after=13)

    mock_make = AsyncMock(side_effect=[rate_limit_exc, rate_limit_exc, canned])
    semaphore = asyncio.Semaphore(1)
    sleep_mock = AsyncMock(return_value=None)

    with (
        patch.object(TemplateForecaster, "_make_prediction", new=mock_make),
        patch("metaculus_bot.ablation.forecasters.asyncio.sleep", new=sleep_mock),
    ):
        slug, payload = await _run_one_forecaster(
            q,
            "research blob",
            one_llm,
            parser_llm,
            cache,
            semaphore=semaphore,
            max_retries=3,
        )

    assert mock_make.await_count == 3
    assert payload["prediction_value"] is not None
    assert payload["prediction_value"]["type"] == "binary"
    assert payload["prediction_value"]["prob"] == pytest.approx(0.42)
    assert payload["errors"] == []


@pytest.mark.asyncio
async def test_run_one_forecaster_honors_retry_after_seconds(
    cache: AblationCache,
    parser_llm: GeneralLlm,
) -> None:
    """``retry_after_seconds`` parsed from the exception payload is passed to asyncio.sleep."""
    from metaculus_bot.ablation.forecasters import _run_one_forecaster

    q = _make_binary_question(qid=9202)
    one_llm = _make_forecaster_llms(count=1)[0]
    canned = ReasonedPrediction(prediction_value=0.42, reasoning="rationale")
    rate_limit_exc = _build_rate_limit_exc(retry_after=13)

    mock_make = AsyncMock(side_effect=[rate_limit_exc, canned])
    semaphore = asyncio.Semaphore(1)
    sleep_mock = AsyncMock(return_value=None)

    with (
        patch.object(TemplateForecaster, "_make_prediction", new=mock_make),
        patch("metaculus_bot.ablation.forecasters.asyncio.sleep", new=sleep_mock),
    ):
        await _run_one_forecaster(
            q,
            "research blob",
            one_llm,
            parser_llm,
            cache,
            semaphore=semaphore,
            max_retries=3,
        )

    # asyncio.sleep called at least once with ~13s (plus jitter <= 0.5s).
    sleep_durations = [call.args[0] for call in sleep_mock.await_args_list if call.args]
    assert any(13.0 <= d <= 13.6 for d in sleep_durations), (
        f"Expected sleep close to 13s honoring retry_after_seconds; got durations {sleep_durations}"
    )


@pytest.mark.asyncio
async def test_run_one_forecaster_falls_back_to_exponential_when_retry_after_missing(
    cache: AblationCache,
    parser_llm: GeneralLlm,
) -> None:
    """When ``retry_after_seconds`` cannot be parsed, jittered exponential backoff is used."""
    from metaculus_bot.ablation.forecasters import _run_one_forecaster

    q = _make_binary_question(qid=9203)
    one_llm = _make_forecaster_llms(count=1)[0]
    canned = ReasonedPrediction(prediction_value=0.42, reasoning="rationale")
    rate_limit_exc = _build_rate_limit_exc(retry_after=None)

    mock_make = AsyncMock(side_effect=[rate_limit_exc, canned])
    semaphore = asyncio.Semaphore(1)
    sleep_mock = AsyncMock(return_value=None)

    with (
        patch.object(TemplateForecaster, "_make_prediction", new=mock_make),
        patch("metaculus_bot.ablation.forecasters.asyncio.sleep", new=sleep_mock),
    ):
        await _run_one_forecaster(
            q,
            "research blob",
            one_llm,
            parser_llm,
            cache,
            semaphore=semaphore,
            max_retries=3,
        )

    # Exponential backoff for attempt 0: 2**0 + jitter[0,1) = [1, 2). Capped at 60s.
    sleep_durations = [call.args[0] for call in sleep_mock.await_args_list if call.args]
    assert any(1.0 <= d < 2.0 for d in sleep_durations), (
        f"Expected exponential backoff sleep in [1, 2) when retry_after missing; got durations {sleep_durations}"
    )


@pytest.mark.asyncio
async def test_run_one_forecaster_exhausts_retries_records_errors_no_raise(
    cache: AblationCache,
    parser_llm: GeneralLlm,
) -> None:
    """Mock failing 4 times with RateLimitError; max_retries=3 → returns normally with all 4 attempts in errors."""
    from metaculus_bot.ablation.forecasters import _run_one_forecaster

    q = _make_binary_question(qid=9204)
    one_llm = _make_forecaster_llms(count=1)[0]
    rate_limit_exc = _build_rate_limit_exc(retry_after=13)

    # 4 failures: initial attempt + 3 retries = 4 total invocations.
    mock_make = AsyncMock(side_effect=[rate_limit_exc] * 4)
    semaphore = asyncio.Semaphore(1)
    sleep_mock = AsyncMock(return_value=None)

    with (
        patch.object(TemplateForecaster, "_make_prediction", new=mock_make),
        patch("metaculus_bot.ablation.forecasters.asyncio.sleep", new=sleep_mock),
    ):
        slug, payload = await _run_one_forecaster(
            q,
            "research blob",
            one_llm,
            parser_llm,
            cache,
            semaphore=semaphore,
            max_retries=3,
        )

    assert mock_make.await_count == 4
    assert payload["prediction_value"] is None
    # All 4 RateLimitError attempts recorded in errors.
    assert len(payload["errors"]) == 4
    assert all("RateLimitError" in e for e in payload["errors"])


@pytest.mark.asyncio
async def test_run_one_forecaster_non_rate_limit_error_not_retried(
    cache: AblationCache,
    parser_llm: GeneralLlm,
) -> None:
    """A plain ``RuntimeError`` (not 429) must NOT trigger the retry loop — single attempt only."""
    from metaculus_bot.ablation.forecasters import _run_one_forecaster

    q = _make_binary_question(qid=9205)
    one_llm = _make_forecaster_llms(count=1)[0]
    runtime_exc = RuntimeError("model exploded — not a rate limit")

    mock_make = AsyncMock(side_effect=[runtime_exc])
    semaphore = asyncio.Semaphore(1)
    sleep_mock = AsyncMock(return_value=None)

    with (
        patch.object(TemplateForecaster, "_make_prediction", new=mock_make),
        patch("metaculus_bot.ablation.forecasters.asyncio.sleep", new=sleep_mock),
    ):
        slug, payload = await _run_one_forecaster(
            q,
            "research blob",
            one_llm,
            parser_llm,
            cache,
            semaphore=semaphore,
            max_retries=3,
        )

    # Only the initial attempt — no retries on non-429.
    assert mock_make.await_count == 1
    assert payload["prediction_value"] is None
    assert len(payload["errors"]) == 1
    assert "RuntimeError" in payload["errors"][0]


# ---------------------------------------------------------------------------
# M1 — _parse_retry_after_seconds also accepts HTTP-date form
#
# RFC 7231 Retry-After allows both integer-seconds AND HTTP-date format.
# Some upstreams proxy the date form through OpenRouter; without parsing it,
# we fast-fall back to exponential backoff (cap 60s) and may burn the
# retry budget while a real >60s window applies.
# ---------------------------------------------------------------------------


def test_parse_retry_after_seconds_handles_integer_form() -> None:
    from metaculus_bot.ablation.forecasters import _parse_retry_after_seconds

    exc = Exception('"Retry-After":"30","retry_after_seconds":30')
    assert _parse_retry_after_seconds(exc) == pytest.approx(30.0)


def test_parse_retry_after_seconds_handles_http_date_form() -> None:
    """An HTTP-date Retry-After header parses to a positive float seconds."""
    from metaculus_bot.ablation.forecasters import _parse_retry_after_seconds

    exc = Exception(
        'OpenrouterException - {"error":{"code":429,"metadata":'
        '{"headers":{"Retry-After":"Wed, 21 Oct 2099 07:28:00 GMT"}}}}'
    )
    parsed = _parse_retry_after_seconds(exc)
    assert parsed is not None
    assert parsed > 0.0


def test_parse_retry_after_seconds_returns_none_for_garbage_date() -> None:
    """Garbage value in Retry-After must not crash; parser returns None so
    the caller falls back to exponential backoff."""
    from metaculus_bot.ablation.forecasters import _parse_retry_after_seconds

    exc = Exception('"Retry-After":"garbage-not-a-date"')
    assert _parse_retry_after_seconds(exc) is None


def test_parse_retry_after_seconds_clamps_negative_past_date_to_zero() -> None:
    """A past HTTP-date (already elapsed) parses to 0 — the caller sleeps
    nothing and retries immediately."""
    from metaculus_bot.ablation.forecasters import _parse_retry_after_seconds

    exc = Exception('"Retry-After":"Wed, 21 Oct 2000 07:28:00 GMT"')
    parsed = _parse_retry_after_seconds(exc)
    assert parsed == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# C1 — Soft-deadline timeout
#
# Production wraps ``_make_prediction`` in ``asyncio.wait_for(... ,
# timeout=FORECASTER_SOFT_DEADLINE)`` (main.py:1063). Without that wrapper,
# a single Anthropic stall can hold a question for litellm timeout(480) *
# allowed_tries(3) ≈ 24 min — at 50q with serial forecasters this parks
# the whole batch for hours. The wrapper bounds each call at
# FORECASTER_SOFT_DEADLINE (10 min) and re-records the timeout in the
# forecaster's ``errors`` so downstream surfaces it as a normal failure.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_one_forecaster_enforces_soft_deadline_timeout(
    cache: AblationCache,
    parser_llm: GeneralLlm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck ``_make_prediction`` is killed by FORECASTER_SOFT_DEADLINE.

    Mocks the call to sleep ``FORECASTER_SOFT_DEADLINE + 5`` seconds; the
    runner must return within ``FORECASTER_SOFT_DEADLINE * 1.1`` with a
    TimeoutError recorded in the payload's ``errors``.
    """
    from metaculus_bot.ablation import forecasters as forecasters_module
    from metaculus_bot.ablation.forecasters import _run_one_forecaster

    # Shrink the deadline so the test is fast (3s instead of 600s). The
    # production constant only matters for the wall-clock cap; the wrap
    # behavior under test is identical at any value.
    monkeypatch.setattr(forecasters_module, "FORECASTER_SOFT_DEADLINE", 3)

    q = _make_binary_question(qid=9501)
    one_llm = _make_forecaster_llms(count=1)[0]

    async def stall(self, question, research, llm) -> ReasonedPrediction:  # type: ignore[no-untyped-def]  # noqa: ASYNC124
        await asyncio.sleep(8)
        return ReasonedPrediction(prediction_value=0.42, reasoning="never reached")  # noqa: ASYNC910

    semaphore = asyncio.Semaphore(1)
    start = asyncio.get_event_loop().time()
    with patch.object(TemplateForecaster, "_make_prediction", new=stall):
        slug, payload = await _run_one_forecaster(
            q,
            "research blob",
            one_llm,
            parser_llm,
            cache,
            semaphore=semaphore,
            max_retries=0,
        )
    elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < 3 * 1.5, f"runner did not honor soft deadline; elapsed={elapsed:.1f}s"
    assert payload["prediction_value"] is None
    assert payload["errors"], "expected at least one error recorded for timeout"
    assert any("TimeoutError" in e for e in payload["errors"]), payload["errors"]


@pytest.mark.asyncio
async def test_run_one_forecaster_soft_deadline_does_not_retry_after_timeout(
    cache: AblationCache,
    parser_llm: GeneralLlm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout is not a rate limit; the runner must NOT retry it."""
    from metaculus_bot.ablation import forecasters as forecasters_module
    from metaculus_bot.ablation.forecasters import _run_one_forecaster

    monkeypatch.setattr(forecasters_module, "FORECASTER_SOFT_DEADLINE", 1)

    q = _make_binary_question(qid=9502)
    one_llm = _make_forecaster_llms(count=1)[0]
    invocations = 0

    async def stall(self, question, research, llm) -> ReasonedPrediction:  # type: ignore[no-untyped-def]  # noqa: ASYNC124
        nonlocal invocations
        invocations += 1
        await asyncio.sleep(5)
        return ReasonedPrediction(prediction_value=0.42, reasoning="never")  # noqa: ASYNC910

    semaphore = asyncio.Semaphore(1)
    with patch.object(TemplateForecaster, "_make_prediction", new=stall):
        await _run_one_forecaster(
            q,
            "research blob",
            one_llm,
            parser_llm,
            cache,
            semaphore=semaphore,
            max_retries=3,
        )

    assert invocations == 1, f"timeout must not be retried; got {invocations} attempts"


@pytest.mark.asyncio
async def test_run_forecasters_for_question_threads_max_retries(
    cache: AblationCache,
    parser_llm: GeneralLlm,
) -> None:
    """``max_retries`` kwarg on ``run_forecasters_for_question`` reaches ``_run_one_forecaster``."""
    from metaculus_bot.ablation.forecasters import run_forecasters_for_question

    q = _make_binary_question(qid=9206)
    one_llm = _make_forecaster_llms(count=1)
    canned = ReasonedPrediction(prediction_value=0.42, reasoning="rationale")
    rate_limit_exc = _build_rate_limit_exc(retry_after=13)

    # Fail 5 times then succeed; max_retries=5 should make the 6th attempt land.
    mock_make = AsyncMock(side_effect=[rate_limit_exc] * 5 + [canned])
    sleep_mock = AsyncMock(return_value=None)

    with (
        patch.object(TemplateForecaster, "_make_prediction", new=mock_make),
        patch("metaculus_bot.ablation.forecasters.asyncio.sleep", new=sleep_mock),
    ):
        result = await run_forecasters_for_question(
            q,
            "research blob",
            cache,
            forecaster_llms=one_llm,
            parser_llm=parser_llm,
            max_retries=5,
        )

    assert mock_make.await_count == 6
    payload = next(iter(result.values()))
    assert payload["prediction_value"] is not None
    assert payload["errors"] == []


@pytest.mark.asyncio
async def test_run_forecasters_batch_threads_max_retries(
    cache: AblationCache,
    parser_llm: GeneralLlm,
) -> None:
    """``max_retries`` kwarg on ``run_forecasters_batch`` flows through to per-question runner."""
    from metaculus_bot.ablation import forecasters as forecasters_module
    from metaculus_bot.ablation.forecasters import run_forecasters_batch

    q1 = _make_binary_question(qid=9207)
    captured_kwargs: list[dict] = []

    async def spy_runner(question, *args, **kwargs):  # type: ignore[no-untyped-def]  # noqa: ASYNC124
        captured_kwargs.append(kwargs)
        return {}  # noqa: ASYNC910

    with patch.object(forecasters_module, "run_forecasters_for_question", new=spy_runner):
        await run_forecasters_batch(
            [(q1, "blob 1")],
            cache,
            forecaster_llms=_make_forecaster_llms(count=1),
            parser_llm=parser_llm,
            max_retries=5,
        )

    assert captured_kwargs
    assert captured_kwargs[0].get("max_retries") == 5


@pytest.mark.asyncio
async def test_max_sleep_cap_honors_retry_after_90s(
    cache: AblationCache,
    parser_llm: GeneralLlm,
) -> None:
    """``Retry-After: 90`` arrives → we sleep at LEAST 90 seconds.

    Free-tier providers occasionally signal long recovery windows (>60s).
    The runner must trust that signal — capping below it would shed forecasters
    that would otherwise succeed on the next attempt.
    """
    from metaculus_bot.ablation.forecasters import _run_one_forecaster

    q = _make_binary_question(qid=9301)
    one_llm = _make_forecaster_llms(count=1)[0]
    canned = ReasonedPrediction(prediction_value=0.42, reasoning="rationale")
    rate_limit_exc = _build_rate_limit_exc(retry_after=90)

    mock_make = AsyncMock(side_effect=[rate_limit_exc, canned])
    semaphore = asyncio.Semaphore(1)
    sleep_mock = AsyncMock(return_value=None)

    with (
        patch.object(TemplateForecaster, "_make_prediction", new=mock_make),
        patch("metaculus_bot.ablation.forecasters.asyncio.sleep", new=sleep_mock),
    ):
        await _run_one_forecaster(
            q,
            "research blob",
            one_llm,
            parser_llm,
            cache,
            semaphore=semaphore,
            max_retries=3,
        )

    sleep_durations = [call.args[0] for call in sleep_mock.await_args_list if call.args]
    assert any(d >= 90.0 for d in sleep_durations), (
        f"Expected sleep >= 90s honoring Retry-After: 90; got durations {sleep_durations}"
    )


@pytest.mark.asyncio
async def test_max_sleep_cap_bounds_runaway_retry_after(
    cache: AblationCache,
    parser_llm: GeneralLlm,
) -> None:
    """``Retry-After: 3600`` (one hour) is bounded by the runner's cap.

    A misbehaving upstream could send Retry-After: 3600 on a hot-tail throttle
    that's actually transient, parking the runner indefinitely. The cap protects
    against that runaway. New cap is 120s (more generous than the 60s exponential-
    backoff cap to honor legitimate long Retry-After signals).
    """
    from metaculus_bot.ablation.forecasters import _run_one_forecaster

    q = _make_binary_question(qid=9302)
    one_llm = _make_forecaster_llms(count=1)[0]
    canned = ReasonedPrediction(prediction_value=0.42, reasoning="rationale")
    rate_limit_exc = _build_rate_limit_exc(retry_after=3600)

    mock_make = AsyncMock(side_effect=[rate_limit_exc, canned])
    semaphore = asyncio.Semaphore(1)
    sleep_mock = AsyncMock(return_value=None)

    with (
        patch.object(TemplateForecaster, "_make_prediction", new=mock_make),
        patch("metaculus_bot.ablation.forecasters.asyncio.sleep", new=sleep_mock),
    ):
        await _run_one_forecaster(
            q,
            "research blob",
            one_llm,
            parser_llm,
            cache,
            semaphore=semaphore,
            max_retries=3,
        )

    sleep_durations = [call.args[0] for call in sleep_mock.await_args_list if call.args]
    rate_limit_sleeps = [d for d in sleep_durations if d > 1.0]
    assert rate_limit_sleeps, f"Expected at least one rate-limit sleep; got {sleep_durations}"
    assert all(d <= 121.0 for d in rate_limit_sleeps), (
        f"Expected sleep capped at 120s (+jitter <=1s) for runaway Retry-After: 3600; got {rate_limit_sleeps}"
    )


@pytest.mark.asyncio
async def test_retry_log_includes_provider_name_and_attempt(
    cache: AblationCache,
    parser_llm: GeneralLlm,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The retry log line surfaces upstream ``provider_name`` (e.g. Venice) plus attempt counter.

    Free-tier rate-limit pain is provider-scoped — Venice and OpenInference each have
    their own throttling windows. When several free-tier forecasters all share an
    upstream provider, the operator needs to see which provider is causing pain.
    """
    import logging  # noqa: PLC0415  - test-local import

    from metaculus_bot.ablation.forecasters import _run_one_forecaster

    q = _make_binary_question(qid=9303)
    one_llm = _make_forecaster_llms(count=1)[0]
    canned = ReasonedPrediction(prediction_value=0.42, reasoning="rationale")
    rate_limit_exc = _build_rate_limit_exc(retry_after=13)

    mock_make = AsyncMock(side_effect=[rate_limit_exc, canned])
    semaphore = asyncio.Semaphore(1)
    sleep_mock = AsyncMock(return_value=None)

    with (
        caplog.at_level(logging.INFO, logger="metaculus_bot.ablation.forecasters"),
        patch.object(TemplateForecaster, "_make_prediction", new=mock_make),
        patch("metaculus_bot.ablation.forecasters.asyncio.sleep", new=sleep_mock),
    ):
        await _run_one_forecaster(
            q,
            "research blob",
            one_llm,
            parser_llm,
            cache,
            semaphore=semaphore,
            max_retries=3,
        )

    retry_log_lines = [r.message for r in caplog.records if "rate-limited (retrying)" in r.message]
    assert retry_log_lines, f"No retry log line emitted. All logs: {[r.message for r in caplog.records]}"
    combined = " | ".join(retry_log_lines)
    # Provider name from the exception JSON metadata (provider_name="Venice").
    assert "Venice" in combined, f"Expected provider name 'Venice' in retry log; got: {combined}"
    # Attempt counter (e.g. 'attempt=1/3' or 'attempt 1 of 3').
    assert "attempt" in combined.lower(), f"Expected 'attempt' counter in retry log; got: {combined}"
