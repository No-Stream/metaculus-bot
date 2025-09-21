from unittest.mock import MagicMock

from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.numeric_pipeline import build_numeric_distribution, sanitize_percentiles


def _build_question(**overrides) -> NumericQuestion:
    base_kwargs = dict(
        id_of_question=1,
        id_of_post=1,
        page_url="https://example.com/q/1",
        question_text="Test numeric question",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="units",
        zero_point=0.0,
    )
    base_kwargs.update(overrides)
    return NumericQuestion(**base_kwargs)


def test_sanitize_percentiles_orders_and_jitters(monkeypatch):
    question = _build_question(cdf_size=50)

    raw_values = [
        (0.8, 60.0),
        (0.025, 5.0),
        (0.95, 90.0),
        (0.5, 40.0),
        (0.2, 20.0),
        (0.4, 35.0),
        (0.6, 35.0),  # duplicate to trigger jitter
        (0.1, 10.0),
        (0.05, 7.5),
        (0.975, 95.0),
        (0.9, 80.0),
    ]
    percentiles = [Percentile(percentile=p, value=v) for p, v in raw_values]

    sanitized, zero_point = sanitize_percentiles(percentiles, question)

    # Ensure sorted by percentile
    sorted_percentiles = sorted([p for p, _ in raw_values])
    assert [p.percentile for p in sanitized] == sorted_percentiles

    # Values should be strictly increasing after jitter/clamp
    sanitized_values = [p.value for p in sanitized]
    assert all(b > a for a, b in zip(sanitized_values, sanitized_values[1:]))

    # Discrete question (cdf_size != default) should force zero_point to None
    assert zero_point is None


def test_build_numeric_distribution_fallback(monkeypatch):
    question = _build_question()
    sanitized = [
        Percentile(percentile=p, value=float(i * 10))
        for i, p in enumerate([0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.975], start=1)
    ]

    failing = MagicMock(side_effect=RuntimeError("boom"))
    fallback = MagicMock(return_value="fallback")
    validate = MagicMock()

    monkeypatch.setattr(
        "metaculus_bot.numeric_pipeline.generate_pchip_cdf_with_smoothing",
        failing,
    )
    monkeypatch.setattr(
        "metaculus_bot.numeric_pipeline.create_fallback_numeric_distribution",
        fallback,
    )
    monkeypatch.setattr(
        "metaculus_bot.numeric_pipeline.validate_cdf_construction",
        validate,
    )

    result = build_numeric_distribution(sanitized, question, zero_point=None)

    assert result == "fallback"
    fallback.assert_called_once_with(sanitized, question, None)
    validate.assert_called_once_with("fallback", question)


def test_build_numeric_distribution_success(monkeypatch):
    question = _build_question()
    sanitized = [
        Percentile(percentile=p, value=float(i * 10))
        for i, p in enumerate([0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.975], start=1)
    ]

    mock_cdf = [i / 10 for i in range(len(sanitized))]
    generator = MagicMock(return_value=(mock_cdf, False, False))
    create_dist = MagicMock(return_value="pchip")
    validate = MagicMock()

    monkeypatch.setattr(
        "metaculus_bot.numeric_pipeline.generate_pchip_cdf_with_smoothing",
        generator,
    )
    monkeypatch.setattr(
        "metaculus_bot.numeric_pipeline.create_pchip_numeric_distribution",
        create_dist,
    )
    monkeypatch.setattr(
        "metaculus_bot.numeric_pipeline.validate_cdf_construction",
        validate,
    )

    result = build_numeric_distribution(sanitized, question, zero_point=None)

    assert result == "pchip"
    generator.assert_called_once_with(sanitized, question, None)
    create_dist.assert_called_once_with(mock_cdf, sanitized, question, None)
    validate.assert_called_once_with("pchip", question)
