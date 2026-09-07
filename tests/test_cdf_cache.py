"""Behavior pins for ``ensemble_analysis.cdf_cache.NumericCdfCache``.

The safe-CDF accessor tries the declared CDF and then a PCHIP rebuild. A prediction
that reaches neither path is unscoreable; these tests pin that it is excluded rather
than assigned a fabricated distribution. They also pin memoization and failure
bookkeeping so the ladder can be restructured without hiding unusable inputs.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.ensemble_analysis.cdf_cache import NumericCdfCache
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid


def _question(
    qid: int = 7, lower: float = 0.0, upper: float = 100.0, zero_point: float | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        id_of_question=qid,
        lower_bound=lower,
        upper_bound=upper,
        open_lower_bound=False,
        open_upper_bound=False,
        zero_point=zero_point,
        cdf_size=201,
        page_url="https://example.com/q",
    )


class _RaisingCdf:
    """Prediction whose ``.cdf`` property raises, forcing the ladder past tier 1."""

    def __init__(self, declared: list[Percentile] | None = None) -> None:
        self.declared_percentiles = declared

    @property
    def cdf(self):
        raise ValueError("cdf unavailable")


def _declared(values: dict[float, float]) -> list[Percentile]:
    """Percentile objects keyed by fraction-of-one percentile, as ft emits them."""
    return [Percentile(percentile=p, value=v) for p, v in values.items()]


class TestTierOneDirectCdf:
    def test_percentile_objects_pass_through_verbatim(self):
        cache = NumericCdfCache()
        raw = [Percentile(percentile=0.0, value=0.0), Percentile(percentile=1.0, value=100.0)]
        out = cache.get_safe_numeric_cdf("m", _question(), SimpleNamespace(cdf=raw))

        assert out is not None
        assert [(p.percentile, p.value) for p in out] == [(0.0, 0.0), (1.0, 100.0)]

    def test_bare_float_cdf_gets_a_synthesized_value_grid(self):
        cache = NumericCdfCache()
        out = cache.get_safe_numeric_cdf("m", _question(), SimpleNamespace(cdf=[0.0, 0.5, 1.0]))

        assert out is not None
        assert [pt.value for pt in out] == [0.0, 50.0, 100.0]
        assert [pt.percentile for pt in out] == [0.0, 0.5, 1.0]

    def test_percentile_only_namespaces_get_a_synthesized_value_grid(self):
        cache = NumericCdfCache()
        raw = [SimpleNamespace(percentile=0.0), SimpleNamespace(percentile=1.0)]
        out = cache.get_safe_numeric_cdf("m", _question(), SimpleNamespace(cdf=raw))

        assert out is not None
        assert [(pt.value, pt.percentile) for pt in out] == [(0.0, 0.0), (100.0, 1.0)]

    def test_bare_cdf_uses_the_canonical_geometric_value_grid(self):
        cache = NumericCdfCache()
        question = _question(lower=1.0, upper=1000.0, zero_point=0.0)
        out = cache.get_safe_numeric_cdf("m", question, SimpleNamespace(cdf=[0.0, 0.5, 1.0]))

        assert out is not None
        expected = build_cdf_value_grid(1.0, 1000.0, 0.0, 3)
        assert [point.value for point in out] == pytest.approx(expected.tolist())

    def test_result_is_memoized_per_model_and_question(self):
        cache = NumericCdfCache()
        question = _question(qid=11)
        first = cache.get_safe_numeric_cdf("m", question, SimpleNamespace(cdf=[0.0, 1.0]))
        # A second call whose prediction would fail outright still returns the cached answer.
        second = cache.get_safe_numeric_cdf("m", question, _RaisingCdf())

        assert first is not None
        assert second is not None
        assert [pt.value for pt in second] == [pt.value for pt in first]

    def test_a_too_short_cdf_is_unscoreable(self):
        # len(raw) < 2 is not a usable CDF, and there are no declared percentiles
        # from which to rebuild one.
        cache = NumericCdfCache()
        out = cache.get_safe_numeric_cdf("m", _question(), SimpleNamespace(cdf=[0.5], declared_percentiles=None))

        assert out is None
        assert cache._numeric_cdf_stats["failures"] == {("m", 7)}


class TestTierTwoPchipRebuild:
    def test_rebuilds_from_declared_percentiles_when_cdf_raises(self):
        cache = NumericCdfCache()
        prediction = _RaisingCdf(_declared({0.1: 20.0, 0.25: 35.0, 0.5: 50.0, 0.75: 65.0, 0.9: 80.0}))
        out = cache.get_safe_numeric_cdf("m", _question(), prediction)

        assert out is not None
        assert len(out) == 201
        values = [pt.value for pt in out]
        assert values[0] == pytest.approx(0.0)
        assert values[-1] == pytest.approx(100.0)
        probs = [pt.percentile for pt in out]
        assert probs == sorted(probs)
        assert probs[0] == pytest.approx(0.0)
        assert probs[-1] == pytest.approx(1.0)
        # Median declared at the range midpoint, so F(50) ~ 0.5.
        assert probs[100] == pytest.approx(0.5, abs=0.05)

    def test_rebuild_uses_the_canonical_geometric_value_grid(self):
        cache = NumericCdfCache()
        question = _question(lower=1.0, upper=1000.0, zero_point=0.0)
        prediction = _RaisingCdf(_declared({0.1: 20.0, 0.5: 100.0, 0.9: 900.0}))
        out = cache.get_safe_numeric_cdf("m", question, prediction)

        assert out is not None
        expected = build_cdf_value_grid(1.0, 1000.0, 0.0, len(out))
        assert [point.value for point in out] == pytest.approx(expected.tolist())

    def test_rebuild_is_counted_as_built_not_failure(self):
        cache = NumericCdfCache()
        prediction = _RaisingCdf(_declared({0.1: 20.0, 0.5: 50.0, 0.9: 80.0}))
        cache.get_safe_numeric_cdf("m", _question(), prediction)

        stats = cache._numeric_cdf_stats
        assert stats["safe_cdf_built"] == {("m", 7)}
        assert stats["failures"] == set()

    def test_first_failure_warns_once_per_pair(self, caplog: pytest.LogCaptureFixture):
        cache = NumericCdfCache()
        question = _question(qid=3)
        prediction = _RaisingCdf(_declared({0.1: 20.0, 0.5: 50.0, 0.9: 80.0}))
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.ensemble_analysis.cdf_cache"):
            cache.get_safe_numeric_cdf("m", question, prediction)
            cache._safe_cdf_cache.clear()  # force a recompute without resetting the warn-once set
            cache.get_safe_numeric_cdf("m", question, prediction)

        assert sum("Numeric CDF access failed" in r.message for r in caplog.records) == 1


class TestFailure:
    def test_nothing_usable_is_unscoreable(self):
        cache = NumericCdfCache()
        out = cache.get_safe_numeric_cdf("m", _question(), _RaisingCdf(declared=None))

        assert out is None
        assert cache._numeric_cdf_stats["failures"] == {("m", 7)}

    def test_missing_bounds_returns_none_and_caches_the_failure(self):
        cache = NumericCdfCache()
        question = SimpleNamespace(id_of_question=9)  # no bounds anywhere on the ladder
        assert cache.get_safe_numeric_cdf("m", question, _RaisingCdf(declared=None)) is None
        assert cache._numeric_cdf_stats["failures"] == {("m", 9)}
        assert cache._safe_cdf_cache[("m", 9)] is None

    def test_absent_question_id_collapses_to_minus_one(self):
        cache = NumericCdfCache()
        question = _question()
        question.id_of_question = None
        cache.get_safe_numeric_cdf("m", question, SimpleNamespace(cdf=[0.0, 1.0]))

        assert ("m", -1) in cache._safe_cdf_cache


class TestBookkeeping:
    def test_every_call_records_an_attempt(self):
        cache = NumericCdfCache()
        cache.get_safe_numeric_cdf("a", _question(qid=1), SimpleNamespace(cdf=[0.0, 1.0]))
        cache.get_safe_numeric_cdf("b", _question(qid=2), SimpleNamespace(cdf=[0.0, 1.0]))

        assert cache._numeric_cdf_stats["attempt_pairs"] == {("a", 1), ("b", 2)}

    def test_clear_drops_cache_and_counters(self):
        cache = NumericCdfCache()
        cache.get_safe_numeric_cdf("a", _question(qid=1), SimpleNamespace(cdf=[0.0, 1.0]))
        cache.clear()

        assert cache._safe_cdf_cache == {}
        assert all(not bucket for bucket in cache._numeric_cdf_stats.values())

    def test_summary_reports_the_tier_counts(self, caplog: pytest.LogCaptureFixture):
        cache = NumericCdfCache()
        cache.get_safe_numeric_cdf("a", _question(qid=1), _RaisingCdf(declared=None))
        with caplog.at_level(logging.INFO, logger="metaculus_bot.ensemble_analysis.cdf_cache"):
            cache.log_numeric_cdf_summary()

        assert "attempts=1" in caplog.text
        assert "ramp=0" in caplog.text
        assert "failures=1" in caplog.text

    def test_summary_is_silent_with_no_attempts(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.INFO, logger="metaculus_bot.ensemble_analysis.cdf_cache"):
            NumericCdfCache().log_numeric_cdf_summary()

        assert caplog.text == ""
