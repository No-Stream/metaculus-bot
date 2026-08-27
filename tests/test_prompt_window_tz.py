"""Timezone-robustness pinning tests for ``prompts._forecasting_window_str``.

Seam: ``_forecasting_window_str`` computes ``today - question.open_time`` (and
``scheduled_resolution_time - today``) to render the "days ago" / "days from
now" window anchor injected into every forecasting prompt. On the currently
installed forecasting-tools 0.2.54, ``MetaculusQuestion._parse_api_date`` uses
``datetime.strptime`` with the trailing ``Z`` as a literal character, so
question datetimes come back NAIVE (holding UTC wall-clock values). The 0.2.92
upgrade switches to ``pendulum.parse`` plus an ``add_timezone_to_dates``
validator, making those datetimes tz-AWARE UTC — at which point subtracting a
naive ``datetime.now()`` would raise ``TypeError: can't subtract offset-naive
and offset-aware datetimes``.

The fix normalizes both operands to tz-aware UTC (``datetime.now(timezone.utc)``
+ ``_as_utc`` on each question datetime), treating naive question datetimes as
UTC. That assumption is verified against 0.2.54's ``_parse_api_date`` (naive-UTC
wall clock) and 0.2.92's validator (naive → ``replace(tzinfo=UTC)``), so the two
representations render byte-identical window strings.

These tests are GREEN on 0.2.54 and enforce the migration:
  * naive question datetimes (0.2.54 reality) render the correct window string
    with NO behavior change vs. the pre-fix output;
  * tz-aware question datetimes (0.2.92 reality) render the SAME string and do
    NOT raise TypeError.
Self-contained: no network, no API keys, ``datetime.now`` is frozen via a
monkeypatched clock so the deltas are deterministic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import cast

import pytest
from forecasting_tools import NumericQuestion

from metaculus_bot import prompts as prompts_module

# Frozen "now" the window helper will see. Chosen so both deltas are clean,
# positive round numbers regardless of the host machine's clock or timezone.
_FROZEN_NOW_UTC = datetime(2026, 3, 2, 12, 0, 0, tzinfo=UTC)
_OPEN = datetime(2026, 1, 1, 0, 0, 0)  # 60 full days before frozen now (UTC)
_RESOLVE = datetime(2026, 5, 1, 0, 0, 0)  # 59 full days after frozen now (UTC)

# Expected deltas for the frozen clock above. Jan 1 00:00 -> Mar 2 12:00 is
# 60 days 12h -> .days == 60; Mar 2 12:00 -> May 1 00:00 is 59 days 12h -> 59.
_EXPECTED_ELAPSED_DAYS = 60
_EXPECTED_REMAINING_DAYS = 59


class _FrozenDatetime(datetime):
    """``datetime`` subclass whose ``now`` returns a fixed instant.

    ``_forecasting_window_str`` calls ``datetime.now(timezone.utc)``; freezing it
    makes the rendered deltas deterministic and lets the naive-vs-aware fixtures
    be compared byte-for-byte. Only ``now`` is overridden — arithmetic and
    ``strftime`` fall through to the real ``datetime`` so the subtraction the
    seam exercises is the genuine one.
    """

    @classmethod
    def now(cls, tz: object | None = None) -> datetime:  # type: ignore[override]
        if tz is None:
            return _FROZEN_NOW_UTC.replace(tzinfo=None)
        return _FROZEN_NOW_UTC.astimezone(cast(timezone, tz))


@pytest.fixture
def frozen_now(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze ``prompts.datetime.now`` at ``_FROZEN_NOW_UTC``."""
    monkeypatch.setattr(prompts_module, "datetime", _FrozenDatetime)


def _question(open_time: datetime, scheduled_resolution_time: datetime) -> NumericQuestion:
    """Minimal question stub carrying only the two datetimes the helper reads.

    A ``SimpleNamespace`` avoids constructing a full ``NumericQuestion`` (and its
    required bounds) — ``_forecasting_window_str`` only touches ``open_time`` and
    ``scheduled_resolution_time``. Cast for the type checker; the helper never
    calls a real ``NumericQuestion`` method.
    """
    return cast(
        NumericQuestion,
        SimpleNamespace(open_time=open_time, scheduled_resolution_time=scheduled_resolution_time),
    )


class TestForecastingWindowTzRobustness:
    """The window string is identical for naive (0.2.54) and aware (0.2.92) inputs."""

    def test_naive_datetimes_render_expected_window(self, frozen_now: None) -> None:
        """0.2.54 reality: naive question datetimes produce the correct window block."""
        output = prompts_module._forecasting_window_str(_question(_OPEN, _RESOLVE))

        assert "Today: 2026-03-02" in output
        assert f"Question opened: 2026-01-01 ({_EXPECTED_ELAPSED_DAYS} days ago)" in output
        assert f"Scheduled to resolve: 2026-05-01 ({_EXPECTED_REMAINING_DAYS} days from now)" in output

    def test_aware_datetimes_render_identical_window(self, frozen_now: None) -> None:
        """0.2.92 reality: tz-aware UTC question datetimes must not raise and must
        render the SAME string as the naive fixture (naive == UTC assumption)."""
        naive_output = prompts_module._forecasting_window_str(_question(_OPEN, _RESOLVE))
        aware_output = prompts_module._forecasting_window_str(
            _question(_OPEN.replace(tzinfo=UTC), _RESOLVE.replace(tzinfo=UTC))
        )

        assert aware_output == naive_output

    def test_mixed_awareness_does_not_raise(self, frozen_now: None) -> None:
        """Defensive: one aware + one naive question datetime still normalizes cleanly.

        ft only ever produces uniformly-naive (0.2.54) or uniformly-aware (0.2.92)
        questions, but the per-operand ``_as_utc`` normalization means a mixed pair
        must not raise either — proving both sides are normalized, not just one.
        """
        output = prompts_module._forecasting_window_str(_question(_OPEN, _RESOLVE.replace(tzinfo=UTC)))
        assert f"({_EXPECTED_ELAPSED_DAYS} days ago)" in output
        assert f"({_EXPECTED_REMAINING_DAYS} days from now)" in output

    def test_negative_control_wrong_now_shifts_deltas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-vacuity: a different frozen ``now`` changes the rendered deltas.

        Guards against the assertions above passing on a stubbed/empty output —
        moving the clock one day earlier must shift both day counts by one.
        """

        class _OneDayEarlier(datetime):
            @classmethod
            def now(cls, tz: object | None = None) -> datetime:  # type: ignore[override]
                shifted = _FROZEN_NOW_UTC.replace(day=1)
                return shifted if tz is not None else shifted.replace(tzinfo=None)

        monkeypatch.setattr(prompts_module, "datetime", _OneDayEarlier)
        output = prompts_module._forecasting_window_str(_question(_OPEN, _RESOLVE))

        # Mar 1 12:00 vs Mar 2 12:00: one fewer elapsed day, one more remaining day.
        assert f"({_EXPECTED_ELAPSED_DAYS - 1} days ago)" in output
        assert f"({_EXPECTED_REMAINING_DAYS + 1} days from now)" in output


class TestAsUtcHelper:
    """Direct coverage of the ``_as_utc`` normalization the seam relies on."""

    def test_naive_assumed_utc(self) -> None:
        naive = datetime(2026, 1, 1, 6, 30, 0)
        result = prompts_module._as_utc(naive)
        assert result.tzinfo is UTC
        assert result == datetime(2026, 1, 1, 6, 30, 0, tzinfo=UTC)

    def test_aware_utc_unchanged(self) -> None:
        aware = datetime(2026, 1, 1, 6, 30, 0, tzinfo=UTC)
        assert prompts_module._as_utc(aware) == aware

    def test_aware_non_utc_converted(self) -> None:
        # A tz-aware, non-UTC instant must be converted, not merely relabeled.
        plus_two = timezone(timedelta(hours=2))
        aware = datetime(2026, 1, 1, 8, 30, 0, tzinfo=plus_two)
        result = prompts_module._as_utc(aware)
        assert result.tzinfo is UTC
        assert result == datetime(2026, 1, 1, 6, 30, 0, tzinfo=UTC)
