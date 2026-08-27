"""Regression tests pinning numeric_config defaults to their empirically chosen values.

These tests guard against accidental reversion of the tail-widening defaults flipped
2026-05-12 in response to `scratch_docs_and_planning/tail_widening_empirical_calibration.md`.
On 43 resolved numerics (Feb-May 2026), k_tail=1.0 produced PIT std closest to the
ideal 0.289 in every segment; k_tail=1.25 moved away from ideal in every segment.
The span_floor_gamma floor never bound on real ensemble-averaged declared percentiles,
so the default was dropped to 0.0 (the floor enforcement at tail_widening.py:171/178
stays correctly gated on `> 0` and re-enables if a forecaster sets it back).
"""

from metaculus_bot.numeric import config as numeric_config


def test_standard_percentiles_is_13_with_p1_and_p99():
    """The standard set is the 13 percentiles incl. P1 (0.01) and P99 (0.99), sorted ascending.

    P1/P99 were added (11 -> 13) to give forecasters finer tail anchors so they can
    express probability mass below an open lower bound (the Minions & Monsters miss).
    """
    expected = [0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.975, 0.99]
    assert expected == numeric_config.STANDARD_PERCENTILES
    assert numeric_config.EXPECTED_PERCENTILE_COUNT == 13
    assert len(numeric_config.STANDARD_PERCENTILES) == 13
    assert sorted(numeric_config.STANDARD_PERCENTILES) == numeric_config.STANDARD_PERCENTILES
    assert 0.01 in numeric_config.STANDARD_PERCENTILES
    assert 0.99 in numeric_config.STANDARD_PERCENTILES


def test_standard_percentiles_csv_is_generated_from_constant():
    """The CSV label string used in prompts/errors is derived from STANDARD_PERCENTILES, not hardcoded."""
    assert numeric_config.STANDARD_PERCENTILES_CSV == "1,2.5,5,10,20,40,50,60,80,90,95,97.5,99"
    # Must track the constant: every label is its percentile * 100 formatted with %g.
    assert (
        ",".join(f"{p * 100:g}" for p in numeric_config.STANDARD_PERCENTILES) == numeric_config.STANDARD_PERCENTILES_CSV
    )


def test_tail_widen_k_tail_default_is_one():
    """TAIL_WIDEN_K_TAIL default must be 1.0 (no widening) per empirical calibration.

    See scratch_docs_and_planning/tail_widening_empirical_calibration.md.
    """
    assert numeric_config.TAIL_WIDEN_K_TAIL == 1.0


def test_tail_widen_span_floor_gamma_default_is_zero():
    """TAIL_WIDEN_SPAN_FLOOR_GAMMA default must be 0.0 (floor check disabled).

    Floor enforcement at tail_widening.py:171/178 is gated on `> 0`; the floor
    never bound on 2026 data. See
    scratch_docs_and_planning/tail_widening_empirical_calibration.md section 3.
    """
    assert numeric_config.TAIL_WIDEN_SPAN_FLOOR_GAMMA == 0.0


def test_tail_widening_enable_flag_still_present():
    """The enable flag stays available so tests and env overrides can re-enable widening."""
    assert hasattr(numeric_config, "TAIL_WIDENING_ENABLE")
    assert isinstance(numeric_config.TAIL_WIDENING_ENABLE, bool)
