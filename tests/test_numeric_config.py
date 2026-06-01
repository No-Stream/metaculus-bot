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
