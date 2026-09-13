"""The value grid every PIT is read off, reconstructed from a record's ``scaling`` block.

Covers ``performance_analysis.scaling``: which ``zero_point`` sentinel means log scale, and
whether the grid comes from the API's ``continuous_range`` or is rebuilt.
"""

import numpy as np

from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.performance_analysis.scaling import cdf_and_grid, grid_zero_point
from tests.width_monitor_fakes import GRID_N


class TestLogScaleGrid:
    """Regression: a log-scale question serializes ``zero_point == 0`` with a
    positive ``range_min`` (the geometric grid uses ``ratio = range_max /
    range_min``). The old code treated 0 as the linear sentinel and rebuilt a
    linear grid, corrupting the value grid by up to ~0.55 span-normalized on
    9 real questions in the 2026-07-18 width audit. The fix (a) prefers the
    API's grid-exact ``continuous_range`` when present, (b) otherwise
    reconstructs the geometric grid instead of a linear one.
    """

    def test_grid_zero_point_treats_zero_as_log_when_range_min_positive(self):
        """``zero_point == 0`` with a positive floor is a genuine log scale, so the 0.0 is kept."""
        assert grid_zero_point(0, 100.0) == 0.0
        assert grid_zero_point(0.0, 100.0) == 0.0
        # zero_point==0 with a non-positive floor can't be a log transform => drop.
        assert grid_zero_point(0, 0.0) is None
        assert grid_zero_point(0, -5.0) is None
        # A genuinely-absent zero_point is linear.
        assert grid_zero_point(None, 100.0) is None
        # A real (nonzero) zero_point is passed through.
        assert grid_zero_point(50, 100.0) == 50.0

    def test_reconstructed_grid_is_geometric_for_zero_point_zero(self):
        """With no ``continuous_range`` in the record, ``cdf_and_grid`` must rebuild the GEOMETRIC
        grid (ratio = range_max / range_min), not a linear ramp."""
        lower, upper = 100.0, 1000.0
        cdf = np.linspace(0.0, 1.0, GRID_N).tolist()
        rec = {
            "type": "numeric",
            "our_forecast_values": cdf,
            "resolution_parsed": 500.0,
            "scaling": {"range_min": lower, "range_max": upper, "zero_point": 0},
        }
        built = cdf_and_grid(rec)
        assert built is not None
        _cdf_arr, grid = built
        expected_geometric = build_cdf_value_grid(lower, upper, 0.0, GRID_N)
        expected_linear = build_cdf_value_grid(lower, upper, None, GRID_N)
        # Matches the geometric grid, and is materially different from linear.
        np.testing.assert_allclose(grid, expected_geometric, rtol=0, atol=1e-9)
        assert float(np.max(np.abs(expected_geometric - expected_linear))) > 1.0

    def test_continuous_range_is_preferred_when_present(self):
        """The API grid is used verbatim when present, whatever the ``zero_point`` sentinel says.

        It is already log/linear correct.
        """
        lower, upper = 100.0, 1000.0
        api_grid = build_cdf_value_grid(lower, upper, 0.0, GRID_N)
        cdf = np.linspace(0.0, 1.0, GRID_N).tolist()
        rec = {
            "type": "numeric",
            "our_forecast_values": cdf,
            "resolution_parsed": 500.0,
            "scaling": {
                "range_min": lower,
                "range_max": upper,
                "zero_point": 0,
                "continuous_range": api_grid.tolist(),
            },
        }
        built = cdf_and_grid(rec)
        assert built is not None
        _cdf_arr, grid = built
        np.testing.assert_array_equal(grid, api_grid)
