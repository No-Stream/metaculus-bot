"""Fixed-point number rendering, shared by the FRED block and the inline-chart rung.

Stdlib only, deliberately. The two consumers are ``research/fred_rendering.py``, which
imports pandas and fredapi, and ``research/resolution_chart_data.py``, which imports
nothing outside the standard library and ``constants``; putting the rule in either one
would drag that module's dependencies into the other.

The rule itself is one piece of reasoning. A resolution source graded on an index level
reads 331.893, not "331.9" and not "3.31893e+02", so a value renders fixed-point at up to
six decimals with trailing zeros stripped, and never in scientific notation. Six decimals
covers everything FRED publishes (index levels at three, most rates at two, a few series
at four). It replaced ``:.4g``, which rounded a Case-Shiller print of 331.893 to "331.9"
on a question whose displayed range was four index points wide with 0.02-point buckets, so
the digits it threw away were the whole forecast, and which turned the Fed balance sheet
into "6.7e+06".

This is NOT the timeseries anchor's formatter (``ts_render._fmt``), which uses three
decimals above 100 and ``:.4g`` below on purpose: that surface also renders an estimated
P10/P50/P90 band, where six decimals would be fabricated precision.
"""

DECIMAL_PLACES = 6


def format_decimal_value(value: float) -> str:
    """``value`` at its published precision: fixed-point, no scientific notation.

    Cleans up float subtraction artifacts for free, so a change computed as
    331.893 - 331.020 = 0.8729999999999905 renders "0.873". Never returns "-0": the sign
    of a quantity too small to show at this precision is not information.
    """
    text = f"{float(value):.{DECIMAL_PLACES}f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-", "-0"} else "0"


def format_decimal_change(change: float) -> str:
    """A signed change at the same precision (``:+`` cannot drive a custom formatter).

    A change that rounds to zero at this precision renders "+0", never "-0", for the same
    reason as above.
    """
    magnitude = format_decimal_value(abs(float(change)))
    sign = "-" if float(change) < 0 and magnitude != "0" else "+"
    return f"{sign}{magnitude}"
