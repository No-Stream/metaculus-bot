"""Label-addressed value object for a complete set of forecast percentiles.

The forecasting pipeline historically indexed ``list[Percentile]`` positionally
(e.g. ``pcts[2]`` to mean "the P10 value"). That is a latent foot-gun: when the
standard percentile set grows (11 -> 13, adding P1/P99), every hardcoded index
silently shifts and points at the wrong percentile with no error.

``PercentileSet`` makes that bug class impossible. It validates completeness
against ``STANDARD_PERCENTILES`` at construction and only exposes label-addressed
access (``value_at(0.10)``) plus explicit ordered views. There is deliberately no
integer ``__getitem__``.
"""

from __future__ import annotations

from collections.abc import Iterable

from forecasting_tools.data_models.numeric_report import Percentile
from pydantic import BaseModel, ConfigDict

from metaculus_bot.numeric.config import STANDARD_PERCENTILES

# Canonical rounding convention for float-keyed percentile matching, shared by
# every module that compares percentile labels (spread_metrics, validation) so
# lookups are robust to float noise.
PERCENTILE_KEY_DECIMALS: int = 6

# The canonical standard-percentile label set, rounded to lookup keys. Single
# source of truth — reuse this instead of re-deriving from STANDARD_PERCENTILES.
EXPECTED_KEYS: frozenset[float] = frozenset(round(p, PERCENTILE_KEY_DECIMALS) for p in STANDARD_PERCENTILES)


def percentile_key(percentile: float) -> float:
    """Round a percentile label to the canonical lookup key."""
    return round(float(percentile), PERCENTILE_KEY_DECIMALS)


class PercentileSet(BaseModel):
    """A complete, label-addressed set of the standard forecast percentiles.

    Construct via :meth:`from_percentiles` or :meth:`from_mapping`; both validate
    that the key set exactly equals ``STANDARD_PERCENTILES``. Access values by
    their percentile label with :meth:`value_at` — never by list position.
    """

    model_config = ConfigDict(frozen=True)

    values_by_percentile: dict[float, float]

    @classmethod
    def from_mapping(cls, mapping: dict[float, float]) -> PercentileSet:
        keyed = {percentile_key(p): float(v) for p, v in mapping.items()}
        if len(keyed) != len(mapping):
            raise ValueError(
                f"PercentileSet: duplicate percentile labels in {sorted(percentile_key(p) for p in mapping)}"
            )
        _validate_keys(keyed.keys())
        return cls(values_by_percentile=keyed)

    @classmethod
    def from_percentiles(cls, percentiles: list[Percentile]) -> PercentileSet:
        keyed = {percentile_key(p.percentile): float(p.value) for p in percentiles}
        if len(keyed) != len(percentiles):
            raise ValueError(
                f"PercentileSet: duplicate percentile labels in {sorted(percentile_key(p.percentile) for p in percentiles)}"
            )
        _validate_keys(keyed.keys())
        return cls(values_by_percentile=keyed)

    def value_at(self, percentile: float) -> float:
        """Return the value declared at ``percentile`` (matched via rounding).

        Raises ``KeyError`` on an unknown label — never returns a neighbor.
        """
        key = percentile_key(percentile)
        if key not in self.values_by_percentile:
            raise KeyError(f"PercentileSet has no percentile {key}; known labels: {sorted(self.values_by_percentile)}")
        return self.values_by_percentile[key]

    def values_sorted(self) -> list[float]:
        """Values in ascending percentile order (for CDF builders)."""
        return [self.values_by_percentile[k] for k in sorted(self.values_by_percentile)]

    def as_percentile_list(self) -> list[Percentile]:
        """Reconstruct the ``list[Percentile]`` form in ascending percentile order."""
        return [Percentile(percentile=k, value=self.values_by_percentile[k]) for k in sorted(self.values_by_percentile)]


def _validate_keys(keys: Iterable[float]) -> None:
    actual = frozenset(keys)
    if actual == EXPECTED_KEYS:
        return
    missing = sorted(EXPECTED_KEYS - actual)
    extra = sorted(actual - EXPECTED_KEYS)
    problems: list[str] = []
    if missing:
        problems.append(f"missing {missing}")
    if extra:
        problems.append(f"extra {extra}")
    raise ValueError(
        f"PercentileSet requires exactly the standard percentiles {sorted(EXPECTED_KEYS)}; " + "; ".join(problems)
    )
