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

from forecasting_tools.data_models.numeric_report import Percentile
from pydantic import BaseModel, ConfigDict

from metaculus_bot.numeric.config import STANDARD_PERCENTILES

# Match the rounding convention used for float-keyed percentile matching in
# metaculus_bot/numeric/validation.py so lookups are robust to float noise.
_KEY_DECIMALS: int = 6

_EXPECTED_KEYS: frozenset[float] = frozenset(round(p, _KEY_DECIMALS) for p in STANDARD_PERCENTILES)


def _key(percentile: float) -> float:
    """Round a percentile label to the canonical lookup key."""
    return round(float(percentile), _KEY_DECIMALS)


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
        keyed = {_key(p): float(v) for p, v in mapping.items()}
        _validate_keys(keyed.keys())
        return cls(values_by_percentile=keyed)

    @classmethod
    def from_percentiles(cls, percentiles: list[Percentile]) -> PercentileSet:
        keyed = {_key(p.percentile): float(p.value) for p in percentiles}
        if len(keyed) != len(percentiles):
            raise ValueError(
                f"PercentileSet: duplicate percentile labels in {sorted(_key(p.percentile) for p in percentiles)}"
            )
        _validate_keys(keyed.keys())
        return cls(values_by_percentile=keyed)

    def value_at(self, percentile: float) -> float:
        """Return the value declared at ``percentile`` (matched via rounding).

        Raises ``KeyError`` on an unknown label — never returns a neighbor.
        """
        key = _key(percentile)
        if key not in self.values_by_percentile:
            raise KeyError(f"PercentileSet has no percentile {key}; known labels: {sorted(self.values_by_percentile)}")
        return self.values_by_percentile[key]

    def values_sorted(self) -> list[float]:
        """Values in ascending percentile order (for CDF builders)."""
        return [self.values_by_percentile[k] for k in sorted(self.values_by_percentile)]

    def as_percentile_list(self) -> list[Percentile]:
        """Reconstruct the ``list[Percentile]`` form in ascending percentile order."""
        return [Percentile(percentile=k, value=self.values_by_percentile[k]) for k in sorted(self.values_by_percentile)]


def _validate_keys(keys: object) -> None:
    actual = frozenset(keys)  # type: ignore[arg-type]
    if actual == _EXPECTED_KEYS:
        return
    missing = sorted(_EXPECTED_KEYS - actual)
    extra = sorted(actual - _EXPECTED_KEYS)
    problems: list[str] = []
    if missing:
        problems.append(f"missing {missing}")
    if extra:
        problems.append(f"extra {extra}")
    raise ValueError(
        f"PercentileSet requires exactly the standard percentiles {sorted(_EXPECTED_KEYS)}; " + "; ".join(problems)
    )
