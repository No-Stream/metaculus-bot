"""Safe numeric-CDF access + fallback bookkeeping for ensemble simulation.

``NumericCdfCache`` owns the ``_safe_cdf_cache`` (per (model, qid) memoization of a
usable CDF) and the ``_numeric_cdf_stats`` counters that track how often we fall
back to a PCHIP rebuild or a monotone ramp. Extracted from ``CorrelationAnalyzer``
as the "CDF cache" concern; the analyzer holds one instance and delegates
``_get_safe_numeric_cdf`` / ``log_numeric_cdf_summary`` to it.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.numeric.pchip_cdf import generate_pchip_cdf, percentiles_to_pchip_format

logger = logging.getLogger(__name__)

# The final-fallback ramp is built on the standard 201-point Metaculus grid with the
# server's 201-grid min step, so it satisfies the strictly-increasing constraint.
_RAMP_POINTS = 201
_RAMP_MIN_STEP = 5e-05


class NumericCdfCache:
    """Memoizing safe-CDF accessor with PCHIP-rebuild and monotone-ramp fallbacks.

    NOTE: callers that mutate the underlying benchmark set (e.g.
    ``CorrelationAnalyzer.add_benchmark_results`` / ``filter_models_inplace``) do NOT
    currently clear this cache. If benchmarks change in a way that reuses the same
    (model_name, question_id) keys with different predictions, cached CDFs could go
    stale. In practice a fresh ``CorrelationAnalyzer`` is constructed per analysis run,
    so this has not bitten us; call ``clear()`` if that assumption ever changes.
    """

    def __init__(self) -> None:
        self._safe_cdf_cache: dict[tuple[str, int], list[Any] | None] = {}
        self._numeric_cdf_stats: dict[str, Any] = {
            "attempt_pairs": set(),  # set[(model, qid)]
            "safe_cdf_built": set(),  # set[(model, qid)]
            "safe_cdf_ramp": set(),  # set[(model, qid)]
            "failures": set(),  # set[(model, qid)]
            "first_warnings_emitted": set(),  # set[(model, qid)]
        }

    def clear(self) -> None:
        """Drop all memoized CDFs and reset fallback counters."""
        self._safe_cdf_cache.clear()
        for v in self._numeric_cdf_stats.values():
            v.clear()

    def get_safe_numeric_cdf(self, model_name: str, question: Any, prediction: Any) -> list[Any] | None:
        """Return a safe numeric CDF as a list of objects with `.percentile` and `.value`.

        Attempts `prediction.cdf` first. If items are floats or missing `.value`, synthesize a
        reasonable x-grid from question bounds. If `prediction.cdf` raises, rebuild from
        declared percentiles via PCHIP; as last resort, return a monotone ramp. All paths return
        objects convertible to the NumericDistribution "Percentile"-like shape required by
        downstream scoring (which only reads `.percentile`).

        Each rung is a deliberate soft-fail: one unusable prediction must degrade to the
        next rung (and ultimately to ``None``) rather than abort an analysis run. Which
        rung answered is recorded in ``_numeric_cdf_stats`` and reported by
        :meth:`log_numeric_cdf_summary`.
        """
        qid = getattr(question, "id_of_question", None)
        if qid is None:
            qid = -1
        key = (model_name, int(qid))

        stats = self._numeric_cdf_stats
        stats["attempt_pairs"].add(key)

        if key in self._safe_cdf_cache:
            return self._safe_cdf_cache[key]

        declared_cdf = self._cdf_read_directly(question, prediction, key=key, model_name=model_name, qid=qid)
        if declared_cdf is not None:
            self._safe_cdf_cache[key] = declared_cdf
            return declared_cdf

        rebuilt_cdf = self._cdf_rebuilt_from_percentiles(question, prediction, key=key, model_name=model_name, qid=qid)
        if rebuilt_cdf is not None:
            self._safe_cdf_cache[key] = rebuilt_cdf
            stats["safe_cdf_built"].add(key)
            return rebuilt_cdf

        ramp_cdf = self._monotone_ramp_cdf(question)
        if ramp_cdf is not None:
            self._safe_cdf_cache[key] = ramp_cdf
            stats["safe_cdf_ramp"].add(key)
            return ramp_cdf

        stats["failures"].add(key)
        self._safe_cdf_cache[key] = None
        return None

    def _warn_once(self, key: tuple[str, int], message: str, *args: object) -> None:
        """Log ``message`` the first time a (model, question) pair needs a fallback."""
        emitted = self._numeric_cdf_stats["first_warnings_emitted"]
        if key not in emitted:
            logger.warning(message, *args)
            emitted.add(key)

    def _cdf_read_directly(
        self,
        question: Any,
        prediction: Any,
        *,
        key: tuple[str, int],
        model_name: str,
        qid: int,
    ) -> list[Any] | None:
        """Rung 1: the prediction's own ``.cdf``, given an x-grid when it carries none.

        Returns None both when ``.cdf`` raises (warned, so the caller falls through to
        the rebuild) and when it is too short to be a CDF (silent — nothing failed, the
        prediction simply has nothing to read).
        """
        try:
            raw = prediction.cdf
            if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                return None

            first = raw[0]
            has_percentile = isinstance(first, (Percentile, SimpleNamespace)) and hasattr(first, "percentile")
            has_value = isinstance(first, (Percentile, SimpleNamespace)) and hasattr(first, "value")
            if has_percentile and has_value:
                return list(raw)

            x = np.linspace(float(question.lower_bound), float(question.upper_bound), len(raw))
            if has_percentile:
                return [
                    SimpleNamespace(value=float(xi), percentile=float(p.percentile))
                    for xi, p in zip(x, raw, strict=True)
                ]
            # Percentiles as bare floats
            return [SimpleNamespace(value=float(xi), percentile=float(pi)) for xi, pi in zip(x, raw, strict=True)]
        except Exception as e:  # noqa: BLE001  # soft-fail rung: any bad `.cdf` degrades to the rebuild below
            self._warn_once(
                key,
                "Numeric CDF access failed for model=%s q=%s: %s — attempting safe rebuild",
                model_name,
                qid,
                e,
            )
            return None

    def _cdf_rebuilt_from_percentiles(
        self,
        question: Any,
        prediction: Any,
        *,
        key: tuple[str, int],
        model_name: str,
        qid: int,
    ) -> list[Any] | None:
        """Rung 2: rebuild a 201-point CDF from the model's declared percentiles via PCHIP."""
        try:
            lower = getattr(question, "lower_bound", None)
            upper = getattr(question, "upper_bound", None)
            if lower is None or upper is None:
                raise ValueError("missing bounds")

            declared = getattr(prediction, "declared_percentiles", None)
            if not declared:
                raise ValueError("no declared_percentiles to rebuild from")

            # Convert to pchip format and rebuild CDF values
            pv = percentiles_to_pchip_format(declared)
            # Use open-bound flags if available
            open_lower = bool(getattr(question, "open_lower_bound", False))
            open_upper = bool(getattr(question, "open_upper_bound", False))
            # zero_point should be None for discrete or unknown
            zero_point = getattr(question, "zero_point", None)
            # For discrete numeric (non-201 bins), ignore zero_point to avoid singularities
            cdf_size = int(getattr(question, "cdf_size", 201) or 201)
            zp = None if cdf_size != 201 else zero_point
            cdf_vals, _ = generate_pchip_cdf(
                pv,
                open_upper_bound=open_upper,
                open_lower_bound=open_lower,
                upper_bound=float(upper),
                lower_bound=float(lower),
                zero_point=zp,
                num_points=201,
                question_id=qid,
                question_url=getattr(question, "page_url", None),
            )
            # Ensure monotone and within [0,1]
            cdf_vals = list(np.maximum.accumulate(np.clip(np.array(cdf_vals, dtype=float), 0.0, 1.0)))
            x = np.linspace(float(lower), float(upper), len(cdf_vals))
            return [SimpleNamespace(value=float(xi), percentile=float(pi)) for xi, pi in zip(x, cdf_vals, strict=True)]
        except Exception as e:  # noqa: BLE001  # soft-fail rung: any rebuild failure degrades to the ramp below
            self._warn_once(
                key,
                "Numeric CDF rebuild failed for model=%s q=%s: %s — using monotone ramp",
                model_name,
                qid,
                e,
            )
            return None

    def _monotone_ramp_cdf(self, question: Any) -> list[Any] | None:
        """Rung 3: a uniform ramp over the question's range, min-step enforced.

        Returns None only when the question exposes no usable bounds at all, which is
        the one case with nothing left to fall back to.
        """
        try:
            vals = list(np.linspace(0.0, 1.0, _RAMP_POINTS))
            for i in range(1, _RAMP_POINTS):
                if vals[i] < vals[i - 1] + _RAMP_MIN_STEP:
                    vals[i] = min(1.0, vals[i - 1] + _RAMP_MIN_STEP)
            if vals[-1] > 1.0:
                vals[-1] = 1.0
            x = np.linspace(float(question.lower_bound), float(question.upper_bound), _RAMP_POINTS)
            return [SimpleNamespace(value=float(xi), percentile=float(pi)) for xi, pi in zip(x, vals, strict=True)]
        except Exception:  # noqa: BLE001  # last soft-fail rung: an unusable question yields None, never a crash
            return None

    def log_numeric_cdf_summary(self) -> None:
        """Log a one-line summary of numeric CDF safety fallbacks to detect systemic issues."""
        s = self._numeric_cdf_stats
        try:
            attempts = len(s["attempt_pairs"]) or 0
            built = len(s["safe_cdf_built"]) or 0
            ramp = len(s["safe_cdf_ramp"]) or 0
            fails = len(s["failures"]) or 0
            if attempts > 0:
                logger.info(
                    "Numeric CDF safety summary: attempts=%d, rebuilt=%d, ramp=%d, failures=%d",
                    attempts,
                    built,
                    ramp,
                    fails,
                )
        except (KeyError, TypeError):
            logger.debug("Failed to compute numeric CDF summary statistics")
