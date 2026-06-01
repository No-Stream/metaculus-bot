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

logger = logging.getLogger(__name__)


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
        """
        # Local import to avoid module-level dependency and to satisfy linters for this scope
        from metaculus_bot.numeric.pchip_cdf import generate_pchip_cdf, percentiles_to_pchip_format

        qid = getattr(question, "id_of_question", None)
        if qid is None:
            qid = -1
        key = (model_name, int(qid))

        # Stats bookkeeping
        stats = self._numeric_cdf_stats
        stats["attempt_pairs"].add(key)

        # Cache lookup
        if key in self._safe_cdf_cache:
            return self._safe_cdf_cache[key]

        # Try direct access
        try:
            raw = prediction.cdf
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                first = raw[0]
                has_percentile = isinstance(first, (Percentile, SimpleNamespace)) and hasattr(first, "percentile")
                has_value = isinstance(first, (Percentile, SimpleNamespace)) and hasattr(first, "value")
                if has_percentile and has_value:
                    self._safe_cdf_cache[key] = list(raw)
                    return list(raw)
                elif has_percentile:
                    lower = getattr(question, "lower_bound", 0.0)
                    upper = getattr(question, "upper_bound", 1.0)
                    n = len(raw)
                    x = np.linspace(float(lower), float(upper), n)
                    out = []
                    for xi, p in zip(x, raw):
                        out.append(SimpleNamespace(value=float(xi), percentile=float(p.percentile)))
                    self._safe_cdf_cache[key] = out
                    return out
                else:
                    # Percentiles as bare floats
                    lower = getattr(question, "lower_bound", 0.0)
                    upper = getattr(question, "upper_bound", 1.0)
                    n = len(raw)
                    x = np.linspace(float(lower), float(upper), n)
                    out = [SimpleNamespace(value=float(xi), percentile=float(pi)) for xi, pi in zip(x, raw)]
                    self._safe_cdf_cache[key] = out
                    return out
        except Exception as e:
            if key not in stats["first_warnings_emitted"]:
                logger.warning(
                    "Numeric CDF access failed for model=%s q=%s: %s — attempting safe rebuild",
                    model_name,
                    qid,
                    e,
                )
                stats["first_warnings_emitted"].add(key)

        # Rebuild from declared percentiles via PCHIP
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
            out = [SimpleNamespace(value=float(xi), percentile=float(pi)) for xi, pi in zip(x, cdf_vals)]
            self._safe_cdf_cache[key] = out
            stats["safe_cdf_built"].add(key)
            return out
        except Exception as e:
            if key not in stats["first_warnings_emitted"]:
                logger.warning(
                    "Numeric CDF rebuild failed for model=%s q=%s: %s — using monotone ramp",
                    model_name,
                    qid,
                    e,
                )
                stats["first_warnings_emitted"].add(key)

        # Final fallback: monotone ramp respecting min step
        try:
            n = 201
            vals = list(np.linspace(0.0, 1.0, n))
            # Enforce min step 5e-05
            min_step = 5e-05
            for i in range(1, n):
                if vals[i] < vals[i - 1] + min_step:
                    vals[i] = min(1.0, vals[i - 1] + min_step)
            if vals[-1] > 1.0:
                vals[-1] = 1.0
            lower = getattr(question, "lower_bound", 0.0)
            upper = getattr(question, "upper_bound", 1.0)
            x = np.linspace(float(lower), float(upper), n)
            out = [SimpleNamespace(value=float(xi), percentile=float(pi)) for xi, pi in zip(x, vals)]
            self._safe_cdf_cache[key] = out
            stats["safe_cdf_ramp"].add(key)
            return out
        except Exception:
            stats["failures"].add(key)
            self._safe_cdf_cache[key] = None
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
        except Exception:
            logger.debug("Failed to compute numeric CDF summary statistics")
