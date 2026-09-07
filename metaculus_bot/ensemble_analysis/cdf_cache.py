"""Safe numeric-CDF access and failure bookkeeping for ensemble simulation.

``NumericCdfCache`` owns the ``_safe_cdf_cache`` (per (model, qid) memoization of a
usable CDF) and the ``_numeric_cdf_stats`` counters that track rebuilds and failures.
The legacy ``safe_cdf_ramp`` bucket remains in the stats payload at zero so archived
diagnostic summaries retain their field shape; fabricated ramp CDFs are no longer built.
``CorrelationAnalyzer`` shares one cache with its ensemble simulator.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.numeric.pchip_cdf import (
    build_cdf_value_grid,
    generate_pchip_cdf,
    percentiles_to_pchip_format,
)

logger = logging.getLogger(__name__)


class NumericCdfCache:
    """Memoizing safe-CDF accessor with a PCHIP-rebuild fallback.

    Callers that mutate the underlying benchmark set (e.g.
    ``CorrelationAnalyzer.add_benchmark_results`` / ``filter_models_inplace``) clear
    this cache before reusing it. If another caller replaces predictions under the
    same ``(model_name, question_id)`` keys, it must call ``clear()`` as well.
    """

    def __init__(self) -> None:
        self._safe_cdf_cache: dict[tuple[str, int], list[Any] | None] = {}
        self._numeric_cdf_stats: dict[str, Any] = {
            "attempt_pairs": set(),  # set[(model, qid)]
            "safe_cdf_built": set(),  # set[(model, qid)]
            # Retained as an always-empty compatibility bucket for summary consumers.
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
        value grid from question bounds. If `prediction.cdf` raises, rebuild from declared
        percentiles via PCHIP. When neither representation is usable, return ``None`` so the
        caller can exclude the prediction from scoring.

        Which path answered is recorded in ``_numeric_cdf_stats`` and reported by
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

            question_values = build_cdf_value_grid(
                float(question.lower_bound),
                float(question.upper_bound),
                getattr(question, "zero_point", None),
                len(raw),
            )
            if has_percentile:
                return [
                    SimpleNamespace(value=float(question_value), percentile=float(point.percentile))
                    for question_value, point in zip(question_values, raw, strict=True)
                ]
            # Percentiles as bare floats
            return [
                SimpleNamespace(value=float(question_value), percentile=float(probability))
                for question_value, probability in zip(question_values, raw, strict=True)
            ]
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
            question_values = build_cdf_value_grid(float(lower), float(upper), zp, len(cdf_vals))
            return [
                SimpleNamespace(value=float(question_value), percentile=float(probability))
                for question_value, probability in zip(question_values, cdf_vals, strict=True)
            ]
        except Exception as e:  # noqa: BLE001  # soft-fail rung: an unusable prediction is excluded
            self._warn_once(key, "Numeric CDF rebuild failed for model=%s q=%s: %s", model_name, qid, e)
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
