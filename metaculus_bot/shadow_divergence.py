"""A0 shadow-divergence logging: JSON block vs RAW parser extraction.

Extracted from ``forecaster.py`` (F3) so the comparator lives in a small,
importable module instead of bloating the bot class's file. Observability
only — nothing here ever affects the prediction pipeline. FUTURE.md uses
the SHADOW_DIVERGENCE read-out to gate promoting the structured JSON block
to the authoritative prediction source.
"""

from __future__ import annotations

import logging

from forecasting_tools import MetaculusQuestion, PredictedOptionList
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.question_types import question_type_of
from metaculus_bot.structured_output_schema import (
    BinaryStructured,
    MultipleChoiceStructured,
    NumericStructured,
    extract_json_block,
    parse_structured_block,
)

logger = logging.getLogger(__name__)


def log_parser_vs_block_divergence(
    question: MetaculusQuestion,
    raw_parser_value: float | PredictedOptionList | list[Percentile] | None,
    reasoning: str,
    model_name: str,
) -> None:
    """Compare the JSON block's declared values against the RAW parser extraction.

    Called from the three runners in ``forecaster_runners`` at the point where
    the parser output still exists untouched by post-processing (F6):

    - binary: the raw ``prediction_in_decimal`` float, BEFORE the
      [BINARY_PROB_MIN, BINARY_PROB_MAX] clamp;
    - multiple choice: the parsed ``PredictedOptionList``, BEFORE
      ``clamp_and_renormalize_mc`` (on the fallback parse path, the
      ``build_mc_prediction`` output — the closest-to-raw value available);
    - numeric: the parser's raw ``list[Percentile]``, BEFORE
      ``sanitize_percentiles`` / distribution building. ``None`` (parser
      failed) logs ``max_abs_diff=N/A``. Keys are percentile labels
      (0.1/0.5/0.9, ...), matching the block's ``declared_percentiles`` keys
      — unlike the post-processed distribution, whose discrete resampling
      rekeyed onto cumulative CDF probabilities and made the comparison
      meaningless on DISCRETE questions.

    ``max_abs_diff`` therefore reflects TRUE parser-vs-block drift; no
    deterministic post-processing (clamps, renormalization, jitter, discrete
    CDF resampling) is folded in.

    Logs a structured INFO line per forecaster per question. This is
    observability only — it NEVER affects the prediction pipeline. Wrapped in
    a broad except so any failure is logged at WARNING and swallowed — a
    systematic bug in this comparison would otherwise be invisible and
    quietly bias the A0 read-out toward "no drift observed" (F10).
    """
    try:
        q_type_str = question_type_of(question)
        if q_type_str is None:
            return

        # F4: distinguish "no fenced JSON block at all" from "block present but
        # parse_structured_block rejected it" — parse_structured_block returns
        # None for both, so we probe extract_json_block independently.
        block_present = extract_json_block(reasoning) is not None
        block = parse_structured_block(reasoning, q_type_str)
        block_valid = block is not None

        max_abs_diff: float | None = None

        if block is not None:
            if isinstance(block, BinaryStructured) and isinstance(raw_parser_value, float):
                max_abs_diff = abs(block.posterior_prob - raw_parser_value)

            elif isinstance(block, MultipleChoiceStructured) and isinstance(raw_parser_value, PredictedOptionList):
                # Compare on canonicalized option names; options missing from
                # one side default to 0.0.
                parser_probs: dict[str, float] = {
                    opt.option_name.strip().lower(): opt.probability for opt in raw_parser_value.predicted_options
                }
                block_probs: dict[str, float] = {k.strip().lower(): v for k, v in block.option_probs.items()}
                diffs: list[float] = []
                all_keys = set(parser_probs.keys()) | set(block_probs.keys())
                for key in all_keys:
                    p_val = parser_probs.get(key, 0.0)
                    b_val = block_probs.get(key, 0.0)
                    diffs.append(abs(p_val - b_val))
                max_abs_diff = max(diffs) if diffs else 0.0

            elif (
                isinstance(block, NumericStructured)
                and block.declared_percentiles
                and isinstance(raw_parser_value, list)
            ):
                parser_pctiles: dict[float, float] = {float(p.percentile): float(p.value) for p in raw_parser_value}
                diffs_numeric: list[float] = []
                for pct_key, block_val in block.declared_percentiles.items():
                    parser_val = parser_pctiles.get(pct_key)
                    if parser_val is None:
                        # Tolerate small float-key drift between the two sources.
                        for p_key, p_val in parser_pctiles.items():
                            if abs(p_key - pct_key) < 0.001:
                                parser_val = p_val
                                break
                    if parser_val is not None:
                        diffs_numeric.append(abs(parser_val - block_val))
                max_abs_diff = max(diffs_numeric) if diffs_numeric else None

        qid = question.id_of_question
        logger.info(
            "SHADOW_DIVERGENCE: qid=%s model=%s type=%s block_present=%s block_valid=%s max_abs_diff=%s",
            qid,
            model_name,
            q_type_str,
            block_present,
            block_valid,
            f"{max_abs_diff:.6f}" if max_abs_diff is not None else "N/A",
        )
    except Exception:  # noqa: BLE001, HARNESS-SCAN-EXEMPT-broad-except  # observability-only; must never crash pipeline
        # F10: WARNING (not DEBUG) so a systematic bug in the A0 comparison
        # code doesn't hide behind a silent "no drift" read-out.
        logger.warning("Shadow divergence logging failed", exc_info=True)
