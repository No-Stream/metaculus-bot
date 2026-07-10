"""F6: tests for metaculus_bot.shadow_divergence.log_parser_vs_block_divergence.

The function is observability-only — it logs a SHADOW_DIVERGENCE line per
forecaster per question comparing the RAW parser extraction (pre-clamp for
binary, pre-renormalize for MC, pre-sanitize percentile list for numeric)
against the fenced JSON block's declared values. This test file locks in:

  - binary: hand-computed max_abs_diff on a valid block
  - binary regression (F6): raw 0.99 vs block 0.99 → 0.000000 — under the old
    post-processed contract the [0.02, 0.98] clamp manufactured a 0.01 diff
  - absent block: block_present=False block_valid=False
  - invalid block (F4 distinction): fence exists but validation fails →
    block_present=True block_valid=False
  - numeric (F5/F6): raw list[Percentile] vs the block's declared_percentiles,
    keyed on percentile labels (0.1/0.5/0.9), with hand-computed max_abs_diff —
    including on a DISCRETE-typed block, where the old post-processed contract
    compared against a cumulative-probability CDF grid and was meaningless
  - numeric parser failure: raw_parser_value=None → max_abs_diff=N/A
  - multiple choice (F2): hand-computed max over canonicalized option keys,
    option-name normalization, and missing-option-defaults-to-0.0

All at INFO level via caplog.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion, PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.shadow_divergence import log_parser_vs_block_divergence


def _binary_question(qid: int = 42) -> BinaryQuestion:
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = qid
    return q


def _numeric_question(qid: int = 43) -> NumericQuestion:
    q = MagicMock(spec=NumericQuestion)
    q.id_of_question = qid
    return q


def _mc_question(qid: int = 44) -> MultipleChoiceQuestion:
    q = MagicMock(spec=MultipleChoiceQuestion)
    q.id_of_question = qid
    return q


def _mc_option_list(options: dict[str, float]) -> PredictedOptionList:
    """Build a real PredictedOptionList, bypassing the sum-to-1 validator.

    ``model_construct`` skips validation so invalid-sum constructions (the
    missing-option case, where a partially-parsed prediction doesn't sum to 1)
    still produce a genuine PredictedOptionList — the isinstance the MC branch
    now requires of its raw parser input.
    """
    return PredictedOptionList.model_construct(
        predicted_options=[PredictedOption(option_name=name, probability=prob) for name, prob in options.items()]
    )


def _extract_shadow_record(caplog: pytest.LogCaptureFixture) -> logging.LogRecord:
    """Return the single SHADOW_DIVERGENCE INFO record in caplog."""
    matches = [r for r in caplog.records if "SHADOW_DIVERGENCE" in r.getMessage()]
    assert len(matches) == 1, f"expected exactly one SHADOW_DIVERGENCE record, got {len(matches)}: {caplog.records}"
    return matches[0]


class TestBinaryDivergence:
    def test_valid_block_hand_computed_diff(self, caplog: pytest.LogCaptureFixture):
        """A valid binary block with a known posterior_prob → max_abs_diff matches by hand."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.shadow_divergence")
        # Raw parser extracted 0.60, block declared 0.75 → |0.75 - 0.60| = 0.15
        raw_parser_value = 0.60
        reasoning = 'Analysis text.\n\n```json\n{"question_type": "binary", "posterior_prob": 0.75}\n```'
        log_parser_vs_block_divergence(_binary_question(101), raw_parser_value, reasoning, "test-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert rec.levelno == logging.INFO
        assert "qid=101" in msg
        assert "model=test-model" in msg
        assert "type=binary" in msg
        assert "block_present=True" in msg
        assert "block_valid=True" in msg
        assert "max_abs_diff=0.150000" in msg

    def test_raw_value_beyond_clamp_shows_zero_divergence(self, caplog: pytest.LogCaptureFixture):
        """F6 regression: raw parser 0.99 vs block 0.99 → exactly 0.000000.

        Under the old contract the comparison used the POST-PROCESSED value —
        0.99 clamped to BINARY_PROB_MAX=0.98 — so perfect parser-block
        agreement at the extremes logged a phantom max_abs_diff=0.010000,
        biasing the A0 promotion gate. The raw-input contract must report zero.
        """
        caplog.set_level(logging.INFO, logger="metaculus_bot.shadow_divergence")
        reasoning = 'Analysis.\n\n```json\n{"question_type": "binary", "posterior_prob": 0.99}\n```'
        log_parser_vs_block_divergence(_binary_question(104), 0.99, reasoning, "test-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "block_valid=True" in msg
        assert "max_abs_diff=0.000000" in msg

    def test_absent_block(self, caplog: pytest.LogCaptureFixture):
        """No fenced JSON block → block_present=False block_valid=False max_abs_diff=N/A."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.shadow_divergence")
        log_parser_vs_block_divergence(_binary_question(102), 0.42, "no fence here", "test-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "qid=102" in msg
        assert "type=binary" in msg
        assert "block_present=False" in msg
        assert "block_valid=False" in msg
        assert "max_abs_diff=N/A" in msg

    def test_invalid_block_present_but_validation_fails(self, caplog: pytest.LogCaptureFixture):
        """F4: fence exists but Pydantic validation fails → block_present=True block_valid=False."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.shadow_divergence")
        # Wrong question_type (says "numeric" while we probe as binary) — the
        # question-type-mismatch guard in parse_structured_block rejects it.
        reasoning = 'text\n```json\n{"question_type": "numeric", "posterior_prob": 0.5}\n```'
        log_parser_vs_block_divergence(_binary_question(103), 0.42, reasoning, "test-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "block_present=True" in msg
        assert "block_valid=False" in msg
        assert "max_abs_diff=N/A" in msg


class TestNumericDivergence:
    def test_raw_percentile_list_hand_computed_diff(self, caplog: pytest.LogCaptureFixture):
        """F6 numeric branch: raw list[Percentile] vs block.declared_percentiles."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.shadow_divergence")

        # Raw parser output is a plain list[Percentile] (pre-sanitize).
        # Block declares three percentiles at the required keys; parser matches
        # two exactly and diverges at 0.5 by 10.0. Include only the required
        # {0.1, 0.5, 0.9} keys (schema validator).
        raw_percentiles = [
            Percentile(percentile=0.1, value=10.0),
            Percentile(percentile=0.5, value=50.0),
            Percentile(percentile=0.9, value=90.0),
        ]
        reasoning = (
            "rationale\n"
            "```json\n"
            '{"question_type": "numeric",'
            ' "declared_percentiles": {"0.1": 10.0, "0.5": 60.0, "0.9": 90.0}}\n'
            "```"
        )
        log_parser_vs_block_divergence(_numeric_question(201), raw_percentiles, reasoning, "num-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "qid=201" in msg
        assert "type=numeric" in msg
        assert "block_present=True" in msg
        assert "block_valid=True" in msg
        # Hand-computed: max(|10-10|, |50-60|, |90-90|) = 10.0
        assert "max_abs_diff=10.000000" in msg

    def test_discrete_block_compares_on_percentile_labels(self, caplog: pytest.LogCaptureFixture):
        """F6 regression: a DISCRETE-typed block still compares on percentile labels.

        Under the old contract the numeric comparison read the post-processed
        distribution's declared_percentiles — which, for DISCRETE questions,
        had been resampled onto a CDF grid keyed by cumulative probabilities
        (0..1 grid steps), NOT percentile labels. The key spaces were disjoint,
        so every lookup missed and the metric was meaningless. With the raw
        percentile list the keys are the labels (0.1/0.5/0.9) on both sides and
        the diff is a hand-computable comparison: max(|12-10|, |50-50|, |90-90|) = 2.0.
        """
        caplog.set_level(logging.INFO, logger="metaculus_bot.shadow_divergence")

        raw_percentiles = [
            Percentile(percentile=0.1, value=12.0),
            Percentile(percentile=0.5, value=50.0),
            Percentile(percentile=0.9, value=90.0),
        ]
        reasoning = (
            "rationale\n"
            "```json\n"
            '{"question_type": "numeric", "outcome_type": "discrete_integer",'
            ' "declared_percentiles": {"0.1": 10.0, "0.5": 50.0, "0.9": 90.0}}\n'
            "```"
        )
        log_parser_vs_block_divergence(_numeric_question(203), raw_percentiles, reasoning, "num-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "block_valid=True" in msg
        assert "max_abs_diff=2.000000" in msg

    def test_parser_failure_logs_na(self, caplog: pytest.LogCaptureFixture):
        """Numeric parser failure (raw_parser_value=None) → max_abs_diff=N/A, block flags intact."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.shadow_divergence")
        reasoning = (
            "rationale\n"
            "```json\n"
            '{"question_type": "numeric",'
            ' "declared_percentiles": {"0.1": 10.0, "0.5": 50.0, "0.9": 90.0}}\n'
            "```"
        )
        log_parser_vs_block_divergence(_numeric_question(204), None, reasoning, "num-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "type=numeric" in msg
        assert "block_present=True" in msg
        assert "block_valid=True" in msg
        assert "max_abs_diff=N/A" in msg

    def test_numeric_absent_block(self, caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.INFO, logger="metaculus_bot.shadow_divergence")
        raw_percentiles = [
            Percentile(percentile=0.5, value=1.0),
        ]
        log_parser_vs_block_divergence(_numeric_question(202), raw_percentiles, "no fence", "num-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "type=numeric" in msg
        assert "block_present=False" in msg
        assert "block_valid=False" in msg
        assert "max_abs_diff=N/A" in msg


class TestMcDivergence:
    def test_valid_block_hand_computed_diff(self, caplog: pytest.LogCaptureFixture):
        """F2 MC branch: max over per-option |parser - block| on canonicalized keys."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.shadow_divergence")

        # Block: {Alpha: 0.50, Beta: 0.30, Gamma: 0.20} (sums to 1.0, schema-valid).
        # Raw parser: {Alpha: 0.62, Beta: 0.28, Gamma: 0.10}.
        # Hand-computed diffs: |0.62-0.50|=0.12, |0.28-0.30|=0.02, |0.10-0.20|=0.10
        # → max = 0.12.
        raw_parser_value = _mc_option_list({"Alpha": 0.62, "Beta": 0.28, "Gamma": 0.10})
        reasoning = (
            "rationale\n"
            "```json\n"
            '{"question_type": "multiple_choice",'
            ' "option_probs": {"Alpha": 0.50, "Beta": 0.30, "Gamma": 0.20}}\n'
            "```"
        )
        log_parser_vs_block_divergence(_mc_question(301), raw_parser_value, reasoning, "mc-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "qid=301" in msg
        assert "type=multiple_choice" in msg
        assert "block_present=True" in msg
        assert "block_valid=True" in msg
        assert "max_abs_diff=0.120000" in msg

    def test_option_name_normalization(self, caplog: pytest.LogCaptureFixture):
        """Option names differing only in case/whitespace must map to the same canonical key.

        Block declares "Yes"/"No"; parser declares "yes "/" NO". If canonicalization
        (strip + lower) failed, the key sets would be disjoint and every option
        would compare against a 0.0 default, giving max = 0.60. The asserted 0.05
        is only reachable when the keys match.
        """
        caplog.set_level(logging.INFO, logger="metaculus_bot.shadow_divergence")

        # Hand-computed on canonical keys: |0.55-0.60|=0.05, |0.45-0.40|=0.05 → max = 0.05.
        raw_parser_value = _mc_option_list({"yes ": 0.55, " NO": 0.45})
        reasoning = (
            'rationale\n```json\n{"question_type": "multiple_choice", "option_probs": {"Yes": 0.6, "No": 0.4}}\n```'
        )
        log_parser_vs_block_divergence(_mc_question(302), raw_parser_value, reasoning, "mc-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "block_valid=True" in msg
        assert "max_abs_diff=0.050000" in msg

    def test_missing_option_defaults_to_zero(self, caplog: pytest.LogCaptureFixture):
        """An option present in the block but absent from the parser output compares against 0.0.

        Block: {A: 0.5, B: 0.5}; parser declares ONLY A=0.7 (model_construct
        bypasses PredictedOptionList's sum-to-1 validator, matching how a
        partially-parsed prediction would look). Union of keys = {a, b}:
        |0.7-0.5| = 0.20 on A, |0.0-0.5| = 0.50 on B → max = 0.50, NOT 0.20.
        """
        caplog.set_level(logging.INFO, logger="metaculus_bot.shadow_divergence")

        raw_parser_value = _mc_option_list({"A": 0.7})
        reasoning = (
            'rationale\n```json\n{"question_type": "multiple_choice", "option_probs": {"A": 0.5, "B": 0.5}}\n```'
        )
        log_parser_vs_block_divergence(_mc_question(303), raw_parser_value, reasoning, "mc-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "block_valid=True" in msg
        assert "max_abs_diff=0.500000" in msg
