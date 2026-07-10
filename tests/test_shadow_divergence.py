"""F6: tests for metaculus_bot.shadow_divergence.log_parser_vs_block_divergence.

The function is observability-only — it logs a SHADOW_DIVERGENCE line per
forecaster per question comparing the pipeline's post-processed prediction
against the fenced JSON block's declared values. This test file locks in:

  - binary: hand-computed max_abs_diff on a valid block
  - absent block: block_present=False block_valid=False
  - invalid block (F4 distinction): fence exists but validation fails →
    block_present=True block_valid=False
  - numeric (F5): parser's prediction.declared_percentiles vs the block's
    declared_percentiles, with hand-computed max_abs_diff
  - multiple choice (F2): hand-computed max over canonicalized option keys,
    option-name normalization, and missing-option-defaults-to-0.0

All at INFO level via caplog.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
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


def _mc_parser_prediction(options: dict[str, float]) -> MagicMock:
    """Build a parser-prediction stand-in with ``.predicted_options`` like PredictedOptionList.

    Uses real PredictedOption items (mirrors the numeric tests' real Percentile
    objects) inside a MagicMock holder so invalid-sum constructions (the
    missing-option case) don't trip PredictedOptionList's sum validator.
    """
    pred = MagicMock()
    pred.predicted_options = [PredictedOption(option_name=name, probability=prob) for name, prob in options.items()]
    return pred


def _extract_shadow_record(caplog: pytest.LogCaptureFixture) -> logging.LogRecord:
    """Return the single SHADOW_DIVERGENCE INFO record in caplog."""
    matches = [r for r in caplog.records if "SHADOW_DIVERGENCE" in r.getMessage()]
    assert len(matches) == 1, f"expected exactly one SHADOW_DIVERGENCE record, got {len(matches)}: {caplog.records}"
    return matches[0]


class TestBinaryDivergence:
    def test_valid_block_hand_computed_diff(self, caplog: pytest.LogCaptureFixture):
        """A valid binary block with a known posterior_prob → max_abs_diff matches by hand."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.shadow_divergence")
        # Parser extracted 0.60, block declared 0.75 → |0.75 - 0.60| = 0.15
        parser_pred = 0.60
        reasoning = 'Analysis text.\n\n```json\n{"question_type": "binary", "posterior_prob": 0.75}\n```'
        log_parser_vs_block_divergence(_binary_question(101), parser_pred, reasoning, "test-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert rec.levelno == logging.INFO
        assert "qid=101" in msg
        assert "model=test-model" in msg
        assert "type=binary" in msg
        assert "block_present=True" in msg
        assert "block_valid=True" in msg
        assert "max_abs_diff=0.150000" in msg

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
    def test_declared_percentiles_hand_computed_diff(self, caplog: pytest.LogCaptureFixture):
        """F5 numeric branch: parser.declared_percentiles vs block.declared_percentiles."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.shadow_divergence")

        # Parser's prediction has .declared_percentiles as list[Percentile].
        # Block declares three percentiles at the required keys; parser matches
        # two exactly and diverges at 0.5 by 10.0. Include only the required
        # {0.1, 0.5, 0.9} keys (schema validator).
        parser_pred = MagicMock()
        parser_pred.declared_percentiles = [
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
        log_parser_vs_block_divergence(_numeric_question(201), parser_pred, reasoning, "num-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "qid=201" in msg
        assert "type=numeric" in msg
        assert "block_present=True" in msg
        assert "block_valid=True" in msg
        # Hand-computed: max(|10-10|, |50-60|, |90-90|) = 10.0
        assert "max_abs_diff=10.000000" in msg

    def test_numeric_absent_block(self, caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.INFO, logger="metaculus_bot.shadow_divergence")
        parser_pred = MagicMock()
        parser_pred.declared_percentiles = [
            Percentile(percentile=0.5, value=1.0),
        ]
        log_parser_vs_block_divergence(_numeric_question(202), parser_pred, "no fence", "num-model")

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
        # Parser: {Alpha: 0.62, Beta: 0.28, Gamma: 0.10}.
        # Hand-computed diffs: |0.62-0.50|=0.12, |0.28-0.30|=0.02, |0.10-0.20|=0.10
        # → max = 0.12.
        parser_pred = _mc_parser_prediction({"Alpha": 0.62, "Beta": 0.28, "Gamma": 0.10})
        reasoning = (
            "rationale\n"
            "```json\n"
            '{"question_type": "multiple_choice",'
            ' "option_probs": {"Alpha": 0.50, "Beta": 0.30, "Gamma": 0.20}}\n'
            "```"
        )
        log_parser_vs_block_divergence(_mc_question(301), parser_pred, reasoning, "mc-model")

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
        parser_pred = _mc_parser_prediction({"yes ": 0.55, " NO": 0.45})
        reasoning = (
            'rationale\n```json\n{"question_type": "multiple_choice", "option_probs": {"Yes": 0.6, "No": 0.4}}\n```'
        )
        log_parser_vs_block_divergence(_mc_question(302), parser_pred, reasoning, "mc-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "block_valid=True" in msg
        assert "max_abs_diff=0.050000" in msg

    def test_missing_option_defaults_to_zero(self, caplog: pytest.LogCaptureFixture):
        """An option present in the block but absent from the parser output compares against 0.0.

        Block: {A: 0.5, B: 0.5}; parser declares ONLY A=0.7 (the MagicMock holder
        bypasses PredictedOptionList's sum-to-1 validator, matching how a
        partially-parsed prediction would look). Union of keys = {a, b}:
        |0.7-0.5| = 0.20 on A, |0.0-0.5| = 0.50 on B → max = 0.50, NOT 0.20.
        """
        caplog.set_level(logging.INFO, logger="metaculus_bot.shadow_divergence")

        parser_pred = _mc_parser_prediction({"A": 0.7})
        reasoning = (
            'rationale\n```json\n{"question_type": "multiple_choice", "option_probs": {"A": 0.5, "B": 0.5}}\n```'
        )
        log_parser_vs_block_divergence(_mc_question(303), parser_pred, reasoning, "mc-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "block_valid=True" in msg
        assert "max_abs_diff=0.500000" in msg
