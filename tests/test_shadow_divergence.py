"""F6: tests for metaculus_bot.forecaster._log_parser_vs_block_divergence.

The function is observability-only — it logs a SHADOW_DIVERGENCE line per
forecaster per question comparing the parser's extracted prediction against
the fenced JSON block's declared values. This test file locks in:

  - binary: hand-computed max_abs_diff on a valid block
  - absent block: block_present=False block_valid=False
  - invalid block (F4 distinction): fence exists but validation fails →
    block_present=True block_valid=False
  - numeric (F5): parser's prediction.declared_percentiles vs the block's
    declared_percentiles, with hand-computed max_abs_diff

All at INFO level via caplog.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from forecasting_tools import BinaryQuestion, NumericQuestion
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.forecaster import _log_parser_vs_block_divergence


def _binary_question(qid: int = 42) -> BinaryQuestion:
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = qid
    return q


def _numeric_question(qid: int = 43) -> NumericQuestion:
    q = MagicMock(spec=NumericQuestion)
    q.id_of_question = qid
    return q


def _extract_shadow_record(caplog: pytest.LogCaptureFixture) -> logging.LogRecord:
    """Return the single SHADOW_DIVERGENCE INFO record in caplog."""
    matches = [r for r in caplog.records if "SHADOW_DIVERGENCE" in r.getMessage()]
    assert len(matches) == 1, f"expected exactly one SHADOW_DIVERGENCE record, got {len(matches)}: {caplog.records}"
    return matches[0]


class TestBinaryDivergence:
    def test_valid_block_hand_computed_diff(self, caplog: pytest.LogCaptureFixture):
        """A valid binary block with a known posterior_prob → max_abs_diff matches by hand."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.forecaster")
        # Parser extracted 0.60, block declared 0.75 → |0.75 - 0.60| = 0.15
        parser_pred = 0.60
        reasoning = 'Analysis text.\n\n```json\n{"question_type": "binary", "posterior_prob": 0.75}\n```'
        _log_parser_vs_block_divergence(_binary_question(101), parser_pred, reasoning, "test-model")

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
        caplog.set_level(logging.INFO, logger="metaculus_bot.forecaster")
        _log_parser_vs_block_divergence(_binary_question(102), 0.42, "no fence here", "test-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "qid=102" in msg
        assert "type=binary" in msg
        assert "block_present=False" in msg
        assert "block_valid=False" in msg
        assert "max_abs_diff=N/A" in msg

    def test_invalid_block_present_but_validation_fails(self, caplog: pytest.LogCaptureFixture):
        """F4: fence exists but Pydantic validation fails → block_present=True block_valid=False."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.forecaster")
        # Wrong question_type (says "numeric" while we probe as binary) — the
        # question-type-mismatch guard in parse_structured_block rejects it.
        reasoning = 'text\n```json\n{"question_type": "numeric", "posterior_prob": 0.5}\n```'
        _log_parser_vs_block_divergence(_binary_question(103), 0.42, reasoning, "test-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "block_present=True" in msg
        assert "block_valid=False" in msg
        assert "max_abs_diff=N/A" in msg


class TestNumericDivergence:
    def test_declared_percentiles_hand_computed_diff(self, caplog: pytest.LogCaptureFixture):
        """F5 numeric branch: parser.declared_percentiles vs block.declared_percentiles."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.forecaster")

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
        _log_parser_vs_block_divergence(_numeric_question(201), parser_pred, reasoning, "num-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "qid=201" in msg
        assert "type=numeric" in msg
        assert "block_present=True" in msg
        assert "block_valid=True" in msg
        # Hand-computed: max(|10-10|, |50-60|, |90-90|) = 10.0
        assert "max_abs_diff=10.000000" in msg

    def test_numeric_absent_block(self, caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.INFO, logger="metaculus_bot.forecaster")
        parser_pred = MagicMock()
        parser_pred.declared_percentiles = [
            Percentile(percentile=0.5, value=1.0),
        ]
        _log_parser_vs_block_divergence(_numeric_question(202), parser_pred, "no fence", "num-model")

        rec = _extract_shadow_record(caplog)
        msg = rec.getMessage()
        assert "type=numeric" in msg
        assert "block_present=False" in msg
        assert "block_valid=False" in msg
        assert "max_abs_diff=N/A" in msg
