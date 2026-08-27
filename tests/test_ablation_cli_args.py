"""Argparse surface of the ablation CLI: flag defaults, ``--stages`` validation,
logging configuration and the rate-limit-mode kwargs mapping.

Split out of the original monolithic ``test_ablation_cli.py``. These tests build a
parser (or call ``_configure_logging`` / ``_rate_limit_mode_kwargs``) directly and
never run a stage, so they need none of the stage mocks in
``tests/ablation_cli_fakes.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import ablation_cli_fakes as _fakes

# Fixture bound by assignment rather than imported: see the ablation_cli_fakes docstring.
cache_dir = _fakes.cache_dir

# ---------------------------------------------------------------------------
# Argparse tests
# ---------------------------------------------------------------------------


class TestParser:
    def test_parser_default_tournaments_is_spring_aib_2026(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--num-binary", "1"])
        assert args.tournaments == ["spring-aib-2026"]

    def test_parser_default_resolved_after_is_2026_01_01(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--num-binary", "1"])
        assert args.resolved_after == "2026-01-01"

    def test_parser_stages_default_is_all(self):
        from metaculus_bot.ablation.cli import STAGES, _build_parser

        args = _build_parser().parse_args([])
        assert args.stages == STAGES

    def test_parser_qids_parses_csv(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--qids", "1,2,3"])
        assert args.qids == [1, 2, 3]

    def test_parser_force_stages_subset(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--force-stages", "research,score"])
        assert args.force_stages == ["research", "score"]

    def test_parser_qa_research_flag(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--qa-research"])
        assert args.qa_research is True

    def test_parser_default_cache_dir_is_backtests_ablation(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.cache_dir == "backtests/ablation"

    def test_parser_concurrency_default(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.concurrency == 4

    def test_parser_seed_default(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.seed == 0

    def test_parser_per_question_sleep_default(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.per_question_sleep == 30

    def test_per_question_sleep_help_text_documents_inter_stage_behavior(self):
        """Help text must clarify the sleep is per-stage, not literally per-question.

        Despite the flag name, the implementation only sleeps once between stages
        (research -> forecast -> stack). The help string should be honest about that
        so a user setting --per-question-sleep=30 for a 30-question run doesn't
        expect 30s * 30 = 900s total pause.
        """
        from metaculus_bot.ablation.cli import _build_parser

        parser = _build_parser()
        help_text = parser.format_help()
        # The help string for --per-question-sleep must mention BETWEEN STAGES
        # (case-insensitive match permits "between stages", "BETWEEN STAGES", etc.).
        assert "between stages" in help_text.lower(), (
            f"--per-question-sleep help must document per-stage (not per-question) behavior; got: {help_text}"
        )

    def test_parser_gap_fill_max_gaps_default(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.gap_fill_max_gaps == 3

    def test_parser_invalid_stage_rejected(self):
        from metaculus_bot.ablation.cli import _build_parser

        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--stages", "fetch,nonsense"])

    def test_parser_default_gemini_model(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.gemini_model == "gemini-2.5-flash"

    def test_parser_default_gap_fill_off(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.gap_fill is False

    def test_parser_explicit_gap_fill_on(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--gap-fill"])
        assert args.gap_fill is True

    def test_parser_explicit_gap_fill_off(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--no-gap-fill"])
        assert args.gap_fill is False

    def test_parser_gap_fill_mutex(self):
        from metaculus_bot.ablation.cli import _build_parser

        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--gap-fill", "--no-gap-fill"])

    def test_parser_rate_limit_mode_default_is_patient(self):
        """Default flipped from ``gentle`` to ``patient`` on 2026-05-14 (Phase A.3 Package 3a).

        At 50q × 5 forecasters = 250 calls per arm, ``gentle`` (concurrency=2,
        max_retries=3) was thrashing free-tier per-minute throttles. ``patient``
        (concurrency=1, max_retries=8) is the new default for any non-trivial run;
        operators with a smoke (≤4q) workload can opt back into ``gentle`` or ``fast``.
        """
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.rate_limit_mode == "patient"

    def test_parser_rate_limit_mode_accepts_fast(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--rate-limit-mode", "fast"])
        assert args.rate_limit_mode == "fast"

    def test_parser_rate_limit_mode_accepts_slow(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--rate-limit-mode", "slow"])
        assert args.rate_limit_mode == "slow"

    def test_parser_rate_limit_mode_accepts_patient(self):
        """The ``patient`` preset is "slow but persistent" — concurrency=1, max_retries bumped above ``slow``."""
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--rate-limit-mode", "patient"])
        assert args.rate_limit_mode == "patient"

    def test_parser_rate_limit_mode_rejects_invalid(self):
        from metaculus_bot.ablation.cli import _build_parser

        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--rate-limit-mode", "nonsense"])

    def test_log_level_default_is_info(self) -> None:
        """The ablation CLI emits per-stage and per-qid INFO logs that we want
        captured by default. Python's root logger defaults to WARNING which
        silently drops all the rich INFO diagnostics.
        """
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.log_level == "INFO"

    def test_log_level_flag_accepts_debug(self) -> None:
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_log_level_flag_accepts_warning(self) -> None:
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--log-level", "WARNING"])
        assert args.log_level == "WARNING"

    def test_log_level_rejects_invalid(self) -> None:
        from metaculus_bot.ablation.cli import _build_parser

        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--log-level", "TRACE"])


class TestLoggingConfiguration:
    """The ablation CLI must wire up file-archived INFO logging so smoke runs
    are reviewable. The audit at backtests/ablation/audit_smoke_20260515.md
    flagged a 38-line smoke log because no ``logging.basicConfig`` was called
    anywhere in ``metaculus_bot/ablation/`` and the root logger defaulted to
    WARNING.
    """

    def test_configure_logging_creates_logs_subdir_under_cache(self, tmp_path: Path) -> None:
        from metaculus_bot.ablation.cli import _build_parser, _configure_logging

        args = _build_parser().parse_args(["--cache-dir", str(tmp_path)])
        _configure_logging(args, tmp_path)
        logs_dir = tmp_path / "logs"
        assert logs_dir.is_dir(), "expected logs/ subdirectory under cache root"

    def test_configure_logging_returns_log_file_path_in_logs_dir(self, tmp_path: Path) -> None:
        from metaculus_bot.ablation.cli import _build_parser, _configure_logging

        args = _build_parser().parse_args(["--cache-dir", str(tmp_path)])
        log_path = _configure_logging(args, tmp_path)
        assert log_path.parent == tmp_path / "logs"
        assert log_path.name.startswith("run_")
        assert log_path.name.endswith(".log")

    def test_configure_logging_writes_info_messages_to_file(self, tmp_path: Path) -> None:
        """Emit a logger.info AFTER configuration; assert it appears in the file."""
        import logging

        from metaculus_bot.ablation.cli import _build_parser, _configure_logging

        args = _build_parser().parse_args(["--cache-dir", str(tmp_path), "--log-level", "INFO"])
        log_path = _configure_logging(args, tmp_path)

        test_logger = logging.getLogger("test_ablation_logging_demo")
        test_logger.info("hello-from-test-info-message")

        # Force flush across handlers so the file is written.
        for handler in logging.root.handlers:
            handler.flush()

        contents = log_path.read_text(encoding="utf-8")
        assert "hello-from-test-info-message" in contents

    def test_configure_logging_respects_log_level_flag(self, tmp_path: Path) -> None:
        """--log-level WARNING should drop INFO messages from the file."""
        import logging

        from metaculus_bot.ablation.cli import _build_parser, _configure_logging

        args = _build_parser().parse_args(["--cache-dir", str(tmp_path), "--log-level", "WARNING"])
        log_path = _configure_logging(args, tmp_path)

        test_logger = logging.getLogger("test_ablation_logging_demo_warning_filter")
        test_logger.info("info-must-not-appear")
        test_logger.warning("warning-must-appear")

        for handler in logging.root.handlers:
            handler.flush()

        contents = log_path.read_text(encoding="utf-8")
        assert "info-must-not-appear" not in contents
        assert "warning-must-appear" in contents


class TestRateLimitModeMapping:
    """The CLI flag maps to (per_forecaster_concurrency, max_retries) tuples.

    The mapping itself lives in ``cli._RATE_LIMIT_MODE_TO_KWARGS`` so individual
    stages and tests share one source of truth.
    """

    def test_fast_mode_high_concurrency_low_retries(self):
        from metaculus_bot.ablation.cli import _rate_limit_mode_kwargs

        kwargs = _rate_limit_mode_kwargs("fast")
        assert kwargs == {"per_forecaster_concurrency": 4, "max_retries": 1}

    def test_gentle_mode_balanced(self):
        from metaculus_bot.ablation.cli import _rate_limit_mode_kwargs

        kwargs = _rate_limit_mode_kwargs("gentle")
        assert kwargs == {"per_forecaster_concurrency": 2, "max_retries": 3}

    def test_slow_mode_low_concurrency_high_retries(self):
        from metaculus_bot.ablation.cli import _rate_limit_mode_kwargs

        kwargs = _rate_limit_mode_kwargs("slow")
        assert kwargs == {"per_forecaster_concurrency": 1, "max_retries": 5}

    def test_rate_limit_mode_patient_maps_to_concurrency_1_retries_8(self):
        """``patient`` keeps concurrency=1 (matching ``slow``) but bumps the retry budget to 8.

        The motivation: free-tier providers (qwen, minimax, gemma-4-26b) frequently shed
        forecasters under tight retry budgets even though successive attempts often
        succeed. ``patient`` adds retry budget without dropping concurrency further;
        it's "slow but persistent" rather than "even slower."
        """
        from metaculus_bot.ablation.cli import _rate_limit_mode_kwargs

        kwargs = _rate_limit_mode_kwargs("patient")
        assert kwargs == {"per_forecaster_concurrency": 1, "max_retries": 8}
