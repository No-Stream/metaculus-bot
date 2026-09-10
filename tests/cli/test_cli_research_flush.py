"""The research batch reaches disk on both of cli's exit paths.

Driven through the REAL ``ResearchPersistenceWriter`` that cli hands the forecaster, so the
whole write path is covered rather than a mock call.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from metaculus_bot.cli import main as cli_main
from metaculus_bot.constants import PERSIST_RESEARCH_ENABLED_ENV
from tests.cli_test_helpers import _cli_main_test_mode, asyncio_run_stub


class TestCliResearchFlush:
    """The research batch reaches disk on BOTH exit paths.

    Records accumulate in memory for the whole run and are written once at the end, so
    before the flush moved inside the ``finally`` any exception escaping ``asyncio.run``
    — an OSError, the invalid-run-mode ValueError, a ``timeout-minutes`` SIGTERM —
    discarded every question's research. A 40-question tournament run that died on the
    last question archived nothing, and since GHA deletes artifacts at 90 days that hole
    is permanent. The same ``finally`` already protected the credit telemetry, which is
    what made the omission easy to miss.

    Both tests drive the REAL ``ResearchPersistenceWriter`` through the sink cli hands the
    forecaster, so they cover the whole write path (sink -> accumulate -> flush -> JSONL)
    rather than asserting that a mock got called.
    """

    @staticmethod
    def _forecaster_class() -> MagicMock:
        """A ``TemplateForecaster`` class stub that also exposes the constructor kwargs.

        The helper's own patch discards its mock, and these tests need the
        ``research_sink`` cli built and passed in. ``alertable_count`` is pinned to a real
        int because the normal-path test runs off the end of ``main``, into the
        ``alertable > 0`` comparison.
        """
        forecaster_class = MagicMock()
        forecaster_class.return_value.alertable_count = 0
        return forecaster_class

    @staticmethod
    def _record_two(sink: Callable[..., None]) -> None:
        """Record two questions' research through cli's own sink callback."""
        for qid in (43613, 50001):
            sink(
                qid=qid,
                page_url=f"https://www.metaculus.com/questions/{qid}/",
                question_text=f"Question {qid}?",
                research_text=f"## News Articles (AskNews)\nResearch for {qid}.",
                providers_used=["asknews"],
                gap_fill_used=False,
            )

    @staticmethod
    def _flushed_records(tmp_path: Path) -> list[dict]:
        """Every record in the JSONL the writer flushed into ``research_outputs/``."""
        written = sorted((tmp_path / "research_outputs").glob("research_*.jsonl"))
        assert len(written) == 1, f"expected exactly one flushed JSONL, got {written}"
        return [json.loads(line) for line in written[0].read_text().strip().splitlines()]

    def test_flush_runs_when_the_forecast_loop_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(PERSIST_RESEARCH_ENABLED_ENV, "true")
        monkeypatch.chdir(tmp_path)  # writer.flush() writes research_outputs/ under CWD

        forecaster_class = self._forecaster_class()

        def _record_then_crash(*_args: object, **_kwargs: object) -> None:
            """Two questions researched, then the run dies before returning: the shape that lost the batch."""
            self._record_two(forecaster_class.call_args.kwargs["research_sink"])
            raise RuntimeError("forecast loop blew up")

        with (
            _cli_main_test_mode(alertable_count=0),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_record_then_crash)),
            pytest.raises(RuntimeError, match="forecast loop blew up"),
        ):
            # The original exception must still propagate: the flush is a rescue, not a swallow.
            cli_main()

        assert [r["qid"] for r in self._flushed_records(tmp_path)] == [43613, 50001]

    def test_flush_still_runs_on_the_normal_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(PERSIST_RESEARCH_ENABLED_ENV, "true")
        monkeypatch.chdir(tmp_path)

        forecaster_class = self._forecaster_class()

        def _record_then_return(*_args: object, **_kwargs: object) -> list[object]:
            self._record_two(forecaster_class.call_args.kwargs["research_sink"])
            return []

        with (
            _cli_main_test_mode(alertable_count=0),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_record_then_return)),
        ):
            cli_main()

        assert [r["qid"] for r in self._flushed_records(tmp_path)] == [43613, 50001]

    def test_nothing_is_written_when_the_flag_is_off(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No writer, no sink: the forecaster is handed None, so the finally-block flush stays guarded."""
        monkeypatch.delenv(PERSIST_RESEARCH_ENABLED_ENV, raising=False)
        monkeypatch.chdir(tmp_path)

        def _crash(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("boom")

        forecaster_class = self._forecaster_class()
        with (
            _cli_main_test_mode(alertable_count=0),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_crash)),
            pytest.raises(RuntimeError, match="boom"),
        ):
            cli_main()

        assert forecaster_class.call_args.kwargs["research_sink"] is None
        assert not (tmp_path / "research_outputs").exists()
