"""The two labels cli stamps on every research-archive record: ``tournament_id`` and ``platform``.

Both follow the run mode, and nothing else on a record says which competition or which platform
a question came from, so a wrong label is silent data corruption rather than a cosmetic slip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar, get_args
from unittest.mock import patch

import pytest
from forecasting_tools import MetaculusApi

from metaculus_bot.cli import RunMode, persisted_platform, persisted_tournament_id
from metaculus_bot.cli import main as cli_main
from metaculus_bot.constants import (
    MANTIC_TOURNAMENT_ID,
    METACULUS_CUP_ID,
    PERSIST_RESEARCH_ENABLED_ENV,
    TOURNAMENT_ID,
)
from tests.cli_test_helpers import _cli_main_test_mode, _forecaster_class, _mantic_env, asyncio_run_stub


class TestPersistedTournamentId:
    """The research archive's ``tournament_id`` label follows the RUN MODE.

    Until 2026-09-03 cli stamped ``TOURNAMENT_ID`` on every record whatever mode was
    running, so enabling the Metaculus Cup workflow would have filed cup questions under
    the bot tournament's slug — inside its config-era buckets and inside the supply probe's
    per-slug rows. That is silent data corruption, not a cosmetic label: nothing else on the
    record says which competition a question came from, since ``run_mode`` names the
    pipeline rather than the object.
    """

    EXPECTED_LABEL: ClassVar[dict[str, str]] = {
        "tournament": TOURNAMENT_ID,
        "minibench": str(MetaculusApi.CURRENT_MINIBENCH_ID),
        "quarterly_cup": METACULUS_CUP_ID,
        "metaculus_cup": METACULUS_CUP_ID,
        "mantic": MANTIC_TOURNAMENT_ID,
        # No label fits the evergreen set; retained so the archive's existing test-run records stay comparable.
        "test_questions": TOURNAMENT_ID,
    }

    def test_every_run_mode_has_a_decided_label(self) -> None:
        """Derived from RunMode, so a mode added without a decision fails here instead of inheriting a slug."""
        assert set(get_args(RunMode)) == set(self.EXPECTED_LABEL)

    @pytest.mark.parametrize(("run_mode", "expected"), sorted(EXPECTED_LABEL.items()))
    def test_label_per_run_mode(self, run_mode: RunMode, expected: str) -> None:
        assert persisted_tournament_id(run_mode) == expected

    def test_the_competitions_do_not_share_a_label(self) -> None:
        """The bot tournament, the cup and the Mantic tournament must stay distinguishable in the archive."""
        labels = {persisted_tournament_id(run_mode) for run_mode in ("tournament", "metaculus_cup", "mantic")}
        assert len(labels) == 3, labels

    def test_an_unknown_mode_raises_rather_than_mislabelling(self) -> None:
        with pytest.raises(ValueError, match="Invalid run mode"):
            persisted_tournament_id("world_cup")  # type: ignore[arg-type]  # deliberately outside RunMode

    def test_a_cup_run_archives_its_records_under_the_cup_slug(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end through cli's own writer, not just the helper: mode -> label -> JSONL."""
        monkeypatch.setenv(PERSIST_RESEARCH_ENABLED_ENV, "true")
        monkeypatch.chdir(tmp_path)  # writer.flush() writes research_outputs/ under CWD

        forecaster_class = _forecaster_class()

        def _record_then_return(*_args: object, **_kwargs: object) -> list[object]:
            forecaster_class.call_args.kwargs["research_sink"](
                qid=45500,
                page_url="https://www.metaculus.com/questions/45500/",
                question_text="A fall cup question?",
                research_text="## News Articles (AskNews)\nResearch for 45500.",
                providers_used=["asknews"],
                gap_fill_used=False,
            )
            return []

        with (
            _cli_main_test_mode(alertable_count=0, mode="metaculus_cup", forecaster_class=forecaster_class),
            patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_record_then_return)),
        ):
            cli_main()

        written = sorted((tmp_path / "research_outputs").glob("research_*.jsonl"))
        assert len(written) == 1, f"expected exactly one flushed JSONL, got {written}"
        records = [json.loads(line) for line in written[0].read_text().strip().splitlines()]
        assert [(r["run_mode"], r["tournament_id"], r["platform"]) for r in records] == [
            ("metaculus_cup", METACULUS_CUP_ID, "metaculus")
        ]

    def test_a_mantic_run_archives_its_records_under_the_mantic_slug_and_platform(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same end-to-end shape for the Mantic mode: the Mantic slug, and ``platform`` set to
        ``mantic`` so a post id in the 600s is never read as a Metaculus id."""
        _mantic_env(monkeypatch)
        monkeypatch.setenv(PERSIST_RESEARCH_ENABLED_ENV, "true")
        monkeypatch.chdir(tmp_path)

        forecaster_class = _forecaster_class()

        def _record_then_return(*_args: object, **_kwargs: object) -> list[object]:
            forecaster_class.call_args.kwargs["research_sink"](
                qid=650,
                page_url="https://competitions.mantic.com/questions/650/",
                question_text="What will the price of bitcoin be?",
                research_text="## News Articles (AskNews)\nResearch for 650.",
                providers_used=["asknews"],
                gap_fill_used=False,
            )
            return []

        with (
            _cli_main_test_mode(alertable_count=0, mode="mantic", forecaster_class=forecaster_class),
            patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_record_then_return)),
        ):
            cli_main()

        written = sorted((tmp_path / "research_outputs").glob("research_*.jsonl"))
        assert len(written) == 1, f"expected exactly one flushed JSONL, got {written}"
        records = [json.loads(line) for line in written[0].read_text().strip().splitlines()]
        assert [(r["run_mode"], r["tournament_id"], r["platform"]) for r in records] == [
            ("mantic", MANTIC_TOURNAMENT_ID, "mantic")
        ]


class TestPersistedPlatform:
    """The archive's additive ``platform`` field: which question platform a record's ids belong to.

    Filenames are not namespaced and the archive groups on the bare qid, so this field is what
    separates the two platforms' records and what an analysis keyed on bare post ids must
    filter on; the id gap today runs from the Mantic counter (~650) to 14333, the next
    Metaculus key.
    """

    def test_mantic_mode_is_the_only_mantic_platform(self) -> None:
        assert persisted_platform("mantic") == "mantic"
        for run_mode in set(get_args(RunMode)) - {"mantic"}:
            assert persisted_platform(run_mode) == "metaculus", run_mode
