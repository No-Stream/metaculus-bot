"""Guards on what ``backtest.py --research-dir`` replays and what it says about it.

The id-resolution rungs of ``_load_research_from_archive`` are covered in
``tests/test_research_persistence_e2e.py::TestReadPathE2E``; this file owns the
provenance side. It matters because the two record sources are not interchangeable: a
comment-backfilled record is middle-trimmed text with sections re-headed one level
deeper and ``## Provider Diagnostics`` leaked into the forecaster-visible body, while an
artifact record is the exact research the forecasters saw. Replaying one is not
comparable with replaying the other, and until the run logged its own split, working
that out meant re-deriving it from ``run_id`` prefixes after the fact.
"""

from __future__ import annotations

import json
import logging
import types
from pathlib import Path

import pytest

from backtest import _load_research_from_archive


def _question(qid: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(id_of_question=qid, id_of_post=qid)


def _write(latest_dir: Path, qid: int, **fields: object) -> None:
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / f"{qid}.json").write_text(json.dumps({"qid": qid, "research_text": f"R{qid}", **fields}))


def _summary_line(caplog: pytest.LogCaptureFixture) -> str:
    return next(msg for msg in (r.getMessage() for r in caplog.records) if "cached research records" in msg)


class TestReplayProvenanceIsLogged:
    def test_log_names_the_source_split_of_what_was_loaded(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        latest_dir = tmp_path / "latest"
        _write(latest_dir, 43613, source="artifact")
        _write(latest_dir, 50001, source="comment_backfill")

        with caplog.at_level(logging.INFO, logger="backtest"):
            cache = _load_research_from_archive(str(latest_dir), [_question(43613), _question(50001)])

        assert len(cache) == 2
        summary = _summary_line(caplog)
        assert "'artifact': 1" in summary, summary
        assert "'comment_backfill': 1" in summary, summary

    def test_a_record_without_a_source_field_counts_as_unknown(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # ``source`` is stamped by ``build_archive``; a record from an archive built
        # before that is unattributed and must not be silently credited to either side.
        latest_dir = tmp_path / "latest"
        _write(latest_dir, 43613)

        with caplog.at_level(logging.INFO, logger="backtest"):
            _load_research_from_archive(str(latest_dir), [_question(43613)])

        assert "'unknown': 1" in _summary_line(caplog)

    def test_uncached_questions_are_not_counted_as_a_source(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The split describes what was REPLAYED. A question with no archive record runs
        # live research, so folding it into the counts would overstate the frozen share.
        latest_dir = tmp_path / "latest"
        _write(latest_dir, 43613, source="artifact")

        with caplog.at_level(logging.INFO, logger="backtest"):
            _load_research_from_archive(str(latest_dir), [_question(43613), _question(99999)])

        summary = _summary_line(caplog)
        assert "Loaded 1 cached research records" in summary, summary
        assert "(1 questions uncached)" in summary, summary
        assert "{'artifact': 1}" in summary, summary
