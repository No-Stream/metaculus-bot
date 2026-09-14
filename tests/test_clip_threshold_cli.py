"""Tests for the sweep's command line: it writes JSON, prints markdown and names its dataset."""

from __future__ import annotations

import json

import pytest

from metaculus_bot.constants import MC_PROB_MIN
from metaculus_bot.performance_analysis.clip_threshold import main
from metaculus_bot.performance_analysis.clip_threshold_sweep import BINARY_FLOOR_GRID, MC_FLOOR_GRID
from metaculus_bot.performance_analysis.clip_threshold_tables import compute_report
from metaculus_bot.performance_analysis.clip_threshold_windows import window_labels
from tests.clip_threshold_fakes import AS_OF, cli_dataset


class TestCli:
    """The CLI writes JSON, prints markdown and honours ``--exclude-qids``."""

    def _write(self, tmp_path, records: list[dict]) -> str:
        path = tmp_path / "data.json"
        path.write_text(json.dumps(records))
        return str(path)

    def test_a_dataset_has_to_be_named(self, capsys):
        """``--cached`` used to default to ``scratch/residual_2026-09-01/perf_all_tagged.json``.

        A fresh clone has no ``scratch/`` at all, and on the operator's machine that round
        directory still resolves after it is superseded, so a bare invocation silently swept a
        stale dataset and reported it as the current round.
        """
        with pytest.raises(SystemExit) as exit_info:
            main([])

        assert exit_info.value.code == 2
        assert "--cached" in capsys.readouterr().err

    def test_writes_json_and_markdown(self, tmp_path, capsys):
        path = self._write(tmp_path, cli_dataset())
        out_json = str(tmp_path / "sweep.json")
        main(["--cached", path, "--as-of", "2026-09-02T00:00:00Z", "--output-json", out_json])
        markdown = capsys.readouterr().out
        payload = json.loads((tmp_path / "sweep.json").read_text())
        assert {"binary", "multiple_choice"} <= set(payload)
        for label in window_labels("binary") + window_labels("multiple_choice"):
            assert label in markdown
        assert "2026-09-02" in markdown
        assert str(path) in markdown
        # Every grid level is rendered for both types.
        assert all(f"{c:.4f}" in markdown for c in BINARY_FLOOR_GRID)
        assert all(f"{c:.4f}" in markdown for c in MC_FLOOR_GRID)

    def test_exclude_qids_drops_the_cohort(self, tmp_path, capsys):
        path = self._write(tmp_path, cli_dataset())
        main(["--cached", path, "--as-of", "2026-09-02T00:00:00Z"])
        baseline = capsys.readouterr().out
        main(["--cached", path, "--as-of", "2026-09-02T00:00:00Z", "--exclude-qids", "degraded_run"])
        excluded = capsys.readouterr().out
        assert "n=41" in baseline
        assert "n=40" in excluded

    def test_report_object_carries_the_same_numbers(self, tmp_path):
        records = cli_dataset()
        report = compute_report(records, dataset_path="mem", as_of=AS_OF, exclude_qids=frozenset())
        payload = report.to_dict()
        assert payload["binary"]["n"] == 41
        assert payload["multiple_choice"]["n"] == 12
        assert payload["meta"]["as_of"].startswith("2026-09-02")
        binary_rows = payload["binary"]["sweep"]
        assert {row["c"] for row in binary_rows} == set(BINARY_FLOOR_GRID)

    def test_mc_sub_shippable_rows_are_labelled(self, tmp_path):
        report = compute_report(cli_dataset(), dataset_path="mem", as_of=AS_OF, exclude_qids=frozenset())
        mc = next(t for t in report.types if t.question_type == "multiple_choice")
        below = [row for row in mc.sweep if row.c < MC_PROB_MIN]
        assert below
        assert all(row.shippable is False for row in below)
        assert all(row.shippable is True for row in mc.sweep if row.c >= MC_PROB_MIN)
