"""Tests for the rendered report: every section reaches the markdown and every table the JSON."""

from __future__ import annotations

from metaculus_bot.performance_analysis.clip_threshold_report import render_report
from metaculus_bot.performance_analysis.clip_threshold_sweep import sweep_row
from metaculus_bot.performance_analysis.clip_threshold_tables import compute_report
from metaculus_bot.performance_analysis.clip_threshold_windows import WINDOW_ALL
from tests.clip_threshold_fakes import AS_OF, binary_record, cli_dataset, one_binary


class TestIdentityRows:
    """A candidate that moves nothing is the do-nothing map, and the report must say so."""

    def test_identity_flag_and_zero_interval(self):
        records = [one_binary(question_id=9001, p_yes=0.30, resolution=True)]
        row = sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.05)
        assert row.identity is True
        assert row.n_affected == 0
        assert (row.ci_lo, row.ci_hi) == (0.0, 0.0)
        moved = sweep_row(
            [one_binary(question_id=9002, p_yes=0.02, resolution=False)],
            question_type="binary",
            side="floor_only",
            window=WINDOW_ALL,
            c=0.05,
        )
        assert moved.identity is False

    def test_rendered_ci_cell_reads_identity_not_an_interval(self):
        report = compute_report(
            [binary_record(question_id=9003, p_yes=0.30, resolution=True)],
            dataset_path="synthetic",
            as_of=AS_OF,
            exclude_qids=frozenset(),
        )
        markdown = render_report(report)
        assert "| identity |" in markdown
        assert "[+0.000, +0.000]" not in markdown


class TestRenderedSections:
    """The report carries every new section and the JSON carries every new table."""

    def test_markdown_and_json_carry_the_new_tables(self, tmp_path):
        """Every selection-derived quantity the markdown prints is in the JSON too: the plateau and
        the censored ties per (side, window), and the sign counts behind the cross-check sentence.
        """
        records = cli_dataset()
        report = compute_report(records, dataset_path="mem", as_of=AS_OF, exclude_qids=frozenset())
        markdown = render_report(report)
        for heading in (
            "Out-of-bag value of the fitted argmax",
            "Affected-set nesting",
            "Insurance view",
            "Single-survivor publishes and the thin publish floor",
            "live clamp regime",
            "at_c (members)",
            "fit n_aff",
        ):
            assert heading in markdown
        payload = report.to_dict()
        for key in (
            "insurance",
            "nesting",
            "oob",
            "regime_span",
            "thin_floor",
            "argmax",
            "older_regime",
            "replay_cohorts",
            "cross_check_summary",
        ):
            assert key in payload["binary"]
        binary = report.type_report("binary")
        assert {(sel["side"], sel["window"]) for sel in payload["binary"]["argmax"]} == {
            (side, w.label) for side in ("floor_only", "ceiling_only", "symmetric") for w in binary.populated_windows
        }
        assert set(payload["binary"]["cross_check_summary"]) == {"n_differing", "n_replay_more_negative"}
        assert "n_replayable" not in payload["binary"]["cross_check"][0]
        assert {c["window"] for c in payload["binary"]["replay_cohorts"]} == {w.label for w in binary.populated_windows}
        assert payload["multiple_choice"]["thin_floor"] is None
        assert payload["meta"]["bootstrap"]["oob_B"] > 0
        assert {row["window"] for row in payload["binary"]["oob"]} == {
            w.label for w in report.type_report("binary").populated_windows
        }
