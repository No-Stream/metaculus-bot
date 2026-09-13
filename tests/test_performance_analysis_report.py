"""Tests for ``generate_report``: the markdown skeleton the residual CLI prints."""

from __future__ import annotations

import numpy as np

from metaculus_bot import performance_analysis
from tests.performance_analysis_fakes import _binary_record, _numeric_record


class TestGenerateReport:
    """Pins the markdown skeleton of ``generate_report`` — the CLI's whole output.

    Each section is gated on its own count, so an empty cut must leave no heading
    behind, and the section ORDER (binary, per-model, numeric, MC) is what a reader
    diffs across residual rounds.
    """

    @staticmethod
    def _mc_record(post_id: int, log_score: float) -> dict:
        return {
            "post_id": post_id,
            "type": "multiple_choice",
            "mc_log_score": log_score,
            "resolution_parsed": "B",
            "options": ["A", "B"],
            "our_forecast_values": [0.3, 0.7],
            "brier_score": None,
            "log_score": None,
            "numeric_log_score": None,
            "per_model_forecasts": {},
            "metadata": {"category": None},
        }

    def test_empty_dataset_renders_only_the_header(self):
        report = performance_analysis.generate_report([])

        assert report.splitlines()[0] == "# Performance Analysis Report"
        assert "**Total questions:** 0" in report
        assert "## Binary Questions" not in report
        assert "## Per-Model Binary Scores" not in report
        assert "## Numeric Questions" not in report
        assert "## Multiple Choice Questions" not in report

    def test_type_counts_are_listed_alphabetically(self):
        data = [
            _binary_record(1, 0.7, True),
            self._mc_record(2, -0.5),
            _numeric_record(3, list(np.linspace(0.0, 1.0, 201)), 50.0),
        ]
        report = performance_analysis.generate_report(data)

        assert "**Total questions:** 3" in report
        assert "- binary: 1" in report
        assert "- multiple_choice: 1" in report
        assert "- numeric: 1" in report
        counts_block = report.split("**Total questions:** 3\n")[1].splitlines()[:3]
        assert counts_block == ["- binary: 1", "- multiple_choice: 1", "- numeric: 1"]

    def test_sections_appear_in_a_fixed_order(self):
        data = [
            _binary_record(1, 0.7, True, per_model={"model-a": "70.0%"}),
            _binary_record(2, 0.2, False, per_model={"model-a": "20.0%"}),
            self._mc_record(3, -0.5),
            _numeric_record(4, list(np.linspace(0.0, 1.0, 201)), 50.0),
        ]
        report = performance_analysis.generate_report(data)

        headings = [line for line in report.splitlines() if line.startswith("## ")]
        assert headings == [
            "## Binary Questions",
            "## Per-Model Binary Scores",
            "## Numeric Questions",
            "## Multiple Choice Questions",
        ]

    def test_binary_section_carries_summary_lines_and_a_calibration_table(self):
        data = [_binary_record(1, 0.7, True), _binary_record(2, 0.2, False)]
        report = performance_analysis.generate_report(data)

        assert "- Count: 2" in report
        assert "- Mean Brier: 0.0650" in report
        assert "- Direction Accuracy: 100.0%" in report
        assert "- Base Rate: 50.0%" in report
        assert "| Bucket | Predicted | Actual | Count |" in report

    def test_per_model_section_lists_each_recovered_member(self):
        data = [
            _binary_record(1, 0.7, True, per_model={"model-a": "70.0%", "model-b": "60.0%"}),
            _binary_record(2, 0.2, False, per_model={"model-a": "20.0%", "model-b": "30.0%"}),
        ]
        report = performance_analysis.generate_report(data)

        assert "### Calibration" in report
        assert "| model-a |" in report
        assert "| model-b |" in report

    def test_numeric_section_renders_ten_pit_histogram_bins(self):
        data = [_numeric_record(1, list(np.linspace(0.0, 1.0, 201)), 50.0)]
        report = performance_analysis.generate_report(data)

        assert "### PIT Histogram" in report
        assert "| 0.0-0.1 | 0 |" in report
        assert "| 0.5-0.6 | 1 |" in report
        assert sum(1 for line in report.splitlines() if line.startswith("| 0.")) == 10

    def test_mc_section_reports_accuracy_and_mean_scores(self):
        report = performance_analysis.generate_report([self._mc_record(1, -0.5)])

        assert "## Multiple Choice Questions" in report
        assert "- Accuracy (top pick correct): 100.0%" in report
        assert "- Mean Prob on Correct: 0.70" in report
        assert "- Mean MC Log Score: -0.50" in report
