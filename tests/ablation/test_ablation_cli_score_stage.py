"""The score stage: ``--stages score`` only, the payloads it hands to scoring, arm-count
routing (3-arm / 5-arm / 6-arm), the deterministic median arm, the report shim's numeric
sorting, and the zero-overlap guard.

Split out of the original monolithic ``test_ablation_cli.py``.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from tests.ablation_cli_fakes import (
    _binary_forecaster_payload,
    _binary_stacker_payload,
    _build_question_set,
    _install_full_stack_mocks,
    _make_binary_ground_truth,
    _make_binary_question,
    _make_numeric_question,
    _populate_full_cache_for_qid,
)


class TestStagesScoreOnly:
    @pytest.mark.asyncio
    async def test_stages_score_only_uses_existing_caches(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        # Pre-populate caches manually.
        cache = AblationCache(cache_dir)

        q1 = _make_binary_question(601)
        gt1 = _make_binary_ground_truth(601, outcome=True)

        # Manifest must reference the qid; schema mirrors _build_manifest_entry's writer.
        cache.write_qids_manifest(
            {
                601: {
                    "type": "binary",
                    "tournament": "spring-aib-2026",
                    "question_text": "Will Q601 happen?",
                    "page_url": "https://example.com/q/601",
                    "resolution_criteria": "Resolves YES if it happens.",
                    "fine_print": "",
                    "background_info": "",
                    "ground_truth": {
                        "question_id": 601,
                        "question_type": "binary",
                        "resolution": True,
                        "resolution_string": "YES",
                        "actual_resolution_time": "2026-05-01T00:00:00",
                        "question_text": "Will Q601 happen?",
                        "page_url": "https://example.com/q/601",
                    },
                    "question_metadata": {
                        "open_time": "2026-01-01T00:00:00",
                        "scheduled_resolution_time": "2026-05-01T00:00:00",
                    },
                }
            }
        )

        cache.write_research(601, "blob 601", {"sources": 1})
        cache.write_leakage_screen(
            601,
            {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        )
        for i in range(3):
            slug = model_slug_to_filename(f"openrouter/test/m{i}")
            cache.write_forecaster_output(
                qid=601, model_slug=slug, payload=_binary_forecaster_payload(f"openrouter/test/m{i}", 0.5)
            )
        cache.write_stacker_output(qid=601, arm="stack", payload=_binary_stacker_payload("stack", 0.6))
        cache.write_stacker_output(qid=601, arm="stack_aug", payload=_binary_stacker_payload("stack_aug", 0.7))
        cache.write_stacker_output(qid=601, arm="median", payload=_binary_stacker_payload("median", 0.65))

        # Mocks installed; none should fire.
        question_set = _build_question_set([(q1, gt1)])
        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(["--stages", "score", "--cache-dir", str(cache_dir)])
        exit_code = await run_ablation(args)
        assert exit_code == 0

        # Scoring-only: no fetch, research, screen, forecast, or stacker calls.
        assert mocks["fetch"].await_count == 0
        assert mocks["research"].await_count == 0
        assert mocks["screen"].await_count == 0
        assert mocks["forecasters"].await_count == 0
        assert mocks["stacker"].await_count == 0

        # Summary file written.
        summaries = list((cache_dir / "scores").glob("summary_*.md"))
        assert len(summaries) == 1

    @pytest.mark.asyncio
    async def test_stages_score_only_errors_when_prerequisites_missing(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        # Empty cache: no manifest, no anything.
        question_set = _build_question_set([])
        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(["--stages", "score", "--cache-dir", str(cache_dir)])
        exit_code = await run_ablation(args)
        assert exit_code != 0


# ---------------------------------------------------------------------------
# Stacker payloads → scoring (confounder surface)
# ---------------------------------------------------------------------------


class TestStagePassesPayloadsToScoring:
    @pytest.mark.asyncio
    async def test_stage_score_passes_stacker_payloads_to_scoring(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The cached arm A/B stacker payloads must populate confounder fields in the summary."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(1301)
        q2 = _make_binary_question(1302)
        gt1 = _make_binary_ground_truth(1301, outcome=True)
        gt2 = _make_binary_ground_truth(1302, outcome=False)
        question_set = _build_question_set([(q1, gt1), (q2, gt2)])

        verdicts = {
            1301: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
            1302: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            qid: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            }
            for qid in (1301, 1302)
        }

        # Q1301 — both arms primary, arm B has cross_model_aggregation (tools fired).
        # Q1302 — arm A used fallback, arm B used primary, arm B has empty cross_model_aggregation.
        a1 = _binary_stacker_payload("stack", 0.6)
        b1 = _binary_stacker_payload("stack_aug", 0.7)
        b1["cross_model_aggregation"] = "## Cross-model aggregation\n(numbers)\n"
        a2 = _binary_stacker_payload("stack", 0.55)
        a2["stacker_model_used"] = "fallback"
        a2["n_forecasters_used"] = 4
        b2 = _binary_stacker_payload("stack_aug", 0.65)
        b2["cross_model_aggregation"] = ""
        b2["n_forecasters_used"] = 5

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={1301: ("blob 1301", {}), 1302: ("blob 1302", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={1301: a1, 1302: a2},
            stacker_b_results={1301: b1, 1302: b2},
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(
            ["--num-binary", "2", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "advisory"]
        )
        await run_ablation(args)

        summary_path = next((cache_dir / "scores").glob("summary_*.md"))
        text = summary_path.read_text(encoding="utf-8")

        # Confounder section is present.
        assert "Confounder summary" in text
        # Arm A: 1 primary (q1301), 1 fallback (q1302).
        assert "1/2 primary" in text
        assert "1/2 fallback" in text
        # Treatment activation: pdf arm fired tools on 1/2.
        assert "stack_aug fired tools on 1/2 questions" in text
        # The empty-aggregation count message should mention the empty case.
        assert "empty cross_model_aggregation" in text
        # Per-question diagnostic table includes the marker columns.
        assert "stack_model" in text
        assert "stack_aug_model" in text
        # Renamed from "B_tools" to "stack_aug_tools" in the 3-arm summary refactor.
        assert "stack_aug_tools" in text


# ---------------------------------------------------------------------------
# Bug-2 regression (post-Bucket-1): numeric stacker payloads round-trip through
# ``deserialize_prediction_value`` to a PchipNumericDistribution, whose ``.cdf``
# is already constraint-enforced (PCHIP guarantees strict monotonicity by
# construction). The pre-Bucket-1 version of this test stress-tested the
# defensive sort in ``_build_report_shim`` against out-of-order declared
# percentiles; that defense is now upstream — declared percentiles flow into
# the PCHIP pipeline at forecaster time, and what comes out is a 201-point
# monotonic CDF. Test now asserts the shim produces a monotonic CDF for valid
# input — and that the shim's reliance on ``deserialize_prediction_value`` is
# wired correctly.
# ---------------------------------------------------------------------------


class TestBuildReportShimNumericSorting:
    def test_build_report_shim_returns_monotonic_cdf(self) -> None:
        """The shim's CDF must be strictly monotonic — required by ``np.trapezoid``
        in ``numeric_crps_from_report``.

        The new ``_build_report_shim`` reads ``stacker_prediction`` as a
        post-Bucket-1 full-CDF payload (declared_percentiles + cdf_probabilities
        + bounds + zero_point + cdf_size). It deserializes via
        ``deserialize_prediction_value`` to a PchipNumericDistribution and uses
        its ``.cdf`` directly — which is monotonic by PCHIP construction.
        """
        from metaculus_bot.ablation.cli import _build_report_shim

        q = _make_numeric_question(7001)
        # Build a valid Bucket-1 payload. declared_percentiles must be
        # strictly monotonic in BOTH percentile and value (Pydantic validator
        # on NumericDistribution enforces this); the synthetic 201-point CDF
        # spans the bounds linearly.
        declared_percentiles = [
            {"percentile": 0.1, "value": 30.0},
            {"percentile": 0.5, "value": 50.0},
            {"percentile": 0.9, "value": 70.0},
        ]
        cdf_probabilities = [0.001 + (0.998 * i / 200) for i in range(201)]
        payload = {
            "stacker_prediction": {
                "type": "numeric",
                "declared_percentiles": declared_percentiles,
                "cdf_probabilities": cdf_probabilities,
                "lower_bound": 0.0,
                "upper_bound": 100.0,
                "open_lower_bound": False,
                "open_upper_bound": False,
                "zero_point": None,
                "cdf_size": 201,
            },
        }
        report = _build_report_shim(7001, q, payload)

        cdf_values = [point.value for point in report.prediction.cdf]
        assert len(cdf_values) == 201, f"Expected 201-point CDF, got {len(cdf_values)}"
        # PCHIP guarantees strict monotonicity in the value axis; assert it.
        for prev, current in pairwise(cdf_values):
            assert current > prev, f"CDF values not strictly increasing: {prev} >= {current}"


# ---------------------------------------------------------------------------
# M1: score_only must error when arm_A and arm_B have zero qid overlap.
#
# The current "either dict empty" check passes when {1,2,3} union {4,5,6}, then
# _stage_score takes the intersection (empty) and produces an "n=0 success".
# ---------------------------------------------------------------------------


class TestScoreOnlyZeroOverlapCheck:
    @pytest.mark.asyncio
    async def test_score_only_succeeds_with_zero_comparisons_when_disjoint(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Disjoint single-arm sets produce 0 comparisons but exit 0 (valid empty result).

        With per-comparison N, a qid needs >= 2 arms for any comparison. When
        each qid only has 1 arm, no comparisons are possible — the summary is
        empty but the run succeeds (exit 0). This is correct: the data simply
        doesn't support any pairwise comparison.
        """
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        cache = AblationCache(cache_dir)
        # Pre-populate manifests for all 6 qids; only arm_A for {1,2,3} and arm_B for {4,5,6}.
        for qid in (1, 2, 3, 4, 5, 6):
            _populate_full_cache_for_qid(cache, qid)
        # Wipe stacker outputs and rewrite them as disjoint single-arm sets.
        for qid in (1, 2, 3, 4, 5, 6):
            (cache.root / "stacker_outputs" / str(qid) / "arm_stack.json").unlink(missing_ok=True)
            (cache.root / "stacker_outputs" / str(qid) / "arm_stack_aug.json").unlink(missing_ok=True)
            (cache.root / "stacker_outputs" / str(qid) / "arm_median.json").unlink(missing_ok=True)
        for qid in (1, 2, 3):
            cache.write_stacker_output(qid=qid, arm="stack", payload=_binary_stacker_payload("stack", 0.6))
        for qid in (4, 5, 6):
            cache.write_stacker_output(qid=qid, arm="stack_aug", payload=_binary_stacker_payload("stack_aug", 0.7))

        question_set = _build_question_set([])
        _install_full_stack_mocks(monkeypatch, fetch_question_set=question_set)
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", AsyncMock(return_value=None))

        args = _build_parser().parse_args(["--stages", "score", "--cache-dir", str(cache_dir)])
        exit_code = await run_ablation(args)

        # Per-comparison N: each qid has only 1 arm, so no comparisons are
        # possible. Exit 0 is correct — it's not a config error, just empty data.
        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_score_only_succeeds_when_arms_overlap(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sanity check: full overlap still produces a summary (didn't break the happy path)."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        cache = AblationCache(cache_dir)
        for qid in (10, 11):
            _populate_full_cache_for_qid(cache, qid)

        question_set = _build_question_set([])
        _install_full_stack_mocks(monkeypatch, fetch_question_set=question_set)
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", AsyncMock(return_value=None))

        args = _build_parser().parse_args(["--stages", "score", "--cache-dir", str(cache_dir)])
        exit_code = await run_ablation(args)

        assert exit_code == 0
        summaries = list((cache_dir / "scores").glob("summary_*.md"))
        assert len(summaries) == 1


# ---------------------------------------------------------------------------
# ARM_MEDIAN end-to-end via _stage_stack
#
# ARM_MEDIAN bypasses the stacker LLM entirely and runs deterministic median
# aggregation per question. These tests verify that:
#
# * The "median" stage is wired into STAGES.
# * ``_stage_stack(arm=ARM_MEDIAN, ...)`` writes a structurally-correct ``arm_median.json``
#   cache file when run on a synthetic working set.
# * ``WorkingSet.stacker_payloads["median"]`` is populated end-to-end.
# ---------------------------------------------------------------------------


class TestArmMedianStageStack:
    def test_stages_includes_stack_c(self) -> None:
        from metaculus_bot.ablation.cli import STAGES

        assert "median" in STAGES
        # median sits between pdf and score in the canonical pipeline order
        # so the orchestrator runs it after both LLM stackers but before scoring.
        stack_b_idx = STAGES.index("stack_aug")
        stack_c_idx = STAGES.index("median")
        score_idx = STAGES.index("score")
        assert stack_b_idx < stack_c_idx < score_idx

    @pytest.mark.asyncio
    async def test_stage_stack_arm_median_writes_arm_median_json_to_cache(
        self,
        cache_dir: Path,
    ) -> None:
        """Synthetic working set + ``_stage_stack(arm=ARM_MEDIAN)`` should emit ``arm_median.json``."""
        from metaculus_bot.ablation.cli import (
            SpendReport,
            WorkingSet,
            _build_parser,
            _stage_stack,
        )
        from metaculus_bot.ablation.run_stacker import ARM_MEDIAN

        cache = AblationCache(cache_dir)
        qid = 99001
        question = _make_binary_question(qid)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(f"openrouter/test/m{i}", 0.5)
            for i in range(3)
        }

        working = WorkingSet()
        working.questions[qid] = question
        working.forecaster_payloads[qid] = forecaster_payloads
        working.research_blobs[qid] = "research blob"

        spend = SpendReport()
        args = _build_parser().parse_args(["--num-binary", "1", "--cache-dir", str(cache_dir)])

        await _stage_stack(args, cache, working, arm=ARM_MEDIAN, force=False, spend=spend)

        # Cache file written.
        cached = cache.read_stacker_output(qid=qid, arm=ARM_MEDIAN)
        assert cached is not None
        assert cached["arm"] == "median"
        assert cached["success"] is True
        assert cached["stacker_model_used"] == "simple_aggregation"
        assert cached["tools_enabled_at_runtime"] is False
        # WorkingSet populated. The cache adds a ``cache_schema_version`` field on
        # read; the in-memory working set carries the raw payload _stage_stack
        # received before the cache round-trip. Compare on the structural fields
        # both should agree on rather than the full dict.
        assert qid in working.stacker_payloads.get("median", {})
        for key in ("arm", "success", "stacker_model_used", "stacker_prediction"):
            assert working.stacker_payloads["median"][qid][key] == cached[key]
        # ARM_MEDIAN does NOT consume LLM-call counters; only cache-hit counter (zero on first run).
        assert spend.stacker_llm_calls_stack == 0
        assert spend.stacker_llm_calls_stack_aug == 0
        assert spend.cached_stacker_hits.get("median", 0) == 0  # first run, no cache hit

    @pytest.mark.asyncio
    async def test_stage_stack_arm_c_uses_cache_on_second_call(
        self,
        cache_dir: Path,
    ) -> None:
        """A second invocation should hit the cache and bump ``cached_stacker_median_hits``."""
        from metaculus_bot.ablation.cli import (
            SpendReport,
            WorkingSet,
            _build_parser,
            _stage_stack,
        )
        from metaculus_bot.ablation.run_stacker import ARM_MEDIAN

        cache = AblationCache(cache_dir)
        qid = 99002
        question = _make_binary_question(qid)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(f"openrouter/test/m{i}", 0.5)
            for i in range(3)
        }

        working = WorkingSet()
        working.questions[qid] = question
        working.forecaster_payloads[qid] = forecaster_payloads
        working.research_blobs[qid] = "research blob"

        args = _build_parser().parse_args(["--num-binary", "1", "--cache-dir", str(cache_dir)])

        # First call: writes cache.
        spend1 = SpendReport()
        await _stage_stack(args, cache, working, arm=ARM_MEDIAN, force=False, spend=spend1)
        assert spend1.cached_stacker_hits.get("median", 0) == 0

        # Second call: hits cache.
        working2 = WorkingSet()
        working2.questions[qid] = question
        working2.forecaster_payloads[qid] = forecaster_payloads
        working2.research_blobs[qid] = "research blob"
        spend2 = SpendReport()
        await _stage_stack(args, cache, working2, arm=ARM_MEDIAN, force=False, spend=spend2)
        assert spend2.cached_stacker_hits.get("median", 0) == 1
        assert qid in working2.stacker_payloads.get("median", {})


# ---------------------------------------------------------------------------
# _stage_score arm-count routing: 3-arm / 5-arm / 6-arm selection
# ---------------------------------------------------------------------------


def _build_six_arm_working_set(qids: list[int]) -> Any:
    """A WorkingSet carrying all six arm payloads for each binary qid."""
    from metaculus_bot.ablation.cli import WorkingSet
    from metaculus_bot.ablation.run_stacker import (
        ARM_MEAN,
        ARM_MEDIAN,
        ARM_PDF_MIN1,
        ARM_PDF_MIN2,
        ARM_STACK,
        ARM_STACK_AUG,
    )

    working = WorkingSet()
    arm_values = {
        ARM_STACK: 0.55,
        ARM_STACK_AUG: 0.6,
        ARM_PDF_MIN1: 0.62,
        ARM_PDF_MIN2: 0.64,
        ARM_MEDIAN: 0.58,
        ARM_MEAN: 0.59,
    }
    for arm in arm_values:
        working.stacker_payloads[arm] = {}
    for qid in qids:
        working.questions[qid] = _make_binary_question(qid)
        working.ground_truths[qid] = _make_binary_ground_truth(qid, outcome=True)
        for arm, value in arm_values.items():
            working.stacker_payloads[arm][qid] = _binary_stacker_payload(arm, value)
    return working


class TestStageScoreArmCountRouting:
    def test_stage_score_routes_to_six_arm_when_pdf_and_mean_present(self, cache_dir: Path) -> None:
        """When both pdf AND mean payloads are present, _stage_score scores the mean arm.

        The rendered summary should carry the five mean comparisons and a `mean`
        per-arm raw-stats row.
        """
        from metaculus_bot.ablation.cli import _build_parser, _stage_score

        cache = AblationCache(cache_dir)
        working = _build_six_arm_working_set([7001, 7002, 7003])
        args = _build_parser().parse_args(["--num-binary", "3", "--cache-dir", str(cache_dir)])

        summary_path = _stage_score(args, cache, working)
        text = summary_path.read_text(encoding="utf-8")

        for mean_comparison in (
            "mean-median",
            "mean-stack",
            "mean-stack_aug",
            "mean-pdf_min1",
            "mean-pdf_min2",
        ):
            assert mean_comparison in text
        # Per-arm raw-stats table includes a mean row.
        assert "| mean |" in text

    def test_stage_score_stays_five_arm_when_mean_absent(self, cache_dir: Path) -> None:
        """Old free-tier data (pdf arms, NO mean) must keep using the 5-arm path.

        This is the load-bearing backward-compat property: a re-run on the
        88-question free-tier cache (0 arm_mean.json) must still score in 5-arm
        mode without emitting mean comparisons.
        """
        from metaculus_bot.ablation.cli import _build_parser, _stage_score
        from metaculus_bot.ablation.run_stacker import ARM_MEAN

        cache = AblationCache(cache_dir)
        working = _build_six_arm_working_set([7101, 7102, 7103])
        # Drop the mean payloads entirely — emulate the old free-tier cache dir.
        del working.stacker_payloads[ARM_MEAN]
        args = _build_parser().parse_args(["--num-binary", "3", "--cache-dir", str(cache_dir)])

        summary_path = _stage_score(args, cache, working)
        text = summary_path.read_text(encoding="utf-8")

        # No mean comparisons, no mean per-arm row.
        assert "mean-median" not in text
        assert "| mean |" not in text
        # But the 5-arm pdf comparisons must still be present.
        assert "median-pdf_min1" in text
        assert "pdf_min1-stack" in text

    def test_stage_score_stays_three_arm_when_no_pdf_or_mean(self, cache_dir: Path) -> None:
        """Stack/stack_aug/median only (no pdf, no mean) must keep using the 3-arm path."""
        from metaculus_bot.ablation.cli import _build_parser, _stage_score
        from metaculus_bot.ablation.run_stacker import ARM_MEAN, ARM_PDF_MIN1, ARM_PDF_MIN2

        cache = AblationCache(cache_dir)
        working = _build_six_arm_working_set([7201, 7202, 7203])
        for arm in (ARM_PDF_MIN1, ARM_PDF_MIN2, ARM_MEAN):
            del working.stacker_payloads[arm]
        args = _build_parser().parse_args(["--num-binary", "3", "--cache-dir", str(cache_dir)])

        summary_path = _stage_score(args, cache, working)
        text = summary_path.read_text(encoding="utf-8")

        assert "mean-median" not in text
        assert "median-pdf_min1" not in text
        # Core 3-arm comparison present.
        assert "median-stack" in text
