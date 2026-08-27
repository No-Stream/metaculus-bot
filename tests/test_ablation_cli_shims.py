"""Manifest serde round-trips: the question shim rehydrated from a manifest entry (time
fields, content fields, numeric ``cdf_size``), ground-truth out-of-bounds values, and
tz-aware resolution times through the question filter.

Split out of the original monolithic ``test_ablation_cli.py``. These assert that what the
manifest writer persisted is what the reader reconstructs.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from forecasting_tools import BinaryQuestion, NumericQuestion

from metaculus_bot.backtest.scoring import GroundTruth
from tests.ablation_cli_fakes import (
    _OPEN,
    _RESOLVE,
    _make_binary_ground_truth,
    _make_binary_question,
    _make_numeric_ground_truth,
    _make_numeric_question,
)

# ---------------------------------------------------------------------------
# Bug-3 regression: OutOfBoundsResolution must round-trip through ground-truth
# serialize/deserialize. Without an explicit tag, the JSON writer's
# ``default=str`` produces "OutOfBoundsResolution.ABOVE_UPPER_BOUND" and the
# reload path's ``float(raw_resolution)`` raises ValueError.
# ---------------------------------------------------------------------------


class TestGroundTruthOutOfBoundsRoundTrip:
    def test_ground_truth_round_trip_with_out_of_bounds_resolution(self) -> None:
        """A numeric ``GroundTruth`` with ``OutOfBoundsResolution`` must reload exactly.

        The score-only path hydrates the working set entirely from cache. If the
        serialize/deserialize pair turns ``OutOfBoundsResolution.ABOVE_UPPER_BOUND``
        into a string then tries ``float(...)`` on it, the reload crashes with
        ``ValueError: could not convert string to float``. This test pins the
        round-trip contract.
        """
        from forecasting_tools.data_models.questions import OutOfBoundsResolution

        from metaculus_bot.ablation.cli import _deserialize_ground_truth, _serialize_ground_truth

        original = GroundTruth(
            question_id=8001,
            question_type="numeric",
            resolution=OutOfBoundsResolution.ABOVE_UPPER_BOUND,
            resolution_string="above upper bound",
            community_prediction=None,
            actual_resolution_time=datetime(2026, 5, 1),
            question_text="What value?",
            page_url="https://example.com/q/8001",
        )

        serialized = _serialize_ground_truth(original)
        # Round-trip via JSON to mimic the cache writer's behavior (default=str).
        import json

        round_tripped = json.loads(json.dumps(serialized, default=str))
        restored = _deserialize_ground_truth(round_tripped)

        assert restored.question_id == original.question_id
        assert restored.question_type == "numeric"
        assert restored.resolution == OutOfBoundsResolution.ABOVE_UPPER_BOUND
        assert restored.resolution_string == original.resolution_string
        assert restored.actual_resolution_time == original.actual_resolution_time


# ---------------------------------------------------------------------------
# Bug-4 regression: question shims hydrated from a manifest must carry
# ``open_time`` and ``scheduled_resolution_time`` so ``compute_mid_window_today``
# can perform real datetime arithmetic. A MagicMock(spec=...) without those
# attributes set returns sub-MagicMocks that pass the ``is not None`` assert
# but crash on subtraction.
# ---------------------------------------------------------------------------


class TestQuestionShimTimeFields:
    def test_question_shim_carries_open_time_and_scheduled_resolution_time(self) -> None:
        """``open_time`` and ``scheduled_resolution_time`` must round-trip through the manifest.

        ``compute_mid_window_today`` (used by the window patch) does
        ``scheduled_resolution_time - open_time``. If the manifest omits those
        fields, the score-only and ``--qids`` paths produce shims with
        sub-``MagicMock`` attributes that pass ``is not None`` but raise
        ``TypeError`` during datetime arithmetic. Tests pin both serialize and
        deserialize to round-trip the values as real datetimes.
        """
        from metaculus_bot.ablation.cli import (
            _build_manifest_entry,
            _build_question_shim_from_manifest_entry,
        )

        original_open = datetime(2026, 2, 1, 12, 0, 0)
        original_resolve = datetime(2026, 6, 15, 12, 0, 0)
        q = _make_binary_question(8501)
        q.open_time = original_open
        q.scheduled_resolution_time = original_resolve

        gt = _make_binary_ground_truth(8501, outcome=True)

        entry = _build_manifest_entry(q, gt, "spring-aib-2026")

        # Round-trip through JSON to mirror what cache.write does.
        import json

        round_tripped_entry = json.loads(json.dumps(entry, default=str))
        shim = _build_question_shim_from_manifest_entry(8501, round_tripped_entry)

        assert isinstance(shim.open_time, datetime), (
            f"shim.open_time must be a datetime, got {type(shim.open_time).__name__}"
        )
        assert isinstance(shim.scheduled_resolution_time, datetime), (
            f"shim.scheduled_resolution_time must be a datetime, got {type(shim.scheduled_resolution_time).__name__}"
        )
        # ft 0.2.92's add_timezone_to_dates validator coerces the naive manifest
        # datetimes to tz-aware UTC when the shim is constructed (old on-disk manifests
        # still hold naive ISO strings), so the rehydrated values are aware UTC.
        assert shim.open_time == original_open.replace(tzinfo=UTC)
        assert shim.scheduled_resolution_time == original_resolve.replace(tzinfo=UTC)
        # Sanity: subtraction (the operation that would crash on a sub-MagicMock) works.
        delta = shim.scheduled_resolution_time - shim.open_time
        assert delta.days > 0


# ---------------------------------------------------------------------------
# Bug-6 regression: question shims hydrated from a manifest must carry every
# attribute downstream code reads. Pydantic BaseModel fields are NOT class
# attributes, so ``MagicMock(spec=BinaryQuestion)`` raises AttributeError on
# ``question.resolution_criteria`` / ``fine_print`` / ``background_info`` /
# ``unit_of_measure`` unless they're explicitly set on the shim. The leakage
# detector at ``backtest/leakage.py:86`` reads ``question.resolution_criteria``;
# the stacker prompts read all of background_info/resolution_criteria/fine_print
# (and unit_of_measure for numeric).
# ---------------------------------------------------------------------------


class TestQuestionShimContentFields:
    """Round-trip every question content attribute downstream code reads."""

    def test_question_shim_carries_resolution_criteria(self) -> None:
        """``resolution_criteria`` must round-trip through the manifest.

        The leakage detector at ``backtest/leakage.py:86`` reads
        ``question.resolution_criteria`` to render its prompt. Without
        explicit set on the shim, ``MagicMock(spec=BinaryQuestion)`` raises
        AttributeError because Pydantic model fields aren't class attributes.
        """
        from metaculus_bot.ablation.cli import (
            _build_manifest_entry,
            _build_question_shim_from_manifest_entry,
        )

        criteria_text = "Resolves YES if the SEC files an enforcement action by 2026-12-31."
        q = _make_binary_question(9501)
        q.resolution_criteria = criteria_text

        gt = _make_binary_ground_truth(9501, outcome=True)
        entry = _build_manifest_entry(q, gt, "spring-aib-2026")

        import json

        round_tripped_entry = json.loads(json.dumps(entry, default=str))
        shim = _build_question_shim_from_manifest_entry(9501, round_tripped_entry)

        assert shim.resolution_criteria == criteria_text

    def test_question_shim_carries_fine_print(self) -> None:
        """``fine_print`` must round-trip through the manifest.

        The stacker prompts (`stacking_binary_prompt`, `stacking_multiple_choice_prompt`,
        `stacking_numeric_prompt`) embed `question.fine_print` directly. Without
        explicit set on the shim, attribute access raises AttributeError.
        """
        from metaculus_bot.ablation.cli import (
            _build_manifest_entry,
            _build_question_shim_from_manifest_entry,
        )

        fine_print_text = "If the SEC issues a no-action letter instead, this resolves NO."
        q = _make_binary_question(9502)
        q.fine_print = fine_print_text

        gt = _make_binary_ground_truth(9502, outcome=True)
        entry = _build_manifest_entry(q, gt, "spring-aib-2026")

        import json

        round_tripped_entry = json.loads(json.dumps(entry, default=str))
        shim = _build_question_shim_from_manifest_entry(9502, round_tripped_entry)

        assert shim.fine_print == fine_print_text

    def test_question_shim_carries_background_info(self) -> None:
        """``background_info`` must round-trip through the manifest.

        Every stacker prompt (binary/MC/numeric) embeds `question.background_info`.
        """
        from metaculus_bot.ablation.cli import (
            _build_manifest_entry,
            _build_question_shim_from_manifest_entry,
        )

        background = "The SEC has been investigating Acme Corp since 2025-03-15."
        q = _make_binary_question(9503)
        q.background_info = background

        gt = _make_binary_ground_truth(9503, outcome=True)
        entry = _build_manifest_entry(q, gt, "spring-aib-2026")

        import json

        round_tripped_entry = json.loads(json.dumps(entry, default=str))
        shim = _build_question_shim_from_manifest_entry(9503, round_tripped_entry)

        assert shim.background_info == background

    def test_question_shim_carries_unit_of_measure_for_numeric(self) -> None:
        """``unit_of_measure`` must round-trip for numeric questions.

        ``stacking_numeric_prompt`` and ``numeric_prompt`` both read
        ``question.unit_of_measure`` to format the bounds-and-units block.
        """
        from metaculus_bot.ablation.cli import (
            _build_manifest_entry,
            _build_question_shim_from_manifest_entry,
        )

        units = "barrels per day"
        q = _make_numeric_question(9504)
        q.unit_of_measure = units

        gt = _make_numeric_ground_truth(9504, value=42.0)
        entry = _build_manifest_entry(q, gt, "spring-aib-2026")

        import json

        round_tripped_entry = json.loads(json.dumps(entry, default=str))
        shim = _build_question_shim_from_manifest_entry(9504, round_tripped_entry)

        assert shim.unit_of_measure == units

    def test_question_shim_supports_leakage_detector_prompt_construction(self) -> None:
        """The shim from a manifest entry must work with ``_check_single_question_leakage``.

        Reproduces the live crash: the live ablation hit
        ``AttributeError: Mock object has no attribute 'resolution_criteria'``
        when the screen stage's leakage detector tried to render its prompt
        against a manifest-rehydrated shim. The fix is to round-trip
        resolution_criteria so the shim has the real string.
        """
        from metaculus_bot.ablation.cli import (
            _build_manifest_entry,
            _build_question_shim_from_manifest_entry,
        )
        from metaculus_bot.backtest.leakage import _check_single_question_leakage

        criteria_text = "Resolves YES if the launch occurs before 2026-12-31."
        q = _make_binary_question(9505)
        q.resolution_criteria = criteria_text

        gt = _make_binary_ground_truth(9505, outcome=True)
        entry = _build_manifest_entry(q, gt, "spring-aib-2026")

        import json

        round_tripped_entry = json.loads(json.dumps(entry, default=str))
        shim = _build_question_shim_from_manifest_entry(9505, round_tripped_entry)

        # Detector LLM whose ``invoke`` returns "NO" and records the prompt
        # via ``await_args_list``.
        detector = MagicMock()
        detector.invoke = AsyncMock(return_value="NO - clean.")

        is_leaked = asyncio.run(_check_single_question_leakage(shim, gt, "research blob text", detector))

        assert is_leaked is False
        # Crucial: the prompt construction did not crash with AttributeError, AND the
        # resolution criteria string flowed through to the prompt verbatim.
        assert detector.invoke.await_count == 1
        prompt_arg = detector.invoke.await_args_list[0].args[0]
        assert criteria_text in prompt_arg, (
            f"Resolution criteria must appear in detector prompt; got: {prompt_arg[:500]}"
        )


class TestNumericCdfSizeShimRoundTrip:
    """``cdf_size`` persists through the manifest so the rehydrated shim carries the real grid length.

    Discrete numeric questions carry ``cdf_size != 201`` (e.g. 17 for an integer-count 0..15
    question like the real qid 42752). The manifest writer now persists it and the shim reader
    restores it, so the ARM_PDF structured-math arm builds its CDF on the right grid. Older
    manifests (schema_version 1) that predate the field fall back to NumericQuestion's 201
    default rather than crashing.
    """

    def _discrete_numeric_question(self, qid: int = 42752) -> NumericQuestion:
        return NumericQuestion(
            id_of_question=qid,
            id_of_post=qid,
            question_text="How many items?",
            background_info="",
            resolution_criteria="Integer count 0..15.",
            fine_print="",
            lower_bound=-0.5,
            upper_bound=15.5,
            open_lower_bound=False,
            open_upper_bound=True,
            zero_point=None,
            unit_of_measure="Items",
            page_url=f"https://example.com/q/{qid}",
            open_time=_OPEN,
            scheduled_resolution_time=_RESOLVE,
            cdf_size=17,
        )

    def test_serialize_question_metadata_persists_cdf_size(self) -> None:
        from metaculus_bot.ablation.cli import _serialize_question_metadata

        metadata = _serialize_question_metadata(self._discrete_numeric_question())
        assert metadata["cdf_size"] == 17

    def test_serialize_question_metadata_persists_default_cdf_size_for_continuous(self) -> None:
        from metaculus_bot.ablation.cli import _serialize_question_metadata

        metadata = _serialize_question_metadata(_make_numeric_question(1))
        assert metadata["cdf_size"] == 201

    def test_shim_round_trips_discrete_cdf_size(self) -> None:
        from metaculus_bot.ablation.cli import _build_manifest_entry, _build_question_shim_from_manifest_entry

        question = self._discrete_numeric_question(qid=42752)
        gt = GroundTruth(
            question_id=42752,
            question_type="numeric",
            resolution=3.0,
            resolution_string="3",
            community_prediction=None,
            actual_resolution_time=_RESOLVE,
            question_text="How many items?",
            page_url="https://example.com/q/42752",
        )
        entry = _build_manifest_entry(question, gt, "spring-aib-2026")
        shim = _build_question_shim_from_manifest_entry(42752, entry)
        assert isinstance(shim, NumericQuestion)
        assert shim.cdf_size == 17

    def test_shim_defaults_cdf_size_when_manifest_omits_it(self) -> None:
        """An older manifest entry (no ``cdf_size`` key) rehydrates with the 201 default, not None."""
        from metaculus_bot.ablation.cli import _build_question_shim_from_manifest_entry

        entry = {
            "type": "numeric",
            "tournament": "spring-aib-2026",
            "question_text": "What value?",
            "page_url": "https://example.com/q/1",
            "id_of_post": 1,
            "ground_truth": {},
            "resolution_criteria": "Resolves to a number.",
            "fine_print": "",
            "background_info": "",
            "question_metadata": {
                "open_time": "2026-01-01T00:00:00",
                "scheduled_resolution_time": "2026-05-01T00:00:00",
                "lower_bound": 0.0,
                "upper_bound": 100.0,
                "open_lower_bound": False,
                "open_upper_bound": False,
                "zero_point": None,
                "unit_of_measure": None,
                # cdf_size deliberately absent — pre-fix schema_version 1 manifest.
            },
        }
        shim = _build_question_shim_from_manifest_entry(1, entry)
        assert isinstance(shim, NumericQuestion)
        assert shim.cdf_size == 201


# ---------------------------------------------------------------------------
# ft 0.2.92 tz-aware resolution-time window filter (lives here per the W2 brief:
# tests/test_backtest_question_prep.py — the natural home for question_prep —
# was outside this worker's scope, so the comparison-path test is parked in an
# owned file). ft 0.2.92 coerces actual_resolution_time to tz-aware UTC at
# construction; backtest window boundaries come from a naive strptime, so the
# filter must normalize both sides or the naive-vs-aware ordering raises
# TypeError and aborts every backtest fetch. The existing question_prep fixtures
# set actual_resolution_time POST-construction (which skips the validator, so it
# stays naive) and thus never exercise the aware path that fires in production.
# ---------------------------------------------------------------------------


class TestFilterHandlesTzAwareResolutionTime:
    @pytest.mark.asyncio
    async def test_fetch_filters_mixed_tz_aware_and_naive_resolution_times(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Aware (production) and naive (local fixture) resolution times both filter cleanly.

        Regression for the 0.2.92 naive-vs-aware TypeError: the aware questions mirror
        real API-sourced questions; the naive one exercises OUR ``_as_utc`` normalization
        of a naive value (the tz-robust path that must stay covered).
        """
        from forecasting_tools.data_models.questions import QuestionState

        from metaculus_bot.backtest.question_prep import fetch_resolved_questions

        def _binary(qid: int, resolved_at: datetime) -> BinaryQuestion:
            return BinaryQuestion(
                question_text=f"Q{qid}?",
                id_of_question=qid,
                resolution_string="yes",
                community_prediction_at_access_time=0.6,
                state=QuestionState.RESOLVED,
                api_json={},
                actual_resolution_time=resolved_at,
            )

        aware_in_window = _binary(9101, datetime(2026, 3, 1, tzinfo=UTC))
        aware_after_upper = _binary(9102, datetime(2026, 6, 1, tzinfo=UTC))  # >= upper => excluded
        naive_in_window = _binary(9103, datetime(2026, 3, 15))  # naive => _as_utc must normalize

        mock_fetch = AsyncMock(return_value=[aware_in_window, aware_after_upper, naive_in_window])
        monkeypatch.setattr("metaculus_bot.backtest.question_prep._fetch_with_retries", mock_fetch)

        result = await fetch_resolved_questions(
            total_questions=10,
            resolved_after="2026-01-01",
            resolved_before="2026-05-01",
        )

        qids = {q.id_of_question for q in result.questions}
        assert qids == {9101, 9103}
