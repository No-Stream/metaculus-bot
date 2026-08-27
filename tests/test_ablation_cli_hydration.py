"""Working-set hydration from cache: content-aware forecast-cache reads, ``--qids``
bypassing fetch, smoke→expand manifest growth, and how prune failures and the
``--qids`` filter interact with an already-hydrated set.

Split out of the original monolithic ``test_ablation_cli.py``. The shared thread is what
the CLI reconstructs from disk before any stage fires.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from metaculus_bot.backtest.scoring import GroundTruth
from tests import ablation_cli_fakes as _fakes
from tests.ablation_cli_fakes import (
    _binary_forecaster_payload,
    _binary_stacker_payload,
    _build_question_set,
    _install_full_stack_mocks,
    _make_binary_ground_truth,
    _make_binary_question,
    _make_mc_ground_truth,
    _make_mc_question,
    _make_numeric_ground_truth,
    _make_numeric_question,
    _mc_forecaster_payload,
    _mc_stacker_payload,
    _numeric_forecaster_payload,
    _numeric_stacker_payload,
    _populate_full_cache_for_qid,
)

# Fixture bound by assignment rather than imported: see the ablation_cli_fakes docstring.
cache_dir = _fakes.cache_dir

# ---------------------------------------------------------------------------
# Task #23: forecast cache check is CONTENT-AWARE
#
# The previous all-or-nothing check treated any non-empty list_forecaster_outputs
# result as a cache hit. This let mock-poisoned (errors-only) payloads serve as
# valid forecasts and the stacker downstream cached "insufficient_forecasters"
# permanently. The diagnosis at skipped_qids_diagnosis_20260515.md:69 documents
# the exact failure mode for qids 43077/43148/43150.
# ---------------------------------------------------------------------------


class TestForecastCacheContentAware:
    @pytest.mark.asyncio
    async def test_forecast_re_runs_when_all_cached_payloads_have_errors(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mutation test for Task #23: if the content-aware check is reverted,
        the test fails because the stage treats error-only payloads as cached.
        """
        from metaculus_bot.ablation.cli import _build_parser, run_ablation
        from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS

        # Pre-populate manifest + research + pruned + screen cache for qid.
        cache = AblationCache(cache_dir)
        _populate_full_cache_for_qid(cache, 12001)
        # Wipe the (synthetic) stacker outputs from _populate_full_cache_for_qid;
        # we want this run to actually re-forecast.
        for arm in ("stack", "B"):
            (cache.root / "stacker_outputs" / "12001" / f"arm_{arm}.json").unlink(missing_ok=True)
        # Replace the synthetic forecaster cache with all-error payloads from the
        # CURRENT lineup (so lineup_filter doesn't drop them).
        for stem in list((cache.root / "forecaster_outputs" / "12001").glob("*.json")):
            stem.unlink()
        for model in FREE_FORECASTER_MODELS:
            slug = model_slug_to_filename(model)
            cache.write_forecaster_output(
                qid=12001,
                model_slug=slug,
                payload={
                    "model": model,
                    "prediction_value": None,
                    "reasoning": "",
                    "errors": ["Mock object has no attribute 'id_of_post'"],
                    "ran_at": "2026-05-13T20:59:58Z",
                    "duration_seconds": 1e-5,
                },
            )

        question_set = _build_question_set([])
        forecaster_results = {
            12001: {
                model_slug_to_filename(model): _binary_forecaster_payload(model, 0.5)
                for model in FREE_FORECASTER_MODELS
            },
        }
        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            forecaster_results=forecaster_results,
            stacker_a_results={12001: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={12001: _binary_stacker_payload("stack_aug", 0.7)},
        )
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", AsyncMock(return_value=None))

        # Run only forecast (and downstream) — the upstream caches are pre-populated.
        args = _build_parser().parse_args(
            [
                "--stages",
                "forecast,stack,stack_aug,pdf,median,score",
                "--cache-dir",
                str(cache_dir),
            ]
        )
        await run_ablation(args)

        # Forecasters mock SHOULD have been called (the cached payloads were
        # all-errors, so the content-aware check forced a re-run).
        assert mocks["forecasters"].await_count == 1, (
            "Task #23: forecast cache hit must check that surviving forecasters "
            "(non-error, prediction_value!=None) reach ABLATION_MIN_FORECASTERS; "
            "otherwise a stale all-errors cache poisons every downstream stage."
        )

    @pytest.mark.asyncio
    async def test_forecast_uses_cache_when_surviving_meets_threshold(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sanity check: a fully-valid cache is honored (didn't break the happy path)."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation
        from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS

        cache = AblationCache(cache_dir)
        _populate_full_cache_for_qid(cache, 12002)
        # Replace synthetic 3-model forecaster cache with current-lineup valid payloads.
        for stem in list((cache.root / "forecaster_outputs" / "12002").glob("*.json")):
            stem.unlink()
        for model in FREE_FORECASTER_MODELS:
            slug = model_slug_to_filename(model)
            cache.write_forecaster_output(qid=12002, model_slug=slug, payload=_binary_forecaster_payload(model, 0.5))

        question_set = _build_question_set([])
        mocks = _install_full_stack_mocks(monkeypatch, fetch_question_set=question_set)
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", AsyncMock(return_value=None))

        args = _build_parser().parse_args(["--stages", "forecast", "--cache-dir", str(cache_dir)])
        await run_ablation(args)

        # Forecasters mock NOT called — cached payloads are valid.
        assert mocks["forecasters"].await_count == 0


class TestQidsBypassesFetch:
    @pytest.mark.asyncio
    async def test_qids_arg_bypasses_fetch_stage(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--qids 100,200 with empty cache: fetch mock NOT called, downstream uses those qids."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        # We're going to mock fetch_resolved_questions_stratified; if the runner
        # calls it, the test fails.
        question_set = _build_question_set([])

        # The orchestrator must build questions some other way. We mock
        # MetaculusApi.get_question_by_post_id (or whatever it uses) to return
        # canned questions for qids 100 and 200. The CLI is responsible for
        # routing qid → MetaculusQuestion. We fake-mock at the routing point.
        from metaculus_bot.ablation import cli as cli_module

        q100 = _make_binary_question(100)
        q200 = _make_binary_question(200)

        async def fake_load_qids(qids: list[int]) -> tuple[list[Any], dict[int, GroundTruth]]:
            questions: list[Any] = []
            gts: dict[int, GroundTruth] = {}
            await asyncio.sleep(0)
            for qid in qids:
                if qid == 100:
                    questions.append(q100)
                    gts[100] = _make_binary_ground_truth(100, outcome=True)
                elif qid == 200:
                    questions.append(q200)
                    gts[200] = _make_binary_ground_truth(200, outcome=False)
            return questions, gts

        monkeypatch.setattr(cli_module, "load_questions_by_qids", fake_load_qids)

        verdicts = {
            100: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
            200: {
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
                    f"openrouter/test/m{i}", value=0.5
                )
                for i in range(3)
            }
            for qid in (100, 200)
        }
        stacker_a = {qid: _binary_stacker_payload("stack", 0.6) for qid in (100, 200)}
        stacker_b = {qid: _binary_stacker_payload("stack_aug", 0.7) for qid in (100, 200)}

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={100: ("blob 100", {}), 200: ("blob 200", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results=stacker_a,
            stacker_b_results=stacker_b,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(
            ["--qids", "100,200", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "advisory"]
        )
        exit_code = await run_ablation(args)
        assert exit_code == 0

        # Fetch never called.
        assert mocks["fetch"].await_count == 0

        # Downstream stages saw qids 100, 200 (not whatever fake fetch would have).
        assert mocks["research"].await_count == 1
        # Inspect the questions arg passed to research.
        research_call_args = mocks["research"].await_args
        assert research_call_args is not None
        questions_arg = (
            research_call_args.args[0] if research_call_args.args else research_call_args.kwargs["questions"]
        )
        ids = sorted(q.id_of_question for q in questions_arg)
        assert ids == [100, 200]


# ---------------------------------------------------------------------------
# Manifest expansion
# ---------------------------------------------------------------------------


class TestManifestExpansion:
    @pytest.mark.asyncio
    async def test_smoke_then_expand_appends_qids(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """First run 1/1/1; second run 5/5/5. Manifest grows from 3 to 15 qids without re-sampling."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        # First fetch: 1/1/1.
        binary_q1 = _make_binary_question(1001)
        mc_q1 = _make_mc_question(1002)
        numeric_q1 = _make_numeric_question(1003)
        first_question_set = _build_question_set(
            [
                (binary_q1, _make_binary_ground_truth(1001)),
                (mc_q1, _make_mc_ground_truth(1002, "Red")),
                (numeric_q1, _make_numeric_ground_truth(1003, 50.0)),
            ]
        )

        first_verdicts = {
            qid: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            }
            for qid in (1001, 1002, 1003)
        }
        first_research: dict[int, tuple[str, dict] | None] = {qid: (f"blob {qid}", {}) for qid in (1001, 1002, 1003)}

        binary_payloads_first = {
            1001: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            }
        }
        mc_payloads_first = {
            1002: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _mc_forecaster_payload(f"openrouter/test/m{i}")
                for i in range(3)
            }
        }
        numeric_payloads_first = {
            1003: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _numeric_forecaster_payload(
                    f"openrouter/test/m{i}", 50.0
                )
                for i in range(3)
            }
        }
        forecasters_first = {**binary_payloads_first, **mc_payloads_first, **numeric_payloads_first}
        stacker_a_first = {
            1001: _binary_stacker_payload("stack", 0.6),
            1002: _mc_stacker_payload("A"),
            1003: _numeric_stacker_payload("A"),
        }
        stacker_b_first = {
            1001: _binary_stacker_payload("stack_aug", 0.7),
            1002: _mc_stacker_payload("B"),
            1003: _numeric_stacker_payload("B"),
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=first_question_set,
            research_results=first_research,
            leakage_verdicts=first_verdicts,
            forecaster_results=forecasters_first,
            stacker_a_results=stacker_a_first,
            stacker_b_results=stacker_b_first,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        argv_smoke = [
            "--num-binary",
            "1",
            "--num-multiple-choice",
            "1",
            "--num-numeric",
            "1",
            "--cache-dir",
            str(cache_dir),
            "--qa-iterate-mode",
            "advisory",
        ]
        await run_ablation(_build_parser().parse_args(argv_smoke))

        cache = AblationCache(cache_dir)
        manifest_after_smoke = cache.read_qids_manifest()
        assert set(manifest_after_smoke.keys()) == {1001, 1002, 1003}

        # Second fetch: 5/5/5. Re-install fetch mock to return only the NEW
        # questions (the orchestrator should ask for the delta: 4/4/4).
        new_binary_qs = [_make_binary_question(2000 + i) for i in range(4)]
        new_mc_qs = [_make_mc_question(2100 + i) for i in range(4)]
        new_numeric_qs = [_make_numeric_question(2200 + i) for i in range(4)]
        new_pairs = (
            [(q, _make_binary_ground_truth(cast(int, q.id_of_question))) for q in new_binary_qs]
            + [(q, _make_mc_ground_truth(cast(int, q.id_of_question), "Red")) for q in new_mc_qs]
            + [(q, _make_numeric_ground_truth(cast(int, q.id_of_question), 50.0)) for q in new_numeric_qs]
        )
        second_question_set = _build_question_set(new_pairs)

        all_new_qids = [cast(int, q.id_of_question) for q, _ in new_pairs]
        second_research: dict[int, tuple[str, dict] | None] = {qid: (f"blob {qid}", {}) for qid in all_new_qids}
        second_verdicts = {
            qid: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            }
            for qid in all_new_qids
        }
        forecasters_second: dict[int, dict[str, dict]] = {}
        for q in new_binary_qs:
            forecasters_second[cast(int, q.id_of_question)] = {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            }
        for q in new_mc_qs:
            forecasters_second[cast(int, q.id_of_question)] = {
                model_slug_to_filename(f"openrouter/test/m{i}"): _mc_forecaster_payload(f"openrouter/test/m{i}")
                for i in range(3)
            }
        for q in new_numeric_qs:
            forecasters_second[cast(int, q.id_of_question)] = {
                model_slug_to_filename(f"openrouter/test/m{i}"): _numeric_forecaster_payload(
                    f"openrouter/test/m{i}", 50.0
                )
                for i in range(3)
            }

        stacker_a_second = {}
        stacker_b_second = {}
        for q in new_binary_qs:
            stacker_a_second[q.id_of_question] = _binary_stacker_payload("stack", 0.6)
            stacker_b_second[q.id_of_question] = _binary_stacker_payload("stack_aug", 0.7)
        for q in new_mc_qs:
            stacker_a_second[q.id_of_question] = _mc_stacker_payload("A")
            stacker_b_second[q.id_of_question] = _mc_stacker_payload("B")
        for q in new_numeric_qs:
            stacker_a_second[q.id_of_question] = _numeric_stacker_payload("A")
            stacker_b_second[q.id_of_question] = _numeric_stacker_payload("B")

        mocks2 = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=second_question_set,
            research_results=second_research,
            leakage_verdicts=second_verdicts,
            forecaster_results=forecasters_second,
            stacker_a_results=stacker_a_second,
            stacker_b_results=stacker_b_second,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        argv_small = [
            "--num-binary",
            "5",
            "--num-multiple-choice",
            "5",
            "--num-numeric",
            "5",
            "--cache-dir",
            str(cache_dir),
            "--qa-iterate-mode",
            "advisory",
        ]
        await run_ablation(_build_parser().parse_args(argv_small))

        # Fetch was called with delta counts (4/4/4), excluding existing qids.
        assert mocks2["fetch"].await_count == 1
        # Check the manifest grew correctly.
        cache = AblationCache(cache_dir)
        new_manifest = cache.read_qids_manifest()
        # Original 3 + 12 new = 15 qids.
        assert len(new_manifest) == 15
        # Original qids preserved.
        assert {1001, 1002, 1003}.issubset(set(new_manifest.keys()))


# ---------------------------------------------------------------------------
# Regression: hydration must not promote raw research to research_blobs.
#
# The original prune stage drops qids with prune-validation failures from
# ``working.research_blobs`` so they never reach forecast/stack/score. The
# hydration path (used when re-running downstream stages from cache) must
# mirror this drop, otherwise re-running ``--stages forecast,stack,...``
# would forecast on raw (potentially leaked) research that the prune stage
# refused to sanitize.
# ---------------------------------------------------------------------------


class TestHydrationRespectsPruneFailures:
    @pytest.mark.asyncio
    async def test_hydration_skips_qid_with_no_pruned_blob(
        self,
        tmp_path: Path,
    ) -> None:
        """A qid present in research/ but absent from research_pruned/ MUST
        NOT appear in ``working.research_blobs`` after hydration.

        This pins the behavior that ``research_pruned`` is the canonical
        "ready for downstream consumption" gate. Raw research stays on disk
        for QA-dump purposes only.
        """
        from metaculus_bot.ablation.cli import (
            WorkingSet,
            _build_manifest_entry,
            _hydrate_working_set_from_cache,
        )

        cache = AblationCache(str(tmp_path))
        q_clean = _make_binary_question(qid=4001)
        q_unsanitized = _make_binary_question(qid=4002)
        gt_clean = _make_binary_ground_truth(4001, outcome=True)
        gt_unsanitized = _make_binary_ground_truth(4002, outcome=False)

        # Both qids in manifest with raw research; only the clean one has
        # a pruned blob.
        cache.append_qids_manifest(
            {
                4001: _build_manifest_entry(q_clean, gt_clean, "test-tournament"),
                4002: _build_manifest_entry(q_unsanitized, gt_unsanitized, "test-tournament"),
            }
        )
        cache.write_research(qid=4001, blob="raw research clean", meta={"provider": "gemini"})
        cache.write_research(qid=4002, blob="raw research with leak", meta={"provider": "gemini"})
        cache.write_pruned_research(qid=4001, sanitized_blob="sanitized clean", meta={"validation": "pass"})
        # Deliberately NO write_pruned_research for 4002 — simulates prune
        # validation failure.

        working = WorkingSet()
        await _hydrate_working_set_from_cache(cache, working)

        # Both questions in manifest are loaded as question objects.
        assert set(working.questions.keys()) == {4001, 4002}
        # But only the qid with a sanitized blob is eligible for downstream stages.
        assert 4001 in working.research_blobs
        assert 4002 not in working.research_blobs, (
            "qid 4002 had no sanitized blob (prune failure); hydration must NOT promote raw research downstream"
        )
        assert working.research_blobs[4001] == "sanitized clean"


# ---------------------------------------------------------------------------
# C2: --qids filter applies even when "fetch" is not in --stages.
#
# Without this, --qids 100 --stages stack,pdf,score loads the FULL
# manifest in hydration and runs every stage on every qid — operator pays
# 50x stacker spend when they meant to re-score one qid. See cli_audit
# (C2) for the full operator-footgun analysis.
# ---------------------------------------------------------------------------


class TestQidsFilterAppliedAfterHydration:
    @pytest.mark.asyncio
    async def test_qids_filter_when_fetch_not_in_stages_restricts_working_set(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--qids 100 --stages score must filter manifest to {100} only.

        Mutation test for C2: if the post-hydration filter is removed, the
        score summary will reference qid 200 too.
        """
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        cache = AblationCache(cache_dir)
        _populate_full_cache_for_qid(cache, 100)
        _populate_full_cache_for_qid(cache, 200)

        question_set = _build_question_set([])
        _install_full_stack_mocks(monkeypatch, fetch_question_set=question_set)
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", AsyncMock(return_value=None))

        args = _build_parser().parse_args(["--qids", "100", "--stages", "score", "--cache-dir", str(cache_dir)])
        exit_code = await run_ablation(args)
        assert exit_code == 0

        # Only qid 100's score should appear; qid 200 must be filtered out.
        summaries = sorted((cache_dir / "scores").glob("summary_*.md"))
        assert len(summaries) == 1
        text = summaries[-1].read_text(encoding="utf-8")
        # Per-question diagnostic table renders qids as `| <qid> |` rows.
        assert "| 100 |" in text
        assert "| 200 |" not in text, (
            f"--qids 100 must filter the working set even with --stages score; qid 200 leaked into summary:\n{text}"
        )
        # The metadata header records n_questions; should be 1, not 2.
        assert "N questions: 1" in text, f"expected N questions: 1; got summary:\n{text}"

    @pytest.mark.asyncio
    async def test_qids_filter_logs_error_for_qids_missing_from_manifest(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """--qids 999 (not in manifest) must log an error after hydration."""
        import logging

        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        cache = AblationCache(cache_dir)
        _populate_full_cache_for_qid(cache, 100)

        question_set = _build_question_set([])
        _install_full_stack_mocks(monkeypatch, fetch_question_set=question_set)
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", AsyncMock(return_value=None))

        caplog.set_level(logging.ERROR, logger="metaculus_bot.ablation.cli")
        args = _build_parser().parse_args(["--qids", "999", "--stages", "score", "--cache-dir", str(cache_dir)])
        await run_ablation(args)

        assert any("999" in record.message and "qids" in record.message.lower() for record in caplog.records), (
            f"expected error log mentioning qid 999; got: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_qids_filter_subset_via_stack_a_path(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--qids 100 --stages stack must run stacker only on qid 100, not on qid 200.

        Tests the path where qids filter applies BEFORE stack stages run.
        """
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        cache = AblationCache(cache_dir)
        _populate_full_cache_for_qid(cache, 100)
        _populate_full_cache_for_qid(cache, 200)

        question_set = _build_question_set([])
        mocks = _install_full_stack_mocks(monkeypatch, fetch_question_set=question_set)
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", AsyncMock(return_value=None))

        # --force-stages stack so we don't pick up the cached stack payload
        # and instead inspect what qids would be sent to the stacker mock.
        args = _build_parser().parse_args(
            [
                "--qids",
                "100",
                "--stages",
                "stack",
                "--force-stages",
                "stack",
                "--cache-dir",
                str(cache_dir),
            ]
        )
        await run_ablation(args)

        # Stacker was called with qid 100 only.
        assert mocks["stacker"].await_count >= 1
        for call in mocks["stacker"].await_args_list:
            qid_to_data = call.args[0] if call.args else call.kwargs["qid_to_data"]
            assert set(qid_to_data.keys()) <= {100}, (
                f"--qids 100 must restrict stacker to {{100}}; got {set(qid_to_data.keys())}"
            )
