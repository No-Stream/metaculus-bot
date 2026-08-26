"""The audit's per-question rankings inherit the per-model cohort guards.

``analysis.py`` was given ``per_model_cohort`` because a stacker-fired record's
per-model slot holds the stacker's aggregate and an anonymous ``Forecaster N``
key is a positional bucket, not a model. ``audit.py`` — which builds the
per-question dossier tables and the closest-to-truth synthesis tally, the
mandatory phase of every residual round — never adopted it: over 423 archived
binary records ``Forecaster 1`` was the third most frequent "best model" (36
wins) and 16 confirmed-stacker records still produced per-model rankings.

The numeric ranker had a second gap of its own: a declared curve is PCHIP'd into
a full CDF before scoring, so a 3-anchor recovery is scored as a distribution the
model never declared. On q43729 that curve ranked #1 at +92.01 against five
11-anchor siblings; on q43826 the same shape ranked last at -135.86.

These tests pin all three exclusions plus the disclosure that goes with them.
"""

from __future__ import annotations

import logging

import pytest

from metaculus_bot.performance_analysis.audit import (
    emit_miss_markdown,
    emit_synthesis,
    rank_our_models_by_accuracy,
    ranking_caveats,
)
from metaculus_bot.performance_analysis.parsing import anonymous_model_key
from metaculus_bot.performance_analysis.ranking_cohort import (
    MIN_SCOREABLE_ANCHORS,
    declared_anchors,
    per_model_ranking_cohort,
)
from metaculus_bot.performance_analysis.stacker_detection import detect_stacker_fired
from tests.test_performance_analysis_audit import _binary_record, _numeric_record

# The pre-2026-07 standard declared set (11 labels). Any curve built from it clears
# MIN_SCOREABLE_ANCHORS, so it stands in for a fully-recovered member.
_DENSE_LABELS: tuple[float, ...] = (2.5, 5.0, 10.0, 20.0, 40.0, 50.0, 60.0, 80.0, 90.0, 95.0, 97.5)

# What a trimmed comment leaves behind: the three percentiles the summary restates.
_SPARSE_LABELS: tuple[float, ...] = (10.0, 50.0, 90.0)


def _dense_curve(center: float, half_width: float = 20.0) -> list[tuple[float, float]]:
    """An 11-anchor declared curve centred on ``center``, strictly increasing."""
    return [(label, center + half_width * (label - 50.0) / 47.5) for label in _DENSE_LABELS]


def _sparse_curve(center: float, half_width: float = 5.0) -> list[tuple[float, float]]:
    """A 3-anchor declared curve — the partial-recovery shape."""
    return [(label, center + half_width * (label - 50.0) / 40.0) for label in _SPARSE_LABELS]


class TestAnonymousKeysExcludedFromRankings:
    """H5: a positional ``Forecaster N`` bucket is not an ensemble member."""

    @staticmethod
    def _record_where_the_anonymous_bullet_is_closest() -> dict:
        # Resolves YES; the unattributed bullet at 90% is nearest the truth, so
        # before the fix it took rank 1 and the "times best" tally credited it.
        return _binary_record(
            9001,
            0.30,
            True,
            per_model={"gpt-5.5": "40%", anonymous_model_key(1): "90%"},
        )

    def test_positional_key_is_not_ranked_as_a_model(self):
        ranked = rank_our_models_by_accuracy(self._record_where_the_anonymous_bullet_is_closest())
        assert [r["model"] for r in ranked] == ["gpt-5.5"]

    def test_positional_key_cannot_win_the_synthesis_best_model_tally(self, tmp_path):
        rec = self._record_where_the_anonymous_bullet_is_closest()
        out = tmp_path / "synthesis.md"
        emit_synthesis([{"record": rec, "ranked": rank_our_models_by_accuracy(rec)}], out)
        text = out.read_text()
        assert "gpt-5.5" in text
        assert anonymous_model_key(1) not in text

    def test_base_model_positional_variant_is_excluded_too(self):
        # ``Forecaster N base`` keys come from the stacker-combined body's base
        # sub-blocks and are equally positional.
        rec = _binary_record(
            9002,
            0.30,
            True,
            per_model={"gpt-5.5": "40%", anonymous_model_key(1, is_base_model=True): "90%"},
        )
        assert [r["model"] for r in rank_our_models_by_accuracy(rec)] == ["gpt-5.5"]

    def test_exclusion_is_stated_in_the_dossier(self, tmp_path):
        rec = self._record_where_the_anonymous_bullet_is_closest()
        out = tmp_path / "miss.md"
        emit_miss_markdown(
            rec,
            rank_our_models_by_accuracy(rec),
            per_model_reasoning={},
            audit_dir=tmp_path,
            out_path=out,
        )
        text = out.read_text()
        assert "Excluded 1 unattributed bullet(s)" in text
        assert "positional key is not a model" in text


class TestStackerFiredRecordsExcludedFromRankings:
    """H5, other half: on a stacked record every per-model slot is an aggregate."""

    def test_confirmed_stacker_binary_record_is_not_ranked(self):
        rec = _binary_record(9101, 0.30, True, per_model={"gpt-5.5": "40%", "claude-opus-4.8": "20%"})
        rec["was_stacked"] = True
        assert detect_stacker_fired(rec) == "confirmed_stacker"
        assert rank_our_models_by_accuracy(rec) == []

    def test_confirmed_stacker_numeric_record_is_not_ranked(self):
        rec = _numeric_record(
            9102,
            resolution=50.0,
            per_model_percentiles={"gpt-5.5": _dense_curve(50.0), "claude-opus-4.8": _dense_curve(60.0)},
        )
        rec["was_stacked"] = True
        assert detect_stacker_fired(rec) == "confirmed_stacker"
        assert rank_our_models_by_accuracy(rec) == []

    def test_dossier_gives_the_stacker_reason_not_no_forecasts_parsed(self, tmp_path):
        rec = _binary_record(9103, 0.30, True, per_model={"gpt-5.5": "40%", "claude-opus-4.8": "20%"})
        rec["was_stacked"] = True
        out = tmp_path / "miss.md"
        emit_miss_markdown(rec, [], per_model_reasoning={}, audit_dir=tmp_path, out_path=out)
        text = out.read_text()
        assert "the stacker produced this question's published value" in text
        assert "No per-model forecasts parsed" not in text

    def test_likely_stacker_is_still_ranked(self):
        # Deliberate non-exclusion, mirroring per_model_cohort: ``likely_stacker``
        # is a spread-plus-delta heuristic that also matches an ordinary MEAN-era
        # aggregate, so honoring it would delete the high-disagreement questions
        # these rankings exist to read.
        rec = _binary_record(9104, 0.20, True, per_model={"gpt-5.5": "80%", "claude-opus-4.8": "10%"})
        assert detect_stacker_fired(rec) == "likely_stacker"
        assert {r["model"] for r in rank_our_models_by_accuracy(rec)} == {"gpt-5.5", "claude-opus-4.8"}


class TestSparseAnchorGate:
    """H6: a sparse recovery may not be log-scored against a full curve."""

    @staticmethod
    def _mixed_density_record() -> dict:
        # The archived shape: several fully-recovered members plus one 3-anchor
        # recovery, here tight around the truth so it would top the table.
        return _numeric_record(
            9201,
            resolution=50.0,
            per_model_percentiles={
                "gpt-5.4": _sparse_curve(50.0),
                "gemini-3.1-pro": _dense_curve(50.0),
                "claude-opus-4.8": _dense_curve(58.0),
            },
        )

    def test_sparse_curve_excluded_when_siblings_are_dense(self):
        ranked = rank_our_models_by_accuracy(self._mixed_density_record())
        assert {r["model"] for r in ranked} == {"gemini-3.1-pro", "claude-opus-4.8"}

    def test_the_excluded_curve_would_have_topped_the_table(self):
        # Proves the exclusion is load-bearing rather than cosmetic: scored on its
        # own the 3-anchor curve beats every 11-anchor member, which is the
        # q43729 artifact (+92.01 at rank 1) in miniature.
        mixed = self._mixed_density_record()
        ranked = rank_our_models_by_accuracy(mixed)
        alone = _numeric_record(9202, resolution=50.0, per_model_percentiles={"gpt-5.4": _sparse_curve(50.0)})
        [sparse_scored] = rank_our_models_by_accuracy(alone)
        assert sparse_scored["score"] > max(r["score"] for r in ranked)

    def test_exclusion_is_stated_with_the_anchor_count(self, tmp_path):
        rec = self._mixed_density_record()
        out = tmp_path / "miss.md"
        emit_miss_markdown(
            rec, rank_our_models_by_accuracy(rec), per_model_reasoning={}, audit_dir=tmp_path, out_path=out
        )
        text = out.read_text()
        assert f"insufficient anchors (fewer than {MIN_SCOREABLE_ANCHORS}" in text
        assert "gpt-5.4 (3 anchors)" in text

    def test_uniformly_sparse_record_is_still_ranked_and_labelled(self):
        # The fall-2025 8-percentile era: every member is equally sparse, so the
        # within-question ranking compares equals. An absolute floor alone would
        # delete 11 valid archived rankings to fix the 5 mixed-density ones.
        eight_labels = (5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 90.0, 95.0)
        curves = {
            model: [(label, center + (label - 50.0) / 5.0) for label in eight_labels]
            for model, center in (("gpt-5.2", 50.0), ("claude-4.6-opus", 55.0), ("grok-4.5", 45.0))
        }
        rec = _numeric_record(9203, resolution=50.0, per_model_percentiles=curves)
        cohort = per_model_ranking_cohort(rec)
        assert cohort.sparse_era is True
        assert cohort.sparse_anchors == {}
        assert {r["model"] for r in rank_our_models_by_accuracy(rec)} == set(curves)
        assert any("Sparse-era question" in note for note in ranking_caveats(cohort))

    def test_anchor_count_is_rendered_in_the_numeric_table(self, tmp_path):
        # _summarize_percentiles renders only P10/P50/P90, so without this column
        # a sparse curve is indistinguishable from a full one in the dossier.
        rec = _numeric_record(
            9204,
            resolution=50.0,
            per_model_percentiles={"gemini-3.1-pro": _dense_curve(50.0), "claude-opus-4.8": _dense_curve(58.0)},
        )
        out = tmp_path / "miss.md"
        emit_miss_markdown(
            rec, rank_our_models_by_accuracy(rec), per_model_reasoning={}, audit_dir=tmp_path, out_path=out
        )
        text = out.read_text()
        assert "| anchors |" in text
        assert f"| {len(_DENSE_LABELS)} |" in text

    def test_a_restated_curve_counts_distinct_anchors_not_lines(self):
        # One archived curve carries its 11-point set twice (22 pairs). Counting
        # lines would let a 3-anchor set restated twice pass a 6-anchor floor.
        restated = _sparse_curve(50.0) * 2
        rec = _numeric_record(
            9205,
            resolution=50.0,
            per_model_percentiles={"gpt-5.4": restated, "gemini-3.1-pro": _dense_curve(50.0)},
        )
        cohort = per_model_ranking_cohort(rec)
        assert cohort.sparse_anchors == {"gpt-5.4": len(_SPARSE_LABELS)}
        assert list(cohort.entries) == ["gemini-3.1-pro"]

    def test_conflicting_restatement_is_logged(self, caplog):
        conflicting = [*_dense_curve(50.0), (50.0, 999.0)]
        rec = _numeric_record(9206, resolution=50.0, per_model_percentiles={"gpt-5.4": conflicting})
        with caplog.at_level(logging.WARNING):
            rank_our_models_by_accuracy(rec)
        assert any("Conflicting percentile restatement" in r.getMessage() for r in caplog.records)


class TestDeclaredAnchors:
    def test_keys_by_label_and_counts_disagreements(self):
        anchors, conflicts = declared_anchors([(10.0, 1.0), (50.0, 2.0), (10.0, 1.0)])
        assert anchors == {10.0: 1.0, 50.0: 2.0}
        assert conflicts == 0

        anchors, conflicts = declared_anchors([(10.0, 1.0), (10.0, 7.0)])
        assert anchors == {10.0: 7.0}  # last value wins, as the PCHIP build does
        assert conflicts == 1

    def test_multiple_choice_record_has_no_ranking_cohort(self):
        cohort = per_model_ranking_cohort({"type": "multiple_choice", "per_model_forecasts": {"gpt-5.5": {"A": 0.5}}})
        assert cohort.entries == {}
        assert cohort.stacker_fired is False


class TestSynthesisDeltaTable:
    def test_missing_ensemble_brier_is_omitted_not_rendered_as_perfect(self, tmp_path):
        # The old fallback rendered an unscored ensemble as Brier 0.000 with
        # delta +0.000 — a perfect ensemble that lost nothing by aggregating.
        scored = _binary_record(9301, 0.30, True, per_model={"gpt-5.5": "40%"})
        unscored = _binary_record(9302, 0.30, True, per_model={"gpt-5.5": "40%"})
        unscored["brier_score"] = None
        entries = [{"record": r, "ranked": rank_our_models_by_accuracy(r)} for r in (scored, unscored)]
        out = tmp_path / "synthesis.md"
        emit_synthesis(entries, out)
        text = out.read_text()
        delta_section = text.split("Ensemble-vs-best delta")[-1]
        assert "| 9301 |" in delta_section
        assert "| 9302 |" not in delta_section
        assert "9302" in text  # named as omitted rather than silently dropped
        assert "0.000 | 0.010 | +0.000" not in text


@pytest.mark.parametrize("q_type", ["numeric", "discrete"])
def test_both_continuous_types_get_the_anchor_gate(q_type: str):
    rec = _numeric_record(
        9401,
        resolution=50.0,
        per_model_percentiles={"gpt-5.4": _sparse_curve(50.0), "gemini-3.1-pro": _dense_curve(50.0)},
        q_type=q_type,
    )
    assert [r["model"] for r in rank_our_models_by_accuracy(rec)] == ["gemini-3.1-pro"]
