"""Tests for the audit-specific additions to performance_analysis.

Covers:
- parse_per_model_reasoning_text: extracts prose per forecaster, stripping the
  Model: line so the body is just what the model produced.
- audit.load_combined_dataset: merges Q1+Q2 with Q2-preferring dedupe.
- audit.select_worst_misses: picks top-N by score per question type.
- audit.rank_our_models_by_accuracy: ranks ensemble members by their individual
  forecasts against the resolution.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from metaculus_bot.performance_analysis.parsing import (
    parse_per_model_forecasts,
    parse_per_model_reasoning_text,
)


def _build_comment(
    bullets: list[tuple[int, str]],
    rationales: list[tuple[int, str, str]],
) -> str:
    """Build a bot-comment string with prose bodies.

    bullets:    list of (idx, value) for Report 1 Summary
    rationales: list of (idx, model, prose_body)
    """
    bullet_lines = "\n".join(f"*Forecaster {i}*: {v}" for i, v in bullets)
    rationale_sections = "\n\n".join(
        f"## R1: Forecaster {i} Reasoning\nModel: {m}\n\n{body}" for i, m, body in rationales
    )
    return f"""# SUMMARY
*Question*: ?

## Report 1 Summary
### Forecasts
{bullet_lines}


### Research Summary
Some research text.

================================================================================
FORECAST SECTION:

{rationale_sections}
"""


class TestParsePerModelReasoningText:
    def test_extracts_prose_keyed_by_model(self):
        comment = _build_comment(
            bullets=[(1, "73.0%"), (2, "75.0%")],
            rationales=[
                (1, "openrouter/openai/gpt-5.2", "Analysis by gpt.\n\nProbability: 73.0%"),
                (2, "openrouter/anthropic/claude-opus-4.5", "Reasoning from claude.\n\nProbability: 75.0%"),
            ],
        )
        result = parse_per_model_reasoning_text(comment)
        assert set(result.keys()) == {"gpt-5.2", "claude-opus-4.5"}
        assert "Analysis by gpt." in result["gpt-5.2"]
        assert "Probability: 73.0%" in result["gpt-5.2"]
        assert "Reasoning from claude." in result["claude-opus-4.5"]

    def test_strips_model_line_from_body(self):
        # The Model: line must not leak into the prose — that's metadata, not reasoning.
        comment = _build_comment(
            bullets=[(1, "50%")],
            rationales=[(1, "openrouter/openai/gpt-5.2", "The actual prose starts here.")],
        )
        result = parse_per_model_reasoning_text(comment)
        assert "Model:" not in result["gpt-5.2"]
        assert "openrouter" not in result["gpt-5.2"]
        assert result["gpt-5.2"].startswith("The actual prose")

    def test_keys_match_parse_per_model_forecasts(self):
        # Invariant for audit: reasoning keys and forecast keys should line up
        # so we can pair {model: reasoning} with {model: forecast}.
        comment = _build_comment(
            bullets=[(1, "73.0%"), (2, "75.0%"), (3, "85.0%")],
            rationales=[
                (1, "openrouter/openai/gpt-5.2", "one"),
                (2, "openrouter/openai/gpt-5.1", "two"),
                (3, "openrouter/anthropic/claude-4.6-opus", "three"),
            ],
        )
        reasoning = parse_per_model_reasoning_text(comment)
        forecasts = parse_per_model_forecasts(comment)
        assert set(reasoning.keys()) == set(forecasts.keys())

    def test_empty_prose_skipped(self):
        # A section with Model: line but no body should not produce an empty entry.
        comment = "## R1: Forecaster 1 Reasoning\nModel: openrouter/openai/gpt-5.2\n\n"
        assert parse_per_model_reasoning_text(comment) == {}

    def test_missing_model_line_uses_anonymized_key(self):
        comment = "## R1: Forecaster 1 Reasoning\nactual prose here.\n"
        result = parse_per_model_reasoning_text(comment)
        assert result == {"Forecaster 1": "actual prose here."}

    def test_explicit_model_names_override(self):
        comment = (
            "## R1: Forecaster 1 Reasoning\nsome prose for one\n\n## R1: Forecaster 2 Reasoning\nsome prose for two\n"
        )
        result = parse_per_model_reasoning_text(comment, model_names=["alpha", "beta"])
        assert result == {"alpha": "some prose for one", "beta": "some prose for two"}

    def test_preserves_multiline_body(self):
        body = "Line 1\nLine 2\n\n### Sub-section\nLine 3\n\nProbability: 50%"
        comment = _build_comment(
            bullets=[(1, "50%")],
            rationales=[(1, "openrouter/openai/gpt-5.2", body)],
        )
        result = parse_per_model_reasoning_text(comment)
        assert "Line 1" in result["gpt-5.2"]
        assert "Line 2" in result["gpt-5.2"]
        assert "### Sub-section" in result["gpt-5.2"]
        assert "Probability: 50%" in result["gpt-5.2"]

    def test_empty_comment(self):
        assert parse_per_model_reasoning_text("") == {}


# ---------------------------------------------------------------------------
# audit module — orchestration
# ---------------------------------------------------------------------------


def _binary_record(post_id: int, prob_yes: float, resolution: bool, **kw) -> dict:
    brier = (prob_yes - (1.0 if resolution else 0.0)) ** 2
    return {
        "post_id": post_id,
        "type": "binary",
        "title": kw.get("title", f"Q{post_id}"),
        "resolution_raw": "yes" if resolution else "no",
        "resolution_parsed": resolution,
        "our_prob_yes": prob_yes,
        "our_forecast_values": [1.0 - prob_yes, prob_yes],
        "per_model_forecasts": kw.get("per_model", {}),
        "per_model_numeric_percentiles": {},
        "comment_text": kw.get("comment_text", ""),
        "brier_score": brier,
        "log_score": 0.0,
        "numeric_log_score": None,
        "mc_log_score": None,
        "options": None,
        "scaling": {},
        "open_lower_bound": False,
        "open_upper_bound": False,
        "metadata": {"category": kw.get("category"), "nr_forecasters": kw.get("nr_forecasters", 0)},
    }


class TestLoadCombinedDataset:
    def test_q2_preferred_on_duplicate_post_id(self, tmp_path):
        from metaculus_bot.performance_analysis.audit import load_combined_dataset

        q1_path = tmp_path / "q1.json"
        q2_path = tmp_path / "q2.json"
        q1_rec = _binary_record(100, 0.2, True, title="q1-version")
        q2_rec = _binary_record(100, 0.2, True, title="q2-version")
        q1_only = _binary_record(200, 0.5, False, title="q1-unique")
        q2_only = _binary_record(300, 0.7, True, title="q2-unique")
        q1_path.write_text(json.dumps([q1_rec, q1_only]))
        q2_path.write_text(json.dumps([q2_rec, q2_only]))

        merged = load_combined_dataset(str(q1_path), str(q2_path))
        by_pid = {r["post_id"]: r for r in merged}
        assert set(by_pid.keys()) == {100, 200, 300}
        assert by_pid[100]["title"] == "q2-version"
        assert by_pid[100]["_cohort"] == "Q2"
        assert by_pid[200]["_cohort"] == "Q1"
        assert by_pid[300]["_cohort"] == "Q2"


class TestSelectWorstMisses:
    def test_takes_top_n_per_type_and_drops_none_scores(self):
        from metaculus_bot.performance_analysis.audit import select_worst_misses

        records = [
            _binary_record(1, 0.05, True),  # brier 0.9025
            _binary_record(2, 0.10, True),  # brier 0.81
            _binary_record(3, 0.95, True),  # brier 0.0025 — should NOT be selected
            _binary_record(4, 0.5, True),  # brier 0.25
            {**_binary_record(5, 0.0, True), "brier_score": None},  # drop
        ]
        worst = select_worst_misses(records, n_binary=2, n_numeric=0, n_mc=0)
        pids = [r["post_id"] for r in worst]
        assert pids == [1, 2]

    def test_numeric_uses_log_score_ascending(self):
        from metaculus_bot.performance_analysis.audit import select_worst_misses

        rs = [
            {"post_id": 10, "type": "numeric", "numeric_log_score": -30.0, "brier_score": None, "mc_log_score": None},
            {"post_id": 11, "type": "numeric", "numeric_log_score": -10.0, "brier_score": None, "mc_log_score": None},
            {"post_id": 12, "type": "numeric", "numeric_log_score": 5.0, "brier_score": None, "mc_log_score": None},
            {"post_id": 13, "type": "discrete", "numeric_log_score": -20.0, "brier_score": None, "mc_log_score": None},
        ]
        worst = select_worst_misses(rs, n_binary=0, n_numeric=2, n_mc=0)
        assert [r["post_id"] for r in worst] == [10, 13]

    def test_mc_uses_log_score_ascending(self):
        from metaculus_bot.performance_analysis.audit import select_worst_misses

        rs = [
            {
                "post_id": 20,
                "type": "multiple_choice",
                "mc_log_score": -50.0,
                "brier_score": None,
                "numeric_log_score": None,
            },
            {
                "post_id": 21,
                "type": "multiple_choice",
                "mc_log_score": 10.0,
                "brier_score": None,
                "numeric_log_score": None,
            },
        ]
        worst = select_worst_misses(rs, n_binary=0, n_numeric=0, n_mc=1)
        assert [r["post_id"] for r in worst] == [20]

    def test_peer_score_beats_brier_when_both_present(self):
        # Record A has worse Brier but better peer (crowd did even worse). Record B
        # has better Brier but worse peer (crowd beat us). Peer ranking should pick B first.
        from metaculus_bot.performance_analysis.audit import select_worst_misses

        a = _binary_record(1, 0.10, True)  # Brier = 0.81
        a["metaculus_scores"] = {"peer_score": -5.0}
        b = _binary_record(2, 0.30, True)  # Brier = 0.49, but peer -50 (worse vs crowd)
        b["metaculus_scores"] = {"peer_score": -50.0}
        worst = select_worst_misses([a, b], n_binary=2, n_numeric=0, n_mc=0)
        assert [r["post_id"] for r in worst] == [2, 1]

    def test_falls_back_to_brier_when_peer_missing(self):
        from metaculus_bot.performance_analysis.audit import select_worst_misses

        a = _binary_record(1, 0.05, True)  # Brier = 0.9025 — worst
        b = _binary_record(2, 0.50, True)  # Brier = 0.25
        # Neither has metaculus_scores → must use fallback Brier ordering.
        worst = select_worst_misses([a, b], n_binary=2, n_numeric=0, n_mc=0)
        assert [r["post_id"] for r in worst] == [1, 2]

    def test_peer_ranks_across_question_types_within_same_bucket(self):
        # A binary with very negative peer (big crowd regret) should beat a
        # binary with mild peer, regardless of absolute Brier difference.
        from metaculus_bot.performance_analysis.audit import select_worst_misses

        mild = _binary_record(1, 0.45, True)
        mild["metaculus_scores"] = {"peer_score": -1.0}
        severe = _binary_record(2, 0.40, True)
        severe["metaculus_scores"] = {"peer_score": -80.0}
        worst = select_worst_misses([mild, severe], n_binary=1, n_numeric=0, n_mc=0)
        assert worst[0]["post_id"] == 2

    def test_extra_post_ids_appended_deduplicated(self):
        # Extras that aren't in the auto-selected top-N should be appended;
        # extras already present should not be duplicated.
        from metaculus_bot.performance_analysis.audit import select_worst_misses

        auto_pick = _binary_record(1, 0.10, True)  # Brier 0.81 — will be picked
        auto_pick["metaculus_scores"] = {"peer_score": -60.0}
        minor = _binary_record(2, 0.40, True)
        minor["metaculus_scores"] = {"peer_score": -5.0}
        extra_only = _binary_record(3, 0.45, True)
        extra_only["metaculus_scores"] = {"peer_score": -2.0}

        # n_binary=1 so only post 1 is auto-picked; 3 is added via extras;
        # 1 is also in extras list but must not duplicate.
        worst = select_worst_misses(
            [auto_pick, minor, extra_only],
            n_binary=1,
            n_numeric=0,
            n_mc=0,
            extra_post_ids=[1, 3],
        )
        pids = [r["post_id"] for r in worst]
        assert pids == [1, 3]  # auto first, then new extra; no dupes

    def test_extra_post_ids_missing_record_is_skipped(self):
        from metaculus_bot.performance_analysis.audit import select_worst_misses

        rec = _binary_record(1, 0.10, True)
        rec["metaculus_scores"] = {"peer_score": -10.0}
        # extra_post_ids references a post_id not in records — must not crash
        worst = select_worst_misses([rec], n_binary=1, n_numeric=0, n_mc=0, extra_post_ids=[999])
        assert [r["post_id"] for r in worst] == [1]


class TestRankOurModelsByAccuracy:
    def test_ranks_binary_per_model_forecasts_closest_first(self):
        from metaculus_bot.performance_analysis.audit import rank_our_models_by_accuracy

        rec = _binary_record(
            1,
            0.2,
            True,
            per_model={"gpt-5.2": "80%", "claude-opus-4.5": "10%", "gemini-3.1": "50%"},
        )
        ranked = rank_our_models_by_accuracy(rec)
        assert [r["model"] for r in ranked] == ["gpt-5.2", "gemini-3.1", "claude-opus-4.5"]
        assert ranked[0]["prob"] == pytest.approx(0.80)
        # Brier: (0.8-1)^2=0.04, (0.5-1)^2=0.25, (0.1-1)^2=0.81
        assert ranked[0]["score"] == pytest.approx(0.04)

    def test_handles_missing_per_model(self):
        from metaculus_bot.performance_analysis.audit import rank_our_models_by_accuracy

        rec = _binary_record(1, 0.2, True, per_model={})
        assert rank_our_models_by_accuracy(rec) == []

    def test_skips_unparseable_values(self):
        from metaculus_bot.performance_analysis.audit import rank_our_models_by_accuracy

        rec = _binary_record(
            1,
            0.2,
            True,
            per_model={"gpt-5.2": "80%", "malformed": "not-a-number", "claude": "60%"},
        )
        ranked = rank_our_models_by_accuracy(rec)
        pids = [r["model"] for r in ranked]
        assert "malformed" not in pids
        assert set(pids) == {"gpt-5.2", "claude"}


class TestEmitSynthesisSpreadOrdering:
    def test_high_spread_uses_prob_range_not_rank_order(self, tmp_path):
        # Guard against a bug where spread = ranked[-1].prob - ranked[0].prob.
        # When question resolves YES and the best model said 13% while the
        # worst said 5%, Brier-ranking puts 13 first and 5 last, so
        # ranked[-1] - ranked[0] = -8pp. Spread must be absolute: 8pp.
        from metaculus_bot.performance_analysis.audit import emit_synthesis, rank_our_models_by_accuracy

        rec = _binary_record(
            43131,
            prob_yes=0.065,
            resolution=True,
            title="Opus release question",
            per_model={"gpt-5.2": "13%", "gemini": "5%", "claude": "7%"},
        )
        ranked = rank_our_models_by_accuracy(rec)
        out = tmp_path / "synthesis.md"
        emit_synthesis([{"record": rec, "ranked": ranked}], out)
        text = out.read_text()
        assert "8.0pp" in text
        assert "-8.0pp" not in text


class TestEmitSynthesisMixedCohort:
    """Phase 0 added numeric ranking via _rank_numeric, which produces entries
    with ``percentiles`` instead of ``prob``. The high-spread section was
    binary-only; passing a mixed binary+numeric cohort previously raised
    ``KeyError: 'prob'``. The fix is a type-aware skip in the spread-section
    loop. This test locks the fix.
    """

    def test_mixed_binary_and_numeric_does_not_raise(self, tmp_path):
        from metaculus_bot.performance_analysis.audit import emit_synthesis, rank_our_models_by_accuracy

        binary_rec = _binary_record(
            41001,
            prob_yes=0.2,
            resolution=True,
            title="binary q",
            per_model={"gpt-5.2": "13%", "claude": "20%"},
        )
        per_model_percentiles = {
            "gpt-5.2": [(10.0, 40.0), (50.0, 50.0), (90.0, 60.0)],
            "claude": [(10.0, 30.0), (50.0, 45.0), (90.0, 55.0)],
        }
        numeric_rec = _numeric_record(41002, resolution=50.0, per_model_percentiles=per_model_percentiles)

        binary_ranked = rank_our_models_by_accuracy(binary_rec)
        numeric_ranked = rank_our_models_by_accuracy(numeric_rec)

        # Mixed cohort: should NOT raise KeyError.
        out = tmp_path / "synthesis.md"
        emit_synthesis(
            [
                {"record": binary_rec, "ranked": binary_ranked},
                {"record": numeric_rec, "ranked": numeric_ranked},
            ],
            out,
        )
        text = out.read_text()

        # Spread section appears (with the binary entry).
        assert "spread" in text.lower()
        # Binary record's post id appears in the synthesis.
        assert "41001" in text
        # Numeric record is silently skipped from the spread section — its
        # post id is NOT in the spread table (numeric percentile-based spread
        # isn't directly comparable to binary prob-range).
        spread_section = text.split("High-spread")[-1] if "High-spread" in text else ""
        assert "41002" not in spread_section


class TestEmitMissMarkdown:
    """Exercises audit.emit_miss_markdown — the per-question audit file."""

    def test_emits_expected_headers_and_fields(self, tmp_path):
        from metaculus_bot.performance_analysis.audit import emit_miss_markdown, rank_our_models_by_accuracy

        rec = _binary_record(
            1234,
            prob_yes=0.10,
            resolution=True,
            title="Example question",
            per_model={"gpt-5.5": "15%", "claude-opus-4.7": "25%"},
            nr_forecasters=50,
        )
        rec["_cohort"] = "Q2"
        rec["was_stacked"] = True
        rec["comment_text"] = "*Forecaster 1 (gpt-5.5)*: 15%\n*Forecaster 2 (claude-opus-4.7)*: 25%\n"
        ranked = rank_our_models_by_accuracy(rec)
        per_model_reasoning = {
            "gpt-5.5": "GPT reasoned carefully.",
            "claude-opus-4.7": "Claude took a different angle.",
        }

        out_path = tmp_path / "miss_1234.md"
        emit_miss_markdown(rec, ranked, per_model_reasoning, audit_dir=tmp_path, out_path=out_path)
        content = out_path.read_text()

        assert content.startswith("# Example question")
        assert "## Per-model accuracy ranking" in content
        assert "## Per-model reasoning" in content
        assert "## Diff notes" in content
        assert "- **post_id**: 1234" in content
        assert "**cohort**: Q2" in content
        assert "**type**: binary" in content
        # was_stacked line renders True/False literally (not "unknown").
        assert "- **was_stacked**: True" in content
        # Per-model prose appears under ### headings keyed by model name.
        assert "### gpt-5.5" in content
        assert "GPT reasoned carefully." in content
        assert "### claude-opus-4.7" in content
        assert "Claude took a different angle." in content

    def test_was_stacked_unknown_when_none(self, tmp_path):
        from metaculus_bot.performance_analysis.audit import emit_miss_markdown

        rec = _binary_record(1, 0.3, True, per_model={})
        rec["was_stacked"] = None
        out_path = tmp_path / "miss.md"
        emit_miss_markdown(rec, ranked_models=[], per_model_reasoning={}, audit_dir=tmp_path, out_path=out_path)
        content = out_path.read_text()
        assert "- **was_stacked**: unknown" in content

    def test_was_stacked_false_literal(self, tmp_path):
        from metaculus_bot.performance_analysis.audit import emit_miss_markdown

        rec = _binary_record(1, 0.3, True, per_model={})
        rec["was_stacked"] = False
        out_path = tmp_path / "miss.md"
        emit_miss_markdown(rec, ranked_models=[], per_model_reasoning={}, audit_dir=tmp_path, out_path=out_path)
        content = out_path.read_text()
        assert "- **was_stacked**: False" in content

    def test_inlines_external_comment_when_present(self, tmp_path):
        from metaculus_bot.performance_analysis.audit import EXTERNAL_COMMENTS_DIRNAME, emit_miss_markdown

        rec = _binary_record(4242, 0.1, True, per_model={}, title="X")
        # Create a real external-comments file with curated content.
        ext_dir = tmp_path / EXTERNAL_COMMENTS_DIRNAME
        ext_dir.mkdir()
        curated = "Curated human comment body.\n\nWith multiple lines so it passes the >2-newline check.\n"
        (ext_dir / "4242.md").write_text(curated)

        out_path = tmp_path / "miss_4242.md"
        emit_miss_markdown(rec, ranked_models=[], per_model_reasoning={}, audit_dir=tmp_path, out_path=out_path)
        content = out_path.read_text()
        assert "## External forecaster comments" in content
        assert "Curated human comment body." in content

    def test_external_comment_skipped_when_only_placeholder(self, tmp_path):
        # The placeholder stub (short + starts with "<!-- PLACEHOLDER") should
        # NOT be inlined — it's an empty scaffold, not a curated comment.
        from metaculus_bot.performance_analysis.audit import EXTERNAL_COMMENTS_DIRNAME, emit_miss_markdown

        rec = _binary_record(9999, 0.1, True, per_model={})
        ext_dir = tmp_path / EXTERNAL_COMMENTS_DIRNAME
        ext_dir.mkdir()
        (ext_dir / "9999.md").write_text("<!-- PLACEHOLDER -->\n# foo\n")

        out_path = tmp_path / "miss_9999.md"
        emit_miss_markdown(rec, ranked_models=[], per_model_reasoning={}, audit_dir=tmp_path, out_path=out_path)
        content = out_path.read_text()
        assert "## External forecaster comments" not in content

    def test_truncation_warning_when_summary_model_absent_from_reasoning(self, tmp_path):
        # When the comment summary lists a model but the reasoning dict doesn't
        # have a corresponding entry (e.g., because Metaculus truncated the
        # comment at COMMENT_CHAR_LIMIT and dropped the rationale), a caution
        # line should be emitted so the reader knows the audit is partial.
        from metaculus_bot.performance_analysis.audit import emit_miss_markdown

        rec = _binary_record(
            55,
            0.1,
            True,
            per_model={"gpt-5.5": "20%", "claude-opus-4.7": "30%"},
        )
        rec["comment_text"] = "*Forecaster 1 (gpt-5.5)*: 20%\n*Forecaster 2 (claude-opus-4.7)*: 30%\n"
        # Only gpt-5.5 has prose; claude-opus-4.7 is missing.
        per_model_reasoning = {"gpt-5.5": "Body."}

        out_path = tmp_path / "miss_55.md"
        emit_miss_markdown(
            rec, ranked_models=[], per_model_reasoning=per_model_reasoning, audit_dir=tmp_path, out_path=out_path
        )
        content = out_path.read_text()
        assert "Reasoning missing for ['claude-opus-4.7']" in content
        assert "COMMENT_CHAR_LIMIT" in content

    def test_no_truncation_warning_when_all_summary_models_present(self, tmp_path):
        from metaculus_bot.performance_analysis.audit import emit_miss_markdown

        rec = _binary_record(
            56,
            0.1,
            True,
            per_model={"gpt-5.5": "20%"},
        )
        rec["comment_text"] = "*Forecaster 1 (gpt-5.5)*: 20%\n"
        per_model_reasoning = {"gpt-5.5": "Body."}
        out_path = tmp_path / "miss_56.md"
        emit_miss_markdown(
            rec, ranked_models=[], per_model_reasoning=per_model_reasoning, audit_dir=tmp_path, out_path=out_path
        )
        content = out_path.read_text()
        assert "Reasoning missing" not in content


class TestEmitExternalCommentStub:
    """Exercises audit.emit_external_comment_stub — the placeholder file."""

    def test_creates_placeholder_when_missing(self, tmp_path):
        from metaculus_bot.performance_analysis.audit import emit_external_comment_stub

        rec = _binary_record(321, 0.1, True, title="Pending question")
        out_path = tmp_path / "321.md"
        assert not out_path.exists()

        emit_external_comment_stub(rec, out_path)
        assert out_path.exists()
        content = out_path.read_text()
        # Stub contents: marker + header + metadata block.
        assert content.startswith("<!-- PLACEHOLDER")
        assert "# Pending question" in content
        assert "https://www.metaculus.com/questions/321/" in content
        assert "- Resolved:" in content
        assert "## Comments" in content

    def test_does_not_overwrite_existing_curated_file(self, tmp_path):
        # Early-return is load-bearing: once a human has pasted real comments,
        # subsequent audit runs MUST NOT clobber that work.
        from metaculus_bot.performance_analysis.audit import emit_external_comment_stub

        rec = _binary_record(321, 0.1, True)
        out_path = tmp_path / "321.md"
        curated = "# My curated notes\n\nReal content here.\n"
        out_path.write_text(curated)

        emit_external_comment_stub(rec, out_path)
        # File content unchanged — no overwrite.
        assert out_path.read_text() == curated


class TestEmitCombinedReport:
    """Exercises audit.emit_combined_report — the merged output file."""

    def _prepare_miss_files(self, tmp_path, post_ids: list[int]) -> tuple[list[dict], list]:
        from metaculus_bot.performance_analysis.audit import emit_miss_markdown, rank_our_models_by_accuracy

        entries: list[dict] = []
        paths: list = []
        for i, pid in enumerate(post_ids):
            rec = _binary_record(pid, 0.10 + 0.05 * i, True, title=f"Question {pid}")
            rec["_cohort"] = "Q2"
            ranked = rank_our_models_by_accuracy(rec)
            path = tmp_path / f"miss_{pid}.md"
            emit_miss_markdown(rec, ranked, per_model_reasoning={}, audit_dir=tmp_path, out_path=path)
            entries.append({"record": rec, "ranked": ranked})
            paths.append(path)
        return entries, paths

    def test_toc_anchors_match_body_anchors(self, tmp_path):
        from metaculus_bot.performance_analysis.audit import emit_combined_report, emit_synthesis

        entries, miss_paths = self._prepare_miss_files(tmp_path, [111, 222])
        synthesis_path = tmp_path / "synthesis.md"
        emit_synthesis(entries, synthesis_path)
        combined_path = tmp_path / "combined.md"
        emit_combined_report(entries, miss_paths, synthesis_path, combined_path)

        content = combined_path.read_text()

        # TOC links are #miss-<pid>; body anchors are <a id='miss-<pid>'>.
        for pid in (111, 222):
            assert f"(#miss-{pid})" in content
            assert f"<a id='miss-{pid}'></a>" in content

        # Synthesis anchor too.
        assert "(#synthesis)" in content
        assert "<a id='synthesis'></a>" in content

    def test_post_ids_appear_in_toc_and_body(self, tmp_path):
        from metaculus_bot.performance_analysis.audit import emit_combined_report, emit_synthesis

        entries, miss_paths = self._prepare_miss_files(tmp_path, [111, 222])
        synthesis_path = tmp_path / "synthesis.md"
        emit_synthesis(entries, synthesis_path)
        combined_path = tmp_path / "combined.md"
        emit_combined_report(entries, miss_paths, synthesis_path, combined_path)

        content = combined_path.read_text()
        toc_section, _, body_section = content.partition("<a id='miss-111'></a>")

        # Each post_id appears in TOC AND body sections.
        assert "111" in toc_section
        assert "222" in toc_section
        assert "Question 111" in body_section
        assert "Question 222" in body_section

    def test_ordering_matches_input_order(self, tmp_path):
        # emit_combined_report zips entries + miss_paths in input order. The
        # selection logic (not under test here) is responsible for ordering;
        # this function preserves that ordering verbatim.
        from metaculus_bot.performance_analysis.audit import emit_combined_report, emit_synthesis

        # Deliberately pass them in a specific order to confirm it's preserved.
        entries, miss_paths = self._prepare_miss_files(tmp_path, [555, 111, 333])
        synthesis_path = tmp_path / "synthesis.md"
        emit_synthesis(entries, synthesis_path)
        combined_path = tmp_path / "combined.md"
        emit_combined_report(entries, miss_paths, synthesis_path, combined_path)

        content = combined_path.read_text()
        # Check body anchor order (555 first, then 111, then 333).
        idx_555 = content.index("<a id='miss-555'></a>")
        idx_111 = content.index("<a id='miss-111'></a>")
        idx_333 = content.index("<a id='miss-333'></a>")
        assert idx_555 < idx_111 < idx_333

        # TOC order matches body order.
        toc_555 = content.index("(#miss-555)")
        toc_111 = content.index("(#miss-111)")
        toc_333 = content.index("(#miss-333)")
        assert toc_555 < toc_111 < toc_333

    def test_includes_synthesis_content(self, tmp_path):
        from metaculus_bot.performance_analysis.audit import emit_combined_report, emit_synthesis

        entries, miss_paths = self._prepare_miss_files(tmp_path, [111])
        synthesis_path = tmp_path / "synthesis.md"
        emit_synthesis(entries, synthesis_path)
        combined_path = tmp_path / "combined.md"
        emit_combined_report(entries, miss_paths, synthesis_path, combined_path)

        content = combined_path.read_text()
        # Synthesis header emitted by emit_synthesis should appear in combined.
        assert "Audit synthesis" in content


# ---------------------------------------------------------------------------
# Helpers for new (Phase 0) tests below.
# ---------------------------------------------------------------------------


def _numeric_record(
    post_id: int,
    resolution: float | str,
    per_model_percentiles: dict[str, list[Any]] | None = None,
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 100.0,
    open_lower: bool = False,
    open_upper: bool = False,
    zero_point: float | None = None,
    q_type: str = "numeric",
    numeric_log_score: float | None = 0.0,
    title: str | None = None,
) -> dict:
    return {
        "post_id": post_id,
        "type": q_type,
        "title": title or f"Q{post_id}",
        "resolution_raw": str(resolution),
        "resolution_parsed": resolution,
        "our_prob_yes": None,
        "our_forecast_values": [i / 200 for i in range(201)],
        "per_model_forecasts": {},
        "per_model_numeric_percentiles": per_model_percentiles or {},
        "scaling": {"range_min": lower_bound, "range_max": upper_bound, "zero_point": zero_point},
        "open_lower_bound": open_lower,
        "open_upper_bound": open_upper,
        "options": None,
        "comment_text": "",
        "brier_score": None,
        "log_score": None,
        "numeric_log_score": numeric_log_score,
        "mc_log_score": None,
        "metadata": {"category": None, "nr_forecasters": 0},
    }


# ---------------------------------------------------------------------------
# select_cohort: best mode (Change 2)
# ---------------------------------------------------------------------------


class TestSelectCohortBest:
    def test_best_mode_takes_highest_peer_first(self):
        from metaculus_bot.performance_analysis.audit import select_cohort

        a = _binary_record(1, 0.10, True)  # Brier 0.81 — but huge POSITIVE peer
        a["metaculus_scores"] = {"peer_score": 80.0}
        b = _binary_record(2, 0.95, True)  # Brier 0.0025, mild peer
        b["metaculus_scores"] = {"peer_score": 5.0}
        c = _binary_record(3, 0.5, True)
        c["metaculus_scores"] = {"peer_score": -5.0}
        result = select_cohort([a, b, c], mode="best", n_binary=2, n_numeric=0, n_mc=0)
        assert [r["post_id"] for r in result] == [1, 2]

    def test_best_mode_falls_back_to_lowest_brier_when_peer_missing(self):
        from metaculus_bot.performance_analysis.audit import select_cohort

        a = _binary_record(1, 0.05, True)  # Brier 0.9025 — worst hit
        b = _binary_record(2, 0.95, True)  # Brier 0.0025 — best hit
        c = _binary_record(3, 0.50, True)  # Brier 0.25
        result = select_cohort([a, b, c], mode="best", n_binary=2, n_numeric=0, n_mc=0)
        assert [r["post_id"] for r in result] == [2, 3]

    def test_best_mode_numeric_falls_back_to_highest_log_score_when_peer_missing(self):
        # _numeric_record's default has no metaculus_scores → no peer_score, so
        # _rank_key_best_logscore's fallback path (sort descending by
        # numeric_log_score) is what's exercised here. Locks the fallback.
        from metaculus_bot.performance_analysis.audit import select_cohort

        rs = [
            _numeric_record(10, resolution=50.0, numeric_log_score=-30.0),
            _numeric_record(11, resolution=50.0, numeric_log_score=20.0),
            _numeric_record(12, resolution=50.0, numeric_log_score=5.0),
        ]
        result = select_cohort(rs, mode="best", n_binary=0, n_numeric=2, n_mc=0)
        assert [r["post_id"] for r in result] == [11, 12]

    def test_best_mode_numeric_with_peer_score_takes_highest_peer(self):
        # Sibling to the fallback test above: when peer_score is populated, it
        # wins over the numeric_log_score fallback (peer is more comparable
        # across question types and accounts for difficulty).
        from metaculus_bot.performance_analysis.audit import select_cohort

        a = _numeric_record(20, resolution=50.0, numeric_log_score=-30.0)
        a["metaculus_scores"] = {"peer_score": 80.0}  # great peer despite bad raw log
        b = _numeric_record(21, resolution=50.0, numeric_log_score=20.0)
        b["metaculus_scores"] = {"peer_score": -5.0}
        c = _numeric_record(22, resolution=50.0, numeric_log_score=5.0)
        c["metaculus_scores"] = {"peer_score": 10.0}
        result = select_cohort([a, b, c], mode="best", n_binary=0, n_numeric=2, n_mc=0)
        # peer_score order desc: a(80) > c(10) > b(-5)
        assert [r["post_id"] for r in result] == [20, 22]

    def test_best_mode_extra_post_ids_appended(self):
        from metaculus_bot.performance_analysis.audit import select_cohort

        auto = _binary_record(1, 0.95, True)  # auto-picked best
        auto["metaculus_scores"] = {"peer_score": 50.0}
        weak = _binary_record(2, 0.3, True)
        weak["metaculus_scores"] = {"peer_score": -10.0}
        extra = _binary_record(3, 0.6, True)
        extra["metaculus_scores"] = {"peer_score": -2.0}
        result = select_cohort(
            [auto, weak, extra],
            mode="best",
            n_binary=1,
            n_numeric=0,
            n_mc=0,
            extra_post_ids=[3],
        )
        assert [r["post_id"] for r in result] == [1, 3]


# ---------------------------------------------------------------------------
# select_cohort: middle mode (Change 2)
# ---------------------------------------------------------------------------


class TestSelectCohortMiddle:
    def _spread_records(self) -> list[dict]:
        # 20 binary records, peer_score from -19 to 0 (one each), so the
        # 20-80 percentile band will contain peer_scores roughly -16..-3.
        records = []
        for i in range(20):
            r = _binary_record(i, 0.5, True)
            r["metaculus_scores"] = {"peer_score": -float(i)}
            records.append(r)
        return records

    def test_middle_mode_excludes_top_and_bottom_extremes(self):
        from metaculus_bot.performance_analysis.audit import select_cohort

        records = self._spread_records()
        # Middle 60% of 20 = 12 records. We sample 5 from those 12.
        # The four most-negative (worst tail, post_ids 16..19) and four most-
        # positive (best tail, post_ids 0..3) must be excluded.
        result = select_cohort(records, mode="middle", n_binary=5, n_numeric=0, n_mc=0, seed=42)
        pids = {r["post_id"] for r in result}
        forbidden_extremes = {0, 1, 2, 3, 16, 17, 18, 19}
        assert pids.isdisjoint(forbidden_extremes)
        assert len(pids) == 5

    def test_middle_mode_reproducible_under_same_seed(self):
        from metaculus_bot.performance_analysis.audit import select_cohort

        records = self._spread_records()
        a = select_cohort(records, mode="middle", n_binary=5, n_numeric=0, n_mc=0, seed=42)
        b = select_cohort(records, mode="middle", n_binary=5, n_numeric=0, n_mc=0, seed=42)
        assert [r["post_id"] for r in a] == [r["post_id"] for r in b]

    def test_middle_mode_different_seed_gives_different_sample(self):
        from metaculus_bot.performance_analysis.audit import select_cohort

        records = self._spread_records()
        a = select_cohort(records, mode="middle", n_binary=5, n_numeric=0, n_mc=0, seed=42)
        b = select_cohort(records, mode="middle", n_binary=5, n_numeric=0, n_mc=0, seed=1)
        assert [r["post_id"] for r in a] != [r["post_id"] for r in b]

    def test_middle_mode_respects_n_binary_n_numeric_n_mc(self):
        from metaculus_bot.performance_analysis.audit import select_cohort

        binaries = []
        for i in range(20):
            r = _binary_record(i, 0.5, True)
            r["metaculus_scores"] = {"peer_score": -float(i)}
            binaries.append(r)
        numerics = []
        for i in range(20):
            r = _numeric_record(100 + i, resolution=50.0, numeric_log_score=-float(i))
            r["metaculus_scores"] = {"peer_score": -float(i)}
            numerics.append(r)
        mcs = []
        for i in range(20):
            r = {
                "post_id": 200 + i,
                "type": "multiple_choice",
                "mc_log_score": -float(i),
                "brier_score": None,
                "numeric_log_score": None,
                "metaculus_scores": {"peer_score": -float(i)},
            }
            mcs.append(r)
        result = select_cohort(binaries + numerics + mcs, mode="middle", n_binary=3, n_numeric=2, n_mc=1, seed=42)
        type_counts: dict[str, int] = {}
        for r in result:
            type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1
        assert type_counts.get("binary", 0) == 3
        assert type_counts.get("numeric", 0) == 2
        assert type_counts.get("multiple_choice", 0) == 1

    def test_middle_mode_returns_empty_when_no_peer_scores(self):
        # Middle mode is peer-score-anchored — records without peer_score
        # have no "middle" to land in. The cohort should be empty (not raise).
        from metaculus_bot.performance_analysis.audit import select_cohort

        records = []
        for i in range(10):
            r = _binary_record(i, 0.5, True)
            # No metaculus_scores → no peer_score
            records.append(r)
        result = select_cohort(records, mode="middle", n_binary=5, n_numeric=0, n_mc=0, seed=42)
        assert result == []

    def test_middle_mode_handles_small_n_without_crashing(self):
        # 3 records: lower_idx = int(0.2*3) = 0, upper_idx = int(0.8*3) = 2,
        # so middle = records[0:2] (2 records). Sample 5 from a pool of 2 must
        # not raise — `min(n_binary, len(pool))` caps the request.
        from metaculus_bot.performance_analysis.audit import select_cohort

        records = []
        for i in range(3):
            r = _binary_record(i, 0.5, True)
            r["metaculus_scores"] = {"peer_score": -float(i)}
            records.append(r)
        result = select_cohort(records, mode="middle", n_binary=5, n_numeric=0, n_mc=0, seed=42)
        # Must not raise; result is bounded by the middle-band size.
        assert len(result) <= 2

    def test_middle_mode_n_larger_than_pool_returns_pool_size(self):
        # When n_binary > middle-band size, we get the whole band, no crash.
        from metaculus_bot.performance_analysis.audit import select_cohort

        records = []
        for i in range(10):
            r = _binary_record(i, 0.5, True)
            r["metaculus_scores"] = {"peer_score": -float(i)}
            records.append(r)
        # Middle 60% of 10 = 6 records. Asking for 20 should return ≤6.
        result = select_cohort(records, mode="middle", n_binary=20, n_numeric=0, n_mc=0, seed=42)
        assert 1 <= len(result) <= 6


class TestSelectCohortValidation:
    def test_invalid_mode_raises_value_error(self):
        from typing import cast

        from metaculus_bot.performance_analysis.audit import select_cohort

        # cast() to defeat the Literal type so the runtime guard is reachable
        # in the test (otherwise the type checker rejects "random" at edit time).
        bad_mode = cast(Any, "random")
        with pytest.raises(ValueError, match="Unknown mode"):
            select_cohort([], mode=bad_mode)


# ---------------------------------------------------------------------------
# rank_our_models_by_accuracy: numeric / discrete (Change 1)
# ---------------------------------------------------------------------------


class TestRankNumericPerModel:
    def test_numeric_record_ranks_truth_centered_first(self):
        # Three models against resolution=50. The model centered on truth
        # must rank first (robust by symmetry); the relative order of the
        # two off-center models depends on PCHIP construction details +
        # asymmetric log-score weighting, so we don't pin it. Instead we
        # check that all three models are present and that the truth-
        # centered one is best.
        from metaculus_bot.performance_analysis.audit import rank_our_models_by_accuracy

        per_model = {
            "tight_around_truth": [(10, 45.0), (50, 50.0), (90, 55.0)],
            "tight_around_80": [(10, 75.0), (50, 80.0), (90, 85.0)],
            "tight_around_10": [(10, 5.0), (50, 10.0), (90, 15.0)],
        }
        rec = _numeric_record(1, resolution=50.0, per_model_percentiles=per_model)
        ranked = rank_our_models_by_accuracy(rec)
        assert ranked[0]["model"] == "tight_around_truth"
        assert {r["model"] for r in ranked} == set(per_model.keys())
        # higher = better, so first beats last
        assert ranked[0]["score"] > ranked[-1]["score"]

    def test_numeric_skips_model_with_only_one_valid_percentile(self):
        from metaculus_bot.performance_analysis.audit import rank_our_models_by_accuracy

        per_model = {
            "good": [(10, 45.0), (50, 50.0), (90, 55.0)],
            "broken": [(50, 50.0)],  # only 1 valid percentile -> PCHIP raises
        }
        rec = _numeric_record(1, resolution=50.0, per_model_percentiles=per_model)
        ranked = rank_our_models_by_accuracy(rec)
        assert {r["model"] for r in ranked} == {"good"}

    def test_numeric_above_upper_bound_resolution(self):
        # PCHIP doesn't extrapolate beyond the highest provided percentile, so
        # cdf[-1] (and therefore the residual above-upper-bound mass) is set by
        # the *largest percentile label* given. wide_upper_tail provides P10/
        # P50/P75 → cdf[-1] ≈ 0.75, leaving 25% of probability mass above the
        # upper bound. narrow_upper_tail provides P10/P50/P95 → cdf[-1] ≈ 0.95
        # leaving only 5% above upper. When the question resolves
        # above_upper_bound, the model that left more residual mass up there
        # wins.
        from metaculus_bot.performance_analysis.audit import rank_our_models_by_accuracy

        per_model = {
            "narrow_upper_tail": [(10, 5.0), (50, 10.0), (95, 15.0)],
            "wide_upper_tail": [(10, 90.0), (50, 95.0), (75, 99.0)],
        }
        rec = _numeric_record(
            1,
            resolution="above_upper_bound",
            per_model_percentiles=per_model,
            open_upper=True,
            lower_bound=0.0,
            upper_bound=100.0,
        )
        ranked = rank_our_models_by_accuracy(rec)
        assert ranked[0]["model"] == "wide_upper_tail"

    def test_numeric_below_lower_bound_resolution(self):
        from metaculus_bot.performance_analysis.audit import rank_our_models_by_accuracy

        per_model = {
            "narrow_low": [(10, 0.5), (50, 1.0), (90, 5.0)],
            "narrow_high": [(10, 90.0), (50, 95.0), (90, 99.0)],
        }
        rec = _numeric_record(
            1,
            resolution="below_lower_bound",
            per_model_percentiles=per_model,
            open_lower=True,
            lower_bound=0.0,
            upper_bound=100.0,
        )
        ranked = rank_our_models_by_accuracy(rec)
        assert ranked[0]["model"] == "narrow_low"

    def test_discrete_type_is_handled(self):
        from metaculus_bot.performance_analysis.audit import rank_our_models_by_accuracy

        per_model = {
            "good": [(10, 45.0), (50, 50.0), (90, 55.0)],
            "bad": [(10, 5.0), (50, 10.0), (90, 15.0)],
        }
        rec = _numeric_record(
            1,
            resolution=50.0,
            per_model_percentiles=per_model,
            q_type="discrete",
        )
        ranked = rank_our_models_by_accuracy(rec)
        assert ranked[0]["model"] == "good"

    def test_numeric_returns_empty_when_no_per_model_percentiles(self):
        from metaculus_bot.performance_analysis.audit import rank_our_models_by_accuracy

        rec = _numeric_record(1, resolution=50.0, per_model_percentiles={})
        assert rank_our_models_by_accuracy(rec) == []

    def test_numeric_returns_empty_when_resolution_is_none(self):
        from metaculus_bot.performance_analysis.audit import rank_our_models_by_accuracy

        per_model = {"good": [(10, 45.0), (50, 50.0), (90, 55.0)]}
        rec = _numeric_record(1, resolution=50.0, per_model_percentiles=per_model)
        rec["resolution_parsed"] = None
        assert rank_our_models_by_accuracy(rec) == []

    def test_numeric_returns_empty_when_bounds_missing(self):
        from metaculus_bot.performance_analysis.audit import rank_our_models_by_accuracy

        per_model = {"good": [(10, 45.0), (50, 50.0), (90, 55.0)]}
        rec = _numeric_record(1, resolution=50.0, per_model_percentiles=per_model)
        rec["scaling"] = {"range_min": None, "range_max": None, "zero_point": None}
        assert rank_our_models_by_accuracy(rec) == []

    def test_numeric_raw_field_summarizes_percentiles(self):
        from metaculus_bot.performance_analysis.audit import rank_our_models_by_accuracy

        per_model = {"m1": [(10, 25.0), (50, 50.0), (90, 75.0)]}
        rec = _numeric_record(1, resolution=50.0, per_model_percentiles=per_model)
        ranked = rank_our_models_by_accuracy(rec)
        raw = ranked[0]["raw"]
        assert "P10" in raw
        assert "P50" in raw
        assert "P90" in raw

    def test_numeric_skips_model_when_scoring_raises(self, monkeypatch):
        # numeric_log_score may raise ValueError or ZeroDivisionError on
        # degenerate CDFs that PCHIP nonetheless accepts. The skip path at
        # audit.py's `except (ValueError, ZeroDivisionError)` block must drop
        # the offending model and continue ranking the others.
        from metaculus_bot.performance_analysis import audit

        per_model = {
            "good": [(10, 45.0), (50, 50.0), (90, 55.0)],
            "bad_score": [(10, 40.0), (50, 50.0), (90, 60.0)],
        }
        rec = _numeric_record(1, resolution=50.0, per_model_percentiles=per_model)

        original = audit.numeric_log_score
        call_state = {"call_idx": 0}

        def flaky_score(*args, **kwargs):
            # Iteration order over `per_model` is insertion-order in CPython 3.7+,
            # so "good" is scored first (call 0), "bad_score" second (call 1).
            call_state["call_idx"] += 1
            if call_state["call_idx"] == 2:
                raise ValueError("degenerate CDF")
            return original(*args, **kwargs)

        monkeypatch.setattr(audit, "numeric_log_score", flaky_score)
        ranked = audit.rank_our_models_by_accuracy(rec)
        models_ranked = [r["model"] for r in ranked]
        assert models_ranked == ["good"]
        assert "bad_score" not in models_ranked


# ---------------------------------------------------------------------------
# emit_synthesis framing parameter (Change 3)
# ---------------------------------------------------------------------------


class TestEmitSynthesisFraming:
    def _entries(self) -> list[dict]:
        from metaculus_bot.performance_analysis.audit import rank_our_models_by_accuracy

        rec = _binary_record(
            42,
            prob_yes=0.10,
            resolution=True,
            per_model={"gpt-5.5": "15%", "claude-opus-4.7": "25%"},
        )
        ranked = rank_our_models_by_accuracy(rec)
        return [{"record": rec, "ranked": ranked}]

    def test_default_framing_unchanged_when_none_passed(self, tmp_path):
        from metaculus_bot.performance_analysis.audit import emit_synthesis

        out = tmp_path / "synthesis.md"
        emit_synthesis(self._entries(), out)
        text = out.read_text()
        assert "Audit synthesis — bot misses vs per-model dissent" in text
        assert "binary worst misses" in text
        assert "times best" in text
        assert "times worst" in text
        assert "High-spread misses" in text

    def test_custom_framing_swaps_labels(self, tmp_path):
        from metaculus_bot.performance_analysis.audit import emit_synthesis

        out = tmp_path / "synthesis.md"
        framing = {
            "title": "Audit synthesis — bot best hits",
            "intro_paragraph": "For each best-scored question, which member led?",
            "cohort_name": "binary best hits",
            "tally_best_label": "times closest",
            "tally_worst_label": "times farthest",
            "delta_section_label": "Ensemble-vs-best delta (binary, sorted by widest hit margin)",
            "spread_section_label": "High-spread hits (dissenting-model signal)",
        }
        emit_synthesis(self._entries(), out, framing=framing)
        text = out.read_text()
        assert "Audit synthesis — bot best hits" in text
        assert "binary best hits" in text
        assert "times closest" in text
        assert "times farthest" in text
        assert "widest hit margin" in text
        assert "High-spread hits" in text
        # Default labels must be absent.
        assert "binary worst misses" not in text
        assert "times worst" not in text
        assert "High-spread misses" not in text

    def test_partial_framing_falls_back_to_defaults_for_unset_keys(self, tmp_path):
        from metaculus_bot.performance_analysis.audit import emit_synthesis

        out = tmp_path / "synthesis.md"
        # Override only title; everything else should default.
        emit_synthesis(self._entries(), out, framing={"title": "Custom title"})
        text = out.read_text()
        assert "Custom title" in text
        # Defaults still present.
        assert "binary worst misses" in text


# ---------------------------------------------------------------------------
# select_worst_misses backward-compat alias (Change 2)
# ---------------------------------------------------------------------------


class TestSelectWorstMissesAlias:
    def test_alias_matches_select_cohort_worst(self):
        from metaculus_bot.performance_analysis.audit import select_cohort, select_worst_misses

        records = []
        for i in range(10):
            r = _binary_record(i, 0.10 + 0.05 * i, True)
            r["metaculus_scores"] = {"peer_score": -float(i)}
            records.append(r)
        # extra_post_ids passed through identically.
        a = select_worst_misses(records, n_binary=5, n_numeric=0, n_mc=0, extra_post_ids=[9])
        b = select_cohort(records, mode="worst", n_binary=5, n_numeric=0, n_mc=0, extra_post_ids=[9])
        assert [r["post_id"] for r in a] == [r["post_id"] for r in b]
