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
