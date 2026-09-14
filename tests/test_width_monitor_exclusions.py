"""Question exclusion: the cohort constants, the ``--exclude-qids`` parser and what ``main``
reports about a pass.

One concern end to end, because the cohorts are only worth anything if the shorthand expands, the
ids coerce to the collector's int ``question_id``, and every drop is visible in the rendered table
or the log.
"""

import json
import logging

import pytest

from metaculus_bot.performance_analysis.cohorts import (
    DEGRADED_RUN_QIDS,
    EXCLUSION_COHORTS,
    KNOWN_BUG_QIDS,
    PARTIAL_DEGRADED_QIDS,
    parse_exclude_qids,
)
from metaculus_bot.performance_analysis.width_monitor import compute_all_eras, main, render_markdown
from tests.width_monitor_fakes import _record_with_pit


class TestExcludeQids:
    """``exclude_qids`` drops named questions from every row, and says so.

    The known-pipeline-bug cohort is excluded from every other dimension of the
    residual analysis; the width monitor was the one place that still counted it.
    43746/43747 (Minions / Toy Story 5 opening-weekend gross) are both PIT-extreme
    and sit in opposite tails, so leaving them in makes the active era read mildly
    too narrow.
    """

    def test_known_bug_qids_pins_the_documented_cohort(self):
        """Membership is a deliberate, dated decision per question, so it is pinned
        here rather than left to whatever a caller happens to pass.

        43913 (WSOP bracelets) joined 2026-08-25: pre-`9f1175c` discrete max-step cap,
        with all six forecasters stating 79.5-83% on the outcome that resolved while
        the published CDF carried 20.00% on that bin — pinned at exactly 0.200000, the
        201-grid ceiling misapplied to an 11-point grid. Receipts in
        `scratch/residual_2026-08-24/dossiers/43913_dossier.md`.

        43147 and 41798 joined 2026-09-01: the same defect on pre_flip discrete
        records (34- and 12-point grids, true caps 1.0), flagged by the shipped
        `max_step_clamp_screen`. Receipts in
        `scratch/residual_2026-08-31/dim_numeric-width.md`.
        """
        assert frozenset({"43746", "43747", "43913", "43147", "41798"}) == KNOWN_BUG_QIDS

    def test_43913_drops_from_the_rows_it_was_added_for(self):
        """43913 leaves the rows, which needs the int coercion and the discrete/numeric type gate.

        The reclassification is only worth anything if the id actually matches: the collector
        writes question_id as an int, and 43913 is a discrete record.
        """
        data = [
            _record_with_pit(0.5, created_at="2026-06-11T00:00:00Z"),
            _record_with_pit(0.99, created_at="2026-06-11T00:00:00Z", question_id=43913),
        ]
        by_label = {m.label: m for m in compute_all_eras(data, exclude_qids=KNOWN_BUG_QIDS)}
        assert by_label["all"].n_pit == 1
        assert by_label["all"].n_excluded == 1

    def test_default_keeps_every_record(self):
        """Exclusion is opt-in: callers pass the set explicitly."""
        data = [
            _record_with_pit(0.5, created_at="2026-06-01T00:00:00Z"),
            _record_with_pit(0.025, created_at="2026-06-01T00:00:00Z", question_id=43746),
            _record_with_pit(0.975, created_at="2026-06-01T00:00:00Z", question_id=43747),
        ]
        by_label = {m.label: m for m in compute_all_eras(data)}
        assert by_label["widening_off (k_tail=1.0)"].n_pit == 3
        assert by_label["all"].n_pit == 3
        assert by_label["all"].n_excluded == 0

    def test_excluded_qids_drop_from_era_and_all_rows(self):
        """Integer ``question_id`` is the real dataset shape — the collector
        writes ``q["id"]`` straight through — so the match must coerce rather
        than compare an int against a string set and silently no-op."""
        data = [
            _record_with_pit(0.5, created_at="2026-06-01T00:00:00Z"),
            _record_with_pit(0.5, created_at="2026-03-01T00:00:00Z"),
            _record_with_pit(0.025, created_at="2026-06-01T00:00:00Z", question_id=43746),
            _record_with_pit(0.975, created_at="2026-06-01T00:00:00Z", question_id="43747"),
        ]
        by_label = {m.label: m for m in compute_all_eras(data, exclude_qids=KNOWN_BUG_QIDS)}
        assert by_label["widening_off (k_tail=1.0)"].n_pit == 1
        assert by_label["widening_off (k_tail=1.0)"].n_excluded == 2
        assert by_label["all"].n_pit == 2
        assert by_label["all"].n_excluded == 2
        # The untouched era is unaffected and reports no exclusions.
        assert by_label["widening_on (k_tail=1.25)"].n_pit == 1
        assert by_label["widening_on (k_tail=1.25)"].n_excluded == 0

    def test_excluded_count_surfaces_in_rendered_table(self):
        """A silent exclusion is the same failure mode as a silent degradation:
        the reader must be able to see that rows were dropped."""
        data = [
            _record_with_pit(0.5, created_at="2026-06-01T00:00:00Z"),
            _record_with_pit(0.025, created_at="2026-06-01T00:00:00Z", question_id=43746),
        ]
        md = render_markdown(compute_all_eras(data, exclude_qids=KNOWN_BUG_QIDS))
        assert "excl" in md
        # The dropped record is visible as a count, not just absent.
        assert "| 1 | 1 |" in md


class TestParseExcludeQids:
    """The ``known_bug`` shorthand COMPOSES with explicit ids.

    It used to expand only as the whole argument, so ``--exclude-qids known_bug,43800``
    produced the literal ``{"known_bug", "43800"}``: no question id matches the word, so the
    bug pair stayed in every row while the ``excl`` column reported one exclusion and made the
    run look like the shorthand had worked. That is exactly the silent-exclusion failure the
    column exists to prevent.
    """

    def test_empty_excludes_nothing(self):
        assert parse_exclude_qids("") == frozenset()
        assert parse_exclude_qids("  ,  ") == frozenset()

    def test_shorthand_alone_expands_to_the_pair(self):
        assert parse_exclude_qids("known_bug") == KNOWN_BUG_QIDS
        assert parse_exclude_qids("  known_bug  ") == KNOWN_BUG_QIDS

    def test_shorthand_mixed_with_explicit_ids_expands_and_keeps_both(self):
        assert parse_exclude_qids("known_bug,43800") == KNOWN_BUG_QIDS | {"43800"}
        assert parse_exclude_qids("43800, known_bug ,43801") == KNOWN_BUG_QIDS | {"43800", "43801"}

    def test_the_shorthand_token_never_survives_as_a_literal_id(self):
        """The word itself must not reach ``compute_all_eras`` — it matches no question id, so
        its only effect there is an exclusion the table reports and never performs."""
        assert "known_bug" not in parse_exclude_qids("known_bug,43800")

    def test_explicit_ids_alone_are_passed_through(self):
        assert parse_exclude_qids("43800,43801") == frozenset({"43800", "43801"})

    def test_the_mixed_form_actually_drops_all_three_questions(self):
        """End-to-end through the metrics, not just the parse: the shorthand's ids and the
        explicit id all leave the rows."""
        data = [
            _record_with_pit(0.5, created_at="2026-06-01T00:00:00Z"),
            _record_with_pit(0.025, created_at="2026-06-01T00:00:00Z", question_id=43746),
            _record_with_pit(0.975, created_at="2026-06-01T00:00:00Z", question_id=43747),
            _record_with_pit(0.1, created_at="2026-06-01T00:00:00Z", question_id=43800),
        ]
        by_label = {m.label: m for m in compute_all_eras(data, exclude_qids=parse_exclude_qids("known_bug,43800"))}

        assert by_label["all"].n_pit == 1
        assert by_label["all"].n_excluded == 3

    def test_the_help_text_states_that_the_shorthand_composes(self, capsys):
        """The composing behavior is only discoverable from ``--help``, and a help string that
        still described the sole-value form is what made the old bug invisible."""
        with pytest.raises(SystemExit):
            main(["--help"])

        help_text = capsys.readouterr().out
        assert "composes" in help_text
        assert "known_bug,43800" in help_text

    def test_the_help_text_names_every_cohort_shorthand(self, capsys):
        """A cohort nobody can discover from ``--help`` gets hardcoded in a round script
        instead, which is how the degraded-run ids ended up copied three times."""
        with pytest.raises(SystemExit):
            main(["--help"])

        help_text = capsys.readouterr().out
        for name in EXCLUSION_COHORTS:
            assert name in help_text

    def test_a_dataset_has_to_be_named(self, capsys):
        """``--cached`` used to default to ``scratch/coherence_2026-07-15/perf_all_tagged.json``.

        A fresh clone has no ``scratch/`` at all, and on the operator's machine that round
        directory still resolves months after it was superseded, so a bare invocation silently
        reported a stale dataset as the current one.
        """
        with pytest.raises(SystemExit) as exit_info:
            main([])

        assert exit_info.value.code == 2
        assert "--cached" in capsys.readouterr().err


class TestDegradedRunCohorts:
    """The dry-donated-key incident cohorts (2026-07-26 .. 07-28), now tracked constants.

    They were standing scoring exclusions living only in playbook prose, and three separate
    analysis rounds hardcoded private copies of the ids. Membership is a dated decision per
    question, so it is pinned here rather than left to whatever a caller retypes.
    """

    def test_degraded_run_qids_pins_the_eight_one_of_three_publishes(self):
        assert frozenset({"44870", "44871", "44872", "44873", "44874", "44875", "44876", "44877"}) == DEGRADED_RUN_QIDS

    def test_partial_degraded_qids_pins_the_three_two_of_three_publishes(self):
        assert frozenset({"44841", "44856", "44912"}) == PARTIAL_DEGRADED_QIDS

    def test_the_cohorts_are_disjoint_from_each_other_and_from_the_bug_pair(self):
        """Overlap would double-count a question in the excluded tally and make the two
        forecaster-count arms non-exclusive."""
        assert not DEGRADED_RUN_QIDS & PARTIAL_DEGRADED_QIDS
        assert not DEGRADED_RUN_QIDS & KNOWN_BUG_QIDS
        assert not PARTIAL_DEGRADED_QIDS & KNOWN_BUG_QIDS

    def test_the_ids_are_question_ids_not_the_post_ids_of_the_same_questions(self):
        """The eight questions carry post ids 44721-44728. Storing those instead would make
        every question-id-keyed join miss, and minibench POST ids 44873-44877 sit inside the
        question-id range, so a "match either id" join admits five unrelated questions."""
        post_ids = {str(pid) for pid in range(44721, 44729)}
        assert not DEGRADED_RUN_QIDS & post_ids

    def test_every_cohort_is_reachable_by_its_shorthand(self):
        assert EXCLUSION_COHORTS == {
            "known_bug": KNOWN_BUG_QIDS,
            "degraded_run": DEGRADED_RUN_QIDS,
            "partial_degraded": PARTIAL_DEGRADED_QIDS,
        }

    def test_each_shorthand_expands_and_composes(self):
        assert parse_exclude_qids("degraded_run") == DEGRADED_RUN_QIDS
        assert parse_exclude_qids("partial_degraded") == PARTIAL_DEGRADED_QIDS
        assert parse_exclude_qids("degraded_run,partial_degraded") == DEGRADED_RUN_QIDS | PARTIAL_DEGRADED_QIDS
        assert parse_exclude_qids("known_bug, degraded_run ,43800") == (KNOWN_BUG_QIDS | DEGRADED_RUN_QIDS | {"43800"})

    def test_an_unrecognized_non_numeric_token_raises_instead_of_excluding_nothing(self):
        """With one shorthand a typo was survivable; with three, ``degraded`` would drop
        nothing while the ``excl`` column read 0 — indistinguishable from a cohort whose
        questions aren't in this pull."""
        with pytest.raises(ValueError, match="neither a question id nor a cohort shorthand"):
            parse_exclude_qids("degraded")
        with pytest.raises(ValueError, match="known_bug"):
            parse_exclude_qids("43800,knownbug")
        # A fullwidth digit passes str.isdigit() and matches no question id: the no-op the guard blocks.
        fullwidth_43800 = "".join(chr(0xFF10 + int(digit)) for digit in "43800")
        with pytest.raises(ValueError, match="neither a question id"):
            parse_exclude_qids(fullwidth_43800)

    def test_a_degraded_run_question_actually_leaves_the_rows(self):
        """End-to-end through the metrics: the constant is only worth anything if the id
        matches the int question_id the collector writes."""
        data = [
            _record_with_pit(0.5, created_at="2026-08-01T00:00:00Z"),
            _record_with_pit(0.025, created_at="2026-08-01T00:00:00Z", question_id=44872),
            _record_with_pit(0.975, created_at="2026-08-01T00:00:00Z", question_id=44841),
        ]
        by_label = {
            m.label: m for m in compute_all_eras(data, exclude_qids=parse_exclude_qids("degraded_run,partial_degraded"))
        }
        assert by_label["all"].n_pit == 1
        assert by_label["all"].n_excluded == 2


class TestExcludeQidsCliReporting:
    """``main`` reports requested-vs-matched and warns only on the id-space confusion.

    A bare numeric id matching no record used to be a silent no-op — pasting the
    degraded cohort's POST ids rendered byte-identically to ``--exclude-qids ''``. The
    numeric half of the failure the shorthand raise closed stays reportable here without
    alarming on a cohort id that simply isn't in the pull.
    """

    def _write(self, tmp_path, records: list[dict]) -> str:
        path = tmp_path / "data.json"
        path.write_text(json.dumps(records))
        return str(path)

    def test_reports_requested_and_matched_counts(self, tmp_path, caplog):
        records = [
            _record_with_pit(0.5, created_at="2026-08-01T00:00:00Z", question_id=99999),
            _record_with_pit(0.5, created_at="2026-08-01T00:00:00Z", question_id=44872),
        ]
        path = self._write(tmp_path, records)
        with caplog.at_level(logging.INFO):
            main(["--cached", path, "--exclude-qids", "degraded_run"])
        # 8 requested (the whole degraded_run cohort), 1 present in this pull.
        assert any("8 requested id(s), 1 matched" in r.message for r in caplog.records)

    def test_warns_when_an_explicit_id_is_a_post_id_not_a_question_id(self, tmp_path, caplog):
        """44721 is the POST id of question 44870, a degraded_run member.

        Pasting post ids is the collision the cohort constants warn about.
        """
        record = _record_with_pit(0.5, created_at="2026-08-01T00:00:00Z", question_id=44870)
        record["post_id"] = 44721
        path = self._write(tmp_path, [record])
        with caplog.at_level(logging.WARNING):
            main(["--cached", path, "--exclude-qids", "44721"])
        assert any(
            "matched no question_id but IS a post_id" in r.message
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    def test_no_post_id_warning_on_a_correct_cohort_pass(self, tmp_path, caplog):
        record = _record_with_pit(0.5, created_at="2026-08-01T00:00:00Z", question_id=44870)
        record["post_id"] = 44721
        path = self._write(tmp_path, [record])
        with caplog.at_level(logging.WARNING):
            main(["--cached", path, "--exclude-qids", "degraded_run"])
        assert not any("IS a post_id" in r.message for r in caplog.records if r.levelno == logging.WARNING)
