"""Tests for the residual-round pull audit (scripts/verify_pull.py).

Every check runs offline against dict fixtures: post payloads from ``tests/supply_probe_fakes.py``
(the same shape the pull checkpoints) and records built here. Nothing opens a socket, and the CLI is
driven end to end over files in ``tmp_path`` rather than against a round directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.verify_pull import (
    VERDICT_ANNULLED,
    VERDICT_INVESTIGATE,
    VERDICT_KNOWN_RECORDLESS,
    audit_coverage,
    audit_parse_parity,
    audit_stability,
    checkpoint_path_for,
    diff_against_prior,
    main,
    recordless_post_ids,
    render_report,
)
from tests.supply_probe_fakes import _group_post, _post, _question

_PER_MODEL_FORECASTS = {"opus": 0.4}


def _checkpoint(*posts: dict) -> dict[str, dict]:
    """Post id (as a string) -> raw payload, the contract resilient_pull.py checkpoints."""
    return {str(post["id"]): post for post in posts}


def _record(
    question_id: int,
    post_id: int,
    *,
    qtype: str = "binary",
    spot: float | None = -5.0,
    peer: float | None = -4.0,
    coverage: float | None = 0.9,
    per_model: Any = _PER_MODEL_FORECASTS,
    comment_text: str | None = "rationale",
    resolution_set_time: str = "2026-09-08T00:00:00Z",
    actual_resolve_time: str = "2026-09-08T00:00:00Z",
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "post_id": post_id,
        "type": qtype,
        "title": f"Question {question_id}?",
        "bot_comment_created_at": "2026-08-01T00:00:00Z",
        "per_model_forecasts": per_model,
        "per_base_model_forecasts": None,
        "per_model_numeric_percentiles": None,
        "comment_text": comment_text,
        "metaculus_scores": {
            "spot_peer_score": spot,
            "peer_score": peer,
            "coverage": coverage,
            "baseline_score": 10.0,
            "spot_baseline_score": 11.0,
        },
        "metadata": {
            "open_time": "2026-07-01T00:00:00Z",
            "scheduled_resolve_time": "2026-09-01T00:00:00Z",
            "actual_resolve_time": actual_resolve_time,
            "resolution_set_time": resolution_set_time,
        },
    }


def _resolved_post(post_id: int, question_id: int, *, resolution: object, forecast: bool | None = True) -> dict:
    post = _post(post_id, _question(question_id, resolution=resolution, forecast=forecast))
    return post | {"status": "resolved", "comment_count": 12}


class TestRecordlessDerivation:
    def test_posts_without_a_record_are_the_recordless_set(self):
        checkpoint = _checkpoint(
            _resolved_post(100, 200, resolution="yes"),
            _resolved_post(101, 201, resolution="annulled"),
        )
        assert recordless_post_ids(checkpoint, [_record(200, 100)]) == {101}

    def test_checkpoint_path_sits_beside_the_records_file(self):
        records = Path("scratch/residual_2026-09-09/perf_summer-futureeval-2026.json")
        expected = Path("scratch/residual_2026-09-09/perf_summer-futureeval-2026_checkpoint.json")
        assert checkpoint_path_for(records) == expected


class TestCoverage:
    def test_annulled_is_benign_and_a_new_recordless_post_is_investigated(self):
        checkpoint = _checkpoint(
            _resolved_post(100, 200, resolution="yes"),
            _resolved_post(101, 201, resolution="annulled"),
            _resolved_post(102, 202, resolution=None, forecast=False),
            _resolved_post(103, 203, resolution=None, forecast=False),
        )
        coverage = audit_coverage(checkpoint, [_record(200, 100)], prior_recordless=frozenset({102}))

        verdicts = {post.post_id: post.verdict for post in coverage.recordless}
        assert verdicts == {
            101: VERDICT_ANNULLED,
            102: VERDICT_KNOWN_RECORDLESS,
            103: VERDICT_INVESTIGATE,
        }
        assert coverage.investigate == (103,)
        assert coverage.checkpoint_posts == 4
        assert coverage.records == 1
        assert coverage.record_posts == 1

    def test_a_recordless_post_carries_its_forecast_state_and_comment_count(self):
        checkpoint = _checkpoint(_resolved_post(101, 201, resolution=None, forecast=False))
        (post,) = audit_coverage(checkpoint, [], prior_recordless=frozenset()).recordless

        assert post.forecast_states == ("no_forecast",)
        assert post.comment_count == 12
        assert post.status == "resolved"
        assert post.resolutions == (None,)

    def test_group_members_resolved_without_a_record_are_named(self):
        group = _group_post(
            110,
            [
                _question(300, resolution="yes"),
                _question(301, resolution="no"),
                _question(302, resolution="annulled"),
                _question(303, resolution=None),
            ],
        )
        coverage = audit_coverage(_checkpoint(group), [_record(300, 110)], prior_recordless=frozenset())

        assert coverage.group_posts == 1
        assert coverage.recordless == ()
        assert [row.question_id for row in coverage.missing_sub_questions] == [301]

    def test_a_whole_group_post_with_no_record_is_check_ones_business(self):
        group = _group_post(110, [_question(300, resolution="yes"), _question(301, resolution="yes")])
        coverage = audit_coverage(_checkpoint(group), [], prior_recordless=frozenset())

        assert [post.post_id for post in coverage.recordless] == [110]
        assert coverage.missing_sub_questions == ()


class TestPriorDiff:
    def test_added_and_lost_records_and_their_histograms(self):
        prior = [_record(200, 100), _record(201, 101)]
        records = [
            _record(200, 100),
            _record(202, 102, qtype="numeric", resolution_set_time="2026-09-09T04:00:00Z"),
            _record(203, 103, spot=None, peer=None),
        ]
        diff = diff_against_prior(records, prior)

        assert [entry["question_id"] for entry in diff.added] == [202, 203]
        assert diff.lost == ((201, 101),)
        assert diff.added_types == {"numeric": 1, "binary": 1}
        assert diff.without_spot_peer == 1
        assert diff.resolution_set_days == {"2026-09-08": 1, "2026-09-09": 1}
        assert diff.actual_resolve_days == {"2026-09-08": 2}

    def test_an_added_entry_carries_the_fields_the_round_reads(self):
        (entry,) = diff_against_prior([_record(202, 102)], []).added

        assert entry["spot_peer_score"] == -5.0
        assert entry["peer_score"] == -4.0
        assert entry["coverage"] == 0.9
        assert entry["actual_resolve_time"] == "2026-09-08T00:00:00Z"
        assert entry["bot_comment_created_at"] == "2026-08-01T00:00:00Z"


class TestStability:
    def test_identical_scores_reproduce(self):
        records = [_record(200, 100), _record(201, 101)]
        stability = audit_stability(records, [dict(record) for record in records])

        assert stability.overlap == 2
        assert stability.expected_overlap == 2
        assert stability.mismatches == ()
        assert stability.checked["spot_peer_score"] == 2

    def test_a_re_resolved_score_is_named_per_field(self):
        prior = [_record(200, 100, spot=5.41, peer=5.0)]
        records = [_record(200, 100, spot=-5.42, peer=-5.0)]
        stability = audit_stability(records, prior)

        assert {mismatch.field for mismatch in stability.mismatches} == {"spot_peer_score", "peer_score"}
        assert stability.mismatches_by_field["spot_peer_score"] == 1
        assert stability.mismatches[0].question_id == 200

    def test_a_score_appearing_or_vanishing_is_a_mismatch_and_two_absences_are_not_checked(self):
        appeared = audit_stability([_record(200, 100, spot=1.0)], [_record(200, 100, spot=None)])
        assert [mismatch.field for mismatch in appeared.mismatches] == ["spot_peer_score"]

        absent = audit_stability([_record(200, 100, spot=None)], [_record(200, 100, spot=None)])
        assert absent.mismatches == ()
        assert absent.checked.get("spot_peer_score", 0) == 0


class TestParseParity:
    def test_a_field_lost_on_a_re_pull_is_reported_with_its_question(self):
        prior = [_record(200, 100), _record(201, 101)]
        records = [_record(200, 100, per_model=None), _record(201, 101, comment_text=None)]
        rows = {row.field: row for row in audit_parse_parity(records, prior)}

        assert rows["per_model_forecasts"].prior_nonempty == 2
        assert rows["per_model_forecasts"].now_nonempty == 1
        assert rows["per_model_forecasts"].lost == (200,)
        assert rows["comment_text"].lost == (201,)
        assert rows["per_base_model_forecasts"].prior_nonempty == 0

    def test_records_absent_from_the_prior_pull_are_not_parity_evidence(self):
        rows = {row.field: row for row in audit_parse_parity([_record(202, 102)], [_record(200, 100)])}
        assert rows["per_model_forecasts"].prior_nonempty == 0
        assert rows["per_model_forecasts"].now_nonempty == 0


class TestRendering:
    def _coverage(self, *, prior_recordless: frozenset[int]):
        checkpoint = _checkpoint(_resolved_post(103, 203, resolution=None, forecast=False))
        return audit_coverage(checkpoint, [], prior_recordless=prior_recordless)

    def test_investigate_posts_and_the_no_prior_caveat_reach_the_report(self):
        report = render_report(self._coverage(prior_recordless=frozenset()), None, None, None, prior_pull_known=False)

        assert VERDICT_INVESTIGATE in report
        assert "need investigation: [103]" in report
        assert "--prior-records" in report
        assert "=== 3." not in report

    def test_a_moved_score_prints_the_re_read_instruction(self):
        stability = audit_stability([_record(200, 100, spot=-5.42)], [_record(200, 100, spot=5.41)])
        report = render_report(self._coverage(prior_recordless=frozenset({103})), None, stability, None)

        assert "MISMATCH q200 spot_peer_score" in report
        assert "re-resolved in place" in report

    def test_the_worst_rows_preview_discloses_what_it_hid(self):
        records = [_record(200 + index, 100 + index, spot=-float(index)) for index in range(5)]
        report = render_report(
            self._coverage(prior_recordless=frozenset({103})),
            diff_against_prior(records, []),
            None,
            None,
            max_worst_rows=2,
        )

        assert "+3 more scored new records" in report


class TestCli:
    def _round_dir(self, tmp_path: Path, name: str, checkpoint: dict, records: list[dict]) -> Path:
        round_dir = tmp_path / name
        round_dir.mkdir()
        records_path = round_dir / "perf_slug.json"
        records_path.write_text(json.dumps(records))
        checkpoint_path_for(records_path).write_text(json.dumps(checkpoint))
        return records_path

    def test_the_checkpoints_are_found_beside_the_records_and_the_cohort_is_written(self, tmp_path, capsys):
        prior = self._round_dir(
            tmp_path,
            "residual_prior",
            _checkpoint(
                _resolved_post(100, 200, resolution="yes"),
                _resolved_post(102, 202, resolution=None, forecast=False),
            ),
            [_record(200, 100)],
        )
        current = self._round_dir(
            tmp_path,
            "residual_now",
            _checkpoint(
                _resolved_post(100, 200, resolution="yes"),
                _resolved_post(102, 202, resolution=None, forecast=False),
                _resolved_post(103, 203, resolution="no"),
            ),
            [_record(200, 100), _record(203, 103)],
        )
        output = tmp_path / "new_since_prior.json"

        main(["--records", str(current), "--prior-records", str(prior), "--output", str(output)])

        report = capsys.readouterr().out
        assert "need investigation: []" in report
        assert VERDICT_KNOWN_RECORDLESS in report
        assert "NEW since prior: 1" in report
        assert "TOTAL mismatches across fields: 0" in report
        assert [entry["question_id"] for entry in json.loads(output.read_text())] == [203]

    def test_without_a_prior_pull_everything_recordless_is_investigated(self, tmp_path, capsys):
        current = self._round_dir(
            tmp_path,
            "residual_now",
            _checkpoint(_resolved_post(102, 202, resolution=None, forecast=False)),
            [],
        )

        main(["--records", str(current)])

        report = capsys.readouterr().out
        assert "need investigation: [102]" in report
        assert "=== 4." not in report

    def test_a_missing_prior_checkpoint_classifies_nothing_as_known(self, tmp_path, capsys, caplog):
        prior_records = tmp_path / "prior.json"
        prior_records.write_text(json.dumps([_record(200, 100)]))
        current = self._round_dir(
            tmp_path,
            "residual_now",
            _checkpoint(_resolved_post(102, 202, resolution=None, forecast=False)),
            [_record(200, 100)],
        )

        main(["--records", str(current), "--prior-records", str(prior_records)])

        assert "will read INVESTIGATE" in caplog.text
        assert "need investigation: [102]" in capsys.readouterr().out
