"""Guards on the research archive's merge stage: which record wins, and what a rebuild keeps.

Every class here locks behavior that was measurably WRONG on the live archive:

1. ``latest/<qid>.json`` was chosen by a raw-string timestamp sort, and a question's
   comment is published minutes AFTER its research runs, so the lossy comment
   reconstruction beat the authoritative GHA artifact on 255 of the 256 questions that
   had both — leaving ``latest/`` at 25 artifact / 989 backfill.
2. Ranking by source class exposed a THIRD writer hiding inside the artifact class:
   ``backfill_research_from_logs`` keys ``qid`` on the POST id, so promoting it made
   ``latest/43592`` serve question 43591's research (and ``latest/43602`` serve 43599's),
   with a post_id healed off a sibling forging the id that made the reader accept it.
3. ``--skip-download`` rebuilt from the backfill dir alone while ``build_archive``
   overwrote ``by_qid/`` wholesale. Since artifact records live nowhere else, one
   invocation took the live archive from 280 artifact records to 27, unrecoverably
   without a re-download.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from metaculus_bot.performance_analysis.id_mapping import QuestionIds
from scripts.download_research import (
    SOURCE_ARTIFACT,
    SOURCE_COMMENT_BACKFILL,
    SOURCE_LOG_BACKFILL,
    _merge_latest,
    build_archive,
    deduplicate_records,
    guard_against_truncation,
    load_existing_by_qid,
    record_precedence_key,
    record_source,
)
from scripts.download_research import main as download_research_main
from scripts.research_sync.verify_completeness import (
    classify_gap_or_empty,
    unpersisted_artifacts,
    unpromoted_artifact_questions,
)
from scripts.sync_all import _build_research_archive

# The real qid-44220 pair that exposed the bug: the artifact ran at 13:13 and the comment
# published at 13:16, a 3m56s margin the plain timestamp sort resolved the wrong way.
ARTIFACT_RUN_ID = "28519934396"
ARTIFACT_TIMESTAMP = "2026-07-01T13:13:00.264950+00:00"
BACKFILL_RUN_ID = "comment-926248"
BACKFILL_TIMESTAMP = "2026-07-01T13:16:56.332333Z"
QID = 44220
POST_ID = 44208


def _artifact(
    qid: int = QID, run_id: str = ARTIFACT_RUN_ID, timestamp: str = ARTIFACT_TIMESTAMP, **extra: object
) -> dict:
    """A GHA-artifact-sourced record: numeric run_id, no ``on_post``, often no ``post_id``.

    ``qid`` is the QUESTION id while ``page_url`` is the POST url — the real qid-44220
    record is exactly this shape (``.../questions/44208``), and the two ids diverge on 984
    of 1,006 questions.
    """
    return {
        "qid": qid,
        "run_id": run_id,
        "timestamp": timestamp,
        "page_url": f"https://www.metaculus.com/questions/{POST_ID}",
        "question_text": "Will the thing happen?",
        "run_mode": "tournament",
        "tournament_id": "summer-futureeval-2026",
        "research_text": "## News Articles (AskNews)\nThe authoritative capture.",
        "providers_used": ["asknews", "native_search"],
        **extra,
    }


def _backfill(
    qid: int = QID, run_id: str = BACKFILL_RUN_ID, timestamp: str = BACKFILL_TIMESTAMP, **extra: object
) -> dict:
    """A comment-backfilled record: ``comment-<id>`` run_id, ``on_post`` from the comments API.

    Its ``page_url`` is the QUESTION url (990 of 990 on the live archive), unlike the two
    writers whose ``page_url`` is the post url.
    """
    return {
        "qid": qid,
        "run_id": run_id,
        "timestamp": timestamp,
        "on_post": POST_ID,
        "page_url": f"https://www.metaculus.com/questions/{qid}/",
        "research_text": "### News Articles (AskNews)\nThe trimmed reconstruction.",
        "providers_used": ["asknews"],
        **extra,
    }


def _log_backfill(qid: int, run_id: str, timestamp: str = "2026-05-19T11:20:34Z", **extra: object) -> dict:
    """A record parsed out of a run LOG by ``scripts/backfill_research_from_logs.py``.

    Two properties make this writer dangerous and both are reproduced here: its ``qid`` is
    the POST id (parsed from its own ``page_url``), and it mimics live capture closely
    enough that only ``run_mode`` plus the two empty fields tell them apart.
    """
    return {
        "schema_version": 1,
        "qid": qid,
        "run_id": run_id,
        "timestamp": timestamp,
        "page_url": f"https://www.metaculus.com/questions/{qid}/",
        "question_text": "",
        "run_mode": "backfill_from_logs",
        "tournament_id": "",
        "research_text": "## News Articles (AskNews)\nResearch for the POST, untrimmed.",
        "providers_used": ["asknews"],
        **extra,
    }


def _latest(archive_dir: Path, qid: int = QID) -> dict:
    return json.loads((archive_dir / "latest" / f"{qid}.json").read_text())


def _by_qid(archive_dir: Path, qid: int = QID) -> list[dict]:
    lines = (archive_dir / "by_qid" / f"{qid}.jsonl").read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


def _write_by_qid(archive_dir: Path, records: list[dict], qid: int = QID) -> None:
    """Seed ``by_qid/<qid>.jsonl`` as a previous ``build_archive`` would have left it."""
    by_qid_dir = archive_dir / "by_qid"
    by_qid_dir.mkdir(parents=True, exist_ok=True)
    with open(by_qid_dir / f"{qid}.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _write_backfill_dir(backfill_dir: Path, records: list[dict]) -> None:
    backfill_dir.mkdir(parents=True, exist_ok=True)
    with open(backfill_dir / "comments_backfill.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


class TestRecordSource:
    def test_the_three_writers_are_told_apart(self) -> None:
        assert record_source(_artifact()) == SOURCE_ARTIFACT
        assert record_source(_backfill()) == SOURCE_COMMENT_BACKFILL
        assert record_source(_log_backfill(43592, "26093553404")) == SOURCE_LOG_BACKFILL

    def test_legacy_log_backfill_records_are_recognized_without_the_run_mode_marker(self) -> None:
        # The 11 records already on disk predate the honest run_mode and claim
        # ``run_mode="tournament"``, so they are identified by the two fields this writer
        # leaves empty. Without this rung they classify as artifacts and outrank the
        # question-keyed record they share a by_qid group with.
        legacy = _log_backfill(43592, "26093553404", run_mode="tournament")
        assert record_source(legacy) == SOURCE_LOG_BACKFILL

    def test_a_live_capture_is_not_mistaken_for_a_log_backfill(self) -> None:
        # The fingerprint's negative control: live capture always names a tournament and
        # always carries question_text, so neither empty-field test can fire on it.
        assert record_source(_artifact()) == SOURCE_ARTIFACT
        assert record_source(_artifact(question_text="")) == SOURCE_ARTIFACT
        assert record_source(_artifact(tournament_id="")) == SOURCE_ARTIFACT

    def test_unmarked_records_default_to_artifact_class(self) -> None:
        # A local (non-GHA) run still writes real research, so it is artifact-class, and a
        # record with no run_id at all is not silently demoted.
        assert record_source({"run_id": "local"}) == SOURCE_ARTIFACT
        assert record_source({}) == SOURCE_ARTIFACT


class TestLogBackfillNeverDisplacesAQuestionKeyedRecord:
    """The keyspace collision, reproduced from the three real groups it corrupted.

    ``backfill_research_from_logs`` keys ``qid`` on the POST id parsed from the page URL,
    while live capture and comment backfill key on the QUESTION id. Post and question ids
    share one integer space, so ``by_qid/43592.jsonl`` legitimately holds BOTH the research
    for post 43592 (which wraps question 43591) and the research for question 43592 (whose
    post is 43593). ``latest/43592.json`` is read question-id-first, so the question-keyed
    record has to win — ranking log backfill with the artifacts served question 43591's
    research to question 43592, and question 43599's to question 43602.
    """

    def test_comment_record_wins_a_group_the_log_backfill_only_shares_by_integer(self, tmp_path: Path) -> None:
        # The real 43592 shape: the log record is NEWER by nothing that matters and would
        # win on any artifact-class ranking.
        log_record = _log_backfill(43592, "26093553404", timestamp="2026-05-19T11:20:34Z")
        comment_record = _backfill(qid=43592, run_id="comment-855787", timestamp="2026-05-19T13:58:49.658926Z")
        comment_record["on_post"] = 43593
        build_archive([log_record, comment_record], tmp_path)

        latest = _latest(tmp_path, qid=43592)
        assert latest["run_id"] == "comment-855787"
        assert latest["source"] == SOURCE_COMMENT_BACKFILL

    def test_a_log_backfill_record_still_wins_when_it_is_the_only_one(self, tmp_path: Path) -> None:
        # Last place is not exclusion: 8 of the 11 are the sole record for their key, and
        # the post-id lookup rung in backtest.py is how a question reaches them.
        build_archive([_log_backfill(43613, "26176160830")], tmp_path)

        latest = _latest(tmp_path, qid=43613)
        assert latest["run_id"] == "26176160830"
        assert latest["source"] == SOURCE_LOG_BACKFILL

    def test_a_genuine_artifact_still_outranks_a_log_backfill(self, tmp_path: Path) -> None:
        build_archive([_log_backfill(QID, "26093553404"), _artifact()], tmp_path)
        assert _latest(tmp_path)["source"] == SOURCE_ARTIFACT

    def test_the_winner_is_not_stamped_with_a_foreign_post_id(self) -> None:
        # The forgery, asserted directly against _merge_latest rather than through
        # build_archive: precedence alone keeps the comment record on top in this group, so
        # going through the archive would pass even if the heal were still borrowing from
        # siblings. If a log-backfill winner took the comment sibling's on_post (43593),
        # matches_archive_record would ACCEPT it for question 43592 and replay post 43592's
        # research — question 43591's — as 43592's.
        log_record = _log_backfill(43592, "26093553404", timestamp="2026-06-01T00:00:00Z")
        comment_record = _backfill(qid=43592, run_id="comment-855787", timestamp="2026-05-19T13:58:49Z")
        comment_record["on_post"] = 43593

        winner = _merge_latest([log_record, comment_record])
        assert winner["post_id"] == 43592, "a log-backfill record's post id is the one in its own URL"
        assert winner["post_id"] != 43593

    def test_a_foreign_log_backfill_record_is_rejected_by_the_archive_reader(self) -> None:
        # The end-to-end consequence, at the seam that matters: backtest replay. Question
        # 43592 lives in post 43593; the record filed under 43592 is post 43592's. With an
        # honest post_id the reader rejects it and the question runs live research instead
        # of silently replaying another question's.
        log_record = _log_backfill(43592, "26093553404")
        winner = _merge_latest([log_record])
        ids = QuestionIds(post_id=43593, question_id=43592)

        assert not ids.matches_archive_record(winner)


class TestDedupAgreesWithPrecedence:
    """Dedup and the merge stage must rank records the SAME way, or dedup decides alone.

    ``deduplicate_records`` collapses a (qid, run_id) collision before ``build_archive`` ever
    sorts, so a loser it picks is gone — precedence never sees the record at all. It used to
    compare timestamps as RAW STRINGS while ``record_precedence_key`` parses to datetimes and
    ranks the source class first, and the archive's two real timestamp formats invert under a
    string compare: at the same second the bare ``...Z`` form (the log-backfill writer's)
    outranks ``...+00:00`` with fractional seconds (live capture's), because ``Z`` (0x5A) beats
    ``.`` (0x2E). So the log-backfill record survived and the artifact — the authoritative
    capture, and the question-keyed one — was discarded, in EITHER arrival order.
    """

    COLLIDING_RUN_ID = "26093553404"

    def _colliding_pair(self) -> tuple[dict, dict]:
        artifact = _artifact(qid=43592, run_id=self.COLLIDING_RUN_ID, timestamp="2026-05-19T11:20:34.500000+00:00")
        log_record = _log_backfill(43592, self.COLLIDING_RUN_ID, timestamp="2026-05-19T11:20:34Z")
        assert log_record["timestamp"] > artifact["timestamp"], "precondition: lexicographic order is inverted here"
        return artifact, log_record

    @pytest.mark.parametrize("log_first", [False, True], ids=["artifact-first", "log-backfill-first"])
    def test_the_artifact_survives_dedup_in_either_arrival_order(self, log_first: bool) -> None:
        artifact, log_record = self._colliding_pair()
        records = [log_record, artifact] if log_first else [artifact, log_record]

        deduped = deduplicate_records(records)

        assert len(deduped) == 1
        assert record_source(deduped[0]) == SOURCE_ARTIFACT

    def test_the_surviving_record_is_the_one_precedence_would_have_picked(self) -> None:
        # The property that makes the two stages consistent, stated directly: whatever dedup
        # keeps is the max under the merge stage's own ordering.
        artifact, log_record = self._colliding_pair()
        deduped = deduplicate_records([log_record, artifact])

        assert deduped[0] is max([artifact, log_record], key=record_precedence_key)

    def test_dedup_still_keeps_the_newest_of_two_same_class_records(self, tmp_path: Path) -> None:
        # The ordinary case dedup exists for: one run re-uploaded, newest text wins. Parsed
        # timestamps, so the two formats compare as instants rather than as strings.
        older = _artifact(run_id="run-1", timestamp="2026-05-20T10:00:00.100000+00:00", research_text="old")
        newer = _artifact(run_id="run-1", timestamp="2026-05-20T12:00:00Z", research_text="newer")

        deduped = deduplicate_records([newer, older])

        assert len(deduped) == 1
        assert deduped[0]["research_text"] == "newer"

    def test_distinct_run_ids_are_not_collapsed(self) -> None:
        records = [_artifact(run_id="run-1"), _artifact(run_id="run-2"), _artifact(qid=50001, run_id="run-1")]
        assert len(deduplicate_records(records)) == 3

    def test_records_without_a_qid_are_dropped(self) -> None:
        assert deduplicate_records([{"run_id": "run-1", "timestamp": ARTIFACT_TIMESTAMP}]) == []


class TestLatestPrecedence:
    """Source class outranks the timestamp; newest wins only WITHIN a class."""

    def test_artifact_beats_a_later_comment_backfill(self, tmp_path: Path) -> None:
        # The exact failure: the backfill is 3m56s NEWER and still must lose.
        build_archive([_artifact(), _backfill()], tmp_path)

        latest = _latest(tmp_path)
        assert latest["run_id"] == ARTIFACT_RUN_ID
        assert latest["source"] == SOURCE_ARTIFACT
        assert latest["research_text"] == "## News Articles (AskNews)\nThe authoritative capture."

    def test_manifest_records_the_winning_source_and_every_source_seen(self, tmp_path: Path) -> None:
        # So a consumer can filter 1,014 questions without opening 1,014 files.
        build_archive([_artifact(), _backfill()], tmp_path)

        entry = json.loads((tmp_path / "manifest.json").read_text())[str(QID)]
        assert entry["latest_source"] == SOURCE_ARTIFACT
        assert entry["sources"] == [SOURCE_ARTIFACT, SOURCE_COMMENT_BACKFILL]
        assert entry["latest_timestamp"] == ARTIFACT_TIMESTAMP
        assert entry["versions_count"] == 2

    def test_input_order_does_not_decide_the_winner(self, tmp_path: Path) -> None:
        # Backfill first in the list — the sort, not the arrival order, must settle it.
        build_archive([_backfill(), _artifact()], tmp_path)
        assert _latest(tmp_path)["run_id"] == ARTIFACT_RUN_ID

    def test_newest_wins_between_two_artifacts(self, tmp_path: Path) -> None:
        # A re-forecast's research is the research behind the newer published forecast.
        older = _artifact(run_id="28000000000", timestamp="2026-06-30T10:00:00+00:00")
        newer = _artifact(run_id="28519934396", timestamp="2026-07-01T13:13:00+00:00")
        build_archive([newer, older], tmp_path)

        assert _latest(tmp_path)["run_id"] == "28519934396"

    def test_newest_wins_between_two_backfills(self, tmp_path: Path) -> None:
        older = _backfill(run_id="comment-900000", timestamp="2026-06-30T10:00:00Z")
        newer = _backfill(run_id="comment-926248", timestamp="2026-07-01T13:16:56Z")
        build_archive([older, newer], tmp_path)

        latest = _latest(tmp_path)
        assert latest["run_id"] == "comment-926248"
        assert latest["source"] == SOURCE_COMMENT_BACKFILL

    def test_bare_z_and_fractional_offset_forms_sort_chronologically(self, tmp_path: Path) -> None:
        # The archive's two real formats: 11 records are bare `...Z` (the log-backfilled
        # 2026-05 era) and 269 are `...+00:00` WITH fractional seconds. At the same second
        # the strings first differ at index 19, 'Z' (0x5A) vs '.' (0x2E), so a string sort
        # ranks the bare form higher while it is the EARLIER instant. Both records are
        # artifact-class, so only the parsed timestamp can break the tie.
        bare_z = _artifact(run_id="28000000001", timestamp="2026-07-01T13:13:00Z")
        fractional = _artifact(run_id="28000000002", timestamp="2026-07-01T13:13:00.264950+00:00")
        assert bare_z["timestamp"] > fractional["timestamp"], "precondition: lexicographic order is inverted here"

        build_archive([bare_z, fractional], tmp_path)
        assert _latest(tmp_path)["run_id"] == "28000000002"

    def test_a_non_utc_offset_is_compared_as_an_instant(self, tmp_path: Path) -> None:
        # Future-proofing rather than a shape on disk today: every record is currently
        # UTC. A writer on a non-UTC clock would invert a string sort by a full 5 hours
        # ('Z' > '-'), which is the version of this trap that silently picks stale research.
        utc = _artifact(run_id="28000000003", timestamp="2026-07-01T13:00:00Z")
        five_hours_later = _artifact(run_id="28000000004", timestamp="2026-07-01T13:00:00-05:00")
        assert utc["timestamp"] > five_hours_later["timestamp"], "precondition: lexicographic order is inverted here"

        build_archive([utc, five_hours_later], tmp_path)
        assert _latest(tmp_path)["run_id"] == "28000000004"

    def test_naive_timestamp_is_read_as_utc(self, tmp_path: Path) -> None:
        # Comparing a naive datetime against an aware one raises TypeError, which would
        # crash the whole sort rather than mis-order one question.
        naive = _artifact(run_id="28000000005", timestamp="2026-07-01T15:00:00")
        build_archive([naive, _artifact()], tmp_path)
        assert _latest(tmp_path)["run_id"] == "28000000005"

    def test_unparseable_timestamp_loses_to_a_real_one(self, tmp_path: Path) -> None:
        build_archive([_artifact(run_id="28000000003", timestamp="not-a-timestamp"), _artifact()], tmp_path)
        assert _latest(tmp_path)["run_id"] == ARTIFACT_RUN_ID

    def test_single_backfill_record_still_wins_its_own_question(self, tmp_path: Path) -> None:
        # The ~734 older questions for which no artifact was ever written.
        build_archive([_backfill(qid=90000)], tmp_path)

        latest = _latest(tmp_path, qid=90000)
        assert latest["source"] == SOURCE_COMMENT_BACKFILL


class TestPostIdHealing:
    """Promoting an artifact must not drop its post-id evidence, or invent the wrong one.

    229 of 280 artifact records predate the ``post_id`` field, so without healing the
    promotion would leave 255 questions with no explicit post id at all. The heal reads the
    winner's OWN post url rather than a sibling's ``on_post`` — see
    ``TestLogBackfillNeverDisplacesAQuestionKeyedRecord`` for the corruption the sibling
    version caused.
    """

    def test_missing_post_id_recovered_from_the_records_own_post_url(self, tmp_path: Path) -> None:
        build_archive([_artifact(), _backfill()], tmp_path)

        latest = _latest(tmp_path)
        assert latest["run_id"] == ARTIFACT_RUN_ID
        assert latest["post_id"] == POST_ID

    def test_heal_works_with_no_sibling_at_all(self, tmp_path: Path) -> None:
        # Self-healing means an artifact-only question gets its post id too, which the
        # sibling version could never do.
        build_archive([_artifact()], tmp_path)
        assert _latest(tmp_path)["post_id"] == POST_ID

    def test_healed_record_still_validates_against_the_true_question_ids(self, tmp_path: Path) -> None:
        # The healed post_id becomes AUTHORITATIVE in matches_archive_record (an explicit
        # post id short-circuits the page_url check), so a wrong heal would reject a good
        # record. Assert the promoted winner validates for the divergent 44208/44220 post.
        build_archive([_artifact(), _backfill()], tmp_path)

        ids = QuestionIds(post_id=POST_ID, question_id=QID)
        assert ids.matches_archive_record(_latest(tmp_path))

    def test_explicit_none_post_id_is_healed_too(self, tmp_path: Path) -> None:
        # The live writer always emits the key, so most artifact records carry post_id=None
        # rather than omitting it; a `key in record` test would miss every one of them.
        build_archive([_artifact(post_id=None), _backfill()], tmp_path)
        assert _latest(tmp_path)["post_id"] == POST_ID

    def test_winners_own_post_id_is_never_overwritten(self, tmp_path: Path) -> None:
        build_archive([_artifact(post_id=44208, page_url="https://www.metaculus.com/questions/99999/")], tmp_path)
        assert _latest(tmp_path)["post_id"] == 44208

    def test_a_comment_winners_post_id_is_not_read_off_its_question_url(self, tmp_path: Path) -> None:
        # A comment record's page_url is the QUESTION url, so parsing a post id out of it
        # would write the question id into post_id — a self-inflicted version of the same
        # forgery. It carries on_post already, and matches_archive_record reads that.
        record = _backfill(qid=90000)
        del record["on_post"]
        build_archive([record], tmp_path)

        assert _latest(tmp_path, qid=90000).get("post_id") is None

    def test_no_usable_url_leaves_the_field_absent(self, tmp_path: Path) -> None:
        # Nothing is invented: page_url-only validation stays the fallback.
        build_archive([_artifact(page_url="")], tmp_path)
        assert _latest(tmp_path).get("post_id") is None


class TestByQidStaysVerbatim:
    """``source`` is a latest/-and-manifest field; the stored history keeps writers' records."""

    def test_both_records_survive_with_no_injected_source_key(self, tmp_path: Path) -> None:
        build_archive([_artifact(), _backfill()], tmp_path)

        versions = _by_qid(tmp_path)
        assert {v["run_id"] for v in versions} == {ARTIFACT_RUN_ID, BACKFILL_RUN_ID}
        assert all("source" not in v for v in versions), "by_qid/ must stay byte-faithful to its writers"

    def test_by_qid_is_ordered_best_first(self, tmp_path: Path) -> None:
        build_archive([_backfill(), _artifact()], tmp_path)
        assert [v["run_id"] for v in _by_qid(tmp_path)] == [ARTIFACT_RUN_ID, BACKFILL_RUN_ID]

    def test_healing_does_not_mutate_the_input_record(self, tmp_path: Path) -> None:
        artifact = _artifact()
        build_archive([artifact, _backfill()], tmp_path)
        assert "source" not in artifact
        assert "post_id" not in artifact


class TestRebuildIsNonDestructive:
    """A rebuild re-ingests what by_qid/ already holds instead of overwriting it away."""

    def test_rebuild_only_keeps_on_disk_artifacts_and_they_still_win(self, tmp_path: Path) -> None:
        archive_dir = tmp_path / "archive"
        backfill_dir = tmp_path / "backfill"
        # The archive holds both records; the backfill dir holds only the comment one.
        _write_by_qid(archive_dir, [_artifact(), _backfill()])
        _write_backfill_dir(backfill_dir, [_backfill()])

        argv = [
            "download_research.py",
            "--rebuild-only",
            "--backfill-dir",
            str(backfill_dir),
            "--output-dir",
            str(archive_dir),
        ]
        with mock.patch("sys.argv", argv):
            download_research_main()

        assert {v["run_id"] for v in _by_qid(archive_dir)} == {ARTIFACT_RUN_ID, BACKFILL_RUN_ID}
        assert _latest(archive_dir)["run_id"] == ARTIFACT_RUN_ID
        assert _latest(archive_dir)["source"] == SOURCE_ARTIFACT

    def test_deprecated_skip_download_alias_is_equally_non_destructive(self, tmp_path: Path) -> None:
        # The old flag name is what the operator's muscle memory and older docs use; it
        # must not still mean "delete every artifact record".
        archive_dir = tmp_path / "archive"
        backfill_dir = tmp_path / "backfill"
        _write_by_qid(archive_dir, [_artifact(), _backfill()])
        _write_backfill_dir(backfill_dir, [_backfill()])

        argv = [
            "download_research.py",
            "--skip-download",
            "--backfill-dir",
            str(backfill_dir),
            "--output-dir",
            str(archive_dir),
        ]
        with mock.patch("sys.argv", argv):
            download_research_main()

        assert {v["run_id"] for v in _by_qid(archive_dir)} == {ARTIFACT_RUN_ID, BACKFILL_RUN_ID}

    def test_load_existing_by_qid_reads_every_question_file(self, tmp_path: Path) -> None:
        _write_by_qid(tmp_path, [_artifact(), _backfill()])
        _write_by_qid(tmp_path, [_backfill(qid=90000, run_id="comment-999000")], qid=90000)

        assert len(load_existing_by_qid(tmp_path)) == 3

    def test_load_existing_by_qid_tolerates_a_missing_archive(self, tmp_path: Path) -> None:
        assert load_existing_by_qid(tmp_path / "not-created-yet") == []

    def test_sync_all_build_also_re_ingests_on_disk_artifacts(self, tmp_path: Path) -> None:
        # The single-pass driver has its own copy of the merge stage, so it needs the same
        # guarantee: a sync whose download returned nothing must not wipe the archive.
        archive_dir = tmp_path / "archive"
        backfill_dir = tmp_path / "backfill"
        _write_by_qid(archive_dir, [_artifact(), _backfill()])
        _write_backfill_dir(backfill_dir, [_backfill()])

        questions, records = _build_research_archive([], backfill_dir, archive_dir)

        assert (questions, records) == (1, 2)
        assert _latest(archive_dir)["run_id"] == ARTIFACT_RUN_ID


class TestCompletenessCheckSeesTheMergeStage:
    """The operator's QA gate reported PASS all through the precedence bug.

    It only ever compared GHA run_ids against ``by_qid/``, and ``by_qid/`` was always
    right — so "captured" was verified while "promoted" never was. The manifest now carries
    both facts, and the gate reads them.
    """

    def test_a_demoted_artifact_question_is_reported(self) -> None:
        # The pre-fix state, rebuilt from the manifest's own vocabulary.
        manifest = {
            "44220": {"sources": [SOURCE_ARTIFACT, SOURCE_COMMENT_BACKFILL], "latest_source": SOURCE_COMMENT_BACKFILL},
            "90000": {"sources": [SOURCE_COMMENT_BACKFILL], "latest_source": SOURCE_COMMENT_BACKFILL},
        }
        assert unpromoted_artifact_questions(manifest) == ["44220"]

    def test_a_correctly_merged_archive_reports_nothing(self, tmp_path: Path) -> None:
        build_archive([_artifact(), _backfill(), _backfill(qid=90000, run_id="comment-999")], tmp_path)
        manifest = json.loads((tmp_path / "manifest.json").read_text())

        assert unpromoted_artifact_questions(manifest) == []

    def test_a_comment_only_question_is_never_flagged(self) -> None:
        # The ~734 questions with no artifact ever written are correctly comment-sourced,
        # so flagging them would make the gate cry wolf on two thirds of the archive.
        manifest = {"90000": {"sources": [SOURCE_COMMENT_BACKFILL], "latest_source": SOURCE_COMMENT_BACKFILL}}
        assert unpromoted_artifact_questions(manifest) == []

    def test_a_log_backfill_winner_over_a_comment_is_not_flagged(self) -> None:
        # log_backfill losing to a comment record is the CORRECT outcome (see the keyspace
        # class above), so the gate must key on the artifact class only.
        manifest = {
            "43592": {"sources": [SOURCE_COMMENT_BACKFILL, SOURCE_LOG_BACKFILL], "latest_source": SOURCE_LOG_BACKFILL}
        }
        assert unpromoted_artifact_questions(manifest) == []


class TestCompletenessCheckWatchesThePersistedStore:
    """The gate's other blind spot: an artifact GitHub has that local disk does not.

    The archive check can pass on an artifact that was never persisted — 632 of the 859
    live artifacts hold run logs and no research at all, so "not represented in the
    archive" is the ordinary case for them and says nothing about whether their bytes are
    safe. Only the store answers that, and it is the copy that survives the 90-day
    retention, so the store check has to be its own signal.
    """

    def _persisted(self, store: Path, name: str, *, research: bool) -> None:
        run_dir = store / name
        run_dir.mkdir(parents=True)
        (run_dir / "_meta.json").write_text(
            json.dumps({"artifact_id": "1", "name": name, "created_at": "2026-07-01T00:00:00Z", "run_id": "100"})
        )
        if research:
            (run_dir / "research_100.jsonl").write_text(json.dumps({"qid": 44220, "run_id": "100"}) + "\n")

    def test_a_live_artifact_absent_from_the_store_is_reported(self, tmp_path: Path) -> None:
        live = [{"name": "research-100", "run_id": 100}, {"name": "research-200", "run_id": 200}]
        self._persisted(tmp_path, "research-100", research=True)

        assert [a["name"] for a in unpersisted_artifacts(live, tmp_path)] == ["research-200"]

    def test_a_fully_persisted_set_reports_nothing(self, tmp_path: Path) -> None:
        live = [{"name": "research-100", "run_id": 100}]
        self._persisted(tmp_path, "research-100", research=True)

        assert unpersisted_artifacts(live, tmp_path) == []

    def test_an_incomplete_store_dir_counts_as_missing(self, tmp_path: Path) -> None:
        """No ``_meta.json`` means the extraction never finished, so the copy isn't trustworthy."""
        (tmp_path / "research-100").mkdir(parents=True)
        assert [a["name"] for a in unpersisted_artifacts([{"name": "research-100", "run_id": 100}], tmp_path)] == [
            "research-100"
        ]

    def test_gap_vs_empty_is_classified_off_the_store_without_downloading(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        self._persisted(store, "research-100", research=True)  # has records -> a genuine GAP
        self._persisted(store, "research-200", research=False)  # logs only -> legitimately EMPTY
        missing = [{"name": "research-100", "run_id": 100}, {"name": "research-200", "run_id": 200}]

        with mock.patch("scripts.research_sync.verify_completeness._download_artifact_to") as dl_mock:
            gaps, empties = classify_gap_or_empty(missing, "repo", store, tmp_path / "dl")

        assert dl_mock.call_count == 0, "the bytes are already on disk; re-downloading them is waste"
        assert [a["name"] for a in gaps] == ["research-100"]
        assert [a["name"] for a in empties] == ["research-200"]

    def test_an_unpersisted_artifact_still_falls_back_to_a_download(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        store.mkdir()

        def fake_download(run_id, repo, artifact_name, dest_dir):  # noqa: ANN001, ANN202
            run_dir = Path(dest_dir) / str(run_id)
            run_dir.mkdir(parents=True)
            (run_dir / "research_100.jsonl").write_text(json.dumps({"qid": 44220, "run_id": "100"}) + "\n")
            return run_dir

        with mock.patch(
            "scripts.research_sync.verify_completeness._download_artifact_to", side_effect=fake_download
        ) as dl_mock:
            gaps, empties = classify_gap_or_empty(
                [{"name": "research-100", "run_id": 100}], "repo", store, tmp_path / "dl"
            )

        assert dl_mock.call_count == 1
        assert ([a["name"] for a in gaps], empties) == (["research-100"], [])


class TestTruncationGuard:
    """Fail loud rather than degrade: losing an un-refetchable record is unrecoverable."""

    def test_raises_when_the_rebuilt_set_holds_fewer_artifacts(self, tmp_path: Path) -> None:
        _write_by_qid(tmp_path, [_artifact(), _backfill()])

        with pytest.raises(RuntimeError, match=r"would drop un-refetchable records 1 -> 0"):
            guard_against_truncation(tmp_path, [_backfill()])

    def test_log_backfill_records_are_protected_too(self, tmp_path: Path) -> None:
        # They are the LEAST refetchable class in the archive: parsed from run logs whose
        # artifacts expired long ago, so by_qid/ is their only copy anywhere. Ranking them
        # last for `latest/` is about which question they describe, not about disposability.
        _write_by_qid(tmp_path, [_log_backfill(43613, "26176160830")], qid=43613)

        with pytest.raises(RuntimeError, match=r"would drop un-refetchable records 1 -> 0"):
            guard_against_truncation(tmp_path, [])

    def test_allows_an_unchanged_or_growing_protected_set(self, tmp_path: Path) -> None:
        _write_by_qid(tmp_path, [_artifact()])

        guard_against_truncation(tmp_path, [_artifact()])
        guard_against_truncation(tmp_path, [_artifact(), _artifact(run_id="28000000009")])

    def test_swapping_a_class_for_the_other_protected_one_still_passes(self, tmp_path: Path) -> None:
        # Both sides count the same two classes, so the guard measures the protected TOTAL.
        # A per-class comparison would have to pick an order and would fire spuriously.
        _write_by_qid(tmp_path, [_artifact()])
        guard_against_truncation(tmp_path, [_log_backfill(43613, "26176160830")])

    def test_backfill_only_loss_is_not_the_guards_business(self, tmp_path: Path) -> None:
        # Comment records are re-derivable from Metaculus at will, so losing one is not a
        # data-loss event and must not block a rebuild.
        _write_by_qid(tmp_path, [_artifact(), _backfill()])
        guard_against_truncation(tmp_path, [_artifact()])

    def test_a_download_that_came_back_short_aborts_the_sync_build(self, tmp_path: Path) -> None:
        # The live hazard: sync_all proceeds as long as ANYTHING loaded, so a download that
        # failed midway used to truncate silently. Simulated by a re-ingest that returns
        # nothing (a moved/renamed by_qid, or the pre-fix code path) while the archive on
        # disk still holds the artifact the rebuild would erase.
        archive_dir = tmp_path / "archive"
        backfill_dir = tmp_path / "backfill"
        _write_by_qid(archive_dir, [_artifact(), _backfill()])
        _write_backfill_dir(backfill_dir, [_backfill()])

        with (
            mock.patch("scripts.sync_all.load_existing_by_qid", return_value=[]),
            pytest.raises(RuntimeError, match="Refusing to rebuild"),
        ):
            _build_research_archive([], backfill_dir, archive_dir)

        # And the archive is untouched: the artifact record is still there to re-merge.
        assert {v["run_id"] for v in _by_qid(archive_dir)} == {ARTIFACT_RUN_ID, BACKFILL_RUN_ID}
