"""Tests for the post-id <-> question-id mapping (metaculus_bot.performance_analysis.id_mapping).

Grounding fixture — the divergent question from a real prod run
(scratch/gha_test_bot_2026_07_19.md, https://www.metaculus.com/questions/38880):

* POST id 38880 — appears in the URL and in "Posted comment on post 38880";
  emitted by GAP_FILL_V2 / GHOST_FORECAST / GHOST_FORECAST_JSON markers.
* QUESTION id 38195 — appears in "Question: 38195" LLM-OUTPUT headers and in
  EXTRACTION_RUNG / OPEN_BOUND_PILING / CLOSE_MARGIN markers.

A single-id lookup on 38880 misses the forecaster rationales (keyed 38195); one on
38195 misses the URL-keyed records. These tests prove both id forms find the same
question in every lookup path the module offers.
"""

from __future__ import annotations

import types

from metaculus_bot.performance_analysis.id_mapping import (
    PAGE_URL_ID_PATTERN,
    QID_KIND_POST_ID,
    QID_KIND_QUESTION_ID,
    QuestionIdMap,
    QuestionIds,
    marker_records_for_question,
)
from scripts import backfill_research_from_logs
from scripts.telemetry import markers as telemetry_markers

# The 38880/38195 divergent question, and a non-divergent one (post == question).
DIVERGENT_POST_ID = 38880
DIVERGENT_QUESTION_ID = 38195
SAME_ID = 44620  # a real single-question post where post_id == question_id


def _perf_record(post_id: int, question_id: int) -> dict:
    return {"post_id": post_id, "question_id": question_id, "title": "t"}


class TestQuestionIds:
    def test_from_perf_record(self):
        ids = QuestionIds.from_perf_record(_perf_record(DIVERGENT_POST_ID, DIVERGENT_QUESTION_ID))
        assert ids.post_id == DIVERGENT_POST_ID
        assert ids.question_id == DIVERGENT_QUESTION_ID

    def test_from_question_reads_both_framework_fields(self):
        q = types.SimpleNamespace(id_of_post=DIVERGENT_POST_ID, id_of_question=DIVERGENT_QUESTION_ID)
        ids = QuestionIds.from_question(q)
        assert ids.post_id == DIVERGENT_POST_ID
        assert ids.question_id == DIVERGENT_QUESTION_ID

    def test_all_ids_divergent_returns_both(self):
        ids = QuestionIds(post_id=DIVERGENT_POST_ID, question_id=DIVERGENT_QUESTION_ID)
        assert ids.all_ids() == {DIVERGENT_POST_ID, DIVERGENT_QUESTION_ID}

    def test_all_ids_collapses_when_equal(self):
        ids = QuestionIds(post_id=SAME_ID, question_id=SAME_ID)
        assert ids.all_ids() == {SAME_ID}

    def test_id_for_kind(self):
        ids = QuestionIds(post_id=DIVERGENT_POST_ID, question_id=DIVERGENT_QUESTION_ID)
        assert ids.id_for_kind(QID_KIND_POST_ID) == DIVERGENT_POST_ID
        assert ids.id_for_kind(QID_KIND_QUESTION_ID) == DIVERGENT_QUESTION_ID
        assert ids.id_for_kind(None) is None
        assert ids.id_for_kind("bogus") is None

    def test_lookup_order_question_id_first_then_post(self):
        ids = QuestionIds(post_id=DIVERGENT_POST_ID, question_id=DIVERGENT_QUESTION_ID)
        # question_id is tried first (dominant research-archive writers key on it),
        # post_id second (log-backfill fallback).
        assert ids.lookup_order() == [DIVERGENT_QUESTION_ID, DIVERGENT_POST_ID]

    def test_lookup_order_dedups_when_ids_equal(self):
        ids = QuestionIds(post_id=SAME_ID, question_id=SAME_ID)
        assert ids.lookup_order() == [SAME_ID]

    def test_string_and_float_ids_coerced(self):
        ids = QuestionIds.from_perf_record({"post_id": "38880", "question_id": 38195.0})
        assert ids.post_id == DIVERGENT_POST_ID
        assert ids.question_id == DIVERGENT_QUESTION_ID


class TestMatchesArchiveRecord:
    """QuestionIds.matches_archive_record — rejects foreign records a filename hit would admit.

    Archive filenames mix both id spaces (question-id from live/comment writers, post-id
    from the log backfill) with no qid_kind, so a filename match alone is the blind
    "match either id" this module documents as unsafe. The validator checks the record's
    self-declared identity (explicit ``post_id`` field, or the id embedded in ``page_url``).
    """

    IDS = QuestionIds(post_id=DIVERGENT_POST_ID, question_id=DIVERGENT_QUESTION_ID)

    def test_accepts_page_url_carrying_post_id(self):
        # A genuine log-backfill record: page_url embeds OUR post id.
        record = {"qid": DIVERGENT_POST_ID, "page_url": f"https://www.metaculus.com/questions/{DIVERGENT_POST_ID}/x/"}
        assert self.IDS.matches_archive_record(record)

    def test_accepts_page_url_carrying_question_id(self):
        # A comment-backfill record: page_url is built from the QUESTION id.
        record = {"page_url": f"https://www.metaculus.com/questions/{DIVERGENT_QUESTION_ID}/"}
        assert self.IDS.matches_archive_record(record)

    def test_rejects_page_url_of_foreign_post(self):
        # The collision: our post id names a file that holds question B's record.
        record = {"qid": 99999, "page_url": "https://www.metaculus.com/questions/99999/other-question/"}
        assert not self.IDS.matches_archive_record(record)

    def test_explicit_post_id_field_is_authoritative(self):
        # Schema-v2 live records carry an explicit post_id; it decides regardless of page_url.
        ours = {"post_id": DIVERGENT_POST_ID, "page_url": "https://www.metaculus.com/questions/99999/"}
        foreign = {"post_id": 99999, "page_url": f"https://www.metaculus.com/questions/{DIVERGENT_POST_ID}/"}
        assert self.IDS.matches_archive_record(ours)
        assert not self.IDS.matches_archive_record(foreign)

    def test_on_post_field_disambiguates_comment_backfill_records(self):
        # Comment-backfill records build page_url from the QUESTION id, so a foreign one
        # whose question id equals OUR post id passes the URL check — but its explicit
        # on_post field (the foreign post's id) betrays it.
        foreign = {
            "on_post": 99999,
            "page_url": f"https://www.metaculus.com/questions/{DIVERGENT_POST_ID}/",
            "run_id": "comment-1",
        }
        ours = {
            "on_post": DIVERGENT_POST_ID,
            "page_url": f"https://www.metaculus.com/questions/{DIVERGENT_QUESTION_ID}/",
            "run_id": "comment-2",
        }
        assert not self.IDS.matches_archive_record(foreign)
        assert self.IDS.matches_archive_record(ours)

    def test_accepts_record_with_no_identity_fields(self):
        # Legacy records with neither post_id nor a parseable page_url pass (current behavior).
        assert self.IDS.matches_archive_record({"research_text": "r"})
        assert self.IDS.matches_archive_record({"page_url": "", "post_id": None})
        assert self.IDS.matches_archive_record({"page_url": "https://example.com/not-metaculus/123/"})

    def test_community_page_url_form_is_parsed(self):
        record = {"page_url": f"https://www.metaculus.com/c/some-community/{DIVERGENT_POST_ID}/slug/"}
        assert self.IDS.matches_archive_record(record)
        foreign = {"page_url": "https://www.metaculus.com/c/some-community/99999/slug/"}
        assert not self.IDS.matches_archive_record(foreign)

    def test_page_url_pattern_pinned_to_log_backfill_extractor(self):
        # id_mapping duplicates the log-backfill qid regex (scripts stay package-import-free);
        # pin them equal so the two never drift apart silently.
        assert PAGE_URL_ID_PATTERN.pattern == backfill_research_from_logs.QID_PATTERN.pattern


class TestQuestionIdMap:
    def test_resolve_both_id_forms_find_same_question(self):
        # THE headline: a lookup by EITHER id resolves to the same (post, question) pair.
        m = QuestionIdMap.from_perf_records([_perf_record(DIVERGENT_POST_ID, DIVERGENT_QUESTION_ID)])
        by_post = m.resolve(DIVERGENT_POST_ID, QID_KIND_POST_ID)
        by_question = m.resolve(DIVERGENT_QUESTION_ID, QID_KIND_QUESTION_ID)
        assert by_post == by_question
        assert by_post == QuestionIds(post_id=DIVERGENT_POST_ID, question_id=DIVERGENT_QUESTION_ID)

    def test_resolve_with_kind_is_collision_proof(self):
        # Two questions where one's post_id equals the other's question_id (the exact
        # namespace collision a blind "match either id" would get wrong).
        q_a = _perf_record(post_id=100, question_id=200)
        q_b = _perf_record(post_id=200, question_id=300)  # q_b.post_id == q_a.question_id
        m = QuestionIdMap.from_perf_records([q_a, q_b])
        # 200 as a QUESTION id is q_a; 200 as a POST id is q_b — kind disambiguates.
        assert m.resolve(200, QID_KIND_QUESTION_ID) == QuestionIds(100, 200)
        assert m.resolve(200, QID_KIND_POST_ID) == QuestionIds(200, 300)

    def test_resolve_marker_record_uses_qid_kind(self):
        m = QuestionIdMap.from_perf_records([_perf_record(DIVERGENT_POST_ID, DIVERGENT_QUESTION_ID)])
        # An EXTRACTION_RUNG record (question-id space) carrying the QUESTION id.
        extraction = {"marker": "extraction_rung", "qid": DIVERGENT_QUESTION_ID, "qid_kind": QID_KIND_QUESTION_ID}
        # A GAP_FILL_V2 record (post-id space) carrying the POST id.
        gap_fill = {"marker": "gap_fill_v2", "qid": DIVERGENT_POST_ID, "qid_kind": QID_KIND_POST_ID}
        assert m.resolve_marker_record(extraction) == m.resolve_marker_record(gap_fill)

    def test_resolve_no_kind_falls_back_question_then_post(self):
        m = QuestionIdMap.from_perf_records([_perf_record(DIVERGENT_POST_ID, DIVERGENT_QUESTION_ID)])
        assert m.resolve(DIVERGENT_QUESTION_ID) == QuestionIds(DIVERGENT_POST_ID, DIVERGENT_QUESTION_ID)
        assert m.resolve(DIVERGENT_POST_ID) == QuestionIds(DIVERGENT_POST_ID, DIVERGENT_QUESTION_ID)

    def test_resolve_missing_returns_none(self):
        m = QuestionIdMap.from_perf_records([_perf_record(1, 2)])
        assert m.resolve(999, QID_KIND_POST_ID) is None
        assert m.resolve_marker_record({"qid": None}) is None


class TestMarkerRecordsForQuestion:
    def test_collects_both_id_spaces_for_divergent_question(self):
        ids = QuestionIds(post_id=DIVERGENT_POST_ID, question_id=DIVERGENT_QUESTION_ID)
        records = [
            # question-id-keyed markers
            {"marker": "extraction_rung", "qid": DIVERGENT_QUESTION_ID, "qid_kind": QID_KIND_QUESTION_ID},
            {"marker": "close_margin", "qid": DIVERGENT_QUESTION_ID, "qid_kind": QID_KIND_QUESTION_ID},
            # post-id-keyed markers
            {"marker": "gap_fill_v2", "qid": DIVERGENT_POST_ID, "qid_kind": QID_KIND_POST_ID},
            {"marker": "ghost_forecast", "qid": DIVERGENT_POST_ID, "qid_kind": QID_KIND_POST_ID},
            # a different question's records — must NOT be collected
            {"marker": "extraction_rung", "qid": 99999, "qid_kind": QID_KIND_QUESTION_ID},
        ]
        found = marker_records_for_question(records, ids)
        assert len(found) == 4
        assert {r["marker"] for r in found} == {"extraction_rung", "close_margin", "gap_fill_v2", "ghost_forecast"}

    def test_rejects_cross_question_namespace_collision(self):
        # ids for question A (post 100 / question 200). A GAP_FILL_V2 record whose POST
        # qid is 200 belongs to a DIFFERENT question (post 200) — it must NOT be matched
        # to A just because 200 == A.question_id. This is the false match qid_kind blocks.
        ids = QuestionIds(post_id=100, question_id=200)
        records = [{"marker": "gap_fill_v2", "qid": 200, "qid_kind": QID_KIND_POST_ID}]
        assert marker_records_for_question(records, ids) == []

    def test_legacy_record_without_qid_kind_falls_back_to_either_id(self):
        # Archived-before-qid_kind records match on either id (best-effort).
        ids = QuestionIds(post_id=DIVERGENT_POST_ID, question_id=DIVERGENT_QUESTION_ID)
        records = [
            {"marker": "extraction_rung", "qid": DIVERGENT_QUESTION_ID},  # no qid_kind
            {"marker": "gap_fill_v2", "qid": DIVERGENT_POST_ID},  # no qid_kind
        ]
        assert len(marker_records_for_question(records, ids)) == 2


class TestQidKindConstantsMatchMarkers:
    """The telemetry parser keeps LOCAL copies of the qid_kind literals (to stay
    stdlib-pure). Pin them equal to the canonical ones here so the two never drift —
    a drift would silently re-open the split-brain the qid_kind field closes.
    """

    def test_qid_kind_constants_match_markers(self):
        assert telemetry_markers.QID_KIND_POST_ID == QID_KIND_POST_ID
        assert telemetry_markers.QID_KIND_QUESTION_ID == QID_KIND_QUESTION_ID

    def test_every_question_bearing_marker_declares_a_qid_kind(self):
        # Guards the invariant that closes the split-brain: any marker that carries a
        # ``question`` capture group MUST declare qid_kind, else its harvested records
        # would have no id space and be un-joinable. Credit markers (no question ref)
        # correctly have qid_kind=None.
        for spec in telemetry_markers.MARKER_SPECS:
            has_question_group = "question" in spec.regex.groupindex
            if has_question_group:
                assert spec.qid_kind in (QID_KIND_POST_ID, QID_KIND_QUESTION_ID), (
                    f"{spec.name} captures a question ref but declares qid_kind={spec.qid_kind!r}"
                )
            else:
                assert spec.qid_kind is None, f"{spec.name} has no question ref but declares qid_kind={spec.qid_kind!r}"
