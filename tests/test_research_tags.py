"""Tests for the research-archive treatment tags (performance_analysis/research_tags.py).

The tags drive treated/untreated calibration cuts, so the load-bearing behaviors are:
depth-agnostic header detection (artifact ``##`` vs comment-backfill ``###``), the
diagnostics-corroborated confidence ladder on a False anchor read, the gfv2
section-vs-transcript distinction (the section is the treatment marker; the loop banks
a transcript even when it soft-fails), and all-None — never False — for a question
with no archive record.
"""

import json
from pathlib import Path

from metaculus_bot.performance_analysis.research_tags import (
    attach_research_tags,
    research_tags_for_qid,
    research_tags_for_record,
)

DIAG_BLOCK_EMPTY = "## Provider Diagnostics\n- timeseries_anchor: empty | 0 chars | 30 ms\n"
DIAG_BLOCK_OK = "## Provider Diagnostics\n- timeseries_anchor: ok | 855 chars | 1171 ms\n"


def _record(
    text: str,
    *,
    source: str = "artifact",
    diagnostics_block: str | None = None,
    gap_fill_v2: dict | None = None,
    is_trimmed: bool | None = None,
) -> dict:
    return {
        "research_text": text,
        "source": source,
        "provider_diagnostics_block": diagnostics_block,
        "gap_fill_v2": gap_fill_v2,
        "is_trimmed": is_trimmed,
    }


class TestAnchorHeaderDetection:
    def test_artifact_header_form(self):
        tags = research_tags_for_record(_record("## Time Series Anchor\nFRED series band...\n"))
        assert tags["anchor_present"] is True
        assert tags["anchor_confidence"] == "header"

    def test_comment_backfill_reheaded_form(self):
        # Backfill records re-head every section one level deeper.
        tags = research_tags_for_record(
            _record("### Time Series Anchor\nFRED series band...\n", source="comment_backfill")
        )
        assert tags["anchor_present"] is True
        assert tags["research_source_class"] == "comment_backfill"

    def test_title_must_match_exactly(self):
        # A prose mention or an extended title is not the section header.
        text = "The Time Series Anchor fired.\n## Time Series Anchor Notes\n"
        assert research_tags_for_record(_record(text))["anchor_present"] is False


class TestAnchorConfidenceLadder:
    def test_diag_empty_confirms_absent(self):
        tags = research_tags_for_record(_record("## News Articles (AskNews)\n", diagnostics_block=DIAG_BLOCK_EMPTY))
        assert tags["anchor_present"] is False
        assert tags["anchor_confidence"] == "diag_confirms_absent"

    def test_diag_ok_with_missing_header_is_flagged(self):
        # The provider ran (diag ok) but the text lost the section (e.g. trimming):
        # treatment is genuinely unclear and must not read as a confirmed absence.
        tags = research_tags_for_record(_record("tail of a trimmed bundle", diagnostics_block=DIAG_BLOCK_OK))
        assert tags["anchor_present"] is False
        assert tags["anchor_confidence"] == "diag_ok_header_missing"

    def test_diag_block_read_from_research_text_fallback(self):
        # Older records carry the diagnostics block only inline in the text.
        tags = research_tags_for_record(_record("some research\n" + DIAG_BLOCK_EMPTY))
        assert tags["anchor_confidence"] == "diag_confirms_absent"

    def test_trimmed_record_without_diag_is_ambiguous(self):
        # Trimming keeps header + tail, so a leading section can be eaten silently.
        tags = research_tags_for_record(_record("tail only", is_trimmed=True))
        assert tags["anchor_confidence"] == "ambiguous_trimmed_no_diag"

    def test_untrimmed_record_without_diag(self):
        assert research_tags_for_record(_record("no sections"))["anchor_confidence"] == "absent_no_diag"


class TestGfv2SectionVsTranscript:
    def test_section_present_and_loop_ran(self):
        tags = research_tags_for_record(
            _record("## Agentic Research Findings\nfindings...\n", gap_fill_v2={"steps": 7})
        )
        assert tags["gfv2_present"] is True
        assert tags["gfv2_loop_ran"] is True

    def test_transcript_without_section_is_not_treatment(self):
        # The v2 driver banks a transcript even when it soft-fails and contributes no
        # section (the dry-key cohort's shape). Only the SECTION marks treatment.
        tags = research_tags_for_record(_record("## News Articles (AskNews)\n", gap_fill_v2={"steps": 0}))
        assert tags["gfv2_present"] is False
        assert tags["gfv2_loop_ran"] is True

    def test_neither(self):
        tags = research_tags_for_record(_record("## News Articles (AskNews)\n"))
        assert tags["gfv2_present"] is False
        assert tags["gfv2_loop_ran"] is False

    def test_reheaded_gfv2_section(self):
        tags = research_tags_for_record(_record("### Agentic Research Findings\n", source="comment_backfill"))
        assert tags["gfv2_present"] is True


class TestAbsentArchiveRecord:
    def test_missing_file_yields_all_none(self, tmp_path: Path):
        tags = research_tags_for_qid(99999, tmp_path)
        assert tags == {
            "anchor_present": None,
            "gfv2_present": None,
            "gfv2_loop_ran": None,
            "anchor_confidence": None,
            "research_source_class": None,
        }

    def test_none_qid_yields_all_none(self, tmp_path: Path):
        assert research_tags_for_qid(None, tmp_path)["anchor_present"] is None

    def test_unreadable_record_yields_all_none(self, tmp_path: Path):
        (tmp_path / "123.json").write_text("{not json")
        assert research_tags_for_qid(123, tmp_path)["anchor_present"] is None


class TestAttachResearchTags:
    def test_stamps_tags_by_question_id(self, tmp_path: Path):
        (tmp_path / "101.json").write_text(
            json.dumps(
                {
                    "research_text": "## Time Series Anchor\nband\n## Agentic Research Findings\nfindings\n",
                    "source": "artifact",
                    "gap_fill_v2": {"steps": 3},
                }
            )
        )
        records = [
            {"question_id": 101, "type": "numeric"},
            {"question_id": 202, "type": "binary"},  # no archive record
        ]
        attach_research_tags(records, tmp_path)
        assert records[0]["anchor_present"] is True
        assert records[0]["gfv2_present"] is True
        assert records[0]["research_source_class"] == "artifact"
        assert records[1]["anchor_present"] is None
        assert records[1]["gfv2_present"] is None
