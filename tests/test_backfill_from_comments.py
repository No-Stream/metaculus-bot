"""Tests for backfill_research_from_comments.py — extracting research from bot comments."""

# --- Sample comment bodies mimicking real bot output ---

SAMPLE_COMMENT_FULL = """\
# SUMMARY

*Question*: Will X happen by 2026?
*URL*: https://www.metaculus.com/questions/43613/

Brief summary of the forecast.

# RESEARCH
## Report 1 Research
## News Articles (AskNews)
Here are the relevant news articles about topic X.

Source 1: Article about X happening.
Source 2: Analysis of X trends.

---

## Web Research (Native Search)
Web search found additional context about X.

Multiple sources confirm the trend is accelerating.

---

## Web Research (Google Search via Gemini)
Gemini grounded search results here.

Key finding: the timeline is shorter than expected.

---

## Targeted Gap-Fill (second pass)
Gap 1: What is the current regulatory status?
Answer: Regulations were passed in March 2026.

# FORECASTS
## R1: Forecaster 1 Reasoning
I think the probability is 75% because...

## R2: Forecaster 2 Reasoning
I estimate 60% based on...
"""

SAMPLE_COMMENT_NO_RESEARCH = """\
# SUMMARY

*Question*: Will Y happen?

This question could not be forecasted due to errors.

# FORECASTS
No forecasts available.
"""

SAMPLE_COMMENT_TRIMMED = """\
# SUMMARY

*Question*: Will Z happen?

# RESEARCH
## Report 1 Research
## News Articles (AskNews)
Very long research text here...

[... trimmed for length]

---

## Web Research (Native Search)
More research content.

# FORECASTS
## R1: Forecaster 1
Prediction: 50%
"""

SAMPLE_COMMENT_NO_FORECASTS_MARKER = """\
# SUMMARY

*Question*: Will W happen?

# RESEARCH
## Report 1 Research
## News Articles (AskNews)
Research with no forecasts section following.

## Web Research (Native Search)
More content at the end.
"""

SAMPLE_COMMENT_MINIMAL_RESEARCH = """\
# SUMMARY

Summary text.

# RESEARCH
## Report 1 Research
## News Articles (AskNews)
Just one provider.

# FORECASTS
## R1: Reasoning
Done.
"""


class TestExtractResearchText:
    """Test extracting research text from structured bot comments."""

    def test_extracts_full_research_section(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text(SAMPLE_COMMENT_FULL)
        assert result is not None
        assert "News Articles (AskNews)" in result
        assert "Web Research (Native Search)" in result
        assert "Web Research (Google Search via Gemini)" in result
        assert "Targeted Gap-Fill (second pass)" in result
        assert "Here are the relevant news articles about topic X." in result
        assert "Regulations were passed in March 2026." in result

    def test_excludes_forecasts_section(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text(SAMPLE_COMMENT_FULL)
        assert result is not None
        assert "Forecaster 1 Reasoning" not in result
        assert "I think the probability is 75%" not in result

    def test_excludes_summary_section(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text(SAMPLE_COMMENT_FULL)
        assert result is not None
        assert "Brief summary of the forecast" not in result
        assert "*Question*: Will X happen" not in result

    def test_returns_none_when_no_research_marker(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text(SAMPLE_COMMENT_NO_RESEARCH)
        assert result is None

    def test_handles_missing_forecasts_marker(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text(SAMPLE_COMMENT_NO_FORECASTS_MARKER)
        assert result is not None
        assert "Research with no forecasts section following." in result
        assert "More content at the end." in result

    def test_strips_report_1_research_header(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text(SAMPLE_COMMENT_FULL)
        assert result is not None
        # The "## Report 1 Research" line itself should be stripped as a header
        assert not result.startswith("## Report 1 Research")

    def test_minimal_research_single_provider(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text(SAMPLE_COMMENT_MINIMAL_RESEARCH)
        assert result is not None
        assert "Just one provider." in result

    def test_empty_string_returns_none(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text("")
        assert result is None

    def test_research_marker_only_returns_none(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text("# RESEARCH\n\n# FORECASTS\n")
        assert result is None


class TestDetectTrimmed:
    """Test detection of trimmed research content."""

    def test_detects_standard_trim_marker(self):
        from scripts.backfill_research_from_comments import detect_trimmed

        assert detect_trimmed("Some text [... trimmed for length] more text") is True

    def test_detects_alternate_trim_marker(self):
        from scripts.backfill_research_from_comments import detect_trimmed

        assert detect_trimmed("Content [...trimmed for length]") is True

    def test_no_trim_marker_returns_false(self):
        from scripts.backfill_research_from_comments import detect_trimmed

        assert detect_trimmed("Normal research text without any trimming.") is False

    def test_empty_string_returns_false(self):
        from scripts.backfill_research_from_comments import detect_trimmed

        assert detect_trimmed("") is False

    def test_trimmed_in_sample_comment(self):
        from scripts.backfill_research_from_comments import detect_trimmed, extract_research_text

        research = extract_research_text(SAMPLE_COMMENT_TRIMMED)
        assert research is not None
        assert detect_trimmed(research) is True


class TestProviderDetectionFromComments:
    """Test that detect_providers works on comment-extracted research."""

    def test_detects_all_providers_in_full_comment(self):
        from scripts.backfill_research_from_comments import extract_research_text
        from scripts.backfill_research_from_logs import detect_providers

        research = extract_research_text(SAMPLE_COMMENT_FULL)
        assert research is not None
        providers = detect_providers(research)
        assert "asknews" in providers
        assert "native_search" in providers
        assert "gemini_search" in providers

    def test_detects_gap_fill_in_full_comment(self):
        from scripts.backfill_research_from_comments import extract_research_text
        from scripts.backfill_research_from_logs import detect_gap_fill

        research = extract_research_text(SAMPLE_COMMENT_FULL)
        assert research is not None
        assert detect_gap_fill(research) is True

    def test_no_gap_fill_in_minimal_comment(self):
        from scripts.backfill_research_from_comments import extract_research_text
        from scripts.backfill_research_from_logs import detect_gap_fill

        research = extract_research_text(SAMPLE_COMMENT_MINIMAL_RESEARCH)
        assert research is not None
        assert detect_gap_fill(research) is False


class TestBuildRecord:
    """Test record construction from a comment dict."""

    def test_builds_valid_record(self):
        from scripts.backfill_research_from_comments import build_record

        comment = {
            "id": 12345,
            "on_post": 99999,
            "text": SAMPLE_COMMENT_FULL,
            "created_at": "2026-05-20T15:48:05Z",
        }
        record = build_record(comment, question_id=43613)
        assert record is not None
        assert record["schema_version"] == 1
        assert record["qid"] == 43613
        assert record["on_post"] == 99999
        assert record["run_id"] == "comment-12345"
        assert record["timestamp"] == "2026-05-20T15:48:05Z"
        assert record["run_mode"] == "tournament"
        assert record["research_chars"] == len(record["research_text"])
        assert record["is_trimmed"] is False
        assert "asknews" in record["providers_used"]
        assert record["gap_fill_used"] is True
        assert record["page_url"] == "https://www.metaculus.com/questions/43613/"

    def test_builds_record_with_none_qid(self):
        from scripts.backfill_research_from_comments import build_record

        comment = {
            "id": 111,
            "on_post": 222,
            "text": SAMPLE_COMMENT_MINIMAL_RESEARCH,
            "created_at": "2026-05-01T00:00:00Z",
        }
        record = build_record(comment, question_id=None)
        assert record is not None
        assert record["qid"] is None
        assert record["page_url"] == ""

    def test_returns_none_for_no_research(self):
        from scripts.backfill_research_from_comments import build_record

        comment = {
            "id": 111,
            "on_post": 222,
            "text": SAMPLE_COMMENT_NO_RESEARCH,
            "created_at": "2026-05-01T00:00:00Z",
        }
        record = build_record(comment, question_id=43613)
        assert record is None

    def test_detects_trimmed_in_record(self):
        from scripts.backfill_research_from_comments import build_record

        comment = {
            "id": 333,
            "on_post": 444,
            "text": SAMPLE_COMMENT_TRIMMED,
            "created_at": "2026-05-15T10:00:00Z",
        }
        record = build_record(comment, question_id=55555)
        assert record is not None
        assert record["is_trimmed"] is True
