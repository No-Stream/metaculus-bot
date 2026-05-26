"""Tests for the research log parser used by backfill_research_from_logs.py."""

# Sample log data for testing
SAMPLE_LOG_MAIN_LOGGER = """\
forecast_job\tUNKNOWN STEP\t2026-05-20T15:48:05.5335094Z 2026-05-20 15:48:05,533 - main - INFO - Found Research for URL https://www.metaculus.com/questions/43613:
forecast_job\tUNKNOWN STEP\t2026-05-20T15:48:05.5335906Z ## News Articles (AskNews)
forecast_job\tUNKNOWN STEP\t2026-05-20T15:48:05.5336341Z Here are the relevant news articles about topic X.
forecast_job\tUNKNOWN STEP\t2026-05-20T15:48:05.5336672Z
forecast_job\tUNKNOWN STEP\t2026-05-20T15:48:05.5336863Z ---
forecast_job\tUNKNOWN STEP\t2026-05-20T15:48:05.5337169Z
forecast_job\tUNKNOWN STEP\t2026-05-20T15:48:05.5337500Z ## Web Research (Native Search)
forecast_job\tUNKNOWN STEP\t2026-05-20T15:48:05.5337800Z Some web research content here.
forecast_job\tUNKNOWN STEP\t2026-05-20T15:50:12.1234567Z 2026-05-20 15:50:12,123 - main - INFO - Using research providers: ['asknews', 'native_search']
"""

SAMPLE_LOG_NEW_LOGGER = """\
forecast_job\tUNKNOWN STEP\t2026-05-24T10:00:01.0000000Z 2026-05-24 10:00:01,000 - metaculus_bot.research_orchestrator - INFO - Found Research for URL https://www.metaculus.com/questions/50001/will-ai-pass-turing-test/:
forecast_job\tUNKNOWN STEP\t2026-05-24T10:00:01.0000001Z ## News Articles (AskNews)
forecast_job\tUNKNOWN STEP\t2026-05-24T10:00:01.0000002Z AI progress is accelerating.
forecast_job\tUNKNOWN STEP\t2026-05-24T10:00:01.0000003Z ## Web Research (Google Search via Gemini)
forecast_job\tUNKNOWN STEP\t2026-05-24T10:00:01.0000004Z Gemini found some relevant results.
forecast_job\tUNKNOWN STEP\t2026-05-24T10:00:01.0000005Z ## Targeted Gap-Fill (second pass)
forecast_job\tUNKNOWN STEP\t2026-05-24T10:00:01.0000006Z Filling knowledge gaps about timeline.
forecast_job\tUNKNOWN STEP\t2026-05-24T10:05:00.0000000Z 2026-05-24 10:05:00,000 - metaculus_bot.research_orchestrator - INFO - Starting forecaster fan-out
"""

SAMPLE_LOG_COMMUNITY_URL = """\
forecast_job\tRun forecasts\t2026-05-18T08:30:00.0000000Z 2026-05-18 08:30:00,000 - main - INFO - Found Research for URL https://www.metaculus.com/c/diffusion-community/38880/some-question-slug/:
forecast_job\tRun forecasts\t2026-05-18T08:30:00.0000001Z ## News Articles (AskNews)
forecast_job\tRun forecasts\t2026-05-18T08:30:00.0000002Z Community question research here.
forecast_job\tRun forecasts\t2026-05-18T08:30:05.0000000Z 2026-05-18 08:30:05,000 - main - INFO - Next log message
"""

SAMPLE_LOG_MULTIPLE_QUESTIONS = """\
forecast_job\tUNKNOWN STEP\t2026-05-20T15:00:00.0000000Z 2026-05-20 15:00:00,000 - main - INFO - Found Research for URL https://www.metaculus.com/questions/11111:
forecast_job\tUNKNOWN STEP\t2026-05-20T15:00:00.0000001Z ## News Articles (AskNews)
forecast_job\tUNKNOWN STEP\t2026-05-20T15:00:00.0000002Z First question research.
forecast_job\tUNKNOWN STEP\t2026-05-20T15:05:00.0000000Z 2026-05-20 15:05:00,000 - main - INFO - Found Research for URL https://www.metaculus.com/questions/22222/second-question/:
forecast_job\tUNKNOWN STEP\t2026-05-20T15:05:00.0000001Z ## Web Research (Native Search)
forecast_job\tUNKNOWN STEP\t2026-05-20T15:05:00.0000002Z Second question research.
forecast_job\tUNKNOWN STEP\t2026-05-20T15:05:00.0000003Z ## Financial & Economic Data
forecast_job\tUNKNOWN STEP\t2026-05-20T15:05:00.0000004Z GDP data here.
forecast_job\tUNKNOWN STEP\t2026-05-20T15:10:00.0000000Z 2026-05-20 15:10:00,000 - main - INFO - Done processing questions
"""

SAMPLE_LOG_NO_RESEARCH = """\
forecast_job\tUNKNOWN STEP\t2026-05-20T15:00:00.0000000Z 2026-05-20 15:00:00,000 - main - INFO - Starting bot run
forecast_job\tUNKNOWN STEP\t2026-05-20T15:00:01.0000000Z 2026-05-20 15:00:01,000 - main - INFO - Skipping 45 previously forecasted questions
forecast_job\tUNKNOWN STEP\t2026-05-20T15:00:02.0000000Z 2026-05-20 15:00:02,000 - main - INFO - No new questions to forecast
"""


class TestResearchStartRegex:
    """Test the RESEARCH_START pattern against various log line formats."""

    def test_old_logger_format(self):
        from scripts.backfill_research_from_logs import RESEARCH_START

        line = (
            "forecast_job\tUNKNOWN STEP\t2026-05-20T15:48:05.5335094Z "
            "2026-05-20 15:48:05,533 - main - INFO - "
            "Found Research for URL https://www.metaculus.com/questions/43613:"
        )
        match = RESEARCH_START.match(line)
        assert match is not None
        assert match.group(1) == "2026-05-20T15:48:05.5335094Z"
        assert match.group(2) == "https://www.metaculus.com/questions/43613"

    def test_new_logger_format(self):
        from scripts.backfill_research_from_logs import RESEARCH_START

        line = (
            "forecast_job\tUNKNOWN STEP\t2026-05-24T10:00:01.0000000Z "
            "2026-05-24 10:00:01,000 - metaculus_bot.research_orchestrator - INFO - "
            "Found Research for URL https://www.metaculus.com/questions/50001/will-ai-pass-turing-test/:"
        )
        match = RESEARCH_START.match(line)
        assert match is not None
        assert match.group(1) == "2026-05-24T10:00:01.0000000Z"
        assert match.group(2) == "https://www.metaculus.com/questions/50001/will-ai-pass-turing-test/"

    def test_url_without_slug(self):
        from scripts.backfill_research_from_logs import RESEARCH_START

        line = (
            "forecast_job\tRun forecasts\t2026-05-20T12:00:00.0000000Z "
            "2026-05-20 12:00:00,000 - main - INFO - "
            "Found Research for URL https://www.metaculus.com/questions/99999:"
        )
        match = RESEARCH_START.match(line)
        assert match is not None
        assert match.group(2) == "https://www.metaculus.com/questions/99999"

    def test_community_question_url(self):
        from scripts.backfill_research_from_logs import RESEARCH_START

        line = (
            "forecast_job\tRun forecasts\t2026-05-18T08:30:00.0000000Z "
            "2026-05-18 08:30:00,000 - main - INFO - "
            "Found Research for URL https://www.metaculus.com/c/diffusion-community/38880/some-question-slug/:"
        )
        match = RESEARCH_START.match(line)
        assert match is not None
        assert match.group(2) == "https://www.metaculus.com/c/diffusion-community/38880/some-question-slug/"

    def test_named_step(self):
        from scripts.backfill_research_from_logs import RESEARCH_START

        line = (
            "forecast_job\tRun forecasts\t2026-05-20T12:00:00.0000000Z "
            "2026-05-20 12:00:00,000 - main - INFO - "
            "Found Research for URL https://www.metaculus.com/questions/12345:"
        )
        match = RESEARCH_START.match(line)
        assert match is not None

    def test_non_research_line_no_match(self):
        from scripts.backfill_research_from_logs import RESEARCH_START

        line = (
            "forecast_job\tUNKNOWN STEP\t2026-05-20T15:48:05.5335094Z "
            "2026-05-20 15:48:05,533 - main - INFO - Starting bot run"
        )
        assert RESEARCH_START.match(line) is None


class TestAppLogLineRegex:
    """Test the APP_LOG_LINE pattern that signals end of research."""

    def test_info_level(self):
        from scripts.backfill_research_from_logs import APP_LOG_LINE

        line = (
            "forecast_job\tUNKNOWN STEP\t2026-05-20T15:50:12.1234567Z "
            "2026-05-20 15:50:12,123 - main - INFO - Using research providers"
        )
        assert APP_LOG_LINE.match(line) is not None

    def test_warning_level(self):
        from scripts.backfill_research_from_logs import APP_LOG_LINE

        line = (
            "forecast_job\tUNKNOWN STEP\t2026-05-20T15:50:12.1234567Z "
            "2026-05-20 15:50:12,123 - main - WARNING - Something went wrong"
        )
        assert APP_LOG_LINE.match(line) is not None

    def test_error_level(self):
        from scripts.backfill_research_from_logs import APP_LOG_LINE

        line = (
            "forecast_job\tUNKNOWN STEP\t2026-05-20T15:50:12.1234567Z "
            "2026-05-20 15:50:12,123 - main - ERROR - Failed to process"
        )
        assert APP_LOG_LINE.match(line) is not None

    def test_research_content_no_match(self):
        from scripts.backfill_research_from_logs import APP_LOG_LINE

        line = "forecast_job\tUNKNOWN STEP\t2026-05-20T15:48:05.5335906Z ## News Articles (AskNews)"
        assert APP_LOG_LINE.match(line) is None

    def test_blank_continuation_no_match(self):
        from scripts.backfill_research_from_logs import APP_LOG_LINE

        line = "forecast_job\tUNKNOWN STEP\t2026-05-20T15:48:05.5336672Z "
        assert APP_LOG_LINE.match(line) is None


class TestQidExtraction:
    """Test extracting question IDs from Metaculus URLs."""

    def test_plain_questions_url(self):
        from scripts.backfill_research_from_logs import extract_qid

        assert extract_qid("https://www.metaculus.com/questions/43613") == 43613

    def test_questions_url_with_slug(self):
        from scripts.backfill_research_from_logs import extract_qid

        assert extract_qid("https://www.metaculus.com/questions/50001/will-ai-pass-turing-test/") == 50001

    def test_community_url(self):
        from scripts.backfill_research_from_logs import extract_qid

        assert extract_qid("https://www.metaculus.com/c/diffusion-community/38880/some-question-slug/") == 38880

    def test_questions_url_trailing_slash(self):
        from scripts.backfill_research_from_logs import extract_qid

        assert extract_qid("https://www.metaculus.com/questions/12345/") == 12345

    def test_invalid_url_returns_none(self):
        from scripts.backfill_research_from_logs import extract_qid

        assert extract_qid("https://www.metaculus.com/about/") is None


class TestProviderDetection:
    """Test detection of research providers from ## headers."""

    def test_asknews_header(self):
        from scripts.backfill_research_from_logs import detect_providers

        text = "## News Articles (AskNews)\nSome content"
        assert "asknews" in detect_providers(text)

    def test_native_search_header(self):
        from scripts.backfill_research_from_logs import detect_providers

        text = "## Web Research (Native Search)\nSome content"
        assert "native_search" in detect_providers(text)

    def test_gemini_search_header(self):
        from scripts.backfill_research_from_logs import detect_providers

        text = "## Web Research (Google Search via Gemini)\nSome content"
        assert "gemini_search" in detect_providers(text)

    def test_financial_data_header(self):
        from scripts.backfill_research_from_logs import detect_providers

        text = "## Financial & Economic Data\nGDP data"
        assert "financial_data" in detect_providers(text)

    def test_multiple_providers(self):
        from scripts.backfill_research_from_logs import detect_providers

        text = "## News Articles (AskNews)\nContent\n## Web Research (Native Search)\nMore"
        providers = detect_providers(text)
        assert providers == ["asknews", "native_search"]

    def test_no_providers(self):
        from scripts.backfill_research_from_logs import detect_providers

        text = "Some plain text without headers"
        assert detect_providers(text) == []

    def test_exa_header(self):
        from scripts.backfill_research_from_logs import detect_providers

        text = "## Web Research (Exa)\nExa content"
        assert "exa" in detect_providers(text)

    def test_perplexity_header(self):
        from scripts.backfill_research_from_logs import detect_providers

        text = "## Web Research (Perplexity)\nPerplexity content"
        assert "perplexity" in detect_providers(text)


class TestGapFillDetection:
    """Test detection of gap-fill usage from research text."""

    def test_gap_fill_present(self):
        from scripts.backfill_research_from_logs import detect_gap_fill

        text = "## News Articles (AskNews)\nContent\n## Targeted Gap-Fill (second pass)\nFilling gaps"
        assert detect_gap_fill(text) is True

    def test_gap_fill_absent(self):
        from scripts.backfill_research_from_logs import detect_gap_fill

        text = "## News Articles (AskNews)\nContent\n## Web Research (Native Search)\nMore"
        assert detect_gap_fill(text) is False

    def test_gap_fill_alternate_casing(self):
        from scripts.backfill_research_from_logs import detect_gap_fill

        text = "## Targeted Gap-Fill\nSome gap fill content"
        assert detect_gap_fill(text) is True


class TestParseResearchBlocks:
    """Test the full log parser on synthetic log snippets."""

    def test_single_research_block_old_logger(self):
        from scripts.backfill_research_from_logs import parse_research_blocks

        records = parse_research_blocks(SAMPLE_LOG_MAIN_LOGGER, run_id="12345")
        assert len(records) == 1
        record = records[0]
        assert record["qid"] == 43613
        assert record["page_url"] == "https://www.metaculus.com/questions/43613"
        assert record["timestamp"] == "2026-05-20T15:48:05Z"
        assert record["run_id"] == "12345"
        assert "## News Articles (AskNews)" in record["research_text"]
        assert "## Web Research (Native Search)" in record["research_text"]
        assert "Some web research content here." in record["research_text"]
        assert "asknews" in record["providers_used"]
        assert "native_search" in record["providers_used"]
        assert record["gap_fill_used"] is False
        assert record["schema_version"] == 1
        assert record["question_text"] == ""
        assert record["run_mode"] == "tournament"
        assert record["tournament_id"] == ""
        assert record["research_chars"] == len(record["research_text"])

    def test_new_logger_with_gap_fill(self):
        from scripts.backfill_research_from_logs import parse_research_blocks

        records = parse_research_blocks(SAMPLE_LOG_NEW_LOGGER, run_id="67890")
        assert len(records) == 1
        record = records[0]
        assert record["qid"] == 50001
        assert record["page_url"] == "https://www.metaculus.com/questions/50001/will-ai-pass-turing-test/"
        assert record["timestamp"] == "2026-05-24T10:00:01Z"
        assert "asknews" in record["providers_used"]
        assert "gemini_search" in record["providers_used"]
        assert record["gap_fill_used"] is True

    def test_community_url_parsing(self):
        from scripts.backfill_research_from_logs import parse_research_blocks

        records = parse_research_blocks(SAMPLE_LOG_COMMUNITY_URL, run_id="99999")
        assert len(records) == 1
        record = records[0]
        assert record["qid"] == 38880
        assert record["page_url"] == "https://www.metaculus.com/c/diffusion-community/38880/some-question-slug/"
        assert "Community question research here." in record["research_text"]

    def test_multiple_questions_in_one_log(self):
        from scripts.backfill_research_from_logs import parse_research_blocks

        records = parse_research_blocks(SAMPLE_LOG_MULTIPLE_QUESTIONS, run_id="55555")
        assert len(records) == 2
        assert records[0]["qid"] == 11111
        assert records[1]["qid"] == 22222
        assert "asknews" in records[0]["providers_used"]
        assert "native_search" in records[1]["providers_used"]
        assert "financial_data" in records[1]["providers_used"]

    def test_no_research_in_log(self):
        from scripts.backfill_research_from_logs import parse_research_blocks

        records = parse_research_blocks(SAMPLE_LOG_NO_RESEARCH, run_id="00000")
        assert len(records) == 0

    def test_research_text_strips_gha_prefix(self):
        from scripts.backfill_research_from_logs import parse_research_blocks

        records = parse_research_blocks(SAMPLE_LOG_MAIN_LOGGER, run_id="12345")
        record = records[0]
        assert "forecast_job" not in record["research_text"]
        assert "2026-05-20T15:48:05.5335906Z" not in record["research_text"]

    def test_research_text_does_not_include_terminating_log_line(self):
        from scripts.backfill_research_from_logs import parse_research_blocks

        records = parse_research_blocks(SAMPLE_LOG_MAIN_LOGGER, run_id="12345")
        record = records[0]
        assert "Using research providers" not in record["research_text"]


class TestTimestampNormalization:
    """Test that GHA timestamps are normalized to ISO 8601 without fractional seconds."""

    def test_fractional_seconds_stripped(self):
        from scripts.backfill_research_from_logs import normalize_timestamp

        assert normalize_timestamp("2026-05-20T15:48:05.5335094Z") == "2026-05-20T15:48:05Z"

    def test_already_normalized(self):
        from scripts.backfill_research_from_logs import normalize_timestamp

        assert normalize_timestamp("2026-05-20T15:48:05Z") == "2026-05-20T15:48:05Z"

    def test_single_fractional_digit(self):
        from scripts.backfill_research_from_logs import normalize_timestamp

        assert normalize_timestamp("2026-05-20T10:00:01.0Z") == "2026-05-20T10:00:01Z"
