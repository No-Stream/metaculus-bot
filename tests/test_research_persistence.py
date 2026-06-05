"""Tests for the research persistence write path."""

import json
from pathlib import Path

from metaculus_bot.research.persistence import RESEARCH_SCHEMA_VERSION, ResearchPersistenceWriter


class TestResearchPersistenceWriter:
    """Tests for ResearchPersistenceWriter accumulate-and-flush lifecycle."""

    def test_flush_with_no_records_returns_none(self, tmp_path: Path) -> None:
        writer = ResearchPersistenceWriter(run_mode="tournament", tournament_id="test-123", run_id="local")
        result = writer.flush(output_dir=str(tmp_path / "empty_output"))
        assert result is None

    def test_single_record_writes_valid_jsonl(self, tmp_path: Path) -> None:
        writer = ResearchPersistenceWriter(run_mode="tournament", tournament_id="summer-2026", run_id="gh-42")
        writer.record(
            qid=12345,
            page_url="https://www.metaculus.com/questions/12345/will-x-happen/",
            question_text="Will X happen by 2027?",
            research_text="Some research about X happening.",
            providers_used=["asknews", "native_search"],
            gap_fill_used=True,
        )

        result = writer.flush(output_dir=str(tmp_path))
        assert result is not None
        assert result.exists()
        assert result.suffix == ".jsonl"

        lines = result.read_text().strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["qid"] == 12345
        assert record["page_url"] == "https://www.metaculus.com/questions/12345/will-x-happen/"
        assert record["question_text"] == "Will X happen by 2027?"
        assert record["research_text"] == "Some research about X happening."
        assert record["providers_used"] == ["asknews", "native_search"]
        assert record["gap_fill_used"] is True
        assert record["run_mode"] == "tournament"
        assert record["tournament_id"] == "summer-2026"
        assert record["run_id"] == "gh-42"
        assert record["research_chars"] == len("Some research about X happening.")
        assert record["schema_version"] == RESEARCH_SCHEMA_VERSION

    def test_multiple_records_produce_multiple_lines(self, tmp_path: Path) -> None:
        writer = ResearchPersistenceWriter(run_mode="minibench", tournament_id="t-1", run_id="local")
        for i in range(5):
            writer.record(
                qid=100 + i,
                page_url=f"https://metaculus.com/q/{100 + i}/",
                question_text=f"Question {i}",
                research_text=f"Research text for question {i}",
                providers_used=["gemini_search"],
                gap_fill_used=False,
            )

        result = writer.flush(output_dir=str(tmp_path))
        assert result is not None

        lines = result.read_text().strip().splitlines()
        assert len(lines) == 5

        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["qid"] == 100 + i
            assert record["question_text"] == f"Question {i}"

    def test_schema_fields_present_and_typed(self, tmp_path: Path) -> None:
        writer = ResearchPersistenceWriter(run_mode="tournament", tournament_id="tid", run_id="rid")
        writer.record(
            qid=1,
            page_url="https://example.com/q/1/",
            question_text="Test?",
            research_text="data",
            providers_used=["p1", "p2"],
            gap_fill_used=False,
        )

        result = writer.flush(output_dir=str(tmp_path))
        assert result is not None
        record = json.loads(result.read_text().strip())

        assert isinstance(record["schema_version"], int)
        assert isinstance(record["qid"], int)
        assert isinstance(record["page_url"], str)
        assert isinstance(record["question_text"], str)
        assert isinstance(record["research_text"], str)
        assert isinstance(record["providers_used"], list)
        assert isinstance(record["run_mode"], str)
        assert isinstance(record["tournament_id"], str)
        assert isinstance(record["timestamp"], str)
        assert isinstance(record["run_id"], str)
        assert isinstance(record["research_chars"], int)
        assert isinstance(record["gap_fill_used"], bool)

    def test_flush_creates_output_directory(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "deeply" / "nested" / "output"
        assert not nested_dir.exists()

        writer = ResearchPersistenceWriter(run_mode="tournament", tournament_id="t", run_id="r")
        writer.record(
            qid=1,
            page_url="https://example.com/q/1/",
            question_text="Q?",
            research_text="R",
            providers_used=[],
            gap_fill_used=False,
        )

        result = writer.flush(output_dir=str(nested_dir))
        assert result is not None
        assert nested_dir.exists()
        assert result.parent == nested_dir

    def test_timestamp_is_iso_format(self, tmp_path: Path) -> None:
        """Verify timestamp is valid ISO 8601 UTC."""
        from datetime import datetime, timezone

        writer = ResearchPersistenceWriter(run_mode="tournament", tournament_id="t", run_id="r")
        writer.record(
            qid=1,
            page_url="u",
            question_text="q",
            research_text="r",
            providers_used=[],
            gap_fill_used=False,
        )

        result = writer.flush(output_dir=str(tmp_path))
        assert result is not None
        record = json.loads(result.read_text().strip())

        ts = datetime.fromisoformat(record["timestamp"])
        assert ts.tzinfo == timezone.utc

    def test_unicode_content_preserved(self, tmp_path: Path) -> None:
        writer = ResearchPersistenceWriter(run_mode="tournament", tournament_id="t", run_id="r")
        unicode_text = "Probability of event: 73.2% — source: “Forecast Journal” \U0001f4c8"
        writer.record(
            qid=99,
            page_url="https://example.com/q/99/",
            question_text="Unicode test?",
            research_text=unicode_text,
            providers_used=["asknews"],
            gap_fill_used=True,
        )

        result = writer.flush(output_dir=str(tmp_path))
        assert result is not None
        record = json.loads(result.read_text().strip())
        assert record["research_text"] == unicode_text
