"""Tests for the raw research-provider payload logger (metaculus_bot/research/raw_log.py).

Network-free and paid-API-free: exercises serialization, truncation, no-op-when-
disabled, and the write-failure guard against temp files. The logger's whole
contract is "capture the raw payload durably, but NEVER break a forecast", so the
guard tests (unserializable payload, unwritable dir) are load-bearing.
"""

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from metaculus_bot.constants import (
    RAW_RESEARCH_LOG_DIR_ENV,
    RAW_RESEARCH_LOG_ENABLED_ENV,
    RAW_RESEARCH_MAX_PAYLOAD_CHARS,
)
from metaculus_bot.research.raw_log import record_raw_research


@pytest.fixture
def enabled_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Enable raw logging and point it at a tmp dir; return the dir."""
    monkeypatch.setenv(RAW_RESEARCH_LOG_ENABLED_ENV, "true")
    monkeypatch.setenv(RAW_RESEARCH_LOG_DIR_ENV, str(tmp_path))
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    return tmp_path


def _read_records(log_dir: Path) -> list[dict]:
    files = list(log_dir.glob("raw_research_*.jsonl"))
    assert len(files) == 1, f"expected exactly one raw-research log, got {files}"
    return [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]


class TestRecordsWhenEnabled:
    def test_writes_one_record_with_expected_fields(self, enabled_log_dir: Path):
        record_raw_research(qid=999, provider="asknews", phase="hot", payload={"articles": [1, 2, 3]})

        records = _read_records(enabled_log_dir)
        assert len(records) == 1
        rec = records[0]
        assert rec["qid"] == 999
        assert rec["provider"] == "asknews"
        assert rec["phase"] == "hot"
        assert rec["payload"] == {"articles": [1, 2, 3]}
        assert rec["truncated"] is False
        assert rec["payload_chars"] > 0
        assert "fetched_at" in rec

    def test_filename_uses_github_run_id(self, enabled_log_dir: Path):
        record_raw_research(qid=1, provider="native_search", payload="text")
        assert (enabled_log_dir / "raw_research_12345.jsonl").exists()

    def test_falls_back_to_local_run_id(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(RAW_RESEARCH_LOG_ENABLED_ENV, "true")
        monkeypatch.setenv(RAW_RESEARCH_LOG_DIR_ENV, str(tmp_path))
        monkeypatch.delenv("GITHUB_RUN_ID", raising=False)

        record_raw_research(qid=1, provider="native_search", payload="text")
        assert (tmp_path / "raw_research_local.jsonl").exists()

    def test_phase_defaults_to_none(self, enabled_log_dir: Path):
        record_raw_research(qid=1, provider="native_search", payload="hello")
        assert _read_records(enabled_log_dir)[0]["phase"] is None

    def test_multiple_records_append(self, enabled_log_dir: Path):
        record_raw_research(qid=1, provider="asknews", phase="hot", payload={"a": 1})
        record_raw_research(qid=1, provider="asknews", phase="historical", payload={"b": 2})
        record_raw_research(qid=2, provider="native_search", payload="x")

        records = _read_records(enabled_log_dir)
        assert len(records) == 3
        assert [r["phase"] for r in records] == ["hot", "historical", None]


class TestNoopWhenDisabled:
    def test_no_file_written_when_env_unset(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(RAW_RESEARCH_LOG_ENABLED_ENV, raising=False)
        monkeypatch.setenv(RAW_RESEARCH_LOG_DIR_ENV, str(tmp_path))

        record_raw_research(qid=1, provider="asknews", phase="hot", payload={"a": 1})
        assert list(tmp_path.glob("*")) == []

    def test_no_file_written_when_env_false(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(RAW_RESEARCH_LOG_ENABLED_ENV, "false")
        monkeypatch.setenv(RAW_RESEARCH_LOG_DIR_ENV, str(tmp_path))

        record_raw_research(qid=1, provider="asknews", phase="hot", payload={"a": 1})
        assert list(tmp_path.glob("*")) == []


class TestSerialization:
    def test_pydantic_like_payload_uses_model_dump(self, enabled_log_dir: Path):
        class FakeArticle:
            def model_dump(self, mode: str = "python"):
                assert mode == "json"
                return {"title": "Hi", "pub_date": "2026-07-19T00:00:00"}

        record_raw_research(qid=1, provider="asknews", phase="hot", payload=[FakeArticle()])

        rec = _read_records(enabled_log_dir)[0]
        assert rec["payload"] == [{"title": "Hi", "pub_date": "2026-07-19T00:00:00"}]

    def test_dataclass_with_datetime_serializes(self, enabled_log_dir: Path):
        @dataclasses.dataclass
        class Match:
            name: str
            close_time: datetime | None

        payload = [Match(name="m1", close_time=datetime(2026, 7, 19, tzinfo=timezone.utc))]
        record_raw_research(qid=1, provider="prediction_market", payload=payload)

        rec = _read_records(enabled_log_dir)[0]
        assert rec["payload"][0]["name"] == "m1"
        assert rec["payload"][0]["close_time"].startswith("2026-07-19T00:00:00")

    def test_plain_string_payload(self, enabled_log_dir: Path):
        record_raw_research(qid=1, provider="native_search", payload="raw completion text")
        assert _read_records(enabled_log_dir)[0]["payload"] == "raw completion text"


class TestTruncation:
    def test_oversized_payload_is_truncated(self, enabled_log_dir: Path):
        big = "x" * (RAW_RESEARCH_MAX_PAYLOAD_CHARS + 5000)
        record_raw_research(qid=1, provider="gemini_search", payload=big)

        rec = _read_records(enabled_log_dir)[0]
        assert rec["truncated"] is True
        assert rec["payload"]["_truncated"] is True
        # payload_chars reflects the ORIGINAL serialized length, not the preview
        assert rec["payload_chars"] > RAW_RESEARCH_MAX_PAYLOAD_CHARS
        # the stored preview is bounded
        assert len(rec["payload"]["_preview"]) <= RAW_RESEARCH_MAX_PAYLOAD_CHARS

    def test_small_payload_not_truncated(self, enabled_log_dir: Path):
        record_raw_research(qid=1, provider="gemini_search", payload="small")
        rec = _read_records(enabled_log_dir)[0]
        assert rec["truncated"] is False
        assert rec["payload"] == "small"


class TestNeverBreaksForecast:
    def test_unserializable_payload_is_swallowed(self, enabled_log_dir: Path):
        # A dict with a tuple key raises TypeError in json.dumps regardless of
        # the `default=` encoder (keys, not values). Must be caught, logged, and
        # swallowed — a serialization bug must never propagate into a forecast.
        record_raw_research(qid=1, provider="asknews", phase="hot", payload={(1, 2): "bad"})

        # No exception, and no corrupt line written.
        files = list(enabled_log_dir.glob("raw_research_*.jsonl"))
        if files:
            assert files[0].read_text().strip() == ""

    def test_unwritable_dir_is_swallowed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file, not a dir")
        monkeypatch.setenv(RAW_RESEARCH_LOG_ENABLED_ENV, "true")
        # A path under a regular file: mkdir(parents=True) raises NotADirectoryError (OSError).
        monkeypatch.setenv(RAW_RESEARCH_LOG_DIR_ENV, str(blocker / "sub"))

        # Must not raise.
        record_raw_research(qid=1, provider="asknews", phase="hot", payload={"a": 1})

    def test_payload_encoder_raising_arbitrary_error_is_swallowed(self, enabled_log_dir: Path):
        # A payload whose model_dump raises a non-(TypeError|ValueError) error
        # (RuntimeError here) is invoked by json.dumps via `default=`. The old
        # serialization guard only caught (TypeError, ValueError), so this
        # escaped and broke the forecast. Must now be caught, logged, swallowed.
        class ExplodingPayload:
            def model_dump(self, mode: str = "python") -> dict:
                raise RuntimeError("boom from model_dump")

        record_raw_research(qid=1, provider="asknews", phase="hot", payload=[ExplodingPayload()])

        files = list(enabled_log_dir.glob("raw_research_*.jsonl"))
        if files:
            assert files[0].read_text().strip() == ""

    def test_lone_surrogate_payload_is_swallowed(self, enabled_log_dir: Path):
        # A lone surrogate serializes fine under json.dumps(ensure_ascii=False)
        # but raises UnicodeEncodeError (a ValueError subclass, NOT OSError) at
        # f.write() on the utf-8 file handle. The old write guard only caught
        # OSError, so this escaped and broke the forecast. Must now be swallowed.
        record_raw_research(qid=1, provider="asknews", phase="hot", payload={"text": "bad\ud800surrogate"})

        # No exception; the write failed, so the log is empty (partial writes to
        # a text stream flush nothing before the encode raises).
        files = list(enabled_log_dir.glob("raw_research_*.jsonl"))
        if files:
            assert files[0].read_text().strip() == ""
