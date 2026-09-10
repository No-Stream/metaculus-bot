"""Research persistence write path — captures research text during production runs as JSONL for backtest replay."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

RESEARCH_SCHEMA_VERSION = 2

__all__ = ["RESEARCH_SCHEMA_VERSION", "ResearchPersistenceWriter"]


class ResearchPersistenceWriter:
    """Accumulates research records during a bot run and flushes to JSONL."""

    def __init__(self, run_mode: str, platform: str, tournament_id: str, run_id: str) -> None:
        self._run_mode = run_mode
        self._platform = platform
        self._tournament_id = tournament_id
        self._run_id = run_id
        self._records: list[dict] = []

    def record(
        self,
        *,
        qid: int,
        page_url: str,
        question_text: str,
        research_text: str,
        providers_used: list[str],
        gap_fill_used: bool,
        provider_results: list[dict] | None = None,
        providers_attempted: list[str] | None = None,
        providers_succeeded: list[str] | None = None,
        gap_fill_v2: dict | None = None,
        provider_diagnostics_block: str | None = None,
        asknews_raw: str | None = None,
        post_id: int | None = None,
    ) -> None:
        """Record a single question's research output.

        Several arguments are archive fields whose names do not say what they mean:
        ``qid`` versus ``post_id``, the legacy ``providers_used``, the optional
        ``gap_fill_v2`` / ``provider_diagnostics_block`` / ``asknews_raw`` keys, and the
        ``platform`` tag. All of them are additive and safe for passthrough readers.

        Field semantics: docs/performance_analysis.md "The research record's fields".
        """
        record: dict[str, object] = {
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "qid": qid,
            "post_id": post_id,
            "page_url": page_url,
            "question_text": question_text,
            "research_text": research_text,
            "providers_used": providers_used,
            "providers_attempted": providers_attempted if providers_attempted is not None else [],
            "providers_succeeded": providers_succeeded if providers_succeeded is not None else [],
            "provider_results": provider_results if provider_results is not None else [],
            "run_mode": self._run_mode,
            "platform": self._platform,
            "tournament_id": self._tournament_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self._run_id,
            "research_chars": len(research_text),
            "gap_fill_used": gap_fill_used,
        }
        if gap_fill_v2 is not None:
            record["gap_fill_v2"] = gap_fill_v2
        if provider_diagnostics_block:
            record["provider_diagnostics_block"] = provider_diagnostics_block
        if asknews_raw:
            record["asknews_raw"] = asknews_raw
        self._records.append(record)

    def flush(self, output_dir: str = "research_outputs") -> Path | None:
        """Write accumulated records to a JSONL file. Returns the path written, or None if no records."""
        if not self._records:
            logger.info("No research records to persist")
            return None

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        filename = out_path / f"research_{timestamp}.jsonl"

        with open(filename, "w", encoding="utf-8") as f:
            for record in self._records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Persisted {len(self._records)} research record(s) to {filename}")
        return filename
