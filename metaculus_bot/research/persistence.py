"""Research persistence write path — captures research text during production runs as JSONL for backtest replay."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

RESEARCH_SCHEMA_VERSION = 1


class ResearchPersistenceWriter:
    """Accumulates research records during a bot run and flushes to JSONL."""

    def __init__(self, run_mode: str, tournament_id: str, run_id: str) -> None:
        self._run_mode = run_mode
        self._tournament_id = tournament_id
        self._run_id = run_id
        self._records: list[dict] = []

    def record(
        self,
        qid: int,
        page_url: str,
        question_text: str,
        research_text: str,
        providers_used: list[str],
        gap_fill_used: bool,
    ) -> None:
        """Record a single question's research output."""
        self._records.append(
            {
                "schema_version": RESEARCH_SCHEMA_VERSION,
                "qid": qid,
                "page_url": page_url,
                "question_text": question_text,
                "research_text": research_text,
                "providers_used": providers_used,
                "run_mode": self._run_mode,
                "tournament_id": self._tournament_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": self._run_id,
                "research_chars": len(research_text),
                "gap_fill_used": gap_fill_used,
            }
        )

    def flush(self, output_dir: str = "research_outputs") -> Path | None:
        """Write accumulated records to a JSONL file. Returns the path written, or None if no records."""
        if not self._records:
            logger.info("No research records to persist")
            return None

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = out_path / f"research_{timestamp}.jsonl"

        with open(filename, "w", encoding="utf-8") as f:
            for record in self._records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Persisted {len(self._records)} research record(s) to {filename}")
        return filename
