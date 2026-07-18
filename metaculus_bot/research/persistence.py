"""Research persistence write path — captures research text during production runs as JSONL for backtest replay."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

RESEARCH_SCHEMA_VERSION = 2


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
        provider_results: list[dict] | None = None,
        providers_attempted: list[str] | None = None,
        providers_succeeded: list[str] | None = None,
        gap_fill_v2: dict | None = None,
        provider_diagnostics_block: str | None = None,
    ) -> None:
        """Record a single question's research output.

        ``providers_used`` is legacy and ambiguous — in live-capture records it
        meant "attempted", in comment-backfill records "succeeded-with-output".
        It is kept for back-compat with old archive readers; ``provider_results``
        is the authoritative per-provider outcome, with ``providers_attempted`` /
        ``providers_succeeded`` as unambiguous derived lists. The new args default
        to None so older callers (and backfill paths) keep working.

        ``gap_fill_v2`` carries the agentic-loop trace (``transcript`` +
        ``telemetry`` dicts) when the v2 loop ran; the key is written only in
        that case so records stay compact when the flag is off.

        ``provider_diagnostics_block`` is the rendered ``## Provider Diagnostics``
        markdown. Since the diagnostics seam (2026-07) it is no longer embedded in
        ``research_text`` (forecasters must not see it), so it is archived as its
        own field to keep records self-contained for grep-based triage.
        """
        record: dict[str, object] = {
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "qid": qid,
            "page_url": page_url,
            "question_text": question_text,
            "research_text": research_text,
            "providers_used": providers_used,
            "providers_attempted": providers_attempted if providers_attempted is not None else [],
            "providers_succeeded": providers_succeeded if providers_succeeded is not None else [],
            "provider_results": provider_results if provider_results is not None else [],
            "run_mode": self._run_mode,
            "tournament_id": self._tournament_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self._run_id,
            "research_chars": len(research_text),
            "gap_fill_used": gap_fill_used,
        }
        if gap_fill_v2 is not None:
            record["gap_fill_v2"] = gap_fill_v2
        if provider_diagnostics_block:
            record["provider_diagnostics_block"] = provider_diagnostics_block
        self._records.append(record)

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
