"""Score gap-fill v2 GHOST_FORECAST markers against resolutions (paired vs published).

The agentic gap-fill loop (v2) logs an unpublished "ghost" forecast per question for
telemetry (``GHOST_FORECAST: qtype=... summary=...``). Comparing those ghosts to the
actually-published forecast on resolved questions is the NAMED GATE for retiring v1
gap-fill: if the v2-driven ghost consistently out-scores the published forecast, v2 is
carrying its weight.

This is scaffold quality by design. v2 shipped to prod 2026-07-17, so there are ~0
resolved v2-era questions today; the scorer must run cleanly and report ``n=0, waiting
on resolutions`` rather than error. It will be hardened once real deltas exist.

INPUTS:
* ghosts   — ``ghost_forecast.jsonl`` from ``backtests/telemetry_archive/`` (harvested
             by ``make sync_telemetry``).
* records  — resolved-question records: either a pre-built performance-analysis dataset
             JSON (``--perf-json``) or a live read-only pull (``--tournament``, free).

Only binary ghosts (full posterior) and MC ghosts (full option probs) are scoreable
from the marker. Numeric ghosts expose only a median in the marker, so no numeric log
score is computable from telemetry alone — they're counted, not scored.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from statistics import mean

from metaculus_bot.scoring_common import binary_log_score, mc_log_score
from scripts.telemetry.archive import load_marker_records

logger = logging.getLogger(__name__)

_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_BINARY_RE = re.compile(rf"posterior_prob=(?P<p>{_FLOAT})")
_MEDIAN_RE = re.compile(rf"median=(?P<m>{_FLOAT})")
_MC_PAIR_RE = re.compile(rf"(?P<name>[^=,]+)=(?P<prob>{_FLOAT})")


def parse_ghost_summary(qtype: str, summary: str) -> float | dict[str, float] | None:
    """Parse a GHOST_FORECAST ``summary=`` payload into a scoreable value.

    * binary          -> ``float`` posterior probability
    * multiple_choice -> ``{option_name: prob}``
    * numeric         -> ``{"median": float}`` (median only; not enough to log-score)
    * anything else / empty -> ``None``
    """
    summary = (summary or "").strip()
    if not summary:
        return None
    if qtype == "binary":
        match = _BINARY_RE.search(summary)
        return float(match.group("p")) if match else None
    if qtype == "multiple_choice":
        probs = {name.strip(): float(prob) for name, prob in _MC_PAIR_RE.findall(summary)}
        return probs or None
    if qtype == "numeric":
        match = _MEDIAN_RE.search(summary)
        return {"median": float(match.group("m"))} if match else None
    return None


def _latest_ghost_per_qid(ghosts: list[dict]) -> dict[int, dict]:
    """Keep the newest ghost per question id (by run_date, then line_ts, then seq)."""
    latest: dict[int, dict] = {}
    for ghost in ghosts:
        qid = ghost.get("qid")
        if qid is None:
            continue
        key = (str(ghost.get("run_date", "")), str(ghost.get("line_ts") or ""), ghost.get("seq", 0))
        current = latest.get(qid)
        if current is None or key > (
            str(current.get("run_date", "")),
            str(current.get("line_ts") or ""),
            current.get("seq", 0),
        ):
            latest[qid] = ghost
    return latest


def _score_binary(ghost: dict, record: dict) -> dict | None:
    """Paired binary log-score row (ghost vs published), or None if not scoreable."""
    resolution = record.get("resolution_parsed")
    if not isinstance(resolution, bool):
        return None
    ghost_p = parse_ghost_summary("binary", ghost.get("summary", ""))
    published_p = record.get("our_prob_yes")
    if not isinstance(ghost_p, float) or published_p is None:
        return None
    ghost_ls = binary_log_score(ghost_p, resolution)
    published_ls = binary_log_score(float(published_p), resolution)
    return {
        "qid": ghost["qid"],
        "resolution": resolution,
        "ghost_prob": ghost_p,
        "published_prob": float(published_p),
        "ghost_log_score": ghost_ls,
        "published_log_score": published_ls,
        "delta": ghost_ls - published_ls,
    }


def _score_mc(ghost: dict, record: dict) -> dict | None:
    """Paired MC log-score row (ghost vs published), or None if not scoreable."""
    resolution = record.get("resolution_parsed")
    options = record.get("options") or []
    published_values = record.get("our_forecast_values")
    ghost_probs = parse_ghost_summary("multiple_choice", ghost.get("summary", ""))
    if not isinstance(resolution, str) or resolution not in options:
        return None
    if not isinstance(ghost_probs, dict) or not all(opt in ghost_probs for opt in options):
        return None
    if not published_values or len(published_values) != len(options):
        return None
    correct_idx = options.index(resolution)
    ghost_ordered = [ghost_probs[opt] for opt in options]
    ghost_ls = mc_log_score(ghost_ordered, correct_idx)
    published_ls = mc_log_score(list(published_values), correct_idx)
    return {
        "qid": ghost["qid"],
        "resolution": resolution,
        "ghost_log_score": ghost_ls,
        "published_log_score": published_ls,
        "delta": ghost_ls - published_ls,
    }


def _summarize_rows(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "mean_delta": mean(r["delta"] for r in rows) if rows else None,
        "rows": rows,
    }


def join_and_score(ghosts: list[dict], records: list[dict]) -> dict:
    """Join latest ghosts to resolved records and compute paired log-score deltas.

    Returns a summary dict with per-qtype breakdowns. Everything is pure/in-memory so
    the n=0 path (no resolved v2-era questions yet) is exercised in tests.
    """
    records_by_qid = {r.get("question_id"): r for r in records}
    latest = _latest_ghost_per_qid(ghosts)

    binary_rows: list[dict] = []
    mc_rows: list[dict] = []
    numeric_joined = 0
    numeric_unscoreable = 0
    n_joined = 0

    for qid, ghost in latest.items():
        record = records_by_qid.get(qid)
        if record is None:
            continue
        n_joined += 1
        qtype = ghost.get("qtype")
        if qtype == "binary":
            row = _score_binary(ghost, record)
            if row is not None:
                binary_rows.append(row)
        elif qtype == "multiple_choice":
            row = _score_mc(ghost, record)
            if row is not None:
                mc_rows.append(row)
        elif qtype == "numeric":
            numeric_joined += 1
            numeric_unscoreable += 1

    return {
        "n_ghosts": len(latest),
        "n_joined": n_joined,
        "n_scored": len(binary_rows) + len(mc_rows),
        "binary": _summarize_rows(binary_rows),
        "multiple_choice": _summarize_rows(mc_rows),
        "numeric": {"n_joined": numeric_joined, "n_unscoreable": numeric_unscoreable},
    }


def render_report(summary: dict) -> str:
    """Human-readable summary. A positive mean delta = ghost out-scores published."""
    lines = [
        "=== Ghost-forecast scoring (gap-fill v2 ghost vs published) ===",
        f"Ghosts (latest per qid): {summary['n_ghosts']}",
        f"Joined to resolved-question dataset: {summary['n_joined']}",
    ]
    if summary["n_scored"] == 0:
        lines.append("Scored ghosts: n=0 — waiting on resolutions (v2 shipped 2026-07-17; expected today).")
    else:
        lines.append(f"Scored ghosts: {summary['n_scored']}")
    for qtype in ("binary", "multiple_choice"):
        block = summary[qtype]
        if block["n"]:
            delta = block["mean_delta"]
            verdict = "ghost better" if delta > 0 else "published better"
            lines.append(f"  {qtype}: n={block['n']} mean_delta={delta:+.4f} ({verdict})")
    numeric = summary["numeric"]
    if numeric["n_joined"]:
        lines.append(
            f"  numeric: {numeric['n_joined']} joined, {numeric['n_unscoreable']} unscoreable "
            "(marker exposes median only)"
        )
    return "\n".join(lines)


def _load_records(perf_json: str | None, tournament: str | None) -> list[dict]:
    """Load resolved-question records from a perf JSON, else a live read-only pull, else []."""
    if perf_json:
        from metaculus_bot.performance_analysis.collector import (
            load_dataset,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import  # lazy: keep the pure scoring core decoupled from the collector's import chain
        )

        records = load_dataset(perf_json)
        logger.info(f"Loaded {len(records)} records from {perf_json}")
        return records
    if tournament:
        from metaculus_bot.performance_analysis.collector import (
            build_performance_dataset,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import  # lazy: keep the pure scoring core decoupled from the collector's import chain
        )

        logger.info(f"Building performance dataset from tournament '{tournament}' (read-only Metaculus pull)...")
        return build_performance_dataset(tournament=tournament)
    logger.warning("No --perf-json or --tournament given; reporting ghost inventory only (nothing to score against).")
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Score gap-fill v2 ghost forecasts vs published on resolutions.")
    parser.add_argument("--archive-dir", default="backtests/telemetry_archive", help="Telemetry archive dir")
    parser.add_argument("--perf-json", default=None, help="Pre-built performance-analysis dataset JSON")
    parser.add_argument("--tournament", default=None, help="Live read-only pull for this tournament slug (free)")
    parser.add_argument("--output", default=None, help="Optional path to dump the summary JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    ghosts = load_marker_records(Path(args.archive_dir), "ghost_forecast")
    logger.info(f"Loaded {len(ghosts)} ghost_forecast record(s) from {args.archive_dir}")

    records = _load_records(args.perf_json, args.tournament)
    summary = join_and_score(ghosts, records)
    print(render_report(summary))

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
        logger.info(f"Summary written to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
