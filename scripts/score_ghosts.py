"""Score gap-fill v2 ghost forecasts against resolutions (paired vs published).

The agentic gap-fill loop (v2) logs an unpublished "ghost" forecast per question for
telemetry. Comparing those ghosts to the actually-published forecast on resolved
questions is the NAMED GATE for retiring v1 gap-fill: if the v2-driven ghost
consistently out-scores the published forecast, v2 is carrying its weight.

Two marker sources, in preference order:

* ``GHOST_FORECAST_JSON`` — the full-fidelity companion marker (a compact JSON blob:
  binary posterior, complete MC option probs, or the complete percentile set + median
  for numeric). Preferred when present — it makes numeric ghosts scoreable, not just
  countable.
* ``GHOST_FORECAST``      — the legacy lossy summary line. Falls back to this for the
  pre-upgrade era (binary/MC scoreable from the summary; numeric exposes a median only,
  so it stays unscoreable there).

A third marker, ``GHOST_PRE_JSON`` (the turn-one, PRE-research dry run), feeds the
identity split rather than the scoring: on 7 of the first 12 scored pairs the
concluding ghost was byte-identical to the pre-research dry run (2026-08-24 residual
round), so a pooled ghost delta mixes measurements of the driver's PRIOR with
measurements of the loop's research. The report therefore splits the delta by whether
the loop moved the driver's own forecast, and the retirement gate should read the
loop-moved subset, never the pool.

All are harvested into ``backtests/telemetry_archive/`` by ``make sync_telemetry``.
Resolved-question records come from a pre-built performance-analysis dataset JSON
(``--perf-json``) or a live read-only pull (``--tournament``, free).

This is scaffold quality by design. v2 reached prod 2026-07-21 (merge ``b4e9df0``; it was
authored 2026-07-17 on the july15 branch), so there are ~0 resolved v2-era questions today;
the scorer must run cleanly and report ``n=0, waiting on resolutions`` rather than error. It
will be hardened once real deltas exist.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from statistics import mean

from metaculus_bot.scoring_common import binary_log_score, mc_log_score, numeric_log_score
from scripts.telemetry.archive import load_marker_records

logger = logging.getLogger(__name__)

_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_BINARY_RE = re.compile(rf"posterior_prob=(?P<p>{_FLOAT})")
_MEDIAN_RE = re.compile(rf"median=(?P<m>{_FLOAT})")
_MC_PAIR_RE = re.compile(rf"(?P<name>[^=,]+)=(?P<prob>{_FLOAT})")


def parse_ghost_summary(qtype: str, summary: str) -> float | dict[str, float] | None:
    """Parse a legacy ``GHOST_FORECAST`` ``summary=`` payload into a scoreable value.

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


def _normalize_json_ghost(record: dict) -> dict | None:
    """Turn a harvested ``ghost_forecast_json`` record into a normalized ghost, or None.

    The ``forecast_json`` field is a raw JSON string (never coerced by the marker
    parser). None if it is missing / malformed so the caller can fall back to any
    legacy ghost for the same qid.
    """
    raw = record.get("forecast_json")
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {
        "qid": record.get("qid"),
        "qtype": payload.get("qtype"),
        "source": "json",
        "payload": payload,
        # Run identity, so the pre/post split can require both halves to come
        # from the SAME run (a qid can be forecast in several archived runs).
        "run_id": record.get("run_id"),
    }


def _select_ghost_per_qid(json_ghosts: list[dict], legacy_ghosts: list[dict]) -> dict[int, dict]:
    """Pick one normalized ghost to score per qid: JSON-source preferred, legacy fallback.

    JSON ghosts carry the full forecast (deterministically parsable); legacy ghosts
    carry only the lossy summary string. Within each source we keep the latest per
    qid; a (well-formed) JSON ghost always wins over a legacy one for the same qid.
    """
    selected: dict[int, dict] = {}
    for qid, record in _latest_ghost_per_qid(legacy_ghosts).items():
        selected[qid] = {
            "qid": qid,
            "qtype": record.get("qtype"),
            "source": "legacy",
            "summary": record.get("summary", ""),
        }
    for qid, record in _latest_ghost_per_qid(json_ghosts).items():
        normalized = _normalize_json_ghost(record)
        if normalized is not None:
            selected[qid] = normalized
    return selected


def _binary_prob(ghost: dict) -> float | None:
    """Ghost's posterior probability, from either source."""
    if ghost["source"] == "json":
        prob = ghost["payload"].get("prob")
        return float(prob) if isinstance(prob, (int, float)) and not isinstance(prob, bool) else None
    parsed = parse_ghost_summary("binary", ghost.get("summary", ""))
    return parsed if isinstance(parsed, float) else None


def _mc_probs(ghost: dict) -> dict[str, float] | None:
    """Ghost's option->prob dict, from either source."""
    if ghost["source"] == "json":
        option_probs = ghost["payload"].get("option_probs")
        if not isinstance(option_probs, dict) or not option_probs:
            return None
        try:
            return {str(name): float(prob) for name, prob in option_probs.items()}
        except (TypeError, ValueError):
            return None
    parsed = parse_ghost_summary("multiple_choice", ghost.get("summary", ""))
    return parsed if isinstance(parsed, dict) else None


def _numeric_percentiles(ghost: dict) -> dict[float, float] | None:
    """Ghost's full percentile->value map (fraction keys in [0, 1]), or None.

    Only JSON-source ghosts carry the full percentile set; the legacy marker exposes
    a median only and so can't be turned into a CDF.
    """
    if ghost["source"] != "json":
        return None
    declared = ghost["payload"].get("declared_percentiles")
    if not isinstance(declared, dict) or len(declared) < 2:
        return None
    try:
        return {float(pct): float(value) for pct, value in declared.items()}
    except (TypeError, ValueError):
        return None


def _score_binary(ghost: dict, record: dict) -> dict | None:
    """Paired binary log-score row (ghost vs published), or None if not scoreable."""
    resolution = record.get("resolution_parsed")
    if not isinstance(resolution, bool):
        return None
    ghost_p = _binary_prob(ghost)
    published_p = record.get("our_prob_yes")
    if ghost_p is None or published_p is None:
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
    ghost_probs = _mc_probs(ghost)
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


def _score_numeric(ghost: dict, record: dict) -> dict:
    """Attempt a paired numeric log-score (ghost vs published).

    Builds the ghost's CDF from its declared percentiles with the production PCHIP
    builder, on the SAME grid length as the published CDF, then scores both with the
    same Metaculus PMF-bucket log score on identical bounds/scaling — so the delta is
    a clean paired comparison. Always returns a dict with ``scoreable``; when False,
    ``reason`` names the gap so the report can surface it instead of silently dropping.

    Discrete handling — two prod mechanisms, handled differently:

    * Native-discrete questions (Metaculus ``type == "discrete"``) publish a CDF on a
      reduced grid (``cdf_size != 201``). Prod resamples the aggregate onto that grid
      with ``generate_pchip_cdf`` (see ``numeric/pipeline.build_numeric_distribution``
      and ``numeric/utils._postprocess_ensemble_cdf``). We mirror it exactly: the ghost
      is built with ``num_points=len(published_cdf)``, so both sides share the native
      grid and the pairing stays clean. No integer-snap is involved (``discrete_snap``
      explicitly skips ``cdf_size != 201``).
    * Continuous questions (``cdf_size == 201``) are integer-*snapped* by prod only when
      a strict majority of the ensemble's forecasters vote the outcome is integer-valued
      (``post_processing.maybe_snap_to_integers`` → ``discrete_snap.snap_distribution_to_integers``).
      That per-forecaster ``outcome_type`` vote is prod-side state absent from both the
      resolved record and the ghost payload, and the snap is not reliably recoverable
      from the published CDF's shape (peaked distributions get smeared back toward smooth
      by the max-step cap, so a snapped CDF can be indistinguishable from a smooth one).
      We therefore score the ghost as the smooth distribution it declared — a documented
      approximation. Residual effect: on the integer-outcome minority the snapped
      published forecast concentrates a little extra mass on the resolution bucket, so
      those deltas are mildly biased against the ghost; it is bounded and affects only
      that minority, not the continuous questions that make up the bulk of the gate.
    """
    qid = ghost["qid"]
    percentiles = _numeric_percentiles(ghost)
    if percentiles is None:
        reason = "legacy_median_only" if ghost["source"] == "legacy" else "no_declared_percentiles"
        return {"qid": qid, "scoreable": False, "reason": reason}

    published_cdf = record.get("our_forecast_values")
    if not published_cdf or len(published_cdf) < 2:
        return {"qid": qid, "scoreable": False, "reason": "no_published_cdf"}

    # Lazy imports: the PCHIP builder pulls numpy/scipy and the collector helper drags
    # the collector's heavy import chain (requests, env loading). Keep the n=0 path
    # dependency-light — numeric scoring only runs once real records join.
    from metaculus_bot.numeric.config import (  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import  # lazy: pair the grid-step rule with the CDF builder it feeds
        grid_step_constraints,
    )
    from metaculus_bot.numeric.pchip_cdf import (  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import  # lazy: keep the n=0 path free of numpy/scipy
        generate_pchip_cdf,
    )
    from metaculus_bot.performance_analysis.collector import (  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import  # lazy: keep the pure scoring core decoupled from the collector's import chain
        resolve_numeric_record_to_score_inputs,
    )

    score_inputs = resolve_numeric_record_to_score_inputs(record)
    if score_inputs is None:
        return {"qid": qid, "scoreable": False, "reason": "no_score_inputs"}
    res_float, lower, upper, zero_point = score_inputs
    open_lower = bool(record.get("open_lower_bound", False))
    open_upper = bool(record.get("open_upper_bound", False))

    # generate_pchip_cdf expects percentile keys in (0, 100); ghosts carry fraction
    # keys in [0, 1]. Match the ghost grid to the published grid length so both score
    # with identical PMF bucketing; grid_step_constraints scales BOTH the min and max
    # per-bin step to that length, so on a native-discrete grid (num_points < 201) the
    # ghost isn't clipped by the 201-grid 0.2 max-step while the (prod-built) published
    # side stays uncapped — that asymmetry biases the paired score against the ghost.
    pct_values = {frac * 100.0: value for frac, value in percentiles.items()}
    num_points = len(published_cdf)
    min_step, max_step = grid_step_constraints(num_points)
    try:
        ghost_cdf, _ = generate_pchip_cdf(
            pct_values,
            open_upper,
            open_lower,
            upper,
            lower,
            zero_point,
            min_step=min_step,
            max_step=max_step,
            num_points=num_points,
        )
    except (ValueError, RuntimeError):
        return {"qid": qid, "scoreable": False, "reason": "cdf_build_failed"}

    try:
        ghost_ls = numeric_log_score(ghost_cdf, res_float, lower, upper, open_lower, open_upper, zero_point)
        published_ls = numeric_log_score(
            list(published_cdf), res_float, lower, upper, open_lower, open_upper, zero_point
        )
    except (ValueError, ZeroDivisionError):
        return {"qid": qid, "scoreable": False, "reason": "score_failed"}

    return {
        "qid": qid,
        "scoreable": True,
        "reason": None,
        "resolution": res_float,
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


def _pre_identity(ghost: dict, pre_by_qid: dict[int, dict]) -> bool | None:
    """Whether the concluding ghost is byte-identical to the pre-research dry run.

    ``True``/``False`` only when both halves are JSON payloads FROM THE SAME RUN
    (the two markers serialize through the same ``_summarize_ghost`` path, so dict
    equality IS the byte-identity the split needs). ``None`` when there is nothing
    to compare: no ``GHOST_PRE_JSON`` for this qid, a legacy summary-only ghost
    (the pre marker and the JSON ghost landed in the same merge, so a legacy ghost
    predates both), or a pre-ghost from a DIFFERENT run — a run can emit one
    marker without the other (a schema-invalid dry run suppresses GHOST_PRE_JSON;
    a deadline-hit run can bank a pre with no concluding ghost), and comparing
    across runs would file a cross-run pair into ``pre_identical``/``loop_moved``.
    """
    if ghost["source"] != "json":
        return None
    pre = pre_by_qid.get(ghost["qid"])
    if pre is None or pre.get("run_id") != ghost.get("run_id"):
        return None
    return pre["payload"] == ghost["payload"]


def _split_bucket(pre_identical: bool | None) -> str:
    if pre_identical is None:
        return "no_pre_marker"
    return "pre_identical" if pre_identical else "loop_moved"


def join_and_score(
    json_ghosts: list[dict],
    legacy_ghosts: list[dict],
    records: list[dict],
    pre_ghosts: list[dict] | None = None,
) -> dict:
    """Join the latest ghost per qid to resolved records and compute paired log-score deltas.

    JSON-source ghosts (full forecast) are preferred over legacy summary-only ghosts
    for the same qid. Binary, MC, and — new with the JSON marker — numeric ghosts are
    all scoreable. Everything is pure/in-memory so the n=0 path (no resolved v2-era
    questions yet) is exercised in tests.

    Join key is ``post_id``, NOT ``question_id``. A ghost's qid is parsed by
    ``qid_from_ref`` from the marker's ``question=`` field, which the gap-fill v2 seam
    sets to ``question.page_url`` (``.../questions/{post_id}``) — a Metaculus POST id.
    The collector keys ``question_id`` on the sub-question id (``q["id"]``) and emits
    ``post_id`` separately; the two id spaces are disjoint on real data, so keying on
    ``question_id`` here would make every join silently miss. (Group/conditional posts
    hold several sub-question records under one ``post_id``; both the ghost marker and
    this dict collapse those to one — a known limitation of the post-level ghost ref,
    not something the join key can resolve.)
    """
    records_by_post_id = {r.get("post_id"): r for r in records}
    selected = _select_ghost_per_qid(json_ghosts, legacy_ghosts)
    # Latest GHOST_PRE_JSON per qid (same post-id space as the concluding ghosts —
    # both markers carry the same log_prefix ref), normalized to comparable payloads.
    pre_by_qid: dict[int, dict] = {}
    for qid, record in _latest_ghost_per_qid(pre_ghosts or []).items():
        normalized = _normalize_json_ghost(record)
        if normalized is not None:
            pre_by_qid[qid] = normalized

    source_counts: dict[str, int] = {"json": 0, "legacy": 0}
    for ghost in selected.values():
        source_counts[ghost["source"]] = source_counts.get(ghost["source"], 0) + 1

    binary_rows: list[dict] = []
    mc_rows: list[dict] = []
    numeric_rows: list[dict] = []
    numeric_unscoreable: dict[str, int] = {}
    numeric_joined = 0
    n_joined = 0
    split_rows: dict[str, list[dict]] = {"pre_identical": [], "loop_moved": [], "no_pre_marker": []}

    def _record_scored_row(ghost: dict, row: dict) -> None:
        row["pre_identical"] = _pre_identity(ghost, pre_by_qid)
        split_rows[_split_bucket(row["pre_identical"])].append(row)

    for qid, ghost in selected.items():
        record = records_by_post_id.get(qid)
        if record is None:
            continue
        n_joined += 1
        qtype = ghost.get("qtype")
        if qtype == "binary":
            row = _score_binary(ghost, record)
            if row is not None:
                binary_rows.append(row)
                _record_scored_row(ghost, row)
        elif qtype == "multiple_choice":
            row = _score_mc(ghost, record)
            if row is not None:
                mc_rows.append(row)
                _record_scored_row(ghost, row)
        elif qtype == "numeric":
            numeric_joined += 1
            outcome = _score_numeric(ghost, record)
            if outcome["scoreable"]:
                numeric_rows.append(outcome)
                _record_scored_row(ghost, outcome)
            else:
                reason = outcome["reason"]
                numeric_unscoreable[reason] = numeric_unscoreable.get(reason, 0) + 1

    return {
        "n_ghosts": len(selected),
        "n_joined": n_joined,
        "n_scored": len(binary_rows) + len(mc_rows) + len(numeric_rows),
        "source_counts": source_counts,
        "binary": _summarize_rows(binary_rows),
        "multiple_choice": _summarize_rows(mc_rows),
        "numeric": {
            **_summarize_rows(numeric_rows),
            "n_joined": numeric_joined,
            "n_unscoreable": sum(numeric_unscoreable.values()),
            "unscoreable_reasons": numeric_unscoreable,
        },
        # Pooled across types, keyed on whether the concluding ghost equals the
        # pre-research dry run. The identical bucket measures the driver's PRIOR
        # (the loop's findings never moved its own number), so only the loop_moved
        # bucket says anything about v2's research — see the module docstring.
        "split_by_pre_identity": {
            bucket: {"n": len(rows), "mean_delta": mean(r["delta"] for r in rows) if rows else None}
            for bucket, rows in split_rows.items()
        },
    }


def render_report(summary: dict) -> str:
    """Human-readable summary. A positive mean delta = ghost out-scores published."""
    source_counts = summary["source_counts"]
    lines = [
        "=== Ghost-forecast scoring (gap-fill v2 ghost vs published) ===",
        f"Ghosts (latest per qid): {summary['n_ghosts']}",
        f"  by source: json={source_counts.get('json', 0)} legacy={source_counts.get('legacy', 0)}",
        f"Joined to resolved-question dataset: {summary['n_joined']}",
    ]
    if summary["n_scored"] == 0:
        lines.append("Scored ghosts: n=0 — waiting on resolutions (v2 live in prod 2026-07-21; expected today).")
    else:
        lines.append(f"Scored ghosts: {summary['n_scored']}")
    for qtype in ("binary", "multiple_choice", "numeric"):
        block = summary[qtype]
        if block["n"]:
            delta = block["mean_delta"]
            verdict = "ghost better" if delta > 0 else "published better"
            lines.append(f"  {qtype}: n={block['n']} mean_delta={delta:+.4f} ({verdict})")
    numeric = summary["numeric"]
    if numeric["n_joined"]:
        lines.append(
            f"  numeric coverage: {numeric['n_joined']} joined, "
            f"{numeric['n']} scored, {numeric['n_unscoreable']} unscoreable"
        )
        for reason, count in sorted(numeric["unscoreable_reasons"].items()):
            lines.append(f"    - {reason}: {count}")
    split = summary.get("split_by_pre_identity")
    if split and summary["n_scored"]:
        bucket_labels = {
            "pre_identical": "byte-identical to the pre-research dry run (measures the driver's prior)",
            "loop_moved": "loop moved the driver's forecast (the only bucket that measures v2's research)",
            "no_pre_marker": "no GHOST_PRE_JSON to compare (predates the marker, or legacy ghost)",
        }
        lines.append("Ghost vs pre-research dry run (pooled across types — do not quote the pooled delta):")
        for bucket, label in bucket_labels.items():
            block = split[bucket]
            if not block["n"]:
                continue
            lines.append(f"  {label}: n={block['n']} mean_delta={block['mean_delta']:+.4f}")
    return "\n".join(lines)


def _load_records(perf_json: str | None, tournament: str | None) -> list[dict]:
    """Load resolved-question records from a perf JSON, else a live read-only pull, else []."""
    if perf_json:
        from metaculus_bot.performance_analysis.collector import (
            load_dataset,
        )

        records = load_dataset(perf_json)
        logger.info(f"Loaded {len(records)} records from {perf_json}")
        return records
    if tournament:
        from metaculus_bot.api_preflight import (
            verify_metaculus_api_identity,
        )
        from metaculus_bot.performance_analysis.collector import (
            build_performance_dataset,
        )

        # Confirm the host is the real Metaculus before the token-sending pull.
        verify_metaculus_api_identity()
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

    json_ghosts = load_marker_records(Path(args.archive_dir), "ghost_forecast_json")
    legacy_ghosts = load_marker_records(Path(args.archive_dir), "ghost_forecast")
    pre_ghosts = load_marker_records(Path(args.archive_dir), "ghost_pre_json")
    logger.info(
        f"Loaded {len(json_ghosts)} ghost_forecast_json + {len(legacy_ghosts)} legacy ghost_forecast "
        f"+ {len(pre_ghosts)} ghost_pre_json record(s) from {args.archive_dir}"
    )

    records = _load_records(args.perf_json, args.tournament)
    summary = join_and_score(json_ghosts, legacy_ghosts, records, pre_ghosts=pre_ghosts)
    print(render_report(summary))

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
        logger.info(f"Summary written to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
