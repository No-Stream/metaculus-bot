#!/usr/bin/env python3
"""
Standalone script to analyze correlations from existing benchmark results.

Usage:
    python analyze_correlations.py benchmarks/benchmarks_2025-08-10_15-04-51.jsonl
    python analyze_correlations.py benchmarks/ --max-cost 0.3 --max-size 3

Or via Makefile:
    make analyze_correlations FILE=benchmarks/benchmarks_2025-08-10_15-04-51.jsonl
    make analyze_correlations_latest
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot
from pydantic import ValidationError

from metaculus_bot.ensemble_analysis.correlation_analysis import CorrelationAnalyzer
from metaculus_bot.scoring_patches import apply_scoring_patches
from metaculus_bot.scoring_patches import calculate_multiple_choice_baseline_score as _score_mc
from metaculus_bot.scoring_patches import calculate_numeric_baseline_score as _score_num

logger = logging.getLogger(__name__)

# Standard question-type order used for every per-type display.
_QTYPE_ORDER = ("binary", "numeric", "multiple_choice")


def extract_timestamp_from_filename(filepath: str) -> str | None:
    """Extract timestamp from benchmark filename like 'benchmarks_2025-08-10_15-04-51.jsonl'"""
    filename = Path(filepath).name
    # Match pattern: benchmarks_YYYY-MM-DD_HH-MM-SS
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", filename)
    return match.group(1) if match else None


def _load_benchmarks_from_file(path: Path) -> list[BenchmarkForBot]:
    """Parse one benchmark file (.jsonl = one per line, .json = object or list).

    An unreadable or malformed file yields no benchmarks rather than raising: a
    directory scan must survive one bad file, since this is an analysis-only script.
    """
    try:
        with open(path) as f:
            if path.suffix == ".jsonl":
                payloads = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                payloads = data if isinstance(data, list) else [data]
        return [BenchmarkForBot.model_validate(payload) for payload in payloads]
    except (OSError, json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Could not load {path}: {e}")
        return []


def _load_benchmarks_from_dir(path: Path) -> list[BenchmarkForBot]:
    """Load every .json / .jsonl benchmark file in a directory.

    Skips this script's own ``correlation_*`` report outputs, which live alongside the
    benchmarks and are not benchmark payloads.
    """
    benchmarks: list[BenchmarkForBot] = []
    for pattern in ("*.json", "*.jsonl"):
        for json_file in path.glob(pattern):
            if json_file.name.startswith("correlation_"):
                continue
            benchmarks.extend(_load_benchmarks_from_file(json_file))
    return benchmarks


def load_benchmarks_from_path(benchmark_path: str) -> list[BenchmarkForBot]:
    """Load benchmark data from a file or directory."""
    path = Path(benchmark_path)
    if path.is_file():
        benchmarks = _load_benchmarks_from_file(path)
    elif path.is_dir():
        benchmarks = _load_benchmarks_from_dir(path)
    else:
        logger.error(f"Path does not exist: {benchmark_path}")
        return []

    logger.info(f"Loaded {len(benchmarks)} benchmarks from {benchmark_path}")
    return benchmarks


class _NumericFallbackFilter(logging.Filter):
    """Suppress noisy numeric-fallback warnings from scoring_patches while counting them.

    Analysis-only: keeps the console clean during a correlation run but records how many
    lines were suppressed and which question ids triggered them so the summary can report it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.suppressed_lines = 0
        self.qids: set[str] = set()
        self._pat_qid1 = re.compile(r"Numeric Question (\d+)")
        self._pat_qid2 = re.compile(r"Numeric q=(\d+)")

    def filter(self, record: logging.LogRecord) -> bool:  # True keeps, False drops
        if record.name == "metaculus_bot.scoring_patches":
            msg = str(record.getMessage())
            low = msg.lower()
            if ("cannot compute model cdf" in low) or ("using percentile fallback" in low):
                self.suppressed_lines += 1
                m = self._pat_qid1.search(msg) or self._pat_qid2.search(msg)
                if m:
                    self.qids.add(m.group(1))
                return False
        return True


# --- Shared identity / classification helpers (route through analyzer wrappers) ---------------


def _model_name_for(analyzer: CorrelationAnalyzer, benchmark: Any) -> str:
    """Clean model name for a benchmark."""
    return str(analyzer._extract_model_name(benchmark))  # type: ignore[attr-defined]


def _is_stacking_bench(analyzer: CorrelationAnalyzer, benchmark: Any) -> bool:
    """True if the benchmark used STACKING aggregation."""
    return bool(analyzer._is_stacking_benchmark(benchmark))  # type: ignore[attr-defined]


def _detect_type_from_report(analyzer: CorrelationAnalyzer, report: Any) -> str:
    """Question type for a report; delegates to analyzer's helper (avoids touching .cdf)."""
    return analyzer._get_question_type(report)  # type: ignore[attr-defined]


def _summary_stats(vals: list[float]) -> dict[str, float]:
    """n / mean / mean|abs| / min / max for a list of scores (NaN-filled when empty)."""
    if not vals:
        return {
            "n": 0,
            "mean": float("nan"),
            "mean_abs": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    arr = np.array(vals, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "mean_abs": float(np.mean(np.abs(arr))),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _summarize_buckets(buckets: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Apply :func:`_summary_stats` to each {qtype: [scores]} bucket."""
    return {qtype: _summary_stats(vals) for qtype, vals in buckets.items()}


def _print_type_stats(summary: dict[str, dict[str, float]], indent: str = "") -> None:
    """Print n/mean/mean|abs|/min/max for each question type present in ``summary``."""
    for qtype in _QTYPE_ORDER:
        st = summary.get(qtype)
        if not st or st.get("n", 0) == 0:
            continue
        print(
            f"{indent}{qtype:16} n={int(st['n']):4d} | mean={st['mean']:7.2f} | "
            f"mean|score|={st['mean_abs']:7.2f} | min={st['min']:7.2f} | max={st['max']:7.2f}"
        )


def _aggregate(values: list[float], agg: str) -> float:
    """Mean or median of ``values``, per the ``agg`` label the ensemble was built with."""
    return float(np.mean(values)) if agg == "mean" else float(np.median(values))


def _aggregate_binary_score(
    analyzer: CorrelationAnalyzer,
    m2r: dict[str, Any],
    models_list: list[str],
    agg: str,
) -> float | None:
    """Community log score of the aggregated binary probability, or None if unscoreable.

    DEPRECATED data path: ``community_prediction_at_access_time`` is always None for
    newly-fetched questions (Metaculus removed aggregations from the list API), so this
    only scores archived benchmarks that still carry it.
    """
    del analyzer  # analyzer hooks aren't needed for the binary path
    rep0 = next(iter(m2r.values()))
    community = getattr(rep0.question, "community_prediction_at_access_time", None)
    if community is None:
        return None
    agg_p = _aggregate([float(m2r[m].prediction) for m in models_list], agg)
    p = max(0.001, min(0.999, agg_p))
    return float(100.0 * (community * (np.log2(p) + 1.0) + (1.0 - community) * (np.log2(1.0 - p) + 1.0)))


def _aggregate_mc_option_probs(m2r: dict[str, Any], models_list: list[str], agg: str) -> tuple[list[str], list[float]]:
    """Option names (first model's ballot order) and the renormalized aggregate probabilities.

    Options are matched BY NAME across models, not by position, since ballots can arrive
    in different orders. An option no model quoted contributes 0.0, and a ballot that
    sums to 0 degrades to uniform rather than dividing by zero.
    """
    first = m2r[models_list[0]].prediction
    option_names = [getattr(o, "option_name", str(o)) for o in first.predicted_options]
    agg_probs: list[float] = []
    for name in option_names:
        values: list[float] = []
        for model in models_list:
            for opt in m2r[model].prediction.predicted_options:
                if getattr(opt, "option_name", str(opt)) == name:
                    values.append(float(getattr(opt, "probability", 0.0)))
                    break
        agg_probs.append(_aggregate(values, agg) if values else 0.0)
    total = sum(agg_probs)
    if total > 0:
        return option_names, [p / total for p in agg_probs]
    return option_names, [1.0 / len(option_names)] * len(option_names)


def _aggregate_mc_score(
    analyzer: CorrelationAnalyzer,
    m2r: dict[str, Any],
    models_list: list[str],
    agg: str,
) -> float | None:
    """Baseline score of the aggregated MC ballot, or None when there's nothing to score."""
    del analyzer  # analyzer hooks aren't needed for the MC path
    rep0 = next(iter(m2r.values()))
    first = m2r[models_list[0]].prediction
    if not hasattr(first, "predicted_options") or not first.predicted_options:
        return None
    option_names, agg_probs = _aggregate_mc_option_probs(m2r, models_list, agg)
    pred_obj = SimpleNamespace(
        predicted_options=[
            SimpleNamespace(option_name=name, probability=prob)
            for name, prob in zip(option_names, agg_probs, strict=True)
        ]
    )
    score = _score_mc(SimpleNamespace(question=rep0.question, prediction=pred_obj))
    return float(score) if score is not None else None


def _aggregate_numeric_score(
    analyzer: CorrelationAnalyzer,
    m2r: dict[str, Any],
    models_list: list[str],
    agg: str,
) -> float | None:
    """Baseline score of the pointwise-aggregated CDF, or None when any model's CDF is missing.

    One unrecoverable CDF disqualifies the whole question rather than shrinking the
    ensemble silently. Grids are truncated to the shortest one before aggregating, so a
    longer grid's extra points are dropped rather than zero-padded.
    """
    rep0 = next(iter(m2r.values()))
    cdfs: list[list[Any]] = []
    for model in models_list:
        cdf = analyzer._get_safe_numeric_cdf(model, rep0.question, m2r[model].prediction)  # type: ignore[attr-defined]
        if cdf is None:
            return None
        cdfs.append(cdf)
    if not cdfs:
        return None

    min_len = min(len(c) for c in cdfs)
    cdfs = [c[:min_len] for c in cdfs]
    percs = np.array([[float(getattr(pt, "percentile", 0.0)) for pt in c] for c in cdfs])
    agg_percs = percs.mean(axis=0) if agg == "mean" else np.median(percs, axis=0)
    x = [float(getattr(pt, "value", i)) for i, pt in enumerate(cdfs[0][:min_len])]
    pred_obj = SimpleNamespace(
        cdf=[SimpleNamespace(value=xi, percentile=float(pi)) for xi, pi in zip(x, agg_percs, strict=True)]
    )
    score = _score_num(SimpleNamespace(question=rep0.question, prediction=pred_obj))
    return float(score) if score is not None else None


def _ensemble_per_type(
    analyzer: CorrelationAnalyzer,
    benches_filtered: list[Any],
    models_list: list[str],
    agg: str,
) -> dict[str, dict[str, float]]:
    """Aggregate-and-score per question across ``models_list``, broken down by question type.

    Builds a qid -> {model: report} index over the questions every model in the list answered,
    aggregates predictions per question (mean or median per ``agg``), scores each aggregate with
    the appropriate baseline scorer, and returns {qtype: summary-stats}. Skips questions any
    model is missing or that lack the data needed to score.
    """
    qmap: dict[int, dict[str, Any]] = {}
    for b in benches_filtered:
        m = _model_name_for(analyzer, b)
        if m not in models_list:
            continue
        for r in b.forecast_reports:
            qid = getattr(getattr(r, "question", None), "id_of_question", None)
            if not isinstance(qid, int):
                continue
            qmap.setdefault(qid, {})[m] = r

    scorers = {
        "binary": _aggregate_binary_score,
        "multiple_choice": _aggregate_mc_score,
        "numeric": _aggregate_numeric_score,
    }
    stats: dict[str, list[float]] = {"binary": [], "numeric": [], "multiple_choice": []}
    for m2r in qmap.values():
        if any(m not in m2r for m in models_list):
            continue  # need all models
        rep0 = next(iter(m2r.values()))
        qtype = _detect_type_from_report(analyzer, rep0)
        scorer = scorers.get(qtype)
        if scorer is None:
            continue
        score = scorer(analyzer, m2r, models_list, agg)
        if score is not None:
            stats[qtype].append(score)

    return {qt: _summary_stats(vals) for qt, vals in stats.items() if vals}


# --- Pipeline stages ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze model correlations from benchmark results")
    parser.add_argument("benchmark_path", help="Path to benchmark file (.json/.jsonl) or directory")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for correlation report (default: correlation_analysis.md)",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=1.0,
        help="Maximum cost per question for ensemble recommendations",
    )
    parser.add_argument("--max-size", type=int, default=7, help="Maximum ensemble size")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--question-types",
        nargs="*",
        choices=["binary", "numeric", "multiple_choice"],
        help="Filter analysis to specific question types",
    )
    parser.add_argument(
        "--score-stats",
        dest="score_stats",
        action="store_true",
        default=True,
        help="Print score scaling stats by question type (default: on)",
    )
    parser.add_argument(
        "--no-score-stats",
        dest="score_stats",
        action="store_false",
        help="Disable printing score scaling stats",
    )
    parser.add_argument(
        "--score-stats-per-question",
        action="store_true",
        default=False,
        help="Also compute per-question stats (average across models per question)",
    )
    parser.add_argument(
        "--score-stats-json",
        type=str,
        default=None,
        help="Optional path to write score stats JSON (includes per-report and per-question if requested)",
    )
    parser.add_argument(
        "--model-stats-json",
        type=str,
        default=None,
        help="Optional path to write per-model, per-type score stats JSON",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=None,
        help=("Exclude models by substring match (case-insensitive). Example: --exclude-models grok-4 gemini-2.5-pro"),
    )
    parser.add_argument(
        "--include-models",
        nargs="*",
        default=None,
        help=(
            "Only include models matching these substrings (case-insensitive). "
            "Mutually exclusive with --exclude-models."
        ),
    )
    return parser.parse_args()


def _load_and_prepare(args: argparse.Namespace) -> tuple[CorrelationAnalyzer, _NumericFallbackFilter]:
    """Load benchmarks, apply scoring patches + log filter, build analyzer, apply model filters.

    Exits the process (matching prior behavior) when too few benchmarks load, the include/exclude
    flags conflict, or fewer than two models remain after filtering.
    """
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load benchmarks
    try:
        benchmarks = load_benchmarks_from_path(args.benchmark_path)
    except Exception as e:  # noqa: BLE001  # CLI entry boundary: any load failure becomes exit 1 with a readable message
        logger.error(f"Failed to load benchmarks: {e}")
        sys.exit(1)

    if len(benchmarks) < 2:
        logger.error("Need at least 2 benchmark results for correlation analysis")
        sys.exit(1)

    # Apply scoring patches for mixed question types
    apply_scoring_patches()

    # Suppress (but count) noisy numeric fallback warnings from scoring_patches.
    fallback_filter = _NumericFallbackFilter()
    logging.getLogger("metaculus_bot.scoring_patches").addFilter(fallback_filter)

    # Perform analysis
    analyzer = CorrelationAnalyzer()
    analyzer.add_benchmark_results(benchmarks)

    _apply_model_filters(analyzer, args)
    return analyzer, fallback_filter


def _apply_model_filters(analyzer: CorrelationAnalyzer, args: argparse.Namespace) -> None:
    """Apply --include-models / --exclude-models in place, reporting what matched.

    Exits the process on conflicting flags (2) or when fewer than two models survive (1).
    """
    if args.include_models and args.exclude_models:
        logger.error("--include-models and --exclude-models are mutually exclusive")
        sys.exit(2)

    filter_summary = analyzer.filter_models_inplace(include=args.include_models, exclude=args.exclude_models)
    if args.include_models or args.exclude_models:
        print("Applied model filters:")
        if args.include_models:
            print(f"  include tokens: {args.include_models}")
        if args.exclude_models:
            print(f"  exclude tokens: {args.exclude_models}")
        for label in ("includes", "excludes"):
            unmatched = filter_summary.get(f"unmatched_{label}", [])
            if unmatched:
                print(f"  unmatched {label[:-1]} tokens: {unmatched}")

    try:
        remaining_models = analyzer.get_model_names()  # type: ignore[attr-defined]
    except Exception:
        # An analyzer build without this hook just skips the >=2-models guard.
        logger.debug("analyzer.get_model_names() failed", exc_info=True)
        remaining_models = None

    if isinstance(remaining_models, (list, tuple, set)) and len(remaining_models) < 2:
        logger.error(
            f"Analysis requires ≥2 models after filtering. Remaining: {remaining_models if remaining_models else 'none'}"
        )
        sys.exit(1)


def _per_report_score_buckets(analyzer: CorrelationAnalyzer, benchmarks: list[Any]) -> dict[str, list[float]]:
    """Every report's baseline score, bucketed by question type."""
    buckets: dict[str, list[float]] = {"binary": [], "numeric": [], "multiple_choice": []}
    for benchmark in benchmarks:
        for report in benchmark.forecast_reports:
            score = getattr(report, "expected_baseline_score", None)
            if score is None:
                continue
            buckets.setdefault(_detect_type_from_report(analyzer, report), []).append(float(score))
    return buckets


def _per_question_score_buckets(analyzer: CorrelationAnalyzer, benchmarks: list[Any]) -> dict[str, list[float]]:
    """Per-question mean baseline score across models, bucketed by question type.

    Averaging within a question first keeps a question every model answered from
    outweighing one only a couple of models reached.
    """
    per_q: dict[tuple[int, str], list[float]] = {}
    for benchmark in benchmarks:
        for report in benchmark.forecast_reports:
            score = getattr(report, "expected_baseline_score", None)
            qid = getattr(getattr(report, "question", None), "id_of_question", None)
            if score is None or qid is None:
                continue
            per_q.setdefault((int(qid), _detect_type_from_report(analyzer, report)), []).append(float(score))

    buckets: dict[str, list[float]] = {"binary": [], "numeric": [], "multiple_choice": []}
    for (_qid, qtype), values in per_q.items():
        if values:
            buckets.setdefault(qtype, []).append(float(np.mean(values)))
    return buckets


def _write_stats_json(path: str, blob: dict[str, Any], *, label: str) -> None:
    """Write an analysis stats blob, WARNing instead of aborting when the write fails."""
    try:
        with open(path, "w") as f:
            json.dump(blob, f, indent=2)
    except (OSError, TypeError) as e:
        logger.warning(f"Failed to write {label.lower()} JSON: {e}")
        return
    print(f"\n{label} written to: {path}")


def _compute_score_stats(analyzer: CorrelationAnalyzer, args: argparse.Namespace) -> None:
    """Print per-report (and optionally per-question) score-scaling stats; optional JSON export."""
    if not args.score_stats:
        return

    benches_filtered = getattr(analyzer, "benchmarks", [])

    per_report_summary = _summarize_buckets(_per_report_score_buckets(analyzer, benches_filtered))
    print("\n" + "=" * 60)
    print("SCORE SCALING (After Filters) — Per-Report")
    print("=" * 60)
    _print_type_stats(per_report_summary)

    per_question_summary: dict[str, dict[str, float]] | None = None
    if args.score_stats_per_question:
        per_question_summary = _summarize_buckets(_per_question_score_buckets(analyzer, benches_filtered))
        print("\nSCORE SCALING — Per-Question (average across models per question)")
        print("-" * 60)
        _print_type_stats(per_question_summary)

    if args.score_stats_json:
        blob: dict[str, Any] = {"per_report": per_report_summary}
        if per_question_summary is not None:
            blob["per_question"] = per_question_summary
        _write_stats_json(args.score_stats_json, blob, label="Score stats")


def _compute_model_stats(analyzer: CorrelationAnalyzer, args: argparse.Namespace) -> None:
    """Print per-model, per-type score stats (excluding stacking benchmarks); optional JSON export."""
    try:
        per_model = _per_model_type_stats(analyzer, getattr(analyzer, "benchmarks", []))
        if per_model:
            print("\n" + "=" * 60)
            print("MODEL STATS BY TYPE (After Filters)")
            print("=" * 60)
            for mname in sorted(per_model.keys()):
                print(f"\n{mname}")
                _print_type_stats(per_model[mname], indent="  ")

        if args.model_stats_json:
            _write_stats_json(args.model_stats_json, per_model, label="Model stats")
    except Exception as e:  # noqa: BLE001  # soft-fail boundary: one diagnostic section must not abort an analysis run
        logger.warning(f"Failed to compute per-model stats: {e}")


def _per_model_type_stats(
    analyzer: CorrelationAnalyzer,
    benchmarks: list[Any],
) -> dict[str, dict[str, dict[str, float]]]:
    """model -> question type -> summary stats, excluding stacking benchmarks.

    Stacking benchmarks are excluded because their score belongs to the stacker, not to
    any one base model, so pooling them would mix a stacker's row into a model's.
    """
    buckets: dict[tuple[str, str], list[float]] = {}
    for benchmark in benchmarks:
        if _is_stacking_bench(analyzer, benchmark):
            continue
        mname = _model_name_for(analyzer, benchmark)
        for report in benchmark.forecast_reports:
            score = getattr(report, "expected_baseline_score", None)
            if score is None:
                continue
            buckets.setdefault((mname, _detect_type_from_report(analyzer, report)), []).append(float(score))

    per_model: dict[str, dict[str, dict[str, float]]] = {}
    for (mname, qtype), values in buckets.items():
        per_model.setdefault(mname, {})[qtype] = _summary_stats(values)
    return per_model


def _resolve_output_path(args: argparse.Namespace) -> str | Path:
    """Resolve the report output path: explicit --output, else timestamped name beside input."""
    if args.output:
        return args.output

    benchmark_path = Path(args.benchmark_path)
    timestamp = extract_timestamp_from_filename(args.benchmark_path)

    if timestamp:
        filename = f"correlation_analysis_{timestamp}.md"
    else:
        # Fallback to current timestamp if can't extract from input
        current_timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"correlation_analysis_{current_timestamp}.md"

    if benchmark_path.is_file():
        return benchmark_path.parent / filename
    return benchmark_path / filename


def _print_report(analyzer: CorrelationAnalyzer, output_file: str | Path) -> str:
    """Generate + print the correlation report, returning the report text."""
    report = analyzer.generate_correlation_report(str(output_file))
    print("=" * 60)
    print("CORRELATION ANALYSIS RESULTS")
    print("=" * 60)
    print(report)
    return report


def _run_ablation(analyzer: CorrelationAnalyzer, args: argparse.Namespace) -> None:
    """Print ensemble recommendations plus per-type and leave-one-out ablations for the top K."""
    print("\n" + "=" * 60)
    print("ENSEMBLE RECOMMENDATIONS")
    print("=" * 60)

    optimal_ensembles = analyzer.find_optimal_ensembles(
        max_ensemble_size=args.max_size, max_cost_per_question=args.max_cost
    )

    if not optimal_ensembles:
        print("No ensembles found meeting the cost constraint.")
        return

    print(f"\nTop 10 Ensembles (Both Aggregation Strategies, Cost ≤ ${args.max_cost}/question):")
    for i, ensemble in enumerate(optimal_ensembles[:10], 1):
        models = " + ".join(ensemble.model_names)
        print(f"{i:2}. {models} ({ensemble.aggregation_strategy.upper()})")
        print(
            f"    Score: {ensemble.avg_performance:.2f} | "
            f"Cost: ${ensemble.avg_cost:.3f} | "
            f"Diversity: {ensemble.diversity_score:.3f} | "
            f"Overall: {ensemble.ensemble_score:.3f}"
        )

    # Ablations and per-type diagnostics for top K
    try:
        benches_filtered = getattr(analyzer, "benchmarks", [])
        top_k = min(5, len(optimal_ensembles))
        print(f"\nENSEMBLE ABLATIONS (Top {top_k} by Overall)")
        for idx in range(top_k):
            _print_ensemble_ablation(analyzer, benches_filtered, optimal_ensembles[idx], rank=idx + 1)
    except Exception as e:  # noqa: BLE001  # soft-fail boundary: one diagnostic section must not abort an analysis run
        logger.warning(f"Failed to compute ensemble ablations: {e}")


def _model_qid_sets(analyzer: CorrelationAnalyzer, benchmarks: list[Any], models: list[str]) -> dict[str, set[int]]:
    """The set of question ids each named model actually answered."""
    qsets: dict[str, set[int]] = {}
    for benchmark in benchmarks:
        model = _model_name_for(analyzer, benchmark)
        if model not in models:
            continue
        answered = qsets.setdefault(model, set())
        answered.update(
            qid
            for qid in (
                getattr(getattr(report, "question", None), "id_of_question", None)
                for report in benchmark.forecast_reports
            )
            if isinstance(qid, int)
        )
    return qsets


def _print_ensemble_ablation(
    analyzer: CorrelationAnalyzer,
    benchmarks: list[Any],
    ensemble: Any,
    *,
    rank: int,
) -> None:
    """Print one ensemble's baseline, per-type split, and leave-one-out impacts.

    ``Q=`` counts are the INTERSECTION of the members' answered questions, so dropping a
    model can raise the question count as well as change the score — both are reported.
    """
    base_models = list(ensemble.model_names)
    agg = ensemble.aggregation_strategy
    base_score = analyzer._simulate_ensemble_performance(base_models, agg)  # type: ignore[attr-defined]

    qsets = _model_qid_sets(analyzer, benchmarks, base_models)
    inter_base = set.intersection(*(qsets[m] for m in base_models)) if base_models else set()
    print(f"\n{rank}. {' + '.join(base_models)} ({agg.upper()})  | baseline={base_score:.2f} | Q={len(inter_base)}")

    per_type = _ensemble_per_type(analyzer, benchmarks, base_models, agg)
    if per_type:
        print("  per-type:")
        _print_type_stats(per_type, indent="    ")

    contribs = []
    for model in base_models:
        subset = [x for x in base_models if x != model]
        score_wo = analyzer._simulate_ensemble_performance(subset, agg)  # type: ignore[attr-defined]
        subset_q = len(set.intersection(*(qsets[x] for x in subset))) if subset else 0
        contribs.append((model, score_wo, base_score - score_wo, subset_q))
    contribs.sort(key=lambda row: row[2], reverse=True)
    print("  leave-one-out impacts (Δscore):")
    for model, score_wo, delta, subset_q in contribs:
        print(f"    - {model:20} Δ={delta:+6.2f} | score_wo={score_wo:6.2f} | Q={subset_q}")


def _print_correlation_highlights(analyzer: CorrelationAnalyzer, has_mixed: bool) -> None:
    """Print most-independent and most-correlated model pairs from the correlation matrix."""
    if has_mixed:
        corr_matrix = analyzer.calculate_correlation_matrix_by_components()
    else:
        corr_matrix = analyzer.calculate_correlation_matrix()
    print(f"\n{'-' * 40}")
    print("CORRELATION HIGHLIGHTS")
    print(f"{'-' * 40}")

    least_correlated = corr_matrix.get_least_correlated_pairs(threshold=0.8)
    print("\nMost Independent Model Pairs:")
    for model1, model2, corr in least_correlated[:8]:
        print(f"  {model1:20} ↔ {model2:20} | r = {corr:6.3f}")

    # Also show most correlated pairs (by absolute r), excluding self and near-1.0
    try:
        pm = corr_matrix.pearson_matrix
        pairs = []
        names = list(pm.columns)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                val = float(pm.iloc[i, j])  # type: ignore[arg-type]  # pandas Scalar -> float
                if np.isnan(val):
                    continue
                if abs(val) >= 0.999:  # skip trivial self/near-identity
                    continue
                pairs.append((names[i], names[j], val))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        print("\nMost Correlated Model Pairs:")
        for model1, model2, corr in pairs[:8]:
            print(f"  {model1:20} ↔ {model2:20} | r = {corr:6.3f}")
    except Exception:
        logger.debug("Failed to compute most-correlated pairs", exc_info=True)


def _print_fallback_summary(fallback_filter: _NumericFallbackFilter) -> None:
    """Print the analysis-only summary of suppressed numeric fallback log lines."""
    try:
        suppressed = getattr(fallback_filter, "suppressed_lines", 0)
        qids = getattr(fallback_filter, "qids", set())
        if suppressed:
            print(f"\n[analysis] Suppressed numeric fallback warnings: {suppressed} lines across {len(qids)} questions")
    except Exception:
        logger.debug("Failed to print numeric fallback summary", exc_info=True)


def _run_all_model_ensemble(analyzer: CorrelationAnalyzer) -> None:
    """Compare mean/median ensembles across all remaining (non-stacking) base models, with per-type splits."""
    try:
        benches_filtered = getattr(analyzer, "benchmarks", [])
        model_to_qids = _base_model_qid_sets(analyzer, benches_filtered)
        # Models with no questions at all can't take part in an intersection.
        models = [m for m, qids in model_to_qids.items() if qids]

        if len(models) < 2:
            print("\nALL-MODEL ENSEMBLE: skipped (need ≥2 base models after filters)")
            return

        # Questions used = intersection across all included models.
        all_sets = [model_to_qids[m] for m in models]
        inter = set.intersection(*all_sets)
        uni = set.union(*all_sets)

        # Computed before any printing so a simulator failure lands in the outer
        # handler with no half-written section above it.
        avg_cost = _avg_cost_per_question(analyzer, models)
        mean_score = analyzer._simulate_ensemble_performance(models, "mean")  # type: ignore[attr-defined]
        median_score = analyzer._simulate_ensemble_performance(models, "median")  # type: ignore[attr-defined]

        print("\n" + "=" * 60)
        print("ALL-MODEL ENSEMBLE (After Filters)")
        print("=" * 60)
        print(f"Models included ({len(models)}): {_short_model_list(models)}")
        coverage = (len(inter) / max(len(uni), 1)) * 100.0 if uni else 0.0
        print(f"Questions used: {len(inter)} of {len(uni)} ({coverage:.1f}% coverage)")
        print(f"Avg cost per question: ${avg_cost:.3f}")
        print(f"MEAN   ensemble score: {mean_score:.2f}")
        print(f"MEDIAN ensemble score: {median_score:.2f}")

        for agg_name in ("MEAN", "MEDIAN"):
            per_type = _ensemble_per_type(analyzer, benches_filtered, models, agg_name.lower())
            if per_type:
                print(f"{agg_name} per-type:")
                _print_type_stats(per_type, indent="  ")
    except Exception as e:  # noqa: BLE001  # soft-fail boundary: one diagnostic section must not abort an analysis run
        logger.warning(f"Failed to compute ALL-MODEL ensemble summary: {e}")


def _base_model_qid_sets(analyzer: CorrelationAnalyzer, benchmarks: list[Any]) -> dict[str, set[int]]:
    """Answered-question ids per NON-stacking model, in first-seen order.

    Stacking benchmarks are excluded: a stacker isn't a base ensemble member.
    """
    model_to_qids: dict[str, set[int]] = {}
    for benchmark in benchmarks:
        if _is_stacking_bench(analyzer, benchmark):
            continue
        answered = model_to_qids.setdefault(_model_name_for(analyzer, benchmark), set())
        answered.update(
            qid
            for qid in (
                getattr(getattr(report, "question", None), "id_of_question", None)
                for report in benchmark.forecast_reports
            )
            if isinstance(qid, int)
        )
    return model_to_qids


def _avg_cost_per_question(analyzer: CorrelationAnalyzer, models: list[str]) -> float:
    """Mean per-question cost across ``models``, NaN when the analyzer publishes no stats."""
    try:
        stats = analyzer._calculate_model_statistics()  # type: ignore[attr-defined]
    except Exception:
        logger.debug("analyzer._calculate_model_statistics() failed", exc_info=True)
        stats = {}
    avg_costs = [stats[m]["avg_cost"] for m in models if m in stats]
    return float(np.mean(avg_costs)) if avg_costs else float("nan")


def _short_model_list(names: list[str], max_len: int = 6) -> str:
    """Comma-joined model names, elided past ``max_len`` with a remaining count."""
    if len(names) <= max_len:
        return ", ".join(names)
    return ", ".join(names[:max_len]) + f" … (+{len(names) - max_len} more)"


def main() -> None:
    args = _parse_args()
    analyzer, fallback_filter = _load_and_prepare(args)

    _compute_score_stats(analyzer, args)
    _compute_model_stats(analyzer, args)

    has_mixed_types = analyzer._has_mixed_question_types()
    if has_mixed_types:
        logger.info("Detected mixed question types - using component-wise correlation analysis")
        type_breakdown = analyzer._get_question_type_breakdown()
        logger.info(f"Question type distribution: {type_breakdown}")
    else:
        logger.info("Using traditional correlation analysis for binary questions")

    output_file = _resolve_output_path(args)
    _print_report(analyzer, output_file)

    _run_ablation(analyzer, args)
    _print_correlation_highlights(analyzer, has_mixed_types)

    print(f"\nDetailed report saved to: {output_file}")
    _print_fallback_summary(fallback_filter)

    _run_all_model_ensemble(analyzer)


if __name__ == "__main__":
    main()
