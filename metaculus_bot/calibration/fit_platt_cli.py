"""Standalone CLI: fit Platt scaling parameters from historical bot performance data.

Reads a ``performance_data.json`` produced by ``performance_analysis`` and fits
a Platt recalibration curve for binary and/or multiple-choice questions.
Writes a JSON dump of fit metadata, an optional pre/post calibration plot, and
a Markdown narrative report.

The fit math lives in ``metaculus_bot.calibration.platt``. This module is just
data loading + scoring + reporting glue. It deliberately does NOT touch
``metaculus_bot/calibration/params.py`` — the user wants to review the report
and hand-edit the params themselves.

Usage::

    python -m metaculus_bot.calibration.fit_platt_cli \\
        --binary-from <path-to-performance-data.json> \\
        --mc-from    <path-to-performance-data.json> \\
        --output-dir <output-dir>

``--binary-from`` and ``--mc-from`` may point at the same file (typical case)
or different files. If only one is provided, only that fit runs.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metaculus_bot.calibration.platt import PlattParams, apply_binary_platt, fit_platt
from metaculus_bot.performance_analysis.analysis import CALIBRATION_BUCKET_EDGES
from metaculus_bot.scoring_common import brier_score

logger: logging.Logger = logging.getLogger(__name__)

# Default tag stamped into the fit JSON so future re-runs / diffs can identify
# which methodology version produced the params. Override per-run via
# ``--version`` to capture the specific dataset (e.g. "platt_v1_spring_aib_2026"
# or "platt_v1_fall_aib_2025"). Bump the major version (``platt_v2_*``) when
# the math itself changes.
DEFAULT_FIT_VERSION: str = "platt_v1"

# Probabilities at which we evaluate the fitted curve in the report. The
# extreme tails (0.01, 0.99) are deliberately included so the user sees what
# the fit wants to do at the edges before committing to a deviation cap.
CURVE_EVAL_POINTS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)

# Heuristic threshold for "fit is aggressive": if the curve moves any
# probability in [0.05, 0.95] by more than this much, the fit is materially
# different from the bot's raw output. Used only for a soft note in the
# report's "Cap notes" section — does not gate any decision.
AGGRESSIVE_FIT_DEVIATION: float = 0.10


@dataclass(frozen=True)
class FitOutcome:
    """Result of one Platt fit + in-sample evaluation pass.

    Either ``params`` is set (fit succeeded) or ``refusal_reason`` is set
    (fit_platt raised ValueError on degenerate input). Never both.
    """

    label: str  # "binary" or "mc", used for filenames + report sections
    n_train: int
    params: PlattParams | None
    refusal_reason: str | None
    baseline_mean_brier: float | None
    post_fit_mean_brier: float | None
    mean_brier_delta: float | None
    curve_evaluation: dict[str, float] | None
    calibration_buckets_pre: dict[str, dict[str, float | int]] | None
    calibration_buckets_post: dict[str, dict[str, float | int]] | None
    raw_probs: list[float]
    outcomes: list[bool]


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list at {path}, got {type(data).__name__}")
    return data


def _extract_binary_pairs(records: list[dict[str, Any]]) -> tuple[list[float], list[bool]]:
    """Filter to clean binary records and return ``(raw_probs, outcomes)``."""
    raw_probs: list[float] = []
    outcomes: list[bool] = []
    for r in records:
        if r.get("type") != "binary":
            continue
        resolution = r.get("resolution_parsed")
        if resolution is not True and resolution is not False:
            logger.warning(
                "Skipping binary record qid=%s: non-bool resolution_parsed=%r", r.get("question_id"), resolution
            )
            continue
        prob = r.get("our_prob_yes")
        if not isinstance(prob, (int, float)):
            logger.warning("Skipping binary record qid=%s: our_prob_yes=%r is not a number", r.get("question_id"), prob)
            continue
        raw_probs.append(float(prob))
        outcomes.append(bool(resolution))
    return raw_probs, outcomes


def _extract_mc_pairs(records: list[dict[str, Any]]) -> tuple[list[float], list[bool]]:
    """One-vs-rest decompose MC records into binary subproblems.

    Each MC question becomes ``len(options)`` rows: one per option, with
    ``raw_prob = our_forecast_values[i]`` and ``outcome = (option == winner)``.
    Returns the concatenated lists across all MC questions.
    """
    raw_probs: list[float] = []
    outcomes: list[bool] = []
    for r in records:
        if r.get("type") != "multiple_choice":
            continue
        resolution = r.get("resolution_parsed")
        if not isinstance(resolution, str) or not resolution:
            logger.warning(
                "Skipping MC record qid=%s: resolution_parsed=%r is not a non-empty string",
                r.get("question_id"),
                resolution,
            )
            continue
        options = r.get("options")
        forecasts = r.get("our_forecast_values")
        if not isinstance(options, list) or not isinstance(forecasts, list):
            logger.warning("Skipping MC record qid=%s: options/forecasts not lists", r.get("question_id"))
            continue
        if len(options) != len(forecasts) or len(options) == 0:
            logger.warning(
                "Skipping MC record qid=%s: options len=%d != forecasts len=%d (or empty)",
                r.get("question_id"),
                len(options),
                len(forecasts),
            )
            continue
        if resolution not in options:
            logger.warning(
                "Skipping MC record qid=%s: resolved option %r not in options list", r.get("question_id"), resolution
            )
            continue
        for option_name, prob in zip(options, forecasts):
            if not isinstance(prob, (int, float)):
                logger.warning(
                    "Skipping MC option qid=%s option=%r: prob=%r is not a number",
                    r.get("question_id"),
                    option_name,
                    prob,
                )
                continue
            raw_probs.append(float(prob))
            outcomes.append(option_name == resolution)
    return raw_probs, outcomes


def _bucket_calibration(probs: list[float], outcomes: list[bool]) -> dict[str, dict[str, float | int]]:
    """Compute calibration buckets matching ``analysis.binary_summary``.

    Returns a dict keyed by ``"0.0-0.1"`` etc., each value containing
    ``predicted_mean``, ``actual_rate``, and ``count``. Empty buckets are
    omitted to match the existing analysis convention.
    """
    edges = CALIBRATION_BUCKET_EDGES
    buckets: dict[str, dict[str, float | int]] = {}
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        label = f"{low:.1f}-{high:.1f}"
        # Match analysis.binary_summary: lower-inclusive, upper-exclusive,
        # with the final bucket also including the upper edge (1.0).
        in_bucket = [(p, o) for p, o in zip(probs, outcomes) if low <= p < high or (high == 1.0 and p == 1.0)]
        if not in_bucket:
            continue
        bucket_probs = [p for p, _ in in_bucket]
        bucket_outcomes = [o for _, o in in_bucket]
        buckets[label] = {
            "predicted_mean": sum(bucket_probs) / len(bucket_probs),
            "actual_rate": sum(1 for o in bucket_outcomes if o) / len(bucket_outcomes),
            "count": len(bucket_outcomes),
        }
    return buckets


def _round_floats(obj: Any, ndigits: int = 6) -> Any:
    """Recursively round floats so JSON output stays machine-readable but lean."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def _run_fit(
    label: str,
    raw_probs: list[float],
    outcomes: list[bool],
) -> FitOutcome:
    """Fit Platt + compute pre/post in-sample diagnostics for one type."""
    n = len(raw_probs)
    if n == 0:
        logger.error("%s fit: no usable records found, refusing.", label)
        return FitOutcome(
            label=label,
            n_train=0,
            params=None,
            refusal_reason="No usable records after filtering.",
            baseline_mean_brier=None,
            post_fit_mean_brier=None,
            mean_brier_delta=None,
            curve_evaluation=None,
            calibration_buckets_pre=None,
            calibration_buckets_post=None,
            raw_probs=raw_probs,
            outcomes=outcomes,
        )

    logger.info("%s fit: %d training rows, %d positive outcomes", label, n, sum(outcomes))

    try:
        params = fit_platt(raw_probs, outcomes)
    except ValueError as exc:
        logger.error("%s fit refused: %s", label, exc)
        return FitOutcome(
            label=label,
            n_train=n,
            params=None,
            refusal_reason=str(exc),
            baseline_mean_brier=None,
            post_fit_mean_brier=None,
            mean_brier_delta=None,
            curve_evaluation=None,
            calibration_buckets_pre=None,
            calibration_buckets_post=None,
            raw_probs=raw_probs,
            outcomes=outcomes,
        )

    logger.info("%s fit: bias=%.4f slope=%.4f", label, params.bias, params.slope)

    # In-sample Brier — apply the curve with NO cap so the user sees what the
    # unconstrained fit wants. The cap is a separate user-tuned safety rail.
    baseline_briers = [brier_score(p, o) for p, o in zip(raw_probs, outcomes)]
    adjusted_probs = [apply_binary_platt(p, params, max_abs_deviation=None) for p in raw_probs]
    post_briers = [brier_score(p, o) for p, o in zip(adjusted_probs, outcomes)]
    baseline_mean = sum(baseline_briers) / len(baseline_briers)
    post_mean = sum(post_briers) / len(post_briers)
    delta = baseline_mean - post_mean

    curve = {f"{p:.2f}": apply_binary_platt(p, params, max_abs_deviation=None) for p in CURVE_EVAL_POINTS}

    pre_buckets = _bucket_calibration(raw_probs, outcomes)
    post_buckets = _bucket_calibration(adjusted_probs, outcomes)

    return FitOutcome(
        label=label,
        n_train=n,
        params=params,
        refusal_reason=None,
        baseline_mean_brier=baseline_mean,
        post_fit_mean_brier=post_mean,
        mean_brier_delta=delta,
        curve_evaluation=curve,
        calibration_buckets_pre=pre_buckets,
        calibration_buckets_post=post_buckets,
        raw_probs=raw_probs,
        outcomes=outcomes,
    )


def _outcome_to_json(outcome: FitOutcome, version: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": version,
        "n_train": outcome.n_train,
    }
    if outcome.refusal_reason is not None:
        payload["refused"] = True
        payload["refusal_reason"] = outcome.refusal_reason
        return payload

    assert outcome.params is not None  # mypy / refusal_reason invariant
    payload["refused"] = False
    payload["bias"] = outcome.params.bias
    payload["slope"] = outcome.params.slope
    payload["baseline_mean_brier"] = outcome.baseline_mean_brier
    payload["post_fit_mean_brier"] = outcome.post_fit_mean_brier
    payload["mean_brier_delta"] = outcome.mean_brier_delta
    payload["curve_evaluation"] = outcome.curve_evaluation
    payload["calibration_buckets_pre"] = outcome.calibration_buckets_pre
    payload["calibration_buckets_post"] = outcome.calibration_buckets_post
    return payload


def _write_json(outcome: FitOutcome, path: Path, version: str) -> None:
    payload = _round_floats(_outcome_to_json(outcome, version))
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    logger.info("Wrote %s", path)


def _try_plot(
    binary_outcome: FitOutcome | None,
    mc_outcome: FitOutcome | None,
    out_path: Path,
) -> str | None:
    """Render the 2-panel pre/post calibration figure.

    Returns ``None`` on success or a human-readable note string on skip
    (e.g., matplotlib missing, no fits succeeded). The note flows into the
    Markdown report so the reader knows why the PNG isn't there.
    """
    fits = [(o, color) for o, color in [(binary_outcome, "C0"), (mc_outcome, "C1")] if o is not None]
    if not fits:
        return "No fits ran; skipping calibration plot."

    try:
        import matplotlib  # pyright: ignore[reportMissingImports]  # matplotlib genuinely optional for CLI plotting; CLI works without it

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]  # optional CLI dep, guarded by ImportError
    except ImportError as exc:
        logger.warning("matplotlib unavailable: %s; skipping plot.", exc)
        return f"matplotlib unavailable ({exc}); calibration plot skipped."

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = [(binary_outcome, axes[0], "Binary", "C0"), (mc_outcome, axes[1], "MC", "C1")]

    for outcome, ax, panel_label, color in panels:
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Actual rate")
        ax.set_aspect("equal")
        if outcome is None:
            ax.set_title(f"{panel_label}: not run")
            continue
        if outcome.refusal_reason is not None:
            ax.set_title(f"{panel_label}: REFUSED\n{outcome.refusal_reason[:60]}")
            continue
        assert outcome.params is not None
        pre = outcome.calibration_buckets_pre or {}
        post = outcome.calibration_buckets_post or {}
        # Gray dots: raw calibration. Color dots: post-fit. Marker size scales
        # with bucket count so sparse buckets don't dominate the eye.
        for buckets, marker_color, label in (
            (pre, "gray", "raw"),
            (post, color, "post-fit"),
        ):
            xs = [b["predicted_mean"] for b in buckets.values()]
            ys = [b["actual_rate"] for b in buckets.values()]
            sizes = [20 + 4 * int(b["count"]) for b in buckets.values()]
            ax.scatter(xs, ys, s=sizes, color=marker_color, alpha=0.7, label=label, edgecolors="black", linewidths=0.5)
        ax.set_title(
            f"{panel_label}: bias={outcome.params.bias:.3f} slope={outcome.params.slope:.3f} n={outcome.n_train}"
        )
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
    return None


def _outcome_summary(outcome: FitOutcome | None) -> str:
    """One-line factual summary for the report header.

    Reports the in-sample Brier delta as-is. Deliberately does NOT include a
    "ship / do NOT ship" verdict — in-sample is optimistic, and the shipping
    decision belongs upstream (cross-round stability check, manual review).
    """
    if outcome is None:
        return "not run"
    if outcome.refusal_reason is not None:
        return f"REFUSED ({outcome.refusal_reason})"
    assert outcome.mean_brier_delta is not None
    assert outcome.params is not None
    return (
        f"n={outcome.n_train}, bias={outcome.params.bias:+.4f}, "
        f"slope={outcome.params.slope:.4f}, in-sample mean Brier delta={outcome.mean_brier_delta:+.4f}"
    )


def _cap_recommendation(outcome: FitOutcome | None) -> str:
    if outcome is None or outcome.refusal_reason is not None or outcome.curve_evaluation is None:
        return "n/a"
    aggressive = False
    for label, adj in outcome.curve_evaluation.items():
        raw = float(label)
        if 0.05 <= raw <= 0.95 and abs(adj - raw) > AGGRESSIVE_FIT_DEVIATION:
            aggressive = True
            break
    if aggressive:
        return (
            f"fit moves probabilities by >{AGGRESSIVE_FIT_DEVIATION:.2f} in [0.05, 0.95]; consider tighter cap of 0.05"
        )
    return "fit is gentle in [0.05, 0.95]; default cap of 0.10 is fine"


def _format_curve_table(outcome: FitOutcome) -> str:
    if outcome.curve_evaluation is None:
        return "(no curve — fit refused)"
    rows = ["| raw | adjusted | delta |", "| --- | --- | --- |"]
    for label, adj in outcome.curve_evaluation.items():
        raw = float(label)
        rows.append(f"| {raw:.2f} | {adj:.4f} | {adj - raw:+.4f} |")
    return "\n".join(rows)


def _format_buckets_table(buckets: dict[str, dict[str, float | int]] | None) -> str:
    if not buckets:
        return "(empty)"
    rows = ["| bucket | predicted_mean | actual_rate | count |", "| --- | --- | --- | --- |"]
    for label, b in buckets.items():
        rows.append(
            f"| {label} | {float(b['predicted_mean']):.4f} | {float(b['actual_rate']):.4f} | {int(b['count'])} |"
        )
    return "\n".join(rows)


def _format_section(outcome: FitOutcome | None, title: str) -> str:
    lines: list[str] = [f"## {title} fit"]
    if outcome is None:
        lines.append("\n_Not run (no `--{0}-from` argument)._".format(title.lower()))
        return "\n".join(lines)

    lines.append(f"\n- n_train: **{outcome.n_train}**")
    if outcome.refusal_reason is not None:
        lines.append(f"- **REFUSED**: {outcome.refusal_reason}")
        return "\n".join(lines)

    assert outcome.params is not None
    lines.append(f"- bias = **{outcome.params.bias:.4f}**, slope = **{outcome.params.slope:.4f}**")
    assert outcome.baseline_mean_brier is not None
    assert outcome.post_fit_mean_brier is not None
    assert outcome.mean_brier_delta is not None
    lines.append(f"- baseline mean Brier: {outcome.baseline_mean_brier:.6f}")
    lines.append(f"- post-fit mean Brier: {outcome.post_fit_mean_brier:.6f}")
    lines.append(f"- mean Brier delta (baseline - post): **{outcome.mean_brier_delta:+.6f}** (positive = improvement)")

    lines.append("\n### Curve evaluation\n")
    lines.append(_format_curve_table(outcome))

    lines.append("\n### Calibration buckets — pre-fit\n")
    lines.append(_format_buckets_table(outcome.calibration_buckets_pre))

    lines.append("\n### Calibration buckets — post-fit\n")
    lines.append(_format_buckets_table(outcome.calibration_buckets_post))

    return "\n".join(lines)


def _write_report(
    binary_outcome: FitOutcome | None,
    mc_outcome: FitOutcome | None,
    plot_skip_note: str | None,
    path: Path,
    version: str,
) -> None:
    binary_summary = _outcome_summary(binary_outcome)
    mc_summary = _outcome_summary(mc_outcome)
    binary_cap = _cap_recommendation(binary_outcome)
    mc_cap = _cap_recommendation(mc_outcome)

    sections: list[str] = []
    sections.append("# Platt scaling fit report\n")
    sections.append(f"_Version: `{version}`_\n")

    sections.append("## Summary\n")
    sections.append(f"- Binary: {binary_summary}")
    sections.append(f"- MC: {mc_summary}\n")

    sections.append(_format_section(binary_outcome, "Binary"))
    sections.append("")
    sections.append(_format_section(mc_outcome, "MC"))
    sections.append("")

    sections.append("## Cap notes\n")
    sections.append(f"- Binary: {binary_cap}")
    sections.append(f"- MC: {mc_cap}")
    sections.append(
        "\nNote: in-sample Brier deltas overstate generalization. Before changing "
        "`metaculus_bot/calibration/params.py`, also run a cross-round stability check "
        "(re-fit on a different tournament's `performance_data.json` and compare params) "
        "and an out-of-sample backtest. The deviation cap (`max_abs_deviation`) lives "
        "in `metaculus_bot.constants`; the fit's curve is shown unconstrained above."
    )
    if plot_skip_note is not None:
        sections.append(f"\n_Plot status_: {plot_skip_note}")
    else:
        sections.append("\nSee `calibration_curves_pre_post.png` for the pre/post calibration scatter.")

    path.write_text("\n".join(sections) + "\n")
    logger.info("Wrote %s", path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m metaculus_bot.calibration.fit_platt_cli",
        description="Fit Platt recalibration parameters from historical performance data.",
    )
    parser.add_argument(
        "--binary-from",
        type=Path,
        default=None,
        help="Path to performance_data.json with binary records to fit on.",
    )
    parser.add_argument(
        "--mc-from",
        type=Path,
        default=None,
        help="Path to performance_data.json with MC records to fit on (may be the same file).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for binary_fit.json, mc_fit.json, calibration_curves_pre_post.png, report.md.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_FIT_VERSION,
        help=(
            f"Version tag stamped into the fit JSON output (default: {DEFAULT_FIT_VERSION!r}). "
            "Override per-run to capture the dataset, e.g. 'platt_v1_spring_aib_2026'."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.binary_from is None and args.mc_from is None:
        logger.error("At least one of --binary-from or --mc-from is required.")
        return 2

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    version: str = args.version

    binary_outcome: FitOutcome | None = None
    if args.binary_from is not None:
        logger.info("Loading binary records from %s", args.binary_from)
        records = _load_records(args.binary_from)
        raw_probs, outcomes = _extract_binary_pairs(records)
        binary_outcome = _run_fit("binary", raw_probs, outcomes)
        _write_json(binary_outcome, output_dir / "binary_fit.json", version)

    mc_outcome: FitOutcome | None = None
    if args.mc_from is not None:
        logger.info("Loading MC records from %s", args.mc_from)
        records = _load_records(args.mc_from)
        raw_probs, outcomes = _extract_mc_pairs(records)
        mc_outcome = _run_fit("mc", raw_probs, outcomes)
        _write_json(mc_outcome, output_dir / "mc_fit.json", version)

    plot_skip_note = _try_plot(binary_outcome, mc_outcome, output_dir / "calibration_curves_pre_post.png")
    _write_report(binary_outcome, mc_outcome, plot_skip_note, output_dir / "report.md", version)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    raise SystemExit(run(parser.parse_args()))
