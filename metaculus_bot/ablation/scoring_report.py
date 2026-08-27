"""Markdown rendering for the probabilistic-tools ablation paired-difference report.

Pure rendering functions: PairedScore/PairedStats -> markdown text.
Extracted from scoring.py to keep that module focused on scoring logic.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from metaculus_bot.ablation.scoring import (
    COMPARISONS_3ARM,
    PairedScore,
    PairedStats,
    _saturation_pair_for_comparison,
    aggregate_paired,
)

__all__ = ["render_summary_markdown"]

LOW_N_THRESHOLD: int = 30


def _format_score(x: float) -> str:
    if math.isnan(x):
        return "NaN"
    return f"{x:.4f}"


def _format_p(x: float | None) -> str:
    if x is None:
        return "N/A"
    if math.isnan(x):
        return "NaN"
    return f"{x:.3f}"


def _format_n(n: int) -> str:
    return f"{n:.0f}"


def _stats_table_header() -> list[str]:
    return [
        (
            "| Metric | Type | N | Mean Δ | Mean 95% CI | Median Δ | Median 95% CI "
            "| NoSat Δ (n_clean) | Sign p | Wilcoxon p | Better when Δ |"
        ),
        (
            "|--------|------|---|--------|-------------|----------|---------------"
            "|-------------------|--------|------------|---------------|"
        ),
    ]


def _format_clean_mean(stats: PairedStats) -> str:
    if stats.n_clean == 0:
        return f"NaN ({stats.n_clean})"
    return f"{_format_score(stats.mean_delta_clean)} ({stats.n_clean})"


def _direction_label(s: PairedStats) -> str:
    """Which arm wins when delta has the favorable sign for this comparison + metric."""
    left = s.comparison.split("-")[0]
    arrow = "Δ > 0" if s.higher_is_better else "Δ < 0"
    return f"{left} when {arrow}"


def _stats_row(s: PairedStats) -> str:
    qtype = s.question_type if s.question_type is not None else "overall"
    mean_ci_str = f"[{_format_score(s.bootstrap_ci_low)}, {_format_score(s.bootstrap_ci_high)}]"
    median_ci_str = f"[{_format_score(s.median_ci_low)}, {_format_score(s.median_ci_high)}]"
    n_str = _format_n(s.n)
    if s.question_type is not None and s.n < LOW_N_THRESHOLD:
        n_str = f"{n_str}*"
    return (
        f"| {s.metric} "
        f"| {qtype} "
        f"| {n_str} "
        f"| {_format_score(s.mean_delta)} "
        f"| {mean_ci_str} "
        f"| {_format_score(s.median_delta)} "
        f"| {median_ci_str} "
        f"| {_format_clean_mean(s)} "
        f"| {_format_p(s.sign_test_p)} "
        f"| {_format_p(s.wilcoxon_p)} "
        f"| {_direction_label(s)} |"
    )


def _has_confounder_data(scores: list[PairedScore]) -> bool:
    """True iff at least one PairedScore has a confounder field populated."""
    return any(
        s.stack_model_used is not None
        or s.stack_aug_model_used is not None
        or s.median_model_used is not None
        or s.stack_aug_tools_fired is not None
        or s.median_tools_fired is not None
        or s.n_forecasters_stack is not None
        or s.n_forecasters_stack_aug is not None
        or s.n_forecasters_median is not None
        for s in scores
    )


def _md_escape_cell(value: Any) -> str:
    """Escape pipes in a markdown table cell."""
    if value is None:
        return "-"
    return str(value).replace("|", "\\|")


def _per_question_diagnostic_table(scores: list[PairedScore]) -> list[str]:
    """One row per (qid, metric) -- dedupe across the three comparisons."""
    show_confounders = _has_confounder_data(scores)
    if show_confounders:
        lines = [
            (
                "| qid | type | metric | stack | stack_aug | median | Δ_stack_aug-stack | Δ_median-stack | Δ_median-stack_aug "
                "| sat_stack | sat_stack_aug | sat_median | stack_model | stack_aug_model | median_model | stack_aug_tools | median_tools |"
            ),
            (
                "|-----|------|--------|-------|-----|--------|-------------|----------------|-------------"
                "|-----------|---------|-----------|-------------|-----------|--------------|-----------|--------------|"
            ),
        ]
    else:
        lines = [
            (
                "| qid | type | metric | stack | stack_aug | median | Δ_stack_aug-stack | Δ_median-stack | Δ_median-stack_aug "
                "| sat_stack | sat_stack_aug | sat_median |"
            ),
            (
                "|-----|------|--------|-------|-----|--------|-------------|----------------|-------------"
                "|-----------|---------|-----------|"
            ),
        ]

    by_key: dict[tuple[int, str], PairedScore] = {}
    for s in scores:
        key = (s.qid, s.metric)
        if key not in by_key or s.comparison == "stack_aug-stack":
            by_key[key] = s

    def _sat_label(flag: bool) -> str:
        return "Y" if flag else "n"

    rows = list(by_key.values())
    rows.sort(key=lambda s: abs(s.score_stack_aug - s.score_stack), reverse=True)
    for s in rows:
        delta_pdf_stack = s.score_stack_aug - s.score_stack
        delta_median_stack = s.score_median - s.score_stack
        delta_median_pdf = s.score_median - s.score_stack_aug
        base_row = (
            f"| {s.qid} "
            f"| {s.question_type} "
            f"| {s.metric} "
            f"| {_format_score(s.score_stack)} "
            f"| {_format_score(s.score_stack_aug)} "
            f"| {_format_score(s.score_median)} "
            f"| {_format_score(delta_pdf_stack)} "
            f"| {_format_score(delta_median_stack)} "
            f"| {_format_score(delta_median_pdf)} "
            f"| {_sat_label(s.is_saturated_stack)} "
            f"| {_sat_label(s.is_saturated_stack_aug)} "
            f"| {_sat_label(s.is_saturated_median)} "
        )
        if show_confounders:
            stack_marker = _md_escape_cell(s.stack_model_used)
            pdf_marker = _md_escape_cell(s.stack_aug_model_used)
            median_marker = _md_escape_cell(s.median_model_used)
            pdf_tools_marker = "-" if s.stack_aug_tools_fired is None else ("yes" if s.stack_aug_tools_fired else "no")
            median_tools_marker = "-" if s.median_tools_fired is None else ("yes" if s.median_tools_fired else "no")
            lines.append(
                f"{base_row}| {stack_marker} | {pdf_marker} | {median_marker} "
                f"| {pdf_tools_marker} | {median_tools_marker} |"
            )
        else:
            lines.append(f"{base_row}|")
    return lines


def _mean_or_none(values: list[int | None]) -> float | None:
    """Mean of the non-None entries, or None when every entry is missing."""
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _avg_forecaster_line(sample: list[PairedScore]) -> list[str]:
    """The 'Average n_forecasters_used' line, degrading when an arm carries no counts."""
    avg_stack = _mean_or_none([s.n_forecasters_stack for s in sample])
    avg_pdf = _mean_or_none([s.n_forecasters_stack_aug for s in sample])
    avg_median = _mean_or_none([s.n_forecasters_median for s in sample])
    if avg_stack is not None and avg_pdf is not None and avg_median is not None:
        return [
            f"- Average n_forecasters_used: stack {avg_stack:.2f}, stack_aug {avg_pdf:.2f}, median {avg_median:.2f}."
        ]
    if avg_stack is not None and avg_pdf is not None:
        return [f"- Average n_forecasters_used: stack {avg_stack:.2f}, stack_aug {avg_pdf:.2f}."]
    return []


def _treatment_activation_lines(sample: list[PairedScore]) -> list[str]:
    """Per-arm 'fired tools on N/M' lines, omitted for arms with no observations."""
    lines: list[str] = []
    pdf_observations = [s.stack_aug_tools_fired for s in sample if s.stack_aug_tools_fired is not None]
    if pdf_observations:
        fired = sum(1 for observed in pdf_observations if observed)
        line_pdf = f"- Treatment activation: stack_aug fired tools on {fired}/{len(pdf_observations)} questions"
        empty = len(pdf_observations) - fired
        if empty > 0:
            line_pdf += f" ({empty} had empty cross_model_aggregation — likely structured-block parse failures)"
        lines.append(line_pdf + ".")
    median_observations = [s.median_tools_fired for s in sample if s.median_tools_fired is not None]
    if median_observations:
        fired = sum(1 for observed in median_observations if observed)
        lines.append(
            f"- Treatment activation: median fired tools on"
            f" {fired}/{len(median_observations)} "
            "(deterministic aggregation — no LLM tools)."
        )
    return lines


def _confounder_summary_lines(scores: list[PairedScore]) -> list[str]:
    """Build the per-arm fallback rate + treatment-activation block from paired scores.

    One representative row per qid — the confounder fields are per-question, so
    counting every metric/comparison row would multiply each question by its row count.
    """
    by_qid: dict[int, PairedScore] = {}
    for score in scores:
        by_qid.setdefault(score.qid, score)
    sample = list(by_qid.values())
    if not sample:
        return []

    total = len(sample)

    def _count(arm_attr: str, value: str) -> int:
        return sum(1 for s in sample if getattr(s, arm_attr) == value)

    lines: list[str] = ["", "## Confounder summary"]
    lines.append(
        f"- stack: {_count('stack_model_used', 'primary')}/{total} primary, "
        f"{_count('stack_model_used', 'fallback')}/{total} fallback."
    )
    lines.append(
        f"- stack_aug: {_count('stack_aug_model_used', 'primary')}/{total} primary, "
        f"{_count('stack_aug_model_used', 'fallback')}/{total} fallback."
    )
    median_simple = _count("median_model_used", "simple_aggregation")
    if median_simple > 0:
        lines.append(f"- median: {median_simple}/{total} simple_aggregation (deterministic, no fallback).")
    lines.extend(_avg_forecaster_line(sample))
    lines.extend(_treatment_activation_lines(sample))
    return lines


def _per_arm_raw_stats_lines(scores: list[PairedScore]) -> list[str]:
    """Per-arm raw mean/median score per (arm, metric, type), sanity-check section."""
    by_key: dict[tuple[int, str, str], PairedScore] = {}
    for s in scores:
        by_key[(s.qid, s.metric, s.question_type)] = s

    if not by_key:
        return []

    has_pdf_arms = any(not math.isnan(s.score_pdf_min1) for s in by_key.values())
    has_mean_arm = any(not math.isnan(s.score_mean) for s in by_key.values())

    arm_extractors: dict[str, Any] = {
        "stack": lambda s: s.score_stack,
        "stack_aug": lambda s: s.score_stack_aug,
        "median": lambda s: s.score_median,
    }
    if has_pdf_arms:
        arm_extractors["pdf_min1"] = lambda s: s.score_pdf_min1
        arm_extractors["pdf_min2"] = lambda s: s.score_pdf_min2
    if has_mean_arm:
        arm_extractors["mean"] = lambda s: s.score_mean
    bucket: dict[tuple[str, str, str], list[float]] = {}
    for s in by_key.values():
        for arm, extractor in arm_extractors.items():
            score = extractor(s)
            if not math.isnan(score):
                bucket.setdefault((arm, s.metric, s.question_type), []).append(score)
                bucket.setdefault((arm, s.metric, "overall"), []).append(score)

    lines: list[str] = [
        "",
        "## Per-arm raw stats",
        "",
        "| Arm | Metric | Type | N | Mean | Median |",
        "|-----|--------|------|---|------|--------|",
    ]
    for (arm, metric, qtype), values in sorted(bucket.items()):
        n = len(values)
        if n == 0:
            continue
        mean = float(np.mean(values))
        median = float(np.median(values))
        lines.append(f"| {arm} | {metric} | {qtype} | {n} | {_format_score(mean)} | {_format_score(median)} |")
    return lines


def _dual_panel_caveat_lines() -> list[str]:
    """The standing caveat about structured-block survival on the free-model ensemble."""
    return [
        "## PDF arm: dual-panel comparison",
        "",
        "> **Caveat:** the 'pdf' arm uses strict structured-math-only per forecaster. On the n=88 "
        "free-OpenRouter ensemble, only ~58% of (forecaster, question) pairs emit parseable "
        "structured blocks (mean 2.32 of 4 forecasters survive). We render two views:",
        ">",
        "> - **min_forecasters=1** (any-structured): qids where >=1 forecaster emitted structured math. "
        "Single-forecaster qids are NOT aggregation -- they are that single forecaster's structured prediction.",
        "> - **min_forecasters=2** (proper aggregation): qids where >=2 forecasters survived. "
        "These are honest pointwise-median aggregations.",
        ">",
        "> A re-run on the prod ensemble (Opus 4.6/4.7, GPT-5.5, Gemini 3 Pro, etc.) is expected "
        "to produce structured blocks much more reliably -- these free-model results should be "
        "interpreted as a lower bound on what the pdf arm can do.",
        "",
    ]


def _structured_survival_lines(paired_scores: list[PairedScore]) -> list[str]:
    """Per-qid structured-forecaster survival table plus the policy-impact sentence.

    On qids where both policies succeed the predictions are identical by construction
    (same forecasters, same aggregation), so the single-forecaster qids are the whole
    policy-relevant difference between min=1 and min=2.
    """
    by_qid_pdf: dict[int, PairedScore] = {}
    for score in paired_scores:
        if score.qid not in by_qid_pdf:
            by_qid_pdf[score.qid] = score

    n_both_fail = 0
    n_min1_only = 0
    n_both_succeed = 0
    for score in by_qid_pdf.values():
        has_min1 = not math.isnan(score.score_pdf_min1)
        has_min2 = not math.isnan(score.score_pdf_min2)
        if has_min1 and has_min2:
            n_both_succeed += 1
        elif has_min1 and not has_min2:
            n_min1_only += 1
        else:
            n_both_fail += 1

    return [
        "| Survival | n_qids | pdf_min1 | pdf_min2 |",
        "|----------|--------|----------|----------|",
        f"| 0 surviving forecasters | {n_both_fail} | fail | fail |",
        f"| 1 surviving forecaster | {n_min1_only} | success (single) | fail |",
        f"| 2+ surviving forecasters | {n_both_succeed} | success | success |",
        "",
        f"Policy impact: pdf_min1 covers {n_min1_only + n_both_succeed} qids, "
        f"pdf_min2 covers {n_both_succeed} qids. The {n_min1_only} single-forecaster "
        "qids are the policy-relevant difference.",
        "",
    ]


def _dual_panel_pdf_section(paired_scores: list[PairedScore], stats: list[PairedStats]) -> list[str]:
    """Render the dual-panel PDF arm comparison section."""
    lines: list[str] = _dual_panel_caveat_lines()
    pdf_min1_comparisons = {"pdf_min1-stack", "pdf_min1-stack_aug", "median-pdf_min1"}
    pdf_min2_comparisons = {"pdf_min2-stack", "pdf_min2-stack_aug", "median-pdf_min2"}

    overall_stats = [s for s in stats if s.question_type is None]

    min1_stats = [s for s in overall_stats if s.comparison in pdf_min1_comparisons]
    if min1_stats:
        n_min1 = min1_stats[0].n if min1_stats else 0
        lines.append(f"### min_forecasters=1 panel (n={n_min1})")
        lines.extend(_stats_table_header())
        for s in sorted(min1_stats, key=lambda x: (x.comparison, x.metric)):
            lines.append(_stats_row(s))
        lines.append("")

    min2_stats = [s for s in overall_stats if s.comparison in pdf_min2_comparisons]
    if min2_stats:
        n_min2 = min2_stats[0].n if min2_stats else 0
        lines.append(f"### min_forecasters=2 panel (n={n_min2})")
        lines.extend(_stats_table_header())
        for s in sorted(min2_stats, key=lambda x: (x.comparison, x.metric)):
            lines.append(_stats_row(s))
        lines.append("")

    lines.append("### Structured-forecaster survival distribution")
    lines.append("")
    lines.append(
        "Shows per-qid structured-forecaster survival counts and which policy "
        "(min=1 vs min=2) produces a prediction. On qids where both succeed, "
        "predictions are identical by construction (same forecasters, same aggregation)."
    )
    lines.append("")

    lines.extend(_structured_survival_lines(paired_scores))
    return lines


def _metadata_header_lines(metadata: dict) -> list[str]:
    """Report title plus one bullet per metadata key.

    ``timestamp`` / ``n_questions`` / ``model_lineup`` are rendered first in that fixed
    order; every other key follows in insertion order so a run's own flags stay visible
    without this function needing to know about them.
    """
    lines = [
        "# Probabilistic-Tools Ablation — Paired-Difference Summary",
        "",
        f"- Generated: {metadata['timestamp']}",
        f"- N questions: {metadata['n_questions']}",
    ]
    if "model_lineup" in metadata:
        lineup = metadata["model_lineup"]
        lineup_str = ", ".join(str(m) for m in lineup) if isinstance(lineup, (list, tuple)) else str(lineup)
        lines.append(f"- Model lineup: {lineup_str}")
    lines.extend(
        f"- {key}: {value}"
        for key, value in metadata.items()
        if key not in {"timestamp", "n_questions", "model_lineup"}
    )
    lines.append("")
    return lines


def _comparison_tables(
    stats_subset: list[PairedStats],
    active_comparisons: list[str],
    *,
    per_type: bool,
) -> list[str]:
    """One ``### Comparison: X-Y`` stats table per comparison that has rows."""
    lines: list[str] = []
    for comparison in active_comparisons:
        comp_stats = [s for s in stats_subset if s.comparison == comparison]
        if not comp_stats:
            continue
        lines.append(f"### Comparison: {comparison}")
        lines.extend(_stats_table_header())
        if per_type:
            comp_stats = sorted(comp_stats, key=lambda x: (x.question_type or "", x.metric))
        else:
            comp_stats = sorted(comp_stats, key=lambda x: x.metric)
        lines.extend(_stats_row(s) for s in comp_stats)
        lines.append("")
    return lines


def _stats_section(
    heading: str,
    stats_subset: list[PairedStats],
    active_comparisons: list[str],
    *,
    per_type: bool,
    empty_note: str,
) -> list[str]:
    """A ``## heading`` section: per-comparison tables, or ``empty_note`` when there are none."""
    lines = [heading]
    if stats_subset:
        lines.extend(_comparison_tables(stats_subset, active_comparisons, per_type=per_type))
    else:
        lines.append(empty_note)
        lines.append("")
    return lines


def _saturation_excluded_sections(paired_scores: list[PairedScore], active_comparisons: list[str]) -> list[str]:
    """Overall + per-type panels recomputed with saturated pairs dropped per comparison.

    Saturation can dominate means without reflecting calibration differences, so these
    panels re-aggregate from scratch over the non-saturated rows only. Empty when every
    row saturated.
    """
    nosat_scores = [s for s in paired_scores if not _saturation_pair_for_comparison(s)]
    if not nosat_scores:
        return []
    nosat_stats = aggregate_paired(nosat_scores, n_bootstrap=5000, seed=0)
    lines = _stats_section(
        "## Overall summary (saturation-excluded)",
        [s for s in nosat_stats if s.question_type is None],
        active_comparisons,
        per_type=False,
        empty_note="(no non-saturated scores; n=0)",
    )
    lines.extend(
        _stats_section(
            "## Per-type breakdown (saturation-excluded)",
            [s for s in nosat_stats if s.question_type is not None],
            active_comparisons,
            per_type=True,
            empty_note="(no non-saturated per-type scores; n=0)",
        )
    )
    return lines


def _caveat_lines(stats: list[PairedStats]) -> list[str]:
    """The standing interpretation caveats, plus the low-n warnings this run earns."""
    lines: list[str] = ["## Caveats"]
    overall_min_n = min((s.n for s in stats if s.question_type is None), default=0)
    per_type_min_n = min((s.n for s in stats if s.question_type is not None), default=0)

    if stats and overall_min_n < LOW_N_THRESHOLD:
        lines.append(
            f"- **Overall directional only (min overall n={overall_min_n}).** "
            "Bootstrap CIs are wide and significance tests have limited power. "
            "Append more questions before drawing conclusions."
        )
    if any(s.question_type is not None for s in stats) and per_type_min_n < LOW_N_THRESHOLD:
        lines.append(
            f"- **Per-type directional only (min per-type n={per_type_min_n}).** "
            f"Per-type rows marked with `*` have n<{LOW_N_THRESHOLD};"
            " treat their CIs and p-values as descriptive. "
            "The overall row aggregates across types and is the primary read."
        )
    lines.append(
        f"- **Multiple testing.** k={len(stats)} tests reported below;"
        " no multiple-testing correction. "
        "Treat individual p-values as descriptive, not as evidence of significance. "
        "To call something significant, look at the overall (metric, all-types) row,"
        " not a per-type cell."
    )
    lines.append(
        "- **Percentile-bootstrap CIs** may under-cover the nominal level when n<30,"
        " especially with skewed Δ. "
        "Treat width as a lower bound on uncertainty."
    )
    lines.append(
        "- **Selection bias.** Survivors of the leakage screen may be systematically"
        " harder or more obscure "
        "than the original tournament population. Δ generalizes to *this* class of"
        " questions, not to the "
        "unscreened population. Leakage rate (proportion of qids dropped) is a good"
        " proxy for how much "
        "selection bias matters."
    )
    lines.append(
        "- **Saturation flagging.** A score is *saturated* when it sits within"
        " ε of the schema's "
        "max-confident-wrong floor (e.g., a numeric log score of -220 means the"
        " resolution fell in a "
        "bucket where the prediction had ~min-step PMF mass — there's no way to"
        " be more wrong on this "
        "schema). Both-saturated rows are mechanical Δ≈0 draws and don't carry"
        " signal; they inflate "
        "sign-test power without informing direction. The `NoSat Δ (n_clean)` column"
        " reports the mean "
        "Δ over only the pairs where neither arm in the comparison saturated."
    )
    lines.append(
        "- **(saturation-excluded) panels** filter rows per comparison: a row is"
        " dropped if either "
        "arm in that comparison saturated (numeric_log_score near -220, brier near"
        " 1.0, etc., where "
        "the schema's max-confident-wrong floor is hit). Use these panels when"
        " interpreting numeric "
        "metrics — saturation can dominate means without reflecting calibration"
        " differences. N varies "
        "by comparison because saturation patterns differ across arms."
    )
    lines.append("- Higher is better for log scores; lower is better for Brier and CRPS.")
    lines.append("- Δ for comparison X-Y = score_X - score_Y. Bootstrap is paired (resampling qids with replacement).")
    return lines


def render_summary_markdown(stats: list[PairedStats], paired_scores: list[PairedScore], metadata: dict) -> str:
    """Generate the human-readable paired-difference summary.

    Sections:
      - Header (metadata)
      - Overall summary, three subsections (B-A, C-A, C-B)
      - Per-type breakdown, three subsections
      - Per-arm raw stats (single table, all arms x metrics x types)
      - Per-question diagnostic (one row per qid+metric, all three deltas inline)
      - Confounder summary (when payloads present)
      - Caveats
    """
    active_comparisons = sorted({s.comparison for s in stats}) if stats else list(COMPARISONS_3ARM)

    lines = _metadata_header_lines(metadata)
    lines.extend(
        _stats_section(
            "## Overall summary",
            [s for s in stats if s.question_type is None],
            active_comparisons,
            per_type=False,
            empty_note="(no scores; n=0)",
        )
    )
    lines.extend(
        _stats_section(
            "## Per-type breakdown",
            [s for s in stats if s.question_type is not None],
            active_comparisons,
            per_type=True,
            empty_note="(no per-type scores; n=0)",
        )
    )
    lines.extend(_saturation_excluded_sections(paired_scores, active_comparisons))

    if paired_scores:
        lines.extend(_per_arm_raw_stats_lines(paired_scores))
        lines.append("")

    if any(not math.isnan(s.score_pdf_min1) for s in paired_scores):
        lines.extend(_dual_panel_pdf_section(paired_scores, stats))
        lines.append("")

    lines.append("## Per-question diagnostic")
    if paired_scores:
        lines.extend(_per_question_diagnostic_table(paired_scores))
    else:
        lines.append("(no per-question scores; n=0)")
    lines.append("")

    if _has_confounder_data(paired_scores):
        lines.extend(_confounder_summary_lines(paired_scores))
        lines.append("")

    lines.extend(_caveat_lines(stats))
    return "\n".join(lines)
