"""Markdown rendering for the clip-threshold sweep.

Presentation only: every number here comes off a :class:`ClipSweepReport` unchanged, and
``--output-json`` writes the same objects, so a reader and a script can never disagree.
Three rendering rules earn their place. A cell whose value is unknown prints ``n/a`` rather
than 0, because a measured zero and an unmeasurable quantity are different claims. A row
that moved no record prints ``identity`` where its CI would go, because the bootstrap of an
all-zero vector is not a precision statement. And a row carrying censored records is marked
``cen``, with its bounds in the loosening table rather than a point estimate, so nobody
reads an unobservable candidate as a neutral one.
"""

from __future__ import annotations

from metaculus_bot.constants import MC_PROB_MIN
from metaculus_bot.performance_analysis.clip_threshold_selection import OOB_BOOTSTRAP_B, OobArgmax
from metaculus_bot.performance_analysis.clip_threshold_sweep import (
    BINARY,
    BOOTSTRAP_B,
    MC_UNSHIPPABLE_NOTE,
    REPLAY_DISAGREE_ATOL,
    SweepRow,
)
from metaculus_bot.performance_analysis.clip_threshold_tables import (
    MIN_OOS_COMPLEMENT_N,
    ArgmaxSelection,
    ClipSweepReport,
    ExtremeBin,
    InsuranceRow,
    OosRow,
    RegimeSpan,
    ThinFloorReport,
    TypeReport,
)
from metaculus_bot.performance_analysis.clip_threshold_windows import (
    LOOKBACK_DAYS,
    WINDOW_ALL,
    WINDOW_ERA_PRE_FLIP,
    WINDOW_TRIPLE_ERA,
    Window,
    nested_windows,
    window_labels,
)
from metaculus_bot.performance_analysis.markdown import markdown_table

_NA = "n/a"
_IDENTITY = "identity"


def _num(value: float | None, spec: str = "+.2f") -> str:
    return _NA if value is None else format(value, spec)


def _ci(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return _NA
    return f"[{lo:+.3f}, {hi:+.3f}]"


def _ci_or_identity(row: SweepRow) -> str:
    """A row that moved nothing has no interval worth printing; say so instead."""
    return _IDENTITY if row.identity else _ci(row.ci_lo, row.ci_hi)


def _window_summary(window: Window) -> str:
    note = f", oversize: asked {window.requested_n}" if window.oversize else ""
    return f"{window.label} n={len(window.records)}{note}"


def _regime_line(span: RegimeSpan, unit: str) -> str:
    bound = (
        "so the clamp now in force has bound NO publish since it went live: the loosening question is "
        "unmeasurable here, and every tightening row is priced on records the live clamp never touched"
        if span.n_at_or_below_floor == 0 and span.n_at_or_above_ceiling == 0
        else "so the clamp now in force has bound at least one publish in this regime"
    )
    if span.n == 0:
        return f"  - live clamp regime (window {span.window}): no records"
    return (
        f"  - live clamp regime (window {span.window}, n={span.n}): published {unit} values span "
        f"[{_num(span.min_value, '.3f')}, {_num(span.max_value, '.3f')}]; at or below the {span.floor} floor: "
        f"{span.n_at_or_below_floor}; at or above the {span.ceiling} ceiling: {span.n_at_or_above_ceiling}; {bound}."
    )


def _header(report: ClipSweepReport) -> list[str]:
    exclusions = (
        f"{len(report.exclude_qids)} question id(s) requested, {report.n_excluded} matched"
        if report.exclude_qids
        else "none"
    )
    lines = [
        "# Counterfactual clip-threshold sweep",
        "",
        f"- dataset: `{report.dataset_path}`",
        f"- as_of: {report.as_of.isoformat()} (the `last_{LOOKBACK_DAYS}d` window is the {LOOKBACK_DAYS} days before this)",
        f"- exclusions: {exclusions}",
        "- spot-peer convention: a move of OUR mass on the resolving outcome is worth 100*ln(new/old) "
        "for binary and multiple_choice (neither is halved)",
        f"- bootstrap: B={BOOTSTRAP_B} percentile CI of the mean delta, resampling questions; "
        f"the out-of-bag argmax check refits on B={OOB_BOOTSTRAP_B} resamples",
    ]
    for type_report in report.types:
        cohort = type_report.cohort
        unit = "p_yes" if type_report.question_type == BINARY else "option"
        lines.append(
            f"- {type_report.question_type}: n={len(cohort.records)} "
            f"(skipped {cohort.n_skipped}, undated {cohort.n_no_timestamp}, "
            f"published at the in-force floor {cohort.n_at_floor}, with a clamped member in a median "
            f"position {cohort.n_member_censored_floor}, at the in-force ceiling {cohort.n_at_ceiling}); "
            f"grid c = {', '.join(f'{c:.4f}' for c in type_report.grid)}"
        )
        lines.append(f"  - windows: {'; '.join(_window_summary(w) for w in type_report.windows)}")
        if type_report.regime_span is not None:
            lines.append(_regime_line(type_report.regime_span, unit))
    lines += [
        "",
        "## How to read this",
        "",
        "- `sum_delta` is the window's whole spot-peer gain from moving to floor `c`; `mean_delta` "
        "divides it by the window's n (not by the affected count), so it reads as points per "
        "question forecast.",
        "- `hits+` counts the AFFECTED records where the clip moved mass toward the outcome that "
        "resolved: the count that decides whether a floor paid.",
        "- `top1` is the largest |delta| over the sum of |delta|, and `top1 question` names it with "
        "its real spot peer in brackets. At 1.00 a single question is the entire row; read it beside "
        "`n_aff`, since a row with one affected record is always 1.00.",
        f"- `{_IDENTITY}` in a CI cell means the candidate moved no record: `sum_delta` is a hard 0 and "
        "the interval would be the bootstrap of an all-zero vector, which is not a precision statement.",
        "- `cen` counts records published exactly at the clamp in force, so the raw member value is "
        "gone; `cen_m` counts records where a clamped MEMBER sat where it could move the publish "
        "(a median position, or any position under a mean aggregator), which is the count that "
        "actually bounds a looser clip and is never smaller than `cen`. A row with either above 0 is "
        "NOT exact for a looser candidate: read its bounds in the loosening section, and note the "
        "in-window argmax skips any row with either count above 0.",
        "- `infeas` counts MC records with more options than `1 / c`, on which the floor cannot be "
        "delivered at all (eleven options each at least 0.10 already exceed 1): the live clamp "
        "returns its sub-floor fallback there, so the cell prices that fallback, not the floor it "
        "is labelled with. Always 0 on a binary row.",
        "- The `all`, `last_N`, `last_90d`, `current_clamp_regime` and `triple_era` windows are NESTED "
        "(each is a subset of `all`), so agreement across them is one measurement re-counted, not "
        "replication; the `era_*` windows are the disjoint config-era slices, with `triple_era` as the "
        "third slice.",
        "- The insurance table per window says what a floor would have to be insuring against: "
        "`break-even r*` is the clipped-side hit rate at which the floor pays, `E[sum] own p` is what "
        "the same clip costs a perfectly calibrated forecaster under the bot's own prices (spot peer is "
        "proper, so this is never positive), and `best case` is the most the floor could have earned "
        "had every moved record resolved the clip's way.",
        "- An MC candidate below "
        f"{MC_PROB_MIN} is reported but marked `{MC_UNSHIPPABLE_NOTE}`: ft 0.2.92's option "
        "validator re-clamps to [0.01, 0.99] whatever we send.",
        "",
    ]
    return lines


def _extreme_bin_section(type_report: TypeReport) -> list[str]:
    unit = "question" if type_report.question_type == BINARY else "option"
    lines = [
        f"### Extreme-bin calibration ({unit}s, window {WINDOW_ALL})",
        "",
        "`hits` is the outcome the extreme price nearly ruled out: a YES for a low bin, a NO for a "
        "high one. `implied` is the bin midpoint expressed as the rate of that same event, and `exp "
        "hits` is what the bot's OWN published prices summed to for the records in the bin. A hit "
        "count ABOVE `exp hits` means the bot's extreme calls were too confident and a tighter floor "
        "pays; below, and the floor costs.",
        "",
    ]
    if type_report.question_type != BINARY:
        lines += [
            "MC bins are per OPTION, and options inside one question are correlated, so these "
            "intervals are not question-clustered. Only the low bins are reported: no archived "
            "option is priced high enough to enter the mirrored table.",
            "",
        ]
    rows = [
        [
            b.label,
            b.side,
            str(b.n),
            str(b.hits),
            f"{b.expected_hits:.2f}",
            _num(b.hit_rate, ".3f"),
            _ci(b.ci_lo, b.ci_hi).replace("+", ""),
            f"{b.implied_rate:.3f}",
        ]
        for b in type_report.extreme_bins_for(WINDOW_ALL)
    ]
    lines += markdown_table(
        ["bin", "side", "n", "hits", "exp hits (own p)", "hit rate", "Jeffreys 95%", "implied"], rows
    )
    lines.append("")
    lines += _extreme_bin_per_window_table(type_report)
    return lines


def _extreme_bin_per_window_table(type_report: TypeReport) -> list[str]:
    """The same bins per window as ``n/hits``, because the pooled table mixes clamp regimes.

    The pooled table above is dominated by the pre-flip era, where the floor in force was
    0.01; the recent windows are the ones a clip decision applies to, and there the extreme
    bins are nearly empty. Counts only: a Jeffreys interval on an n of 0-3 says nothing the
    counts do not, and the full per-window intervals are in the JSON. Every populated window
    carries every bin label, so a missing key here is a construction bug and raises.
    """
    windows = [w.label for w in type_report.populated_windows]
    labels = [b.label for b in type_report.extreme_bins_for(WINDOW_ALL)]
    by_key = {(b.window, b.label): b for b in type_report.extreme_bins}
    rows = [[label] + [_bin_cell(by_key[window, label]) for window in windows] for label in labels]
    return [
        "Per window, as `n/hits` (the pooled table above is mostly the pre-flip era):",
        "",
        *markdown_table(["bin", *windows], rows),
        "",
    ]


def _bin_cell(b: ExtremeBin) -> str:
    return f"{b.n}/{b.hits}"


def _sweep_row_cells(row: SweepRow) -> list[str]:
    return [
        f"{row.c:.4f}",
        str(row.n),
        str(row.n_affected),
        str(row.censored_n),
        str(row.member_censored_n),
        str(row.infeasible_n),
        _num(row.sum_delta),
        _num(row.mean_delta, "+.4f"),
        _ci_or_identity(row),
        str(row.hits_on_clipped_side),
        _num(row.top1_share, ".2f"),
        _driver_cell(row),
        "yes" if row.shippable else "NO",
    ]


def _driver_cell(row: SweepRow) -> str:
    """The question carrying the largest |delta| in the row, with its real spot peer."""
    if row.top1_question_id is None:
        return "-"
    peer = "" if row.top1_spot_peer is None else f" ({row.top1_spot_peer:+.1f})"
    return f"q{row.top1_question_id}{peer}"


_SWEEP_HEADER = [
    "c",
    "n",
    "n_aff",
    "cen",
    "cen_m",
    "infeas",
    "sum_delta",
    "mean_delta",
    "95% CI of mean",
    "hits+",
    "top1",
    "top1 question (spot peer)",
    "shippable",
]


def _full_sweep_section(type_report: TypeReport, side: str, title: str) -> list[str]:
    lines = [f"### {title}", ""]
    for window in type_report.populated_windows:
        rows = type_report.sweep_rows(side=side, window=window.label)
        lines.append(f"#### {_window_summary(window)}")
        lines.append("")
        lines += markdown_table(_SWEEP_HEADER, [_sweep_row_cells(row) for row in rows])
        lines.append("")
        if side == "floor_only":
            lines += _insurance_table(type_report.insurance_rows(window.label))
    return lines


def _insurance_cells(row: InsuranceRow) -> list[str]:
    verdict = _NA
    if row.rejected_at_ci_upper is not None:
        verdict = "rejected at CI upper" if row.rejected_at_ci_upper else "not closed by the CI"
    return [
        f"{row.c:.4f}",
        str(row.n_affected),
        str(row.hits),
        _num(row.hit_rate, ".4f"),
        _ci(row.ci_lo, row.ci_hi).replace("+", ""),
        _num(row.break_even_rate, ".4f"),
        _num(row.p_hits_at_most_if_rate_c, ".4f"),
        _num(row.expected_sum_delta),
        _num(row.best_case_sum_delta),
        _num(row.sum_delta),
        verdict,
    ]


def _insurance_table(rows: list[InsuranceRow]) -> list[str]:
    moved = [row for row in rows if row.n_affected]
    if not moved:
        return ["Insurance view: no candidate moves a record in this window.", ""]
    return [
        "Insurance view (floor_only, candidates that move at least one record; break-even and the "
        "binomial column are binary-only):",
        "",
        *markdown_table(
            [
                "c",
                "n_aff",
                "hits+",
                "hit rate",
                "Jeffreys 95%",
                "break-even r*",
                "P(hits<=obs | rate=c)",
                "E[sum] own p",
                "best case",
                "realized",
                "verdict",
            ],
            [_insurance_cells(row) for row in moved],
        ),
        "",
    ]


def _tie_cell(selection: ArgmaxSelection) -> str:
    """The argmax plateau, plus the censored candidates that scored the same and lost on censoring."""
    parts: list[str] = []
    if len(selection.tied_c) > 1:
        parts.append(f"{len(selection.tied_c)}: {selection.tied_c[0]:.4f}-{selection.tied_c[-1]:.4f}")
    if selection.censored_tie_c:
        parts.append(
            f"+{len(selection.censored_tie_c)} censored at the same score "
            f"({', '.join(f'{c:.4f}' for c in selection.censored_tie_c)})"
        )
    return "; ".join(parts) if parts else "-"


def _argmax_section(type_report: TypeReport, side: str, title: str) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "The best EXACT candidate per window (rows with censored records are skipped, since their "
        "`sum_delta` of 0 would read as neutral rather than unmeasurable; a censored candidate that "
        "scored the same as the winner is named in `tied` so a strict win and a win-by-exclusion are "
        "not confused). `tied` is otherwise the plateau: every candidate at or below a window's "
        "in-force floor scores exactly 0 when no publish in that window was clamped, so the reported "
        "`argmax c` is the smallest member of the tie and carries no preference for loosening over "
        "the status quo.",
        "",
    ]
    rows: list[list[str]] = []
    for window in type_report.populated_windows:
        selection = type_report.argmax_for(side=side, window=window.label)
        best = (
            None
            if selection.argmax_c is None
            else next(r for r in type_report.sweep_rows(side=side, window=window.label) if r.c == selection.argmax_c)
        )
        rows.append(
            [
                window.label,
                str(len(window.records)),
                _NA if best is None else f"{best.c:.4f}",
                _tie_cell(selection),
                _NA if best is None else _num(best.sum_delta),
                _NA if best is None else _num(best.mean_delta, "+.4f"),
                _NA if best is None else _ci_or_identity(best),
                _NA if best is None else str(best.n_affected),
                _NA if best is None else str(best.hits_on_clipped_side),
                _NA if best is None else _driver_cell(best),
            ]
        )
    lines += markdown_table(
        ["window", "n", "argmax c", "tied", "sum_delta", "mean_delta", "95% CI", "n_aff", "hits+", "top1 question"],
        rows,
    )
    lines.append("")
    return lines


def _oob_cells(row: OobArgmax) -> list[str]:
    return [
        row.window,
        str(row.n),
        _NA if row.in_window_c is None else f"{row.in_window_c:.4f}",
        _num(row.in_window_mean_delta, "+.4f"),
        _num(row.oob_mean_delta, "+.4f"),
        _ci(row.oob_ci_lo, row.oob_ci_hi),
        _num(row.shrinkage, "+.4f"),
        str(row.n_candidates),
    ]


def _oob_section(type_report: TypeReport) -> list[str]:
    return [
        "### Out-of-bag value of the fitted argmax (floor_only)",
        "",
        "The CI on the argmax row above ignores that the row was CHOSEN over the grid. This refits "
        f"the argmax on each of {OOB_BOOTSTRAP_B} bootstrap resamples of the window and scores the "
        'fitted candidate on the records the resample left out, so `oob mean` is what "pick the '
        'best floor, then apply it" is actually worth per question, and `shrinkage` is the part of '
        "the in-window figure that was selection. On a window whose argmax is the do-nothing "
        "candidate both columns are 0 by construction.",
        "",
        *markdown_table(
            ["window", "n", "in-window c", "in-window mean", "oob mean", "oob 95%", "shrinkage", "candidates"],
            [_oob_cells(row) for row in type_report.oob],
        ),
        "",
    ]


def _nesting_section(type_report: TypeReport) -> list[str]:
    nested = [w.label for w in nested_windows(type_report.populated_windows)]
    rows = [
        [f"{row.c:.4f}"] + [str(row.n_affected_by_window[label]) for label in nested] + [str(row.n_distinct)]
        for row in type_report.nesting
        if row.n_distinct
    ]
    lines = [
        "### Affected-set nesting (floor_only)",
        "",
        "How many questions each nested window moves at `c`, and how many DISTINCT questions that is "
        "across all of them. The distinct count equals the `all` count because every other nested "
        "window is a subset of it: the windows re-count one set of records at several sample sizes. "
        f"The disjoint slices are `{WINDOW_ERA_PRE_FLIP}`, `era_post_flip` and `{WINDOW_TRIPLE_ERA}`.",
        "",
    ]
    if rows:
        lines += markdown_table(["c", *nested, "distinct"], rows)
    else:
        lines.append("- No candidate moves a record in any window.")
    lines.append("")
    return lines


def _compact_sweep_section(type_report: TypeReport, side: str, title: str) -> list[str]:
    """Only the rows that moved something, per window, plus an explicit nothing-moved line."""
    lines = [f"### {title}", ""]
    if side == "ceiling_only":
        lines += [
            "A ceiling-only tightening is the hard-clip form of a YES-side shrink, a layer this repo "
            "has measured and killed as era-local before; read a positive row here against the "
            "`era_*` slices and the out-of-bag table, not on its own.",
            "",
        ]
    for window in type_report.populated_windows:
        rows = [r for r in type_report.sweep_rows(side=side, window=window.label) if r.n_affected or r.censored_n]
        if not rows:
            lines.append(f"- {window.label}: no record is affected or censored at any candidate.")
            continue
        lines.append(f"#### {_window_summary(window)}")
        lines.append("")
        lines += markdown_table(_SWEEP_HEADER, [_sweep_row_cells(row) for row in rows])
        lines.append("")
    lines.append("")
    return lines


def _live_floor_on_older_regime(type_report: TypeReport) -> list[str]:
    """The floor now in force, priced on the records published under the OLDER clamp.

    This is the one floor comparison the archive MEASURES rather than bounds: the older
    records were published under a looser clamp, so today's floor is a tightening on them.
    The row is ``TypeReport.older_regime``, computed on the current-regime window's complement
    at the type's own in-force floor. It is era-confounded (a different roster, a different
    pipeline) and says so.
    """
    span = type_report.regime_span
    row = type_report.older_regime
    if span is None or row is None or row.identity:
        return []
    return [
        f"The floor now in force ({row.c}) priced on the {row.n} records published before the clamp in force "
        f"went live (window `{row.window}`), under the older, looser clamp: sum_delta {_num(row.sum_delta)} over "
        f"{row.n_affected} affected records with {row.hits_on_clipped_side} resolving on the clipped side (CI of "
        f"the mean {_ci(row.ci_lo, row.ci_hi)}). This is the only floor comparison the archive measures rather "
        "than bounds, and it is era-confounded: those records came from a different roster and pipeline.",
        "",
    ]


def _loosening_section(type_report: TypeReport) -> list[str]:
    lines = [
        "### Loosening bounds (candidates below the clamp in force)",
        "",
        "`at_floor` assumes every censored raw value was exactly the floor, so nothing moves; "
        "`at_c` assumes every censored PUBLISHED value was at or below `c`, so all of them move the "
        "whole way; `at_c (members)` runs the same scenario through the members (every clamped member "
        "assumed at `c`, re-aggregated under the record's own aggregator), which is the bound that "
        "reaches the `cen_m` records the published-value rule cannot see. The published-value pair are "
        "NOT a bracket when the signs disagree (a censored NO gains from a looser floor, a censored "
        "YES loses), which is what `bracket` is for: each censored record contributes its own best and "
        "worst case. For MC the bracket is the two-scenario span per record and the true set can be "
        "marginally wider, because the floored options could have differed from one another.",
        "",
    ]
    rows: list[list[str]] = []
    for window in type_report.populated_windows:
        for row in type_report.sweep_rows(side="floor_only", window=window.label):
            if not row.n_loosening:
                continue
            rows.append(
                [
                    window.label,
                    f"{row.c:.4f}",
                    str(row.n),
                    str(row.n_loosening),
                    str(row.censored_n),
                    str(row.member_censored_n),
                    _num(row.sum_delta),
                    _num(row.sum_delta_lower),
                    _num(row.sum_delta_upper),
                    _num(row.sum_delta_upper_members),
                    f"[{row.bracket_lo:+.2f}, {row.bracket_hi:+.2f}]",
                ]
            )
    if rows:
        lines += markdown_table(
            [
                "window",
                "c",
                "n",
                "n_loosen",
                "cen",
                "cen_m",
                "exact part",
                "at_floor",
                "at_c",
                "at_c (members)",
                "bracket",
            ],
            rows,
        )
    else:
        lines.append("- No candidate on either grid is looser than the clamp in force for any record.")
    lines.append("")
    lines += _live_floor_on_older_regime(type_report)
    return lines


def _oos_cells(row: OosRow) -> list[str]:
    fit_c = _NA if row.c_star is None else f"{row.c_star:.4f}"
    if row.fit_is_identity:
        fit_c += " (moves nothing)"
    return [
        row.window,
        str(row.n_window),
        str(row.n_complement),
        fit_c,
        _num(row.fit_sum_delta),
        _NA if row.fit_n_affected is None else str(row.fit_n_affected),
        _num(row.carried_sum_delta),
        _num(row.carried_mean_delta, "+.4f"),
        _IDENTITY if row.carried_n_affected == 0 else _ci(row.carried_ci_lo, row.carried_ci_hi),
        _NA if row.in_window_c_star is None else f"{row.in_window_c_star:.4f}",
        _num(row.in_window_sum_delta),
        str(row.n_tied_in_window),
        _num(row.carry_gap),
    ]


def _oos_section(type_report: TypeReport) -> list[str]:
    lines = [
        "### Out-of-sample carry (floor_only)",
        "",
        "A clip level is a fitted calibration layer, so it ships only if a fit on the PAST carries "
        f"into the window. `c*` is the argmax over the records older than the window; a complement "
        f"under {MIN_OOS_COMPLEMENT_N} records prints `{_NA}` rather than a number, and undated records "
        "sit in no complement (which is why a complement can be smaller than `all` minus the window). "
        "Read `carry gap` (in-window best minus what `c*` scored inside the window), NOT a mismatch "
        "between the two `c*` labels: when several candidates tie in-window the labels differ while "
        "the scores do not. And read `fit n_aff` first: a fitted `c*` that moves NOTHING in its own "
        "complement is the do-nothing map, its carry of 0 is vacuous, and the row says nothing about "
        "whether a fitted floor generalises. Only a fit with `fit n_aff > 0` tests the era rule.",
        "",
    ]
    lines += markdown_table(
        [
            "window",
            "n_W",
            "n_complement",
            "fit c*",
            "fit sum_delta",
            "fit n_aff",
            "carried sum_delta on W",
            "carried mean",
            "carried 95% CI",
            "in-window c*",
            "in-window sum_delta",
            "in-window ties",
            "carry gap",
        ],
        [_oos_cells(row) for row in type_report.oos],
    )
    lines.append("")
    return lines


def _replay_sign_sentence(type_report: TypeReport) -> str:
    """The sign of the shortcut's error, derived from the counts rather than asserted."""
    summary = type_report.cross_check_summary
    if summary.n_differing == 0:
        return (
            "- sign of the shortcut's error: no (window, c) cell differs between the two paths, so the "
            "published-median shortcut is exact on this cohort."
        )
    direction = "understates" if summary.n_replay_more_negative * 2 > summary.n_differing else "overstates"
    return (
        f"- sign of the shortcut's error: the replay path is MORE negative than the published-vector "
        f"path in {summary.n_replay_more_negative} of the {summary.n_differing} (window, c) cells where the "
        f"two differ, so where the shortcut is wrong it mostly {direction} the cost of tightening."
    )


def _cross_check_section(type_report: TypeReport) -> list[str]:
    lines = [
        "### Per-model replay cross-check (floor_only)",
        "",
        "Clamping the PUBLISHED median is exact only because clamping is monotone and the median of "
        "an ODD number of members is an order statistic; with an even count the published value "
        "averages the two middle members, so the two paths can differ. This replays the members "
        "through the candidate clamp, under the aggregator that rebuilds each record's published "
        "value (the median, or the mean for the 2025-09 mean-aggregator era), and reports the gap "
        "instead of assuming it away. Stacked records and records with no recovered members are not "
        "replayable.",
        "",
    ]
    windows = [w.label for w in type_report.populated_windows]
    cohorts = {label: type_report.replay_cohort_for(label) for label in windows}
    by_key = {(row.window, row.c): row for row in type_report.cross_check}
    lines.append(
        "- replayable per window: "
        + "; ".join(f"{label} {cohorts[label].n_replayable}/{cohorts[label].n_records}" for label in windows)
    )
    lines.append(
        "- even member count (the approximate case) per window: "
        + "; ".join(f"{label} {cohorts[label].n_even_members}/{cohorts[label].n_replayable}" for label in windows)
    )
    lines.append(
        "- aggregator detected per window, mean / unknown: "
        + "; ".join(
            f"{label} {cohorts[label].n_mean_aggregator} / {cohorts[label].n_unknown_aggregator}" for label in windows
        )
        + ". A mean-aggregator record is replayed with the mean, so its baseline reproduces the "
        "publish; an unknown one is replayed as a median against its own baseline and is what the "
        "next line counts."
    )
    lines.append(
        "- replay does NOT reproduce the published value with no clip applied on: "
        + "; ".join(
            f"{label} {cohorts[label].n_baseline_mismatch} (max {cohorts[label].max_baseline_gap:.3f})"
            for label in windows
        )
    )
    lines.append(_replay_sign_sentence(type_report))
    if type_report.question_type == BINARY:
        lines.append(
            "- routing caveat: the counterfactual holds the stacking route fixed, but a tighter member "
            "clamp compresses the binary spread CONDITIONAL_STACKING routes on. Records whose spread "
            "would fall below the threshold under the candidate, per window: "
            + "; ".join(
                f"{label} "
                + ", ".join(
                    f"c={c:.3f}: {by_key[label, c].n_routing_flips}"
                    for c in type_report.grid
                    if (by_key[label, c].n_routing_flips or 0) > 0
                )
                for label in windows
                if any((by_key[label, c].n_routing_flips or 0) > 0 for c in type_report.grid)
            )
            + "."
        )
    lines.append("")
    lines.append(f"Records disagreeing by more than {REPLAY_DISAGREE_ATOL} of resolving mass (max gap in brackets):")
    lines.append("")
    lines += markdown_table(
        ["c", *windows],
        [
            [f"{c:.4f}"] + [f"{by_key[label, c].n_disagree} ({by_key[label, c].max_abs_gap:.4f})" for label in windows]
            for c in type_report.grid
        ],
    )
    lines.append("")
    lines.append("Replay `sum_delta` (published-vector `sum_delta` in brackets):")
    lines.append("")
    lines += markdown_table(
        ["c", *windows],
        [
            [f"{c:.4f}"]
            + [
                f"{by_key[label, c].sum_delta_replay:+.2f} ({by_key[label, c].sum_delta_published:+.2f})"
                for label in windows
            ]
            for c in type_report.grid
        ],
    )
    lines.append("")
    return lines


def _thin_floor_section(report: ThinFloorReport) -> list[str]:
    lines = [
        "### Single-survivor publishes and the thin publish floor",
        "",
        f"`apply_thin_publish_floor` clamps a binary publish into [{report.floor}, {report.ceiling}] ONLY "
        "when exactly one forecaster survived, so its cost is the sum over these records and no "
        "others. A one-member record counts as a genuine single-survivor publish only from "
        f"{report.boundary} (the merge that let MIN_FORECASTERS_TO_PUBLISH fall to 1); "
        f"{report.n_single_member_before_boundary} one-member record(s) predate it and are trimmed-comment "
        "parse artifacts, counted here and not priced. The dataset carries no `stacker_skip_reason` "
        "marker, so the member count plus the merge timestamp is the detector.",
        "",
    ]
    if not report.rows:
        lines += [
            "- No genuine single-survivor publish in this cohort. If exclusions are active, note that "
            "every such publish in the archive so far came from a degraded run, so the strict cohort "
            "excludes the one shape the thin floor exists for.",
            "",
        ]
        return lines
    lines += markdown_table(
        ["question", "published", "p_yes", "resolved", "thin-floored p_yes", "delta", "spot peer"],
        [
            [
                f"q{row.question_id}",
                (row.created_at or _NA)[:10],
                f"{row.p_yes:.3f}",
                "YES" if row.resolved_yes else "NO",
                f"{row.clamped_p_yes:.3f}",
                _num(row.delta),
                _num(row.spot_peer, "+.1f"),
            ]
            for row in report.rows
        ],
    )
    lines.append("")
    lines.append(
        f"Thin-floor sum_delta over the {len(report.rows)} genuine single-survivor publish(es): {_num(report.sum_delta)}."
    )
    lines.append("")
    return lines


def _type_section(type_report: TypeReport) -> list[str]:
    lines = [f"## {type_report.question_type}", ""]
    lines += _extreme_bin_section(type_report)
    lines += _argmax_section(type_report, "floor_only", "In-window argmax floor (floor_only)")
    lines += _oob_section(type_report)
    lines += _nesting_section(type_report)
    lines += _full_sweep_section(type_report, "floor_only", "Floor-only sweep, per window")
    lines += _compact_sweep_section(type_report, "ceiling_only", "Ceiling-only sweep (affected rows only)")
    lines += _argmax_section(type_report, "symmetric", "In-window argmax (symmetric, both sides at once)")
    lines += _full_sweep_section_all_window(type_report, "symmetric")
    lines += _loosening_section(type_report)
    lines += _oos_section(type_report)
    lines += _cross_check_section(type_report)
    if type_report.thin_floor is not None:
        lines += _thin_floor_section(type_report.thin_floor)
    return lines


def _full_sweep_section_all_window(type_report: TypeReport, side: str) -> list[str]:
    """The symmetric side in full for ``all`` only; every window of it is in the JSON."""
    rows = type_report.sweep_rows(side=side, window=WINDOW_ALL)
    lines = [
        f"#### Symmetric sweep, window {WINDOW_ALL} (every window is in the JSON)",
        "",
    ]
    lines += markdown_table(_SWEEP_HEADER, [_sweep_row_cells(row) for row in rows])
    lines.append("")
    return lines


def render_report(report: ClipSweepReport) -> str:
    """The whole sweep as markdown, header first, then one section per question type."""
    lines = _header(report)
    for type_report in report.types:
        lines += _type_section(type_report)
    missing = [
        label
        for type_report in report.types
        for label in window_labels(type_report.question_type)
        if label not in {w.label for w in type_report.populated_windows}
    ]
    if missing:
        lines += [
            "## Empty windows",
            "",
            "These window labels carry no record in this pull, so they have no sweep rows: "
            + ", ".join(sorted(set(missing)))
            + ".",
            "",
        ]
    return "\n".join(lines)
