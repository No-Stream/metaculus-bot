"""Audit: compare per-model reasoning on questions the bot got wrong.

The standard performance-analysis pipeline scores the ensemble's aggregate and
writes narrative post-mortems against the full bot comment. This module adds
a cut that ranks the *individual ensemble members* on each wrong question so
the per-question reasoning can be diffed — e.g. "gpt-5.2 got closest at 55%,
the ensemble dragged it down to 22%."

External human-forecaster comments were the original intent but the Metaculus
``/api/comments/`` endpoint is restricted to the bot's own author id and
staff. As a workaround, this module also looks for manually-curated comments
at ``<audit_dir>/human_comments/<post_id>.md`` and inlines them verbatim.
"""

from __future__ import annotations

import logging
from pathlib import Path

from metaculus_bot.performance_analysis.collector import load_dataset
from metaculus_bot.performance_analysis.parsing import (
    _parse_probability,
    parse_per_model_forecasts,
)
from metaculus_bot.performance_analysis.scoring import brier_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def load_combined_dataset(q1_path: str, q2_path: str) -> list[dict]:
    """Load Q1 and Q2 datasets, dedupe by post_id (Q2 wins) and tag cohort."""
    q1 = load_dataset(q1_path)
    q2 = load_dataset(q2_path)
    q2_pids = {r["post_id"] for r in q2}

    merged: dict[int, dict] = {r["post_id"]: dict(r) for r in q1}
    for r in q2:
        merged[r["post_id"]] = dict(r)

    for pid, rec in merged.items():
        rec["_cohort"] = "Q2" if pid in q2_pids else "Q1"
    return list(merged.values())


def _record_peer_score(record: dict) -> float | None:
    """Return the Metaculus peer score on a record, or None if unavailable."""
    sd = record.get("metaculus_scores") or {}
    return sd.get("peer_score")


def select_worst_misses(
    records: list[dict],
    n_binary: int = 10,
    n_numeric: int = 5,
    n_mc: int = 2,
    extra_post_ids: list[int] | None = None,
) -> list[dict]:
    """Pick the worst records per question type, ranked by peer score.

    Peer score compares our log score to the crowd's mean log score (negative =
    worse than peers). When populated on ``metaculus_scores.peer_score``, it's
    the canonical ranking signal because it's comparable across question types
    and accounts for question difficulty.

    Records missing peer score fall back to absolute scores:
    - Binary: highest Brier (worse = farther from truth).
    - Numeric/discrete/MC: lowest log score (worse = more confidently wrong).

    Records that are None on both peer and the type-specific fallback are
    dropped.

    ``extra_post_ids`` unions a manually-curated list of post_ids onto the
    selection (deduplicated, appended in input order). Use this to include
    questions the user wants audited regardless of the automatic ranking —
    e.g. spot_peer_score outliers not caught by the peer_score top-N.

    Returns concatenated list in (binary, numeric, mc, extras) order.
    """

    def _rank_key_binary(r: dict) -> tuple[int, float]:
        peer = _record_peer_score(r)
        if peer is not None:
            return (0, peer)  # primary: peer ascending (most negative = worst)
        brier = r.get("brier_score")
        if brier is None:
            return (2, 0.0)
        return (1, -brier)  # fallback: higher Brier = worse

    def _rank_key_logscore(r: dict, field: str) -> tuple[int, float]:
        peer = _record_peer_score(r)
        if peer is not None:
            return (0, peer)
        log = r.get(field)
        if log is None:
            return (2, 0.0)
        return (1, log)  # fallback: lower log = worse

    def _has_rankable_score(r: dict, fallback_field: str) -> bool:
        return _record_peer_score(r) is not None or r.get(fallback_field) is not None

    worst_binary = sorted(
        [r for r in records if r.get("type") == "binary" and _has_rankable_score(r, "brier_score")],
        key=_rank_key_binary,
    )[:n_binary]
    worst_numeric = sorted(
        [
            r
            for r in records
            if r.get("type") in ("numeric", "discrete") and _has_rankable_score(r, "numeric_log_score")
        ],
        key=lambda r: _rank_key_logscore(r, "numeric_log_score"),
    )[:n_numeric]
    worst_mc = sorted(
        [r for r in records if r.get("type") == "multiple_choice" and _has_rankable_score(r, "mc_log_score")],
        key=lambda r: _rank_key_logscore(r, "mc_log_score"),
    )[:n_mc]

    selected = worst_binary + worst_numeric + worst_mc
    if extra_post_ids:
        seen = {r["post_id"] for r in selected}
        by_pid = {r["post_id"]: r for r in records}
        for pid in extra_post_ids:
            if pid in seen:
                continue
            rec = by_pid.get(pid)
            if rec is None:
                logger.warning(f"extra_post_ids includes {pid} but no matching record — skipped")
                continue
            selected.append(rec)
            seen.add(pid)
    return selected


# ---------------------------------------------------------------------------
# Per-model accuracy ranking
# ---------------------------------------------------------------------------


def rank_our_models_by_accuracy(record: dict) -> list[dict]:
    """Rank the ensemble members on a single binary record by Brier score.

    Returns a list of ``{model, prob, score}`` dicts ordered best-first.
    Only binary records are supported for now — numeric/MC per-model scoring
    would require the full CDF/probability vector per model which isn't
    always parseable from the comment.
    """
    if record.get("type") != "binary":
        return []
    resolution = record.get("resolution_parsed")
    if not isinstance(resolution, bool):
        return []
    per_model = record.get("per_model_forecasts") or {}

    ranked: list[dict] = []
    for model, raw in per_model.items():
        prob = _parse_probability(raw)
        if prob is None:
            logger.debug(f"Skipping unparseable per-model value: {model}={raw!r}")
            continue
        ranked.append(
            {
                "model": model,
                "prob": prob,
                "score": brier_score(prob, resolution),
                "raw": raw,
            }
        )
    ranked.sort(key=lambda d: d["score"])
    return ranked


# ---------------------------------------------------------------------------
# Markdown emission
# ---------------------------------------------------------------------------


def _format_score_header(record: dict) -> str:
    q_type = record["type"]
    parts: list[str] = []
    sd = record.get("metaculus_scores") or {}
    peer = sd.get("peer_score")
    spot_peer = sd.get("spot_peer_score")
    if peer is not None:
        parts.append(f"peer **{peer:+.1f}**")
    if spot_peer is not None:
        parts.append(f"spot peer **{spot_peer:+.1f}**")
    if q_type == "binary":
        b = record.get("brier_score")
        log = record.get("log_score")
        if b is not None:
            parts.append(f"Brier {b:.3f}")
        if log is not None:
            parts.append(f"log {log:.2f}")
    elif q_type in ("numeric", "discrete"):
        nlog = record.get("numeric_log_score")
        if nlog is not None:
            parts.append(f"numeric log {nlog:.2f}")
    elif q_type == "multiple_choice":
        mlog = record.get("mc_log_score")
        if mlog is not None:
            parts.append(f"MC log {mlog:.2f}")
    return " | ".join(parts) if parts else "no score"


def _format_our_prediction(record: dict) -> str:
    q_type = record["type"]
    if q_type == "binary":
        p = record.get("our_prob_yes")
        return f"P(YES)={p * 100:.1f}%" if p is not None else "unknown"
    if q_type == "multiple_choice":
        options = record.get("options") or []
        fvs = record.get("our_forecast_values") or []
        if options and len(fvs) == len(options):
            pairs = sorted(zip(options, fvs, strict=False), key=lambda t: -t[1])
            return ", ".join(f"{o}={p * 100:.1f}%" for o, p in pairs[:3]) + (", ..." if len(pairs) > 3 else "")
        return str(fvs)[:80]
    if q_type in ("numeric", "discrete"):
        fvs = record.get("our_forecast_values") or []
        if not fvs:
            return "unknown"
        # Show rough 10/50/90 from CDF if it's a 201-point CDF.
        if len(fvs) >= 201:
            return f"P10≈{fvs[20]:.3f}  P50≈{fvs[100]:.3f}  P90≈{fvs[180]:.3f}  (as CDF prob)"
        return str(fvs)[:80]
    return "unknown"


def _format_resolution(record: dict) -> str:
    raw = record.get("resolution_raw", "?")
    parsed = record.get("resolution_parsed")
    if isinstance(parsed, bool):
        return "YES" if parsed else "NO"
    if parsed is None:
        return str(raw)
    return f"{parsed} (raw: {raw})"


def _format_model_ranking(ranked: list[dict], record: dict) -> str:
    """Render the per-model ranking table."""
    if not ranked:
        return "_No per-model forecasts parsed from the comment._\n"

    lines = ["| rank | model | forecast | Brier | delta vs ensemble |", "|---|---|---|---|---|"]
    our = record.get("our_prob_yes")
    for i, r in enumerate(ranked, 1):
        delta = ""
        if our is not None:
            delta_pp = (r["prob"] - our) * 100.0
            delta = f"{delta_pp:+.1f}pp"
        lines.append(f"| {i} | {r['model']} | {r['raw']} | {r['score']:.3f} | {delta} |")
    return "\n".join(lines) + "\n"


EXTERNAL_COMMENTS_DIRNAME = "external_forecaster_comments"


def _load_external_comments(post_id: int, audit_dir: Path) -> str | None:
    """Return the body of ``<audit_dir>/external_forecaster_comments/<post_id>.md`` if present.

    "External" means any forecaster other than our own bot — other competing
    bots and a smaller number of humans. Metaculus's ``/api/comments/``
    endpoint is restricted to the bot's own author id, so these comments are
    curated manually (pasted from the web UI) rather than fetched.
    """
    path = audit_dir / EXTERNAL_COMMENTS_DIRNAME / f"{post_id}.md"
    if not path.exists():
        return None
    text = path.read_text().strip()
    # Skip if file is still the placeholder stub (only a header, no body).
    if text.count("\n") < 2 or text.startswith("<!-- PLACEHOLDER"):
        return None
    return text


def emit_miss_markdown(
    record: dict,
    ranked_models: list[dict],
    per_model_reasoning: dict[str, str],
    audit_dir: Path,
    out_path: Path,
) -> None:
    """Write a per-question audit markdown file."""
    post_id = record["post_id"]
    title = record.get("title", "(no title)")
    q_type = record["type"]
    cohort = record.get("_cohort", "?")
    url = f"https://www.metaculus.com/questions/{post_id}/"
    meta = record.get("metadata") or {}

    lines: list[str] = []
    lines.append(f"# {title}\n")
    lines.append(f"- **Metaculus**: {url}")
    lines.append(f"- **post_id**: {post_id}  |  **cohort**: {cohort}  |  **type**: {q_type}")
    lines.append(f"- **Resolved**: {_format_resolution(record)}")
    lines.append(f"- **Our prediction**: {_format_our_prediction(record)}")
    lines.append(f"- **Score**: {_format_score_header(record)}")
    lines.append(f"- **nr_forecasters**: {meta.get('nr_forecasters', 'n/a')}")
    was_stacked = record.get("was_stacked")
    lines.append(f"- **was_stacked**: {'unknown' if was_stacked is None else was_stacked}")
    category = meta.get("category")
    if category:
        lines.append(f"- **category**: {category}")
    lines.append("")

    # Per-model ranking (binary only for now — ranking other types is harder
    # and would need the full CDF/probability vector per model).
    lines.append("## Per-model accuracy ranking")
    lines.append("")
    lines.append(_format_model_ranking(ranked_models, record))

    # Flag truncation so the reader knows per-model reasoning may be partial.
    summary_models = set(parse_per_model_forecasts(record.get("comment_text") or "").keys())
    reasoning_models = set(per_model_reasoning.keys())
    missing = summary_models - reasoning_models
    if missing:
        lines.append(f"> ⚠️ Reasoning missing for {sorted(missing)} — comment likely trimmed to COMMENT_CHAR_LIMIT.")
        lines.append("")

    # Per-model reasoning, ordered by rank when available so best-first.
    lines.append("## Per-model reasoning")
    lines.append("")
    ordered_models: list[str]
    if ranked_models:
        ordered_models = [r["model"] for r in ranked_models if r["model"] in per_model_reasoning]
        # Append any reasoning-only models (ranking skipped them — non-binary or unparseable).
        for m in per_model_reasoning:
            if m not in ordered_models:
                ordered_models.append(m)
    else:
        ordered_models = list(per_model_reasoning.keys())

    for model in ordered_models:
        prose = per_model_reasoning.get(model, "")
        forecast_str = ""
        for r in ranked_models:
            if r["model"] == model:
                forecast_str = f" — forecast **{r['raw']}** (Brier {r['score']:.3f})"
                break
        lines.append(f"### {model}{forecast_str}\n")
        lines.append(prose)
        lines.append("")

    # Optional external-forecaster-comment inclusion (other bots + humans,
    # curated manually since the comments API is restricted).
    external = _load_external_comments(post_id, audit_dir)
    if external is not None:
        lines.append("## External forecaster comments (other bots + humans, manually curated)")
        lines.append("")
        lines.append(external)
        lines.append("")

    lines.append("## Diff notes")
    lines.append("")
    lines.append("<!-- Add cross-model and bot-vs-external observations during synthesis. -->")
    lines.append("")

    out_path.write_text("\n".join(lines))


def emit_external_comment_stub(record: dict, out_path: Path) -> None:
    """Write a placeholder external_forecaster_comments/<post_id>.md if none exists."""
    if out_path.exists():
        return
    post_id = record["post_id"]
    title = record.get("title", "(no title)")
    url = f"https://www.metaculus.com/questions/{post_id}/"
    resolution = _format_resolution(record)
    out_path.write_text(
        f"<!-- PLACEHOLDER: paste top comments from other bots / humans below this line, then delete this marker. -->\n\n"
        f"# {title}\n\n"
        f"- URL: {url}\n"
        f"- Resolved: {resolution}\n\n"
        f"## Comments\n\n"
        f"_Paste 2-3 top comments here (usually other bots, occasionally humans)._\n"
    )


# ---------------------------------------------------------------------------
# Synthesis + combined report
# ---------------------------------------------------------------------------


def emit_synthesis(records_with_rankings: list[dict], out_path: Path) -> None:
    """Write a cross-cutting synthesis of model-level patterns across all misses."""
    lines: list[str] = []
    lines.append("# Audit synthesis — bot misses vs per-model dissent\n")
    lines.append(
        "For each worst-scored question, which ensemble members were closest to the truth, "
        "and how far did the ensemble aggregate sit from the best member?\n"
    )

    # Per-model "best finisher" tally — how often was each model the closest?
    best_model_tally: dict[str, int] = {}
    avoided_model_tally: dict[str, int] = {}
    binary_count = 0
    ensemble_vs_best_deltas: list[tuple[int, str, float, float, float]] = []
    for entry in records_with_rankings:
        rec = entry["record"]
        ranked = entry["ranked"]
        if rec.get("type") != "binary" or not ranked:
            continue
        binary_count += 1
        best = ranked[0]
        worst = ranked[-1]
        best_model_tally[best["model"]] = best_model_tally.get(best["model"], 0) + 1
        avoided_model_tally[worst["model"]] = avoided_model_tally.get(worst["model"], 0) + 1
        our = rec.get("our_prob_yes")
        if our is not None:
            ensemble_brier = rec.get("brier_score")
            best_brier = best["score"]
            delta = ensemble_brier - best_brier if ensemble_brier is not None else 0.0
            ensemble_vs_best_deltas.append((rec["post_id"], best["model"], ensemble_brier or 0.0, best_brier, delta))

    lines.append(f"## Scope\n\n- {binary_count} binary misses ranked.\n")

    if best_model_tally:
        lines.append("## Closest-to-truth tally (binary worst misses)\n")
        lines.append("| model | times best | times worst |")
        lines.append("|---|---|---|")
        all_models = sorted(set(best_model_tally) | set(avoided_model_tally))
        for m in sorted(all_models, key=lambda k: -best_model_tally.get(k, 0)):
            lines.append(f"| {m} | {best_model_tally.get(m, 0)} | {avoided_model_tally.get(m, 0)} |")
        lines.append("")

    if ensemble_vs_best_deltas:
        lines.append("## Ensemble-vs-best delta (binary, sorted by worst regret)\n")
        lines.append("| post | best model | ensemble Brier | best-model Brier | Δ (lost by aggregating) |")
        lines.append("|---|---|---|---|---|")
        for pid, best_m, ens, best, delta in sorted(ensemble_vs_best_deltas, key=lambda t: -t[4]):
            lines.append(f"| {pid} | {best_m} | {ens:.3f} | {best:.3f} | {delta:+.3f} |")
        lines.append("")

    # Spread buckets — large per-model spread + wrong ensemble = stacking case.
    high_spread_miss = []
    for entry in records_with_rankings:
        ranked = entry["ranked"]
        if len(ranked) < 2:
            continue
        probs = [r["prob"] for r in ranked]
        spread = max(probs) - min(probs)
        high_spread_miss.append(
            (
                entry["record"]["post_id"],
                entry["record"].get("title", "")[:60],
                spread,
                entry["record"].get("brier_score"),
                entry["record"].get("was_stacked"),
            )
        )
    if high_spread_miss:
        lines.append("## High-spread misses (dissenting-model signal)\n")
        lines.append("| post | title | best-vs-worst spread | Brier | was_stacked |")
        lines.append("|---|---|---|---|---|")
        for pid, t, s, b, st in sorted(high_spread_miss, key=lambda x: -x[2])[:10]:
            st_s = "unknown" if st is None else str(st)
            b_s = f"{b:.3f}" if b is not None else "n/a"
            lines.append(f"| {pid} | {t} | {s * 100:.1f}pp | {b_s} | {st_s} |")
        lines.append("")

    lines.append("## Manual synthesis\n")
    lines.append("<!-- Populate by hand after reading per-question miss_*.md files. -->\n")

    out_path.write_text("\n".join(lines))


def emit_combined_report(
    records_with_rankings: list[dict],
    miss_paths: list[Path],
    synthesis_path: Path,
    out_path: Path,
) -> None:
    """Write one combined markdown report that inlines each per-question file plus synthesis."""
    lines: list[str] = []
    lines.append("# Audit: bot misses (combined report)\n")
    lines.append("## Table of contents\n")
    for entry, path in zip(records_with_rankings, miss_paths, strict=False):
        rec = entry["record"]
        anchor = f"miss-{rec['post_id']}"
        lines.append(f"- [{rec['post_id']} — {rec.get('title', '')[:70]}](#{anchor})")
    lines.append("- [Synthesis](#synthesis)")
    lines.append("")

    for entry, path in zip(records_with_rankings, miss_paths, strict=False):
        rec = entry["record"]
        lines.append(f"<a id='miss-{rec['post_id']}'></a>\n")
        lines.append(path.read_text())
        lines.append("\n---\n")

    lines.append("<a id='synthesis'></a>\n")
    lines.append(synthesis_path.read_text())

    out_path.write_text("\n".join(lines))
