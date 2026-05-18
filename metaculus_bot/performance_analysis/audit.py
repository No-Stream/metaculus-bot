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
import random
from pathlib import Path
from typing import Literal

from metaculus_bot.pchip_cdf import generate_pchip_cdf
from metaculus_bot.performance_analysis.collector import load_dataset, resolve_numeric_record_to_score_inputs
from metaculus_bot.performance_analysis.parsing import (
    _parse_probability,
    parse_per_model_forecasts,
)
from metaculus_bot.performance_analysis.scoring import brier_score, numeric_log_score

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


def _has_rankable_score(r: dict, fallback_field: str) -> bool:
    return _record_peer_score(r) is not None or r.get(fallback_field) is not None


def _filter_by_type(records: list[dict], q_types: tuple[str, ...], fallback_field: str) -> list[dict]:
    return [r for r in records if r.get("type") in q_types and _has_rankable_score(r, fallback_field)]


def _rank_key_worst_binary(r: dict) -> tuple[int, float]:
    peer = _record_peer_score(r)
    if peer is not None:
        return (0, peer)  # primary: peer ascending (most negative = worst)
    brier = r.get("brier_score")
    if brier is None:
        return (2, 0.0)
    return (1, -brier)  # fallback: higher Brier = worse


def _rank_key_worst_logscore(r: dict, field: str) -> tuple[int, float]:
    peer = _record_peer_score(r)
    if peer is not None:
        return (0, peer)
    log = r.get(field)
    if log is None:
        return (2, 0.0)
    return (1, log)  # fallback: lower log = worse


def _rank_key_best_binary(r: dict) -> tuple[int, float]:
    peer = _record_peer_score(r)
    if peer is not None:
        return (0, -peer)  # primary: peer descending (most positive = best)
    brier = r.get("brier_score")
    if brier is None:
        return (2, 0.0)
    return (1, brier)  # fallback: lower Brier = better


def _rank_key_best_logscore(r: dict, field: str) -> tuple[int, float]:
    peer = _record_peer_score(r)
    if peer is not None:
        return (0, -peer)
    log = r.get(field)
    if log is None:
        return (2, 0.0)
    return (1, -log)  # fallback: higher log = better


def _middle_band(records: list[dict]) -> list[dict]:
    """Return the records whose peer_score falls in the 20-80 percentile band.

    Records without a peer_score are dropped — middle-mode is peer-anchored
    only. This is intentional: the cohort is meant to surface "ordinary
    questions" relative to the crowd, which requires peer comparability.
    """
    with_peer = [r for r in records if _record_peer_score(r) is not None]
    if not with_peer:
        return []
    sorted_records = sorted(with_peer, key=lambda r: _record_peer_score(r) or 0.0)
    n = len(sorted_records)
    lower_idx = int(0.2 * n)
    upper_idx = max(lower_idx, int(0.8 * n))
    return sorted_records[lower_idx:upper_idx]


def select_cohort(
    records: list[dict],
    mode: Literal["worst", "best", "middle"],
    n_binary: int = 10,
    n_numeric: int = 5,
    n_mc: int = 2,
    extra_post_ids: list[int] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Pick records per question type, ranked by peer score under one of three modes.

    Peer score compares our log score to the crowd's mean log score (negative =
    worse than peers). When populated on ``metaculus_scores.peer_score``, it's
    the canonical ranking signal because it's comparable across question types
    and accounts for question difficulty.

    Modes:
    - ``"worst"``: peer ascending (most negative = worst). Fallback: highest
      Brier (binary) / lowest log score (numeric, MC).
    - ``"best"``: peer descending (most positive = best). Fallback: lowest
      Brier / highest log score.
    - ``"middle"``: rank by peer_score, drop the bottom 20% and top 20% as
      extremes, then ``random.Random(seed).sample(...)`` to draw N per type.
      Middle 60% (not pure random) so the cohort surfaces "ordinary"
      questions; pure random risks repeatedly drawing near-extremes that
      defeat the purpose. Records without peer_score are dropped from middle
      mode (no peer comparability → no "middle").

    Records missing both peer score and the type-specific fallback are
    dropped (worst/best modes). Middle mode additionally drops records
    without peer score regardless of fallback availability.

    ``extra_post_ids`` unions a manually-curated list of post_ids onto the
    selection (deduplicated, appended in input order). Useful for spot_peer
    outliers or known-interesting questions outside the auto-selected top-N.

    Returns concatenated list in (binary, numeric, mc, extras) order.
    """
    if mode == "worst":
        binary_pool = _filter_by_type(records, ("binary",), "brier_score")
        numeric_pool = _filter_by_type(records, ("numeric", "discrete"), "numeric_log_score")
        mc_pool = _filter_by_type(records, ("multiple_choice",), "mc_log_score")
        sel_binary = sorted(binary_pool, key=_rank_key_worst_binary)[:n_binary]
        sel_numeric = sorted(numeric_pool, key=lambda r: _rank_key_worst_logscore(r, "numeric_log_score"))[:n_numeric]
        sel_mc = sorted(mc_pool, key=lambda r: _rank_key_worst_logscore(r, "mc_log_score"))[:n_mc]
    elif mode == "best":
        binary_pool = _filter_by_type(records, ("binary",), "brier_score")
        numeric_pool = _filter_by_type(records, ("numeric", "discrete"), "numeric_log_score")
        mc_pool = _filter_by_type(records, ("multiple_choice",), "mc_log_score")
        sel_binary = sorted(binary_pool, key=_rank_key_best_binary)[:n_binary]
        sel_numeric = sorted(numeric_pool, key=lambda r: _rank_key_best_logscore(r, "numeric_log_score"))[:n_numeric]
        sel_mc = sorted(mc_pool, key=lambda r: _rank_key_best_logscore(r, "mc_log_score"))[:n_mc]
    elif mode == "middle":
        rng = random.Random(seed)
        middle = _middle_band(records)
        binary_pool = [r for r in middle if r.get("type") == "binary"]
        numeric_pool = [r for r in middle if r.get("type") in ("numeric", "discrete")]
        mc_pool = [r for r in middle if r.get("type") == "multiple_choice"]
        sel_binary = rng.sample(binary_pool, min(n_binary, len(binary_pool)))
        sel_numeric = rng.sample(numeric_pool, min(n_numeric, len(numeric_pool)))
        sel_mc = rng.sample(mc_pool, min(n_mc, len(mc_pool)))
    else:
        raise ValueError(  # type: ignore[reportUnreachable]
            f"Unknown mode {mode!r}; expected 'worst', 'best', or 'middle'"
        )

    selected = sel_binary + sel_numeric + sel_mc
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


def select_worst_misses(
    records: list[dict],
    n_binary: int = 10,
    n_numeric: int = 5,
    n_mc: int = 2,
    extra_post_ids: list[int] | None = None,
) -> list[dict]:
    """Backward-compat alias for ``select_cohort(mode='worst', ...)``.

    Preserved so the April audit driver script (``scratch/audit_2026-04/run_audit.py``)
    keeps working unchanged. Identical behavior to the worst-mode path of
    ``select_cohort``.
    """
    return select_cohort(
        records,
        mode="worst",
        n_binary=n_binary,
        n_numeric=n_numeric,
        n_mc=n_mc,
        extra_post_ids=extra_post_ids,
    )


# ---------------------------------------------------------------------------
# Per-model accuracy ranking
# ---------------------------------------------------------------------------


def rank_our_models_by_accuracy(record: dict) -> list[dict]:
    """Rank the ensemble members on a single record by per-model score.

    Binary: each ``{model: percentage_string}`` entry from
    ``per_model_forecasts`` is parsed to a probability and scored with Brier
    (lower is better). Sort is **ascending** by Brier (best first).

    Numeric / discrete: each ``{model: [(percentile, value), ...]}`` entry
    from ``per_model_numeric_percentiles`` is converted to a 201-point CDF
    via PCHIP, then scored with Metaculus-style ``numeric_log_score`` (higher
    is better). Sort is **descending** by score (best first). The return
    semantics differ between binary and numeric — both list "best first" but
    via different score directions; downstream consumers should treat the
    list as ordered, not the score values directly.

    For numeric / discrete, **we score the raw model output — pre tail-
    widening, pre validation**. This isolates *model judgment* from
    *post-processing* effects. The bot's actual published CDF goes through
    additional ``numeric_pipeline.py`` post-processing (tail widening, etc.)
    which is intentionally excluded from this audit-side scoring.

    Returns a list of dicts. Binary entries: ``{model, prob, score, raw}``.
    Numeric/discrete entries: ``{model, percentiles, score, raw}`` where
    ``raw`` is a short summary like ``"P10≈X P50≈Y P90≈Z"``.

    Empty list when:
    - Type isn't binary/numeric/discrete.
    - Resolution isn't usable (None binary; non-recognized numeric resolution).
    - Numeric scaling bounds are missing.
    - No parseable per-model entries.
    """
    q_type = record.get("type")
    if q_type == "binary":
        return _rank_binary(record)
    if q_type in ("numeric", "discrete"):
        return _rank_numeric(record)
    return []


def _rank_binary(record: dict) -> list[dict]:
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


def _rank_numeric(record: dict) -> list[dict]:
    score_inputs = resolve_numeric_record_to_score_inputs(record)
    if score_inputs is None:
        return []
    res_float, lower_bound, upper_bound, zero_point = score_inputs

    open_lower = bool(record.get("open_lower_bound", False))
    open_upper = bool(record.get("open_upper_bound", False))

    per_model = record.get("per_model_numeric_percentiles") or {}
    skipped_count = 0
    ranked: list[dict] = []
    post_id = record.get("post_id")
    for model, percentile_pairs in per_model.items():
        percentile_dict = {float(p): float(v) for p, v in percentile_pairs}
        try:
            cdf, _ = generate_pchip_cdf(
                percentile_dict,
                open_upper_bound=open_upper,
                open_lower_bound=open_lower,
                upper_bound=upper_bound,
                lower_bound=lower_bound,
                zero_point=zero_point,
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning(f"Per-model PCHIP failure post={post_id} model={model}: {exc}")
            skipped_count += 1
            continue

        try:
            score = numeric_log_score(
                cdf,
                res_float,
                lower_bound,
                upper_bound,
                open_lower,
                open_upper,
                zero_point,
            )
        except (ValueError, ZeroDivisionError) as exc:
            logger.warning(f"Per-model scoring failure post={post_id} model={model}: {exc}")
            skipped_count += 1
            continue

        ranked.append(
            {
                "model": model,
                "percentiles": percentile_pairs,
                "score": score,
                "raw": _summarize_percentiles(percentile_dict),
            }
        )

    # If the majority of models failed, the audit output for this question
    # would be misleading (empty or near-empty ranking) — surface loudly
    # rather than silently produce a degraded report.
    if per_model and skipped_count >= len(per_model) / 2:
        logger.error(
            f"Per-model numeric scoring degraded post={post_id}: "
            f"{skipped_count}/{len(per_model)} models failed; ranking unreliable"
        )

    ranked.sort(key=lambda d: -d["score"])
    return ranked


def _summarize_percentiles(percentile_dict: dict[float, float]) -> str:
    """Render a tiny "P10≈X P50≈Y P90≈Z" preview for the audit Markdown."""
    parts: list[str] = []
    for target in (10, 50, 90):
        nearest_key = min(percentile_dict.keys(), key=lambda k: abs(k - target))
        parts.append(f"P{target}≈{percentile_dict[nearest_key]:g}")
    return " ".join(parts)


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


_DEFAULT_SYNTHESIS_FRAMING: dict[str, str] = {
    "title": "Audit synthesis — bot misses vs per-model dissent",
    "intro_paragraph": (
        "For each worst-scored question, which ensemble members were closest to the truth, "
        "and how far did the ensemble aggregate sit from the best member?"
    ),
    "cohort_name": "binary worst misses",
    "tally_best_label": "times best",
    "tally_worst_label": "times worst",
    "delta_section_label": "Ensemble-vs-best delta (binary, sorted by worst regret)",
    "spread_section_label": "High-spread misses (dissenting-model signal)",
}


def emit_synthesis(
    records_with_rankings: list[dict],
    out_path: Path,
    framing: dict[str, str] | None = None,
) -> None:
    """Write a cross-cutting synthesis of model-level patterns across all misses.

    ``framing`` lets callers override the misses-cohort defaults so the same
    function can emit synthesis files for hits or middle cohorts (where labels
    like "times best / times worst" or "High-spread misses" no longer fit).
    Missing keys fall back to defaults; ``framing=None`` is identical to the
    pre-parameterization behavior.
    """
    f = {**_DEFAULT_SYNTHESIS_FRAMING, **(framing or {})}

    lines: list[str] = []
    lines.append(f"# {f['title']}\n")
    lines.append(f"{f['intro_paragraph']}\n")

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

    lines.append(f"## Scope\n\n- {binary_count} binary records ranked.\n")

    if best_model_tally:
        lines.append(f"## Closest-to-truth tally ({f['cohort_name']})\n")
        lines.append(f"| model | {f['tally_best_label']} | {f['tally_worst_label']} |")
        lines.append("|---|---|---|")
        all_models = sorted(set(best_model_tally) | set(avoided_model_tally))
        for m in sorted(all_models, key=lambda k: -best_model_tally.get(k, 0)):
            lines.append(f"| {m} | {best_model_tally.get(m, 0)} | {avoided_model_tally.get(m, 0)} |")
        lines.append("")

    if ensemble_vs_best_deltas:
        lines.append(f"## {f['delta_section_label']}\n")
        lines.append("| post | best model | ensemble Brier | best-model Brier | Δ (lost by aggregating) |")
        lines.append("|---|---|---|---|---|")
        for pid, best_m, ens, best, delta in sorted(ensemble_vs_best_deltas, key=lambda t: -t[4]):
            lines.append(f"| {pid} | {best_m} | {ens:.3f} | {best:.3f} | {delta:+.3f} |")
        lines.append("")

    # Spread buckets — large per-model spread + wrong ensemble = stacking case.
    # Only meaningful for binary records: "spread" here is max - min over the
    # per-model probabilities (a number in [0, 1]). Numeric/discrete entries
    # produced by ``_rank_numeric`` carry ``percentiles`` rather than ``prob``,
    # and a percentile-based spread isn't directly comparable, so we skip them
    # here rather than synthesize a number that the reader would misread.
    high_spread_miss = []
    for entry in records_with_rankings:
        ranked = entry["ranked"]
        rec = entry["record"]
        if rec.get("type") != "binary":
            continue
        if len(ranked) < 2:
            continue
        probs = [r["prob"] for r in ranked]
        spread = max(probs) - min(probs)
        high_spread_miss.append(
            (
                rec["post_id"],
                rec.get("title", "")[:60],
                spread,
                rec.get("brier_score"),
                rec.get("was_stacked"),
            )
        )
    if high_spread_miss:
        lines.append(f"## {f['spread_section_label']}\n")
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
