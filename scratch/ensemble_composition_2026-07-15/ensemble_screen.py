"""Retrospective ensemble-composition screening (2026-07-15).

Recovers per-forecaster predictions from the bot's own published Metaculus comments
(via the collector dataset cached at scratch/coherence_2026-07-15/perf_all_tagged.json,
summer pulled fresh 2026-07-15; spring/fall reused from the 07-08 pull of those closed
tournaments) and asks: how would the published ensemble aggregate have scored under
alternative compositions?

Method:
  - Baseline is the REPLICA aggregate (recomputed from recovered per-model values with
    the era-appropriate combine: mean for the earliest 3-model era, median after), not
    the published number. Subset-vs-replica paired deltas therefore control for
    recovery error; published-vs-replica agreement is reported separately as validation.
  - Era bucketing on forecaster-roster changes (submit timestamp = bot comment
    created_at). Model families: lineages pooled (grok-4-fast/4.1/4.3 -> grok, etc.).
  - Comparisons per era: leave-one-family-out for every family with enough coverage,
    hypothesis subsets (drop_grok, drop_gemini, drop_both), and a leave-one-question-out
    top-3-family selection. Paired per-question deltas on the Metaculus log-score family
    (binary_log_score / mc_log_score / numeric_log_score, all higher-better), bootstrap
    CIs over questions (4000 reps).
  - Numeric counterfactuals: rebuild each member's 201-point PCHIP CDF from recovered
    percentiles, aggregate pointwise in CDF space (median/mean), pin/monotone/min-step
    like the prod ensemble postprocess, score with numeric_log_score. Discrete questions
    are scored on the 201-grid for BOTH arms (prod resamples to the native bin count;
    consistent-grid deltas remain meaningful, levels differ from Metaculus's).

100% offline: no API calls, no LLM calls. Run from repo root:
  uv run python scratch/ensemble_composition_2026-07-15/ensemble_screen.py
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from metaculus_bot.numeric.pchip_cdf import generate_pchip_cdf
from metaculus_bot.performance_analysis.collector import resolve_numeric_record_to_score_inputs
from metaculus_bot.performance_analysis.parsing import _parse_probability
from metaculus_bot.scoring_common import binary_log_score, brier_score, mc_log_score, numeric_log_score

logging.basicConfig(level=logging.WARNING)
logging.getLogger("metaculus_bot.numeric.pchip_cdf").setLevel(logging.ERROR)
logger = logging.getLogger("ensemble_screen")

HERE = Path(__file__).parent
DATA = Path("scratch/coherence_2026-07-15/perf_all_tagged.json")
OUT_JSON = HERE / "results.json"
OUT_TABLES = HERE / "tables.md"

BOOT_REPS = 4000
SEED = 20260715
NUMERIC_TYPES = ("numeric", "discrete")
ANON_RE = re.compile(r"^Forecaster \d+( base)?$")

# ---------------------------------------------------------------------------
# Model canonicalization + family mapping
# ---------------------------------------------------------------------------

MODEL_ALIASES = {"claude-4.6-opus": "claude-opus-4.6"}


def canonicalize(raw: str) -> str:
    name = raw.strip()
    if ANON_RE.match(name):
        return name
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    if name.endswith(":free"):
        name = name[: -len(":free")]
    return MODEL_ALIASES.get(name, name)


FAMILY_RULES: list[tuple[str, str]] = [
    # (prefix, family) — first match wins. Judgment call, stated in RESULTS.md:
    # opus 4.5-4.8 + sonnet 4/4.5 pooled as one "anthropic" family (same vendor slot
    # lineage; some eras run two anthropic slots — LOO then drops both, n_members noted).
    ("gpt-5", "openai-gpt5"),
    ("o3", "openai-o3"),
    ("claude", "anthropic"),
    ("gemini", "gemini"),
    ("grok", "grok"),
    ("kimi", "kimi"),
    ("qwen", "qwen"),
]


def family_of(model: str) -> str | None:
    if ANON_RE.match(model):
        return None
    if model.startswith("unattributed:"):
        return model.split(":", 1)[1]
    for prefix, fam in FAMILY_RULES:
        if model.startswith(prefix):
            return fam
    return f"other:{model}"


# ---------------------------------------------------------------------------
# Era definitions (roster changes = era boundaries; submit-ts driven)
# ---------------------------------------------------------------------------

# (era, start_date, end_date_inclusive, combine_method)
ERAS: list[tuple[str, str, str, str]] = [
    ("fall_mean3", "2025-08-01", "2025-09-20", "mean"),  # 3-model MEAN era (roster-gated below)
    ("fall_5m", "2025-09-12", "2025-09-20", "median"),  # +kimi/qwen, median switch
    ("fall_6m", "2025-09-21", "2025-12-31", "median"),  # +grok lineage; in-family sub-swaps
    ("spring_5m_a", "2026-01-01", "2026-02-07", "median"),  # opus4.5 + 2x gemini + gpt-5/5.2
    ("spring_trans", "2026-02-08", "2026-02-20", "median"),  # transitional roster churn
    ("spring_5m_b", "2026-02-21", "2026-04-03", "median"),  # 2x claude, 2x gpt, gemini-3.1-pro
    ("spring_6m", "2026-04-04", "2026-05-18", "median"),  # +grok-4.1-fast/gpt-5.4; stacking era
    ("summer_6m", "2026-05-19", "2026-07-14", "median"),  # current 6-slot roster
]

MEAN3_ROSTER = frozenset({"gpt-5", "o3", "claude-sonnet-4"})


def era_of(record: dict, member_models: set[str]) -> str | None:
    ts = (record.get("bot_comment_created_at") or "")[:10]
    if not ts:
        return None
    # Boundary disambiguation: 3-model roster records through 09-20 are the MEAN era
    # (mean verified exact on all of them); 5-model records from 09-12 are median.
    if ts <= "2025-09-20":
        return (
            "fall_mean3"
            if member_models and member_models <= MEAN3_ROSTER
            else ("fall_5m" if ts >= "2025-09-12" else None)
        )
    for era, start, end, _ in ERAS:
        if era in ("fall_mean3", "fall_5m"):
            continue
        if start <= ts <= end:
            return era
    return None


ERA_METHOD = {era: method for era, _, _, method in ERAS}
ERA_ORDER = [era for era, *_ in ERAS]

# ---------------------------------------------------------------------------
# Record -> usable members
# ---------------------------------------------------------------------------

DROPS: Counter = Counter()


def extract_members(record: dict) -> tuple[dict[str, object], bool] | None:
    """Return ({canonical_model: value}, published_was_stacked) or None if unusable.

    value: float prob (binary) | dict option->prob (MC) | list[(pct, val)] (numeric).
    For stacked-published questions the per-model summary collapses to the stacker's
    aggregate, so base values come from per_base_model_forecasts (binary/MC) — numeric
    stacked questions are unrecoverable (percentile keys collapse to 'Forecaster 1').
    """
    q_type = record["type"]
    stacked = record.get("stacker_outcome") in ("primary", "fallback_llm")

    if q_type in NUMERIC_TYPES:
        raw = record.get("per_model_numeric_percentiles") or {}
        if stacked:
            named = {k: v for k, v in raw.items() if not ANON_RE.match(k)}
            if not named:
                DROPS["stacked_numeric_unrecoverable"] += 1
                return None
            raw = named
        members: dict[str, object] = {}
        for m, pairs in raw.items():
            cm = canonicalize(m)
            dedup = {}
            for p, v in pairs:
                dedup[float(p)] = float(v)
            if len(dedup) < 5:
                DROPS["numeric_member_too_few_percentiles"] += 1
                continue
            members[cm] = sorted(dedup.items())
        return (members, stacked) if members else None

    src = record.get("per_base_model_forecasts") if stacked else record.get("per_model_forecasts")
    src = src or {}
    if stacked and not src:
        DROPS[f"stacked_{q_type}_unrecoverable"] += 1
        return None

    members = {}
    if q_type == "binary":
        for m, v in src.items():
            if not isinstance(v, str):
                DROPS["binary_member_nonstring"] += 1
                continue
            p = _parse_probability(v)
            if p is None:
                DROPS["binary_member_unparseable"] += 1
                continue
            members[canonicalize(m)] = p
    elif q_type == "multiple_choice":
        opts = set(record.get("options") or [])
        for m, d in src.items():
            if not isinstance(d, dict) or set(d) != opts:
                DROPS["mc_member_option_mismatch"] += 1
                continue
            members[canonicalize(m)] = {str(k): float(v) for k, v in d.items()}
    return (members, stacked) if members else None


def eliminate_anon(era_records: list[dict]) -> int:
    """Attribute a single anonymous member by elimination against the era's modal roster.

    Fires when the record has exactly one anon member and exactly one modal-roster model
    absent from its named members. Version guard: when the era contains multiple same-
    family versions (e.g. gpt-5 -> gpt-5.1 mid-era) and the record's timestamp falls
    outside the candidate model's observed date window, the value is provably the right
    FAMILY but the wrong version label — attribute as ``unattributed:{family}`` instead.
    Returns count of successful attributions.
    """
    model_freq: Counter = Counter()
    windows: dict[str, tuple[str, str]] = {}
    for r in era_records:
        for m in r["members"]:
            if ANON_RE.match(m):
                continue
            model_freq[m] += 1
            lo, hi = windows.get(m, (r["ts"], r["ts"]))
            windows[m] = (min(lo, r["ts"]), max(hi, r["ts"]))
    n = len(era_records)
    modal = {m for m, c in model_freq.items() if c >= 0.3 * n}
    fixed = 0
    for r in era_records:
        anons = [m for m in r["members"] if ANON_RE.match(m)]
        if len(anons) != 1:
            continue
        named = {m for m in r["members"] if not ANON_RE.match(m)}
        missing = modal - named
        if len(missing) != 1:
            continue
        target = missing.pop()
        fam = family_of(target)
        has_sibling_versions = any(m != target and family_of(m) == fam for m in model_freq)
        lo, hi = windows[target]
        if has_sibling_versions and not (lo <= r["ts"] <= hi):
            target = f"unattributed:{fam}"
        r["members"][target] = r["members"].pop(anons[0])
        fixed += 1
    return fixed


# ---------------------------------------------------------------------------
# Aggregation + scoring (mirrors prod combine rules)
# ---------------------------------------------------------------------------

_CDF_CACHE: dict[tuple[int, str], np.ndarray | None] = {}


def member_cdf(record: dict, model: str, pairs: list[tuple[float, float]]) -> np.ndarray | None:
    key = (record["question_id"], model)
    if key in _CDF_CACHE:
        return _CDF_CACHE[key]
    si = resolve_numeric_record_to_score_inputs(record)
    if si is None:
        _CDF_CACHE[key] = None
        return None
    _, lower, upper, zero_point = si
    try:
        cdf, _ = generate_pchip_cdf(
            dict(pairs),
            open_upper_bound=bool(record.get("open_upper_bound")),
            open_lower_bound=bool(record.get("open_lower_bound")),
            upper_bound=upper,
            lower_bound=lower,
            zero_point=zero_point,
        )
        arr = np.asarray(cdf, dtype=float)
    except (ValueError, RuntimeError) as exc:
        logger.debug(f"member CDF failed q={record['question_id']} {model}: {exc}")
        DROPS["numeric_member_pchip_failed"] += 1
        arr = None
    _CDF_CACHE[key] = arr
    return arr


def postprocess_cdf(agg: np.ndarray, record: dict) -> np.ndarray:
    """Prod-style ensemble CDF postprocess: clip, monotone, pin endpoints, ramp min-step."""
    p = np.maximum.accumulate(np.clip(agg, 0.0, 1.0))
    open_lower = bool(record.get("open_lower_bound"))
    open_upper = bool(record.get("open_upper_bound"))

    def pin(v: np.ndarray) -> None:
        v[0] = max(v[0], 0.001) if open_lower else 0.0
        v[-1] = min(v[-1], 0.999) if open_upper else 1.0

    pin(p)
    min_step = 5e-5
    if len(p) > 1 and float(np.min(np.diff(p))) < min_step:
        p = np.maximum.accumulate(p + np.linspace(0.0, min_step * 3.0, len(p)))
        pin(p)
    return p


def score_subset(record: dict, models: list[str], method: str) -> float | None:
    """Metaculus-family log score of the subset aggregate on this record. None = not scorable."""
    q_type = record["type"]
    members = record["members"]
    vals = [members[m] for m in models if m in members]
    if len(vals) < 2:
        return None

    if q_type == "binary":
        res = record["resolution_parsed"]
        agg = sum(vals) / len(vals) if method == "mean" else statistics.median(vals)
        return binary_log_score(round(agg, 3), res)

    if q_type == "multiple_choice":
        options = record["options"]
        agg_fn = (lambda xs: sum(xs) / len(xs)) if method == "mean" else statistics.median
        agg = [agg_fn([v[o] for v in vals]) for o in options]
        total = sum(agg)
        if total <= 0:
            return None
        agg = [a / total for a in agg]
        return mc_log_score(agg, options.index(record["resolution_parsed"]))

    # numeric / discrete
    cdfs = [c for m in models if m in members for c in [member_cdf(record, m, members[m])] if c is not None]
    if len(cdfs) < 2:
        return None
    stack = np.vstack(cdfs)
    agg = np.mean(stack, axis=0) if method == "mean" else np.median(stack, axis=0)
    p = postprocess_cdf(agg, record)
    si = resolve_numeric_record_to_score_inputs(record)
    if si is None:
        return None
    res_float, lower, upper, zero_point = si
    try:
        return numeric_log_score(
            list(p),
            res_float,
            lower,
            upper,
            bool(record.get("open_lower_bound")),
            bool(record.get("open_upper_bound")),
            zero_point,
        )
    except (ValueError, ZeroDivisionError):
        return None


def member_score(record: dict, model: str) -> float | None:
    """Individual member's own log score (for descriptive ranking + top-3 selection)."""
    q_type = record["type"]
    v = record["members"].get(model)
    if v is None:
        return None
    if q_type == "binary":
        return binary_log_score(v, record["resolution_parsed"])
    if q_type == "multiple_choice":
        options = record["options"]
        total = sum(v.values())
        probs = [v[o] / total for o in options]
        return mc_log_score(probs, options.index(record["resolution_parsed"]))
    cdf = member_cdf(record, model, v)
    if cdf is None:
        return None
    si = resolve_numeric_record_to_score_inputs(record)
    if si is None:
        return None
    res_float, lower, upper, zero_point = si
    try:
        return numeric_log_score(
            list(cdf),
            res_float,
            lower,
            upper,
            bool(record.get("open_lower_bound")),
            bool(record.get("open_upper_bound")),
            zero_point,
        )
    except (ValueError, ZeroDivisionError):
        return None


def binary_brier_subset(record: dict, models: list[str], method: str) -> float | None:
    if record["type"] != "binary":
        return None
    vals = [record["members"][m] for m in models if m in record["members"]]
    if len(vals) < 2:
        return None
    agg = sum(vals) / len(vals) if method == "mean" else statistics.median(vals)
    return brier_score(round(agg, 3), record["resolution_parsed"])


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_ci(deltas: list[float], rng: np.random.Generator) -> tuple[float, float, float]:
    arr = np.asarray(deltas, dtype=float)
    n = len(arr)
    idx = rng.integers(0, n, size=(BOOT_REPS, n))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def is_scorable(record: dict) -> bool:
    q_type = record["type"]
    if q_type == "binary":
        return isinstance(record.get("resolution_parsed"), bool)
    if q_type == "multiple_choice":
        opts = record.get("options") or []
        return isinstance(record.get("resolution_parsed"), str) and record["resolution_parsed"] in opts
    if q_type in NUMERIC_TYPES:
        return resolve_numeric_record_to_score_inputs(record) is not None
    return False


def main() -> None:
    with open(DATA) as f:
        raw_records = json.load(f)
    print(f"loaded {len(raw_records)} records from {DATA}")

    # --- build usable records ---
    by_era: dict[str, list[dict]] = defaultdict(list)
    validation = {"binary_exact": 0, "binary_close": 0, "binary_off": 0, "published_replica_absdiff": []}
    n_stacked_kept = 0
    for r in raw_records:
        if not is_scorable(r):
            DROPS[f"unscorable_{r['type']}"] += 1
            continue
        ext = extract_members(r)
        if ext is None:
            DROPS[f"no_members_{r['type']}"] += 1
            continue
        members, stacked = ext
        named = {m for m in members if not ANON_RE.match(m)}
        era = era_of(r, named)
        if era is None:
            DROPS["no_era_assignable"] += 1
            continue
        rec = {
            "question_id": r["question_id"],
            "post_id": r["post_id"],
            "type": r["type"],
            "resolution_parsed": r["resolution_parsed"],
            "options": r.get("options"),
            "open_lower_bound": r.get("open_lower_bound"),
            "open_upper_bound": r.get("open_upper_bound"),
            "scaling": r.get("scaling"),
            "members": dict(members),
            "stacked_published": stacked,
            "era": era,
            "ts": (r.get("bot_comment_created_at") or "")[:10],
            "published_prob": r.get("our_prob_yes"),
        }
        n_stacked_kept += int(stacked)
        by_era[era].append(rec)

    # --- anon elimination within each era ---
    anon_fixed = {era: eliminate_anon(recs) for era, recs in by_era.items()}
    # Drop remaining anon members (unattributable), then records with <2 members.
    for era, recs in by_era.items():
        kept = []
        for rec in recs:
            rec["members"] = {m: v for m, v in rec["members"].items() if not ANON_RE.match(m)}
            if len(rec["members"]) >= 2:
                kept.append(rec)
            else:
                DROPS["fewer_than_2_attributed_members"] += 1
        by_era[era] = kept

    # --- validation: replica vs published (binary, non-stacked) ---
    for era, recs in by_era.items():
        method = ERA_METHOD[era]
        for rec in recs:
            if rec["type"] != "binary" or rec["stacked_published"] or rec["published_prob"] is None:
                continue
            vals = list(rec["members"].values())
            agg = round(sum(vals) / len(vals) if method == "mean" else statistics.median(vals), 3)
            d = abs(agg - rec["published_prob"])
            validation["published_replica_absdiff"].append(d)
            if d < 1e-9:
                validation["binary_exact"] += 1
            elif d < 0.005:
                validation["binary_close"] += 1
            else:
                validation["binary_off"] += 1

    # --- per-era analysis ---
    rng = np.random.default_rng(SEED)
    results: dict[str, dict] = {}
    for era in ERA_ORDER:
        recs = by_era.get(era, [])
        if not recs:
            continue
        method = ERA_METHOD[era]
        fam_members: dict[str, set[str]] = defaultdict(set)
        for rec in recs:
            for m in rec["members"]:
                fam = family_of(m)
                if fam:
                    fam_members[fam].add(m)

        # replica baseline per record
        for rec in recs:
            rec["replica_score"] = score_subset(rec, list(rec["members"]), method)
            rec["replica_brier"] = binary_brier_subset(rec, list(rec["members"]), method)
        scored = [r for r in recs if r["replica_score"] is not None]

        # member-level scores (descriptive + top3 selection)
        mstats: dict[str, list[float]] = defaultdict(list)
        for rec in scored:
            for m in rec["members"]:
                s = member_score(rec, m)
                if s is not None:
                    mstats[m].append(s)

        comparisons: dict[str, dict] = {}

        def run_comparison(name: str, subset_fn, sample=None) -> None:
            deltas, briers, n_removed = [], [], []
            pool = sample if sample is not None else scored
            for rec in pool:
                keep = subset_fn(rec)
                if keep is None:
                    continue
                s = score_subset(rec, keep, method)
                if s is None or rec["replica_score"] is None:
                    continue
                deltas.append(s - rec["replica_score"])
                n_removed.append(len(rec["members"]) - len(keep))
                if rec["type"] == "binary":
                    b = binary_brier_subset(rec, keep, method)
                    if b is not None and rec["replica_brier"] is not None:
                        briers.append(b - rec["replica_brier"])
            if len(deltas) < 8:
                comparisons[name] = {"n": len(deltas), "skipped": "n<8"}
                return
            mean_d, lo, hi = bootstrap_ci(deltas, rng)
            entry = {
                "n": len(deltas),
                "mean_delta_log": mean_d,
                "ci_lo": lo,
                "ci_hi": hi,
                "mean_members_removed": float(np.mean(n_removed)),
            }
            if briers:
                mb, blo, bhi = bootstrap_ci(briers, rng)
                entry.update({"n_binary": len(briers), "mean_delta_brier": mb, "brier_ci_lo": blo, "brier_ci_hi": bhi})
            comparisons[name] = entry

        # (a) LOO per family
        for fam in sorted(fam_members):
            fam_set = fam_members[fam]

            def loo(rec, fam_set=fam_set):
                if not (set(rec["members"]) & fam_set):
                    return None  # family didn't forecast this question
                keep = [m for m in rec["members"] if m not in fam_set]
                return keep if len(keep) >= 2 else None

            run_comparison(f"drop_{fam}", loo)

        # (b) hypothesis subsets
        gg = fam_members.get("grok", set()) | fam_members.get("gemini", set())
        if fam_members.get("grok") and fam_members.get("gemini"):

            def drop_both(rec, gg=gg):
                if not (set(rec["members"]) & gg):
                    return None
                keep = [m for m in rec["members"] if m not in gg]
                return keep if len(keep) >= 2 else None

            run_comparison("drop_grok+gemini", drop_both)

        # (c) top-3 families, leave-one-question-out selection
        fam_scores_by_q: dict[int, dict[str, float]] = {}
        for rec in scored:
            per_fam: dict[str, list[float]] = defaultdict(list)
            for m in rec["members"]:
                s = member_score(rec, m)
                fam = family_of(m)
                if s is not None and fam:
                    per_fam[fam].append(s)
            fam_scores_by_q[rec["question_id"]] = {f: float(np.mean(v)) for f, v in per_fam.items()}

        fam_totals: dict[str, float] = defaultdict(float)
        fam_counts: dict[str, int] = defaultdict(int)
        for qs in fam_scores_by_q.values():
            for f, s in qs.items():
                fam_totals[f] += s
                fam_counts[f] += 1

        def top3(rec):
            qid = rec["question_id"]
            own = fam_scores_by_q.get(qid, {})
            if len(fam_counts) <= 3:
                return None  # top-3 of <=3 families == full ensemble; comparison is a no-op
            ranking = []
            for f in fam_counts:
                cnt = fam_counts[f] - (1 if f in own else 0)
                if cnt < 5:
                    continue
                tot = fam_totals[f] - own.get(f, 0.0)
                ranking.append((tot / cnt, f))
            if len(ranking) < 3:
                return None
            chosen = {f for _, f in sorted(ranking, reverse=True)[:3]}
            keep = [m for m in rec["members"] if family_of(m) in chosen]
            return keep if len(keep) >= 2 else None

        run_comparison("top3_families_LOQO", top3)

        # sensitivity: unstacked-only split for eras containing stacked-published records
        unstacked = [r for r in scored if not r["stacked_published"]]
        if len(unstacked) < len(scored):
            for fam in sorted(fam_members):
                fam_set = fam_members[fam]

                def loo_u(rec, fam_set=fam_set):
                    if not (set(rec["members"]) & fam_set):
                        return None
                    keep = [m for m in rec["members"] if m not in fam_set]
                    return keep if len(keep) >= 2 else None

                run_comparison(f"unstacked_only/drop_{fam}", loo_u, sample=unstacked)

        results[era] = {
            "n_records": len(recs),
            "n_scored": len(scored),
            "n_stacked_published": sum(r["stacked_published"] for r in recs),
            "by_type": dict(Counter(r["type"] for r in scored)),
            "method": method,
            "anon_attributed_by_elimination": anon_fixed.get(era, 0),
            "families": {f: sorted(ms) for f, ms in fam_members.items()},
            "member_mean_log": {
                m: {"n": len(v), "mean": float(np.mean(v))} for m, v in sorted(mstats.items()) if len(v) >= 5
            },
            "comparisons": comparisons,
        }

    # --- pooled per-family rows (descriptive; per-era is primary) ---
    pooled: dict[str, dict] = {}
    for fam in ("grok", "gemini", "anthropic", "openai-gpt5", "openai-o3", "kimi", "qwen"):
        deltas: list[float] = []
        eras_used = []
        for era in ERA_ORDER:
            recs = by_era.get(era, [])
            if not recs:
                continue
            method = ERA_METHOD[era]
            fam_set = {m for r in recs for m in r["members"] if family_of(m) == fam}
            if not fam_set:
                continue
            got = 0
            for rec in recs:
                if rec.get("replica_score") is None or not (set(rec["members"]) & fam_set):
                    continue
                keep = [m for m in rec["members"] if m not in fam_set]
                if len(keep) < 2:
                    continue
                s = score_subset(rec, keep, method)
                if s is None:
                    continue
                deltas.append(s - rec["replica_score"])
                got += 1
            if got:
                eras_used.append(era)
        if len(deltas) >= 8:
            mean_d, lo, hi = bootstrap_ci(deltas, rng)
            pooled[f"drop_{fam}"] = {
                "n": len(deltas),
                "eras": eras_used,
                "mean_delta_log": mean_d,
                "ci_lo": lo,
                "ci_hi": hi,
            }

    n_comparisons = sum(
        1 for era in results.values() for c in era["comparisons"].values() if "mean_delta_log" in c
    ) + len(pooled)

    out = {
        "generated": "2026-07-15",
        "data_source": str(DATA),
        "bootstrap_reps": BOOT_REPS,
        "seed": SEED,
        "n_comparisons_reported": n_comparisons,
        "validation_binary_replica_vs_published": {
            "exact": validation["binary_exact"],
            "close_lt_0.005": validation["binary_close"],
            "off_ge_0.005": validation["binary_off"],
            "mean_absdiff": float(np.mean(validation["published_replica_absdiff"]))
            if validation["published_replica_absdiff"]
            else None,
        },
        "drop_counts": dict(sorted(DROPS.items())),
        "eras": results,
        "pooled_descriptive": pooled,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT_JSON}")

    # --- markdown tables ---
    lines: list[str] = ["# Ensemble-composition screening — generated tables\n"]
    v = out["validation_binary_replica_vs_published"]
    lines.append(
        f"Replica-vs-published validation (binary, non-stacked): exact={v['exact']}, "
        f"close<0.005={v['close_lt_0.005']}, off≥0.005={v['off_ge_0.005']}, "
        f"mean |Δp|={v['mean_absdiff']:.5f}. Total CI-bearing comparisons reported: {n_comparisons}.\n"
    )
    for era in ERA_ORDER:
        if era not in results:
            continue
        e = results[era]
        lines.append(
            f"\n## {era}  (method={e['method']}, n_scored={e['n_scored']}, "
            f"types={e['by_type']}, stacked_published={e['n_stacked_published']}, "
            f"anon_recovered={e['anon_attributed_by_elimination']})\n"
        )
        lines.append("Families: " + "; ".join(f"{f}={ms}" for f, ms in sorted(e["families"].items())) + "\n")
        lines.append("| comparison | n_q | mean Δlog vs replica | 95% CI | ΔBrier (binary) | Brier CI |")
        lines.append("|---|---|---|---|---|---|")
        for name, c in e["comparisons"].items():
            if "mean_delta_log" not in c:
                lines.append(f"| {name} | {c['n']} | — skipped ({c.get('skipped')}) | | | |")
                continue
            brier = f"{c['mean_delta_brier']:+.4f} (n={c['n_binary']})" if "mean_delta_brier" in c else ""
            brier_ci = f"[{c['brier_ci_lo']:+.4f}, {c['brier_ci_hi']:+.4f}]" if "mean_delta_brier" in c else ""
            lines.append(
                f"| {name} | {c['n']} | {c['mean_delta_log']:+.2f} | "
                f"[{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}] | {brier} | {brier_ci} |"
            )
        lines.append("\nPer-member mean log score (n≥5):\n")
        lines.append("| model | n | mean log |")
        lines.append("|---|---|---|")
        for m, s in sorted(e["member_mean_log"].items(), key=lambda kv: -kv[1]["mean"]):
            lines.append(f"| {m} | {s['n']} | {s['mean']:+.2f} |")

    lines.append("\n## Pooled per-family LOO (descriptive — per-era tables are primary)\n")
    lines.append("| comparison | n_q | eras | mean Δlog | 95% CI |")
    lines.append("|---|---|---|---|---|")
    for name, c in pooled.items():
        lines.append(
            f"| {name} | {c['n']} | {','.join(c['eras'])} | {c['mean_delta_log']:+.2f} | "
            f"[{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}] |"
        )
    lines.append("\n## Drop counts\n")
    for k, cnt in sorted(DROPS.items()):
        lines.append(f"- {k}: {cnt}")
    OUT_TABLES.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_TABLES}")


if __name__ == "__main__":
    main()
