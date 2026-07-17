"""Analysis 1 — symmetric anchor-vs-final scoring.

For each mapped (question, model) anchor pair, score the model's OUTSIDE-VIEW ANCHOR
and its FINAL forecast against the resolution, and measure whether deviating from the
anchor helped or hurt. No thumb on the scale: log score AND Brier, deviation-magnitude
splits, verified-accuracy splits, tercile splits, and a bootstrap CI on the mean delta.

Score convention (higher = better): log score = log(p_correct); Brier = -(p_correct-1)^2
so that larger (less negative) is better for BOTH metrics. delta = final_score - anchor_score:
delta > 0 means the deviation HELPED (final beat the anchor); delta < 0 means it HURT.

Free/local only.
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from anchor_map import ROWS

EPS = 1e-6  # clamp to keep log finite; matches nothing in prod, just for scoring math


def clamp(p: float) -> float:
    return min(1 - EPS, max(EPS, p))


def p_correct(p_yes: float, resolved_yes: bool) -> float:
    """Prob the model put on the realized outcome."""
    return clamp(p_yes) if resolved_yes else clamp(1 - p_yes)


def log_score(p_yes: float, resolved_yes: bool) -> float:
    return math.log(p_correct(p_yes, resolved_yes))


def brier_score(p_yes: float, resolved_yes: bool) -> float:
    # negated Brier so higher = better (aligns sign with log score)
    pc = p_correct(p_yes, resolved_yes)
    return -((pc - 1.0) ** 2)


def odds(p: float) -> float:
    p = clamp(p)
    return p / (1 - p)


def odds_ratio(a: float, b: float) -> float:
    """Symmetric odds ratio between anchor and final (>=1)."""
    oa, ob = odds(a), odds(b)
    r = oa / ob
    return max(r, 1 / r)


def magnitude_bucket(anchor: float, final: float) -> str:
    r = odds_ratio(anchor, final)
    if r < 1.5:
        return "small (<1.5x odds)"
    if r <= 3.0:
        return "medium (1.5-3x)"
    return "large (>3x odds)"


def bootstrap_ci(vals: list[float], n: int = 20000, seed: int = 20260716) -> tuple[float, float]:
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    k = len(vals)
    for _ in range(n):
        s = sum(vals[rng.randrange(k)] for _ in range(k)) / k
        means.append(s)
    means.sort()
    return (means[int(0.025 * n)], means[int(0.975 * n)])


# Resolution -> tercile lookup from manifest
def load_tercile() -> dict[str, tuple[str, str]]:
    man = json.loads(
        (Path(__file__).resolve().parents[1] / "base_rate_audit_2026-07-16" / "sample_manifest.json").read_text()
    )
    return {str(m["post_id"]): (m["tercile"], m["config_era"]) for m in man}


def build_records() -> list[dict]:
    terc = load_tercile()
    recs = []
    for pid, model, qtype, ap, fp, res_yes, mapping, cluster, verif, note in ROWS:
        a_log, f_log = log_score(ap, res_yes), log_score(fp, res_yes)
        a_bri, f_bri = brier_score(ap, res_yes), brier_score(fp, res_yes)
        tname, era = terc.get(pid, ("?", "?"))
        recs.append(
            {
                "pid": pid,
                "model": model,
                "qtype": qtype,
                "anchor_p": ap,
                "final_p": fp,
                "res_yes": res_yes,
                "mapping": mapping,
                "cluster": cluster,
                "verif": verif,
                "tercile": tname,
                "era": era,
                "note": note,
                "anchor_log": a_log,
                "final_log": f_log,
                "anchor_bri": a_bri,
                "final_bri": f_bri,
                "dlog": f_log - a_log,
                "dbri": f_bri - a_bri,
                "or": odds_ratio(ap, fp),
                "mag": magnitude_bucket(ap, fp),
                "moved": abs(ap - fp) > 1e-9,
            }
        )
    return recs


def summarize(recs: list[dict], label: str) -> dict:
    moved = [r for r in recs if r["moved"]]
    dlog = [r["dlog"] for r in moved]
    dbri = [r["dbri"] for r in moved]
    helped = sum(1 for d in dlog if d > 1e-9)
    hurt = sum(1 for d in dlog if d < -1e-9)
    tie = len(dlog) - helped - hurt
    mean_dlog = sum(dlog) / len(dlog) if dlog else float("nan")
    med_dlog = sorted(dlog)[len(dlog) // 2] if dlog else float("nan")
    ci = bootstrap_ci(dlog)
    return {
        "label": label,
        "n_all": len(recs),
        "n_moved": len(moved),
        "helped": helped,
        "hurt": hurt,
        "tie": tie,
        "mean_dlog": mean_dlog,
        "median_dlog": med_dlog,
        "ci_dlog": ci,
        "mean_dbri": sum(dbri) / len(dbri) if dbri else float("nan"),
        "sum_dlog": sum(dlog),
        "mean_anchor_log": sum(r["anchor_log"] for r in moved) / len(moved) if moved else float("nan"),
        "mean_final_log": sum(r["final_log"] for r in moved) / len(moved) if moved else float("nan"),
    }


def pr(s: dict) -> None:
    print(
        f"  {s['label']:34s} n={s['n_moved']:2d}/{s['n_all']:2d}  "
        f"helped={s['helped']:2d} hurt={s['hurt']:2d} tie={s['tie']:2d}  "
        f"mean dlog={s['mean_dlog']:+.4f} [{s['ci_dlog'][0]:+.3f},{s['ci_dlog'][1]:+.3f}]  "
        f"med={s['median_dlog']:+.4f}  meanBrier d={s['mean_dbri']:+.4f}  "
        f"anchorLL={s['mean_anchor_log']:+.3f}->finalLL={s['mean_final_log']:+.3f}"
    )


def main() -> None:
    recs = build_records()
    print(
        f"Mapped anchor pairs: {len(recs)}  (binary={sum(1 for r in recs if r['qtype'] == 'binary')}, "
        f"MC={sum(1 for r in recs if r['qtype'] == 'multiple_choice')})"
    )
    print(
        f"mapping: direct={sum(1 for r in recs if r['mapping'] == 'direct')}, "
        f"borderline={sum(1 for r in recs if r['mapping'] == 'borderline')}"
    )
    print()

    print("=== OVERALL (higher dlog = deviation from anchor HELPED) ===")
    pr(summarize(recs, "all mapped"))
    pr(summarize([r for r in recs if r["mapping"] == "direct"], "direct-only (drop borderline)"))
    pr(summarize([r for r in recs if r["qtype"] == "binary"], "binary only"))
    pr(summarize([r for r in recs if r["qtype"] == "multiple_choice"], "MC only"))
    print()

    print("=== BY DEVIATION MAGNITUDE (odds ratio anchor vs final) ===")
    for mag in ("small (<1.5x odds)", "medium (1.5-3x)", "large (>3x odds)"):
        pr(summarize([r for r in recs if r["mag"] == mag], mag))
    print()

    print("=== BY VERIFIED ACCURACY OF THE ANCHOR (audit's 22 clusters) ===")
    for v in ("accurate", "wrong", "uncertain"):
        pr(summarize([r for r in recs if r["verif"] == v], f"anchor verified={v}"))
    pr(summarize([r for r in recs if r["verif"] is None], "unverified (no cluster)"))
    print()

    print("=== BY TERCILE (sample oversamples 'worst'; report strata separately) ===")
    for t in ("worst", "middling", "good"):
        pr(summarize([r for r in recs if r["tercile"] == t], f"tercile={t}"))
    print()

    print("=== BY ERA ===")
    for e in ("pre_flip", "post_flip"):
        pr(summarize([r for r in recs if r["era"] == e], f"era={e}"))
    print()

    # Reweight: equal weight per tercile stratum (undo oversampling of worst)
    print("=== TERCILE-REWEIGHTED overall mean dlog (equal weight per stratum) ===")
    strata = defaultdict(list)
    for r in recs:
        if r["moved"]:
            strata[r["tercile"]].append(r["dlog"])
    stratum_means = {t: sum(v) / len(v) for t, v in strata.items() if v}
    if stratum_means:
        reweighted = sum(stratum_means.values()) / len(stratum_means)
        print("  per-tercile means: " + ", ".join(f"{t}={m:+.4f}" for t, m in stratum_means.items()))
        print(f"  equal-weight-across-terciles mean dlog = {reweighted:+.4f}")
    print()

    print("=== PER-PAIR DETAIL (sorted by dlog; most-hurt first) ===")
    for r in sorted([r for r in recs if r["moved"]], key=lambda r: r["dlog"]):
        arrow = "HELPED" if r["dlog"] > 1e-9 else ("HURT" if r["dlog"] < -1e-9 else "tie")
        print(
            f"  {r['pid']} {r['model'][:20]:20s} {r['qtype'][:3]} "
            f"a={r['anchor_p']:.3f} f={r['final_p']:.3f} res={'Y' if r['res_yes'] else 'N'} "
            f"OR={r['or']:.2f} {r['mag'][:6]:6s} v={str(r['verif'])[:5]:5s} "
            f"dlog={r['dlog']:+.4f} {arrow:6s} | {r['note'][:52]}"
        )


if __name__ == "__main__":
    main()
