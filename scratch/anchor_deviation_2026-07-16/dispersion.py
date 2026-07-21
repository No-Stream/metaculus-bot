"""Analysis 2 — cross-model reference-class dispersion: bug or feature?

Two grains:
  2a. QUESTION-LEVEL ANCHOR dispersion. For each question with >=3 mapped anchors
      (from anchor_map.ROWS), measure how much models disagree on the outside-view
      base rate for the SAME event, then relate that to:
        - question peer score (guessing-signature hypothesis: high dispersion -> worse)
        - ensemble forecast spread of the FINAL per-model forecasts
        - the MEDIAN's log-score advantage over the mean individual model
          (diversity-as-feature hypothesis: does the median absorb diverse anchors?)
  2b. NAMED-CLASS dispersion from the audit's verified clusters (same named class,
      >=3 models) + a descriptive tally of whether the class is a citable dataset.

Dispersion metric: odds-spread = max_i odds(p_i) / min_i odds(p_i) across models
(symmetric, handles probs near 0/1). Also report the raw prob range.

Free/local only.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

from anchor_map import ROWS

AUDIT = Path(__file__).resolve().parents[1] / "base_rate_audit_2026-07-16"
PERF = Path(__file__).resolve().parents[1] / "coherence_2026-07-15" / "perf_all_tagged.json"
EPS = 1e-6


def clamp(p: float) -> float:
    return min(1 - EPS, max(EPS, p))


def odds(p: float) -> float:
    p = clamp(p)
    return p / (1 - p)


def log_score(p_correct: float) -> float:
    return math.log(clamp(p_correct))


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation, no scipy."""
    n = len(xs)
    if n < 3:
        return float("nan")

    def rank(a: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: a[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    return num / (dx * dy) if dx and dy else float("nan")


def main() -> None:
    man = {str(m["post_id"]): m for m in json.loads((AUDIT / "sample_manifest.json").read_text())}
    perf = {str(r["post_id"]): r for r in json.loads(PERF.read_text())}

    # ---- 2a. question-level anchor dispersion ----
    anchors_by_q: dict[str, list[float]] = defaultdict(list)
    qtype: dict[str, str] = {}
    for pid, model, qt, ap, fp, res_yes, mapping, cluster, verif, note in ROWS:
        anchors_by_q[pid].append(ap)
        qtype[pid] = qt

    rows = []
    for pid, aps in anchors_by_q.items():
        if len(aps) < 3:
            continue
        m = man[pid]
        rec = perf[pid]
        anchor_odds_spread = max(odds(p) for p in aps) / min(odds(p) for p in aps)
        anchor_prob_range = max(aps) - min(aps)

        # FINAL per-model forecasts, scored against resolution
        res = m["resolution"]
        if qtype[pid] == "binary":
            pmf = rec.get("per_model_forecasts") or {}
            finals = [float(v.rstrip("%")) / 100.0 for v in pmf.values() if isinstance(v, str) and v.endswith("%")]
            res_yes = bool(res)
            pcorr = [(p if res_yes else 1 - p) for p in finals]
            final_spread = (max(finals) - min(finals)) if finals else float("nan")
            median_p = sorted(finals)[len(finals) // 2] if finals else float("nan")
            median_pcorr = median_p if res_yes else 1 - median_p
        else:
            pbmf = rec.get("per_base_model_forecasts") or {}
            pmf0 = rec.get("per_model_forecasts") or {}
            src = pbmf if (any(k.startswith("Forecaster") for k in pmf0) and pbmf) else pmf0
            dists = [v for k, v in src.items() if isinstance(v, dict)]
            finals = [d.get(res, 0.0) for d in dists]  # P(resolved option) per model
            pcorr = finals
            final_spread = (max(finals) - min(finals)) if finals else float("nan")
            # ensemble median distribution's P(resolved option)
            if dists:
                opts = m["options"]
                med_dist = {o: sorted(d.get(o, 0.0) for d in dists)[len(dists) // 2] for o in opts}
                s = sum(med_dist.values()) or 1.0
                median_pcorr = med_dist.get(res, 0.0) / s
            else:
                median_pcorr = float("nan")

        indiv_ll = [log_score(pc) for pc in pcorr]
        mean_indiv_ll = sum(indiv_ll) / len(indiv_ll) if indiv_ll else float("nan")
        median_ll = log_score(median_pcorr)
        median_adv = median_ll - mean_indiv_ll

        rows.append(
            {
                "pid": pid,
                "qtype": qtype[pid],
                "tercile": m["tercile"],
                "peer": float(m["peer_score"]),
                "n_anchors": len(aps),
                "anchor_odds_spread": anchor_odds_spread,
                "anchor_prob_range": anchor_prob_range,
                "final_spread": final_spread,
                "median_adv": median_adv,
                "mean_indiv_ll": mean_indiv_ll,
                "median_ll": median_ll,
            }
        )

    rows.sort(key=lambda r: -r["anchor_odds_spread"])
    print(f"=== 2a. QUESTION-LEVEL ANCHOR DISPERSION (n={len(rows)} questions with >=3 mapped anchors) ===\n")
    print(
        f"{'pid':6s} {'type':4s} {'terc':5s} {'peer':>7s} {'nA':>3s} "
        f"{'anchOddsSpr':>11s} {'anchPrRange':>11s} {'finalSpread':>11s} {'medianAdv':>10s}"
    )
    for r in rows:
        print(
            f"{r['pid']:6s} {r['qtype'][:4]:4s} {r['tercile'][:4]:5s} {r['peer']:7.1f} {r['n_anchors']:3d} "
            f"{r['anchor_odds_spread']:11.2f} {r['anchor_prob_range']:11.3f} {r['final_spread']:11.3f} "
            f"{r['median_adv']:+10.4f}"
        )

    print("\n--- correlations (Spearman; n small, descriptive only) ---")
    disp = [math.log(r["anchor_odds_spread"]) for r in rows]  # log odds-spread as the dispersion scale
    print(f"  n = {len(rows)}")
    print(
        f"  Spearman(anchor dispersion, peer score)      = {spearman(disp, [r['peer'] for r in rows]):+.3f}   "
        f"(guessing-signature: expect NEGATIVE)"
    )
    print(
        f"  Spearman(anchor dispersion, final spread)    = {spearman(disp, [r['final_spread'] for r in rows]):+.3f}   "
        f"(does anchor disagreement propagate to final spread?)"
    )
    print(
        f"  Spearman(anchor dispersion, median advantage)= {spearman(disp, [r['median_adv'] for r in rows]):+.3f}   "
        f"(diversity-as-feature: POSITIVE = median absorbs diverse anchors better as dispersion grows)"
    )

    # median advantage overall and split by dispersion tertile
    print("\n--- MEDIAN advantage over mean-individual (log score), by anchor-dispersion tercile ---")
    srt = sorted(rows, key=lambda r: r["anchor_odds_spread"])
    n = len(srt)
    thirds = {"low disp": srt[: n // 3], "mid disp": srt[n // 3 : 2 * n // 3], "high disp": srt[2 * n // 3 :]}
    for lbl, grp in thirds.items():
        if not grp:
            continue
        madv = sum(r["median_adv"] for r in grp) / len(grp)
        osp = sum(r["anchor_odds_spread"] for r in grp) / len(grp)
        print(f"  {lbl:9s} n={len(grp):2d}  mean odds-spread={osp:6.2f}  mean median-advantage={madv:+.4f}")
    all_madv = sum(r["median_adv"] for r in rows) / len(rows)
    print(f"  ALL       n={len(rows):2d}                        mean median-advantage={all_madv:+.4f}")


if __name__ == "__main__":
    main()
