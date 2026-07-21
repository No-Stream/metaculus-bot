"""Inter-rater check: my curated anchor_p vs the blind adjudicator(s)' anchor_p.

The confirmation-bias worry (operator-flagged): did I pick anchor values that make
deviation look bad? An independent Opus adjudicator mapped the SAME rationales
resolution-blind. For every (pid, model) pair I scored, compare my anchor_p to the
adjudicator's, and re-run the headline helped/hurt + mean-Δlog on the ADJUDICATOR's
anchors to see if the conclusion is stable to who did the extraction.

Free/local only.
"""

import json
from pathlib import Path

from anchor_map import ROWS
from score_anchor import log_score

HERE = Path(__file__).resolve().parent


def norm_model(m: str) -> str:
    m = m.split("/")[-1].strip().lower()
    # unify a couple of naming variants seen across files
    return m


def load_adj(name: str) -> dict[tuple[str, str], dict]:
    p = HERE / name
    if not p.exists():
        return {}
    out = {}
    for r in json.loads(p.read_text()):
        key = (str(r["post_id"]), norm_model(r["model"]))
        out[key] = r
    return out


def adj_anchor_scalar(r: dict, qtype: str, maps_to: str | None) -> float | None:
    """Reduce an adjudicator anchor_p to a scalar comparable to my mapped anchor.

    For binary: it's already P(YES). For MC: my anchor is P(resolved-relevant option);
    the adjudicator stored either that option's prob or a dict — but blind, it can't
    know which option resolved. So MC scalar comparison is only meaningful where the
    adjudicator recorded a single mapped option AND it matches my maps_to. We therefore
    restrict the quantitative re-score to BINARY, and report MC agreement qualitatively.
    """
    ap = r.get("anchor_p")
    if ap is None:
        return None
    if isinstance(ap, (int, float)):
        return float(ap)
    return None  # dict-valued MC anchor: skip scalar compare


def main() -> None:
    B = load_adj("_adj_B.json")
    A = load_adj("_adj_A.json")
    adjs = {"B": B, "A": A}

    # My scored rows keyed the same way
    mine = {}
    for pid, model, qtype, ap, fp, res_yes, mapping, cluster, verif, note in ROWS:
        mine[(pid, norm_model(model))] = {
            "qtype": qtype,
            "anchor_p": ap,
            "final_p": fp,
            "res_yes": res_yes,
            "mapping": mapping,
        }

    for tag, ADJ in adjs.items():
        if not ADJ:
            print(f"=== adjudicator {tag}: (not available) ===\n")
            continue
        print(f"=== INTER-RATER vs adjudicator {tag} (binary pairs; blind anchor extraction) ===")
        rows = []
        for key, mrow in mine.items():
            if mrow["qtype"] != "binary":
                continue
            ar = ADJ.get(key)
            if not ar:
                continue
            a_scalar = adj_anchor_scalar(ar, "binary", None)
            if a_scalar is None:
                continue
            rows.append((key, mrow, a_scalar, ar.get("mapping_verdict")))

        if not rows:
            print("  no overlapping binary scalar anchors\n")
            continue

        diffs = [abs(m["anchor_p"] - a) for _, m, a, _ in rows]
        print(f"  overlapping binary pairs: {len(rows)}")
        print(f"  mean |my_anchor - {tag}_anchor| = {sum(diffs) / len(diffs):.3f}")
        print(
            f"  pairs within 0.05: {sum(1 for d in diffs if d <= 0.05)}/{len(rows)}; "
            f"within 0.10: {sum(1 for d in diffs if d <= 0.10)}/{len(rows)}"
        )

        # Re-run headline on adjudicator anchors (final + resolution unchanged — those aren't judgment)
        helped = hurt = 0
        dsum = 0.0
        for _, m, a, _ in rows:
            dlog = log_score_final_minus_anchor(m["final_p"], a, m["res_yes"])
            dsum += dlog
            if dlog > 1e-9:
                helped += 1
            elif dlog < -1e-9:
                hurt += 1
        print(
            f"  headline on {tag}'s anchors (binary only): helped={helped} hurt={hurt} "
            f"mean Δlog={dsum / len(rows):+.4f}"
        )
        # compare to mine on the SAME binary subset
        mh = mhu = 0
        md = 0.0
        for _, m, _, _ in rows:
            dlog = log_score_final_minus_anchor(m["final_p"], m["anchor_p"], m["res_yes"])
            md += dlog
            if dlog > 1e-9:
                mh += 1
            elif dlog < -1e-9:
                mhu += 1
        print(f"  headline on MY anchors (same subset):      helped={mh} hurt={mhu} mean Δlog={md / len(rows):+.4f}")
        print("  biggest anchor disagreements:")
        for key, m, a, verdict in sorted(rows, key=lambda t: -abs(t[1]["anchor_p"] - t[2]))[:6]:
            print(f"    {key[0]} {key[1][:18]:18s} mine={m['anchor_p']:.3f} {tag}={a:.3f} ({tag}-verdict={verdict})")
        print()


def log_score_final_minus_anchor(final_p: float, anchor_p: float, res_yes: bool) -> float:
    return log_score(final_p, res_yes) - log_score(anchor_p, res_yes)


if __name__ == "__main__":
    main()
