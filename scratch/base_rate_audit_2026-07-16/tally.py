"""Tally extracted base-rate claims and join with scores from the manifest.

Reads extracted/claims_batch*.json + sample_manifest.json, prints:
- lean rates (fraction of (question, model) rationales resting on >=1 base rate)
- rate_basis / load_bearing / verifiable breakdowns
- per-question claim counts joined with peer scores
- candidate list for the verification subsample (load-bearing, numeric, verifiable)

Free/local only.
"""

import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main() -> None:
    manifest = {str(m["post_id"]): m for m in json.loads((OUT / "sample_manifest.json").read_text())}

    claims: list[dict] = []
    summaries: list[dict] = []
    # Fallback batches are named claims_batchNb.json; prefer the original if it
    # exists (a stalled original may still land later), else take the fallback.
    files = {f.stem: f for f in (OUT / "extracted").glob("claims_batch*.json")}
    for stem in sorted(files):
        if stem.endswith("b") and stem[:-1] in files:
            continue
        blob = json.loads(files[stem].read_text())
        claims.extend(blob["claims"])
        summaries.extend(blob["question_summaries"])

    print(f"claims: {len(claims)}  (question, model) rationales read: {len(summaries)}")
    print(f"questions covered: {len({c['post_id'] for c in claims})} / {len(manifest)}")
    print()

    lean = [s for s in summaries if s["leans_on_base_rate"]]
    print(
        f"rationales leaning on >=1 base rate: {len(lean)}/{len(summaries)} ({100 * len(lean) / len(summaries):.0f}%)"
    )
    q_lean = {s["post_id"] for s in lean}
    q_all = {s["post_id"] for s in summaries}
    print(f"questions with >=1 leaning rationale: {len(q_lean)}/{len(q_all)} ({100 * len(q_lean) / len(q_all):.0f}%)")
    print()

    for field in ("rate_type", "rate_basis", "load_bearing", "verifiable"):
        print(field, dict(Counter(c.get(field) for c in claims)))
    print()
    print("dominant_rate_basis per rationale:", dict(Counter(s.get("dominant_rate_basis") for s in summaries)))
    print()

    # memory-based share among load-bearing claims
    lb = [c for c in claims if c["load_bearing"] in ("anchor", "adjustment")]
    print(f"load-bearing (anchor+adjustment) claims: {len(lb)}")
    print("  rate_basis:", dict(Counter(c["rate_basis"] for c in lb)))
    print("  source_cited non-null:", sum(1 for c in lb if c.get("source_cited")))
    print()

    # verification candidates: numeric, load-bearing anchor, verifiable, memory-based first
    cands = [c for c in claims if c["rate_type"] == "numeric" and c["verifiable"] and c["load_bearing"] == "anchor"]
    cands.sort(key=lambda c: (c["rate_basis"] != "memory", str(c["post_id"])))
    print(f"verification candidates (numeric anchor, verifiable): {len(cands)}")
    for c in cands:
        m = manifest.get(str(c["post_id"]), {})
        print(
            f"  {c['post_id']} [{m.get('tercile', '?'):8s} peer={m.get('peer_score', 0):7.1f}] "
            f"{c['model'].split('/')[-1][:22]:22s} {c['rate_basis']:8s} | {c['stated_rate']} | "
            f"{c['reference_class'][:90]}"
        )


if __name__ == "__main__":
    main()
