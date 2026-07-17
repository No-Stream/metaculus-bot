"""Descriptive association between base-rate usage/basis and peer score.

Joins extracted claims with the manifest and prints:
- per-tercile counts of memory vs research/computed anchors
- mean peer score for questions whose anchors are memory-only vs any-sourced
- per-question claim/basis profile with peer score (for the writeup table)

Small n — descriptive only, no significance tests.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

OUT = Path(__file__).resolve().parent


class QuestionProfile(NamedTuple):
    post_id: str
    qtype: str
    tercile: str
    peer: float
    n_loadbearing: int
    memory_share: float
    memory_only: bool
    any_sourced: bool


def load_claims() -> list[dict]:
    claims: list[dict] = []
    files = {f.stem: f for f in (OUT / "extracted").glob("claims_batch*.json")}
    for stem in sorted(files):
        if stem.endswith("b") and stem[:-1] in files:
            continue
        claims.extend(json.loads(files[stem].read_text())["claims"])
    return claims


def main() -> None:
    manifest = {str(m["post_id"]): m for m in json.loads((OUT / "sample_manifest.json").read_text())}
    claims = load_claims()

    per_q: dict[str, dict[str, int]] = defaultdict(lambda: {"memory": 0, "research": 0, "computed": 0, "anchors": 0})
    for c in claims:
        pid = str(c["post_id"])
        if c["load_bearing"] == "passing":
            continue
        per_q[pid][c["rate_basis"]] += 1
        if c["load_bearing"] == "anchor":
            per_q[pid]["anchors"] += 1

    rows: list[QuestionProfile] = []
    for pid, prof in per_q.items():
        m = manifest.get(pid)
        if not m:
            continue
        total = prof["memory"] + prof["research"] + prof["computed"]
        if total == 0:
            continue  # per_q only holds non-passing claims, so this shouldn't happen
        rows.append(
            QuestionProfile(
                post_id=pid,
                qtype=str(m["type"]),
                tercile=str(m["tercile"]),
                peer=float(m["peer_score"]),
                n_loadbearing=total,
                memory_share=prof["memory"] / total,
                memory_only=prof["research"] == 0 and prof["computed"] == 0 and prof["memory"] > 0,
                any_sourced=(prof["research"] + prof["computed"]) > 0,
            )
        )

    print("tercile x dominant basis (load-bearing claims):")
    agg: dict[str, list[QuestionProfile]] = defaultdict(list)
    for r in rows:
        agg[r.tercile].append(r)
    for terc in ("worst", "middling", "good"):
        rs = agg[terc]
        mem_only = [r for r in rs if r.memory_only]
        sourced = [r for r in rs if r.any_sourced]
        mean_mem_share = sum(r.memory_share for r in rs) / len(rs)
        print(
            f"  {terc:9s} n={len(rs):2d}  memory-only Qs={len(mem_only):2d}  any-sourced Qs={len(sourced):2d}  "
            f"mean memory share={mean_mem_share:.2f}"
        )

    mem_only_scores = [r.peer for r in rows if r.memory_only]
    sourced_scores = [r.peer for r in rows if r.any_sourced]
    print()
    print(
        f"memory-only questions: n={len(mem_only_scores)}, mean peer={sum(mem_only_scores) / len(mem_only_scores):.1f}"
    )
    print(f"any-sourced questions: n={len(sourced_scores)}, mean peer={sum(sourced_scores) / len(sourced_scores):.1f}")
    print()
    print("per-question profile (sorted by peer):")
    for r in sorted(rows, key=lambda r: r.peer):
        print(
            f"  {r.post_id} {r.qtype[:3]} {r.tercile[:4]:4s} peer={r.peer:7.1f} "
            f"n={r.n_loadbearing:2d} mem_share={r.memory_share:.2f} "
            f"{'MEMORY-ONLY' if r.memory_only else ''}"
        )


if __name__ == "__main__":
    main()
