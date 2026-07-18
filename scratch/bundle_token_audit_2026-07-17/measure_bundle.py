"""Research-bundle token budget audit (2026-07-17).

Measures per-section char/token contributions to the research bundle every
forecaster reads, across:
  1. The 64 schema-2 (untrimmed) prod artifacts in backtests/research_archive/by_qid/
  2. The 2026-07-17 Zambia smoke bundle extracted from /tmp/v2-smoke.log
     (saved as zambia_bundle.txt next to this script if the log is gone)

Also runs a cheap cross-provider redundancy screen (shared numbers/entities/domains
between AskNews vs Native Search vs Gemini sections) and a rationale-citation scan
over scratch/coherence_2026-07-15/perf_all_tagged.json.

Token estimate = chars / 4 (crude; ~±15% for English prose with markdown).

Run: uv run python scratch/bundle_token_audit_2026-07-17/measure_bundle.py
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/Users/flatljan/personal/metaculus-bot")
ARCHIVE = REPO / "backtests/research_archive/by_qid"
PERF = REPO / "scratch/coherence_2026-07-15/perf_all_tagged.json"
OUT_DIR = Path(__file__).parent

# Known top-level section headers (orchestrator._provider_header + gap-fill/agentic/diagnostics)
KNOWN_HEADERS = [
    "## News Articles (AskNews)",
    "## Web Research (Native Search)",
    "## Web Research (Google Search via Gemini)",
    "## Financial & Economic Data",
    "## Time Series Anchor",
    "## Prediction Market Snapshot",
    "## Resolution Source Snapshot",
    "## Web Research (Exa)",
    "## Web Research (Perplexity)",
    "## Web Research (OpenRouter)",
    "## Targeted Gap-Fill (second pass)",
    "## Agentic Research Findings",
    "## Provider Diagnostics",
]
SHORT = {
    "## News Articles (AskNews)": "asknews",
    "## Web Research (Native Search)": "native_search",
    "## Web Research (Google Search via Gemini)": "gemini_search",
    "## Financial & Economic Data": "financial_data",
    "## Time Series Anchor": "ts_anchor",
    "## Prediction Market Snapshot": "prediction_market",
    "## Resolution Source Snapshot": "resolution_source",
    "## Web Research (Exa)": "exa",
    "## Web Research (Perplexity)": "perplexity",
    "## Web Research (OpenRouter)": "openrouter",
    "## Targeted Gap-Fill (second pass)": "gap_fill_v1",
    "## Agentic Research Findings": "gap_fill_v2",
    "## Provider Diagnostics": "diagnostics",
}


def split_sections(text: str) -> dict[str, str]:
    """Attribute every line to the most recent KNOWN header.

    Inner '## Finding' headers emitted by gap-fill search results (not demoted)
    stay attributed to gap_fill_v1 — split on known headers only.
    """
    sections: dict[str, list[str]] = defaultdict(list)
    current = "preamble"
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped in SHORT:
            current = SHORT[stripped]
        sections[current].append(line)
    return {k: "".join(v) for k, v in sections.items()}


def load_schema2_artifacts() -> list[dict]:
    rows = []
    for f in sorted(ARCHIVE.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("schema_version") == 2 and d.get("research_text"):
                rows.append(d)
    return rows


def pct(vals: list[int] | list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[idx]


def stats_row(vals: list[int], n_total: int) -> dict:
    nonzero = [v for v in vals if v > 0]
    return {
        "present_n": len(nonzero),
        "present_frac": round(len(nonzero) / n_total, 2) if n_total else 0,
        "mean_when_present": int(statistics.mean(nonzero)) if nonzero else 0,
        "median_when_present": int(statistics.median(nonzero)) if nonzero else 0,
        "p90": int(pct(nonzero, 0.9)) if nonzero else 0,
        "max": max(nonzero) if nonzero else 0,
        "mean_over_all_qs": int(statistics.mean(vals)) if vals else 0,
    }


# --- redundancy helpers -----------------------------------------------------

NUM_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")
ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\b")
DOMAIN_RE = re.compile(r"https?://(?:www\.)?([^/\s)\]]+)")
STOP_ENTITIES = {
    "The",
    "This",
    "These",
    "Bottom Line",
    "Historical Context",
    "Recent Developments",
    "Key Factors",
    "Research Summary",
    "Intelligence Briefing",
    "United States",
}


def fingerprint(text: str) -> tuple[set[str], set[str], set[str]]:
    nums = {m.group(0) for m in NUM_RE.finditer(text)}
    # drop trivial small integers / years alone are still meaningful, keep
    nums = {n for n in nums if len(n) > 1}
    ents = {m.group(1) for m in ENTITY_RE.finditer(text)} - STOP_ENTITIES
    domains = {m.group(1).lower() for m in DOMAIN_RE.finditer(text)}
    return nums, ents, domains


def overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# --- main -------------------------------------------------------------------


def main() -> None:
    artifacts = load_schema2_artifacts()
    n = len(artifacts)
    print(
        f"schema-2 untrimmed artifacts: n={n} ({artifacts[0]['timestamp'][:10]} .. {artifacts[-1]['timestamp'][:10]})"
    )

    per_section: dict[str, list[int]] = defaultdict(list)
    totals = []
    redund = defaultdict(list)
    for d in artifacts:
        secs = split_sections(d["research_text"])
        totals.append(len(d["research_text"]))
        for name in list(SHORT.values()) + ["preamble"]:
            per_section[name].append(len(secs.get(name, "")))
        # redundancy fingerprints across the three news-ish providers
        fps = {}
        for prov in ("asknews", "native_search", "gemini_search"):
            if secs.get(prov):
                fps[prov] = fingerprint(secs[prov])
        pairs = [("asknews", "native_search"), ("asknews", "gemini_search"), ("native_search", "gemini_search")]
        for a, b in pairs:
            if a in fps and b in fps:
                redund[(a, b, "numbers")].append(overlap(fps[a][0], fps[b][0]))
                redund[(a, b, "entities")].append(overlap(fps[a][1], fps[b][1]))
                redund[(a, b, "domains")].append(overlap(fps[a][2], fps[b][2]))

    print(f"\n=== Per-section size stats (chars; tokens ~= chars/4) over {n} prod questions ===")
    order = [
        "asknews",
        "native_search",
        "gemini_search",
        "gap_fill_v1",
        "prediction_market",
        "resolution_source",
        "financial_data",
        "gap_fill_v2",
        "ts_anchor",
        "diagnostics",
        "preamble",
    ]
    header = f"{'section':<20}{'present':>8}{'mean':>9}{'median':>9}{'p90':>9}{'max':>9}{'mean_all':>10}{'tok_mean':>9}"
    print(header)
    rows_out = {}
    for name in order:
        vals = per_section[name]
        s = stats_row(vals, n)
        rows_out[name] = s
        print(
            f"{name:<20}{s['present_frac']:>8}{s['mean_when_present']:>9}"
            f"{s['median_when_present']:>9}{s['p90']:>9}{s['max']:>9}"
            f"{s['mean_over_all_qs']:>10}{s['mean_over_all_qs'] // 4:>9}"
        )
    print(
        f"\nTOTAL bundle: mean={int(statistics.mean(totals))} "
        f"median={int(statistics.median(totals))} p90={int(pct(totals, 0.9))} "
        f"max={max(totals)} (tokens mean ~{int(statistics.mean(totals)) // 4})"
    )

    print("\n=== Cross-provider redundancy (overlap coefficient, mean over questions) ===")
    for (a, b, kind), vals in sorted(redund.items()):
        print(f"{a:>14} vs {b:<14} {kind:<9} mean={statistics.mean(vals):.2f} p90={pct(vals, 0.9):.2f} (n={len(vals)})")

    # --- Zambia smoke bundle -------------------------------------------------
    smoke_path = Path("/tmp/zambia_bundle.txt")
    if smoke_path.exists():
        bundle = smoke_path.read_text()
        (OUT_DIR / "zambia_bundle.txt").write_text(bundle)
        secs = split_sections(bundle)
        print(
            f"\n=== Zambia smoke bundle (2026-07-17, gap-fill v1+v2 both ON) "
            f"total={len(bundle)} chars ~{len(bundle) // 4} tokens ==="
        )
        for name, body in sorted(secs.items(), key=lambda kv: -len(kv[1])):
            print(f"{name:<20}{len(body):>8} chars {len(body) // 4:>7} tokens")

    # --- rationale citation scan ---------------------------------------------
    perf = json.loads(PERF.read_text())
    recent = [x for x in perf if (x.get("bot_comment_created_at") or "") >= "2026-05-01" and x.get("comment_text")]
    print(f"\n=== Rationale citation scan (comments since 2026-05-01, n={len(recent)}) ===")
    # Distinctive markers per source. We scan only the FORECASTS part (rationales),
    # not the RESEARCH echo, to see what forecasters actually leaned on.
    markers = {
        "asknews": [r"AskNews", r"[Ii]ntelligence [Bb]riefing", r"news briefing"],
        "native_search": [r"[Nn]ative [Ss]earch", r"[Ww]eb [Rr]esearch"],
        "gemini_search": [r"Gemini", r"Google Search"],
        "prediction_market": [r"Polymarket", r"Kalshi", r"Manifold", r"PredictIt", r"prediction market"],
        "resolution_source": [r"[Rr]esolution [Ss]ource [Ss]napshot"],
        "financial_data": [r"FRED", r"yfinance", r"Financial & Economic Data"],
        "gap_fill": [r"[Gg]ap[- ][Ff]ill", r"second pass", r"targeted search"],
        "generic_research": [r"[Tt]he research", r"[Pp]rovided research", r"[Bb]riefing"],
    }
    cite_counts = Counter()
    q_with_any = Counter()
    for x in recent:
        c = x["comment_text"]
        idx = c.find("# FORECASTS")
        rationales = c[idx:] if idx >= 0 else c
        for src, pats in markers.items():
            hits = sum(len(re.findall(p, rationales)) for p in pats)
            cite_counts[src] += hits
            if hits:
                q_with_any[src] += 1
    for src in markers:
        print(f"{src:<20} mentions={cite_counts[src]:>5}  questions_with_mention={q_with_any[src]:>3}/{len(recent)}")

    json.dump(
        {
            "per_section": rows_out,
            "total": {
                "mean": int(statistics.mean(totals)),
                "median": int(statistics.median(totals)),
                "p90": int(pct(totals, 0.9)),
                "max": max(totals),
            },
            "n": n,
        },
        open(OUT_DIR / "section_stats.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
