"""Deterministic boilerplate + verbatim-duplication scan over all 64 schema-2 bundles.

Complements the judged 10-question sample with corpus-wide numbers:
  1. Boilerplate apparatus density per section (Source:/Credibility:/hedging filler).
  2. Verbatim 12-gram cross-section duplication (stricter than the token audit's
     number/entity fingerprints — catches copy-paste-level restatement only).
  3. Within-section 12-gram self-repetition (the category-(c) signature).

Free and read-only. Run:
  uv run python scratch/bundle_content_audit_2026-07-17/boilerplate_scan.py
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/Users/flatljan/personal/metaculus-bot")
ARCHIVE = REPO / "backtests/research_archive/by_qid"
OUT = Path(__file__).parent

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

BOILERPLATE_PATTERNS = {
    "credibility_tag": re.compile(r"^\s*[-*]?\s*\**Credibility[:*]", re.M),
    "source_tag": re.compile(r"^\s*[-*]?\s*\**Source[s]?[:*]", re.M),
    "not_specified_filler": re.compile(
        r"(?:research|sources?|articles?|briefing|reporting) (?:provided )?"
        r"(?:do(?:es)? not|did not|doesn'?t) (?:specify|state|report|include|mention|confirm)",
        re.I,
    ),
    "no_information_filler": re.compile(
        r"no (?:specific |direct |additional |further )?(?:information|data|reporting|evidence|articles?) "
        r"(?:was |were |is |are )?(?:found|available|provided|identified|located)",
        re.I,
    ),
    "as_of_hedge": re.compile(r"as of (?:the latest|this writing|the most recent)", re.I),
}


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = defaultdict(list)
    current = "preamble"
    for line in text.splitlines(keepends=True):
        if line.strip() in SHORT:
            current = SHORT[line.strip()]
        sections[current].append(line)
    return {k: "".join(v) for k, v in sections.items()}


def ngrams(text: str, n: int = 12) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9%$.,]+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def ngram_list(text: str, n: int = 12) -> list[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9%$.,]+", text.lower())
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def main() -> None:
    artifacts = []
    for f in sorted(ARCHIVE.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("schema_version") == 2 and d.get("research_text"):
                artifacts.append(d)
    n = len(artifacts)
    print(f"artifacts: {n}")

    boiler_counts: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    boiler_chars: dict[str, list[float]] = defaultdict(list)
    cross_dup: dict[tuple[str, str], list[float]] = defaultdict(list)
    self_rep: dict[str, list[float]] = defaultdict(list)

    news_sections = ("asknews", "native_search", "gemini_search", "gap_fill_v1")

    for d in artifacts:
        secs = split_sections(d["research_text"])
        for name in news_sections:
            body = secs.get(name, "")
            if len(body) < 500:
                continue
            # boilerplate hits + chars consumed by boilerplate lines
            hit_chars = 0
            for pname, pat in BOILERPLATE_PATTERNS.items():
                ms = list(pat.finditer(body))
                boiler_counts[name][pname].append(len(ms))
                for m in ms:
                    # charge the whole line containing the match
                    start = body.rfind("\n", 0, m.start()) + 1
                    end = body.find("\n", m.end())
                    end = len(body) if end == -1 else end
                    hit_chars += end - start
            boiler_chars[name].append(hit_chars / len(body))
            # within-section verbatim self-repetition (12-gram appearing 2+ times)
            grams = ngram_list(body)
            if grams:
                c = Counter(grams)
                repeated = sum(v - 1 for v in c.values() if v > 1)
                self_rep[name].append(repeated / len(grams))
        # cross-section verbatim duplication
        fps = {s: ngrams(secs[s]) for s in news_sections if len(secs.get(s, "")) > 500}
        keys = sorted(fps)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if fps[a] and fps[b]:
                    cross_dup[(a, b)].append(len(fps[a] & fps[b]) / min(len(fps[a]), len(fps[b])))

    print("\n=== Boilerplate apparatus (per question, when section present) ===")
    for name in news_sections:
        parts = [
            f"{pname}: mean {statistics.mean(v):.1f}"
            for pname, v in sorted(boiler_counts[name].items())
            if v and statistics.mean(v) >= 0.05
        ]
        frac = statistics.mean(boiler_chars[name]) if boiler_chars[name] else 0
        print(f"{name:<16} boilerplate-line char share: {frac:.1%}   ({'; '.join(parts)})")

    print("\n=== Verbatim 12-gram cross-section duplication (overlap coeff; copy-paste level) ===")
    for (a, b), vals in sorted(cross_dup.items()):
        print(f"{a:>14} vs {b:<14} mean={statistics.mean(vals):.3f} max={max(vals):.3f} (n={len(vals)})")

    print("\n=== Within-section verbatim self-repetition (share of 12-grams that are repeats) ===")
    for name in news_sections:
        vals = self_rep[name]
        if vals:
            print(f"{name:<16} mean={statistics.mean(vals):.3f} p90={sorted(vals)[int(0.9 * (len(vals) - 1))]:.3f}")

    json.dump(
        {
            "boilerplate_char_share": {k: statistics.mean(v) for k, v in boiler_chars.items()},
            "cross_dup_12gram": {f"{a}|{b}": statistics.mean(v) for (a, b), v in cross_dup.items()},
            "self_repetition_12gram": {k: statistics.mean(v) for k, v in self_rep.items()},
            "n": n,
        },
        open(OUT / "boilerplate_stats.json", "w"),
        indent=2,
    )
    print("\nwrote boilerplate_stats.json")


if __name__ == "__main__":
    main()
