"""Date-profile the AskNews section of archived research bundles.

For each qid: extract the '### News Articles (AskNews)' section from the
schema-2 archive record, pull every date mention, and report the histogram of
cited dates relative to the run timestamp. Used to classify staleness failures
as FETCH (raw article set stale) vs SUMMARIZE (fresh inputs distorted).
All local/free — reads backtests/research_archive/by_qid/ only.
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[2] / "backtests" / "research_archive" / "by_qid"

MONTHS = {
    m: i + 1
    for i, m in enumerate(
        "january february march april may june july august september october november december".split()
    )
}
# **June 29, 2026** / June 29, 2026 / July 1 2026
DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})(?:,)?\s+(\d{4})\b",
    re.IGNORECASE,
)
ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


SECTION_HEADER_RE = re.compile(r"^## (?!#)", re.M)


def extract_asknews_section(research_text: str) -> str:
    """Slice the bundle between top-level '## ' headers.

    Can't split on the '\\n\\n---\\n\\n' joiner: the summarizer's briefing
    contains its own '---' horizontal rules, which truncates the section.
    Inner headings are demoted so only true section headers start with '## '.
    """
    starts = [m.start() for m in SECTION_HEADER_RE.finditer(research_text)]
    for i, s in enumerate(starts):
        header_line = research_text[s : research_text.index("\n", s)]
        if "News Articles (AskNews)" in header_line:
            end = starts[i + 1] if i + 1 < len(starts) else len(research_text)
            return research_text[s:end]
    return ""


def profile(qid: int) -> None:
    path = ARCHIVE / f"{qid}.jsonl"
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rec = next(r for r in recs if r["schema_version"] == 2)
    run_dt = datetime.fromisoformat(rec["timestamp"].replace("Z", "+00:00"))
    section = extract_asknews_section(rec["research_text"])
    if not section:
        print(f"q{qid}: NO asknews section found")
        return

    dates: list[datetime] = []
    for mo, d, y in DATE_RE.findall(section):
        try:
            dates.append(datetime(int(y), MONTHS[mo.lower()], int(d), tzinfo=timezone.utc))
        except ValueError:
            pass
    for y, mo, d in ISO_RE.findall(section):
        try:
            dates.append(datetime(int(y), int(mo), int(d), tzinfo=timezone.utc))
        except ValueError:
            pass
    # Keep only plausible article-ish dates (not far-future resolution dates)
    cited = [dt for dt in dates if dt <= run_dt]
    future = [dt for dt in dates if dt > run_dt]
    ages = sorted((run_dt - dt).days for dt in cited)
    n = len(ages)
    print(f"\n=== q{qid} run {run_dt:%Y-%m-%d} | asknews {len(section)} chars | {n} past-date mentions ===")
    if n:

        def pct(p: float) -> int:
            return ages[min(n - 1, int(p * n))]

        buckets = {
            "<=3d": sum(a <= 3 for a in ages),
            "4-14d": sum(3 < a <= 14 for a in ages),
            "15-45d": sum(14 < a <= 45 for a in ages),
            "46-120d": sum(45 < a <= 120 for a in ages),
            ">120d": sum(a > 120 for a in ages),
        }
        print(f"  age days: min={ages[0]} p25={pct(0.25)} median={pct(0.5)} p75={pct(0.75)} max={ages[-1]}")
        print(f"  buckets: {buckets}  (future-dated mentions excl.: {len(future)})")
        freshest = sorted(set(cited), reverse=True)[:4]
        print(f"  freshest cited dates: {[f'{d:%Y-%m-%d}' for d in freshest]}")
    # Publish-date lines survive only in raw fallback; count Source:/Credibility: apparatus
    print(
        f"  apparatus: Source: x{section.count('Source:')}  Credibility: x{section.count('Credibility:')}"
        f"  [PRE-WINDOW x{section.count('[PRE-WINDOW')}  [SINGLE-SOURCE x{section.count('[SINGLE-SOURCE')}"
    )
    for marker in ("Bottom line", "Key facts", "Contradiction", "If you want"):
        c = section.count(marker)
        if c:
            print(f"  marker '{marker}': x{c}")


if __name__ == "__main__":
    qids = [int(a) for a in sys.argv[1:]] or [44219, 44255, 44512, 44551, 44555]
    for q in qids:
        profile(q)
