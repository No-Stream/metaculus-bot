"""Decompose AskNews-section padding into mechanical categories, corpus-wide.

Categories measured deterministically (chars):
- apparatus: per-fact Source:/Credibility:/Date: label lines, [PRE-WINDOW]/[SINGLE-SOURCE]
  tags, tier tags
- tail_restatement: everything from the first tail-summary header (Key facts /
  Key Quantitative / Expert Opinions / Contradictions / credibility notes /
  Practical ... conclusion / Assessment for forecasting) to the end of the section
- implication_blocks: per-article "Implication/Relevance/Forecasting significance"
  editorial paragraphs
- offers: "If you want, I can..." chatbot lines
The residual padding (vs the judged 57%) is topical-drift article content, which
needs semantic judgment and is taken from the content-audit judgments.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from date_profile import ARCHIVE, extract_asknews_section  # noqa: E402

TAIL_HEADERS = re.compile(
    r"^#{2,4} .*(Key facts|Key Quantitative|Expert Opinions and Attributions|"
    r"Contradictions|credibility notes|Practical forecasting|Assessment for forecasting|"
    r"Key Forecasting Takeaways|Summary of key)",
    re.I | re.M,
)
APPARATUS_LINE = re.compile(
    r"^\s*[-*]?\s*\*{0,2}(Source|Credibility|Date|Publish date|Original language)\s*:",
    re.M,
)
TAGS = re.compile(r"\[(?:PRE-WINDOW[^\]]*|SINGLE-SOURCE|[ABCD]: [^\]]{1,30})\]")
IMPLICATION = re.compile(
    r"^\*{0,2}(Implication[s]? for forecasting|Relevance|Forecasting significance|Interpretation)\b.*?(?=\n#{2,4} |\n---|\Z)",
    re.M | re.S,
)
OFFER = re.compile(r"^.*If you want, I can.*$", re.M)


def decompose(section: str) -> dict[str, float]:
    body = section
    n = len(body)
    if n == 0:
        return {}
    # Only count a summary header as the briefing TAIL when it appears in the
    # final 60% of the section — per-article "Key facts" subsections early in
    # the body are the briefing's normal structure, not tail restatement.
    tail_m = None
    for m in TAIL_HEADERS.finditer(body):
        if m.start() >= 0.4 * n:
            tail_m = m
            break
    tail_chars = n - tail_m.start() if tail_m else 0
    head = body[: tail_m.start()] if tail_m else body  # avoid double count with tail
    apparatus_chars = sum(len(line) for line in APPARATUS_LINE.findall(head))
    # apparatus measured as full matched lines in head:
    apparatus_chars = 0
    for m in APPARATUS_LINE.finditer(head):
        line_end = head.find("\n", m.start())
        apparatus_chars += (line_end if line_end != -1 else len(head)) - m.start()
    tag_chars = sum(len(t) for t in TAGS.findall(head))
    impl_chars = sum(len(m.group(0)) for m in IMPLICATION.finditer(head))
    offer_chars = sum(len(m) for m in OFFER.findall(body))
    return {
        "chars": n,
        "tail_restatement": tail_chars / n,
        "apparatus_lines": apparatus_chars / n,
        "inline_tags": tag_chars / n,
        "implication_blocks": impl_chars / n,
        "offers": offer_chars / n,
    }


def main() -> None:
    rows = []
    for path in sorted(ARCHIVE.glob("*.jsonl")):
        recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rec = next((r for r in recs if r.get("schema_version") == 2), None)
        if rec is None:
            continue
        section = extract_asknews_section(rec.get("research_text") or "")
        if len(section) < 2000:
            continue
        d = decompose(section)
        d["qid"] = rec["qid"]
        rows.append(d)

    keys = ["tail_restatement", "apparatus_lines", "inline_tags", "implication_blocks", "offers"]
    print(f"n={len(rows)} asknews sections (schema-2, >=2k chars)")
    print(f"{'metric':<20} {'mean%':>7} {'median%':>8} {'p90%':>6}")
    for k in keys:
        vals = sorted(r[k] for r in rows)
        mean = sum(vals) / len(vals)
        med = vals[len(vals) // 2]
        p90 = vals[int(0.9 * len(vals))]
        print(f"{k:<20} {mean * 100:7.1f} {med * 100:8.1f} {p90 * 100:6.1f}")
    total = [sum(r[k] for k in keys) for r in rows]
    total.sort()
    print(
        f"{'TOTAL mechanical':<20} {sum(total) / len(total) * 100:7.1f} {total[len(total) // 2] * 100:8.1f} {total[int(0.9 * len(total))] * 100:6.1f}"
    )

    print("\nPer-question (the 5 failure cases):")
    for r in rows:
        if r["qid"] in (44219, 44255, 44512, 44551, 44555):
            parts = " ".join(f"{k}={r[k] * 100:.1f}%" for k in keys)
            print(f"  q{r['qid']} ({r['chars']} chars): {parts}")


if __name__ == "__main__":
    main()
