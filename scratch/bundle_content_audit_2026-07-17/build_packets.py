"""Build judging packets for the bundle CONTENT audit (2026-07-17).

Follow-on to scratch/bundle_token_audit_2026-07-17/ (token counts + fingerprint
redundancy). That audit found 40-47% numeric-fact overlap across the three
news-ish sections; the operator wants a content-level judgment before cutting:
per section, per question, what fraction is (a) uniquely load-bearing,
(b) corroborative duplication, (c) pure repetition/padding.

This script builds one markdown packet per sampled question containing the
question metadata (title, type, resolution criteria, fine print — fetched once
from the free read-only Metaculus API) plus the archived research bundle split
into its per-provider sections. Judge subagents read the packets; no LLM or
paid API is invoked here.

Run: uv run python scratch/bundle_content_audit_2026-07-17/build_packets.py
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path("/Users/flatljan/personal/metaculus-bot")
ARCHIVE = REPO / "backtests/research_archive/by_qid"
OUT = Path(__file__).parent
PACKETS = OUT / "packets"

# Sample: 3 fat bundles (>p90=53,664 chars), mid-range mix across qtypes/topics,
# one small bundle, and the two questions the token audit used as redundancy
# examples (44255, 44563) so fingerprint overlap can be checked against judged
# content overlap.
SAMPLE_QIDS = [
    44773,  # Brent crude price — numeric, financial, FAT (61,371)
    44558,  # Level-4 travel advisories count — numeric/discrete, FAT (61,115)
    44555,  # Collins vs Dem polling lead — binary, election, FAT (60,357)
    44563,  # Trump midwest-state visits — binary, token-audit example (48,404)
    44512,  # Australia Commonwealth gold medals — numeric, sports (47,425)
    44551,  # hottest-temperature state — MC, weather (44,028)
    44219,  # highest Artificial Analysis lab — MC, AI (44,059)
    44255,  # H.R.6644 becomes law — binary, RESOLVED, token-audit example (40,630)
    44453,  # July 2026 US jobs added — numeric, macro (37,096)
    44225,  # arXiv agentic-paper count — numeric, small bundle (30,876)
]

# Same section splitter as bundle_token_audit_2026-07-17/measure_bundle.py.
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
SECTION_ORDER = [
    "preamble",
    "asknews",
    "native_search",
    "gemini_search",
    "financial_data",
    "ts_anchor",
    "prediction_market",
    "resolution_source",
    "gap_fill_v1",
    "gap_fill_v2",
    "diagnostics",
]


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = defaultdict(list)
    current = "preamble"
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped in SHORT:
            current = SHORT[stripped]
        sections[current].append(line)
    return {k: "".join(v) for k, v in sections.items()}


def load_artifact(qid: int) -> dict:
    f = ARCHIVE / f"{qid}.jsonl"
    best = None
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("schema_version") == 2 and d.get("research_text"):
            best = d  # last schema-2 row wins (most recent run)
    if best is None:
        raise ValueError(f"no schema-2 artifact for qid {qid}")
    return best


def metaculus_token() -> str:
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("METACULUS_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("METACULUS_TOKEN not found in .env")


def fetch_question_meta(post_url: str, qid: int, token: str) -> dict:
    """Fetch title/type/resolution_criteria/fine_print for a question.

    page_url in the artifact points at the POST (group posts contain several
    sub-questions); locate the sub-question whose id matches qid. Responses
    are cached to meta_cache/ so reruns after a 429 don't re-hit the API.
    """
    post_id = post_url.rstrip("/").split("/")[-1]
    cache_dir = OUT / "meta_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"post_{post_id}.json"
    if cache_path.exists():
        post = json.loads(cache_path.read_text())
    else:
        req = urllib.request.Request(
            f"https://www.metaculus.com/api/posts/{post_id}/",
            headers={"Authorization": f"Token {token}", "User-Agent": "metaculus-bot-audit"},
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    post = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 3:
                    wait = 20 * (attempt + 1)
                    print(f"  429 on post {post_id}, backing off {wait}s")
                    time.sleep(wait)
                else:
                    raise
        cache_path.write_text(json.dumps(post))
        time.sleep(3)  # politeness between fresh fetches

    candidates = []
    if post.get("question"):
        candidates.append(post["question"])
    gq = post.get("group_of_questions") or {}
    candidates.extend(gq.get("questions") or [])
    for q in candidates:
        if q.get("id") == qid:
            return {
                "post_id": post_id,
                "post_title": post.get("title", ""),
                "question_title": q.get("title", ""),
                "type": q.get("type", ""),
                "description": q.get("description", ""),
                "resolution_criteria": q.get("resolution_criteria", ""),
                "fine_print": q.get("fine_print", ""),
                "scheduled_resolve_time": q.get("scheduled_resolve_time", ""),
            }
    raise ValueError(f"question {qid} not found in post {post_id}")


def main() -> None:
    PACKETS.mkdir(exist_ok=True)
    token = metaculus_token()
    manifest = []
    for qid in SAMPLE_QIDS:
        art = load_artifact(qid)
        meta = fetch_question_meta(art["page_url"], qid, token)
        secs = split_sections(art["research_text"])

        lines = [
            f"# Judging packet — Q{qid}",
            "",
            f"- **Question:** {meta['question_title'] or meta['post_title']}",
            f"- **Post title:** {meta['post_title']}",
            f"- **Type:** {meta['type']}",
            f"- **Scheduled resolve:** {meta['scheduled_resolve_time'][:10]}",
            f"- **Bundle size:** {art['research_chars']} chars (run {art['timestamp'][:10]})",
            "",
            "## Question description",
            "",
            meta["description"].strip() or "(none)",
            "",
            "## Resolution criteria",
            "",
            meta["resolution_criteria"].strip() or "(none)",
            "",
            "## Fine print",
            "",
            meta["fine_print"].strip() or "(none)",
            "",
            "---",
            "",
            "# RESEARCH BUNDLE (split per section, verbatim)",
        ]
        section_sizes = {}
        for name in SECTION_ORDER:
            body = secs.get(name, "")
            if not body.strip():
                continue
            section_sizes[name] = len(body)
            lines += [
                "",
                f"<<<SECTION: {name} | {len(body)} chars>>>",
                "",
                body.rstrip(),
                "",
                f"<<<END SECTION: {name}>>>",
            ]
        out_path = PACKETS / f"q{qid}.md"
        out_path.write_text("\n".join(lines))
        manifest.append(
            {
                "qid": qid,
                "type": meta["type"],
                "question_title": meta["question_title"] or meta["post_title"],
                "bundle_chars": art["research_chars"],
                "sections": section_sizes,
                "packet": str(out_path.relative_to(REPO)),
            }
        )
        print(f"q{qid}: {meta['type']:<16} {art['research_chars']:>6} chars  {out_path.name}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(manifest)} packets + manifest.json")


if __name__ == "__main__":
    main()
