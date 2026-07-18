"""Dump the AskNews section of each failure bundle to a text file for close reading."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from date_profile import ARCHIVE, extract_asknews_section  # noqa: E402

OUT = Path(__file__).parent / "sections"
OUT.mkdir(exist_ok=True)

for qid in [44219, 44255, 44512, 44551, 44555]:
    recs = [json.loads(line) for line in (ARCHIVE / f"{qid}.jsonl").read_text().splitlines() if line.strip()]
    rec = next(r for r in recs if r["schema_version"] == 2)
    section = extract_asknews_section(rec["research_text"])
    (OUT / f"asknews_{qid}.md").write_text(section)
    print(qid, len(section))
