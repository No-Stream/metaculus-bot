"""Build blinded judge packets for the research-role model audit.

Adapted from scratch/driver_replay_2026-07-17/make_blind_packets.py. One packet
per (role, arm) containing ONLY the arm's output text — no wall time, token
counts, or cost (all three correlate with model tier and would deanonymize).
The arm->letter mapping is shuffled independently PER ROLE and saved to
mapping_SECRET.json (NOT committed).

Shared, unblinded context files are copied into blind/ for the judge:
- ``judge_context_question.md``  — question text / criteria / fine print
- ``judge_context_asknews_raw.md`` — the raw article dump (summarizer ground truth)
- ``judge_context_crux_inputs.md`` — the six rationales (crux ground truth),
  model tags stripped.

Usage:
    uv run python scratch/research_role_audit_2026-07-17/make_blind_packets.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent
ARMS_DIR = BASE_DIR / "arms"
BLIND_DIR = BASE_DIR / "blind"
INPUTS_DIR = BASE_DIR / "inputs"

ROLES = ["summarizer", "native_search", "crux"]
ARMS = ["sol_low", "terra_low", "luna_medium"]

# Belt-and-suspenders scrub: outputs were verified to contain no model
# self-identification, but scrub precise markers anyway. Word-boundary
# patterns so 'resolution'/'solely' survive.
MODEL_MARKERS = [
    r"openrouter/openai/gpt-5\.6-(?:sol|terra|luna)",
    r"openai/gpt-5\.6-(?:sol|terra|luna)",
    r"gpt-5\.6-(?:sol|terra|luna)",
    r"gpt-5\.6",
    r"gpt-5",
    r"\bopenai\b",
    r"\bsol\b",
    r"\bterra\b",
    r"\bluna\b",
]
_SCRUB_RE = re.compile("|".join(MODEL_MARKERS), re.IGNORECASE)


def scrub(text: str) -> str:
    return _SCRUB_RE.sub("[MODEL-REDACTED]", text)


def write_context_files() -> None:
    meta = json.loads((INPUTS_DIR / "question_meta.json").read_text(encoding="utf-8"))
    (BLIND_DIR / "judge_context_question.md").write_text(
        f"# Question (Metaculus Q44229)\n\n{meta['question_text']}\n\n"
        f"## Resolution criteria\n{meta['resolution_criteria']}\n\n"
        f"## Fine print\n{meta['fine_print']}\n\n"
        f"Question opened: {meta['open_date']}\n",
        encoding="utf-8",
    )
    (BLIND_DIR / "judge_context_asknews_raw.md").write_text(
        (INPUTS_DIR / "asknews_raw.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    crux_payload = json.loads((INPUTS_DIR / "crux_base_texts.json").read_text(encoding="utf-8"))
    parts = [f"## Forecaster {i + 1} Analysis\n\n{entry['reasoning']}\n" for i, entry in enumerate(crux_payload)]
    (BLIND_DIR / "judge_context_crux_inputs.md").write_text(
        "# Crux-analyzer inputs: the six forecaster rationales (model tags stripped)\n\n" + "\n".join(parts),
        encoding="utf-8",
    )


def main() -> None:
    BLIND_DIR.mkdir(exist_ok=True)
    write_context_files()

    rng = random.Random()  # OS-seeded; mapping recorded in the secret file
    mapping: dict[str, dict[str, str]] = {}
    for role in ROLES:
        letters = ["A", "B", "C"]
        rng.shuffle(letters)
        role_map = dict(zip(ARMS, letters, strict=True))
        mapping[role] = role_map
        for arm, letter in role_map.items():
            output = (ARMS_DIR / role / f"{arm}.md").read_text(encoding="utf-8")
            packet = f"# Role: {role} — Packet {letter}\n\n{scrub(output)}\n"
            leftover = re.findall(r"(?i)\b(sol|terra|luna|gpt-5)\b", packet.replace("[MODEL-REDACTED]", ""))
            assert not leftover, f"scrub leak in {role}/{arm}: {set(leftover)}"
            (BLIND_DIR / f"{role}_packet_{letter}.md").write_text(packet, encoding="utf-8")
            print(f"wrote {role}_packet_{letter}.md  <- {arm}")

    (BASE_DIR / "mapping_SECRET.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print("mapping saved to mapping_SECRET.json (do NOT commit)")


if __name__ == "__main__":
    main()
