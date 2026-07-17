"""Build blinded judge packets from replay arm outputs.

Each packet contains: findings artifact, full tool-call trace (tool choices,
queries/URLs, result status/method, budget usage), ghost summary, and per-arm
stats — with every model identifier scrubbed. The arm→letter mapping is
shuffled and saved to mapping_SECRET.json (NOT committed).

Usage:
    uv run python scratch/driver_replay_2026-07-17/make_blind_packets.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).parent
BLIND_DIR = BASE_DIR / "blind"

ARMS = [
    "arm_terra_low",
    "arm_terra_medium",
    "arm_sol_low",
    "arm_sonnet5_medium",
    "arm_luna_medium",
]

# Model-identifying strings to scrub from any judge-visible text, checked
# case-insensitively. Order matters: longer/more specific first.
MODEL_MARKERS = [
    "openrouter/openai/gpt-5.6-terra",
    "openrouter/openai/gpt-5.6-sol",
    "openrouter/openai/gpt-5.6-luna",
    "openrouter/anthropic/claude-sonnet-5",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-luna",
    "anthropic/claude-sonnet-5",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "gpt-5.6-luna",
    "claude-sonnet-5",
    "gpt-5.6",
    "gpt-5",
    "sonnet-5",
    "sonnet 5",
    "claude",
    "anthropic",
    "openai",
    "terra",
    "luna",
]
_SCRUB_RE = re.compile("|".join(re.escape(m) for m in MODEL_MARKERS), re.IGNORECASE)


def scrub(text: str) -> str:
    return _SCRUB_RE.sub("[MODEL-REDACTED]", text)


def _parse_tool_args(raw: str) -> dict[str, Any] | str:
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else raw
    except json.JSONDecodeError:
        return raw


def render_tool_trace(transcript: list[dict[str, Any]]) -> str:
    """Human-readable step-by-step trace of assistant actions + tool results."""
    lines: list[str] = []
    step = 0
    results_by_id: dict[str, dict[str, Any]] = {
        str(msg.get("tool_call_id")): msg for msg in transcript if msg.get("role") == "tool"
    }
    for msg in transcript:
        role = msg.get("role")
        if role == "assistant":
            step += 1
            tool_calls = msg.get("tool_calls") or []
            content = (msg.get("content") or "").strip()
            lines.append(f"### Step {step} (assistant)")
            if content:
                lines.append(f"Assistant text: {content[:800]}")  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # display truncation, not data subsampling
            for tc in tool_calls:
                name = tc["function"]["name"]
                args = _parse_tool_args(tc["function"]["arguments"])
                if name == "record_findings" and isinstance(args, dict):
                    n = len(args.get("findings", []))
                    lines.append(f"- TOOL CALL: record_findings ({n} finding(s) banked)")
                elif name == "conclude" and isinstance(args, dict):
                    n_final = len(args.get("final_findings", []) or [])
                    n_leads = len(args.get("pending_leads", []) or [])
                    lines.append(f"- TOOL CALL: conclude (final_findings={n_final}, pending_leads={n_leads})")
                else:
                    args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                    lines.append(f"- TOOL CALL: {name}({args_str[:500]})")  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # display truncation, not data subsampling
                result = results_by_id.get(str(tc.get("id")))
                if result is not None:
                    body = str(result.get("content") or "")
                    header_lines = [
                        line
                        for line in body.splitlines()
                        if line.startswith(("tool:", "status:", "method:", "[budget:"))
                    ]
                    preview = body.strip().splitlines()
                    content_preview = " / ".join(
                        line for line in preview if line and not line.startswith(("tool:", "status:", "method:"))
                    )[:300]
                    lines.append(f"  RESULT: {'; '.join(header_lines)}")
                    if content_preview:
                        lines.append(f"  RESULT PREVIEW: {content_preview}")
            lines.append("")
        elif role == "user" and msg.get("content") not in (None, "") and step > 0:
            content = str(msg.get("content"))
            # ghost prompt or nudge — note it without dumping the whole thing
            label = "GHOST PROMPT" if "research phase is closed" in content else "USER NUDGE"
            lines.append(f"### ({label} injected)")
            lines.append("")
    return "\n".join(lines)


def per_tool_counts_str(counts: dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"


def build_packet(arm: str, letter: str) -> str:
    arm_dir = BASE_DIR / arm
    telemetry = json.loads((arm_dir / "telemetry.json").read_text(encoding="utf-8"))
    meta = json.loads((arm_dir / "meta.json").read_text(encoding="utf-8"))
    findings = (arm_dir / "findings.md").read_text(encoding="utf-8")
    transcript = json.loads((arm_dir / "transcript.json").read_text(encoding="utf-8"))
    ghost_raw = json.loads((arm_dir / "ghost.json").read_text(encoding="utf-8"))

    trace = render_tool_trace(transcript)
    ghost_summary = (
        f"qtype={ghost_raw.get('qtype')}, parsed={ghost_raw.get('parsed_summary')}\n\n"
        f"Ghost structured block:\n{ghost_raw.get('raw_text') or '(none)'}"
        if ghost_raw
        else "(no ghost forecast produced)"
    )

    stats = f"""\
| stat | value |
|---|---|
| steps | {telemetry["steps"]} |
| tool_calls | {telemetry["tool_calls"]} |
| per-tool counts | {per_tool_counts_str(telemetry["per_tool_counts"])} |
| dup_tool_calls | {telemetry["dup_tool_calls"]} |
| deadline_hit | {telemetry["deadline_hit"]} |
| concluded_early | {telemetry["concluded_early"]} |
| wall_s (loop) | {telemetry["wall_s"]:.1f} |
| findings | {telemetry["findings_count"]} |
| pending_leads | {telemetry["pending_leads_count"]} |
| lint_rejections | {telemetry["lint_rejections"]} |
| llm calls | {meta["n_llm_calls"]} |
| prompt tokens | {meta["prompt_tokens_total"]} |
| completion tokens | {meta["completion_tokens_total"]} |
| est. cost USD | {meta["est_cost_usd"]} |
"""

    packet = f"""\
# Packet {letter}

Replay of the same gap-fill v2 research task (identical system prompt, user
brief, tools, and budgets) by an anonymized driver model.

## Per-run stats

{stats}

## Findings artifact (what the forecasting panel would receive)

{findings}

## Ghost forecast (driver's own private forecast after research freeze)

{ghost_summary}

## Full tool-call trace

{trace}
"""
    return scrub(packet)


def main() -> None:
    BLIND_DIR.mkdir(exist_ok=True)
    letters = ["A", "B", "C", "D", "E"]
    rng = random.Random()  # OS-seeded; mapping recorded in the secret file
    shuffled = letters[:]
    rng.shuffle(shuffled)
    mapping = dict(zip(ARMS, shuffled, strict=True))

    for arm, letter in mapping.items():
        packet = build_packet(arm, letter)
        leftover = re.findall(
            r"(?i)(terra|luna|sonnet|claude|anthropic|openai|gpt-5)",
            packet.replace("[MODEL-REDACTED]", ""),
        )
        assert not leftover, f"scrub leak in {arm}: {set(leftover)}"
        (BLIND_DIR / f"packet_{letter}.md").write_text(packet, encoding="utf-8")
        print(f"wrote packet_{letter}.md  <- {arm}")

    (BASE_DIR / "mapping_SECRET.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print("mapping saved to mapping_SECRET.json (do NOT commit)")


if __name__ == "__main__":
    main()
