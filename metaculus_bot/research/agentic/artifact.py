from __future__ import annotations

import re
from collections import defaultdict

from metaculus_bot.research.agentic.types import Finding

_BANNED_REGISTER_PATTERNS: tuple[str, ...] = (
    r"likely",
    r"unlikely",
    r"probably",
    r"probable",
    r"suggests",
    r"indicates that",
    r"we believe",
    r"we expect",
    r"expected to (?:win|pass|fail|resolve)",
    r"points to (?:a|the)",
    r"this implies",
    r"bullish",
    r"bearish",
    r"i think",
    r"in our view",
    r"odds are",
)
_BANNED_REGISTER_RE = re.compile(r"\b(?:" + "|".join(_BANNED_REGISTER_PATTERNS) + r")\b", re.IGNORECASE)


def detachment_lint(finding: Finding) -> list[str]:
    violations: list[str] = []
    for field_name, value in (("claim", finding.claim), ("topic", finding.topic)):
        matches = [match.group(0) for match in _BANNED_REGISTER_RE.finditer(value)]
        for match in matches:
            violations.append(f"{field_name} contains banned register {match!r}")
    return violations


def _quote_lines(quote: str) -> list[str]:
    return [f"> {line}" if line else ">" for line in quote.splitlines() or [""]]


def render_findings(findings: list[Finding], pending_leads: list[str]) -> str:
    if not findings and not pending_leads:
        return ""

    corrections = [finding for finding in findings if finding.discrepancy]
    grouped: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        if finding.discrepancy:
            continue
        grouped[finding.topic].append(finding)

    lines = ["## Agentic Research Findings"]
    if corrections:
        lines.append("")
        lines.append("### ⚠ Corrections to the briefing")
        lines.append(
            "The verified findings below contradict the research briefing and supersede the corresponding briefing content."
        )
        lines.append("")
        for finding in corrections:
            lines.append(f"Claim: {finding.claim}")
            lines.append(f"Source: {finding.source_url}")
            lines.append("Quote:")
            lines.extend(_quote_lines(finding.quote))
            lines.append(f"Date: {finding.date}")
            lines.append(f"Retrieved how: {finding.retrieved_how}")
            lines.append("")
        lines.pop()

    for topic in sorted(grouped):
        lines.append("")
        lines.append(f"### {topic}")
        for finding in grouped[topic]:
            lines.append(f"Claim: {finding.claim}")
            lines.append(f"Source: {finding.source_url}")
            lines.append("Quote:")
            lines.extend(_quote_lines(finding.quote))
            lines.append(f"Date: {finding.date}")
            lines.append(f"Retrieved how: {finding.retrieved_how}")
            lines.append("")
        lines.pop()

    if pending_leads:
        lines.append("")
        lines.append("Pending leads:")
        for lead in pending_leads:
            lines.append(f"- {lead}")

    return "\n".join(lines)
