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

# Label prefixing a finding that carries a ``derivation`` (W3) — flags the
# arithmetic synthesis as ours so the panel weights it accordingly.
_DERIVED_ANALYSIS_LABEL = "Derived analysis (arithmetic from quoted sources)"


def detachment_lint(finding: Finding) -> list[str]:
    # Only ``claim`` and ``topic`` are scanned. ``derivation`` (W3) is
    # deliberately EXEMPT — it is arithmetic-only synthesis over the finding's
    # own quoted numbers (a saw-tooth-style bound/rate table), which needs no
    # likelihood language and would false-positive on incidental register words
    # inside a computation. Do NOT add ``derivation`` here; the contract that
    # keeps it safe (every input quoted, arithmetic only, no new facts) is a
    # prompt/render-side convention, not a lint rule. ``quote`` stays unscanned
    # too — it is verbatim source text, not the driver's own register.
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
            "The sourced findings below contradict the research briefing and supersede the corresponding briefing content."
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
            if finding.derivation:
                # W3: label our arithmetic synthesis so the panel weights it as
                # a derived analysis, not a source claim. Deterministic: emitted
                # iff the field is set, in stable field order.
                lines.append(_DERIVED_ANALYSIS_LABEL)
            lines.append(f"Claim: {finding.claim}")
            lines.append(f"Source: {finding.source_url}")
            lines.append("Quote:")
            lines.extend(_quote_lines(finding.quote))
            if finding.derivation:
                lines.append(f"Derivation: {finding.derivation}")
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
