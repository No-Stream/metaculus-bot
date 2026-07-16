from __future__ import annotations

from metaculus_bot.research.agentic.artifact import detachment_lint, render_findings
from metaculus_bot.research.agentic.types import Finding


def test_render_findings_empty_returns_empty_string() -> None:
    assert render_findings([], []) == ""


def test_render_findings_is_deterministic_and_topic_sorted() -> None:
    findings = [
        Finding(
            claim="Treasury released the report.",
            source_url="https://example.com/b",
            quote="Report published on July 1.",
            date="2026-07-01",
            retrieved_how="fetch",
            topic="zeta",
        ),
        Finding(
            claim="The ministry published a corrigendum.",
            source_url="https://example.com/a1",
            quote="Corrigendum posted the next day.",
            date="2026-07-02",
            retrieved_how="read_document",
            topic="alpha",
        ),
        Finding(
            claim="A regulator archived the filing.",
            source_url="https://example.com/a2",
            quote="Archived filing includes the relevant clause.",
            date="2026-07-03",
            retrieved_how="fetch",
            topic="alpha",
        ),
    ]

    rendered = render_findings(findings, ["Check the appendix PDF."])

    assert rendered.startswith("## Agentic Research Findings")
    assert rendered.index("### alpha") < rendered.index("### zeta")
    assert rendered.index("The ministry published a corrigendum.") < rendered.index("A regulator archived the filing.")
    assert "Pending leads:\n- Check the appendix PDF." in rendered


def test_detachment_lint_flags_claim_and_topic_only() -> None:
    finding = Finding(
        claim="This likely resolves in 2026.",
        source_url="https://example.com",
        quote="Analyst note says the odds are improving.",
        topic="Bullish updates",
    )

    violations = detachment_lint(finding)

    assert "claim contains banned register 'likely'" in violations
    assert "topic contains banned register 'Bullish'" in violations
    assert all("odds are" not in violation for violation in violations)
