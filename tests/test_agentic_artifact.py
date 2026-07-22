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


def test_render_findings_hoists_discrepancies_to_corrections_block_first() -> None:
    findings = [
        Finding(
            claim="The briefing says the rate is 7%, but the source says 5.2%.",
            source_url="https://example.com/correction",
            quote="The unemployment rate was 5.2%.",
            date="2026-07-10",
            retrieved_how="fetch",
            topic="labor",
            discrepancy=True,
        ),
        Finding(
            claim="The ministry published a bulletin.",
            source_url="https://example.com/bulletin",
            quote="Bulletin issued July 10.",
            date="2026-07-10",
            retrieved_how="fetch",
            topic="labor",
        ),
    ]

    rendered = render_findings(findings, [])

    assert "### ⚠ Corrections to the briefing" in rendered
    assert rendered.index("### ⚠ Corrections to the briefing") < rendered.index("### labor")
    assert (
        "The sourced findings below contradict the research briefing and supersede the corresponding briefing content."
        in rendered
    )
    assert "The briefing says the rate is 7%, but the source says 5.2%." in rendered


def test_render_findings_without_discrepancies_has_no_corrections_block() -> None:
    rendered = render_findings(
        [
            Finding(
                claim="The ministry published a bulletin.",
                source_url="https://example.com/bulletin",
                quote="Bulletin issued July 10.",
                date="2026-07-10",
                retrieved_how="fetch",
                topic="labor",
            )
        ],
        [],
    )

    assert "### ⚠ Corrections to the briefing" not in rendered
    assert rendered.startswith("## Agentic Research Findings\n\n### labor")


def test_detachment_lint_flags_discrepancy_claims_too() -> None:
    violations = detachment_lint(
        Finding(
            claim="The briefing likely overstates the rate.",
            source_url="https://example.com/correction",
            quote="The rate was 5.2%.",
            discrepancy=True,
        )
    )

    assert "claim contains banned register 'likely'" in violations


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
