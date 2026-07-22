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


def test_detachment_lint_exempts_derivation_field() -> None:
    """W3: the ``derivation`` field is arithmetic-only synthesis (saw-tooth-class
    tables/bounds) and is exempt from the banned-register scan. The SAME phrase
    that trips the lint in ``claim`` passes when it appears only in derivation."""
    banned_phrase = "the annual record probably tops the prior year by 1"
    exempt = Finding(
        claim="Yearly oldest-human records reconstructed from the quoted table.",
        source_url="https://example.com/table",
        quote="1997: 122; 1998: 116; 1999: 119.",
        derivation=banned_phrase,
    )
    linted = Finding(
        claim=banned_phrase,
        source_url="https://example.com/table",
        quote="1997: 122; 1998: 116; 1999: 119.",
    )

    assert detachment_lint(exempt) == []
    assert any("probably" in violation for violation in detachment_lint(linted))


def test_render_labels_derived_findings() -> None:
    """W3: a finding carrying a derivation gets a visible label in its topic block
    so forecasters weight it as our synthesis, not a source claim."""
    findings = [
        Finding(
            claim="Per-year upper bound on the oldest living human, from the quoted record.",
            source_url="https://example.com/grg",
            quote="Verified oldest person 1990-2000: 122, 122, 122, 122, 119, 117, 116, 115, 114, 113.",
            date="2026-07-01",
            retrieved_how="fetch",
            topic="oldest-human bounds",
            derivation="Max verified age per calendar year: 1990=122 ... 2000=113; annual step never exceeds +1.",
        )
    ]

    rendered = render_findings(findings, [])

    assert "### oldest-human bounds" in rendered
    assert "Derived analysis (arithmetic from quoted sources)" in rendered
    assert "Max verified age per calendar year" in rendered
    # The label sits inside the topic block, not before it.
    assert rendered.index("### oldest-human bounds") < rendered.index(
        "Derived analysis (arithmetic from quoted sources)"
    )


def test_render_omits_derivation_label_for_plain_findings() -> None:
    """A finding with no derivation renders exactly as before — no stray label."""
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

    assert "Derived analysis" not in rendered


def test_render_with_derivation_is_deterministic() -> None:
    """Determinism guard extended to the derivation label: repeated renders of the
    same findings produce byte-identical output."""
    findings = [
        Finding(
            claim="Derived bound table.",
            source_url="https://example.com/a",
            quote="Row values: 10, 11, 12.",
            date="2026-07-01",
            retrieved_how="fetch",
            topic="alpha",
            derivation="10 + 1 = 11; 11 + 1 = 12.",
        ),
        Finding(
            claim="A plain source claim.",
            source_url="https://example.com/b",
            quote="Report published.",
            date="2026-07-02",
            retrieved_how="fetch",
            topic="beta",
        ),
    ]

    assert render_findings(findings, ["lead"]) == render_findings(findings, ["lead"])
