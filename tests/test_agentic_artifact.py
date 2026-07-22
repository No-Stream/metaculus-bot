from __future__ import annotations

from typing import Literal

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
            # W4: a fetched-tier discrepancy keeps the supersede banner; this test
            # exercises the hoist-above-topics behavior on the supersede block.
            verification_tier="fetched",
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


def _discrepancy(
    tier: Literal["fetched", "snippet"] | None, *, claim: str, url: str = "https://example.com/correction"
) -> Finding:
    return Finding(
        claim=claim,
        source_url=url,
        quote="The source states the corrected figure.",
        date="2026-07-10",
        retrieved_how="fetch",
        topic="labor",
        discrepancy=True,
        verification_tier=tier,
    )


def test_fetched_discrepancy_keeps_supersede_block() -> None:
    """W4: a discrepancy stamped ``fetched`` (primary source pulled) renders under
    the supersede header with its supersede banner — full authority."""
    rendered = render_findings([_discrepancy("fetched", claim="Briefing says 7%; source says 5.2%.")], [])

    assert "### ⚠ Corrections to the briefing" in rendered
    assert "### ⚠ Possible corrections (snippet-sourced — recheck advised)" not in rendered
    assert (
        "The sourced findings below contradict the research briefing and supersede the corresponding briefing content."
        in rendered
    )
    assert "Briefing says 7%; source says 5.2%." in rendered


def test_snippet_discrepancy_renders_demoted_block_not_supersede() -> None:
    """W4 core fix (131.3): a discrepancy stamped ``snippet`` renders under the
    demoted header with a banner stating it does NOT supersede the briefing — the
    supersede header/banner must be absent."""
    rendered = render_findings([_discrepancy("snippet", claim="Briefing says 133; a snippet says 131.3.")], [])

    assert "### ⚠ Possible corrections (snippet-sourced — recheck advised)" in rendered
    assert "### ⚠ Corrections to the briefing" not in rendered
    assert "do NOT supersede the briefing" in rendered
    assert "treat them as leads that contradict it" in rendered
    assert "Briefing says 133; a snippet says 131.3." in rendered
    # The supersede banner text must NOT appear anywhere.
    assert "supersede the corresponding briefing content" not in rendered


def test_untiered_discrepancy_is_demoted() -> None:
    """A discrepancy with no tier (None — its URL was never retrieved through a
    tool, or only via a failed fetch) is demoted conservatively, same as snippet."""
    rendered = render_findings([_discrepancy(None, claim="Briefing says X; an untiered source says Y.")], [])

    assert "### ⚠ Possible corrections (snippet-sourced — recheck advised)" in rendered
    assert "### ⚠ Corrections to the briefing" not in rendered


def test_mixed_discrepancies_render_both_blocks_supersede_first() -> None:
    """W4: a run with both a fetched and a snippet discrepancy renders BOTH
    correction blocks, supersede first, then demoted, both above the topic
    blocks; non-discrepancy findings stay in their topic block."""
    findings = [
        _discrepancy("snippet", claim="Snippet-only correction.", url="https://example.com/snippet"),
        _discrepancy("fetched", claim="Fetched correction.", url="https://example.com/fetched"),
        Finding(
            claim="The ministry published a bulletin.",
            source_url="https://example.com/bulletin",
            quote="Bulletin issued July 10.",
            date="2026-07-10",
            retrieved_how="fetch",
            topic="labor",
            verification_tier="fetched",
        ),
    ]

    rendered = render_findings(findings, [])

    supersede_at = rendered.index("### ⚠ Corrections to the briefing")
    demoted_at = rendered.index("### ⚠ Possible corrections (snippet-sourced — recheck advised)")
    topic_at = rendered.index("### labor")
    assert supersede_at < demoted_at < topic_at
    assert "Fetched correction." in rendered
    assert "Snippet-only correction." in rendered
    assert "The ministry published a bulletin." in rendered


def test_verification_tier_renders_compactly_on_topic_findings() -> None:
    """W4: a non-discrepancy finding carrying a tier shows it inline alongside
    retrieved_how, so the panel sees each finding's retrieval quality."""
    findings = [
        Finding(
            claim="The agency reported the figure.",
            source_url="https://example.com/fetched",
            quote="The figure is 4.1 percent.",
            date="2026-07-01",
            retrieved_how="fetch",
            topic="labor",
            verification_tier="fetched",
        ),
        Finding(
            claim="A secondary source echoed a figure.",
            source_url="https://example.com/snippet",
            quote="Reported at about 4 percent.",
            date="2026-07-02",
            retrieved_how="search_web",
            topic="labor",
            verification_tier="snippet",
        ),
    ]

    rendered = render_findings(findings, [])

    assert "Retrieved how: fetch [verification: fetched]" in rendered
    assert "Retrieved how: search_web [verification: snippet]" in rendered


def test_untiered_topic_finding_omits_verification_token() -> None:
    """A finding with no tier (briefing-grounded, never tool-retrieved) renders
    the plain retrieved-how line — no stray verification token."""
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

    assert "Retrieved how: fetch" in rendered
    assert "[verification:" not in rendered


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
