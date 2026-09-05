"""The research-side prompts, the clauses gated on what the research actually carries, and the
benchmarking leakage guards.

Critical for the benchmarking carve-out: the higher-level gap-fill tests mock ``_run_analyzer``
entirely, so without these the benchmarking warning could be deleted from a prompt and no test
would catch it, and backtest scores would silently get polluted with prediction-market data.
"""

from datetime import datetime
from typing import ClassVar

import pytest

from metaculus_bot.prompts import (
    _AUTO_ANNOTATED_CITATION_CLAUSE,
    _OUTSIDE_VENUE_MARKET_ODDS_BULLET,
    _SOURCE_TIER_TAG_INSTRUCTION,
    MARKET_SNAPSHOT_SECTION_HEADER,
    TS_ANCHOR_SECTION_HEADER,
    binary_prompt,
    gap_fill_analyzer_prompt,
    gap_fill_search_prompt,
    multiple_choice_prompt,
    numeric_prompt,
    stacking_binary_prompt,
    stacking_multiple_choice_prompt,
    stacking_numeric_prompt,
    web_research_prompt,
)
from metaculus_bot.research.section_format import PROVIDER_SECTION_HEADERS
from tests.prompt_builders import (
    _RESEARCH_WITH_MARKETS,
    _binary_q,
    _flat,
    _mc_q,
    _numeric_q,
    _summarizer_prompt,
)


class TestGapFillAnalyzerPrompt:
    """Covers the benchmarking carve-out in the analyzer prompt."""

    def test_benchmarking_true_includes_warning_and_bans_prediction_markets(self) -> None:
        """When benchmarking, the prompt must BOTH mark itself as benchmarking AND ban prediction markets."""
        result = gap_fill_analyzer_prompt(
            question_text="Will X happen?",
            resolution_criteria="resolves YES if X",
            fine_print="see bls.gov",
            first_pass_research="some research",
            is_benchmarking=True,
        )

        lowered = result.lower()
        # The benchmarking marker must be present so downstream reviewers can grep for it.
        assert "benchmarking run" in lowered
        # The carve-out must explicitly instruct the model to avoid prediction-market data.
        # We verify both the "DO NOT" directive AND that it's attached to "prediction market".
        assert "do not flag prediction-market" in lowered or "do not request searches for prediction markets" in lowered
        # Data-leakage framing must be present so the model understands *why*.
        assert "data leakage" in lowered

    def test_benchmarking_false_has_no_warning(self) -> None:
        """Without benchmarking, there must be no benchmarking marker or prediction-market ban."""
        result = gap_fill_analyzer_prompt(
            question_text="Will X happen?",
            resolution_criteria="resolves YES if X",
            fine_print="see bls.gov",
            first_pass_research="some research",
            is_benchmarking=False,
        )

        lowered = result.lower()
        assert "benchmarking run" not in lowered
        # Neither the DO-NOT directive nor the leakage framing should leak in.
        assert "data leakage" not in lowered
        assert "prediction-market" not in lowered
        assert "prediction markets" not in lowered

    def test_threads_question_and_resolution_criteria(self) -> None:
        """Sanity: the question text, resolution criteria, and fine print all appear in the prompt."""
        result = gap_fill_analyzer_prompt(
            question_text="Will BTC exceed $200k by 2027?",
            resolution_criteria="Resolves YES if BTC USD > 200k on Jan 1 2027.",
            fine_print="Data source: Coinbase Pro BTC-USD.",
            first_pass_research="first-pass text",
            is_benchmarking=False,
        )

        assert "Will BTC exceed $200k by 2027?" in result
        assert "Resolves YES if BTC USD > 200k on Jan 1 2027." in result
        assert "Data source: Coinbase Pro BTC-USD." in result
        assert "first-pass text" in result

    def test_none_resolution_and_fine_print_render_as_placeholder(self) -> None:
        """Optional fields default to "(none provided)" rather than crashing."""
        result = gap_fill_analyzer_prompt(
            question_text="Question?",
            resolution_criteria=None,
            fine_print=None,
            first_pass_research="x",
            is_benchmarking=False,
        )

        assert "(none provided)" in result


class TestGapFillSearchPrompt:
    """Covers the benchmarking carve-out in the per-gap search prompt."""

    def test_benchmarking_true_includes_warning_and_bans_prediction_markets(self) -> None:
        """The per-gap search prompt also needs the benchmarking carve-out."""
        result = gap_fill_search_prompt(
            gap="What was the 2025 GDP?",
            search_query="US 2025 GDP BEA",
            question_text="Will GDP exceed 30T?",
            is_benchmarking=True,
        )

        lowered = result.lower()
        assert "benchmarking run" in lowered
        # The search-side carve-out text uses slightly different wording than the analyzer.
        assert "do not search for or include" in lowered
        assert "prediction-market" in lowered or "prediction market" in lowered
        assert "data leakage" in lowered

    def test_benchmarking_false_has_no_warning(self) -> None:
        """Without benchmarking, no benchmarking marker or prediction-market ban."""
        result = gap_fill_search_prompt(
            gap="What was the 2025 GDP?",
            search_query="US 2025 GDP BEA",
            question_text="Will GDP exceed 30T?",
            is_benchmarking=False,
        )

        lowered = result.lower()
        assert "benchmarking run" not in lowered
        assert "data leakage" not in lowered
        assert "prediction-market" not in lowered
        assert "prediction market" not in lowered

    def test_threads_gap_and_question(self) -> None:
        """Sanity: gap text, search query, and question text all appear in the prompt."""
        result = gap_fill_search_prompt(
            gap="Was the treaty signed?",
            search_query="treaty signing Sept 2026",
            question_text="Will the treaty be in force by 2027?",
            is_benchmarking=False,
        )

        assert "Was the treaty signed?" in result
        assert "treaty signing Sept 2026" in result
        assert "Will the treaty be in force by 2027?" in result


class TestTsAnchorClause:
    """The numeric prompt surfaces the time-series-anchor pointer ONLY when the
    research actually carries the anchor section header. Binary/MC never mention
    it (the anchor routes to numeric questions only in v1).

    The clause is deliberately NEUTRAL: it describes precisely what the section
    contains and leaves the forecaster to decide how to use it. The old directive
    framing ("CALIBRATED REFERENCE EVIDENCE", "keep your interval close", "do NOT
    widen") was removed 2026-07-18 — the operator chose to state the facts and
    trust the forecasters rather than prescribe usage on an untested feature."""

    _MARKER = "## Time Series Anchor"
    # Directive phrases that must NOT reappear in the neutral clause.
    _BANNED = ("CALIBRATED", "keep your interval", "do not widen", "sharpen your distribution")

    def _assert_no_directives(self, text: str) -> None:
        lowered = text.lower()
        for phrase in self._BANNED:
            assert phrase.lower() not in lowered, f"directive phrase leaked back in: {phrase!r}"

    def test_clause_present_when_section_in_research(self) -> None:
        research = f"Some news.\n\n{TS_ANCHOR_SECTION_HEADER}\n**DGS10** — latest 4.20\n- band ..."
        result = numeric_prompt(_numeric_q(), research=research, lower_bound_message="lbm", upper_bound_message="ubm")
        # Neutral description present: it points at the section and says what the band IS,
        # including the independent-window caveat, without telling the model how to weigh it.
        assert self._MARKER in result
        lowered = result.lower()
        assert "empirical distribution of the series' own past changes" in lowered
        assert "independent" in lowered
        self._assert_no_directives(result)

    def test_clause_absent_when_section_missing(self) -> None:
        result = numeric_prompt(
            _numeric_q(), research="Just some news, no anchor.", lower_bound_message="lbm", upper_bound_message="ubm"
        )
        # No anchor section -> neither the pointer nor any directive language appears.
        assert "empirical distribution of the series' own past changes" not in result.lower()
        self._assert_no_directives(result)

    def test_clause_not_in_binary_or_mc_even_with_section(self) -> None:
        research = f"{TS_ANCHOR_SECTION_HEADER}\n**DGS10** — latest 4.20"
        assert (
            "empirical distribution of the series' own past changes"
            not in binary_prompt(_binary_q(), research=research).lower()
        )
        assert (
            "empirical distribution of the series' own past changes"
            not in multiple_choice_prompt(_mc_q(), research=research).lower()
        )

    def test_stacking_binary_injects_window(self) -> None:
        q = _binary_q(
            open_time=datetime(2026, 2, 2),
            resolve_time=datetime(2028, 2, 2),
        )
        result = stacking_binary_prompt(q, research="r", base_predictions=["a1", "a2"])
        assert "2026-02-02" in result
        assert "2028-02-02" in result
        assert "Forecasting window" in result

    def test_stacking_multiple_choice_injects_window(self) -> None:
        q = _mc_q(
            open_time=datetime(2026, 4, 4),
            resolve_time=datetime(2028, 4, 4),
        )
        result = stacking_multiple_choice_prompt(q, research="r", base_predictions=["a1", "a2"])
        assert "2026-04-04" in result
        assert "2028-04-04" in result
        assert "Forecasting window" in result

    def test_stacking_numeric_injects_window(self) -> None:
        q = _numeric_q(
            open_time=datetime(2026, 5, 5),
            resolve_time=datetime(2028, 5, 5),
        )
        result = stacking_numeric_prompt(
            q,
            research="r",
            base_predictions=["a1", "a2"],
            lower_bound_message="lbm",
            upper_bound_message="ubm",
        )
        assert "2026-05-05" in result
        assert "2028-05-05" in result
        assert "Forecasting window" in result

    def test_stacking_binary_asserts_on_missing_timestamps(self) -> None:
        q = _binary_q()
        q.open_time = None
        with pytest.raises(AssertionError):
            stacking_binary_prompt(q, research="r", base_predictions=["a"])


class TestWebResearchPromptPrimarySources:
    """The first-pass web-research prompt must steer the model toward primary
    sources (government stats, SEC filings, official docs, scientific
    registries, central banks) — the pattern observed in Preseen-Atlas's
    winning spring-AIB-2026 comments. Matches the primary-source hints
    already present in targeted_search_prompt and gap_fill_search_prompt."""

    # Domain examples we expect to see called out somewhere in the block.
    # We assert ≥3 of these 4 show up so the list can evolve without
    # breaking the test on single-domain renames.
    _EXAMPLE_DOMAINS = (".gov", "sec.gov", "docs.", "who.int")

    def _assert_primary_sources_block_present(self, prompt: str) -> None:
        upper = prompt.upper()
        assert "PRIMARY SOURCES" in upper, "expected a PRIMARY SOURCES block header"
        hits = sum(1 for d in self._EXAMPLE_DOMAINS if d in prompt)
        assert hits >= 3, f"expected ≥3 of {self._EXAMPLE_DOMAINS} in prompt; got {hits}"

    def test_non_benchmarking_contains_primary_sources_block(self) -> None:
        result = web_research_prompt("Will X happen?", is_benchmarking=False)
        self._assert_primary_sources_block_present(result)

    def test_benchmarking_also_contains_primary_sources_block(self) -> None:
        """The primary-source steer is orthogonal to the benchmarking carve-out
        — it must apply during backtests too (Atlas-style sourcing helps
        regardless of whether prediction markets are banned)."""
        result = web_research_prompt("Will X happen?", is_benchmarking=True)
        self._assert_primary_sources_block_present(result)

    def test_benchmarking_warning_still_present_when_benchmarking(self) -> None:
        """Regression: adding the primary-sources block must not displace the
        benchmarking warning."""
        result = web_research_prompt("Will X happen?", is_benchmarking=True)
        lowered = result.lower()
        assert "benchmarking run" in lowered
        assert "data leakage" in lowered

    def test_market_odds_bullet_only_when_not_benchmarking(self) -> None:
        """Regression: the FOCUS AREAS market-odds bullet must still appear only when
        is_benchmarking=False (backtests must not see market data at all)."""
        non_bench = web_research_prompt("Q?", is_benchmarking=False)
        bench = web_research_prompt("Q?", is_benchmarking=True)

        assert _OUTSIDE_VENUE_MARKET_ODDS_BULLET in non_bench
        assert _OUTSIDE_VENUE_MARKET_ODDS_BULLET not in bench
        assert "Market-implied or crowd odds" not in bench

    def test_market_odds_bullet_points_away_from_the_covered_venues(self) -> None:
        """Across 42 ranked-era bundles the bullet's covered-venue half produced one
        content-redundant retrieval and three stale prices that contradicted correct
        live snapshot rows, while every realized instance of decisive market evidence
        came from a venue OUTSIDE the four the live snapshot covers (GJO q44869, CME
        FedWatch q45401, Metaculus q20683). So the bullet is narrowed, not removed:
        the outside venues stay named, the four covered ones are ruled out, and the
        price must carry an observation date."""
        prompt = web_research_prompt("Q?", is_benchmarking=False)
        collapsed = " ".join(prompt.split())

        for outside_venue in ("Metaculus", "Good Judgment Open", "CME FedWatch", "bookmakers"):
            assert outside_venue in collapsed
        assert "Do NOT report Polymarket/Kalshi/Manifold/PredictIt prices from search results" in collapsed
        # The item-2 vintage rule landing on this surface too.
        assert "always name the market and the date you observed the price" in collapsed
        assert "usually days stale" in collapsed

    def test_auto_annotated_style_bans_model_authored_citation_indices(self) -> None:
        """Half of all archived gemini sections (173 of 323) carry the model's own
        hierarchical [1.2.3] indices alongside the [N] markers our formatter splices
        from real grounding metadata, so a forecaster cannot tell which brackets are
        checkable. The formatter strips them; this tells the model not to write them.
        Gemini-only: the markdown branch (native search) is untouched."""
        auto = web_research_prompt("Q?", citation_style="auto_annotated")
        markdown = web_research_prompt("Q?", citation_style="markdown")

        lowered = " ".join(auto.lower().split())
        assert "do not write your own citation markers" in lowered
        assert "[1.2.3]" in auto
        assert "[1.2.3]" not in markdown
        assert "do not write your own citation markers" not in " ".join(markdown.lower().split())

    def test_citation_index_ban_carves_out_the_source_tier_tags_it_ships_with(self) -> None:
        """The ban and the SOURCE TIER TAGS block ride the SAME rendered prompt, 26 lines
        apart, and the tier block orders exactly what the ban's second half appears to
        forbid: a bracketed, model-authored source annotation. A literal reader that
        over-complies stops tagging, which costs the forecaster prompts the tier signal
        they weight on and leaves gemini_attribution's unsupported-attribution check
        nothing to check — the direction nothing downstream guards, unlike the dotted
        indices _strip_model_citation_indices removes. So the carve-out ships in the same
        clause, and it says "still applies" rather than "required", because the tier
        block's own closing line licenses leaving an unclear claim untagged."""
        auto = web_research_prompt("Q?", citation_style="auto_annotated")
        collapsed = " ".join(auto.split())

        assert "do NOT write your own citation markers" in collapsed
        assert "This bans invented CITATION indices only" in collapsed
        assert "the SOURCE TIER TAGS instruction below still applies" in collapsed
        # The instruction the carve-out names, in the same prompt and BELOW the ban.
        assert collapsed.index("SOURCE TIER TAGS instruction below") < collapsed.index(
            "SOURCE TIER TAGS: annotate each factual claim"
        )
        assert '"[A: official]"' in collapsed
        # Not phrased as a requirement, which would contradict the tier block's softener.
        assert "requir" not in _AUTO_ANNOTATED_CITATION_CLAUSE.lower()
        assert "leave a claim untagged if unsure" in collapsed

    def test_vintage_clause_present_for_both_citation_styles(self) -> None:
        """qid 44872: gemini searched correctly, Google attached no grounding, and it
        answered from memory of 2021/2022 OCEARCH press releases restamped as 2026
        plans. The prompt had "say so explicitly" and "DO NOT hallucinate sources"
        and no date discipline at all, so nothing in it made an undated recollection
        look wrong. Shared by both consumers (native search + gemini) on purpose."""
        for citation_style in ("markdown", "auto_annotated"):
            for is_benchmarking in (False, True):
                result = web_research_prompt(
                    "Will X happen?",
                    citation_style=citation_style,
                    is_benchmarking=is_benchmarking,
                )
                lowered = " ".join(result.lower().split())
                assert "publication date" in lowered
                assert "state when and where it was announced" in lowered
                assert "never present an undated recollection as a current fact" in lowered

    def test_reference_class_frequency_instruction_present(self) -> None:
        """The prompt must ask for historical frequencies (with source and
        denominator) on reference-class questions — prioritizing niche,
        regional, or conditional rates and skipping common knowledge."""
        for is_benchmarking in (False, True):
            result = web_research_prompt("Will X happen?", is_benchmarking=is_benchmarking)
            assert "reference-class reasoning" in result
            assert "historical frequency with its source and denominator" in result
            assert "niche, regional, or conditional" in result
            assert "common knowledge" in result


class TestPredictionMarketFraming:
    """The forecaster prompts must frame prediction markets as STRONG EVIDENCE
    to weight heavily — not the old "not beholden" footnote — with a precise
    conditional adjustment: anchor when the market's resolution criteria AND
    date match the question, discount proportionally to any specific mismatch,
    and extrapolate across a date-only mismatch.

    The PM clause must NOT carry a "you may deviate from a market" carve-out:
    that sentence undercut the strong-evidence framing. The general principle
    that a forecaster may supplement the research with its own training
    knowledge is a SEPARATE, prompt-wide directive — not a market-specific one.

    Gate: the whole clause renders ONLY when the research carries the rendered
    ``## Prediction Market Snapshot`` section (``MARKET_SNAPSHOT_SECTION_HEADER``), the
    same way the numeric prompt gates its TS-anchor clause. Prod-neutral: the header is
    emitted whenever the provider rendered anything, including the deliberate-empty "no
    relevant market" sentence, and it is absent only when the provider returned ``""``
    (benchmarking, flag off, soft-fail) — exactly the prompts where the market policy
    had nothing to bear on. That also makes the leakage story simpler than it was: a
    benchmarking prompt no longer carries three paragraphs about markets it cannot see.
    The mode-dependent leakage guard on the RESEARCH side still lives on
    ``web_research_prompt`` (see ``test_market_ask_present_non_benchmarking_absent_benchmarking``).

    Notation vs policy: the rendered table's own legend (``MARKET_SIGNAL_LEGEND``) defines the
    relation tiers, the evidential order, RESOLVED, the ``↳`` rows and ``[remaining N]``; the
    prompt keeps only the three READING rules the legend does not carry.
    """

    def _assert_strong_evidence_framing(self, prompt: str) -> None:
        # Collapse whitespace so assertions don't depend on where clean_indents wraps lines.
        lowered = " ".join(prompt.lower().split())
        assert "strong evidence" in lowered
        assert "weight them heavily" in lowered
        # The conditional adjustment: match anchors, mismatch discounts.
        assert "resolution date" in lowered
        assert "match" in lowered
        assert "discount" in lowered
        # Date-only mismatch must trigger an explicit extrapolation, not a vague haircut.
        assert "extrapolate" in lowered
        assert "constant-hazard" in lowered or "base-rate-over-time" in lowered
        assert "show the arithmetic" in lowered
        # Reading rule 1: an other-cut market is the same quantity at another date/threshold/source,
        # so it is something to extrapolate from, not to haircut.
        assert "`same_quantity_other_cut`" in lowered
        assert "extrapolate from it rather than discount it vaguely" in lowered
        # Reading rule 2: which label wins when the two axes disagree. A tight relation on a THIN
        # market is the shape that cost q45189: all three forecasters imported a thin single-strike
        # price at full weight. The rule carries its reason (a thin price is noisy however tight its
        # relation) and is directional — widen around the implied value rather than transplant it.
        assert "the liquidity warning governs" in lowered
        assert "a thin price is noisy however tight its relation" in lowered
        assert "widen around its implied value rather than transplant its price" in lowered
        # Reading rule 3: a family of `↳` rows is a distribution over the market's own question, so
        # reading one bracket as an equality constraint on a tail is a category error — the other
        # half of q45189 (all three cut the resolving bucket below their own prior that way).
        assert "is a distribution over that market's own question" in lowered
        assert "read the whole ladder" in lowered
        assert "never treat one outcome's price as an equality constraint" in lowered
        assert "cut the resolving bucket below the forecaster's own prior" in lowered
        # NOTATION is the legend's job, stated beside the table; the prompt must not re-teach it.
        # These are the phrases the pre-2026-09 clause carried that duplicated MARKET_SIGNAL_LEGEND.
        assert "weight each market/crowd signal by its stated liquidity/participation label" not in lowered
        assert "listed in order of evidential value" not in lowered
        assert "realized outcome rather than a forecast" not in lowered
        assert "a market with several outcomes has no single price" not in lowered
        assert "[remaining n]" not in lowered
        assert "inside a counted group with its summed price" not in lowered
        # The old "not beholden" footnote must be gone.
        assert "not beholden" not in lowered
        # The mis-scoped "you may deviate from a market" carve-out must NOT be present —
        # it undercut the strong-evidence framing. The general expertise principle is
        # asserted separately below.
        assert "deviate from a market" not in lowered

    def _assert_market_clause_absent(self, prompt: str) -> None:
        lowered = " ".join(prompt.lower().split())
        assert "prediction markets are strong evidence" not in lowered
        assert "weight them heavily" not in lowered
        assert "equality constraint" not in lowered
        assert "liquidity warning" not in lowered

    def _assert_general_expertise_principle(self, prompt: str) -> None:
        """The prompt-wide directive that a forecaster may draw on its own training
        knowledge to fill research gaps — distinct from any market-specific clause."""
        lowered = " ".join(prompt.lower().split())
        assert "use your own expertise and knowledge, not only the provided research" in lowered
        assert "you are not required to ground every claim in the research" in lowered

    def test_binary_prompt_frames_markets_as_strong_evidence(self) -> None:
        self._assert_strong_evidence_framing(binary_prompt(_binary_q(), research=_RESEARCH_WITH_MARKETS))

    def test_multiple_choice_prompt_frames_markets_as_strong_evidence(self) -> None:
        self._assert_strong_evidence_framing(multiple_choice_prompt(_mc_q(), research=_RESEARCH_WITH_MARKETS))

    def test_numeric_prompt_frames_markets_as_strong_evidence(self) -> None:
        result = numeric_prompt(
            _numeric_q(), research=_RESEARCH_WITH_MARKETS, lower_bound_message="lbm", upper_bound_message="ubm"
        )
        self._assert_strong_evidence_framing(result)

    def test_market_clause_absent_when_research_carries_no_snapshot(self) -> None:
        """No snapshot section, no market policy: the clause is ~1.5k chars of instruction about
        a table the forecaster does not have, and before the gate every benchmarking, flag-off and
        soft-failed prompt paid for it."""
        self._assert_market_clause_absent(binary_prompt(_binary_q(), research="Just some news."))
        self._assert_market_clause_absent(multiple_choice_prompt(_mc_q(), research="Just some news."))
        self._assert_market_clause_absent(
            numeric_prompt(
                _numeric_q(), research="Just some news.", lower_bound_message="lbm", upper_bound_message="ubm"
            )
        )

    def test_market_clause_not_triggered_by_another_providers_header(self) -> None:
        research = f"{TS_ANCHOR_SECTION_HEADER}\n**DGS10** — latest 4.20"
        self._assert_market_clause_absent(
            numeric_prompt(_numeric_q(), research=research, lower_bound_message="lbm", upper_bound_message="ubm")
        )

    def test_gate_header_is_the_one_the_provider_actually_renders(self) -> None:
        """Pinned against the provider-to-header map rather than a copied string: if the market
        provider's section header ever changes, the gate has to move with it or every prod prompt
        silently loses the clause."""
        assert PROVIDER_SECTION_HEADERS["prediction_market"] is MARKET_SNAPSHOT_SECTION_HEADER
        assert MARKET_SNAPSHOT_SECTION_HEADER == "## Prediction Market Snapshot"

    def test_binary_prompt_carries_general_expertise_principle(self) -> None:
        self._assert_general_expertise_principle(binary_prompt(_binary_q(), research="r"))

    def test_multiple_choice_prompt_carries_general_expertise_principle(self) -> None:
        self._assert_general_expertise_principle(multiple_choice_prompt(_mc_q(), research="r"))

    def test_numeric_prompt_carries_general_expertise_principle(self) -> None:
        result = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        self._assert_general_expertise_principle(result)

    def test_market_ask_present_non_benchmarking_absent_benchmarking(self) -> None:
        """Leakage guard: the market/crowd-odds ask in the research prompt is present
        only when NOT benchmarking. This is the mode-dependent surface — the
        forecaster prompts above are mode-agnostic because the provider data is
        suppressed upstream during backtests. Since the bullet was narrowed (it now
        names venues on both sides of the line), a benchmarking prompt must name
        none of them."""
        non_bench = web_research_prompt("Will X happen?", is_benchmarking=False)
        bench = web_research_prompt("Will X happen?", is_benchmarking=True)

        assert "crowd odds" in non_bench
        assert "crowd odds" not in bench
        for venue in ("Polymarket", "Kalshi", "Manifold", "PredictIt", "Metaculus", "CME FedWatch"):
            assert venue in non_bench, f"expected {venue} named in the narrowed bullet"
            assert venue not in bench, f"{venue} leaked into a benchmarking prompt"


class TestPresentTenseInstrumentGaps:
    """The Nebraska/Texas natural experiment (44554 miss vs 44556 control — same
    template, same day, same roster): Texas's gap-fill asked what the polling
    tracker reads NOW and got a live value; Nebraska asked what the tracker would
    display on the resolution date and got "August 31, 2026 has not occurred".
    Scoped to questions resolving off a live data source."""

    def _analyzer(self) -> str:
        return gap_fill_analyzer_prompt(
            "Will the challenger lead the polling average on 2026-08-31?",
            "Resolves YES if the tracker's average shows the challenger ahead.",
            "The tracker is RaceToTheWH.",
            "First pass: no current average retrieved.",
            is_benchmarking=False,
        )

    def test_analyzer_requires_gaps_answerable_from_todays_sources(self) -> None:
        lowered = " ".join(self._analyzer().lower().split())
        assert "answerable from sources that exist today" in lowered
        # The live-data-source scope, named by instrument kind.
        assert "tracker, index" in lowered
        assert "dashboard" in lowered

    def test_analyzer_requires_one_present_tense_current_value_gap(self) -> None:
        lowered = " ".join(self._analyzer().lower().split())
        assert "at least one gap must ask what that source reads now" in lowered
        assert "present tense" in lowered

    def test_analyzer_forbids_future_dated_gaps(self) -> None:
        lowered = " ".join(self._analyzer().lower().split())
        assert "never phrase a gap as that source's value on the resolution date" in lowered
        assert "rewrite it as the present-tense observable or drop it" in lowered

    def test_analyzer_does_not_spend_a_slot_on_a_reading_the_briefing_already_dates(self) -> None:
        """The present-tense rule says what the resolver has to come back WITH, not that the
        slot must be spent regardless. On a question whose first pass already reports the
        source's current dated reading, re-asking buys a value the briefing holds, and this
        pass is roughly 44% of research spend. The carve-out also keeps the block from
        fighting the prompt's own "DO NOT invent gaps" discipline."""
        lowered = " ".join(self._analyzer().lower().split())
        assert "already states the source's current reading with its as-of date counts as answered" in lowered
        assert "spend the slot on something the briefing lacks" in lowered
        # The two conditions that still earn a verify-style gap, in one clause.
        assert "re-ask only if the stated reading is undated or older than the source's own update cadence" in lowered

    def test_analyzer_states_the_present_tense_mandate_with_its_reason(self) -> None:
        lowered = " ".join(self._analyzer().lower().split())
        assert "the current reading is the single fact that most often decides these questions" in lowered


class TestGapFillAnalyzerSlotDiscipline:
    """The analyzer fills every slot whatever it is told: 55-77% of archived records sit at
    the cap and only 8 of 308 returned one or two gaps, so the old "Most questions have 0-2
    real gaps; a few have 3-5" was behaviourally dead and "3-5" was stale against
    GAP_FILL_MAX_GAPS = 4. What earns its place is the discipline with its reason (each gap
    is a paid search) and the ordering contract the cap relies on (order is the ranking, no
    rank fields). Receipt: scratch/prompt_bloat_audit_2026-09-02.md section 8."""

    def _analyzer(self) -> str:
        return gap_fill_analyzer_prompt(
            "Will X happen?",
            "Resolves YES if X.",
            "fp",
            "First pass.",
            is_benchmarking=False,
            max_gaps=4,
        )

    def test_gap_counts_are_gone_and_the_discipline_keeps_its_reason(self) -> None:
        flat = " ".join(self._analyzer().lower().split())
        assert "most questions have 0-2" not in flat
        assert "a few have 3-5" not in flat
        assert "be thorough but selective" not in flat
        assert "do not invent gaps for completeness" in flat
        assert "each gap is a paid search" in flat

    def test_ordering_contract_is_two_sentences(self) -> None:
        flat = " ".join(self._analyzer().lower().split())
        assert "order the gaps most forecast-moving first" in flat
        assert "the list order is the ranking" in flat
        assert "do not add rank fields or scores" in flat
        assert "compare the candidate gaps against each other" not in flat
        assert "order the gaps by decision-relevance" not in flat


class TestLastRealApplicationClause:
    """qid 45215: the question turned on how an electoral threshold cashed out at the
    last real election, a training-data fact no research bundle held and no gap asked
    for. Analyzer-only — it directs a search slot, which a forecaster prompt cannot.

    It shipped as a standalone block whose "one gap MUST ask" mandate pre-committed a paid
    slot; since the analyzer fills every slot regardless, a mandate does not add spend, it
    DISPLACES other gaps, and a second mandate beside ANSWERABLE NOW's pre-committed half
    the slots on a question with both a live instrument and an institutional rule. The gap
    type is right, so it now lives as a candidate clause inside gap type 6 (base rate /
    reference class), where the analyzer weighs it against the other gap types."""

    def _analyzer(self) -> str:
        return gap_fill_analyzer_prompt(
            "Will party X clear the electoral threshold?",
            "Resolves YES if the commission seats party X.",
            "Threshold is 5% nationally.",
            "First pass: polling summary only.",
            is_benchmarking=False,
        )

    def test_last_real_application_is_a_candidate_inside_gap_type_six(self) -> None:
        flat = _flat(self._analyzer())
        assert "how that rule actually applied at its most recent real application" in flat
        assert "as a realized count or outcome" in flat
        assert "an electoral threshold, a quota, an allocation formula, a cut-off score" in flat
        # Inside type 6, between the base-rate item and the expert-opinion item.
        type_six_at = flat.index("6. missing base rate / reference class")
        type_seven_at = flat.index("7. missing expert opinion")
        assert type_six_at < flat.index("most recent real application") < type_seven_at

    def test_clause_separates_the_institution_rule_from_the_question_threshold(self) -> None:
        """The gap is about how the BODY applies its own rule, not a restatement of the
        question's resolution threshold, which the analyzer already has in front of it."""
        flat = _flat(self._analyzer())
        assert "a different fact from the question's own resolution threshold" in flat

    def test_the_clause_keeps_its_scope_limiter(self) -> None:
        """The condition is what stops the clause fighting the same prompt's "DO NOT invent
        gaps" discipline: it earns a paid slot only where a discontinuous institutional rule is
        actually in play. Without this pin a reword could keep the examples and the realized-count
        wording, drop the condition, and stay green while the clause started firing on every
        question with a number in it."""
        flat = _flat(self._analyzer())
        assert "where the question resolves through an institutional rule" in flat

    def test_the_standalone_mandate_is_gone(self) -> None:
        prompt = self._analyzer()
        assert "LAST REAL APPLICATION OF THE RULE" not in prompt
        flat = " ".join(prompt.lower().split())
        assert "one gap must ask how that rule" not in flat
        # ANSWERABLE NOW keeps the one slot mandate the prompt still carries.
        assert flat.count("must ask") == 1

    def test_forecaster_prompts_do_not_carry_the_gap_clause(self) -> None:
        """Scope guard: this one is an instruction to the gap AUDITOR."""
        for prompt in (
            binary_prompt(_binary_q(), research="r"),
            multiple_choice_prompt(_mc_q(), research="r"),
            numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm"),
        ):
            assert "most recent real application" not in prompt


class TestResearchPromptsCarryMcOptions:
    """Every research-side prompt that shows the question must show an MC ballot with it.

    On q44952 (World Yo-Yo champion, MC) no research stage ever saw the candidate list —
    the prompts carried only question_text — and AskNews returned zero mentions of the
    eventual winner even though the ballot named him. A searching model can only query
    names it has been shown, and a summarizer can only relevance-gate articles against
    candidates it knows exist.
    """

    _BALLOT: ClassVar[list[str]] = ["Mir Kim", "Hunter Feuerstein", "Other"]
    _LINE = "Options (in resolution order): Mir Kim | Hunter Feuerstein | Other"

    def test_web_research_prompt_names_the_ballot(self) -> None:
        assert self._LINE in web_research_prompt("Who wins?", options=self._BALLOT)

    def test_summarizer_prompt_names_the_ballot(self) -> None:
        assert self._LINE in _summarizer_prompt(options=self._BALLOT)

    def test_gap_fill_analyzer_prompt_names_the_ballot(self) -> None:
        prompt = gap_fill_analyzer_prompt(
            question_text="Who wins?",
            resolution_criteria="rc",
            fine_print="fp",
            first_pass_research="research",
            options=self._BALLOT,
        )
        assert self._LINE in prompt

    @pytest.mark.parametrize("options", [None, [], ()])
    def test_non_mc_questions_carry_no_options_line(self, options) -> None:
        # Binary/numeric questions have no ballot; an empty "Options" header would invite
        # the model to invent one.
        assert "Options (in resolution order)" not in web_research_prompt("Will X happen?", options=options)
        assert "Options (in resolution order)" not in _summarizer_prompt(options=options)


class TestSourceTierTagging:
    """Both TRADITIONAL research prompts (web research + AskNews summarizer)
    must instruct the model to tag factual claims with the A-D source-tier
    vocabulary — otherwise a C-tier aggregator claim arrives in the briefing
    looking identical to a B-tier wire fact and the forecaster prompts'
    provenance ladder has nothing left to weight."""

    def _assert_tier_tag_instruction(self, prompt: str) -> None:
        # Collapse whitespace so assertions don't depend on where clean_indents wraps lines.
        collapsed = " ".join(prompt.split())
        assert "SOURCE TIER TAGS" in collapsed
        # Inline tag examples using the shared vocabulary.
        for example in ('"[A: official]"', '"[B: Reuters]"', '"[C: aggregator]"', '"[D: social]"'):
            assert example in collapsed, f"missing tag example {example}"
        # The condensed A-D definitions mirror the forecaster ladder's vocabulary.
        lowered = collapsed.lower()
        assert "official / primary" in lowered
        assert "wire services and papers of record" in lowered
        assert "aggregators, advocacy or partisan outlets" in lowered
        assert "anonymous, social, rumor" in lowered
        # Tag only when clear; never drop a low-tier fact.
        assert "tag only when the tier is reasonably clear" in lowered
        assert "never discard a fact because its tier is low" in lowered

    def test_web_research_prompt_carries_tier_tag_instruction(self) -> None:
        self._assert_tier_tag_instruction(web_research_prompt("Will X happen?", is_benchmarking=False))

    def test_web_research_prompt_carries_tier_tag_instruction_when_benchmarking(self) -> None:
        """The tier-tag steer is orthogonal to the benchmarking carve-out."""
        self._assert_tier_tag_instruction(web_research_prompt("Will X happen?", is_benchmarking=True))

    def test_summarizer_prompt_carries_tier_tag_instruction(self) -> None:
        self._assert_tier_tag_instruction(_summarizer_prompt())

    def test_instruction_is_shared_constant_verbatim(self) -> None:
        """Both prompts must interpolate the SAME module-level constant — a
        drift between the two vocabularies would let the summarizer emit tags
        the forecaster ladder doesn't recognize."""
        collapsed_instruction = " ".join(_SOURCE_TIER_TAG_INSTRUCTION.split())
        for prompt in (
            web_research_prompt("Q?", is_benchmarking=False),
            _summarizer_prompt(),
        ):
            assert collapsed_instruction in " ".join(prompt.split())


class TestAskNewsSummarizerPrompt:
    """The summarizer prompt moved from ResearchOrchestrator._summarize_asknews
    to prompts.py (2026-07). These lock the load-bearing content that
    tests/test_research_orchestrator.py asserts through the orchestrator, plus
    argument threading now that the text is a standalone function."""

    def test_threads_all_arguments(self) -> None:
        prompt = _summarizer_prompt(
            question_text="Will the treaty be in force by 2027?",
            resolution_criteria="Resolves YES if in force.",
            fine_print="Per the depositary's records.",
            open_date="2026-01-02",
            research="ARTICLE BODY HERE",
        )
        assert "Will the treaty be in force by 2027?" in prompt
        assert "Resolves YES if in force." in prompt
        assert "Per the depositary's records." in prompt
        assert "opened on 2026-01-02" in " ".join(prompt.split())
        assert "<research>\nARTICLE BODY HERE\n</research>" in prompt

    def test_retains_window_stamping_and_no_forecast_rules(self) -> None:
        """Regression: the move + tier-tag insertion must not displace the
        summarizer's existing critical rules."""
        collapsed = " ".join(_summarizer_prompt().split())
        assert "Date every fact precisely" in collapsed
        # First occurrence carries the full tag; repeats use the short tag (display
        # compression only — the pre-window warning semantics must stay intact).
        assert "[PRE-WINDOW — occurred before question open, cannot itself satisfy the criteria]" in collapsed
        assert "FIRST time such a flag appears in the briefing, use the full tag" in collapsed
        assert 'for every subsequent occurrence use the short tag "[PRE-WINDOW]"' in collapsed
        assert "[SINGLE-SOURCE]" in collapsed
        assert "NEVER promote a single-source claim to a confirmed or factual statement" in collapsed
        assert "NEVER include your own forecast, probability estimate, or probability distribution" in collapsed

    def test_preserves_conditionality_rule(self) -> None:
        """BTF-2 lesson: research found a crucial hedge and synthesis dropped the
        conditionality. The summarizer must keep a condition attached to its
        claim, never flattening 'X if Y' into an unconditional 'X'."""
        collapsed = " ".join(_summarizer_prompt().split())
        assert "Preserve conditionality" in collapsed
        assert "keep the condition attached to the claim" in collapsed
        assert "never report a conditional statement as an unconditional one" in collapsed

    def test_supersession_and_deadline_arithmetic_rule(self) -> None:
        """2026-07-18 AskNews audit R2: newer facts govern, superseded ones are
        compressed, and deadline questions must surface the arithmetic inputs
        rather than an unsupported conclusion (the q44255 failure — briefing
        had every input to 'Jun 29 + 10 non-Sundays > Jul 4', did no math, and
        asserted Yes anyway)."""
        collapsed = " ".join(_summarizer_prompt().split())
        assert "state which version governs" in collapsed
        assert "compress the superseded version to one line" in collapsed
        assert "QUOTE the relevant inputs explicitly" in collapsed
        assert "verify the arithmetic" in collapsed
        assert "do not assert a deadline conclusion without showing the facts it rests on" in collapsed

    def test_evidence_age_disclosure_opens_briefing(self) -> None:
        """2026-07-18 AskNews audit R4: the briefing must open with the age of
        the newest directly-relevant evidence, or say no article directly
        reports the resolution quantity (the q44219 failure — a 67-day-old
        leaderboard read as current)."""
        collapsed = " ".join(_summarizer_prompt().split())
        assert "Newest directly-relevant article" in collapsed
        assert "DIRECTLY bears on the resolution" in collapsed
        assert "background rather than signal" in collapsed
        # The disclosure is requirement #1, ahead of the extraction bullets.
        assert collapsed.index("Newest directly-relevant article") < collapsed.index("Extracts ALL facts")

    def test_proportionality_rule_scopes_comprehensiveness(self) -> None:
        """2026-07-18 AskNews audit (operator-designed proportionality rule):
        length tracks decision-relevant content; tangential-only article sets
        yield a short briefing with a screened-out list, and the
        comprehensiveness mandate is scoped to decision-relevant material so
        the two rules can't be read as contradicting."""
        collapsed = " ".join(_summarizer_prompt().split())
        assert "Length must track decision-relevant content, not article count" in collapsed
        assert "Screened out as not decision-relevant" in collapsed
        assert "do not pad with tangential material to appear thorough" in collapsed
        # Comprehensiveness survives, but scoped to decision-relevant material.
        assert "Be COMPREHENSIVE about DECISION-RELEVANT material" in collapsed
        assert "Be COMPREHENSIVE — do not omit relevant details" not in collapsed

    def test_hard_relevance_gate_drops_offtopic_articles(self) -> None:
        """2026-07-18 AskNews audit R3: a hard per-article screen against the
        resolution criteria (the semantic-drift padding class — Microsoft
        surveys and crypto incidents extracted in full for unrelated
        questions). Off-topic articles are DROPPED to a one-line list, not
        summarized 'briefly'."""
        collapsed = " ".join(_summarizer_prompt().split())
        assert "screen each article for relevance to the resolution criteria" in collapsed
        assert "NO direct bearing on how this question resolves" in collapsed
        assert "must be DROPPED entirely" in collapsed
        assert "plausibly affect a forecaster's reasoning on THIS question" in collapsed

    def test_recency_first_ordering_replaces_input_mirroring(self) -> None:
        """2026-07-18 AskNews audit R1: the briefing leads with the newest,
        most resolution-relevant facts instead of mirroring AskNews's
        Historical/Recent input structure (the q44555 failure — obsolete
        chronology buried the live matchup)."""
        collapsed = " ".join(_summarizer_prompt().split())
        assert "lead with the most recent and most resolution-relevant facts" in collapsed
        assert "Do not mirror the raw input's section structure" in collapsed
        assert "organize by recency and relevance to the question" in collapsed
        # The old input-mirroring instruction it replaced must be gone.
        assert "Maintains the section structure" not in collapsed
