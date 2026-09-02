"""Direct unit tests for prompt-building functions in ``metaculus_bot.prompts``.

These exercise the prompt strings themselves (not the pipelines that call
them). Critical for the benchmarking carve-out: the higher-level gap-fill
tests mock ``_run_analyzer`` entirely, so without these the benchmarking
warning could be deleted from a prompt and no test would catch it —
backtest scores would silently get polluted with prediction-market data.
"""

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from metaculus_bot.prompts import (
    _AUTO_ANNOTATED_CITATION_CLAUSE,
    _OUTSIDE_VENUE_MARKET_ODDS_BULLET,
    _SOURCE_TIER_TAG_INSTRUCTION,
    MARKET_SNAPSHOT_SECTION_HEADER,
    TS_ANCHOR_SECTION_HEADER,
    asknews_summarizer_prompt,
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
from metaculus_bot.research.gemini_attribution import UNVERIFIED_ATTRIBUTION_MARKER
from metaculus_bot.research.section_format import PROVIDER_SECTION_HEADERS

# Research that carries a rendered prediction-market section. The market clause in the three
# forecaster prompts is gated on this header (see TestPredictionMarketFraming), so tests that
# assert on the clause must hand the prompt research that would actually have carried a snapshot.
_RESEARCH_WITH_MARKETS = f"Some news.\n\n{MARKET_SNAPSHOT_SECTION_HEADER}\n| venue | market | prob |\n| k | m | 0.4 |"

# gap_fill_analyzer_prompt


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


# gap_fill_search_prompt


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


# Forecasting-window anchor (binary / MC / numeric + stacking variants)


def _binary_q(
    open_time: datetime | None = None,
    resolve_time: datetime | None = None,
) -> MagicMock:
    """Minimal question stub with the attributes the prompts read."""
    q = MagicMock()
    q.question_text = "Will X occur by 2030?"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = "fp"
    q.open_time = open_time if open_time is not None else datetime.now() - timedelta(days=30)
    q.scheduled_resolution_time = resolve_time if resolve_time is not None else datetime.now() + timedelta(days=365)
    return q


def _mc_q(**kwargs) -> MagicMock:
    q = _binary_q(**kwargs)
    q.options = ["A", "B", "C"]
    return q


def _numeric_q(**kwargs) -> MagicMock:
    q = _binary_q(**kwargs)
    q.unit_of_measure = "widgets"
    q.lower_bound = 0
    q.upper_bound = 1000
    return q


class TestForecastingWindowAnchor:
    """Every forecasting prompt must surface the open date, today, and
    resolution date so the LLM anchors on the forecasting window and does
    NOT treat pre-open historical events as already-resolved (e.g., the
    classic 1945-Japan-detonation → "Will a detonation occur by 2030?"
    auto-YES error)."""

    # -- binary ------------------------------------------------------------

    def test_binary_injects_open_and_resolution_dates(self) -> None:
        q = _binary_q(
            open_time=datetime(2026, 1, 15),
            resolve_time=datetime(2030, 12, 31),
        )
        result = binary_prompt(q, research="r")
        assert "2026-01-15" in result
        assert "2030-12-31" in result
        assert "Forecasting window" in result
        assert "days ago" in result
        assert "days from now" in result
        assert "BEFORE the open date" in result

    def test_binary_phase0_reinforcement_present(self) -> None:
        """The Phase 0 resolution check must remind the model about the open timestamp."""
        result = binary_prompt(_binary_q(), research="r")
        assert "open timestamp" in result
        assert "1945" in result  # the worked example
        assert "pre-dating the open date" in result

    def test_binary_asserts_on_missing_open_time(self) -> None:
        """Missing timestamps are a data bug, not a graceful-degrade path."""
        q = _binary_q()
        q.open_time = None
        with pytest.raises(AssertionError):
            binary_prompt(q, research="r")

    def test_binary_asserts_on_missing_scheduled_resolution(self) -> None:
        q = _binary_q()
        q.scheduled_resolution_time = None
        with pytest.raises(AssertionError):
            binary_prompt(q, research="r")

    # -- multiple choice ---------------------------------------------------

    def test_multiple_choice_injects_window(self) -> None:
        q = _mc_q(
            open_time=datetime(2025, 3, 1),
            resolve_time=datetime(2027, 3, 1),
        )
        result = multiple_choice_prompt(q, research="r")
        assert "2025-03-01" in result
        assert "2027-03-01" in result
        assert "Forecasting window" in result
        assert "BEFORE the open date" in result

    def test_multiple_choice_asserts_on_missing_timestamps(self) -> None:
        q = _mc_q()
        q.open_time = None
        with pytest.raises(AssertionError):
            multiple_choice_prompt(q, research="r")

    # -- numeric -----------------------------------------------------------

    def test_numeric_injects_window(self) -> None:
        q = _numeric_q(
            open_time=datetime(2024, 6, 1),
            resolve_time=datetime(2026, 6, 1),
        )
        result = numeric_prompt(q, research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        assert "2024-06-01" in result
        assert "2026-06-01" in result
        assert "Forecasting window" in result
        assert "BEFORE the open date" in result

    def test_numeric_asserts_on_missing_timestamps(self) -> None:
        q = _numeric_q()
        q.scheduled_resolution_time = None
        with pytest.raises(AssertionError):
            numeric_prompt(q, research="r", lower_bound_message="lbm", upper_bound_message="ubm")

    # -- stacking variants -------------------------------------------------


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


# web_research_prompt


class TestMcPromptInterpolatesRealOptionNames:
    """Strict parsers (e.g. gemma-4-31b-it) refuse to map literal ``Option_A``
    placeholders onto real option names in the allowed-list — they correctly
    emit ``<<NOT_FOUND>>`` because the prompt example does not contain anything
    semantically tied to the question's actual options.

    Fix: the STRUCTURED FORECAST JSON block in both ``multiple_choice_prompt``
    and ``stacking_multiple_choice_prompt`` must use the REAL option names as
    JSON keys so the LLM emits text the parser can directly recognize.

    Post-refactor: the trailing prose "{opt}: NN%" answer lines are gone;
    ``option_probs`` in the JSON block is the sole forecast surface.
    """

    def test_stacking_mc_prompt_emits_real_option_names_in_json_block(self) -> None:
        q = _mc_q()
        q.options = ["Apple", "Banana", "Cherry"]

        result = stacking_multiple_choice_prompt(q, research="r", base_predictions=["a1", "a2"])

        structured_section = result[result.find("STRUCTURED FORECAST") :]
        for opt in ("Apple", "Banana", "Cherry"):
            assert f'"{opt}"' in structured_section, f"option {opt!r} missing from option_probs JSON example"
        # Trailing prose per-option lines must be gone.
        assert "Apple: NN%" not in result
        assert "Banana: NN%" not in result
        assert "Cherry: NN%" not in result

    def test_stacking_mc_prompt_drops_literal_option_a_b_placeholders(self) -> None:
        q = _mc_q()
        q.options = ["Apple", "Banana", "Cherry"]

        result = stacking_multiple_choice_prompt(q, research="r", base_predictions=["a1", "a2"])

        assert '"Option_A"' not in result
        assert '"Option_B"' not in result
        assert '"Option_N"' not in result
        # Also the old prose placeholders.
        assert "Option_A: NN%" not in result
        assert "Option_B: NN%" not in result
        assert "Option_N: NN%" not in result

    def test_multiple_choice_prompt_emits_real_option_names_in_json_block(self) -> None:
        q = _mc_q()
        q.options = ["Apple", "Banana", "Cherry"]

        result = multiple_choice_prompt(q, research="r")

        structured_section = result[result.find("STRUCTURED FORECAST") :]
        for opt in ("Apple", "Banana", "Cherry"):
            assert f'"{opt}"' in structured_section, f"option {opt!r} missing from option_probs JSON example"
        assert "Apple: NN%" not in result
        assert "Banana: NN%" not in result
        assert "Cherry: NN%" not in result

    def test_multiple_choice_prompt_drops_literal_option_a_b_placeholders(self) -> None:
        q = _mc_q()
        q.options = ["Apple", "Banana", "Cherry"]

        result = multiple_choice_prompt(q, research="r")

        # No literal "Option_A" placeholders anywhere — not in the JSON block
        # (real option names go there), not in prose (prose forecast lines gone).
        assert '"Option_A"' not in result
        assert '"Option_B"' not in result
        assert '"Option_N"' not in result
        assert "Option_A: NN%" not in result
        assert "Option_B: NN%" not in result
        assert "Option_N: NN%" not in result

    def test_stacking_mc_prompt_preserves_options_in_order_in_json_block(self) -> None:
        """The JSON-block ``option_probs`` example must list options in the same
        order as ``question.options`` — a strict parser matching on positional
        alignment depends on that ordering."""
        q = _mc_q()
        q.options = ["Manufacturing PMI higher", "Services PMI higher", "Equal"]

        result = stacking_multiple_choice_prompt(q, research="r", base_predictions=["a1", "a2"])

        idx_mfg = result.find('"Manufacturing PMI higher"')
        idx_svc = result.find('"Services PMI higher"')
        idx_eq = result.find('"Equal"')
        assert idx_mfg >= 0
        assert idx_svc >= 0
        assert idx_eq >= 0
        assert idx_mfg < idx_svc < idx_eq


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


# Prediction-market framing (strong-evidence, criteria/date-matched weighting)


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


# Source-provenance / motivation trust ladder


class TestSourceProvenanceLadder:
    """The Source-analysis section of every forecaster prompt must carry the
    provenance trust ladder: rank claims by proximity to the primary record and
    adjust by source motivation (discount self-interest, treat statements
    against interest as strong evidence). These are lightweight "did the text
    land" guards — we expect them to break when the prompt is revised."""

    def _assert_ladder_present(self, prompt: str) -> None:
        # Collapse whitespace so assertions don't depend on where clean_indents wraps lines.
        lowered = " ".join(prompt.lower().split())
        assert "proximity to the primary record" in lowered
        assert "against the speaker's interest" in lowered
        # The A-D DEFINITIONS live once, in the research-side _SOURCE_TIER_TAG_INSTRUCTION, and
        # the briefing arrives carrying the tags (99 of 1,069 archive records, every one since the
        # tagging landed in prod). The forecaster ladder names the tag shape and keeps only the two
        # USAGE clauses the tag instruction does not state: (C) facts-not-framing, (D) suggestive only.
        assert "arrive tagged by source tier" in lowered
        assert "[a: ...]" in lowered
        assert "use their cited facts, not their framing" in lowered
        assert "suggestive only" in lowered
        assert "government statistics, regulatory filings" not in lowered, (
            "the tier definitions were re-teaching what the research-side tag instruction already defines"
        )

    def _assert_unverified_attribution_defined(self, prompt: str) -> None:
        """`[unverified attribution]` reaches the forecaster in gemini research sections
        (`research/gemini_attribution.py` writes it over a tier tag whose named outlet the
        grounding record cannot back), and it lands on a bundle whose ladder tells the model to
        weight by tier. Undefined, it is a token the forecaster has to guess at, on exactly the
        claims where the guess matters."""
        lowered = " ".join(prompt.lower().split())
        assert "[unverified attribution]" in lowered
        assert "could not match against its own retrieval record" in lowered
        # The two halves that keep it from reading as "this fact is false" or as a tier grade.
        assert "the claim itself may still be correct" in lowered
        assert "untiered, unattributed evidence rather than as a named outlet's authority" in lowered

    def test_binary_prompt_defines_the_unverified_attribution_token(self) -> None:
        self._assert_unverified_attribution_defined(binary_prompt(_binary_q(), research="r"))

    def test_multiple_choice_prompt_defines_the_unverified_attribution_token(self) -> None:
        self._assert_unverified_attribution_defined(multiple_choice_prompt(_mc_q(), research="r"))

    def test_numeric_prompt_defines_the_unverified_attribution_token(self) -> None:
        result = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        self._assert_unverified_attribution_defined(result)

    def test_the_definition_matches_the_marker_the_rewriter_actually_writes(self) -> None:
        """Pinned against the emitting constant rather than a copied string: if the rewriter's
        wording changes, the forecaster-facing definition has to move with it or the prompt
        defines a token no section carries."""
        assert f"[{UNVERIFIED_ATTRIBUTION_MARKER}]" in binary_prompt(_binary_q(), research="r")

    def test_binary_prompt_carries_provenance_ladder(self) -> None:
        self._assert_ladder_present(binary_prompt(_binary_q(), research="r"))

    def test_multiple_choice_prompt_carries_provenance_ladder(self) -> None:
        self._assert_ladder_present(multiple_choice_prompt(_mc_q(), research="r"))

    def test_numeric_prompt_carries_provenance_ladder(self) -> None:
        result = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        self._assert_ladder_present(result)

    def test_numeric_prompt_preserves_data_anchor(self) -> None:
        """Regression: appending the ladder must not displace the load-bearing
        data-anchor bullet in the numeric Source-analysis section."""
        result = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        lowered = " ".join(result.lower().split())
        assert "most recent authoritative measurement" in lowered
        assert "centered near this value" in lowered


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
        assert "if the first pass already states that source's current reading with the date it was read" in lowered
        assert "no slot should be spent re-fetching a value the briefing holds" in lowered
        # The two conditions that still earn a verify-style gap.
        assert "carries no as-of date" in lowered
        assert "older than the source's own update cadence" in lowered


class TestNullResultReadingClause:
    """A search that found nothing licenses "could not find evidence of X", never
    "X did not happen". On qid 44799 four of six forecasters read the gap-fill
    resolver's "I found no authoritative public record" as "the permit is absent",
    and the two members that discounted it scored best in the ensemble. Every base
    forecaster prompt must carry the reading rule; the gap-fill analyzer carries the
    auditor's version of it."""

    def _assert_forecaster_clause_present(self, prompt: str) -> None:
        # Collapse whitespace so assertions don't depend on where clean_indents wraps lines.
        lowered = " ".join(prompt.lower().split())
        assert "read a null search result as a null search result" in lowered
        assert "could not find evidence of x" in lowered
        # Coverage-conditioned strength, and the demonstrated-capability discount.
        assert "weight the absence by how well the topic is covered" in lowered
        assert "already demonstrated the capability" in lowered

    def test_binary_prompt_carries_null_result_clause(self) -> None:
        self._assert_forecaster_clause_present(binary_prompt(_binary_q(), research="r"))

    def test_multiple_choice_prompt_carries_null_result_clause(self) -> None:
        self._assert_forecaster_clause_present(multiple_choice_prompt(_mc_q(), research="r"))

    def test_numeric_prompt_carries_null_result_clause(self) -> None:
        result = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        self._assert_forecaster_clause_present(result)

    def test_clause_sits_with_the_evidence_weighting_rubric(self) -> None:
        """The clause is an evidence-weighting rule, so it must land after the
        Strong/Moderate/Weak rubric rather than displacing it."""
        prompt = binary_prompt(_binary_q(), research="r")
        rubric_at = prompt.index("Weak: anecdotes")
        clause_at = prompt.index("Read a null search result")
        assert rubric_at < clause_at

    def test_stacking_prompts_do_not_carry_the_clause(self) -> None:
        """Scope guard: the clause ships to the base prompts only (stacking is
        prod-disabled, so the diff stays minimal)."""
        stacked = stacking_binary_prompt(_binary_q(), research="r", base_predictions=["a1", "a2"])
        assert "Read a null search result" not in stacked

    def test_gap_fill_analyzer_carries_the_auditor_version(self) -> None:
        """Auditor phrasing, not forecaster phrasing: the analyzer must not bank a
        first-pass "found nothing" as an established negative, and must point the
        gap at the authoritative place that would hold the record."""
        prompt = gap_fill_analyzer_prompt(
            "Will the permit be granted?",
            "Resolves YES if the regulator lists the permit.",
            "See the regulator's public register.",
            "First pass: no authoritative public record found.",
            is_benchmarking=False,
        )
        lowered = " ".join(prompt.lower().split())
        assert "null results are search outcomes" in lowered
        assert "not as an established negative fact" in lowered
        assert "name that source in the search query" in lowered


class TestOutsideViewRubricRules:
    """The three 2026-09-01 outside-view rules, all from verified misses in that
    residual round. qid 43837: six members applied a monthly announcement rate over the
    FULL question window when 16 days had already elapsed event-free, then OR-ed it with
    a specific scheduled path the rate already contained. qid 44557: four of six wrote a
    17-25% base rate and published 35-55% on soft schedule signals. qid 44561: all six
    built a "no failure announced yet, so Poisson(1.0)" schedule model instead of the
    pooled FDIC failure rate. Each rule is a named module constant, so there is exactly
    one place to read the wording."""

    @staticmethod
    def _flat(prompt: str) -> str:
        # Collapse whitespace: the constants are pre-indented for clean_indents, so
        # assertions must not depend on where the lines wrap.
        return " ".join(prompt.lower().split())

    def _binary(self) -> str:
        return binary_prompt(_binary_q(), research="r")

    def _mc(self) -> str:
        return multiple_choice_prompt(_mc_q(), research="r")

    def _numeric(self) -> str:
        return numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")

    # -- remaining exposure + disjointness (binary, MC) --------------------

    def _assert_remaining_exposure(self, prompt: str) -> None:
        flat = self._flat(prompt)
        assert "outside-view rates apply to the exposure that remains" in flat
        assert "apply it from now until the question's deadline" in flat
        assert "already elapsed without the event as observed" in flat
        # The double-count half: a named path and the rate that contains it.
        assert "the two must be disjoint" in flat
        assert "remove that path's own instances from the rate before combining" in flat

    def test_binary_prompt_carries_remaining_exposure_rule(self) -> None:
        self._assert_remaining_exposure(self._binary())

    def test_multiple_choice_prompt_carries_remaining_exposure_rule(self) -> None:
        self._assert_remaining_exposure(self._mc())

    def test_binary_union_line_requires_disjoint_paths(self) -> None:
        """The union instruction itself is what invited the qid 43837 double count, so
        the disjointness requirement lands where the union is computed, not only in the
        rule below it."""
        flat = self._flat(self._binary())
        assert "1 - product of (1-p_i)" in flat
        assert "union only over paths that cannot be the same event" in flat

    # -- anchor consistency (binary, MC) -----------------------------------

    def _assert_anchor_consistency(self, prompt: str) -> None:
        flat = self._flat(prompt)
        assert "state the outside-view number you computed" in flat
        assert "more than about 15 percentage points" in flat
        assert "naming the specific evidence that justifies the gap" in flat
        # And the reverse failure: drifting off your own number on disposition alone.
        assert "do not move off your own number on a general feeling" in flat

    def test_binary_prompt_carries_anchor_consistency_rule(self) -> None:
        self._assert_anchor_consistency(self._binary())

    def test_multiple_choice_prompt_carries_anchor_consistency_rule(self) -> None:
        self._assert_anchor_consistency(self._mc())

    # -- count-in-period reference class (all three types) -----------------

    def _assert_count_in_period(self, prompt: str) -> None:
        flat = self._flat(prompt)
        assert "how many events of a kind occur in a period" in flat
        assert "pooled realized rate of that event over the longest comparable history" in flat
        # A known-candidate schedule updates the rate rather than replacing it.
        assert "it does not replace it" in flat

    def test_binary_prompt_carries_count_in_period_class(self) -> None:
        self._assert_count_in_period(self._binary())

    def test_multiple_choice_prompt_carries_count_in_period_class(self) -> None:
        self._assert_count_in_period(self._mc())

    def test_numeric_prompt_carries_count_in_period_class(self) -> None:
        """Count questions arrive as all three question types, so this one rule ships to
        the numeric prompt too — unlike the other two, which are probability-shaped."""
        self._assert_count_in_period(self._numeric())

    # -- placement + scope -------------------------------------------------

    def test_rules_sit_in_the_outside_view_phase(self) -> None:
        """All three are outside-view rules: they must land inside PHASE 1, before the
        inside-view update, so the model reads them while computing the number."""
        prompt = self._binary()
        phase1_at = prompt.index("PHASE 1: OUTSIDE VIEW")
        phase2_at = prompt.index("PHASE 2: INSIDE VIEW UPDATE")
        for phrase in (
            "For questions asking how many events",
            "Outside-view rates apply to the exposure that REMAINS",
            "State the outside-view number you computed",
        ):
            assert phase1_at < prompt.index(phrase) < phase2_at, f"{phrase!r} outside PHASE 1"

    def test_stacking_prompts_do_not_carry_the_rules(self) -> None:
        """Same scope guard as the null-result clause: base prompts only, since stacking
        is prod-disabled."""
        stacked = [
            stacking_binary_prompt(_binary_q(), research="r", base_predictions=["a1", "a2"]),
            stacking_multiple_choice_prompt(_mc_q(), research="r", base_predictions=["a1", "a2"]),
            stacking_numeric_prompt(
                _numeric_q(),
                research="r",
                base_predictions=["a1", "a2"],
                lower_bound_message="lbm",
                upper_bound_message="ubm",
            ),
        ]
        for prompt in stacked:
            assert "Outside-view rates apply to the exposure that REMAINS" not in prompt
            assert "State the outside-view number you computed" not in prompt
            assert "For questions asking how many events of a kind occur in a period" not in prompt

    def test_numeric_prompt_does_not_carry_the_probability_shaped_rules(self) -> None:
        """Scope: the exposure and anchor rules are worded in probabilities, while the
        numeric prompt anchors on a range. Only the reference-class rule ships there."""
        numeric = self._numeric()
        assert "Outside-view rates apply to the exposure that REMAINS" not in numeric
        assert "State the outside-view number you computed" not in numeric


class TestLastRealApplicationGap:
    """qid 45215: the question turned on how an electoral threshold cashed out at the
    last real election, a training-data fact no research bundle held and no gap asked
    for. Analyzer-only — it directs a search slot, which a forecaster prompt cannot."""

    def _analyzer(self) -> str:
        return gap_fill_analyzer_prompt(
            "Will party X clear the electoral threshold?",
            "Resolves YES if the commission seats party X.",
            "Threshold is 5% nationally.",
            "First pass: polling summary only.",
            is_benchmarking=False,
        )

    def test_analyzer_requires_a_last_real_application_gap(self) -> None:
        flat = " ".join(self._analyzer().lower().split())
        assert "last real application of the rule" in flat
        assert "discontinuous institutional rule applied by a body" in flat
        assert "at its most recent real application" in flat
        assert "as a realized count or outcome" in flat

    def test_analyzer_separates_the_institution_rule_from_the_question_threshold(self) -> None:
        """The gap is about how the BODY applies its own rule, not a restatement of the
        question's resolution threshold, which the analyzer already has in front of it."""
        flat = " ".join(self._analyzer().lower().split())
        assert "not about the question's own resolution threshold" in flat

    def test_rule_does_not_override_the_do_not_invent_gaps_discipline(self) -> None:
        flat = " ".join(self._analyzer().lower().split())
        assert "do not invent gaps" in flat
        assert "only when such a rule is actually in play" in flat

    def test_rule_sits_with_the_answerable_now_block(self) -> None:
        prompt = self._analyzer()
        assert prompt.index("ANSWERABLE NOW") < prompt.index("LAST REAL APPLICATION OF THE RULE")

    def test_forecaster_prompts_do_not_carry_the_gap_rule(self) -> None:
        """Scope guard: this one is an instruction to the gap AUDITOR."""
        for prompt in (
            binary_prompt(_binary_q(), research="r"),
            multiple_choice_prompt(_mc_q(), research="r"),
            numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm"),
        ):
            assert "LAST REAL APPLICATION OF THE RULE" not in prompt


class TestMcExampleBlockEscaping:
    """The MC example block is the forecaster's authoritative template, so it has to stay
    valid JSON for option names carrying quotes or backslashes — the escaping is done by
    ``_option_probs_example``, and a break there is invisible to substring assertions."""

    @pytest.mark.parametrize("options", [["Red", "Blue"], ['He said "yes"', r"C:\Windows"]])
    def test_mc_example_block_stays_valid_json(self, options: list[str]) -> None:
        q = _mc_q()
        q.options = options
        prompt = multiple_choice_prompt(q, research="r")
        body = re.findall(r"```json\s*\n(.*?)\n\s*```", prompt, re.DOTALL)[-1]
        parsed = json.loads(body)
        assert list(parsed["option_probs"].keys()) == options


class TestStatusQuoDerivation:
    """Every forecaster prompt must open PHASE 0 with a mandatory status-quo
    DERIVATION — a question the model answers itself before reviewing any
    research — not a rule/warning. Rules-as-warnings get argued away as
    boilerplate (qid 42024: 4/5 models dismissed the "not yet satisfied"
    line); a derivation the model writes in its own words is stickier."""

    def _assert_derivation_present(self, prompt: str) -> None:
        lowered = " ".join(prompt.lower().split())
        assert "status-quo derivation" in lowered
        # The model must state the platform-state premise in its own words...
        assert "open and unresolved as of" in lowered
        # ...with today's date interpolated so the statement is concrete. The
        # prompt uses UTC (see prompts._forecasting_window_str, which uses
        # datetime.now(timezone.utc) to stay tz-aware against ft's aware question
        # datetimes), so assert the UTC date too — a naive datetime.now() flakes
        # in the evening-local/next-day-UTC window.
        assert datetime.now(UTC).strftime("%Y-%m-%d") in prompt
        # Moving off the status quo requires naming a post-open trigger.
        assert "post-open event" in lowered
        # And an explicit commitment about the window.
        assert "no qualifying event has yet occurred inside the window" in lowered

    def _assert_derivation_before_outside_view(self, prompt: str) -> None:
        idx_derivation = prompt.find("Status-quo derivation")
        idx_phase1 = prompt.find("PHASE 1")
        assert idx_derivation >= 0
        assert idx_phase1 >= 0
        assert idx_derivation < idx_phase1, "status-quo derivation must come before the outside view"

    def test_binary_prompt_has_status_quo_derivation_first(self) -> None:
        prompt = binary_prompt(_binary_q(), research="r")
        self._assert_derivation_present(prompt)
        self._assert_derivation_before_outside_view(prompt)
        # In binary, the derivation must be the TOP of PHASE 0 — before the resolution check.
        assert prompt.find("Status-quo derivation") < prompt.find("Resolution check")

    def test_multiple_choice_prompt_has_status_quo_derivation_first(self) -> None:
        prompt = multiple_choice_prompt(_mc_q(), research="r")
        self._assert_derivation_present(prompt)
        self._assert_derivation_before_outside_view(prompt)

    def test_numeric_prompt_has_status_quo_derivation_first(self) -> None:
        prompt = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        self._assert_derivation_present(prompt)
        self._assert_derivation_before_outside_view(prompt)

    def test_binary_resolution_check_retained_after_derivation(self) -> None:
        """Regression: adding the derivation must not displace the 0a resolution
        check or the 0b decomposition."""
        prompt = binary_prompt(_binary_q(), research="r")
        assert "Resolution check" in prompt
        assert "Resolution decomposition" in prompt


class TestConjunctiveCriteriaPricing:
    """Change #2: the binary prompt upgrades the qualitative Boolean-product
    decomposition with a NUMERIC pricing table — placed LATE in the flow
    (after evidence review and red-team) so the red-team can causally affect
    the clause probabilities. PHASE 0b keeps only the qualitative listing."""

    def test_binary_prompt_has_pricing_table_section(self) -> None:
        prompt = binary_prompt(_binary_q(), research="r")
        lowered = " ".join(prompt.lower().split())
        assert "conjunctive criteria pricing" in lowered
        assert "one row per resolution clause" in lowered
        assert "product" in lowered

    def test_pricing_table_comes_after_red_team_and_before_final_rationale(self) -> None:
        """ORDERING IS LOAD-BEARING: clause numbers must be emitted after the
        red-team/bear-bull section so red-teaming can move them, and before
        the final rationale that reconciles against the product."""
        prompt = binary_prompt(_binary_q(), research="r")
        idx_red_team = prompt.find("Red-team both")
        idx_pricing = prompt.find("Conjunctive criteria pricing")
        idx_final = prompt.find("Final rationale and calibration")
        assert 0 <= idx_red_team < idx_pricing < idx_final

    def test_phase0b_keeps_listing_but_defers_numbers(self) -> None:
        """PHASE 0b still lists/structures the clauses (early structure
        identification is fine) but must explicitly defer the probabilities
        to the late pricing step."""
        prompt = binary_prompt(_binary_q(), research="r")
        assert "Resolution decomposition" in prompt
        assert "Boolean product" in prompt
        lowered = " ".join(prompt.lower().split())
        assert "do not assign probabilities to the clauses yet" in lowered
        # The deferral must appear inside 0b, i.e. before PHASE 1.
        idx_defer = prompt.find("Do NOT assign probabilities to the clauses yet")
        assert 0 <= idx_defer < prompt.find("PHASE 1")

    def test_reconciliation_requires_named_clause_dependence(self) -> None:
        """pgodzinai 42855 failure mode: a computed clause product coexisting
        with free-form narrative adjustment gets nullified (82% computed →
        87% via 'season-specific upward adjustment'). Any deviation from the
        product must operate through the clause probabilities themselves, a
        named clause dependence, or a corrected clause decomposition; overrides
        that route around the clauses are explicitly forbidden."""
        prompt = binary_prompt(_binary_q(), research="r")
        lowered = " ".join(prompt.lower().split())
        assert "you have exactly three valid moves" in lowered
        assert "revise the clause probabilities themselves and recompute" in lowered
        assert "name a specific dependence between clauses" in lowered
        # The third sanctioned move: the decomposition itself was wrong.
        assert "revise the clause decomposition from 0b" in lowered
        assert (
            "all hedging and adjustment must operate through the clauses, their dependence, or a corrected "
            "decomposition, not around them" in lowered
        )
        assert "if none applies, stay at the product" in lowered


class TestResolutionMetricEcho:
    """PHASE 0 resolution-metric echo: when the resolution criteria name an
    official statistical series, force the forecaster to name the exact
    resolving series and enumerate its variants (component vs total, etc.)
    before forecasting. Motivated by qid 44211 — all six models priced the
    USBP-apprehensions *component* of a series that resolves on the *total*,
    even though the research carried the wedge, the historical conversion, and
    an explicit provider warning. Scoped to the numeric prompt (which also
    serves discrete-integer questions — outcome_type is decided in the block,
    there is no separate discrete prompt) and the binary prompt. MC and the
    prod-disabled stacking prompts are deliberately untouched."""

    _HEADER = "resolution-metric echo (named-series questions only)"
    _INERT = "no named series, metric echo skipped"

    @staticmethod
    def _collapsed(prompt: str) -> str:
        # Collapse whitespace so assertions don't depend on where clean_indents wraps lines.
        return " ".join(prompt.lower().split())

    def _assert_core_block(self, prompt: str) -> None:
        c = self._collapsed(prompt)
        assert self._HEADER in c
        assert "name the exact series that resolves this question" in c
        assert "enumerate the plausible variants" in c
        assert "component vs total" in c
        # Inert escape: questions with no named series skip the step.
        assert self._INERT in c
        # Don't-discard-on-one-implausible-estimate: the gemini-31k poisoning lesson.
        assert "do not discard a candidate variant" in c
        assert "recompute the candidate from its components" in c

    def test_numeric_prompt_has_metric_echo(self) -> None:
        prompt = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        self._assert_core_block(prompt)
        c = self._collapsed(prompt)
        # Numeric reconciles against the displayed range — but NOT as an oracle
        # (the 44211 trap: the true total was the bounds midpoint).
        assert "reconcile each candidate against the displayed range" in c
        assert 'do not read "inside the range" as confirming' in c
        # Points at both numeric-available research sections (synergy, not duplication).
        assert "## Resolution Source Snapshot" in prompt
        assert "## Time Series Anchor" in prompt

    def test_binary_prompt_has_metric_echo(self) -> None:
        prompt = binary_prompt(_binary_q(), research="r")
        self._assert_core_block(prompt)
        c = self._collapsed(prompt)
        # Binary reconciles against the criteria's threshold/comparison (no displayed range).
        assert "reconcile each candidate against the threshold or comparison" in c
        # Points at the resolution-source snapshot only — the TS anchor is numeric-only.
        assert "## Resolution Source Snapshot" in prompt
        assert "## Time Series Anchor" not in prompt

    def test_binary_metric_echo_after_decomposition_before_phase1(self) -> None:
        prompt = binary_prompt(_binary_q(), research="r")
        idx_decomp = prompt.find("Resolution decomposition")
        idx_echo = prompt.find("Resolution-metric echo")
        idx_phase1 = prompt.find("PHASE 1")
        assert 0 <= idx_decomp < idx_echo < idx_phase1

    def test_numeric_metric_echo_after_status_quo_before_phase1(self) -> None:
        prompt = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        idx_sq = prompt.find("Status-quo derivation")
        idx_echo = prompt.find("Resolution-metric echo")
        idx_phase1 = prompt.find("PHASE 1")
        assert 0 <= idx_sq < idx_echo < idx_phase1

    def test_mc_prompt_omits_metric_echo(self) -> None:
        # MC questions resolve to an enumerated option, not a value read off a
        # named statistical series, so the component-vs-total ambiguity does not
        # arise. The measured miss family (44211 numeric, 42018/41801 binary)
        # contains zero MC cases — scoping decision locked here.
        prompt = multiple_choice_prompt(_mc_q(), research="r")
        assert "Resolution-metric echo" not in prompt

    def test_stacking_prompts_omit_metric_echo(self) -> None:
        # Stacking is prod-disabled; those prompts are left untouched.
        binary = stacking_binary_prompt(_binary_q(), research="r", base_predictions=["a1", "a2"])
        mc = stacking_multiple_choice_prompt(_mc_q(), research="r", base_predictions=["a1", "a2"])
        numeric = stacking_numeric_prompt(
            _numeric_q(),
            research="r",
            base_predictions=["a1", "a2"],
            lower_bound_message="lbm",
            upper_bound_message="ubm",
        )
        for p in (binary, mc, numeric):
            assert "Resolution-metric echo" not in p


class TestNumericPromptThirteenPercentiles:
    """The numeric prompts must document all 13 standard percentile keys in
    the STRUCTURED FORECAST JSON schema example (P1 first, P99 last) and
    never tell the model to emit exactly 11.

    Post-refactor: the ONLY forecast surface is ``declared_percentiles`` in the
    JSON block — the old trailing "Percentile X: [value]" prose lines are gone.
    We assert on the JSON-key form ("0.01" .. "0.99")."""

    _PERCENTILE_KEYS = (
        "0.01",
        "0.025",
        "0.05",
        "0.1",
        "0.2",
        "0.4",
        "0.5",
        "0.6",
        "0.8",
        "0.9",
        "0.95",
        "0.975",
        "0.99",
    )

    def test_numeric_prompt_json_block_has_all_thirteen_keys_in_order(self) -> None:
        result = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        structured_section = result[result.find("STRUCTURED FORECAST") :]
        indices = []
        for key in self._PERCENTILE_KEYS:
            token = f'"{key}"'
            assert token in structured_section, f"missing percentile key {token} in declared_percentiles example"
            indices.append(structured_section.find(token))
        # Keys appear in the declared order: 0.01 < 0.025 < ... < 0.99.
        assert indices == sorted(indices), f"percentile keys out of order: {indices}"
        # No trailing prose "Percentile 1: [value]" block anywhere.
        assert "Percentile 1:" not in result
        assert "Percentile 99:" not in result

    def test_numeric_prompt_says_13_not_11(self) -> None:
        result = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        lowered = " ".join(result.lower().split())
        assert "all 13 percentiles" in lowered or "all 13 standard" in lowered
        assert "13 standard percentiles" in lowered
        assert "11 percentiles" not in lowered
        assert "11 standard percentiles" not in lowered

    def test_stacking_numeric_prompt_says_13_not_11(self) -> None:
        result = stacking_numeric_prompt(
            _numeric_q(),
            research="r",
            base_predictions=["a1", "a2"],
            lower_bound_message="lbm",
            upper_bound_message="ubm",
        )
        lowered = " ".join(result.lower().split())
        assert "all 13 percentiles" in lowered or "all 13 standard" in lowered
        assert "11 percentiles" not in lowered

    def test_stacking_numeric_prompt_json_block_has_all_thirteen_keys_in_order(self) -> None:
        result = stacking_numeric_prompt(
            _numeric_q(),
            research="r",
            base_predictions=["a1", "a2"],
            lower_bound_message="lbm",
            upper_bound_message="ubm",
        )
        structured_section = result[result.find("STRUCTURED FORECAST") :]
        indices = []
        for key in self._PERCENTILE_KEYS:
            token = f'"{key}"'
            assert token in structured_section, (
                f"missing percentile key {token} in stacking declared_percentiles example"
            )
            indices.append(structured_section.find(token))
        assert indices == sorted(indices), f"stacking percentile keys out of order: {indices}"
        # No trailing prose "Percentile 1: [value]" block.
        assert "Percentile 1:" not in result
        assert "Percentile 99:" not in result


def _extract_last_json_block(prompt: str) -> str:
    """Return the body of the LAST fenced ```json block in a prompt.

    Every forecasting prompt ends on its STRUCTURED FORECAST example, and the ladder
    (``value_extraction``) reads that block as the authoritative forecast, so the example
    is what teaches the model the schema. One regex for both classes below.
    """
    blocks = re.findall(r"```json\s*\n(.*?)\n\s*```", prompt, re.DOTALL)
    assert blocks, "no fenced json block found in prompt"
    return blocks[-1]


class TestOptionProbsExampleJsonValidity:
    """The MC schema example is the forecaster's authoritative template — it must
    be VALID JSON for any real option names, including ones carrying quotes,
    backslashes, or newlines (F2). A naive f-string concat emitted invalid JSON
    for those and silently taught the model a broken schema."""

    @pytest.mark.parametrize(
        "options",
        [
            ['He said "yes"', r"C:\Windows", "Option C"],
            ["Line\nbreak", "Plain"],
        ],
    )
    def test_mc_prompt_block_example_parses_for_special_char_options(self, options: list[str]) -> None:
        q = _mc_q()
        q.options = options
        prompt = multiple_choice_prompt(q, research="r")
        body = _extract_last_json_block(prompt)
        parsed = json.loads(body)
        assert list(parsed["option_probs"].keys()) == options

    def test_stacking_mc_prompt_block_example_parses_for_special_char_options(self) -> None:
        options = ['He said "yes"', r"C:\Windows", "Option C"]
        q = _mc_q()
        q.options = options
        prompt = stacking_multiple_choice_prompt(q, research="r", base_predictions=["a1"])
        body = _extract_last_json_block(prompt)
        parsed = json.loads(body)
        assert list(parsed["option_probs"].keys()) == options

    def test_example_probs_valid_for_large_option_count(self) -> None:
        options = [f"Bucket {i}" for i in range(12)]
        q = _mc_q()
        q.options = options
        prompt = multiple_choice_prompt(q, research="r")
        parsed = json.loads(_extract_last_json_block(prompt))
        probs = list(parsed["option_probs"].values())
        assert sum(probs) == pytest.approx(1.0, abs=0.02)
        assert all(0.0 < p < 1.0 for p in probs)


_EXAMPLE_BLOCK_BUILDERS = [
    pytest.param(lambda: binary_prompt(_binary_q(), research="r"), id="binary"),
    pytest.param(lambda: multiple_choice_prompt(_mc_q(), research="r"), id="multiple_choice"),
    pytest.param(
        lambda: numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm"),
        id="numeric",
    ),
    pytest.param(
        lambda: stacking_binary_prompt(_binary_q(), research="r", base_predictions=["a1", "a2"]),
        id="stacking_binary",
    ),
    pytest.param(
        lambda: stacking_multiple_choice_prompt(_mc_q(), research="r", base_predictions=["a1", "a2"]),
        id="stacking_multiple_choice",
    ),
    pytest.param(
        lambda: stacking_numeric_prompt(
            _numeric_q(),
            research="r",
            base_predictions=["a1", "a2"],
            lower_bound_message="lbm",
            upper_bound_message="ubm",
        ),
        id="stacking_numeric",
    ),
]


class TestStructuredForecastExampleBlocks:
    """Every builder's STRUCTURED FORECAST example must PARSE, and must not re-grow a
    key the 2026-09-02 de-bloat retired.

    These examples are static literals, so the only thing that breaks them is a source
    edit — and until this test existed most of them were guarded by substring checks only,
    which a dropped comma sails straight past. Dropping the comma before a field in the
    binary block once left the whole suite green. Parsing all six closes that gap in one
    place, and the retired-key check means re-adding a post-hoc telemetry slot to any
    example has to come through here (the reasoning behind the removals is in
    ``scratch/schema_bloat_audit_2026-09-02.md``: the block is written after the forecast
    is fixed, so a slot in it cannot scaffold reasoning).
    """

    # Retired by the schema de-bloat: each was read only by dormant telemetry, and every one
    # of them asked the model for post-hoc admin rather than for its forecast.
    _RETIRED_KEYS = (
        "remaining_window_days",
        "base_rate_anchor",
        "criteria_clauses",
        "other_mass",
        "concentration",
    )

    @pytest.mark.parametrize("build_prompt", _EXAMPLE_BLOCK_BUILDERS)
    def test_example_block_parses(self, build_prompt: Callable[[], str]) -> None:
        parsed = json.loads(_extract_last_json_block(build_prompt()))
        assert parsed["question_type"] in {"binary", "multiple_choice", "numeric"}

    @pytest.mark.parametrize("build_prompt", _EXAMPLE_BLOCK_BUILDERS)
    def test_example_block_carries_no_retired_key(self, build_prompt: Callable[[], str]) -> None:
        parsed = json.loads(_extract_last_json_block(build_prompt()))
        assert not [key for key in self._RETIRED_KEYS if key in parsed]

    def test_only_the_base_numeric_prompt_asks_for_outcome_type(self) -> None:
        """``outcome_type`` gates discrete snapping and saves a parser call, so the BASE
        numeric prompt keeps it. The stacker's vote is never read — the discrete decision is
        the base members' majority — so asking the stacker for it was pure admin, dropped
        2026-09-02."""
        base = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        stacking = stacking_numeric_prompt(
            _numeric_q(),
            research="r",
            base_predictions=["a1", "a2"],
            lower_bound_message="lbm",
            upper_bound_message="ubm",
        )
        assert json.loads(_extract_last_json_block(base))["outcome_type"] == "continuous"
        assert "outcome_type" not in json.loads(_extract_last_json_block(stacking))
        # The stacking prompt still DESCRIBES the field where it names what the base
        # members' own blocks carry; what went is its own schema instruction.
        stacking_schema = stacking[stacking.rfind("STRUCTURED FORECAST") :]
        assert "outcome_type" not in stacking_schema

    def test_the_thirteen_percentile_requirement_is_stated_once_per_numeric_prompt(self) -> None:
        """It was printed in the schema header line and again a few lines below in the
        Notes, in both numeric prompts. The header keeps it (it is the definition)."""
        for build in (
            lambda: numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm"),
            lambda: stacking_numeric_prompt(
                _numeric_q(),
                research="r",
                base_predictions=["a1", "a2"],
                lower_bound_message="lbm",
                upper_bound_message="ubm",
            ),
        ):
            schema_section = build()
            schema_section = schema_section[schema_section.rfind("STRUCTURED FORECAST") :]
            assert schema_section.count("MUST contain all") == 1


def _summarizer_prompt(**overrides) -> str:
    """Build the AskNews summarizer prompt with representative defaults."""
    kwargs = {
        "question_text": "Will X happen by 2027?",
        "resolution_criteria": "Resolves YES if X happens",
        "fine_print": "fp",
        "open_date": "2026-03-15",
        "research": "raw asknews articles",
    }
    kwargs.update(overrides)
    return asknews_summarizer_prompt(**kwargs)


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
