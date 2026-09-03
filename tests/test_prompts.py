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
    _HISTORY_DISCHARGED_RULE,
    _OUTSIDE_VENUE_MARKET_ODDS_BULLET,
    _SOFT_CLOCK_RULE,
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

    def test_binary_pre_open_rule_is_stated_twice_not_three_times(self) -> None:
        """The pre-open footgun has cost the bot badly, so the rule is deliberately stated
        TWICE: the forecasting-window line ("events before the open date do NOT resolve YES")
        and the status-quo derivation's demand to name the specific POST-OPEN event. The third
        statement, a 447-char 0a restatement with the 1945-detonation worked example, was
        retired as pure repetition (its receipt is the window line's own docstring)."""
        result = binary_prompt(_binary_q(), research="r")
        assert "BEFORE the open date" in result
        assert "POST-OPEN event" in result
        assert "1945" not in result
        assert "pre-dating the open date" not in result
        assert "open timestamp" not in result

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

    def test_numeric_status_quo_derivation_is_the_one_anchor_to_latest_statement(self) -> None:
        """The numeric prompt used to tell the model five times how to pick a center: the
        step-0 status-quo derivation, a step-1 "centered near this value" push, a step-3
        "status-quo outcome" line, step-3 trend continuation and a step-7 trajectory check.
        One anchor statement (step 0, which already says to move off the latest measurement
        only for a named post-open event) and one trend statement (step 3) remain."""
        result = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        lowered = " ".join(result.lower().split())
        assert lowered.count("most recent authoritative measurement") == 1
        assert lowered.count("trend continuation") == 1
        assert "centered near this value" not in lowered
        assert "status-quo outcome" not in lowered
        assert "trajectory check" not in lowered
        # Step 1 no longer claims a "data anchor" it does not carry.
        assert "source analysis and data anchor" not in lowered


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
        # Coverage-conditioned strength.
        assert "weight the absence by how well the topic is covered" in lowered
        # The retired third bullet ("weaker still where the actor has already demonstrated the
        # behavior") carried no receipt of its own and pushed the wrong way on qid 43837, where
        # Metaculus had announced eleven prior tournaments, none was found, and the answer was NO.
        assert "already demonstrated the capability" not in lowered

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
    """The outside-view corrections from the 2026-09-01 residual round, each stated ONCE.

    qid 43837: six members applied a monthly announcement rate over the FULL question
    window when 16 days had already elapsed event-free, then OR-ed it with a specific
    scheduled path the rate already contained. The correction survives in two places
    that were already there: the remaining-exposure sentence rides the binary
    conditional-hazard bullet (and stands alone as one MC bullet, since MC has no
    hazard bullet), and the disjointness half lives in the binary union line. The
    standalone `_REMAINING_EXPOSURE_RULE` that restated both was retired.

    qid 44557: four of six wrote a 17-25% base rate and published 35-55% on soft
    schedule signals. What survives is the SIZE: the existing "Anchor on your math"
    bullet now says a move of more than about 15 points needs a named reason. The
    standalone `_ANCHOR_CONSISTENCY_RULE` was retired — its first bullet was the fourth
    request to state the anchor, and its second ("do not move off your number on a
    general feeling that history counsels caution") priced at zero on the whole archive
    and would suppress good moves.

    qid 44561: all six built a "no failure announced yet, so Poisson(1.0)" schedule
    model instead of the pooled FDIC failure rate; `_COUNT_IN_PERIOD_REFERENCE_CLASS`
    stays verbatim in all three prompts.

    Receipts: scratch/prompt_bloat_audit_2026-09-02.md, scratch/prompt_debloat_2026-09-02/receipts.md."""

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

    # -- remaining exposure (binary, MC): once each --------------------------

    _EXPOSURE_KEY = "rates apply to the exposure that remains"

    def _assert_remaining_exposure_once(self, prompt: str) -> None:
        flat = self._flat(prompt)
        assert flat.count(self._EXPOSURE_KEY) == 1, "the exposure rule must be stated exactly once"
        assert "apply it from now until the deadline" in flat
        assert "treating the elapsed event-free part of the window as observed" in flat
        # The retired constant's phrasings must not come back beside the surviving statement.
        assert "outside-view rates apply to the exposure" not in flat
        assert "the two must be disjoint" not in flat
        assert "remove that path's own instances from the rate before combining" not in flat

    def test_binary_prompt_states_remaining_exposure_once_inside_the_hazard_bullet(self) -> None:
        prompt = self._binary()
        self._assert_remaining_exposure_once(prompt)
        # Folded INTO the conditional-hazard bullet, which is the same rule for recurring events.
        flat = self._flat(prompt)
        bullet_at = flat.index(self._EXPOSURE_KEY)
        window = flat[bullet_at : bullet_at + 900]
        # The check keeps its label: without it the trailing skip token names something the
        # bullet never introduced, and "Otherwise write ..." reads as scoping over the
        # exposure sentence, which applies to every rate-based question and is never skipped.
        assert "conditional-hazard check: for a recurring event with a history of inter-arrival gaps" in window
        assert "no event in the t days already elapsed" in window
        assert "non-recurring, conditional-hazard skipped" in window

    def test_multiple_choice_prompt_states_remaining_exposure_once(self) -> None:
        self._assert_remaining_exposure_once(self._mc())

    def test_binary_union_line_requires_disjoint_paths(self) -> None:
        """The union instruction itself is what invited the qid 43837 double count, so
        the disjointness requirement lives where the union is computed — and only there,
        since MC never computes a union."""
        flat = self._flat(self._binary())
        assert "1 - product of (1-p_i)" in flat
        assert "union only over paths that cannot be the same event" in flat
        assert "union only over paths" not in self._flat(self._mc())

    # -- anchor adherence (binary, MC): one rule, with a size ----------------

    _ANCHOR_SIZE = "more than about 15 points"

    def _assert_single_anchor_rule_with_size(self, prompt: str) -> None:
        flat = self._flat(prompt)
        assert "anchor on your math" in flat
        assert flat.count(self._ANCHOR_SIZE) == 1, "the 15-point size belongs to exactly one bullet"
        assert "named, specific" in flat
        # The retired `_ANCHOR_CONSISTENCY_RULE` phrasings.
        assert "state the outside-view number you computed" not in flat
        assert "more than about 15 percentage points" not in flat
        assert "do not move off your own number on a general feeling" not in flat
        assert "history counsels caution" not in flat

    def test_binary_prompt_has_one_anchor_rule_sized_at_15_points(self) -> None:
        self._assert_single_anchor_rule_with_size(self._binary())

    def test_multiple_choice_prompt_has_one_anchor_rule_sized_at_15_points(self) -> None:
        self._assert_single_anchor_rule_with_size(self._mc())

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
        the numeric prompt too — unlike the exposure rule, which is probability-shaped."""
        self._assert_count_in_period(self._numeric())

    # -- placement + scope -------------------------------------------------

    def test_outside_view_rules_sit_in_the_outside_view_phase(self) -> None:
        """Both outside-view rules must land inside PHASE 1, before the inside-view update,
        so the model reads them while computing the number. The anchor size rides the
        final-rationale bullet in PHASE 2, where the final number is written."""
        prompt = self._binary()
        phase1_at = prompt.index("PHASE 1: OUTSIDE VIEW")
        phase2_at = prompt.index("PHASE 2: INSIDE VIEW UPDATE")
        for phrase in ("For questions asking how many events", "Rates apply to the exposure that REMAINS"):
            assert phase1_at < prompt.index(phrase) < phase2_at, f"{phrase!r} outside PHASE 1"
        assert prompt.index("Anchor on your math") > phase2_at

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
            flat = self._flat(prompt)
            assert self._EXPOSURE_KEY not in flat
            assert self._ANCHOR_SIZE not in flat
            assert "anchor on your math" not in flat
            assert "how many events of a kind occur in a period" not in flat

    def test_numeric_prompt_does_not_carry_the_probability_shaped_rules(self) -> None:
        """Scope: the exposure rule and the 15-point size are worded in probabilities, while
        the numeric prompt anchors on a range. Only the reference-class rule ships there."""
        flat = self._flat(self._numeric())
        assert self._EXPOSURE_KEY not in flat
        assert self._ANCHOR_SIZE not in flat


class TestSoftClockAndHistoryDischargedRules:
    """The two Phase 1 rules the 2026-09-02 failure-mode audit produced
    (scratch/failure_mode_audit_2026-09-02/AUDIT_SYNTHESIS.md; Items A and C of
    scratch_docs_and_planning/announced_unscheduled_fix_plan_2026-09-02.md).

    `_SOFT_CLOCK_RULE` (lens A, the announced-but-unscheduled event). On "will X happen before
    D" questions whose only route to X is a target date the responsible actor has ANNOUNCED but
    is not BOUND to, members decomposed P(target lands in window) x P(X | target) and set the
    first term near 1 because the target had been announced. 52 of 815 STRICT records (6.4%;
    8.3% of binaries; coder kappa 0.74); on the 37 flagged binaries the bot published a mean 0.44
    for events that happened 8% of the time, 13 records above 0.5 resolved NO and none went the
    other way, and flagged records score 18.7 spot-peer points worse (95% CI 5.9 to 33.4).
    Receipts 43837 / 44424 / 44557; the contrast is 45217, where a statutory clock existed,
    members computed the date and scored +45. The "measured record of meeting" carve-out is
    load-bearing: on qid 42305 a weekly bulletin with a measured publication lag WAS a binding
    clock and a near-1 timing term was right. It supersedes the two 2026-09-02 rules Item B
    retired, and it deliberately adds no structured-block field.

    `_HISTORY_DISCHARGED_RULE` (lens C, history repeats past an acknowledged change). A member
    names a reason the historical cadence has been discharged and keeps it as the center anyway:
    12.1% of rationales, about 7 spot per flagged record (95% CI 2.7 to 12.2), the pattern failed
    in 13 of 13 live-triple fires; coder agreement 0.59 and partly hindsight-contaminated, so
    upper bounds. Shipped on the plan's recommendation with the operator's final say pending.

    Both ship to binary and MC only and sit in Phase 1 beside the reference-class bullets."""

    _RUBRIC = "Strong: multiple independent sources"
    _COUNT_IN_PERIOD = "For questions asking how many events"
    _SOFT_CLOCK_OPENER = "A target date the responsible actor has not bound itself to"
    _HISTORY_OPENER = "If your own analysis names a reason the historical cadence has been discharged"

    @staticmethod
    def _flat(text: str) -> str:
        # Collapse whitespace: the constants are pre-indented for clean_indents, so assertions
        # must not depend on where the lines wrap.
        return " ".join(text.lower().split())

    def _binary(self) -> str:
        return binary_prompt(_binary_q(), research="r")

    def _mc(self) -> str:
        return multiple_choice_prompt(_mc_q(), research="r")

    def _numeric(self) -> str:
        return numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")

    def _stacked(self) -> list[str]:
        return [
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

    def _assert_verbatim_once(self, prompt: str, constant: str) -> None:
        flat_prompt = self._flat(prompt)
        flat_constant = self._flat(constant)
        assert flat_constant in flat_prompt, "the rule must land verbatim (modulo line wrapping)"
        assert flat_prompt.count(flat_constant) == 1, "the rule must be stated exactly once"

    # -- presence, verbatim, once (binary and MC) -----------------------------

    def test_binary_prompt_carries_the_soft_clock_rule_verbatim(self) -> None:
        self._assert_verbatim_once(self._binary(), _SOFT_CLOCK_RULE)

    def test_multiple_choice_prompt_carries_the_soft_clock_rule_verbatim(self) -> None:
        self._assert_verbatim_once(self._mc(), _SOFT_CLOCK_RULE)

    def test_binary_prompt_carries_the_history_discharged_rule_verbatim(self) -> None:
        self._assert_verbatim_once(self._binary(), _HISTORY_DISCHARGED_RULE)

    def test_multiple_choice_prompt_carries_the_history_discharged_rule_verbatim(self) -> None:
        self._assert_verbatim_once(self._mc(), _HISTORY_DISCHARGED_RULE)

    # -- wording pins: each load-bearing clause, by name ----------------------

    def test_soft_clock_rule_keeps_every_load_bearing_clause(self) -> None:
        """Each clause earns its place: the carve-out keeps a measured cadence (q42305) from
        being read as soft; "as its own number" is the move the audit measured; the list of
        things that do NOT raise it is the q44557 / q43837 evidence class; "say which clock"
        is what q45217's members did; the parenthetical is the receipt (principle 3 of the fix
        plan: a receipted rule carries its reason)."""
        flat = self._flat(_SOFT_CLOCK_RULE)
        assert "no statute, no contract, no published schedule it has a measured record of meeting" in flat
        assert "is evidence that a target exists, not that it will hold" in flat
        assert "price the probability that the target lands inside the question window as its own number" in flat
        assert "derived from that actor's record of slips and scrubs for this kind of event" in flat
        assert "an announcement, plan, tracker page or partner page does not raise it" in flat
        assert "where a binding clock exists, compute the date from it and say which clock" in flat
        assert "forecasts averaged 44% on events that happened 8% of the time" in flat

    def test_history_discharged_rule_is_conditional_on_the_members_own_acknowledgment(self) -> None:
        """The condition is the member's OWN written acknowledgment, so the rule cannot fire on
        a question where nothing has changed; and the cadence becomes a BOUND, not a center,
        which is the whole correction. The receipt rides as a short parenthetical."""
        flat = self._flat(_HISTORY_DISCHARGED_RULE)
        assert flat.startswith("• if your own analysis names a reason the historical cadence has been discharged")
        assert "(its driver was met, the deadline passed, the rule changed)" in flat
        assert "that cadence is a bound on your estimate, not its center" in flat
        assert "state the post-change estimate and what it rests on" in flat
        assert "held in 0 of 13 recent cases" in flat

    def test_rules_do_not_revive_the_retired_anchor_consistency_wording(self) -> None:
        """Item B retired "do not move off your number when history counsels caution" because it
        pulled against exactly this correction; the two new rules must not smuggle it back."""
        for prompt in (self._binary(), self._mc()):
            flat = self._flat(prompt)
            assert "history counsels caution" not in flat
            assert "do not move off your own number on a general feeling" not in flat

    # -- placement: Phase 1, after the reference-class bullets, before the rubric ----

    @pytest.mark.parametrize(
        ("build", "timeframe_step"),
        [
            pytest.param(lambda self: self._binary(), "3) Timeframe reasoning", id="binary"),
            pytest.param(lambda self: self._mc(), "(3) Timeframe reasoning", id="multiple_choice"),
        ],
    )
    def test_rules_sit_in_the_reference_class_step_of_phase_1(self, build, timeframe_step: str) -> None:
        """Both rules are about how to read a base rate you just computed, so they land inside
        the reference-class step (after `_COUNT_IN_PERIOD_REFERENCE_CLASS`, before the timeframe
        step), inside PHASE 1 and before the Strong/Moderate/Weak evidence rubric in PHASE 2.
        Soft-clock first, history-discharged directly after it."""
        prompt = build(self)
        phase1_at = prompt.index("PHASE 1: OUTSIDE VIEW")
        count_at = prompt.index(self._COUNT_IN_PERIOD)
        soft_clock_at = prompt.index(self._SOFT_CLOCK_OPENER)
        history_at = prompt.index(self._HISTORY_OPENER)
        timeframe_at = prompt.index(timeframe_step)
        rubric_at = prompt.index(self._RUBRIC)
        phase2_at = prompt.index("PHASE 2: INSIDE VIEW UPDATE")
        assert phase1_at < count_at < soft_clock_at < history_at < timeframe_at < phase2_at < rubric_at

    # -- scope: not numeric, not stacking, no block field ---------------------------

    def test_numeric_prompt_does_not_carry_the_rules(self) -> None:
        """Both rules are worded in probabilities over an event-by-deadline question; the
        numeric prompt anchors on a range and gets neither."""
        flat = self._flat(self._numeric())
        assert self._flat(self._SOFT_CLOCK_OPENER) not in flat
        assert self._flat(self._HISTORY_OPENER) not in flat
        assert "measured record of meeting" not in flat
        assert "historical cadence has been discharged" not in flat

    def test_stacking_prompts_do_not_carry_the_rules(self) -> None:
        """Same scope guard as every other base-prompt rule: stacking is prod-disabled."""
        for prompt in self._stacked():
            flat = self._flat(prompt)
            assert self._flat(self._SOFT_CLOCK_OPENER) not in flat
            assert self._flat(self._HISTORY_OPENER) not in flat
            assert "measured record of meeting" not in flat
            assert "historical cadence has been discharged" not in flat

    def test_rules_add_no_structured_block_field(self) -> None:
        """The schema audit rejected a `target_holds_probability` / `clock_type` slot: the block
        is written after the forecast is fixed, so a field there cannot scaffold the reasoning,
        and the number lives in the rationale. The example blocks stay exactly the forecast."""
        assert set(json.loads(_extract_last_json_block(self._binary()))) == {"question_type", "posterior_prob"}
        assert set(json.loads(_extract_last_json_block(self._mc()))) == {"question_type", "option_probs"}
        for constant in (_SOFT_CLOCK_RULE, _HISTORY_DISCHARGED_RULE):
            assert "target_holds_probability" not in constant
            assert "clock_type" not in constant

    def test_rules_keep_the_base_prompts_mode_agnostic(self) -> None:
        """The benchmarking leakage guard lives on the research side; the forecaster prompts stay
        mode-agnostic, so neither rule may name a market venue or crowd source."""
        for constant in (_SOFT_CLOCK_RULE, _HISTORY_DISCHARGED_RULE):
            for venue in ("Polymarket", "Kalshi", "Manifold", "PredictIt", "Metaculus", "CME FedWatch"):
                assert venue not in constant


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
        prompt = self._analyzer()
        flat = " ".join(prompt.lower().split())
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
        flat = " ".join(self._analyzer().lower().split())
        assert "a different fact from the question's own resolution threshold" in flat

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
        that route around the clauses are explicitly forbidden. Kept through the
        2026-09 de-bloat by operator decision, with its reason attached: the
        criteria stay consumed as constraints."""
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
        assert "so the criteria stay consumed as constraints" in lowered
        assert "if none applies, stay at the product" in lowered

    def test_clause_product_is_named_as_the_anchor_on_multi_clause_questions(self) -> None:
        """Three rules used to say how the final number may differ from a computed one and
        none said WHICH computation anchors when a Step-2 base rate and a 5b clause product
        both exist. One sentence where the product is computed now says: the product, because
        it is more specific than the step-2 base rate, which is the comparand a bare "more
        specific" left the model to infer; the three valid moves are the ways to leave it."""
        prompt = binary_prompt(_binary_q(), research="r")
        lowered = " ".join(prompt.lower().split())
        assert 'this product is the number the "anchor on your math" check in step 6 anchors to' in lowered
        assert "because it is more specific than the step-2 base rate" in lowered
        # The anchor bullet names the product among the computations it covers.
        assert "clause product" in lowered
        # And the sentence sits in 5b, before the reconciliation bullet.
        assert prompt.index("more specific than the step-2 base rate") < prompt.index("exactly three valid moves")


class TestTemplateStatesEachCheckOnce:
    """The binary and MC templates said several things twice: an odds check and a small-delta
    check that both asked how a ±10% shift would sit; a trailing "Brief checklist" whose items
    re-asked for outputs the template had already produced (0b paraphrases the criteria, step 2
    states the base rate, step 4 lists the evidence, step 5 red-teams). The two checklist items
    with no template twin, the bait-and-switch check and the consistency line, moved INTO the
    template as its final numbered step: an instruction that shapes the answer's structure
    belongs in the template, not in a post-hoc reminder. MC also carried three statements of
    "sum to 100%", two of them in integer percent while the schema demands decimals summing to
    1.0; the schema line is the one the parser reads, so it is the one that stays. The
    every-option floor was on the percent scale too ("a probability (1-99%)", "at least 1%") and
    now interpolates MC_PROB_MIN, since a percent-scale option_probs is hard-rejected and drops
    the whole ballot to the paid LLM salvage rung (q44558)."""

    @staticmethod
    def _flat(prompt: str) -> str:
        return " ".join(prompt.lower().split())

    def _binary(self) -> str:
        return binary_prompt(_binary_q(), research="r")

    def _mc(self) -> str:
        return multiple_choice_prompt(_mc_q(), research="r")

    def _numeric(self) -> str:
        return numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")

    def test_final_checks_is_the_last_template_step_in_every_base_prompt(self) -> None:
        for prompt in (self._binary(), self._mc(), self._numeric()):
            flat = self._flat(prompt)
            assert flat.count("bait-and-switch check") == 1
            assert "brief checklist" not in flat
            # Retired checklist echoes of template outputs.
            assert "paraphrase the resolution criteria" not in flat
            assert "paraphrase options" not in flat
            assert "top 3-5 evidence items" not in flat
            assert "top 3 to 5 evidence items" not in flat
            assert "blind-spot scenario most likely" not in flat
            assert "blind-spot statement" not in flat
            assert "blind spot scenario and expected effect" not in flat
            assert "state the outside-view base rate you anchored to" not in flat
            assert "state the outside-view distribution used as anchor" not in flat
            assert "state the outside view baseline used" not in flat
            # Placement: inside the template, after the final-rationale step, before the block.
            final_checks_at = flat.index("final checks")
            assert flat.index("final rationale") < final_checks_at < flat.index("structured forecast")

    def test_binary_final_checks_keep_the_consistency_line(self) -> None:
        flat = self._flat(self._binary())
        assert 'consistency line: "x out of 100 times, [criteria] happens." sensible?' in flat

    def test_mc_final_checks_keep_the_consistency_line(self) -> None:
        flat = self._flat(self._mc())
        assert "most likely: __; least likely: __; coherent with rationale?" in flat

    def test_numeric_final_checks_keep_units_and_the_percentile_consistency_line(self) -> None:
        """Numeric adds the units bullet: the unit-mismatch guard withholds a forecaster that gets
        the units wrong, so it is the one checklist item with a pipeline consequence."""
        flat = self._flat(self._numeric())
        final_checks = flat[flat.index("final checks") :]
        assert "units: what are the units of the output values and why?" in final_checks
        assert "which percentile corresponds to the status quo or trend" in final_checks

    def test_numeric_step_seven_asks_for_a_central_estimate_not_a_probability(self) -> None:
        """ "My base rate was X% ... moving to Y%" was a probability template copy-pasted onto a
        distribution question (the sibling numeric odds check was cut for the same reason)."""
        flat = self._flat(self._numeric())
        assert (
            "state your outside-view central estimate and range, then say what the current evidence moved and why"
            in flat
        )
        assert "my base rate was x%" not in flat
        assert "odds check" not in flat

    def test_numeric_outcome_type_is_defined_once_in_the_schema_notes(self) -> None:
        """The field was defined three times (step 9, schema notes, example). The schema note is
        the definition; step 9 is a one-line pointer to it."""
        flat = self._flat(self._numeric())
        assert "outcome type classification" not in flat
        assert flat.count("counts, rankings, number of events") == 1
        assert "record it in `outcome_type`" in flat
        assert "definition in the schema notes" in flat

    def test_odds_and_delta_are_one_check(self) -> None:
        for prompt in (self._binary(), self._mc()):
            flat = self._flat(prompt)
            assert flat.count("odds and delta check") == 1
            assert "odds check:" not in flat
            assert "small-delta check" not in flat
            assert "9:1" in flat

    def test_mc_states_the_sum_constraint_once_in_decimals(self) -> None:
        prompt = self._mc()
        flat = self._flat(prompt)
        assert "sum to 1.0" in flat
        assert "sum to 100" not in flat
        assert "use integers" not in flat
        assert "remember:" not in flat
        # The every-option requirement is extraction-critical and stays, but on the DECIMAL
        # scale: a percent-scale option_probs is hard-rejected by _check_option_probs and
        # cannot be repaired, so the whole ballot drops to the paid LLM salvage rung.
        assert "you must assign a probability to every single option" in flat
        assert "assign it at least 0.01" in flat
        assert "1-99%" not in flat


class TestKeptScaffoldingBullets:
    """Six bullets the 2026-09-02 de-bloat deliberately KEPT as reasoning scaffolding but
    that no other test named, so a later cut could have taken any of them and stayed green.
    AGENTS.md claims every surviving rule has a presence pin on its wording; these are the
    pins that make that claim true. Each phrase is the bullet's own opening clause, so a
    reword lands here rather than in a vague substring."""

    @staticmethod
    def _flat(prompt: str) -> str:
        return " ".join(prompt.lower().split())

    def _binary(self) -> str:
        return binary_prompt(_binary_q(), research="r")

    def _mc(self) -> str:
        return multiple_choice_prompt(_mc_q(), research="r")

    def _numeric(self) -> str:
        return numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")

    def test_binary_worked_examples_keep_their_reason(self) -> None:
        """Step 0b's meta-justification is an operator keep: it tells the model WHY it is
        writing two worked examples, which is what stops them becoming a paraphrase."""
        assert (
            "this is mechanical bait-and-switch protection: it forces the resolution criteria to be "
            "consumed as structured constraints rather than treated as a prose paraphrase" in self._flat(self._binary())
        )

    def test_binary_and_numeric_keep_the_question_specific_base_rate_bullet(self) -> None:
        assert (
            "question-specific base rate: the relevant base rate is the historical frequency for questions "
            "like this one" in self._flat(self._binary())
        )
        assert (
            "question-specific base rate: anchor on the historical frequency, trend, or variance for this "
            "specific indicator" in self._flat(self._numeric())
        )

    def test_binary_keeps_the_trajectory_check(self) -> None:
        """Kept in binary and removed from numeric, where it restated step-3 trend
        continuation. The numeric absence is pinned by
        test_numeric_status_quo_derivation_is_the_one_anchor_to_latest_statement."""
        flat = self._flat(self._binary())
        assert (
            'trajectory check: consider whether the "status quo" means "nothing changes" or '
            '"the current trajectory reaches its natural conclusion"' in flat
        )
        assert "justify predictions that diverge from the most likely trajectory" in flat

    def test_mc_keeps_the_blind_spot_and_calibration_audit_bullets(self) -> None:
        flat = self._flat(self._mc())
        assert (
            "blind-spot consideration: if the resolution is unexpected, what would likely be the reason, "
            "and how should that affect confidence spreads?" in flat
        )
        assert "calibration audit: if one option is genuinely dominant, commit to it" in flat

    def test_numeric_keeps_the_small_delta_check(self) -> None:
        """Numeric's own delta check survives the binary/MC odds-and-delta merge: there is no
        odds check on a distribution question for it to merge with."""
        assert "small delta check: would +/- 10 percent on key percentiles still fit the reasoning?" in self._flat(
            self._numeric()
        )


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
