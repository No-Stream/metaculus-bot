"""Direct unit tests for prompt-building functions in ``metaculus_bot.prompts``.

These exercise the prompt strings themselves (not the pipelines that call
them). Critical for the benchmarking carve-out: the higher-level gap-fill
tests mock ``_run_analyzer`` entirely, so without these the benchmarking
warning could be deleted from a prompt and no test would catch it —
backtest scores would silently get polluted with prediction-market data.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from metaculus_bot.prompts import (
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

# ---------------------------------------------------------------------------
# gap_fill_analyzer_prompt
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# gap_fill_search_prompt
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Forecasting-window anchor (binary / MC / numeric + stacking variants)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# web_research_prompt
# ---------------------------------------------------------------------------


class TestMcPromptInterpolatesRealOptionNames:
    """Strict parsers (e.g. gemma-4-31b-it) refuse to map literal ``Option_A: NN%``
    placeholders onto real option names in the allowed-list — they correctly
    emit ``<<NOT_FOUND>>`` because the prompt example does not contain anything
    semantically tied to the question's actual options.

    Fix: the example block in both ``multiple_choice_prompt`` and
    ``stacking_multiple_choice_prompt`` must interpolate the real option names
    so the LLM emits text the parser can directly recognize.
    """

    def test_stacking_mc_prompt_emits_real_option_names(self) -> None:
        q = _mc_q()
        q.options = ["Apple", "Banana", "Cherry"]

        result = stacking_multiple_choice_prompt(q, research="r", base_predictions=["a1", "a2"])

        assert "Apple: NN%" in result
        assert "Banana: NN%" in result
        assert "Cherry: NN%" in result

    def test_stacking_mc_prompt_drops_literal_option_a_b_placeholders(self) -> None:
        q = _mc_q()
        q.options = ["Apple", "Banana", "Cherry"]

        result = stacking_multiple_choice_prompt(q, research="r", base_predictions=["a1", "a2"])

        assert "Option_A: NN%" not in result
        assert "Option_B: NN%" not in result
        assert "Option_N: NN%" not in result

    def test_multiple_choice_prompt_emits_real_option_names(self) -> None:
        q = _mc_q()
        q.options = ["Apple", "Banana", "Cherry"]

        result = multiple_choice_prompt(q, research="r")

        assert "Apple: NN%" in result
        assert "Banana: NN%" in result
        assert "Cherry: NN%" in result

    def test_multiple_choice_prompt_drops_literal_option_a_b_placeholders(self) -> None:
        q = _mc_q()
        q.options = ["Apple", "Banana", "Cherry"]

        result = multiple_choice_prompt(q, research="r")

        # Note: the JSON schema example block still uses Option_A/B/C as JSON
        # keys to illustrate the mapping shape; that's fine. What we don't want
        # is the literal "Option_A: NN%" answer-line example, since that's the
        # text the parser actually has to map onto real option names.
        assert "Option_A: NN%" not in result
        assert "Option_B: NN%" not in result
        assert "Option_N: NN%" not in result

    def test_stacking_mc_prompt_preserves_options_in_order(self) -> None:
        """The example answer lines must list options in the same order as
        ``question.options`` — the trailing answer lines downstream rely on
        that ordering to carry into the LLM's output."""
        q = _mc_q()
        q.options = ["Manufacturing PMI higher", "Services PMI higher", "Equal"]

        result = stacking_multiple_choice_prompt(q, research="r", base_predictions=["a1", "a2"])

        idx_mfg = result.find("Manufacturing PMI higher: NN%")
        idx_svc = result.find("Services PMI higher: NN%")
        idx_eq = result.find("Equal: NN%")
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

    def test_prediction_market_nudge_only_when_not_benchmarking(self) -> None:
        """Regression: the existing FOCUS AREAS 'Prediction market odds' bullet
        must still appear only when is_benchmarking=False."""
        non_bench = web_research_prompt("Q?", is_benchmarking=False)
        bench = web_research_prompt("Q?", is_benchmarking=True)

        assert "Prediction market odds" in non_bench
        assert "Prediction market odds" not in bench
