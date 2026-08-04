"""The ranking stage: the prompt, the parser, and the deterministic fail-open.

``test_order_is_preserved_exactly`` is the regression guard for the whole port. The defect
being fixed is that four per-venue keep/drop calls cannot say which kept row matters most, so
something downstream invented an order — a round-robin venue-fairness pass — and that
invention evicted 43 of 58 wanted rows. Any code that re-sorts, re-scores or "improves" the
model's output re-introduces it.

Second in importance is the ``[]``-versus-fail-open distinction. An empty array is the model
saying nothing here bears on the question, which it did on the two true negatives; conflating
it with a failed call would render 8 rows of near-misses on exactly the questions where the
right answer is silence, and would delete the adaptive-width mechanism the design rests on.
"""

from __future__ import annotations

import pytest

from metaculus_bot.research.market_retrieval.ranking import (
    RENDER_BUDGET,
    RULES_CHARS,
    TIER_UNSPECIFIED,
    TIERS,
    WHY_CHARS,
    Pick,
    RankerQuestion,
    RankingUnusable,
    apply_picks,
    build_ranker_prompt,
    fail_open_slate,
    parse_ranking,
    render_candidate_line,
)
from metaculus_bot.research.market_retrieval.types import MarketMatch, SettlementSource
from tests.test_market_retrieval_generation import Platform

QUESTION = RankerQuestion(
    title="Will US unemployment exceed 4.5% in June 2026?",
    qtype="binary",
    unit="percent",
    resolution_criteria="Resolves per the BLS U-3 release.",
    fine_print="Revisions do not count.",
)


def _row(
    platform: Platform = "kalshi",
    *,
    title: str = "A market",
    market_id: str = "M-1",
    rules: str = "",
    sub_title: str = "",
    resolved: bool = False,
    sources: tuple[SettlementSource, ...] = (),
    volume: float | None = None,
    bettors: int | None = None,
) -> MarketMatch:
    return MarketMatch(
        platform=platform,
        market_title=title,
        market_url=f"https://example.test/{market_id}",
        implied_prob_yes=0.5,
        bid=None,
        ask=None,
        spread=None,
        volume_24h=None,
        close_time=None,
        is_resolved=resolved,
        match_confidence=1.0,
        raw_rules=rules,
        total_volume=volume,
        num_bettors=bettors,
        venue_market_id=market_id,
        sub_title=sub_title,
        settlement_sources=sources,
    )


def _pool(n: int, platform: Platform = "kalshi") -> list[MarketMatch]:
    return [_row(platform, title=f"Market {i}", market_id=f"{platform}-{i}") for i in range(n)]


class TestParseRanking:
    def test_an_empty_array_is_a_valid_answer(self) -> None:
        """The whole adaptive-width mechanism. It must NOT raise, because raising is what
        triggers fail-open, and a fail-open on a true negative renders 8 near-misses."""
        assert parse_ranking("[]", 10) == []

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "I could not find any relevant markets.",
            '{"i": 0}',
            "[unclosed",
        ],
    )
    def test_unreadable_output_fails_open(self, text: str) -> None:
        with pytest.raises(RankingUnusable):
            parse_ranking(text, 10)

    def test_an_array_wrapped_in_an_object_is_still_read(self) -> None:
        """Leniently, and on purpose: the widest bracket pair is extracted, so a model that
        wraps its answer does not cost the question its whole ranking."""
        assert [pick.index for pick in parse_ranking('{"picks": [{"i": 0}, {"i": 2}]}', 10)] == [0, 2]

    def test_order_is_preserved_exactly(self) -> None:
        """THE regression guard: nothing may re-sort the model's ranking, not by tier, not by
        index, not by venue."""
        text = """[
          {"i": 7, "tier": "weak", "why": "loose correlate"},
          {"i": 0, "tier": "same_quantity_same_date", "why": "same series, same month"},
          {"i": 3, "tier": "driver_or_consequence", "why": "a driver"}
        ]"""

        picks = parse_ranking(text, 10)

        assert [pick.index for pick in picks] == [7, 0, 3]
        assert [pick.tier for pick in picks] == ["weak", "same_quantity_same_date", "driver_or_consequence"]

    def test_out_of_range_indices_drop(self) -> None:
        """A hallucinated index names no real market, so there is nothing to render for it."""
        picks = parse_ranking('[{"i": 0}, {"i": 99}, {"i": -1}, {"i": 2}]', 3)

        assert [pick.index for pick in picks] == [0, 2]

    def test_duplicate_indices_collapse_to_the_first_occurrence(self) -> None:
        """First occurrence is the best-ranked one, so keeping the later duplicate would
        silently demote the row."""
        text = '[{"i": 1, "why": "first"}, {"i": 1, "why": "second"}]'

        picks = parse_ranking(text, 5)

        assert [(pick.index, pick.why) for pick in picks] == [(1, "first")]

    def test_an_unknown_tier_becomes_unspecified_without_dropping_the_row(self) -> None:
        """Recall-first: a typo in a label is not evidence the market is irrelevant."""
        picks = parse_ranking('[{"i": 0, "tier": "sort_of_related", "why": "x"}, {"i": 1}]', 5)

        assert [pick.tier for pick in picks] == [TIER_UNSPECIFIED, TIER_UNSPECIFIED]
        assert [pick.index for pick in picks] == [0, 1]

    def test_every_documented_tier_survives_verbatim(self) -> None:
        text = "[" + ", ".join(f'{{"i": {i}, "tier": "{tier}"}}' for i, tier in enumerate(TIERS)) + "]"

        assert [pick.tier for pick in parse_ranking(text, len(TIERS))] == list(TIERS)

    def test_output_truncates_at_the_budget(self) -> None:
        text = "[" + ", ".join(f'{{"i": {i}}}' for i in range(20)) + "]"

        picks = parse_ranking(text, 20)

        assert len(picks) == RENDER_BUDGET == 8
        assert [pick.index for pick in picks] == list(range(8))

    def test_a_fenced_array_parses(self) -> None:
        assert [pick.index for pick in parse_ranking('```json\n[{"i": 2}]\n```', 5)] == [2]

    def test_prose_around_the_array_is_tolerated(self) -> None:
        assert [pick.index for pick in parse_ranking('Here you go:\n[{"i": 1}]\nHope that helps.', 5)] == [1]

    def test_the_why_phrase_is_capped(self) -> None:
        picks = parse_ranking('[{"i": 0, "why": "' + "x" * 400 + '"}]', 1)

        assert len(picks[0].why) == WHY_CHARS

    def test_entries_without_a_usable_index_are_skipped(self) -> None:
        picks = parse_ranking('[{"tier": "weak"}, "not an object", {"i": "abc"}, {"i": 1}]', 5)

        assert [pick.index for pick in picks] == [1]

    def test_an_empty_pool_drops_every_index(self) -> None:
        assert parse_ranking('[{"i": 0}]', 0) == []


class TestApplyPicks:
    def test_rows_are_stamped_in_the_models_order(self) -> None:
        pool = _pool(5)
        picks = [Pick(index=3, tier="weak", why="third"), Pick(index=0, tier="same_quantity_same_date", why="best")]

        rows = apply_picks(pool, picks)

        assert [row.venue_market_id for row in rows] == ["kalshi-3", "kalshi-0"]
        assert [row.rank for row in rows] == [0, 1]
        assert [row.relation_tier for row in rows] == ["weak", "same_quantity_same_date"]
        assert [row.relevance_label for row in rows] == ["third", "best"]

    def test_the_pool_rows_are_not_mutated(self) -> None:
        """Provider-health reads field presence off the POOL after ranking, and a rendered row
        is also a pool row."""
        pool = _pool(2)

        apply_picks(pool, [Pick(index=0, tier="weak", why="w")])

        assert pool[0].rank is None
        assert pool[0].relation_tier == ""


class TestFailOpenSlate:
    def test_the_slate_is_the_head_of_what_the_model_was_shown(self) -> None:
        """A fail-open must be a TRUNCATION of the input, not a different pipeline — which is
        also why the prompt presents candidates in pool order."""
        pool = _pool(20)

        slate = fail_open_slate(pool)

        assert [row.venue_market_id for row in slate] == [f"kalshi-{i}" for i in range(RENDER_BUDGET)]
        assert [row.rank for row in slate] == list(range(RENDER_BUDGET))

    def test_the_slate_carries_no_tier_or_label(self) -> None:
        """An unranked slate must not present as a confident ranking: the empty tier is what
        selects the neutral preamble and suppresses the strong header downstream."""
        slate = fail_open_slate(_pool(3))

        assert {row.relation_tier for row in slate} == {""}
        assert {row.relevance_label for row in slate} == {""}

    def test_an_empty_pool_yields_an_empty_slate(self) -> None:
        assert fail_open_slate([]) == []


class TestCandidateLine:
    def test_the_segments_appear_in_the_documented_order(self) -> None:
        row = _row(
            title="US unemployment rate",
            sub_title="Above 4.5%",
            rules="Resolves per BLS U-3.",
            resolved=True,
            sources=(SettlementSource(name="Bureau of Labor Statistics", url="https://bls.gov"),),
            volume=100_000.0,
        )
        row.close_time = __import__("datetime").datetime(2026, 6, 30, tzinfo=__import__("datetime").timezone.utc)

        line = render_candidate_line(4, row)

        assert line.startswith("[4] (kalshi) US unemployment rate | Above 4.5% | closes: 2026-06-30 | RESOLVED |")
        segments = [segment.strip() for segment in line.split("|")]
        assert segments[4].startswith("liquidity: deep")
        assert segments[5] == "settles via: Bureau of Labor Statistics"
        assert segments[6] == "rules: Resolves per BLS U-3."

    def test_empty_segments_are_omitted(self) -> None:
        line = render_candidate_line(0, _row(title="Bare row"))

        assert line == "[0] (kalshi) Bare row | liquidity: no-liquidity-data"

    def test_a_sub_title_equal_to_the_title_is_omitted(self) -> None:
        line = render_candidate_line(0, _row(title="Same", sub_title="Same"))

        assert line.count("Same") == 1

    def test_only_the_first_three_settlement_sources_render(self) -> None:
        """Kalshi lists as many as a dozen and the tail is boilerplate."""
        sources = tuple(SettlementSource(name=f"Source {i}", url=f"https://s{i}.test") for i in range(10))

        line = render_candidate_line(0, _row(sources=sources))

        assert "settles via: Source 0; Source 1; Source 2 |" in line + " |"
        assert "Source 3" not in line

    def test_a_long_settlement_source_name_is_capped(self) -> None:
        sources = (SettlementSource(name="N" * 500, url="https://s.test"),)

        line = render_candidate_line(0, _row(sources=sources))

        assert "N" * 90 in line
        assert "N" * 91 not in line

    def test_rules_are_capped_per_venue(self) -> None:
        """Set at measured distributions: Kalshi's p90 is 174 chars so 700 truncates nothing
        real, while Polymarket's median description is 900 and is the biggest token lever."""
        for platform in ("kalshi", "polymarket", "manifold"):
            line = render_candidate_line(0, _row(platform, rules="r" * 5000))  # type: ignore[arg-type]
            rendered = line.split("rules: ", 1)[1]
            assert len(rendered) == RULES_CHARS[platform]

    def test_predictit_omits_the_constant_liquidity_segment(self) -> None:
        """The label is a constant for this venue, so repeating it on ~197 candidate lines
        spends prompt tokens carrying zero information. The venue-block header says it once."""
        line = render_candidate_line(0, _row("predictit", rules="contracts: A, B"))

        assert "liquidity:" not in line
        assert "no-liquidity-data" not in line

    def test_a_manifold_row_still_carries_its_liquidity_label(self) -> None:
        line = render_candidate_line(0, _row("manifold", bettors=5))

        assert "liquidity: thin" in line


class TestRankerPrompt:
    def test_the_operator_relevance_rule_is_present(self) -> None:
        prompt = build_ranker_prompt(QUESTION, _pool(2))

        assert "THE RELEVANCE RULE YOU ARE APPLYING, stated by the operator this list is built for:" in prompt
        assert "Recall matters far more than precision." in prompt
        assert "a related driver, consequence, or sibling contest IS" in prompt
        assert "included markets are a good trade for one wrongly excluded one." in prompt

    def test_the_sibling_contest_example_is_a_keep_not_a_drop(self) -> None:
        """The change that recovered an entire question: a pool full of other Florida races is
        exactly what the old DROP clause told the model to exclude, and it returned an empty
        array in both replicates."""
        prompt = build_ranker_prompt(QUESTION, _pool(2))

        assert "different OFFICE" not in prompt
        assert "a different RACE's election result" not in prompt
        assert "A SIBLING CONTEST in the same electorate and the same cycle belongs here too" in prompt

    def test_the_different_entity_drop_examples_survive(self) -> None:
        """These are what keep Mexico's unemployment rate out of an Australia question; the arm
        retained 0 of 39 labelled no-bearing rows with them in place."""
        prompt = build_ranker_prompt(QUESTION, _pool(2))

        assert "a different COUNTRY's version of the same statistic" in prompt
        assert "a different COMPANY's output or production" in prompt

    def test_all_four_tier_names_and_the_three_signals_are_present(self) -> None:
        prompt = build_ranker_prompt(QUESTION, _pool(2))

        for tier in TIERS:
            assert f'"{tier}"' in prompt
        assert "THREE SIGNALS IN THE CANDIDATE BLOCK:" in prompt
        assert "`settles via`" in prompt
        assert "`liquidity` is a QUALITY signal, never a relevance signal." in prompt
        assert "`RESOLVED` means the market has already settled." in prompt

    def test_the_empty_array_is_offered_as_a_valid_answer(self) -> None:
        prompt = build_ranker_prompt(QUESTION, _pool(2))

        assert "Return an EMPTY ARRAY [] if nothing here bears on the question." in prompt
        assert "An empty array is a valid and sometimes correct answer." in prompt

    def test_the_budget_is_substituted_everywhere(self) -> None:
        prompt = build_ranker_prompt(QUESTION, _pool(2))

        assert "{budget}" not in prompt
        assert f"from 0 up to {RENDER_BUDGET} rows" in prompt

    def test_no_placeholder_survives_substitution(self) -> None:
        prompt = build_ranker_prompt(QUESTION, _pool(3))

        for placeholder in ("{title}", "{qtype}", "{unit}", "{rc}", "{fp}", "{candidates}", "{venue_summary}"):
            assert placeholder not in prompt

    def test_the_question_header_is_capped(self) -> None:
        long_question = RankerQuestion(
            title="T", resolution_criteria="c" * 5000, fine_print="f" * 5000, qtype="binary", unit="percent"
        )

        prompt = build_ranker_prompt(long_question, _pool(1))

        assert "c" * 1200 in prompt and "c" * 1201 not in prompt
        assert "f" * 600 in prompt and "f" * 601 not in prompt

    def test_candidates_are_grouped_by_venue_in_pool_order(self) -> None:
        """No interleaving. The grouping is derived from the pool's own order so the prompt and
        the fail-open slate can never disagree."""
        pool = [*_pool(2, "kalshi"), *_pool(2, "polymarket"), *_pool(1, "manifold")]

        prompt = build_ranker_prompt(QUESTION, pool)

        assert prompt.index("-- kalshi (2 candidates) --") < prompt.index("-- polymarket (2 candidates) --")
        assert prompt.index("-- polymarket (2 candidates) --") < prompt.index("-- manifold (1 candidates) --")
        assert "5 candidates from 3 venue(s)" in prompt
        assert "grouped by venue: kalshi 2, polymarket 2, manifold 1" in prompt

    def test_candidate_indices_are_pool_positions(self) -> None:
        """The parser's indices address this list, so a renumbering per venue block would make
        every pick name the wrong market."""
        pool = [*_pool(2, "kalshi"), *_pool(2, "polymarket")]

        prompt = build_ranker_prompt(QUESTION, pool)

        assert "[0] (kalshi)" in prompt
        assert "[2] (polymarket)" in prompt
        assert "[0] (polymarket)" not in prompt

    def test_the_predictit_block_header_states_the_liquidity_fact_once(self) -> None:
        pool = [*_pool(2, "kalshi"), *_pool(3, "predictit")]

        prompt = build_ranker_prompt(QUESTION, pool)

        assert "-- predictit (3 candidates; this venue publishes no liquidity data) --" in prompt
        assert prompt.count("no liquidity data") == 1

    def test_a_venue_with_no_candidates_gets_no_block(self) -> None:
        prompt = build_ranker_prompt(QUESTION, _pool(2, "kalshi"))

        assert "-- polymarket" not in prompt
        assert "2 candidates from 1 venue(s)" in prompt

    def test_a_brace_in_a_question_title_does_not_break_substitution(self) -> None:
        """`.replace` rather than `.format`, because the emitted-object example is literal JSON
        braces and a question title is arbitrary text."""
        prompt = build_ranker_prompt(RankerQuestion(title="Will {x} happen?"), _pool(1))

        assert "Will {x} happen?" in prompt
        assert '{"i": <index>, "tier":' in prompt

    def test_an_empty_pool_still_produces_a_well_formed_prompt(self) -> None:
        prompt = build_ranker_prompt(QUESTION, [])

        assert "0 candidates from 0 venue(s)" in prompt
