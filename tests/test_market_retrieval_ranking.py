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

import logging

import pytest

from metaculus_bot.research.market_retrieval.generation import RETRIEVAL_WIDTH
from metaculus_bot.research.market_retrieval.ranking import (
    FP_CHARS,
    RC_CHARS,
    RENDER_BUDGET,
    RULES_CHARS,
    SETTLEMENT_SOURCE_CHARS,
    SETTLEMENT_SOURCES_RENDERED,
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
from metaculus_bot.research.market_retrieval.venues import (
    MANIFOLD_ANSWER_TEXT_MAX_CHARS,
    MANIFOLD_TOP_ANSWERS_RENDERED,
)
from tests.test_market_retrieval_generation import Platform

# Chars per token, the measured calibration for this prompt shape. Used instead of a tokenizer
# so the budget test stays dependency-free and deterministic.
CHARS_PER_TOKEN = 3.88

# Ceiling on the ranker prompt at the pool scale the design was MEASURED at: ~100 Kalshi + ~30
# Polymarket + ~15 Manifold (the venue searches return ~10 rows per query and dedup hard, so
# their width-60 ceilings are rarely approached) + the full 197-market PredictIt universe. That
# pool measures 35,875 estimated tokens (2026-08-05, was 35,760 before the multi-outcome answers
# segment: 5 of the 15 Manifold rows now carry one, for +84 tokens), which lands inside the spec's
# own 36.3k-median / 39.4k-worst arithmetic — so 50,000 is ~39% headroom and catches silent
# line-format growth.
#
# For scale, not as the binding constraint: luna's context window is 1,050,000 tokens (verified
# live 2026-08-04), so this prompt is ~3% of it. The ceiling guards OUR line format, not the
# model's limits.
MARKET_RANKER_PROMPT_TOKEN_CEILING = 50_000

# Ceiling on the degenerate case: every per-venue width saturated (100/60/60) AND every field at
# its cap simultaneously. That measures 86,316 estimated tokens — 2.4x the measured-scale pool —
# because the caps sit well above the real distributions on purpose (Kalshi `rules_primary` is
# p50=134 / p90=174 against a 700 cap, so the cap "truncates nothing real" rather than describing
# a typical row). This bound pins the CAPS and the widths themselves, which the measured-scale
# assertion above cannot see.
#
# 90,000 is now only ~4% headroom, down from ~8%: the multi-outcome answers segment costs 3,294
# tokens here (all 60 Manifold rows multi-outcome, 3 answers each at the 60-char cap) against 84
# at measured scale, because the saturation assumption is 20x the real 30% multi-outcome share and
# 5x the real 10-13-char answer texts. The bound was left at 90,000 deliberately rather than
# nudged, so it stays a live tripwire — but it is close enough now that the next raised cap or
# width has to RE-DERIVE it from a measurement, not add slack to clear itself.
#
# Worth knowing before either number is edited: the two assertions catch different regressions. A
# new candidate-line segment moves both. Raising a rules cap or a retrieval width moves only this
# one. And a fully-saturated pool does NOT fit the 50,000 ceiling — that ceiling was sized
# against the measured (unsaturated) pools, which is why it is asserted against one.
MARKET_RANKER_PROMPT_SATURATED_TOKEN_CEILING = 90_000

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
    answers: tuple[tuple[str, float], ...] = (),
) -> MarketMatch:
    return MarketMatch(
        platform=platform,
        market_title=title,
        market_url=f"https://example.test/{market_id}",
        implied_prob_yes=None if answers else 0.5,
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
        top_answers=answers,
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
        """Leniently, and on purpose: the first array of ranking objects is decoded, so a model
        that wraps its answer does not cost the question its whole ranking."""
        assert [pick.index for pick in parse_ranking('{"picks": [{"i": 0}, {"i": 2}]}', 10)] == [0, 2]

    def test_a_bracket_inside_a_narrated_string_does_not_shadow_the_real_array(self) -> None:
        """The array scan walks candidates instead of taking the first `[` it sees, because a
        bracket inside a JSON string literal is a `[` like any other. `[3]` decodes to a list of
        ints, which cannot be a ranking, so the scan advances to the array that holds picks."""
        assert [pick.index for pick in parse_ranking('{"note": "see [3]", "picks": [{"i": 1}]}', 10)] == [1]

    @pytest.mark.parametrize(
        "text",
        [
            '{"excluded": [], "picks": [{"i": 1}, {"i": 4}]}',
            '{"picks": [{"i": 1}, {"i": 4}], "excluded": []}',
        ],
    )
    def test_an_empty_helper_array_does_not_shadow_the_real_picks(self, text: str) -> None:
        """An empty array is a valid ANSWER but it is not evidence there is no better one.

        The scan used to return the first empty array the moment it decoded one, so a wrapped
        completion carrying a helper key rendered ZERO markets whenever that key happened to sort
        before the picks — the two orderings here returned `[]` and `[1, 4]` respectively off the
        same content. Silent, total, and decided by nothing but JSON key order, which is why both
        orderings are pinned rather than just the broken one.
        """
        assert [pick.index for pick in parse_ranking(text, 10)] == [1, 4]

    @pytest.mark.parametrize("text", ["[]", '{"picks": []}', '{"excluded": [], "picks": []}'])
    def test_a_completion_whose_arrays_are_all_empty_is_the_valid_empty_answer(self, text: str) -> None:
        """The other half of the contract, and the reason the fix cannot simply skip empties.

        Deferring an empty array must not DISCARD it: when no dict-bearing array turns up, the
        remembered empty one is the model's "nothing bears on this" and has to parse rather than
        raise, because raising is what triggers fail-open and a fail-open on a true negative
        renders 8 near-misses.
        """
        assert parse_ranking(text, 10) == []

    @pytest.mark.parametrize("literal", ["Infinity", "-Infinity", "1e400"])
    def test_a_non_finite_index_skips_that_entry_and_keeps_its_siblings(self, literal: str) -> None:
        """`json.loads` accepts the bare literals `Infinity` / `-Infinity` and overflowing float
        literals, and `int(inf)` raises OverflowError — which is neither TypeError nor ValueError,
        so it escaped `_rank_pool`'s `except RankingUnusable`, reached the snapshot-level net and
        discarded the WHOLE prediction-market snapshot. One bad entry must cost that entry."""
        picks = parse_ranking(f'[{{"i": {literal}}}, {{"i": 2}}]', 10)

        assert [pick.index for pick in picks] == [2]

    def test_an_index_past_the_int_conversion_cap_fails_open_rather_than_raising_valueerror(self) -> None:
        """CPython caps int-from-string conversion at 4300 digits and raises a BARE ValueError
        from `json.loads` itself — which `json.JSONDecodeError` does not cover, so a narrower
        catch on the decode rung would let it escape exactly as the OverflowError above did.
        Fail-open is the correct outcome: the array never decoded, so there are no picks."""
        with pytest.raises(RankingUnusable):
            parse_ranking('[{"i": ' + "9" * 5000 + "}]", 10)

    def test_a_well_formed_array_followed_by_bracket_bearing_prose_still_parses(self) -> None:
        """The measured weakness of the retired widest-bracket slice: it ran from the first `[`
        to the LAST `]` anywhere in the output, so ordinary narration citing indices made a
        perfectly good ranking unreadable and cost the question its whole snapshot."""
        text = '[{"i": 1}, {"i": 4}]\nExcluded: [3] and [7] as off-topic.'

        assert [pick.index for pick in parse_ranking(text, 10)] == [1, 4]

    def test_an_array_of_bare_indices_is_unusable_rather_than_an_empty_answer(self) -> None:
        """A shape regression, and one the "usable array" rule catches outright: bare ints carry
        no tier and no `i`, so the scan finds no array of ranking objects and fails open to the
        deterministic slate. That is strictly better than the old behaviour, where it returned
        `[]` and was indistinguishable from the model's valid "nothing bears on this"."""
        with pytest.raises(RankingUnusable):
            parse_ranking("[0, 1, 2]", 10)

    def test_a_renamed_index_key_warns_instead_of_passing_as_an_empty_answer(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The residue the fail-open cannot catch. An array of OBJECTS with the wrong key is a
        usable array, so it parses, yields no pick, and reaches the caller as `ok(0)` — byte
        identical to a genuine `[]` in the token, the telemetry line and the render. The WARN is
        the only place the two are distinguishable."""
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.market_retrieval.ranking"):
            assert parse_ranking('[{"index": 0}, {"index": 1}]', 10) == []
        assert any("yielded no usable pick" in message for message in caplog.messages)

    def test_a_genuinely_empty_array_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """The control. `[]` is the adaptive-width mechanism working, so it must stay silent —
        a WARN on every true negative is how an operator learns to ignore the channel."""
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.market_retrieval.ranking"):
            assert parse_ranking("[]", 10) == []
        assert caplog.messages == []

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

    def test_a_multi_outcome_row_carries_its_leading_answers(self) -> None:
        """The one segment on this line that quotes a price, and the only thing a multi-outcome
        row has to offer beyond its title: Manifold publishes no market-level probability for one,
        so without this the ranker sees a bare question and no evidence of what it measures."""
        row = _row(
            "manifold",
            title="How high will the US gas price get in 2026?",
            bettors=47,
            rules="Per AAA.",
            answers=(("Over $4.60", 0.4992), ("Over $4.65", 0.3723), ("Over $4.70", 0.2765)),
        )

        line = render_candidate_line(0, row)

        assert "answers: Over $4.60 50% | Over $4.65 37% | Over $4.70 28%" in line
        segments = [segment.strip() for segment in line.split("|")]
        assert segments.index("answers: Over $4.60 50%") < segments.index("rules: Per AAA.")

    def test_a_binary_row_gains_no_answers_segment(self) -> None:
        """The flip must be invisible on the ~97% of the pool that is not multi-outcome, or every
        Kalshi, Polymarket and PredictIt line pays for it."""
        for platform in ("kalshi", "polymarket", "manifold", "predictit"):
            line = render_candidate_line(0, _row(platform, rules="r"))  # type: ignore[arg-type]

            assert "answers:" not in line

    def test_an_integral_answer_probability_renders_without_a_decimal_tail(self) -> None:
        """A resolved ladder rung prices at exactly 1.0, which is most of the answers on a real
        threshold market — `100%`, not `100.0%`, so the segment stays one glanceable phrase."""
        row = _row("manifold", bettors=47, answers=(("Over $4.00", 1.0), ("Over $4.60", 0.4992)))

        assert "answers: Over $4.00 100% | Over $4.60 50%" in render_candidate_line(0, row)


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


class TestRankerPromptTokenBudget:
    """Validate the counts; don't trust the estimates.

    The ranker prompt is the one place in this port where input size grows with the pool, and it
    grows in two independent ways: the per-candidate LINE FORMAT (a new segment costs ~400 rows
    worth of chars) and the per-venue CAPS AND WIDTHS. Two pools are asserted because a single
    one cannot see both — a measured-scale pool would miss a raised rules cap, and a saturated
    one has so much slack it would miss a new segment.

    The other direction is the RENDERED snapshot, which is budgeted far more tightly
    (~6,000 chars) in tests/test_market_retrieval_rendering.py: it goes to the expensive
    forecaster models, while this prompt goes to a cheap one and is where recall lives.
    """

    def _row_at(
        self,
        platform: Platform,
        *,
        title: str,
        rules: str,
        sub_title: str = "",
        sources: tuple[SettlementSource, ...] = (),
        answers: tuple[tuple[str, float], ...] = (),
    ) -> MarketMatch:
        return _row(
            platform,
            title=title,
            rules=rules,
            sub_title=sub_title,
            sources=sources,
            answers=answers,
            volume=123456.0,
            bettors=250 if platform == "manifold" else None,
        )

    def _maxed_question(self) -> RankerQuestion:
        return RankerQuestion(
            title="T" * 200,
            qtype="numeric",
            unit="percent",
            resolution_criteria="C" * (RC_CHARS * 2),
            fine_print="F" * (FP_CHARS * 2),
        )

    def _measured_scale_pool(self) -> list[MarketMatch]:
        """The pool scale the design was measured at, with realistic per-field content."""
        real_sources = (
            SettlementSource(name="Bureau of Labor Statistics", url="https://bls.gov"),
            SettlementSource(name="The Washington Post", url="https://washingtonpost.com"),
            SettlementSource(name="Reuters", url="https://reuters.com"),
        )
        pool = [
            self._row_at(
                "kalshi",
                title="Will the US unemployment rate be above 4.5% in June 2026?",
                rules="R" * 174,  # measured p90; p50 is 134 and the cap is 700
                sub_title="Above 4.5%",
                sources=real_sources,
            )
            for _ in range(100)
        ]
        pool += [
            self._row_at(
                "polymarket", title="US unemployment above 4.5% in June?", rules="R" * RULES_CHARS["polymarket"]
            )
            for _ in range(30)
        ]
        # 30% of Manifold's labeled-wanted universe is multi-outcome (the share the contractType
        # flip admits), so ~5 of 15 carry an answers segment, at MEASURED answer texts: 10-13 chars
        # each on the committed live fixture, not the 60-char cap.
        pool += [
            self._row_at("manifold", title="Will unemployment top 4.5%?", rules="R" * RULES_CHARS["manifold"])
            for _ in range(10)
        ]
        pool += [
            self._row_at(
                "manifold",
                title="How high will the US unemployment rate get in 2026?",
                rules="R" * RULES_CHARS["manifold"],
                answers=(("Over $4.60", 0.4992), ("$3.80 - $4.19", 0.5083), ("Nov-Dec 2026", 0.445)),
            )
            for _ in range(5)
        ]
        pool += [
            self._row_at(
                "predictit",
                title="Which party will win the 2026 Senate race in Ohio?",
                rules="contracts: " + ", ".join(["Republican Candidate Name"] * 8),
            )
            for _ in range(197)
        ]
        return pool

    def _saturated_pool_at_caps(self) -> list[MarketMatch]:
        """Every width saturated and every field at its cap at once. A bound, not a forecast."""
        maxed_sources = tuple(
            SettlementSource(name="N" * SETTLEMENT_SOURCE_CHARS, url="https://s.test")
            for _ in range(SETTLEMENT_SOURCES_RENDERED)
        )
        pool = [
            self._row_at(
                "kalshi", title="M" * 80, rules="R" * RULES_CHARS["kalshi"], sub_title="S" * 60, sources=maxed_sources
            )
            for _ in range(RETRIEVAL_WIDTH["kalshi"])
        ]
        pool += [
            self._row_at("polymarket", title="M" * 80, rules="R" * RULES_CHARS["polymarket"], sub_title="S" * 60)
            for _ in range(RETRIEVAL_WIDTH["polymarket"])
        ]
        # Every Manifold row multi-outcome with its answers at the cap too — the flip's worst case,
        # since a real pool is ~30% multi-outcome and real answer texts are 10-13 chars.
        maxed_answers = tuple(("A" * MANIFOLD_ANSWER_TEXT_MAX_CHARS, 1.0) for _ in range(MANIFOLD_TOP_ANSWERS_RENDERED))
        pool += [
            self._row_at(
                "manifold",
                title="M" * 80,
                rules="R" * RULES_CHARS["manifold"],
                sub_title="S" * 60,
                answers=maxed_answers,
            )
            for _ in range(RETRIEVAL_WIDTH["manifold"])
        ]
        pool += [
            self._row_at(
                "predictit",
                title="M" * 80,
                rules="contracts: " + ", ".join(["C" * 40] * 8),
                sub_title="S" * 60,
            )
            for _ in range(197)
        ]
        return pool

    def test_the_measured_scale_pool_fits_the_token_ceiling(self) -> None:
        prompt = build_ranker_prompt(self._maxed_question(), self._measured_scale_pool())
        estimated = len(prompt) / CHARS_PER_TOKEN

        assert estimated < MARKET_RANKER_PROMPT_TOKEN_CEILING, (
            f"ranker prompt grew to ~{estimated:,.0f} estimated tokens at measured pool scale, over the "
            f"{MARKET_RANKER_PROMPT_TOKEN_CEILING:,} ceiling. Most likely a new candidate-line segment."
        )

    def test_the_width_saturated_pool_at_every_cap_stays_bounded(self) -> None:
        """The bound that pins the caps and the widths, which the measured-scale pool cannot see."""
        prompt = build_ranker_prompt(self._maxed_question(), self._saturated_pool_at_caps())
        estimated = len(prompt) / CHARS_PER_TOKEN

        assert estimated < MARKET_RANKER_PROMPT_SATURATED_TOKEN_CEILING, (
            f"ranker prompt grew to ~{estimated:,.0f} estimated tokens with every width and cap maxed, over the "
            f"{MARKET_RANKER_PROMPT_SATURATED_TOKEN_CEILING:,} bound. Most likely a raised rules cap or "
            f"retrieval width."
        )

    def test_the_question_header_is_a_negligible_share_of_the_prompt(self) -> None:
        """RC_CHARS + FP_CHARS is fixed overhead, so it must not be where the budget goes: the
        candidate block is ~99% of the prompt and is the only part worth policing."""
        pool = self._measured_scale_pool()
        prompt = build_ranker_prompt(self._maxed_question(), pool)
        candidate_lines = [line for line in prompt.split("\n") if line.startswith("[")]

        assert len(candidate_lines) == len(pool)
        assert sum(len(line) for line in candidate_lines) > 0.9 * len(prompt)
