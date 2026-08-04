"""The prediction-market query machinery: deterministic set, digit stripping, LLM author, scorer.

Two properties here are load-bearing rather than stylistic, and each has a measured failure
behind it:

- **The query author is ADDITIVE.** A *replacing* author's failure mode is an empty query set,
  which is indistinguishable from "no markets exist" — the silent failure that hid the Manifold
  breakage for 17+ days. So ``parse_query_author`` returns ``()`` on every unreadable shape and
  the deterministic set fires regardless.
- **``fuzzy_best`` has no score floor.** It ORDERS the enumerable venues' catalogues so the top
  N can fit in a prompt; it never drops anything. The retired floors
  (``KALSHI_MIN_FUZZY_SCORE`` / ``PREDICTIT_MIN_FUZZY_SCORE``) are what discarded the
  adjacent-cut markets the ranked design exists to surface, so the parameter is absent rather
  than defaulted — a floor cannot come back by passing an argument.

The ``manifold_relaxation_terms`` cases are ported from
``tests/test_prediction_market_liquidity_contract.py`` with the function's move; the ladder
itself is unchanged and those assertions still describe it correctly.
"""

from __future__ import annotations

import inspect
import json

import pytest

from metaculus_bot.research.market_retrieval.queries import (
    _RELEVANCE_STOPWORDS,
    MANIFOLD_RELAXATION_MAX_TOKENS,
    QUERY_AUTHOR_RC_CHARS,
    build_query_author_prompt,
    deterministic_queries,
    fuzzy_best,
    manifold_relaxation_terms,
    parse_query_author,
    strip_dates_and_numbers,
)

_AUSTRALIA_TITLE = "What will the seasonally adjusted unemployment rate be in Australia for July 2026?"


def _content_tokens(term: str) -> int:
    """Manifold requires every CONTENT token to appear in a market's text; stopwords are
    measured not to constrain the match (see the D2 diagnosis), so they don't count."""
    return len([t for t in term.split() if t.lower().strip(".,'") not in _RELEVANCE_STOPWORDS])


class TestStripDatesAndNumbers:
    """Manifold's ``term`` is a strict conjunction, so one date token no market's text carries
    zeroes the result set. Stripping happens in code, not by asking the model nicely."""

    def test_drops_every_token_containing_a_digit(self):
        """Whole tokens go, not just their digits: `Q3` and `$90.50` are as unsatisfiable as a year."""
        assert strip_dates_and_numbers("US unemployment rate July 2026") == "US unemployment rate July"
        assert strip_dates_and_numbers("Brent crude above $90.50 in Q3") == "Brent crude above in"

    def test_collapses_whitespace_and_survives_an_all_numeric_query(self):
        assert strip_dates_and_numbers("  2026   2027  ") == ""
        assert strip_dates_and_numbers("") == ""


class TestDeterministicQueries:
    """``[title, natural-language framing, *relaxation rungs]``, deduped case-insensitively.

    The FULL TITLE leads the set and is issued unconditionally. The validated design measured
    every rung as a first-class query rather than a fallback, and dropping the full-length query
    would cost the high-precision hit on questions that already match at full length.
    """

    def test_leads_with_the_full_title(self):
        queries = deterministic_queries(_AUSTRALIA_TITLE)
        assert queries[0] == _AUSTRALIA_TITLE

    def test_carries_the_framing_and_every_relaxation_rung(self):
        queries = deterministic_queries(_AUSTRALIA_TITLE)
        assert "What will the seasonally adjusted unemployment rate be in Australia for July 2026" in queries
        for rung in manifold_relaxation_terms(_AUSTRALIA_TITLE):
            assert rung in queries

    def test_does_not_strip_digits(self):
        """The Kalshi and PredictIt fuzzy channels score on the UN-stripped set — a year is
        real signal against a catalogue of dated market titles. Only the conjunctive venues
        strip, at their own call site."""
        assert any("2026" in q for q in deterministic_queries(_AUSTRALIA_TITLE))

    def test_dedupes_case_insensitively(self):
        """A title with no '?' makes the framing identical to the title; it must appear once."""
        title = "US CPI year-over-year"
        queries = deterministic_queries(title)
        assert queries.count(title) == 1
        assert len(queries) == len({q.casefold() for q in queries})

    def test_returns_nothing_for_an_empty_title(self):
        assert deterministic_queries("") == []
        assert deterministic_queries("   ") == []


class TestParseQueryAuthor:
    def test_reads_synonyms_and_framings_in_order(self):
        out = parse_query_author(
            '{"synonyms": ["jobless rate", "labour force survey"], "framings": ["Australia jobs"]}'
        )
        assert out == ("jobless rate", "labour force survey", "Australia jobs")

    def test_reads_a_fenced_json_block(self):
        text = '```json\n{"synonyms": ["ABS labour force"], "framings": []}\n```'
        assert parse_query_author(text) == ("ABS labour force",)

    def test_strips_digits_and_caps_query_length(self):
        """Digits are stripped at PARSE time, not only at search time: a numeric token in a
        synonym would otherwise survive into the Kalshi fuzzy channel, which does not strip."""
        long_synonym = "unemployment " * 20
        out = parse_query_author(f'{{"synonyms": ["CPI 2026 print", "{long_synonym}"], "framings": []}}')
        assert out[0] == "CPI print"
        assert all(len(q) <= 80 for q in out)

    def test_caps_at_eight_synonyms_and_three_framings(self):
        """The prompt states both ceilings; they are enforced in code so a runaway completion
        cannot blow up the pool."""
        payload = json.dumps(
            {
                "synonyms": [f"synonym {chr(ord('a') + i)}" for i in range(12)],
                "framings": [f"framing {chr(ord('a') + i)}" for i in range(6)],
            }
        )
        out = parse_query_author(payload)
        assert len(out) == 8 + 3
        assert out[8].startswith("framing")

    def test_dedupes_case_insensitively_across_both_keys(self):
        out = parse_query_author('{"synonyms": ["jobless rate", "Jobless Rate"], "framings": ["JOBLESS RATE"]}')
        assert out == ("jobless rate",)

    def test_a_synonym_whose_content_is_numeric_drops_out(self):
        """The known cost of stripping at parse time, pinned so nobody discovers it in prod: a
        series-code synonym like "U-3" is ONE token containing a digit, so the whole synonym
        disappears rather than being trimmed. Accepted — the alternative is a date token reaching
        the conjunctive venues, which measurably zeroes their result set — but it does mean the
        author cannot contribute ticker-shaped vocabulary ("U-3", "S&P 500", "CPI-U")."""
        assert parse_query_author('{"synonyms": ["U-3", "unemployment"], "framings": []}') == ("unemployment",)
        assert parse_query_author('{"synonyms": ["S&P 500 index"], "framings": []}') == ("S&P index",)

    @pytest.mark.parametrize(
        ("shape", "text"),
        [
            ("empty", "   "),
            ("no_json_object", "Sure! Here are some queries: unemployment, jobless."),
            ("malformed_json", '{"synonyms": ["U-3",}'),
            ("wrong_type", '["U-3", "jobless rate"]'),
            ("neither_key", '{"queries": ["U-3"], "notes": "nothing usable"}'),
            ("keys_present_but_empty", '{"synonyms": [], "framings": []}'),
            ("non_string_items", '{"synonyms": [1, 2, 3], "framings": [{"a": 1}]}'),
            ("digits_only", '{"synonyms": ["2026", "44796"], "framings": []}'),
        ],
    )
    def test_unusable_output_returns_an_empty_tuple(self, shape: str, text: str):
        """Every unreadable shape degrades to "no extra queries" — never to an exception and
        never to a query set the deterministic half has been replaced by."""
        assert parse_query_author(text) == (), shape


class TestQueryAuthorPrompt:
    """The author asks for the vocabulary a token match on the question cannot reach, and asks
    for it WITHOUT dates. The digit ban is asked for here and ENFORCED in
    ``parse_query_author`` — the code is the guarantee, since a model that ignores the
    instruction would otherwise zero a Manifold conjunction."""

    def test_the_additive_framing_and_the_digit_ban_are_stated(self):
        prompt = build_query_author_prompt("Will unemployment exceed 4.5%?", "Resolves per BLS.")

        assert "Recall is the objective" in prompt
        assert "Your queries are ADDED to a deterministic query set" in prompt
        assert "Do NOT include dates, years, or numbers in any string" in prompt
        assert '"synonyms"' in prompt and '"framings"' in prompt

    def test_the_question_is_substituted_and_the_criteria_capped(self):
        prompt = build_query_author_prompt("Will unemployment exceed 4.5%?", "c" * 5000)

        assert "title: Will unemployment exceed 4.5%?" in prompt
        assert "{title}" not in prompt and "{rc}" not in prompt
        assert "c" * QUERY_AUTHOR_RC_CHARS in prompt
        assert "c" * (QUERY_AUTHOR_RC_CHARS + 1) not in prompt

    def test_a_brace_in_a_title_does_not_break_substitution(self):
        prompt = build_query_author_prompt("Will {x} happen?", "")

        assert "title: Will {x} happen?" in prompt


class TestFuzzyBest:
    """0.7 × title + 0.3 × rules ``token_set_ratio``, maxed over the query set. The floors died
    with the keyword era; this is only a ranker."""

    def test_scores_an_exact_title_match_at_the_ceiling(self):
        assert fuzzy_best(["US unemployment rate"], "US unemployment rate", "") == pytest.approx(70.0)

    def test_rules_text_lifts_a_weak_title_match(self):
        title_only = fuzzy_best(["seasonally adjusted U-3"], "Jobs report", "")
        with_rules = fuzzy_best(
            ["seasonally adjusted U-3"], "Jobs report", "Settles on the seasonally adjusted U-3 rate"
        )
        assert with_rules > title_only

    def test_has_no_floor_so_a_zero_scoring_candidate_still_ranks_last(self):
        """A floor is what dropped the adjacent-cut markets. The scorer must return a usable
        number for an unrelated candidate so the caller can rank it last rather than lose it."""
        queries = ["Australian unemployment rate"]
        scored = sorted(
            [
                ("unrelated", fuzzy_best(queries, "zzzz", "")),
                ("wanted", fuzzy_best(queries, "Australia unemployment rate July", "")),
            ],
            key=lambda pair: pair[1],
            reverse=True,
        )
        assert [name for name, _ in scored] == ["wanted", "unrelated"]
        assert scored[-1][1] == pytest.approx(0.0, abs=25.0)

    def test_takes_no_min_score_argument(self):
        """Pinned so the retired floor cannot return by way of a defaulted parameter."""
        assert "min_score" not in inspect.signature(fuzzy_best).parameters


class TestManifoldRelaxationLadder:
    """Ported with the function's move out of ``prediction_market.py``. Manifold's
    ``/v0/search-markets`` treats ``term`` as a strict conjunction: every content token must
    appear in a market's text, and one absent token returns ``[]``. Measured cliff sits at ~4
    content tokens, so 9-15-token question sentences were nearly always unsatisfiable —
    Manifold contributed zero rows to any bundle for 17+ days while reporting healthy."""

    def test_relaxation_ladder_descends_to_a_satisfiable_length(self):
        terms = manifold_relaxation_terms(_AUSTRALIA_TITLE)
        assert terms, "no relaxation rungs produced"
        assert [_content_tokens(t) for t in terms] == sorted((_content_tokens(t) for t in terms), reverse=True), (
            "rungs must descend from most specific to most general"
        )
        assert _content_tokens(terms[-1]) <= 2
        assert _content_tokens(terms[0]) <= MANIFOLD_RELAXATION_MAX_TOKENS

    def test_ladder_keeps_the_salient_entity_not_the_scaffolding(self):
        terms = manifold_relaxation_terms("Will a bot finish in the top 5 of the Summer 2026 Metaculus Cup?")
        assert any("Metaculus" in t for t in terms)
        assert not any(t.lower().startswith("will ") for t in terms)

    def test_ladder_drops_duplicate_tokens(self):
        """A title repeating an entity must not yield 'Sturgis Sturgis Motorcycle' — a
        duplicated token spends a rung slot without narrowing anything."""
        terms = manifold_relaxation_terms(
            "How many vehicles will enter Sturgis for the 86th Annual Sturgis Motorcycle Rally?"
        )
        for term in terms:
            lowered = [t.lower() for t in term.split()]
            assert len(lowered) == len(set(lowered)), f"duplicate token in rung {term!r}"

    def test_ladder_is_empty_for_a_title_with_no_content_tokens(self):
        assert manifold_relaxation_terms("Will it?") == []
