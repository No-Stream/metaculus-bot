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
    MAX_FRAMINGS,
    MAX_QUERY_CHARS,
    MAX_SYNONYMS,
    QUERY_AUTHOR_PROMPT,
    QUERY_AUTHOR_RC_CHARS,
    build_query_author_prompt,
    deterministic_queries,
    fuzzy_best,
    fuzzy_best_many,
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

    def test_a_well_formed_object_followed_by_brace_bearing_prose_still_parses(self):
        """The measured weakness of the retired widest-brace slice: it ran from the first `{` to
        the LAST `}` anywhere in the output, so ordinary trailing prose containing a brace made a
        perfectly good payload unreadable and reported this additive stage lost for nothing. The
        canonical extractor is string-literal-aware and stops at the object's own closing brace."""
        text = '{"synonyms": ["jobless rate"], "framings": []}\nNote: use {care} with these.'

        assert parse_query_author(text) == ("jobless rate",)

    def test_drops_digit_bearing_synonyms_and_caps_query_length(self):
        """Digits are handled at PARSE time, not only at search time: a numeric token in a synonym
        would otherwise survive into the Kalshi fuzzy channel, which does not strip. The synonym
        is DROPPED rather than trimmed — see the drop-the-remnant test below for why."""
        long_synonym = "unemployment " * 20
        out = parse_query_author(f'{{"synonyms": ["CPI 2026 print", "{long_synonym}"], "framings": []}}')

        assert "CPI print" not in out, "a digit-bearing synonym must not survive as a generic remnant"
        assert len(out) == 1
        assert out[0].startswith("unemployment unemployment")
        assert len(out[0]) <= MAX_QUERY_CHARS

    def test_caps_at_eight_synonyms_and_three_framings(self):
        """The prompt states both ceilings; they are enforced in code so a runaway completion
        cannot blow up the pool. Asserted against the constants, and the PROMPT's own stated
        numbers are checked against them below — a ceiling raised in code while the prompt still
        asks for the old one silently discards the extra strings the model was invited to send."""
        payload = json.dumps(
            {
                "synonyms": [f"synonym {chr(ord('a') + i)}" for i in range(MAX_SYNONYMS + 4)],
                "framings": [f"framing {chr(ord('a') + i)}" for i in range(MAX_FRAMINGS + 3)],
            }
        )
        out = parse_query_author(payload)
        assert len(out) == MAX_SYNONYMS + MAX_FRAMINGS
        assert out[MAX_SYNONYMS].startswith("framing")

    def test_dedupes_case_insensitively_across_both_keys(self):
        out = parse_query_author('{"synonyms": ["jobless rate", "Jobless Rate"], "framings": ["JOBLESS RATE"]}')
        assert out == ("jobless rate",)

    def test_a_synonym_carrying_any_digit_drops_rather_than_leaving_a_remnant(self):
        """The known cost of handling digits at parse time, pinned so nobody discovers it in prod:
        the author cannot contribute ticker-shaped vocabulary at all ("U-3", "S&P 500", "CPI-U").

        Dropping is the LESSER cost. The remnant this used to keep ("S&P 500 index" → "S&P index",
        "U-3 rate" → "rate") pollutes the query set the Kalshi fuzzy channel scores on: that
        channel does not strip, `fuzzy_best` maxes with no floor, and `token_set_ratio` gives a
        bare generic word ~100 against every event whose rules text contains it. Measured on the
        real 9,762-event catalogue, `"rate"` scores >=99 on 52 events and moved the first wanted
        row from pool rank 2 to rank 31, replacing the whole 8-row slate with Fed-funds markets.
        Passing digit-bearing synonyms through verbatim instead is worse still — a bare "2026"
        scores 100 against every dated Kalshi title.
        """
        assert parse_query_author('{"synonyms": ["U-3", "unemployment"], "framings": []}') == ("unemployment",)
        assert parse_query_author('{"synonyms": ["U-3 rate", "jobless"], "framings": []}') == ("jobless",)
        assert parse_query_author('{"synonyms": ["S&P 500 index"], "framings": []}') == ()

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

    def test_the_stated_ceilings_match_the_enforced_ones(self):
        """The prompt asks for the ceilings in prose and the parser enforces them in code, so the
        two can drift apart silently — raising ``MAX_SYNONYMS`` alone wastes the extra slots the
        model was never invited to fill, and lowering it alone discards strings it was."""
        assert f"Up to {MAX_SYNONYMS} short strings" in QUERY_AUTHOR_PROMPT
        assert f"{MAX_FRAMINGS} alternate short phrasings" in QUERY_AUTHOR_PROMPT


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


class TestFuzzyBestMany:
    """The batched form the full-catalogue scan runs on. The equivalence is what the whole
    optimization rests on: `process.cdist` at `dtype=np.float64` is bit-identical to the scalar
    loop, while the float32 default drifts ~5e-06 — enough to reorder ties in a 9,762-event sort
    and silently change which 100 candidates reach the ranker."""

    def test_it_agrees_with_the_scalar_form_exactly(self):
        queries = ["Australian unemployment rate", "ABS labour force", "jobless"]
        titles = [
            "Australia unemployment rate July",
            "US CPI year over year",
            "zzzz",
            "Will the RBA cut rates?",
        ]
        rules = [
            "Settles on the ABS seasonally adjusted rate",
            "Settles on the BLS CPI print",
            "",  # an empty rules row: the scalar form special-cases it, cdist must agree
            "Resolves on the RBA cash rate decision",
        ]

        batched = fuzzy_best_many(queries, titles, rules)
        scalar = [fuzzy_best(queries, title, rule) for title, rule in zip(titles, rules, strict=True)]

        assert batched == scalar

    def test_an_empty_query_set_scores_every_title_zero(self):
        """`deterministic_queries("")` returns `[]`, and `.max(axis=0)` over a zero-row array
        raises `ValueError` — so the guard is load-bearing rather than defensive noise."""
        assert fuzzy_best_many([], ["a title", "another"], ["", ""]) == [0.0, 0.0]

    def test_an_empty_catalogue_scores_nothing(self):
        assert fuzzy_best_many(["a query"], [], []) == []


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
