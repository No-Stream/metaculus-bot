"""Tests for the per-provider research-observability primitives.

Covers the ``ProviderResult`` frozen dataclass and the compact, pipe-delimited
diagnostics block that gets appended to the combined research string (and lands
in the Metaculus comment via that string).
"""

from dataclasses import FrozenInstanceError

import pytest

from metaculus_bot.research.provider_diagnostics import (
    ProviderResult,
    format_provider_diagnostics_block,
    pop_provider_detail,
    record_provider_detail,
)


class TestProviderResult:
    def test_fields_and_defaults(self) -> None:
        result = ProviderResult(name="asknews", status="ok", chars=12483, latency_ms=8231)
        assert result.name == "asknews"
        assert result.status == "ok"
        assert result.chars == 12483
        assert result.latency_ms == 8231
        assert result.error_type is None
        assert result.error_message is None
        assert result.details == {}

    def test_is_frozen(self) -> None:
        result = ProviderResult(name="asknews", status="ok", chars=1, latency_ms=1)
        with pytest.raises(FrozenInstanceError):
            result.status = "errored"  # type: ignore[misc]

    def test_details_default_is_independent_per_instance(self) -> None:
        first = ProviderResult(name="a", status="ok", chars=1, latency_ms=1)
        second = ProviderResult(name="b", status="ok", chars=1, latency_ms=1)
        assert first.details is not second.details


class TestFormatProviderDiagnosticsBlock:
    def test_empty_results_returns_empty_string(self) -> None:
        assert format_provider_diagnostics_block([]) == ""

    def test_one_line_per_provider_pipe_delimited(self) -> None:
        results = [
            ProviderResult(name="asknews", status="ok", chars=12483, latency_ms=8231),
            ProviderResult(name="native_search", status="ok", chars=9044, latency_ms=41210),
        ]
        block = format_provider_diagnostics_block(results)

        assert "## Provider Diagnostics" in block
        assert "- asknews: ok | 12483 chars | 8231 ms" in block
        assert "- native_search: ok | 9044 chars | 41210 ms" in block
        # Pipe-delimited, one line per provider.
        provider_lines = [line for line in block.splitlines() if line.startswith("- ")]
        assert len(provider_lines) == 2

    def test_block_is_separated_by_horizontal_rule(self) -> None:
        results = [ProviderResult(name="asknews", status="ok", chars=10, latency_ms=5)]
        block = format_provider_diagnostics_block(results)
        # Mirrors the gap-fill addendum: a leading `---` so it renders as its own section.
        assert block.startswith("---")

    def test_error_type_appended_only_for_errored(self) -> None:
        results = [
            ProviderResult(
                name="gemini_search",
                status="errored",
                chars=0,
                latency_ms=360002,
                error_type="TimeoutError",
                error_message="timed out after 360s",
            ),
            ProviderResult(name="financial_data", status="empty", chars=0, latency_ms=1102),
        ]
        block = format_provider_diagnostics_block(results)

        assert "- gemini_search: errored | 0 chars | 360002 ms | TimeoutError" in block
        # No trailing error_type segment on a non-errored provider.
        assert "- financial_data: empty | 0 chars | 1102 ms" in block
        financial_line = next(line for line in block.splitlines() if line.startswith("- financial_data"))
        assert financial_line.count("|") == 2  # chars, ms — no error_type segment

    def test_inactive_and_fallback_statuses_render(self) -> None:
        results = [
            ProviderResult(name="asknews", status="inactive", chars=0, latency_ms=120),
            ProviderResult(name="asknews", status="fallback", chars=842, latency_ms=5012),
        ]
        block = format_provider_diagnostics_block(results)
        assert "- asknews: inactive | 0 chars | 120 ms" in block
        assert "- asknews: fallback | 842 chars | 5012 ms" in block


def _line_for(block: str, name: str) -> str:
    return next(line for line in block.splitlines() if line.startswith(f"- {name}:"))


class TestPartialDegradationRendering:
    """A multi-source provider that lost an upstream source must not render identically
    to one that got everything. The partial-loss suffix appears ONLY when a source failed,
    so healthy lines stay byte-identical to the pre-existing format (archive/comment stable)."""

    def test_kalshi_dropped_renders_partial_signal_not_bare_ok(self) -> None:
        """The confirmed 2026-07-25 case: prediction_market is `ok` (Polymarket/Manifold
        contributed) but Kalshi's series catalogue was dropped over the size cap."""
        results = [
            ProviderResult(
                name="prediction_market",
                status="ok",
                chars=3067,
                latency_ms=2836,
                details={
                    "sources": {
                        "polymarket": "ok(2)",
                        "kalshi": "dropped(size_cap)",
                        "manifold": "ok(1)",
                        "predictit": "none",
                    }
                },
            )
        ]
        line = _line_for(format_provider_diagnostics_block(results), "prediction_market")

        # Still carries the base fields.
        assert line.startswith("- prediction_market: ok | 3067 chars | 2836 ms")
        # The dropped source is visible at a glance, with its reason.
        assert "kalshi:dropped(size_cap)" in line
        # 2 of 4 platforms contributed matches.
        assert "sources=2/4" in line
        # Contributing / benign-empty platforms are NOT listed as lost.
        lost_segment = line.split("lost=", 1)[1]
        assert "polymarket" not in lost_segment
        assert "manifold" not in lost_segment
        assert "predictit" not in lost_segment  # `none` = queried, no match — benign, not a loss

    def test_healthy_multi_source_has_no_degradation_suffix(self) -> None:
        """Every source contributed → line is byte-identical to a provider with no details."""
        healthy = ProviderResult(
            name="prediction_market",
            status="ok",
            chars=3067,
            latency_ms=2836,
            details={"sources": {"polymarket": "ok(2)", "kalshi": "ok(1)", "manifold": "ok(3)"}},
        )
        no_details = ProviderResult(name="prediction_market", status="ok", chars=3067, latency_ms=2836)

        healthy_line = _line_for(format_provider_diagnostics_block([healthy]), "prediction_market")
        plain_line = _line_for(format_provider_diagnostics_block([no_details]), "prediction_market")

        assert healthy_line == plain_line
        assert "lost=" not in healthy_line
        assert healthy_line.count("|") == 2  # chars, ms — no extra segments

    def test_benign_empty_only_is_not_degradation(self) -> None:
        """A provider whose every queried source simply had no match (all `none`) is healthy —
        no source FAILED, so no degradation suffix."""
        results = [
            ProviderResult(
                name="prediction_market",
                status="empty",
                chars=0,
                latency_ms=1500,
                details={"sources": {"polymarket": "none", "kalshi": "none"}},
            )
        ]
        line = _line_for(format_provider_diagnostics_block(results), "prediction_market")
        assert "lost=" not in line

    def test_all_sources_lost_renders_zero_contributed(self) -> None:
        """resolution_source with its one URL blocked: status `ok` (it emits a notice), but
        0/1 fetched — the partial signal must still fire so it doesn't read as healthy."""
        results = [
            ProviderResult(
                name="resolution_source",
                status="ok",
                chars=142,
                latency_ms=5011,
                details={"sources": {"cbp.gov": "blocked"}},
            )
        ]
        line = _line_for(format_provider_diagnostics_block(results), "resolution_source")
        assert "sources=0/1" in line
        assert "cbp.gov:blocked" in line

    def test_multiple_lost_sources_all_listed(self) -> None:
        results = [
            ProviderResult(
                name="resolution_source",
                status="ok",
                chars=900,
                latency_ms=4000,
                details={"sources": {"a.gov": "ok", "b.org": "js_wall", "c.com": "blocked"}},
            )
        ]
        line = _line_for(format_provider_diagnostics_block(results), "resolution_source")
        # A fetched URL is normalized to "ok"; js_wall/blocked are losses.
        assert "sources=1/3" in line
        assert "b.org:js_wall" in line
        assert "c.com:blocked" in line

    def test_many_lost_sources_are_capped_with_an_overflow_count(self) -> None:
        """A provider can lose more sources than the one-liner should carry (resolution_source
        takes up to 5 URLs, financial_data an unbounded ticker list). Past the render cap the
        rest must collapse to a `+N more` count, so the total loss stays visible without the
        line growing without bound."""
        lost = {f"src{i}.com": "blocked" for i in range(11)}
        results = [
            ProviderResult(
                name="resolution_source",
                status="ok",
                chars=900,
                latency_ms=4000,
                details={"sources": {"good.gov": "ok", **lost}},
            )
        ]
        line = _line_for(format_provider_diagnostics_block(results), "resolution_source")

        assert "sources=1/12" in line  # the true totals are unaffected by the render cap
        lost_segment = line.split("lost=", 1)[1]
        assert lost_segment.count(":blocked") == 8  # only the cap's worth is spelled out
        assert "+3 more" in lost_segment  # ...and the remainder is counted, not dropped

    def test_long_reason_token_is_bounded(self) -> None:
        """A pathologically long token must not blow up the compact one-line format."""
        results = [
            ProviderResult(
                name="resolution_source",
                status="ok",
                chars=10,
                latency_ms=10,
                details={"sources": {"x.com": "error(" + "z" * 500 + ")"}},
            )
        ]
        line = _line_for(format_provider_diagnostics_block(results), "resolution_source")
        # Bounded well under the raw 500-char token.
        assert len(line) < 200

    def test_details_without_sources_key_is_ignored(self) -> None:
        """A details dict that carries no `sources` map renders no suffix (forward-compat)."""
        results = [ProviderResult(name="financial_data", status="ok", chars=50, latency_ms=30, details={"foo": "bar"})]
        line = _line_for(format_provider_diagnostics_block(results), "financial_data")
        assert line == "- financial_data: ok | 50 chars | 30 ms"


class TestInternalCountsRendering:
    """``details["counts"]`` is the second detail convention: provider-internal quantities
    that are neither a source outcome nor a failure. First user is ``gemini_search``'s
    ``unsupported_attributions`` — the tier-tag attributions a response's own grounding
    record could not back. A zero must render nothing, because the count is recorded on
    EVERY checked response (so the archive can tell "ran, found none" from "never ran")
    and a `=0` on every gemini line would be noise in the published comment.
    """

    def test_nonzero_count_rides_the_line(self) -> None:
        results = [
            ProviderResult(
                name="gemini_search",
                status="ok",
                chars=3535,
                latency_ms=21044,
                details={"counts": {"unsupported_attributions": 3}},
            )
        ]
        line = _line_for(format_provider_diagnostics_block(results), "gemini_search")
        assert line == "- gemini_search: ok | 3535 chars | 21044 ms | unsupported_attributions=3"

    def test_zero_count_renders_nothing(self) -> None:
        clean = ProviderResult(
            name="gemini_search",
            status="ok",
            chars=3535,
            latency_ms=21044,
            details={"counts": {"unsupported_attributions": 0}},
        )
        no_details = ProviderResult(name="gemini_search", status="ok", chars=3535, latency_ms=21044)
        block = format_provider_diagnostics_block([clean])
        assert _line_for(block, "gemini_search") == _line_for(
            format_provider_diagnostics_block([no_details]), "gemini_search"
        )

    def test_counts_and_a_source_loss_coexist(self) -> None:
        """Both suffixes are additive, and the loss tail stays last so its `+N more`
        truncation is still the end of the line."""
        results = [
            ProviderResult(
                name="gemini_search",
                status="ok",
                chars=900,
                latency_ms=1000,
                details={
                    "counts": {"unsupported_attributions": 2},
                    "sources": {"grounding": "error(ungrounded_suppressed)"},
                },
            )
        ]
        line = _line_for(format_provider_diagnostics_block(results), "gemini_search")
        assert line.index("unsupported_attributions=2") < line.index("lost=grounding:")


class TestProviderDetailRegistry:
    """The seam that carries a provider's per-source outcome from the provider (which knows
    it) to the orchestrator's _run_one (which builds the ProviderResult). Keyed by
    (qid, provider); record-once / pop-once, mirroring the record_raw_research sink pattern."""

    def test_record_then_pop_round_trips(self) -> None:
        detail = {"sources": {"polymarket": "ok(2)", "kalshi": "dropped(size_cap)"}}
        record_provider_detail(4242, "prediction_market", detail)
        assert pop_provider_detail(4242, "prediction_market") == detail

    def test_pop_clears_the_entry(self) -> None:
        record_provider_detail(4243, "resolution_source", {"sources": {"a.gov": "blocked"}})
        assert pop_provider_detail(4243, "resolution_source") != {}
        # Second pop is empty — the entry was drained, so it can't leak into a later call.
        assert pop_provider_detail(4243, "resolution_source") == {}

    def test_pop_absent_returns_empty_dict(self) -> None:
        assert pop_provider_detail(999999, "nonexistent") == {}

    def test_keyed_by_qid_and_provider(self) -> None:
        record_provider_detail(4244, "prediction_market", {"sources": {"p": "ok"}})
        record_provider_detail(4244, "resolution_source", {"sources": {"u": "blocked"}})
        # Distinct provider on the same qid does not collide.
        assert pop_provider_detail(4244, "prediction_market") == {"sources": {"p": "ok"}}
        assert pop_provider_detail(4244, "resolution_source") == {"sources": {"u": "blocked"}}

    def test_none_qid_is_a_noop(self) -> None:
        # Mirrors the record_raw_research / comment-diagnostics qid=None handling: skip.
        record_provider_detail(None, "prediction_market", {"sources": {"p": "ok"}})
        assert pop_provider_detail(None, "prediction_market") == {}
