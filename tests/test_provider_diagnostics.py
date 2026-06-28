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
