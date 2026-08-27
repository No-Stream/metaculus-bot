"""Markdown assembly for the research bundle: provider headers and heading levels.

Split out of ``orchestrator.py``: how a provider's text is LABELLED and stitched into
one document is presentation, and it is independent of how the providers were chosen,
run, or budgeted. Everything here is pure — no I/O, no orchestrator state — which is
also why the two heading helpers are directly unit-testable.

``ResearchOrchestrator`` mixes in ``ResearchSectionFormatting``, so
``ResearchOrchestrator._provider_header`` keeps working for the callers (and tests)
that reach it through the class, and ``orchestrator._demote_inner_headings`` stays
importable from its original module path.
"""

import re

from metaculus_bot.prompts import TS_ANCHOR_SECTION_HEADER
from metaculus_bot.research.provider_diagnostics import ProviderResult

_LEADING_HEADING_RE = re.compile(r"^(#{1,2})(?=\s|$)", re.MULTILINE)


def _demote_inner_headings(text: str) -> str:
    """Shift any in-body h1/h2 heading down by two levels (h1→h3, h2→h4).

    Provider headers are h2 (``_provider_header``). If an LLM-written body emits
    its own h1/h2 (e.g. ``# Historical Context``), it sits at/above the provider
    header and breaks the framework's ``report_sections_to_markdown``
    renormalization, which degrades to the ugly ``[Hashtag]`` fallback. Demoting
    keeps every provider header the minimum-level section.
    """
    return _LEADING_HEADING_RE.sub(lambda m: "##" + m.group(1), text)


class ResearchSectionFormatting:
    """Mixin: renders provider output into the research bundle's markdown sections."""

    @staticmethod
    def _provider_header(name: str) -> str:
        headers = {
            "asknews": "## News Articles (AskNews)",
            "native_search": "## Web Research (Native Search)",
            "gemini_search": "## Web Research (Google Search via Gemini)",
            "financial_data": "## Financial & Economic Data",
            "timeseries_anchor": TS_ANCHOR_SECTION_HEADER,
            "prediction_market": "## Prediction Market Snapshot",
            "resolution_source": "## Resolution Source Snapshot",
            "exa": "## Web Research (Exa)",
            "perplexity": "## Web Research (Perplexity)",
            "openrouter": "## Web Research (OpenRouter)",
            "custom": "## Research (Custom)",
        }
        return headers.get(name, f"## Research ({name})")

    @classmethod
    def _assemble_provider_sections(cls, results: list[tuple[str, ProviderResult]]) -> tuple[str, list[ProviderResult]]:
        """Join each provider's text under its own h2 header; also return the results in order.

        ``results`` arrives in SELECTION order (``_await_providers_within_deadline``
        rebuilds it from the provider list rather than from ``asyncio.wait``'s
        unordered sets), and that order is both the section order in the bundle and
        the row order in the diagnostics block. A provider that produced nothing is
        skipped in the text but still returned in ``provider_results``, so the
        diagnostics block and the research archive still name it.
        """
        combined_parts = []
        provider_results: list[ProviderResult] = []
        for raw, provider_result in results:
            provider_results.append(provider_result)
            if raw and raw.strip():
                # Label the section with whoever actually produced the text. On a
                # fallback that is NOT provider_result.name (which keeps the primary's
                # identity for the diagnostics line) — rendering Perplexity prose under
                # "## News Articles (AskNews)" mislabelled the source in the published
                # comment and in the archive.
                header = cls._provider_header(provider_result.fallback_provider or provider_result.name)
                combined_parts.append(f"{header}\n{_demote_inner_headings(raw)}")

        combined = "\n\n---\n\n".join(combined_parts) if combined_parts else ""
        return combined, provider_results
