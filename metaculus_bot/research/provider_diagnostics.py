"""Per-provider research-observability primitives.

A ``ProviderResult`` captures, for one research provider's call, what the
orchestrator's ``_run_one`` chokepoint already sees: outcome status, output
size, wall-clock latency, and any exception detail. The orchestrator both
persists these to the durable research artifact (schema v2) and renders them
into a compact, pipe-delimited block appended to the combined research string
(which also becomes the Metaculus comment), so failures/empties/timing survive
beyond the ephemeral GHA run-log.
"""

from dataclasses import dataclass, field
from typing import Literal

# Status of one provider call. ``ok``: non-empty output. ``empty``: returned but
# blank/whitespace. ``errored``: raised an unexpected exception. ``inactive``:
# AskNews off-season subscription error (expected, not alertable). ``fallback``:
# AskNews failed but a prose fallback provider supplied the result.
ProviderStatus = Literal["ok", "empty", "errored", "inactive", "fallback"]

# Markdown header for the diagnostics section. Named so the leakage screen can
# recognize a diagnostics-only blob without hardcoding the literal in two places.
PROVIDER_DIAGNOSTICS_HEADER = "## Provider Diagnostics"


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """Structured outcome of a single research-provider call."""

    name: str
    status: ProviderStatus
    chars: int
    latency_ms: int
    error_type: str | None = None
    error_message: str | None = None
    # Reserved for provider-INTERNAL detail (financial tickers/FRED series,
    # prediction-market per-platform match counts, gap-fill gaps). Empty in
    # Phase 1; tasks #6/#7 populate it.
    details: dict = field(default_factory=dict)


# Statuses that count as "this provider contributed usable research". ``ok`` =
# its own non-empty output; ``fallback`` = AskNews failed but a prose fallback
# supplied the result. The orchestrator derives ``providers_succeeded`` from this
# set, so it lives here as the single source of truth.
SUCCEEDED_STATUSES: tuple[ProviderStatus, ...] = ("ok", "fallback")


def _format_one(result: ProviderResult) -> str:
    line = f"- {result.name}: {result.status} | {result.chars} chars | {result.latency_ms} ms"
    if result.status == "errored" and result.error_type is not None:
        line += f" | {result.error_type}"
    return line


def format_provider_diagnostics_block(results: list[ProviderResult]) -> str:
    """Render the compact provider-diagnostics block, or "" when there are none.

    One ``- <name>: <status> | <chars> chars | <ms> ms[ | <error_type>]`` line per
    provider, pipe-delimited so it greps/parses trivially. The leading ``---``
    mirrors the gap-fill addendum so the block renders as its own comment section.
    """
    if not results:
        return ""
    lines = [_format_one(result) for result in results]
    return f"---\n\n{PROVIDER_DIAGNOSTICS_HEADER}\n\n" + "\n".join(lines)
