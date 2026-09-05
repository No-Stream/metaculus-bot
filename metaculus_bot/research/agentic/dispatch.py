"""Turning one assistant turn's tool calls into exactly one tool message each.

Three stages, in order. ADMISSION (``_admit_tool_calls``) applies the plan gate,
the call budget and duplicate detection, bumping the accepted calls' counters
before any handler runs. ABSORPTION (``_absorb_tool_results``) folds a batch's
provenance URLs, verification tiers and per-method counters into the loop state.
EMISSION (``_append_tool_messages``) writes one tool message per
``tool_call_id`` in the assistant's original order — anything else and the next
LLM turn 400s. The content builders for a refused, unknown or failed call live
here too, since a rejection has to render in exactly the shape a real outcome
does.

The per-CALL handler dispatch stays in ``loop.py``: it reaches the internal
``set_research_plan`` / ``record_findings`` / ``conclude`` handlers, which emit
the loop's own telemetry markers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from metaculus_bot.research.agentic.gates import _PLAN_REQUIRED_NUDGE
from metaculus_bot.research.agentic.loop_state import (
    _LoopState,
    _ToolCall,
    _ToolExecutionResult,
)
from metaculus_bot.research.agentic.provenance import _TIER_RANK, _normalize_quote_text
from metaculus_bot.research.agentic.tool_schemas import _INTERNAL_TOOL_NAMES
from metaculus_bot.research.agentic.types import LoopConfig, ToolOutcome


def _truncate_content(content: str, max_chars: int) -> tuple[str, bool]:
    if len(content) <= max_chars:
        return content, False
    clipped = content[:max_chars].rstrip()
    return f"{clipped}\n[truncated at {len(content)} chars]", True


def _format_tool_content(tool_name: str, outcome: ToolOutcome, max_chars: int) -> str:
    body, truncated = _truncate_content(outcome.content_markdown, max_chars)
    effective = outcome.model_copy(update={"content_markdown": body, "truncated": outcome.truncated or truncated})
    lines = [f"tool: {tool_name}", f"status: {effective.status}"]
    if effective.method:
        lines.append(f"method: {effective.method}")
    if effective.links:
        lines.append("links:")
        lines.extend(f"- {link}" for link in effective.links)
    if effective.truncated and "[truncated at " not in effective.content_markdown:
        lines.append("truncated: true")
    if effective.content_markdown:
        lines.append("")
        lines.append(effective.content_markdown)
    return "\n".join(lines)


def _parse_arguments(arguments: str) -> dict[str, Any]:
    if not arguments.strip():
        return {}
    parsed = json.loads(arguments)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _tool_error_result(tool_call: _ToolCall, message: str, max_result_chars: int) -> _ToolExecutionResult:
    """Synthesize the error tool-response for a call that never reached its handler."""
    outcome = ToolOutcome(content_markdown=message, method="internal", status="error")
    return _ToolExecutionResult(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=_format_tool_content(tool_call.name, outcome, max_result_chars),
    )


def _normalized_call_key(tool_call: _ToolCall) -> tuple[str, str]:
    """(tool, normalized-args) identity for exact-duplicate detection.

    JSON args are re-serialized with sorted keys so key-order shuffles still
    count as the same call; unparseable args fall back to the raw string.
    """
    try:
        normalized = json.dumps(json.loads(tool_call.arguments or "{}"), sort_keys=True)
    except (json.JSONDecodeError, ValueError):
        normalized = tool_call.arguments
    return (tool_call.name, normalized)


_DUPLICATE_CALL_WARNING = (
    "\n[note: this exact tool call was already made earlier in this run — "
    "its result will not have changed. Vary the query/URL or move on.]"
)


def _budget_rejected_content(tool_name: str, config: LoopConfig) -> str:
    outcome = ToolOutcome(
        content_markdown=(
            f"Tool call rejected: the {config.max_tool_calls}-call research budget is exhausted. "
            "No further external tool calls will run — call conclude to finish, "
            "or record_findings to bank what you already have."
        ),
        method="internal",
        status="error",
    )
    return _format_tool_content(tool_name, outcome, config.max_result_chars)


def _plan_rejected_content(tool_name: str, config: LoopConfig) -> str:
    outcome = ToolOutcome(
        content_markdown=f"Tool call rejected: {_PLAN_REQUIRED_NUDGE}.",
        method="internal",
        status="error",
    )
    return _format_tool_content(tool_name, outcome, config.max_result_chars)


@dataclass
class _AdmittedCalls:
    """Which calls in one batch run, and why each of the rest was refused."""

    accepted: list[_ToolCall]
    duplicate_call_ids: set[str]
    rejected_call_ids: set[str]
    plan_rejected_call_ids: set[str]


def _admit_tool_calls(tool_calls: list[_ToolCall], state: _LoopState, config: LoopConfig) -> _AdmittedCalls:
    """Apply the plan gate, the call budget, and duplicate detection to one batch.

    Mutates ``state`` exactly as the inline version did — the accepted calls'
    counters are bumped here, before any handler runs, and the plan nudge is
    recorded once per gated batch.
    """
    admitted = _AdmittedCalls(
        accepted=[], duplicate_call_ids=set(), rejected_call_ids=set(), plan_rejected_call_ids=set()
    )

    # Plan gate (W1): external tool calls are rejected until set_research_plan
    # has run. Checked once per batch (before gather), so a parallel batch of
    # external calls emitted before any plan is all rejected together and counts
    # as a single nudge — a driver gets config.max_plan_nudges turns to plan,
    # after which the loop soft-continues (plan_skipped) rather than wedging.
    # Internal tools (set_research_plan/record_findings/conclude) are never
    # plan-gated, so the driver can always plan, bank, or finish.
    plan_gate_active = state.research_plan is None and not state.plan_skipped

    # Clamp the batch to the remaining call slots. With parallel_tool_calls a
    # single turn can emit more calls than budget allows; without this an
    # over-budget batch executes (and bills) every external call, overshooting
    # the max_tool_calls anytime ceiling. Internal bookkeeping tools
    # (record_findings/conclude) are never rejected so the driver can always
    # bank/finish. Rejected calls are NOT counted as executed, so
    # telemetry.tool_calls stays consistent with _must_conclude's gate.
    for tool_call in tool_calls:
        is_internal = tool_call.name in _INTERNAL_TOOL_NAMES
        if not is_internal and plan_gate_active:
            admitted.plan_rejected_call_ids.add(tool_call.id)
            continue
        if not is_internal and state.telemetry.tool_calls >= config.max_tool_calls:
            admitted.rejected_call_ids.add(tool_call.id)
            continue

        state.telemetry.tool_calls += 1
        state.telemetry.per_tool_counts[tool_call.name] = state.telemetry.per_tool_counts.get(tool_call.name, 0) + 1
        call_key = _normalized_call_key(tool_call)
        if call_key in state.seen_tool_calls:
            state.telemetry.dup_tool_calls += 1
            admitted.duplicate_call_ids.add(tool_call.id)
        else:
            state.seen_tool_calls.add(call_key)
        admitted.accepted.append(tool_call)

    # Record the plan nudge (once per gated batch) and flip to soft-continue
    # once the cap is hit, so the NEXT batch's external calls run un-gated.
    if admitted.plan_rejected_call_ids:
        state.plan_nudges += 1
        if state.plan_nudges >= config.max_plan_nudges:
            state.plan_skipped = True
            state.telemetry.plan_skipped = True
    return admitted


def _absorb_tool_results(state: _LoopState, results: Sequence[_ToolExecutionResult]) -> None:
    """Fold one batch's provenance, verification tiers, and counters into loop state."""
    provenance_texts: list[str] = []
    for result in results:
        if result.method == "rendered":
            state.telemetry.rendered_fetches += 1
        # Accumulate provenance so a LATER turn's record_findings/conclude can
        # verify a finding's source_url against what the driver actually
        # retrieved. Internal tools contribute nothing (see
        # provenance._harvest_provenance).
        state.tool_seen_urls.update(result.provenance_urls)
        # Merge per-call verification tiers, keeping the best tier seen per URL
        # (fetched outranks snippet) — a URL first seen via search then fetched
        # upgrades to "fetched" (W4).
        for url, tier in result.provenance_tiers.items():
            existing = state.url_best_tier.get(url)
            if existing is None or _TIER_RANK[tier] > _TIER_RANK[existing]:
                state.url_best_tier[url] = tier
        if result.provenance_text:
            provenance_texts.append(result.provenance_text)
    if provenance_texts:
        state.tool_content_normalized = _normalize_quote_text(
            f"{state.tool_content_normalized} {' '.join(provenance_texts)}"
        )


def _append_tool_messages(
    tool_calls: list[_ToolCall],
    admitted: _AdmittedCalls,
    results: Sequence[_ToolExecutionResult],
    *,
    state: _LoopState,
    config: LoopConfig,
    budget_line: str,
) -> None:
    """Emit exactly one tool message per tool_call_id, in the assistant's original order.

    Anything else and the next LLM turn 400s. Rejected calls get a synthetic
    plan-gate / budget-exhausted error response. This is also the only stage that
    can see an outcome, so it owns the one duplicate-detection exemption the
    admission stage cannot make: a throttled fetch's key is evicted from
    ``state.seen_tool_calls`` here (see the inline comment below).
    """
    results_by_id = {result.tool_call_id: result for result in results}
    for tool_call in tool_calls:
        if tool_call.id in admitted.plan_rejected_call_ids:
            content = _plan_rejected_content(tool_call.name, config)
        elif tool_call.id in admitted.rejected_call_ids:
            content = _budget_rejected_content(tool_call.name, config)
        else:
            result = results_by_id[tool_call.id]
            # Forget a throttled call so its retry isn't called a duplicate. The
            # throttle outcome is deliberately not cached and its message tells the
            # driver to fetch the same URL again later in the run
            # (tools._throttled_fetch_outcome), so _DUPLICATE_CALL_WARNING's "its
            # result will not have changed. Vary the query/URL or move on." is false
            # exactly here and steers the driver off the URL. Advisory bookkeeping
            # only: max_tool_calls still caps a throttle spin, a re-throttled retry
            # re-registers at admission and is evicted again here, and a retry that
            # succeeds leaves its key in place so a THIRD identical call is still
            # warned.
            if result.method == "throttled":
                state.seen_tool_calls.discard(_normalized_call_key(tool_call))
            content = result.content + (_DUPLICATE_CALL_WARNING if tool_call.id in admitted.duplicate_call_ids else "")
        state.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": content + budget_line,
            }
        )
