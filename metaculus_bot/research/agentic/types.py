from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolOutcome(BaseModel):
    content_markdown: str
    links: list[str] = Field(default_factory=list)
    method: str = ""
    status: str = "ok"
    truncated: bool = False


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[ToolOutcome]]
    timeout_s: float


class Finding(BaseModel):
    claim: str
    source_url: str
    quote: str
    date: str = ""
    retrieved_how: str = ""
    topic: str = "general"
    discrepancy: bool = False
    # Arithmetic-only synthesis over the finding's own quoted numbers (W3): a
    # derived table, bound, or rate whose every input appears as a quoted value
    # with URL in this finding's quote/source fields. Exempt from the
    # detachment lint (arithmetic + its result, no likelihood language, no new
    # facts — see artifact.detachment_lint); rendered under a "Derived analysis"
    # label so the panel weights it as our synthesis, not a source claim.
    derivation: str | None = None


class GhostForecast(BaseModel):
    qtype: str
    raw_text: str
    parsed_summary: str = ""


class PlannedGap(BaseModel):
    """One ranked research gap the driver commits to in set_research_plan (W1).

    ``id`` is the driver-chosen handle W2's conclude-time accounting keys on;
    ``question`` is the factual question to resolve; ``why_decision_relevant``
    is the ranking rationale (the trailing slot holds the least forecast-moving
    gap). Both verify-targets (assumptions to check) and fill-targets (facts
    absent from the briefing) live here — the debiased union of v1's and v2's
    targeting questions.
    """

    id: str
    question: str
    why_decision_relevant: str = ""


class ResearchPlan(BaseModel):
    """The driver's turn-one research plan, emitted via set_research_plan (W1).

    The private dry run stays in the driver's reasoning; its outputs surface
    here. ``dry_run_forecast`` is the qtype-shaped structured-block payload (same
    schema family as the ghost) used only for the GHOST_PRE telemetry delta;
    ``sensitive_assumptions`` are the 3-5 assumptions that would most move the
    forecast if wrong; ``gaps`` is the ranked work-list (capped at
    ``GAP_FILL_V2_MAX_GAPS``). W2 reads ``gaps`` for conclude-time accounting.
    """

    dry_run_forecast: dict[str, Any] | None = None
    sensitive_assumptions: list[str] = Field(default_factory=list)
    gaps: list[PlannedGap] = Field(default_factory=list)


# Terminal state the driver assigns each plan gap at conclude time (W2). A gap
# is either resolved (the fact was found), parked (attempted, unresolvable this
# run — a pending lead), or dismissed on inspection (turned out not to move the
# forecast once looked at). All three are honest outcomes; the conclude gate
# cares that every gap got AN entry with SOME action, not which status it is.
GapStatus = Literal["resolved", "unresolved_parked", "not_decision_relevant_on_inspection"]


class GapAccountingEntry(BaseModel):
    """One plan gap's disposition, supplied in conclude's ``gap_accounting`` (W2).

    ``gap_id`` keys back to a ``PlannedGap.id`` from the turn-one research plan;
    ``actions_taken`` is the driver's free-text note of what it did for this gap
    (searches run, pages fetched, why it parked it); ``status`` is the terminal
    disposition. The conclude gate rejects an early conclusion unless every plan
    gap id appears here (plus the global tool-call and fetch-floor invariants);
    see loop._conclude_tool.
    """

    gap_id: str
    actions_taken: str = ""
    status: GapStatus = "resolved"


@dataclass(slots=True)
class LoopConfig:
    model: str
    reasoning_effort: str = "medium"
    max_tool_calls: int = 14
    wall_deadline_s: float = 540.0
    conclude_threshold_s: float = 90.0
    max_result_chars: int = 8000
    max_steps: int = 20
    # Max ranked gaps set_research_plan accepts (W1). Extra gaps beyond this are
    # dropped (the driver ranks them, so the tail holds the least valuable).
    max_gaps: int = 4
    # How many times the plan-required gate may reject an external tool call
    # before the loop soft-continues without a plan (W1). Prevents a driver that
    # never plans from wedging the loop.
    max_plan_nudges: int = 2
    # How many times the conclude gate (W2) may reject an early conclusion before
    # accepting it unconditionally. Mirrors max_plan_nudges: a pathological driver
    # that can't satisfy the gap-accounting / fetch-floor invariants can't loop
    # forever on conclude attempts. Budget-exhaustion conclusions bypass the gate
    # entirely and never count against this cap.
    max_conclude_gate_rejections: int = 2


@dataclass(slots=True)
class LoopTelemetry:
    model: str = ""
    steps: int = 0
    tool_calls: int = 0
    per_tool_counts: dict[str, int] = field(default_factory=dict)
    # fetch calls whose outcome came from the headless-Chromium rung — a
    # per-method count (per_tool_counts can't see which rung served a fetch).
    rendered_fetches: int = 0
    # Exact-duplicate (tool, normalized-args) repeats within the run — plan
    # §3.1 v1-lite stuck-detection (counter + gentle warning, no enforcement).
    dup_tool_calls: int = 0
    deadline_hit: bool = False
    concluded_early: bool = False
    wall_s: float = 0.0
    findings_count: int = 0
    pending_leads_count: int = 0
    lint_rejections: int = 0
    # Findings dropped because their cited source_url never appeared in a tool
    # result this run (and, for discrepancies, was not tool-sourced) — the hard
    # provenance gate. See loop._validate_findings_payload.
    provenance_rejections: int = 0
    # Findings ACCEPTED despite their quote not being found verbatim in the
    # per-loop tool contents — a warn-only signal (read_document paraphrases and
    # ellipsis-joined quotes make a hard gate too false-positive-prone; we
    # measure the real miss rate first).
    quote_mismatch_warnings: int = 0
    # Ranked gaps the driver registered via set_research_plan (W1). 0 when no
    # plan was set (see plan_skipped).
    plan_gaps: int = 0
    # True when the driver never called set_research_plan and the plan-nudge cap
    # was hit, so the loop soft-continued without a plan (W1). A pathological
    # driver can't wedge the loop on the plan gate; this flags the degraded run.
    plan_skipped: bool = False
    # Early conclusions the W2 conclude gate rejected (missing gap accounting,
    # too few tool calls, or an unmet fetch floor) before the loop accepted one.
    # Persistent 2s in prod flag a gate that's too strict or a prompt that's
    # unclear; see loop._conclude_tool and the GAP_FILL_V2 completion marker.
    conclude_gate_rejections: int = 0


@dataclass(slots=True)
class LoopResult:
    findings_markdown: str
    ghost: GhostForecast | None
    telemetry: LoopTelemetry
    transcript: list[dict[str, Any]]
