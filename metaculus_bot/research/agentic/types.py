from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

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


class GhostForecast(BaseModel):
    qtype: str
    raw_text: str
    parsed_summary: str = ""


@dataclass(slots=True)
class LoopConfig:
    model: str
    reasoning_effort: str = "medium"
    max_tool_calls: int = 14
    wall_deadline_s: float = 540.0
    conclude_threshold_s: float = 90.0
    max_result_chars: int = 8000
    max_steps: int = 20


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


@dataclass(slots=True)
class LoopResult:
    findings_markdown: str
    ghost: GhostForecast | None
    telemetry: LoopTelemetry
    transcript: list[dict[str, Any]]
