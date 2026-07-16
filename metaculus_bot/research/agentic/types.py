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
    steps: int = 0
    tool_calls: int = 0
    per_tool_counts: dict[str, int] = field(default_factory=dict)
    deadline_hit: bool = False
    concluded_early: bool = False
    wall_s: float = 0.0
    findings_count: int = 0
    pending_leads_count: int = 0
    lint_rejections: int = 0


@dataclass(slots=True)
class LoopResult:
    findings_markdown: str
    ghost: GhostForecast | None
    telemetry: LoopTelemetry
    transcript: list[dict[str, Any]]
