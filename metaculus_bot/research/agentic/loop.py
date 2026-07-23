from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import ValidationError

from metaculus_bot.research.agentic.artifact import detachment_lint, render_findings
from metaculus_bot.research.agentic.llm import build_default_llm_call
from metaculus_bot.research.agentic.types import (
    Finding,
    GapAccountingEntry,
    GhostForecast,
    LoopConfig,
    LoopResult,
    LoopTelemetry,
    PlannedGap,
    ResearchPlan,
    ToolOutcome,
    ToolSpec,
)
from metaculus_bot.structured_output_schema import (
    BinaryStructured,
    MultipleChoiceStructured,
    NumericStructured,
    extract_json_block,
    parse_structured_block,
)

logger = logging.getLogger(__name__)

LlmCall = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], Awaitable[Any]]

_INTERNAL_TOOL_TIMEOUT_S = 5.0
_INTERNAL_TOOL_NAMES = ("set_research_plan", "record_findings", "conclude")
_NUDGE = "call conclude or use tools"
# Returned in place of an external tool's result until set_research_plan has run
# (W1). Mirrors _NUDGE mechanics: the driver sees why the call was rejected and
# what to do instead. Capped at LoopConfig.max_plan_nudges (then soft-continue).
_PLAN_REQUIRED_NUDGE = (
    "call set_research_plan first — register your dry-run forecast, sensitive "
    "assumptions, and ranked research gaps before using any research tool"
)

# Provenance gate (source_url grounding + quote spot-check). A finding is
# rendered under a "supersedes-the-briefing" banner and shown to every base
# forecaster, so a hallucinated/mistyped citation from the low-effort driver
# would silently override correct research. The URL check is a HARD gate; the
# quote check is WARN-ONLY (read_document paraphrases and ellipsis-joined
# quotes make a hard quote gate too false-positive-prone). The banner says
# "sourced" (not "verified") because only the URL is gated — see artifact.py.
_URL_IN_TEXT_RE = re.compile(r"https?://[^\s<>\"'\)\]]+")
_URL_TRAILING_PUNCT = ".,;:)]}>\"'"
_TRACKER_PARAM_PREFIXES = ("utm_",)
_TRACKER_PARAM_NAMES = frozenset({"gclid", "fbclid", "mc_cid", "mc_eid", "ref", "ref_src", "igshid", "spm"})
# All straight/curly quote glyphs and backticks collapse to one token so a
# driver quote using curly quotes still matches straight-quoted source text.
_QUOTE_GLYPHS_RE = re.compile(r"[\"'‘’“”`]")
_WHITESPACE_RE = re.compile(r"\s+")

# Retrieval-quality tiers (W4). ToolOutcome.method records HOW a URL's content
# reached the driver; we collapse the seven method values into two tiers so a
# finding stamped from the URL->best-method map carries honest authority. A
# "fetched" URL is one whose actual page/document we pulled (document/rendered/
# plain/cache); a "snippet" URL was only seen in a search/news result. A
# discrepancy resting on a snippet must NOT supersede the briefing (the 131.3
# failure mode: a crowd-median "correction" from a search snippet after the
# direct fetch 403'd, which every forecaster then adopted). Methods absent here
# (internal bookkeeping, error/blocked outcomes) contribute no tier — the URL
# still counts for the provenance gate, but the finding stays untiered and a
# discrepancy on it is demoted, conservatively. See artifact.render_findings.
_METHOD_TO_TIER: dict[str, str] = {
    "document": "fetched",
    "rendered": "fetched",
    "plain": "fetched",
    "cache": "fetched",
    "search": "snippet",
    "news": "snippet",
}
# fetched outranks snippet, so a URL seen via search THEN fetched upgrades.
_TIER_RANK: dict[str, int] = {"snippet": 0, "fetched": 1}


def _outranks(candidate: str | None, incumbent: str | None) -> bool:
    """True when ``candidate`` is a strictly better verification tier than
    ``incumbent``. None (untiered — the URL was never retrieved through a tool)
    ranks below every real tier."""
    return (-1 if candidate is None else _TIER_RANK[candidate]) > (-1 if incumbent is None else _TIER_RANK[incumbent])


def _method_to_tier(method: str) -> str | None:
    """Map a ToolOutcome.method to a verification tier ("fetched"/"snippet"), or
    None when the method isn't a real content retrieval (internal bookkeeping,
    error/blocked states) and so grants no retrieval authority."""
    return _METHOD_TO_TIER.get(method)


def _normalize_url(url: str) -> str | None:
    """Canonicalize a URL for provenance comparison.

    Lowercases scheme + host, drops the fragment, strips a trailing slash from
    the path, and removes common tracker query params (``utm_*``, ``gclid``,
    ``fbclid``, ...). Returns ``None`` for non-http(s) or unparseable input, so
    those never count as provenance.
    """
    candidate = url.strip().rstrip(_URL_TRAILING_PUNCT)
    try:
        parts = urlsplit(candidate)
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return None
    host = parts.netloc.lower()
    if not host:
        return None
    path = parts.path.rstrip("/")
    kept_params = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith(_TRACKER_PARAM_PREFIXES) and key.lower() not in _TRACKER_PARAM_NAMES
    ]
    return urlunsplit((scheme, host, path, urlencode(kept_params), ""))


def _iter_normalized_urls(text: str) -> Iterator[str]:
    """Yield the normalized form of every http(s) URL found in free text."""
    for match in _URL_IN_TEXT_RE.finditer(text):
        normalized = _normalize_url(match.group(0))
        if normalized is not None:
            yield normalized


def _normalize_quote_text(text: str) -> str:
    """Lowercase, collapse whitespace, and unify quote glyphs for substring matching."""
    return _WHITESPACE_RE.sub(" ", _QUOTE_GLYPHS_RE.sub("'", text)).strip().lower()


def _quote_is_grounded(quote: str, tool_content_normalized: str) -> bool:
    """True when the finding's quote appears (normalized) in the tool contents.

    An empty quote is treated as grounded — there is nothing to verify.
    """
    normalized_quote = _normalize_quote_text(quote)
    if not normalized_quote:
        return True
    return normalized_quote in tool_content_normalized


class _FindingsValidation(NamedTuple):
    accepted: list[Finding]
    rejected: list[str]
    lint_rejections: int
    provenance_rejections: int
    quote_mismatch_warnings: int


@dataclass(slots=True)
class _LoopState:
    messages: list[dict[str, Any]]
    started_at_s: float
    deadline_at_s: float
    telemetry: LoopTelemetry = field(default_factory=LoopTelemetry)
    findings: list[Finding] = field(default_factory=list)
    # Canonical-JSON identity -> index in ``findings`` for already-banked
    # findings, so re-recording the same finding (record_findings then a re-list
    # in conclude's final_findings) restamps the stored copy's tier instead of
    # double-appending. See _bank_findings.
    seen_finding_keys: dict[str, int] = field(default_factory=dict)
    pending_leads: list[str] = field(default_factory=list)
    seen_tool_calls: set[tuple[str, str]] = field(default_factory=set)
    # Provenance gate accumulators (see the _normalize_url block). Normalized
    # URLs the driver actually saw via a TOOL this run — fetch/read call
    # arguments plus every URL in a tool result's content/links. Discrepancy
    # findings must cite one of these (a fresh primary-source check).
    tool_seen_urls: set[str] = field(default_factory=set)
    # Normalized URL -> best retrieval tier seen this run ("fetched" outranks
    # "snippet"), stamped onto each finding's verification_tier at banking time
    # (W4). Only successful (status=ok) tool outcomes contribute a tier, so a
    # 403'd fetch never grants "fetched" authority — a later search snippet of
    # the same fact lands "snippet" and its discrepancy is demoted. A URL only
    # in the briefing (never retrieved) is absent here -> untiered finding.
    url_best_tier: dict[str, str] = field(default_factory=dict)
    # Normalized URLs embedded in the frozen briefing bundle. Non-discrepancy
    # findings may cite these too; discrepancies may NOT.
    briefing_urls: set[str] = field(default_factory=set)
    # Concatenated, normalized tool-result contents — the corpus the warn-only
    # quote spot-check searches.
    tool_content_normalized: str = ""
    nudged_for_no_action: bool = False
    explicit_conclude: bool = False
    stop_loop: bool = False
    # Per-run log prefix (question ref), so internal-tool handlers can emit
    # markers keyed the same way as the ghost phase (GHOST_PRE at plan-set time).
    log_prefix: str = ""
    # Turn-one research plan (W1). None until set_research_plan runs; external
    # tool calls are rejected with _PLAN_REQUIRED_NUDGE until it exists (unless
    # the plan-nudge cap forces a soft-continue). W2 reads research_plan.gaps.
    research_plan: ResearchPlan | None = None
    # Count of external tool calls rejected by the plan gate, and whether the cap
    # was hit so we soft-continued without a plan (telemetry plan_skipped).
    plan_nudges: int = 0
    plan_skipped: bool = False


@dataclass(slots=True)
class _ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class _ToolExecutionResult:
    tool_call_id: str
    tool_name: str
    content: str
    method: str = ""
    # Provenance harvested from an EXTERNAL tool call (never internal
    # record_findings/conclude, whose echoed rejection text would otherwise let
    # a hallucinated URL launder itself into the seen-set): the normalized URLs
    # the driver saw/requested this call, and the normalized result text the
    # warn-only quote check searches. Accumulated into loop state post-gather.
    provenance_urls: list[str] = field(default_factory=list)
    provenance_text: str = ""
    # Normalized URL -> verification tier this call established (W4). Only
    # populated for successful retrievals: a fetched-class call tiers the URLs it
    # actually retrieved (its arguments) "fetched"; a snippet-class call tiers
    # every URL it surfaced "snippet". Merged into state.url_best_tier (best-tier
    # wins) post-gather. Empty when the outcome granted no retrieval authority.
    provenance_tiers: dict[str, str] = field(default_factory=dict)


def _tool_schema(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _internal_tool_schemas() -> list[dict[str, Any]]:
    finding_schema = {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "source_url": {"type": "string"},
                        "quote": {"type": "string"},
                        "date": {"type": "string"},
                        "retrieved_how": {"type": "string"},
                        "topic": {"type": "string"},
                        "discrepancy": {"type": "boolean"},
                        "derivation": {
                            "type": "string",
                            "description": (
                                "OPTIONAL arithmetic-only synthesis over THIS finding's quoted numbers "
                                "(a derived table, bound, or rate). Every input number must appear as a "
                                "quoted value with URL in this finding's quote/source. Arithmetic and its "
                                "result only — no likelihood language, no new facts."
                            ),
                        },
                    },
                    "required": ["claim", "source_url", "quote"],
                    "additionalProperties": True,
                },
            }
        },
        "required": ["findings"],
        "additionalProperties": False,
    }
    conclude_schema = {
        "type": "object",
        "properties": {
            "pending_leads": {"type": "array", "items": {"type": "string"}},
            "final_findings": {
                "type": "array",
                "items": finding_schema["properties"]["findings"]["items"],
            },
            "gap_accounting": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "gap_id": {"type": "string"},
                        "actions_taken": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": [
                                "resolved",
                                "unresolved_parked",
                                "not_decision_relevant_on_inspection",
                            ],
                        },
                    },
                    "required": ["gap_id", "actions_taken", "status"],
                    "additionalProperties": True,
                },
                "description": (
                    "REQUIRED before concluding early: one entry per research-plan gap "
                    "(gap_id, what you did, and its status). An early conclude is rejected "
                    "until every plan gap is accounted for and the fetch floor is met."
                ),
            },
        },
        "additionalProperties": False,
    }
    plan_schema = {
        "type": "object",
        "properties": {
            "dry_run_forecast": {
                "type": "object",
                "description": (
                    "Your private dry-run forecast as the panel's STRUCTURED FORECAST block "
                    "(same shape as the template: question_type + posterior_prob / option_probs / "
                    "declared_percentiles). Telemetry only — never shown to the panel."
                ),
                "additionalProperties": True,
            },
            "sensitive_assumptions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 assumptions that would most move your forecast if wrong.",
            },
            "gaps": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "question": {"type": "string"},
                        "why_decision_relevant": {"type": "string"},
                    },
                    "required": ["id", "question"],
                    "additionalProperties": True,
                },
                "description": (
                    "Ranked research gaps (most forecast-moving first): verify-targets "
                    "(assumptions to check) AND fill-targets (facts absent from the briefing). "
                    "At least one gap is required — a plan with no gaps is rejected."
                ),
            },
        },
        "required": ["gaps"],
        "additionalProperties": False,
    }
    return [
        _tool_schema(
            "set_research_plan",
            "Register your turn-one research plan (dry-run forecast, sensitive assumptions, ranked gaps). "
            "REQUIRED before any research tool — external tool calls are rejected until this is set.",
            plan_schema,
        ),
        _tool_schema(
            "record_findings",
            "Bank detached findings. Claims must stay citation-only and avoid likelihood or verdict language. "
            "Optional derivation field carries arithmetic-only synthesis over the finding's own quoted numbers.",
            finding_schema,
        ),
        _tool_schema(
            "conclude",
            "Finish the loop, optionally banking final findings and leaving pending leads for follow-up telemetry.",
            conclude_schema,
        ),
    ]


def _tool_schemas(tools: list[ToolSpec], must_conclude: bool) -> list[dict[str, Any]]:
    internal = _internal_tool_schemas()
    if must_conclude:
        return internal
    return internal + [_tool_schema(tool.name, tool.description, tool.parameters) for tool in tools]


def _get_field(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _parse_response_message(response: Any) -> dict[str, Any]:
    choices = _get_field(response, "choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response missing choices[0].message")
    message = _get_field(choices[0], "message")
    if message is None:
        raise ValueError("LLM response missing choices[0].message")

    content = _get_field(message, "content")
    tool_calls_raw = _get_field(message, "tool_calls")
    tool_calls: list[dict[str, Any]] = []
    if tool_calls_raw:
        if not isinstance(tool_calls_raw, list):
            raise ValueError("assistant tool_calls was not a list")
        tool_calls = [_normalize_tool_call(entry, index) for index, entry in enumerate(tool_calls_raw)]

    assistant: dict[str, Any] = {"role": "assistant", "content": content if isinstance(content, str) else ""}
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    return assistant


def _normalize_tool_call(raw: Any, index: int) -> dict[str, Any]:
    function = _get_field(raw, "function")
    name = _get_field(function, "name")
    arguments = _get_field(function, "arguments")
    if not isinstance(name, str) or not name:
        raise ValueError(f"assistant tool call {index} missing function.name")
    call_id = _get_field(raw, "id")
    normalized: dict[str, Any] = {
        "id": call_id if isinstance(call_id, str) and call_id else f"tool_{index}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments if isinstance(arguments, str) else "{}",
        },
    }
    return normalized


def _extract_tool_calls(assistant_message: dict[str, Any]) -> list[_ToolCall]:
    raw_calls = assistant_message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    return [
        _ToolCall(
            id=str(entry["id"]),
            name=str(entry["function"]["name"]),
            arguments=str(entry["function"]["arguments"]),
        )
        for entry in raw_calls
    ]


def _remaining_s(state: _LoopState, now: Callable[[], float]) -> float:
    return max(0.0, state.deadline_at_s - now())


def _must_conclude(state: _LoopState, config: LoopConfig, now: Callable[[], float]) -> bool:
    return _remaining_s(state, now) < config.conclude_threshold_s or state.telemetry.tool_calls >= config.max_tool_calls


def _truncate_content(content: str, max_chars: int) -> tuple[str, bool]:
    if len(content) <= max_chars:
        return content, False
    clipped = content[:max_chars].rstrip()
    return f"{clipped}\n[truncated at {len(content)} chars]", True


def _unaddressed_gaps_suffix(state: _LoopState) -> str:
    """Render the driver's outstanding gap work-list for the budget line (W1).

    W1 accounting is deliberately coarse: it lists EVERY plan gap id until W2's
    conclude-time gap_accounting lands (the plan's W2 section explicitly says to
    build strict per-call attribution there, not here — a per-call gap_id param
    would mean touching every external tool's schema). So the suffix shows the
    full work-list debt as a standing reminder, not a live-shrinking count.
    """
    if state.research_plan is None or not state.research_plan.gaps:
        return ""
    gap_ids = ", ".join(gap.id for gap in state.research_plan.gaps)
    return f" unaddressed_gaps=[{gap_ids}]"


def _budget_line(state: _LoopState, config: LoopConfig, now: Callable[[], float]) -> str:
    remaining = int(_remaining_s(state, now))
    gaps = _unaddressed_gaps_suffix(state)
    if _must_conclude(state, config, now):
        return f"\n[budget: {remaining}s remaining — you must conclude now{gaps}]"
    return (
        f"\n[budget: {remaining}s remaining, "
        f"{state.telemetry.tool_calls}/{config.max_tool_calls} tool calls used{gaps}]"
    )


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


def _check_url_provenance(finding: Finding, state: _LoopState) -> str | None:
    """Return a rejection reason if the finding's source_url isn't grounded, else None.

    A non-discrepancy finding is grounded when its normalized URL appears
    anywhere the driver could have seen it — a tool result this run OR the
    frozen briefing. A discrepancy finding must rest on a fresh primary-source
    check (per the driver prompt), so only a TOOL-sourced URL counts; a
    briefing-embedded URL does not.
    """
    normalized = _normalize_url(finding.source_url)
    if normalized is None:
        return f"source_url {finding.source_url!r} is not a valid http(s) URL"
    if normalized in state.tool_seen_urls:
        return None
    if finding.discrepancy:
        return (
            f"discrepancy source_url {finding.source_url!r} was not seen in any tool result this run; "
            "a discrepancy must rest on a fresh primary-source check (search/fetch/read), "
            "not a URL already in the briefing"
        )
    if normalized in state.briefing_urls:
        return None
    return (
        f"source_url {finding.source_url!r} did not appear in any tool result or the briefing this run; "
        "cite a URL you actually retrieved"
    )


def _validate_findings_payload(
    raw_findings: Any,
    state: _LoopState,
    *,
    label: Literal["findings", "final_findings"],
) -> _FindingsValidation:
    if raw_findings is None:
        return _FindingsValidation([], [], 0, 0, 0)
    if not isinstance(raw_findings, list):
        return _FindingsValidation([], [f"{label} must be a list"], 0, 0, 0)

    accepted: list[Finding] = []
    rejected: list[str] = []
    lint_rejections = 0
    provenance_rejections = 0
    quote_mismatch_warnings = 0
    for index, raw_finding in enumerate(raw_findings):
        try:
            finding = Finding.model_validate(raw_finding)
        except ValidationError as exc:
            rejected.append(f"{label}[{index}] invalid: {exc.errors()[0]['msg']}")
            continue
        violations = detachment_lint(finding)
        if violations:
            lint_rejections += 1
            rejected.append(f"{label}[{index}] rejected: {'; '.join(violations)}")
            continue
        provenance_reason = _check_url_provenance(finding, state)
        if provenance_reason is not None:
            provenance_rejections += 1
            rejected.append(f"{label}[{index}] rejected: {provenance_reason}")
            continue
        # WARN-ONLY quote spot-check: a miss is logged and counted but the
        # finding is still accepted (read_document paraphrases, ellipsis joins).
        if not _quote_is_grounded(finding.quote, state.tool_content_normalized):
            quote_mismatch_warnings += 1
            logger.warning(
                "GAP_FILL_V2 quote_mismatch: source_url=%s quote=%r not found verbatim in tool contents",
                finding.source_url,
                finding.quote,
            )
        accepted.append(finding)
    return _FindingsValidation(accepted, rejected, lint_rejections, provenance_rejections, quote_mismatch_warnings)


def _coerce_pending_leads(raw_pending_leads: Any) -> tuple[list[str], list[str]]:
    if raw_pending_leads is None:
        return [], []
    if not isinstance(raw_pending_leads, list):
        return [], ["pending_leads must be a list of strings"]

    pending_leads: list[str] = []
    issues: list[str] = []
    for index, item in enumerate(raw_pending_leads):
        if isinstance(item, str):
            pending_leads.append(item)
        else:
            issues.append(f"pending_leads[{index}] invalid: expected string")
    return pending_leads, issues


def _stamp_verification_tier(finding: Finding, state: _LoopState) -> Finding:
    """Return a copy of ``finding`` with ``verification_tier`` set from the
    URL->best-tier map (W4) — CODE-derived, so any driver-supplied tier is
    overwritten. A source_url the driver never retrieved through a tool (e.g. a
    briefing-only URL) is absent from the map and stays untiered (None)."""
    normalized = _normalize_url(finding.source_url)
    tier = state.url_best_tier.get(normalized) if normalized is not None else None
    return finding.model_copy(update={"verification_tier": tier})


def _bank_findings(state: _LoopState, accepted: list[Finding]) -> tuple[int, int]:
    """Append findings to ``state.findings``, skipping ones already banked this run.

    Returns ``(banked, duplicates)``. Banking is idempotent by full-field
    identity — the finding's canonical JSON serialization EXCLUDING the
    loop-stamped ``verification_tier``: a driver that records findings
    incrementally with ``record_findings`` and then re-lists the same ones in
    ``conclude``'s ``final_findings`` (observed on Q578 — 8 findings rendered 16
    times) no longer doubles the list. Serializing every OTHER field keeps
    genuinely distinct findings that happen to share a source/quote (a different
    ``claim`` or ``topic``) separate, and stays correct if ``Finding`` gains a
    field. Each banked finding carries the code-derived tier (W4).

    A duplicate still RESTAMPS the stored finding's tier: when the URL's tier
    upgraded between the two banks (banked from a search snippet, then the same
    URL was successfully fetched, then re-listed in conclude), the affirmative
    re-record carries the fetched authority forward — otherwise a verified
    discrepancy would stay demoted to the "possible corrections" block. Only an
    explicit re-record upgrades (never a bare fetch alone: the driver may have
    fetched, seen the claim was wrong, and deliberately not re-listed it), and
    the best-tier merge in ``url_best_tier`` makes the restamp monotonic — a
    tier never downgrades.
    """
    banked = 0
    duplicates = 0
    for finding in accepted:
        # Key on the driver-stable identity (tier is loop-stamped, so exclude it)
        # so dedup survives a between-bank tier upgrade.
        key = finding.model_dump_json(exclude={"verification_tier"})
        stamped = _stamp_verification_tier(finding, state)
        existing_index = state.seen_finding_keys.get(key)
        if existing_index is not None:
            duplicates += 1
            if _outranks(stamped.verification_tier, state.findings[existing_index].verification_tier):
                state.findings[existing_index] = stamped
            continue
        state.seen_finding_keys[key] = len(state.findings)
        state.findings.append(stamped)
        banked += 1
    return banked, duplicates


def _apply_findings_telemetry(state: _LoopState, validation: _FindingsValidation) -> None:
    state.telemetry.lint_rejections += validation.lint_rejections
    state.telemetry.provenance_rejections += validation.provenance_rejections
    state.telemetry.quote_mismatch_warnings += validation.quote_mismatch_warnings


async def _record_findings_tool(state: _LoopState, arguments: dict[str, Any]) -> ToolOutcome:
    validation = _validate_findings_payload(arguments.get("findings"), state, label="findings")
    banked, duplicates = _bank_findings(state, validation.accepted)
    _apply_findings_telemetry(state, validation)

    lines = [f"Recorded {banked} finding(s)."]
    if duplicates:
        lines.append(f"Skipped {duplicates} finding(s) already recorded earlier in this run.")
    if validation.rejected:
        lines.append("Rejected:")
        lines.extend(f"- {item}" for item in validation.rejected)
    return ToolOutcome(content_markdown="\n".join(lines), method="internal")


def _coerce_planned_gaps(raw_gaps: Any, *, max_gaps: int) -> tuple[list[PlannedGap], list[str]]:
    """Validate the plan's ``gaps`` list, capping at ``max_gaps`` (the tail — least
    forecast-moving — is dropped since the driver ranks them). Returns
    ``(gaps, issues)``; malformed entries are skipped with a reason."""
    if raw_gaps is None or not isinstance(raw_gaps, list):
        return [], ["gaps must be a list of {id, question, why_decision_relevant}"]
    gaps: list[PlannedGap] = []
    issues: list[str] = []
    for index, raw_gap in enumerate(raw_gaps):
        try:
            gaps.append(PlannedGap.model_validate(raw_gap))
        except ValidationError as exc:
            issues.append(f"gaps[{index}] invalid: {exc.errors()[0]['msg']}")
    if len(gaps) > max_gaps:
        issues.append(f"kept the top {max_gaps} of {len(gaps)} gaps (ranked); dropped the rest")
        gaps = gaps[:max_gaps]
    return gaps, issues


def _summarize_dry_run(dry_run_forecast: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize the plan's dry-run forecast dict into the GHOST_FORECAST_JSON payload
    shape, reusing the ghost parser so GHOST_PRE_JSON and GHOST_FORECAST_JSON are
    directly comparable. Returns ``None`` when absent or unparseable (the JSON
    marker line is then suppressed, same as the ghost path)."""
    if not isinstance(dry_run_forecast, dict):
        return None
    # Wrap the dict as a fenced block so the tested _summarize_ghost path parses
    # it identically to a real ghost — no second parsing code path.
    raw_text = f"```json\n{json.dumps(dry_run_forecast, separators=(',', ':'))}\n```"
    _qtype, _summary, forecast = _summarize_ghost(raw_text)
    return forecast


async def _set_research_plan_tool(state: _LoopState, arguments: dict[str, Any], config: LoopConfig) -> ToolOutcome:
    """Register the driver's turn-one research plan (W1).

    Stores ``state.research_plan`` (W2 reads its gaps for conclude-time
    accounting) and emits the GHOST_PRE / GHOST_PRE_JSON telemetry — the pre-
    research counterpart to the concluding ghost, so the pre/post delta measures
    whether v2's research moved its own view.
    """
    gaps, gap_issues = _coerce_planned_gaps(arguments.get("gaps"), max_gaps=config.max_gaps)
    # A plan with zero valid gaps is rejected (F3a): storing it would flip W1's
    # plan_gate_active off (opening external tools) while _evaluate_conclude_gate
    # returns None on `not plan.gaps` — disabling the W2 gate entirely and letting
    # a driver conclude with zero research. Leave research_plan untouched (None, or
    # a prior valid plan) so the W1 gate stays armed and re-planning to empty can't
    # clobber an existing plan; nudge the driver to register real gaps.
    if not gaps:
        notes = "; ".join(gap_issues) if gap_issues else "provide at least one ranked research gap"
        return ToolOutcome(
            content_markdown=f"Research plan rejected: {_PLAN_REQUIRED_NUDGE} ({notes}).",
            method="internal",
            status="error",
        )

    sensitive = arguments.get("sensitive_assumptions")
    sensitive_assumptions = [item for item in sensitive if isinstance(item, str)] if isinstance(sensitive, list) else []
    dry_run_forecast = arguments.get("dry_run_forecast")
    dry_run_forecast = dry_run_forecast if isinstance(dry_run_forecast, dict) else None

    state.research_plan = ResearchPlan(
        dry_run_forecast=dry_run_forecast,
        sensitive_assumptions=sensitive_assumptions,
        gaps=gaps,
    )
    state.telemetry.plan_gaps = len(gaps)

    forecast = _summarize_dry_run(dry_run_forecast)
    logger.info(
        "%sGHOST_PRE: gaps=%s sensitive_assumptions=%s",
        state.log_prefix,
        len(gaps),
        len(sensitive_assumptions),
    )
    if forecast is not None:
        logger.info("%sGHOST_PRE_JSON: %s", state.log_prefix, json.dumps(forecast, separators=(",", ":")))

    lines = [f"Research plan set: {len(gaps)} gap(s), {len(sensitive_assumptions)} sensitive assumption(s)."]
    if gap_issues:
        lines.append("Notes:")
        lines.extend(f"- {item}" for item in gap_issues)
    return ToolOutcome(content_markdown="\n".join(lines), method="internal")


# Fetch floor (W2, clause c): the run must reach primary sources, not stop at
# search snippets. The global fallback clause is satisfied at this many
# successful fetched-tier retrievals; the per-gap clause reads the driver's own
# actions_taken note for a fetch/read mention.
_FETCH_FLOOR_MIN_CALLS = 2
# Word-boundary match on fetch/read verbs so common narration words don't count:
# a bare substring "read" fired on "already"/"spread"/"thread"/"ready", and
# "could not fetch the source" (an honest failed-fetch note) satisfied the floor
# with zero real retrievals (F2). \b anchors kill the "read"-in-a-word matches;
# the fetch verb is required CONJUGATED (fetched/fetches/fetching) so bare
# present-tense "fetch" — overwhelmingly negated/attempted ("could not fetch",
# "try to fetch") in a past-actions note — doesn't clear, while any real
# completed fetch/read mention still does. (bare `read` stays: its past tense is
# spelled identically, so "read the PDF" must count.)
_FETCH_ACTION_RE = re.compile(r"\b(?:fetch(?:ed|es|ing)|read(?:s|ing|_document)?)\b", re.IGNORECASE)


def _external_tool_call_count(state: _LoopState) -> int:
    """Accepted EXTERNAL tool calls this run — the internal bookkeeping tools
    (set_research_plan / record_findings / conclude) don't count toward the
    per-gap research floor."""
    return sum(count for name, count in state.telemetry.per_tool_counts.items() if name not in _INTERNAL_TOOL_NAMES)


def _coerce_gap_accounting(raw_accounting: Any) -> list[GapAccountingEntry]:
    """Validate conclude's ``gap_accounting`` into entries, skipping malformed
    ones (a skipped entry just reads as a still-unaccounted gap in the gate)."""
    if not isinstance(raw_accounting, list):
        return []
    entries: list[GapAccountingEntry] = []
    for raw_entry in raw_accounting:
        try:
            entries.append(GapAccountingEntry.model_validate(raw_entry))
        except ValidationError:
            continue
    return entries


def _actions_cite_fetch(actions_taken: str) -> bool:
    """True when the driver's free-text ``actions_taken`` mentions fetching or
    reading a source (the fetch floor's per-gap self-report clause). This prose
    signal is no longer trusted on its own: a failure note ("fetched the source
    but got a 403") also matches, so the conclude gate now counts it only
    alongside >=1 successful fetched-tier retrieval (see ``_conclude_gate_debts``);
    the telemetry-based clause is the robust one. Matches fetch/read verbs on word
    boundaries so "already"/"spread"/"could not fetch" don't false-positive."""
    return bool(_FETCH_ACTION_RE.search(actions_taken))


def _conclude_gate_debts(state: _LoopState, entries: list[GapAccountingEntry]) -> list[str]:
    """Outstanding conclude-gate requirements (W2); empty means the early
    conclusion may proceed. Only called when a research plan with gaps exists.

    Enforces: (a) every plan gap appears in the accounting; (b) at least one
    external tool call per plan gap (the loop can't attribute calls to gaps, so
    this is the cheap global invariant); (c) the fetch floor — either the top-2
    ranked gaps' accounting each cites a fetch/read action AND the run made >=1
    successful fetched-tier retrieval, OR the run made ``_FETCH_FLOOR_MIN_CALLS``+
    fetches/reads. The >=1-retrieval conjunct stops prose alone (which can cite a
    fetch verb even in a failure note) from clearing the floor with zero pages
    reached.
    """
    plan = state.research_plan
    assert plan is not None and plan.gaps  # caller guards; keeps the type narrow
    gaps = plan.gaps
    debts: list[str] = []

    accounted_ids = {entry.gap_id for entry in entries}
    missing_ids = [gap.id for gap in gaps if gap.id not in accounted_ids]
    if missing_ids:
        debts.append(f"gap_accounting is missing entries for gap(s): {', '.join(missing_ids)}")

    external_calls = _external_tool_call_count(state)
    if external_calls < len(gaps):
        debts.append(
            f"only {external_calls} external tool call(s) made for {len(gaps)} plan gap(s) — "
            "research each gap at least once before concluding"
        )

    # Count only successfully-retrieved primary sources, not fetch/read tool
    # CALLS: per_tool_counts increments at accept time regardless of outcome, so
    # two 403'd fetches would clear the floor though they reached nothing — the
    # exact 131.3 mechanism, and exactly what W4's tier map already excludes
    # (url_best_tier is populated only from status=="ok" fetched-class outcomes).
    # Reusing it makes W2's "fetched" agree with W4's (F4). (Distinct URLs, so a
    # single page fetched twice counts once — the intent is breadth of primary
    # sources reached.)
    fetches_reads = sum(1 for tier in state.url_best_tier.values() if tier == "fetched")
    entry_by_id = {entry.gap_id: entry for entry in entries}
    # Top-2 ranked gaps: a finite-N fetch-floor clause, not subsampling.
    top_gaps = gaps[:2]  # HARNESS-SCAN-EXEMPT-subsampling
    top_cite_fetch = bool(top_gaps) and all(
        gap.id in entry_by_id and _actions_cite_fetch(entry_by_id[gap.id].actions_taken) for gap in top_gaps
    )
    # The per-gap prose clause counts ONLY alongside >=1 real successful retrieval:
    # a fetch-verb note ("fetched the source but got a 403") also matches the
    # actions regex, so prose alone could clear the floor with zero pages reached.
    # Requiring one real fetched-tier retrieval closes that hole while preserving
    # the clause's purpose (one load-bearing fetch + honest per-gap notes beats the
    # global 2-fetch bar). The global clause remains a fetch-count-only fallback.
    if not ((top_cite_fetch and fetches_reads >= 1) or fetches_reads >= _FETCH_FLOOR_MIN_CALLS):
        debts.append(
            "fetch floor unmet: a top-ranked gap's fetch/read_document citation now counts only "
            f"alongside at least one successful fetched-tier retrieval (the run made {fetches_reads}); "
            f"otherwise make {_FETCH_FLOOR_MIN_CALLS}+ fetches/reads — fetch a primary source before "
            "concluding on the load-bearing gaps"
        )
    return debts


def _evaluate_conclude_gate(
    state: _LoopState, arguments: dict[str, Any], config: LoopConfig, now: Callable[[], float]
) -> ToolOutcome | None:
    """The W2 anti-satisficing gate. Returns a blocking error ToolOutcome (loop
    continues, ``conclude_gate_rejections`` bumped) when an EARLY conclusion
    hasn't met the work-list floor, else ``None`` (let the conclude proceed).

    Bypasses entirely — never blocks — when the deadline/budget forces a
    conclusion (``_must_conclude``) or once the rejection cap
    (``max_conclude_gate_rejections``) is hit (both guarantee no wedge).

    When a real research plan exists (gaps registered) its work-list floor is
    enforced regardless of ``plan_skipped`` — a valid plan set AFTER the plan-
    nudge cap fired re-arms the gate (F3c: ``plan_skipped`` is checked only in the
    no-plan branch below, not before the plan). When NO plan was set, a driver
    that legitimately soft-continued past the plan-nudge cap (``plan_skipped``) is
    let through, but an early conclude with no plan at all is rejected with a
    plan-first nudge (F3b: a driver can't skip planning by never calling it).
    """
    if _must_conclude(state, config, now):
        return None
    if state.telemetry.conclude_gate_rejections >= config.max_conclude_gate_rejections:
        return None

    plan = state.research_plan
    if plan is not None and plan.gaps:
        debts = _conclude_gate_debts(state, _coerce_gap_accounting(arguments.get("gap_accounting")))
    elif state.plan_skipped:
        return None
    else:
        debts = ["no research plan was set — call set_research_plan with ranked gaps before concluding"]
    if not debts:
        return None

    state.telemetry.conclude_gate_rejections += 1
    lines = [
        "Conclude rejected — resolve the outstanding research debt below before concluding "
        "(a forced deadline conclusion overrides this):"
    ]
    lines.extend(f"- {debt}" for debt in debts)
    return ToolOutcome(content_markdown="\n".join(lines), method="internal", status="error")


async def _conclude_tool(
    state: _LoopState, arguments: dict[str, Any], config: LoopConfig, now: Callable[[], float]
) -> ToolOutcome:
    gate_block = _evaluate_conclude_gate(state, arguments, config, now)
    if gate_block is not None:
        return gate_block

    validation = _validate_findings_payload(arguments.get("final_findings"), state, label="final_findings")
    pending_leads, pending_errors = _coerce_pending_leads(arguments.get("pending_leads"))
    banked, duplicates = _bank_findings(state, validation.accepted)
    state.pending_leads = pending_leads
    _apply_findings_telemetry(state, validation)
    state.explicit_conclude = True
    state.stop_loop = True

    lines = [f"Concluded with {banked} final finding(s) and {len(pending_leads)} pending lead(s)."]
    if duplicates:
        lines.append(f"Skipped {duplicates} final finding(s) already recorded earlier in this run.")
    if validation.rejected or pending_errors:
        lines.append("Rejected:")
        lines.extend(f"- {item}" for item in [*validation.rejected, *pending_errors])
    return ToolOutcome(content_markdown="\n".join(lines), method="internal")


async def _execute_one_tool_call(
    tool_call: _ToolCall,
    tools_by_name: dict[str, ToolSpec],
    state: _LoopState,
    config: LoopConfig,
    now: Callable[[], float],
) -> _ToolExecutionResult:
    try:
        arguments = _parse_arguments(tool_call.arguments)
    except (json.JSONDecodeError, ValueError) as exc:
        outcome = ToolOutcome(
            content_markdown=f"Invalid tool arguments: {exc}",
            method="internal",
            status="error",
        )
        return _ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=_format_tool_content(tool_call.name, outcome, config.max_result_chars),
        )

    if tool_call.name in _INTERNAL_TOOL_NAMES:
        timeout_s = _INTERNAL_TOOL_TIMEOUT_S
    else:
        spec = tools_by_name.get(tool_call.name)
        if spec is None:
            outcome = ToolOutcome(
                content_markdown=f"Unknown tool: {tool_call.name}",
                method="internal",
                status="error",
            )
            return _ToolExecutionResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=_format_tool_content(tool_call.name, outcome, config.max_result_chars),
            )
        timeout_s = spec.timeout_s

    try:
        # Instantiate the handler coroutine INSIDE the boundary. External-tool
        # handlers have concrete signatures, and async-def binds kwargs eagerly:
        # a missing/typo'd/extra key in the LLM-emitted `arguments` raises
        # TypeError at bind time, before any await. Doing the bind here means
        # that failure becomes a status="error" outcome (via the except below)
        # instead of escaping the batch gather and aborting the whole pass —
        # matching the unknown-tool path. Internal tools bind positionally and
        # can't hit this.
        if tool_call.name == "set_research_plan":
            handler = _set_research_plan_tool(state, arguments, config)
        elif tool_call.name == "record_findings":
            handler = _record_findings_tool(state, arguments)
        elif tool_call.name == "conclude":
            handler = _conclude_tool(state, arguments, config, now)
        else:
            handler = tools_by_name[tool_call.name].handler(**arguments)
        raw_outcome = await asyncio.wait_for(handler, timeout=timeout_s)
        outcome = ToolOutcome.model_validate(raw_outcome)
    except asyncio.TimeoutError:
        outcome = ToolOutcome(
            content_markdown=f"Tool timed out after {timeout_s:.2f}s.",
            method="internal",
            status="timeout",
        )
    except ValidationError as exc:
        outcome = ToolOutcome(
            content_markdown=f"Tool returned invalid outcome: {exc.errors()[0]['msg']}",
            method="internal",
            status="error",
        )
    except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # tool-execution boundary: any tool failure becomes an error outcome, never a loop crash
        outcome = ToolOutcome(
            content_markdown=f"{type(exc).__name__}: {exc}",
            method="internal",
            status="error",
        )

    provenance_urls, provenance_text = _harvest_provenance(tool_call.name, arguments, outcome)
    provenance_tiers = _harvest_verification_tiers(tool_call.name, arguments, outcome)
    return _ToolExecutionResult(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=_format_tool_content(tool_call.name, outcome, config.max_result_chars),
        method=outcome.method,
        provenance_urls=provenance_urls,
        provenance_text=provenance_text,
        provenance_tiers=provenance_tiers,
    )


def _surfaced_urls(arguments: dict[str, Any], outcome: ToolOutcome) -> list[str]:
    """Normalized URLs an EXTERNAL tool call actually surfaced.

    Exactly the ``url`` the driver asked the tool to retrieve (fetch/read_document
    take one — a search's ``query`` is a search term, not a retrieval target)
    plus every URL in the result body and link list. A URL merely typed into a
    free-text argument (read_document's ``ask``, a ``query``) was NOT retrieved by
    the tool, so it is deliberately excluded: that is what stops a URL a driver
    pastes into ``ask`` from laundering itself into provenance or a "fetched" tier
    (F1). Both ``_harvest_provenance`` and the snippet tier path share this so a
    URL can never be provenance-seen without a matching tier, or vice versa (F7).
    """
    urls: list[str] = []
    requested = arguments.get("url")
    if isinstance(requested, str):
        urls.extend(_iter_normalized_urls(requested))
    urls.extend(_iter_normalized_urls(outcome.content_markdown))
    for link in outcome.links:
        normalized = _normalize_url(link)
        if normalized is not None:
            urls.append(normalized)
    return urls


def _harvest_provenance(tool_name: str, arguments: dict[str, Any], outcome: ToolOutcome) -> tuple[list[str], str]:
    """Collect the normalized URLs and result text a single EXTERNAL tool call surfaced.

    Internal bookkeeping tools contribute nothing: their echoed content restates
    the driver's own rejected findings, so harvesting them would let a
    hallucinated URL launder itself into ``tool_seen_urls``.
    """
    if tool_name in _INTERNAL_TOOL_NAMES:
        return [], ""
    return _surfaced_urls(arguments, outcome), outcome.content_markdown


def _harvest_verification_tiers(tool_name: str, arguments: dict[str, Any], outcome: ToolOutcome) -> dict[str, str]:
    """Assign a retrieval tier to the URLs this EXTERNAL tool call established (W4).

    Only a successful (``status == "ok"``) outcome grants a tier — a 403'd/
    blocked fetch confers no authority, which is the exact 131.3 mechanism (the
    real fetch failed, so a later search snippet must not inherit "fetched").

    A **fetched-class** call (document/rendered/plain/cache) tiers ONLY the page
    it actually retrieved — the ``url`` argument the driver asked for — as
    "fetched". A URL merely named in a free-text ``ask`` (or in the result
    body/links) is a lead, not a page we read, so it earns no tier from this call
    (F1: an ``ask``-URL must not inherit fetched authority). A **snippet-class**
    call (search/news) tiers every URL it surfaced (the exact set provenance
    harvests — requested ``url``, body, links) as "snippet": the driver saw only
    the excerpt, never the page.
    """
    if tool_name in _INTERNAL_TOOL_NAMES or outcome.status != "ok":
        return {}
    tier = _method_to_tier(outcome.method)
    if tier is None:
        return {}
    if tier == "fetched":
        # Only the requested page (the `url` argument) counts as retrieved.
        requested = arguments.get("url")
        if not isinstance(requested, str):
            return {}
        return {normalized: tier for normalized in _iter_normalized_urls(requested)}
    # snippet: every surfaced URL was seen only as an excerpt. Reuse provenance's
    # URL set so tier and provenance can't drift (F7).
    return {normalized: tier for normalized in _surfaced_urls(arguments, outcome)}


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


async def _execute_tool_batch(
    tool_calls: list[_ToolCall],
    *,
    tools_by_name: dict[str, ToolSpec],
    state: _LoopState,
    config: LoopConfig,
    now: Callable[[], float],
) -> None:
    duplicate_call_ids: set[str] = set()
    rejected_call_ids: set[str] = set()
    plan_rejected_call_ids: set[str] = set()
    accepted: list[_ToolCall] = []

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
            plan_rejected_call_ids.add(tool_call.id)
            continue
        if not is_internal and state.telemetry.tool_calls >= config.max_tool_calls:
            rejected_call_ids.add(tool_call.id)
            continue

        state.telemetry.tool_calls += 1
        state.telemetry.per_tool_counts[tool_call.name] = state.telemetry.per_tool_counts.get(tool_call.name, 0) + 1
        call_key = _normalized_call_key(tool_call)
        if call_key in state.seen_tool_calls:
            state.telemetry.dup_tool_calls += 1
            duplicate_call_ids.add(tool_call.id)
        else:
            state.seen_tool_calls.add(call_key)
        accepted.append(tool_call)

    # Record the plan nudge (once per gated batch) and flip to soft-continue
    # once the cap is hit, so the NEXT batch's external calls run un-gated.
    if plan_rejected_call_ids:
        state.plan_nudges += 1
        if state.plan_nudges >= config.max_plan_nudges:
            state.plan_skipped = True
            state.telemetry.plan_skipped = True

    results = await asyncio.gather(
        *[_execute_one_tool_call(tool_call, tools_by_name, state, config, now) for tool_call in accepted]
    )
    provenance_texts: list[str] = []
    for result in results:
        if result.method == "rendered":
            state.telemetry.rendered_fetches += 1
        # Accumulate provenance so a LATER turn's record_findings/conclude can
        # verify a finding's source_url against what the driver actually
        # retrieved. Internal tools contribute nothing (see _harvest_provenance).
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
    results_by_id = {result.tool_call_id: result for result in results}

    # Exactly one tool message per tool_call_id, in the assistant's original
    # order, or the next LLM turn 400s. Rejected calls get a synthetic
    # budget-exhausted error response.
    budget_line = _budget_line(state, config, now)
    for tool_call in tool_calls:
        if tool_call.id in plan_rejected_call_ids:
            content = _plan_rejected_content(tool_call.name, config)
        elif tool_call.id in rejected_call_ids:
            content = _budget_rejected_content(tool_call.name, config)
        else:
            result = results_by_id[tool_call.id]
            content = result.content + (_DUPLICATE_CALL_WARNING if tool_call.id in duplicate_call_ids else "")
        state.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": content + budget_line,
            }
        )


def _freeze_result(state: _LoopState, findings_markdown: str, ghost: GhostForecast | None) -> LoopResult:
    state.telemetry.findings_count = len(state.findings)
    state.telemetry.pending_leads_count = len(state.pending_leads)
    state.telemetry.concluded_early = state.explicit_conclude and not state.telemetry.deadline_hit
    state.telemetry.wall_s = max(0.0, state.telemetry.wall_s)
    return LoopResult(
        findings_markdown=findings_markdown,
        ghost=ghost,
        telemetry=state.telemetry,
        transcript=copy.deepcopy(state.messages),
    )


def _log_completion(state: _LoopState, log_prefix: str) -> None:
    # Marker shape per plan §6: model + per-surface counters make the run_logs
    # grep enough for the driver vibe-eval (no research-archive JSON needed).
    per_tool = state.telemetry.per_tool_counts
    searches = per_tool.get("search_news", 0) + per_tool.get("search_web", 0)
    logger.info(
        "%sGAP_FILL_V2: model=%s steps=%s tool_calls=%s searches=%s fetches=%s rendered=%s reads=%s "
        "dup_tool_calls=%s deadline_hit=%s concluded_early=%s wall_s=%.2f findings=%s "
        "pending_leads=%s lint_rejections=%s provenance_rejections=%s quote_mismatch_warnings=%s "
        "plan_gaps=%s plan_skipped=%s conclude_gate_rejections=%s error=%s",
        log_prefix,
        state.telemetry.model,
        state.telemetry.steps,
        state.telemetry.tool_calls,
        searches,
        per_tool.get("fetch", 0),
        state.telemetry.rendered_fetches,
        per_tool.get("read_document", 0),
        state.telemetry.dup_tool_calls,
        state.telemetry.deadline_hit,
        state.explicit_conclude and not state.telemetry.deadline_hit,
        state.telemetry.wall_s,
        len(state.findings),
        len(state.pending_leads),
        state.telemetry.lint_rejections,
        state.telemetry.provenance_rejections,
        state.telemetry.quote_mismatch_warnings,
        state.telemetry.plan_gaps,
        state.telemetry.plan_skipped,
        state.telemetry.conclude_gate_rejections,
        state.telemetry.error,
    )


_GHOST_QTYPES: tuple[Literal["binary", "multiple_choice", "numeric"], ...] = (
    "binary",
    "multiple_choice",
    "numeric",
)


def _declared_qtype(raw_text: str) -> Literal["binary", "multiple_choice", "numeric"] | None:
    """Peek the ghost block's self-declared ``question_type`` without parsing it.

    The ghost emits exactly one structured block that names its own
    ``question_type`` (the schema discriminator). Reading it lets
    ``_summarize_ghost`` parse only the matching type instead of trying all
    three — the two non-matching attempts would otherwise trip
    ``parse_structured_payload``'s ``question_type`` mismatch guard, which WARNs
    by construction (a numeric ghost logged 2 such WARNs, an MC ghost 1). Returns
    ``None`` when no block is present, the JSON is malformed, or the declared
    type is missing/unsupported, so the caller can fall back to trying every type.
    """
    raw = extract_json_block(raw_text)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    declared = payload.get("question_type")
    return declared if declared in _GHOST_QTYPES else None


def _summarize_ghost(raw_text: str) -> tuple[str, str, dict[str, Any] | None]:
    """Parse the ghost's structured block once into ``(qtype, legacy_summary, forecast)``.

    ``legacy_summary`` is the lossy human-readable string kept for the
    back-compat ``GHOST_FORECAST`` marker. ``forecast`` is the full,
    deterministically-parsable payload for the ``GHOST_FORECAST_JSON`` marker:
    the posterior probability (binary), the complete option->prob dict (MC), or
    the complete percentile->value dict plus the median (numeric). ``None`` when
    no block parses — the JSON marker line is then suppressed and only the
    legacy ``unknown``/``""`` marker is emitted.

    Only the block's self-declared ``question_type`` is parsed (see
    ``_declared_qtype``); the all-types fallback runs only when no type could be
    read, keeping the mismatch-WARN path off the normal ghost flow.
    """
    declared = _declared_qtype(raw_text)
    candidate_qtypes = (declared,) if declared is not None else _GHOST_QTYPES
    for qtype in candidate_qtypes:
        block = parse_structured_block(raw_text, qtype)
        if isinstance(block, BinaryStructured):
            prob = float(block.posterior_prob)
            return "binary", f"posterior_prob={prob:.4f}", {"qtype": "binary", "prob": prob}
        if isinstance(block, MultipleChoiceStructured):
            option_probs = {name: float(prob) for name, prob in block.option_probs.items()}
            summary = ", ".join(f"{name}={prob:.3f}" for name, prob in sorted(option_probs.items()))
            return "multiple_choice", summary, {"qtype": "multiple_choice", "option_probs": option_probs}
        if isinstance(block, NumericStructured) and block.declared_percentiles:
            declared = {float(pct): float(value) for pct, value in block.declared_percentiles.items()}
            median = declared.get(0.5)
            summary = "" if median is None else f"median={median}"
            return "numeric", summary, {"qtype": "numeric", "declared_percentiles": declared, "median": median}
    return "unknown", "", None


async def _run_ghost_phase(
    *,
    state: _LoopState,
    ghost_prompt: str,
    llm_call: LlmCall,
    log_prefix: str,
) -> GhostForecast | None:
    state.messages.append({"role": "user", "content": ghost_prompt})
    try:
        response = await asyncio.wait_for(llm_call(state.messages, None), timeout=60.0)
        assistant_message = _parse_response_message(response)
        state.messages.append(assistant_message)
    except asyncio.TimeoutError:
        logger.warning("%sGhost phase timed out after 60s", log_prefix)
        return None
    except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # telemetry-only phase
        logger.warning("%sGhost phase failed: %s: %s", log_prefix, type(exc).__name__, exc)
        return None

    raw_text = assistant_message["content"]
    qtype, parsed_summary, forecast = _summarize_ghost(raw_text)
    ghost = GhostForecast(qtype=qtype, raw_text=raw_text, parsed_summary=parsed_summary)
    logger.info("%sGHOST_FORECAST: qtype=%s summary=%s", log_prefix, ghost.qtype, ghost.parsed_summary)
    # Additive full-fidelity companion marker: the legacy line above stays
    # byte-identical (harvested archive + other tests depend on it); this one
    # carries the complete, deterministically-parsable forecast so numeric
    # ghosts (not just their median) are scoreable. qid is carried the same way
    # as GHOST_FORECAST — via ``log_prefix`` — so the harvester derives it
    # identically. Suppressed when no block parsed (nothing to serialize).
    if forecast is not None:
        logger.info("%sGHOST_FORECAST_JSON: %s", log_prefix, json.dumps(forecast, separators=(",", ":")))
    return ghost


async def _run_loop_body(
    *,
    state: _LoopState,
    tools: list[ToolSpec],
    config: LoopConfig,
    llm_call: LlmCall,
    ghost_prompt: str | None,
    log_prefix: str,
    now: Callable[[], float],
) -> LoopResult:
    tools_by_name = {tool.name: tool for tool in tools}

    while not state.stop_loop and state.telemetry.steps < config.max_steps:
        tools_json = _tool_schemas(tools, _must_conclude(state, config, now))
        response = await llm_call(state.messages, tools_json)
        assistant_message = _parse_response_message(response)
        state.messages.append(assistant_message)
        state.telemetry.steps += 1

        tool_calls = _extract_tool_calls(assistant_message)
        if tool_calls:
            await _execute_tool_batch(tool_calls, tools_by_name=tools_by_name, state=state, config=config, now=now)
            continue

        if state.nudged_for_no_action:
            state.stop_loop = True
            break

        state.messages.append({"role": "user", "content": _NUDGE})
        state.nudged_for_no_action = True

    state.telemetry.wall_s = now() - state.started_at_s
    findings_markdown = render_findings(state.findings, state.pending_leads)
    ghost: GhostForecast | None = None
    if ghost_prompt is not None and state.explicit_conclude:
        ghost = await _run_ghost_phase(state=state, ghost_prompt=ghost_prompt, llm_call=llm_call, log_prefix=log_prefix)

    _log_completion(state, log_prefix)
    return _freeze_result(state, findings_markdown, ghost)


async def run_agentic_loop(
    system_prompt: str,
    user_brief: str,
    tools: list[ToolSpec],
    config: LoopConfig,
    llm_call: LlmCall | None = None,
    ghost_prompt: str | None = None,
    *,
    log_prefix: str = "",
    now: Callable[[], float] | None = None,
) -> LoopResult:
    now_fn = now or time.monotonic
    call = llm_call or build_default_llm_call(config)
    state = _LoopState(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_brief},
        ],
        started_at_s=now_fn(),
        deadline_at_s=now_fn() + config.wall_deadline_s,
        log_prefix=log_prefix,
    )
    state.telemetry.model = config.model
    # Seed the provenance sets from the frozen brief: its embedded URLs
    # (resolution-source snapshot, market snapshot, AskNews digests) are things
    # the driver saw, so a NON-discrepancy finding may cite them. The system
    # prompt is a fixed template that embeds no question URLs. A discrepancy
    # finding may NOT lean on these — see _check_url_provenance.
    state.briefing_urls = set(_iter_normalized_urls(user_brief))

    try:
        return await asyncio.wait_for(
            _run_loop_body(
                state=state,
                tools=tools,
                config=config,
                llm_call=call,
                ghost_prompt=ghost_prompt,
                log_prefix=log_prefix,
                now=now_fn,
            ),
            timeout=config.wall_deadline_s,
        )
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        state.telemetry.deadline_hit = True
        state.telemetry.wall_s = now_fn() - state.started_at_s
        findings_markdown = render_findings(state.findings, state.pending_leads)
        _log_completion(state, log_prefix)
        return _freeze_result(state, findings_markdown, None)
    except Exception as exc:  # noqa: BLE001, HARNESS-SCAN-EXEMPT-broad-except  # sanctioned package boundary: mirror v1 soft-fail contract and never raise past the harness except on cancellation
        logger.exception("%sAgentic loop failed; soft-failing to banked findings if any", log_prefix)
        # Stamp the crash so the completion marker (error=...) and the
        # orchestrator's alertable counter can tell this apart from an idle
        # "found nothing" run — the TimeoutError branch above deliberately does
        # NOT set this (a deadline hit is expected degradation, not a crash).
        state.telemetry.error = repr(exc)
        state.telemetry.wall_s = now_fn() - state.started_at_s
        findings_markdown = render_findings(state.findings, state.pending_leads)
        _log_completion(state, log_prefix)
        return _freeze_result(state, findings_markdown, None)


__all__ = ["LlmCall", "run_agentic_loop"]
