"""Replay the archived fetch outcomes through the unified fetch ladder's rung selection.

A local pre-merge check for ``metaculus_bot/research/fetch_ladder/``. It reads the gitignored
research archive, reconstructs each archived per-URL fetch outcome as the direct
:class:`FetchResult` behind it, and runs the new dispatcher's escalation over that result with every
rung stubbed to record its attempt and then decline. Nothing is dialed, so the run is free.

What this measures. Every rung declines by construction, so the replayed final status always
equals the reconstructed direct status and no rescue can happen. What the replay says is which
rungs each policy preset would CONSULT for each archived direct outcome and in what order, and
which archived rescues (impersonate, rendered, derived_api, wayback, url_context) the new
dispatcher would no longer attempt at all. A rung counts as consulted when its own pure trigger
predicate passes; the environment gates behind that trigger (a flag, an API key, a remembered
derived endpoint, the robots pre-check, the wall budget, the per-question caps) each need a live
run and are not modelled, so the replayed route is the LAST rung consulted, not one that served
bytes.

Two archive limits shape the loop half. A gap-fill v2 tool result carries no HTTP status header, so
it is recovered from the two body templates that disclose one and read through the fetcher's own
table; without that, no loop refusal could reach the impersonated retry, whose trigger is a 403.
And an outcome produced by the loop's own rescue rung, memo or document ladder names the rescuing
method but not the direct status it escalated from, so those are counted rather than replayed.

Usage:
    uv run python scripts/fetch_ladder_replay.py
    uv run python scripts/fetch_ladder_replay.py --archive-dir ../metaculus-bot/backtests/research_archive
    uv run python scripts/fetch_ladder_replay.py --policy RESOLUTION_SOURCE_POLICY --format json
    make replay_ladder ARGS="--limit 200"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, get_args
from unittest.mock import patch
from urllib.parse import urlparse

from metaculus_bot.research.agentic.fetch_outcomes import DOCUMENT_NEEDED_METHOD
from metaculus_bot.research.fetch_ladder import context, ladder, rungs
from metaculus_bot.research.fetch_ladder import policy as ladder_policy
from metaculus_bot.research.fetch_ladder.policy import LadderPolicy
from metaculus_bot.research.resolution_fetch_result import (
    _NON_OK_FETCH_STATUS,
    FetchResult,
    FetchRoute,
    FetchStatus,
    FetchStatusReason,
)

logger = logging.getLogger(__name__)

DEFAULT_ARCHIVE_DIR = Path(__file__).resolve().parents[1] / "backtests" / "research_archive"
RAW_SUBDIR = "raw"
BY_QID_SUBDIR = "by_qid"
FETCHER_PROVIDER = "resolution_source"
LOOP_TRANSCRIPT_FIELD = "gap_fill_v2"
LOOP_FETCH_TOOLS = ("fetch", "read_document")
CHANGED_CELL_LIMIT = 20

_FETCH_STATUSES: frozenset[str] = frozenset(get_args(FetchStatus))
_FETCH_ROUTES: frozenset[str] = frozenset(get_args(FetchRoute))
_FETCH_STATUS_REASONS: frozenset[str] = frozenset(get_args(FetchStatusReason))

# The rungs the ESCALATION runs, which is every route the direct fetch cannot produce itself.
ESCALATION_RUNGS: frozenset[str] = _FETCH_ROUTES - {"direct", "meta_refresh", "pdf_local"}

# No trigger predicate reads the text; only FetchResult's success-implies-content invariant needs it.
_REPLAY_TEXT_STANDIN = "[archived body not replayed; the ladder's rung triggers read only the status]"

# Anchored on `agentic.fetch_outcomes._non_ok_status_result`, whose two templates are f-strings with no constant to import.
_LOOP_HTTP_STATUS_RE = re.compile(r"^Fetch (?:blocked|failed) with HTTP (\d{3})\.$", re.MULTILINE)
_LOOP_HEADER_RE = re.compile(r"^(status|method): (.+)$", re.MULTILINE)

# A stable substring of `_PLATFORM_FETCH_BLOCK_MSG`, whose tail gained Mantic after these transcripts were written.
_LOOP_PLATFORM_REFUSAL_MARKER = "already reflected in the question brief"

# `agentic.dispatch._format_tool_content` inserts a blank line before a body and nothing when there is none.
_LOOP_BODY_SEPARATOR = "\n\n"


# Named rather than a bool so each table row says which of the two it is at the call site.
LoopFidelity = Literal["exact", "ambiguous"]


@dataclass(frozen=True, slots=True)
class LoopStatusMapping:
    """One gap-fill v2 tool status read back as the fetcher status vocabulary."""

    fetch_status: FetchStatus | None
    fidelity: LoopFidelity
    note: str

    @property
    def ambiguous(self) -> bool:
        return self.fidelity == "ambiguous"


# A module constant so the loop's forward mapping can be checked against this inverse of it.
LOOP_STATUS_TO_FETCH_STATUS: dict[str, LoopStatusMapping] = {
    "ok": LoopStatusMapping("success", "exact", "the loop read text, which is what makes a fetch a success"),
    "blocked": LoopStatusMapping(
        "blocked", "exact", "the same token and the same producers, 403/406/429 and the self-reference refusal"
    ),
    "error": LoopStatusMapping(
        "error", "exact", "the loop's catch-all for a transport fault or a non-200 with no table entry"
    ),
    "empty": LoopStatusMapping(
        "js_wall", "ambiguous", "one loop token spans the fetcher's js_wall and its no_resolving_content verdicts"
    ),
    "throttled": LoopStatusMapping(
        "js_wall", "ambiguous", "Tier 1 has no throttle-phrase check, so such a 200 classifies as js_wall or thin_page"
    ),
    "robots_disallowed": LoopStatusMapping(
        None,
        "ambiguous",
        "the fetcher has no page-level robots refusal; only the paid rung pre-checks, and records a skip",
    ),
    "timeout": LoopStatusMapping(
        "error", "ambiguous", "the fetcher's wall cancels the fetch rather than giving it a status of its own"
    ),
}

# `document_needed` is a METHOD carried on an `ok` status, so it is read before the table above.
LOOP_METHOD_TO_FETCH_STATUS: dict[str, LoopStatusMapping] = {
    DOCUMENT_NEEDED_METHOD: LoopStatusMapping(
        "unsupported_type",
        "ambiguous",
        "nothing was read yet; the loop escalates to read_document where the fetcher reads locally",
    ),
}

# The plain rung's own verdicts: the only loop outcomes that ARE a direct fetch's outcome.
LOOP_PLAIN_METHODS = ("plain", "empty", "throttled")

# The loop's own rescue, memo and document-ladder methods: the transcript never records what they escalated from.
LOOP_RESCUE_METHODS = (
    "cache",
    "rendered",
    "impersonate",
    "derived_api",
    "wayback",
    "pdf_local",
    "document",
    "digest_local",
)

# A tool that timed out or was rejected before its handler ran, which is no fetch outcome at all.
LOOP_INTERNAL_METHODS = ("internal",)


@dataclass(frozen=True, slots=True)
class ArchivedFetch:
    """One archived per-URL fetch outcome, read back as the direct result behind it."""

    source: Literal["fetcher", "loop"]
    qid: str
    url: str
    direct_status: FetchStatus | None
    status_reason: FetchStatusReason | None
    http_status: int | None
    content_type: str | None
    archived_route: str
    notes: tuple[str, ...]
    archived_status: str = ""
    route_recorded: bool = False
    # False once a rung that SERVED BYTES overwrote the direct fetch's http_status with its own.
    http_status_is_direct: bool = True

    @property
    def replayable(self) -> bool:
        return self.direct_status is not None

    @property
    def host(self) -> str:
        return urlparse(self.url).netloc or "-"


@dataclass(frozen=True, slots=True)
class Replay:
    """What one preset's dispatcher did with one archived outcome."""

    record: ArchivedFetch
    preset: str
    sequence: tuple[str, ...]
    skips: tuple[str, ...]
    final_status: str
    final_route: str

    @property
    def _rescue_unattempted(self) -> bool:
        return self.record.archived_route in ESCALATION_RUNGS and self.record.archived_route not in self.sequence

    @property
    def rescue_lost(self) -> bool:
        """The archive's rescuing rung is one this preset's dispatcher would never even consult."""
        return self._rescue_unattempted and self.record.http_status_is_direct

    @property
    def rescue_undecidable(self) -> bool:
        """The rung is absent, but its trigger reads a direct http_status the rescue overwrote."""
        return self._rescue_unattempted and not self.record.http_status_is_direct


@dataclass(frozen=True, slots=True)
class Report:
    presets: tuple[str, ...]
    records: tuple[ArchivedFetch, ...]
    replays: tuple[Replay, ...]
    parse_failures: tuple[str, ...]


class _DecliningRungs:
    """Every escalation rung, stubbed to record its attempt on the context and then decline.

    Each stub honours that rung's own pure trigger predicate and nothing else, which is what makes
    the recorded sequence the dispatcher's rung SELECTION rather than a fixed list. The signatures
    mirror the real ones exactly so a rung signature change breaks the replay loudly.
    """

    def __init__(self, real: ModuleType) -> None:
        self._real = real

    def _consult(self, rung: FetchRoute, *, applies: bool, direct: FetchResult, url: str, ctx: Any) -> None:
        if applies:
            ctx.start_rung(rung, direct.status, url)

    def _rendered_rung_applies(self, direct: FetchResult) -> bool:
        return self._real._rendered_rung_applies(direct)

    async def _impersonate_rung(self, url: str, direct: FetchResult, *, host_sems: Any, ctx: Any) -> None:
        return self._consult(
            "impersonate", applies=self._real._impersonate_rung_applies(direct), direct=direct, url=url, ctx=ctx
        )

    async def _derived_api_rung(self, session: Any, url: str, direct: FetchResult, *, host_sems: Any, ctx: Any) -> None:
        return self._consult(
            "derived_api", applies=self._real._rendered_rung_applies(direct), direct=direct, url=url, ctx=ctx
        )

    async def _rendered_rung(self, url: str, direct: FetchResult, host_sems: Any, ctx: Any) -> None:
        return self._consult(
            "rendered", applies=self._real._rendered_rung_applies(direct), direct=direct, url=url, ctx=ctx
        )

    async def _wayback_rung(self, session: Any, url: str, direct: FetchResult, *, host_sems: Any, ctx: Any) -> None:
        triggers = self._real._WAYBACK_TRIGGER_STATUSES
        return self._consult("wayback", applies=direct.status in triggers, direct=direct, url=url, ctx=ctx)

    async def _url_context_rung(self, session: Any, url: str, direct: FetchResult, *, host_sems: Any, ctx: Any) -> None:
        return self._consult(
            "url_context", applies=self._real._url_context_rung_applies(direct), direct=direct, url=url, ctx=ctx
        )


def discover_presets(names: Sequence[str] | None) -> dict[str, LadderPolicy]:
    """Every ``LadderPolicy`` attribute of the policy module, or the named subset.

    Discovered at runtime rather than imported by name so a preset added beside
    ``RESOLUTION_SOURCE_POLICY`` is replayed the moment it lands.
    """
    found = {
        name: value
        for name in dir(ladder_policy)
        if not name.startswith("_") and isinstance(value := getattr(ladder_policy, name), LadderPolicy)
    }
    if names is None:
        if not found:
            sys.exit(f"no LadderPolicy preset found on {ladder_policy.__name__}")
        return found
    missing = [name for name in names if name not in found]
    if missing:
        sys.exit(
            f"--policy names no preset on {ladder_policy.__name__}: {', '.join(missing)} (found: {', '.join(sorted(found))})"
        )
    return {name: found[name] for name in names}


def _validated(value: Any, allowed: frozenset[str], label: str, failures: list[str]) -> str | None:
    """``value`` when it is a member of a pinned token vocabulary, else None with a failure noted."""
    if value is None:
        return None
    if not isinstance(value, str) or value not in allowed:
        failures.append(f"{label} is not a known token: {value!r}")
        return None
    return value


def _direct_from_attempts(attempts: Any) -> tuple[str | None, tuple[str, ...]]:
    """The direct status behind a rescued record, taken off its first ESCALATION attempt.

    An escalation rung's ``from_status`` is the direct fetch's own outcome by contract, which is
    what the ``FetchResult.status`` accounting note points a reader at. The direct
    ``status_reason`` is not recoverable and is dropped: a rung whose trigger excludes a reason
    cannot have fired on it, so None is faithful to every trigger predicate.
    """
    if not isinstance(attempts, list):
        return None, ()
    for attempt in attempts:
        if isinstance(attempt, dict) and attempt.get("rung") in ESCALATION_RUNGS and not attempt.get("skipped_reason"):
            return attempt.get("from_status"), (f"direct status read off the {attempt['rung']} attempt's from_status",)
    return None, ()


def _fetcher_record(entry: dict[str, Any], qid: str, failures: list[str]) -> ArchivedFetch | None:
    archived_status = _validated(entry.get("status"), _FETCH_STATUSES, "fetcher status", failures)
    if archived_status is None:
        return None
    url = entry.get("url") or ""
    if archived_status == "success" and not (entry.get("text") or ""):
        failures.append(f"a success with no text violates the FetchResult invariant: {url}")
        return None
    archived_route = _validated(entry.get("route"), _FETCH_ROUTES, "fetcher route", failures)
    notes = () if archived_route else ("the record predates the route field, so its route reads as direct",)
    from_status, attempt_notes = _direct_from_attempts(entry.get("rung_attempts"))
    served_bytes = archived_route in ESCALATION_RUNGS and archived_status == "success"
    if served_bytes:
        attempt_notes += (f"http_status is the {archived_route} rescue's own, so the direct fetch's is lost",)
    direct_status = _validated(from_status, _FETCH_STATUSES, "escalation from_status", failures) or archived_status
    reason = (
        None if attempt_notes else _validated(entry.get("status_reason"), _FETCH_STATUS_REASONS, "reason", failures)
    )
    return ArchivedFetch(
        source="fetcher",
        qid=qid,
        url=url,
        direct_status=direct_status,  # pyright: ignore[reportArgumentType]  # validated against get_args(FetchStatus)
        status_reason=reason,  # pyright: ignore[reportArgumentType]  # validated against get_args(FetchStatusReason)
        http_status=None if served_bytes else entry.get("http_status"),
        content_type=entry.get("content_type"),
        archived_route=archived_route or "direct",
        notes=notes + attempt_notes,
        archived_status=archived_status,
        route_recorded=archived_route is not None,
        http_status_is_direct=not served_bytes,
    )


def _jsonl_lines(path: Path, failures: list[str]) -> Iterator[dict[str, Any]]:
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(f"{path.name}:{number} is not JSON: {exc}")
            continue
        if isinstance(parsed, dict):
            yield parsed


def fetcher_records(archive_dir: Path, failures: list[str] | None = None) -> Iterator[ArchivedFetch]:
    """Every per-URL resolution-source fetch outcome in the archive's raw run files."""
    failures = [] if failures is None else failures
    for path in sorted((archive_dir / RAW_SUBDIR).glob("*.jsonl")):
        for entry in _jsonl_lines(path, failures):
            if entry.get("provider") != FETCHER_PROVIDER or not isinstance(entry.get("payload"), list):
                continue
            qid = str(entry.get("qid", "-"))
            for item in entry["payload"]:
                record = _fetcher_record(item, qid, failures) if isinstance(item, dict) else None
                if record is not None:
                    yield record


def parse_tool_headers(content: str) -> tuple[str, str]:
    """The ``status:`` and ``method:`` headers a gap-fill v2 tool result opens with."""
    headers = dict(_LOOP_HEADER_RE.findall(content))
    return headers.get("status", "").strip(), headers.get("method", "").strip()


def recover_http_status(content: str) -> int | None:
    """The HTTP status a blocked or failed loop fetch names in its body, or None."""
    match = _LOOP_HTTP_STATUS_RE.search(content)
    return int(match.group(1)) if match else None


def _loop_mapping(loop_status: str, method: str, content: str) -> tuple[LoopStatusMapping, int | None]:
    """The fetcher status behind one loop tool result, and the HTTP status it disclosed."""
    if method in LOOP_METHOD_TO_FETCH_STATUS:
        return LOOP_METHOD_TO_FETCH_STATUS[method], None
    http_status = recover_http_status(content)
    if http_status is not None:
        mapped = _NON_OK_FETCH_STATUS.get(http_status, "error")
        return LoopStatusMapping(
            mapped, "exact", f"HTTP {http_status} read through the fetcher's own status table"
        ), http_status
    if loop_status not in LOOP_STATUS_TO_FETCH_STATUS:
        return LoopStatusMapping(None, "ambiguous", f"no inverse entry for the loop status {loop_status!r}"), None
    return LOOP_STATUS_TO_FETCH_STATUS[loop_status], None


def _loop_record(content: str, url: str, qid: str) -> ArchivedFetch:
    loop_status, method = parse_tool_headers(content)
    if method not in LOOP_PLAIN_METHODS and method not in LOOP_METHOD_TO_FETCH_STATUS:
        note = (
            f"the transcript records the {method} outcome but not the direct status it escalated from"
            if method in LOOP_RESCUE_METHODS or method in LOOP_INTERNAL_METHODS
            else f"unknown gap-fill v2 fetch method {method!r}"
        )
        return ArchivedFetch(
            "loop", qid, url, None, None, None, None, method or "-", (note,), loop_status, route_recorded=True
        )
    mapping, http_status = _loop_mapping(loop_status, method, content)
    reason = "metaculus_self_ref" if _LOOP_PLATFORM_REFUSAL_MARKER in content else None
    notes = (mapping.note,) if mapping.ambiguous or mapping.fetch_status is None else ()
    if mapping.fetch_status == "success" and _LOOP_BODY_SEPARATOR not in content:
        notes += ("the loop reported ok with no body, which the fetcher would call empty_body or js_wall",)
    return ArchivedFetch(
        source="loop",
        qid=qid,
        url=url,
        direct_status=mapping.fetch_status,
        status_reason=reason,
        http_status=http_status,
        content_type=None,
        archived_route=method or "-",
        notes=notes,
        archived_status=loop_status,
        route_recorded=True,
    )


def _tool_call_urls(transcript: Sequence[Any]) -> dict[str, str]:
    """Tool-call id to fetched URL, off the assistant turns that requested the fetches."""
    urls: dict[str, str] = {}
    for message in transcript:
        if not isinstance(message, dict) or not isinstance(message.get("tool_calls"), list):
            continue
        for call in message["tool_calls"]:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            if function.get("name") not in LOOP_FETCH_TOOLS:
                continue
            arguments = json.loads(function.get("arguments") or "{}")
            urls[call.get("id", "")] = arguments.get("url") or ""
    return urls


def loop_records(archive_dir: Path, failures: list[str] | None = None) -> Iterator[ArchivedFetch]:
    """Every gap-fill v2 fetch or document-read tool result in the archive's per-question files."""
    failures = [] if failures is None else failures
    for path in sorted((archive_dir / BY_QID_SUBDIR).glob("*.jsonl")):
        for entry in _jsonl_lines(path, failures):
            transcript = (entry.get(LOOP_TRANSCRIPT_FIELD) or {}).get("transcript")
            if not isinstance(transcript, list):
                continue
            urls = _tool_call_urls(transcript)
            qid = str(entry.get("qid", "-"))
            for message in transcript:
                if message.get("role") != "tool" or message.get("name") not in LOOP_FETCH_TOOLS:
                    continue
                yield _loop_record(message.get("content") or "", urls.get(message.get("tool_call_id", ""), ""), qid)


def direct_result(record: ArchivedFetch) -> FetchResult:
    """The direct :class:`FetchResult` an archived outcome was produced by."""
    if record.direct_status is None:
        raise ValueError(f"{record.url} carries no recoverable direct status")
    return FetchResult(
        url=record.url,
        status=record.direct_status,
        text=_REPLAY_TEXT_STANDIN if record.direct_status == "success" else "",
        http_status=record.http_status,
        content_type=record.content_type,
        status_reason=record.status_reason,
    )


async def _replay_one(record: ArchivedFetch, preset: str, preset_policy: LadderPolicy) -> Replay:
    ctx = context.LadderContext(policy=preset_policy, host_sems={})
    direct = direct_result(record)
    escalated = await ladder._escalate_unresolved(None, record.url, direct, host_sems={}, ctx=ctx)
    final = context._stamped_with_route(escalated, ctx)
    return Replay(
        record=record,
        preset=preset,
        sequence=tuple(a.rung for a in ctx.rungs if not a.skipped_reason),
        skips=tuple(f"{a.rung}:{a.skipped_reason}" for a in ctx.rungs if a.skipped_reason),
        final_status=final.status,
        final_route=final.route,
    )


async def replay_archive(archive_dir: Path, presets: dict[str, LadderPolicy], limit: int | None = None) -> Report:
    """Read every archived outcome and replay the replayable ones under each preset."""
    failures: list[str] = []
    records: list[ArchivedFetch] = []
    for stream in (fetcher_records(archive_dir, failures), loop_records(archive_dir, failures)):
        records.extend(islice(stream, limit))
    replays: list[Replay] = []
    with patch.object(ladder, "rungs", _DecliningRungs(rungs)):
        for preset, preset_policy in presets.items():
            for record in records:
                if record.replayable:
                    replays.append(await _replay_one(record, preset, preset_policy))
    return Report(tuple(presets), tuple(records), tuple(replays), tuple(failures))


def _cell_key(replay: Replay) -> tuple[str, str, str, str, str]:
    sequence = " -> ".join(replay.sequence) if replay.sequence else "(none)"
    return (
        replay.record.archived_status or "-",
        replay.record.archived_route,
        replay.final_status,
        replay.final_route,
        sequence,
    )


def _cell_rows(replays: Sequence[Replay]) -> list[tuple[tuple[str, str, str, str, str], int, str, bool, bool]]:
    """One row per before/after cell: the key, its count, an example host, lost, undecidable."""
    counts: Counter[tuple[str, str, str, str, str]] = Counter()
    examples: dict[tuple[str, str, str, str, str], str] = {}
    lost: dict[tuple[str, str, str, str, str], bool] = {}
    undecided: dict[tuple[str, str, str, str, str], bool] = {}
    for replay in replays:
        key = _cell_key(replay)
        counts[key] += 1
        examples.setdefault(key, replay.record.host)
        lost[key] = lost.get(key, False) or replay.rescue_lost
        undecided[key] = undecided.get(key, False) or replay.rescue_undecidable
    rows = [(key, count, examples[key], lost[key], undecided[key]) for key, count in counts.items()]
    rows.sort(key=lambda row: (not row[3], not row[4], -row[1]))
    return rows


def _source_summary(records: Sequence[ArchivedFetch]) -> list[str]:
    lines: list[str] = []
    for source in ("fetcher", "loop"):
        subset = [record for record in records if record.source == source]
        if not subset:
            continue
        routed = sum(1 for record in subset if record.route_recorded)
        lines.append(
            f"  {source}: {len(subset)} records, {routed} naming the rung that produced them, "
            f"{sum(1 for record in subset if record.replayable)} replayable, "
            f"{sum(1 for record in subset if record.notes)} carrying a fidelity note"
        )
    return lines


def _unreplayable_summary(records: Sequence[ArchivedFetch]) -> list[str]:
    counts = Counter(
        f"{record.source}/{record.archived_status or '-'}/{record.archived_route}"
        for record in records
        if not record.replayable
    )
    if not counts:
        return []
    lines = ["", "Recorded but NOT replayed (the archive does not carry the direct status behind them):"]
    lines.extend(f"  {label:<40}{count:>6}" for label, count in counts.most_common())
    return lines


def render_report(report: Report) -> str:
    """The whole replay as text. Pure: no clock read, no IO."""
    lines = [
        "Fetch-ladder archive replay. Every rung is stubbed to record its attempt and DECLINE, so the",
        "replayed status always equals the reconstructed direct status and the replayed route is the last",
        "rung consulted, not one that served bytes. What this measures is the rung SEQUENCE each preset",
        "would attempt per archived outcome, and the archived rescues its dispatcher no longer attempts.",
        "",
        f"Presets found on {ladder_policy.__name__}: {', '.join(report.presets)}",
        "",
        "Records read:",
        *_source_summary(report.records),
        *_unreplayable_summary(report.records),
    ]
    for preset in report.presets:
        replays = [replay for replay in report.replays if replay.preset == preset]
        rows = _cell_rows(replays)
        lost = sum(1 for replay in replays if replay.rescue_lost)
        undecidable = sum(1 for replay in replays if replay.rescue_undecidable)
        lines.extend(
            [
                "",
                f"=== {preset} === {len(replays)} replayed, {len(rows)} distinct cells, {lost} lost rescues, "
                f"{undecidable} undecidable (the rescue overwrote the direct http_status its trigger reads)",
                f"  {'archived':<22}{'route':<12}{'replayed':<22}{'route':<12}{'n':>6}  rung sequence / example host",
            ]
        )
        for key, count, host, is_lost, is_undecidable in rows[:CHANGED_CELL_LIMIT]:
            archived_status, archived_route, final_status, final_route, sequence = key
            flag = "  <- archived rescue never attempted" if is_lost else ""
            flag = flag or ("  <- rescue verdict undecidable: the direct http_status is lost" if is_undecidable else "")
            lines.append(
                f"  {archived_status:<22}{archived_route:<12}{final_status:<22}{final_route:<12}{count:>6}  "
                f"{sequence} [{host}]{flag}"
            )
        if len(rows) > CHANGED_CELL_LIMIT:
            lines.append(
                f"  ... {len(rows) - CHANGED_CELL_LIMIT} further cells, all unchanged-rank; --format json for the rest"
            )
    if report.parse_failures:
        lines.extend(["", f"Did not parse ({len(report.parse_failures)}):"])
        lines.extend(f"  {failure}" for failure in sorted(set(report.parse_failures))[:CHANGED_CELL_LIMIT])
    return "\n".join(lines)


def render_json(report: Report) -> str:
    """The whole replay as JSON, carrying full URLs where the table prints hostnames."""
    return json.dumps(
        {
            "presets": list(report.presets),
            "loop_status_inverse_mapping": {
                status: {"fetch_status": mapping.fetch_status, "ambiguous": mapping.ambiguous, "note": mapping.note}
                for status, mapping in LOOP_STATUS_TO_FETCH_STATUS.items()
            },
            "replays": [
                {
                    "preset": replay.preset,
                    "source": replay.record.source,
                    "qid": replay.record.qid,
                    "url": replay.record.url,
                    "archived_status": replay.record.archived_status,
                    "archived_route": replay.record.archived_route,
                    "direct_status": replay.record.direct_status,
                    "replayed_status": replay.final_status,
                    "replayed_route_last_consulted": replay.final_route,
                    "sequence": list(replay.sequence),
                    "skips": list(replay.skips),
                    "rescue_lost": replay.rescue_lost,
                    "rescue_undecidable": replay.rescue_undecidable,
                    "notes": list(replay.record.notes),
                }
                for replay in report.replays
            ],
            "not_replayed": [
                {
                    "source": record.source,
                    "qid": record.qid,
                    "url": record.url,
                    "archived_status": record.archived_status,
                    "archived_route": record.archived_route,
                    "notes": list(record.notes),
                }
                for record in report.records
                if not record.replayable
            ],
            "parse_failures": list(report.parse_failures),
        },
        indent=2,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Replay the archived fetch outcomes through the unified ladder's rung selection. Free: no network."
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help="research archive root holding raw/ and by_qid/ (default: %(default)s)",
    )
    parser.add_argument(
        "--policy",
        action="append",
        metavar="NAME",
        help="LadderPolicy preset to replay, repeatable (default: every preset on the policy module)",
    )
    parser.add_argument("--limit", type=int, help="cap each of the two record streams at this many records")
    parser.add_argument(
        "--format", choices=("table", "json"), default="table", help="output shape (default: %(default)s)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

    if not args.archive_dir.is_dir():
        sys.exit(f"no research archive at {args.archive_dir} — pass --archive-dir")
    absent = [name for name in (RAW_SUBDIR, BY_QID_SUBDIR) if not (args.archive_dir / name).is_dir()]
    if absent:
        logger.warning(f"the archive has no {'/'.join(absent)} subdirectory, so those records are absent")
    presets = discover_presets(args.policy)
    report = asyncio.run(replay_archive(args.archive_dir, presets, args.limit))
    print(render_json(report) if args.format == "json" else render_report(report))


if __name__ == "__main__":
    main()
