"""Marker-spec registry + pure parser for bot run-log telemetry.

Each :class:`MarkerSpec` pairs a marker name (which is also its archive-file stem)
with a regex whose named groups are the marker's fields. The regexes are written
against the ACTUAL emitted format strings (the source of truth):

* ``EXTRACTION_RUNG``   — ``metaculus_bot/value_extraction.py`` ``_log_extraction``
* ``GAP_FILL_V2``       — ``metaculus_bot/research/agentic/loop.py`` ``_log_completion``
* ``GHOST_FORECAST``    — ``metaculus_bot/research/agentic/loop.py`` ``_run_ghost_phase``
* ``OPEN_BOUND_PILING`` — ``metaculus_bot/numeric/diagnostics.py``
* ``CREDIT_BALANCE`` / ``CREDIT_SPEND`` / ``CREDIT_FLOOR_BREACH`` — ``metaculus_bot/credit_telemetry.py``
* ``STACKER_OUTCOME`` / ``TOOLS_USED`` / ``ANCHOR_OVERSHOOT_PP`` /
  ``CLAUSE_PRODUCT_DIVERGENCE_PP`` — ``metaculus_bot/comment/markers.py``

NOTE ON THE HTML-COMMENT MARKERS: the last four are ``<!-- ... -->`` markers
injected into the *published Metaculus comment*, not logged to stdout/stderr (the
framework logs only ``Posted comment on post N``, never the comment body). They
are therefore almost never present in run logs — their durable source is the
comment itself, which ``metaculus_bot.performance_analysis`` already parses. Their
specs live here so the parser stays complete if a run ever does log a comment
body, and because STACKER_OUTCOME/TOOLS_USED/ANCHOR/CLAUSE are all dormant in prod
anyway (stacking + probabilistic-tools disabled). Don't read their absence from
the telemetry archive as signal.

The parser matches on the marker TOKEN via ``re.search``, so it is agnostic to the
log-line prefix (the prod ``%(asctime)s - %(name)s - %(levelname)s - %(message)s``
format and the ablation ``%(asctime)s %(levelname)s %(name)s | %(message)s`` format
both work).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Fields captured as free text / references — never numerically coerced. ``question``
# is a raw ref (URL or bare id); ``summary`` is the ghost-forecast free-text summary.
_RAW_FIELDS: frozenset[str] = frozenset({"question", "summary"})

# Values that mean "no data" in the marker formats (``_fmt`` renders ``None`` as
# "n/a"; ``question_id`` renders as "None"; a stray "null" is defensive).
_NONE_SENTINELS: frozenset[str] = frozenset({"none", "n/a", "null"})

_INT_RE = re.compile(r"[+-]?\d+")
_LINE_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3})")
_QID_URL_RE = re.compile(r"/questions/(\d+)")
_BARE_INT_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class MarkerSpec:
    """One telemetry marker: its archive-file stem + the regex extracting its fields."""

    name: str
    regex: re.Pattern[str]


def coerce_value(raw: str | None) -> object:
    """Coerce a captured field string to bool / None / int / float, else keep the string.

    ``"True"``/``"true"`` -> ``True``; ``"n/a"``/``"None"``/``"null"`` -> ``None``;
    integer-looking -> ``int``; float-looking -> ``float``; everything else (model
    names, ``bound=upper``, ``qtype=binary``, ...) stays a ``str``.
    """
    if raw is None:
        return None
    text = raw.strip()
    low = text.lower()
    if low in _NONE_SENTINELS:
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if _INT_RE.fullmatch(text):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def qid_from_ref(ref: str | None) -> int | None:
    """Extract an integer question id from a Metaculus URL or a bare id string."""
    if ref is None:
        return None
    text = str(ref).strip()
    if text.lower() in _NONE_SENTINELS:
        return None
    url_match = _QID_URL_RE.search(text)
    if url_match:
        return int(url_match.group(1))
    if _BARE_INT_RE.fullmatch(text):
        return int(text)
    return None


def _parse_line_ts(line: str) -> str | None:
    """Extract the ``%(asctime)s`` prefix as an ISO-8601 string, or None if absent."""
    match = _LINE_TS_RE.match(line.lstrip())
    if not match:
        return None
    date, clock, millis = match.groups()
    return f"{date}T{clock}.{millis}000"


# --- Marker registry ---------------------------------------------------------
# ``question=`` on GAP_FILL_V2 / GHOST_FORECAST comes from ``log_prefix`` (see
# ``agentic_gap_fill.py``: ``f"question={ref} "``) and is prepended BEFORE the
# marker token, so it's an optional leading group there. On EXTRACTION_RUNG /
# OPEN_BOUND_PILING the ``question=`` is a normal field AFTER the token.
MARKER_SPECS: list[MarkerSpec] = [
    MarkerSpec(
        "extraction_rung",
        re.compile(
            r"EXTRACTION_RUNG:\s*question=(?P<question>\S+)\s+model=(?P<model>.+?)"
            r"\s+qtype=(?P<qtype>\S+)\s+rung=(?P<rung>\S+)\s+block_present=(?P<block_present>\S+)"
        ),
    ),
    MarkerSpec(
        "gap_fill_v2",
        re.compile(
            r"(?:question=(?P<question>\S+)\s+)?GAP_FILL_V2:\s*model=(?P<model>.+?)"
            r"\s+steps=(?P<steps>\S+)\s+tool_calls=(?P<tool_calls>\S+)\s+searches=(?P<searches>\S+)"
            r"\s+fetches=(?P<fetches>\S+)\s+rendered=(?P<rendered>\S+)\s+reads=(?P<reads>\S+)"
            r"\s+dup_tool_calls=(?P<dup_tool_calls>\S+)\s+deadline_hit=(?P<deadline_hit>\S+)"
            r"\s+concluded_early=(?P<concluded_early>\S+)\s+wall_s=(?P<wall_s>\S+)"
            r"\s+findings=(?P<findings>\S+)\s+pending_leads=(?P<pending_leads>\S+)"
            r"\s+lint_rejections=(?P<lint_rejections>\S+)"
        ),
    ),
    MarkerSpec(
        "ghost_forecast",
        re.compile(
            r"(?:question=(?P<question>\S+)\s+)?GHOST_FORECAST:\s*qtype=(?P<qtype>\S+)\s+summary=(?P<summary>.*)$"
        ),
    ),
    MarkerSpec(
        "open_bound_piling",
        re.compile(
            r"OPEN_BOUND_PILING:\s*question=(?P<question>\S+)\s+model=(?P<model>.+?)"
            r"\s+bound=(?P<bound>\S+)\s+bin_mass=(?P<bin_mass>\S+)"
            r"\s+declared_edge=(?P<declared_edge>\S+)\s+bound_value=(?P<bound_value>\S+)"
        ),
    ),
    MarkerSpec(
        "credit_balance",
        re.compile(
            r"CREDIT_BALANCE:\s*key=(?P<key>\S+)\s+phase=(?P<phase>\S+)"
            r"(?:\s+remaining=(?P<remaining>\S+)\s+usage=(?P<usage>\S+))?"
        ),
    ),
    MarkerSpec(
        "credit_spend",
        re.compile(
            r"CREDIT_SPEND:\s*key=(?P<key>\S+)\s+run_delta_usd=(?P<run_delta_usd>\S+)\s+remaining=(?P<remaining>\S+)"
        ),
    ),
    MarkerSpec(
        "credit_floor_breach",
        re.compile(r"CREDIT_FLOOR_BREACH:\s*key=(?P<key>\S+)\s+remaining=(?P<remaining>\S+)\s+floor=(?P<floor>\S+)"),
    ),
    MarkerSpec(
        "stacker_outcome",
        re.compile(
            r"<!--\s*STACKER_OUTCOME=(?P<outcome>primary|fallback_llm|fallback_median|fallback_mean|skipped)\s*-->",
            re.IGNORECASE,
        ),
    ),
    MarkerSpec("tools_used", re.compile(r"<!--\s*TOOLS_USED=(?P<value>true|false)\s*-->", re.IGNORECASE)),
    MarkerSpec(
        "anchor_overshoot_pp",
        re.compile(r"<!--\s*ANCHOR_OVERSHOOT_PP=(?P<pp>[+-]?\d+(?:\.\d+)?)\s*-->", re.IGNORECASE),
    ),
    MarkerSpec(
        "clause_product_divergence_pp",
        re.compile(r"<!--\s*CLAUSE_PRODUCT_DIVERGENCE_PP=(?P<pp>[+-]?\d+(?:\.\d+)?)\s*-->", re.IGNORECASE),
    ),
]


def _build_record(
    spec: MarkerSpec,
    match: re.Match[str],
    line: str,
    seq: int,
    meta: dict[str, str],
) -> dict:
    """Assemble one archive record from a regex match + run metadata."""
    record: dict = {
        "marker": spec.name,
        "run_id": meta["run_id"],
        "workflow": meta["workflow"],
        "artifact": meta["artifact"],
        "run_date": meta["run_date"],
        "log_file": meta["log_file"],
        "seq": seq,
        "line_ts": _parse_line_ts(line),
    }
    for field, raw in match.groupdict().items():
        record[field] = raw if field in _RAW_FIELDS else coerce_value(raw)
    if "question" in record:
        record["qid"] = qid_from_ref(record["question"])
    return record


def parse_log_text(
    text: str,
    *,
    run_id: str,
    workflow: str,
    artifact: str,
    run_date: str,
    log_file: str,
) -> dict[str, list[dict]]:
    """Parse all telemetry markers from one log-text blob into per-marker record lists.

    ``seq`` is a per-marker ordinal within this blob; because a run's logs are parsed
    in stable order, re-harvesting produces byte-identical records — which the archive
    merge relies on for idempotent replace-by-run (see :mod:`scripts.telemetry.archive`).
    """
    meta = {
        "run_id": run_id,
        "workflow": workflow,
        "artifact": artifact,
        "run_date": run_date,
        "log_file": log_file,
    }
    harvested: dict[str, list[dict]] = {spec.name: [] for spec in MARKER_SPECS}
    counters: dict[str, int] = {spec.name: 0 for spec in MARKER_SPECS}

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        for spec in MARKER_SPECS:
            match = spec.regex.search(line)
            if match:
                harvested[spec.name].append(_build_record(spec, match, line, counters[spec.name], meta))
                counters[spec.name] += 1
                break  # marker tokens are mutually exclusive — one marker per line
    return harvested
