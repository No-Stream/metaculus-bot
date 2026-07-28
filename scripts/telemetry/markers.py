"""Marker-spec registry + pure parser for bot run-log telemetry.

Each :class:`MarkerSpec` pairs a marker name (which is also its archive-file stem)
with a regex whose named groups are the marker's fields. The regexes are written
against the ACTUAL emitted format strings (the source of truth):

* ``EXTRACTION_RUNG``   — ``metaculus_bot/value_extraction.py`` ``_log_extraction``
* ``GAP_FILL_V2``       — ``metaculus_bot/research/agentic/loop.py`` ``_log_completion``
* ``GHOST_PRE`` / ``GHOST_PRE_JSON`` — ``metaculus_bot/research/agentic/loop.py``
  ``_set_research_plan_tool`` (the pre-research counterpart to the concluding
  ghost pair below; the GHOST_PRE↔GHOST_FORECAST delta measures whether v2's
  research moved its own view)
* ``GHOST_FORECAST``    — ``metaculus_bot/research/agentic/loop.py`` ``_run_ghost_phase``
* ``GHOST_FORECAST_JSON`` — ``metaculus_bot/research/agentic/loop.py`` ``_run_ghost_phase``
  (additive full-fidelity companion to ``GHOST_FORECAST``; the ``forecast_json``
  field is a compact single-line JSON blob the ghost scorer ``json.loads``)
* ``OPEN_BOUND_PILING`` — ``metaculus_bot/numeric/diagnostics.py``
* ``FORECASTER_DROPS`` / ``Degradation counters`` — ``metaculus_bot/forecaster.py``
  (per-RUN summaries: which models dropped and why, and the counter set that
  decides CI color)
* ``FORECASTERS_SURVIVED`` — ``metaculus_bot/forecaster.py``
  ``_research_and_make_predictions`` (per-QUESTION positive survivor count; the
  drop marker above is silent on a healthy question, and its comment-side twin
  ``FORECASTERS_USED`` never reaches stdout)
* ``CLOSE_MARGIN``      — ``metaculus_bot/close_margin.py`` (emitted at submit time in ``forecaster.py``)
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

POST-ID vs QUESTION-ID (the ``qid_kind`` field): Metaculus posts contain questions,
and the two ids DIVERGE on newer posts (post 38880 wraps question 38195). Marker
types are keyed in DIFFERENT spaces — ``EXTRACTION_RUNG`` / ``OPEN_BOUND_PILING`` /
``CLOSE_MARGIN`` emit ``question.id_of_question`` (the QUESTION id) while
``GAP_FILL_V2`` / ``GHOST_PRE`` / ``GHOST_PRE_JSON`` / ``GHOST_FORECAST`` /
``GHOST_FORECAST_JSON`` emit ``question.page_url`` (a POST id). Each :class:`MarkerSpec` therefore declares
``qid_kind`` and every harvested record carries it, so a residual join keyed on one
id can TRANSLATE into the record's own space rather than silently dropping the
records keyed on the other (see :mod:`metaculus_bot.performance_analysis.id_mapping`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The two Metaculus id spaces a marker's ``question=`` ref can live in. Kept as
# LOCAL literals (not imported from ``metaculus_bot``) so this parser and the whole
# ``scripts.telemetry`` archive stack stay pure-stdlib — ``make sync_telemetry`` must
# not drag in forecasting_tools/numpy/streamlit just to grep logs. The canonical
# definitions live in ``metaculus_bot.performance_analysis.id_mapping``; the two are
# pinned equal by ``tests/test_id_mapping.py::test_qid_kind_constants_match_markers``.
QID_KIND_POST_ID = "post_id"
QID_KIND_QUESTION_ID = "question_id"

# Fields captured as free text / references — never numerically coerced. ``question``
# is a raw ref (URL or bare id); ``summary`` is the ghost-forecast free-text summary;
# ``forecast_json`` is the compact ghost-forecast JSON blob (kept verbatim for the
# scorer to ``json.loads`` — coercion would mangle it).
_RAW_FIELDS: frozenset[str] = frozenset({"question", "summary", "forecast_json", "detail"})

# Values that mean "no data" in the marker formats (``_fmt`` renders ``None`` as
# "n/a"; ``question_id`` renders as "None"; a stray "null" is defensive).
_NONE_SENTINELS: frozenset[str] = frozenset({"none", "n/a", "null"})

_INT_RE = re.compile(r"[+-]?\d+")
_LINE_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3})")
_QID_URL_RE = re.compile(r"/questions/(\d+)")
_BARE_INT_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class MarkerSpec:
    """One telemetry marker: its archive-file stem, the field regex, and its id space.

    ``qid_kind`` names which Metaculus id space the marker's ``question=`` ref lives
    in (``"post_id"`` or ``"question_id"``); ``None`` for markers with no question
    ref (the credit markers). It is stamped onto every harvested record so a
    residual join knows how to translate a query into the record's id space instead
    of guessing (see the module docstring + ``performance_analysis.id_mapping``).
    """

    name: str
    regex: re.Pattern[str]
    qid_kind: str | None = None


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
# ``question=`` on GAP_FILL_V2 / GHOST_PRE / GHOST_PRE_JSON / GHOST_FORECAST /
# GHOST_FORECAST_JSON comes from ``log_prefix`` (see ``agentic_gap_fill.py``:
# ``f"question={ref} "``) and is prepended BEFORE the marker token, so it's an
# optional leading group there. On EXTRACTION_RUNG / OPEN_BOUND_PILING /
# CLOSE_MARGIN the ``question=`` is a normal field AFTER the token.
MARKER_SPECS: list[MarkerSpec] = [
    MarkerSpec(
        "extraction_rung",
        re.compile(
            r"EXTRACTION_RUNG:\s*question=(?P<question>\S+)\s+model=(?P<model>.+?)"
            r"\s+qtype=(?P<qtype>\S+)\s+rung=(?P<rung>\S+)\s+block_present=(?P<block_present>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # value_extraction.py emits question.id_of_question
    ),
    MarkerSpec(
        "gap_fill_v2",
        # The trailing counter group (provenance_rejections .. conclude_gate_rejections,
        # added 2026-07-21) is OPTIONAL: re-harvesting replays pre-branch logs whose
        # lines end at lint_rejections, and a mandatory tail would drop every one of
        # those records on the next replace-by-run sync. Missing groups coerce to None.
        # ``error`` (added 2026-07-23) is nested one level deeper — it is always
        # emitted alongside the 2026-07-21 counters, so it can only appear after
        # conclude_gate_rejections; it captures greedily to end-of-line because it
        # holds ``repr(exc)`` which contains spaces (``error=None`` on healthy runs).
        re.compile(
            r"(?:question=(?P<question>\S+)\s+)?GAP_FILL_V2:\s*model=(?P<model>.+?)"
            r"\s+steps=(?P<steps>\S+)\s+tool_calls=(?P<tool_calls>\S+)\s+searches=(?P<searches>\S+)"
            r"\s+fetches=(?P<fetches>\S+)\s+rendered=(?P<rendered>\S+)\s+reads=(?P<reads>\S+)"
            r"\s+dup_tool_calls=(?P<dup_tool_calls>\S+)\s+deadline_hit=(?P<deadline_hit>\S+)"
            r"\s+concluded_early=(?P<concluded_early>\S+)\s+wall_s=(?P<wall_s>\S+)"
            r"\s+findings=(?P<findings>\S+)\s+pending_leads=(?P<pending_leads>\S+)"
            r"\s+lint_rejections=(?P<lint_rejections>\S+)"
            r"(?:\s+provenance_rejections=(?P<provenance_rejections>\S+)"
            r"\s+quote_mismatch_warnings=(?P<quote_mismatch_warnings>\S+)"
            r"\s+plan_gaps=(?P<plan_gaps>\S+)\s+plan_skipped=(?P<plan_skipped>\S+)"
            r"\s+conclude_gate_rejections=(?P<conclude_gate_rejections>\S+)"
            r"(?:\s+error=(?P<error>.*))?)?"
        ),
        qid_kind=QID_KIND_POST_ID,  # agentic_gap_fill.py emits question.page_url (post id)
    ),
    # Pre-research counterparts to the GHOST_FORECAST pair below, emitted by
    # ``_set_research_plan_tool`` at plan-set time. ``question=`` comes from the
    # same ``log_prefix`` leading group, and the ``GHOST_PRE:`` token requires the
    # colon so it can't collide with ``GHOST_PRE_JSON`` under the one-marker-per-line
    # ``break`` (same mechanism as the GHOST_FORECAST / GHOST_FORECAST_JSON pair).
    MarkerSpec(
        "ghost_pre",
        re.compile(
            r"(?:question=(?P<question>\S+)\s+)?GHOST_PRE:\s*gaps=(?P<gaps>\S+)"
            r"\s+sensitive_assumptions=(?P<sensitive_assumptions>\S+)"
        ),
        qid_kind=QID_KIND_POST_ID,  # agentic_gap_fill.py log_prefix = question.page_url (post id)
    ),
    MarkerSpec(
        "ghost_pre_json",
        re.compile(r"(?:question=(?P<question>\S+)\s+)?GHOST_PRE_JSON:\s*(?P<forecast_json>\{.*\})\s*$"),
        qid_kind=QID_KIND_POST_ID,  # agentic_gap_fill.py log_prefix = question.page_url (post id)
    ),
    MarkerSpec(
        "ghost_forecast",
        re.compile(
            r"(?:question=(?P<question>\S+)\s+)?GHOST_FORECAST:\s*qtype=(?P<qtype>\S+)\s+summary=(?P<summary>.*)$"
        ),
        qid_kind=QID_KIND_POST_ID,  # agentic_gap_fill.py log_prefix = question.page_url (post id)
    ),
    # Additive full-fidelity companion to ``ghost_forecast``. ``question=`` comes
    # from ``log_prefix`` (same leading-group mechanism as GAP_FILL_V2 /
    # GHOST_FORECAST), and ``forecast_json`` greedily captures the compact
    # single-line JSON payload to the final ``}``. The ``GHOST_FORECAST_JSON``
    # token can't collide with ``GHOST_FORECAST:`` (the latter requires ``:``
    # immediately after ``GHOST_FORECAST``), so the two specs stay mutually
    # exclusive under the one-marker-per-line ``break``.
    MarkerSpec(
        "ghost_forecast_json",
        re.compile(r"(?:question=(?P<question>\S+)\s+)?GHOST_FORECAST_JSON:\s*(?P<forecast_json>\{.*\})\s*$"),
        qid_kind=QID_KIND_POST_ID,  # agentic_gap_fill.py log_prefix = question.page_url (post id)
    ),
    MarkerSpec(
        "open_bound_piling",
        re.compile(
            r"OPEN_BOUND_PILING:\s*question=(?P<question>\S+)\s+model=(?P<model>.+?)"
            r"\s+bound=(?P<bound>\S+)\s+bin_mass=(?P<bin_mass>\S+)"
            r"\s+declared_edge=(?P<declared_edge>\S+)\s+bound_value=(?P<bound_value>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # numeric/diagnostics.py emits question.id_of_question
    ),
    MarkerSpec(
        "close_margin",
        re.compile(
            r"CLOSE_MARGIN:\s*question=(?P<question>\S+)\s+close_time=(?P<close_time>\S+)"
            r"\s+submitted_at=(?P<submitted_at>\S+)\s+window_s=(?P<window_s>\S+)"
            r"\s+margin_s=(?P<margin_s>\S+)\s+margin_frac=(?P<margin_frac>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # close_margin.py emits question.id_of_question
    ),
    MarkerSpec(
        "forecaster_drops",
        # Per-run ensemble-drop summary emitted by
        # forecaster.py:_emit_forecaster_drop_telemetry. No per-question ref (it
        # aggregates a whole run), so qid_kind stays None. ``detail`` is a compact
        # model->cause->count JSON blob captured verbatim (it is in _RAW_FIELDS) so
        # the '/'-laden OpenRouter slugs and nested counts survive; ``systematic`` is
        # a comma-joined model list (or the "none" sentinel -> None).
        re.compile(
            r"FORECASTER_DROPS:\s*total=(?P<total>\S+)\s+systematic=(?P<systematic>\S+)\s+detail=(?P<detail>\{.*\})\s*$"
        ),
    ),
    MarkerSpec(
        "forecasters_survived",
        # The POSITIVE per-question counterpart to forecaster_drops above, emitted by
        # forecaster.py's _research_and_make_predictions once the survivor set is
        # known. Unlike the ``forecasters_used`` HTML marker further down — which
        # carries the same count but lives in the published COMMENT and so is
        # effectively never in a run log — this one is on stdout, which makes
        # historical survivor counts queryable from the telemetry archive alone.
        #
        # ``models`` is a comma-joined slug list; OpenRouter slugs contain no spaces,
        # so ``\S+`` takes the whole field. It is deliberately NOT in _RAW_FIELDS: a
        # comma-joined string coerces to itself (``coerce_value`` only converts
        # numeric-looking text), and the "unknown" sentinel is not in
        # _NONE_SENTINELS, so it survives verbatim either way.
        re.compile(
            r"FORECASTERS_SURVIVED:\s*question=(?P<question>\S+)\s+survived=(?P<survived>\d+)/(?P<configured>\d+)"
            r"\s+models=(?P<models>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # forecaster.py emits question.id_of_question
    ),
    MarkerSpec(
        "degradation_counters",
        # The per-run summary that DECIDES CI COLOR (cli.py exits non-zero on a
        # positive alertable_count), emitted by forecaster.py's forecast_questions.
        # No per-question ref — it aggregates a whole run — so qid_kind stays None.
        #
        # The trailing keys are OPTIONAL-group wrapped for the same reason as
        # gap_fill_v2 above: replace-by-run re-harvesting replays pre-rename logs
        # (``research_provider_timeouts``, no ``summarizer_failures``,
        # ``prediction_market_platform_failures`` as the tail), and a mandatory tail
        # would drop each of those records wholesale instead of harvesting the
        # counters it does carry. The rename pairs are alternations so one group name
        # can't cover both spellings; missing groups coerce to None, which reads as
        # "this era didn't emit it" rather than a measured zero.
        re.compile(
            r"Degradation counters:\s*forecasters_dropped=(?P<forecasters_dropped>\S+?),"
            r"\s*questions_failed_to_publish=(?P<questions_failed_to_publish>\S+?),"
            r"\s*stacker_primary_failed=(?P<stacker_primary_failed>\S+?),"
            r"\s*stacker_fallback_used=(?P<stacker_fallback_used>\S+?),"
            r"\s*stacker_fallback_failed=(?P<stacker_fallback_failed>\S+?),"
            r"\s*(?:research_provider_failures=(?P<research_provider_failures>\S+?)"
            r"|research_provider_timeouts=(?P<research_provider_timeouts>\S+?)),"
            r"(?:\s*summarizer_failures=(?P<summarizer_failures>\S+?),)?"
            r"\s*gap_fill_v2_errors=(?P<gap_fill_v2_errors>\S+?)"
            r"(?:,\s*prediction_market_degraded=(?P<prediction_market_degraded>\S+?))?"
            r"(?:,\s*(?:prediction_market_source_losses=(?P<prediction_market_source_losses>\S+)"
            r"|prediction_market_platform_failures=(?P<prediction_market_platform_failures>\S+)))?"
            r"\s*$"
        ),
    ),
    MarkerSpec(
        "gemini_ungrounded_suppressed",
        # Gemini grounded-search suppression (research/gemini_search.py
        # _format_grounded_response): google_search returned no grounding chunks and
        # no url_context read succeeded, so the section is dropped as ungrounded
        # parametric output. The orchestrator then records status="empty", which is
        # NOT alertable and bumps no counter — so this WARN is the only signal, and
        # without a spec the suppression rate was unmeasurable from the archive.
        re.compile(
            r"GEMINI_UNGROUNDED_SUPPRESSED:\s*question=(?P<question>\S+)\s+model=(?P<model>.+?)"
            r"\s+queries=(?P<queries>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # gemini_search.py passes question.id_of_question
    ),
    MarkerSpec(
        "agentic_document_ungrounded_suppressed",
        # The read_document twin of the marker above (research/agentic/tools.py
        # read_document): Gemini's url_context tool retrieved nothing, so the answer would
        # be unsourced recall and the "fetched" verification tier is withheld. Worth
        # measuring separately because a "fetched" document discrepancy is the only kind
        # that enters the artifact's SUPERSEDE block, i.e. the one that tells every
        # forecaster to override the briefing. Carries no question id — read_document is a
        # per-URL tool with no question in scope — so the URL is the only field.
        re.compile(r"AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED:\s*url=(?P<url>\S+)"),
    ),
    MarkerSpec(
        "gap_fill_analyzer_failed",
        # Gap-fill v1's analyzer (research/targeted.py run_gap_fill_pass) died, which
        # GATES the whole pass — the addendum is silently "" and the run looks identical
        # to a question that legitimately had no gaps. Gap-fill isn't one of the
        # orchestrator's _run_one providers, so it has no ProviderResult and no `lost=`
        # token; this marker is the only durable signal, and v1's searches are one of the
        # largest research spend lines (~44%). ``detail`` captures greedily to end-of-line
        # because it holds the exception's str.
        re.compile(
            r"GAP_FILL_ANALYZER_FAILED:\s*question=(?P<question>\S+)\s+error=(?P<error>\S+)"
            r"(?:\s+detail=(?P<detail>.*))?$"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # targeted.py passes question.id_of_question
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
            r"<!--\s*STACKER_OUTCOME="
            r"(?P<outcome>primary|fallback_llm|fallback_median|fallback_mean|skipped_config_off|skipped)"
            r"\s*-->",
            re.IGNORECASE,
        ),
    ),
    MarkerSpec("tools_used", re.compile(r"<!--\s*TOOLS_USED=(?P<value>true|false)\s*-->", re.IGNORECASE)),
    # Ensemble-size disclosure (metaculus_bot/comment/markers.py). Like the other
    # HTML-comment markers this lives in the published comment, not stdout — its
    # durable consumer is performance_analysis.parsing; the spec is here so the
    # run-log parser stays complete if a comment body is ever logged. ``used`` /
    # ``configured`` are the contributed / configured forecaster counts.
    MarkerSpec(
        "forecasters_used",
        re.compile(r"<!--\s*FORECASTERS_USED=(?P<used>\d+)/(?P<configured>\d+)\s*-->", re.IGNORECASE),
    ),
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
        # ``qid_kind`` names which Metaculus id space ``qid`` lives in, so a residual
        # join can translate a query into this record's space instead of guessing
        # (see the module docstring). Only meaningful for question-bearing markers.
        record["qid_kind"] = spec.qid_kind
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
