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
* ``AGENTIC_FETCH_THROTTLED`` — ``metaculus_bot/research/agentic/tools.py``
  ``_throttled_fetch_outcome`` (per-FETCH: a host answered the gap-fill v2 ladder with a
  rate-limit interstitial under HTTP 200. Silent before this spec, and silent in the worst
  way — the interstitial reached the driver as a successful fetch and was cached)
* ``AGENTIC_FETCH_LOCAL_DOC`` — ``metaculus_bot/research/agentic/local_document.py``
  ``log_local_document_read`` (per-DOCUMENT: the gap-fill v2 ladder read a document
  locally instead of paying a Gemini ``url_context`` call for it — a PDF's extracted
  text on a ``fetch``, or a BM25 passage digest on a ``read_document``)
* ``OPEN_BOUND_PILING`` — ``metaculus_bot/numeric/diagnostics.py``
* ``FORECASTER_DROPS`` — ``metaculus_bot/drop_telemetry.py`` ``emit_drop_telemetry``
  (per-RUN summary: which models dropped and why)
* ``Degradation counters`` — ``metaculus_bot/degradation_counters.py``
  ``format_degradation_summary`` (per-RUN counter set that decides CI color)
* ``FORECASTERS_SURVIVED`` — ``metaculus_bot/forecaster.py``
  ``_research_and_make_predictions`` (per-QUESTION positive survivor count; the
  drop marker above is silent on a healthy question, and its comment-side twin
  ``FORECASTERS_USED`` never reaches stdout)
* ``MEMBER_FORECAST``   — ``metaculus_bot/member_forecast.py`` ``format_member_forecast_marker``,
  emitted from ``forecaster_runners.py`` (each member, all three types), ``stacking.py``
  (stacker binary / MC) and ``aggregation_pipeline.py`` (stacker numeric): per-VALUE
  record of what the ladder extracted and what the runner handed on, both as compact
  JSON. The one marker that carries a member's forecast value on every question; before
  it the raw value lived only in the trim-lossy published comment
* ``CLOSE_MARGIN``      — ``metaculus_bot/close_margin.py`` (emitted at submit time in ``forecaster.py``)
* ``MARKET_RANKING``    — ``metaculus_bot/research/prediction_market.py``
  ``_log_ranking_telemetry`` (per-QUESTION ranked-retrieval outcome: pool size,
  ranker outcome, and every rendered row's ``venue:pool_index@rank``)
* ``MARKET_CHILD_RENDER`` — ``metaculus_bot/research/prediction_market.py``
  ``_log_child_render_telemetry`` (per-QUESTION multi-outcome child accounting:
  how many outcomes were individually named versus collapsed into a counted
  group, and ``withheld``, the count of venue-manufactured prices the parsers
  refused — the field the Kalshi no-price spread threshold gets retuned on)
* ``MARKET_RANKING_DEGRADED`` — ``metaculus_bot/research/prediction_market.py``
  ``_rank_pool`` (per-QUESTION ranker fail-open, and WHY: ``shape_regression``
  means our own prompt/parser contract broke, which used to pass silently as
  ``ok(0)``, i.e. as a deliberate "we reviewed the markets and none bore on it")
* ``MARKET_TIER_CAPPED`` — ``metaculus_bot/research/prediction_market.py``
  ``_log_tier_caps`` (per-QUESTION staleness tier cap: the ranker graded a market
  that stopped trading long before the question opened as ``same_quantity_same_date``
  and the deterministic pass refused it that top tier; silent otherwise, and it
  fires on nothing in the archive, so a first record is itself the finding)
* ``NUMERIC_DEGENERATE_DECLARATION`` — ``metaculus_bot/numeric/pipeline.py``
  ``_apply_jitter_and_clamp`` (per-FORECASTER point-mass numeric declaration that
  is no longer cluster-spread into a width nobody stated — a fabrication-attempt
  rate, since the unit-mismatch guard then withholds that forecaster)
* ``NUMERIC_AGGREGATE_GRID_MISMATCH`` — ``metaculus_bot/numeric/utils.py``
  ``aggregate_numeric`` (per-MODEL CDF whose grid length disagreed with the
  question's; expect zero in prod, so any record means a length drifted)
* ``CDF_MAXSTEP_CLIP`` — ``metaculus_bot/numeric/pchip_cdf.py`` ``safe_cdf_bounds``
  (per-CDF-BUILD max-step clip: a declared single-bin mass the platform's per-bin
  cap cannot hold, and where the displaced mass went — the repair that reshaped
  47% of q45065's published forecast while logging at DEBUG)
* ``PCHIP CDF construction failed`` (spec ``numeric_pchip_fallback``) —
  ``metaculus_bot/numeric/diagnostics.py`` ``log_pchip_fallback`` (per-QUESTION
  PCHIP build failure that fell back to forecasting-tools' own CDF builder; the
  one numeric repair surface with a confirmed prod fire)

NUMERIC REPAIR TIERS, DELIBERATELY UNHARVESTED (sentinel_value_audit M16 asked for
one ``numeric_repair`` marker over five repair surfaces): the repair-tier WARNs in
``numeric/bounds_clamping.py`` — ``Corrected numeric distribution``, ``Heavy bound
clamping``, ``Cluster spread applied`` — never fire on real model output, because
``generate_pchip_cdf``'s uniform-mixture construction pre-enforces the min-step
before any repair tier is reached (0 of 1182 archived numeric forecasts; see
AGENTS.md "Repair-tier WARN signals are effectively dead code"). Registering specs
for them would archive permanently-empty files that read as signal, so the omission
is a decision, not a miss. M16's incidence question is answered by the narrower
markers above — ``NUMERIC_DEGENERATE_DECLARATION``, ``NUMERIC_AGGREGATE_GRID_MISMATCH``,
``SPREAD_UNDEFINED`` — plus ``numeric_pchip_fallback`` for the surface that does fire;
the unit-mismatch withhold rides ``FORECASTER_DROPS`` rather than its own marker.
* ``SPREAD_UNDEFINED`` — ``metaculus_bot/spread_metrics.py``
  ``numeric_percentile_spread`` (per-QUESTION unmeasurable spread: the routing
  decision was made on no measurement at all)
* ``TS_ANCHOR_ROUTE``   — ``metaculus_bot/research/ts_routing.py`` ``route_question``
  (per-QUESTION timeseries-anchor routing decision: routed/skipped, the series
  involved, and the branch/reject step — the marker that made anchor coverage
  queryable; before it, 27 of the triple era's 30 route-level misses were the
  silent ``kw_no_keyword_hit`` return and left no log line at all)
* ``FINANCIAL_STALE_LATEST`` — ``metaculus_bot/research/financial_data.py``
  ``_fetch_yfinance_data`` and ``metaculus_bot/research/ts_render.py``
  ``_render_single`` (per-IDENTIFIER stale "latest" disclosure: the newest
  observation is older than its own cadence explains, so the rendered latest
  value — and anything anchored on it — was flagged stale to the forecaster;
  informational data-quality signal, NOT alertable)
* ``FINANCIAL_NOISE_FLAG`` — ``metaculus_bot/research/financial_data.py``
  ``_volatility_lines`` and ``metaculus_bot/research/ts_render.py``
  ``_realized_vol_lines`` (per-IDENTIFIER vendor-noise disclosure: the series'
  variance ratio says most of each day's move is reversed the next, so every
  volatility computed from one-day returns is inflated and the block leads with the
  noise-robust multi-period figure instead; informational, NOT alertable)
* ``RESOLUTION_SOURCE_FETCH`` — ``metaculus_bot/research/resolution_source.py``
  ``_log_fetch_outcome_markers`` (per-URL Tier-1 page fetch AND Tier-2 Datawrapper
  dataset hop: the outcome, the HTTP code, and the routeless data-embed providers
  found in the page's raw HTML. Before this the per-URL outcomes lived only in
  free-text log lines and the comment's provider-diagnostics block, so a cut like
  "cdc.gov is 0 successes in 1,069 fetch records" meant re-scraping GHA logs that
  expire at 90 days)
* ``RESOLUTION_SOURCE_ESCALATION`` — ``metaculus_bot/research/resolution_source.py``
  (per-ESCALATED URL: a Tier-1 fetch the direct route could not read, which rung of
  the escalation ladder was tried, what came back, and how long the rung cost. Its
  sibling ``RESOLUTION_SOURCE_FETCH`` records the FINAL outcome per URL and so is
  silent on the path taken to it, which is what decides whether a rung earns its
  latency)
* ``RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP`` / ``RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED``
  / ``RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED`` — ``metaculus_bot/research/resolution_source.py``
  ``_url_context_admission`` and ``_url_context_rung`` (per-URL accounting for the ladder's
  one PAID rung: a read the free robots pre-check refused to make, a paid read discarded
  for retrieving nothing, and a paid read that retrieved the page and found nothing on
  the ask. Registered when ``RESOLUTION_SOURCE_URL_CONTEXT_ENABLED`` went on in every bot
  workflow; before that none of the three could fire in production. Parallel to the
  gap-fill v2 reader's ``AGENTIC_URLCONTEXT_ROBOTS_SKIP`` /
  ``AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED``)
* ``RENDERED_FETCH_OFF_HOST`` — ``metaculus_bot/research/rendered_fetch.py``
  ``render_page`` (per-REFUSED RENDER: the headless-Chromium main frame landed on a host
  the DNS pin does not cover, so its DOM was refused unread. The transport is shared, so
  the row covers both callers and ``scope`` says which one asked for the render)
* ``GEMINI_USAGE`` — ``metaculus_bot/research/gemini_search.py``,
  ``metaculus_bot/research/agentic/tool_backends.py`` and
  ``metaculus_bot/research/resolution_source.py`` (per-CALL google-genai token
  and grounded-query accounting for all three Gemini surfaces: grounded search,
  gap-fill v2's ``read_document``, and the resolution-source ladder's paid
  ``url_context`` rung. None bills through OpenRouter, so none
  appears in ``CREDIT_ROLE_SPEND`` — before this marker the Google AI Studio side of
  a run's spend was unmeasurable from the archive, and the monthly grounded-prompt
  allowance is what a feature multiplying grounded calls eats. ``role`` is the
  surface; the ``question=`` ref is optional because ``read_document`` and the
  resolution-source rung both run with no question in scope)
* ``PROVIDER_DEGRADATION`` — ``metaculus_bot/research/provider_health.py``
  ``log_provider_degradation_summary`` (per-RUN: which venue/signal degraded, and
  whether it counted toward the exit code)
* ``PAID PERSONAL-KEY FALLBACK`` — ``metaculus_bot/fallback_openrouter.py``
  ``_log_fallback`` (per-CALL: which model fell back off the donated key, and why)
* ``PUBLISH_HARDENING`` — ``metaculus_bot/publish_hardening.py``
  ``_wrap_with_timeout_retry`` (per-ATTEMPT publish failure: which POST method,
  which attempt of how many, and the timeout or exception that killed it — the
  q45085 405-closed shape left no harvestable trace before this spec)
* ``PUBLISH_SKIPPED_CLOSED`` — ``metaculus_bot/publish_gate.py``
  ``skip_publish_if_closed`` (per-QUESTION pre-publish skip: the question whose
  window had already closed, why, and by how many seconds it missed)
* ``Run completed [clean] with N alertable...`` — ``metaculus_bot/cli.py`` (the
  end-of-run breakdown, emitted on EVERY path — degraded, fully-suppressed green,
  crashed, and fully clean — so the archive holds one record per run; the ``clean``
  variant is the 2026-08-25 addition that keeps a healthy run in the census)
* ``CREDIT_BALANCE`` / ``CREDIT_SPEND`` / ``CREDIT_ROLE_SPEND`` / ``CREDIT_FLOOR_BREACH`` — ``metaculus_bot/credit_telemetry.py``
  (``CREDIT_ROLE_SPEND`` is per-RUN, per-(role, key): where the run's OpenRouter dollars went)
* ``LITELLM_CALLBACK_DRAIN_TIMEOUT`` — ``metaculus_bot/credit_telemetry.py``
  ``drain_litellm_callbacks`` (per-RUN, at most one line: the callback drain hit its
  bound, so the ``CREDIT_ROLE_SPEND`` rows of that run are a lower bound)
* ``STACKER_OUTCOME`` / ``STACKER_SKIP_REASON`` / ``TOOLS_USED`` — ``metaculus_bot/comment/markers.py``

NOTE ON THE HTML-COMMENT MARKERS: the ones on that last line are ``<!-- ... -->``
markers injected into the *published Metaculus comment*, not logged to stdout/stderr (the
framework logs only ``Posted comment on post N``, never the comment body). They
are therefore almost never present in run logs — their durable source is the
comment itself, which ``metaculus_bot.performance_analysis`` already parses. Their
specs live here so the parser stays complete if a run ever does log a comment
body, and because STACKER_OUTCOME/STACKER_SKIP_REASON/TOOLS_USED are all dormant in
prod anyway (stacking + probabilistic-tools disabled). Don't read their absence from
the telemetry archive as signal. (The ANCHOR_OVERSHOOT_PP / CLAUSE_PRODUCT_DIVERGENCE_PP
specs were deleted 2026-09-02 with the fields that fed them; both had 0 archived rows.)

The parser matches on the marker TOKEN via ``re.search``, so it is agnostic to the
log-line prefix (the prod ``%(asctime)s - %(name)s - %(levelname)s - %(message)s``
format and the ablation ``%(asctime)s %(levelname)s %(name)s | %(message)s`` format
both work).

POST-ID vs QUESTION-ID (the ``qid_kind`` field): Metaculus posts contain questions,
and the two ids DIVERGE on newer posts (post 38880 wraps question 38195). Marker
types are keyed in DIFFERENT spaces, so each :class:`MarkerSpec` declares its own
``qid_kind`` and every harvested record carries it — a residual join keyed on one id
can then TRANSLATE into the record's own space instead of silently dropping the
records keyed on the other (see :mod:`metaculus_bot.performance_analysis.id_mapping`).
The split is mechanical: a marker that logs ``question.id_of_question`` is
``question_id`` and one that logs ``question.page_url`` is ``post_id`` (the gap-fill
v2 / ghost family, whose ``log_prefix`` carries the page URL).

That per-spec field is the ONLY membership statement. This docstring used to also
enumerate which markers were in which space, and the list rotted to 12 of the 26
question-keyed specs, which made the partiality read as if the unlisted ones were
keyed some third way. The current membership is one filter over the registry:

    [spec.name for spec in MARKER_SPECS if spec.qid_kind == QID_KIND_QUESTION_ID]
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

# One ``key=value`` token of a generic counter tail (see the degradation_counters
# spec): keys are Python identifiers, values run to the next comma/whitespace.
_KV_PAIR_RE = re.compile(r"(\w+)=([^,\s]+)")


@dataclass(frozen=True)
class MarkerSpec:
    """One telemetry marker: its archive-file stem, the field regex, and its id space.

    ``qid_kind`` names which Metaculus id space the marker's ``question=`` ref lives
    in (``"post_id"`` or ``"question_id"``); ``None`` for markers with no question
    ref (the credit markers). It is stamped onto every harvested record so a
    residual join knows how to translate a query into the record's id space instead
    of guessing (see the module docstring + ``performance_analysis.id_mapping``).

    ``raw_fields`` names fields of THIS spec kept verbatim rather than coerced, on top of
    the global ``_RAW_FIELDS``. That set is keyed by field name alone, so adding a name
    there changes its meaning on every spec that uses it (``thin_publish_floor.raw`` is a
    float, ``member_forecast.raw`` a JSON literal); a per-spec set keeps the two apart.
    """

    name: str
    regex: re.Pattern[str]
    qid_kind: str | None = None
    raw_fields: frozenset[str] = frozenset()


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
# CLOSE_MARGIN / MARKET_RANKING the ``question=`` is a normal field AFTER the token.
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
        "agentic_fetch_throttled",
        # Per-FETCH: the gap-fill v2 fetch ladder read a 200-OK body that was the host's
        # rate-limit interstitial rather than the page it asked for
        # (research/agentic/tools.py:_throttled_fetch_outcome). Registered because the event
        # had NO trace at all before it and its whole failure mode is looking like a success:
        # on q45191 two throttled ogimet.com fetches were served to the driver as
        # `status: ok`, cached, and replayed on its own retry, so the exact-date reference
        # class it published came to 4 years instead of 6. No `question=` — the tool handlers
        # run below the loop's log_prefix and have no question id, exactly like the credit
        # markers, so a join goes through the run id.
        #
        # `phrase` is the entry of ``fetch_outcomes.FETCH_THROTTLE_PHRASES`` that fired and
        # is last because it contains spaces; with `chars` (the body's length) it is what
        # lets a prod fire be graded true or false positive, and the phrase list and the
        # ``FETCH_THROTTLE_PAGE_MAX_CHARS`` cap retuned on evidence rather than taste.
        re.compile(
            r"AGENTIC_FETCH_THROTTLED:\s*url=(?P<url>\S+)\s+method=(?P<method>\S+)"
            r"\s+chars=(?P<chars>\S+)\s+phrase=(?P<phrase>.*)"
        ),
    ),
    MarkerSpec(
        "agentic_fetch_local_doc",
        # Per-DOCUMENT: the gap-fill v2 ladder read a document without paying for it
        # (research/agentic/local_document.py:log_local_document_read). Registered because it
        # is how the whole local-first change gets measured: before it, every PDF the driver
        # met went to a paid Gemini url_context read, and the only trace of one was the spend.
        # `method` separates the two local routes — `pdf_local` is a fetch serving a PDF's
        # extracted text (which paginates, so it selects nothing), `digest_local` is a
        # read_document answering an ask from BM25-selected passages of text we hold.
        #
        # `chars` is the local text HELD, not the window or digest block handed to the driver,
        # so one figure is comparable across both routes and against URL_CONTEXT_SIZE_GATE_TOKENS
        # (chars / 4). `pages` is n/a for a page with no page structure; `passages` is n/a on a
        # pdf_local line and, on a digest_local one, is the field that says whether the digest
        # actually answered — 0 means the document does not discuss what was asked, which reads
        # in the block itself as an ordinary successful read. No `question=`: the tool handlers
        # run below the loop's log_prefix and have no question id, exactly like the throttle
        # marker above, so a join goes through the run id.
        re.compile(
            r"AGENTIC_FETCH_LOCAL_DOC:\s*url=(?P<url>\S+)\s+method=(?P<method>\S+)"
            r"\s+chars=(?P<chars>\S+)\s+pages=(?P<pages>\S+)\s+passages=(?P<passages>\S+)"
        ),
    ),
    MarkerSpec(
        "agentic_urlcontext_robots_skip",
        # Per-URL: the gap-fill v2 PAID document read was skipped before it spent anything,
        # because the host's robots.txt disallows `Google-Extended` — the product token Gemini's
        # url_context retrieval identifies as, so that read is refused at the host and returns
        # nothing whatever it costs (proven live 2026-09-03 on internationalaisafetyreport.org,
        # against a robots-allowed host that retrieved on the identical call). Registered because
        # the pre-check spends one free request per host to save a paid call, and only these lines
        # say how often it fires: a fire is a call NOT billed, and a suspiciously high rate would
        # mean the group parser is over-matching and withholding reads we could have had.
        #
        # `host` rides beside `url` because the verdict is cached and applied PER HOST, so the
        # host is the unit any rate is computed over. No `question=` — the tool handlers run below
        # the loop's log_prefix and have no question id, exactly like the two markers above, so a
        # join goes through the run id.
        re.compile(r"AGENTIC_URLCONTEXT_ROBOTS_SKIP:\s*url=(?P<url>\S+)\s+host=(?P<host>\S+)"),
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
        "market_ranking",
        # Per-question ranked market-retrieval outcome
        # (research/prediction_market.py:_log_ranking_telemetry). This is the port's own
        # post-ship instrument and the reason it needs a spec rather than a manual grep:
        # `rendered`'s pool INDICES answer whether the ranker's attention decays down a
        # ~400-candidate prompt and whether Manifold detail enrichment shifts which rows
        # get picked, and `prompt_chars` gives the free prod distribution against the
        # ranker's prompt ceiling. Run logs expire from GHA at 90 days, so an unharvested
        # line is an unanswerable question later.
        #
        # `rendered` is a comma-joined `venue:pool_index@rank` list with no spaces, so
        # `\S+` takes the whole field; the "none" sentinel (no rows rendered) coerces to
        # None, which reads correctly alongside rows=0. A pool index of -1 means the row
        # could not be traced back to a pool entry.
        re.compile(
            r"MARKET_RANKING:\s*question=(?P<question>\S+)\s+pool=(?P<pool>\S+)"
            r"\s+outcome=(?P<outcome>\S+)\s+rows=(?P<rows>\S+)"
            r"\s+prompt_chars=(?P<prompt_chars>\S+)\s+rendered=(?P<rendered>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # prediction_market.py emits question.id_of_question
    ),
    MarkerSpec(
        "market_child_render",
        # Per-question multi-outcome CHILD render accounting
        # (research/prediction_market.py:_log_child_render_telemetry). A separate line rather than
        # extra fields on `market_ranking`, because that regex is not end-anchored and a separate
        # spec keeps this harvester change purely additive.
        #
        # Two fields carry the questions this exists to answer. `withheld` counts the prices the
        # venue parsers REFUSED as manufactured — an empty Kalshi book, a Polymarket placeholder leg
        # at Gamma's `["0.5","0.5"]` default, a Manifold answer at its untouched prior. The Kalshi
        # half of that is gated on `KALSHI_NO_PRICE_SPREAD`, a threshold calibrated on eleven fixture
        # strikes, so its prod incidence has to be a query rather than a guess. `max_stage` and
        # `ladder_chars` say whether the ladder's section allowance binds on real slates (0 = every
        # outcome named, 99 = the per-family hard bound).
        #
        # `named` + `collapsed` == `outcomes` is the completeness invariant the render guarantees, so
        # a harvested line where those disagree is a render bug and not a tuning signal.
        re.compile(
            r"MARKET_CHILD_RENDER:\s*question=(?P<question>\S+)\s+families=(?P<families>\S+)"
            r"\s+full_rows=(?P<full_rows>\S+)\s+ladder_rows=(?P<ladder_rows>\S+)"
            r"\s+outcomes=(?P<outcomes>\S+)\s+named=(?P<named>\S+)"
            r"\s+collapsed=(?P<collapsed>\S+)\s+withheld=(?P<withheld>\S+)"
            r"\s+max_stage=(?P<max_stage>\S+)\s+ladder_chars=(?P<ladder_chars>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # prediction_market.py emits question.id_of_question
    ),
    MarkerSpec(
        "market_ranking_degraded",
        # Per-question ranker FAIL-OPEN (research/prediction_market.py:_rank_pool), the
        # discriminating sibling of `market_ranking`'s `outcome=failopen`. That field says a
        # fail-open happened; only this line says which failure, and the distinction is the
        # whole finding: `reason=shape_regression` means the ranker's output was structurally
        # unreadable to US (a renamed index key, indices all out of range), which before
        # 2026-08-25 was reported as `ok(0)` — i.e. indistinguishable from the model
        # deliberately answering "none of these markets bear on the question", which the
        # render turns into an affirmative forecaster-facing sentence. `reason=unreadable`
        # means the completion was not a ranking array at all.
        #
        # NOT end-anchored: `detail=` is free text holding the exception's str (spaces,
        # semicolons, quotes), captured verbatim via _RAW_FIELDS and optional so a terser
        # future form still harvests rather than dropping the record.
        re.compile(
            r"MARKET_RANKING_DEGRADED:\s*question=(?P<question>\S+)\s+pool=(?P<pool>\S+)"
            r"\s+reason=(?P<reason>\S+)(?:\s+detail=(?P<detail>.*))?"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # prediction_market.py emits question.id_of_question
    ),
    MarkerSpec(
        "market_tier_capped",
        # Per-question staleness tier cap (research/prediction_market.py:_log_tier_caps,
        # over `market_retrieval.ranking.cap_stale_top_tier`). Silent on the no-cap case, so
        # a harvested record means the ranker graded a market that stopped trading more than
        # MARKET_STALENESS_TIER_CAP_DAYS before the question opened as `same_quantity_same_date`
        # — the claim a long-closed market cannot make. The demotion also rides the archived
        # snapshot as `MarketMatch.tier_cap_note`, so the incidence is answerable offline
        # too; this line is the prod-log half and the one that survives a snapshot the
        # research archive never captured.
        #
        # It fires on NOTHING in the 102 archived snapshots, and would not have fired on
        # q45163 either (that row was graded one tier lower — see AGENTS.md's
        # prediction-market paragraph), so a first record is itself the finding. `capped` is
        # a comma-joined `venue@rank` list with no spaces, so `\S+` takes the whole field.
        re.compile(
            r"MARKET_TIER_CAPPED:\s*question=(?P<question>\S+)\s+rows=(?P<rows>\S+)"
            r"\s+capped=(?P<capped>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # prediction_market.py emits question.id_of_question
    ),
    MarkerSpec(
        "numeric_degenerate_declaration",
        # Per-FORECASTER point-mass numeric declaration (numeric/pipeline.py
        # _apply_jitter_and_clamp): the model put (near-)identical values at every
        # percentile, so the cluster spreader is deliberately NOT applied and the honest
        # zero span reaches the unit-mismatch guard, which withholds that forecaster. The
        # count is therefore a per-model fabrication-ATTEMPT rate: before 2026-08-25 the
        # spreader manufactured a ±6-unit distribution from it and that width was exactly
        # what let it pass the guard, so the published forecast stated a width nobody
        # declared. The resulting drop shows up as UnitMismatchError in FORECASTER_DROPS;
        # this line is the only place the CAUSE is named.
        #
        # ``model`` names the forecaster (or the stacker, on the aggregation path). All three
        # sanitize_percentiles callers pass it, so "unknown" now means a NEW caller forgot to
        # — it is not in _NONE_SENTINELS, so it stays a readable string rather than coercing
        # into an absent field.
        re.compile(
            r"NUMERIC_DEGENERATE_DECLARATION:\s*question=(?P<question>\S+)\s+model=(?P<model>.+?)"
            r"\s+n_unique=(?P<n_unique>\S+)\s+span=(?P<span>\S+)\s+value_eps=(?P<value_eps>\S+)"
            r"\s+spread_applied=(?P<spread_applied>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # numeric/pipeline.py emits question.id_of_question
    ),
    MarkerSpec(
        "numeric_aggregate_grid_mismatch",
        # Per-MODEL grid-length disagreement inside ensemble aggregation
        # (numeric/utils.py aggregate_numeric). Expect ZERO records in prod: every model's
        # CDF is built on the question's own point count, so a record means a length
        # drifted — which used to matter far more than a resample, because the old
        # group-by-VALUE aggregation silently medianed over a rotating SUBSET of the
        # ensemble whenever an ft-fallback distribution mixed with PCHIP ones (their x-axes
        # differ by float rounding, and on a log-scaled question by construction).
        # Aggregation is positional now, so the mismatch is handled rather than silent —
        # this marker is what keeps "handled" from meaning "unnoticed".
        re.compile(
            r"NUMERIC_AGGREGATE_GRID_MISMATCH:\s*question=(?P<question>\S+)"
            r"\s+model_index=(?P<model_index>\S+)\s+got_points=(?P<got_points>\S+)"
            r"\s+expected_points=(?P<expected_points>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # numeric/utils.py emits question.id_of_question
    ),
    MarkerSpec(
        "cdf_maxstep_clip",
        # Per-CDF-BUILD max-step clip (numeric/pchip_cdf.py safe_cdf_bounds): a bin whose
        # mass exceeded the server's per-bin cap (0.2 * 200 / N) was clipped and the excess
        # moved elsewhere. NOT a bot defect and NOT alertable — the cap is the platform's,
        # so a spike above it is simply unpublishable. What this measures is how much of a
        # published forecast's shape the repair OWNS: q45065 (2026-08-01) capped three
        # forecasters who all declared ~0.72 on the resolving count and, under the old
        # slack-proportional policy, scattered 47% of the mass past 35 deaths where the
        # ensemble had put ~2%. It logged at DEBUG, so the 2026-07-15 "repair-tier WARNs
        # never fire" audit never saw it and the reshaping left no trace in any run log.
        #
        # ``bins_displaced`` and ``max_offset_bins`` are the fields that make the policy
        # itself auditable: nearest-first packing puts the excess a bin or two away (q45065:
        # 4 bins, offset 2), while the retired policy touched nearly every bin on the grid.
        # A record whose ``max_offset_bins`` runs into the tens means the neighbours were
        # already at cap, i.e. a genuinely wide declaration, not a scattered spike.
        #
        # ``model`` is the forecaster whose declaration was clipped, or an ``ensemble_*``
        # label (``ensemble_median`` / ``ensemble_mean`` / ``ensemble_discrete_snap``) for
        # the aggregation stages; the ablation/pooling callers pass none and read "unknown".
        re.compile(
            r"CDF_MAXSTEP_CLIP:\s*question=(?P<question>\S+)\s+model=(?P<model>.+?)"
            r"\s+clipped_mass=(?P<clipped_mass>\S+)\s+over_cap_bins=(?P<over_cap_bins>\S+)"
            r"\s+bins_displaced=(?P<bins_displaced>\S+)\s+max_offset_bins=(?P<max_offset_bins>\S+)"
            r"\s+pre_max_step=(?P<pre_max_step>\S+)\s+max_step=(?P<max_step>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # numeric/pchip_cdf.py is handed question.id_of_question
    ),
    MarkerSpec(
        "numeric_pchip_fallback",
        # Per-question PCHIP CDF build failure (numeric/diagnostics.py log_pchip_fallback):
        # the distribution the forecasters see came from forecasting-tools' fallback CDF
        # builder, not our PCHIP pipeline. The one numeric repair surface with a confirmed
        # prod fire (the repair-tier WARNs upstream of it are dead code on real output —
        # see the module docstring's M16 note), so its absence from the archive was the
        # one genuine blind spot in that family. The question ref is ``id_of_question``,
        # rendered "N/A" when absent (a _NONE_SENTINELS member, coerces to None).
        re.compile(
            r"Question (?P<question>\S+): PCHIP CDF construction failed "
            r"\((?P<error>.*)\), falling back to forecasting-tools default"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # numeric/diagnostics.py emits question.id_of_question
    ),
    MarkerSpec(
        "spread_undefined",
        # Per-question unmeasurable disagreement spread (spread_metrics.py
        # numeric_percentile_spread): the normalizing denominator was non-positive, so no
        # spread could be computed. It returns inf now — before 2026-08-25 it returned 0.0,
        # which route_after_forecasts reads as "the models agree", and the comment marker
        # then claimed `spread_below_threshold`. A measurement FAILURE read as an
        # affirmative agreement signal.
        #
        # Latent in prod while the three per-type stacking gates are off, but it fires in
        # backtests and ablation, where it marks a question whose routing decision was made
        # on no measurement at all. ``qtype`` is a field rather than a literal so a future
        # binary/MC variant of the same shape harvests with no spec change.
        re.compile(
            r"SPREAD_UNDEFINED:\s*question=(?P<question>\S+)\s+qtype=(?P<qtype>\S+)"
            r"\s+denominator=(?P<denominator>\S+)\s+models=(?P<models>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # spread_metrics.py emits question.id_of_question
    ),
    MarkerSpec(
        "ts_anchor_route",
        # Per-question timeseries-anchor routing decision (research/ts_routing.py
        # route_question). This exists because routing was near-unauditable from telemetry:
        # route_question logged only the ambiguous/guard branches, so of the triple era's 30
        # route-level misses exactly 2 left any line in 1,800 run logs, and part (2) of the
        # research-archive-qa dimension could only be written by re-running the router
        # offline. One line per numeric/discrete question makes anchor coverage a query.
        #
        # `decision` is routed|skipped; `step` names the deciding branch on a route
        # (url_single / url_spread / kw_single) or the reject reason on a skip
        # (url_ambiguous, url_quantity_gate, url_change_vs_level_guard,
        # url_no_relative_return_wording, kw_no_keyword_hit, kw_derivation_gate,  # noqa: ERA001  # prose list of route/skip tokens, not code
        # kw_ambiguous, kw_change_vs_level_guard). `series` is the series involved where one
        # is known — comma-joined on ambiguity, slash-joined on a spread, the "none" sentinel
        # (-> None) on a plain keyword miss. All values are spaceless, so `\S+` takes each.
        #
        # DENOMINATOR CAVEAT: the marker covers routing-ELIGIBLE questions, not every
        # numeric/discrete question — build_anchor_section returns before route_question
        # when scheduled_resolution_time is missing/non-datetime, and a disabled
        # TS_ANCHOR_ENABLED run emits nothing. A coverage query must reconcile against the
        # run's question list rather than treat absent lines as skips.
        re.compile(
            r"TS_ANCHOR_ROUTE:\s*question=(?P<question>\S+)\s+decision=(?P<decision>\S+)"
            r"\s+series=(?P<series>\S+)\s+step=(?P<step>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # ts_routing.py emits question.id_of_question
    ),
    MarkerSpec(
        "financial_stale_latest",
        # Stale "latest" observation behind a rendered anchor value, WARNING-level and
        # informational — NOT alertable (the render already tells the forecaster to treat
        # the value as stale; this line makes each surface's prod incidence a query
        # instead of a guess). Two emitters share one shape because they share the
        # estimator (``ts_estimators.stale_latest_age_days``): ``surface=financial_data``
        # is financial_data.py's ``_fetch_yfinance_data``, ``surface=ts_anchor`` is
        # ts_render.py's ``_render_single``.
        #
        # ``symbol`` is a ticker or FRED series id — carets and dots (^GSPC, BRK.B) are
        # spaceless, so ``\S+`` takes it and coerce_value keeps it a string. ``cadence``
        # is the daily-step unit the age was judged against (trading-day /
        # calendar-day). No question ref — the fetch is per-identifier, and one
        # question can fire several — so qid_kind stays None.
        re.compile(
            r"FINANCIAL_STALE_LATEST:\s*surface=(?P<surface>\S+)\s+symbol=(?P<symbol>\S+)"
            r"\s+age_d=(?P<age_d>\S+)\s+cadence=(?P<cadence>\S+)"
        ),
    ),
    MarkerSpec(
        "financial_noise_flag",
        # Vendor-noise flag on a rendered volatility, the sibling of financial_stale_latest
        # and non-alertable for the same reason: the render already tells the forecaster the
        # one-day-return volatility is inflated, so this line exists to make each surface's
        # prod incidence a query rather than a guess. The two emitters again share one shape
        # because they share the estimator (``ts_estimators.variance_ratio`` /
        # ``multi_period_annualized_vol_pct``): ``surface=financial_data`` is
        # financial_data.py's ``_volatility_lines``, ``surface=ts_anchor`` is ts_render.py's
        # ``_realized_vol_lines``.
        #
        # ``vr`` is the Lo-MacKinlay overlapping variance ratio at lag ``vr_lag`` over the
        # provider's full held history; a random walk reads ~1.0 and the flag fires below
        # ``floor``. ``robust_vol`` is the volatility measured on overlapping ``vr_lag``-step
        # returns (the flagged block's headline figure) and reads "None" when the estimator
        # refused the sample. ``long_vol`` is the long-horizon one-day-return volatility and
        # reads "None" on the ts_anchor surface, which computes no long window at all, exactly
        # as it does on a yfinance series too short to hold one — read ``surface`` to tell
        # those apart. Every field is REQUIRED: one shared emitter
        # (``research/noise_flag.py`` ``noise_flag_line``) means one shape, so a future field
        # reorder harvests as a clean zero rather than recording None for a value that WAS
        # emitted, which an optional group in the middle of same-shaped ``\S+`` fields does.
        #
        # ``symbol`` is the ticker or FRED series id the flagged volatility was computed on,
        # in the same field position its stale-latest sibling carries it. It is REQUIRED, not
        # optional-wrapped: the marker ships in the same diff as this spec, so no archived
        # record predates the field. Without it every record was anonymous, and the fan-out is
        # one thread per ticker up to MAX_FINANCIAL_IDENTIFIERS with nondeterministic line
        # order, so two flagged tickers in one run were byte-identical apart from ``seq`` — no
        # join to the stale-latest record for the same series, and no way to tell a pegged-cross
        # true positive from a `^GSPC` false positive at n=1.
        #
        # No question ref: like its stale-latest sibling the flag is per-IDENTIFIER (one
        # question can fire several) and neither call site has the question in scope, so
        # qid_kind stays None.
        re.compile(
            r"FINANCIAL_NOISE_FLAG:\s*surface=(?P<surface>\S+)\s+symbol=(?P<symbol>\S+)"
            r"\s+vr_lag=(?P<vr_lag>\S+)\s+vr=(?P<vr>\S+)\s+floor=(?P<floor>\S+)"
            r"\s+short_vol=(?P<short_vol>\S+)\s+long_vol=(?P<long_vol>\S+)"
            r"\s+robust_vol=(?P<robust_vol>\S+)"
        ),
    ),
    MarkerSpec(
        "fred_unknown_series",
        # A FRED series id that does not exist, as FRED itself reports it (``400 "The series does
        # not exist"``, surfaced by fredapi as a ``ValueError`` carrying that body). Emitted by
        # ``fred_rendering._fetch_fred_data`` on the live path only; the keyless benchmarking
        # fetcher cannot tell a bad id from a vintage predating the series, so it stays silent
        # rather than guessing.
        #
        # NOT alertable: a hallucinated id is the classifier's habit, not a bot crash, and the
        # provider degrades to whatever its other identifiers returned. What the marker buys is
        # the incidence — q45363 lost its whole financial block to ``DEXBOUS`` with only the
        # ambiguous ``DEXBOUS:empty`` source token to show for it, which reads identically to a
        # live series with no observations.
        #
        # ``proposed_by`` splits the two causes, which want different responses: ``classifier``
        # means an LLM invented the id (the prompt's FX routing rule is the fix point), while
        # ``resolution_url`` means the QUESTION's own resolution criteria link a dead FRED page,
        # which is a fact about the question rather than about us.
        #
        # No question ref: the fetch runs in a per-identifier ``to_thread`` worker with no
        # question in scope, the same limitation its ``financial_stale_latest`` /
        # ``financial_noise_flag`` siblings carry, so one question can fire several lines and
        # ``qid_kind`` stays None.
        re.compile(r"FRED_UNKNOWN_SERIES:\s*series_id=(?P<series_id>\S+)\s+proposed_by=(?P<proposed_by>\S+)"),
    ),
    MarkerSpec(
        "resolution_source_fetch",
        # One line per FETCHED URL, emitted at the per-question aggregation point in the
        # provider (that is where the question id exists — threading it down through the
        # monkeypatched fetch surface would change every signature for a log line). The
        # free-text `resolution_source fetched <netloc> (<status>)` lines it replaces were
        # deleted, so no fetch is recorded twice.
        #
        # ``status`` is ``ok`` for a success and the verbatim ``FetchStatus`` otherwise
        # (``blocked`` / ``js_wall`` / ``no_resolving_content`` / ``stale_data`` / ...), the
        # same token the provider-diagnostics source map uses, minus that map's
        # dataset-``stale_data``-to-``none`` amnesty — telemetry keeps the reason verbatim.
        # Since the escalation ladder (2026-09-03) it may be a RUNG's verdict rather than the
        # direct fetch's: the Wayback rung's ``stale_data`` where the direct fetch said
        # ``blocked`` / ``error`` / ``not_found``, the paid reader's ``ungrounded`` where it
        # said ``blocked`` / ``js_wall`` / ``error`` / ``no_resolving_content``. An
        # era-bucketed ``blocked`` rate off this field alone shows a drop at that merge that is
        # bookkeeping, not hosts refusing us less; the direct outcome is ``from_status`` on the
        # sibling escalation line, and ``route`` partitions the two populations.
        # ``http`` is ``n/a`` when no response ever arrived (timeout, client error, SSRF
        # rejection). ``embeds`` names the routeless data-embed providers found in the
        # page's raw HTML, or the ``none`` sentinel (which harvests as None); it is what
        # makes an unreadable-embed page queryable on the qids 44554/44556 shape, where the
        # page carried real prose and the fetch was a legitimate ``ok``.
        #
        # Tier-2 dataset hops ride this marker too and are told apart by ``url``: every
        # dataset is ``static.dwcdn.net/data/<chart_id>.csv``, a host reachable no other
        # way, so a query partitions cited pages from hop artifacts on it.
        #
        # ``reason`` (optional, 2026-09-02) disambiguates a status that has more than one
        # rule behind it: ``no_resolving_content`` is ``embed_shell`` when the page named a
        # routeless data embed, ``thin_page`` when the extraction was simply under the
        # chrome floor (the population the floor gained when it stopped being gated on a
        # named provider), and ``no_matching_passage`` when a cited document read in full
        # discusses nothing the question asks about — the one member that is a document
        # rather than a page. ``unreadable_document`` splits into ``no_text_layer`` /
        # ``encrypted`` / ``malformed``, and ``unsupported_type`` carries
        # ``budget_skipped`` / ``parse_contention`` when it was a document we were holding
        # and declined to parse. The provider appends it only where it applies, so the group
        # is optional in BOTH directions — absent on every line the archive already holds,
        # and absent on a fresh line whose status carries no reason.
        #
        # ``route`` (optional) names which rung of the escalation ladder produced the
        # recorded outcome — ``direct`` for the plain fetch, and ``meta_refresh`` /
        # ``impersonate`` / ``pdf_local`` / ``derived_api`` / ``rendered`` / ``wayback`` /
        # ``url_context`` for an escalated one. Without it a rescued page is indistinguishable
        # from one the direct route read, so "what did the ladder actually buy" is not a query.
        # BOTH optional groups are keyed and at the TAIL, in that order: an optional group
        # sitting BETWEEN same-shaped ``\S+`` fields silently records None for a value that
        # WAS emitted, and a keyed tail group cannot mis-claim its neighbour's value, so a
        # line carrying ``route`` but no ``reason`` parses correctly.
        # ``failure_class`` / ``exc`` / ``server`` (all optional, 2026-09-03) are the failure
        # diagnostics that separate an egress-reputation refusal from a host fault: a small token
        # vocabulary (``http_403`` / ``http_4xx`` / ``http_5xx`` off the response, ``tls`` /
        # ``dns`` / ``timeout`` / ``connection`` / ``decode`` / ``malformed_response`` off the
        # transport exception, the last added 2026-09-04 for a response aiohttp's parser refused
        # — an undecodable Content-Encoding, an oversized header — which recorded as
        # ``connection`` before), the
        # exception class name, and the ``Server`` header lower-cased with internal spaces
        # collapsed to ``_``. Keyed and TAIL-positioned after ``route`` in that fixed order, each
        # emitted only when present, so an old parser and every archived line still parse and a
        # line carrying a later field but not an earlier one cannot mis-claim a neighbour's value.
        re.compile(
            r"RESOLUTION_SOURCE_FETCH:\s*question=(?P<question>\S+)\s+url=(?P<url>\S+)"
            r"\s+status=(?P<status>\S+)\s+http=(?P<http>\S+)\s+embeds=(?P<embeds>\S+)"
            r"(?:\s+reason=(?P<reason>\S+))?(?:\s+route=(?P<route>\S+))?"
            r"(?:\s+failure_class=(?P<failure_class>\S+))?(?:\s+exc=(?P<exc>\S+))?(?:\s+server=(?P<server>\S+))?"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # resolution_source.py emits question.id_of_question
    ),
    MarkerSpec(
        "resolution_source_escalation",
        # One line per ESCALATED URL-rung attempt: the direct fetch could not read the page,
        # so the ladder tried a heavier route. Its sibling ``resolution_source_fetch`` records
        # only the FINAL per-URL outcome, so on its own it cannot say whether a rung rescued
        # the page, how many rungs were spent, or what the attempt cost — ``wall_s`` is the
        # field that decides whether a rung earns its place on a question under a close-derived
        # time budget.
        #
        # ``from_status`` is the verbatim ``FetchStatus`` that triggered the escalation, so the
        # trigger population is queryable without joining back to the fetch marker. Its domain
        # is per rung, and the pairs are disjoint by construction: ``js_wall`` /
        # ``no_resolving_content`` for ``meta_refresh``, ``derived_api`` and ``rendered`` (a page
        # that answered 200 with nothing readable); ``unsupported_type`` for ``pdf_local`` (the
        # content-type router's verdict before the ``%PDF-`` sniff); ``blocked`` / ``error`` /
        # ``not_found`` for ``wayback`` (a page our address never read); and ``blocked`` /
        # ``js_wall`` / ``error`` / ``no_resolving_content`` for ``url_context``. ``blocked``
        # never pairs with a browser rung, since Chromium dials from the same address. ``rung``
        # names the route tried. ``outcome`` and ``wall_s`` are THAT RUNG's own, stamped as it
        # closes (``RungAttempt``): ``outcome`` is the status that stood once the rung was over
        # — its rescue, its verdict (``stale_data``, ``ungrounded``), or the direct status it
        # left standing when it declined — and ``wall_s`` is what that rung alone cost. So on a
        # page where a dead feed GET was followed by a rescuing render, the first line reads the
        # direct status and the second reads ``success``, which is what keeps a rung that fires
        # often but rescues nothing distinguishable from one that never fires at all. Two rungs
        # measure ``wall_s`` narrower than their whole footprint: the local PDF read stamps it
        # inside the parse gate, so queueing for a slot is not billed to the parse, and the paid
        # rung opens its attempt only after its ``Google-Extended`` robots.txt pre-check (a real
        # request, bounded at ``ROBOTS_FETCH_TIMEOUT_S``), so that pre-check is not in its
        # ``wall_s`` — a 15-30% under-count against the rung's 15 s floor when the pre-check has
        # to fetch. Skipped attempts emit no line at all and ride ``details["counts"]`` instead.
        #
        # The token cannot collide with ``RESOLUTION_SOURCE_FETCH``: both specs match on their
        # own full marker word plus the colon, and neither word is a prefix of the other, so
        # the one-marker-per-line ``break`` in ``parse_log_text`` cannot mis-route either line
        # whichever order they sit in.
        re.compile(
            r"RESOLUTION_SOURCE_ESCALATION:\s*question=(?P<question>\S+)\s+url=(?P<url>\S+)"
            r"\s+from_status=(?P<from_status>\S+)\s+rung=(?P<rung>\S+)\s+outcome=(?P<outcome>\S+)"
            r"\s+wall_s=(?P<wall_s>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # resolution_source.py emits question.id_of_question
    ),
    MarkerSpec(
        "resolution_source_urlcontext_robots_skip",
        # Per-URL: the resolution-source ladder's PAID url_context read was skipped before it
        # spent anything, because the host's robots.txt disallows ``Google-Extended``, the product
        # token Gemini's retrieval identifies as, so the read would have been spend with a
        # known-zero return (research/resolution_source.py _url_context_admission). The
        # resolution-source twin of ``agentic_urlcontext_robots_skip`` above: the same pre-check
        # and the same per-host cache (research/robots_policy.py), kept as a separate spec because
        # the two surfaces have different trigger populations, and this one fires only for a cited
        # resolution URL every free rung failed to read. Registered on 2026-09-04, when
        # RESOLUTION_SOURCE_URL_CONTEXT_ENABLED went on in every bot workflow; until then the line
        # could not fire in production and a spec would have archived an always-empty column, so
        # no run from before that merge carries a record.
        #
        # A fire is a paid call NOT billed, so it must not read as a failure; a rate far above the
        # handful of hosts publishing the directive would mean the group parser is over-matching
        # and withholding reads we could have had. ``host`` rides beside ``url`` because the
        # verdict is cached and applied per host, so the host is the unit any rate is taken over.
        # No ``question=``: the rung runs per cited URL inside its provider with no question in
        # scope, exactly like the GEMINI_USAGE row for the same read, so a join goes through the
        # run id.
        re.compile(r"RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP:\s*url=(?P<url>\S+)\s+host=(?P<host>\S+)"),
    ),
    MarkerSpec(
        "resolution_source_urlcontext_ungrounded_suppressed",
        # Per-URL: a PAID url_context read on the resolution-source ladder came back with zero
        # successful retrievals, so its text was discarded as ``ungrounded`` rather than rendered
        # under the primary-grading-evidence caption (research/resolution_source.py
        # _url_context_rung). Gemini answers fluently out of parametric memory when every
        # retrieval failed, and this section tells the forecasters what the resolution source
        # says, so the suppression is the same floor ``gemini_ungrounded_suppressed`` and
        # ``agentic_document_ungrounded_suppressed`` apply, and the three rates read as one
        # family. The read WAS billed, so each record is money spent on nothing served, which is
        # the figure that says whether the rung's trigger population earns its cost. Registered
        # with its two siblings on 2026-09-04, for the same reason.
        #
        # ``statuses`` is the comma-joined list of ``url_retrieval_status`` values the SDK
        # reported, or the ``none`` sentinel when it attached no url_metadata at all (which
        # harvests as None through ``coerce_value``). It splits a retrieval that was attempted and
        # failed for a nameable reason from one the tool never made. Required rather than optional
        # (the v2 twin's tail is optional for archived pre-field lines): the emitter has always
        # written it, and no production line exists from before this spec. No ``question=``, for
        # the same reason as the robots-skip line above.
        re.compile(
            r"RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED:\s*url=(?P<url>\S+)\s+statuses=(?P<statuses>\S+)"
        ),
    ),
    MarkerSpec(
        "resolution_source_urlcontext_not_addressed",
        # Per-URL: a PAID url_context read retrieved the page but answered with the prompt's
        # ``NOT_ADDRESSED`` sentinel, the model's designed reply when the page does not discuss
        # the ask, so the read was withheld as ``no_resolving_content`` / ``not_addressed``
        # instead of rendered (research/resolution_source.py _url_context_rung). Rendered, it was
        # prose standing in for an absent section, the shape the PDF digest closes with
        # ``no_matching_passage``. Distinct from the ungrounded line above: the page WAS
        # retrieved, so Gemini can reach the host, and the money bought a true negative rather
        # than nothing. ``host`` because the rollout question is which hosts Gemini reaches but
        # finds nothing on, the population a sharper ask or a different rung would recover.
        # Registered with its two siblings on 2026-09-04; no ``question=``, as on both of them.
        re.compile(r"RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED:\s*url=(?P<url>\S+)\s+host=(?P<host>\S+)"),
    ),
    MarkerSpec(
        "rendered_fetch_off_host",
        # Per-REFUSED RENDER: headless Chromium's main frame ended up on a host other than the one
        # its DNS pin covers, so the DOM was refused unread, or discarded unpublished when the
        # navigation committed during the read itself (research/rendered_fetch.py render_page; the
        # landing is checked before and after ``page.content()``). Read fail-shut, so Chromium's own
        # error document after a failed navigation (``landed_host=chromewebdata``) is refused too,
        # which makes the record an upper bound on hostile landings. A server-side redirect hop is dialed by the
        # browser with no route handler of ours involved, so the landing is reached through
        # Chromium's own resolver, outside every check the transport makes; a record therefore says
        # a cited page tried to send us somewhere the pin does not cover, which is a
        # security-relevant event rather than a data-quality one.
        #
        # This is the ONLY per-event record of it. The resolution-source caller counts the refusal
        # under ``render_off_host_skips`` in its ``details["counts"]``, which is a per-question
        # total and names neither the pinned host nor the landing, and a skipped rung emits no
        # ``RESOLUTION_SOURCE_ESCALATION`` line at all; the gap-fill v2 caller has no count of its
        # own. Registered 2026-09-04 with the check itself, so no archived run carries a record and
        # a first one is itself the finding.
        #
        # ``scope`` is the transport's ``memo_scope``, ``resolution_source`` or ``gap_fill_v2``:
        # the render path is shared by both callers, whose URL populations differ, and the count
        # only one of them keeps cannot tell them apart. ``landed_host`` is a HOSTNAME and never
        # the landing URL, which can carry a session token or a credential, and it is ``None``
        # (harvested as no data) for an http(s) landing with no hostname at all, the shape that
        # matches no pin and so fails closed into this refusal. No ``question=``: the transport
        # runs per URL with no question in scope, so a join goes through the run id.
        re.compile(
            r"RENDERED_FETCH_OFF_HOST:\s*scope=(?P<scope>\S+)\s+pinned_host=(?P<pinned_host>\S+)"
            r"\s+landed_host=(?P<landed_host>\S+)"
        ),
    ),
    MarkerSpec(
        "forecaster_drops",
        # Per-run ensemble-drop summary emitted by
        # drop_telemetry.py:emit_drop_telemetry. No per-question ref (it
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
        "extreme_call",
        # Per-MEMBER extreme binary call, emitted by metaculus_bot/extreme_call.py's
        # format_extreme_call_markers from forecaster.py's _research_and_make_predictions,
        # right after the survivor count above (which is this marker's denominator: only
        # extreme members get a line, so a rate needs the survivor list from
        # forecasters_survived plus the question's type).
        #
        # ``lone`` is the finding, not ``p``: a member at 0.03 with nobody else in the
        # extreme band on the SAME side behaves nothing like one whose neighbour agrees
        # (4 of 9 right versus 21 of 23 —
        # scratch/residual_2026-08-31/gemini_review/RECOMMENDATION.md §2), and it was
        # re-derived by hand from parsed comments every residual round before this line
        # existed. ``survivors`` rides along because "lone" is vacuous at k=1, so a cut
        # can drop those records instead of joining out to another marker to find them.
        # Those 4-of-9 / 21-of-23 counts come from the memo's own scripts, which read lone
        # as "no other extreme member on EITHER side"; this marker uses the same-side rule
        # the memo's prose states, and extreme_call.py's docstring measures where the two
        # part company before anyone pools old and new counts.
        #
        # Binary questions only — a dominant MC option is a different measurement. The
        # ``model`` field is a bare display-name slug (spaceless), "unknown" when the
        # forecaster's reasoning carried no ``Model:`` prefix; ``lone`` is rendered
        # lowercase and coerce_value lowercases before its bool test, so it harvests as a
        # bool.
        re.compile(
            r"EXTREME_CALL:\s*question=(?P<question>\S+)\s+model=(?P<model>\S+)\s+p=(?P<p>\S+)"
            r"\s+side=(?P<side>\S+)\s+lone=(?P<lone>\S+)\s+survivors=(?P<survivors>\d+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # forecaster.py emits question.id_of_question
    ),
    MarkerSpec(
        "thin_publish_floor",
        # Per-QUESTION single-survivor binary publish floor, emitted by
        # aggregation_pipeline.py's _floor_single_survivor_binary from the base-combine
        # re-entry, ONLY when the lone survivor's value actually moved: ``raw`` is the
        # member's declared probability (what the comment's summary bullet still
        # carries) and ``clamped`` is what was published (THIN_PUBLISH_BINARY_FLOOR /
        # _CEIL in constants.py). A lone value already inside the band leaves no line,
        # so this marker's count IS the floor's prod incidence; the single-survivor
        # EVENT itself is forecasters_survived's ``survived=1``.
        #
        # ``survivors`` is always 1 today and rides along so the record stays
        # self-describing if the k<=2 generalisation the receipt discusses
        # (scratch/residual_2026-08-31/gemini_review/RECOMMENDATION.md §3, "1=") is ever
        # enabled — a cut can then split the two regimes without a join. ``raw`` and
        # ``clamped`` are %.4f, so coerce_value reads them as floats.
        re.compile(
            r"THIN_PUBLISH_FLOOR:\s*question=(?P<question>\S+)\s+raw=(?P<raw>\S+)\s+clamped=(?P<clamped>\S+)"
            r"\s+survivors=(?P<survivors>\d+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # same id space as forecasters_survived / extreme_call
    ),
    MarkerSpec(
        "member_forecast",
        # Per-VALUE record of every forecast that leaves a runner
        # (metaculus_bot/member_forecast.py format_member_forecast_marker): ``raw`` is what
        # the extraction ladder read off the rationale, ``published`` what the runner
        # returned after its clamp (binary), clamp-and-renormalise (MC) or sanitise
        # (numeric). ``role`` is ``member`` for an ensemble forecaster and ``stacker`` for
        # the meta-forecaster, whose numeric line is emitted by aggregation_pipeline.py
        # where its percentiles are sanitised.
        #
        # This is the only marker that carries a member's VALUE on every question. Before
        # it (2026-09-02) the raw value existed solely inside the published comment's
        # per-rationale fenced block — middle-trimmed at COMMENT_CHAR_LIMIT, only present
        # since 2026-05, and recoverable for 74 of 451 resolved binaries when the
        # clip-threshold re-read needed it. EXTREME_CALL's ``p`` covers only members past
        # the extreme band and THIN_PUBLISH_FLOOR only the lone-survivor case.
        #
        # ``raw`` and ``published`` are compact JSON literals with NO whitespace (a float,
        # ``[p1,p2,...]`` in question.options order, or ``[[percentile,value],...]`` with
        # the percentile as a decimal), so ``\S+`` takes each whole; they are in this
        # spec's ``raw_fields`` so the archive holds them verbatim and a consumer always
        # ``json.loads`` — otherwise a binary line would coerce to a float while the MC
        # and numeric vectors stayed strings. ``model`` is ``.+?`` like extraction_rung's,
        # since the same ``forecaster_llm.model`` feeds both.
        re.compile(
            r"MEMBER_FORECAST:\s*question=(?P<question>\S+)\s+model=(?P<model>.+?)\s+role=(?P<role>\S+)"
            r"\s+qtype=(?P<qtype>\S+)\s+raw=(?P<raw>\S+)\s+published=(?P<published>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # every emitter passes question.id_of_question
        raw_fields=frozenset({"raw", "published"}),
    ),
    MarkerSpec(
        "degradation_counters",
        # The per-run summary that DECIDES CI COLOR (cli.py exits non-zero on a
        # positive alertable_count), emitted by forecaster.py's forecast_questions.
        # No per-question ref — it aggregates a whole run — so qid_kind stays None.
        #
        # The tail is parsed GENERICALLY: ``kv_pairs`` captures the whole
        # ``key=value, key=value`` list and ``_build_record`` tokenizes it, so a
        # new counter in ``format_degradation_summary`` harvests with NO change
        # here (the old 17-named-group ``$``-anchored regex needed a coordinated
        # two-file edit per counter, and getting it wrong dropped the whole
        # line's harvest). Historic renames survive as their own keys exactly as
        # before (``research_provider_timeouts`` era-records keep that spelling).
        # One deliberate delta from the old spec: a key absent from an era's line
        # is now ABSENT from the record rather than explicitly None — read
        # ``record.get(key)`` and treat both as "this era didn't emit it", never
        # as a measured zero.
        re.compile(r"Degradation counters:\s*(?P<kv_pairs>.*)$"),
    ),
    MarkerSpec(
        "provider_degradation",
        # Per-run provider-degradation summary (metaculus_bot/research/provider_health.py
        # log_provider_degradation_summary), the positive/negative counterpart to the
        # ``provider_degradation`` counter in the line above. Aggregates a whole run, so
        # qid_kind stays None. Emitted even at ``findings=0``, which makes a measured
        # zero a recorded fact rather than an absent line.
        #
        # ``detail`` is a compact JSON ARRAY of findings captured verbatim (it is in
        # _RAW_FIELDS): venue and field names are delimiter-hostile, and residual
        # analysis json.loads it. The trailing suppression clause is free text after
        # the JSON, so ``detail`` stops at the array's closing bracket.
        #
        # The observation denominators (venues_observed / catalogues_observed /
        # pool_rows, added 2026-08-24) are OPTIONAL-group wrapped: re-harvesting
        # replays the ~1039 archived lines that predate them, and on those a missing
        # group coerces to None — which reads correctly as "not recorded", never as a
        # measured zero. They exist because ``findings=0`` alone is byte-identical
        # between a run that evaluated 400 pool rows and one that evaluated nothing.
        re.compile(
            r"PROVIDER_DEGRADATION:\s*run=(?P<run>\S+)\s+findings=(?P<findings>\d+)"
            r"\s+alertable=(?P<alertable>\d+)\s+suppressed=(?P<suppressed>\d+)"
            r"(?:\s+venues_observed=(?P<venues_observed>\d+)"
            r"\s+catalogues_observed=(?P<catalogues_observed>\d+)"
            r"\s+pool_rows=(?P<pool_rows>\d+))?"
            r"\s+detail=(?P<detail>\[.*?\])"
        ),
    ),
    MarkerSpec(
        "publish_hardening",
        # Per-attempt publish failure WARN (metaculus_bot/publish_hardening.py
        # _wrap_with_timeout_retry). Two emitted shapes share the prefix:
        #   "PUBLISH_HARDENING: <method> attempt N/M timed out after Ts"  # noqa: ERA001  # documented marker format, not commented-out code
        #   "PUBLISH_HARDENING: <method> attempt N/M failed (<ExcType>: <msg>)"  # noqa: ERA001  # documented marker format, not commented-out code
        # The ``attempt N/M`` clause is what keeps this spec off the OTHER
        # PUBLISH_HARDENING-prefixed strings in that module (the applied-INFO line,
        # the seam-moved AttributeErrors, the loop-exited RuntimeError). Exactly one
        # of ``timeout_s`` / (``error_type``, ``error``) is populated per record; the
        # counter it complements is ``publish_attempt_failures`` in the degradation
        # line, but only this marker names WHICH method died and with what. No
        # question ref (the wrapper sees only the POST), so qid_kind stays None.
        re.compile(
            r"PUBLISH_HARDENING:\s*(?P<method>\S+)\s+attempt\s+(?P<attempt>\d+)/(?P<attempts>\d+)\s+"
            r"(?:timed out after (?P<timeout_s>\d+)s"
            r"|failed \((?P<error_type>[^:()]+):\s*(?P<error>.*)\))\s*$"
        ),
    ),
    MarkerSpec(
        "publish_skipped_closed",
        # Per-QUESTION pre-publish skip WARN (metaculus_bot/publish_gate.py). The
        # counterpart to publish_hardening above: that marker fires when a POST was
        # attempted and died, this one when the gate saw the window had closed and
        # made no POST at all. Both point at the same underlying problem (latency
        # against a question's close deadline), and this is the one that names the
        # question and by how many seconds it missed, which is what a latency
        # analysis needs and what CLOSE_MARGIN alone cannot say (a negative margin
        # there does not distinguish "published late but accepted" from "never
        # published"). ``overdue_s`` can be negative under reason=state_closed,
        # meaning the question was shut ahead of its scheduled close.
        re.compile(
            r"PUBLISH_SKIPPED_CLOSED:\s*question=(?P<question>\S+)\s+reason=(?P<reason>\S+)"
            r"\s+close_time=(?P<close_time>\S+)\s+now=(?P<now>\S+)"
            r"\s+overdue_s=(?P<overdue_s>\S+)\s+state=(?P<state>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # publish_gate.py emits question.id_of_question
    ),
    MarkerSpec(
        "time_budget",
        # Per-QUESTION budget grant INFO (metaculus_bot/time_budget.py), emitted for
        # EVERY question including the roomy ones. That is the point: CLOSE_MARGIN,
        # the only other close-time telemetry, is emitted after a SUCCESSFUL
        # submission, so it is censored on exactly the thin-window questions the
        # budget exists for (q45085 had 22 seconds of headroom and appears in no
        # CLOSE_MARGIN record). This marker is the uncensored denominator: how often a
        # window is actually thin, and how often the fast path fires.
        #
        # ``close_limited`` says the close time — not the static
        # PER_QUESTION_WALL_CLOCK_DEADLINE — set the budget; ``fast_path`` says the
        # optional research stages were dropped, and is the per-question detail behind
        # the ``time_budget_fast_path`` counter in the degradation line.
        re.compile(
            r"TIME_BUDGET:\s*question=(?P<question>\S+)\s+budget_s=(?P<budget_s>\S+)"
            r"\s+close_time=(?P<close_time>\S+)\s+close_limited=(?P<close_limited>\S+)"
            r"\s+fast_path=(?P<fast_path>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # time_budget.py emits question.id_of_question
    ),
    MarkerSpec(
        "time_budget_fast_path",
        # Per-QUESTION WARN (forecaster.py) when the close-derived budget dropped the
        # optional research stages. The INFO TIME_BUDGET line above carries the same
        # fact as a field; this is the loud half docs/operations.md tells the operator
        # to grep, and without a spec it vanished at the 90-day GHA log expiry.
        # ``close_time`` is a datetime repr with an internal space, so it captures up
        # to the semicolon rather than as one \S+ token.
        re.compile(
            r"TIME_BUDGET_FAST_PATH:\s*qid=(?P<question>\S+)\s+budget=(?P<budget_s>[\d.]+)s\s+close_time=(?P<close_time>[^;]+);"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # forecaster.py emits question.id_of_question
    ),
    MarkerSpec(
        "research_phase_deadline",
        # Research-phase deadline WARN (research/provider_fanout.py
        # await_providers_within_deadline): the outer budget bound cancelled
        # straggler providers. Carries no question ref — the line names counts and
        # provider names only — so qid_kind stays None; the cancelled providers also
        # survive as status="deadline" rows in the archive's provider_results, which
        # is where per-question attribution lives.
        re.compile(
            r"RESEARCH_PHASE_DEADLINE:\s*cancelled (?P<cancelled>\d+)/(?P<total>\d+) providers"
            r" after (?P<deadline_s>[\d.]+)s \((?P<providers>[^)]*)\)"
        ),
    ),
    MarkerSpec(
        "gap_fill_skipped_for_budget",
        # Per-QUESTION gap-fill skip (research/gap_fill_stages.py): both passes dropped
        # up front, either on the fast path or because the research phase had no
        # budget left. ``research_phase_remaining`` is "n/a" (fast path — never
        # computed) or "NNNs".
        re.compile(
            r"GAP_FILL_SKIPPED_FOR_BUDGET:\s*question=(?P<question>\S+)"
            r"\s+fast_path=(?P<fast_path>\S+)\s+research_phase_remaining=(?P<research_phase_remaining>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # orchestrator logs question.id_of_question
    ),
    MarkerSpec(
        "gap_fill_cut_for_budget",
        # Per-QUESTION mid-phase gap-fill cut (research/gap_fill_stages.py): the pass
        # STARTED and was then cancelled at the research-phase deadline — the one
        # budget event recoverable from nothing else once GHA logs expire (the
        # up-front skip above and the fast path both have their own records).
        # ``gap_fill_pass`` is V1 or V2.
        re.compile(r"GAP_FILL_(?P<gap_fill_pass>V1|V2)_CUT_FOR_BUDGET:\s*question=(?P<question>\S+);"),
        qid_kind=QID_KIND_QUESTION_ID,  # orchestrator logs question.id_of_question
    ),
    MarkerSpec(
        "paid_personal_key_fallback",
        # Per-CALL donated->personal key fallback WARN (fallback_openrouter.py
        # _log_fallback). The counters it feeds are already in degradation_counters /
        # the cli summary, but only this line names WHICH MODEL fell back and with
        # what error, which is what separates "one flaky Gemini call" from "every
        # forecaster ran on the paid key". ``error`` captures greedily to end-of-line
        # because it holds the exception's str.
        re.compile(
            r"PAID PERSONAL-KEY FALLBACK:\s*donated OpenRouter key failed for model=(?P<model>\S+?),"
            r".*?error=(?P<error_type>[^:]+):\s*(?P<error>.*)$"
        ),
    ),
    MarkerSpec(
        "run_alertable_summary",
        # The end-of-run alertable breakdown (cli.py), emitted on EVERY path — the
        # green fully-suppressed case is exactly the one that would otherwise leave no
        # record (the 2026-07-26 drained-key run read alertable=0 alongside real
        # degradation). ``donated_key`` is the /auth/key probe verdict and is
        # OPTIONAL-group wrapped twice over: it is omitted entirely when no spend-cap
        # failure made the wrapper probe, and the suppression clause between it and
        # ``credit`` only appears mid-window.
        #
        # ``outcome`` captures the literal "clean" that cli.py adds when nothing
        # degraded at all (2026-08-25; before then such a run logged no line, so the
        # census counted only degraded runs). It is None on every other shape,
        # including all pre-2026-08-25 records — the alternation is purely additive.
        # The token is what marks a run clean, NOT all-zero fields: a run that lost a
        # question to a raising log_report_summary emits all zeros too (q45085's
        # shape) and deliberately keeps the plain phrase.
        re.compile(
            r"Run completed (?:(?P<outcome>clean) )?with (?P<alertable>\S+) alertable degradation event\(s\)\s*"
            r"\(bot=(?P<bot>\S+?), personal_key_fallback=(?P<personal_key_fallback>\S+?) of which "
            r"donated_404=(?P<donated_404>\S+?), credit=(?P<credit>\S+?)"
            r"(?: with (?P<suppressed_credit>\S+?) credit event\(s\) suppressed until (?P<resume_date>\S+?))?"
            r"(?:, donated_key=(?P<donated_key>\S+?))?\);"
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
        "gemini_grounding_density",
        # The floor's complement (research/gemini_search.py _format_grounded_response): one row
        # per response that PASSED the grounded-chunk floor, carrying how thinly the passing text
        # is attributed. Post-floor the median response has one grounding support per ~872 chars
        # and 41% of passers carry <=3 supports, which is the surface the floor cannot see and
        # where the embellishment rate lives. Deliberately telemetry and never a gate — a decisive
        # true figure once came out of a 1-support response — so nothing keys on these values;
        # they exist so "did embellishment move" is a query over the archive. ``chars`` is the raw
        # model text, so supports/chars reproduces the audit's density denominator.
        re.compile(
            r"GEMINI_GROUNDING_DENSITY:\s*question=(?P<question>\S+)\s+chunks=(?P<chunks>\S+)"
            r"\s+supports=(?P<supports>\S+)\s+chars=(?P<chars>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # gemini_search.py passes question.id_of_question
    ),
    MarkerSpec(
        "gemini_unsupported_attribution",
        # The embellishment channel, per response (research/gemini_search.py
        # _check_attributions): outlet-named source-tier tags — ``[A: NASA]``, ``[B: Reuters]``
        # — that the SAME response's own grounded-domain list does not name, rewritten to
        # ``[unverified attribution]`` at format time. 70% (478 of 681) of the outlet-named tier
        # attributions in the 323 archived Gemini sections are that shape under the shipped
        # keep-biased matcher (86% under the audit's looser rule; receipts in
        # scratch/next_season_bundle_2026-09/item4_attribution_check/VALIDATION.md) and the
        # zero-chunk floor cannot see any
        # of them (it fires only when nothing grounded at all), so before this the rate was a
        # hand audit. ``labels`` is load-bearing context, not decoration: the same
        # ``unsupported`` count reads completely differently against it — q38195 named 21
        # outlets over ONE grounded domain — and ``groups`` is the render footprint, below
        # ``unsupported`` because several unsupported names in one bracket collapse to a single
        # marker. Emitted only when ``unsupported`` > 0; a checked response with none logs
        # nothing and carries its zero in the research archive's provider details instead.
        # NOT alertable: an absent outlet is the model's habit, not a bot defect.
        re.compile(
            r"GEMINI_UNSUPPORTED_ATTRIBUTION:\s*question=(?P<question>\S+)\s+tagged=(?P<tagged>\S+)"
            r"\s+unsupported=(?P<unsupported>\S+)\s+groups=(?P<groups>\S+)\s+labels=(?P<labels>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # gemini_search.py passes question.id_of_question
    ),
    MarkerSpec(
        "gemini_usage",
        # Per-CALL google-genai accounting for ALL THREE Gemini surfaces: grounded search
        # (research/gemini_search.py, role ``grounded_search``), gap-fill v2's read_document
        # (research/agentic/tool_backends.py, role ``read_document``) and the resolution-source
        # ladder's paid url_context rung (research/resolution_source.py, role
        # ``resolution_source``, which emits only from runs with
        # RESOLUTION_SOURCE_URL_CONTEXT_ENABLED on: every bot workflow since 2026-09-04, and no
        # run from before that merge). None routes through OpenRouter, so none
        # shows up in CREDIT_ROLE_SPEND and the whole Google AI Studio side of a run's spend
        # was unmeasurable from the archive — which matters because grounding is metered
        # against a monthly grounded-prompt allowance per project, billed per QUERY on
        # overage, and any feature that multiplies grounded calls re-eats that pool (the
        # spring-2026 billing arc). ``role`` names the surface, so they are separable
        # without keying on the model.
        #
        # Every token field can read ``n/a``: the SDK's usage_metadata fields are individually
        # optional, and a missing count must harvest as None rather than as a measured zero,
        # which is exactly what the ``n/a`` sentinel does through ``coerce_value``.
        # ``search_queries`` is the grounded-query count (the billable unit on overage). On the
        # read_document surface it reads a genuine ``0``, NOT ``n/a``: the SDK omits
        # ``web_search_queries`` when the search tool issued none, and an absent list IS a count
        # of none, which is the honest reading for a url_context-only read. The resolution-source
        # rung reads it the same way, being url_context-only too. It reads ``n/a`` only
        # when the grounding metadata could not be walked at all. So a spend query filters the
        # surfaces on ``role``, never on 0-versus-n/a in this field.
        #
        # The ledger covers COMPLETED responses only. ``log_gemini_usage`` runs after the SDK
        # returns, so a Gemini call that timed out or raised billed unknown tokens and emitted no
        # row — 14 of 154 archived read_document calls (9.1%) hit that handler. A spend total from
        # these rows is therefore a LOWER bound, biased toward undercounting the largest calls;
        # the denominator is ``provider_results['gemini_search'].status`` per question plus
        # ``research_provider_failures``, never this marker's row count.
        #
        # ``question`` is OPTIONAL and last: the grounded-search call site has the question in
        # scope and passes ``question.id_of_question`` (hence ``qid_kind``, matching its
        # ``gemini_grounding_density`` sibling), while read_document runs as a per-URL tool
        # below the loop's log prefix with no question at all, and the resolution-source rung
        # runs per cited URL inside its provider with none either. A keyed tail group is what
        # lets one spec serve all three without recording None for a field that WAS emitted.
        re.compile(
            r"GEMINI_USAGE:\s*role=(?P<role>\S+)\s+model=(?P<model>\S+)"
            r"\s+prompt_tokens=(?P<prompt_tokens>\S+)"
            r"\s+tool_use_prompt_tokens=(?P<tool_use_prompt_tokens>\S+)"
            r"\s+candidates_tokens=(?P<candidates_tokens>\S+)"
            r"\s+thoughts_tokens=(?P<thoughts_tokens>\S+)\s+total_tokens=(?P<total_tokens>\S+)"
            r"\s+search_queries=(?P<search_queries>\S+)(?:\s+question=(?P<question>\S+))?"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # gemini_search.py passes question.id_of_question
    ),
    MarkerSpec(
        "agentic_document_ungrounded_suppressed",
        # The read_document twin of GEMINI_UNGROUNDED_SUPPRESSED (research/agentic/tools.py
        # read_document): Gemini's url_context tool retrieved nothing, so the answer would
        # be unsourced recall and the "fetched" verification tier is withheld. Worth
        # measuring separately because a "fetched" document discrepancy is the only kind
        # that enters the artifact's SUPERSEDE block, i.e. the one that tells every
        # forecaster to override the briefing. Carries no question id — read_document is a
        # per-URL tool with no question in scope — so the URL was for a long time its only field.
        #
        # ``statuses`` (optional) is the comma-joined list of url_context retrieval statuses the
        # SDK reported for that call, or the ``none`` sentinel when it reported none at all
        # (which harvests as None, the same reading an archived pre-field line gets). It splits
        # the two causes a bare suppression cannot: the tool tried and the fetch failed for a
        # nameable reason, versus the tool never retrieved anything to report on. Optional and
        # at the tail so every line the archive already holds parses byte-identically.
        re.compile(r"AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED:\s*url=(?P<url>\S+)(?:\s+statuses=(?P<statuses>\S+))?"),
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
        # ``source`` (added 2026-07-27) names which branch produced the delta and so
        # how much to trust it: ``remaining_delta`` is reliable,
        # ``usage_delta_unsettled`` is a LOWER BOUND (settlement lag — see
        # credit_telemetry's module docstring), ``unavailable`` means no delta. The
        # group is OPTIONAL because re-harvesting replays pre-2026-07-27 logs whose
        # lines end at ``remaining=``; a mandatory tail would drop every one of those
        # records on the next replace-by-run sync. Missing coerces to None, which
        # reads correctly as "this run predates the field".
        re.compile(
            r"CREDIT_SPEND:\s*key=(?P<key>\S+)\s+run_delta_usd=(?P<run_delta_usd>\S+)\s+remaining=(?P<remaining>\S+)"
            r"(?:\s+source=(?P<source>\S+))?"
        ),
    ),
    MarkerSpec(
        "credit_role_spend",
        # Per-(role, key) decomposition of the run's OpenRouter spend, read off
        # OpenRouter's own per-call usage accounting (credit_telemetry.py "Per-role
        # dollar attribution"). ``usd`` is ``n/a`` when no call of that row carried cost
        # data — never a fabricated zero — and ``costed_calls`` says how many of
        # ``calls`` the sum covers. ``byok_usd`` is the upstream-provider component,
        # i.e. the part the donated key books as ``byok_usage``; the personal key is
        # not BYOK, so its rows carry ``byok_usd=0.0000``. Roles are the names in
        # ``credit_telemetry.llm_call_metadata`` (``forecaster:<vendor>``, ``parser``,
        # ``native_search``, ...); ``untagged`` means a completion nobody stamped.
        re.compile(
            r"CREDIT_ROLE_SPEND:\s*role=(?P<role>\S+)\s+key=(?P<key>\S+)\s+usd=(?P<usd>\S+)\s+calls=(?P<calls>\d+)"
            r"\s+costed_calls=(?P<costed_calls>\d+)\s+byok_usd=(?P<byok_usd>\S+)"
        ),
    ),
    MarkerSpec(
        "credit_floor_breach",
        re.compile(r"CREDIT_FLOOR_BREACH:\s*key=(?P<key>\S+)\s+remaining=(?P<remaining>\S+)\s+floor=(?P<floor>\S+)"),
    ),
    MarkerSpec(
        "litellm_callback_drain_timeout",
        # Per-RUN completeness flag on the ``credit_role_spend`` rows above, emitted at most once
        # per run from ``credit_telemetry.drain_litellm_callbacks``: litellm's logging worker did
        # not deliver every queued success callback inside the drain's bound, so the role ledger
        # logged beside it is a LOWER BOUND, missing that run's last few completions. Without this
        # row a low ``reconcile_credit_spend.py --roles`` coverage ratio has two readings — a
        # genuine gap in OpenRouter's per-call cost data, or a drain that gave up — and the archive
        # cannot tell them apart, because no field on the ``credit_role_spend`` row carries the
        # caveat. Registered 2026-09-04; the WARN is new in the 2026-09 bundle and has never fired,
        # so a first record is itself the finding.
        #
        # ``timeout_s`` is the bound the run used, i.e. the ``LITELLM_CALLBACK_DRAIN_TIMEOUT_S``
        # constant unless a caller overrode it, so the row's value is its PRESENCE rather than that
        # near-constant. The same field name sits on the ``publish_hardening`` spec meaning the
        # per-attempt POST timeout; one JSONL per marker keeps the two apart, but a pooled
        # cross-marker query has to key on ``marker``.
        #
        # The regex stays loose either side of the ``within <n>s`` clause so a reword of the
        # surrounding prose keeps harvesting, but that clause is now part of the contract:
        # rewording it would zero the harvest, which the seam pin in tests/test_credit_telemetry.py
        # catches. The line names ``CREDIT_ROLE_SPEND`` in its own prose and cannot be stolen by
        # that spec, which demands ``CREDIT_ROLE_SPEND:\s*role=``, nor by ``credit_spend``.
        re.compile(r"LITELLM_CALLBACK_DRAIN_TIMEOUT:.*?within (?P<timeout_s>[\d.]+)s"),
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
    # Additive skip-reason companion to stacker_outcome (comment/markers.py):
    # separates the mechanisms the plain "skipped" outcome conflates — spread below
    # threshold, per-type config gate off, single-forecaster short-circuit (computes no
    # spread at all), wall-clock budget, and ``spread_undefined`` (the spread could not
    # be MEASURED; it must not read as an affirmative agreement). HTML-comment marker
    # like its parent, so the same rarely-in-run-logs caveat applies.
    MarkerSpec(
        "stacker_skip_reason",
        re.compile(
            r"<!--\s*STACKER_SKIP_REASON=(?P<reason>spread_below_threshold|spread_undefined"
            r"|config_off|single_forecaster|wall_clock_budget)\s*-->",
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
]


def _build_record(
    spec: MarkerSpec,
    match: re.Match[str],
    *,
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
        if field == "kv_pairs":
            # Generic ``key=value`` tail (degradation_counters): every key the
            # emitter writes harvests with no spec change; a key an era's line
            # never emitted is absent from the record, never a measured zero.
            for key, value in _KV_PAIR_RE.findall(raw or ""):
                record[key] = coerce_value(value)
            continue
        record[field] = raw if (field in _RAW_FIELDS or field in spec.raw_fields) else coerce_value(raw)
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
                harvested[spec.name].append(_build_record(spec, match, line=line, seq=counters[spec.name], meta=meta))
                counters[spec.name] += 1
                break  # marker tokens are mutually exclusive — one marker per line
    return harvested
