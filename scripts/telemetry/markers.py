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
* ``FORECASTER_DROPS`` — ``metaculus_bot/drop_telemetry.py`` ``emit_drop_telemetry``
  (per-RUN summary: which models dropped and why)
* ``Degradation counters`` — ``metaculus_bot/degradation_counters.py``
  ``format_degradation_summary`` (per-RUN counter set that decides CI color)
* ``FORECASTERS_SURVIVED`` — ``metaculus_bot/forecaster.py``
  ``_research_and_make_predictions`` (per-QUESTION positive survivor count; the
  drop marker above is silent on a healthy question, and its comment-side twin
  ``FORECASTERS_USED`` never reaches stdout)
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
* ``NUMERIC_DEGENERATE_DECLARATION`` — ``metaculus_bot/numeric/pipeline.py``
  ``_apply_jitter_and_clamp`` (per-FORECASTER point-mass numeric declaration that
  is no longer cluster-spread into a width nobody stated — a fabrication-attempt
  rate, since the unit-mismatch guard then withholds that forecaster)
* ``NUMERIC_AGGREGATE_GRID_MISMATCH`` — ``metaculus_bot/numeric/utils.py``
  ``aggregate_numeric`` (per-MODEL CDF whose grid length disagreed with the
  question's; expect zero in prod, so any record means a length drifted)
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
* ``CREDIT_BALANCE`` / ``CREDIT_SPEND`` / ``CREDIT_FLOOR_BREACH`` — ``metaculus_bot/credit_telemetry.py``
* ``STACKER_OUTCOME`` / ``STACKER_SKIP_REASON`` / ``TOOLS_USED`` /
  ``ANCHOR_OVERSHOOT_PP`` / ``CLAUSE_PRODUCT_DIVERGENCE_PP`` — ``metaculus_bot/comment/markers.py``

NOTE ON THE HTML-COMMENT MARKERS: the ones on that last line are ``<!-- ... -->``
markers injected into the *published Metaculus comment*, not logged to stdout/stderr (the
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
``CLOSE_MARGIN`` / ``MARKET_RANKING`` / ``MARKET_RANKING_DEGRADED`` /
``NUMERIC_DEGENERATE_DECLARATION`` / ``NUMERIC_AGGREGATE_GRID_MISMATCH`` /
``SPREAD_UNDEFINED`` / ``numeric_pchip_fallback`` emit ``question.id_of_question``
(the QUESTION id) while
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
        # _await_providers_within_deadline): the outer budget bound cancelled
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
                harvested[spec.name].append(_build_record(spec, match, line=line, seq=counters[spec.name], meta=meta))
                counters[spec.name] += 1
                break  # marker tokens are mutually exclusive — one marker per line
    return harvested
