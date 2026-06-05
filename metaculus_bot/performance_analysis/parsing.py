"""Comment text parsing and resolution parsing utilities.

Per-model attribution
---------------------
Metaculus bot comments contain two relevant sections:

1. A summary with ``*Forecaster N*: value`` bullets (one per ensemble member).
2. Rationales sections ``## R1: Forecaster N Reasoning`` whose first line is
   ``Model: openrouter/<provider>/<model>``, injected by
   ``main.TemplateForecaster._run_forecast_on`` when wrapping each prediction.

We key per-model results by the model display name pulled from the rationales'
``Model:`` line, not by the current ``FORECASTER_MODEL_NAMES`` list. This
avoids the failure mode where a roster change in ``llm_configs.py`` silently
relabels historical forecasts.

Stacked comments
----------------
When a question is stacked, the framework collapses all base predictions into a
single ``ReasonedPrediction`` — so there's only ONE ``## R1: Forecaster 1 Reasoning``
block, whose body contains both the stacker's meta-analysis and the base models'
reasonings folded in under a ``## Base Model Reasoning (inputs to stacker)`` sub-header.
See ``metaculus_bot.stacking.combine_stacker_and_base_reasoning`` for the exact
format. Per-base-model parsers below split on that delimiter to recover
attribution for each base model; the summary bullets only show the stacker's
aggregate, so per-base-model values are only recoverable from reasoning prose.
"""

import logging
import re
from collections.abc import Iterator

from metaculus_bot.comment.markers import (
    HISTORICAL_STACKER_SIGNATURE_RE,
    STACKED_BASE_REASONING_HEADER,
    STACKED_MARKER_RE,
    STACKER_OUTCOME_RE,
)


def parse_stacker_outcome_marker(comment_text: str) -> str | None:
    """Return the STACKER_OUTCOME literal in ``comment_text``, else None.

    Returns one of ``"primary"``, ``"fallback_llm"``, ``"fallback_median"``,
    ``"skipped"`` (always lower-cased), or ``None`` if no marker is present.
    Older comments predating the tri-state marker return ``None``.
    """
    match = STACKER_OUTCOME_RE.search(comment_text)
    if match is None:
        return None
    return match.group(1).lower()


def detect_historical_stacker_signature(comment_text: str) -> bool:
    """Return True if the comment carries the pre-marker stacker body signature.

    The stacking commit at 2026-04-02 (`c6d1ab3`) collapsed base predictions
    into a single Forecaster 1 whose reasoning began with `## Meta-Analysis`
    (later renamed to `## Stacker Meta-Analysis` on 2026-04-27, `95c4fff`).
    Comments published in that ~25-day window AND any earlier code variants
    that emitted the same shape carry no explicit `STACKED=` or
    `STACKER_OUTCOME=` marker, but the body alone is recognizable.

    Match conditions: the FIRST `## R1: Forecaster 1 Reasoning` block must
    open with `## (Stacker )?Meta-Analysis` (modulo a possible `Model:` line
    and whitespace). A bare meta-analysis header inside an ordinary forecaster
    body is NOT signal — that's a model's own reasoning structure.

    Returns False on comments that don't match the pattern (including all
    non-stacked comments, all post-marker comments, and the very oldest
    pre-stacking-commit comments).
    """
    return HISTORICAL_STACKER_SIGNATURE_RE.search(comment_text) is not None


def parse_inferred_stacker_outcome(comment_text: str) -> tuple[str | None, str]:
    """Return ``(outcome, source)`` combining marker and historical signature.

    Source is a string explaining how the outcome was determined:

    * ``"marker_outcome"`` — explicit ``STACKER_OUTCOME=...`` marker present.
    * ``"marker_legacy"`` — explicit ``STACKED=true|false`` marker only;
      outcome inferred to ``"primary"`` (true) or ``"skipped"`` (false). The
      legacy marker can't distinguish primary from fallback_llm or skipped
      from fallback_median, so this is a lossy mapping kept for back-compat.
    * ``"historical_body"`` — no marker, but the comment body carries the
      pre-marker stacker signature (`## R1: Forecaster 1 Reasoning` opening
      with `## (Stacker )?Meta-Analysis`). Outcome inferred to ``"primary"``
      since the body shape was only produced when the stacker LLM ran
      successfully — failed-stacker / median-fallback paths in old code did
      NOT collapse to a single Forecaster-1-with-Meta-Analysis shape.
    * ``"none"`` — neither marker nor historical signature present. Returns
      outcome=None, leaving downstream interpretation to the caller (it could
      be a non-stacking strategy, a skipped trigger, or an old comment from
      pre-stacking days).

    Use this when analyzing a dataset that spans multiple code versions —
    e.g., the spring-aib-2026 closing dataset where all forecasts predate
    the explicit markers and the only signal is body shape.
    """
    marker_outcome = parse_stacker_outcome_marker(comment_text)
    if marker_outcome is not None:
        return marker_outcome, "marker_outcome"
    legacy = parse_stacked_marker(comment_text)
    if legacy is True:
        return "primary", "marker_legacy"
    if legacy is False:
        return "skipped", "marker_legacy"
    if detect_historical_stacker_signature(comment_text):
        return "primary", "historical_body"
    return None, "none"


logger: logging.Logger = logging.getLogger(__name__)

# Matches bullet lines like: *Forecaster 3*: 72.0%
# Also matches the annotated form: *Forecaster 3 (gpt-5.5)*: 72.0%
#
# F10: anchor to start-of-line (or start-of-string) so stray ``*Forecaster N*:``
# patterns inside reasoning prose ("...*Forecaster 3*: 50% would have been
# wrong") don't get parsed as real bullets. The bot only ever emits these
# bullets at column 0; quoted occurrences inside prose always have leading
# context. We additionally split on the summary-section boundary in
# ``_summary_section_for_bullets`` to limit the regex to the right section.
_FORECASTER_RE: re.Pattern[str] = re.compile(r"(?m)^\*Forecaster\s+(\d+)(?:\s*\(([^)]+)\))?\*\s*:\s*(.+)")

# Header that follows the summary section (``### Forecasts``) and marks the
# end of the bullet region. Comments are structured as
# ``## Report 1 Summary / ### Forecasts / *Forecaster N*: ... / ### Research
# Summary / ...`` (see ``metaculus_bot.comment.trimming._SUMMARY_END_MARKER``).
# Splitting on this boundary prevents the parser from picking up bullet-shaped
# strings inside research prose or rationale bodies.
_SUMMARY_END_MARKER: str = "### Research Summary"

# Secondary boundary: the rationales divider ``================... FORECAST
# SECTION:``. Used as a backup when the research-summary marker is absent
# (e.g. if it's been trimmed away) but the rationale section is still present.
_FORECAST_SECTION_MARKER_RE: re.Pattern[str] = re.compile(r"^=+\s*\nFORECAST SECTION:", re.MULTILINE)


def _summary_section_for_bullets(comment_text: str) -> str:
    """Return the prefix of ``comment_text`` that contains the summary bullets.

    Splits on ``### Research Summary`` (the canonical end-of-summary marker).
    Falls back to splitting on the ``FORECAST SECTION:`` divider if the
    summary marker is missing. If neither is present, logs a warning and
    returns the full text — caller will fall back to legacy unanchored matching
    (still safer than mislabeling, since the regex itself is now line-anchored).
    """
    marker_idx = comment_text.find(_SUMMARY_END_MARKER)
    if marker_idx >= 0:
        return comment_text[:marker_idx]
    fallback = _FORECAST_SECTION_MARKER_RE.search(comment_text)
    if fallback is not None:
        return comment_text[: fallback.start()]
    logger.warning(
        "No summary-section boundary marker found in comment; falling back to line-anchored matching across full text"
    )
    return comment_text


# Matches the leading "Model: openrouter/..." line prepended to each
# ReasonedPrediction.reasoning by main.TemplateForecaster._make_prediction.
_REASONING_MODEL_PREFIX_RE: re.Pattern[str] = re.compile(r"\AModel:[ \t]*([^\n]*)")

# Shared subpattern for the R1 Forecaster section header. Both _R1_MODEL_RE
# and _R1_SECTION_RE anchor on this — extract it once to avoid lockstep drift
# if the framework ever renames the header.
_R1_HEADER_SUBPATTERN: str = r"##\s+R1:\s+Forecaster\s+(\d+)\s+Reasoning"

# Matches the R1 Forecaster N Reasoning header followed by a Model: line.
# Only Report 1 matters — the summary bullets are always for report 1.
# Horizontal whitespace [ \t]* is used between "Model:" and the value so an
# empty "Model:" line doesn't eat through to the next block.
_R1_MODEL_RE: re.Pattern[str] = re.compile(
    rf"^[ \t]*{_R1_HEADER_SUBPATTERN}[ \t]*\n"
    r"[ \t]*Model:[ \t]*([^\n]*?)[ \t]*$",
    re.MULTILINE,
)

SKIP_RESOLUTIONS: frozenset[str] = frozenset({"annulled", "ambiguous"})

# Matches a leading signed decimal number, optionally followed by '%'. Used by
# ``_parse_probability`` to pull a value out of a forecast string that may have
# surrounding prose (e.g. ``"about 72%"``) — standard comment bullets contain
# just the bare value, but we tolerate stray text since the downstream cost of
# dropping a parseable forecast is higher than the cost of accepting a weird one.
_PROBABILITY_VALUE_RE: re.Pattern[str] = re.compile(r"(-?[0-9]+(?:\.[0-9]+)?)\s*%?")


def _parse_probability(raw: str) -> float | None:
    """Parse a per-model forecast string into a probability in [0, 1].

    Accepts values like ``"72.0%"`` and ``"0.72"``. Heuristic for bare numbers:

    * Explicit ``%`` in the source string → always divide by 100.
    * Bare value in ``[0, 1]`` → treat as already-scaled probability.
    * Bare value in ``[1.5, 100]`` → treat as percentage (the bot's
      missing-``%`` form on values like ``"72"``); divide by 100.
    * Bare value in ``(1.0, 1.5)`` → ambiguous (too high for a valid
      probability, too low to confidently be a percentage). Reject as a
      parse error to avoid silently corrupting Brier/spread calculations
      with a ~2× scaled value (the F11 bug pre-fix).
    * Bare value > 100 or value < 0 → reject as out-of-range parse error.

    Returns ``None`` for any rejected case and logs a WARNING when an
    in-range but ambiguous value is dropped, so operators can grep the
    skip rate for upstream parse drift.
    """
    match = _PROBABILITY_VALUE_RE.search(raw)
    if match is None:
        return None
    try:
        num = float(match.group(1))
    except ValueError:
        return None
    has_percent = "%" in raw
    if has_percent:
        scaled = num / 100.0
    elif 0.0 <= num <= 1.0:
        scaled = num
    elif 1.5 <= num <= 100.0:
        scaled = num / 100.0
    else:
        # Either negative, in the (1.0, 1.5) ambiguous zone, or > 100.
        # The (1.0, 1.5) zone is the load-bearing fix: ``"1.2"`` was being
        # silently coerced to 0.012 — a parse error masquerading as a tiny
        # probability that contaminated Brier / spread metrics. Drop it
        # explicitly with a WARNING so the skip rate is auditable.
        logger.warning(
            "Dropping out-of-range probability value: %r (parsed as %s; "
            "neither a valid decimal probability nor a confident percentage)",
            raw,
            num,
        )
        return None
    if scaled < 0.0 or scaled > 1.0:
        return None
    return scaled


# Matches each R1 Forecaster N Reasoning section, capturing the section body
# up to (but not including) the next ``## R`` heading or end-of-string. Used to
# extract per-forecaster percentile blocks on numeric questions.
_R1_SECTION_RE: re.Pattern[str] = re.compile(
    rf"{_R1_HEADER_SUBPATTERN}\s*\n(.*?)(?=\n##\s+R\d+:|\Z)",
    re.DOTALL,
)

# Matches lines like "Percentile 2.5: 1234.5" (trailing whitespace OK).
_PERCENTILE_LINE_RE: re.Pattern[str] = re.compile(
    r"^\s*Percentile\s+([0-9]+(?:\.[0-9]+)?)\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*$",
    re.MULTILINE,
)

# Matches a ``Model: openrouter/...`` line that starts a base-model sub-block
# inside a stacker-combined body. Must anchor at the start of a line so it
# doesn't match stray mentions inside prose; the value is captured for
# attribution. The value is required to contain ``/`` so we don't accidentally
# split on a narrative line like ``Model: previous version`` inside a base
# reasoning's prose — the bot-injected prefix always uses a slash-delimited
# OpenRouter path (e.g. ``Model: openrouter/openai/gpt-5.5``).
_BASE_MODEL_SUBBLOCK_SPLIT_RE: re.Pattern[str] = re.compile(
    r"(?m)^[ \t]*Model:[ \t]*([^\n]*/[^\n]*?)[ \t]*$",
)


def _split_stacker_combined_body(body: str) -> tuple[str, list[tuple[str | None, str]]] | None:
    """Split a stacker-combined R1 body into (stacker_meta, base_sub_blocks).

    Returns ``None`` if the body does not contain the stacker delimiter — caller
    should fall back to single-block handling.

    On a match, ``stacker_meta`` is everything before the delimiter (with the
    leading stacker ``Model:`` line already stripped, since it's the stacker's
    own model name). ``base_sub_blocks`` is a list of ``(model_name, prose)``
    tuples, one per ``Model: openrouter/...`` line found after the delimiter.
    ``prose`` excludes the leading ``Model:`` line itself.

    Note: when the body has been trimmed mid-base-block by Metaculus's comment
    char limit, the trailing sub-block may be truncated. We still return what
    we can — a partial prose body is usually still useful.
    """
    if STACKED_BASE_REASONING_HEADER not in body:
        return None
    stacker_portion, base_portion = body.split(STACKED_BASE_REASONING_HEADER, 1)

    # Strip the stacker's own "Model:" prefix from the stacker portion so the
    # meta text we return is just the prose.
    stacker_lstripped = stacker_portion.lstrip()
    model_match = _REASONING_MODEL_PREFIX_RE.match(stacker_lstripped)
    if model_match:
        stacker_meta = stacker_lstripped[model_match.end() :].lstrip("\r\n")
    else:
        stacker_meta = stacker_lstripped
    stacker_meta = stacker_meta.rstrip()

    # Walk the base portion, splitting on each line that starts with "Model:".
    # Each match starts a new sub-block; the body of a sub-block runs until
    # the next "Model:" line or end-of-portion.
    sub_blocks: list[tuple[str | None, str]] = []
    matches = list(_BASE_MODEL_SUBBLOCK_SPLIT_RE.finditer(base_portion))
    for i, match in enumerate(matches):
        raw_model = match.group(1).strip()
        model_name: str | None
        if raw_model:
            model_name = raw_model.rsplit("/", 1)[-1].strip() or None
        else:
            model_name = None
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(base_portion)
        prose = base_portion[start:end].strip("\r\n").rstrip()
        sub_blocks.append((model_name, prose))

    return stacker_meta, sub_blocks


def parse_stacked_marker(comment_text: str) -> bool | None:
    """Return True/False if a STACKED=true/false marker is present, else None.

    Older comments without the marker return None. Collectors can use the
    tri-state return to distinguish "known stacked", "known not stacked",
    and "unknown".
    """
    match = STACKED_MARKER_RE.search(comment_text)
    if match is None:
        return None
    return match.group(1).lower() == "true"


def _iter_per_model_blocks(
    comment_text: str,
    model_names: list[str] | None = None,
) -> Iterator[tuple[str, str, bool]]:
    """Yield ``(model_display_name, block_body_text, is_stacker_meta)`` tuples.

    Walks R1 sections, handling both plain and stacker-combined bodies. For
    stacker-combined bodies, yields one entry for the stacker's meta text
    (``is_stacker_meta=True``) followed by one entry per base model
    (``is_stacker_meta=False``). For plain bodies, yields a single entry per R1
    section with ``is_stacker_meta=False``.

    ``block_body_text`` is always the prose with any leading ``Model:`` line
    stripped — callers can feed it directly to line-level regexes without
    worrying about the prefix. Keys use the same attribution fallback chain
    as the public parsers:

    1. explicit ``model_names`` (indexed 1..N),
    2. ``Model:`` line inside each R1 section / base sub-block,
    3. ``Forecaster N`` / ``Forecaster N base`` anonymized fallback.
    """
    if model_names is not None:
        fallback_map: dict[int, str] = {i + 1: name for i, name in enumerate(model_names)}
    else:
        fallback_map = parse_forecaster_model_map(comment_text)

    for match in _R1_SECTION_RE.finditer(comment_text):
        idx = int(match.group(1))
        body = match.group(2)
        body_lstripped = body.lstrip()

        split = _split_stacker_combined_body(body)
        if split is not None:
            stacker_meta, base_sub_blocks = split

            stacker_name = extract_model_display_name_from_reasoning(body_lstripped)
            if stacker_name is None:
                stacker_name = fallback_map.get(idx) or f"Forecaster {idx}"
            yield stacker_name, stacker_meta, True

            for base_model_name, prose in base_sub_blocks:
                key = base_model_name or f"Forecaster {idx} base"
                yield key, prose, False
            continue

        key = extract_model_display_name_from_reasoning(body_lstripped)
        model_match = _REASONING_MODEL_PREFIX_RE.match(body_lstripped)
        if model_match:
            prose = body_lstripped[model_match.end() :].lstrip("\r\n").rstrip()
        else:
            prose = body_lstripped.rstrip()
        if key is None:
            key = fallback_map.get(idx) or f"Forecaster {idx}"
        yield key, prose, False


def parse_per_model_numeric_percentiles(
    comment_text: str,
    model_names: list[str] | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """Extract per-forecaster percentile lists from a numeric/discrete comment.

    Walks each ``## R1: Forecaster N Reasoning`` block and collects every
    ``Percentile P: V`` line inside it. Returns a dict mapping model display
    name → list of (percentile, value) tuples (percentiles are the raw numbers
    from the text, e.g. 2.5, 5, 10 — not normalized to [0, 1]).

    Attribution mirrors ``parse_per_model_forecasts``:
    1. explicit ``model_names`` (indexed 1..N),
    2. ``Model:`` line inside each R1 section,
    3. ``Forecaster N`` anonymized fallback.

    Empty dict if no sections match. Sections without percentile lines are
    skipped (stacker meta-blocks that only reason, no distribution).

    Stacked blocks
    --------------
    When a single R1 block contains ``## Base Model Reasoning (inputs to stacker)``
    (the stacker-combined format emitted by
    ``metaculus_bot.stacking.combine_stacker_and_base_reasoning``), this function
    treats each base-model sub-block as its own attribution target: the dict may
    therefore contain more entries than there are R1 headers. Percentile lines
    appearing in the stacker meta-analysis portion (above the delimiter) are
    also captured under the stacker's model name when present — some stackers
    explicitly restate ``Percentile X: Y`` in their prose and those are useful
    signal.
    """
    result: dict[str, list[tuple[float, float]]] = {}
    for key, body_text, _is_stacker_meta in _iter_per_model_blocks(comment_text, model_names):
        percentiles = [(float(m.group(1)), float(m.group(2))) for m in _PERCENTILE_LINE_RE.finditer(body_text)]
        if not percentiles:
            continue
        result[key] = percentiles
    return result


def extract_model_display_name_from_reasoning(reasoning: str) -> str | None:
    """Return the model display name injected at the top of a reasoning block.

    Returns the last slash-segment of the ``Model: openrouter/<provider>/<name>``
    prefix (e.g. ``gpt-5.2``), or None if the prefix is absent.
    """
    match = _REASONING_MODEL_PREFIX_RE.match(reasoning)
    if match is None:
        return None
    raw = match.group(1).strip()
    return raw.rsplit("/", 1)[-1].strip() or None


def annotate_forecaster_bullets_with_models(
    text: str,
    model_names_by_index: dict[int, str],
) -> str:
    """Rewrite ``*Forecaster N*: value`` bullets to include the model name.

    Idempotent: bullets that already include a parenthesized name are left
    untouched. Indices without a known model are also left untouched.
    """

    def _replace(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        existing_name = m.group(2)
        value = m.group(3)
        if existing_name is not None:
            return m.group(0)
        name = model_names_by_index.get(idx)
        if name is None:
            return m.group(0)
        return f"*Forecaster {idx} ({name})*: {value}"

    return _FORECASTER_RE.sub(_replace, text)


def parse_forecaster_model_map(comment_text: str) -> dict[int, str]:
    """Extract {forecaster_index: model_display_name} from a bot comment.

    Reads ``## R1: Forecaster N Reasoning\\nModel: openrouter/.../name`` blocks
    and returns the index→name map. Model display name is the last
    slash-segment of the OpenRouter path (e.g. ``openrouter/openai/gpt-5.2``
    → ``gpt-5.2``).

    Returns an empty dict if no ``Model:`` lines are found inside R1 headers.
    Malformed entries (empty model value) are skipped.

    Stacked comments
    ----------------
    On stacked comments the single R1 header's ``Model:`` line is the stacker,
    so this map has one entry (the stacker) instead of one-per-base-model. The
    SUMMARY bullet also belongs to the stacker, not the base models. To recover
    per-base-model info from a stacked comment use
    ``parse_per_model_reasoning_text`` and ``parse_per_model_numeric_percentiles``,
    which split the combined R1 body on the
    ``## Base Model Reasoning (inputs to stacker)`` delimiter.
    """
    result: dict[int, str] = {}
    for match in _R1_MODEL_RE.finditer(comment_text):
        idx = int(match.group(1))
        raw_model = match.group(2).strip()
        if not raw_model:
            continue
        display_name = raw_model.rsplit("/", 1)[-1].strip()
        if not display_name:
            continue
        result[idx] = display_name
    return result


def parse_per_model_reasoning_text(
    comment_text: str,
    model_names: list[str] | None = None,
) -> dict[str, str]:
    """Extract per-forecaster reasoning prose from ``## R1: Forecaster N Reasoning`` blocks.

    Returns ``{model_display_name: body_text}``. The leading ``Model: ...`` line
    is stripped so the body is just the prose the model produced. Sections
    whose body is empty after stripping are skipped.

    Attribution mirrors ``parse_per_model_numeric_percentiles`` / ``parse_per_model_forecasts``:

    1. explicit ``model_names`` (indexed 1..N),
    2. ``Model:`` line inside each R1 section,
    3. ``Forecaster N`` anonymized fallback.

    Stacked blocks
    --------------
    When a single R1 block contains ``## Base Model Reasoning (inputs to stacker)``
    (the stacker-combined format from
    ``metaculus_bot.stacking.combine_stacker_and_base_reasoning``), the dict will
    include one entry per base model found after the delimiter plus one entry
    for the stacker's own meta-analysis. This means the returned dict can have
    more entries than there are R1 headers.
    """
    result: dict[str, str] = {}
    for key, body_text, _is_stacker_meta in _iter_per_model_blocks(comment_text, model_names):
        if not body_text:
            continue
        result[key] = body_text
    return result


def parse_per_model_forecasts(
    comment_text: str,
    model_names: list[str] | None = None,
) -> dict[str, str]:
    """Extract per-model predictions from the summary section of a comment.

    Returns dict mapping model name → raw value string (e.g. ``"72.0%"``).

    Attribution sources, in order of preference:

    1. ``model_names`` argument, if provided (back-compat; indexed 1..N).
    2. ``Model:`` lines in R1 rationales sections (primary path).
    3. Anonymized fallback keys (``"Forecaster N"``) if neither is available
       — better to leave a forecast unattributed than mislabel it.

    Only returns per-BASE-model forecasts for UNSTACKED questions. For stacked
    questions the bot publishes a single summary bullet with the stacker's
    aggregate value (the base models' individual forecasts are never written to
    the summary), so this function returns ``{stacker_model: aggregate_value}``
    — that single entry. To recover per-base-model info from a stacked comment
    use ``parse_per_model_reasoning_text`` and
    ``parse_per_model_numeric_percentiles``, which operate on the combined
    reasoning body.
    """
    if model_names is not None:
        fallback_map: dict[int, str] = {i + 1: name for i, name in enumerate(model_names)}
    else:
        fallback_map = parse_forecaster_model_map(comment_text)

    summary_text = _summary_section_for_bullets(comment_text)
    result: dict[str, str] = {}
    for match in _FORECASTER_RE.finditer(summary_text):
        idx = int(match.group(1))
        inline_name = match.group(2)
        value = match.group(3).strip()
        if inline_name is not None:
            key = inline_name.strip()
        else:
            key = fallback_map.get(idx) or f"Forecaster {idx}"
        result[key] = value
    return result


# Matches MC option lines like: ``- Option A: 40.0%`` or ``- Yes (>50%): 60.0%``
# Captures (option_name, numeric_value). The option name runs from after ``- ``
# to the last ``:`` before the numeric value. Anchored at start-of-line.
_MC_OPTION_LINE_RE: re.Pattern[str] = re.compile(r"(?m)^[ \t]*-\s+(.+?):\s+([0-9]+(?:\.[0-9]+)?)\s*%")


def parse_per_model_mc_option_probs(
    comment_text: str,
    model_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Extract per-forecaster MC option probability vectors from a comment.

    For multiple-choice questions the bot emits multi-line bullets::

        *Forecaster 1 (gpt-5.5)*:
        - Option A: 40.0%
        - Option B: 30.0%
        ...

    This function captures ALL option lines per forecaster and returns
    ``{model_display_name: {option_name: probability}}`` where probability
    is in [0, 1].

    Returns an empty dict for binary/numeric comments (no option lines found)
    or for empty input.

    Attribution uses the same fallback chain as ``parse_per_model_forecasts``:
    inline name in bullet > ``Model:`` line in R1 section > ``Forecaster N``.
    """
    if not comment_text:
        return {}

    if model_names is not None:
        fallback_map: dict[int, str] = {i + 1: name for i, name in enumerate(model_names)}
    else:
        fallback_map = parse_forecaster_model_map(comment_text)

    summary_text = _summary_section_for_bullets(comment_text)

    # Find all forecaster bullet positions, then collect option lines between them.
    bullet_matches = list(_FORECASTER_RE.finditer(summary_text))
    if not bullet_matches:
        return {}

    result: dict[str, dict[str, float]] = {}
    for i, match in enumerate(bullet_matches):
        idx = int(match.group(1))
        inline_name = match.group(2)

        if inline_name is not None:
            key = inline_name.strip()
        else:
            key = fallback_map.get(idx) or f"Forecaster {idx}"

        # Region: the captured first-line value (group 3) plus everything
        # until the next bullet or end of summary. Group 3 matters because for
        # MC the regex consumes the first option line (e.g. "- Option A: 40.0%")
        # as part of the (.+) capture.
        first_line = match.group(3)
        after_match_end = bullet_matches[i + 1].start() if i + 1 < len(bullet_matches) else len(summary_text)
        region = first_line + "\n" + summary_text[match.end() : after_match_end]

        # Parse option lines in this region.
        options: dict[str, float] = {}
        for opt_match in _MC_OPTION_LINE_RE.finditer(region):
            option_name = opt_match.group(1).strip()
            prob_pct = float(opt_match.group(2))
            options[option_name] = prob_pct / 100.0

        if options:
            result[key] = options

    return result


_PROBABILITY_LINE_RE: re.Pattern[str] = re.compile(
    r"(?i)(?:final\s+)?probability\s*:\s*(.+)",
)


def parse_per_base_model_forecasts(
    comment_text: str,
    q_type: str,
) -> dict[str, str | dict[str, float]]:
    """Extract per-base-model forecasts from a stacker-combined reasoning body.

    For binary questions: returns ``{model_name: "XX.X%"}`` — one entry per
    base-model sub-block, extracted from the LAST "Probability: X%" or
    "Final probability: X%" line in each block's prose.

    For MC questions: returns ``{model_name: {option: probability}}`` — one
    entry per base-model sub-block, extracted from ``- Option: XX.X%`` lines.

    For numeric/discrete: returns ``{}`` — those question types use
    ``parse_per_model_numeric_percentiles`` which already handles stacked bodies.

    Returns ``{}`` for non-stacked comments (no base-model sub-blocks found).
    """
    if not comment_text:
        return {}
    if q_type in ("numeric", "discrete"):
        return {}

    result: dict[str, str | dict[str, float]] = {}
    for model_name, body_text, is_stacker_meta in _iter_per_model_blocks(comment_text):
        if is_stacker_meta:
            continue

        if q_type == "binary":
            prob = _extract_last_probability_from_body(body_text)
            if prob is not None:
                result[model_name] = f"{prob * 100:.1f}%"

        elif q_type == "multiple_choice":
            options: dict[str, float] = {}
            for opt_match in _MC_OPTION_LINE_RE.finditer(body_text):
                option_name = opt_match.group(1).strip()
                prob_pct = float(opt_match.group(2))
                options[option_name] = prob_pct / 100.0
            if options:
                result[model_name] = options

    # Only return non-empty if we found a stacker-combined body (i.e., there
    # was at least one is_stacker_meta=True block). For non-stacked comments,
    # _iter_per_model_blocks yields blocks but none with is_stacker_meta=True,
    # so we'd be extracting from plain per-forecaster blocks (which are already
    # captured by parse_per_model_forecasts). Return empty to avoid duplication.
    has_stacker_meta = any(is_meta for _, _, is_meta in _iter_per_model_blocks(comment_text))
    if not has_stacker_meta:
        return {}
    return result


def _extract_last_probability_from_body(body_text: str) -> float | None:
    """Extract the LAST probability value from a base-model reasoning body.

    Scans for lines matching "Probability: X%" or "Final probability: X%"
    (case-insensitive) and returns the last one found — that's typically the
    model's final answer after any intermediate estimates.
    """
    last_prob: float | None = None
    for match in _PROBABILITY_LINE_RE.finditer(body_text):
        raw_value = match.group(1).strip()
        parsed = _parse_probability(raw_value)
        if parsed is not None:
            last_prob = parsed
    return last_prob


def parse_resolution(
    resolution_raw: str,
    question_type: str,
) -> tuple[bool | float | str | None, bool]:
    """Parse raw resolution string into a typed value.

    Returns (parsed_value, should_skip). should_skip=True means this question
    should be excluded from analysis.
    """
    if resolution_raw in SKIP_RESOLUTIONS:
        return None, True

    if question_type == "binary":
        if resolution_raw == "yes":
            return True, False
        if resolution_raw == "no":
            return False, False
        logger.warning(f"Unexpected binary resolution: {resolution_raw!r}")
        return None, True

    if question_type in ("numeric", "discrete"):
        if resolution_raw == "above_upper_bound":
            return "above_upper_bound", False
        if resolution_raw == "below_lower_bound":
            return "below_lower_bound", False
        try:
            return float(resolution_raw), False
        except (ValueError, TypeError):
            logger.warning(f"Unparseable numeric resolution: {resolution_raw!r}")
            return None, True

    if question_type == "multiple_choice":
        return resolution_raw, False

    logger.warning(f"Unknown question type: {question_type!r}")
    return None, True
