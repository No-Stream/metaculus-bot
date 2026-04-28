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

from metaculus_bot.comment_markers import STACKED_BASE_REASONING_HEADER, STACKED_MARKER_RE

logger: logging.Logger = logging.getLogger(__name__)

# Matches bullet lines like: *Forecaster 3*: 72.0%
# Also matches the annotated form: *Forecaster 3 (gpt-5.5)*: 72.0%
_FORECASTER_RE: re.Pattern[str] = re.compile(r"\*Forecaster\s+(\d+)(?:\s*\(([^)]+)\))?\*\s*:\s*(.+)")

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

    Accepts values like ``"72.0%"`` and ``"0.72"``. Treats bare numbers > 1 as
    percentages (heuristic for inputs like ``"72"`` missing the ``%``). Returns
    None for unparseable strings or values that fall outside [0, 1] after
    scaling — an out-of-range forecast is almost certainly a parse error and
    shouldn't silently enter Brier/spread calculations.
    """
    match = _PROBABILITY_VALUE_RE.search(raw)
    if match is None:
        return None
    try:
        num = float(match.group(1))
    except ValueError:
        return None
    if "%" in raw or num > 1.0:
        num = num / 100.0
    if num < 0.0 or num > 1.0:
        return None
    return num


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

    result: dict[str, str] = {}
    for match in _FORECASTER_RE.finditer(comment_text):
        idx = int(match.group(1))
        inline_name = match.group(2)
        value = match.group(3).strip()
        if inline_name is not None:
            key = inline_name.strip()
        else:
            key = fallback_map.get(idx) or f"Forecaster {idx}"
        result[key] = value
    return result


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
