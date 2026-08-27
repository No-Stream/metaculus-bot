"""Where a bot comment's sections are, and which model each one belongs to.

The locator + attribution layer under the public parsers in ``parsing``: the
summary-bullet regex and the summary/rationale boundary markers, the
``## R1: Forecaster N Reasoning`` header and ``Model:`` prefix regexes, the
stacker-combined body split, the positional ``Forecaster N`` fallback keys, and
``_iter_per_model_blocks`` — the walk that hands each per-model reasoning body
to those parsers already keyed by model. Split out of ``parsing`` so the regexes
that ``comment.trimming`` has to keep byte-stable sit in one small module,
separate from the parsers that consume them.
"""

import logging
import re
from collections.abc import Iterator

from metaculus_bot.comment.markers import (
    BASE_MODEL_SUBBLOCK_SPLIT_RE,
    STACKED_BASE_REASONING_HEADER,
)

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

# Matches each R1 Forecaster N Reasoning section, capturing the section body
# up to (but not including) the next ``## R`` heading or end-of-string. Used to
# extract per-forecaster percentile blocks on numeric questions.
_R1_SECTION_RE: re.Pattern[str] = re.compile(
    rf"{_R1_HEADER_SUBPATTERN}\s*\n(.*?)(?=\n##\s+R\d+:|\Z)",
    re.DOTALL,
)


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
    stacker_meta = stacker_lstripped[model_match.end() :].lstrip("\r\n") if model_match else stacker_lstripped
    stacker_meta = stacker_meta.rstrip()

    # Walk the base portion, splitting on each line that starts with "Model:".
    # Each match starts a new sub-block; the body of a sub-block runs until
    # the next "Model:" line or end-of-portion.
    sub_blocks: list[tuple[str | None, str]] = []
    matches = list(BASE_MODEL_SUBBLOCK_SPLIT_RE.finditer(base_portion))
    for i, match in enumerate(matches):
        raw_model = match.group(1).strip()
        model_name: str | None
        model_name = (raw_model.rsplit("/", 1)[-1].strip() or None) if raw_model else None
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(base_portion)
        prose = base_portion[start:end].strip("\r\n").rstrip()
        sub_blocks.append((model_name, prose))

    return stacker_meta, sub_blocks


# ---------------------------------------------------------------------------
# Anonymous attribution keys
#
# When neither an explicit roster nor a ``Model:`` line identifies a forecaster,
# the comment parsers key its forecast by POSITION instead of by model. Those keys
# are not model names, and on stacking-era comments the position-1 bucket holds
# whatever the stacker published — so pooling them across questions silently
# mixes stacker aggregates with base-model forecasts. Per-model cuts have to be
# able to recognize and drop them; ``is_anonymous_model_key`` is that predicate,
# built from the same pieces ``anonymous_model_key`` writes so a change to the
# key format can't leave the two out of step.
# ---------------------------------------------------------------------------

_ANONYMOUS_KEY_PREFIX: str = "Forecaster "
_ANONYMOUS_KEY_BASE_SUFFIX: str = " base"

_ANONYMOUS_MODEL_KEY_RE: re.Pattern[str] = re.compile(
    rf"\A{re.escape(_ANONYMOUS_KEY_PREFIX)}\d+(?:{re.escape(_ANONYMOUS_KEY_BASE_SUFFIX)})?\Z"
)


def anonymous_model_key(index: int, *, is_base_model: bool = False) -> str:
    """Return the positional attribution key for forecaster ``index`` (1-based).

    ``is_base_model=True`` returns the variant used for base-model sub-blocks
    inside a stacker-combined body, which all share the enclosing R1 header's
    index and so need distinguishing from the stacker's own key.
    """
    suffix = _ANONYMOUS_KEY_BASE_SUFFIX if is_base_model else ""
    return f"{_ANONYMOUS_KEY_PREFIX}{index}{suffix}"


def is_anonymous_model_key(key: str) -> bool:
    """True when ``key`` is a positional fallback key rather than a model name."""
    return _ANONYMOUS_MODEL_KEY_RE.match(key) is not None


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
                stacker_name = fallback_map.get(idx) or anonymous_model_key(idx)
            yield stacker_name, stacker_meta, True

            for base_model_name, prose in base_sub_blocks:
                key = base_model_name or anonymous_model_key(idx, is_base_model=True)
                yield key, prose, False
            continue

        key = extract_model_display_name_from_reasoning(body_lstripped)
        model_match = _REASONING_MODEL_PREFIX_RE.match(body_lstripped)
        prose = body_lstripped[model_match.end() :].lstrip("\r\n").rstrip() if model_match else body_lstripped.rstrip()
        if key is None:
            key = fallback_map.get(idx) or anonymous_model_key(idx)
        yield key, prose, False
