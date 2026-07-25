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

import json
import logging
import math
import re
from collections.abc import Iterator
from typing import Literal, TypeGuard

from metaculus_bot.comment.markers import (
    BASE_MODEL_SUBBLOCK_SPLIT_RE,
    FORECASTERS_USED_MARKER_RE,
    HISTORICAL_STACKER_SIGNATURE_RE,
    STACKED_BASE_REASONING_HEADER,
    STACKED_MARKER_RE,
    STACKER_OUTCOME_RE,
)
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.numeric.percentile_set import PERCENTILE_KEY_DECIMALS
from metaculus_bot.structured_output_schema import (
    BinaryStructured,
    MultipleChoiceStructured,
    NumericStructured,
    StructuredBlock,
    extract_json_block,
    parse_structured_block,
)


def parse_stacker_outcome_marker(comment_text: str) -> str | None:
    """Return the STACKER_OUTCOME literal in ``comment_text``, else None.

    Returns one of ``"primary"``, ``"fallback_llm"``, ``"fallback_median"``,
    ``"fallback_mean"``, ``"skipped"``, ``"skipped_config_off"`` (always
    lower-cased), or ``None`` if no marker is present. Older comments
    predating the tri-state marker return ``None``; comments published before
    ``skipped_config_off`` shipped (2026-07-19) collapse both skip reasons
    into ``"skipped"``.
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
    matches = list(BASE_MODEL_SUBBLOCK_SPLIT_RE.finditer(base_portion))
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


def parse_forecasters_used_marker(comment_text: str) -> tuple[int, int] | None:
    """Return ``(n_used, n_configured)`` from a FORECASTERS_USED marker, else None.

    ``n_used`` is how many forecasters contributed to the published aggregate (==
    the number of per-model summary bullets); ``n_configured`` is the roster size
    that run. When ``n_used < n_configured`` the question published on a degraded
    ensemble (a model dropped), which is what disambiguates a missing bullet from
    a genuine roster change. Older comments predating the marker return None
    (unknown ensemble size), so callers can distinguish "known degraded",
    "known full", and "unknown".
    """
    match = FORECASTERS_USED_MARKER_RE.search(comment_text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


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


# ---------------------------------------------------------------------------
# Structured-block extraction (block-first, prose-regex fallback)
#
# Since 2026-07 the forecaster prompts emit forecast values ONLY inside the
# fenced ```json STRUCTURED FORECAST block; the trailing prose value lines
# ("Probability: NN%", "Percentile X: V", "- Option: NN%") are gone from new
# rationales. Historical comments carry the prose lines and often no block, so
# the per-model reasoning-body parsers below try the JSON block first and fall
# back to the prose regexes — no era detection, just per-body try-then-fallback.
#
# NOTE: this applies only to parsers that read the R1 REASONING bodies (via
# ``_iter_per_model_blocks``). ``parse_per_model_mc_option_probs`` reads the
# SUMMARY bullet region, which is bot-rendered display text
# (``forecasting_tools.forecast_bots.forecast_bot`` formats each parsed
# prediction via ``make_readable_prediction`` → ``- {option}: {prob}%`` lines),
# unaffected by the prompt change — it stays prose-only.
# ---------------------------------------------------------------------------


def _parse_block_in_body(
    body_text: str,
    question_type: Literal["binary", "numeric", "multiple_choice"],
) -> StructuredBlock | None:
    """Parse the structured JSON block in a per-model reasoning body, if any.

    Pre-checks for a fenced block so historical prose-only bodies skip the
    full parse (and its per-body "no JSON block" logging) entirely.
    ``parse_structured_block`` returns None on malformed JSON / validation
    failure, so callers just fall back to the prose regex on None.
    """
    if extract_json_block(body_text) is None:
        return None
    return parse_structured_block(body_text, question_type)


def _numeric_percentiles_from_block(body_text: str) -> list[tuple[float, float]] | None:
    """Return sorted (percentile, value) pairs from the body's JSON block, or None.

    The block stores percentile keys as decimals (0.025, 0.5, 0.9) while the
    prose convention downstream expects the raw percent labels the regex path
    captures (2.5, 50, 90) — convert so consumers see identical shapes
    regardless of comment era. Rounded to 6 places to cancel float noise
    (0.1 * 100 == 10.000000000000002), matching the label rounding in
    ``stacker_detection``.
    """
    block = _parse_block_in_body(body_text, "numeric")
    if not isinstance(block, NumericStructured) or not block.declared_percentiles:
        return None
    return sorted((round(pct * 100.0, 6), value) for pct, value in block.declared_percentiles.items())


def _binary_prob_from_block(body_text: str) -> float | None:
    """Return the declared posterior probability from the body's JSON block, or None."""
    block = _parse_block_in_body(body_text, "binary")
    if not isinstance(block, BinaryStructured):
        return None
    return block.posterior_prob


def _mc_option_probs_from_block(body_text: str) -> dict[str, float] | None:
    """Return {option_name: probability} from the body's JSON block, or None."""
    block = _parse_block_in_body(body_text, "multiple_choice")
    if not isinstance(block, MultipleChoiceStructured):
        return None
    return dict(block.option_probs)


# ---------------------------------------------------------------------------
# Tolerant raw-JSON salvage (last rung, after strict block AND prose regex)
#
# Historical blocks (pre-2026-07-08 prompt era) carry retired tier-2 scaffold
# fields (``mixture_components``, ``tails``, ``distribution_family_hint``, the
# old ``scenarios`` shape) or values the current strict schemas reject
# (``concentration: 0.0``). ``parse_structured_block`` uses extra="forbid"
# schemas, so it returns None for the ENTIRE block even though the declared
# forecast values inside are fine. Models that emitted block-only rationales
# with no prose value lines (gemini-3.1-pro in the 2026-05/06 era) then vanish
# from recovered per-model data entirely — the 2026-07-15 ensemble screening's
# false "gemini missed 5 of 45 summer questions" finding was exactly this
# (4 of the 5 were parse losses, verified against GHA run logs; only 1 was a
# real soft-deadline drop). The salvage below reads just the value field from
# the raw JSON, faithful to what the model emitted, mirroring the tolerant
# read pinned in scratch/coherence_2026-07-15/schema_README.md.
#
# Precedence is deliberately strict-block > prose > tolerant: transition-era
# comments where a sparse/invalid block coexists with full prose value lines
# keep resolving from prose exactly as before.
# ---------------------------------------------------------------------------

_KNOWN_BLOCK_QUESTION_TYPES: frozenset[str] = frozenset({"binary", "numeric", "multiple_choice"})


def _tolerant_block_payload(
    body_text: str,
    question_type: Literal["binary", "numeric", "multiple_choice"],
) -> dict | None:
    """Raw ``json.loads`` of the body's fenced block, bypassing schema validation.

    Returns the payload dict, or None when there is no block, the JSON is
    malformed, or the block declares a *conflicting* known question_type
    (an unknown/absent question_type passes — field extraction filters it).
    """
    raw = extract_json_block(body_text)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    declared_type = payload.get("question_type")
    if declared_type in _KNOWN_BLOCK_QUESTION_TYPES and declared_type != question_type:
        return None
    return payload


def _is_finite_number(value: object) -> TypeGuard[float]:
    """True for real int/float values (bool excluded, NaN/inf excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _numeric_percentiles_from_block_tolerant(body_text: str) -> list[tuple[float, float]] | None:
    """Salvage (percentile, value) pairs from a strict-invalid numeric block.

    Same decimal→raw-percent label conversion as the strict reader. Keys must
    coerce to float in (0, 1); values must be finite numbers. Returns None when
    nothing usable survives.
    """
    payload = _tolerant_block_payload(body_text, "numeric")
    if payload is None:
        return None
    declared = payload.get("declared_percentiles")
    if not isinstance(declared, dict):
        return None
    pairs: list[tuple[float, float]] = []
    for key, value in declared.items():
        try:
            pct = float(key)
        except (TypeError, ValueError):
            continue
        if not (0.0 < pct < 1.0) or not _is_finite_number(value):
            continue
        pairs.append((round(pct * 100.0, 6), float(value)))
    return sorted(pairs) if pairs else None


def _binary_prob_from_block_tolerant(body_text: str) -> float | None:
    """Salvage the posterior probability from a strict-invalid binary block."""
    payload = _tolerant_block_payload(body_text, "binary")
    if payload is None:
        return None
    prob = payload.get("posterior_prob")
    if not _is_finite_number(prob) or not (0.0 <= prob <= 1.0):
        return None
    return float(prob)


def _mc_option_probs_from_block_tolerant(body_text: str) -> dict[str, float] | None:
    """Salvage {option_name: probability} from a strict-invalid MC block.

    Per-value bounds are enforced but the strict sum≈1 check is NOT — the
    downstream consumers (option-set match + renormalization) handle scale.
    """
    payload = _tolerant_block_payload(body_text, "multiple_choice")
    if payload is None:
        return None
    raw_options = payload.get("option_probs")
    if not isinstance(raw_options, dict):
        return None
    options: dict[str, float] = {}
    for key, value in raw_options.items():
        if not isinstance(key, str) or not key.strip():
            return None
        if not _is_finite_number(value) or not (0.0 <= value <= 1.0):
            return None
        options[key] = float(value)
    if not options or sum(options.values()) <= 0.0:
        return None
    return options


# ---------------------------------------------------------------------------
# Percentile-label validation (guard against double-scaled labels)
#
# ``parse_per_model_numeric_percentiles`` emits labels in the PERCENT convention
# (2.5, 5, ..., 97.5, 99) — every legitimate label is strictly inside (0, 100).
# A forecaster that writes its block keys in percent form (2.5, 5, ...) instead
# of the decimal convention (0.025, 0.05, ...) has them multiplied by 100 by the
# block readers, producing double-scaled labels (250, 500, ..., 9750). That was
# the qid=43684 grok-4.3 record the 2026-07-15 coherence study had to hand-drop
# (drop_reason=malformed_percentile_labels_out_of_range). The guard below runs
# after extraction and either passes clean labels through, deterministically
# rescales an exact canonical-set*100 match, or rejects the model's percentiles.
# ---------------------------------------------------------------------------

# Canonical label sets in the percent convention (STANDARD_PERCENTILES * 100).
# The 13-point set is the current standard; the 11-point subset (P1/P99 dropped)
# is the pre-2026-07-07 archive-era shape. Both are derived from the decimal-form
# single source of truth so a percentile-set change stays in lockstep.
_CANONICAL_PERCENT_LABELS_13: frozenset[float] = frozenset(
    round(p * 100.0, PERCENTILE_KEY_DECIMALS) for p in STANDARD_PERCENTILES
)
# P1 (0.01) and P99 (0.99) are the finer tail anchors added when the set grew 11->13.
_CANONICAL_PERCENT_LABELS_11: frozenset[float] = _CANONICAL_PERCENT_LABELS_13 - {
    round(0.01 * 100.0, PERCENTILE_KEY_DECIMALS),
    round(0.99 * 100.0, PERCENTILE_KEY_DECIMALS),
}
_CANONICAL_PERCENT_LABEL_SETS: tuple[frozenset[float], ...] = (
    _CANONICAL_PERCENT_LABELS_13,
    _CANONICAL_PERCENT_LABELS_11,
)


def _validate_percentile_labels(
    percentiles: list[tuple[float, float]],
    *,
    model: str,
    question_id: int | str | None = None,
) -> list[tuple[float, float]] | None:
    """Guard extracted percentile labels against the double-scaled failure mode.

    Runs after label extraction/normalization (block, prose, or tolerant rung).
    The percent-label convention puts every legitimate label strictly inside
    ``(0, 100)`` (P1..P99). This guard:

    * returns the list unchanged when every label is in ``(0, 100)``;
    * deterministically divides every label by 100 (with a WARNING) when the set
      is EXACTLY a canonical label set * 100 (11- or 13-point) — a safe,
      reversible correction of percent-vs-decimal double scaling, not a guess;
    * otherwise returns ``None`` (with a WARNING naming the model and offending
      labels) so the caller drops the model rather than emit out-of-range labels
      into downstream calibration.

    ``question_id`` is optional log context; callers that lack a qid pass None.
    """
    labels = [label for label, _ in percentiles]
    if all(0.0 < label < 100.0 for label in labels):
        return percentiles

    rescaled_label_set = frozenset(round(label / 100.0, PERCENTILE_KEY_DECIMALS) for label in labels)
    if rescaled_label_set in _CANONICAL_PERCENT_LABEL_SETS:
        logger.warning(
            "Rescaling double-scaled percentile labels (label/100 == canonical set): "
            "question=%s model=%s offending_labels=%s",
            question_id,
            model,
            sorted(labels),
        )
        return [(round(label / 100.0, PERCENTILE_KEY_DECIMALS), value) for label, value in percentiles]

    logger.warning(
        "Rejecting numeric percentiles with out-of-range labels: question=%s model=%s offending_labels=%s",
        question_id,
        model,
        sorted(labels),
    )
    return None


def parse_per_model_numeric_percentiles(
    comment_text: str,
    model_names: list[str] | None = None,
    question_id: int | str | None = None,
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

    Every extracted percentile set is run through ``_validate_percentile_labels``
    before it is returned: labels outside ``(0, 100)`` are either deterministically
    rescaled (when the set is an exact canonical-set*100 double-scaling) or the
    model is dropped, so downstream calibration never sees out-of-range labels.
    ``question_id`` is optional context threaded into that guard's WARNING logs.

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
        percentiles = _numeric_percentiles_from_block(body_text)
        if percentiles is None:
            percentiles = [(float(m.group(1)), float(m.group(2))) for m in _PERCENTILE_LINE_RE.finditer(body_text)]
        if not percentiles:
            # Tolerant salvage: historical blocks with retired tier-2 fields
            # fail the strict schema, and block-only rationales have no prose
            # lines — without this rung those models vanish from recovery.
            percentiles = _numeric_percentiles_from_block_tolerant(body_text) or []
        if not percentiles:
            continue
        validated = _validate_percentile_labels(percentiles, model=key, question_id=question_id)
        if validated is None:
            continue
        result[key] = validated
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

    Deliberately prose-only (no structured-block path): this function reads
    the SUMMARY bullet region, which is bot-rendered display text — the
    framework formats each parsed prediction via
    ``MultipleChoiceReport.make_readable_prediction`` into
    ``- {option}: {prob}%`` lines (``forecasting_tools/forecast_bots/
    forecast_bot.py::_format_and_expand_research_summary``). The 2026-07
    block-only prompt change alters what the MODEL writes in its R1 rationale,
    not how the bot renders the summary, so these bullets keep their prose
    form across eras. The forecaster's JSON block lives in the R1 reasoning
    body and is handled by the ``_iter_per_model_blocks`` consumers.
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
            prob = _binary_prob_from_block(body_text)
            if prob is None:
                prob = _extract_last_probability_from_body(body_text)
            if prob is None:
                # Tolerant salvage for historical strict-invalid blocks with
                # no prose value lines (see the salvage-rung comment above).
                prob = _binary_prob_from_block_tolerant(body_text)
            if prob is not None:
                result[model_name] = f"{prob * 100:.1f}%"

        elif q_type == "multiple_choice":
            options = _mc_option_probs_from_block(body_text)
            if options is None:
                options = {}
                for opt_match in _MC_OPTION_LINE_RE.finditer(body_text):
                    option_name = opt_match.group(1).strip()
                    prob_pct = float(opt_match.group(2))
                    options[option_name] = prob_pct / 100.0
            if not options:
                # Tolerant salvage, e.g. blocks with concentration=0.0 that
                # the strict validator rejects wholesale.
                options = _mc_option_probs_from_block_tolerant(body_text) or {}
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
