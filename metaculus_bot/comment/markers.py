"""Shared constants + regex for the stacker-outcome HTML-comment markers.

Two marker families coexist on every stacked comment for one round of back-compat:

* ``STACKER_OUTCOME=<primary|fallback_llm|fallback_median|fallback_mean|skipped|skipped_config_off>``
  — the tri-state-plus marker. ``primary`` and ``fallback_llm`` mean a stacker
  LLM produced the value; ``fallback_median`` means both stacker LLMs failed
  and MEDIAN aggregation was used (CONDITIONAL_STACKING budget-skip path);
  ``fallback_mean`` is the analogous outcome on the regular STACKING budget-
  skip path, where the base-combine re-entry uses MEAN rather than MEDIAN
  (F15 — keeps the marker truthful for residual analysis cuts that bucket on
  aggregation strategy); ``skipped`` means the conditional-stacking trigger
  short-circuited the stacker because spread stayed at/below the threshold;
  ``skipped_config_off`` means spread EXCEEDED the threshold but the per-type
  ``<TYPE>_STACKING_ENABLED`` env gate was off, so the stacker was deliberately
  bypassed (2026-07 residual round: 22 numeric "skipped" suppressions had to be
  re-attributed to config-off via git archaeology — this value makes the reason
  durable in the published record). Comments published before this value
  shipped collapse both skip reasons into ``skipped``; disambiguating those
  requires the workflow-yaml flag history.
* ``STACKED=<true|false>`` — legacy binary marker derived from the new outcome
  (true ↔ outcome ∈ {primary, fallback_llm}, false ↔ outcome ∈ {skipped,
  skipped_config_off, fallback_median, fallback_mean}). Kept around for one
  round so any external parsers don't break the day this fix lands.

Both are injected into each published Metaculus comment by the bot's
``_create_unified_explanation`` override (see ``main.py``), and parsed back out
by the residual-analysis collector via
``metaculus_bot.performance_analysis.parsing.parse_stacked_marker``.

Keeping the literals + regex in one module avoids silent producer/consumer
drift if either side changes the comment shape.
"""

from __future__ import annotations

import re

# Legacy binary marker — retained for one round of back-compat with external
# parsers. New analyses should prefer the STACKER_OUTCOME variant below.
STACKED_MARKER_TRUE: str = "<!-- STACKED=true -->"
STACKED_MARKER_FALSE: str = "<!-- STACKED=false -->"

# Tolerant of surrounding whitespace and casing so accidental reformatting of
# already-published comments (e.g. a markdown editor normalizing whitespace)
# doesn't silently desync the collector.
STACKED_MARKER_RE: re.Pattern[str] = re.compile(
    r"<!--\s*STACKED=(true|false)\s*-->",
    re.IGNORECASE,
)

# Tri-state stacker-outcome marker — replaces the lossy STACKED=true/false
# marker. Distinguishes primary success, fallback-LLM success, median fallback
# (which the old marker silently mislabeled as STACKED=true), and the
# conditional-stacking skip path.
STACKER_OUTCOME_PRIMARY: str = "<!-- STACKER_OUTCOME=primary -->"
STACKER_OUTCOME_FALLBACK_LLM: str = "<!-- STACKER_OUTCOME=fallback_llm -->"
STACKER_OUTCOME_FALLBACK_MEDIAN: str = "<!-- STACKER_OUTCOME=fallback_median -->"
# F15: STACKING budget-skip path uses MEAN base-combine (CONDITIONAL_STACKING uses
# MEDIAN). The original "fallback_median" marker silently mislabeled the STACKING
# variant; this constant gives that path its own bucket so residual analysis cuts
# can separate MEAN-fallback from MEDIAN-fallback without re-deriving the
# strategy from other signals.
STACKER_OUTCOME_FALLBACK_MEAN: str = "<!-- STACKER_OUTCOME=fallback_mean -->"
STACKER_OUTCOME_SKIPPED: str = "<!-- STACKER_OUTCOME=skipped -->"
# Spread exceeded the threshold but the per-type <TYPE>_STACKING_ENABLED gate
# was off — the stacker was config-suppressed, not spread-suppressed.
STACKER_OUTCOME_SKIPPED_CONFIG_OFF: str = "<!-- STACKER_OUTCOME=skipped_config_off -->"

# ``skipped_config_off`` precedes ``skipped`` in the alternation so the longer
# literal wins on first match rather than relying on backtracking after the
# ``skipped`` branch fails the trailing ``-->``.
STACKER_OUTCOME_RE: re.Pattern[str] = re.compile(
    r"<!--\s*STACKER_OUTCOME=(primary|fallback_llm|fallback_median|fallback_mean|skipped_config_off|skipped)\s*-->",
    re.IGNORECASE,
)

# Probabilistic-tools activation marker. Emitted alongside STACKED by
# ``_create_unified_explanation`` so residual analysis can distinguish
# tool-augmented runs from vanilla stacking runs.
TOOLS_USED_MARKER_TRUE: str = "<!-- TOOLS_USED=true -->"
TOOLS_USED_MARKER_FALSE: str = "<!-- TOOLS_USED=false -->"

TOOLS_USED_MARKER_RE: re.Pattern[str] = re.compile(
    r"<!--\s*TOOLS_USED=(true|false)\s*-->",
    re.IGNORECASE,
)

# Per-forecaster anchor / clause telemetry markers (2026-07-08). Emitted by
# ``tool_runner._run_binary_tools`` inside each forecaster's "## Computed
# quantities" block, which is gated behind the ``PROBABILISTIC_TOOLS_ENABLED``
# env flag: ``run_tools_for_forecaster`` returns an empty string when the
# flag is false-y (see ``tool_runner.py``). All three prod workflows pin the
# flag to ``'false'``
# (``.github/workflows/run_bot_on_{tournament,minibench,metaculus_cup}.yaml``),
# so these HTML-comment markers are currently DORMANT in published prod
# comments. NOTE: the ``base_rate_anchor`` / ``criteria_clauses`` fields are
# live in the binary prompt (authored ``30bca2f`` 2026-07-08, live on main
# 2026-07-11T16:37Z in merge ``642b027`` — the merge date is the one to split a
# replay on, per AGENTS.md era-bucketing) — they land unconditionally in every
# prod binary comment's STRUCTURED FORECAST block.
# The COMPUTED markers below (``ANCHOR_OVERSHOOT_PP`` /
# ``CLAUSE_PRODUCT_DIVERGENCE_PP``) are dormant only because they emit from
# ``tool_runner``, which is gated behind ``PROBABILISTIC_TOOLS_ENABLED`` (all
# three prod workflows pin it to ``'false'``). While the flag is off, the
# overshoot / divergence math is trivially replayable offline from the raw
# JSON; the markers become the primary channel if it is ever flipped on.
# (An earlier note here wrongly claimed the elicitation was retired in
# Workstream C2 — that retirement covered the prior/base_rate/hazard/evidence/
# scenario tier-2 fields; the anchor/clause fields shipped the next day as a
# separate, still-live channel. The "0/2203 archived rows" reading was a
# data-window artifact: the archive ended 2026-07-01, before both the authoring
# and the merge date, so the argument holds on either.)
# TELEMETRY ONLY either way: nothing in the pipeline reads these
# back to clamp or mutate a forecast.
ANCHOR_OVERSHOOT_MARKER_PREFIX: str = "ANCHOR_OVERSHOOT_PP"
CLAUSE_DIVERGENCE_MARKER_PREFIX: str = "CLAUSE_PRODUCT_DIVERGENCE_PP"

ANCHOR_OVERSHOOT_MARKER_RE: re.Pattern[str] = re.compile(
    r"<!--\s*ANCHOR_OVERSHOOT_PP=([+-]?\d+(?:\.\d+)?)\s*-->",
    re.IGNORECASE,
)
CLAUSE_DIVERGENCE_MARKER_RE: re.Pattern[str] = re.compile(
    r"<!--\s*CLAUSE_PRODUCT_DIVERGENCE_PP=([+-]?\d+(?:\.\d+)?)\s*-->",
    re.IGNORECASE,
)


# Ensemble-size disclosure marker. Injected into every published comment by
# ``build_unified_explanation``. Records how many forecasters CONTRIBUTED (== the
# number of ``*Forecaster N*`` summary bullets) out of how many were CONFIGURED
# this run. Without it a comment carrying two bullets is ambiguous between a
# 3-model ensemble where one model never answered (a drop) and a genuine 2-model
# era (a roster change) — CLAUDE.md's "fewer than N bullets" gotcha. It rides the
# preserved comment tail alongside the STACKER_OUTCOME markers, so it survives the
# 150k middle-trim. Cause of any drop stays in the run-log FORECASTER_DROPS
# telemetry (operational detail), not the public comment.
FORECASTERS_USED_MARKER_PREFIX: str = "FORECASTERS_USED"

FORECASTERS_USED_MARKER_RE: re.Pattern[str] = re.compile(
    r"<!--\s*FORECASTERS_USED=(\d+)/(\d+)\s*-->",
    re.IGNORECASE,
)


def format_forecasters_used_marker(n_used: int, n_configured: int) -> str:
    """Render the ensemble-size marker: n contributed of N configured (``n/N``)."""
    return f"<!-- {FORECASTERS_USED_MARKER_PREFIX}={n_used}/{n_configured} -->"


def format_anchor_overshoot_marker(overshoot_pp: float) -> str:
    """Render the per-forecaster anchor-overshoot marker (signed, 1 decimal)."""
    return f"<!-- {ANCHOR_OVERSHOOT_MARKER_PREFIX}={overshoot_pp:+.1f} -->"


def format_clause_divergence_marker(divergence_pp: float) -> str:
    """Render the per-forecaster clause-product-divergence marker (signed, 1 decimal)."""
    return f"<!-- {CLAUSE_DIVERGENCE_MARKER_PREFIX}={divergence_pp:+.1f} -->"


# Section headers emitted by ``metaculus_bot.stacking.combine_stacker_and_base_reasoning``
# inside the single R1 body for stacked questions. Shared with
# ``metaculus_bot.performance_analysis.parsing`` which splits the body on
# ``STACKED_BASE_REASONING_HEADER`` to recover per-base-model attribution.
STACKER_META_ANALYSIS_HEADER: str = "## Stacker Meta-Analysis"
STACKED_BASE_REASONING_HEADER: str = "## Base Model Reasoning (inputs to stacker)"

# Splits the base-model portion of a stacker-combined R1 body into per-model
# sub-blocks, one per ``Model: openrouter/<provider>/<name>`` line the bot
# injects ahead of each base reasoning. Shared by two consumers so they can
# never drift: ``metaculus_bot.performance_analysis.parsing`` reads it to
# recover per-base-model attribution, and ``metaculus_bot.comment.trimming``
# uses the same split so a length-trim shrinks each base sub-block from within
# — keeping every base model paired with its own ``Model:`` line and JSON
# forecast block. The value is required to contain ``/`` so a narrative line
# like ``Model: previous version`` inside a base reasoning's prose can't be
# mistaken for a sub-block boundary — the bot-injected prefix is always a
# slash-delimited OpenRouter path (e.g. ``Model: openrouter/openai/gpt-5.5``).
BASE_MODEL_SUBBLOCK_SPLIT_RE: re.Pattern[str] = re.compile(
    r"(?m)^[ \t]*Model:[ \t]*([^\n]*/[^\n]*?)[ \t]*$",
)

# Historical-header signature for stacked comments published before the
# explicit STACKED= / STACKER_OUTCOME= markers existed. Three variants in the
# wild (all stacker-only): ``## Stacker Meta-Analysis`` (current),
# ``## Meta-Analysis`` (older), and ``# Meta-Analysis and Synthesis`` (earliest
# H1). Match condition: meta header is the FIRST heading after
# ``## R1: Forecaster 1 Reasoning`` (modulo a possible ``Model:`` line). A
# bare ``## Meta-Analysis`` deeper in a body isn't signal — that pattern shows
# up inside individual non-stacker forecaster reasoning bodies too.
HISTORICAL_STACKER_META_HEADER: str = "## Meta-Analysis"

HISTORICAL_STACKER_SIGNATURE_RE: re.Pattern[str] = re.compile(
    r"##\s+R1:\s+Forecaster\s+1\s+Reasoning"
    r"(?:\s*\n\s*Model:[^\n]*)?"
    r"\s*\n+"
    r"#{1,2}\s+(?:Stacker\s+)?Meta-Analysis\b",
    re.IGNORECASE,
)

# Sanity-check: the regex must match both literals. If a future edit breaks
# this invariant, fail at import time rather than silently when the collector
# runs weeks later against real comments.
assert STACKED_MARKER_RE.search(STACKED_MARKER_TRUE) is not None, (
    f"STACKED_MARKER_RE does not match STACKED_MARKER_TRUE={STACKED_MARKER_TRUE!r}"
)
assert STACKED_MARKER_RE.search(STACKED_MARKER_FALSE) is not None, (
    f"STACKED_MARKER_RE does not match STACKED_MARKER_FALSE={STACKED_MARKER_FALSE!r}"
)
assert STACKER_OUTCOME_RE.search(STACKER_OUTCOME_PRIMARY) is not None, (
    f"STACKER_OUTCOME_RE does not match STACKER_OUTCOME_PRIMARY={STACKER_OUTCOME_PRIMARY!r}"
)
assert STACKER_OUTCOME_RE.search(STACKER_OUTCOME_FALLBACK_LLM) is not None, (
    f"STACKER_OUTCOME_RE does not match STACKER_OUTCOME_FALLBACK_LLM={STACKER_OUTCOME_FALLBACK_LLM!r}"
)
assert STACKER_OUTCOME_RE.search(STACKER_OUTCOME_FALLBACK_MEDIAN) is not None, (
    f"STACKER_OUTCOME_RE does not match STACKER_OUTCOME_FALLBACK_MEDIAN={STACKER_OUTCOME_FALLBACK_MEDIAN!r}"
)
assert STACKER_OUTCOME_RE.search(STACKER_OUTCOME_FALLBACK_MEAN) is not None, (
    f"STACKER_OUTCOME_RE does not match STACKER_OUTCOME_FALLBACK_MEAN={STACKER_OUTCOME_FALLBACK_MEAN!r}"
)
assert STACKER_OUTCOME_RE.search(STACKER_OUTCOME_SKIPPED) is not None, (
    f"STACKER_OUTCOME_RE does not match STACKER_OUTCOME_SKIPPED={STACKER_OUTCOME_SKIPPED!r}"
)
_skipped_config_off_match = STACKER_OUTCOME_RE.search(STACKER_OUTCOME_SKIPPED_CONFIG_OFF)
assert _skipped_config_off_match is not None, (
    f"STACKER_OUTCOME_RE does not match STACKER_OUTCOME_SKIPPED_CONFIG_OFF={STACKER_OUTCOME_SKIPPED_CONFIG_OFF!r}"
)
# Guard the alternation-order subtlety: the full literal must be captured, not
# just its "skipped" prefix.
assert _skipped_config_off_match.group(1) == "skipped_config_off", (
    f"STACKER_OUTCOME_RE captured {_skipped_config_off_match.group(1)!r} from "
    f"{STACKER_OUTCOME_SKIPPED_CONFIG_OFF!r}; expected 'skipped_config_off'"
)
del _skipped_config_off_match
assert TOOLS_USED_MARKER_RE.search(TOOLS_USED_MARKER_TRUE) is not None, (
    f"TOOLS_USED_MARKER_RE does not match TOOLS_USED_MARKER_TRUE={TOOLS_USED_MARKER_TRUE!r}"
)
assert TOOLS_USED_MARKER_RE.search(TOOLS_USED_MARKER_FALSE) is not None, (
    f"TOOLS_USED_MARKER_RE does not match TOOLS_USED_MARKER_FALSE={TOOLS_USED_MARKER_FALSE!r}"
)
assert ANCHOR_OVERSHOOT_MARKER_RE.search(format_anchor_overshoot_marker(16.2)) is not None, (
    "ANCHOR_OVERSHOOT_MARKER_RE does not match its own formatter output"
)
assert CLAUSE_DIVERGENCE_MARKER_RE.search(format_clause_divergence_marker(-4.0)) is not None, (
    "CLAUSE_DIVERGENCE_MARKER_RE does not match its own formatter output"
)
_forecasters_used_match = FORECASTERS_USED_MARKER_RE.search(format_forecasters_used_marker(2, 3))
assert _forecasters_used_match is not None, "FORECASTERS_USED_MARKER_RE does not match its own formatter output"
assert _forecasters_used_match.group(1) == "2" and _forecasters_used_match.group(2) == "3", (
    f"FORECASTERS_USED_MARKER_RE captured {_forecasters_used_match.groups()!r}; expected ('2', '3')"
)
del _forecasters_used_match
assert STACKER_META_ANALYSIS_HEADER.startswith("## "), (
    f"STACKER_META_ANALYSIS_HEADER must be a markdown H2 header, got {STACKER_META_ANALYSIS_HEADER!r}"
)
assert STACKED_BASE_REASONING_HEADER.startswith("## "), (
    f"STACKED_BASE_REASONING_HEADER must be a markdown H2 header, got {STACKED_BASE_REASONING_HEADER!r}"
)

__all__ = [
    "STACKED_MARKER_TRUE",
    "STACKED_MARKER_FALSE",
    "STACKED_MARKER_RE",
    "STACKER_OUTCOME_PRIMARY",
    "STACKER_OUTCOME_FALLBACK_LLM",
    "STACKER_OUTCOME_FALLBACK_MEDIAN",
    "STACKER_OUTCOME_FALLBACK_MEAN",
    "STACKER_OUTCOME_SKIPPED",
    "STACKER_OUTCOME_SKIPPED_CONFIG_OFF",
    "STACKER_OUTCOME_RE",
    "STACKER_META_ANALYSIS_HEADER",
    "STACKED_BASE_REASONING_HEADER",
    "BASE_MODEL_SUBBLOCK_SPLIT_RE",
    "HISTORICAL_STACKER_META_HEADER",
    "HISTORICAL_STACKER_SIGNATURE_RE",
    "TOOLS_USED_MARKER_TRUE",
    "TOOLS_USED_MARKER_FALSE",
    "TOOLS_USED_MARKER_RE",
    "ANCHOR_OVERSHOOT_MARKER_PREFIX",
    "ANCHOR_OVERSHOOT_MARKER_RE",
    "CLAUSE_DIVERGENCE_MARKER_PREFIX",
    "CLAUSE_DIVERGENCE_MARKER_RE",
    "format_anchor_overshoot_marker",
    "format_clause_divergence_marker",
    "FORECASTERS_USED_MARKER_PREFIX",
    "FORECASTERS_USED_MARKER_RE",
    "format_forecasters_used_marker",
]
