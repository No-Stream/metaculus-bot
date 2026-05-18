"""Shared constants + regex for the stacker-outcome HTML-comment markers.

Two marker families coexist on every stacked comment for one round of back-compat:

* ``STACKER_OUTCOME=<primary|fallback_llm|fallback_median|fallback_mean|skipped>``
  — the tri-state-plus marker. ``primary`` and ``fallback_llm`` mean a stacker
  LLM produced the value; ``fallback_median`` means both stacker LLMs failed
  and MEDIAN aggregation was used (CONDITIONAL_STACKING budget-skip path);
  ``fallback_mean`` is the analogous outcome on the regular STACKING budget-
  skip path, where the base-combine re-entry uses MEAN rather than MEDIAN
  (F15 — keeps the marker truthful for residual analysis cuts that bucket on
  aggregation strategy); ``skipped`` means the conditional-stacking trigger
  short-circuited the stacker entirely.
* ``STACKED=<true|false>`` — legacy binary marker derived from the new outcome
  (true ↔ outcome ∈ {primary, fallback_llm}, false ↔ outcome ∈ {skipped,
  fallback_median, fallback_mean}). Kept around for one round so any external
  parsers don't break the day this fix lands.

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

STACKER_OUTCOME_RE: re.Pattern[str] = re.compile(
    r"<!--\s*STACKER_OUTCOME=(primary|fallback_llm|fallback_median|fallback_mean|skipped)\s*-->",
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


# Section headers emitted by ``metaculus_bot.stacking.combine_stacker_and_base_reasoning``
# inside the single R1 body for stacked questions. Shared with
# ``metaculus_bot.performance_analysis.parsing`` which splits the body on
# ``STACKED_BASE_REASONING_HEADER`` to recover per-base-model attribution.
STACKER_META_ANALYSIS_HEADER: str = "## Stacker Meta-Analysis"
STACKED_BASE_REASONING_HEADER: str = "## Base Model Reasoning (inputs to stacker)"

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
assert TOOLS_USED_MARKER_RE.search(TOOLS_USED_MARKER_TRUE) is not None, (
    f"TOOLS_USED_MARKER_RE does not match TOOLS_USED_MARKER_TRUE={TOOLS_USED_MARKER_TRUE!r}"
)
assert TOOLS_USED_MARKER_RE.search(TOOLS_USED_MARKER_FALSE) is not None, (
    f"TOOLS_USED_MARKER_RE does not match TOOLS_USED_MARKER_FALSE={TOOLS_USED_MARKER_FALSE!r}"
)
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
    "STACKER_OUTCOME_RE",
    "STACKER_META_ANALYSIS_HEADER",
    "STACKED_BASE_REASONING_HEADER",
    "HISTORICAL_STACKER_META_HEADER",
    "HISTORICAL_STACKER_SIGNATURE_RE",
    "TOOLS_USED_MARKER_TRUE",
    "TOOLS_USED_MARKER_FALSE",
    "TOOLS_USED_MARKER_RE",
]
