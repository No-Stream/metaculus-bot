"""Which per-model entries a per-question ranking is allowed to score.

A per-question ranking (``audit.rank_our_models_by_accuracy`` and the dossier
tables built on it) scores ONE ensemble member per row, so it may only consume
entries that really are one member's own forecast. Three kinds of entry are not,
and all three used to be ranked as though they were:

* a stacker-fired record's summary bullet, which holds the stacker's aggregate
  (16 archived binary records);
* an anonymous ``Forecaster N`` key, a positional bucket rather than a model (57
  archived binary bullets — ``Forecaster 1`` was the third most frequent "best
  model" in the synthesis tally, on 36 wins over 423 binary records);
* a declared percentile curve recovered with too few anchors to rebuild the
  distribution the model actually declared.

The first two exclusions come from ``analysis.py``'s ``per_model_cohort``, called
rather than restated so the rule cannot drift between the aggregate cuts and the
dossiers. The third is specific to the audit's own scoring path, where each
member's declared percentiles are PCHIP'd into a full CDF and log-scored.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from metaculus_bot.performance_analysis.analysis import per_model_cohort
from metaculus_bot.performance_analysis.parsing import is_anonymous_model_key

logger: logging.Logger = logging.getLogger(__name__)

# Minimum DISTINCT percentile labels a member's declared curve needs before its
# PCHIP-rebuilt CDF may be log-scored against another member's. A sparse recovery
# gets scored as a distribution the model never declared: on q43729 a 3-anchor
# curve ranked #1 at +92.01 against five 11-anchor siblings, and on q43826 the
# same shape ranked LAST at -135.86 — a ~96-point artifact either way, which is
# what made "gemini was catastrophically worse" a scoring-path artifact in that
# question's dossier. Deliberately the same number as the per-curve gate in
# ``stacker_detection.exceeded_spread_threshold`` (``len(model_pcts) < 9``): the
# two guard the same field for the same reason and should move together.
MIN_SCOREABLE_ANCHORS: int = 9

# Which per-model field a ranking reads, by question type. MC has no ranker.
_PER_MODEL_FIELD_BY_TYPE: dict[str, str] = {
    "binary": "per_model_forecasts",
    "numeric": "per_model_numeric_percentiles",
    "discrete": "per_model_numeric_percentiles",
}


@dataclass(frozen=True)
class PerModelRankingCohort:
    """Member forecasts a ranking may score, plus everything it dropped and why.

    ``entries`` maps model name to that member's own forecast: a probability
    string for binary, a declared ``(percentile, value)`` list for
    numeric/discrete. The other fields exist so a renderer can STATE an exclusion
    instead of emitting a quietly shorter table:

    * ``stacker_fired`` — the whole record was dropped because its published
      value came from the stacker, so every per-model slot is an aggregate.
    * ``anonymous_keys`` — positional ``Forecaster N`` buckets, which carry no
      model attribution.
    * ``sparse_anchors`` — model to distinct-anchor count, for curves below
      ``MIN_SCOREABLE_ANCHORS``.
    * ``sparse_era`` — no curve on the record cleared the floor, so they are
      ranked against each other after all (see ``_split_by_anchor_density``).
    """

    entries: dict[str, Any]
    stacker_fired: bool = False
    anonymous_keys: tuple[str, ...] = ()
    sparse_anchors: dict[str, int] = field(default_factory=dict)
    sparse_era: bool = False


def declared_anchors(pairs: Sequence[Sequence[float]]) -> tuple[dict[float, float], int]:
    """``(label -> value, n_conflicting_restatements)`` for one declared curve.

    Percentile lines are recovered from comment prose, and a member sometimes
    restates its whole set (one archived curve carries a byte-identical 11-point
    set twice, arriving as 22 pairs). Keying by label is what the PCHIP build
    already does, so the count that matters for density is the number of DISTINCT
    labels, never the number of lines — a 3-anchor set restated three times is
    still a 3-anchor set. A restatement that disagrees with itself is counted so
    the caller can report it; the dict build otherwise takes the last value
    silently.
    """
    anchors: dict[float, float] = {}
    conflicts = 0
    for pair in pairs:
        label, value = float(pair[0]), float(pair[1])
        if label in anchors and anchors[label] != value:
            conflicts += 1
        anchors[label] = value
    return anchors, conflicts


def _split_by_anchor_density(curves: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int], bool]:
    """Split declared curves into the scoreable cohort and the too-sparse rest.

    A curve with at least ``MIN_SCOREABLE_ANCHORS`` distinct anchors is
    scoreable. When at least one curve clears the floor, the ones below it are
    excluded: that is the mixed-density case the gate exists for — five 11-anchor
    members beside one 3-anchor recovery, which is the only shape the archived
    3-anchor curves take (5 records, all ``(3, 11, 11, 11, 11, 11)``).

    When NO curve clears the floor the record is sparse-ERA output rather than a
    partial recovery: fall-2025 comments declare an 8-percentile set, and all 11
    such archived records are uniformly 8-anchor. Every member is equally sparse
    there, so the ranking still compares equals and is kept — an absolute floor
    alone would delete those 11 valid rankings to fix the 5 broken ones.
    ``sparse_era`` tells the renderer to say so, because the absolute log scores
    still are not comparable with a denser question's.
    """
    counts = {model: len(declared_anchors(pairs)[0]) for model, pairs in curves.items()}
    dense = {model for model, n_anchors in counts.items() if n_anchors >= MIN_SCOREABLE_ANCHORS}
    if dense:
        scoreable = {model: pairs for model, pairs in curves.items() if model in dense}
        sparse = {model: n_anchors for model, n_anchors in counts.items() if model not in dense}
        return scoreable, sparse, False
    return dict(curves), {}, bool(curves)


def per_model_ranking_cohort(record: dict) -> PerModelRankingCohort:
    """The per-model entries a ranking of ``record`` may score (see module docstring).

    Pure: scoring call sites do the disclosure logging via
    :func:`log_ranking_cohort`, so a renderer can re-derive the same exclusions
    for its caveat lines without doubling the log.
    """
    field_name = _PER_MODEL_FIELD_BY_TYPE.get(str(record.get("type")))
    if field_name is None:
        return PerModelRankingCohort({})
    per_model = record.get(field_name) or {}
    if not per_model_cohort([record], cut="audit_ranking"):
        # per_model_cohort drops a record only on a confirmed-stacker verdict.
        return PerModelRankingCohort({}, stacker_fired=True)
    anonymous = tuple(sorted(key for key in per_model if is_anonymous_model_key(key)))
    attributed = {name: value for name, value in per_model.items() if not is_anonymous_model_key(name)}
    if field_name == "per_model_numeric_percentiles":
        attributed, sparse_anchors, sparse_era = _split_by_anchor_density(attributed)
    else:
        sparse_anchors, sparse_era = {}, False
    return PerModelRankingCohort(
        entries=attributed,
        anonymous_keys=anonymous,
        sparse_anchors=sparse_anchors,
        sparse_era=sparse_era,
    )


def log_ranking_cohort(record: dict, cohort: PerModelRankingCohort, *, cut: str) -> None:
    """Log what a ranking scored and what it dropped, mirroring ``PER_MODEL_COHORT``."""
    logger.info(
        "AUDIT_PER_MODEL_COHORT: cut=%s post=%s scoreable=%d stacker_fired=%s "
        "excluded_anonymous=%d excluded_sparse_anchors=%d sparse_era=%s "
        "reason=a stacked record's per-model slot holds the stacker aggregate, an anonymous "
        "Forecaster-N key is a positional bucket, and a curve under %d anchors is not the "
        "distribution the model declared",
        cut,
        record.get("post_id"),
        len(cohort.entries),
        cohort.stacker_fired,
        len(cohort.anonymous_keys),
        len(cohort.sparse_anchors),
        cohort.sparse_era,
        MIN_SCOREABLE_ANCHORS,
    )


__all__ = [
    "MIN_SCOREABLE_ANCHORS",
    "PerModelRankingCohort",
    "declared_anchors",
    "log_ranking_cohort",
    "per_model_ranking_cohort",
]
