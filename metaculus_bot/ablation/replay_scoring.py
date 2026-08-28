"""Scoring the replayed aggregates and summarizing them under iterated k-fold CV.

One responsibility: turning per-question aggregate predictions into scores and then into the
per-config comparison the harness reports — per-question primary log score (plus the Brier /
CRPS secondaries), the per-config score arrays, the iterated k-fold resample summary, and the
saturation / degeneracy diagnostics that keep a collapsed config from reading as a winner.
Split out of ``ablation.offline_replay`` so the scoring path (pure ``scoring_common``
primitives over score arrays) is readable and testable apart from the cache loader and the
aggregation catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metaculus_bot.ablation.replay_configs import (
    MEDIAN_BASELINE,
    BinaryConfig,
    MCConfig,
    NumericConfig,
)
from metaculus_bot.ablation.replay_dataset import BinaryRecord, MCRecord, NumericRecord
from metaculus_bot.backtest.scoring import numeric_crps
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.scoring_common import binary_log_score, brier_score, mc_log_score, numeric_log_score

# Scoring (Route A — pure primitives)
#
# PRIMARY metric for every type is the Metaculus-style log score (higher = better),
# INCLUDING saturation blowups. Secondary diagnostics: Brier (binary) and CRPS (numeric).
# Saturation = the per-question primary log score falling below SATURATION_THRESHOLD.

# A binary/MC log score of -100 happens at the prob clamp (p ~ 1e-4, log2(p) ~ -13.3 →
# 100*(log2(1e-4)+1) ~ -1230 for binary). We define "saturated" as a deeply negative
# per-question score; -200 captures genuine blowups (confidently-wrong) without flagging
# routine misses. Numeric uses 50*ln(pmf/baseline); a near-empty bucket (~1e-15) gives a
# very negative score. Same -200 threshold.
SATURATION_THRESHOLD: float = -200.0


def score_binary(record: BinaryRecord, predicted_prob: float) -> tuple[float, float]:
    """(primary log score, secondary brier) for a binary aggregate prediction."""
    return binary_log_score(predicted_prob, record.outcome), brier_score(predicted_prob, record.outcome)


def score_mc(record: MCRecord, predicted_probs: list[float]) -> float:
    """Primary MC log score for an aggregate option-prob vector."""
    return mc_log_score(predicted_probs, record.correct_option_index)


def score_numeric(record: NumericRecord, cdf_values: list[float]) -> tuple[float, float]:
    """(primary log score, secondary CRPS) for a numeric aggregate CDF."""
    q = record.question
    log_score = numeric_log_score(
        cdf_values,
        record.resolution_value,
        float(q.lower_bound),
        float(q.upper_bound),
        open_lower_bound=bool(q.open_lower_bound),
        open_upper_bound=bool(q.open_upper_bound),
        zero_point=float(q.zero_point) if q.zero_point is not None else None,
    )
    # CRPS x-values must be the SAME grid the production CDF lives on: geometric for
    # zero_point (log-scaled) questions, linear otherwise. A linear grid here would
    # mis-locate the CDF mass for zero_point questions and bias CRPS.
    zero_point = float(q.zero_point) if q.zero_point is not None else None
    x_values = list(build_cdf_value_grid(float(q.lower_bound), float(q.upper_bound), zero_point, len(cdf_values)))
    # CRPS needs an in-range resolution; clamp the out-of-bounds sentinel back to the grid.
    crps_resolution = min(max(record.resolution_value, x_values[0]), x_values[-1])
    crps = numeric_crps(x_values, cdf_values, crps_resolution)
    return log_score, crps


# Per-question scoring across all configs


def score_all_binary(records: list[BinaryRecord], configs: dict[str, BinaryConfig]) -> dict[str, np.ndarray]:
    """Per-config array of per-question primary log scores, aligned to ``records`` order."""
    return {
        name: np.array([score_binary(r, config(r))[0] for r in records], dtype=float)
        for name, config in configs.items()
    }


def score_all_mc(records: list[MCRecord], configs: dict[str, MCConfig]) -> dict[str, np.ndarray]:
    return {name: np.array([score_mc(r, config(r)) for r in records], dtype=float) for name, config in configs.items()}


def score_all_numeric(records: list[NumericRecord], configs: dict[str, NumericConfig]) -> dict[str, np.ndarray]:
    return {
        name: np.array([score_numeric(r, config(r))[0] for r in records], dtype=float)
        for name, config in configs.items()
    }


# Iterated k-fold cross-validation


@dataclass(frozen=True)
class ConfigCVResult:
    """CV summary for one config vs. the median baseline.

    ``mean_log_score`` / ``std_log_score`` are over the held-out-fold mean log score across
    all (iteration, fold) resamples. ``delta_vs_median_mean`` / ``delta_vs_median_std`` are
    the PAIRED held-out delta (config minus median, per question, averaged within each fold)
    summarized across resamples. ``full_data_log_score`` is the plain mean over all questions
    (no resampling) for a headline number.

    Selection-bias caveat: the harness reuses the SAME data to fit the CV bands and to pick
    the best-delta config (no nested CV, no multiplicity correction across the 4-6 candidates
    per type), so the winning config's reported band is conditional-on-having-won and is
    optimistically biased toward whichever config came out ahead — read it as edge-STABILITY
    across resamples, not as an unbiased estimate of the selected winner's true edge.
    """

    name: str
    full_data_log_score: float
    mean_log_score: float
    std_log_score: float
    delta_vs_median_mean: float
    delta_vs_median_std: float
    n_resamples: int


def _make_folds(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Partition indices [0, n) into ``k`` shuffled folds (sizes differ by at most 1)."""
    indices = rng.permutation(n)
    return [np.asarray(fold) for fold in np.array_split(indices, k)]


def iterated_kfold_cv(
    per_config_scores: dict[str, np.ndarray],
    *,
    baseline: str = MEDIAN_BASELINE,
    k: int = 5,
    iterations: int = 10,
) -> dict[str, ConfigCVResult]:
    """Iterated k-fold CV over per-question scores. Reports held-out fold means + paired deltas.

    ``per_config_scores[name]`` is the per-question primary-log-score array (same question
    order across configs). For each of ``iterations`` iterations we shuffle (seeded by the
    iteration index for reproducibility) and split into ``k`` folds; each fold's held-out
    questions give one resample. We collect, per config, the fold-mean log score and the
    fold-mean paired delta vs. the baseline. This is a variance-estimation / honesty
    mechanism — with this little data it shows whether an edge is stable, not which config
    to pick.

    Returns a per-config :class:`ConfigCVResult`.
    """
    names = list(per_config_scores.values())
    n = len(names[0]) if names else 0
    for arr in per_config_scores.values():
        if len(arr) != n:
            raise ValueError("all configs must have the same number of per-question scores")
    if baseline not in per_config_scores:
        raise ValueError(f"baseline {baseline!r} not among configs {list(per_config_scores)}")

    config_names = list(per_config_scores.keys())
    fold_means: dict[str, list[float]] = {name: [] for name in config_names}
    fold_deltas: dict[str, list[float]] = {name: [] for name in config_names}
    baseline_scores = per_config_scores[baseline]

    if n == 0:
        return {
            name: ConfigCVResult(name, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0)
            for name in config_names
        }

    effective_k = min(k, n)
    for it in range(iterations):
        rng = np.random.default_rng(it)
        folds = _make_folds(n, effective_k, rng)
        for fold in folds:
            if fold.size == 0:
                continue
            for name in config_names:
                fold_means[name].append(float(np.mean(per_config_scores[name][fold])))
                fold_deltas[name].append(float(np.mean(per_config_scores[name][fold] - baseline_scores[fold])))

    results: dict[str, ConfigCVResult] = {}
    for name in config_names:
        means = np.array(fold_means[name], dtype=float)
        deltas = np.array(fold_deltas[name], dtype=float)
        results[name] = ConfigCVResult(
            name=name,
            full_data_log_score=float(np.mean(per_config_scores[name])),
            mean_log_score=float(np.mean(means)),
            std_log_score=float(np.std(means)),
            delta_vs_median_mean=float(np.mean(deltas)),
            delta_vs_median_std=float(np.std(deltas)),
            n_resamples=int(means.size),
        )
    return results


def count_saturation_events(scores: np.ndarray) -> int:
    """Number of questions whose primary log score is a saturation blowup (< threshold)."""
    return int(np.sum(np.asarray(scores, dtype=float) < SATURATION_THRESHOLD))


# A config whose per-question scores barely vary is producing near-identical predictions on
# every question — i.e. it has collapsed to a (near-)uniform / constant distribution that
# ignores the data. Its "score" is then an artifact of the metric's baseline, not a real
# aggregation edge. We flag such configs so they aren't mistaken for winners. The threshold
# is in primary-log-score units; on the prod set a genuine numeric aggregator's per-question
# scores have std ~60-90, while the uniform-collapse tail-floor config has std ~1.1 (its
# scores cluster on the 2-3 boundary-baseline values). 5.0 separates the two regimes with a
# wide margin.
DEGENERATE_SCORE_STD: float = 5.0


def is_degenerate_config(scores: np.ndarray) -> bool:
    """True if a config's per-question scores barely vary (collapsed to a constant prediction).

    Catches the tail-floor failure mode where ``floor_eps`` is large enough relative to the
    201-point grid (floor_eps * 200 ~ 1.0) that flooring every step forces a uniform CDF: every
    question then scores the same small constant regardless of where it resolved, which can
    look like a "win" against a baseline that takes occasional blowups. That is not a real
    improvement and must not drive the re-bench decision.
    """
    arr = np.asarray(scores, dtype=float)
    return bool(arr.size >= 2 and np.std(arr) < DEGENERATE_SCORE_STD)
