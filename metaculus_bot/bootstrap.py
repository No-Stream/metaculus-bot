"""Bootstrap resampling primitives shared by the offline analysis surfaces.

One home for "resample a sample with replacement, B times, under a fixed seed", so the
ablation harness (``ablation/scoring.py``) and the clip-threshold sweep
(``performance_analysis/clip_threshold_sweep.py``) draw identical index matrices for the same
``(n, n_bootstrap, seed)`` and cannot drift on the statistic they share. Each caller keeps
its own return shape and small-sample rule on top; this module owns only the resampling.

The index matrices can be cached (``cache=True``) because the clip sweep calls this for every
``(window, c)`` cell over the same handful of sample sizes, and generating ``4000 x 451``
integers a few hundred times is the sweep's dominant cost. Caching is OFF by default: the
ablation harness derives a distinct seed per scoring group, so for it every call is a miss
that would be retained for the process lifetime (megabytes per group, zero reuse). The cache
is a SPEED measure only: reproducibility comes from re-seeding on every miss, and the key
carries every parameter the draw depends on, so changing the draw count or the seed can never
serve a stale matrix. Indices are stored
as ``int32`` (a sample is indexed by position, and no offline cohort approaches 2**31 rows),
which is value-preserving and a quarter of the ``int64`` numpy generates.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_INDEX_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def bootstrap_indices(n: int, *, n_bootstrap: int, seed: int, cache: bool = False) -> np.ndarray:
    """``(n_bootstrap, n)`` resample indices into a sample of size ``n``, seeded; cached on request.

    Generated as ``int64`` and downcast AFTER the draw rather than by passing ``dtype=`` to
    the generator, because the generator's bounded-integer stream is dtype-dependent and a
    cached matrix has to equal a freshly drawn ``int64`` one element for element.
    """
    key = (n, n_bootstrap, seed)
    if cache and key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    indices = np.random.default_rng(seed).integers(0, n, size=(n_bootstrap, n)).astype(np.int32)
    if cache:
        _INDEX_CACHE[key] = indices
    return indices


def bootstrap_means(
    values: Sequence[float] | np.ndarray, *, n_bootstrap: int, seed: int, cache: bool = False
) -> np.ndarray:
    """The ``n_bootstrap`` resampled means of ``values`` (non-empty), as a float array."""
    sample = np.asarray(values, dtype=float)
    return sample[bootstrap_indices(len(sample), n_bootstrap=n_bootstrap, seed=seed, cache=cache)].mean(axis=1)


def bootstrap_medians(
    values: Sequence[float] | np.ndarray, *, n_bootstrap: int, seed: int, cache: bool = False
) -> np.ndarray:
    """The ``n_bootstrap`` resampled medians of ``values`` (non-empty), as a float array."""
    sample = np.asarray(values, dtype=float)
    indices = bootstrap_indices(len(sample), n_bootstrap=n_bootstrap, seed=seed, cache=cache)
    return np.median(sample[indices], axis=1)
