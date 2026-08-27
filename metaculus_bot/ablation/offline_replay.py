"""Offline replay + iterated-k-fold CV harness for rigorous-PDF aggregation (W5).

Replays the cached per-forecaster predictions for an ablation backtest through the
W1-W3 aggregation primitives ENTIRELY OFFLINE — zero API / network calls — and scores
each candidate aggregation config against real Metaculus resolutions on the tournament
metric (Metaculus-style log score, including saturation blowups). The output is a
per-type comparison of each config vs. the median baseline, under iterated k-fold
cross-validation so we can see whether an edge is stable or evaporates across resamples.

The whole point is correctness: a bug here produces a wrong conclusion about which
aggregation strategy wins. So the data load reuses the EXACT same survivor filter,
deserializer, question shim, and ground-truth deserializer that the live ablation arms
use, and scoring goes straight through the pure ``scoring_common`` primitives.

This module is the harness entry point: it owns the zero-API guard and re-exports the
pieces, which live in three siblings — ``ablation.replay_dataset`` (record shapes +
cache hydration), ``ablation.replay_configs`` (the candidate aggregation arms, over
``ablation.weighted_quantiles`` for the coherence-weighted combine), and
``ablation.replay_scoring`` (scoring, iterated k-fold CV, degeneracy diagnostics).

Zero-API guarantee
------------------
The only inputs are on-disk cache files (forecaster outputs + qids manifest) read via
``ablation.cache.AblationCache``. Nothing here *calls* the forecaster, a research/LLM provider, or
``main.py``: we consume cached predictions + ground truth and run pure aggregation math.

Note on import vs. call: the question-shim + ground-truth helpers the loader reuses live in
``ablation.manifest_serde``, which pulls in nothing heavier than forecasting-tools, but the
prediction deserializer lives in ``ablation.forecasters``, which transitively *imports*
``metaculus_bot.forecaster`` at module load time. Importing a module is not a network
call — instantiating a forecaster and calling ``.forecast()`` would be. So the load-bearing
enforcement is :func:`no_network` — a context manager that monkeypatches ``socket`` so any
outbound connection during replay raises immediately, making a live call impossible by
construction. Run the whole replay inside ``with no_network():`` and a stray provider call
crashes instead of silently spending credits.

Candidate configs (what we are comparing), per type
---------------------------------------------------
* BINARY: ``median_baseline`` (median of per-forecaster probs — the incumbent), fixed-w
  logit shrinkage ``pool_binary(median(p_model), median(p_math), w)`` for w in {0, .1, .25,
  .5}, and a divergence-gated ``adaptive_weight`` config. ``p_math`` per forecaster is
  reconstructed from its structured block via ``reconstruct_p_math``. Also reports the
  overconfidence measurement |logit(median p_math) - logit(median p_model)| per question.
* MC: ``median_baseline`` (per-option median, renormalized — current behavior), geometric
  ``pool_mc``, and ``pool_mc`` + Dirichlet smoothing at a couple of concentration values.
* NUMERIC: ``median_baseline`` (vertical CDF-median — the incumbent), ``mean_baseline``
  (vertical mean), ``vincentize(mean)``, ``vincentize(median)``, ``log_pool``, plus a small
  tail-floor sweep wrapping the vertical-mean baseline.
"""

from __future__ import annotations

import logging
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

from metaculus_bot.ablation.replay_configs import (
    ADAPTIVE_MAX_WEIGHT,  # noqa: F401  # re-export: candidate-arm knob read off this module
    ADAPTIVE_SLOPE,  # noqa: F401  # re-export: candidate-arm knob read off this module
    ADAPTIVE_THRESHOLD,  # noqa: F401  # re-export: candidate-arm knob read off this module
    BINARY_CLAMP_BOUNDS,  # noqa: F401  # re-export: candidate-arm knob read off this module
    BINARY_SHRINKAGE_WEIGHTS,  # noqa: F401  # re-export: candidate-arm knob read off this module
    COHERENCE_SOFTWEIGHT,  # noqa: F401  # re-export: weighted-aggregation tests import it from here
    MC_DIRICHLET_CONCENTRATIONS,  # noqa: F401  # re-export: candidate-arm knob read off this module
    MEDIAN_BASELINE,  # noqa: F401  # re-export: replay + weighted-aggregation tests import it from here
    NUMERIC_TAIL_FLOORS,  # noqa: F401  # re-export: candidate-arm knob read off this module
    BinaryConfig,  # noqa: F401  # re-export: config type alias for harness call sites
    MCConfig,  # noqa: F401  # re-export: config type alias for harness call sites
    NumericConfig,  # noqa: F401  # re-export: config type alias for harness call sites
    WeightLookup,  # noqa: F401  # re-export: coherence-weight type alias for harness call sites
    binary_overconfidence,  # noqa: F401  # re-export: replay tests import it from here
    build_binary_configs,  # noqa: F401  # re-export: replay tests + coherence harness import it from here
    build_mc_configs,  # noqa: F401  # re-export: replay tests + coherence harness import it from here
    build_numeric_configs,  # noqa: F401  # re-export: replay tests + coherence harness import it from here
)
from metaculus_bot.ablation.replay_dataset import (
    BinaryRecord,  # noqa: F401  # re-export: replay + weighted-aggregation tests import it from here
    MCRecord,  # noqa: F401  # re-export: replay + weighted-aggregation tests import it from here
    NumericRecord,  # noqa: F401  # re-export: replay + weighted-aggregation tests import it from here
    ReplayDataset,  # noqa: F401  # re-export: replay tests import it from here
    _mc_correct_index,  # noqa: F401  # re-export: replay tests import it from here
    _resolution_to_float,  # noqa: F401  # re-export: replay tests import it from here
    load_replay_dataset,  # noqa: F401  # re-export: replay tests + coherence harness import it from here
)
from metaculus_bot.ablation.replay_scoring import (
    DEGENERATE_SCORE_STD,  # noqa: F401  # re-export: degeneracy threshold read off this module
    SATURATION_THRESHOLD,  # noqa: F401  # re-export: replay tests import it from here
    ConfigCVResult,  # noqa: F401  # re-export: replay tests import it from here
    count_saturation_events,  # noqa: F401  # re-export: replay tests import it from here
    is_degenerate_config,  # noqa: F401  # re-export: replay tests import it from here
    iterated_kfold_cv,  # noqa: F401  # re-export: replay tests import it from here
    score_all_binary,  # noqa: F401  # re-export: replay tests + coherence harness import it from here
    score_all_mc,  # noqa: F401  # re-export: replay tests + coherence harness import it from here
    score_all_numeric,  # noqa: F401  # re-export: replay tests + coherence harness import it from here
    score_binary,  # noqa: F401  # re-export: replay tests import it from here
    score_mc,  # noqa: F401  # re-export: replay tests import it from here
    score_numeric,  # noqa: F401  # re-export: replay tests import it from here
)
from metaculus_bot.ablation.weighted_quantiles import (
    weighted_cdf_median,  # noqa: F401  # re-export: weighted-aggregation tests + coherence harness import it here
    weighted_quantile,  # noqa: F401  # re-export: weighted-aggregation tests + coherence harness import it here
)

logger: logging.Logger = logging.getLogger(__name__)

QuestionType = Literal["binary", "multiple_choice", "numeric"]


# Zero-API guard


class NetworkAccessDuringReplayError(RuntimeError):
    """Raised when the offline replay attempts any outbound network connection."""


@contextmanager
def no_network() -> Iterator[None]:
    """Block outbound network for the duration of the block (real zero-API enforcement).

    Monkeypatches ``socket.getaddrinfo`` — the universal DNS-resolution chokepoint every
    outbound HTTP/LLM/provider call passes through to resolve a hostname — to raise
    :class:`NetworkAccessDuringReplayError`. Run the whole replay under this so a stray
    forecaster / research / provider call crashes loudly instead of silently spending API
    credits. Resolving ``localhost`` / a literal IP is allowed so local tooling still works.

    This is the load-bearing guarantee — far stronger than an import-graph check, because the
    helper modules we reuse legitimately *import* the forecaster at load time without ever
    *calling* it.
    """
    real_getaddrinfo = socket.getaddrinfo

    def _blocked_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host in (None, "localhost", "127.0.0.1", "::1"):
            return real_getaddrinfo(host, *args, **kwargs)
        raise NetworkAccessDuringReplayError(
            f"offline replay attempted to resolve host {host!r}; this must be zero-API"
        )

    # setattr (not direct assignment) so the type checker doesn't flag the deliberate
    # module-attribute swap as an incompatible reassignment of the stdlib signature.
    socket.getaddrinfo = _blocked_getaddrinfo
    try:
        yield
    finally:
        socket.getaddrinfo = real_getaddrinfo
