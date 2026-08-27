"""Run-wide counters for which community-baseline scoring path each question took.

A leaf by construction: this module imports nothing from ``metaculus_bot``, so both
halves of the community-baseline scorer can depend on it without depending on each
other. ``scoring_patches`` owns the score formulas and the monkey-patch installers;
``scoring_extraction`` parses community forecasts off a question's ``api_json``. The
counters used to live in ``scoring_patches``, which forced the parsers to reach back
into it from inside four function bodies — a real import cycle that only worked
because those imports were deferred to call time.

``COUNTERS`` is a single mutable instance held by every importer, so
``reset_scoring_path_stats`` clears its fields in place and never rebinds the name.
"""

import logging
from dataclasses import dataclass, fields
from typing import Literal

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Which breakdown bucket an MC community-data miss belongs to. Each member maps to the
# ``mc_missing_<member>`` field below by name, which is what keeps ``record_mc_missing``
# free of a second reason -> counter table that could drift from the fields themselves.
McMissingKind = Literal["api_json", "question_node", "aggregations", "prob_yes_per_category"]


# slots=True so a mistyped field name raises instead of silently creating a new attribute
# that nothing reports.
@dataclass(slots=True)
class ScoringPathCounters:
    """One benchmark run's tally of scoring attempts, successes and community-data misses."""

    numeric_pmf_attempts: int = 0
    numeric_pmf_successes: int = 0
    numeric_fallback_attempts: int = 0
    numeric_fallback_successes: int = 0
    mc_attempts: int = 0
    mc_successes: int = 0
    mc_missing_community: int = 0
    mc_missing_api_json: int = 0
    mc_missing_question_node: int = 0
    mc_missing_aggregations: int = 0
    mc_missing_prob_yes_per_category: int = 0


COUNTERS = ScoringPathCounters()


def record_mc_missing(kind: McMissingKind | None = None) -> None:
    """Count one MC question whose community probabilities could not be read.

    ``mc_missing_community`` is the rollup the run log reports as the MC miss rate, so
    every miss bumps it; ``kind`` additionally names the breakdown bucket when the caller
    could attribute the miss to one. Both bumps live here rather than being hand-repeated at
    every failure site in ``scoring_extraction``, so a new failure path cannot land counted
    in the breakdown but absent from the rollup (the same reason
    ``record_donated_key_fallback`` owns the key-fallback counters).
    """
    if kind is not None:
        counter_name = f"mc_missing_{kind}"
        setattr(COUNTERS, counter_name, getattr(COUNTERS, counter_name) + 1)
    COUNTERS.mc_missing_community += 1


def reset_scoring_path_stats() -> None:
    """Zero every counter in place. Importers hold ``COUNTERS`` itself, so never rebind it."""
    for field in fields(COUNTERS):
        setattr(COUNTERS, field.name, 0)


def get_scoring_path_stats() -> dict[str, float | int]:
    total_numeric = COUNTERS.numeric_pmf_attempts + COUNTERS.numeric_fallback_attempts
    total_mc = COUNTERS.mc_attempts
    return {
        "numeric_pmf_attempts": COUNTERS.numeric_pmf_attempts,
        "numeric_pmf_successes": COUNTERS.numeric_pmf_successes,
        "numeric_fallback_attempts": COUNTERS.numeric_fallback_attempts,
        "numeric_fallback_successes": COUNTERS.numeric_fallback_successes,
        "numeric_total": total_numeric,
        "numeric_fallback_rate": ((COUNTERS.numeric_fallback_attempts / total_numeric) if total_numeric > 0 else 0.0),
        "mc_attempts": total_mc,
        "mc_successes": COUNTERS.mc_successes,
        "mc_missing_community": COUNTERS.mc_missing_community,
        "mc_missing_rate": ((COUNTERS.mc_missing_community / total_mc) if total_mc > 0 else 0.0),
        # MC breakdown
        "mc_missing_api_json": COUNTERS.mc_missing_api_json,
        "mc_missing_question_node": COUNTERS.mc_missing_question_node,
        "mc_missing_aggregations": COUNTERS.mc_missing_aggregations,
        "mc_missing_prob_yes_per_category": COUNTERS.mc_missing_prob_yes_per_category,
    }


def log_scoring_path_stats() -> None:
    stats = get_scoring_path_stats()
    logger.info("=== SCORING PATH SUMMARY ===")
    logger.info(
        "Numeric: pmf_attempts=%d pmf_successes=%d fallback_attempts=%d fallback_successes=%d total=%d fallback_rate=%.2f",
        stats["numeric_pmf_attempts"],
        stats["numeric_pmf_successes"],
        stats["numeric_fallback_attempts"],
        stats["numeric_fallback_successes"],
        stats["numeric_total"],
        stats["numeric_fallback_rate"],
    )
    logger.info(
        "MC: attempts=%d successes=%d missing_community=%d missing_rate=%.2f",
        stats["mc_attempts"],
        stats["mc_successes"],
        stats["mc_missing_community"],
        stats["mc_missing_rate"],
    )
    logger.info(
        "MC missing breakdown: api_json=%d question_node=%d aggregations=%d prob_yes_per_category=%d",
        stats["mc_missing_api_json"],
        stats["mc_missing_question_node"],
        stats["mc_missing_aggregations"],
        stats["mc_missing_prob_yes_per_category"],
    )

    # Bright warnings when fallbacks dominate
    if stats["numeric_total"] and stats["numeric_fallback_rate"] >= 0.8:
        logger.warning(
            "⚠️  ALERT: Numeric scoring fallback used for %.0f%% of items. Check that model predictions expose CDFs.",
            100 * stats["numeric_fallback_rate"],
        )
    logger.info("=== END SCORING SUMMARY ===")
