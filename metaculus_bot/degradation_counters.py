"""Run-level degradation accounting: what reddens CI, and the lines that say why.

``TemplateForecaster`` owns the counters themselves (some as plain ints it bumps,
some as pass-through properties onto the research orchestrator and the aggregation
pipeline). This module owns the two things done WITH them: the alertable sum
``cli.py`` turns into an exit status, and the two end-of-run summary lines an
operator greps first.

The formatters are pure and return the message; the caller logs it. That keeps the
run-summary lines emitting from ``metaculus_bot.forecaster`` — the logger
``cli.py`` raises to DEBUG and the telemetry archive harvests — and makes the
exact text assertable without a caplog round-trip. The text is a parsed contract:
``scripts/telemetry/markers.py`` reads both lines, so renaming a key here breaks
harvesting (see the 2026-07-26 ``research_provider_timeouts`` rename).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metaculus_bot.forecaster import TemplateForecaster


def alertable_total(bot: "TemplateForecaster") -> int:
    """Sum of counters whose non-zero value should page us.

    Consumed by ``cli.py`` to decide whether to ``sys.exit(1)`` after all
    publications complete. Any individual non-zero counter is enough to trip the
    alert; the sum is just a convenient single number.

    The conditional-stacking counters are deliberately absent: a triggered or
    skipped stacker is normal operation, not degradation.
    """
    return (
        bot._forecasters_dropped_count
        + bot._questions_failed_to_publish
        + bot._stacker_primary_failed_count
        + bot._stacker_fallback_used_count
        + bot._stacker_fallback_failed_count
        + bot._research_provider_failure_count
        + bot._summarizer_failure_count
        + bot._gap_fill_v2_error_count
        + bot._prediction_market_degraded_count
        + bot._prediction_market_source_loss_count
        + bot._provider_degradation_count
    )


def format_degradation_summary(bot: "TemplateForecaster") -> str:
    """The loud end-of-run degradation line.

    Any non-zero counter here means something got dropped, the stacker fell back,
    or a research provider failed — all states where CI (``cli.py``) should exit
    non-zero so we get paged, but every publishable question has already been
    published. Emitted unconditionally, so a clean run states its zeros rather
    than implying them by an absent line.

    ``provider_degradation`` stays LAST: the telemetry parser wraps it in an
    optional trailing group so the ~290 archived records that predate it still
    harvest their other ten counters on a replace-by-run re-harvest.
    """
    return (
        f"Degradation counters: forecasters_dropped={bot._forecasters_dropped_count}, "
        f"questions_failed_to_publish={bot._questions_failed_to_publish}, "
        f"stacker_primary_failed={bot._stacker_primary_failed_count}, "
        f"stacker_fallback_used={bot._stacker_fallback_used_count}, "
        f"stacker_fallback_failed={bot._stacker_fallback_failed_count}, "
        f"research_provider_failures={bot._research_provider_failure_count}, "
        f"summarizer_failures={bot._summarizer_failure_count}, "
        f"gap_fill_v2_errors={bot._gap_fill_v2_error_count}, "
        f"prediction_market_degraded={bot._prediction_market_degraded_count}, "
        f"prediction_market_source_losses={bot._prediction_market_source_loss_count}, "
        f"provider_degradation={bot._provider_degradation_count}"
    )


def format_conditional_stacking_summary(bot: "TemplateForecaster") -> str:
    """The per-run conditional-stacking tally.

    ``skipped_single_forecaster`` is its own key rather than folded into
    ``skipped``: the single-survivor short-circuit returns above both other
    increment sites, and when it was uncounted a run logged "SKIPPED: single
    forecaster survived" per question and then ``skipped=0`` at the end.
    """
    return (
        f"Conditional stacking summary: triggered={bot._conditional_stacking_triggered_count}, "
        f"skipped={bot._conditional_stacking_skipped_count}, "
        f"skipped_single_forecaster={bot._conditional_stacking_skipped_single_forecaster_count}, "
        f"crux_failures={bot._conditional_stacking_crux_failures}, "
        f"search_failures={bot._conditional_stacking_search_failures}"
    )
