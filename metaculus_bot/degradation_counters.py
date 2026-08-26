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
        + bot._publish_attempt_failures
        + bot._publish_skipped_closed_count
        + bot._time_budget_fast_path_count
    )


def format_degradation_summary(bot: "TemplateForecaster") -> str:
    """The loud end-of-run degradation line.

    Any non-zero counter here means something got dropped, the stacker fell back,
    or a research provider failed — all states where CI (``cli.py``) should exit
    non-zero so we get paged, but every publishable question has already been
    published. Emitted unconditionally, so a clean run states its zeros rather
    than implying them by an absent line.

    The tail keys (``provider_degradation``, then ``publish_attempt_failures``,
    then ``publish_skipped_closed``, then ``time_budget_fast_path``) stay LAST and
    in that order: the telemetry parser wraps each in an optional trailing group so
    archived records that predate any of them still harvest their other counters on
    a replace-by-run re-harvest. Appending a key here without extending that regex
    breaks the whole line's harvest, because the pattern is ``$``-anchored.

    The three publish-side counters mean three different things, which is why
    they are three keys. ``questions_failed_to_publish`` counts questions the
    min-forecasters floor kept from ATTEMPTING publication;
    ``publish_attempt_failures`` counts attempted POSTs that exhausted the
    publish-hardening retry budget (the q45085 405 shape the old counter could
    not see); ``publish_skipped_closed`` counts questions whose publish the
    close-time gate skipped before any POST, i.e. latency cost us the question.
    The oldest key's name is misleading but stays — renaming it would silently
    drop the field from every historical record the parser replays.

    ``time_budget_fast_path`` is the fourth member of that family and the earliest
    of them: it counts questions whose close time was too near for the full
    pipeline's worst case, so the optional research stages were dropped to protect
    the prediction POST. The three above fire once a publish has already failed or
    been withheld; this one fires while the question is still savable, which is why
    it is worth alerting on separately.
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
        f"provider_degradation={bot._provider_degradation_count}, "
        f"publish_attempt_failures={bot._publish_attempt_failures}, "
        f"publish_skipped_closed={bot._publish_skipped_closed_count}, "
        f"time_budget_fast_path={bot._time_budget_fast_path_count}"
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
