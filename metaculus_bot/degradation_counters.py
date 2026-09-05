"""Run-level degradation accounting: what reddens CI, and the lines that say why.

``TemplateForecaster``, its research orchestrator, and its aggregation pipeline own
their respective counters. The forecaster constructs a fresh ``DegradationSnapshot``
at each read. This module
owns the immutable read boundary and the two things done WITH it: the alertable sum
``cli.py`` turns into an exit status, and the two end-of-run summary lines an
operator greps first.

The formatters are pure and return the message; the caller logs it. That keeps the
run-summary lines emitting from ``metaculus_bot.forecaster`` — the logger
``cli.py`` raises to DEBUG and the telemetry archive harvests — and makes the
exact text assertable without a caplog round-trip. The text is a parsed contract:
``scripts/telemetry/markers.py`` reads both lines, so renaming a key here breaks
harvesting (see the 2026-07-26 ``research_provider_timeouts`` rename).
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DegradationSnapshot:
    """Immutable values read from one point in a bot run.

    Conditional-stacking values travel with the snapshot so the summary formatters
    have no dependency on the bot adapter. They are intentionally excluded from
    ``alertable_total``: normal conditional-stacking decisions do not redden CI.
    """

    forecasters_dropped: int
    questions_failed_to_publish: int
    stacker_primary_failed: int
    stacker_fallback_used: int
    stacker_fallback_failed: int
    research_provider_failures: int
    summarizer_failures: int
    gap_fill_v2_errors: int
    prediction_market_degraded: int
    prediction_market_source_losses: int
    provider_degradation: int
    publish_attempt_failures: int
    publish_skipped_closed: int
    time_budget_fast_path: int
    research_budget_cuts: int
    conditional_stacking_triggered: int
    conditional_stacking_skipped: int
    conditional_stacking_skipped_single_forecaster: int
    conditional_stacking_crux_failures: int
    conditional_stacking_search_failures: int


def alertable_total(snapshot: DegradationSnapshot) -> int:
    """Sum of counters whose non-zero value should page us.

    Consumed by ``cli.py`` to decide whether to ``sys.exit(1)`` after all
    publications complete. Any individual non-zero counter is enough to trip the
    alert; the sum is just a convenient single number.

    The conditional-stacking counters are deliberately absent: a triggered or
    skipped stacker is normal operation, not degradation.
    """
    return (
        snapshot.forecasters_dropped
        + snapshot.questions_failed_to_publish
        + snapshot.stacker_primary_failed
        + snapshot.stacker_fallback_used
        + snapshot.stacker_fallback_failed
        + snapshot.research_provider_failures
        + snapshot.summarizer_failures
        + snapshot.gap_fill_v2_errors
        + snapshot.prediction_market_degraded
        + snapshot.prediction_market_source_losses
        + snapshot.provider_degradation
        + snapshot.publish_attempt_failures
        + snapshot.publish_skipped_closed
        + snapshot.time_budget_fast_path
        + snapshot.research_budget_cuts
    )


def format_degradation_summary(snapshot: DegradationSnapshot) -> str:
    """The loud end-of-run degradation line.

    Any non-zero counter here means something got dropped, the stacker fell back,
    or a research provider failed — all states where CI (``cli.py``) should exit
    non-zero so we get paged, but every publishable question has already been
    published. Emitted unconditionally, so a clean run states its zeros rather
    than implying them by an absent line.

    The tail keys (``provider_degradation``, then ``publish_attempt_failures``,
    then ``publish_skipped_closed``, then ``time_budget_fast_path``, then
    ``research_budget_cuts``) stay LAST and in that order: the telemetry parser
    wraps each in an optional trailing group so archived records that predate any
    of them still harvest their other counters on a replace-by-run re-harvest.
    Appending a key here without extending that regex breaks the whole line's
    harvest, because the pattern is ``$``-anchored.

    ``research_budget_cuts`` is the off-fast-path complement of
    ``time_budget_fast_path``: a question whose window cleared the fast-path
    threshold but whose research WINDOW still cut a provider or a gap-fill pass
    (deduplicated per question, orchestrator-side). Without it that band's
    degradation was invisible to the counter contract and the all-clear census.

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
        f"Degradation counters: forecasters_dropped={snapshot.forecasters_dropped}, "
        f"questions_failed_to_publish={snapshot.questions_failed_to_publish}, "
        f"stacker_primary_failed={snapshot.stacker_primary_failed}, "
        f"stacker_fallback_used={snapshot.stacker_fallback_used}, "
        f"stacker_fallback_failed={snapshot.stacker_fallback_failed}, "
        f"research_provider_failures={snapshot.research_provider_failures}, "
        f"summarizer_failures={snapshot.summarizer_failures}, "
        f"gap_fill_v2_errors={snapshot.gap_fill_v2_errors}, "
        f"prediction_market_degraded={snapshot.prediction_market_degraded}, "
        f"prediction_market_source_losses={snapshot.prediction_market_source_losses}, "
        f"provider_degradation={snapshot.provider_degradation}, "
        f"publish_attempt_failures={snapshot.publish_attempt_failures}, "
        f"publish_skipped_closed={snapshot.publish_skipped_closed}, "
        f"time_budget_fast_path={snapshot.time_budget_fast_path}, "
        f"research_budget_cuts={snapshot.research_budget_cuts}"
    )


def format_conditional_stacking_summary(snapshot: DegradationSnapshot) -> str:
    """The per-run conditional-stacking tally.

    ``skipped_single_forecaster`` is its own key rather than folded into
    ``skipped``: the single-survivor short-circuit returns above both other
    increment sites, and when it was uncounted a run logged "SKIPPED: single
    forecaster survived" per question and then ``skipped=0`` at the end.
    """
    return (
        f"Conditional stacking summary: triggered={snapshot.conditional_stacking_triggered}, "
        f"skipped={snapshot.conditional_stacking_skipped}, "
        f"skipped_single_forecaster={snapshot.conditional_stacking_skipped_single_forecaster}, "
        f"crux_failures={snapshot.conditional_stacking_crux_failures}, "
        f"search_failures={snapshot.conditional_stacking_search_failures}"
    )
