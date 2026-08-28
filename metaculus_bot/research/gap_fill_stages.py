"""Both gap-fill passes, wired into whatever the research phase has left to spend.

Split out of ``orchestrator.py``: v1 (targeted search) and v2 (the agentic loop) are
the research phase's largest OPTIONAL cost and its most failure-prone stage, so their
independent failure guards, budget arithmetic and error accounting belong together,
away from provider selection and bundle assembly.

``run_gap_fill_passes`` RETURNS its accounting in a ``GapFillOutcome`` rather than
bumping counters, because the counters (``gap_fill_v2_error_count``,
``research_budget_cut_count``) live on the orchestrator, which is what the forecaster
and the end-of-run degradation line read. The two gap-fill modules stay behind
function-level imports inside the failure guards, so an import error in either one
degrades the question instead of killing the run.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    GAP_FILL_ENABLED_ENV,
    GAP_FILL_MIN_RESEARCH_CHARS,
    GAP_FILL_V2_ENABLED_ENV,
    env_flag_enabled,
)
from metaculus_bot.time_budget import QuestionTimeBudget

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GapFillOutcome:
    """One question's gap-fill result plus the accounting its caller owns.

    ``v2_errors`` and ``budget_cut`` are returned rather than counted here because
    both counters they feed live on the orchestrator. ``budget_cut`` is a single
    boolean, not a count: the orchestrator's own bookkeeping dedupes per question, so
    a question losing v1 AND v2 to the deadline is one degradation, and collapsing the
    three cut sites into one flag is what makes that impossible to double-count.
    """

    research: str
    v2_payload: dict | None
    v2_errors: int
    budget_cut: bool


def _remaining_research_phase_s(time_budget: QuestionTimeBudget | None) -> float | None:
    """Seconds the research phase may still spend, or None for unbounded.

    ``None`` is both "this caller has no budget" and the value ``asyncio.wait`` /
    ``asyncio.wait_for`` already take to mean "no timeout", so it passes straight
    through to them without a sentinel translation.
    """
    if time_budget is None:
        return None
    return time_budget.research_phase_deadline_s()


async def _run_gap_fill_v1(
    question: MetaculusQuestion,
    research: str,
    *,
    active: bool,
    is_benchmarking: bool,
    time_budget: QuestionTimeBudget | None,
) -> tuple[str, bool]:
    """Return ``(addendum, budget_cut)``; the addendum is ``""`` when inactive, cut, or failed.

    Its own failure guard (v2 has a separate one) so a v1 defect can never zero
    v2's findings, and vice versa.
    """
    if not active:
        return "", False
    try:
        from metaculus_bot.research.targeted import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # import stays inside failure guard
            run_gap_fill_pass,
        )

        # Bounded by whatever the research phase has left, so a pass
        # that overruns its own internal deadlines still cannot spend
        # the forecast's time.
        addendum = await asyncio.wait_for(
            run_gap_fill_pass(question, research, is_benchmarking=is_benchmarking),
            timeout=_remaining_research_phase_s(time_budget),
        )
        return addendum, False
    except TimeoutError:
        # Its own branch (like v2's below) because it is not a failure:
        # falling into the generic except would log a traceback under
        # "stage failed" for a deliberate budget cut.
        logger.warning(
            "GAP_FILL_V1_CUT_FOR_BUDGET: question=%s; research phase ran out of budget",
            getattr(question, "id_of_question", None),
        )
        return "", True
    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except — gap-fill is optional; a failure (import error, unhandled raise) must never kill the forecast
        logger.exception("Gap-fill v1 stage failed; proceeding without it")
        return "", False


async def _run_gap_fill_v2(
    question: MetaculusQuestion,
    research: str,
    *,
    active: bool,
    is_benchmarking: bool,
    time_budget: QuestionTimeBudget | None,
    archive_sink: Callable[[dict], None],
) -> tuple[str, bool, int]:
    """Return ``(findings, budget_cut, errors)``; findings is ``""`` when inactive, cut, or failed.

    ``errors`` counts GENUINE v2 crashes only, never an idle "found nothing" run and
    never a deadline hit. Three mutually-exclusive crash paths, one bump each (no
    double-count): (a) the loop-internal soft-fail — detected post-gather by the
    caller via the archive payload's telemetry["error"]; (b) this seam's
    construction-error soft-fail — via run_gap_fill_v2's on_error callback, counted
    into the local below; (c) the import/escape error caught by the generic except.
    (a) and (b) are exclusive because (b)'s error means the loop never ran (no
    payload), and (c) is exclusive of both because the seam swallows all Exception, so
    nothing escapes it once construction succeeds.
    """
    if not active:
        return "", False, 0
    errors = 0

    def _count_error(_exc: BaseException) -> None:
        nonlocal errors
        errors += 1

    try:
        from metaculus_bot.research.agentic_gap_fill import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # import stays inside failure guard
            run_gap_fill_v2,
        )

        # Same research-phase bound as v1 above, on top of v2's own
        # GAP_FILL_V2_WALL_DEADLINE (which measures as never binding:
        # 0 of 103 triple-era records report deadline_hit).
        findings = await asyncio.wait_for(
            run_gap_fill_v2(
                question,
                research,
                is_benchmarking=is_benchmarking,
                archive_sink=archive_sink,
                on_error=_count_error,
            ),
            timeout=_remaining_research_phase_s(time_budget),
        )
        return findings, False, errors
    except TimeoutError:
        # NOT a v2 crash: we cut it to protect the prediction POST, so
        # this must not add to ``errors`` (which exists to redden CI on a
        # dead v2 feature) — the budget decision is alertable via
        # research_budget_cut_count (fast-path questions never reach here;
        # gap-fill is skipped upstream for them).
        logger.warning(
            "GAP_FILL_V2_CUT_FOR_BUDGET: question=%s; research phase ran out of budget",
            getattr(question, "id_of_question", None),
        )
        return "", True, errors
    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except — gap-fill is optional; a failure (import error, unhandled raise) must never kill the forecast
        logger.exception("Gap-fill v2 stage failed; proceeding without it")
        # Path (c): an import failure (or any escape past the seam's own
        # soft-fail) is a crash — count it here since no payload/on_error
        # fires on this path.
        return "", False, errors + 1


async def run_gap_fill_passes(
    question: MetaculusQuestion,
    research: str,
    *,
    fast_path: bool,
    is_benchmarking: bool,
    time_budget: QuestionTimeBudget | None,
) -> GapFillOutcome:
    """Append both gap-fill passes' sections to ``research``; return it plus the accounting.

    Gap-fill v1 and v2 both consume the pre-gap-fill bundle and run
    CONCURRENTLY in one gather (plan doc §2: research-phase wall-clock
    is max(v1, v2), not the sum — v2's GAP_FILL_V2_WALL_DEADLINE fits
    inside v1's worst-case envelope only under this parallelism).
    Consequence: the v2 driver's brief sees the bundle WITHOUT v1's
    addendum. v2's section appends after v1's.

    Both are OPTIONAL, and they are the research phase's largest optional
    cost: v1's configured worst case is 555s (analyzer 135 + resolver wave
    420) and v2 measures 84s at p50 / 293s at its observed max. So the
    fast path drops both — that is where the time for a thin window comes
    from, far more than provider selection.
    """
    gap_fill_budget_s = _remaining_research_phase_s(time_budget)
    skip_optional_gap_fill = fast_path or (gap_fill_budget_s is not None and gap_fill_budget_s <= 0.0)
    gap_fill_v1_active = (
        env_flag_enabled(GAP_FILL_ENABLED_ENV)
        and not skip_optional_gap_fill
        and len(research.strip()) >= GAP_FILL_MIN_RESEARCH_CHARS
    )
    gap_fill_v2_active = env_flag_enabled(GAP_FILL_V2_ENABLED_ENV) and not skip_optional_gap_fill
    budget_cut = False
    if skip_optional_gap_fill and (env_flag_enabled(GAP_FILL_ENABLED_ENV) or env_flag_enabled(GAP_FILL_V2_ENABLED_ENV)):
        logger.warning(
            "GAP_FILL_SKIPPED_FOR_BUDGET: question=%s fast_path=%s research_phase_remaining=%s",
            getattr(question, "id_of_question", None),
            str(fast_path).lower(),
            "n/a" if gap_fill_budget_s is None else f"{gap_fill_budget_s:.0f}s",
        )
        budget_cut = True
    if not (gap_fill_v1_active or gap_fill_v2_active):
        return GapFillOutcome(research=research, v2_payload=None, v2_errors=0, budget_cut=budget_cut)

    gap_fill_v2_payload: dict | None = None

    def _capture_gap_fill_v2(payload: dict) -> None:
        nonlocal gap_fill_v2_payload
        gap_fill_v2_payload = payload

    (addendum, v1_cut), (v2_findings, v2_cut, v2_errors) = await asyncio.gather(
        _run_gap_fill_v1(
            question, research, active=gap_fill_v1_active, is_benchmarking=is_benchmarking, time_budget=time_budget
        ),
        _run_gap_fill_v2(
            question,
            research,
            active=gap_fill_v2_active,
            is_benchmarking=is_benchmarking,
            time_budget=time_budget,
            archive_sink=_capture_gap_fill_v2,
        ),
    )
    budget_cut = budget_cut or v1_cut or v2_cut
    # Path (a): the loop ran but hit its catch-all soft-fail. The
    # loop swallows the crash and returns findings normally, so the
    # only crash signal is the stamped telemetry["error"] on the
    # archive payload. Checked here (not in _run_gap_fill_v2) so it can't
    # double-count with the on_error/except paths there — those
    # produce no payload with a non-None telemetry error.
    if gap_fill_v2_payload is not None:
        v2_telemetry = gap_fill_v2_payload.get("telemetry")
        if isinstance(v2_telemetry, dict) and v2_telemetry.get("error") is not None:
            v2_errors += 1
    if addendum:
        research = f"{research}\n\n---\n\n## Targeted Gap-Fill (second pass)\n\n{addendum}"
    if v2_findings:
        # v2_findings carries its own "## Agentic Research Findings"
        # header (render_findings) — distinct from v1's section.
        research = f"{research}\n\n---\n\n{v2_findings}"
    return GapFillOutcome(research=research, v2_payload=gap_fill_v2_payload, v2_errors=v2_errors, budget_cut=budget_cut)
