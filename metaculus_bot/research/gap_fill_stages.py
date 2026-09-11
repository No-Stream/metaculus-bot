"""Both gap-fill passes, wired into whatever the research phase has left to spend.

Split out of ``orchestrator.py``: v1 (targeted search) and v2 (the agentic loop) are
the research phase's largest OPTIONAL cost and its most failure-prone stage, so their
independent failure guards, budget arithmetic and error accounting belong together,
away from provider selection and bundle assembly.

``run_gap_fill_passes`` RETURNS its accounting in a ``GapFillOutcome`` rather than
bumping counters, because the counters (``gap_fill_v1_error_count``,
``gap_fill_v2_error_count``, ``research_budget_cut_count``) live on the orchestrator, which is what the forecaster
and the end-of-run degradation line read. The two gap-fill modules stay behind
function-level imports inside the failure guards, so an import error in either one
degrades the question instead of killing the run.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    GAP_FILL_ENABLED_ENV,
    GAP_FILL_MIN_RESEARCH_CHARS,
    GAP_FILL_V2_ENABLED_ENV,
    env_flag_enabled,
)
from metaculus_bot.prompts import GAP_FILL_V1_SECTION_HEADER
from metaculus_bot.time_budget import QuestionTimeBudget

if TYPE_CHECKING:
    from metaculus_bot.research.agentic.types import GhostContext, GhostForecast

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GapFillOutcome:
    """One question's gap-fill result plus the accounting its caller owns.

    ``v1_errors``, ``v2_errors`` and ``budget_cut`` are returned rather than counted here because
    the counters they feed live on the orchestrator. ``budget_cut`` is a single
    boolean, not a count: the orchestrator's own bookkeeping dedupes per question, so
    a question losing v1 AND v2 to the deadline is one degradation, and collapsing the
    three cut sites into one flag is what makes that impossible to double-count.
    """

    research: str
    v2_payload: dict | None
    v1_errors: int
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
) -> tuple[str, bool, int]:
    """Return ``(addendum, budget_cut, errors)``; addendum is ``""`` when inactive, cut, or failed.

    Its own failure guard (v2 has a separate one) so a v1 defect can never zero
    v2's findings, and vice versa.
    """
    if not active:
        return "", False, 0
    errors = 0

    def _count_error(_exc: BaseException) -> None:
        nonlocal errors
        errors += 1

    try:
        from metaculus_bot.research.targeted import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # import stays inside failure guard
            run_gap_fill_pass,
        )

        # Bounded by the research phase's remainder, so a pass that overruns its own deadlines cannot spend the forecast's time.
        addendum = await asyncio.wait_for(
            run_gap_fill_pass(
                question,
                research,
                is_benchmarking=is_benchmarking,
                on_error=_count_error,
            ),
            timeout=_remaining_research_phase_s(time_budget),
        )
        return addendum, False, errors
    except TimeoutError:
        # A deliberate budget cut, not a failure: its own branch keeps it out of the "stage failed" traceback.
        logger.warning(
            "GAP_FILL_V1_CUT_FOR_BUDGET: question=%s; research phase ran out of budget",
            getattr(question, "id_of_question", None),
        )
        return "", True, errors
    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except — gap-fill is optional; a failure (import error, unhandled raise) must never kill the forecast
        logger.exception("Gap-fill v1 stage failed; proceeding without it")
        return "", False, errors + 1


async def _run_gap_fill_v2(
    question: MetaculusQuestion,
    research: str,
    *,
    active: bool,
    is_benchmarking: bool,
    time_budget: QuestionTimeBudget | None,
    archive_sink: Callable[[dict], None],
    ghost_context_sink: Callable[[GhostContext], None],
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

        # The research phase's remainder, on top of v2's own wall deadline (which has never bound in prod).
        findings = await asyncio.wait_for(
            run_gap_fill_v2(
                question,
                research,
                is_benchmarking=is_benchmarking,
                archive_sink=archive_sink,
                ghost_context_sink=ghost_context_sink,
                on_error=_count_error,
            ),
            timeout=_remaining_research_phase_s(time_budget),
        )
        return findings, False, errors
    except TimeoutError:
        # A budget cut, not a crash: it protects the prediction POST and is alertable via research_budget_cut_count.
        logger.warning(
            "GAP_FILL_V2_CUT_FOR_BUDGET: question=%s; research phase ran out of budget",
            getattr(question, "id_of_question", None),
        )
        return "", True, errors
    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except — gap-fill is optional; a failure (import error, unhandled raise) must never kill the forecast
        logger.exception("Gap-fill v2 stage failed; proceeding without it")
        # Path (c): an import failure or an escape past the seam's soft-fail, where no payload and no on_error fires.
        return "", False, errors + 1


async def _run_ghost_v1(
    question: MetaculusQuestion,
    context: GhostContext,
    addendum: str,
    *,
    time_budget: QuestionTimeBudget | None,
) -> GhostForecast | None:
    """The v1 ghost once both passes have landed: telemetry only, bounded by the phase's remainder, never raises."""
    remaining_s = _remaining_research_phase_s(time_budget)
    if remaining_s is not None and remaining_s <= 0.0:
        return None
    try:
        from metaculus_bot.research.agentic_gap_fill import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # import stays inside failure guard
            run_gap_fill_v2_ghost_v1,
        )

        return await asyncio.wait_for(run_gap_fill_v2_ghost_v1(question, context, addendum), timeout=remaining_s)
    except TimeoutError:
        logger.warning(
            "Gap-fill v1 ghost cut for budget: question=%s; research phase ran out of budget",
            getattr(question, "id_of_question", None),
        )
        return None
    except (
        Exception
    ):  # HARNESS-SCAN-EXEMPT-broad-except — telemetry only; a failed v1 ghost must never cost the forecast
        logger.exception("Gap-fill v1 ghost failed; proceeding without it")
        return None


async def _stamp_v1_ghost(
    question: MetaculusQuestion,
    payload: dict | None,
    ghost_context: GhostContext | None,
    addendum: str,
    *,
    time_budget: QuestionTimeBudget | None,
) -> None:
    """Run the v1 ghost when both halves of its pair exist, and write it (or None) onto the archive payload."""
    ghost_v1: GhostForecast | None = None
    if addendum and ghost_context is not None:
        ghost_v1 = await _run_ghost_v1(question, ghost_context, addendum, time_budget=time_budget)
    if payload is not None:
        payload["ghost_v1"] = ghost_v1.model_dump() if ghost_v1 is not None else None


async def run_gap_fill_passes(
    question: MetaculusQuestion,
    research: str,
    *,
    fast_path: bool,
    is_benchmarking: bool,
    time_budget: QuestionTimeBudget | None,
) -> GapFillOutcome:
    """Append both gap-fill passes' sections to ``research``; return it plus the accounting.

    v1 and v2 both consume the pre-gap-fill bundle and run CONCURRENTLY in one gather, so the
    research phase's wall clock is max(v1, v2), not the sum, and the v2 driver's brief never
    sees v1's addendum; v2's section appends after v1's. The v1 ghost, one more driver call on
    the loop's cached prefix with v1's section added, therefore runs here after the gather and
    only when both sections exist (docs/agentic_gap_fill.md "The ghost forecast"). Both passes
    are OPTIONAL and the research phase's largest optional cost (v1's configured worst case is
    555s, v2 measures 84s at p50), which is why the fast path drops both.
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
        return GapFillOutcome(research=research, v2_payload=None, v1_errors=0, v2_errors=0, budget_cut=budget_cut)

    gap_fill_v2_payload: dict | None = None
    ghost_context: GhostContext | None = None

    def _capture_gap_fill_v2(payload: dict) -> None:
        nonlocal gap_fill_v2_payload
        gap_fill_v2_payload = payload

    def _capture_ghost_context(context: GhostContext) -> None:
        nonlocal ghost_context
        ghost_context = context

    (addendum, v1_cut, v1_errors), (v2_findings, v2_cut, v2_errors) = await asyncio.gather(
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
            ghost_context_sink=_capture_ghost_context,
        ),
    )
    budget_cut = budget_cut or v1_cut or v2_cut
    # Path (a): the loop's soft-fail leaves telemetry["error"] on the payload; checked here so it cannot double-count.
    if gap_fill_v2_payload is not None:
        v2_telemetry = gap_fill_v2_payload.get("telemetry")
        if isinstance(v2_telemetry, dict) and v2_telemetry.get("error") is not None:
            v2_errors += 1
    await _stamp_v1_ghost(question, gap_fill_v2_payload, ghost_context, addendum, time_budget=time_budget)
    if addendum:
        research = f"{research}\n\n---\n\n{GAP_FILL_V1_SECTION_HEADER}\n\n{addendum}"
    if v2_findings:
        # v2_findings carries its own "## Agentic Research Findings" header (render_findings), distinct from v1's.
        research = f"{research}\n\n---\n\n{v2_findings}"
    return GapFillOutcome(
        research=research,
        v2_payload=gap_fill_v2_payload,
        v1_errors=v1_errors,
        v2_errors=v2_errors,
        budget_cut=budget_cut,
    )
