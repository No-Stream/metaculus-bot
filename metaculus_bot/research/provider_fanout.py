"""Deadline-bounded fan-out of the selected research providers.

Split out of ``orchestrator.py``: the orchestrator decides WHICH providers run and
what to do with their output; this module owns only the concurrency and the outer
time bound — start every provider at once, cancel whoever is still going when the
research window closes, and hand back exactly one result per provider in selection
order. ``_empty_provider`` lives here for the same reason: it is the stub the
fan-out runs when selection produced nothing.

The per-provider wrapper (``_run_one``) deliberately stays in the orchestrator: it
is where AskNews summarization, the fallback ladder and the per-provider detail
registry meet, and its failure logging is pinned to the orchestrator's logger.
``ResearchOrchestrator`` mixes in ``ProviderFanout``, so
``ResearchOrchestrator._await_providers_within_deadline`` keeps working for the
callers (and tests) that reach it through the class.
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine

from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.research.provider_diagnostics import ProviderResult
from metaculus_bot.research.providers import ResearchCallable
from metaculus_bot.time_budget import QuestionTimeBudget

logger = logging.getLogger(__name__)


async def _empty_provider(_: MetaculusQuestion) -> str:
    """Stand-in provider for a selection that produced nothing to run."""
    return ""


class ProviderFanout:
    """Mixin: runs the selected providers concurrently under the research-phase deadline."""

    @staticmethod
    async def _await_providers_within_deadline(
        providers: list[tuple[ResearchCallable, str]],
        run_one: Callable[[ResearchCallable, str], Coroutine[object, object, tuple[str, ProviderResult]]],
        time_budget: QuestionTimeBudget | None,
    ) -> list[tuple[str, ProviderResult]]:
        """Run every provider concurrently, cancelling any still running at the deadline.

        Replaces a bare ``asyncio.gather``, which had no outer bound at all: each
        provider carries its own wall timeout, but the PHASE carried none, so one
        provider whose internal timeout failed to fire could hold the question past
        its close with nothing to stop it. Stragglers are cancelled and recorded as
        ``status="deadline"``, so the partial bundle is used rather than lost and the
        cut providers are named in the diagnostics block and the research archive.

        A cancelled provider does NOT bump ``provider_failure_count``: it did not
        fail, we stopped it. The budget decision is alertable instead through
        ``research_budget_cut_count`` (bumped by the caller when any result comes
        back ``status="deadline"`` off the fast path; fast-path questions are
        already counted by the forecaster's ``time_budget_fast_path`` counter).
        Keeping the two apart is what lets ``research_provider_failures`` keep
        meaning "a provider broke".

        With no budget (every caller outside the per-question pipeline) the wait is
        unbounded and behavior is identical to the old gather.
        """
        if not providers:
            # ``asyncio.gather()`` returned [] on an empty list; ``asyncio.wait(set())``
            # raises. Selection always yields at least the "none" stub, so this is the
            # direct-call path only.
            return []

        tasks = [asyncio.create_task(run_one(provider, name), name=f"research:{name}") for provider, name in providers]
        task_name = {task: name for task, (_, name) in zip(tasks, providers, strict=True)}

        deadline_s = time_budget.research_phase_deadline_s() if time_budget is not None else None
        _done, pending = await asyncio.wait(tasks, timeout=deadline_s, return_when=asyncio.ALL_COMPLETED)
        for task in pending:
            task.cancel()
        if pending:
            # Let the cancellations land so no "task was destroyed but it is pending"
            # warning escapes into the run log.
            await asyncio.wait(pending, timeout=2.0)
            logger.warning(
                "RESEARCH_PHASE_DEADLINE: cancelled %d/%d providers after %.0fs (%s)",
                len(pending),
                len(tasks),
                deadline_s or 0.0,
                ",".join(sorted(task_name.get(task, "unknown") for task in pending)),
            )

        # Provider order is the section order in the research bundle and the row order
        # in the diagnostics block, so rebuild it from `providers` rather than from
        # asyncio.wait's unordered sets.
        results: list[tuple[str, ProviderResult]] = []
        for task, (_, name) in zip(tasks, providers, strict=True):
            if task in pending:
                # Latency IS the deadline for a cancelled provider: every task starts
                # at phase start, so one still running when the deadline lands ran for
                # exactly that long.
                results.append(
                    (
                        "",
                        ProviderResult(
                            name=name,
                            status="deadline",
                            chars=0,
                            latency_ms=round((deadline_s or 0.0) * 1000),
                        ),
                    )
                )
                continue
            exc = task.exception()
            if exc is not None:
                # _run_one converts every provider exception into a ProviderResult, so
                # reaching here means the wrapper itself broke — a bug, not a provider
                # failure, and it must not be swallowed into a fake result.
                raise exc
            results.append(task.result())
        return results
