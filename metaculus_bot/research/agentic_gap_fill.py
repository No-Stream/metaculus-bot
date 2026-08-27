"""Gap-fill v2 seam — wires the bounded agentic research loop into orchestration.

Keeps ``ResearchOrchestrator`` thin: this module owns prompt/tool/config
construction and the soft-fail boundary, and the orchestrator just gathers the
returned section. Contract mirrors v1 (``research/targeted.py``
``run_gap_fill_pass``): never raises, returns ``""`` when disabled,
benchmarking, unsupported question type, or on any failure.

Archive persistence: pass ``archive_sink`` to receive the loop transcript +
telemetry when the loop actually ran (mirrors the orchestrator's
``research_sink`` callback pattern) — the findings string alone is not enough
for the research-archive trace requirement.
"""

import dataclasses
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    GAP_FILL_V2_CONCLUDE_THRESHOLD,
    GAP_FILL_V2_DRIVER_EFFORT,
    GAP_FILL_V2_DRIVER_MODEL,
    GAP_FILL_V2_ENABLED_ENV,
    GAP_FILL_V2_MAX_GAPS,
    GAP_FILL_V2_MAX_TOOL_CALLS,
    GAP_FILL_V2_WALL_DEADLINE,
    env_flag_enabled,
)
from metaculus_bot.research.agentic import LoopConfig, build_gap_fill_tools, run_agentic_loop
from metaculus_bot.research.agentic.driver_prompt import (
    SupportedQuestion as _SupportedQuestion,
)
from metaculus_bot.research.agentic.driver_prompt import (
    build_ghost_prompt,
    build_system_prompt,
    build_user_brief,
)

__all__ = ["run_gap_fill_v2"]

logger = logging.getLogger(__name__)

# Broad by design, same policy and rationale as v1's
# _GAP_FILL_SOFT_FAIL_EXCEPTIONS (research/targeted.py): gap-fill is an
# optional enrichment layer, and a forecast with only first-pass research is
# strictly better than no forecast. run_agentic_loop already soft-fails
# internally; this seam is belt-and-suspenders for prompt/tool construction
# errors. CancelledError propagates (BaseException).
_GAP_FILL_V2_SOFT_FAIL_EXCEPTIONS: tuple[type[BaseException], ...] = (Exception,)


async def run_gap_fill_v2(
    question: MetaculusQuestion,
    bundle_markdown: str,
    *,
    is_benchmarking: bool,
    archive_sink: Callable[[dict[str, Any]], None] | None = None,
    on_error: Callable[[BaseException], None] | None = None,
) -> str:
    """Run the agentic gap-fill v2 loop and return its findings section.

    Returns ``""`` (and makes zero LLM calls) when the ``GAP_FILL_V2_ENABLED``
    flag is off, when benchmarking, or for question types the dry-run scaffold
    has no template for. Soft-fails to ``""`` on any error. When the loop ran,
    ``archive_sink`` (if given) receives
    ``{"transcript": ..., "telemetry": ..., "ghost": ...}`` for research-archive
    persistence — including empty-findings runs, whose telemetry is still worth
    keeping. ``ghost`` is the serialized ghost forecast (or ``None`` when the
    ghost phase did not run or failed).

    ``on_error`` (if given) is called with the exception when this seam's
    prompt/tool-construction step CRASHES — the belt-and-suspenders soft-fail
    below. It fires ONLY on that construction-error path, which produces no
    marker and no archive payload, so it is the only crash signal the caller can
    observe for it (the loop-internal soft-fail is instead observable via the
    archive payload's ``telemetry["error"]``, and an import failure never reaches
    here). It is NOT called on the legitimate flag-off / benchmarking /
    unsupported-qtype early returns — those are skips, not crashes. Mirrors the
    ``archive_sink`` callback pattern so the orchestrator stays thin.
    """
    if not env_flag_enabled(GAP_FILL_V2_ENABLED_ENV):
        return ""
    if is_benchmarking:
        # Same leakage rationale as the prediction-market provider: live search
        # sees post-resolution information on a large fraction of resolved
        # questions, so v2 is hard-off in benchmarking runs.
        return ""
    if not isinstance(question, _SupportedQuestion):
        # The dry-run scaffold embeds the panel's per-qtype template; question
        # types without one (e.g. date questions) skip v2 entirely.
        logger.info(
            "Gap-fill v2 skipped: unsupported question type %s",
            type(question).__name__,
        )
        return ""
    try:
        # UTC so the driver's "today" agrees with the forecaster prompt bundle's
        # "Today:" line (``prompts._forecasting_window_str`` normalizes to UTC),
        # regardless of host timezone. Prod runs on UTC hosts, so unchanged there.
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        system_prompt = build_system_prompt(today)
        user_brief = build_user_brief(question, bundle_markdown)
        tools = build_gap_fill_tools(question.question_text)
        config = LoopConfig(
            model=GAP_FILL_V2_DRIVER_MODEL,
            reasoning_effort=GAP_FILL_V2_DRIVER_EFFORT,
            max_tool_calls=GAP_FILL_V2_MAX_TOOL_CALLS,
            wall_deadline_s=GAP_FILL_V2_WALL_DEADLINE,
            conclude_threshold_s=GAP_FILL_V2_CONCLUDE_THRESHOLD,
            max_gaps=GAP_FILL_V2_MAX_GAPS,
        )
        question_ref = question.page_url or str(question.id_of_question)
        result = await run_agentic_loop(
            system_prompt,
            user_brief,
            tools,
            config,
            ghost_prompt=build_ghost_prompt(),
            log_prefix=f"question={question_ref} ",
        )
        if archive_sink is not None:
            archive_sink(
                {
                    "transcript": result.transcript,
                    "telemetry": dataclasses.asdict(result.telemetry),
                    "ghost": result.ghost.model_dump() if result.ghost is not None else None,
                }
            )
        return result.findings_markdown
    except _GAP_FILL_V2_SOFT_FAIL_EXCEPTIONS as exc:
        logger.exception("Gap-fill v2 seam failed; continuing without v2 findings")
        if on_error is not None:
            on_error(exc)
        return ""
