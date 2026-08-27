"""Operator-facing output for an ablation run.

Two renderers, both terminal artifacts rather than pipeline state:

* ``_stage_qa_research_dump`` — the ``--qa-research`` markdown file: per-qid question,
  ground truth, raw research blob, leakage verdict, sanitized blob and redactions, read
  back off disk so a qid the screen dropped is still reviewable;
* ``_print_spend_report`` — the end-of-run block: API-call counts, cache hits, and the
  surviving-question breakdown by type.

Split out of ``ablation.cli`` because neither feeds a later stage: they read the working
set and the cache and emit text.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

from metaculus_bot.ablation.cache import AblationCache, atomic_write_text
from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS, FREE_PARSER_MODEL
from metaculus_bot.ablation.leakage_screen import DEFAULT_DETECTOR_MODEL
from metaculus_bot.ablation.run_stacker import ARM_MEDIAN, ARM_PDF, ARM_STACK, ARM_STACK_AUG
from metaculus_bot.ablation.run_state import SpendReport, WorkingSet


def _stage_qa_research_dump(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
) -> Path:
    """Dump first N qids' question, ground truth, research blob, leakage verdict."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    target_path = cache.root / f"qa_research_{timestamp}.md"

    qids_in_order = sorted(working.questions.keys())
    selected = qids_in_order[: max(5, args.num_binary + args.num_multiple_choice + args.num_numeric)]

    lines: list[str] = ["# Ablation QA Research Dump", "", f"Generated: {timestamp}", ""]
    for qid in selected:
        gt = working.ground_truths[qid]
        # Read from disk rather than ``working.research_blobs``: the screen
        # stage pops leaked qids from in-memory state to keep them out of
        # downstream forecast/stack stages, but the QA dump exists to let the
        # operator review what the screener flagged. The blob still lives on
        # disk via ``cache.write_research``, so go back to that source.
        cached_research = cache.read_research(qid)
        if cached_research is None:
            # Research stage skipped or never ran for this qid — record it
            # explicitly rather than emit a sentinel string under a
            # normal-looking section.
            lines.append(f"## Q{qid} (skipped — no research blob)")
            lines.append(f"- URL: {gt.page_url}")
            lines.append(f"- Question text: {gt.question_text}")
            lines.append("")
            continue
        research = cached_research[0]
        # The QA dump runs even on partial-pipeline failures so the operator
        # can review whatever artifacts DO exist. A qid may have research
        # cached but no verdict (e.g., prune stage failed for everyone, screen
        # never ran). Surface that as a "no verdict" section rather than
        # crashing — the goal is operator-readable diagnostic output.
        verdict = working.leakage_verdicts.get(qid)

        lines.append(f"## Q{qid}")
        lines.append(f"- URL: {gt.page_url}")
        lines.append(f"- Question text: {gt.question_text}")
        lines.append(f"- Ground truth: {gt.resolution_string}")
        if verdict is None:
            lines.append("- Leaked: (no verdict — screen stage did not run for this qid)")
            lines.append("")
        else:
            lines.append(f"- Leaked: {verdict['is_leaked']}")
            lines.append("")
            lines.append("### Detector verdict")
            lines.append(str(verdict["detector_response"]))
            lines.append("")
        lines.append("### Raw research blob (truncated to 4000 chars)")
        lines.append("```")
        lines.append(research[:4000])
        lines.append("```")
        lines.append("")

        # Surface the redactor's output and metadata so the operator can review
        # what was pruned and verify the redactor's judgment. The pruned blob
        # lives at research_pruned/<qid>.md when the prune stage ran; a missing
        # entry simply means the prune stage didn't process this qid.
        cached_pruned = cache.read_pruned_research(qid)
        if cached_pruned is not None:
            sanitized_blob, prune_meta = cached_pruned
            n_redactions = len(prune_meta.get("redactions", []))
            lines.append(
                f"### Sanitized blob ({prune_meta.get('original_chars', 0)} -> "
                f"{prune_meta.get('sanitized_chars', 0)} chars, {n_redactions} redactions)"
            )
            lines.append("```")
            lines.append(sanitized_blob[:4000])
            lines.append("```")
            lines.append("")
            if prune_meta.get("redactions"):
                lines.append("### Redactions")
                for redaction in prune_meta["redactions"]:
                    excerpt = redaction.get("original_excerpt", "")
                    reason = redaction.get("reason", "")
                    lines.append(f"- `{excerpt}` — {reason}")
                lines.append("")

    atomic_write_text(target_path, "\n".join(lines))
    return target_path


def _print_spend_report(spend: SpendReport, working: WorkingSet, summary_path: Path | None) -> None:
    n_total = len(working.questions)
    # Verdict dicts have a fixed schema (leakage_screen._build_verdict);
    # ``is_leaked`` is always present. Direct subscript surfaces drift.
    n_leaked = sum(1 for v in working.leakage_verdicts.values() if v["is_leaked"])
    # n_clean = qids that reached forecasters (all upstream gates passed: prune,
    # screen, qa_iterate). Filter by leakage verdict because hydration on resume
    # loads all on-disk pruned blobs regardless of whether the screen later
    # marked them leaked. Without this filter, resume invocations report
    # n_dropped_other as a negative count.
    clean_qids = {
        qid for qid in working.research_blobs if not working.leakage_verdicts.get(qid, {}).get("is_leaked", False)
    }
    n_clean = len(clean_qids)
    n_dropped_other = n_total - n_clean - n_leaked

    by_type = {"binary": 0, "multiple_choice": 0, "numeric": 0}
    # ``research_blobs`` and ``questions`` are kept in lockstep by the
    # orchestrator — every qid in research_blobs is in questions. Direct
    # subscript surfaces invariant violations as KeyError.
    for qid in clean_qids:
        question = working.questions[qid]
        if isinstance(question, BinaryQuestion):
            by_type["binary"] += 1
        elif isinstance(question, MultipleChoiceQuestion):
            by_type["multiple_choice"] += 1
        elif isinstance(question, NumericQuestion):
            by_type["numeric"] += 1

    detector_short = DEFAULT_DETECTOR_MODEL.rsplit("/", maxsplit=1)[-1].replace(":free", "")
    parser_short = FREE_PARSER_MODEL.rsplit("/", maxsplit=1)[-1].replace(":free", "")
    n_forecasters = len(FREE_FORECASTER_MODELS)

    border = "=" * 60
    print(border)
    print("ABLATION RUN COMPLETE")
    print(border)
    print("Spend report:")
    print(
        f"  Gemini search        primary: {spend.gemini_research_calls} calls    "
        f"gap-fill: {spend.gemini_gap_fill_calls} calls"
    )
    print(
        f"  Redactor             {spend.redactor_invocations} claude -p invocations "
        f"({spend.prune_validation_failures} validation failures)"
    )
    print(f"  Leakage detector     {spend.leakage_detector_calls} LLM calls (free model: {detector_short})")
    print(f"  Forecasters          {spend.forecaster_llm_calls} LLM calls (free models, {n_forecasters} per question)")
    print(f"  Stacker (stack)      {spend.stacker_llm_calls_stack} calls ({spend.fallback_stacker_stack} fallback)")
    print(
        f"  Stacker (stack_aug)        {spend.stacker_llm_calls_stack_aug} calls ({spend.fallback_stacker_stack_aug} fallback)"
    )
    print(f"  Parser               {spend.parser_llm_calls} calls (free model: {parser_short})")
    print(
        f"  Cache hits           research={spend.cached_research_hits}  "
        f"prune={spend.cached_prune_hits}  "
        f"screen={spend.cached_screen_hits}  forecast={spend.cached_forecaster_hits}  "
        f"stack={spend.cached_stacker_hits.get(ARM_STACK, 0)}  "
        f"stack_aug={spend.cached_stacker_hits.get(ARM_STACK_AUG, 0)}  "
        f"pdf={spend.cached_stacker_hits.get(ARM_PDF, 0)}  "
        f"median={spend.cached_stacker_hits.get(ARM_MEDIAN, 0)}"
    )
    print()
    print(
        f"Results: {n_clean} questions in working set "
        f"({n_leaked} leaked, {n_dropped_other} other drops "
        "(prune/qa_iterate))"
    )
    print(f"  Binary:  {by_type['binary']} questions")
    print(f"  MC:      {by_type['multiple_choice']} questions")
    print(f"  Numeric: {by_type['numeric']} questions")
    if summary_path is not None:
        print()
        print(f"Summary written to: {summary_path}")
        print()
        print(f"Ready for sign-off — please review summary at {summary_path} before expanding the run.")
    print(border)
