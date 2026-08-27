"""CLI orchestrator for the probabilistic-tools ablation benchmark.

Wave-4 integration layer. Pulls together every Wave 1-3 module under one
argparse-driven entry point. The behavioral spec lives in
``scratch_docs_and_planning/`` (Atlas-inspired ablation plan). Highlights:

* Stage-by-stage pipeline: fetch → research → screen → forecast → stack → pdf → median → score.
* Per-stage disk cache: every artifact lives under ``backtests/ablation/`` and a
  re-run is a no-op for cached qids.
* Smoke→expand semantics: re-running with larger ``--num-*`` counts fetches the
  delta only; the manifest is append-extendable.
* ``--qa-research`` halts after the screen stage, dumps a markdown QA file.
* ``--stages`` selects a subset (default all); ``--force-stages`` re-runs the
  listed stages bypassing their cache reads.
* Each stage reads cache where prerequisites permit; missing prerequisites for
  ``--stages score`` exit non-zero with a clear message.

Module-level imports are deliberate: tests monkeypatch the wave-1/2/3 entry
points at ``metaculus_bot.ablation.cli.<name>``. Keep the imports below stable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from forecasting_tools import BinaryQuestion, GeneralLlm, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport
from forecasting_tools.data_models.numeric_report import NumericReport
from forecasting_tools.data_models.questions import OutOfBoundsResolution

from metaculus_bot.ablation.cache import AblationCache, atomic_write_text, model_slug_to_filename
from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS, FREE_PARSER_MODEL
from metaculus_bot.ablation.forecasters import (
    deserialize_prediction_value,
    run_forecasters_batch,
)
from metaculus_bot.ablation.leakage_screen import (
    _EMPTY_BLOB_RESPONSE,
    DEFAULT_DETECTOR_MODEL,
    _research_blob_sha,
    screen_batch,
)
from metaculus_bot.ablation.prune import DEFAULT_BATCH_SIZE as PRUNE_DEFAULT_BATCH_SIZE
from metaculus_bot.ablation.prune import run_prune_for_qids
from metaculus_bot.ablation.qa_iterate import (
    DEFAULT_FORECASTABILITY_THRESHOLD as QA_ITERATE_DEFAULT_FORECASTABILITY_THRESHOLD,
)
from metaculus_bot.ablation.qa_iterate import (
    DEFAULT_LEAKAGE_THRESHOLD as QA_ITERATE_DEFAULT_LEAKAGE_THRESHOLD,
)
from metaculus_bot.ablation.qa_iterate import (
    DEFAULT_MAX_ITERATIONS as QA_ITERATE_DEFAULT_MAX_ITERATIONS,
)
from metaculus_bot.ablation.qa_iterate import (
    IterateOutcome,
    read_manual_rejects,
    render_qa_summary,
    run_qa_iterate_batch,
    serialize_outcome,
    write_manual_rejects,
)
from metaculus_bot.ablation.research import run_gemini_research_for_qids
from metaculus_bot.ablation.run_pdf import (
    ARM_PDF_MIN1,
    ARM_PDF_MIN2,
    run_pdf_for_qid,
)

# Symbols used in _stage_stack for the mean arm and pdf mean variant.
# Function-scoped imports would survive formatter stripping, but these are
# used in multiple places across the orchestrator so top-level is cleaner.
from metaculus_bot.ablation.run_simple_agg import run_mean_for_qid, run_median_for_qid
from metaculus_bot.ablation.run_stacker import (
    ABLATION_MIN_FORECASTERS,
    ARM_MEAN,
    ARM_MEDIAN,
    ARM_PDF,
    ARM_STACK,
    ARM_STACK_AUG,
    run_stacker_batch,
)
from metaculus_bot.ablation.scoring import (
    PairedScore,
    PairedStats,
    aggregate_paired,
    score_arm_for_qid,
)
from metaculus_bot.ablation.scoring_report import render_summary_markdown
from metaculus_bot.aiohttp_cleanup import enable_aiohttp_session_autoclose
from metaculus_bot.api_preflight import verify_metaculus_api_identity
from metaculus_bot.backtest.question_prep import (
    BacktestQuestionSet,
    fetch_resolved_questions_stratified,
)
from metaculus_bot.backtest.scoring import GroundTruth
from metaculus_bot.question_types import question_type_of

logger: logging.Logger = logging.getLogger(__name__)

STAGES: list[str] = [
    "fetch",
    "research",
    "prune",
    "screen",
    "qa_iterate",
    "forecast",
    "stack",
    "stack_aug",
    "pdf",
    "median",
    "mean",
    "score",
]
DEFAULT_TOURNAMENTS: list[str] = ["spring-aib-2026"]
DEFAULT_RESOLVED_AFTER: str = "2026-01-01"
DEFAULT_CACHE_DIR: str = "backtests/ablation"

# Static cascade map: forcing a stage on the left invalidates the caches of
# every stage on the right (transitive closure already pre-computed). Without
# this, --force-stages forecast leaves stale stacker payloads on disk derived
# from the OLD forecaster outputs, and the next score run quietly compares
# fresh-vs-stale arms. See cli_audit_20260515.md (C1) for the operator footgun.
_FORCE_CASCADES: dict[str, set[str]] = {
    "fetch": set(),
    "research": {"prune", "screen", "qa_iterate", "forecast", "stack", "stack_aug", "pdf", "median", "mean"},
    "prune": {"screen", "qa_iterate", "forecast", "stack", "stack_aug", "pdf", "median", "mean"},
    "screen": {"qa_iterate"},
    "qa_iterate": set(),
    "forecast": {"stack", "stack_aug", "pdf", "median", "mean"},
    "stack": set(),
    "stack_aug": set(),
    "pdf": set(),
    "median": set(),
    "mean": set(),
    "score": set(),
}


def _expand_forced_stages(forced: set[str]) -> set[str]:
    """Apply the static cascade map to ``forced`` and return the expansion."""
    expanded = set(forced)
    for stage in forced:
        expanded.update(_FORCE_CASCADES.get(stage, set()))
    return expanded


# ---------------------------------------------------------------------------
# Spend tracking
# ---------------------------------------------------------------------------


@dataclass
class SpendReport:
    """Tracks API-call counts and cache hits across a single CLI run.

    Counters are populated by each stage by snapshotting cache state before the
    stage runs and diffing against post-stage cache state. The orchestrator
    (``run_ablation``) constructs one ``SpendReport`` per CLI run and threads
    it through every stage; ``_print_spend_report`` renders it at the end.
    """

    gemini_research_calls: int = 0
    gemini_gap_fill_calls: int = 0
    leakage_detector_calls: int = 0
    forecaster_llm_calls: int = 0
    stacker_llm_calls_stack: int = 0
    stacker_llm_calls_stack_aug: int = 0
    parser_llm_calls: int = 0
    redactor_invocations: int = 0
    cached_research_hits: int = 0
    cached_prune_hits: int = 0
    cached_screen_hits: int = 0
    cached_forecaster_hits: int = 0
    cached_stacker_hits: dict[str, int] = field(default_factory=dict)
    fallback_stacker_stack: int = 0
    fallback_stacker_stack_aug: int = 0
    prune_validation_failures: int = 0


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


# Rate-limit dial mapping. Each preset trades off wall-clock speed vs. tolerance
# for upstream-provider 429s. ``fast`` is the historical behavior; ``gentle`` is
# the new default and pairs lower per-forecaster parallelism with a longer retry
# budget so a single 429 wave doesn't shed forecasters from the lineup. ``slow``
# serializes per-forecaster runs entirely — useful on medium runs where wall-
# clock matters less than completing every cell. ``patient`` keeps ``slow``'s
# concurrency=1 but bumps the retry budget to 8 — "slow but persistent" for
# free-tier providers (qwen, minimax, gemma-4-26b) that frequently shed
# forecasters under tight retry budgets even though successive attempts often
# succeed.
_RATE_LIMIT_MODES: tuple[str, ...] = ("fast", "gentle", "slow", "patient")
_RATE_LIMIT_MODE_TO_KWARGS: dict[str, dict[str, int]] = {
    "fast": {"per_forecaster_concurrency": 4, "max_retries": 1},
    "gentle": {"per_forecaster_concurrency": 2, "max_retries": 3},
    "slow": {"per_forecaster_concurrency": 1, "max_retries": 5},
    "patient": {"per_forecaster_concurrency": 1, "max_retries": 8},
}


def _rate_limit_mode_kwargs(mode: str) -> dict[str, int]:
    """Return the (per_forecaster_concurrency, max_retries) kwargs for a mode."""
    return dict(_RATE_LIMIT_MODE_TO_KWARGS[mode])


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_strings(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_stages_arg(raw: str) -> list[str]:
    parsed = _parse_csv_strings(raw)
    invalid = [s for s in parsed if s not in STAGES]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid stage(s): {invalid}; valid stages: {STAGES}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ablation",
        description="Probabilistic-tools ablation benchmark — A/B test of PROBABILISTIC_TOOLS_ENABLED.",
    )
    parser.add_argument("--num-binary", type=int, default=0, help="Target binary question count.")
    parser.add_argument("--num-multiple-choice", type=int, default=0, help="Target MC question count.")
    parser.add_argument("--num-numeric", type=int, default=0, help="Target numeric question count.")
    parser.add_argument(
        "--qids",
        type=_parse_csv_ints,
        default=None,
        help=(
            "Comma-separated explicit qid list; bypasses fetching. When combined with "
            "--stages subsets that omit fetch, the working set is filtered to these qids "
            "after manifest hydration so downstream stages run only over the requested qids."
        ),
    )
    parser.add_argument(
        "--tournaments",
        type=_parse_csv_strings,
        default=DEFAULT_TOURNAMENTS,
        help="Comma-separated tournament slugs; default: spring-aib-2026.",
    )
    parser.add_argument(
        "--resolved-after",
        type=str,
        default=DEFAULT_RESOLVED_AFTER,
        help="ISO date YYYY-MM-DD; lower bound on actual_resolution_time.",
    )
    parser.add_argument(
        "--resolved-before",
        type=str,
        default=None,
        help="ISO date YYYY-MM-DD; optional upper bound on actual_resolution_time.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Disk cache root; default: {DEFAULT_CACHE_DIR}.",
    )
    parser.add_argument(
        "--stages",
        type=_parse_stages_arg,
        default=list(STAGES),
        help=f"Comma-separated subset of {STAGES}; default: all stages.",
    )
    parser.add_argument(
        "--qa-research",
        action="store_true",
        help="Halt after the screen stage; dump first 5 qids' research+verdict to a QA markdown file.",
    )
    parser.add_argument(
        "--force-stages",
        type=_parse_stages_arg,
        default=[],
        help=(
            "Comma-separated stages to re-run (bypass cache reads). Other stages still read cache. "
            "Forcing a stage AUTO-CASCADES to every downstream stage whose inputs change: "
            "research → prune,screen,qa_iterate,forecast,stack,stack_aug,pdf,median; "
            "prune → screen,qa_iterate,forecast,stack,stack_aug,pdf,median; "
            "screen → qa_iterate; forecast → stack,pdf. "
            "Without the cascade, downstream caches would silently serve stale outputs "
            "derived from the prior upstream artifact."
        ),
    )
    parser.add_argument(
        "--per-question-sleep",
        type=int,
        default=30,
        help=(
            "Seconds to sleep BETWEEN STAGES, AFTER each API-firing stage "
            "(research, prune, screen, qa_iterate, forecast, stack, pdf). "
            "Total pause for a full pipeline = 7 × value. Despite the name this is "
            "per-stage, not per-question: a 30-question run with --per-question-sleep=30 "
            "pauses ~210s total (7 stages × 30s), not 900s. Set to 0 to disable. "
            "Increase to back off OpenRouter rate limits. "
            "TODO: real per-question pacing would require restructuring run_forecasters_batch's "
            "asyncio.gather into a serial loop (or a per-question post-release sleep); "
            "documented but deliberately not shipped here."
        ),
    )
    parser.add_argument(
        "--gap-fill-max-gaps",
        type=int,
        default=3,
        help="Maximum number of gap-fill searches per question; default: 3 (no-op when --no-gap-fill is set).",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.5-flash",
        help=(
            "Gemini model for research grounded search. Default: gemini-2.5-flash "
            "(fully free at our scale per Google AI Studio rate limits). "
            "Production tournament uses gemini-3-flash-preview which requires Tier 1 billing. "
            "Canonical: this flag overrides any GEMINI_SEARCH_MODEL shell env var for the run."
        ),
    )
    gap_fill_group = parser.add_mutually_exclusive_group()
    gap_fill_group.add_argument(
        "--gap-fill",
        dest="gap_fill",
        action="store_true",
        help=(
            "Enable second-pass gap-fill grounded search. Off by default for the ablation; "
            "gap-fill amplifies leakage on resolved questions because the analyzer hunts for "
            "'factual gaps' which reliably surface resolution-revealing sentences."
        ),
    )
    gap_fill_group.add_argument(
        "--no-gap-fill",
        dest="gap_fill",
        action="store_false",
        help="Disable second-pass gap-fill (default). The benchmark deliberately uses single-pass Gemini.",
    )
    parser.set_defaults(gap_fill=False)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Global ceiling for OpenRouter parallelism; default: 4.",
    )
    # Default is ``patient`` (concurrency=1, max_retries=8) as of 2026-05-14
    # (Phase A.3 Package 3a). At 50q × 5 forecasters = 250 calls per arm,
    # ``gentle`` (concurrency=2, max_retries=3) was thrashing free-tier
    # per-minute throttles (qwen / minimax / gemma-4-26b) and bleeding
    # forecasters off the lineup — `patient`'s extra retry budget rides out
    # the 429 storms at the cost of wall-clock. Operators with a smoke
    # workload (≤4q) can opt back into ``gentle`` or ``fast``.
    parser.add_argument(
        "--rate-limit-mode",
        type=str,
        choices=list(_RATE_LIMIT_MODES),
        default="patient",
        help=(
            "Rate-limit dial. Maps to (per_forecaster_concurrency, max_retries) tuples: "
            "'fast' (4, 1) — historical behavior, lowest wall-clock; "
            "'gentle' (2, 3) — tolerates a single 429 wave per forecaster; "
            "'slow' (1, 5) — full serialization, for medium runs where wall-clock is secondary; "
            "'patient' (1, 8) — current default: slow but persistent, more retry "
            "budget to ride out free-tier 429 storms (qwen / minimax / gemma-4-26b) "
            "at 50q+ scale."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for reproducible bootstrap CIs in scoring; default: 0.",
    )
    parser.add_argument(
        "--qa-iterate-mode",
        type=str,
        choices=["halt", "advisory", "skip"],
        default="halt",
        help=(
            "Mode for the qa_iterate stage. 'halt' (default) writes the QA summary then raises "
            "RuntimeError so the operator can review manual_rejects.json before forecast spend. "
            "'advisory' writes the summary but proceeds to forecast. 'skip' bypasses the stage entirely."
        ),
    )
    parser.add_argument(
        "--qa-iterate-max-iterations",
        type=int,
        default=QA_ITERATE_DEFAULT_MAX_ITERATIONS,
        help=f"Max iterations per qid in qa_iterate; default: {QA_ITERATE_DEFAULT_MAX_ITERATIONS}.",
    )
    parser.add_argument(
        "--qa-iterate-leakage-threshold",
        type=float,
        default=QA_ITERATE_DEFAULT_LEAKAGE_THRESHOLD,
        help=(f"Verifier leakage_risk threshold for accepting a qid; default: {QA_ITERATE_DEFAULT_LEAKAGE_THRESHOLD}."),
    )
    parser.add_argument(
        "--qa-iterate-forecastability-threshold",
        type=float,
        default=QA_ITERATE_DEFAULT_FORECASTABILITY_THRESHOLD,
        help=(
            "Verifier forecastability threshold below which a clean blob is rejected as "
            f"too thin to forecast from; default: {QA_ITERATE_DEFAULT_FORECASTABILITY_THRESHOLD}. "
            "At 50q+ the modal smoke-run forecastability sits near 0.18 — operators can "
            "tighten to 0.25 or relax to 0.15 after seeing the iteration distribution."
        ),
    )
    parser.add_argument(
        "--prune-batch-size",
        type=int,
        default=PRUNE_DEFAULT_BATCH_SIZE,
        help=(
            f"Redactor batch size; default: {PRUNE_DEFAULT_BATCH_SIZE}. Lower = smaller "
            "blast radius on flaky runs (each subprocess failure drops at most batch_size "
            "qids before per-qid recovery kicks in)."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help=(
            "Logging level for stage transitions and per-qid verdicts. Default INFO. "
            "Set DEBUG for subprocess invocations and raw API responses. "
            "Logs are tee'd to stderr and to <cache-dir>/logs/run_<timestamp>.log."
        ),
    )
    parser.add_argument(
        "--lineup",
        type=str,
        choices=["free", "prod"],
        default="free",
        help=(
            "Forecaster ensemble: 'free' for the 4-model free-tier ablation (default), "
            "'prod' for the 3-model paid ensemble (claude-opus-4.6, claude-opus-4.8, "
            "gpt-5.6-sol, all at medium reasoning effort). The 'prod' lineup also selects "
            "the opus-4.8 prod stacker under --plain-llm."
        ),
    )
    parser.add_argument(
        "--plain-llm",
        action="store_true",
        default=False,
        help=(
            "Construct stacker LLMs via plain GeneralLlm (no donated-key wrapper, no "
            "fallback wrapping). Intended for paid benchmark-mode runs where fail-fast "
            "is preferred over cost absorption."
        ),
    )
    parser.add_argument(
        "--no-stacker-fallback",
        action="store_true",
        default=False,
        help=(
            "Disable the stacker fallback chain: on primary stacker failure, propagate "
            "the error immediately instead of trying the fallback LLM or median fallback. "
            "Intended for paid benchmark-mode runs where we want to see failures rather "
            "than silently degrade."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Manifest serialization
# ---------------------------------------------------------------------------


def _question_type_str(question: Any) -> str:
    qtype = question_type_of(question)
    if qtype is None:
        raise ValueError(f"Unsupported question type: {type(question).__name__}")
    return qtype


def _serialize_resolution(resolution: Any) -> Any:
    """Convert a ``GroundTruth.resolution`` to a JSON-safe value.

    ``OutOfBoundsResolution`` is an Enum; the cache writer's ``default=str``
    fallback would emit ``"OutOfBoundsResolution.ABOVE_UPPER_BOUND"`` which the
    ``float(...)`` call in :func:`_deserialize_ground_truth` cannot reverse.
    Tag the enum explicitly with a ``_type`` discriminator so the deserializer
    can reconstruct it via ``OutOfBoundsResolution[...]``.
    """
    if isinstance(resolution, OutOfBoundsResolution):
        return {"_type": "OutOfBoundsResolution", "value": resolution.name}
    if isinstance(resolution, datetime):
        return resolution.isoformat()
    return resolution


def _deserialize_resolution(raw: Any, question_type: str) -> Any:
    """Inverse of :func:`_serialize_resolution`, dispatched by question type."""
    if isinstance(raw, dict) and raw.get("_type") == "OutOfBoundsResolution":
        return OutOfBoundsResolution[raw["value"]]
    if question_type == "binary":
        return bool(raw)
    if question_type == "numeric":
        assert not isinstance(raw, dict)
        return float(raw)
    return str(raw)


def _serialize_ground_truth(gt: GroundTruth) -> dict:
    return {
        "question_id": gt.question_id,
        "question_type": gt.question_type,
        "resolution": _serialize_resolution(gt.resolution),
        "resolution_string": gt.resolution_string,
        "actual_resolution_time": (
            gt.actual_resolution_time.isoformat() if gt.actual_resolution_time is not None else None
        ),
        "question_text": gt.question_text,
        "page_url": gt.page_url,
    }


def _deserialize_ground_truth(payload: dict) -> GroundTruth:
    resolution = _deserialize_resolution(payload["resolution"], payload["question_type"])
    actual_time_raw = payload.get("actual_resolution_time")
    actual_time = datetime.fromisoformat(actual_time_raw) if actual_time_raw else None
    return GroundTruth(
        question_id=int(payload["question_id"]),
        question_type=payload["question_type"],
        resolution=resolution,
        resolution_string=payload["resolution_string"],
        community_prediction=None,
        actual_resolution_time=actual_time,
        question_text=payload["question_text"],
        page_url=payload.get("page_url"),
    )


def _serialize_question_metadata(question: Any) -> dict:
    # ``open_time`` / ``scheduled_resolution_time`` round-trip as ISO strings so
    # the manifest-hydration path gets real datetimes back; ``compute_mid_window_today``
    # subtracts them.
    metadata: dict[str, Any] = {
        "open_time": question.open_time.isoformat() if question.open_time is not None else None,
        "scheduled_resolution_time": (
            question.scheduled_resolution_time.isoformat() if question.scheduled_resolution_time is not None else None
        ),
    }
    if isinstance(question, NumericQuestion):
        metadata["lower_bound"] = float(question.lower_bound)
        metadata["upper_bound"] = float(question.upper_bound)
        metadata["open_lower_bound"] = bool(question.open_lower_bound)
        metadata["open_upper_bound"] = bool(question.open_upper_bound)
        # ``zero_point`` is legitimately Optional on NumericQuestion (None for
        # linear-scale, float for log-scale). Direct attribute access will fail
        # loudly if forecasting-tools renames the field; the None branch handles
        # the legitimate optional case.
        metadata["zero_point"] = float(question.zero_point) if question.zero_point is not None else None
        # ``unit_of_measure`` is read by ``stacking_numeric_prompt`` and ``numeric_prompt``;
        # legitimately Optional (None when the question doesn't specify a unit).
        metadata["unit_of_measure"] = question.unit_of_measure
        # ``cdf_size`` (``inbound_outcome_count + 1``) is 201 for continuous questions and
        # smaller for discrete ones (e.g. 17 for an integer-count 0..15 question). Persist it so
        # the rehydrated shim carries the real grid length instead of NumericQuestion's 201
        # default — the ARM_PDF structured-math arm builds its CDF on this grid, and a wrong
        # length silently mis-scores discrete questions. Older manifests (schema_version 1)
        # predate this field; the shim reader treats it as optional and the ablation arms
        # recover it from the cached per-forecaster CDF for those entries.
        metadata["cdf_size"] = int(question.cdf_size) if question.cdf_size is not None else None
    if isinstance(question, MultipleChoiceQuestion):
        metadata["options"] = list(question.options)
    return metadata


def _build_manifest_entry(question: Any, ground_truth: GroundTruth, tournament: str) -> dict:
    # ``resolution_criteria`` / ``fine_print`` / ``background_info`` are required
    # by every downstream consumer that rehydrates from the manifest:
    # * ``backtest.leakage._check_single_question_leakage`` reads ``resolution_criteria``.
    # * Stacker prompts (binary/MC/numeric) read all three.
    # All three are legitimately Optional on the Pydantic model (default ``None``),
    # so we serialize with that fallback. Direct attribute access surfaces drift
    # if forecasting-tools renames any of the fields.
    return {
        "type": _question_type_str(question),
        "tournament": tournament,
        "question_text": question.question_text,
        "page_url": question.page_url,
        "id_of_post": question.id_of_post,
        "resolution_criteria": question.resolution_criteria,
        "fine_print": question.fine_print,
        "background_info": question.background_info,
        "ground_truth": _serialize_ground_truth(ground_truth),
        "question_metadata": _serialize_question_metadata(question),
    }


def _id_of_post_from_entry(entry: dict) -> int | None:
    """Recover ``id_of_post`` from a manifest entry.

    Newer entries store it directly; older entries (written before the field
    was added) fall back to parsing the trailing integer from
    ``page_url=https://www.metaculus.com/questions/<post_id>``. Returns None
    when neither source yields an int.
    """
    explicit = entry.get("id_of_post")
    if isinstance(explicit, int):
        return explicit
    page_url = entry.get("page_url") or ""
    tail = page_url.rstrip("/").rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _build_question_shim_from_manifest_entry(qid: int, entry: dict) -> Any:
    """Rehydrate a real Pydantic question instance from a manifest entry.

    Function name is preserved for callers + tests; "shim" here means
    "rebuild without an API call," not "MagicMock." Real ``BinaryQuestion`` /
    ``NumericQuestion`` / ``MultipleChoiceQuestion`` instances flow through
    every downstream stage:

    * The forecaster path calls ``framework._initialize_notepad(question)``,
      which constructs ``Notepad(question=question)`` — a Pydantic model with
      ``question: MetaculusQuestion``. A MagicMock fails that validation.
    * The framework's ``_get_notepad`` builds an error message via
      ``question.id_of_post``; MagicMock(spec=...) raises AttributeError on
      any field not explicitly set.

    The manifest is written by ``_build_manifest_entry`` in this same module —
    every key dereferenced here is required by that writer's schema. Missing
    required keys mean schema drift, not optional fields, so direct subscript
    surfaces drift via ``KeyError``. ``zero_point`` and ``unit_of_measure``
    use ``.get`` because they are legitimately optional.
    """
    qtype = entry["type"]
    # Pre-validate required keys so ``KeyError`` surfaces schema drift before
    # we try to construct the Pydantic model (whose ``ValidationError`` would
    # be a less precise signal).
    required_top_level = ("page_url", "resolution_criteria", "fine_print", "background_info", "question_metadata")
    for key in required_top_level:
        if key not in entry:
            raise KeyError(key)
    metadata = entry["question_metadata"]
    required_metadata = ("open_time", "scheduled_resolution_time")
    for key in required_metadata:
        if key not in metadata:
            raise KeyError(key)

    open_time_raw = metadata["open_time"]
    scheduled_resolution_raw = metadata["scheduled_resolution_time"]
    open_time = datetime.fromisoformat(open_time_raw) if open_time_raw is not None else None
    scheduled_resolution_time = (
        datetime.fromisoformat(scheduled_resolution_raw) if scheduled_resolution_raw is not None else None
    )

    common_kwargs: dict[str, Any] = {
        "question_text": entry["question_text"],
        "page_url": entry["page_url"],
        "id_of_post": _id_of_post_from_entry(entry),
        "id_of_question": qid,
        "resolution_criteria": entry["resolution_criteria"],
        "fine_print": entry["fine_print"],
        "background_info": entry["background_info"],
        "open_time": open_time,
        "scheduled_resolution_time": scheduled_resolution_time,
    }

    if qtype == "binary":
        return BinaryQuestion(**common_kwargs)
    if qtype == "multiple_choice":
        # ``options`` is required on MC manifest entries; missing → drift.
        return MultipleChoiceQuestion(options=metadata["options"], **common_kwargs)
    if qtype == "numeric":
        # ``lower_bound`` / ``upper_bound`` / ``open_*_bound`` always written by
        # ``_serialize_question_metadata`` for numerics; missing → drift.
        numeric_kwargs: dict[str, Any] = {
            "lower_bound": metadata["lower_bound"],
            "upper_bound": metadata["upper_bound"],
            "open_lower_bound": metadata["open_lower_bound"],
            "open_upper_bound": metadata["open_upper_bound"],
            # ``zero_point`` is None for linear-scale numerics; legitimately optional.
            "zero_point": metadata.get("zero_point"),
            # ``unit_of_measure`` is None when the question doesn't specify a unit.
            "unit_of_measure": metadata.get("unit_of_measure"),
        }
        # ``cdf_size`` distinguishes a discrete grid (e.g. 17) from the 201-point continuous
        # default. Older manifests (schema_version 1) omit it — pass it only when present so
        # NumericQuestion keeps its 201 default for those entries (it rejects ``cdf_size=None``).
        cdf_size = metadata.get("cdf_size")
        if cdf_size is not None:
            numeric_kwargs["cdf_size"] = int(cdf_size)
        return NumericQuestion(**numeric_kwargs, **common_kwargs)
    raise ValueError(f"Unknown question type {qtype} in manifest entry for qid {qid}")


# ---------------------------------------------------------------------------
# Working set
# ---------------------------------------------------------------------------


@dataclass
class WorkingSet:
    """In-memory state shared across stages.

    ``research_blobs`` holds RAW blobs after the research stage and SANITIZED
    blobs after the prune stage. Raw blobs always remain on disk under
    ``research/<qid>.md``; sanitized blobs land in ``research_pruned/<qid>.md``.
    Downstream stages (screen, forecast, stack) always see sanitized blobs.
    ``prune_metas`` carries per-qid redactor metadata for the QA dump.
    """

    questions: dict[int, Any] = field(default_factory=dict)
    ground_truths: dict[int, GroundTruth] = field(default_factory=dict)
    research_blobs: dict[int, str] = field(default_factory=dict)
    prune_metas: dict[int, dict] = field(default_factory=dict)
    leakage_verdicts: dict[int, dict] = field(default_factory=dict)
    forecaster_payloads: dict[int, dict[str, dict]] = field(default_factory=dict)
    stacker_payloads: dict[str, dict[int, dict]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Question loaders
# ---------------------------------------------------------------------------


async def load_questions_by_qids(qids: list[int]) -> tuple[list[Any], dict[int, GroundTruth]]:
    """Fetch a fixed list of qids individually. Tests monkey-patch this.

    Production path: not implemented yet (real Metaculus per-qid fetch). The
    ablation runner only uses ``--qids`` for cached re-runs, where the manifest
    already has the data; tests cover the bypass-fetch path.
    """
    await asyncio.sleep(0)
    raise NotImplementedError(
        "load_questions_by_qids: production path needs MetaculusApi.get_question_by_post_id wiring; "
        "tests monkey-patch this entry point."
    )


# ---------------------------------------------------------------------------
# Fetch stage
# ---------------------------------------------------------------------------


async def _stage_fetch(args: argparse.Namespace, cache: AblationCache, working: WorkingSet) -> None:
    """Populate the working set with questions + ground truths + manifest.

    Behavior:
    - ``args.qids`` provided → call ``load_questions_by_qids``; do not fetch from tournaments.
    - Else → read existing manifest; fetch ADDITIONAL questions per type if requested counts
      exceed existing counts, excluding already-known qids.
    """
    await asyncio.sleep(0)
    if args.qids:
        await _fetch_explicit_qids(args, cache, working)
    else:
        await _fetch_tournament_delta(args, cache, working)


async def _fetch_explicit_qids(args: argparse.Namespace, cache: AblationCache, working: WorkingSet) -> None:
    """Hydrate ``args.qids`` from the manifest, fetching only the ones missing from it."""
    existing = cache.read_qids_manifest()
    # If every qid already in manifest, no fetch needed — load shims from manifest.
    if all(qid in existing for qid in args.qids):
        for qid in args.qids:
            entry = existing[qid]
            question = _build_question_shim_from_manifest_entry(qid, entry)
            working.questions[qid] = question
            working.ground_truths[qid] = _deserialize_ground_truth(entry["ground_truth"])
        return

    # Some qids missing from manifest: fetch them via the per-qid loader.
    questions, ground_truths = await load_questions_by_qids(args.qids)
    new_entries: dict[int, dict] = {}
    tournament = args.tournaments[0] if args.tournaments else DEFAULT_TOURNAMENTS[0]
    for question in questions:
        qid = question.id_of_question
        gt = ground_truths[qid]
        working.questions[qid] = question
        working.ground_truths[qid] = gt
        new_entries[qid] = _build_manifest_entry(question, gt, tournament)
    cache.append_qids_manifest(new_entries)


async def _fetch_tournament_delta(args: argparse.Namespace, cache: AblationCache, working: WorkingSet) -> None:
    """Tournament fetch with append-extend semantics.

    Hydrates every already-known qid from the manifest, then fetches only the per-type
    shortfall against the requested counts, dropping anything the manifest already has.
    """
    await asyncio.sleep(0)
    existing = cache.read_qids_manifest()
    existing_per_type = {"binary": 0, "multiple_choice": 0, "numeric": 0}
    for entry in existing.values():
        # Manifest entries always have ``type``; direct subscript surfaces drift.
        qtype = entry["type"]
        if qtype in existing_per_type:
            existing_per_type[qtype] += 1
        # Hydrate working set from manifest for already-known qids.
        qid = int(entry["ground_truth"]["question_id"])
        question = _build_question_shim_from_manifest_entry(qid, entry)
        working.questions[qid] = question
        working.ground_truths[qid] = _deserialize_ground_truth(entry["ground_truth"])

    delta_binary = max(0, args.num_binary - existing_per_type["binary"])
    delta_mc = max(0, args.num_multiple_choice - existing_per_type["multiple_choice"])
    delta_numeric = max(0, args.num_numeric - existing_per_type["numeric"])

    if delta_binary == 0 and delta_mc == 0 and delta_numeric == 0:
        logger.info(
            "fetch | manifest already saturated for requested counts | per-type=%s",
            existing_per_type,
        )
        return

    logger.info(
        "fetch | requesting delta | binary=%d mc=%d numeric=%d (existing=%s)",
        delta_binary,
        delta_mc,
        delta_numeric,
        existing_per_type,
    )

    # Confirm the host is the real Metaculus before the token-sending fetch.
    verify_metaculus_api_identity()

    question_set: BacktestQuestionSet = await fetch_resolved_questions_stratified(
        num_binary=delta_binary,
        num_multiple_choice=delta_mc,
        num_numeric=delta_numeric,
        resolved_after=args.resolved_after,
        resolved_before=args.resolved_before,
        tournaments=args.tournaments,
    )

    # Drop any qids already in the manifest (the fetcher doesn't have an exclude-list arg).
    # ``fetch_resolved_questions_stratified`` guarantees ``ground_truths`` covers every
    # qid in ``questions`` — direct subscript surfaces a fetcher invariant violation.
    new_entries = {}
    tournament = args.tournaments[0] if args.tournaments else DEFAULT_TOURNAMENTS[0]
    for question in question_set.questions:
        qid = question.id_of_question
        assert qid is not None
        if qid in existing:
            continue
        gt = question_set.ground_truths[qid]
        working.questions[qid] = question
        working.ground_truths[qid] = gt
        new_entries[qid] = _build_manifest_entry(question, gt, tournament)

    if new_entries:
        cache.append_qids_manifest(new_entries)


# ---------------------------------------------------------------------------
# Research stage
# ---------------------------------------------------------------------------


async def _stage_research(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    force: bool,
    spend: SpendReport,
) -> None:
    """Populate working.research_blobs from cache + live runner.

    Spend counters: a qid cached at stage entry (and not forced) increments
    ``cached_research_hits``. A fresh run for a qid increments
    ``gemini_research_calls`` (whether or not it succeeded — the API was hit)
    and, on success, adds ``meta["gap_count"]`` to ``gemini_gap_fill_calls``
    when ``meta["gap_fill_used"]`` is True.
    """
    await asyncio.sleep(0)
    qids = sorted(working.questions.keys())
    cached_blobs: dict[int, str] = {}
    if not force:
        for qid in qids:
            cached = cache.read_research(qid)
            if cached is not None:
                cached_blobs[qid] = cached[0]
                spend.cached_research_hits += 1

    questions_to_fetch = [working.questions[qid] for qid in qids if qid not in cached_blobs]

    fresh_results: dict[int, tuple[str, dict] | None] = {}
    if questions_to_fetch:
        fresh_results = await run_gemini_research_for_qids(
            questions_to_fetch,
            cache,
            gap_fill_max_gaps=args.gap_fill_max_gaps,
            is_benchmarking=True,
            force=force,
            concurrency=args.concurrency,
            gemini_model=args.gemini_model,
            enable_gap_fill=args.gap_fill,
        )

    for qid, blob in cached_blobs.items():
        working.research_blobs[qid] = blob
    for qid, result in fresh_results.items():
        spend.gemini_research_calls += 1
        if result is None:
            logger.warning("research | qid=%d failed; dropping from downstream", qid)
            continue
        blob, meta = result
        if meta.get("gap_fill_used"):
            spend.gemini_gap_fill_calls += int(meta.get("gap_count", 0))
        working.research_blobs[qid] = blob


# ---------------------------------------------------------------------------
# Prune stage
# ---------------------------------------------------------------------------


async def _stage_prune(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    force: bool,
    spend: SpendReport,
) -> None:
    """Run the redactor over ``working.research_blobs``; replace with sanitized blobs.

    After this stage, ``working.research_blobs[qid]`` holds the SANITIZED blob
    (the screen / forecast / stack stages always operate on sanitized blobs).
    Validation failures (subagent emitted a sanitized blob still containing the
    ground truth) drop the qid from ``working.research_blobs`` entirely; that
    qid then never reaches the screen or any downstream stage.
    """
    await asyncio.sleep(0)
    qids = sorted(working.research_blobs.keys())
    if not qids:
        return

    triples: list[tuple[Any, GroundTruth, str]] = []
    pre_cached_qids: set[int] = set()
    for qid in qids:
        question = working.questions[qid]
        gt = working.ground_truths[qid]
        raw_blob = working.research_blobs[qid]
        if not force and cache.has_pruned_research(qid):
            pre_cached_qids.add(qid)
        triples.append((question, gt, raw_blob))

    spend.cached_prune_hits += len(pre_cached_qids)

    results = await run_prune_for_qids(triples, cache, force=force, batch_size=args.prune_batch_size)

    new_blobs: dict[int, str] = {}
    new_metas: dict[int, dict] = {}
    for qid, result in results.items():
        if result is None:
            spend.prune_validation_failures += 1
            logger.warning("prune | qid=%d failed; dropping from downstream", qid)
            continue
        sanitized_blob, meta = result
        new_blobs[qid] = sanitized_blob
        new_metas[qid] = meta

    # Count fresh redactor invocations: one per qid that was NOT pre-cached.
    # One invocation per batch (default 10 qids per batch).
    fresh_qids = [qid for qid in qids if qid not in pre_cached_qids]
    if fresh_qids:
        batches = (len(fresh_qids) + args.prune_batch_size - 1) // args.prune_batch_size
        spend.redactor_invocations += batches

    working.research_blobs = new_blobs
    working.prune_metas = new_metas


# ---------------------------------------------------------------------------
# Screen stage
# ---------------------------------------------------------------------------


def _partition_screen_cache(
    cache: AblationCache,
    research_blobs: dict[int, str],
    *,
    force: bool,
    spend: SpendReport,
) -> tuple[dict[int, dict], list[int]]:
    """Split the qids into (reusable cached verdicts, qids needing a fresh screen).

    A cached verdict only counts when its ``research_blob_sha`` still matches the blob
    on hand: a pre-C3 entry carrying no sha, or a stale sha after a re-prune, re-screens.
    Bumps ``spend.cached_screen_hits`` for each reuse.
    """
    if force:
        return {}, list(research_blobs.keys())

    cached_verdicts: dict[int, dict] = {}
    qids_needing_screen: list[int] = []
    for qid, blob in research_blobs.items():
        cached = cache.read_leakage_screen(qid)
        if cached is not None and cached.get("research_blob_sha") == _research_blob_sha(blob):
            cached_verdicts[qid] = cached
            spend.cached_screen_hits += 1
        else:
            if cached is not None:
                logger.info("screen | qid=%d cache stale (blob changed); re-screening", qid)
            qids_needing_screen.append(qid)
    return cached_verdicts, qids_needing_screen


async def _stage_screen(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    force: bool,
    spend: SpendReport,
) -> None:
    """Run leakage screen on research blobs; drop leaked qids.

    Spend counters: cached + non-forced verdicts bump ``cached_screen_hits``.
    Fresh verdicts bump ``leakage_detector_calls`` UNLESS the verdict's
    ``detector_response`` equals the empty-blob sentinel — that path is a
    short-circuit in ``screen_research_blob`` that never hits the LLM.
    """
    await asyncio.sleep(0)
    questions_with_research = [working.questions[qid] for qid in working.research_blobs if qid in working.questions]
    ground_truths = {qid: working.ground_truths[qid] for qid in working.research_blobs if qid in working.ground_truths}
    research_blobs = dict(working.research_blobs)

    cached_verdicts, qids_needing_screen = _partition_screen_cache(cache, research_blobs, force=force, spend=spend)

    fresh_verdicts: dict[int, dict] = {}
    if qids_needing_screen:
        questions_subset = [q for q in questions_with_research if q.id_of_question in qids_needing_screen]
        gts_subset = {qid: gt for qid, gt in ground_truths.items() if qid in qids_needing_screen}
        blobs_subset = {qid: blob for qid, blob in research_blobs.items() if qid in qids_needing_screen}
        _clean_qs, _clean_gts, fresh_verdicts = await screen_batch(
            questions_subset,
            gts_subset,
            blobs_subset,
            cache,
            force=force,
            concurrency=args.concurrency,
        )

    for verdict in fresh_verdicts.values():
        # Verdict schema is fixed (leakage_screen._build_verdict);
        # ``detector_response`` is always present.
        if verdict["detector_response"] == _EMPTY_BLOB_RESPONSE:
            continue
        spend.leakage_detector_calls += 1

    all_verdicts = {**cached_verdicts, **fresh_verdicts}
    working.leakage_verdicts = all_verdicts

    # Drop leaked qids from the working set's downstream-stage data.
    # Verdict dicts come from ``leakage_screen._build_verdict`` with a fixed
    # schema; ``is_leaked`` is always present.
    for qid, verdict in all_verdicts.items():
        if verdict["is_leaked"]:
            working.research_blobs.pop(qid, None)


# ---------------------------------------------------------------------------
# QA-iterate stage
# ---------------------------------------------------------------------------


def _qa_iterate_paths(cache: AblationCache, *, timestamp: str) -> tuple[Path, Path]:
    """Return (summary_path, manual_rejects_path) for a qa_iterate run."""
    summary_path = cache.root / f"qa_summary_{timestamp}.md"
    manual_rejects_path = cache.root / "manual_rejects.json"
    return summary_path, manual_rejects_path


async def _stage_qa_iterate(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    force: bool = False,
) -> tuple[dict[int, IterateOutcome], Path | None]:
    """Iterate-until-clean QA over surviving qids.

    Returns ``(outcomes, summary_path)``. ``summary_path`` is ``None`` only when
    mode='skip' or there are zero surviving qids. Otherwise the summary is
    always written, regardless of mode.

    When ``force`` is True (passed via ``--force-stages qa_iterate``), the
    pre-existing ``manual_rejects.json`` is archived to
    ``manual_rejects.bak.<timestamp>.json`` instead of being honored, so the
    operator can re-run the verifier on previously-rejected qids after fixing
    upstream issues (e.g. redactor over-aggression).
    """
    await asyncio.sleep(0)
    if args.qa_iterate_mode == "skip":
        logger.info("stage=qa_iterate SKIPPED (mode=skip)")
        return {}, None

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    summary_path, manual_rejects_path = _qa_iterate_paths(cache, timestamp=timestamp)

    if force and manual_rejects_path.exists():
        # Auto-archive prior rejects on --force-stages qa_iterate. Without this,
        # forcing the stage is a silent no-op (the file is read at stage start
        # and pre-rejected qids are popped from working.research_blobs before
        # the verifier runs).
        backup_path = manual_rejects_path.parent / f"manual_rejects.bak.{timestamp}.json"
        manual_rejects_path.rename(backup_path)
        logger.info("--force-stages qa_iterate: archived prior rejects to %s", backup_path)
        existing_rejects = {}
    else:
        existing_rejects = read_manual_rejects(manual_rejects_path)
    for qid in list(existing_rejects.keys()):
        working.research_blobs.pop(qid, None)

    surviving_qids = sorted(working.research_blobs.keys())
    if not surviving_qids:
        logger.info("stage=qa_iterate no surviving qids after honoring manual_rejects.json")
        return {}, None

    inputs: dict[int, dict[str, Any]] = {}
    for qid in surviving_qids:
        inputs[qid] = {
            "question": working.questions[qid],
            "ground_truth": working.ground_truths[qid],
            "current_blob": working.research_blobs[qid],
            "screen_verdict": working.leakage_verdicts.get(
                qid,
                {"is_leaked": False, "detector_response": "", "detector_model": "", "detector_failed": False},
            ),
        }

    outcomes = await run_qa_iterate_batch(
        inputs,
        cache=cache,
        max_iterations=args.qa_iterate_max_iterations,
        leakage_threshold=args.qa_iterate_leakage_threshold,
        forecastability_threshold=args.qa_iterate_forecastability_threshold,
        concurrency=args.concurrency,
    )

    render_qa_summary(outcomes, summary_path)
    write_manual_rejects(list(outcomes.values()), manual_rejects_path)

    # Per-qid qa_reports: one structured JSON per qid recording final_status,
    # iterations, and verifier_scores. The plan
    # (``scratch_docs_and_planning/ablation_phase_a3_plan.md:290``) specifies
    # these alongside the aggregate summary; the audit at
    # ``backtests/ablation/audit_smoke_20260515.md:243-263`` confirmed they
    # were missing from the smoke run.
    qa_reports_dir = cache.root / "qa_reports"
    for qid, outcome in outcomes.items():
        report_path = qa_reports_dir / f"{qid}.json"
        atomic_write_text(report_path, json.dumps(serialize_outcome(outcome), indent=2, default=str))
        logger.info("qa_iterate | qid=%d report written: %s", qid, report_path)

    rejected_qids = [qid for qid, outcome in outcomes.items() if outcome.final_status != "clean"]
    for qid in rejected_qids:
        working.research_blobs.pop(qid, None)

    n_clean = sum(1 for o in outcomes.values() if o.final_status == "clean")
    logger.info(
        "stage=qa_iterate DONE | clean=%d rejected=%d summary=%s",
        n_clean,
        len(rejected_qids),
        summary_path,
    )
    return outcomes, summary_path


# ---------------------------------------------------------------------------
# Forecast stage
# ---------------------------------------------------------------------------


async def _stage_forecast(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    force: bool,
    spend: SpendReport,
) -> None:
    """Populate working.forecaster_payloads via cache + run_forecasters_batch.

    Spend counters work at the (qid, model_slug) cell level. Per qid we
    pre-snapshot the slugs already on disk; cells in the snapshot AND not
    forced count as ``cached_forecaster_hits``. Cells outside the snapshot
    (or all cells under force) count as ``forecaster_llm_calls``. Cell
    granularity uses the active lineup's model list so the count doesn't
    drift if the orchestrator's coarse all-or-nothing cache check changes
    shape later.
    """
    from metaculus_bot.ablation.forecaster_lineup import get_lineup  # noqa: PLC0415

    await asyncio.sleep(0)
    qids = sorted(working.research_blobs.keys())

    lineup_name: str = getattr(args, "lineup", "free")
    forecaster_llms, lineup_models = get_lineup(lineup_name)

    cached_per_qid: dict[int, dict[str, dict]] = {}
    pre_snapshot_slugs: dict[int, set[str]] = {}
    for qid in qids:
        # m2: filter the on-disk listing to the CURRENT lineup so obsolete
        # files don't pollute cache hit accounting or downstream stacker calls.
        on_disk = cache.list_forecaster_outputs(qid, lineup_filter=lineup_models)
        pre_snapshot_slugs[qid] = set(on_disk.keys())
        if force or not on_disk:
            continue
        # Task #23: content-aware cache hit. A forecaster cache that contains
        # only error payloads (prediction_value=None or non-empty errors) would
        # cause the stacker stage to permanently cache "insufficient_forecasters"
        # and skip the qid at score time. Re-run if the surviving count is below
        # the stacker's downstream threshold.
        n_valid = sum(1 for p in on_disk.values() if p.get("prediction_value") is not None and not p.get("errors"))
        if n_valid < ABLATION_MIN_FORECASTERS:
            logger.info(
                "forecast | qid=%d cache has %d valid forecasters (< %d threshold); re-running",
                qid,
                n_valid,
                ABLATION_MIN_FORECASTERS,
            )
            continue
        cached_per_qid[qid] = on_disk

    needs_run = [qid for qid in qids if qid not in cached_per_qid]
    questions_with_research = [(working.questions[qid], working.research_blobs[qid]) for qid in needs_run]

    fresh_results: dict[int, dict[str, dict]] = {}
    if questions_with_research:
        rate_limit_kwargs = _rate_limit_mode_kwargs(args.rate_limit_mode)
        fresh_results = await run_forecasters_batch(
            questions_with_research,
            cache,
            forecaster_llms=forecaster_llms,
            force=force,
            per_question_concurrency=args.concurrency,
            per_forecaster_concurrency=rate_limit_kwargs["per_forecaster_concurrency"],
            max_retries=rate_limit_kwargs["max_retries"],
        )

    for qid in qids:
        cached_slugs = pre_snapshot_slugs[qid] if not force else set()
        for model in lineup_models:
            slug = model_slug_to_filename(model)
            if slug in cached_slugs:
                spend.cached_forecaster_hits += 1
            else:
                spend.forecaster_llm_calls += 1

    for qid, payloads in cached_per_qid.items():
        working.forecaster_payloads[qid] = payloads
    for qid, payloads in fresh_results.items():
        working.forecaster_payloads[qid] = payloads


# ---------------------------------------------------------------------------
# Stacker stage
# ---------------------------------------------------------------------------


def _active_stacker_slug(args: argparse.Namespace) -> str:
    """Filesystem slug for the stacker this run uses, for per-stacker cache keying.

    Only the LLM-stacker arms (stack / stack_aug) are slugged so a stacker swap
    (e.g. opus-4.5 free-tier vs opus-4.8 prod) never overwrites another stacker's
    results, while deterministic arms (median / mean / pdf_*) stay shared. The
    selection mirrors the stacker construction in ``_stage_llm_stacker``:

    * ``--plain-llm --lineup prod`` → opus-4.8 (``PROD_STACKER_MODEL``).
    * ``--plain-llm`` other lineups → opus-4.5 (``DEFAULT_STACKER_MODEL``).
    * No ``--plain-llm`` → the default donated-key wrapper, whose primary is also
      ``DEFAULT_STACKER_MODEL`` (opus-4.5).

    All three callers (``_stage_llm_stacker`` write/read, ``_stage_score`` read,
    ``_hydrate_working_set_from_cache`` read) route through here so they agree.
    """
    from metaculus_bot.ablation.run_stacker import DEFAULT_STACKER_MODEL, PROD_STACKER_MODEL  # noqa: PLC0415

    use_prod_stacker = getattr(args, "plain_llm", False) and getattr(args, "lineup", "free") == "prod"
    model = PROD_STACKER_MODEL if use_prod_stacker else DEFAULT_STACKER_MODEL
    return model_slug_to_filename(model)


async def _stage_simple_agg(
    arm: str,
    working: WorkingSet,
    cache: AblationCache,
    *,
    force: bool,
    spend: SpendReport,
) -> None:
    """Sequential deterministic aggregation (mean/median). No LLM calls."""
    await asyncio.sleep(0)
    qids = sorted(working.forecaster_payloads.keys())

    cached_payloads: dict[int, dict] = {}
    needs_run: list[int] = []
    for qid in qids:
        if not force:
            cached = cache.read_stacker_output(qid=qid, arm=arm)
            if cached is not None:
                cached_payloads[qid] = cached
                spend.cached_stacker_hits[arm] = spend.cached_stacker_hits.get(arm, 0) + 1
                continue
        needs_run.append(qid)

    fresh_results: dict[int, dict] = {}
    if needs_run:
        run_fn = run_median_for_qid if arm == ARM_MEDIAN else run_mean_for_qid
        for qid in needs_run:
            fresh_results[qid] = await run_fn(
                qid=qid,
                question=working.questions[qid],
                forecaster_payloads=working.forecaster_payloads[qid],
                cache=cache,
                force=force,
            )

    working.stacker_payloads[arm] = {**cached_payloads, **fresh_results}


async def _stage_pdf(
    working: WorkingSet,
    cache: AblationCache,
    force: bool,
    spend: SpendReport,
) -> None:
    """Deterministic structured-math aggregation for all PDF sub-arms."""
    from metaculus_bot.ablation.run_pdf import ARM_PDF_MIN1_MEAN  # noqa: PLC0415

    await asyncio.sleep(0)
    qids = sorted(working.forecaster_payloads.keys())

    # PDF sub-arms share a single cache-read loop for the parent ARM_PDF key.
    cached_payloads: dict[int, dict] = {}
    needs_run: list[int] = []
    for qid in qids:
        if not force:
            cached = cache.read_stacker_output(qid=qid, arm=ARM_PDF)
            if cached is not None:
                cached_payloads[qid] = cached
                spend.cached_stacker_hits[ARM_PDF] = spend.cached_stacker_hits.get(ARM_PDF, 0) + 1
                continue
        needs_run.append(qid)

    results_min1: dict[int, dict] = {}
    results_min2: dict[int, dict] = {}
    for qid in needs_run:
        result_min1 = await run_pdf_for_qid(
            qid=qid,
            question=working.questions[qid],
            forecaster_payloads=working.forecaster_payloads[qid],
            cache=cache,
            force=force,
            min_forecasters=1,
            arm_label=ARM_PDF_MIN1,
        )
        result_min2 = await run_pdf_for_qid(
            qid=qid,
            question=working.questions[qid],
            forecaster_payloads=working.forecaster_payloads[qid],
            cache=cache,
            force=force,
            min_forecasters=2,
            arm_label=ARM_PDF_MIN2,
        )
        # pdf_min1_mean is computed for diagnostic completeness but not scored.
        await run_pdf_for_qid(
            qid=qid,
            question=working.questions[qid],
            forecaster_payloads=working.forecaster_payloads[qid],
            cache=cache,
            force=force,
            min_forecasters=1,
            arm_label=ARM_PDF_MIN1_MEAN,
            aggregation="mean",
        )
        results_min1[qid] = result_min1
        results_min2[qid] = result_min2

    working.stacker_payloads[ARM_PDF_MIN1] = {**cached_payloads, **results_min1}
    working.stacker_payloads[ARM_PDF_MIN2] = {**cached_payloads, **results_min2}


def _partition_stacker_cache(
    cache: AblationCache,
    qids: list[int],
    arm: str,
    *,
    stacker_slug: str | None,
    force: bool,
    spend: SpendReport,
) -> tuple[dict[int, dict], list[int]]:
    """Split the qids into (cached arm payloads, qids needing a stacker run)."""
    if force:
        return {}, list(qids)

    cached_payloads: dict[int, dict] = {}
    needs_run: list[int] = []
    for qid in qids:
        cached = cache.read_stacker_output(qid=qid, arm=arm, stacker_slug=stacker_slug)
        if cached is not None:
            cached_payloads[qid] = cached
            spend.cached_stacker_hits[arm] = spend.cached_stacker_hits.get(arm, 0) + 1
        else:
            needs_run.append(qid)
    return cached_payloads, needs_run


def _stacker_batch_kwargs(
    args: argparse.Namespace,
    *,
    stacker_slug: str | None,
    force: bool,
) -> dict[str, Any]:
    """Wire --plain-llm and --no-stacker-fallback into ``run_stacker_batch``'s kwargs.

    An omitted ``stacker_llm`` / ``fallback_stacker_llm`` key leaves the runner on its
    own donated-key-wrapped defaults, which is why the keys are added conditionally
    rather than passed as None: an explicit None means "no fallback chain".
    ``--lineup prod`` uses the prod stacker (mirroring the prod-ish forecaster posture);
    other lineups keep the opus default.
    """
    stacker_llm_kwarg: GeneralLlm | None = None
    fallback_llm_kwarg: GeneralLlm | None = None
    if getattr(args, "plain_llm", False):
        from metaculus_bot.ablation.run_stacker import (  # noqa: PLC0415
            _OPENAI_STACKER_KWARGS,
            _OPUS_STACKER_KWARGS,
            _PROD_STACKER_KWARGS,
            DEFAULT_STACKER_FALLBACK_MODEL,
            DEFAULT_STACKER_MODEL,
            PROD_STACKER_MODEL,
        )

        if getattr(args, "lineup", "free") == "prod":
            stacker_llm_kwarg = GeneralLlm(model=PROD_STACKER_MODEL, **_PROD_STACKER_KWARGS)
        else:
            stacker_llm_kwarg = GeneralLlm(model=DEFAULT_STACKER_MODEL, **_OPUS_STACKER_KWARGS)
        if not getattr(args, "no_stacker_fallback", False):
            fallback_llm_kwarg = GeneralLlm(model=DEFAULT_STACKER_FALLBACK_MODEL, **_OPENAI_STACKER_KWARGS)

    batch_kwargs: dict[str, Any] = {
        "stacker_slug": stacker_slug,
        "force": force,
        "concurrency": args.concurrency,
    }
    if stacker_llm_kwarg is not None:
        batch_kwargs["stacker_llm"] = stacker_llm_kwarg
    if getattr(args, "no_stacker_fallback", False):
        batch_kwargs["fallback_stacker_llm"] = None
    elif fallback_llm_kwarg is not None:
        batch_kwargs["fallback_stacker_llm"] = fallback_llm_kwarg
    return batch_kwargs


def _record_stacker_spend(fresh_results: dict[int, dict], arm: str, spend: SpendReport) -> None:
    """Count stacker + parser LLM calls and fallback usage from this arm's fresh results."""
    is_arm_a = arm == ARM_STACK
    for payload in fresh_results.values():
        used = payload["stacker_model_used"]
        if used in ("primary", "fallback"):
            if is_arm_a:
                spend.stacker_llm_calls_stack += 1
            else:
                spend.stacker_llm_calls_stack_aug += 1
            spend.parser_llm_calls += 1
        if used == "fallback":
            if is_arm_a:
                spend.fallback_stacker_stack += 1
            else:
                spend.fallback_stacker_stack_aug += 1


async def _stage_llm_stacker(
    arm: str,
    args: argparse.Namespace,
    working: WorkingSet,
    cache: AblationCache,
    *,
    force: bool,
    spend: SpendReport,
) -> None:
    """LLM stacker dispatch (stack/stack_aug). Uses run_stacker_batch."""
    await asyncio.sleep(0)
    qids = sorted(working.forecaster_payloads.keys())

    # Per-stacker cache keying: stack/stack_aug payloads are slugged by the active
    # stacker so a swap never clobbers another stacker's results.
    stacker_slug = _active_stacker_slug(args)

    cached_payloads, needs_run = _partition_stacker_cache(
        cache, qids, arm, stacker_slug=stacker_slug, force=force, spend=spend
    )

    fresh_results: dict[int, dict] = {}
    if needs_run:
        qid_to_data = {
            qid: {
                "question": working.questions[qid],
                "research": working.research_blobs.get(qid, ""),
                "forecaster_payloads": working.forecaster_payloads[qid],
            }
            for qid in needs_run
        }
        fresh_results = await run_stacker_batch(
            qid_to_data,
            arm,
            cache,
            **_stacker_batch_kwargs(args, stacker_slug=stacker_slug, force=force),
        )

    _record_stacker_spend(fresh_results, arm, spend)
    working.stacker_payloads[arm] = {**cached_payloads, **fresh_results}


async def _stage_stack(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    arm: str,
    force: bool,
    spend: SpendReport,
) -> None:
    """Dispatcher for stacker arms. Routes to per-type implementation."""
    if arm in (ARM_MEDIAN, ARM_MEAN):
        await _stage_simple_agg(arm, working, cache, force=force, spend=spend)
    elif arm == ARM_PDF:
        await _stage_pdf(working, cache, force, spend)
    else:
        await _stage_llm_stacker(arm, args, working, cache, force=force, spend=spend)


# ---------------------------------------------------------------------------
# Stage: QA dump
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Score stage
# ---------------------------------------------------------------------------


def _build_report_shim(qid: int, question: Any, payload: dict) -> Any:
    """Build a MagicMock report whose isinstance() check matches the question type.

    The score function in ``metaculus_bot.ablation.scoring`` uses ``isinstance``
    to dispatch to the right metric set; ``MagicMock(spec=BinaryReport)`` etc.
    pass that check.
    """
    stacker_pred = payload["stacker_prediction"]
    pred_type = stacker_pred["type"]

    if pred_type == "binary":
        report = MagicMock(spec=BinaryReport)
        report.prediction = float(stacker_pred["prob"])
        return report

    if pred_type == "multiple_choice":
        report = MagicMock(spec=MultipleChoiceReport)
        prediction = MagicMock()
        predicted_options: list[Any] = []
        for opt in stacker_pred["options"]:
            po = MagicMock()
            po.option_name = opt["option_name"]
            po.probability = float(opt["probability"])
            predicted_options.append(po)
        prediction.predicted_options = predicted_options
        report.prediction = prediction
        report.question = question
        return report

    if pred_type == "numeric":
        # Post-Bucket-1: ``deserialize_prediction_value`` returns a
        # ``PchipNumericDistribution`` whose ``.cdf`` already provides the
        # constraint-enforced 201-point CDF as a list of Percentile objects
        # (monotonic by construction — PCHIP enforces strict monotonicity in
        # the value axis). No defensive sort or duplicate-check needed; that
        # was a workaround for the old ``list[Percentile]`` return type when
        # free-model stackers emitted out-of-order declared percentiles.
        report = MagicMock(spec=NumericReport)
        deserialized = deserialize_prediction_value(stacker_pred, question)
        cdf_points: list[Any] = []
        for percentile in deserialized.cdf:
            point = MagicMock()
            point.value = float(percentile.value)
            point.percentile = float(percentile.percentile)
            cdf_points.append(point)
        prediction = MagicMock()
        prediction.cdf = cdf_points
        report.prediction = prediction
        report.question = question
        return report

    raise ValueError(f"Unknown prediction type {pred_type} for qid {qid}")


# Which cache arm each scoring-arm label reads from, in the canonical order
# ``score_arm_for_qid`` requires its ``arm_reports`` list to arrive in.
_SCORING_ARM_CACHE_KEYS: dict[str, str] = {
    "stack": ARM_STACK,
    "stack_aug": ARM_STACK_AUG,
    "pdf_min1": ARM_PDF_MIN1,
    "pdf_min2": ARM_PDF_MIN2,
    "median": ARM_MEDIAN,
    "mean": ARM_MEAN,
}
_SCORING_ARM_LABELS_6: tuple[str, ...] = ("stack", "stack_aug", "pdf_min1", "pdf_min2", "median", "mean")
_SCORING_ARM_LABELS_5: tuple[str, ...] = ("stack", "stack_aug", "pdf_min1", "pdf_min2", "median")
_SCORING_ARM_LABELS_3: tuple[str, ...] = ("stack", "stack_aug", "median")


def _scoring_arm_labels(*, has_pdf_arms: bool, has_mean_arm: bool) -> tuple[str, ...]:
    """Which arms enter the paired comparison for this cache directory.

    6-arm adds the deterministic mean arm on top of the 5-arm set, and is used only
    when BOTH pdf AND mean payloads are present. Old free-tier cache dirs (pdf, no
    mean) stay on the 5-arm path; pre-pdf data stays on the 3-arm path.
    """
    if has_pdf_arms and has_mean_arm:
        return _SCORING_ARM_LABELS_6
    if has_pdf_arms:
        return _SCORING_ARM_LABELS_5
    return _SCORING_ARM_LABELS_3


def _arm_score_inputs(working: WorkingSet, qid: int, question: Any) -> dict[str, tuple[Any, dict | None]]:
    """(report, payload) per scoring-arm label; report is None when the arm is missing or failed."""
    inputs: dict[str, tuple[Any, dict | None]] = {}
    for label, arm_key in _SCORING_ARM_CACHE_KEYS.items():
        payload = working.stacker_payloads.get(arm_key, {}).get(qid)
        report = _build_report_shim(qid, question, payload) if payload is not None and payload.get("success") else None
        inputs[label] = (report, payload)
    return inputs


def _build_paired_scores(
    working: WorkingSet,
    qids: list[int],
    arm_labels: tuple[str, ...],
) -> tuple[list[PairedScore], int]:
    """Score every qid that has at least 2 present arms; returns (rows, n_scored).

    The 2-arm floor is counted over EVERY arm, not just the ones in ``arm_labels``,
    so which qids clear the gate doesn't move when the cache directory gains an arm.
    ``n_scored`` counts the qids that produced at least one comparison row.
    """
    paired_scores: list[PairedScore] = []
    n_scored = 0
    for qid in qids:
        question = working.questions.get(qid)
        gt = working.ground_truths.get(qid)
        if question is None or gt is None:
            continue

        arm_inputs = _arm_score_inputs(working, qid, question)
        if sum(1 for report, _payload in arm_inputs.values() if report is not None) < 2:
            continue

        scores = score_arm_for_qid([(label, *arm_inputs[label]) for label in arm_labels], gt)
        if scores:
            n_scored += 1
        paired_scores.extend(scores)
    return paired_scores, n_scored


def _score_run_payload(
    metadata: dict,
    stats: list[PairedStats],
    paired_scores: list[PairedScore],
) -> dict:
    """The machine-readable score-run record written alongside the markdown summary."""
    return {
        "metadata": metadata,
        "stats": [
            {
                "metric": s.metric,
                "question_type": s.question_type,
                "n": s.n,
                "mean_delta": s.mean_delta,
                "bootstrap_ci_low": s.bootstrap_ci_low,
                "bootstrap_ci_high": s.bootstrap_ci_high,
                "sign_test_p": s.sign_test_p,
                "wilcoxon_p": s.wilcoxon_p,
                "higher_is_better": s.higher_is_better,
            }
            for s in stats
        ],
        "paired_scores": [
            {
                "qid": s.qid,
                "question_type": s.question_type,
                "metric": s.metric,
                "comparison": s.comparison,
                "score_stack": s.score_stack,
                "score_stack_aug": s.score_stack_aug,
                "score_median": s.score_median,
                "delta": s.delta,
                "higher_is_better": s.higher_is_better,
            }
            for s in paired_scores
        ],
    }


def _stage_score(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
) -> Path:
    """Build paired scores, aggregate, render summary, write run + summary files.

    Uses per-comparison N: each pairwise comparison only requires the two arms in
    that comparison to have succeeded for a qid. A qid is included if at least 2
    arms succeeded (enabling at least one comparison). This avoids collapsing N to
    the 5-way intersection of all arms.
    """
    # Union of all qids that have ANY arm payload.
    qid_set: set[int] = set()
    for arm_key in _SCORING_ARM_CACHE_KEYS.values():
        qid_set.update(working.stacker_payloads.get(arm_key, {}).keys())

    arm_labels = _scoring_arm_labels(
        has_pdf_arms=bool(working.stacker_payloads.get(ARM_PDF_MIN1))
        or bool(working.stacker_payloads.get(ARM_PDF_MIN2)),
        has_mean_arm=bool(working.stacker_payloads.get(ARM_MEAN)),
    )
    paired_scores, n_scored = _build_paired_scores(working, sorted(qid_set), arm_labels)

    stats = aggregate_paired(paired_scores, n_bootstrap=5000, seed=args.seed)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    metadata = {
        "timestamp": timestamp,
        "n_questions": n_scored,
        "tournaments": ", ".join(args.tournaments),
        "resolved_after": args.resolved_after,
    }
    summary_md = render_summary_markdown(stats, paired_scores, metadata)
    cache.write_score_run(timestamp, _score_run_payload(metadata, stats, paired_scores))
    return cache.write_score_summary(timestamp, summary_md)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def _hydrate_research_artifacts(
    cache: AblationCache,
    working: WorkingSet,
    qid: int,
    *,
    spend: SpendReport | None,
) -> None:
    """Load one qid's research + pruned-research artifacts into the working set.

    Only the SANITIZED blob flows downstream — the raw blob stays on disk for QA-dump
    inspection only. A qid whose prune failed (no cached pruned blob) must NOT be
    eligible for forecast or later stages even when those stages are requested without
    ``--stages prune``; the prune stage drops such qids from ``research_blobs``, and
    leaving them out here mirrors that, so re-running downstream stages from cache
    produces the same working set.
    """
    if cache.read_research(qid) is not None and spend is not None:
        spend.cached_research_hits += 1
    cached_pruned = cache.read_pruned_research(qid)
    if cached_pruned is not None:
        working.research_blobs[qid] = cached_pruned[0]
        working.prune_metas[qid] = cached_pruned[1]
        if spend is not None:
            spend.cached_prune_hits += 1


def _hydrate_screen_and_forecasters(
    cache: AblationCache,
    working: WorkingSet,
    qid: int,
    *,
    spend: SpendReport | None,
) -> None:
    """Load one qid's leakage verdict + per-model forecaster payloads."""
    verdict = cache.read_leakage_screen(qid)
    if verdict is not None:
        working.leakage_verdicts[qid] = verdict
        if spend is not None:
            spend.cached_screen_hits += 1
    forecaster_payloads = cache.list_forecaster_outputs(qid)
    if forecaster_payloads:
        working.forecaster_payloads[qid] = forecaster_payloads
        if spend is not None:
            spend.cached_forecaster_hits += len(forecaster_payloads)


def _hydrate_stacker_arms(
    cache: AblationCache,
    working: WorkingSet,
    qid: int,
    *,
    spend: SpendReport | None,
    stacker_slug: str | None,
) -> None:
    """Load one qid's payload for every arm that has one on disk."""
    for arm in (ARM_STACK, ARM_STACK_AUG, ARM_PDF, ARM_PDF_MIN1, ARM_PDF_MIN2, ARM_MEDIAN, ARM_MEAN):
        # Only the LLM-stacker arms are slugged; deterministic arms stay shared.
        arm_slug = stacker_slug if arm in (ARM_STACK, ARM_STACK_AUG) else None
        payload = cache.read_stacker_output(qid=qid, arm=arm, stacker_slug=arm_slug)
        if payload is not None:
            working.stacker_payloads.setdefault(arm, {})[qid] = payload
            if spend is not None:
                spend.cached_stacker_hits[arm] = spend.cached_stacker_hits.get(arm, 0) + 1


async def _hydrate_working_set_from_cache(
    cache: AblationCache,
    working: WorkingSet,
    spend: SpendReport | None = None,
    stacker_slug: str | None = None,
) -> None:
    """For score-only paths: load every artifact from disk.

    When ``spend`` is supplied (score-only path), bumps the relevant
    ``cached_*_hits`` fields so the spend report reflects what was loaded
    rather than reading as a hard zero across the board.

    ``stacker_slug`` is applied ONLY to the LLM-stacker arms (stack / stack_aug)
    so the score-only path reads the active run's stacker outputs; deterministic
    arms (pdf / median / mean) read with ``stacker_slug=None`` (shared, unslugged).
    The in-memory ``working.stacker_payloads`` keys stay the plain arm name —
    only the on-disk filename carries the slug — so scoring/summary code is
    untouched.
    """
    await asyncio.sleep(0)
    for qid_raw, entry in cache.read_qids_manifest().items():
        qid = int(qid_raw)
        working.questions[qid] = _build_question_shim_from_manifest_entry(qid, entry)
        working.ground_truths[qid] = _deserialize_ground_truth(entry["ground_truth"])
        _hydrate_research_artifacts(cache, working, qid, spend=spend)
        _hydrate_screen_and_forecasters(cache, working, qid, spend=spend)
        _hydrate_stacker_arms(cache, working, qid, spend=spend, stacker_slug=stacker_slug)


def _filter_working_set_to_qids(working: WorkingSet, qids: list[int]) -> set[int]:
    """Restrict every per-qid attribute of ``working`` to entries in ``qids``.

    Returns the set of requested qids that were NOT found in the manifest
    (so the caller can log them). Without this filter, --qids X --stages
    stack would silently fan out the stacker over the full manifest.
    """
    requested = set(qids)
    for attr in (
        "questions",
        "ground_truths",
        "research_blobs",
        "prune_metas",
        "leakage_verdicts",
        "forecaster_payloads",
    ):
        existing = getattr(working, attr)
        filtered = {qid: v for qid, v in existing.items() if qid in requested}
        setattr(working, attr, filtered)
    # Filter each arm's payload dict within stacker_payloads.
    for arm_key in list(working.stacker_payloads.keys()):
        working.stacker_payloads[arm_key] = {
            qid: v for qid, v in working.stacker_payloads[arm_key].items() if qid in requested
        }
    return requested - set(working.questions.keys())


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


@dataclass(frozen=True)
class _StagePlan:
    """Which stages this invocation runs, which are forced, and the inter-stage pause."""

    requested: set[str]
    forced: set[str]
    sleep_seconds: float

    def wants(self, stage: str) -> bool:
        return stage in self.requested

    def is_forced(self, stage: str) -> bool:
        return stage in self.forced


@dataclass(frozen=True)
class _ArmStage:
    """One aggregation arm's stage: how to announce it and where its payloads land."""

    name: str  # --stages token and log label
    arm: str  # ARM_* key handed to _stage_stack
    report_arm: str  # arm whose payload count the DONE line reports
    # Deterministic arms do zero API work: fixed ~1 min estimate, no inter-stage sleep.
    # None marks the LLM-backed arms, which get a question-count estimate and a pause.
    deterministic_note: str | None


_ARM_STAGES: tuple[_ArmStage, ...] = (
    _ArmStage("stack", ARM_STACK, ARM_STACK, None),
    _ArmStage("stack_aug", ARM_STACK_AUG, ARM_STACK_AUG, None),
    _ArmStage("pdf", ARM_PDF, ARM_PDF_MIN2, "deterministic structured-math"),
    _ArmStage("median", ARM_MEDIAN, ARM_MEDIAN, "deterministic aggregation"),
    _ArmStage("mean", ARM_MEAN, ARM_MEAN, "deterministic aggregation"),
)


def _qids_with_any_arm_payload(working: WorkingSet) -> set[int]:
    """Union of qids carrying a payload for ANY arm.

    Per-comparison N means scoring only needs SOME qids with >= 2 arm payloads, so the
    gate is "any arm at all", not the intersection across arms.
    """
    all_arm_qids: set[int] = set()
    for arm_payloads in working.stacker_payloads.values():
        all_arm_qids.update(arm_payloads.keys())
    return all_arm_qids


def _apply_qids_filter(working: WorkingSet, qids: list[int] | None) -> None:
    """Restrict the working set to ``qids``, logging any the manifest didn't have."""
    if not qids:
        return
    missing = _filter_working_set_to_qids(working, qids)
    if missing:
        logger.error("--qids filter: %d qids not in working set: %s", len(missing), sorted(missing))


async def _run_score_only(
    args: argparse.Namespace, cache: AblationCache, working: WorkingSet, spend: SpendReport
) -> int:
    """``--stages score``: hydrate every artifact from disk and score, without spending.

    Returns the process exit code — 2 when no qid has any arm payload at all, since
    there is nothing to score and the operator asked for exactly that.
    """
    await _hydrate_working_set_from_cache(cache, working, spend=spend, stacker_slug=_active_stacker_slug(args))
    _apply_qids_filter(working, args.qids)
    if not _qids_with_any_arm_payload(working):
        arm_counts = {arm: len(p) for arm, p in working.stacker_payloads.items()}
        logger.error(
            "Cannot run 'score': zero qids have any stacker outputs %s.",
            arm_counts,
        )
        print("ERROR: --stages score: zero qids have any arm payloads.")
        return 2
    summary_path = _stage_score(args, cache, working)
    _print_spend_report(spend, working, summary_path)
    return 0


async def _run_research_stages(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    spend: SpendReport,
    plan: _StagePlan,
) -> None:
    """Run the requested research / prune / screen stages in order."""
    if plan.wants("research"):
        n = len(working.questions)
        # Gemini grounded search ~30s/qid, parallelism = args.concurrency.
        est_seconds = max(30, n * 30 // max(1, args.concurrency))
        logger.info("stage=research START | est wall-clock ~%d min (n=%d)", est_seconds // 60 + 1, n)
        await _stage_research(args, cache, working, force=plan.is_forced("research"), spend=spend)
        logger.info("stage=research DONE | qids_with_blob=%d", len(working.research_blobs))
        await asyncio.sleep(plan.sleep_seconds)

    if plan.wants("prune"):
        n = len(working.research_blobs)
        # Redactor: ~30s/batch × ceil(n/batch_size).
        n_batches = (n + PRUNE_DEFAULT_BATCH_SIZE - 1) // max(1, PRUNE_DEFAULT_BATCH_SIZE) if n else 0
        est_seconds = max(30, n_batches * 30)
        logger.info(
            "stage=prune START | est wall-clock ~%d min (n=%d, batches=%d)", est_seconds // 60 + 1, n, n_batches
        )
        await _stage_prune(args, cache, working, force=plan.is_forced("prune"), spend=spend)
        logger.info(
            "stage=prune DONE | qids_with_sanitized_blob=%d | validation_failures=%d",
            len(working.research_blobs),
            spend.prune_validation_failures,
        )
        await asyncio.sleep(plan.sleep_seconds)

    if plan.wants("screen"):
        n = len(working.research_blobs)
        est_seconds = max(15, n * 10 // max(1, args.concurrency))
        logger.info("stage=screen START | est wall-clock ~%d min (n=%d)", est_seconds // 60 + 1, n)
        await _stage_screen(args, cache, working, force=plan.is_forced("screen"), spend=spend)
        # Verdict dicts have a fixed schema; ``is_leaked`` always present.
        n_leaked = sum(1 for v in working.leakage_verdicts.values() if v["is_leaked"])
        logger.info("stage=screen DONE | leaked=%d clean=%d", n_leaked, len(working.research_blobs))
        await asyncio.sleep(plan.sleep_seconds)


async def _run_qa_iterate_stage(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    plan: _StagePlan,
) -> None:
    """Run the QA-iterate stage, raising in halt mode so the operator reviews first.

    The inter-stage sleep happens BEFORE the halt-mode raise so the pause symmetry
    matches advisory mode and the operator's preferred backoff is honored even when
    halting right after the verifier batch. Halt is strict — it always blocks, even on
    a fully-clean batch, so the QA summary is read before any forecast spend.
    """
    logger.info("stage=qa_iterate START mode=%s", args.qa_iterate_mode)
    outcomes, qa_summary_path = await _stage_qa_iterate(args, cache, working, force=plan.is_forced("qa_iterate"))
    n_clean = sum(1 for o in outcomes.values() if o.final_status == "clean")
    n_rejected = len(outcomes) - n_clean
    await asyncio.sleep(plan.sleep_seconds)
    if args.qa_iterate_mode == "halt" and qa_summary_path is not None:
        raise RuntimeError(
            f"QA iteration halted: {n_rejected} rejects + {n_clean} clean qids. "
            f"Review {qa_summary_path}. To resume after review:\n"
            f"  1. (Optional) edit {cache.root}/manual_rejects.json to override rejects.\n"
            f"  2. Run: --stages forecast,stack,stack_aug,pdf,median,score (note: this skips qa_iterate; "
            f"manual_rejects is only consulted when qa_iterate is in --stages)."
        )


async def _run_forecast_stage(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    spend: SpendReport,
    plan: _StagePlan,
) -> None:
    """Run the forecaster fan-out stage."""
    n = len(working.research_blobs)
    n_forecasters = len(FREE_FORECASTER_MODELS)
    rl_kwargs = _rate_limit_mode_kwargs(args.rate_limit_mode)
    per_forecaster_concurrency = max(1, rl_kwargs["per_forecaster_concurrency"])
    # Forecaster: ~30s/call serially per question, parallelism =
    # (per_question_concurrency × per_forecaster_concurrency).
    per_question_seconds = (n_forecasters * 30) // per_forecaster_concurrency
    global_parallel = max(1, args.concurrency)
    est_seconds = max(60, n * per_question_seconds // global_parallel)
    logger.info(
        "stage=forecast START | est wall-clock ~%d min (n=%d × n_forecasters=%d / "
        "per_forecaster_concurrency=%d / question_concurrency=%d)",
        est_seconds // 60 + 1,
        n,
        n_forecasters,
        per_forecaster_concurrency,
        global_parallel,
    )
    await _stage_forecast(args, cache, working, force=plan.is_forced("forecast"), spend=spend)
    logger.info("stage=forecast DONE | qids=%d", len(working.forecaster_payloads))
    await asyncio.sleep(plan.sleep_seconds)


async def _run_arm_stages(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    spend: SpendReport,
    plan: _StagePlan,
) -> None:
    """Run each requested aggregation arm, in ``_ARM_STAGES`` order."""
    for stage in _ARM_STAGES:
        if not plan.wants(stage.name):
            continue
        n = len(working.forecaster_payloads)
        if stage.deterministic_note is None:
            est_seconds = max(30, n * 30 // max(1, args.concurrency))
            logger.info("stage=%s START | est wall-clock ~%d min (n=%d)", stage.name, est_seconds // 60 + 1, n)
        else:
            logger.info("stage=%s START | est wall-clock ~1 min (n=%d, %s)", stage.name, n, stage.deterministic_note)
        await _stage_stack(args, cache, working, arm=stage.arm, force=plan.is_forced(stage.name), spend=spend)
        logger.info("stage=%s DONE | qids=%d", stage.name, len(working.stacker_payloads.get(stage.report_arm, {})))
        if stage.deterministic_note is None:
            await asyncio.sleep(plan.sleep_seconds)


def _run_score_stage(args: argparse.Namespace, cache: AblationCache, working: WorkingSet) -> Path | None:
    """Score the run, or return None (after a WARN) when no arm produced any payload."""
    logger.info("stage=score START")
    if not _qids_with_any_arm_payload(working):
        arm_counts = {arm: len(p) for arm, p in working.stacker_payloads.items()}
        logger.warning("score | no qids have any arm payloads %s; skipping", arm_counts)
        return None
    summary_path = _stage_score(args, cache, working)
    logger.info("stage=score DONE | summary=%s", summary_path)
    return summary_path


async def run_ablation(args: argparse.Namespace) -> int:
    """Top-level orchestrator. Returns process exit code (0 OK, 1 partial, 2 fatal config)."""
    if args.rate_limit_mode == "patient" and args.concurrency > 1:
        # patient mode promises "concurrency=1" but --concurrency is the
        # per-question knob (default 4) and stacks multiplicatively with
        # per_forecaster_concurrency. Clamping here keeps the doc-comment
        # honest at 50q+ where free-tier 429s correlate across providers.
        logger.warning(
            "rate_limit_mode=patient overrides --concurrency=%d to 1 (avoids 4-way "
            "free-tier flooding; pass --rate-limit-mode gentle for parallel question fan-out)",
            args.concurrency,
        )
        args.concurrency = 1

    cache = AblationCache(args.cache_dir)
    working = WorkingSet()
    spend = SpendReport()

    forced_explicit = set(args.force_stages)
    forced = _expand_forced_stages(forced_explicit)
    if forced != forced_explicit:
        logger.info("forced stages (after cascade): %s", sorted(forced))
    plan = _StagePlan(requested=set(args.stages), forced=forced, sleep_seconds=args.per_question_sleep)

    if plan.requested == {"score"}:
        return await _run_score_only(args, cache, working, spend)

    # Full pipeline (or any subset that includes upstream stages).
    if plan.wants("fetch"):
        logger.info("stage=fetch START")
        await _stage_fetch(args, cache, working)
        logger.info("stage=fetch DONE | qids=%d", len(working.questions))
    else:
        await _hydrate_working_set_from_cache(cache, working, stacker_slug=_active_stacker_slug(args))
        _apply_qids_filter(working, args.qids)

    await _run_research_stages(args, cache, working, spend=spend, plan=plan)

    if args.qa_research:
        qa_path = _stage_qa_research_dump(args, cache, working)
        print(f"QA research dump written to: {qa_path}")
        _print_spend_report(spend, working, summary_path=None)
        return 0

    if plan.wants("qa_iterate"):
        await _run_qa_iterate_stage(args, cache, working, plan=plan)

    if plan.wants("forecast"):
        await _run_forecast_stage(args, cache, working, spend=spend, plan=plan)

    await _run_arm_stages(args, cache, working, spend=spend, plan=plan)

    summary_path = _run_score_stage(args, cache, working) if plan.wants("score") else None
    _print_spend_report(spend, working, summary_path)
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _configure_logging(args: argparse.Namespace, cache_dir: Path) -> Path:
    """Configure logging to write to both console and a per-invocation file.

    The audit at ``backtests/ablation/audit_smoke_20260515.md`` flagged a
    38-line smoke log because Python's root logger defaults to WARNING and no
    ``logging.basicConfig`` was called anywhere in ``metaculus_bot/ablation/``.
    This wires up file-archived INFO logging so every stage's per-qid
    diagnostics survive on disk for review.

    ``force=True`` replaces any pre-existing handlers (e.g., ones that other
    modules attached during import) so the configured level actually takes
    effect.
    """
    logs_dir = cache_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    # astimezone() attaches the local zone without shifting the wall clock, so the
    # run-log filename stays local-time (what the operator greps) and tz-aware.
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_{timestamp}.log"

    level = getattr(logging, args.log_level.upper())

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(DEFAULT_CACHE_DIR)
    log_path = _configure_logging(args, cache_dir)
    logger.info("ablation run starting; log file: %s", log_path)
    # Suppress aiohttp "Unclosed client session" warnings from litellm's
    # cached HTTP transport (1-hour TTL) — same fix used by backtest.py and
    # community_benchmark.py. Once-per-process noise, no FD leak.
    enable_aiohttp_session_autoclose()
    return asyncio.run(run_ablation(args))


if __name__ == "__main__":
    sys.exit(main())
