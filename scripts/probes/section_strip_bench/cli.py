"""The command line: plan, estimate, the spend gate, the paid run, and the offline rescore."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from metaculus_bot.credit_telemetry import drain_litellm_callbacks, install_role_spend_tracker
from scripts.probes.section_strip_bench.bundle import ARMS, FULL_ARM, load_bench_questions, read_jsonl
from scripts.probes.section_strip_bench.plan import Estimate, PlanItem, build_plan, estimate_spend, print_plan
from scripts.probes.section_strip_bench.report import CallRow, aggregate, render_markdown
from scripts.probes.section_strip_bench.run import (
    SpendMeter,
    build_model_call,
    build_parser_llm,
    parser_ledger_usd,
    personal_key_only_environment,
    run_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAIRS = REPO_ROOT / "scratch" / "cost_pass_2026-09-09" / "v1_vs_v2" / "pairs.jsonl"
DEFAULT_PERF_JSON = REPO_ROOT / "scratch" / "residual_2026-09-09" / "perf_all_tagged.json"
DEFAULT_ARCHIVE_DIR = REPO_ROOT / "backtests" / "research_archive" / "latest"
DEFAULT_OUT_ROOT = REPO_ROOT / "scratch" / "probes"

DEFAULT_MODEL = "meta/muse-spark-1.3-contributor"
# The listed rates at https://openrouter.ai/api/v1/models for the default model; --price-in / --price-out for another.
DEFAULT_PRICE_IN_USD_PER_M = 0.10
DEFAULT_PRICE_OUT_USD_PER_M = 0.20
DEFAULT_SEEDS = 3
DEFAULT_MAX_SPEND_USD = 10.0
DEFAULT_OUTPUT_TOKENS = 800
DEFAULT_CONCURRENCY = 6


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired section-strip bench over the archived gap-fill bundles. SPENDS MONEY on the personal key."
    )
    parser.add_argument(
        "--i-accept-spend", action="store_true", help="Required to run: billed calls on OPENROUTER_API_KEY."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and the estimate, call nothing.")
    parser.add_argument("--rescore", type=Path, help="Rebuild results.json and SUMMARY.md from a run dir; no calls.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"OpenRouter slug without the prefix (default {DEFAULT_MODEL})."
    )
    parser.add_argument("--parser-model", default=None, help="Salvage-parser slug (default: the same as --model).")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="Replicates per (question, arm).")
    parser.add_argument(
        "--questions", type=int, nargs="+", help="Question ids to bench (default: every resolved pair)."
    )
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS), help="Arms to run; full is required.")
    parser.add_argument("--max-spend-usd", type=float, default=DEFAULT_MAX_SPEND_USD)
    parser.add_argument(
        "--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS, help="Estimate's completion tokens per call."
    )
    parser.add_argument(
        "--price-in", type=float, default=DEFAULT_PRICE_IN_USD_PER_M, help="USD per million prompt tokens."
    )
    parser.add_argument(
        "--price-out", type=float, default=DEFAULT_PRICE_OUT_USD_PER_M, help="USD per million completion tokens."
    )
    parser.add_argument(
        "--reasoning-effort", default=None, help="OpenRouter reasoning effort; unset keeps the model default."
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--perf-json", type=Path, default=DEFAULT_PERF_JSON)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Run dirs are created under this.")
    args = parser.parse_args(argv)
    if FULL_ARM not in args.arms:
        parser.error(f"--arms must include {FULL_ARM!r}: the deltas are full minus arm")
    return args


@contextmanager
def _run_logging(run_dir: Path) -> Iterator[None]:
    """Pipeline INFO lines (extraction rungs, CDF repairs) go to the run's log; the console keeps warnings.

    Handlers are attached explicitly rather than through ``basicConfig``, which is a silent no-op once any
    handler exists on the root logger, and detached on exit so nothing outlives the run.
    """
    root = logging.getLogger()
    file_handler = logging.FileHandler(run_dir / "bench.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    previous_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console)
    try:
        yield
    finally:
        root.removeHandler(file_handler)
        root.removeHandler(console)
        file_handler.close()
        root.setLevel(previous_level)


def write_results(run_dir: Path, rows: Sequence[CallRow], run_meta: dict[str, Any]) -> None:
    results = aggregate(rows, run_meta["questions"], arms=run_meta["arms"], bootstrap_seed=run_meta["bootstrap_seed"])
    if run_meta.get("parser_usd") is not None:
        results["spend"]["parser_usd"] = run_meta["parser_usd"]
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = render_markdown(results, model=run_meta["model"], seeds=run_meta["seeds"])
    (run_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")


def rescore(run_dir: Path) -> None:
    run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    rows = [CallRow(**record) for record in read_jsonl(run_dir / "calls.jsonl")]
    write_results(run_dir, rows, run_meta)
    print(f"Rescored {len(rows)} calls into {run_dir / 'results.json'} and {run_dir / 'SUMMARY.md'}")


def _run_meta(args: argparse.Namespace, plan: Sequence[PlanItem], estimate: Estimate) -> dict[str, Any]:
    return {
        "model": args.model,
        "parser_model": args.parser_model or args.model,
        "seeds": args.seeds,
        "arms": list(args.arms),
        "bootstrap_seed": args.bootstrap_seed,
        "reasoning_effort": args.reasoning_effort,
        "max_spend_usd": args.max_spend_usd,
        "started_utc": datetime.now(UTC).isoformat(),
        "estimate": estimate.as_dict(),
        "questions": [item.question.summary() for item in plan if item.arm == FULL_ARM and item.seed == 1],
    }


async def run_bench(args: argparse.Namespace, plan: Sequence[PlanItem], estimate: Estimate, run_dir: Path) -> None:
    """The paid path: personal key only, the ledger installed, rows appended as they land, results at the end."""
    api_key = personal_key_only_environment()
    install_role_spend_tracker()
    run_meta = _run_meta(args, plan, estimate)
    (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    meter = SpendMeter(cap_usd=args.max_spend_usd)
    calls_path = run_dir / "calls.jsonl"

    def _append(row: CallRow) -> None:
        with calls_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(row)) + "\n")

    with _run_logging(run_dir):
        rows = await run_plan(
            plan,
            call=build_model_call(args.model, api_key, reasoning_effort=args.reasoning_effort),
            parser_llm=build_parser_llm(run_meta["parser_model"]),
            model_name=args.model,
            meter=meter,
            concurrency=args.concurrency,
            on_row=_append,
        )
        await drain_litellm_callbacks()
    parser_usd = parser_ledger_usd()
    run_meta.update(
        finished_utc=datetime.now(UTC).isoformat(),
        forecaster_usd=meter.forecaster_usd,
        parser_usd=parser_usd,
        measured_usd=meter.forecaster_usd + parser_usd,
        cap_hit=meter.exhausted,
    )
    (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    write_results(run_dir, rows, run_meta)
    cap_note = " (CAP HIT, remaining calls skipped)" if meter.exhausted else ""
    print(
        f"Done: {dict(Counter(row.status for row in rows))}; measured spend ${run_meta['measured_usd']:.4f}{cap_note}"
    )
    print(f"Results: {run_dir / 'SUMMARY.md'}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rescore is not None:
        rescore(args.rescore)
        return 0
    questions = load_bench_questions(
        args.pairs, args.perf_json, args.archive_dir, only=set(args.questions) if args.questions else None
    )
    plan = build_plan(questions, args.arms, args.seeds, model=args.model)
    estimate = estimate_spend(plan, output_tokens=args.output_tokens, price_in=args.price_in, price_out=args.price_out)
    print_plan(plan, estimate, model=args.model, cap_usd=args.max_spend_usd)
    if args.dry_run:
        return 0
    if not args.i_accept_spend:
        print("Refusing to run: this bench makes billed calls on the personal key. Re-run with --i-accept-spend.")
        return 2
    if estimate.total_usd > args.max_spend_usd:
        print(
            f"Aborting before the first call: the estimate ${estimate.total_usd:.2f} exceeds --max-spend-usd {args.max_spend_usd:.2f}."
        )
        return 2
    run_dir = args.out_root / f"section_strip_bench_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    asyncio.run(run_bench(args, plan, estimate, run_dir))
    return 0
