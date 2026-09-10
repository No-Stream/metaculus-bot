"""Paired section-strip bench: how much do the two gap-fill sections move a cheap forecaster?

Every archived research bundle ends with two appended gap-fill sections, v1 (``## Targeted Gap-Fill
(second pass)``, about $0.76 a question) and v2 (``## Agentic Research Findings``, about $0.36), and
whether v1 stays for the fall turns on whether its text changes forecasts. For every resolved question
where both ran, this bench forecasts four arms of the same frozen bundle with ONE cheap model (``full``
as published, ``minus_v1``, ``minus_v2``, ``minus_both``), several replicates each, scores each forecast
against the resolution with the repo's scoring functions, and reports the paired full-minus-arm deltas.
No research runs: the bundles come off ``backtests/research_archive/latest/``, the question metadata off
the residual round's tagged dataset, and the only network calls are completions on the OPERATOR'S
PERSONAL OpenRouter key (the donated key is removed from the process environment before any client is
built). The default model, Meta's Muse Spark 1.3 Contributor tier, may train on prompts; the operator
accepted that for this open-source repo, and the OpenRouter account's privacy setting must allow
training providers or every call fails on data policy. Runs are gated like ``gemini_verify.py``: a bare
invocation prints the plan and the estimate and refuses, ``--dry-run`` prints them and exits clean,
``--i-accept-spend`` runs under ``--max-spend-usd``, and ``--rescore <run dir>`` rebuilds the results
from ``calls.jsonl`` offline.

    uv run python -m scripts.probes.section_strip_bench --dry-run
    uv run python -m scripts.probes.section_strip_bench --i-accept-spend
"""

from scripts.probes.section_strip_bench.bundle import (
    ARMS,
    FULL_ARM,
    SECTION_SEPARATOR,
    STRIPPED_ARMS,
    V1_SECTION_HEADER,
    V2_SECTION_HEADER,
    BenchQuestion,
    BundleSections,
    arm_texts,
    build_question,
    load_bench_questions,
    split_bundle,
    typed_resolution,
)
from scripts.probes.section_strip_bench.cli import DEFAULT_MODEL, main, parse_args, rescore
from scripts.probes.section_strip_bench.plan import (
    Estimate,
    PlanItem,
    anchored_clock,
    build_plan,
    estimate_spend,
    render_prompt,
    replicate_nonce,
)
from scripts.probes.section_strip_bench.report import (
    STATUS_API_ERROR,
    STATUS_BUILD_FAILED,
    STATUS_EXTRACTION_FAILED,
    STATUS_SCORED,
    STATUS_SKIPPED_SPEND_CAP,
    STATUS_UNIT_MISMATCH,
    CallRow,
    aggregate,
    render_markdown,
)
from scripts.probes.section_strip_bench.run import (
    ModelReply,
    SpendMeter,
    personal_key_only_environment,
    reply_from_response,
    run_plan,
)
from scripts.probes.section_strip_bench.scoring import (
    peer_scale_factor,
    score_binary,
    score_mc,
    score_numeric,
    score_published,
    score_reply,
)

__all__ = [
    "ARMS",
    "DEFAULT_MODEL",
    "FULL_ARM",
    "SECTION_SEPARATOR",
    "STATUS_API_ERROR",
    "STATUS_BUILD_FAILED",
    "STATUS_EXTRACTION_FAILED",
    "STATUS_SCORED",
    "STATUS_SKIPPED_SPEND_CAP",
    "STATUS_UNIT_MISMATCH",
    "STRIPPED_ARMS",
    "V1_SECTION_HEADER",
    "V2_SECTION_HEADER",
    "BenchQuestion",
    "BundleSections",
    "CallRow",
    "Estimate",
    "ModelReply",
    "PlanItem",
    "SpendMeter",
    "aggregate",
    "anchored_clock",
    "arm_texts",
    "build_plan",
    "build_question",
    "estimate_spend",
    "load_bench_questions",
    "main",
    "parse_args",
    "peer_scale_factor",
    "personal_key_only_environment",
    "render_markdown",
    "render_prompt",
    "replicate_nonce",
    "reply_from_response",
    "rescore",
    "run_plan",
    "score_binary",
    "score_mc",
    "score_numeric",
    "score_published",
    "score_reply",
    "split_bundle",
    "typed_resolution",
]
