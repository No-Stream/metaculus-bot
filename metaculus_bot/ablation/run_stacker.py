"""Per-arm stacker runner for the probabilistic-tools ablation benchmark.

Reads cached forecaster rationales, runs the tool-runner (per-rationale
"Computed quantities" + cross-model aggregation), then invokes
``stacking.run_stacking_*``. Two arms differ only by the
``PROBABILISTIC_TOOLS_ENABLED`` env-var state at the moment we call the
tool-runner functions:

* ARM_STACK — flag explicitly unset; both ``run_tools_for_forecaster`` and
  ``build_cross_model_aggregation`` early-return ``""``.
* ARM_STACK_AUG — flag set to ``"1"``; both runners produce real markdown that
  gets piped into the stacker prompt.

Caches per ``(qid, arm)``. On primary-stacker failure, falls back to a
secondary stacker LLM. On both-fail, caches a ``success=False`` payload
and returns; the batch wrapper continues.

Reads canonical (qid, model_slug) cache entries written by
``metaculus_bot.ablation.forecasters`` and dispatches to
``metaculus_bot.stacking.run_stacking_*``.

Default stacker is ``openrouter/openai/gpt-5.5`` routed through
``build_llm_with_openrouter_fallback`` so the Metaculus-donated OpenRouter
key absorbs cost ahead of the operator's paid key. The wrapper handles
donated→paid fallback on credit/auth/allowed-providers errors internally,
so the outer primary→fallback chain in ``run_stacker_for_arm`` is a
defense-in-depth backstop rather than a primary cost-control mechanism.

Cost note (rough order of magnitude — for the operator's mental model only):

* gpt-5.5 with ``reasoning={"effort": "high"}`` runs ~$0.05–0.10 per stacker call.
* 20-question intermediate sweep (40 stacker calls) ≈ $2–4 if everything pays.
* 60-question medium sweep (120 stacker calls) ≈ $6–12 worst case.
* In practice the donated key absorbs almost everything, so the actual
  paid spend is usually a small fraction of the worst-case figure.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

from metaculus_bot import stacking, tool_runner
from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.ablation.env import FEATURE_FLAG_ENV, probabilistic_tools_enabled
from metaculus_bot.ablation.forecasters import (
    deserialize_prediction_value,
    question_type_for_serialization,
    serialize_prediction_value,
)
from metaculus_bot.ablation.window_patch import patched_window_for_question
from metaculus_bot.constants import STACKER_FALLBACK_SOFT_DEADLINE, STACKER_SOFT_DEADLINE
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.numeric_utils import bound_messages

logger: logging.Logger = logging.getLogger(__name__)

ARM_STACK = "stack"  # LLM stacker, rationale only, no probability-math tools
ARM_STACK_AUG = "stack_aug"  # LLM stacker, rationale + computed quantities + cross-model aggregation (augmented)
ARM_PDF = "pdf"  # deterministic structured-math aggregation, no LLM (see metaculus_bot.ablation.run_pdf)
ARM_PDF_MIN1 = "pdf_min1"  # pdf arm with min_forecasters=1 (any structured output qualifies)
ARM_PDF_MIN2 = "pdf_min2"  # pdf arm with min_forecasters=2 (proper aggregation)
ARM_MEDIAN = (
    "median"  # deterministic median over base-forecaster predictions, no LLM (see metaculus_bot.ablation.run_median)
)

# Default stacker mirrors production ``STACKER_LLM`` from ``llm_configs.py``:
# claude-opus-4.5 as primary (donated key allows it; verified) and gpt-5.5 as
# fallback (production STACKER_FALLBACK_LLM choice — different provider so an
# Anthropic stall doesn't take both attempts down). Both are routed through
# ``build_llm_with_openrouter_fallback`` for the donated→paid key fallback the
# wrapper provides on credit/auth/data-policy errors.
#
# History note: a previous iteration tried gpt-5.5 as primary, but the
# operator's local-`.env` donated key returns 404 "no endpoints available"
# (data-policy guardrail) on every gpt-5.5 request. Production with a
# different `OAI_ANTH_OPENROUTER_KEY` GitHub-secret value works, but local
# couldn't. claude-opus-4.5 sidesteps the issue entirely.
DEFAULT_STACKER_MODEL = "openrouter/anthropic/claude-opus-4.5"
DEFAULT_STACKER_FALLBACK_MODEL = "openrouter/openai/gpt-5.5"
DEFAULT_PARSER_MODEL = "openrouter/openai/gpt-oss-120b:free"

# Reasoning config shared by both stacker LLMs. Anthropic models use
# ``reasoning={"max_tokens": ...}`` (explicit thinking budget — see production
# STACKER_LLM at llm_configs.py:122). OpenAI models use ``reasoning={"effort":
# ...}``. We construct each LLM with its own kwargs so the per-provider
# parameter differences are explicit.
_OPUS_STACKER_KWARGS: dict[str, Any] = {
    "reasoning": {"max_tokens": 32_000},
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 64_000,
    "stream": False,
    "timeout": 480,
    "allowed_tries": 1,
}
_GPT_5_5_STACKER_KWARGS: dict[str, Any] = {
    "reasoning": {"effort": "high"},
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 64_000,
    "stream": False,
    "timeout": 480,
    "allowed_tries": 1,
}


def _build_default_stacker_llm() -> GeneralLlm:
    """Primary ablation stacker (claude-opus-4.5 via donated-key wrapper).

    Mirrors production STACKER_LLM. Anthropic models work cleanly on the
    donated key. ``allowed_tries=1`` so we don't double-bill the donated key
    on transient stalls — the wrapper's donated→paid handling is the safety
    net.
    """
    return build_llm_with_openrouter_fallback(model=DEFAULT_STACKER_MODEL, **_OPUS_STACKER_KWARGS)


def _build_default_fallback_stacker_llm() -> GeneralLlm:
    """Fallback ablation stacker (gpt-5.5).

    Mirrors production STACKER_FALLBACK_LLM. Different provider on purpose —
    if Anthropic is thrashing, retrying against Anthropic rarely recovers.
    Goes through the donated-key wrapper too; if the donated key data-policy
    blocks gpt-5.5 (operator-specific), the wrapper's fallback to the paid
    key catches it.
    """
    return build_llm_with_openrouter_fallback(model=DEFAULT_STACKER_FALLBACK_MODEL, **_GPT_5_5_STACKER_KWARGS)


# Re-exported so existing imports of ``probabilistic_tools_enabled`` and
# ``FEATURE_FLAG_ENV`` from this module continue to work. The canonical
# definition lives in ``metaculus_bot.ablation.env`` to keep
# ``forecasters.py`` and this module from importing each other.
__all__ = [
    "ABLATION_MIN_FORECASTERS",
    "ARM_MEDIAN",
    "ARM_PDF",
    "ARM_PDF_MIN1",
    "ARM_PDF_MIN2",
    "ARM_STACK_AUG",
    "ARM_STACK",
    "DEFAULT_PARSER_MODEL",
    "DEFAULT_STACKER_FALLBACK_MODEL",
    "DEFAULT_STACKER_MODEL",
    "FEATURE_FLAG_ENV",
    "probabilistic_tools_enabled",
    "run_stacker_batch",
    "run_stacker_for_arm",
]

# Minimum forecasters required to run the stacker for an ablation arm. The
# production path enforces ``MIN_FORECASTERS_TO_PUBLISH=3``; we relax to 2
# here because the alternative is dropping the question for both arms (which
# defeats the paired-comparison design when one arm could still produce a
# valid stacker output).
ABLATION_MIN_FORECASTERS = 2

# Approximate per-prompt character ceiling for stacker calls. ~4 chars/token
# is the standard rule of thumb for English; both Opus 4.5 (200k) and gpt-5.5
# (128k) need headroom for the model's reasoning + response tokens, so we
# use 128k - 30k headroom = 98k tokens => ~390k chars as the guard threshold.
# Going over surfaces as a 400 ``context_length_exceeded`` on the primary,
# fallback inherits the same prompt and fails too — the question is dropped
# from both arms. Truncating per-rationale (tail-preserving so the
# conclusion survives) is far cheaper than losing both arms.
APPROX_STACKER_CHAR_LIMIT = 4 * (128_000 - 30_000)


def _truncate_long_rationales(base_texts: list[str], char_limit: int) -> list[str]:
    """Tail-truncate ``base_texts`` to fit total within ``char_limit``.

    Preserves the LAST chars of each rationale because that's where
    forecasters typically put their final probability + summary
    judgement. A head-preserving truncation would cut off the conclusion
    which is the most stacker-relevant part.
    """
    if not base_texts:
        return base_texts
    total = sum(len(t) for t in base_texts)
    if total <= char_limit:
        return base_texts
    per_rationale_limit = max(1, char_limit // len(base_texts))
    return [t[-per_rationale_limit:] if len(t) > per_rationale_limit else t for t in base_texts]


# ---------------------------------------------------------------------------
# Window-patch reentrancy lock
# ---------------------------------------------------------------------------
#
# ``patched_window_for_question`` is a global monkey-patch with a RuntimeError
# guard against nested entry. When ``run_stacker_batch`` runs concurrent
# per-question stacker calls, two concurrent enters would collide. Serializing
# the patched section under a module-level asyncio.Lock keeps each stacker
# call inside its own patched region while still letting other batch work
# (cache reads, gathers) overlap across questions. Mirrors the equivalent
# pattern in ``metaculus_bot.ablation.forecasters``.
_WINDOW_PATCH_LOCK: asyncio.Lock | None = None


def _get_window_patch_lock() -> asyncio.Lock:
    global _WINDOW_PATCH_LOCK
    if _WINDOW_PATCH_LOCK is None:
        _WINDOW_PATCH_LOCK = asyncio.Lock()
    return _WINDOW_PATCH_LOCK


# ---------------------------------------------------------------------------
# Forecaster filtering
# ---------------------------------------------------------------------------


def _is_finite_prediction(prediction_value: Any) -> bool:
    """Return True iff every numeric field inside ``prediction_value`` is finite.

    A free-tier forecaster whose parser produces NaN (e.g. on ``Probability:
    ?%``) used to slip through the prediction_value=None / errors=[] filter
    because Python propagates NaN through ``min``/``max`` clamps. Reject
    NaN/inf values explicitly so they don't poison the cross-model
    aggregator and bootstrap CIs downstream.
    """
    import math  # noqa: PLC0415  - keep import resilient against formatter strip

    if not isinstance(prediction_value, dict):
        return False
    payload_type = prediction_value.get("type")
    if payload_type == "binary":
        prob = prediction_value.get("prob")
        return isinstance(prob, (int, float)) and math.isfinite(prob)
    if payload_type == "multiple_choice":
        options = prediction_value.get("options") or []
        for option in options:
            prob = option.get("probability") if isinstance(option, dict) else None
            if not isinstance(prob, (int, float)) or not math.isfinite(prob):
                return False
        return True
    if payload_type == "numeric":
        cdf = prediction_value.get("cdf_probabilities") or []
        if not cdf:
            return False
        for value in cdf:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                return False
        return True
    return False


def _surviving_forecasters(forecaster_payloads: dict[str, dict]) -> dict[str, dict]:
    """Drop forecasters with ``prediction_value=None``, errors, or NaN/inf values.

    A failed forecaster (parse error, LLM timeout, etc.) writes its payload
    with ``prediction_value=None`` and ``errors=[...]``; we filter those out
    before stacking. Also drops payloads whose numeric content carries
    NaN/inf — see ``_is_finite_prediction``.
    """
    surviving: dict[str, dict] = {}
    for slug, payload in forecaster_payloads.items():
        if payload.get("prediction_value") is None:
            continue
        if payload.get("errors"):
            continue
        if not _is_finite_prediction(payload["prediction_value"]):
            continue
        surviving[slug] = payload
    return surviving


# ---------------------------------------------------------------------------
# Stacker dispatch
# ---------------------------------------------------------------------------


async def _dispatch_stacker(
    *,
    question: MetaculusQuestion,
    research: str,
    base_texts: list[str],
    stacker_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    aggregated_tool_output: str | None,
) -> tuple[Any, str]:
    """Call the right ``stacking.run_stacking_*`` based on question type.

    ``aggregated_tool_output`` is forwarded directly to ``stacking.run_stacking_*``.
    """
    if isinstance(question, BinaryQuestion):
        return await stacking.run_stacking_binary(
            stacker_llm,
            parser_llm,
            question,
            research,
            base_texts,
            aggregated_tool_output=aggregated_tool_output,
        )
    if isinstance(question, MultipleChoiceQuestion):
        return await stacking.run_stacking_mc(
            stacker_llm,
            parser_llm,
            question,
            research,
            base_texts,
            aggregated_tool_output=aggregated_tool_output,
        )
    if isinstance(question, NumericQuestion):
        # Function-scoped imports survive Ruff's unused-import pass when added
        # in the same edit as their usage; see AGENTS.md note on ``main.py``'s
        # function-scoped imports for the same reason.
        from metaculus_bot.exceptions import UnitMismatchError  # noqa: PLC0415  # function-scoped: see AGENTS.md
        from metaculus_bot.numeric_pipeline import (  # noqa: PLC0415  # function-scoped: see AGENTS.md
            build_numeric_distribution,
            sanitize_percentiles,
        )
        from metaculus_bot.numeric_validation import (
            detect_unit_mismatch,  # noqa: PLC0415  # function-scoped: see AGENTS.md
        )

        upper_msg, lower_msg = bound_messages(question)
        # Production at main.py:436-468: ``run_stacking_numeric`` returns
        # ``tuple[list[Percentile], str]``, then the percentile list is piped
        # through sanitize → unit-mismatch guard → build_numeric_distribution
        # to produce the final NumericDistribution. Mirror that here so the
        # ablation cache's full-CDF serializer (which requires NumericDistribution)
        # gets the right input shape.
        perc_list, meta_text = await stacking.run_stacking_numeric(
            stacker_llm,
            parser_llm,
            question,
            research,
            base_texts,
            lower_msg,
            upper_msg,
            aggregated_tool_output=aggregated_tool_output,
        )
        percentile_list, zero_point = sanitize_percentiles(list(perc_list), question)
        mismatch, reason = detect_unit_mismatch(percentile_list, question)
        if mismatch:
            raise UnitMismatchError(
                f"Unit mismatch likely; {reason}. Values: {[float(p.value) for p in percentile_list]}"
            )
        prediction = build_numeric_distribution(percentile_list, question, zero_point)
        return prediction, meta_text
    raise ValueError(f"Unsupported question type for stacking: {type(question).__name__}")


# ---------------------------------------------------------------------------
# M3 — Tertiary MEDIAN fallback when both stackers fail
# ---------------------------------------------------------------------------


async def _median_fallback_prediction(
    question: MetaculusQuestion,
    surviving: dict[str, dict],
) -> Any:
    """Return a MEDIAN aggregation of surviving forecaster predictions.

    Mirrors production at main.py:1304-1322: when both primary and
    fallback stackers fail, MEDIAN-aggregate the per-forecaster
    predictions so the question still gets a publishable forecast.
    Per question type:

    * Binary: median of probabilities.
    * Multiple choice: per-option median, renormalized.
    * Numeric: pointwise median of CDFs (via combine_numeric_predictions).

    Marks the failure mode in the caller's logs; no internal logging
    here so the surrounding context (qid, arm) appears in one place.
    """
    from metaculus_bot.aggregation_strategies import (  # noqa: PLC0415
        AggregationStrategy,
        combine_binary_predictions,
        combine_multiple_choice_predictions,
        combine_numeric_predictions,
    )

    deserialized = [
        deserialize_prediction_value(payload["prediction_value"], question) for payload in surviving.values()
    ]
    # Cooperative yield so flake8-async ASYNC910 sees a checkpoint on the
    # binary/MC sync-aggregation branches without changing observable
    # behavior. The numeric branch already awaits inside combine_numeric_predictions.
    await asyncio.sleep(0)
    if isinstance(question, BinaryQuestion):
        return combine_binary_predictions([float(v) for v in deserialized], AggregationStrategy.MEDIAN)
    if isinstance(question, MultipleChoiceQuestion):
        return combine_multiple_choice_predictions(deserialized, AggregationStrategy.MEDIAN)
    if isinstance(question, NumericQuestion):
        return await combine_numeric_predictions(deserialized, question, AggregationStrategy.MEDIAN)
    raise ValueError(f"Unsupported question type for median fallback: {type(question).__name__}")


# ---------------------------------------------------------------------------
# Per-question runner
# ---------------------------------------------------------------------------


async def run_stacker_for_arm(
    question: MetaculusQuestion,
    research_blob: str,
    forecaster_payloads: dict[str, dict],
    arm: str,
    cache: AblationCache,
    *,
    stacker_llm: GeneralLlm | None = None,
    fallback_stacker_llm: GeneralLlm | None = None,
    parser_llm: GeneralLlm | None = None,
    force: bool = False,
) -> dict:
    """Run the stacker for one arm of one question. Cached per ``(qid, arm)``.

    Steps:
    1. Cache check (``read_stacker_output(qid, arm)``). On hit and not ``force``, return as-is.
    2. Filter to surviving forecasters (``prediction_value`` set, no errors). Need
       >= ``ABLATION_MIN_FORECASTERS`` (2) to proceed; below that, cache an
       error payload and return.
    3. Set ``PROBABILISTIC_TOOLS_ENABLED=1`` for arm B; explicitly unset for arm A.
    4. Per surviving forecaster: append ``\\n\\n## Computed quantities\\n<md>``
       to its rationale only when ``run_tools_for_forecaster`` returns non-empty
       (matches production at ``main.py:1131``).
    5. Call ``build_cross_model_aggregation`` once per question over all
       surviving rationales + deserialized prediction values.
    6. Inside ``patched_window_for_question(question)`` (serialized via a
       module-level asyncio.Lock so concurrent stacker batch calls don't
       collide on the global monkey-patch) and the env context manager,
       dispatch to the right ``run_stacking_*``. On primary failure, try
       the fallback LLM. On both fail, cache an error payload.
    7. Cache the success or error payload, return it.

    Note on the min-forecasters threshold: production enforces 3, but the
    ablation relaxes to 2 because the comparison is paired across arms — an
    insufficient-forecaster question fails for both arms, so the relaxation
    only opens a window where both arms might produce a valid stacker output
    that wouldn't have been compared otherwise.
    """
    qid = question.id_of_question
    assert qid is not None, "run_stacker_for_arm requires question.id_of_question"

    if not force:
        cached = cache.read_stacker_output(qid=qid, arm=arm)
        if cached is not None:
            # Yield once so flake8-async (ASYNC910) sees a guaranteed checkpoint
            # on this early-return branch. The return path is otherwise sync.
            await asyncio.sleep(0)
            return cached

    surviving = _surviving_forecasters(forecaster_payloads)
    if len(surviving) < ABLATION_MIN_FORECASTERS:
        # Schema invariant: every stacker payload carries ``stacker_model_used``.
        # ``None`` here means "stacker never ran" (insufficient inputs); other
        # branches set ``"primary"``, ``"fallback"``, or leave ``None`` after a
        # both-stackers-failed exception. Keeping the key present in every
        # payload lets cli.py:_stage_stack read it via direct subscript.
        error_payload = {
            "success": False,
            "arm": arm,
            "reason": "insufficient_forecasters",
            "stacker_model_used": None,
            "n_forecasters_used": len(surviving),
            "ran_at": datetime.now().isoformat(),
            "tools_enabled_at_runtime": arm == ARM_STACK_AUG,
            "errors": [],
        }
        cache.write_stacker_output(qid=qid, arm=arm, payload=error_payload)
        await asyncio.sleep(0)
        return error_payload

    # Provide defaults for any LLMs the caller didn't pass. Tests always pass
    # all three; production callers may rely on the defaults below. The
    # default stacker LLMs go through ``build_llm_with_openrouter_fallback``
    # so the Metaculus-donated key absorbs cost ahead of the operator's paid
    # key; see module docstring for cost expectations.
    if stacker_llm is None:
        stacker_llm = _build_default_stacker_llm()
    if fallback_stacker_llm is None:
        fallback_stacker_llm = _build_default_fallback_stacker_llm()
    if parser_llm is None:
        parser_llm = GeneralLlm(model=DEFAULT_PARSER_MODEL, allowed_tries=1)

    enable_tools = arm == ARM_STACK_AUG

    with probabilistic_tools_enabled(enable_tools):
        # Per-rationale "Computed quantities" augmentation. The runner
        # internally checks the env flag and returns "" when off — so for
        # arm A this loop produces no augmentations.
        per_forecaster_md: dict[str, str] = {}
        augmented_base_texts: list[str] = []
        deserialized_values: list[Any] = []
        for slug, payload in surviving.items():
            raw_rationale = payload["reasoning"]
            # Forecaster payload schema is fixed (forecasters.run_forecasters_batch);
            # ``model`` is always present. Direct subscript surfaces drift.
            forecaster_id = payload["model"]
            computed_md = tool_runner.run_tools_for_forecaster(
                question=question,
                rationale=raw_rationale,
                forecaster_id=forecaster_id,
            )
            if computed_md:
                augmented_rationale = f"{raw_rationale}\n\n## Computed quantities\n{computed_md}"
                per_forecaster_md[slug] = computed_md
            else:
                augmented_rationale = raw_rationale
            # Strip the leading "Model: <name>\n\n" tag — production behavior
            # at main.py:375 inside _run_stacking.
            stripped = stacking.strip_model_tag(augmented_rationale)
            augmented_base_texts.append(stripped)

            deserialized_values.append(deserialize_prediction_value(payload["prediction_value"], question))

        # Cross-model aggregation. Receives the *raw* (with-Model-tag) rationales
        # for parsing — the structured-block extractors don't depend on the tag.
        # Production at main.py:798-803 and 890-897 passes raw rationales.
        cross_model_md = tool_runner.build_cross_model_aggregation(
            question=question,
            rationales=[p["reasoning"] for p in surviving.values()],
            prediction_values=deserialized_values,
        )
        # Production passes ``aggregated_tool_output or None``; mirror that.
        aggregated_for_stacker = cross_model_md or None

        # Prompt-size guard: free-tier forecasters can emit 200k+ char
        # rationales, and 4 of them stacked together blow past Claude/GPT
        # context windows. Truncate per-rationale (tail-preserving) when we
        # exceed APPROX_STACKER_CHAR_LIMIT minus the research budget.
        research_budget = len(research_blob) + len(aggregated_for_stacker or "")
        rationale_budget = APPROX_STACKER_CHAR_LIMIT - research_budget
        total_rationale_chars = sum(len(t) for t in augmented_base_texts)
        if total_rationale_chars > max(0, rationale_budget):
            logger.warning(
                "stacker | qid=%s arm=%s | prompt size %d > %d limit; truncating rationales",
                qid,
                arm,
                total_rationale_chars + research_budget,
                APPROX_STACKER_CHAR_LIMIT,
            )
            augmented_base_texts = _truncate_long_rationales(augmented_base_texts, max(1, rationale_budget))

        errors_list: list[str] = []
        stacker_model_used: str | None = None
        result: tuple[Any, str] | None = None

        # Serialize patched-section entry across concurrent stacker batch calls,
        # mirroring ``metaculus_bot.ablation.forecasters``. ``patched_window_for_question``
        # is a global monkey-patch and raises on nested entry; the asyncio.Lock
        # keeps each call inside its own patched region without colliding.
        async with _get_window_patch_lock():
            with patched_window_for_question(question):
                try:
                    # Soft deadline mirrors production at main.py:1243: a stuck
                    # stacker can hold a question for the entire litellm
                    # timeout(480) when allowed_tries=1, blocking every other
                    # question that waits on the global window-patch lock.
                    result = await asyncio.wait_for(
                        _dispatch_stacker(
                            question=question,
                            research=research_blob,
                            base_texts=augmented_base_texts,
                            stacker_llm=stacker_llm,
                            parser_llm=parser_llm,
                            aggregated_tool_output=aggregated_for_stacker,
                        ),
                        timeout=STACKER_SOFT_DEADLINE,
                    )
                    stacker_model_used = "primary"
                except Exception as primary_exc:  # noqa: BLE001 - translated to cached error payload
                    logger.exception("Primary stacker failed for qid=%s arm=%s", qid, arm)
                    errors_list.append(f"primary: {type(primary_exc).__name__}: {primary_exc!r}")
                    try:
                        # Tighter deadline on fallback mirrors production
                        # main.py:1271 — by the time we're falling back,
                        # we're already late on the critical path.
                        result = await asyncio.wait_for(
                            _dispatch_stacker(
                                question=question,
                                research=research_blob,
                                base_texts=augmented_base_texts,
                                stacker_llm=fallback_stacker_llm,
                                parser_llm=parser_llm,
                                aggregated_tool_output=aggregated_for_stacker,
                            ),
                            timeout=STACKER_FALLBACK_SOFT_DEADLINE,
                        )
                        stacker_model_used = "fallback"
                    except Exception as fallback_exc:  # noqa: BLE001 - translated to cached error payload
                        logger.exception("Fallback stacker failed for qid=%s arm=%s", qid, arm)
                        errors_list.append(f"fallback: {type(fallback_exc).__name__}: {fallback_exc!r}")
                        result = None

    if result is None:
        # Tertiary MEDIAN fallback (mirror of main.py:1287-1322): both stackers
        # failed, but we still have surviving forecasters. Median-aggregate
        # them so the question gets a degraded-but-publishable forecast
        # instead of being lost from both arms. The
        # ``stacker_model_used="median_fallback"`` tag lets confounder
        # analysis bucket these separately from the regular primary/fallback
        # outcomes.
        try:
            median_prediction = await _median_fallback_prediction(question, surviving)
            median_payload = {
                "success": True,
                "arm": arm,
                "stacker_prediction": serialize_prediction_value(
                    median_prediction, question_type_for_serialization(question)
                ),
                "stacker_meta_reasoning": "median_fallback: both stackers failed",
                "computed_quantities": per_forecaster_md,
                "cross_model_aggregation": cross_model_md or "",
                "stacker_model_used": "median_fallback",
                "n_forecasters_used": len(surviving),
                "ran_at": datetime.now().isoformat(),
                "tools_enabled_at_runtime": enable_tools,
                "errors": errors_list,
            }
            logger.warning(
                "Median fallback engaged for qid=%s arm=%s after both stackers failed",
                qid,
                arm,
            )
            cache.write_stacker_output(qid=qid, arm=arm, payload=median_payload)
            return median_payload
        except Exception as median_exc:  # noqa: BLE001 - degrade gracefully
            logger.exception("Median fallback failed for qid=%s arm=%s", qid, arm)
            errors_list.append(f"median_fallback: {type(median_exc).__name__}: {median_exc!r}")
            error_payload = {
                "success": False,
                "arm": arm,
                "reason": "stacker_failed",
                "stacker_prediction": None,
                "stacker_meta_reasoning": "",
                "computed_quantities": per_forecaster_md,
                "cross_model_aggregation": cross_model_md or "",
                "stacker_model_used": stacker_model_used,
                "n_forecasters_used": len(surviving),
                "ran_at": datetime.now().isoformat(),
                "tools_enabled_at_runtime": enable_tools,
                "errors": errors_list,
            }
            cache.write_stacker_output(qid=qid, arm=arm, payload=error_payload)
            await asyncio.sleep(0)
            return error_payload

    stacker_prediction, stacker_meta = result
    serialized_prediction = serialize_prediction_value(stacker_prediction, question_type_for_serialization(question))
    # NaN/inf in the stacker output would corrupt cross-model aggregation,
    # bootstrap CIs, and any cached downstream consumer. Treat it the same
    # as both stackers failing — error payload, no cache pollution.
    if not _is_finite_prediction(serialized_prediction):
        logger.error(
            "Stacker emitted non-finite prediction_value for qid=%s arm=%s; recording failure",
            qid,
            arm,
        )
        errors_list.append(f"{stacker_model_used}: stacker output contained NaN/inf")
        error_payload = {
            "success": False,
            "arm": arm,
            "reason": "stacker_nonfinite_output",
            "stacker_prediction": None,
            "stacker_meta_reasoning": stacker_meta,
            "computed_quantities": per_forecaster_md,
            "cross_model_aggregation": cross_model_md or "",
            "stacker_model_used": stacker_model_used,
            "n_forecasters_used": len(surviving),
            "ran_at": datetime.now().isoformat(),
            "tools_enabled_at_runtime": enable_tools,
            "errors": errors_list,
        }
        cache.write_stacker_output(qid=qid, arm=arm, payload=error_payload)
        await asyncio.sleep(0)
        return error_payload

    success_payload = {
        "success": True,
        "arm": arm,
        "stacker_prediction": serialized_prediction,
        "stacker_meta_reasoning": stacker_meta,
        "computed_quantities": per_forecaster_md,
        "cross_model_aggregation": cross_model_md or "",
        "stacker_model_used": stacker_model_used,
        "n_forecasters_used": len(surviving),
        "ran_at": datetime.now().isoformat(),
        "tools_enabled_at_runtime": enable_tools,
        "errors": errors_list,
    }
    cache.write_stacker_output(qid=qid, arm=arm, payload=success_payload)
    await asyncio.sleep(0)
    return success_payload


# ---------------------------------------------------------------------------
# Batch wrapper
# ---------------------------------------------------------------------------


async def run_stacker_batch(
    qid_to_data: dict[int, dict],
    arm: str,
    cache: AblationCache,
    *,
    stacker_llm: GeneralLlm | None = None,
    fallback_stacker_llm: GeneralLlm | None = None,
    parser_llm: GeneralLlm | None = None,
    force: bool = False,
    concurrency: int = 2,
) -> dict[int, dict]:
    """Run the stacker for one arm across many questions.

    Returns ``{qid: payload}``. The same LLMs are reused across questions
    (built once at the call site or here, then passed straight into each
    per-question runner). Per-question failure (insufficient forecasters,
    both stackers down, etc.) is recorded in that question's payload and
    the batch continues.
    """
    if stacker_llm is None:
        stacker_llm = _build_default_stacker_llm()
    if fallback_stacker_llm is None:
        fallback_stacker_llm = _build_default_fallback_stacker_llm()
    if parser_llm is None:
        parser_llm = GeneralLlm(model=DEFAULT_PARSER_MODEL, allowed_tries=1)

    semaphore = asyncio.Semaphore(concurrency)

    async def _one(qid: int, data: dict) -> tuple[int, dict]:
        async with semaphore:
            payload = await run_stacker_for_arm(
                question=data["question"],
                research_blob=data["research"],
                forecaster_payloads=data["forecaster_payloads"],
                arm=arm,
                cache=cache,
                stacker_llm=stacker_llm,
                fallback_stacker_llm=fallback_stacker_llm,
                parser_llm=parser_llm,
                force=force,
            )
        return qid, payload

    tasks = [_one(qid, data) for qid, data in qid_to_data.items()]
    results = await asyncio.gather(*tasks)
    return dict(results)
