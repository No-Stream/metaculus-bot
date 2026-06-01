"""Forecaster fan-out runner for the probabilistic-tools ablation benchmark.

Runs N free-model forecasters per question ONCE against pre-cached Gemini
research. Caches per-(qid, model) on disk via :class:`AblationCache`. The
question-anchored window patch wraps the prediction call so prompts don't
leak today's date — important for resolved-question backtests where any
"today" reference would tip off the forecaster that the question already
resolved.

The same per-forecaster outputs feed BOTH stacker arms (tools-on, tools-off)
in the downstream A/B benchmark, so we run forecasters once per question per
backtest and cache aggressively. Errors are recorded in the payload, never
re-raised — a single forecaster crash shouldn't poison the batch.

Design choices:

* ``_make_prediction`` is called directly (not the soft-deadline wrapper) so
  test mocks can replace it cleanly. The runner wraps each call with a
  per-(qid, model) timer and exception capture.
* Aggregation is :class:`AggregationStrategy.MEAN` because we want raw
  per-model predictions, not stacker output — the stacker stage is downstream.
* Each bot is per-(question, forecaster) because ``research_cache`` is keyed
  by ``id_of_question`` and we want each instance's cache to contain exactly
  the one entry for the question we're forecasting.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from datetime import datetime
from typing import Any

import litellm
from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile

from main import TemplateForecaster
from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from metaculus_bot.ablation.env import probabilistic_tools_enabled
from metaculus_bot.ablation.forecaster_lineup import (
    build_free_forecaster_llms,
    build_free_parser_llm,
)
from metaculus_bot.ablation.window_patch import patched_window_for_question
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import FORECASTER_SOFT_DEADLINE
from metaculus_bot.llm_configs import RESEARCHER_LLM, SUMMARIZER_LLM

logger = logging.getLogger(__name__)

# Maximum sleep duration for exponential backoff fallback when ``retry_after_seconds``
# is missing from the 429 payload. OpenRouter docs recommend 1s/2s/4s/8s; this caps
# the worst case at 60s so a stuck attempt doesn't park the runner indefinitely.
_RATE_LIMIT_BACKOFF_CAP_SECONDS: float = 60.0

# Maximum sleep when honoring an explicit ``retry_after_seconds`` from the 429
# payload. Free-tier providers occasionally signal long recovery windows (90s+)
# on hot-tail throttles, and we want to honor those — capping below them sheds
# forecasters that would otherwise succeed on the next attempt. The cap exists
# only as a backstop against a misbehaving upstream sending Retry-After: 3600
# (one hour) on a transient throttle, which would park the runner for the rest
# of the batch. 120s is the calibrated middle: comfortably above observed
# legitimate Retry-After values, well below catastrophic.
_MAX_RATE_LIMIT_SLEEP_SECONDS: float = 120.0

# Default max retries on RateLimitError. Matches the "gentle" preset in cli.py;
# the CLI always passes an explicit value, so this default is just for direct
# unit-test invocation and other ad-hoc callers.
_DEFAULT_MAX_RETRIES: int = 3

# Pulled from a litellm-wrapped OpenRouter 429 in /tmp/ablation_phase_a1_v3.log:
#     ..."retry_after_seconds":13,"retry_after_seconds_raw":12.315,...
# The integer key is what we want — ``retry_after_seconds_raw`` is a more precise
# float but ``retry_after_seconds`` matches the upstream Retry-After header. Using
# regex (rather than json.loads) is deliberate: the litellm prefix wraps a JSON
# blob that itself contains nested braces and string-escaped JSON, so a substring
# json.loads is brittle. The regex matches a int or float and returns the first hit.
_RETRY_AFTER_REGEX = re.compile(r'"retry_after_seconds"\s*:\s*(\d+(?:\.\d+)?)')

# Companion to ``_RETRY_AFTER_REGEX`` for HTTP-date-form Retry-After headers
# (RFC 7231). OpenRouter occasionally proxies upstream date-form values
# (e.g. "Wed, 21 Oct 2026 07:28:00 GMT"); without parsing this form, the
# retry path falls through to the 60s exponential backoff cap and burns the
# retry budget while the real recovery window is much longer.
_RETRY_AFTER_DATE_REGEX = re.compile(r'"Retry-After"\s*:\s*"([^"\d][^"]*)"')

# Same regex shape as ``_RETRY_AFTER_REGEX`` but for the upstream provider name.
# Real OpenRouter 429 payload carries ``"provider_name":"Venice"`` (or
# OpenInference, etc.); surfacing this in retry logs lets the operator see
# which upstream is causing pain when several free-tier forecasters share one.
# Quoted-string match — values are short alphanumeric/dash, no need for
# exhaustive escape handling.
_PROVIDER_NAME_REGEX = re.compile(r'"provider_name"\s*:\s*"([^"]+)"')


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if the exception is a 429 / upstream-provider rate limit.

    Defensive about exception class drift: matches both ``litellm.RateLimitError``
    *and* any exception whose stringified message carries ``"code":429`` or
    ``rate-limited upstream``. The upstream-rate-limit text is OpenRouter-specific
    and stable across litellm versions.
    """
    if isinstance(exc, litellm.RateLimitError):
        return True
    msg = str(exc)
    if '"code":429' in msg or "code: 429" in msg:
        return True
    if "rate-limited upstream" in msg.lower():
        return True
    return False


def _parse_retry_after_seconds(exc: BaseException) -> float | None:
    """Extract ``retry_after_seconds`` from an OpenRouter 429 payload.

    Returns None if the field isn't present (caller falls back to exponential
    backoff). Uses regex rather than json.loads because the litellm-wrapped
    string carries a JSON blob with nested braces + string-escaped JSON inside,
    which makes substring json.loads brittle.

    Two upstream forms supported:
    * Integer-seconds in ``retry_after_seconds`` (the common OpenRouter shape).
    * HTTP-date in the ``Retry-After`` header field (RFC 7231 alt form). We
      compute (target_time - now) clamped to >= 0 so a stale past-date sleeps
      nothing and retries immediately.
    """
    import email.utils  # noqa: PLC0415  - keeps import resilient against ruff auto-strip during partial edits
    from datetime import datetime, timezone  # noqa: PLC0415

    text = str(exc)
    match = _RETRY_AFTER_REGEX.search(text)
    if match is not None:
        return float(match.group(1))
    date_match = _RETRY_AFTER_DATE_REGEX.search(text)
    if date_match is None:
        return None
    raw_date = date_match.group(1)
    try:
        target = email.utils.parsedate_to_datetime(raw_date)
    except (ValueError, TypeError) as parse_exc:
        logger.warning("parsing Retry-After date %r: %s", raw_date, parse_exc)
        return None
    if target is None:
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    delta = (target - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, delta)


def _parse_provider_name(exc: BaseException) -> str | None:
    """Extract ``provider_name`` (e.g. ``"Venice"``) from an OpenRouter 429 payload.

    Returns None if absent. Same regex-over-json.loads tradeoff as
    ``_parse_retry_after_seconds`` — the litellm-wrapped exception string is
    not parseable JSON without significant work.
    """
    match = _PROVIDER_NAME_REGEX.search(str(exc))
    if match is None:
        return None
    return match.group(1)


def _backoff_seconds(attempt: int) -> float:
    """Jittered exponential backoff for attempt index ``attempt`` (0-indexed).

    ``2**attempt + uniform[0, 1)`` capped at ``_RATE_LIMIT_BACKOFF_CAP_SECONDS``.
    Matches the OpenRouter docs' 1s/2s/4s/8s recommendation when uncapped.
    """
    return min(_RATE_LIMIT_BACKOFF_CAP_SECONDS, (2**attempt) + random.uniform(0.0, 1.0))


# The window patch monkey-patches `_forecasting_window_str` GLOBALLY, and its
# context manager guards against nested entry with a RuntimeError. When two
# `run_forecasters_for_question` invocations are concurrent (per_question_concurrency>1),
# the second entry would collide with the first's patch state. Serializing the
# patched section with an asyncio.Lock keeps all forecasters within ONE patched
# region per question while still letting other batch work (cache reads, write
# atomic file ops, gathers) overlap across questions.
_WINDOW_PATCH_LOCK: asyncio.Lock | None = None


def _get_window_patch_lock() -> asyncio.Lock:
    global _WINDOW_PATCH_LOCK
    if _WINDOW_PATCH_LOCK is None:
        _WINDOW_PATCH_LOCK = asyncio.Lock()
    return _WINDOW_PATCH_LOCK


__all__ = [
    "deserialize_prediction_value",
    "question_type_for_serialization",
    "run_forecasters_batch",
    "run_forecasters_for_question",
    "serialize_prediction_value",
]


# ---------------------------------------------------------------------------
# Failure-stage classification
# ---------------------------------------------------------------------------


def _infer_failure_stage(exc: Exception, forecaster_model_slug: str) -> str:
    """Heuristically tag which stage of ``_make_prediction`` raised.

    ``_make_prediction`` calls (1) the forecaster LLM via
    ``_run_forecast_on_<type>``, then (2) the parser LLM via
    ``structure_output``. The exception message rarely identifies which model
    raised — exception types are litellm-generic. We use textual heuristics
    that survived first-light:

    * ``"no allowed providers"`` → almost always the parser. The donated-key
      allowed-providers 404 hits the parser specifically (the parser is the
      OAI-prefixed model in the ablation; production forecasters route via
      the donated key too but typically don't 404 at this scale).
    * Rate-limit text → forecaster. Free-tier rate limits are model-specific;
      since forecasters run before the parser is invoked, a rate limit on
      the forecaster prevents the parser from ever being called. The parser
      runs once per forecaster call and shares one parser model across all
      forecasters, so its rate-limit footprint is bounded.

    Returns one of ``"forecaster"``, ``"parser"``, or ``"unknown"``. The full
    exception is always preserved in the payload's ``errors`` list — this tag
    is purely advisory for log readers.
    """
    msg = str(exc).lower()
    # Donated-key allowed-providers 404. Parser uses the OAI-prefixed model
    # in production llm_configs (and historically did in the ablation, before
    # task #16 switched to plain GeneralLlm); forecasters in the ablation
    # are now plain GeneralLlm so they don't 404 on this code path.
    if "no allowed providers" in msg:
        return "parser"
    # Rate limits: provider-throttled before parser is invoked.
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return "forecaster"
    # Model-specific text: if the exception names the forecaster slug, the
    # forecaster invoke itself raised. Parser-side errors typically wrap
    # the parser model's name instead.
    if forecaster_model_slug.lower() in msg:
        return "forecaster"
    return "unknown"


# ---------------------------------------------------------------------------
# Prediction-value (de)serialization
# ---------------------------------------------------------------------------


def question_type_for_serialization(question: MetaculusQuestion) -> str:
    """Return the canonical question-type discriminator string used by
    :func:`serialize_prediction_value`.

    Exposed (rather than ``_question_type``) so sibling modules in
    ``metaculus_bot.ablation`` can serialize prediction values consistently.
    """
    if isinstance(question, BinaryQuestion):
        return "binary"
    if isinstance(question, MultipleChoiceQuestion):
        return "multiple_choice"
    if isinstance(question, NumericQuestion):
        return "numeric"
    raise ValueError(f"Unsupported question type: {type(question).__name__}")


def serialize_prediction_value(value: Any, question_type: str) -> dict[str, Any]:
    """Convert a prediction_value from ``_make_prediction`` into a JSON-safe dict.

    Schema by question type:

    * Binary → ``{"type": "binary", "prob": float}``
    * MC → ``{"type": "multiple_choice", "options": [{"option_name", "probability"}, ...]}``
    * Numeric → full ``NumericDistribution`` round-trip (see below).

    The numeric payload captures *everything* the forecaster's pipeline computed —
    declared percentiles, the constraint-enforced 201-point CDF (probabilities), and
    the bounds/zero_point/cdf_size needed to reconstruct ``PchipNumericDistribution``
    on read. Storing the raw CDF avoids silently re-deriving via
    ``build_numeric_distribution`` at deserialize time, which would couple cached
    artifacts to whatever pipeline code happens to be running when the cache is read.
    """
    if question_type == "binary":
        return {"type": "binary", "prob": float(value)}
    if question_type == "multiple_choice":
        if not isinstance(value, PredictedOptionList):
            raise TypeError(f"Expected PredictedOptionList for MC, got {type(value).__name__}")
        return {
            "type": "multiple_choice",
            "options": [
                {"option_name": opt.option_name, "probability": float(opt.probability)}
                for opt in value.predicted_options
            ],
        }
    if question_type == "numeric":
        # Iterating a NumericDistribution (Pydantic BaseModel) yields
        # ``(field_name, value)`` tuples — that was the bug. Pull declared_percentiles
        # explicitly. The .cdf property returns a list of Percentile objects whose
        # ``percentile`` field carries the probability and ``value`` field carries the
        # corresponding question value (lower_bound + i * (upper-lower)/(N-1)).
        # We only persist the probabilities; the value axis is fully determined by
        # bounds + cdf_size and reconstructed at deserialize time.
        if not isinstance(value, NumericDistribution):
            raise TypeError(f"Expected NumericDistribution for numeric, got {type(value).__name__}")
        cdf_points = value.cdf
        return {
            "type": "numeric",
            "declared_percentiles": [
                {"percentile": float(p.percentile), "value": float(p.value)} for p in value.declared_percentiles
            ],
            "cdf_probabilities": [float(p.percentile) for p in cdf_points],
            "lower_bound": float(value.lower_bound),
            "upper_bound": float(value.upper_bound),
            "open_lower_bound": bool(value.open_lower_bound),
            "open_upper_bound": bool(value.open_upper_bound),
            "zero_point": float(value.zero_point) if value.zero_point is not None else None,
            "cdf_size": int(value.cdf_size) if value.cdf_size is not None else None,
        }
    raise ValueError(f"Unknown question_type for serialization: {question_type}")


def deserialize_prediction_value(payload: dict[str, Any], question: MetaculusQuestion) -> Any:
    """Inverse of ``serialize_prediction_value``. Returns the original Python type.

    For numeric payloads, reconstructs a ``PchipNumericDistribution`` using the
    cached 201-point CDF directly — does NOT re-run ``build_numeric_distribution``,
    so cached artifacts are independent of whatever pipeline code is currently
    loaded. This matters for stacker stages reading forecaster outputs that were
    cached weeks ago.

    Old payload shape (``percentiles`` instead of ``declared_percentiles``, no
    ``cdf_probabilities``) is recognized and rejected loudly — re-run the
    forecaster stage with ``--force-stages forecast`` to upgrade.
    """
    payload_type = payload["type"]
    if payload_type == "binary":
        return float(payload["prob"])
    if payload_type == "multiple_choice":
        return PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=opt["option_name"], probability=float(opt["probability"]))
                for opt in payload["options"]
            ]
        )
    if payload_type == "numeric":
        if "cdf_probabilities" not in payload:
            raise ValueError(
                "Numeric forecaster payload predates the full-CDF round-trip "
                "(missing 'cdf_probabilities' key). Re-run the forecaster stage "
                "with --force-stages forecast to upgrade cached payloads."
            )
        # Local import keeps pchip_processing out of the import-time graph for
        # binary/MC paths (lighter cold-start when only those types are exercised).
        from metaculus_bot.numeric.pchip_processing import (  # noqa: PLC0415  # function-scoped: see AGENTS.md
            create_pchip_numeric_distribution,
        )

        declared = [
            Percentile(percentile=float(p["percentile"]), value=float(p["value"]))
            for p in payload["declared_percentiles"]
        ]
        cdf_probabilities: list[float] = [float(p) for p in payload["cdf_probabilities"]]
        # ``create_pchip_numeric_distribution`` reads bounds/zero_point/cdf_size
        # from ``question`` — but for the ablation, ``question`` is the
        # rehydrated Pydantic shim, which carries those fields verbatim. Using
        # the live question's bounds keeps the .cdf property's value axis
        # consistent with what the forecaster actually computed.
        return create_pchip_numeric_distribution(
            pchip_cdf=cdf_probabilities,
            percentile_list=declared,
            question=question,  # type: ignore[arg-type]  # NumericQuestion at runtime
            zero_point=payload.get("zero_point"),
        )
    raise ValueError(f"Unknown prediction_value payload type: {payload_type}")


# ---------------------------------------------------------------------------
# Bot construction
# ---------------------------------------------------------------------------


def _build_bot(
    *,
    question: MetaculusQuestion,
    research_blob: str,
    forecaster_llm: GeneralLlm,
    parser_llm: GeneralLlm,
) -> TemplateForecaster:
    """Construct a single-forecaster TemplateForecaster keyed to one question.

    ``research_cache={qid: research_blob}`` plus ``is_benchmarking=True`` makes
    ``run_research`` short-circuit and return the cached blob without invoking
    any provider.

    ``aggregation_strategy=MEAN`` ensures the stacker doesn't fire — we just
    want the per-model rationale and prediction value from ``_make_prediction``.
    """
    qid = question.id_of_question
    if qid is None:
        raise ValueError("Question must have id_of_question for ablation forecasting")
    return TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        llms={
            "forecasters": [forecaster_llm],
            "parser": parser_llm,
            "summarizer": SUMMARIZER_LLM,
            "researcher": RESEARCHER_LLM,
        },
        aggregation_strategy=AggregationStrategy.MEAN,
        research_provider=None,
        max_questions_per_run=None,
        is_benchmarking=True,
        max_concurrent_research=1,
        allow_research_fallback=False,
        research_cache={qid: research_blob},
        min_forecasters_to_publish=1,
    )


# ---------------------------------------------------------------------------
# Per-question runner
# ---------------------------------------------------------------------------


async def _run_one_forecaster(
    question: MetaculusQuestion,
    research_blob: str,
    forecaster_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    cache: AblationCache,
    *,
    semaphore: asyncio.Semaphore,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> tuple[str, dict[str, Any]]:
    """Run one forecaster on one question; return (model_slug_filename, payload).

    Errors are captured in the payload's ``errors`` list with
    ``prediction_value=None``. The payload is always written to cache before
    returning so we have a record even on failure.

    On ``litellm.RateLimitError`` (or any 429-shaped exception), the call is
    retried up to ``max_retries`` times. The sleep duration honors
    ``retry_after_seconds`` from the OpenRouter exception payload when present
    (plus a small jitter to dodge thundering-herd wakeups when many forecasters
    are throttled simultaneously); otherwise falls back to capped jittered
    exponential backoff. Non-429 errors are NOT retried — they fall through to
    the existing single-attempt error path.
    """
    model_slug = model_slug_to_filename(forecaster_llm.model)
    qid = question.id_of_question

    async with semaphore:
        bot = _build_bot(
            question=question,
            research_blob=research_blob,
            forecaster_llm=forecaster_llm,
            parser_llm=parser_llm,
        )

        logger.info("ablation forecaster start | qid=%s | model=%s", qid, forecaster_llm.model)
        start = time.monotonic()
        prediction: ReasonedPrediction | None = None
        errors: list[str] = []
        # Mirror the framework's notepad lifecycle from
        # ``ForecastBot._run_individual_question`` (lines 352-354 + 414): every
        # call to ``_make_prediction`` first reads ``_get_notepad(question)``,
        # which raises if no notepad has been registered. ``_run_individual_question``
        # is the framework's normal entry; we bypass it for cache-friendly per-(qid,
        # model) granularity, so we own the lifecycle here. The ``finally`` block
        # ensures notepads don't accumulate across calls when the prediction raises.
        notepad = await bot._initialize_notepad(question)
        async with bot._note_pad_lock:
            bot._note_pads.append(notepad)
        try:
            # Attempt budget: 1 initial attempt + ``max_retries`` retries on 429.
            # Non-429 errors fall through after a single attempt. ``errors``
            # accumulates EVERY failed attempt's stringified exception so a
            # postmortem can see the full sequence. On a success that follows
            # transient 429s, we clear ``errors`` (after ``break`` below) so
            # downstream consumers can treat ``len(errors) == 0`` as "this
            # forecaster delivered" without false positives from recovered retries.
            for attempt in range(max_retries + 1):
                try:
                    # Soft deadline mirrors production at main.py:1063: a single
                    # stuck forecaster used to be able to hold a question for
                    # litellm timeout(480) * allowed_tries(3) ≈ 24 min. Bound
                    # each attempt at FORECASTER_SOFT_DEADLINE (10 min in prod);
                    # the asyncio.TimeoutError is treated as a non-rate-limit
                    # failure below so it doesn't trigger the retry loop.
                    prediction = await asyncio.wait_for(
                        bot._make_prediction(question, research_blob, forecaster_llm),
                        timeout=FORECASTER_SOFT_DEADLINE,
                    )
                    errors = []
                    break
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                    if not _is_rate_limit_error(exc):
                        # Not a rate limit — log with stage heuristic (mirrors the
                        # original error path) and stop. Original misdirection
                        # cost (parser 404 mis-tagged as forecaster) was the
                        # whole reason ``_infer_failure_stage`` exists.
                        stage = _infer_failure_stage(exc, forecaster_llm.model)
                        logger.warning(
                            "ablation forecaster failed | qid=%s | forecaster_model=%s | "
                            "likely_stage=%s | parser_model=%s | %s: %s",
                            qid,
                            forecaster_llm.model,
                            stage,
                            parser_llm.model,
                            type(exc).__name__,
                            exc,
                        )
                        break
                    # Rate-limited path. If we've exhausted retries, log + give up.
                    if attempt >= max_retries:
                        logger.warning(
                            "ablation forecaster rate-limited (retries exhausted) | qid=%s | "
                            "forecaster_model=%s | attempts=%d | %s: %s",
                            qid,
                            forecaster_llm.model,
                            attempt + 1,
                            type(exc).__name__,
                            exc,
                        )
                        break
                    # Honor ``retry_after_seconds`` when present; jitter prevents
                    # thundering-herd wake when many forecasters share an upstream
                    # provider (Venice, OpenInference) and all hit the same window.
                    # Cap at ``_MAX_RATE_LIMIT_SLEEP_SECONDS`` to bound a misbehaving
                    # upstream that signals an unreasonably long Retry-After
                    # (3600s / one hour observed in the wild on hot-tail throttles).
                    retry_after = _parse_retry_after_seconds(exc)
                    if retry_after is not None:
                        sleep_seconds = min(retry_after + random.uniform(0.1, 0.5), _MAX_RATE_LIMIT_SLEEP_SECONDS)
                    else:
                        sleep_seconds = _backoff_seconds(attempt)
                    provider_name = _parse_provider_name(exc) or "unknown"
                    logger.info(
                        "ablation forecaster rate-limited (retrying) | qid=%s | "
                        "forecaster_model=%s | provider=%s | attempt=%d/%d | sleep=%.2fs",
                        qid,
                        forecaster_llm.model,
                        provider_name,
                        attempt + 1,
                        max_retries,
                        sleep_seconds,
                    )
                    await asyncio.sleep(sleep_seconds)
        finally:
            await bot._remove_notepad(question)

        duration = time.monotonic() - start

        # Wrap serialize + cache-write in its own try/except so a single forecaster's
        # post-prediction failure (e.g., the ``tuple`` AttributeError that started
        # this whole exercise) cannot cascade through ``asyncio.gather``'s default
        # ``return_exceptions=False`` and wipe out every other forecaster's already-
        # cached output for this qid. Each forecaster's payload — success or
        # failure — must reach disk independently.
        prediction_value: Any = None
        reasoning: str = ""
        try:
            if prediction is not None:
                qtype = question_type_for_serialization(question)
                prediction_value = serialize_prediction_value(prediction.prediction_value, qtype)
                reasoning = prediction.reasoning
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            # We know the stage here — no heuristic needed. The original tuple bug
            # surfaced as ``AttributeError: 'tuple' object has no attribute 'percentile'``
            # in this exact code path, and tagging it ``serialize`` saves the
            # operator the 30-minute detour of grep-ing through forecaster vs.
            # parser code.
            logger.warning(
                "ablation forecaster failed | qid=%s | forecaster_model=%s | "
                "likely_stage=serialize | parser_model=%s | %s: %s",
                qid,
                forecaster_llm.model,
                parser_llm.model,
                type(exc).__name__,
                exc,
            )
            prediction_value = None
            reasoning = ""

        payload = {
            "model": forecaster_llm.model,
            "prediction_value": prediction_value,
            "reasoning": reasoning,
            "errors": errors,
            "ran_at": datetime.now().isoformat(),
            "duration_seconds": float(duration),
        }
        # Even on serialize failure, persist the (failure) payload so partial-success
        # cases preserve the diagnostic record. Cache-write failures (rare — disk
        # full, permissions) are intentionally NOT caught here; they're a fatal
        # environmental signal the operator needs to see.
        cache.write_forecaster_output(qid=qid, model_slug=model_slug, payload=payload)
        logger.info(
            "ablation forecaster done | qid=%s | model=%s | duration=%.1fs | errors=%d",
            qid,
            forecaster_llm.model,
            duration,
            len(errors),
        )
        return model_slug, payload


async def run_forecasters_for_question(
    question: MetaculusQuestion,
    research_blob: str,
    cache: AblationCache,
    *,
    forecaster_llms: list[GeneralLlm] | None = None,
    parser_llm: GeneralLlm | None = None,
    force: bool = False,
    per_forecaster_concurrency: int = 4,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> dict[str, dict[str, Any]]:
    """Run all forecasters against one question; cache + return per-model payloads.

    Returns a dict keyed by ``model_slug_filename`` (filesystem-safe slug).

    For each forecaster:

    * If a cached payload exists and ``force`` is False, the cached payload is
      returned without calling ``_make_prediction``.
    * Otherwise the forecaster is invoked under
      :func:`patched_window_for_question` so prompt-injected dates are anchored
      to the question's mid-window.

    Per-forecaster errors are caught, recorded in the payload's ``errors``
    list, and persisted to cache with ``prediction_value=None``. The batch
    continues regardless.
    """
    if forecaster_llms is None:
        forecaster_llms = build_free_forecaster_llms()
    if parser_llm is None:
        parser_llm = build_free_parser_llm()

    qid = question.id_of_question
    if qid is None:
        raise ValueError("Question must have id_of_question for ablation forecasting")

    results: dict[str, dict[str, Any]] = {}
    to_run: list[GeneralLlm] = []
    for llm in forecaster_llms:
        slug = model_slug_to_filename(llm.model)
        if not force:
            cached = cache.read_forecaster_output(qid=qid, model_slug=slug)
            if cached is not None:
                results[slug] = cached
                continue
        to_run.append(llm)

    if to_run:
        semaphore = asyncio.Semaphore(per_forecaster_concurrency)
        # Window patch wraps the entire fan-out so prompts inside _make_prediction
        # see the question-anchored "today". The context manager guards against
        # nested entry; the module-level asyncio.Lock serializes patched sections
        # across concurrent batch runs so two questions can't try to enter
        # simultaneously.
        #
        # ``probabilistic_tools_enabled(False)`` disables the env flag for the
        # duration of the fan-out so an operator-shell ``PROBABILISTIC_TOOLS_ENABLED=1``
        # does NOT contaminate the cached rationales. Forecaster rationales are
        # shared across both ablation arms; if ``_make_prediction`` were to append
        # ``## Computed quantities`` while caching, BOTH arms would inherit that
        # contamination and the A/B comparison would be invalid. The original
        # env value is restored on exit so anything outside the forecast stage
        # observes the operator's setting unchanged.
        async with _get_window_patch_lock():
            with patched_window_for_question(question), probabilistic_tools_enabled(False):
                tasks = [
                    _run_one_forecaster(
                        question,
                        research_blob,
                        llm,
                        parser_llm,
                        cache,
                        semaphore=semaphore,
                        max_retries=max_retries,
                    )
                    for llm in to_run
                ]
                for slug, payload in await asyncio.gather(*tasks):
                    results[slug] = payload

    succeeded = sum(1 for p in results.values() if p["prediction_value"] is not None)
    logger.info(
        "ablation forecaster rollup | qid=%s | succeeded=%d/%d",
        qid,
        succeeded,
        len(forecaster_llms),
    )
    return results  # noqa: ASYNC910 -- all-cached path is intentionally checkpoint-free


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


async def run_forecasters_batch(
    questions_with_research: list[tuple[MetaculusQuestion, str]],
    cache: AblationCache,
    *,
    forecaster_llms: list[GeneralLlm] | None = None,
    parser_llm: GeneralLlm | None = None,
    force: bool = False,
    per_question_concurrency: int = 2,
    per_forecaster_concurrency: int = 4,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Run forecasters on a batch of (question, research_blob) pairs.

    Returns ``{qid: {model_slug_filename: payload}}``. Per-question failures
    (the runner itself raising, not individual forecasters) leave the qid
    keyed to an empty dict — the batch continues.
    """
    if forecaster_llms is None:
        forecaster_llms = build_free_forecaster_llms()
    if parser_llm is None:
        parser_llm = build_free_parser_llm()

    semaphore = asyncio.Semaphore(per_question_concurrency)
    # Late binding via module attribute so tests can monkeypatch
    # `run_forecasters_for_question` on the module and have it observed here.
    from metaculus_bot.ablation import forecasters as _self_module

    async def _run_one(question: MetaculusQuestion, blob: str) -> tuple[int, dict[str, dict[str, Any]]]:
        async with semaphore:
            try:
                per_model = await _self_module.run_forecasters_for_question(
                    question,
                    blob,
                    cache,
                    forecaster_llms=forecaster_llms,
                    parser_llm=parser_llm,
                    force=force,
                    per_forecaster_concurrency=per_forecaster_concurrency,
                    max_retries=max_retries,
                )
                return question.id_of_question, per_model
            except Exception as exc:
                logger.warning(
                    "ablation forecaster batch | qid=%s failed entirely | %s: %s",
                    question.id_of_question,
                    type(exc).__name__,
                    exc,
                )
                return question.id_of_question, {}

    tasks = [_run_one(q, blob) for q, blob in questions_with_research]
    results: dict[int, dict[str, dict[str, Any]]] = {}
    for qid, per_model in await asyncio.gather(*tasks):
        results[qid] = per_model
    return results
