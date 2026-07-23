"""Deterministic-first extraction ladder for forecast values.

Forecaster (and stacker) LLMs emit their forecast exactly once: a fenced
```json STRUCTURED FORECAST block as the LAST thing in the rationale. This
module extracts the value with a four-rung ladder:

1. **block** — deterministic fenced-block parse (``parse_structured_block``:
   json.loads + Pydantic validation).
2. **repair** — deterministic JSON repair (``json_repair``) of a malformed
   fenced block, or a balanced-braces scan of the rationale tail when no
   fence survived.
3. **llm** — the existing LLM parser (``parse_structured``) over the full
   rationale, as salvage. Logged loudly; the guardrail against fabrication is
   the strict post-rung validation, not trust.
4. raise ``ValueExtractionError`` — the caller drops/soft-fails the
   forecaster, exactly as parser failures propagated before the ladder.

Every successful extraction emits one ``EXTRACTION_RUNG`` INFO line (this
telemetry supersedes the deleted shadow-divergence comparison): watch for
``rung=llm`` salvages and ``block_present=False`` as the drift signal.

Callers keep their post-processing contracts: binary output is the RAW
pre-clamp decimal; MC output is the pre-``clamp_and_renormalize_mc`` option
list; numeric output feeds ``sanitize_percentiles`` unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from forecasting_tools import BinaryPrediction, GeneralLlm, PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile
from json_repair import repair_json
from pydantic import ValidationError

from metaculus_bot.exceptions import ValueExtractionError
from metaculus_bot.mc_processing import build_mc_prediction, clamp_and_renormalize_probs
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.simple_types import OptionProbability
from metaculus_bot.structured_output_schema import (
    _MAX_STRUCTURED_BLOCK_BYTES,
    _MC_OPTION_PROB_SUM_TOLERANCE,
    BinaryStructured,
    MultipleChoiceStructured,
    NumericStructured,
    StructuredBlock,
    extract_first_balanced_braces,
    extract_json_block,
    parse_structured_block,
    parse_structured_payload,
)
from metaculus_bot.structured_parse import parse_structured

logger = logging.getLogger(__name__)

T = TypeVar("T")
Rung = Literal["block", "repair", "llm"]
QuestionTypeStr = Literal["binary", "numeric", "multiple_choice"]

# How far back from the rationale tail the unfenced-JSON rescue scans. The
# block is prompted to be the LAST output, so a lost fence still leaves the
# payload within the final few KB.
_TAIL_SCAN_CHARS = 4000
# Float tolerance when matching parsed percentile keys against the canonical
# 13-set (guards against 0.1 vs 0.10000000001 drift from JSON round-trips).
_PERCENTILE_KEY_TOLERANCE = 1e-6


@dataclass
class ExtractionOutcome(Generic[T]):
    """A validated forecast value plus which ladder rung produced it."""

    value: T
    rung: Rung
    block_present: bool


def _log_extraction(
    qtype: QuestionTypeStr,
    rung: Rung,
    block_present: bool,
    question_id: int | None,
    model_name: str,
) -> None:
    logger.info(
        "EXTRACTION_RUNG: question=%s model=%s qtype=%s rung=%s block_present=%s",
        question_id,
        model_name,
        qtype,
        rung,
        block_present,
    )


async def _run_ladder(
    *,
    text: str,
    qtype: QuestionTypeStr,
    convert_block: Callable[[StructuredBlock], T],
    validate: Callable[[T], T],
    llm_extract: Callable[[], Awaitable[T]],
    question_id: int | None,
    model_name: str,
) -> ExtractionOutcome[T]:
    """Shared rung driver. ``convert_block``/``validate`` raise ValueError to fail a rung."""
    failures: list[str] = []
    block_body = extract_json_block(text)
    block_present = block_body is not None

    # --- Rung 1: deterministic block parse -------------------------------
    block = parse_structured_block(text, qtype)
    if block is not None:
        try:
            value = validate(convert_block(block))
        except (ValueError, TypeError) as exc:
            failures.append(f"block: {exc}")
        else:
            _log_extraction(qtype, "block", block_present, question_id, model_name)
            return ExtractionOutcome(value=value, rung="block", block_present=block_present)
    else:
        failures.append("block: no schema-valid fenced JSON block")

    # --- Rung 2: deterministic JSON repair --------------------------------
    # Only useful when rung 1 could not even produce a model: if the block
    # parsed to a schema-valid model but conversion/validation failed (e.g. a
    # partial percentile set), repairing already-valid JSON is a no-op — skip
    # straight to the LLM rung. Size-cap violations stay fatal: never feed a
    # >200KB payload to the repairer.
    repair_candidate: str | None = None
    if block is None:
        if block_body is not None:
            if len(block_body) <= _MAX_STRUCTURED_BLOCK_BYTES:
                repair_candidate = block_body
            else:
                failures.append("repair: block exceeds size cap; refusing to repair")
        else:
            repair_candidate = extract_first_balanced_braces(text[-_TAIL_SCAN_CHARS:])
            if repair_candidate is None:
                failures.append("repair: no candidate JSON in rationale tail")
    else:
        failures.append("repair: skipped (block was schema-valid; repair cannot change it)")

    if repair_candidate is not None:
        repaired = repair_json(repair_candidate)
        if isinstance(repaired, str) and repaired.strip():
            if repaired != repair_candidate:
                logger.info(
                    "json_repair modified candidate for qtype=%s question=%s (len %d -> %d)",
                    qtype,
                    question_id,
                    len(repair_candidate),
                    len(repaired),
                )
            payload_model = parse_structured_payload(repaired, qtype)
            if payload_model is not None:
                try:
                    value = validate(convert_block(payload_model))
                except (ValueError, TypeError) as exc:
                    failures.append(f"repair: {exc}")
                else:
                    _log_extraction(qtype, "repair", block_present, question_id, model_name)
                    return ExtractionOutcome(value=value, rung="repair", block_present=block_present)
            else:
                failures.append("repair: repaired JSON failed schema validation")
        else:
            failures.append("repair: json_repair produced no usable output")

    # --- Rung 3: LLM parser salvage ---------------------------------------
    try:
        value = validate(await llm_extract())
    except Exception as exc:  # noqa: BLE001, HARNESS-SCAN-EXEMPT-broad-except  # terminal rung: fold ANY parser failure into the typed ladder error so callers see one exception type
        failures.append(f"llm: {type(exc).__name__}: {exc}")
    else:
        logger.warning(
            "EXTRACTION_RUNG=llm salvage: question=%s model=%s qtype=%s block_present=%s "
            "(deterministic rungs failed: %s)",
            question_id,
            model_name,
            qtype,
            block_present,
            " | ".join(failures),
        )
        _log_extraction(qtype, "llm", block_present, question_id, model_name)
        return ExtractionOutcome(value=value, rung="llm", block_present=block_present)

    # --- Rung 4: typed failure --------------------------------------------
    raise ValueExtractionError(
        f"All extraction rungs failed for qtype={qtype} question={question_id} model={model_name}: "
        + " | ".join(failures)
    )


# ---------------------------------------------------------------------------
# Binary
# ---------------------------------------------------------------------------


def _binary_from_block(block: StructuredBlock) -> float:
    if not isinstance(block, BinaryStructured):
        raise ValueError(f"expected binary block, got {type(block).__name__}")
    return float(block.posterior_prob)


def _validate_binary(value: float) -> float:
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"binary probability {value} outside [0, 1]")
    return value


async def extract_binary(
    text: str,
    parser_llm: GeneralLlm,
    *,
    prompt_notes: str = "",
    question_id: int | None = None,
    model_name: str = "",
) -> ExtractionOutcome[float]:
    """Extract the RAW decimal probability (pre-clamp; caller applies the binary clamp)."""

    async def _llm() -> float:
        prediction: BinaryPrediction = await parse_structured(
            text, BinaryPrediction, parser_llm, prompt_notes=prompt_notes
        )
        return float(prediction.prediction_in_decimal)

    return await _run_ladder(
        text=text,
        qtype="binary",
        convert_block=_binary_from_block,
        validate=_validate_binary,
        llm_extract=_llm,
        question_id=question_id,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Numeric
# ---------------------------------------------------------------------------


def _numeric_from_block(block: StructuredBlock) -> list[Percentile]:
    if not isinstance(block, NumericStructured) or not block.declared_percentiles:
        raise ValueError("block lacks declared_percentiles")
    # Absorbs the old numeric_format_router F5 fallback: lift the block's
    # declared_percentiles dict into Percentile objects.
    return [
        Percentile(percentile=float(pct), value=float(val)) for pct, val in sorted(block.declared_percentiles.items())
    ]


def _validate_numeric(percentiles: list[Percentile]) -> list[Percentile]:
    """Require ALL 13 standard percentiles; return exactly the canonical 13, never padded."""
    matched: dict[float, Percentile] = {}
    for standard in STANDARD_PERCENTILES:
        for p in percentiles:
            if abs(float(p.percentile) - standard) <= _PERCENTILE_KEY_TOLERANCE:
                matched[standard] = p
                break
    missing = [s for s in STANDARD_PERCENTILES if s not in matched]
    if missing:
        raise ValueError(
            f"missing standard percentiles {missing}; got {sorted(float(p.percentile) for p in percentiles)}"
        )
    return [matched[s] for s in STANDARD_PERCENTILES]


async def extract_numeric(
    text: str,
    parser_llm: GeneralLlm,
    *,
    prompt_notes: str = "",
    question_id: int | None = None,
    model_name: str = "",
) -> ExtractionOutcome[list[Percentile]]:
    """Extract the 13 standard percentiles (caller feeds them to ``sanitize_percentiles``).

    ``prompt_notes`` should be ``build_parse_notes(question)`` so the rung-3
    parser keeps today's bound-aware extraction instructions.
    """

    async def _llm() -> list[Percentile]:
        return await parse_structured(text, list[Percentile], parser_llm, prompt_notes=prompt_notes)

    return await _run_ladder(
        text=text,
        qtype="numeric",
        convert_block=_numeric_from_block,
        validate=_validate_numeric,
        llm_extract=_llm,
        question_id=question_id,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Multiple choice
# ---------------------------------------------------------------------------


def _make_mc_from_block(options: list[str]) -> Callable[[StructuredBlock], PredictedOptionList]:
    def _mc_from_block(block: StructuredBlock) -> PredictedOptionList:
        if not isinstance(block, MultipleChoiceStructured):
            raise ValueError(f"expected multiple_choice block, got {type(block).__name__}")
        # Match each block key to a canonical option by case/whitespace-insensitive
        # comparison. We deliberately do NOT route through build_mc_prediction here:
        # its _normalize_name strips a leading "Option " token, which would mangle
        # options literally named "Option A"/"Option B". The block already declares
        # exact per-option probabilities, so we map straight onto the canonical
        # names in question order. Any unmatched key or missing option fails the
        # rung (→ validation → next rung). Clamp + renormalize BEFORE constructing
        # the PredictedOptionList so ft 0.2.92's clamp-and-renormalize validator
        # (which raises on any >0.05 move) is a no-op; the caller still applies
        # clamp_and_renormalize_mc idempotently.
        canonical_by_norm = {opt.strip().lower(): opt for opt in options}
        matched: dict[str, float] = {}
        for key, prob in block.option_probs.items():
            canonical = canonical_by_norm.get(key.strip().lower())
            if canonical is None:
                raise ValueError(f"block option {key!r} does not match any question option {options}")
            matched[canonical] = matched.get(canonical, 0.0) + float(prob)
        total = sum(matched.values())
        if total <= 0:
            raise ValueError(f"block option probabilities sum to {total}")
        ordered = [(name, matched[name]) for name in options if name in matched]
        clamped = clamp_and_renormalize_probs([prob for _, prob in ordered])
        return PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=name, probability=prob) for (name, _), prob in zip(ordered, clamped)
            ]
        )

    return _mc_from_block


def _make_validate_mc(options: list[str]) -> Callable[[PredictedOptionList], PredictedOptionList]:
    def _validate_mc(pol: PredictedOptionList) -> PredictedOptionList:
        names = [o.option_name for o in pol.predicted_options]
        if set(names) != set(options):
            raise ValueError(f"option set mismatch: got {names}, expected {options}")
        for option in pol.predicted_options:
            if not (0.0 <= option.probability <= 1.0):
                raise ValueError(f"option {option.option_name!r} probability {option.probability} outside [0, 1]")
        total = sum(o.probability for o in pol.predicted_options)
        if abs(total - 1.0) > _MC_OPTION_PROB_SUM_TOLERANCE:
            raise ValueError(f"option probabilities sum to {total}, outside 1.0 ± {_MC_OPTION_PROB_SUM_TOLERANCE}")
        return pol

    return _validate_mc


async def extract_mc(
    text: str,
    options: list[str],
    parser_llm: GeneralLlm,
    *,
    prompt_notes: str = "",
    question_id: int | None = None,
    model_name: str = "",
) -> ExtractionOutcome[PredictedOptionList]:
    """Extract a PredictedOptionList mapped onto ``options`` (pre-``clamp_and_renormalize_mc``)."""
    options = list(options)

    async def _llm() -> PredictedOptionList:
        # Mirror the pre-ladder two-stage tolerant parse: strict
        # PredictedOptionList first, then the loose list[OptionProbability]
        # form. BOTH sub-paths route through build_mc_prediction so parser
        # output with case/prefix-variant option names ("option a") is
        # canonicalized onto the question's option set before _validate_mc's
        # exact set comparison — the parser prompt_notes explicitly allow
        # case-insensitive matches, so the strict result can't be trusted to
        # carry canonical spellings.
        try:
            strict = await parse_structured(text, PredictedOptionList, parser_llm, prompt_notes=prompt_notes)
            as_raw = [
                OptionProbability(option_name=o.option_name, probability=o.probability)
                for o in strict.predicted_options
            ]
            return build_mc_prediction(as_raw, options)
        except (ValidationError, ValueError) as exc:
            logger.warning("Primary MC parse failed in llm rung, using tolerant fallback: %s", exc)
            raw: list[OptionProbability] = await parse_structured(
                text, list[OptionProbability], parser_llm, prompt_notes=prompt_notes
            )
            return build_mc_prediction(raw, options)

    return await _run_ladder(
        text=text,
        qtype="multiple_choice",
        convert_block=_make_mc_from_block(options),
        validate=_make_validate_mc(options),
        llm_extract=_llm,
        question_id=question_id,
        model_name=model_name,
    )


__all__ = [
    "ExtractionOutcome",
    "Rung",
    "extract_binary",
    "extract_mc",
    "extract_numeric",
]
