"""Deterministic-first extraction ladder for forecast values.

Forecaster (and stacker) LLMs emit their forecast exactly once: a fenced
```json STRUCTURED FORECAST block as the LAST thing in the rationale. This
module extracts the value with a four-rung ladder:

1. **block** — deterministic fenced-block parse (``parse_structured_payload``:
   json.loads + Pydantic validation).
2. **repair** — deterministic JSON repair (``json_repair``) of a malformed
   fenced block, or a balanced-braces scan of the rationale tail when no
   fence survived.
3. **llm** — the existing LLM parser (``parse_structured``) over the full
   rationale, as salvage. Logged loudly; the guardrail against fabrication is
   the strict post-rung validation, not trust.
4. raise ``ValueExtractionError`` — the caller drops/soft-fails the
   forecaster, exactly as parser failures propagated before the ladder.

The two deterministic rungs run CANDIDATE-major, not rung-major: for each
candidate in selection order (position-last first, since the prompt asks for
the block last) BOTH the strict parse and the repair are tried before a
lower-ranked candidate is considered. Rung-major ordering would publish a
superseded draft — a valid earlier block would satisfy rung 1, so a malformed
final block would never reach the repairer. ``rung`` on the returned outcome
names whichever mechanism produced the value, so the telemetry is unchanged.

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
    extract_json_block_candidates,
    iter_balanced_braces,
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


@dataclass
class _DeterministicHit(Generic[T]):
    """A value recovered from ONE candidate body, plus the rung that produced it."""

    value: T
    rung: Rung


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


def _try_candidate(
    candidate: str,
    *,
    qtype: QuestionTypeStr,
    convert_block: Callable[[StructuredBlock], T],
    validate: Callable[[T], T],
    try_strict: bool,
    label: str,
    failures: list[str],
    question_id: int | None,
) -> _DeterministicHit[T] | None:
    """Strict-parse then ``json_repair`` ONE candidate body; None when neither yields a value.

    Both deterministic mechanisms hit the same candidate before the caller moves
    on, so a malformed block that is plausibly the model's final answer gets
    repaired instead of losing to a lower-ranked valid one.

    ``try_strict=False`` for unfenced tail blobs: there is no block to have
    parsed, so they are reported as ``rung="repair"`` even when the blob happens
    to be well-formed JSON.

    Candidates are probed with ``log_failures=False``: any single one may be a
    junk recap that a lower-ranked candidate recovers from, so the reasons are
    accumulated into ``failures`` (surfaced by the rung-3 log or the ladder
    error) rather than each emitting its own WARNING.
    """
    if try_strict:
        strict = parse_structured_payload(candidate, qtype, log_failures=False)
        if strict is not None:
            try:
                return _DeterministicHit(value=validate(convert_block(strict)), rung="block")
            except (ValueError, TypeError) as exc:
                # Schema-valid but unusable — e.g. a numeric block carrying only
                # the three percentiles the schema demands, not the 13 the
                # pipeline needs. json_repair cannot change already-valid JSON,
                # so this candidate is spent and the caller falls back.
                failures.append(f"block: {label}: {exc}")
                return None

    if len(candidate) > _MAX_STRUCTURED_BLOCK_BYTES:
        failures.append(f"repair: {label}: exceeds size cap; refusing to repair")
        return None
    repaired = repair_json(candidate)
    if not (isinstance(repaired, str) and repaired.strip()):
        failures.append(f"repair: {label}: json_repair produced no usable output")
        return None
    payload_model = parse_structured_payload(repaired, qtype, log_failures=False)
    if payload_model is None:
        failures.append(f"repair: {label}: repaired JSON failed schema validation")
        return None
    try:
        value = validate(convert_block(payload_model))
    except (ValueError, TypeError) as exc:
        failures.append(f"repair: {label}: {exc}")
        return None
    if repaired != candidate:
        logger.info(
            "json_repair modified candidate for qtype=%s question=%s (len %d -> %d)",
            qtype,
            question_id,
            len(candidate),
            len(repaired),
        )
    return _DeterministicHit(value=value, rung="repair")


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
    fenced = extract_json_block_candidates(text)
    block_present = bool(fenced)

    # --- Rungs 1+2: deterministic walk over candidates, best-first ---------
    # Selection order comes from extract_json_block_candidates: tagged ```json
    # before untagged fences, and WITHIN a tier the last block by position first,
    # because the prompt asks for the STRUCTURED FORECAST block last. Position is
    # the primary signal; validity — strict OR repaired — only breaks ties.
    #
    # Each candidate is offered to BOTH deterministic mechanisms before the next
    # one is tried (see _try_candidate). Running them as separate passes over the
    # whole list published superseded drafts: a valid earlier draft block
    # satisfied the strict pass, so a malformed final block never reached the
    # repairer that exists to fix exactly that (a trailing comma).
    walk: list[tuple[str, bool]]
    if block_present:
        walk = [(candidate, True) for candidate in fenced]
    else:
        logger.info("No fenced JSON block in rationale for qtype=%s question=%s", qtype, question_id)
        failures.append("block: no fenced JSON block")
        # No fence survived: rescue bare JSON objects from the rationale tail,
        # LAST first for the same position primacy. Repair-only — calling these
        # rung="block" would contradict block_present=False.
        tail_blobs = list(iter_balanced_braces(text[-_TAIL_SCAN_CHARS:]))
        if not tail_blobs:
            failures.append("repair: no candidate JSON in rationale tail")
        walk = [(blob, False) for blob in reversed(tail_blobs)]

    for rank, (candidate, try_strict) in enumerate(walk):
        hit = _try_candidate(
            candidate,
            qtype=qtype,
            convert_block=convert_block,
            validate=validate,
            try_strict=try_strict,
            label=f"candidate {rank + 1}/{len(walk)}",
            failures=failures,
            question_id=question_id,
        )
        if hit is None:
            continue
        if rank > 0:
            # The value did NOT come from the position-last candidate, so it may
            # not be the model's final answer (the observed case is a trailing
            # schema-example block, which is benign — hence INFO, not WARNING).
            # Watch this alongside EXTRACTION_RUNG: a rise means the prompt's
            # block-last contract is eroding.
            logger.info(
                "BLOCK_FALLBACK: question=%s model=%s qtype=%s skipped=%d rung=%s reasons=%s",
                question_id,
                model_name,
                qtype,
                rank,
                hit.rung,
                " | ".join(failures),
            )
        _log_extraction(qtype, hit.rung, block_present, question_id, model_name)
        return ExtractionOutcome(value=hit.value, rung=hit.rung, block_present=block_present)

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
