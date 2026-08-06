"""Forecaster-drop attribution (systematic-failure observability).

When a forecaster is dropped from the ensemble, we record WHICH model, WHICH
question, and WHY — turning the bare ``forecasters_dropped=N`` scalar into a signal
that can tell a one-off blip from a model going systematically bad (a refusal
class, a routing problem, provider-wide instability). Each cause below is
honestly DETERMINABLE at its recording site (see the ``_record_forecaster_drop``
call sites in ``forecaster.py``); we never guess — "error_other" is the explicit
catch-all.
"""

import json
import logging
from collections import defaultdict
from typing import NamedTuple, Sequence

from metaculus_bot.exceptions import ValueExtractionError
from metaculus_bot.llm_retry import is_zero_output_failure

logger = logging.getLogger(__name__)

DROP_CAUSE_TIMEOUT_WALL_CLOCK = "timeout_wall_clock"  # per-question wall-clock budget expired; task cancelled
DROP_CAUSE_TIMEOUT_SOFT_DEADLINE = "timeout_soft_deadline"  # per-forecaster FORECASTER_SOFT_DEADLINE exceeded
DROP_CAUSE_ZERO_OUTPUT = "zero_output"  # provider returned no usable content (empty/whitespace body)
DROP_CAUSE_PARSE_EXTRACTION = "parse_extraction"  # the value-extraction ladder exhausted every rung
DROP_CAUSE_ERROR_OTHER = "error_other"  # a raised exception outside the classes above


class ForecasterDrop(NamedTuple):
    """One dropped ensemble member, attributed. ``model`` is the GeneralLlm slug,
    ``qid`` the question id (None only when a site genuinely can't know it), and
    ``cause`` one of the ``DROP_CAUSE_*`` categories."""

    model: str
    qid: int | None
    cause: str


def classify_raised_drop_cause(exc: BaseException) -> str:
    """Map an exception that dropped a forecaster to a ``DROP_CAUSE_*`` category.

    Inspects the ALREADY-CAUGHT exception's type/shape — no new try/except. Reuses
    ``llm_retry.is_zero_output_failure`` so this telemetry's "zero_output" label
    agrees with the retry gate's own classification by construction (the 2026-07-25
    OpenRouter whitespace-drip). ``asyncio.TimeoutError`` is recorded at the
    soft-deadline site and excluded before this is called, so it never lands here.
    """
    if is_zero_output_failure(exc):
        return DROP_CAUSE_ZERO_OUTPUT
    if isinstance(exc, ValueExtractionError):
        return DROP_CAUSE_PARSE_EXTRACTION
    return DROP_CAUSE_ERROR_OTHER


def emit_drop_telemetry(drops: Sequence[ForecasterDrop]) -> None:
    """Emit the per-run ``FORECASTER_DROPS`` marker + a human per-model summary,
    and WARN on any single model that dropped across MULTIPLE questions.

    Attribution is CODE-derived (never model-self-reported): every entry was
    stamped at a drop site with the model slug, the question id, and a
    determinable cause. This turns the bare ``forecasters_dropped=N`` scalar
    into "which model failed, how often, and why" answerable in one grep.

    The systematic signal (a single model dropping on >=2 DISTINCT questions)
    is a WARNING, not just a summary line: at the current
    ``MIN_FORECASTERS_TO_PUBLISH`` floor a model going bad silently degrades every
    forecast in the run while CI shows one modest red mark, so it must be visible
    without grepping. It deliberately does NOT change the exit code or block
    publishing (that is the operator's call); ``alertable_count`` already reddens
    CI on ANY drop.
    """
    # model -> cause -> count, plus the set of DISTINCT questions each model
    # dropped on (the systematic-failure key).
    detail: dict[str, dict[str, int]] = {}
    questions_by_model: dict[str, set[int]] = defaultdict(set)
    for drop in drops:
        per_cause = detail.setdefault(drop.model, {})
        per_cause[drop.cause] = per_cause.get(drop.cause, 0) + 1
        if drop.qid is not None:
            questions_by_model[drop.model].add(drop.qid)

    systematic = sorted(model for model, qids in questions_by_model.items() if len(qids) >= 2)
    systematic_field = ",".join(systematic) if systematic else "none"
    # Compact JSON blob: robust to '/'-laden OpenRouter slugs (which defeat any
    # delimiter-based encoding) and json.loads-able by residual analysis.
    detail_json = json.dumps(detail, sort_keys=True, separators=(",", ":"))
    logger.info("FORECASTER_DROPS: total=%d systematic=%s detail=%s", len(drops), systematic_field, detail_json)

    if not drops:
        return

    for model in sorted(detail):
        cause_str = ",".join(f"{cause}:{count}" for cause, count in sorted(detail[model].items()))
        logger.info("Forecaster drops by model: %s=%d [%s]", model, sum(detail[model].values()), cause_str)

    for model in systematic:
        cause_str = ",".join(f"{cause}:{count}" for cause, count in sorted(detail[model].items()))
        qids = ",".join(str(qid) for qid in sorted(questions_by_model[model]))
        logger.warning(
            "SYSTEMATIC_FORECASTER_FAILURE: model=%s dropped_on_questions=%d qids=%s causes=%s — "
            "one model failed across multiple questions this run (likely a refusal class or routing "
            "problem, not a blip); investigate or consider pulling it from the roster.",
            model,
            len(questions_by_model[model]),
            qids,
            cause_str,
        )
