"""Serialization for the ablation qids manifest.

One responsibility: converting between live forecasting-tools question objects /
:class:`GroundTruth` records and the JSON-safe dicts stored in
``<cache-dir>/qids_manifest.json``, and rehydrating a real Pydantic question from a
stored entry so cached re-runs need no Metaculus fetch. Split out of ``ablation.cli``
so the manifest schema — and the drift-surfacing subscripts that enforce it — can be
read and tested without importing the whole CLI orchestrator.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.questions import OutOfBoundsResolution

from metaculus_bot.backtest.scoring import GroundTruth
from metaculus_bot.question_types import question_type_of


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
