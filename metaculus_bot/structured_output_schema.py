"""
Pydantic schemas for structured forecaster output blocks.

Base-forecaster LLMs (binary / multiple-choice / numeric) are asked to append
a fenced ```json block to their free-text rationale that declares structured
fields (prior, base rate, hazard, percentiles, scenarios, etc.). A post-hoc
tool runner extracts these blocks and feeds them to probabilistic tools
(Beta-binomial, log-pooling, distribution fitting).

This module defines the schemas and extraction helpers. It is DORMANT:
nothing here is wired into the prompt or runtime pipeline yet.

Note: ``DiscreteCountStructured`` is defined here but not dispatched by the
current tool runner — discrete-count question dispatch is phase-3 work. The
class remains so that forecaster prompts can be updated first and the
runtime wiring can follow later.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

_HAZARD_FRACTION_TOLERANCE = 0.01
_SCENARIO_PROB_SUM_TOLERANCE = 0.02
_MC_OPTION_PROB_SUM_TOLERANCE = 0.02
_TAIL_MASS_SUM_CEILING = 0.5
_REQUIRED_NUMERIC_PERCENTILES: frozenset[float] = frozenset({0.1, 0.5, 0.9})
# Defensive cap on raw structured-block size. Legitimate blocks are <5KB;
# larger payloads likely indicate a malformed rationale (e.g., an unclosed
# fence accidentally swallowing half the transcript). We cap rather than
# parse-on-trust to keep memory / parse time bounded.
_MAX_STRUCTURED_BLOCK_BYTES: int = 200_000


# ---------------------------------------------------------------------------
# Shared submodels
# ---------------------------------------------------------------------------


class StatedPrior(BaseModel):
    """A forecaster's declared outside-view prior (before updating on evidence)."""

    model_config = ConfigDict(extra="forbid")

    prob: float = Field(ge=0.0, le=1.0)
    source: str = Field(min_length=1)


class StatedBaseRate(BaseModel):
    """Explicit k successes out of n trials in a declared reference class."""

    model_config = ConfigDict(extra="forbid")

    k: int
    n: int
    ref_class: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_k_n(self) -> StatedBaseRate:
        if self.n < 1:
            raise ValueError(f"StatedBaseRate.n must be >= 1, got {self.n}")
        if self.k < 0:
            raise ValueError(f"StatedBaseRate.k must be >= 0, got {self.k}")
        if self.k > self.n:
            raise ValueError(f"StatedBaseRate requires k <= n, got k={self.k}, n={self.n}")
        return self


class StatedHazard(BaseModel):
    """Constant-hazard model: rate per unit time plus the window length in the same units.

    Contract: ``rate_per_unit`` is expressed per ``unit`` (e.g., 0.25/day),
    and ``window_duration_units`` is the full forecast-window length in the
    SAME unit (e.g., 30 for "rate/day over a 30-day window"). Units cancel
    when the tool runner computes the survival integral, so no conversion
    to years (or any other canonical unit) is performed.

    ``elapsed_fraction`` + ``remaining_fraction`` describe how much of
    ``window_duration_units`` has already passed at forecast time.
    """

    model_config = ConfigDict(extra="forbid")

    rate_per_unit: float = Field(ge=0.0)
    unit: Literal["day", "week", "month", "year"]
    window_duration_units: float = Field(gt=0.0)
    elapsed_fraction: float = Field(ge=0.0, le=1.0)
    remaining_fraction: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_fractions_sum(self) -> StatedHazard:
        total = self.elapsed_fraction + self.remaining_fraction
        if abs(total - 1.0) > _HAZARD_FRACTION_TOLERANCE:
            raise ValueError(
                f"StatedHazard.elapsed_fraction + remaining_fraction must be ~1.0 "
                f"(tol {_HAZARD_FRACTION_TOLERANCE}), got {total}"
            )
        return self


class EvidenceItem(BaseModel):
    """A single piece of evidence with direction and strength.

    ``summary`` and ``direction`` are prompt-scaffolding: they structure the
    forecaster's reasoning but are not consumed by the numeric tool runner
    (which uses only ``strength`` and ``likelihood_ratio``). Keep them as
    required fields so prompts continue to demand explicit decomposition.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    direction: Literal["up", "down", "neutral"]
    strength: Literal["strong", "moderate", "weak"]
    likelihood_ratio: float | None = None

    @field_validator("likelihood_ratio")
    @classmethod
    def _check_lr(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"EvidenceItem.likelihood_ratio must be > 0 if set, got {v}")
        return v


class ScenarioBranch(BaseModel):
    """One branch of a declared scenario decomposition."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    prob: float = Field(ge=0.0, le=1.0)
    conditional_outcome: str | None = None


class TailMass(BaseModel):
    """Declared mass outside the question's declared numeric range."""

    model_config = ConfigDict(extra="forbid")

    below_min_expected: float = Field(ge=0.0, le=1.0)
    above_max_expected: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_sum(self) -> TailMass:
        total = self.below_min_expected + self.above_max_expected
        if total >= _TAIL_MASS_SUM_CEILING:
            raise ValueError(
                f"TailMass sum must be < {_TAIL_MASS_SUM_CEILING}, got {total} "
                f"(below_min_expected={self.below_min_expected}, above_max_expected={self.above_max_expected})"
            )
        return self


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _validate_scenario_sum(scenarios: list[ScenarioBranch]) -> list[ScenarioBranch]:
    if not scenarios:
        return scenarios
    total = sum(s.prob for s in scenarios)
    if abs(total - 1.0) > _SCENARIO_PROB_SUM_TOLERANCE:
        raise ValueError(
            f"Non-empty scenarios must have probs summing to ~1.0 (tol {_SCENARIO_PROB_SUM_TOLERANCE}), got {total}"
        )
    return scenarios


# ---------------------------------------------------------------------------
# Per-question-type models
# ---------------------------------------------------------------------------


class BinaryStructured(BaseModel):
    """Structured declaration for a binary question."""

    model_config = ConfigDict(extra="forbid")

    question_type: Literal["binary"]
    prior: StatedPrior | None = None
    base_rate: StatedBaseRate | None = None
    hazard: StatedHazard | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    scenarios: list[ScenarioBranch] = Field(default_factory=list)
    posterior_prob: float = Field(ge=0.0, le=1.0)

    @field_validator("scenarios")
    @classmethod
    def _check_scenarios_sum(cls, v: list[ScenarioBranch]) -> list[ScenarioBranch]:
        return _validate_scenario_sum(v)


class NumericStructured(BaseModel):
    """Structured declaration for a numeric question."""

    model_config = ConfigDict(extra="forbid")

    question_type: Literal["numeric"]
    prior: StatedPrior | None = None
    declared_percentiles: dict[float, float]
    distribution_family_hint: Literal["normal", "lognormal", "student_t", "skew_normal", "beta", "other"] | None = None
    student_t_df: float | None = None
    tails: TailMass | None = None
    scenarios: list[ScenarioBranch] = Field(default_factory=list)

    @field_validator("student_t_df")
    @classmethod
    def _check_df(cls, v: float | None) -> float | None:
        if v is not None and v <= 1:
            raise ValueError(f"NumericStructured.student_t_df must be > 1 if set, got {v}")
        return v

    @field_validator("declared_percentiles")
    @classmethod
    def _check_percentiles(cls, v: dict[float, float]) -> dict[float, float]:
        missing = _REQUIRED_NUMERIC_PERCENTILES - set(v.keys())
        if missing:
            raise ValueError(
                f"NumericStructured.declared_percentiles must include at least "
                f"{sorted(_REQUIRED_NUMERIC_PERCENTILES)}, missing {sorted(missing)}"
            )
        for pct in v.keys():
            if not (0.0 <= pct <= 1.0):
                raise ValueError(f"Percentile keys must be in [0, 1], got {pct}")
        sorted_keys = sorted(v.keys())
        prev_value: float | None = None
        for key in sorted_keys:
            value = v[key]
            if prev_value is not None and value <= prev_value:
                raise ValueError(
                    f"declared_percentiles values must be strictly increasing with percentile; "
                    f"got {value} at pct {key} after {prev_value}"
                )
            prev_value = value
        return v

    @field_validator("scenarios")
    @classmethod
    def _check_scenarios_sum(cls, v: list[ScenarioBranch]) -> list[ScenarioBranch]:
        return _validate_scenario_sum(v)


class MultipleChoiceStructured(BaseModel):
    """Structured declaration for a multiple-choice question."""

    model_config = ConfigDict(extra="forbid")

    question_type: Literal["multiple_choice"]
    prior: StatedPrior | None = None
    option_probs: dict[str, float]
    other_mass: float | None = Field(default=None, ge=0.0, le=1.0)
    concentration: float | None = None

    @field_validator("concentration")
    @classmethod
    def _check_concentration(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"MultipleChoiceStructured.concentration must be > 0 if set, got {v}")
        return v

    @field_validator("option_probs")
    @classmethod
    def _check_option_probs(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("MultipleChoiceStructured.option_probs must be non-empty")
        for key, prob in v.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"MultipleChoiceStructured.option_probs keys must be non-empty strings, got {key!r}")
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"MultipleChoiceStructured.option_probs values must be in [0, 1], got {prob}")
        total = sum(v.values())
        if abs(total - 1.0) > _MC_OPTION_PROB_SUM_TOLERANCE:
            raise ValueError(
                f"MultipleChoiceStructured.option_probs must sum to ~1.0 "
                f"(tol {_MC_OPTION_PROB_SUM_TOLERANCE}), got {total}"
            )
        return v


class DiscreteCountStructured(BaseModel):
    """Structured declaration for a discrete-count question."""

    model_config = ConfigDict(extra="forbid")

    question_type: Literal["discrete_count"]
    prior: StatedPrior | None = None
    mean_estimate: float = Field(ge=0.0)
    dispersion: Literal["poisson", "negbinom", "beta_binom_ceiling"]
    ceiling: int | None = None
    overdispersion_factor: float | None = None
    declared_percentiles: dict[float, float] | None = None

    @field_validator("ceiling")
    @classmethod
    def _check_ceiling(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"DiscreteCountStructured.ceiling must be >= 1 if set, got {v}")
        return v

    @field_validator("overdispersion_factor")
    @classmethod
    def _check_overdispersion(cls, v: float | None) -> float | None:
        if v is not None and v <= 1:
            raise ValueError(f"DiscreteCountStructured.overdispersion_factor must be > 1 if set, got {v}")
        return v

    @model_validator(mode="after")
    def _check_dispersion_requirements(self) -> DiscreteCountStructured:
        if self.dispersion == "beta_binom_ceiling" and self.ceiling is None:
            raise ValueError("DiscreteCountStructured with dispersion='beta_binom_ceiling' requires ceiling to be set")
        return self


StructuredBlock = Annotated[
    Union[BinaryStructured, NumericStructured, MultipleChoiceStructured],
    Field(discriminator="question_type"),
]


# NOTE: ``DiscreteCountStructured`` is intentionally NOT mapped here — the
# runtime tool runner does not dispatch on it yet (phase-3). The class is
# retained in this module so prompts can declare discrete_count blocks and
# future activation work can extend the runner without schema changes.
_QUESTION_TYPE_TO_MODEL: dict[str, type[BaseModel]] = {
    "binary": BinaryStructured,
    "numeric": NumericStructured,
    "multiple_choice": MultipleChoiceStructured,
}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

# Matches fenced blocks of the form ```json ... ```, ```JSON ... ```,
# ``` json ... ``` (with whitespace), or plain ``` ... ``` where the content
# itself starts with `{`.
_FENCE_PATTERN = re.compile(
    r"```[ \t]*(?P<tag>[A-Za-z]*)[ \t]*\r?\n(?P<body>.*?)\r?\n[ \t]*```",
    re.DOTALL,
)


def extract_json_block(rationale_text: str) -> str | None:
    """
    Extract the LAST fenced JSON block from a rationale.

    Preference order:
      1. Explicitly tagged ```json / ```JSON (case-insensitive, any whitespace).
      2. Untagged ``` fence whose body begins with `{`.

    Returns the trimmed body or None if nothing matches.
    """
    if not rationale_text:
        return None

    tagged_matches: list[str] = []
    untagged_json_matches: list[str] = []

    for match in _FENCE_PATTERN.finditer(rationale_text):
        tag = match.group("tag").strip().lower()
        body = match.group("body").strip()
        if not body:
            continue
        if tag == "json":
            tagged_matches.append(body)
        elif tag == "" and body.lstrip().startswith("{"):
            untagged_json_matches.append(body)

    if tagged_matches:
        return tagged_matches[-1]
    if untagged_json_matches:
        return untagged_json_matches[-1]
    return None


def extract_first_balanced_braces(s: str) -> str | None:
    """Return the first balanced ``{...}`` block in ``s``, or None if none exists.

    String-literal-aware: braces inside JSON string literals are not counted.
    Respects backslash escapes so ``"\\""`` does not terminate a string. This
    makes the helper safe on inputs like ``{"foo": "has a } brace"}`` which a
    naive brace-counter would truncate.

    A naive scan that counted every ``{`` / ``}`` would produce malformed
    output on payloads where the LLM embeds literal braces inside string
    values — a common failure mode we silently hit before adding this.
    """
    start_idx = s.find("{")
    if start_idx == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start_idx, len(s)):
        c = s[i]
        if escape_next:
            escape_next = False
            continue
        if in_string:
            if c == "\\":
                escape_next = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start_idx : i + 1]

    return None


def parse_structured_block(
    rationale_text: str,
    question_type: Literal["binary", "numeric", "multiple_choice"],
) -> StructuredBlock | None:
    """
    Extract and validate a structured JSON block from a rationale.

    Returns the parsed Pydantic model or None. None on:
      - No fenced JSON block (logged at DEBUG)
      - Malformed JSON (logged at WARNING)
      - Pydantic validation error (logged at WARNING)
      - question_type mismatch between argument and JSON payload (WARNING)

    ``"discrete_count"`` is intentionally unsupported at runtime — see the
    module docstring.
    """
    raw = extract_json_block(rationale_text)
    if raw is None:
        logger.debug("No JSON block found in rationale for question_type=%s", question_type)
        return None

    if len(raw) > _MAX_STRUCTURED_BLOCK_BYTES:
        logger.warning(
            "Structured block exceeds size cap (%d bytes > %d); refusing to parse (question_type=%s)",
            len(raw),
            _MAX_STRUCTURED_BLOCK_BYTES,
            question_type,
        )
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        snippet = raw[:200].replace("\n", " ")
        logger.warning(
            "Malformed JSON in structured block (question_type=%s): %s. Snippet: %s", question_type, exc, snippet
        )
        return None

    if not isinstance(payload, dict):
        logger.warning(
            "Structured block must decode to a JSON object, got %s (question_type=%s)",
            type(payload).__name__,
            question_type,
        )
        return None

    payload_qtype = payload.get("question_type")
    if payload_qtype is not None and payload_qtype != question_type:
        logger.warning(
            "question_type mismatch: arg=%s, payload=%s. Refusing to parse.",
            question_type,
            payload_qtype,
        )
        return None

    # Inject the expected question_type if missing so the discriminator picks the right model.
    if payload_qtype is None:
        payload = {**payload, "question_type": question_type}

    model_cls = _QUESTION_TYPE_TO_MODEL[question_type]
    try:
        return model_cls.model_validate(payload)  # type: ignore[return-value]
    except ValidationError as exc:
        logger.warning(
            "Structured block failed validation for question_type=%s: %s",
            question_type,
            exc,
        )
        return None
