"""
Pydantic schemas for structured forecaster output blocks.

Base-forecaster LLMs (binary / multiple-choice / numeric) are asked to append
a fenced ```json block to their free-text rationale that declares structured
fields (prior, base rate, hazard, percentiles, scenarios, etc.). A post-hoc
tool runner extracts these blocks and feeds them to probabilistic tools
(Beta-binomial, log-pooling, distribution fitting).

This module defines the schemas and extraction helpers. Active surface,
gated by ``PROBABILISTIC_TOOLS_ENABLED`` env flag and per-question-type
via ``PROBABILISTIC_TOOLS_TYPES``. See ``metaculus_bot/tool_runner.py``
for dispatch and ``metaculus_bot/forecaster.py:_make_prediction`` for the
activation site.

Note: ``DiscreteCountStructured`` is defined here but not dispatched by the
current tool runner — discrete-count question dispatch is phase-3 work. The
class remains so that forecaster prompts can be updated first and the
runtime wiring can follow later.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterator, Mapping
from typing import Annotated, Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

_HAZARD_FRACTION_TOLERANCE = 0.01
_SCENARIO_PROB_SUM_TOLERANCE = 0.02
_MC_OPTION_PROB_SUM_TOLERANCE = 0.02
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


class BaseRateAnchor(BaseModel):
    """The forecaster's stated outside-view base-rate range (archived blocks only).

    Prompted 2026-07-08 to 2026-09-02 and read by an anchor-overshoot telemetry line
    that was deleted with it; it never clamped or mutated a forecast (the 2026-07
    residual experiments buried the anchor-guard clamp for sign-flipping across eras).
    Retained so the 49 published comments carrying the field still strict-parse in
    ``performance_analysis``.
    """

    model_config = ConfigDict(extra="forbid")

    low: float = Field(ge=0.0, le=1.0)
    high: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> BaseRateAnchor:
        if self.low > self.high:
            raise ValueError(f"BaseRateAnchor requires low <= high, got low={self.low}, high={self.high}")
        return self


class CriteriaClause(BaseModel):
    """One priced resolution clause from the conjunctive-criteria table (archived blocks only).

    Same history as ``BaseRateAnchor``: prompted 2026-07-08 to 2026-09-02, read only by a
    clause-product divergence line that is gone, retained for the 12 published comments
    that carry it. The clause-pricing REASONING stays in the binary prompt's step 5b —
    what went is the JSON echo of the table.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    prob: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _readable_optional_float(value: object, *, low: float | None, high: float | None) -> float | None:
    """A finite float inside [low, high], or None when the declaration is unusable.

    The lenient half of the 2026-09-02 retirement of ``other_mass`` / ``concentration``.
    Both were inputs to a Dirichlet tool behind ``PROBABILISTIC_TOOLS_ENABLED`` (off in
    prod) and neither is prompted any more, so a value that fails its range is worth
    nothing and must cost nothing: it reads as absent instead of raising. The strict
    version cost a real forecast — on q45189 gemini wrote ``"concentration": 0.0`` beside
    a valid three-option ballot, the ``> 0`` check rejected the whole block, ``json_repair``
    cannot alter valid JSON, and MC has no telemetry strip-and-retry, so the ballot was
    re-read by the LLM salvage rung (``rung=llm`` in the extraction archive).

    None is the honest reading rather than a clamp: a value outside its own range means the
    model was not declaring the quantity we asked for, and inventing an in-range substitute
    for a dormant field would put a number nobody stated into the archive.
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        # ``json.loads`` decodes an integer literal of any length into an arbitrary-precision
        # int, and ``float()`` on one past ~308 digits raises. Pydantic converts only
        # ValueError and AssertionError into a ValidationError, so this would propagate out of
        # ``model_validate`` and past ``parse_structured_payload``, whose except clause catches
        # ValidationError only -- strictly worse than the strict code it replaced, which turned
        # the same input into a clean rejection the ladder could fall through.
        return None
    if not math.isfinite(number):
        return None
    if low is not None and number < low:
        return None
    if high is not None and number > high:
        return None
    return number


# The DISCRETE-vs-CONTINUOUS vote's vocabulary, declared once. ``_tolerate_unknown_outcome_type``
# reads its accepted set off this Literal via ``get_args`` rather than restating the two strings:
# a restated copy fails asymmetrically, since a third outcome type would pass the annotation and
# then be silently nulled by the validator. A TUPLE, not a set: membership on a tuple compares by
# equality, so an unhashable declaration (a list, a dict) reads as unrecognised instead of raising
# TypeError out of the validator.
NumericOutcomeType = Literal["discrete_integer", "continuous"]
_NUMERIC_OUTCOME_TYPES: tuple[str, ...] = get_args(NumericOutcomeType)


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
    # ARCHIVED BLOCKS ONLY: the stated outside-view range and priced resolution
    # clauses (2026-07-08). No longer prompted since 2026-09-02 — the block is written
    # after the forecast is fixed, so both slots only re-keyed prose we already have, and
    # their only reader was telemetry behind a flag every prod workflow pins off. The
    # fields stay optional and tolerant because 49 + 12 published comments carry them and
    # performance_analysis strict-parses those blocks.
    base_rate_anchor: BaseRateAnchor | None = None
    criteria_clauses: list[CriteriaClause] = Field(default_factory=list)

    @field_validator("scenarios")
    @classmethod
    def _check_scenarios_sum(cls, v: list[ScenarioBranch]) -> list[ScenarioBranch]:
        return _validate_scenario_sum(v)


class NumericStructured(BaseModel):
    """Structured declaration for a numeric question."""

    model_config = ConfigDict(extra="forbid")

    question_type: Literal["numeric"]
    prior: StatedPrior | None = None
    declared_percentiles: dict[float, float] | None = None
    outcome_type: NumericOutcomeType | None = None
    scenarios: list[ScenarioBranch] = Field(default_factory=list)

    @field_validator("outcome_type", mode="before")
    @classmethod
    def _tolerate_unknown_outcome_type(cls, v: object, info: ValidationInfo) -> str | None:
        """An unrecognised spelling reads as absent instead of failing the whole block.

        ``outcome_type`` gates discrete snapping, and the block value exists to save a
        parser-LLM call (``forecaster_runners._resolve_discrete_vote``). Under a bare
        ``Literal`` a near-miss spelling — "integer", "discrete", "count" — took the
        PERCENTILES down with it: the numeric block has no strip-and-retry, so the whole
        forecast dropped to LLM salvage AND the parser call fired anyway for the type.
        Reading the strays as None costs exactly the one parser call the field was meant
        to save, which is the right price for a misspelling.

        Logged at WARNING with the raw value, because a spelling the roster starts using is a
        prompt signal rather than noise, but only where the caller wants failure logging.
        ``parse_structured_payload`` hands its ``log_failures`` down as validation context, so a
        candidate probe that will be discarded silently (``value_extraction``'s ladder probes
        every candidate that way) drops to DEBUG. Without that gate a misspelling warned on
        superseded draft blocks and twice per numeric forecast on the publish path, about a block
        that validates and publishes fine, which is exactly what ``log_failures`` exists to
        prevent. Direct construction passes no context and keeps the WARNING.
        """
        if v is None:
            return None
        if v in _NUMERIC_OUTCOME_TYPES:
            return str(v)
        context = info.context or {}
        log = logger.warning if context.get("log_failures", True) else logger.debug
        log(
            "Unrecognised outcome_type %r in numeric structured block; reading it as absent "
            "(the discrete vote falls back to the parser call)",
            v,
        )
        return None

    @field_validator("declared_percentiles")
    @classmethod
    def _check_percentiles(cls, v: dict[float, float] | None) -> dict[float, float] | None:
        if not v:
            return v
        missing = _REQUIRED_NUMERIC_PERCENTILES - set(v.keys())
        if missing:
            raise ValueError(
                f"NumericStructured.declared_percentiles must include at least "
                f"{sorted(_REQUIRED_NUMERIC_PERCENTILES)}, missing {sorted(missing)}"
            )
        for pct in v:
            if not (0.0 <= pct <= 1.0):
                raise ValueError(f"Percentile keys must be in [0, 1], got {pct}")
        # NON-decreasing, not strictly increasing: a repeated value is a legitimate
        # concentrated (often count-like) declaration — p1 = p2.5 = 0 on a question that
        # usually reads zero — and both downstream layers are built for exactly that
        # (``value_extraction._validate_numeric`` allows ties by name, and
        # ``sanitize_percentiles``'s cluster spreader exists to separate them). While this
        # schema demanded a strict increase, such a block failed rung 1, could not be
        # repaired (it is valid JSON), and reached the pipeline only via LLM salvage. A
        # strict DECREASE with rising percentile still raises: it is incoherent, and
        # ``sort_percentiles_by_value`` sorts by LABEL, so a value-disordered set is
        # force-monotonized rather than reordered.
        #
        # This is a SAFETY NET for non-compliant output, not a licensed shape. The numeric
        # prompt's schema notes still tell the model values must be strictly increasing, and
        # that wording is deliberate (softening it shifts the forecast distribution, so it is
        # the operator's call), which means the relaxation only ever engages for a forecaster
        # that disobeys its instructions. Measured on the archive: 2 of 346 declarations carry
        # any exact tie, both 3-anchor records from the KNOWN_BUG_QIDS cohort.
        #
        # It also admits a WHOLE-set collapse, which lands somewhere different from the
        # partial tie argued for above: a 13-way tie reaches rung 1, ``sanitize_percentiles``
        # deliberately refuses to cluster-spread it (``NUMERIC_DEGENERATE_DECLARATION`` with
        # spread_applied=false), and ``detect_unit_mismatch`` then WITHHOLDS the member as an
        # alertable drop, instead of the salvage rung re-reading a percentile table out of the
        # prose. The withhold is the designed outcome for a member that declared no width, and
        # 0 of 346 archived declarations are all-equal, so this is unobserved in prod. Do NOT
        # add a "require at least two distinct values" branch: that is defensive branching for
        # a zero-instance case inside the extraction fallback ladder.
        sorted_keys = sorted(v.keys())
        prev_value: float | None = None
        for key in sorted_keys:
            value = v[key]
            if prev_value is not None and value < prev_value:
                raise ValueError(
                    f"declared_percentiles values must be non-decreasing with percentile; "
                    f"got {value} at pct {key} after {prev_value}"
                )
            prev_value = value
        return v

    @model_validator(mode="after")
    def _require_percentiles(self) -> NumericStructured:
        if not self.declared_percentiles:
            raise ValueError(
                f"NumericStructured requires declared_percentiles with at least {sorted(_REQUIRED_NUMERIC_PERCENTILES)}"
            )
        return self

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
    # ARCHIVED BLOCKS ONLY, and read leniently — see _readable_optional_float. Both were
    # Dirichlet tool inputs; no longer prompted since 2026-09-02, and an unusable value
    # now reads as absent rather than costing the ballot that carries the forecast.
    other_mass: float | None = None
    concentration: float | None = None

    @field_validator("other_mass", mode="before")
    @classmethod
    def _tolerate_other_mass(cls, v: object) -> float | None:
        return _readable_optional_float(v, low=0.0, high=1.0)

    @field_validator("concentration", mode="before")
    @classmethod
    def _tolerate_concentration(cls, v: object) -> float | None:
        # A concentration is a positive Dirichlet hyperparameter, so 0.0 and negatives are
        # not readings; the widely-copied example value was 20.0, hence no upper bound.
        read = _readable_optional_float(v, low=None, high=None)
        return read if read is not None and read > 0.0 else None

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
    BinaryStructured | NumericStructured | MultipleChoiceStructured,
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


def extract_json_block_candidates(rationale_text: str) -> list[str]:
    """Fenced JSON-block bodies in SELECTION order (best candidate first).

    Preference order (identical to ``extract_json_block``, now expressed as a
    ranking rather than a single pick):
      1. Explicitly tagged ```json / ```JSON (case-insensitive, any whitespace).
      2. Untagged ``` fence whose body begins with `{`.
    WITHIN each tier the LAST block by document position ranks first — the
    prompt asks for the STRUCTURED FORECAST block last, so among equally-valid
    blocks the last one is the intended forecast. Empty-bodied fences are
    skipped.

    Callers that know the ``question_type`` (``parse_structured_block``) walk
    this list and keep the first body that VALIDATES, so a trailing schema-recap
    or example block that doesn't parse no longer shadows the real forecast
    earlier in the rationale. The publish path (``value_extraction._run_ladder``)
    walks the same order but also runs ``json_repair`` on each body before
    dropping to the next, so a malformed-but-repairable final block outranks a
    lower-ranked valid one.
    """
    if not rationale_text:
        return []

    tagged: list[str] = []
    untagged: list[str] = []
    for match in _FENCE_PATTERN.finditer(rationale_text):
        tag = match.group("tag").strip().lower()
        body = match.group("body").strip()
        if not body:
            continue
        if tag == "json":
            tagged.append(body)
        elif tag == "" and body.lstrip().startswith("{"):
            untagged.append(body)
    # Last-by-position first within each tier; the tagged tier ranks ahead of
    # the untagged one.
    return [*reversed(tagged), *reversed(untagged)]


def extract_json_block(rationale_text: str) -> str | None:
    """
    Extract the single best fenced JSON block from a rationale, by POSITION.

    Preference order:
      1. Explicitly tagged ```json / ```JSON (case-insensitive, any whitespace).
      2. Untagged ``` fence whose body begins with `{`.
    Within a tier the LAST block by document position wins. Returns the trimmed
    body or None if nothing matches.

    This helper is schema-blind: it returns the best-positioned candidate
    without checking that it parses. Callers that know the ``question_type``
    should prefer ``parse_structured_block``, which walks ALL candidates
    (``extract_json_block_candidates``) and keeps the first that actually
    validates. This stays for callers that only need a block's raw text — e.g.
    peeking at a self-declared ``question_type`` before the schema is known.
    """
    candidates = extract_json_block_candidates(rationale_text)
    return candidates[0] if candidates else None


def iter_balanced_braces(s: str) -> Iterator[str]:
    """Yield each top-level balanced ``{...}`` block in ``s``, in document order.

    String-literal-aware: braces inside JSON string literals are not counted,
    and backslash escapes are respected so ``"\\""`` does not terminate a
    string. This makes the scan safe on inputs like ``{"foo": "has a } brace"}``
    which a naive brace-counter would truncate. After closing one block the scan
    resumes AFTER it, so a rationale tail with several bare objects surfaces them
    all — the caller (the value-extraction repair rung) repairs+validates each
    and keeps the first that passes, rather than giving up on a junk leading
    blob (the same iterate-to-valid selection the fenced path uses).
    """
    idx = 0
    length = len(s)
    while idx < length:
        start_idx = s.find("{", idx)
        if start_idx == -1:
            return
        end_idx = _scan_to_matching_brace(s, start_idx)
        if end_idx is None:
            # Unbalanced from start_idx to end — no further balanced block can
            # begin inside this run, so stop.
            return
        yield s[start_idx : end_idx + 1]
        idx = end_idx + 1


def _scan_to_matching_brace(s: str, start_idx: int) -> int | None:
    """Index of the ``}`` that closes the ``{`` at ``start_idx``, or None if unbalanced.

    String-literal-aware: braces inside JSON string literals are not counted, and
    backslash escapes are respected so ``"\\""`` does not terminate a string.
    """
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start_idx, len(s)):
        c = s[i]
        if escape_next:
            escape_next = False
        elif in_string:
            if c == "\\":
                escape_next = True
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def extract_first_balanced_braces(s: str) -> str | None:
    """Return the first balanced ``{...}`` block in ``s``, or None if none exists.

    Thin wrapper over ``iter_balanced_braces`` (which yields every top-level
    balanced block); this returns only the first. Kept for callers that want a
    single blob — e.g. gap-list JSON salvage in ``research/targeted.py``.
    """
    return next(iter_balanced_braces(s), None)


def parse_structured_payload(
    raw_json: str,
    question_type: Literal["binary", "numeric", "multiple_choice"],
    *,
    log_failures: bool = True,
) -> StructuredBlock | None:
    """
    Validate a raw JSON payload string against the structured-block schemas.

    Callers that already have the block body in hand (e.g. after
    ``extract_json_block`` or a ``json_repair`` pass) use this to run the
    size cap, ``json.loads``, dict-shape check, ``question_type`` inject /
    mismatch guard, and Pydantic ``model_validate`` (including the binary
    telemetry strip-and-retry). Returns ``None`` on any failure; the calling
    ladder decides how to log and whether to fall through to the next rung.

    ``log_failures`` gates the WARNING lines on the failure paths (bad size /
    JSON / shape / question_type / validation). A caller probing several
    candidate blocks for the first valid one passes ``log_failures=False`` so a
    rejected-then-recovered candidate doesn't emit a scary WARNING as if
    extraction failed; the telemetry-strip RECOVERY log is not gated, since it
    reports on a block that IS returned. Default ``True`` keeps every direct
    caller's logging unchanged. It is also handed to ``model_validate`` as validation
    CONTEXT, because a lenient validator that reads an unusable declaration as absent
    logs from inside the model, where it cannot otherwise see the flag: without that, a
    suppressed probe still emitted an operator-facing WARNING about a block it was about
    to discard.

    ``"discrete_count"`` is intentionally unsupported at runtime — see the
    module docstring.
    """
    payload = _decode_structured_payload(raw_json, question_type, log_failures=log_failures)
    if payload is None:
        return None

    model_cls = _QUESTION_TYPE_TO_MODEL[question_type]
    context: Mapping[str, object] = {"log_failures": log_failures}
    try:
        return model_cls.model_validate(payload, context=context)  # type: ignore[return-value]
    except ValidationError as exc:
        retry = _retry_without_binary_telemetry(model_cls, payload, question_type, exc, context=context)
        if retry is not None:
            return retry
        if log_failures:
            logger.warning(
                "Structured block failed validation for question_type=%s: %s",
                question_type,
                exc,
            )
        return None


def _decode_structured_payload(
    raw_json: str,
    question_type: Literal["binary", "numeric", "multiple_choice"],
    *,
    log_failures: bool,
) -> dict | None:
    """Size-cap, decode, and shape-check a raw structured block into a dict.

    Returns None on any failure (over the byte cap, malformed JSON, a non-object payload, or
    a ``question_type`` that contradicts the caller's). On success the expected
    ``question_type`` is injected when absent, so the Pydantic discriminator resolves.
    """
    if len(raw_json) > _MAX_STRUCTURED_BLOCK_BYTES:
        if log_failures:
            logger.warning(
                "Structured block exceeds size cap (%d bytes > %d); refusing to parse (question_type=%s)",
                len(raw_json),
                _MAX_STRUCTURED_BLOCK_BYTES,
                question_type,
            )
        return None

    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        if log_failures:
            snippet = raw_json[:200].replace("\n", " ")
            logger.warning(
                "Malformed JSON in structured block (question_type=%s): %s. Snippet: %s", question_type, exc, snippet
            )
        return None

    if not isinstance(payload, dict):
        if log_failures:
            logger.warning(
                "Structured block must decode to a JSON object, got %s (question_type=%s)",
                type(payload).__name__,
                question_type,
            )
        return None

    payload_qtype = payload.get("question_type")
    if payload_qtype is not None and payload_qtype != question_type:
        if log_failures:
            logger.warning(
                "question_type mismatch: arg=%s, payload=%s. Refusing to parse.",
                question_type,
                payload_qtype,
            )
        return None

    # Inject the expected question_type if missing so the discriminator picks the right model.
    if payload_qtype is None:
        return {**payload, "question_type": question_type}
    return payload


def _retry_without_binary_telemetry(
    model_cls: type[BaseModel],
    payload: dict,
    question_type: str,
    exc: ValidationError,
    *,
    context: Mapping[str, object] | None = None,
) -> StructuredBlock | None:
    """Re-validate a failed BINARY block with only the telemetry fields dropped.

    Strip-and-retry for malformed BINARY telemetry (2026-07-08). Since 2026-09-02 the
    prompt no longer asks for either field, so on a fresh forecast this never fires; it
    survives for archived blocks and for a model that emits one from habit. The
    ``base_rate_anchor`` and ``criteria_clauses`` fields are TELEMETRY ONLY — nothing reads them
    to clamp or mutate a forecast. But without this, a malformed anchor / clauses payload
    (canonical failure modes: ``criteria_clauses: null`` even though the prompt says "omit";
    a reversed ``{low > high}`` anchor) would make us drop the ENTIRE block — including a
    perfectly good posterior_prob — silently disappearing the forecaster's base-rate blend
    and prior/posterior contributions from the cross-model aggregation. That would let a pure
    formatting bug in a telemetry field shift stacker input, violating the telemetry
    rollout's zero-behavior-change invariant.

    NOT a schema-wide before-validator: those silently coerce bad clause probs and miss the
    reversed-anchor case. Only the telemetry fields are dropped, so any error in a core field
    (posterior_prob, prior, base_rate, hazard, evidence, scenarios) still surfaces as None.
    """
    telemetry_fields = {"base_rate_anchor", "criteria_clauses"}
    if question_type != "binary" or not telemetry_fields & payload.keys():
        return None

    stripped_keys = sorted(telemetry_fields & payload.keys())
    stripped_payload = {k: v for k, v in payload.items() if k not in telemetry_fields}
    try:
        retry = model_cls.model_validate(stripped_payload, context=context)
    except ValidationError:
        return None
    logger.warning(
        "Dropping malformed telemetry fields %s and keeping core binary block (original error: %s)",
        stripped_keys,
        exc,
    )
    return retry  # type: ignore[return-value]


def parse_structured_block(
    rationale_text: str,
    question_type: Literal["binary", "numeric", "multiple_choice"],
) -> StructuredBlock | None:
    """
    Extract and validate a structured JSON block from a rationale.

    Selection is validity-aware: rather than validating only the last block by
    position (which let a trailing schema-recap / example block shadow a valid
    forecast earlier in the rationale), this walks candidates best-first
    (``extract_json_block_candidates``) and keeps the FIRST that validates for
    ``question_type``. Among valid blocks the last-by-position still wins (the
    prompt asks for the forecast block last); tagged ```json blocks still
    outrank untagged fences.

    Returns the parsed Pydantic model or None. None on:
      - No fenced JSON block at all (logged at INFO)
      - No candidate validates (the last-tried candidate's WARNING surfaces the
        reason — malformed JSON / validation / question_type mismatch — exactly
        as before)

    A candidate that failed but was recovered by a later valid one is NOT logged
    as a WARNING; instead a single INFO records that trailing blocks were skipped
    (a signal the prompt's block-last contract is eroding). ``"discrete_count"``
    is intentionally unsupported at runtime — see the module docstring.

    Selection here is STRICT-only, which is why the publish path does not use it:
    ``value_extraction._run_ladder`` also repairs each candidate in place, so a
    malformed final block beats a lower-ranked valid one and no superseded draft
    gets published. Callers of this function read a block for telemetry or
    analysis, where an unrepaired None is the honest answer.
    """
    candidates = extract_json_block_candidates(rationale_text)
    if not candidates:
        logger.info("No JSON block found in rationale for question_type=%s", question_type)
        return None

    last_index = len(candidates) - 1
    for index, candidate in enumerate(candidates):
        # Log a failure only on the LAST candidate we try: earlier failures are
        # silently skipped (a valid block may still follow), while the final
        # failure's WARNING preserves the honest end-state signal. A single-block
        # rationale (the common case) is index==last, so its logging is unchanged.
        parsed = parse_structured_payload(candidate, question_type, log_failures=index == last_index)
        if parsed is not None:
            if index > 0:
                logger.info(
                    "Structured-block selection skipped %d trailing block(s) that did not validate "
                    "for question_type=%s before a usable one; the model may be emitting blocks after "
                    "the forecast block (prompt block-last contract eroding).",
                    index,
                    question_type,
                )
            return parsed
    return None
