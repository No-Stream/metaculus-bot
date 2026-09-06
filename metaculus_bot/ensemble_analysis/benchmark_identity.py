"""Pure identity/type helpers for benchmark and forecast-report objects.

These functions are stateless over a single ``BenchmarkForBot`` (or its forecast
reports): clean model-name derivation, stacking-strategy detection, question-type
classification, and the substring identifiers used for include/exclude filtering.
They were extracted from ``CorrelationAnalyzer`` (which now keeps thin delegating
wrappers) so the identity concern lives apart from correlation math, ingestion,
and reporting.
"""

from __future__ import annotations

import hashlib
import logging
from enum import Enum
from typing import Any

from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution

logger = logging.getLogger(__name__)

# A benchmark config is arbitrary archived JSON, so walking it can hit a missing key,
# a string where a dict was expected, or a short list. Those are the failure modes the
# naming helpers below absorb; anything else is a real bug and propagates.
_MALFORMED_CONFIG_ERRORS = (AttributeError, KeyError, TypeError, IndexError)

# Substring → family token used to name a multi-model ensemble. Order matters: the
# first match wins, so a model name matching two needles takes the earlier token.
_MODEL_FAMILY_TOKENS: tuple[tuple[str, str], ...] = (
    ("qwen3", "qwen3"),
    ("glm", "glm"),
    ("gpt", "gpt5"),
    ("claude", "claude"),
    ("deepseek", "deepseek"),
)


def extract_clean_model_name(model_path: str) -> str:
    """Extract a clean model name from a model path like 'openrouter/deepseek/deepseek-r1-0528:free'."""
    # Split by '/' and take the last part, then split by ':' to remove variant suffixes
    return model_path.split("/")[-1].split(":")[0]


def _looks_like_a_bare_model_name(name: str) -> bool:
    """True for a name already shaped like a model id, e.g. ``qwen3-235b``."""
    return bool(name) and "|" not in name and " " not in name and len(name.split("-")) <= 3


def _model_family_token(model_name: str) -> str:
    """The ensemble-name token for one model, e.g. ``glm-4.5`` → ``glm``."""
    for needle, token in _MODEL_FAMILY_TOKENS:
        if needle in model_name:
            return token
    return model_name.split("-")[0]


def _name_from_default_llm_slot(llms: Any) -> str | None:
    """Name off the modern ``llms.default`` slot; None when that slot says nothing."""
    if not isinstance(llms, dict):
        return None
    default_config = llms.get("default")
    if isinstance(default_config, dict) and "model" in default_config:
        return extract_clean_model_name(default_config["model"])
    return None


def _ensemble_name_from_forecasters(forecasters: list, config: dict) -> str | None:
    """Name a multi-model ensemble as its sorted family tokens plus the strategy."""
    family_tokens: list[str] = []
    for forecaster in forecasters:
        if not isinstance(forecaster, dict):
            continue
        model_key = "original_model" if "original_model" in forecaster else "model"
        if model_key in forecaster:
            family_tokens.append(_model_family_token(forecaster[model_key].split("/")[-1]))

    if not family_tokens:
        return None

    ensemble_base = "_".join(sorted(set(family_tokens)))
    strategy = config.get("aggregation_strategy")
    if isinstance(strategy, Enum):
        return f"{ensemble_base}_{strategy.value}"
    if isinstance(strategy, str):
        return f"{ensemble_base}_{strategy}"
    return ensemble_base


def _name_from_legacy_forecasters(llms: Any, config: dict) -> str | None:
    """Name off the legacy ``llms.forecasters`` array — one model, or an ensemble."""
    forecasters = llms.get("forecasters") if isinstance(llms, dict) else None
    if not forecasters:
        return None

    if len(forecasters) == 1:
        first_forecaster = forecasters[0]
        if isinstance(first_forecaster, dict):
            for model_key in ("original_model", "model"):
                if model_key in first_forecaster:
                    return extract_clean_model_name(first_forecaster[model_key])
        return None

    return _ensemble_name_from_forecasters(forecasters, config)


def extract_model_name(benchmark: BenchmarkForBot) -> str:
    """Extract clean model name from benchmark.

    Four sources, in precedence order: the bot name when it already looks like a
    model id (the new ensemble configs name themselves, e.g. 'qwen3_glm_mean',
    'qwen3-235b'), the modern ``llms.default`` slot, the legacy ``llms.forecasters``
    array, then the third pipe-delimited field of the benchmark name. A deterministic
    digest is the last resort so every benchmark still gets a stable key.
    """
    try:
        simple_name = benchmark.name.strip()
        if _looks_like_a_bare_model_name(simple_name):
            return simple_name

        config = benchmark.forecast_bot_config
        llms = config.get("llms", {})
        for name in (_name_from_default_llm_slot(llms), _name_from_legacy_forecasters(llms, config)):
            if name is not None:
                return name

        name_parts = benchmark.name.split(" | ")
        if len(name_parts) >= 3:
            return name_parts[2]  # Model name is usually third part

    except _MALFORMED_CONFIG_ERRORS as e:
        logger.warning(f"Could not extract model name from benchmark: {e}")

    digest = hashlib.sha256(benchmark.name.encode()).hexdigest()[:12]
    return f"model_{digest}"


def _model_paths_in_llm_config(llms: dict) -> list[str]:
    """Every model path the config names, in default → forecasters → stacker order."""
    paths: list[str] = []

    default_config = llms.get("default")
    if isinstance(default_config, dict) and default_config.get("model"):
        paths.append(str(default_config["model"]))

    forecasters = llms.get("forecasters")
    if isinstance(forecasters, list):
        for forecaster in forecasters:
            if not isinstance(forecaster, dict):
                continue
            paths.extend(str(forecaster[key]) for key in ("original_model", "model") if forecaster.get(key))

    stacker_config = llms.get("stacker")
    if isinstance(stacker_config, dict) and stacker_config.get("model"):
        paths.append(str(stacker_config["model"]))

    return paths


def _deduplicated_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def identifiers_for_benchmark(benchmark: BenchmarkForBot, model_name: str) -> list[str]:
    """Return identifier strings used for substring matching.

    Uses multiple fields for robustness without normalization beyond lowercasing:
    - cleaned model name we derived
    - the benchmark's own name
    - any model path strings found in forecast_bot_config.llms (default/forecasters/stacker)
    """
    idents: list[str] = [model_name]
    try:
        if getattr(benchmark, "name", None):
            idents.append(benchmark.name)
        cfg = getattr(benchmark, "forecast_bot_config", {}) or {}
        llms = cfg.get("llms", {}) if isinstance(cfg, dict) else {}
        if isinstance(llms, dict):
            idents.extend(_model_paths_in_llm_config(llms))
    except _MALFORMED_CONFIG_ERRORS:
        logger.debug(f"Failed to extract identifiers for benchmark {model_name}: unexpected config structure")
    return _deduplicated_nonempty(idents)


def is_stacking_benchmark(benchmark: BenchmarkForBot | None) -> bool:
    """Return True if the provided benchmark used STACKING aggregation.

    Single canonical detection: forecast_bot_config['aggregation_strategy'] == 'stacking'
    (supports enum-like objects with .value or plain strings).
    """
    if benchmark is None:
        return False
    try:
        cfg = benchmark.forecast_bot_config or {}
        strat = cfg.get("aggregation_strategy")
        if isinstance(strat, Enum):
            strat = strat.value
        if isinstance(strat, str):
            return strat.lower() == "stacking"
    except _MALFORMED_CONFIG_ERRORS:
        logger.debug(f"Failed to detect stacking strategy for benchmark: {benchmark.name}")
    return False


def get_question_type(report: Any) -> str:
    """Determine question type from report."""
    prediction = report.prediction

    if isinstance(prediction, (int, float)):
        return "binary"

    if isinstance(prediction, PredictedOptionList):
        return "multiple_choice"

    if isinstance(prediction, NumericDistribution):
        return "numeric"

    return "binary"
