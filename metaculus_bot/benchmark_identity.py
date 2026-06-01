"""Pure identity/type helpers for benchmark and forecast-report objects.

These functions are stateless over a single ``BenchmarkForBot`` (or its forecast
reports): clean model-name derivation, stacking-strategy detection, question-type
classification, and the substring identifiers used for include/exclude filtering.
They were extracted from ``CorrelationAnalyzer`` (which now keeps thin delegating
wrappers) so the identity concern lives apart from correlation math, ingestion,
and reporting.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution

logger = logging.getLogger(__name__)


def extract_clean_model_name(model_path: str) -> str:
    """Extract a clean model name from a model path like 'openrouter/deepseek/deepseek-r1-0528:free'."""
    # Split by '/' and take the last part, then split by ':' to remove variant suffixes
    return model_path.split("/")[-1].split(":")[0]


def extract_model_name(benchmark: BenchmarkForBot) -> str:
    """Extract clean model name from benchmark.

    For the new ensemble configuration, this returns the bot name directly
    (e.g., 'qwen3_glm_mean', 'qwen3-235b') rather than trying to parse
    individual model names from the forecasters list.
    """
    try:
        # First, try simple approach: if benchmark name looks like a model name, use it directly
        simple_name = benchmark.name.strip()
        # Check if it's a simple model name without complex parsing
        if (
            simple_name and "|" not in simple_name and " " not in simple_name and len(simple_name.split("-")) <= 3
        ):  # Simple model names like "qwen3-235b"
            return simple_name

        # Extract from LLM config - handle both old and new formats
        llms = benchmark.forecast_bot_config.get("llms", {})

        # New format: check the "default" LLM which is used for forecasting
        if "default" in llms and isinstance(llms["default"], dict):
            forecaster_config = llms["default"]
            if "model" in forecaster_config:
                model_path = forecaster_config["model"]
                return extract_clean_model_name(model_path)

        # Legacy format: check forecasters array
        if "forecasters" in llms and llms["forecasters"]:
            forecasters = llms["forecasters"]

            # For single model bots, use the model name
            if len(forecasters) == 1:
                first_forecaster = forecasters[0]
                if isinstance(first_forecaster, dict):
                    if "original_model" in first_forecaster:
                        model_path = first_forecaster["original_model"]
                        return extract_clean_model_name(model_path)
                    elif "model" in first_forecaster:
                        model_path = first_forecaster["model"]
                        return extract_clean_model_name(model_path)

            # For multi-model ensembles, generate ensemble name from components
            elif len(forecasters) > 1:
                model_components = []
                for forecaster in forecasters:
                    if isinstance(forecaster, dict):
                        model_key = "original_model" if "original_model" in forecaster else "model"
                        if model_key in forecaster:
                            model_name = forecaster[model_key].split("/")[-1]
                            if "qwen3" in model_name:
                                model_components.append("qwen3")
                            elif "glm" in model_name:
                                model_components.append("glm")
                            elif "gpt" in model_name:
                                model_components.append("gpt5")
                            elif "claude" in model_name:
                                model_components.append("claude")
                            elif "deepseek" in model_name:
                                model_components.append("deepseek")
                            else:
                                # Fallback: use last part of model name
                                model_components.append(model_name.split("-")[0])

                if model_components:
                    ensemble_base = "_".join(sorted(set(model_components)))
                    config = benchmark.forecast_bot_config
                    if "aggregation_strategy" in config:
                        strategy = config["aggregation_strategy"]
                        if isinstance(strategy, Enum):
                            return f"{ensemble_base}_{strategy.value}"
                        elif isinstance(strategy, str):
                            return f"{ensemble_base}_{strategy}"
                    return ensemble_base

        # Fallback to benchmark name parsing
        name_parts = benchmark.name.split(" | ")
        if len(name_parts) >= 3:
            return name_parts[2]  # Model name is usually third part

    except Exception as e:
        logger.warning(f"Could not extract model name from benchmark: {e}")

    return f"model_{hash(benchmark.name) % 10000}"


def identifiers_for_benchmark(benchmark: BenchmarkForBot, model_name: str) -> list[str]:
    """Return identifier strings used for substring matching.

    Uses multiple fields for robustness without normalization beyond lowercasing:
    - cleaned model name we derived
    - the benchmark's own name
    - any model path strings found in forecast_bot_config.llms (default/forecasters/stacker)
    """
    idents: list[str] = []
    try:
        idents.append(model_name)
        if getattr(benchmark, "name", None):
            idents.append(benchmark.name)
        cfg = getattr(benchmark, "forecast_bot_config", {}) or {}
        llms = cfg.get("llms", {}) if isinstance(cfg, dict) else {}
        if isinstance(llms, dict):
            default_cfg = llms.get("default")
            if isinstance(default_cfg, dict) and default_cfg.get("model"):
                idents.append(str(default_cfg.get("model")))
            # forecasters list
            forecasters = llms.get("forecasters")
            if isinstance(forecasters, list):
                for f in forecasters:
                    if isinstance(f, dict):
                        if f.get("original_model"):
                            idents.append(str(f.get("original_model")))
                        if f.get("model"):
                            idents.append(str(f.get("model")))
            # stacker model
            stacker_cfg = llms.get("stacker")
            if isinstance(stacker_cfg, dict) and stacker_cfg.get("model"):
                idents.append(str(stacker_cfg.get("model")))
    except Exception:
        logger.debug(f"Failed to extract identifiers for benchmark {model_name}: unexpected config structure")
    # Deduplicate while preserving order
    seen = set()
    out = []
    for s in idents:
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


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
    except Exception:
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
