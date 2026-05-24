"""Tests for metaculus_bot.ablation.stage_payload factory functions."""

from __future__ import annotations

from datetime import datetime

from metaculus_bot.ablation.stage_payload import make_error_payload, make_success_payload


class TestMakeSuccessPayload:
    def test_minimal_success_payload(self) -> None:
        payload = make_success_payload(
            arm="stack",
            prediction=0.75,
            model_used="primary",
            n_forecasters=4,
        )
        assert payload["success"] is True
        assert payload["arm"] == "stack"
        assert payload["stacker_prediction"] == 0.75
        assert payload["stacker_meta_reasoning"] == ""
        assert payload["computed_quantities"] == {}
        assert payload["cross_model_aggregation"] == ""
        assert payload["stacker_model_used"] == "primary"
        assert payload["n_forecasters_used"] == 4
        assert payload["tools_enabled_at_runtime"] is False
        assert payload["errors"] == []
        datetime.fromisoformat(payload["ran_at"])

    def test_full_success_payload(self) -> None:
        payload = make_success_payload(
            arm="stack_aug",
            prediction={"type": "binary", "prob": 0.42},
            meta_reasoning="The models agree that...",
            computed_quantities={"forecaster_1": "## Math\nBeta posterior = 0.4"},
            cross_model_aggregation="## Cross-model\nMedian = 0.45",
            model_used="fallback",
            n_forecasters=6,
            tools_enabled=True,
            errors=["primary: TimeoutError: stacker hung"],
        )
        assert payload["success"] is True
        assert payload["arm"] == "stack_aug"
        assert payload["stacker_prediction"] == {"type": "binary", "prob": 0.42}
        assert payload["stacker_meta_reasoning"] == "The models agree that..."
        assert payload["computed_quantities"] == {"forecaster_1": "## Math\nBeta posterior = 0.4"}
        assert payload["cross_model_aggregation"] == "## Cross-model\nMedian = 0.45"
        assert payload["stacker_model_used"] == "fallback"
        assert payload["n_forecasters_used"] == 6
        assert payload["tools_enabled_at_runtime"] is True
        assert payload["errors"] == ["primary: TimeoutError: stacker hung"]

    def test_none_cross_model_aggregation(self) -> None:
        payload = make_success_payload(
            arm="pdf",
            prediction=[0.0, 0.01, 0.5, 1.0],
            model_used="structured_math",
            n_forecasters=3,
            cross_model_aggregation=None,
        )
        assert payload["cross_model_aggregation"] is None

    def test_none_model_used(self) -> None:
        payload = make_success_payload(
            arm="median",
            prediction=0.5,
            model_used=None,
            n_forecasters=5,
        )
        assert payload["stacker_model_used"] is None


class TestMakeErrorPayload:
    def test_minimal_error_payload(self) -> None:
        payload = make_error_payload(
            arm="stack",
            reason="insufficient_forecasters",
            model_used=None,
            n_forecasters=1,
        )
        assert payload["success"] is False
        assert payload["arm"] == "stack"
        assert payload["reason"] == "insufficient_forecasters"
        assert payload["stacker_prediction"] is None
        assert payload["stacker_meta_reasoning"] == ""
        assert payload["computed_quantities"] == {}
        assert payload["cross_model_aggregation"] == ""
        assert payload["stacker_model_used"] is None
        assert payload["n_forecasters_used"] == 1
        assert payload["tools_enabled_at_runtime"] is False
        assert payload["errors"] == []
        datetime.fromisoformat(payload["ran_at"])

    def test_full_error_payload(self) -> None:
        payload = make_error_payload(
            arm="stack_aug",
            reason="stacker_failed",
            prediction=None,
            meta_reasoning="median_fallback: both stackers failed",
            computed_quantities={"f1": "md"},
            cross_model_aggregation="## Cross-model",
            model_used="primary",
            n_forecasters=4,
            tools_enabled=True,
            errors=["primary: ValueError: xyz", "fallback: TimeoutError"],
        )
        assert payload["success"] is False
        assert payload["reason"] == "stacker_failed"
        assert payload["stacker_meta_reasoning"] == "median_fallback: both stackers failed"
        assert payload["computed_quantities"] == {"f1": "md"}
        assert payload["cross_model_aggregation"] == "## Cross-model"
        assert payload["stacker_model_used"] == "primary"
        assert payload["n_forecasters_used"] == 4
        assert payload["tools_enabled_at_runtime"] is True
        assert payload["errors"] == ["primary: ValueError: xyz", "fallback: TimeoutError"]

    def test_nonfinite_output_error_with_meta_reasoning(self) -> None:
        payload = make_error_payload(
            arm="stack",
            reason="stacker_nonfinite_output",
            meta_reasoning="The stacker produced NaN",
            model_used="primary",
            n_forecasters=3,
            errors=["primary: stacker output contained NaN/inf"],
        )
        assert payload["success"] is False
        assert payload["reason"] == "stacker_nonfinite_output"
        assert payload["stacker_meta_reasoning"] == "The stacker produced NaN"


class TestPayloadTypeCompatibility:
    """Verify the factory output matches the TypedDict shape."""

    def test_success_payload_is_stage_payload(self) -> None:
        payload = make_success_payload(
            arm="mean",
            prediction=0.5,
            model_used="simple_aggregation",
            n_forecasters=4,
        )
        assert isinstance(payload, dict)
        expected_keys = {
            "success",
            "arm",
            "stacker_prediction",
            "stacker_meta_reasoning",
            "computed_quantities",
            "cross_model_aggregation",
            "stacker_model_used",
            "n_forecasters_used",
            "ran_at",
            "tools_enabled_at_runtime",
            "errors",
        }
        assert set(payload.keys()) == expected_keys

    def test_error_payload_has_reason_key(self) -> None:
        payload = make_error_payload(
            arm="stack",
            reason="stacker_failed",
            model_used=None,
            n_forecasters=2,
        )
        assert "reason" in payload
        assert set(payload.keys()) == {
            "success",
            "arm",
            "reason",
            "stacker_prediction",
            "stacker_meta_reasoning",
            "computed_quantities",
            "cross_model_aggregation",
            "stacker_model_used",
            "n_forecasters_used",
            "ran_at",
            "tools_enabled_at_runtime",
            "errors",
        }
