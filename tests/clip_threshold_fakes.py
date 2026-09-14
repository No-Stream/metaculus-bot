"""Shared record builders for the clip-threshold sweep tests.

Not named ``test_*`` on purpose: pytest must import this module without collecting it. Every
builder here shapes a raw performance record the way the residual collector writes one, so the
split ``test_clip_threshold_*`` modules all sweep the same synthetic archive.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from metaculus_bot.performance_analysis.clip_threshold_sweep import build_clip_records

# A moment inside each clamp regime, so a factory default never straddles a boundary.
BEFORE_WIDENING = "2026-01-15T00:00:00Z"
AFTER_WIDENING = "2026-06-15T00:00:00Z"
AFTER_FT_0292 = "2026-08-01T00:00:00Z"
AS_OF = datetime(2026, 9, 2, tzinfo=UTC)


def binary_record(
    *,
    question_id: int,
    p_yes: float,
    resolution: bool,
    created_at: str | None = AFTER_WIDENING,
    per_model: dict[str, str] | None = None,
    per_base_model: dict[str, str] | None = None,
    stacker_outcome: str | None = "skipped_config_off",
) -> dict:
    """A minimal binary performance record shaped the way the collector writes one."""
    return {
        "question_id": question_id,
        "type": "binary",
        "our_prob_yes": p_yes,
        "our_forecast_values": [1.0 - p_yes, p_yes],
        "resolution_parsed": resolution,
        "bot_comment_created_at": created_at,
        "per_model_forecasts": per_model or {},
        "per_base_model_forecasts": per_base_model or {},
        "stacker_outcome": stacker_outcome,
        "was_stacked": False,
        "metaculus_scores": {"spot_peer_score": 0.0},
    }


def mc_record(
    *,
    question_id: int,
    options: list[str],
    probs: list[float],
    resolution: str,
    created_at: str | None = AFTER_FT_0292,
    per_model: dict[str, dict[str, float]] | None = None,
) -> dict:
    return {
        "question_id": question_id,
        "type": "multiple_choice",
        "our_prob_yes": None,
        "our_forecast_values": list(probs),
        "options": list(options),
        "resolution_parsed": resolution,
        "bot_comment_created_at": created_at,
        "per_model_forecasts": per_model or {},
        "stacker_outcome": "skipped_config_off",
        "was_stacked": False,
        "metaculus_scores": {"spot_peer_score": 0.0},
    }


def one_binary(**kwargs):
    """The single :class:`ClipRecord` built from one hand-made binary dict."""
    [record] = build_clip_records([binary_record(**kwargs)], "binary").records
    return record


def one_mc(**kwargs):
    [record] = build_clip_records([mc_record(**kwargs)], "multiple_choice").records
    return record


def cli_dataset() -> list[dict]:
    """The 53-record synthetic archive the CLI and the rendered-report tests both sweep."""
    records: list[dict] = []
    base = datetime(2026, 5, 1, tzinfo=UTC)
    for i in range(40):
        records.append(
            binary_record(
                question_id=8000 + i,
                p_yes=0.02 if i % 3 == 0 else 0.45,
                resolution=(i % 5 == 0),
                created_at=(base + timedelta(days=2 * i)).isoformat().replace("+00:00", "Z"),
                per_model={"a": "2.0%", "b": "3.0%", "c": "40.0%"},
            )
        )
    for i in range(12):
        records.append(
            mc_record(
                question_id=8500 + i,
                options=["A", "B", "C"],
                probs=[0.70, 0.29, 0.01],
                resolution="A" if i % 2 else "C",
                created_at=(base + timedelta(days=5 * i)).isoformat().replace("+00:00", "Z"),
            )
        )
    # A degraded_run cohort member, so --exclude-qids has something to drop.
    records.append(binary_record(question_id=44870, p_yes=0.02, resolution=False))
    return records
