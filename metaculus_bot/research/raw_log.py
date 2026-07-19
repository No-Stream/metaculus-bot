"""Raw research-provider payload logging — a durable "tape recorder" for GHA artifacts.

Each research provider formats its raw API return into markdown/prose before the
orchestrator combines it into the research bundle; only that combined text (plus a
post-summarizer briefing) reaches the research archive. That makes the raw evidence
behind a forecast — the exact AskNews articles, the grounding sources Gemini used,
the prediction-market contracts — unrecoverable after the fact, and the summarizer
relevance gate un-auditable (you can't see what it dropped).

``record_raw_research`` captures each provider's raw payload as one JSONL record,
appended to ``<RAW_RESEARCH_LOG_DIR>/raw_research_<run_id>.jsonl``. The four workflow
yamls tee stdout to ``run_logs/`` and upload it wholesale as an artifact, so the raw
log rides along and survives the 90-day window without depending on published
comments. ``scripts/download_raw_research.py`` archives it locally.

Contract: this MUST NEVER break a forecast. It is a best-effort side channel —
serialization failures and IO errors are caught, logged, and swallowed. When the
``RAW_RESEARCH_LOG_ENABLED`` env flag is unset (tests, local runs) every call is a
cheap no-op.
"""

import dataclasses
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path

from metaculus_bot.constants import (
    RAW_RESEARCH_LOG_DIR_DEFAULT,
    RAW_RESEARCH_LOG_DIR_ENV,
    RAW_RESEARCH_LOG_ENABLED_ENV,
    RAW_RESEARCH_MAX_PAYLOAD_CHARS,
    env_flag_enabled,
)

logger = logging.getLogger(__name__)

RAW_RESEARCH_SCHEMA_VERSION = 1


def _json_default(obj: object) -> object:
    """Coerce provider payloads to JSON-native form for ``json.dumps(default=...)``.

    Handles the concrete types real providers pass — pydantic models (AskNews
    ``Article``, the Gemini ``GenerateContentResponse``), dataclasses (the
    prediction-market ``MarketSnapshot``/``MarketMatch`` and resolution-source
    ``FetchResult``), and stray datetimes nested inside those. Anything else falls
    back to ``str`` so a payload with an exotic object degrades to its repr rather
    than raising. ``json.dumps`` calls this recursively for nested values.
    """
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)


def _log_path() -> Path:
    log_dir = os.getenv(RAW_RESEARCH_LOG_DIR_ENV, RAW_RESEARCH_LOG_DIR_DEFAULT)
    run_id = os.getenv("GITHUB_RUN_ID", "local")
    return Path(log_dir) / f"raw_research_{run_id}.jsonl"


def record_raw_research(
    *,
    qid: int | None,
    provider: str,
    phase: str | None = None,
    payload: object,
) -> None:
    """Append one raw provider payload to the run's raw-research JSONL log.

    ``phase`` distinguishes multiple raw payloads from one provider (AskNews's
    ``hot``/``historical`` searches); ``None`` for single-payload providers.
    No-op when ``RAW_RESEARCH_LOG_ENABLED`` is unset. Never raises.
    """
    if not env_flag_enabled(RAW_RESEARCH_LOG_ENABLED_ENV):
        return

    try:
        payload_json = json.dumps(payload, default=_json_default, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001, HARNESS-SCAN-EXEMPT-broad-except  # broad by contract: side-channel must never break a forecast (see module docstring)
        # Serialization guard: a non-str dict key `default=` can't rescue (TypeError),
        # or an arbitrary error raised inside a payload's model_dump/__str__
        # (RuntimeError, AttributeError, ...) must not propagate into the forecast.
        logger.warning(
            "raw_research: could not serialize payload for qid=%s provider=%s phase=%s: %s",
            qid,
            provider,
            phase,
            exc,
        )
        return

    payload_chars = len(payload_json)
    truncated = payload_chars > RAW_RESEARCH_MAX_PAYLOAD_CHARS
    if truncated:
        payload_field: object = {
            "_truncated": True,
            "_original_chars": payload_chars,
            "_preview": payload_json[:RAW_RESEARCH_MAX_PAYLOAD_CHARS],
        }
    else:
        # Round-trip so the outer dump sees JSON-native values (no second default=).
        payload_field = json.loads(payload_json)

    record = {
        "schema_version": RAW_RESEARCH_SCHEMA_VERSION,
        "qid": qid,
        "provider": provider,
        "phase": phase,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "payload_chars": payload_chars,
        "truncated": truncated,
        "payload": payload_field,
    }

    path = _log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001, HARNESS-SCAN-EXEMPT-broad-except  # broad by contract: side-channel must never break a forecast (see module docstring)
        # Write guard: OSError from a bad path, but also UnicodeEncodeError (a
        # ValueError, not OSError) when a payload carries a lone surrogate that
        # survives json.dumps(ensure_ascii=False) and only fails at utf-8 encode.
        logger.warning("raw_research: failed to append record to %s: %s", path, exc)
