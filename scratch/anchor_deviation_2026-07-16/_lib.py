"""Shared loaders for the anchor-vs-final deviation analysis (free/local only)."""

import json
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1] / "base_rate_audit_2026-07-16"
PERF = Path(__file__).resolve().parents[1] / "coherence_2026-07-15" / "perf_all_tagged.json"


def basename(m: str) -> str:
    return m.split("/")[-1]


def load_claims() -> list[dict]:
    """Canonical batch selection: prefer originals, skip Nb fallback if original exists."""
    files = {f.stem: f for f in (AUDIT / "extracted").glob("claims_batch*.json")}
    claims = []
    for stem in sorted(files):
        if stem.endswith("b") and stem[:-1] in files:
            continue
        for c in json.loads(files[stem].read_text())["claims"]:
            c["_batch"] = stem
            claims.append(c)
    return claims


def load_manifest() -> dict[str, dict]:
    return {str(m["post_id"]): m for m in json.loads((AUDIT / "sample_manifest.json").read_text())}


def load_perf() -> dict[str, dict]:
    return {str(r["post_id"]): r for r in json.loads(PERF.read_text())}


def per_model_final(rec: dict) -> dict[str, object]:
    """Unified {model_basename: final_forecast} for a question.

    Binary: float prob YES. MC: dict[option->prob]. Prefers per_base_model_forecasts
    when per_model_forecasts is keyed by 'Forecaster N' (stacked) rather than model name.
    """
    pmf = rec.get("per_model_forecasts") or {}
    pbmf = rec.get("per_base_model_forecasts") or {}
    keyed_by_forecaster = any(k.startswith("Forecaster") for k in pmf)
    src = pbmf if (keyed_by_forecaster and pbmf) else pmf
    out = {}
    for k, v in src.items():
        if k.startswith("Forecaster"):
            continue
        if isinstance(v, str) and v.strip().endswith("%"):
            out[k] = float(v.strip().rstrip("%")) / 100.0
        else:
            out[k] = v
    return out
