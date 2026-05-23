"""Tests for the ablation disk-cache foundation.

Caches are JSON for structured payloads, with raw research blobs as ``.md``
sidecars. All writes are atomic (tempfile + ``os.replace``) so a crash mid-write
leaves the cache in either the old state or the new state — never partial.
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from metaculus_bot.ablation.cache import CACHE_SCHEMA_VERSION, AblationCache, model_slug_to_filename

# ``Any`` is referenced inside test-local fail_on_blob_write monkeypatches; mark
# explicitly so the formatter doesn't strip the import on a usage-only edit.
_ = Any


@pytest.fixture
def cache(tmp_path: Path) -> AblationCache:
    return AblationCache(tmp_path / "abl")


# ---------------------------------------------------------------------------
# Atomic writes / round-trips
# ---------------------------------------------------------------------------


def test_atomic_write_creates_directory_tree(cache: AblationCache) -> None:
    cache.write_research(qid=42, blob="hello", meta={"sources": 3})
    assert (cache.root / "research" / "42.md").exists()
    assert (cache.root / "research" / "42.meta.json").exists()


def test_atomic_write_round_trip_research(cache: AblationCache) -> None:
    blob = "## Gemini\n\nLong research blob."
    meta = {"sources": 7, "gap_count": 2, "providers": ["gemini"]}
    cache.write_research(qid=42, blob=blob, meta=meta)

    result = cache.read_research(qid=42)

    assert result is not None
    read_blob, read_meta = result
    assert read_blob == blob
    assert read_meta["sources"] == 7
    assert read_meta["gap_count"] == 2
    assert read_meta["providers"] == ["gemini"]
    assert read_meta["cache_schema_version"] == CACHE_SCHEMA_VERSION


def test_atomic_write_round_trip_leakage_screen(cache: AblationCache) -> None:
    payload = {
        "is_leaked": False,
        "detector_response": "No clear leak found.",
        "detector_model": "openrouter/z-ai/glm-4.5-air:free",
        "screened_at": "2026-05-13T12:00:00",
    }

    cache.write_leakage_screen(qid=42, payload=payload)
    result = cache.read_leakage_screen(qid=42)

    assert result is not None
    assert result["is_leaked"] is False
    assert result["detector_response"] == "No clear leak found."
    assert result["detector_model"] == "openrouter/z-ai/glm-4.5-air:free"
    assert result["cache_schema_version"] == CACHE_SCHEMA_VERSION


def test_atomic_write_round_trip_forecaster_output(cache: AblationCache) -> None:
    payload = {
        "prediction_value": 0.62,
        "reasoning": "## Reasoning\n\nLooks plausible.",
        "errors": [],
        "model": "openrouter/qwen/qwen3-next-80b-a3b-instruct:free",
        "ran_at": "2026-05-13T12:30:00",
    }
    slug = "qwen__qwen3-next-80b-a3b-instruct__free"

    cache.write_forecaster_output(qid=42, model_slug=slug, payload=payload)
    result = cache.read_forecaster_output(qid=42, model_slug=slug)

    assert result is not None
    assert result["prediction_value"] == 0.62
    assert result["reasoning"] == "## Reasoning\n\nLooks plausible."
    assert result["model"] == "openrouter/qwen/qwen3-next-80b-a3b-instruct:free"


def test_atomic_write_round_trip_stacker_output(cache: AblationCache) -> None:
    payload_stack = {
        "stacker_prediction": 0.55,
        "meta_reasoning": "Arm A reasoning.",
        "computed_quantities": "",
        "cross_model_aggregation": "",
        "ran_at": "2026-05-13T13:00:00",
        "stacker_model_used": "primary",
    }
    payload_pdf = {
        "stacker_prediction": 0.60,
        "meta_reasoning": "Arm B reasoning.",
        "computed_quantities": "## Computed quantities\n\nPooled = 0.58",
        "cross_model_aggregation": "Cross-model block.",
        "ran_at": "2026-05-13T13:05:00",
        "stacker_model_used": "fallback",
    }

    cache.write_stacker_output(qid=42, arm="stack", payload=payload_stack)
    cache.write_stacker_output(qid=42, arm="pdf", payload=payload_pdf)

    a = cache.read_stacker_output(qid=42, arm="stack")
    b = cache.read_stacker_output(qid=42, arm="pdf")
    assert a is not None and b is not None
    assert a["stacker_prediction"] == 0.55
    assert b["stacker_prediction"] == 0.60
    assert b["computed_quantities"] == "## Computed quantities\n\nPooled = 0.58"
    assert b["stacker_model_used"] == "fallback"


def test_atomic_write_round_trip_pruned_research(cache: AblationCache) -> None:
    sanitized_blob = "## Background\n\nGenerally relevant context, with the resolution-revealing line stripped."
    meta = {
        "qid": 42,
        "original_chars": 5000,
        "sanitized_chars": 1234,
        "redactions": [
            {"original_excerpt": "weekly total 17,237,442", "reason": "directly states resolution"},
        ],
        "redactor_invocation_id": "abcd1234",
        "pruned_at": "2026-05-13T18:00:00",
    }

    cache.write_pruned_research(qid=42, sanitized_blob=sanitized_blob, meta=meta)
    result = cache.read_pruned_research(qid=42)

    assert result is not None
    read_blob, read_meta = result
    assert read_blob == sanitized_blob
    assert read_meta["original_chars"] == 5000
    assert read_meta["sanitized_chars"] == 1234
    assert read_meta["redactions"][0]["original_excerpt"] == "weekly total 17,237,442"
    assert read_meta["cache_schema_version"] == CACHE_SCHEMA_VERSION


def test_has_pruned_research_reflects_state(cache: AblationCache) -> None:
    assert cache.has_pruned_research(qid=42) is False
    cache.write_pruned_research(
        qid=42,
        sanitized_blob="sanitized content",
        meta={
            "qid": 42,
            "original_chars": 100,
            "sanitized_chars": 50,
            "redactions": [],
            "redactor_invocation_id": "z",
            "pruned_at": "2026-05-13T18:00:00",
        },
    )
    assert cache.has_pruned_research(qid=42) is True


# ---------------------------------------------------------------------------
# Cache miss & partial-state behavior
# ---------------------------------------------------------------------------


def test_cache_miss_returns_none(cache: AblationCache) -> None:
    assert cache.read_research(qid=999) is None
    assert cache.read_leakage_screen(qid=999) is None
    assert cache.read_forecaster_output(qid=999, model_slug="anything") is None
    assert cache.read_stacker_output(qid=999, arm="stack") is None
    assert cache.list_forecaster_outputs(qid=999) == {}
    assert cache.read_pruned_research(qid=999) is None


def test_partial_pruned_research_returns_none(cache: AblationCache) -> None:
    pruned_dir = cache.root / "research_pruned"
    pruned_dir.mkdir(parents=True, exist_ok=True)

    (pruned_dir / "42.md").write_text("sanitized blob without meta")
    assert cache.read_pruned_research(qid=42) is None

    (pruned_dir / "42.md").unlink()
    payload = {"qid": 42, "cache_schema_version": CACHE_SCHEMA_VERSION}
    (pruned_dir / "42.meta.json").write_text(json.dumps(payload))
    assert cache.read_pruned_research(qid=42) is None


def test_partial_research_returns_none(cache: AblationCache, tmp_path: Path) -> None:
    research_dir = cache.root / "research"
    research_dir.mkdir(parents=True, exist_ok=True)

    (research_dir / "42.md").write_text("blob without meta")
    assert cache.read_research(qid=42) is None

    (research_dir / "42.md").unlink()
    payload = {"sources": 3, "cache_schema_version": CACHE_SCHEMA_VERSION}
    (research_dir / "42.meta.json").write_text(json.dumps(payload))
    assert cache.read_research(qid=42) is None


# ---------------------------------------------------------------------------
# Failure rollback
# ---------------------------------------------------------------------------


def test_atomic_write_rolls_back_on_failure(cache: AblationCache, monkeypatch: pytest.MonkeyPatch) -> None:
    cache.write_leakage_screen(qid=42, payload={"is_leaked": False, "detector_response": "ok"})
    target = cache.root / "leakage_screens" / "42.json"
    original_bytes = target.read_bytes()

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated atomic-replace failure")

    monkeypatch.setattr(os, "replace", boom)

    with pytest.raises(OSError, match="simulated atomic-replace failure"):
        cache.write_leakage_screen(qid=42, payload={"is_leaked": True, "detector_response": "leaked"})

    assert target.read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


def test_schema_version_mismatch_raises(cache: AblationCache) -> None:
    leakage_dir = cache.root / "leakage_screens"
    leakage_dir.mkdir(parents=True, exist_ok=True)
    (leakage_dir / "42.json").write_text(
        json.dumps({"is_leaked": False, "detector_response": "stale", "cache_schema_version": 999})
    )

    with pytest.raises(ValueError, match="cache_schema_version"):
        cache.read_leakage_screen(qid=42)


def test_schema_version_auto_injected(cache: AblationCache) -> None:
    cache.write_leakage_screen(qid=42, payload={"is_leaked": True, "detector_response": "leaked"})

    on_disk = json.loads((cache.root / "leakage_screens" / "42.json").read_text())
    assert on_disk["cache_schema_version"] == CACHE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# qids manifest
# ---------------------------------------------------------------------------


def test_qids_manifest_round_trip(cache: AblationCache) -> None:
    manifest = {
        1: {"tournament": "spring-aib-2026", "type": "binary"},
        2: {"tournament": "spring-aib-2026", "type": "numeric"},
    }
    cache.write_qids_manifest(manifest)
    loaded = cache.read_qids_manifest()
    assert loaded == manifest


def test_qids_manifest_append_merges(cache: AblationCache) -> None:
    cache.write_qids_manifest({1: {"type": "binary"}, 2: {"type": "numeric"}})
    merged = cache.append_qids_manifest({3: {"type": "multiple_choice"}, 4: {"type": "binary"}})

    assert set(merged.keys()) == {1, 2, 3, 4}
    on_disk = cache.read_qids_manifest()
    assert set(on_disk.keys()) == {1, 2, 3, 4}
    assert on_disk[3]["type"] == "multiple_choice"


def test_qids_manifest_append_overwrites_existing_qid(cache: AblationCache) -> None:
    cache.write_qids_manifest({1: {"type": "binary", "tournament": "old"}})
    merged = cache.append_qids_manifest({1: {"type": "binary", "tournament": "new"}})

    assert merged[1]["tournament"] == "new"
    on_disk = cache.read_qids_manifest()
    assert on_disk[1]["tournament"] == "new"


def test_qids_manifest_missing_returns_empty(cache: AblationCache) -> None:
    assert cache.read_qids_manifest() == {}


# ---------------------------------------------------------------------------
# Forecaster listing
# ---------------------------------------------------------------------------


def test_list_forecaster_outputs(cache: AblationCache) -> None:
    slugs = [
        "qwen__qwen3-next-80b-a3b-instruct__free",
        "minimax__minimax-m2.5__free",
        "z-ai__glm-4.5-air__free",
    ]
    for i, slug in enumerate(slugs):
        cache.write_forecaster_output(
            qid=42,
            model_slug=slug,
            payload={"prediction_value": 0.5 + 0.01 * i, "reasoning": f"r{i}", "errors": []},
        )

    cache.write_forecaster_output(
        qid=99,
        model_slug="other__slug",
        payload={"prediction_value": 0.1, "reasoning": "different qid", "errors": []},
    )

    listed = cache.list_forecaster_outputs(qid=42)
    assert set(listed.keys()) == set(slugs)
    for slug in slugs:
        assert listed[slug]["reasoning"].startswith("r")


def test_list_forecaster_outputs_filters_obsolete_lineup(cache: AblationCache) -> None:
    """m2: list_forecaster_outputs(lineup_filter=...) ignores files outside the lineup.

    When the lineup rotates, the forecaster_outputs/<qid>/ directory may
    contain stale files for retired models. Without a filter, those payloads
    pollute the stacker prompt and cache-hit accounting.
    """
    current_lineup = [
        "openrouter/qwen/qwen3-next-80b-a3b-instruct:free",
        "openrouter/minimax/minimax-m2.5:free",
    ]
    obsolete_models = [
        "openrouter/openai/gpt-oss-120b:free",
        "openrouter/z-ai/glm-4.5-air:free",
    ]
    for model in current_lineup + obsolete_models:
        cache.write_forecaster_output(
            qid=42,
            model_slug=model_slug_to_filename(model),
            payload={"prediction_value": 0.5, "reasoning": "r", "errors": []},
        )

    # No filter: every file returned.
    unfiltered = cache.list_forecaster_outputs(qid=42)
    assert len(unfiltered) == 4

    # With filter: only the current lineup's files.
    filtered = cache.list_forecaster_outputs(qid=42, lineup_filter=current_lineup)
    assert set(filtered.keys()) == {model_slug_to_filename(m) for m in current_lineup}


# ---------------------------------------------------------------------------
# model_slug_to_filename
# ---------------------------------------------------------------------------


def test_model_slug_to_filename_openrouter_free() -> None:
    slug = model_slug_to_filename("openrouter/qwen/qwen3-next-80b-a3b-instruct:free")
    assert slug == "qwen__qwen3-next-80b-a3b-instruct__free"


def test_model_slug_to_filename_strips_whitespace() -> None:
    slug = model_slug_to_filename("  openrouter/qwen/qwen3-next:free  ")
    assert slug == "qwen__qwen3-next__free"


def test_model_slug_to_filename_no_path_separators() -> None:
    slug = model_slug_to_filename("openrouter/openai/gpt-oss-120b:free")
    assert "/" not in slug
    assert ":" not in slug


def test_model_slug_to_filename_dots_preserved_in_version() -> None:
    slug = model_slug_to_filename("openrouter/minimax/minimax-m2.5:free")
    assert slug == "minimax__minimax-m2.5__free"


def test_model_slug_to_filename_no_openrouter_prefix() -> None:
    slug = model_slug_to_filename("anthropic/claude-opus-4.5")
    assert slug == "anthropic__claude-opus-4.5"


# ---------------------------------------------------------------------------
# Score runs
# ---------------------------------------------------------------------------


def test_score_run_writes_timestamped_file(cache: AblationCache) -> None:
    payload = {"mean_delta": 0.04, "n_questions": 30, "metric": "log_score"}
    path = cache.write_score_run(run_id="20260513_120000", payload=payload)

    assert path.exists()
    assert path.name == "run_20260513_120000.json"
    on_disk = json.loads(path.read_text())
    assert on_disk["mean_delta"] == 0.04
    assert on_disk["n_questions"] == 30
    assert on_disk["cache_schema_version"] == CACHE_SCHEMA_VERSION


def test_score_summary_writes_markdown(cache: AblationCache) -> None:
    markdown = "# Ablation summary\n\n- Mean Δ = 0.04\n"
    path = cache.write_score_summary(run_id="20260513_120000", markdown=markdown)

    assert path.exists()
    assert path.name == "summary_20260513_120000.md"
    assert path.read_text() == markdown


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_has_research_reflects_state(cache: AblationCache) -> None:
    assert cache.has_research(qid=42) is False
    cache.write_research(qid=42, blob="content", meta={"sources": 1})
    assert cache.has_research(qid=42) is True


def test_root_accepts_string_path(tmp_path: Path) -> None:
    cache = AblationCache(str(tmp_path / "abl-str"))
    cache.write_research(qid=1, blob="x", meta={"sources": 0})
    assert (Path(tmp_path / "abl-str") / "research" / "1.md").exists()


def test_default_root() -> None:
    """Default root is the repo-relative ``backtests/ablation`` directory."""
    cache = AblationCache()
    assert Path(cache.root).as_posix().endswith("backtests/ablation")


# ---------------------------------------------------------------------------
# M2: write_research / write_pruned_research must write meta BEFORE blob so
# an interrupted write between the two leaves only the meta on disk (which
# fails has_research → next read returns None → safe state for re-running).
# Without this, writing blob first leaves a (new_blob, old_meta) state and
# read_research returns inconsistent data.
# ---------------------------------------------------------------------------


def test_write_research_meta_persists_when_blob_write_fails(
    cache: AblationCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupting between write_research's two atomic ops must leave a recoverable state.

    Mutation test for M2: if the order is reversed (blob first), this test
    fails because blob_v1 stays on disk with no meta — partial state.
    """
    cache.write_research(qid=1, blob="blob_v1", meta={"version": 1})

    call_count = [0]
    original_replace = os.replace

    def fail_on_blob_write(src: Any, dst: Any) -> Any:
        call_count[0] += 1
        # M2 invariant: meta is written FIRST, blob SECOND. Failing the
        # second os.replace simulates a crash between the two writes.
        if call_count[0] == 2:
            raise RuntimeError("interrupted between meta and blob writes")
        return original_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_on_blob_write)

    with pytest.raises(RuntimeError, match="interrupted between meta and blob writes"):
        cache.write_research(qid=1, blob="blob_v2", meta={"version": 2})

    # After M2 fix (meta first, blob second), the failed write leaves:
    #   blob: still "blob_v1" (old) — blob write was interrupted
    #   meta: "version=2" (new) — meta write succeeded
    # has_research returns True (both exist), but the blob is stale.
    # Critically, the OPPOSITE order would put blob_v2 on disk with meta_v1,
    # which is the silently-corrupt state M2 prevents.
    blob_path = cache.root / "research" / "1.md"
    meta_path = cache.root / "research" / "1.meta.json"
    assert meta_path.exists()
    on_disk_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert on_disk_meta["version"] == 2, (
        "M2: meta must be written FIRST so a partial-write state has the "
        "newer meta on disk; otherwise blob_v2 + meta_v1 is silently corrupt."
    )
    # The blob is still v1 (the second write was interrupted).
    assert blob_path.read_text(encoding="utf-8") == "blob_v1"


def test_write_pruned_research_meta_persists_when_blob_write_fails(
    cache: AblationCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same M2 invariant for write_pruned_research."""
    cache.write_pruned_research(qid=1, sanitized_blob="sanitized_v1", meta={"version": 1})

    call_count = [0]
    original_replace = os.replace

    def fail_on_blob_write(src: Any, dst: Any) -> Any:
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("interrupted")
        return original_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_on_blob_write)

    with pytest.raises(RuntimeError, match="interrupted"):
        cache.write_pruned_research(qid=1, sanitized_blob="sanitized_v2", meta={"version": 2})

    meta_path = cache.root / "research_pruned" / "1.meta.json"
    on_disk_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert on_disk_meta["version"] == 2
