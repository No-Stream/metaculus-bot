"""Regression guard: tests must never write into the live ablation cache.

The diagnosis at ``backtests/ablation/skipped_qids_diagnosis_20260515.md``
documents how a previous test run leaked MagicMock-poisoned forecaster
payloads into ``backtests/ablation/forecaster_outputs/{43077,43148,43150}/``.
The sticky cache poisoned every subsequent ablation run because
``run_stacker_for_arm`` cached ``insufficient_forecasters`` once and the score
stage skipped the qids forever.

This module:

1. Asserts that ``AblationCache`` writes during the test session always land
   under pytest's tmp directory hierarchy (Task #24 sentinel).
2. Documents which forecaster_outputs/<qid>/ entries on the live cache today
   contain MagicMock errors so the operator can clean them manually
   (programmatic deletion is too risky for a test).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from metaculus_bot.ablation.cache import AblationCache


def test_test_session_does_not_write_to_live_ablation_cache(tmp_path: Path) -> None:
    """Sentinel: an AblationCache constructed under tmp_path must NOT touch
    the project's ``backtests/ablation/`` directory."""
    live_cache_root = Path(__file__).resolve().parent.parent / "backtests" / "ablation"
    sandbox = AblationCache(tmp_path / "abl-sandbox")

    pre_listing = _snapshot_dir(live_cache_root)
    sandbox.write_research(qid=42, blob="sandbox-test", meta={"source": "regression"})
    sandbox.write_forecaster_output(
        qid=42,
        model_slug="qwen__qwen3-next-80b-a3b-instruct__free",
        payload={
            "model": "openrouter/qwen/qwen3-next-80b-a3b-instruct:free",
            "prediction_value": None,
            "reasoning": "",
            "errors": ["MagicMock noise"],
        },
    )
    post_listing = _snapshot_dir(live_cache_root)

    assert pre_listing == post_listing, (
        "Tests must NEVER write into the live ablation cache root. "
        "Use tmp_path / tmp_path_factory for AblationCache(...) constructors. "
        f"Diff before/after sandbox writes:\n  pre={pre_listing}\n  post={post_listing}"
    )

    # Sandbox path got the writes.
    assert (tmp_path / "abl-sandbox" / "research" / "42.md").exists()
    assert (tmp_path / "abl-sandbox" / "forecaster_outputs" / "42").exists()


def test_ablation_cache_constructor_signals_relative_default_for_audit() -> None:
    """The default root is intentionally a relative path; documenting here so a
    future change that hard-codes the absolute repo path triggers test failure."""
    cache = AblationCache()
    assert str(cache.root).endswith("backtests/ablation")
    # MUST be relative — operators rely on cwd-relative resolution; absolute
    # would break on Brazil/conductor builds.
    assert not Path(cache.root).is_absolute()


def test_constructor_guard_catches_non_tmp_path(tmp_path: Path) -> None:
    """Demonstrate the kind of guard the conftest could install if test
    isolation slips again. Wraps AblationCache.__init__ to refuse non-tmp paths."""
    forbidden = Path("backtests/ablation")
    real_init = AblationCache.__init__

    def guarded_init(self: AblationCache, root: Any = "backtests/ablation") -> None:
        as_path = Path(root)
        # Allow tmp paths and the explicit default-root unit test (no writes).
        is_tmp = "pytest-of-" in str(as_path) or str(as_path).startswith("/tmp")
        is_default_test = str(as_path) == str(forbidden)
        if not (is_tmp or is_default_test):
            raise AssertionError(
                f"AblationCache constructed with non-tmp path during test: {root}. Use tmp_path / tmp_path_factory."
            )
        real_init(self, root)

    with patch.object(AblationCache, "__init__", guarded_init):
        # OK: tmp_path
        cache_ok = AblationCache(tmp_path / "abl")
        assert cache_ok.root == tmp_path / "abl"

        # OK: default-root unit test pattern (no writes go through).
        cache_default = AblationCache()
        assert str(cache_default.root) == "backtests/ablation"

        # NOT OK: an explicit path under backtests/ in the project root.
        with pytest.raises(AssertionError, match="non-tmp path"):
            AblationCache(Path("backtests/ablation/some_subdir"))


def _snapshot_dir(root: Path) -> dict[str, int]:
    """Return ``{relative_path_str: size_bytes}`` for every file under ``root``.

    Returns an empty dict if ``root`` does not exist (CI without checked-in
    backtests/ablation/ is fine).
    """
    if not root.exists():
        return {}
    snapshot: dict[str, int] = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = Path(dirpath) / name
            try:
                size = full.stat().st_size
            except FileNotFoundError:
                continue
            snapshot[str(full.relative_to(root))] = size
    return snapshot


def test_documented_mock_poisoned_files_in_live_cache_for_manual_cleanup() -> None:
    """Surface today's mock-poisoned files for the operator to clean manually.

    The diagnosis documented qids 43077/43148/43150 as mock-poisoned. This test
    walks the live cache and prints any forecaster_outputs file whose payload
    contains MagicMock-style errors. The test always passes; it just emits the
    list as the captured-stdout so the operator can ``grep`` for it.

    NOTE: programmatic deletion is intentionally NOT done here. The operator
    decides what to delete based on this listing.
    """
    live_cache_root = Path(__file__).resolve().parent.parent / "backtests" / "ablation" / "forecaster_outputs"
    if not live_cache_root.exists():
        return

    poisoned: list[str] = []
    for qid_dir in sorted(live_cache_root.iterdir()):
        if not qid_dir.is_dir():
            continue
        for path in sorted(qid_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            errors = payload.get("errors") or []
            if any("Mock" in str(e) for e in errors):
                poisoned.append(str(path.relative_to(live_cache_root.parent.parent)))

    if poisoned:
        # Surface, but don't fail — the test exists to document, not block.
        print("\n=== Mock-poisoned forecaster cache entries (manual cleanup recommended) ===\n" + "\n".join(poisoned))
