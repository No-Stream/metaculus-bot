"""Common pytest fixtures for the ``test_ablation_run_stacker*`` modules.

Split from ``tests/ablation_stacker_fakes.py`` so each test module can pull the fixtures in
as one contiguous alias-only import block: the aliases keep these module-level names from
colliding with the same-named test-method parameters (ruff F811), while pytest still
registers each fixture under its own name. ``_ensure_flag_unset`` is autouse, so it applies
to whichever module imports it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from metaculus_bot.ablation.cache import AblationCache
from tests.ablation_stacker_fakes import FEATURE_FLAG


@pytest.fixture
def cache(tmp_path: Path) -> AblationCache:
    return AblationCache(tmp_path / "abl")


@pytest.fixture
def stacker_llm() -> MagicMock:
    return MagicMock(name="stacker_llm")


@pytest.fixture
def fallback_stacker_llm() -> MagicMock:
    return MagicMock(name="fallback_stacker_llm")


@pytest.fixture
def parser_llm() -> MagicMock:
    return MagicMock(name="parser_llm")


@pytest.fixture(autouse=True)
def _ensure_flag_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every test start with the flag explicitly unset so no leakage from earlier tests."""
    monkeypatch.delenv(FEATURE_FLAG, raising=False)
