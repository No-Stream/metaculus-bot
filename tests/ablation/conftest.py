"""Directory-scoped fixtures shared by every ``tests/ablation/`` module.

These tests all want the same throwaway ``AblationCache`` (or just its directory) and the
same mock stacker / parser LLMs, so the fixtures live in a conftest instead of being
imported into each module and re-bound at module scope. A conftest also removes the trap
that came with the re-binding: the autouse ``PROBABILISTIC_TOOLS_ENABLED`` reset below now
applies to every module in this directory, where before a new module could silently omit
it and become order-dependent on whatever an earlier module left in the environment.

Scoping to this directory rather than the root ``tests/conftest.py`` is deliberate:
``cache`` would otherwise shadow pytest's builtin ``cache`` fixture for all of ``tests/``,
``parser_llm`` already means something else in ``tests/test_structured_parse.py``, and the
autouse reset would start deleting that env var for the whole suite.

The question factories, payload builders and mock installers these tests share stay in
``tests/ablation_cli_fakes.py`` and ``tests/ablation_stacker_fakes.py`` — plain modules
holding no fixtures, imported normally.
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
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "abl"


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
