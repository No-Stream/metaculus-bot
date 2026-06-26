"""The package sets DISABLE_AIOHTTP_TRANSPORT before litellm is imported.

litellm's aiohttp transport (default since v1.71.x) raises near-instant
connection failures under concurrent async bursts, which litellm re-wraps as a
1ms ``litellm.Timeout`` (the spurious-instant-timeout incident; litellm #14895).
``metaculus_bot/__init__.py`` sets ``DISABLE_AIOHTTP_TRANSPORT=true`` via
``os.environ.setdefault`` at package-init time, before any litellm import.
"""

import importlib
import os


def test_disable_aiohttp_transport_set_after_import() -> None:
    """Importing the package leaves DISABLE_AIOHTTP_TRANSPORT set to a truthy value."""
    import metaculus_bot  # noqa: F401, HARNESS-SCAN-EXEMPT-function-level-import  # import for env side effect

    assert os.environ.get("DISABLE_AIOHTTP_TRANSPORT") == "true"


def test_setdefault_does_not_clobber_explicit_override(monkeypatch) -> None:
    """An explicit pre-set value survives a package re-import (setdefault semantics).

    Reproduces the package-init line against a pre-seeded env so the no-clobber
    contract is asserted without depending on import-order timing: an operator
    who exports DISABLE_AIOHTTP_TRANSPORT=false (e.g. to A/B the aiohttp transport)
    must keep that value.
    """
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "false")

    # Re-execute the package __init__ body; setdefault must leave "false" intact.
    importlib.reload(importlib.import_module("metaculus_bot"))

    assert os.environ["DISABLE_AIOHTTP_TRANSPORT"] == "false"
