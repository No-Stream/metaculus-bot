"""Guards on the DISABLE_AIOHTTP_TRANSPORT default that ``metaculus_bot/__init__.py`` sets.

litellm's aiohttp transport (default since v1.71.x) raises near-instant connection
failures under concurrent async bursts, which litellm re-wraps as a 1ms
``litellm.Timeout`` (the spurious-instant-timeout incident; litellm #14895). The
package sets ``DISABLE_AIOHTTP_TRANSPORT=true`` via ``os.environ.setdefault`` at
package-init time so litellm falls back to the httpx transport.

Mechanism, stated precisely because getting it wrong already misled one test design:
litellm 1.92.0 does NOT read this variable at import time. The ``os.getenv`` call lives
in ``AsyncHTTPHandler._should_use_aiohttp_transport()``
(``litellm/llms/custom_httpx/http_handler.py:941``) and runs once per transport
construction, so a late default still reaches every handler built afterwards. What
depends on ordering is the handful of ``AsyncHTTPHandler``s litellm builds during its
OWN import — four, measured on 1.92.0 — which freeze onto ``LiteLLMAiohttpTransport``
when the default arrives late and onto ``AsyncHTTPTransport`` when it arrives first.
Those handlers are why the invariant guarded here is source ORDER inside the package
``__init__`` rather than any post-import observable: once the package is in
``sys.modules`` no in-process assertion can distinguish the two orders.
"""

import ast
import importlib
import os
from pathlib import Path


def test_package_import_sets_disable_aiohttp_transport() -> None:
    """Importing the package leaves DISABLE_AIOHTTP_TRANSPORT set to a truthy value.

    Catches deletion of the setdefault line, and nothing more: by the time this runs the
    package is a ``sys.modules`` cache hit, so the variable is set no matter when it was
    set. Ordering is guarded by the AST test below.
    """
    import metaculus_bot  # noqa: F401  # HARNESS-SCAN-EXEMPT-function-level-import  # import for env side effect

    assert os.environ.get("DISABLE_AIOHTTP_TRANSPORT") == "true"


def test_setdefault_precedes_every_import_in_package_init() -> None:
    """The setdefault must come before any import that can pull litellm in.

    Guarded on source order because the invariant is carried by the POSITION of one line
    in ``__init__.py``: commit 276ecf2 hoisted ``question_patches``'s own
    ``forecasting_tools`` import to module scope, so the function-scoped import whose
    noqa comment used to carry the invariant mechanically is gone.

    Asserting against EVERY import except ``import os`` (which the setdefault itself
    needs) beats enumerating ``metaculus_bot.*`` / ``forecasting_tools.*``: it needs no
    maintained prefix list and also catches a third-party import hoisted above the line
    that transitively pulls litellm.
    """
    import metaculus_bot  # HARNESS-SCAN-EXEMPT-function-level-import  # kept in-function so the formatter cannot strip it

    body = ast.parse(Path(metaculus_bot.__file__).read_text()).body
    setdefault_idx = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "os.environ.setdefault"
        and getattr(node.value.args[0], "value", None) == "DISABLE_AIOHTTP_TRANSPORT"
    )
    early_imports = [
        ast.unparse(node)
        for node in body[:setdefault_idx]
        if isinstance(node, ast.Import | ast.ImportFrom) and ast.unparse(node) != "import os"
    ]

    assert early_imports == [], f"imports before the DISABLE_AIOHTTP_TRANSPORT setdefault: {early_imports}"


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
