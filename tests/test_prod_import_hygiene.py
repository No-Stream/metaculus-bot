"""The prod entry-point import graph must not pull in a dev-only dependency.

The five bot workflows install with ``uv sync --no-dev``, so a module-scope import of a
dev-group package anywhere on the ``metaculus_bot.cli`` import path is an ImportError at
interpreter startup there — while every local and CI gate stays green, because CI installs
``uv sync --dev --frozen``. The one detector ci.yaml names for this class, deptry's DEP004,
is disabled for the obvious offender by pyproject's blanket
``[tool.deptry].per_rule_ignores.DEP004 = ["matplotlib"]``, so it cannot see a module-scope
matplotlib import anywhere in the shipped package.

An import-linter contract is the wrong instrument: grimp resolves function-scoped imports
too, so a contract would flag the legitimately-lazy ``timeseries_anchor`` → ``ts_chart``
edge, and the ``ignore_imports`` entry needed to silence that today would mask the
regression tomorrow.

Hence a child interpreter. It has to be a child: this suite has already imported matplotlib
(``tests/test_ts_chart.py``), so an in-process ``sys.modules`` check would read the test
session's own imports rather than the package's. Cost is one ~5s process.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

# Top-level import names of the ``[dependency-groups].dev`` packages that shipped code could
# plausibly import, each verified dev-only via `uv tree --invert` (nothing in
# [project].dependencies reaches them). ``yaml`` is deliberately absent despite pyyaml's
# dev-group entry: it also arrives at runtime through litellm → tokenizers →
# huggingface-hub, so it is present under --no-dev and asserting on it would be a false
# positive.
DEV_ONLY_MODULES = (
    "matplotlib",
    "statsmodels",
    "numpyro",
    "jax",  # numpyro's backend, dev-only through it
    "arviz",
    "h5py",
    "h5netcdf",
    "hypothesis",
    "pytest",
    "ipykernel",
    "IPython",
    "deptry",
    "grimp",  # import-linter's graph builder
)

_OK_MARKER = "PROD_IMPORT_OK"
_LEAKED_MARKER = "DEV_ONLY_LEAKED:"

_CHILD_SOURCE = textwrap.dedent(
    f"""
    import sys

    import metaculus_bot.cli  # noqa: F401  # prod entry point; pulls forecaster + the research stack

    leaked = sorted(name for name in {DEV_ONLY_MODULES!r} if name in sys.modules)
    print("{_OK_MARKER}")
    print("{_LEAKED_MARKER}" + ",".join(leaked))
    """
)


def test_prod_entry_point_imports_no_dev_only_module() -> None:
    """``import metaculus_bot.cli`` in a fresh interpreter must leave every dev-only package out.

    Asserted on ``cli`` rather than ``forecaster`` because cli imports forecaster, so one
    child process covers the whole prod entry-point graph. The child prints the leaked names
    and its stderr is surfaced on failure, so a failure says which module leaked instead of
    conflating a leak with the package failing to import at all.
    """
    repo_root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(repo_root), os.environ.get("PYTHONPATH")]))}

    result = subprocess.run(
        [sys.executable, "-c", _CHILD_SOURCE],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, (
        f"child interpreter failed to import metaculus_bot.cli (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert _OK_MARKER in result.stdout, (
        f"child produced no import-completed marker\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    leaked_line = next(line for line in result.stdout.splitlines() if line.startswith(_LEAKED_MARKER))
    leaked = [name for name in leaked_line.removeprefix(_LEAKED_MARKER).split(",") if name]

    assert leaked == [], (
        f"dev-only modules imported by the metaculus_bot.cli graph: {leaked}. "
        "These are absent under the bot workflows' `uv sync --no-dev`, so this import "
        "graph now dies at interpreter startup in prod. Move the import behind a lazy "
        "guard (see research/timeseries_anchor.py) or promote the package to "
        "[project].dependencies."
    )
