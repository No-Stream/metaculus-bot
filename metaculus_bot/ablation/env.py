"""Shared env-var context manager for the probabilistic-tools ablation.

Lives outside ``forecasters.py`` and ``run_stacker.py`` so both modules can
import it without forming a circular dependency. The same primitive is used
by:

* ``forecasters.run_forecasters_for_question`` to FORCE the flag off during
  the forecast stage (so an operator-shell ``PROBABILISTIC_TOOLS_ENABLED=1``
  cannot contaminate cached rationales).
* ``run_stacker.run_stacker_for_arm`` to toggle the flag per arm (off for
  arm A, on for arm B).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

FEATURE_FLAG_ENV = "PROBABILISTIC_TOOLS_ENABLED"


@contextmanager
def probabilistic_tools_enabled(enabled: bool) -> Iterator[None]:
    """Set/unset ``PROBABILISTIC_TOOLS_ENABLED`` for the duration of this context.

    On exit, restores the previous value (or removes the var if it wasn't set).
    The env-flag check ``constants.env_flag_enabled`` returns False for both
    "var unset" and "var set to 'false'/'0'", but explicit deletion is the
    cleanest signal — and it's what lets tests assert "the var was absent
    when the runner was called".
    """
    previous_value = os.environ.get(FEATURE_FLAG_ENV)
    if enabled:
        os.environ[FEATURE_FLAG_ENV] = "1"
    else:
        os.environ.pop(FEATURE_FLAG_ENV, None)
    try:
        yield
    finally:
        if previous_value is None:
            os.environ.pop(FEATURE_FLAG_ENV, None)
        else:
            os.environ[FEATURE_FLAG_ENV] = previous_value
