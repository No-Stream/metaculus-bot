"""Metaculus bot package.

This package will gradually house refactored modules such as CLI, prompts, utils, etc.
"""

import os

# Disable litellm's aiohttp transport (default since litellm v1.71.x). Under
# concurrent async bursts that transport raises near-instant connection failures
# that litellm re-wraps as a 1ms ``litellm.Timeout`` — the root cause behind the
# spurious instant-timeout incident (see litellm issue #14895 and
# ``scratch_docs_and_planning/transient_retry_fix.md``). Falling back to the
# httpx transport avoids the pathology. setdefault so an explicit env override
# (e.g. to re-enable aiohttp for testing) still wins. Must run BEFORE the first
# litellm import, though not because litellm reads the variable at import time — it
# re-reads it per transport construction, in
# ``AsyncHTTPHandler._should_use_aiohttp_transport``. What needs the ordering is the
# handful of handlers litellm builds during its OWN import (four, on 1.92.0), which
# freeze onto the aiohttp transport when the default arrives late. Hence this line
# precedes the submodule import below, which pulls forecasting_tools and thus litellm;
# tests/test_aiohttp_transport_flag.py asserts that source order.
os.environ.setdefault("DISABLE_AIOHTTP_TRANSPORT", "true")

from metaculus_bot.question_patches import apply_question_patches  # must follow the env setdefault above

apply_question_patches()
