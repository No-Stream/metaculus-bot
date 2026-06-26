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
# (e.g. to re-enable aiohttp for testing) still wins. Must run BEFORE any litellm
# import: this is the package __init__, executed before any submodule (and thus
# before forecasting_tools/litellm) is imported.
os.environ.setdefault("DISABLE_AIOHTTP_TRANSPORT", "true")

from metaculus_bot.question_patches import apply_question_patches  # noqa: E402  # must follow the env setdefault above

apply_question_patches()
