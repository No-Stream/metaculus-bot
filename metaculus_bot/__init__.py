"""Metaculus bot package.

This package will gradually house refactored modules such as CLI, prompts, utils, etc.
"""

from metaculus_bot.question_patches import apply_question_patches

apply_question_patches()
