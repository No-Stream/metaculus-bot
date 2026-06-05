"""Utility functions for API key management."""

import os

from metaculus_bot.constants import OAI_ANTH_OPENROUTER_KEY_ENV, OPENROUTER_API_KEY_ENV
from metaculus_bot.fallback_openrouter import should_route_via_donated_key


def get_openrouter_api_key(model: str) -> str | None:
    """Return the donated key for providers it covers (openai/anthropic/google), else the general key.

    See ``DONATED_KEY_PROVIDERS`` in ``fallback_openrouter`` for the donated-key provider set.
    """
    if should_route_via_donated_key(model):
        special_key = os.getenv(OAI_ANTH_OPENROUTER_KEY_ENV)
        if special_key:
            return special_key

    return os.getenv(OPENROUTER_API_KEY_ENV)
