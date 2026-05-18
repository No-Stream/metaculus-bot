"""Utility functions for API key management."""

import os

from metaculus_bot.fallback_openrouter import should_route_via_donated_key


def get_openrouter_api_key(model: str) -> str | None:
    """
    Determine the correct OpenRouter API key based on the model provider.

    Uses Metaculus-donated credits for providers covered by the donated key
    (see ``DONATED_KEY_PROVIDERS`` in ``fallback_openrouter``: openai,
    anthropic, google) via ``OAI_ANTH_OPENROUTER_KEY``. Falls back to the
    operator's general ``OPENROUTER_API_KEY`` for everything else (and when
    the donated key isn't set).

    Args:
        model: The model name (e.g., "openrouter/anthropic/claude-sonnet-4")

    Returns:
        The appropriate API key or None if no key is available
    """
    if should_route_via_donated_key(model):
        special_key = os.getenv("OAI_ANTH_OPENROUTER_KEY")
        if special_key:
            return special_key

    # Fall back to general OpenRouter key
    return os.getenv("OPENROUTER_API_KEY")
