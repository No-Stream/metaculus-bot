"""Which question platform a question belongs to, read off its page URL.

The bot publishes to two platforms with one API shape: Metaculus and Mantic's Crucible
competition, a fork of the Metaculus backend. A question knows which one it came from only
through ``page_url`` (``forecasting_tools`` sets the Metaculus URL, ``metaculus_bot.mantic``
overwrites it with the competition host), so this is the one place that reading lives. The
prompts use it to state the platform's own scoring rule and its measured out-of-range base
rate; nothing here reads a run-mode flag, so a locally constructed question with no URL is a
Metaculus question, which is what every existing test and archive record assumes.
"""

from __future__ import annotations

from urllib.parse import urlparse

from forecasting_tools import MetaculusQuestion

from metaculus_bot.constants import MANTIC_HOST, PLATFORM_MANTIC, PLATFORM_METACULUS


def question_platform(question: MetaculusQuestion) -> str:
    """``PLATFORM_MANTIC`` when ``page_url`` is on the Crucible host, else ``PLATFORM_METACULUS``.

    Matches the host and its subdomains, the way ``research.resolution_url_scan`` matches a
    self-reference. ``page_url`` is ``str | None`` and None on a locally constructed question.
    """
    hostname = urlparse(question.page_url or "").hostname
    if hostname is not None and (hostname == MANTIC_HOST or hostname.endswith(f".{MANTIC_HOST}")):
        return PLATFORM_MANTIC
    return PLATFORM_METACULUS


__all__ = ["question_platform"]
