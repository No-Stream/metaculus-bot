"""Reading ONE group out of a robots.txt: what the host tells ``Google-Extended`` it may read.

Gemini's ``url_context`` retrieval honours the ``Google-Extended`` product token, and a host
that disallows it refuses the fetch on the server side. Proven live 2026-09-03: the same paid
read that retrieved a robots-allowed host came back ``URL_RETRIEVAL_STATUS_ERROR`` on
``internationalaisafetyreport.org``, whose (Cloudflare-managed) robots.txt carries
``User-agent: Google-Extended`` / ``Disallow: /``. Retrying cannot help, so the call is spend
with a known-zero return — which is what makes a pre-check worth one free request.

Only the ``Google-Extended`` group is read. ``urllib.robotparser`` cannot express that:
``can_fetch("Google-Extended", url)`` falls back to the ``User-agent: *`` group when no
Google-Extended group exists (verified on 3.12.12 — a robots.txt carrying only
``User-agent: *`` / ``Disallow: /`` answers False), so it would skip the paid read on every
host that merely disallows generic crawlers. That is a different and much broader policy than
the one this pre-check implements: our own free rungs are unaffected by this file, and the
operator's reading of ``Content-Signal: use=reference`` is that reference use is permitted.

Every ambiguity resolves toward PAYING rather than skipping, because a wrong skip loses a
document we could have read while a wrong pay costs one ordinary reader call: an unreadable
robots.txt, a rule this module does not model (an interior ``*``, a ``$`` anchor), and an
absent group all come back "not disallowed".
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# The product token Google documents for generative-AI retrieval, matched exactly rather than
# as a substring: it is a fixed token, and a loose match would let `Google-Extended-Test` or a
# vendor's own `google-extended-crawler` speak for Gemini.
GOOGLE_EXTENDED_AGENT = "google-extended"

# Bound on the one request the pre-check makes, shared by both callers. Fixed rather than derived
# from a caller's remaining budget because the read is a pre-check on a PAID call: a robots.txt
# that has not answered in five seconds is a host that is not going to make the paid read any
# cheaper, and the caller's own wall budget is re-read after this returns.
ROBOTS_FETCH_TIMEOUT_S: float = 5.0

# One robots.txt per host per run, shared by BOTH paid readers (gap-fill v2's `read_document`
# tool and the Tier-1 resolution-source ladder). Shared rather than one cache each, because a
# host's policy is a property of the host: the two paths routinely reach the same government
# domains in one run, and a second read would spend a request to learn what we already know.
# The value is the fetched text, or None when we could not read it — which proceeds to pay.
#
# Bounded and FIFO rather than a plain dict, because this is process-global state that outlives
# one question and a whole run's hosts would otherwise accumulate here.
ROBOTS_TXT_CACHE_MAX_HOSTS = 50
_ROBOTS_TXT_CACHE: OrderedDict[str, str | None] = OrderedDict()


def robots_host(url: str) -> str:
    """``url``'s netloc with any userinfo dropped: the cache key, and what gets logged.

    The port stays, since a host's policy is served per origin. Userinfo goes because it must
    reach neither a robots.txt request nor the archived telemetry line.
    """
    return urlparse(url).netloc.rpartition("@")[2]


def robots_txt_url(url: str) -> str:
    """The robots.txt URL for ``url``'s origin."""
    return f"{urlparse(url).scheme}://{robots_host(url)}/robots.txt"


def reset_robots_cache() -> None:
    """Forget every host's policy. For tests, so one test's read cannot answer another's."""
    _ROBOTS_TXT_CACHE.clear()


async def google_extended_blocks_url(url: str, *, fetch_text: Callable[[str], Awaitable[str | None]]) -> bool:
    """True when ``url``'s host tells ``Google-Extended`` to stay out of that path.

    ``fetch_text`` is injected because the two callers own different SSRF-guarded clients, and
    neither may be bypassed for a request this module makes: v2 hands over its plain fetch ladder
    and Tier-1 its direct fetch, so the preflight, the filtering resolver, the redirect vetting
    and the body cap all apply unchanged. It returns the robots.txt text, or None when the host's
    policy could not be read.

    Fails toward PAYING in every ambiguous case (see the module docstring): an unreadable
    robots.txt, an absent Google-Extended group, and a rule this module will not judge all come
    back False. A wrong skip loses a document we could have read; a wrong pay costs one call.
    """
    host = robots_host(url)
    if host in _ROBOTS_TXT_CACHE:
        robots_txt = _ROBOTS_TXT_CACHE[host]
    else:
        robots_txt = await fetch_text(robots_txt_url(url))
        _ROBOTS_TXT_CACHE[host] = robots_txt
        _ROBOTS_TXT_CACHE.move_to_end(host)
        while len(_ROBOTS_TXT_CACHE) > ROBOTS_TXT_CACHE_MAX_HOSTS:
            _ROBOTS_TXT_CACHE.popitem(last=False)
    if robots_txt is None:
        return False
    return google_extended_disallows(robots_txt, urlparse(url).path)


def google_extended_disallows(robots_txt: str, path: str) -> bool:
    """True when ``robots_txt``'s ``Google-Extended`` group disallows ``path``.

    Longest matching rule wins and an ``Allow`` wins a tie, which is Google's own precedence.
    A rule this module cannot match by prefix is skipped rather than guessed at, so it can only
    ever fail to disallow.
    """
    target = path or "/"
    longest_allow = -1
    longest_disallow = -1
    for allow, rule_path in _google_extended_rules(robots_txt):
        prefix = _matchable_prefix(rule_path)
        if prefix is None or not target.startswith(prefix):
            continue
        if allow:
            longest_allow = max(longest_allow, len(prefix))
        else:
            longest_disallow = max(longest_disallow, len(prefix))
    return longest_disallow > longest_allow


def _google_extended_rules(robots_txt: str) -> list[tuple[bool, str]]:
    """The ``(is_allow, path)`` rules of every ``Google-Extended`` group, in file order.

    A group is a run of consecutive ``User-agent`` lines plus the rules under it, so a group
    naming several agents (``User-agent: Googlebot`` then ``User-agent: Google-Extended``)
    counts, and the first rule line after a group ends the agent list.
    """
    rules: list[tuple[bool, str]] = []
    group_applies = False
    reading_agents = False
    for raw_line in robots_txt.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        field, separator, value = line.partition(":")
        if not separator:
            continue
        field = field.strip().lower()
        value = value.strip()
        if field == "user-agent":
            if not reading_agents:
                group_applies = False
                reading_agents = True
            group_applies = group_applies or value.lower() == GOOGLE_EXTENDED_AGENT
            continue
        reading_agents = False
        if group_applies and field in ("allow", "disallow"):
            rules.append((field == "allow", value))
    return rules


def _matchable_prefix(rule_path: str) -> str | None:
    """The path prefix a rule matches on, or None when this module will not judge it.

    An empty value states nothing (a bare ``Disallow:`` is the standard's own way of allowing
    everything). A trailing ``*`` is the prefix it decorates. Anything needing real glob
    matching — an interior ``*``, a ``$`` anchor, or a bare ``*`` decorating no path at all —
    is skipped rather than guessed at, so the read proceeds and is paid for.
    """
    if not rule_path:
        return None
    candidate = rule_path[:-1] if rule_path.endswith("*") else rule_path
    if "*" in candidate or "$" in candidate:
        return None
    return candidate or None
