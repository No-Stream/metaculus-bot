"""The JSON feed a rendered page loads its own figures from, and this run's memory of it.

A JavaScript dashboard's numbers are not in its HTML at any wait condition — they arrive over
XHR after the DOM is ready. Measured 2026-09-03 across six such dashboards: grepping the served
HTML for an API URL found ONE candidate (a Maps key), 3 of 4 hand-guessed endpoints were wrong,
and recording the page's own XHR during a render found a working unauthenticated JSON endpoint
for all six. So the discovery half rides the browser rung (``rendered_fetch``) and this module
is the bookkeeping: pick the body worth serving, remember where it came from, and say so.

The memory is per RUN and keyed by HOST, which is what makes the second question on a host free
— but a host's feed is usually parameterised, so the endpoint one page loads is not necessarily
the one another page's figures come from. That is a disclosure problem rather than a reason not
to reuse it: :func:`derived_api_lead` names the endpoint either way and says explicitly when it
was discovered on a DIFFERENT page, because the section it lands in is captioned primary grading
evidence and a forecaster has to be able to check that the feed covers the quantity asked about.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

from metaculus_bot.research.rendered_fetch import HarvestedJson

# Hosts remembered at once. Bounded and FIFO rather than a plain dict for the same reason the
# fetch caches are: this is process-global state that outlives one question, so a whole run's
# hosts would otherwise accumulate.
DERIVED_ENDPOINT_MAX_HOSTS = 50


@dataclass(frozen=True, slots=True)
class DerivedEndpoint:
    """A JSON feed found by rendering, and the page it was found on."""

    endpoint_url: str
    discovered_on: str


_DERIVED_ENDPOINTS: OrderedDict[str, DerivedEndpoint] = OrderedDict()


def _host_of(url: str) -> str:
    return urlparse(url).hostname or ""


def remember_endpoint(page_url: str, endpoint_url: str) -> None:
    """Record that rendering ``page_url`` revealed ``endpoint_url`` as its data feed.

    First find wins for a host: a later page's feed does not overwrite it, because the
    reuse path already discloses that the endpoint came from another page and churning the
    entry would only make that disclosure less stable across a run.
    """
    host = _host_of(page_url)
    if not host or host in _DERIVED_ENDPOINTS:
        return
    _DERIVED_ENDPOINTS[host] = DerivedEndpoint(endpoint_url=endpoint_url, discovered_on=page_url)
    while len(_DERIVED_ENDPOINTS) > DERIVED_ENDPOINT_MAX_HOSTS:
        _DERIVED_ENDPOINTS.popitem(last=False)


def endpoint_for(page_url: str) -> DerivedEndpoint | None:
    """The JSON feed already known for ``page_url``'s host in this run, if any."""
    return _DERIVED_ENDPOINTS.get(_host_of(page_url))


def reset_derived_endpoints() -> None:
    """Forget every discovered endpoint. For tests, so one run's finds cannot leak into another."""
    _DERIVED_ENDPOINTS.clear()


def largest_json(responses: Sequence[HarvestedJson]) -> HarvestedJson | None:
    """The biggest harvested body, or None when nothing was harvested.

    Size is the only signal available without parsing what a given dashboard's schema means: a
    page fetches its config, its feature flags and its data, and the data is the big one. Ties
    resolve to the FIRST, which is document order, so the pick is deterministic across runs.
    """
    if not responses:
        return None
    return max(responses, key=lambda harvested: len(harvested.body))


def derived_api_lead(endpoint: DerivedEndpoint, page_url: str) -> str:
    """The forecaster-facing lead a served feed carries.

    Two shapes, because the two provenances are genuinely different evidence. Harvested during
    THIS page's own render, the feed is what this page displays. Reused from another page on the
    host, it may be parameterised for that other page — so the lead names where it came from and
    asks the reader to check the coverage, rather than presenting it as this page's data.
    """
    if endpoint.discovered_on == page_url:
        return (
            f"[This page's own HTML carried no readable content. The JSON below is the data feed "
            f"the page loads its figures from ({endpoint.endpoint_url}), read directly.]"
        )
    return (
        f"[This page's own HTML carried no readable content. The JSON below is from "
        f"{endpoint.endpoint_url}, the data feed found earlier in this run on a DIFFERENT page of "
        f"the same host ({endpoint.discovered_on}) — check that it covers the quantity this "
        f"question asks about before relying on it.]"
    )
