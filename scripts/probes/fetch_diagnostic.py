"""Measure, from wherever this runs, WHY the resolution-source fetcher gets 403s, and whether
the production ladder recovers them.

Background: the Tier-1 resolution-source fetcher
(``metaculus_bot/research/resolution_source.py``) is refused with HTTP 403 by
Akamai-fronted federal hosts — bls.gov, cdc.gov, fsis.usda.gov — but only when the
bot runs on a GitHub Actions runner. The identical client, on the operator's laptop
and on their EC2 box, gets 200 from the same URLs. Two explanations fit that split
and they call for completely different fixes: the runner's TLS/HTTP fingerprint is
being scored (fixable with Chrome impersonation) or the runner's egress IP range is
blocked outright (not fixable client-side; needs an archival route). The 2026-09-04
run from the runner settled it: the four Akamai hosts are fingerprint-scored and the
Cloudflare, CloudFront and DataDome hosts are IP-blocked.

So this script runs four probes per URL and prints one table plus a ladder block:

  A  the bot's REAL client — ``resolution_source._get_session()``, so the browser
     headers, the SSRF FilteringResolver and the fetcher's own HTTP timeout are
     exactly what a live run uses.
  B  the same GET through curl_cffi with ``impersonate="chrome"``, which presents a
     real Chrome TLS/JA3 + HTTP2 fingerprint. A ROW WHERE A IS 403 AND B IS 200 IS
     THE FINGERPRINT VERDICT; a row where both are 403 is the IP verdict.
  C  the Wayback Machine copy, which is the fallback route if B does not help.
  D  the real Tier-1 provider entry point, ``fetch_resolution_sources``, run on the
     URL alone with no keys in the environment: the direct fetch plus every FREE rung
     of the escalation ladder (the impersonation rung among them), reported as the
     status, the route and the rung outcomes production would record. A row reading
     ``status=success route=impersonate`` with ``rung=impersonate outcome=success``
     on an Akamai host, from the runner, is the live proof that the rung works from
     production egress; B says the fingerprint is the problem, D says the shipped
     code fixes it.

Everything here is free: no LLM call, no API key, no paid provider, and no write of
any kind. It is safe to run on a runner with no secrets in the environment. Column D
reads production code only, and the one PAID rung in that ladder (the Gemini
``url_context`` read) declines on its flag before it looks for its key, and both are
unset here.

Politeness: probes run strictly sequentially, with at least ``_HOST_SPACING_S``
between two requests to the same host, and every request carries a timeout. Column D
is the exception the pacer cannot see inside: the provider serialises same-host
requests on its own per-host gate but does not space them, and a host that refuses
every free rung costs it 1 to 3 extra requests (the impersonated retry, then the
archive lookup, which repeats column C's request).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import NamedTuple
from urllib.parse import urlparse

import aiohttp
from curl_cffi import requests as curl_requests

from metaculus_bot.constants import RESOLUTION_SOURCE_MAX_RESPONSE_BYTES
from metaculus_bot.research.http_fetch import read_body_capped
from metaculus_bot.research.resolution_fetch_result import FetchResult, RungAttempt
from metaculus_bot.research.resolution_source import _get_session, fetch_resolution_sources, is_public_http_url
from metaculus_bot.research.wayback import parse_snapshot_url, snapshot_age_days, wayback_snapshot_url

_HOST_SPACING_S = 1.0
# The Server header is the informative part of a row's note ("AkamaiGHost",
# "cloudflare", "DataDome"); Wikipedia answers with a pod hostname long enough to
# break the column, so it is capped where it is read rather than at render time —
# the Wayback column's note carries the snapshot age and must survive whole.
_SERVER_HEADER_MAX_CHARS = 20
# This diagnostic's own knob for columns B and C, ABOVE the Tier-1 HTTP timeout on purpose:
# the question here is whether a host answers at all, not whether it answers inside the
# provider wall. Column D runs production's own timeouts. Never copy this into production.
_IMPERSONATE_TIMEOUT_S = 25.0
_EGRESS_IP_URL = "https://api.ipify.org"


class ProbeUrl(NamedTuple):
    """A URL under test plus the anti-bot class it is here to represent."""

    url: str
    source_class: str


# One entry per anti-bot class we care about, including three controls whose
# expected answer is known in advance — a control that comes back wrong means the
# harness is broken, not that the host changed its mind.
PROBE_URLS: tuple[ProbeUrl, ...] = (
    # Akamai-fronted federal hosts: the reported failures.
    ProbeUrl("https://www.bls.gov/wsp/", "akamai-fed"),
    ProbeUrl("https://www.bls.gov/news.release/pdf/wkstp.pdf", "akamai-fed"),
    ProbeUrl("https://www.cdc.gov/cyclosporiasis/php/surveillance/index.html", "akamai-fed"),
    ProbeUrl("https://www.fsis.usda.gov/", "akamai-fed"),
    # Federal, different CDN: separates "Akamai scores us" from "federal hosts do".
    ProbeUrl("https://www.congress.gov/bill/119th-congress/house-bill/2913", "fed-other"),
    # Vendor SPA the fetcher has hit: no CDN challenge, content behind JS.
    ProbeUrl("https://tracxn.com/d/companies/deepseek/__1GrZ3pgoi2O-9tMSfF9ka6Sjybc1", "vendor-spa"),
    # DataDome control: expected 403 on every rung, impersonation included.
    ProbeUrl("https://www.sagaftra.org/sag-aftra-strikes-video-games-over-ai", "control-datadome"),
    # Cloudflare-challenge control: expected 403 on the plain rung.
    ProbeUrl("https://www.trueup.io/big-tech-hiring", "control-cloudflare"),
    # Positive controls: expected 200 on every rung.
    ProbeUrl("https://en.wikipedia.org/wiki/Nuri_(rocket)", "control-open"),
    ProbeUrl("https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026", "control-open"),
)


class ProbeOutcome(NamedTuple):
    """One probe's result. ``status`` is None when no HTTP response came back."""

    status: int | None
    n_bytes: int
    note: str

    def render(self) -> str:
        if self.status is None:
            return self.note or "no response"
        size = _render_bytes(self.n_bytes)
        return f"{self.status} {size} {self.note}".strip()


def _render_bytes(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes}B"
    if n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.1f}kB"
    return f"{n_bytes / (1024 * 1024):.1f}MB"


def _short_error(exc: BaseException) -> str:
    """Render an exception as one short table cell, never a traceback."""
    detail = str(exc).strip().splitlines()
    first = detail[0] if detail else ""
    rendered = f"{type(exc).__name__}: {first}" if first else type(exc).__name__
    return rendered[:60]


class HostPacer:
    """Enforce a minimum gap between two requests to the same host."""

    def __init__(self, spacing_s: float) -> None:
        self._spacing_s = spacing_s
        self._last_at: dict[str, float] = {}

    async def wait(self, url: str) -> None:
        host = urlparse(url).netloc
        loop = asyncio.get_running_loop()
        previous = self._last_at.get(host)
        if previous is not None:
            remaining = self._spacing_s - (loop.time() - previous)
            if remaining > 0:
                await asyncio.sleep(remaining)
        self._last_at[host] = loop.time()


async def probe_bot_client(session: aiohttp.ClientSession, url: str) -> ProbeOutcome:
    """Probe A: the bot's own aiohttp client, headers, resolver and timeout.

    The live fetcher walks redirects hop by hop under a per-host semaphore; here one
    GET with ``allow_redirects=True`` is equivalent for the question being asked,
    because a 403 is refused at the first hop.
    """
    if not await is_public_http_url(url):
        return ProbeOutcome(None, 0, "ssrf_blocked")
    try:
        async with session.get(url, allow_redirects=True) as resp:
            body = await read_body_capped(resp, max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES, label="probe")
            server = (resp.headers.get("Server") or "").strip()[:_SERVER_HEADER_MAX_CHARS] if resp.headers else ""
            n_bytes = len(body) if body is not None else 0
            note = server if body is not None else "over size cap"
            return ProbeOutcome(resp.status, n_bytes, note)
    # Narrow on purpose: every network failure this probe exists to record is an
    # aiohttp ClientError, a timeout or an OSError. Anything else is a bug in this
    # script and should crash with a stack trace rather than land in a table cell.
    except (aiohttp.ClientError, TimeoutError, OSError) as exc:
        return ProbeOutcome(None, 0, _short_error(exc))


def probe_impersonated(url: str) -> ProbeOutcome:
    """Probe B: the same GET with a real Chrome TLS/HTTP2 fingerprint.

    Deliberately the floating ``"chrome"`` alias rather than production's pinned profile, so
    this column keeps answering "does the CURRENT Chrome fingerprint get in" while column D
    answers "does the shipped rung get in"; the two diverging is itself a finding.
    """
    try:
        resp = curl_requests.get(url, impersonate="chrome", allow_redirects=True, timeout=_IMPERSONATE_TIMEOUT_S)
    except (curl_requests.RequestsError, OSError) as exc:
        return ProbeOutcome(None, 0, _short_error(exc))
    server = (resp.headers.get("Server") or "").strip()[:_SERVER_HEADER_MAX_CHARS]
    return ProbeOutcome(resp.status_code, len(resp.content), server)


def probe_wayback(url: str) -> ProbeOutcome:
    """Probe C: the Wayback Machine copy, reported with its snapshot age.

    The request URL comes from the rung's own ``wayback_snapshot_url`` rather than a template
    here, so what the probe measures cannot drift from what production asks the archive for; the
    stamp and age in the note are read back through the rung's own parser too (``_snapshot_note``).
    """
    archive_url = wayback_snapshot_url(url, now=datetime.now(UTC))
    try:
        resp = curl_requests.get(
            archive_url, impersonate="chrome", allow_redirects=True, timeout=_IMPERSONATE_TIMEOUT_S
        )
    except (curl_requests.RequestsError, OSError) as exc:
        return ProbeOutcome(None, 0, _short_error(exc))
    return ProbeOutcome(resp.status_code, len(resp.content), _snapshot_note(resp.url))


def _snapshot_note(final_url: str) -> str:
    """Render the capture stamp Wayback redirected to, plus its age in days, as the rung reads them.

    Both the parse and the age rule are production's — ``parse_snapshot_url`` and
    ``snapshot_age_days`` in ``metaculus_bot/research/wayback.py`` — rather than a regex kept
    here, so a stamp this probe reports as usable is one the rung would accept. The two ways a
    local copy drifted: an unanchored ``/web/(\\d{14})`` search could lift a stamp out of a nested
    or unrelated URL that the rung, anchored on the archive's own host, would refuse; and a plain
    ``now - stamp`` reported a capture dated in the FUTURE as the freshest possible copy, where the
    rung's clock-skew rule (``RESOLUTION_SOURCE_CLOCK_SKEW_TOLERANCE``) treats it as a broken
    clock or a misparse and refuses it. That case gets its own note so the operator can tell it
    from a fresh capture.
    """
    snapshot = parse_snapshot_url(final_url)
    if snapshot is None:
        return "no snapshot stamp"
    stamp = f"{snapshot.captured_at:%Y%m%d%H%M%S}"
    age_days = snapshot_age_days(snapshot, datetime.now(UTC))
    if age_days is None:
        return f"{stamp} (future-dated; the rung would refuse it)"
    return f"{stamp} ({int(age_days)}d old)"


class LadderOutcome(NamedTuple):
    """Probe D's result: what the production ladder recorded for the URL."""

    status: str
    route: str
    http_status: int | None
    n_chars: int
    rung_attempts: tuple[RungAttempt, ...]

    def render(self) -> str:
        head = f"status={self.status} route={self.route} http={self.http_status} chars={self.n_chars}"
        if not self.rung_attempts:
            return f"{head} rungs=none"
        return f"{head} {' '.join(_render_rung_attempt(attempt) for attempt in self.rung_attempts)}"

    @property
    def rescued_by_impersonation(self) -> bool:
        """The rung fired on this URL and the page it produced is what was published."""
        return self.status == "success" and self.route == "impersonate"


def _render_rung_attempt(attempt: RungAttempt) -> str:
    if attempt.skipped_reason:
        return f"rung={attempt.rung} skip={attempt.skipped_reason}"
    return f"rung={attempt.rung} outcome={attempt.outcome}"


async def probe_ladder(url: str) -> LadderOutcome:
    """Probe D: the real Tier-1 provider on this one URL, every free rung included.

    ``fetch_resolution_sources`` is the entry point ``resolution_source_provider`` calls for a
    question's cited URLs, so what this column reports is what a live run would record for the
    same page from the same egress: the direct fetch and, when that fails, the meta-refresh hop,
    the impersonated retry, the local PDF read, the browser render (which declines on a runner
    with no Chromium installed, as the ``renderer_unavailable`` skip), the derived feed and the
    archive. The paid ``url_context`` rung declines on its unset flag. ``query`` is empty because
    there is no question here; it only ranks which passages of a document a forecaster sees.
    ``fast_path=False`` so no rung declines for the thin-window mode a real question might be in.

    No exception is caught: a failed fetch comes back as a ``FetchResult`` with its own status,
    and anything that RAISES out of the provider is a bug this diagnostic should crash on.
    """
    results: list[FetchResult] = await fetch_resolution_sources([url], query="", fast_path=False)
    (result,) = results
    return LadderOutcome(
        status=result.status,
        route=result.route,
        http_status=result.http_status,
        n_chars=len(result.text),
        rung_attempts=tuple(result.rung_attempts),
    )


class ProbeRow(NamedTuple):
    probe_url: ProbeUrl
    bot: ProbeOutcome
    impersonated: ProbeOutcome
    wayback: ProbeOutcome
    ladder: LadderOutcome


async def read_egress_ip(session: aiohttp.ClientSession) -> str:
    """Report the egress IP this run presents to every host below."""
    try:
        async with session.get(_EGRESS_IP_URL, allow_redirects=True) as resp:
            body = await resp.read()
            return body.decode("utf-8", errors="replace").strip() or "(empty response)"
    except (aiohttp.ClientError, TimeoutError, OSError) as exc:
        return f"unavailable ({_short_error(exc)})"


async def run_probes(session: aiohttp.ClientSession, pacer: HostPacer) -> list[ProbeRow]:
    rows: list[ProbeRow] = []
    for probe_url in PROBE_URLS:
        await pacer.wait(probe_url.url)
        bot = await probe_bot_client(session, probe_url.url)
        await pacer.wait(probe_url.url)
        impersonated = await asyncio.to_thread(probe_impersonated, probe_url.url)
        archive_url = wayback_snapshot_url(probe_url.url, now=datetime.now(UTC))
        await pacer.wait(archive_url)
        wayback = await asyncio.to_thread(probe_wayback, probe_url.url)
        # The ladder's first request is to the cited host; its later rungs may hit the archive
        # too, which the pacer cannot space. Both are paced on entry at least.
        await pacer.wait(probe_url.url)
        await pacer.wait(archive_url)
        ladder = await probe_ladder(probe_url.url)
        rows.append(ProbeRow(probe_url, bot, impersonated, wayback, ladder))
    return rows


def print_url_list() -> None:
    print("URLs under test")
    for index, probe_url in enumerate(PROBE_URLS, start=1):
        print(f"  {index:>2}  {probe_url.source_class:<18}  {probe_url.url}")
    print()


def print_table(rows: list[ProbeRow]) -> None:
    print("Probe results")
    header = f"  {'#':>2}  {'class':<18}  {'A bot aiohttp':<34}  {'B chrome-impersonate':<34}  C wayback"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for index, row in enumerate(rows, start=1):
        print(
            f"  {index:>2}  {row.probe_url.source_class:<18}  "
            f"{row.bot.render():<34}  {row.impersonated.render():<34}  {row.wayback.render()}"
        )
    print()


def print_ladder(rows: list[ProbeRow]) -> None:
    """Column D on its own lines: the rung pairs are too wide for the table."""
    print("D  production ladder (fetch_resolution_sources, no keys)")
    for index, row in enumerate(rows, start=1):
        print(f"  {index:>2}  {row.probe_url.source_class:<18}  {row.ladder.render()}")
    print()


def _count(subset: list[ProbeRow]) -> str:
    """Render a bucket size with its noun: "1 URL" / "2 URLs"."""
    return "1 URL" if len(subset) == 1 else f"{len(subset)} URLs"


def _subject(subset: list[ProbeRow]) -> str:
    """Render a bucket size as a passive sentence subject: "1 URL was" / "2 URLs were"."""
    return f"{_count(subset)} was" if len(subset) == 1 else f"{_count(subset)} were"


def _hosts(subset: list[ProbeRow]) -> str:
    """Name the distinct hosts in a bucket, in table order, without repeats."""
    seen: list[str] = []
    for row in subset:
        host = urlparse(row.probe_url.url).netloc
        if host not in seen:
            seen.append(host)
    return ", ".join(seen) or "none"


def print_verdict(rows: list[ProbeRow]) -> None:
    """Print the one paragraph the operator actually reads."""
    helped = [r for r in rows if r.bot.status == 403 and r.impersonated.status == 200]
    both_refused = [r for r in rows if r.bot.status == 403 and r.impersonated.status == 403]
    both_ok = [r for r in rows if r.bot.status == 200 and r.impersonated.status == 200]
    archived = [r for r in rows if r.wayback.status == 200]
    rescued = [r for r in rows if r.ladder.rescued_by_impersonation]
    helped_not_rescued = [r for r in helped if not r.ladder.rescued_by_impersonation]

    print("Verdict")
    print(
        f"  Of {len(rows)} URLs, {_subject(helped)} refused with 403 by the bot's own client and served 200 under "
        f"Chrome impersonation ({_hosts(helped)}), which is the signature of TLS/HTTP fingerprint scoring and is "
        f"fixable client-side. {_subject(both_refused)} refused with 403 on both rungs ({_hosts(both_refused)}), "
        "which points at the egress IP's reputation rather than at anything about the request, so impersonation "
        f"would not recover those. {_count(both_ok)} served 200 on both rungs ({_hosts(both_ok)}), needing no fix "
        f"from this egress at all. The Wayback Machine returned 200 for {len(archived)} of {len(rows)} URLs, "
        "which bounds how much of the rest an archival route could recover; read the snapshot age before "
        "treating any of them as a live source."
    )
    print(
        f"  The production ladder published {_count(rescued)} through the impersonation rung "
        f"({_hosts(rescued)}). {_count(helped_not_rescued)} that column B recovered did NOT come back "
        f"route=impersonate from the ladder ({_hosts(helped_not_rescued)}); a nonzero count there is a defect in "
        "the shipped rung rather than in the host, so read that row's rung pairs and the run log for an "
        "ImpersonatePinNotHeld error or a failure_class=tls before merging."
    )


async def main() -> None:
    print(f"Fetch egress diagnostic — {datetime.now(UTC).isoformat(timespec='seconds')}")
    async with _get_session() as session:
        print(f"  egress IP: {await read_egress_ip(session)}")
        print()
        print_url_list()
        rows = await run_probes(session, HostPacer(_HOST_SPACING_S))
    print_table(rows)
    print_ladder(rows)
    print_verdict(rows)


if __name__ == "__main__":
    asyncio.run(main())
