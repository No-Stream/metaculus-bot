"""Measure, from wherever this runs, WHY the resolution-source fetcher gets 403s.

Background: the Tier-1 resolution-source fetcher
(``metaculus_bot/research/resolution_source.py``) is refused with HTTP 403 by
Akamai-fronted federal hosts — bls.gov, cdc.gov, fsis.usda.gov — but only when the
bot runs on a GitHub Actions runner. The identical client, on the operator's laptop
and on their EC2 box, gets 200 from the same URLs. Two explanations fit that split
and they call for completely different fixes: the runner's TLS/HTTP fingerprint is
being scored (fixable with Chrome impersonation) or the runner's egress IP range is
blocked outright (not fixable client-side; needs an archival route).

So this script runs three probes per URL and prints one table:

  A  the bot's REAL client — ``resolution_source._get_session()``, so the browser
     headers, the SSRF FilteringResolver and the fetcher's own HTTP timeout are
     exactly what a live run uses.
  B  the same GET through curl_cffi with ``impersonate="chrome"``, which presents a
     real Chrome TLS/JA3 + HTTP2 fingerprint. A ROW WHERE A IS 403 AND B IS 200 IS
     THE FINGERPRINT VERDICT; a row where both are 403 is the IP verdict.
  C  the Wayback Machine copy, which is the fallback route if B does not help.

Everything here is free: no LLM call, no API key, no paid provider, and no write of
any kind. It is safe to run on a runner with no secrets in the environment.

Politeness: probes run strictly sequentially, with at least ``_HOST_SPACING_S``
between two requests to the same host, and every request carries a timeout.
"""

from __future__ import annotations

import asyncio
import importlib.util
from datetime import UTC, datetime
from typing import NamedTuple
from urllib.parse import urlparse

import aiohttp

from metaculus_bot.constants import RESOLUTION_SOURCE_MAX_RESPONSE_BYTES
from metaculus_bot.research.http_fetch import read_body_capped
from metaculus_bot.research.resolution_source import _get_session, is_public_http_url
from metaculus_bot.research.wayback import parse_snapshot_url, snapshot_age_days, wayback_snapshot_url

_HOST_SPACING_S = 1.0
# The Server header is the informative part of a row's note ("AkamaiGHost",
# "cloudflare", "DataDome"); Wikipedia answers with a pod hostname long enough to
# break the column, so it is capped where it is read rather than at render time —
# the Wayback column's note carries the snapshot age and must survive whole.
_SERVER_HEADER_MAX_CHARS = 20
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


def impersonation_available() -> bool:
    """Whether the optional curl_cffi impersonation rung can run at all."""
    return importlib.util.find_spec("curl_cffi") is not None


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
    """Probe B: the same GET with a real Chrome TLS/HTTP2 fingerprint."""
    try:
        # Genuinely optional dependency: curl_cffi is declared only in pyproject's dev group
        # (for tests/conftest.py's egress guard), and this probe's workflow syncs `--no-dev`
        # and supplies it with `uv run --with curl_cffi`, so this rung has to degrade rather
        # than crash when it is absent.
        from curl_cffi import requests as curl_requests  # noqa: PLC0415
    except ImportError:
        return ProbeOutcome(None, 0, "curl_cffi absent")
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
    try:
        from curl_cffi import requests as curl_requests  # noqa: PLC0415  # optional, see probe_impersonated
    except ImportError:
        return ProbeOutcome(None, 0, "curl_cffi absent")
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


class ProbeRow(NamedTuple):
    probe_url: ProbeUrl
    bot: ProbeOutcome
    impersonated: ProbeOutcome
    wayback: ProbeOutcome


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
        rows.append(ProbeRow(probe_url, bot, impersonated, wayback))
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


async def main() -> None:
    print(f"Fetch egress diagnostic — {datetime.now(UTC).isoformat(timespec='seconds')}")
    if not impersonation_available():
        print("  WARNING: impersonation rung unavailable — curl_cffi is not importable.")
        print("  Re-run as: uv run --with curl_cffi python scripts/probes/fetch_diagnostic.py")
    async with _get_session() as session:
        print(f"  egress IP: {await read_egress_ip(session)}")
        print()
        print_url_list()
        rows = await run_probes(session, HostPacer(_HOST_SPACING_S))
    print_table(rows)
    print_verdict(rows)


if __name__ == "__main__":
    asyncio.run(main())
