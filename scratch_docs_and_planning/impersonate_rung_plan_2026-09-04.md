# TLS-impersonation rung for the resolution-source fetcher (plan, 2026-09-04)

Branch `impersonate-rung`, cut from `main` at 8d5a082, current tip b0dd940 ("Move curl-cffi to the
runtime dependencies for the impersonation rung"). Repo root is
`/Users/flatljan/personal/metaculus-bot`; every path below is relative to it.

Read this whole file before editing anything. Every claim in it was checked against the code on
2026-09-04, and the file:line references are from that reading. Where a fact was believed but not
verified, it says so.

Terms used throughout. **Tier 1** is the resolution-source fetcher's plain HTTP path
(`metaculus_bot/research/resolution_source.py`), which fetches the URL a Metaculus question names
as its resolution source. **The ladder** is `_escalate_unresolved`, the sequence of escalation
**rungs** that run when the direct fetch cannot read a page. **Gap-fill v2** is the bounded agentic
research loop in `metaculus_bot/research/agentic/`, which has its own separate fetch ladder.
**Impersonation** means presenting a real browser's TLS ClientHello and HTTP/2 settings
fingerprint, which `curl_cffi` does by driving a patched libcurl (libcurl-impersonate).

---

## 1. Context and decision

### The measurement that motivates the rung

The resolution-source fetcher gets HTTP 403 from several Akamai-fronted federal hosts, but only
when the bot runs on a GitHub Actions runner. The identical client on the operator's laptop and
EC2 box gets 200 from the same URLs. Two explanations fit that split and they need completely
different fixes: the runner's TLS and HTTP/2 fingerprint is being scored, which is fixable in our
client, or the runner's egress IP range is blocked outright, which is not.

The free diagnostic (`scripts/probes/fetch_diagnostic.py`, dispatched through
`.github/workflows/fetch_diagnostic.yaml`) settled it on 2026-09-04 from the runner, egress IP
4.246.135.160. Column A is the bot's own aiohttp client with its own headers, resolver and
timeout; column B is the same GET through `curl_cffi` with `impersonate="chrome"`.

| Host | Class | A, bot aiohttp | B, chrome impersonation | Verdict |
|---|---|---|---|---|
| www.bls.gov/wsp/ | Akamai federal | 403 AkamaiGHost | 200 | fingerprint |
| www.bls.gov/news.release/pdf/wkstp.pdf | Akamai federal | 403 AkamaiGHost | 200 | fingerprint |
| www.cdc.gov/cyclosporiasis/php/surveillance/index.html | Akamai federal | 403 AkamaiGHost | 200 | fingerprint |
| www.fsis.usda.gov | Akamai federal | 403 AkamaiGHost | 200 | fingerprint |
| www.congress.gov (Cloudflare) | federal, other CDN | 403 | 403 | egress IP |
| tracxn.com (CloudFront) | vendor SPA | 403 | 403 | egress IP |
| www.sagaftra.org (DataDome) | control | 403 | 403 | egress IP |
| www.trueup.io (Cloudflare) | control | 403 | 403 | egress IP |

All four Akamai-fronted federal hosts are recoverable client-side. The four both-403 hosts are
out of scope for this rung: nothing about the request changes their answer, and they stay the
Wayback and paid-reader rungs' population.

Note that one of the four recoverable URLs is a PDF (`wkstp.pdf`). The rung therefore has to
handle a document body, not only HTML, or it drops a quarter of the population it exists for.

### The lead's fixed decisions, restated

These are settled. Do not re-derive or re-litigate them.

1. The rung fires on a direct-fetch 403, that is `FetchResult.status == "blocked"` with
   `http_status == 403`. Section 3 pins the exact predicate and says why 401, 406, 429 and a
   200-with-a-challenge-page are excluded.
2. It sits between the direct fetch and the Wayback rung, so a live page beats a stale capture.
3. It is free. No API key, no model call, no spend.
4. It is self-bounded inside the 45 s Tier-1 provider wall through the existing
   `FetchContext.claim_rung_budget` mechanism.
5. Its route token is `impersonate`, already reserved in
   `metaculus_bot/research/resolution_fetch_result.py:234` (the `FetchRoute` Literal) and `:267`
   (its `ROUTE_CAVEATS` sentence), in `scripts/telemetry/markers.py:816`, and described in
   `docs/research.md:1182`.
6. The rung carries the SSRF invariants itself. `curl_cffi` drives libcurl, which never touches
   aiohttp's connect-time `FilteringResolver`, so the rung pre-resolves the host through the
   repo's own vetting predicate, pins the connection to the vetted address with libcurl's
   `CURLOPT_RESOLVE`, and refuses when it cannot. No automatic redirects; every hop is re-guarded
   and re-pinned under the existing `MAX_REDIRECTS` cap. Same body cap and content-type gating as
   the direct fetch. Per-host politeness.
7. Gap-fill v2 gets the same rung through the same transport if it is cheap. Section 4 prices it
   and recommends doing it.

### Two standing rules this change is directly governed by

From `AGENTS.md`, "Standing rules":

- **Guards and safety.** Any new outbound HTTP path goes through the SSRF invariants that
  `research/http_fetch.py` owns: the `is_public_http_url` preflight, the connect-time
  `FilteringResolver` as the real DNS-rebinding boundary, a bounded manual redirect loop that
  re-guards every hop, and the per-host politeness semaphores. A guard fails SHUT: the worked
  example in that file is the unit-mismatch guard, where wrapping it in try/except made a guard
  crash byte-identical to a passing check. Do not hand-roll a fetch, a render or a reader; this
  change adds a third shared transport beside `rendered_fetch.py` and `url_context_reader.py`,
  which is the sanctioned way to add one.
- **Telemetry and logs.** Every marker and status string is a data contract. `scripts/telemetry/
  markers.py` is the registry and the archive keys off exact spellings, so ADD a token; never
  rename or re-spell one, and never change a field's meaning in place.

---

## 2. Design: the transport module

New file `metaculus_bot/research/impersonated_fetch.py`. The name is already committed to: the
`[tool.deptry]` comment at `pyproject.toml:338-345` names `research/impersonated_fetch.py` as the
module that justifies curl-cffi being a runtime dependency.

The transport owns the sessions, the pin, the redirect loop, the body cap and the politeness gate.
It knows nothing about `FetchResult`, `FetchStatus`, rung attempts or the ladder. That split is
what lets gap-fill v2 call it (section 4) and what keeps `resolution_source.py`, already carrying
a `SMELL-EXEMPT-monolithic-file-loc` header at 3,517 lines, from growing more than a thin rung.

### 2.1 The one public coroutine and its result

```python
async def fetch_impersonated(
    url: str,
    *,
    host_sems: dict[str, asyncio.Semaphore],
    deadline_monotonic_s: float,
    per_hop_timeout_s: float,
    max_bytes: int,
) -> ImpersonatedResponse
```

```python
@dataclass(frozen=True, slots=True)
class ImpersonatedResponse:
    """One completed impersonated HTTP response, body already read and capped."""

    status: int          # the HTTP status, including a non-200 the caller reads as data
    url: str             # the URL this response was read from, i.e. the last hop dialed
    content_type: str    # the Content-Type header lower-cased, "" when absent
    server: str | None   # the raw Server header, unmodified; the caller tokenises it
    body: bytes          # decompressed, at most max_bytes
    elapsed_s: float     # wall clock for the whole call, including redirect hops
    primary_ip: str      # the address libcurl actually connected to, for the pin assertion
```

`host_sems` is the caller's own netloc-to-`Semaphore(1)` map, passed in rather than derived,
exactly as `rendered_fetch.render_page` takes `host_gate` for the same reason: Tier 1's map is
`http_fetch.host_semaphores()`, shared process-wide across concurrent questions, and gap-fill v2's
is its own module global `agentic/tools.py:_FETCH_HOST_SEMAPHORES`. The transport takes the MAP
rather than one semaphore because it walks redirect hops that may land on other hosts, and each
hop must contend on ITS host's gate. Use `http_fetch.semaphore_for_host(url, host_sems)`.

`deadline_monotonic_s` is the instant by which the whole call must be done, so the caller's wall
budget bounds it. `per_hop_timeout_s` is the per-request ceiling, which Tier 1 passes as
`RESOLUTION_SOURCE_HTTP_TIMEOUT`. `max_bytes` is the body cap.

A non-200 response is returned, not raised. The caller reads a still-403 as the fact that the
fingerprint was not the problem. Verified: `curl_cffi` does not raise on 4xx unless the session
was built with `raise_for_status=True`
(`.venv/lib/python3.12/site-packages/curl_cffi/requests/session.py:237`).

### 2.2 The typed declines

Everything that produces no body raises. One base class so the caller can catch the whole family
in one clause, and one subclass per failure shape so the log line and the tests can be exact.
This mirrors `rendered_fetch`'s `RenderTimeout` / `RenderBudgetExpired` / `RenderDomOverCeiling` /
`RenderOffHost` split.

| Class | Meaning | Caller's response |
|---|---|---|
| `ImpersonateDeclined` | Base. Never raised directly. | catch-all for "fired and produced nothing" |
| `ImpersonateUnpinnable` | The initial URL's host is not pinnable or will not resolve to a vetted public address. Raised BEFORE anything is dialed. | its own skip token, `impersonate_unpinnable` |
| `ImpersonatePinNotHeld` | The connection landed on an address other than the pinned one. Logged at ERROR. | decline; a nonzero count is a build defect, not host behaviour |
| `ImpersonateHopRefused` | A redirect target failed `_hop_refusal`. Carries the `HopRefusal` token. | decline, with a warning naming both netlocs |
| `ImpersonateRedirectLimit` | More than `MAX_REDIRECTS` hops. | decline |
| `ImpersonateBodyTooLarge` | The body exceeded `max_bytes` and the transfer was cut. | decline |
| `ImpersonateTransportError` | A libcurl failure. Carries `failure_class: str \| None` and `exc: str`. | decline |

Two traps here, both verified:

- `curl_cffi.requests.exceptions.RequestException` subclasses `curl_cffi.CurlError` **and**
  `OSError` (`.venv/.../curl_cffi/requests/exceptions.py:12-24`). Catch `RequestException`
  specifically inside the transport. Never put an `except OSError` on any code path that wraps
  this transport, and never wrap the pin-and-vet step in a broad except: per `AGENTS.md`, a guard
  that crashes must not be byte-identical to a guard that passed.
- `Response.ok` is a plain instance attribute set once in `_parse_response`, not a property
  derived from `status_code` (`.venv/.../curl_cffi/requests/models.py:99-101`). Verified on this
  venv: `r = Response(); r.status_code = 403` leaves `r.ok is True`. Branch on `status_code`
  only, everywhere, production and tests alike.

### 2.3 The SSRF procedure, step by step

The loop runs at most `MAX_REDIRECTS + 1` times, exactly like
`resolution_source._fetch_direct` (`resolution_source.py:2845`). `MAX_REDIRECTS = 5` and
`REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})` come from
`metaculus_bot/research/http_fetch.py:102-103`, shared so the two SSRF-guarded fetchers cannot
drift. Import them; do not restate either.

Per hop:

1. **Eligibility and vetted address, in one call.** Use the pin helper `rendered_fetch` already
   owns rather than writing a second one:

   ```python
   pinned = await rendered_fetch.resolve_pinned_host(hop_url)   # (host, vetted_ip) or None
   ```

   That function (currently `rendered_fetch._resolve_pinned_host`, `rendered_fetch.py:407-454`)
   already does the whole job: it runs `_pinnable_url_host` for scheme, userinfo, an empty host, a
   non-ASCII host and a trailing-dot host; it handles the IP-literal branch by vetting through
   `resolution_source._ip_is_disallowed` and pinning to itself; and it otherwise calls
   `resolution_source.resolve_vetted_public_ip(host)`, which resolves off the event loop through
   `asyncio.to_thread(socket.getaddrinfo, ...)` and rejects the WHOLE hostname if ANY resolved
   address is disallowed. Its docstring carries the reasoning that transfers verbatim to libcurl:
   a pin pattern built from a unicode or trailing-dot host matches nothing, and reproducing
   Chromium's UTS#46 canonicalisation with the stdlib IDNA2003 codec would emit a pattern that
   looks right and is inert, so it fails closed instead.

   **Rename it as part of this change.** Promote `_resolve_pinned_host` to `resolve_pinned_host`
   and `_pinnable_url_host` to `pinnable_url_host` in `rendered_fetch.py`. Two transports pinning
   DNS makes this a real public seam of the transport layer rather than one module's private
   helper, and reaching across modules for an underscore name is a smell forge will flag. Call
   sites to update, all of them: the two definitions and their internal callers in
   `rendered_fetch.py` (`_resolve_pinned_host` reads `_pinnable_url_host`, and `render_page` reads
   `_resolve_pinned_host`), `tests/playwright_fakes.py:375` and `:411`, and the ten tests in
   `tests/test_agentic_tools.py:1745-1810`. `monkeypatch.setattr` raises on a missing attribute,
   so a missed patch site fails loudly rather than silently unpinning.

   `None` from that call means: on hop 0, raise `ImpersonateUnpinnable`; on a later hop, it cannot
   happen, because a later hop already passed `_hop_refusal` in step 2, whose
   `is_public_http_url` covers the same rejections. Raise `ImpersonateUnpinnable` there too and
   let the test pin that the two agree.

2. **Re-guard a derived hop.** For every hop after the first, the target came out of a `Location`
   header, so it owes the two checks every derived URL owes. Call
   `resolution_source._hop_refusal(next_url)` (`resolution_source.py:1055-1073`): the
   `is_public_http_url` preflight first, then the Metaculus self-reference refusal, in that order,
   because a URL that is both non-public and a self-reference has always recorded as
   `ssrf_blocked` and that ordering is a telemetry contract. A refusal raises
   `ImpersonateHopRefused`. Do not build a terminal `ssrf_blocked` result here: a rung DECLINES on
   a refusal and leaves the direct fetch's own outcome standing, which is what `_rendered_rung`
   (`resolution_source.py:2101-2108`) and `_wayback_snapshot_result`
   (`resolution_source.py:2296-2304`) both do.

   Both `_hop_refusal` and `resolve_vetted_public_ip` live on `resolution_source`, which imports
   this new module at module scope, so the transport needs a function-scoped import:

   ```python
   from metaculus_bot.research import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # real cycle + the guard's single patch surface
       resolution_source,
   )
   ```

   Both of `AGENTS.md`'s justifications hold and both must be named in the comment, exactly as
   `rendered_fetch._resolve_pinned_host` does at `rendered_fetch.py:435-437`: it is a genuine
   circular import, and it is the late binding the suites rely on, because
   `resolution_source.is_public_http_url` / `resolve_vetted_public_ip` / `_ip_is_disallowed` are
   monkeypatched on THAT module by both fetch paths' tests and the guard has exactly one patch
   surface precisely because every reader resolves it there.

3. **Acquire the host gate for THIS hop's host**, release it before the next hop acquires its own.
   Strict per-hop acquire and release, never nested: `asyncio.Semaphore` is not reentrant and an
   A to B to A redirect chain would self-deadlock otherwise. This is the invariant
   `_fetch_direct`'s docstring spells out at `resolution_source.py:2803-2812`.

4. **Size this hop's timeout once the gate is held**, not before, so a hop that queued behind a
   slow host does not then help itself to a fresh ceiling. Same arithmetic as
   `_fetch_one_hop` (`resolution_source.py:1882`):

   ```python
   remaining = deadline_monotonic_s - time.monotonic()
   if remaining <= 0.0:
       raise ImpersonateTransportError(failure_class="timeout", exc="TimeoutError")
   hop_timeout_s = min(per_hop_timeout_s, max(remaining, RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S))
   ```

5. **Build one short-lived session per pinned hop.** This is forced by the library, not a choice.
   Verified on curl-cffi 0.15.0: `curl_options`, the only route to `CURLOPT_RESOLVE`, is a
   `BaseSession` constructor parameter (`.venv/.../requests/session.py:229`, applied last inside
   `set_curl_options` at `.venv/.../requests/utils.py:750-752` so user options can override
   library defaults) and it is reachable on `AsyncSession` through its `**kwargs`. It is NOT a
   request parameter: `inspect.signature(AsyncSession.request)` has no `curl_options`, confirmed
   in this venv. So a pin cannot be set per request on a shared session, and mutating a live
   session's `curl_options` between concurrent requests would be a race. One session per hop is
   also cheap: 20 sequential create, request, close cycles measured 0.020 s total on loopback.

   ```python
   session = AsyncSession(
       impersonate=IMPERSONATE_BROWSER_TARGET,
       allow_redirects=False,
       verify=certifi.where(),
       timeout=hop_timeout_s,
       curl_options={
           CurlOpt.RESOLVE: [f"{host}:{port}:{vetted_ip}"],
           CurlOpt.PROXY: "",
           CurlOpt.NOPROXY: "*",
       },
   )
   ```

   Every safety-relevant option is on the CONSTRUCTOR rather than per request, so no request can
   forget one. Use `async with AsyncSession(...) as session:` so `close()` always runs: that is
   what cancels the 0.1 s `_force_timeout` polling task `AsyncCurl.__init__` spawns for the
   session's lifetime, removes the loop's readers and writers, and closes pooled handles
   (`.venv/.../requests/session.py:971-988`). A session also binds permanently to the first
   running loop that drives it, so never cache one at module scope: it would break across
   pytest-asyncio's per-test loops and leak that task.

   Five details on those options:

   - **`CurlOpt.RESOLVE` takes a LIST, and a bare string is a silent, type-checked bug.**
     `Curl.setopt` special-cases the option and iterates the value, appending each element to a
     libcurl slist (`.venv/.../curl_cffi/curl.py:396-401`, read and confirmed). A bare
     `"host:443:1.2.3.4"` iterates CHARACTER BY CHARACTER, so the request dies at perform time
     with `curl: (49) Couldn't parse CURLOPT_RESOLVE entry 'p'`. Worse, curl-cffi's own type hint
     is `dict[CurlOpt, str]`, so basedpyright will bless the string form. A malformed pin fails
     OPEN in the general case: libcurl falls back to its own resolver and the request still
     succeeds. Only a test asserting the exact operand list will catch a regression here.
   - **Port.** Pin exactly the port this hop will dial: `urlparse(hop_url).port` when present,
     else 443 for https and 80 for http. libcurl matches host and port exactly, so a pin for the
     wrong port is inert. One entry, not two: a redirect that changes scheme on the same host is a
     new hop with a new session and a new pin.
   - **The proxy options are mandatory, and they are not belt and braces.** libcurl reads
     `HTTP_PROXY` / `http_proxy` from the environment itself, and curl-cffi 0.15.0's documented
     `trust_env` parameter is DEAD CODE: `grep -rn trust_env` over the installed package finds
     only its annotation, its default and one assignment (`session.py:75`, `:218`, `:246`) plus two
     docstrings, and nothing reads it. Verified on loopback: with `http_proxy` set, a request
     built with `trust_env=False` still went through the proxy, which received the absolute-form
     request and thus the hostname, so `CURLOPT_RESOLVE` never applied at all and the pin was
     bypassed entirely. `CurlOpt.PROXY: ""` restored the direct pinned connection with zero proxy
     hits, as did `CurlOpt.NOPROXY: "*"`. GitHub runners do not normally set a proxy, but an SSRF
     invariant must not depend on that. Set both.
   - **`verify=certifi.where()`** mirrors `http_fetch.build_session`'s TLS trust pinning
     (`http_fetch.py:179`), which exists because trade.gov failed the handshake against the
     machine's default store and succeeded against certifi's, so which sources are reachable was a
     property of the machine. curl-cffi routes a `str` `verify` to `CurlOpt.CAINFO`
     (`.venv/.../requests/utils.py:631-637`) and otherwise reads `REQUESTS_CA_BUNDLE` /
     `CURL_CA_BUNDLE` out of the environment, so passing it explicitly is also the
     strictly-safer direction. **Unverified:** the wheel bundles its own libcurl 8.15.0 with
     BoringSSL (`curl_cffi.__curl_version__`, read on this laptop's arm64 wheel), and I did not
     test CAINFO against that build offline. If the live QA in section 9 shows all four
     impersonated rows failing with a TLS failure class, dropping the `verify` argument is the
     first thing to try, and that would be a one-line change plus a test update.
   - **Do NOT pass `headers=BROWSER_HEADERS`.** The impersonation profile supplies its own
     complete Chrome header set, including `Accept`, `Accept-Language`, `Priority`, the
     `Sec-Fetch-*` family and the `sec-ch-ua*` client hints. Overriding those with
     `http_fetch.BROWSER_HEADERS`' Safari-like set would present a Chrome TLS fingerprint under
     Safari headers, which is precisely the incoherence an edge scores. This is a deliberate
     divergence from the direct path, and its consequence is that `Accept-Encoding` becomes the
     profile's `gzip, deflate, br, zstd` rather than the pinned `gzip, deflate` at
     `http_fetch.py:136`. That is safe here: libcurl's bundled zlib, brotli and zstd decode in
     process, so no Python decoder dependency is involved and none of the aiohttp path's
     `Accept-Encoding` measurement is disturbed. Say all of this in the module docstring.

6. **Issue exactly one streamed GET, inside an asyncio wall bound.**

   ```python
   async with asyncio.timeout(hop_timeout_s):
       async with session.stream("GET", hop_url) as response:
           ...
   ```

   The outer `asyncio.timeout` is not optional. Verified in curl-cffi's own source
   (`.venv/.../requests/utils.py:548-561`): with `stream=True`, `set_curl_options` deliberately
   does NOT set `TIMEOUT_MS`. It sets `CONNECTTIMEOUT_MS` plus `LOW_SPEED_LIMIT = 1` byte per
   second over `ceil(timeout)` seconds, so a slow-drip server holds a streamed fetch far past the
   `timeout=` value. Measured on loopback: a streamed read with `timeout=1.0` ran to completion,
   pulling 192 KiB in 4.5 s with no exception, because roughly 40 KiB per second never dropped
   below the 1 byte per second floor. Keep `timeout=hop_timeout_s` on the session anyway, because
   it is still the connect bound, and let the `asyncio.timeout` own the wall. An
   `asyncio.TimeoutError` out of that block becomes
   `ImpersonateTransportError(failure_class="timeout", exc="TimeoutError")`.

   `AsyncSession.stream` is `@asynccontextmanager`-decorated and its `finally` calls
   `await rsp.aclose()` (`.venv/.../requests/session.py:999-1011`, read).

7. **Assert the pin held before reading any body.** Read `response.primary_ip`, which is
   `CURLINFO_PRIMARY_IP`, and compare it to `vetted_ip`. On a mismatch, log at ERROR and raise
   `ImpersonatePinNotHeld`. This is the post-hoc detection for an inert pin (the bare-string trap)
   and for proxy interposition, which is how the pin bypass above was diagnosed in the first
   place.

   **Unverified:** whether `primary_ip` is already populated at the moment the streamed response
   object becomes available, or only after the transfer. Implement it as: check immediately, and
   when the value is empty at that point, check again after the body read and refuse then.
   Refusing after a read still closes the SSRF channel, because nothing from a refused response
   is ever returned or published. Note that a host with several A records is a benign source of a
   mismatch when the pin is inert, because `resolve_vetted_public_ip` returns the FIRST vetted
   address while libcurl's own resolver may pick another; every address was vetted, so refusing
   is safe if slightly lossy, and a nonzero count in the live QA means the pin is not working.

8. **Read the body with a cap, and cut the transfer when it is exceeded.**

   ```python
   chunks: list[bytes] = []
   total = 0
   async for chunk in response.aiter_content():
       total += len(chunk)
       if total > max_bytes:
           response.quit_now.set()
           break
       chunks.append(chunk)
   ```

   `quit_now.set()` must happen BEFORE leaving the `async with`, because `aclose()` on its own
   only awaits the download task and drains the rest of the body; the flag is what makes
   curl-cffi's write callback return `CURL_WRITEFUNC_ERROR` and abort the transfer
   (`.venv/.../requests/utils.py:718-732`). Measured on loopback: against a 192 KiB drip served
   16 KiB every 0.4 s, the abort returned 0.83 s in with 32 KiB read and the server had written
   only 4 of its 12 chunks. Then raise `ImpersonateBodyTooLarge`.

   Parity with the direct path is exact on the thing that matters. `http_fetch.read_body_capped`
   (`http_fetch.py:300-321`) cannot be reused because it iterates `resp.content.iter_chunked`,
   which is aiohttp-shaped, but libcurl decompresses BEFORE the write callback, so
   `aiter_content()` yields DECOMPRESSED bytes exactly as aiohttp's `iter_chunked` does after its
   `DeflateBuffer`. Counting decompressed bytes against the same cap reproduces the direct
   fetch's gzip-bomb protection. Verified on loopback: a gzip-encoded body arrived as plaintext in
   `content` while the response header still read `content-encoding: gzip`.

   Do not use the non-stream `content_callback` alternative. It keeps libcurl's hard total
   timeout, but its abort raises a bare `RequestException` with code 23 (`CURLE_WRITE_ERROR`) that
   is indistinguishable from a genuine write failure, `response.content` stays empty so the caller
   has to accumulate anyway, and the last chunk overshoots the cap (measured 49,009 bytes
   accumulated against a 32 KiB trip point).

9. **Redirect or return.** If `status_code in REDIRECT_STATUSES`, resolve the next hop and loop.
   Read `response.redirect_url`, which is `CURLINFO_REDIRECT_URL` and is the ABSOLUTISED target
   resolved against the request URL, populated even with `allow_redirects=False`: verified on
   loopback that a relative `/deep/other?a=1` came back as
   `http://pinned.invalid:<port>/deep/other?a=1` and a scheme-relative `//host/ok` as
   `http://host/ok`, while `headers.get("location")` stayed raw. Fall back to
   `urljoin(hop_url, headers["location"])` when `redirect_url` is empty, and treat a redirect with
   neither as `ImpersonateTransportError(failure_class=None, exc="MissingLocation")`, which
   mirrors the direct path's malformed-redirect branch at `resolution_source.py:1122-1132`.
   Exhausting `MAX_REDIRECTS` raises `ImpersonateRedirectLimit`.

   Otherwise build and return the `ImpersonatedResponse`. Do NOT use curl-cffi 0.15.0's
   `allow_redirects="safe"` / `CurlFollow.SAFE`, advertised as SSRF protection against internal
   and private addresses: I did not verify its semantics, and it would follow hops invisibly to
   our own `_hop_refusal`, our per-hop pin and our telemetry. Say so in a comment so a future
   reader does not "simplify" the manual loop away.

10. **Map libcurl failures to a failure class.** Catch
    `curl_cffi.requests.exceptions.RequestException` and translate by exception class, mirroring
    `resolution_source._network_failure_class` (`resolution_source.py:1138-1171`) so the two
    fetchers speak the same small vocabulary. Verified mapping from
    `.venv/.../requests/exceptions.py:159-200`: `Timeout` (code 28) becomes `timeout`; `SSLError`
    and `CertificateVerifyError` (35, 60 and the other SSL codes) become `tls`; `DNSError`
    (6) becomes `dns`; `ConnectionError` (7, 52, 55, 56) becomes `connection`;
    `IncompleteRead` (18) becomes `decode`; anything else becomes `connection`. Carry the
    exception's class name as `exc`. Do not invent new `failure_class` tokens: the vocabulary is a
    marker contract (`http_403`, `http_4xx`, `http_5xx`, `tls`, `dns`, `timeout`, `connection`,
    `decode`, `malformed_response`) and the direct path's own redirect-cap result carries no
    failure class at all, so neither should ours.

### 2.4 The memo

A host that answered our impersonated client with a block status is not going to answer the next
cited URL on that host differently in the same run. Keep a loop-scoped, process-wide memo in this
module, keyed by NETLOC, with the same three-function shape `rendered_fetch` uses for its render
memos:

```python
def impersonation_refused(url: str) -> bool
def note_impersonation_refused(url: str) -> None
def reset_impersonation_memo() -> None   # for tests
```

Three decisions inside that:

- **Keyed by host, not URL,** because what was learned is the edge's policy toward our fingerprint
  from this address, which is a property of the host.
- **Written only for a block-shaped answer**, that is an impersonated `status_code` in
  `{403, 406, 429}`. A 404 says the path is gone, and a 200 whose body classified as chrome or a
  JavaScript wall means the fingerprint DID get us in, so neither should switch the host off for
  the rest of the run.
- **Not scoped per caller.** `rendered_fetch` needs its `MemoScope` because "rendered to nothing"
  means something weaker in gap-fill v2 than in Tier 1. Here the fact is identical for both
  callers, so one unscoped memo is correct and it saves v2 a request on a host Tier 1 already
  probed.

The memo needs a reset in the test fixtures; see section 7.

### 2.5 Concurrency summary

One `AsyncSession` per pinned hop, created and closed inside that hop's host gate, forced by
`curl_options` being session-only. No shared connection pool and no module-level session. Per-host
politeness through the caller's map, acquired and released strictly per hop. The transport is
genuinely asyncio-native rather than thread-pooled: `AsyncCurl` drives libcurl's multi interface
through `loop.add_reader` / `add_writer` and `loop.call_later`
(`.venv/.../curl_cffi/aio.py:118-192`), so nothing here needs `to_thread` except the DNS
resolution that `resolve_vetted_public_ip` already does off the loop.

---

## 3. The Tier-1 rung

All of this lands in `metaculus_bot/research/resolution_source.py`, modelled on `_wayback_rung`
(`resolution_source.py:2360-2396`), which is the shortest rung in the file and the clearest
template.

### 3.1 The trigger, precisely

```python
def _impersonate_rung_applies(direct: FetchResult) -> bool:
    return direct.status == "blocked" and direct.http_status == 403
```

Both halves are load-bearing. `blocked` has exactly four producers and the status alone is not
specific enough:

- `_NON_OK_FETCH_STATUS` (`resolution_fetch_result.py:299-306`) maps 403, 406 **and** 429 to
  `blocked`.
- `_vetted_hop_target` (`resolution_source.py:1096-1116`) returns `status="blocked"` for a
  Metaculus self-reference hop, carrying the REDIRECT's 301 or 302 as `http_status`. The
  `http_status == 403` test excludes that case exactly, which matters: handing a URL this module
  refused to a second transport is the bypass the guard exists to prevent.

**Recommended and settled by this plan:** 403 only, with the other candidate shapes excluded for
these reasons, each of which belongs in the predicate's docstring.

- **429 is out.** It is a throttle, not a fingerprint verdict. Retrying immediately with a
  different fingerprint against a host that just asked us to slow down is the one shape where the
  retry could make our position worse, and the diagnostic measured impersonation helping only on
  403s.
- **406 is out.** It is a content-negotiation refusal, and impersonation changes the `Accept`
  headers as a side effect, so a 406 rung would be an untested guess. No 406 appeared in the
  diagnostic.
- **401 is out, and it is not even a `blocked` shape.** 401 is absent from
  `_NON_OK_FETCH_STATUS`, so it falls through to `error`. It is an authentication requirement no
  fingerprint changes.
- **A 200 challenge page is NOT a representable trigger today.** Tier 1 has no
  throttle-interstitial detection at all: `matched_throttle_phrase` and `FETCH_THROTTLE_PHRASES`
  exist only in gap-fill v2's `agentic/fetch_outcomes.py:46-92`, and `resolution_source.py` never
  calls either. A 200 carrying a challenge or throttle interstitial goes through
  `_classify_html_body` and lands on `success`, `js_wall`, or
  `no_resolving_content` with reason `thin_page`; an interstitial routinely clears the chrome
  floor. Making it a trigger means porting the phrase check into Tier 1's classifier, which is its
  own change with its own false-positive risk (a real page discussing rate limits) and its own
  measurement. Out of scope. Add a FUTURE.md entry.
- **`error` with `failure_class="tls"` is a plausible future widening** (a TLS handshake reset is
  arguably a fingerprint verdict too) but nothing measured it. Out of scope; same FUTURE.md entry.

### 3.2 Position in the ladder

In `_escalate_unresolved` (`resolution_source.py:2700-2761`), insert the rung immediately after
the `if direct.status == "success": return direct` line and BEFORE the
`if _rendered_rung_applies(direct):` block:

```python
impersonated = await _run_rung(
    ctx, direct.status, _impersonate_rung(url, direct, host_sems=host_sems, ctx=ctx)
)
if impersonated is not None:
    return impersonated
```

Why there rather than after the browser block. Functionally the position among the earlier rungs
does not matter, because the trigger sets are disjoint: `_rendered_rung_applies`
(`resolution_source.py:1912-1931`) fires only on `js_wall` and the `thin_page` shape of
`no_resolving_content`, never on `blocked`, for the stated reason that the edge refused our
address before any HTML existed and Chromium dials from the same address. So a `blocked`-triggered
rung never contends for `ctx.shared.browser_escalation_gate`. Putting it first is the reading
choice: it matches the `FetchRoute` Literal's own ladder order, where `impersonate` sits at
position 3 right after `meta_refresh`, and a reader meets the cheap free retry before the
expensive ones. It is before `_wayback_rung` as the lead fixed, so a live page beats a stale
capture, and before `_url_context_rung`, so a successful impersonation saves the paid read on that
URL entirely.

**The `_run_rung` bracket is mandatory.** `_run_rung` (`resolution_source.py:2681-2699`) reads
`len(ctx.rungs)` before awaiting, awaits, then closes every attempt opened since with the rung's
own wall and outcome. A rung awaited without it still returns its result, but its attempt falls
through to `_stamped_with_route`'s last-resort close, which stamps the LADDER's final status and
the WHOLE-ladder wall onto that attempt, which are the two figures the per-rung close exists to
keep apart. `fallback` is `direct.status` at every existing call site; use that.

### 3.3 The rung body

```python
async def _impersonate_rung(
    url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: FetchContext
) -> FetchResult | None:
```

In this order:

1. `if not _impersonate_rung_applies(direct): return None`.
2. **Which URL to dial.** `retry_url = direct.url`. `_resolution_status_outcome`
   (`resolution_source.py:1173-1189`) sets `url=current_url`, so a 403 result's `direct.url` is
   the hop that ANSWERED 403, which is not necessarily the cited URL. Dial that hop, because it is
   the URL the host actually refused. When `retry_url != url`, re-vet it through `_hop_refusal`
   and DECLINE with a warning on a refusal. Keep the rung ATTEMPT keyed on the cited `url`, which
   is what the escalation marker names. All three behaviours mirror `_rendered_rung`
   (`resolution_source.py:2099-2108`).
3. **Memo.** `if impersonation_refused(retry_url): ctx.skip_rung("impersonate", direct.status, url,
   "impersonate_host_refused"); return None`.
4. **No fast-path skip.** Do not call `_skip_for_fast_path`. That token exists to separate "the
   question's close left no room" from "this rung ran out of the provider's own clock", and its
   docstring (`resolution_source.py:994-1011`) reserves it for the two EXPENSIVE rungs. This rung
   is one GET, exactly the cost of the meta-refresh hop, and the cheap rungs run on the fast path
   unchanged, pinned by
   `tests/resolution_source/test_resolution_source_dispatch.py::TestFastPath::test_the_cheap_rungs_still_run_on_the_fast_path`.
   Say this in the docstring so nobody adds the gate later "for symmetry".
5. **Budget.**
   `budget_s = ctx.claim_rung_budget("impersonate", direct.status, url,
   RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S)`; `None` means the skip is already recorded, so
   return None. **`claim_rung_budget` indexes `_RUNG_WALL_SKIP_PHRASE[rung]`
   (`resolution_source.py:932`), so shipping without a phrase entry raises `KeyError` from inside
   the rung, and the provider's `asyncio.gather(..., return_exceptions=False)` turns that into
   losing every cited page for the question.** Add the entry in the same edit; see section 3.6.
6. `attempt = ctx.start_rung("impersonate", direct.status, url)`.
7. Call the transport:

   ```python
   response = await fetch_impersonated(
       retry_url,
       host_sems=host_sems,
       deadline_monotonic_s=time.monotonic() + budget_s,
       per_hop_timeout_s=RESOLUTION_SOURCE_HTTP_TIMEOUT,
       max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
   )
   ```

   Handle the declines, `ImpersonateUnpinnable` FIRST because it subclasses the base:

   - `ImpersonateUnpinnable`: `attempt.skipped_reason = "impersonate_unpinnable"` and return None.
     Stamp the skip on the attempt already started rather than appending a second one, which is
     the pattern `_rendered_rung` uses for `render_non_200`
     (`resolution_source.py:2127`). Note this path should be near-impossible in practice: the
     direct fetch got a 403 from this host, so the host DID resolve through the filtering resolver
     moments earlier. A nonzero count means a DNS flake or a rebinding host that flipped.
   - `ImpersonateDeclined` (everything else, including `ImpersonatePinNotHeld`): log at the level
     the shape deserves (WARNING for a refused hop or an oversized body, INFO for a transport
     error, and the transport already logged ERROR for a pin that did not hold) and return None.
     The attempt stays FIRED and `_run_rung` closes it on `direct.status`, which is exactly the
     `route=impersonate status=blocked` record the archive wants.
8. **A non-200 answer.** Memoize when `response.status in {403, 406, 429}`, stamp
   `attempt.outcome = _NON_OK_FETCH_STATUS.get(response.status, "error")`, and return None. For a
   still-403 that outcome is `blocked`, which reproduces the synthetic marker line already sitting
   in `tests/test_telemetry_markers.py:1341-1344`:
   `from_status=blocked rung=impersonate outcome=blocked wall_s=3.07`.
9. **A 200 answer.** Route on content type through the shared body classification described in
   3.4, stamp `attempt.outcome` with the resulting status, and return the result ONLY when it is
   `success`; otherwise return None and leave the direct `blocked` standing.

   Why decline on a 200 that classified as unreadable: replacing `blocked` with `js_wall` on the
   page's own record would change the fetch line's status without giving any later rung a way to
   act on it, because the dispatcher's browser block keys on `direct`, not on a rung's result. The
   escalation line still tells the whole story (`rung=impersonate outcome=js_wall`), and the paid
   `url_context` rung still gets its turn because `blocked` is in `_URL_CONTEXT_TRIGGER_STATUSES`.
   Stamping the rung's own outcome before deciding is the pattern `_rendered_rung` uses at
   `resolution_source.py:2155`. Add a FUTURE.md entry: an impersonated 200 that carries no
   readable text is a genuine candidate trigger for the browser rung, and wiring it up means
   letting the dispatcher escalate on a RUNG's result rather than only on `direct`, which is a
   restructure with its own review.

### 3.4 Classifying the impersonated body, without a second classification path

The invariant that makes the whole ladder work is that a rescued page is indistinguishable
downstream from a directly-fetched one, because every rung's body goes through the SAME
classification path. `_classify_html_body` (`resolution_source.py:1367-1465`) is already
transport-agnostic (bytes in) and is the documented one home for an HTML body whichever rung got
it. The raw-text and PDF branches are not: `_resolution_text_outcome` and `_resolution_pdf_outcome`
take an aiohttp response object.

So extract the bytes-level tails. Three small refactors in `resolution_source.py`, each leaving
the existing response-shaped function as a thin read-then-delegate wrapper, so there is exactly one
copy of each rule:

1. From `_resolution_text_outcome` (`resolution_source.py:1509-1555`), extract everything after
   `read_body_capped`:
   `_raw_body_outcome(body: bytes, current_url: str, content_type: str, *, http_status: int) ->
   FetchResult`. That is the charset-honouring `decode_text_body`, the markup strip on the
   `_RAW_TEXT_CONTENT_TYPES` branch, `vacuous_body_status`, its reason log line, and the
   `_truncate_with_marker` success.
2. From `_resolution_pdf_outcome` (`resolution_source.py:1621-1689`), extract everything after
   `read_body_capped`:
   `_document_outcome(body, current_url, content_type, ctx, *, http_status, from_status) ->
   FetchResult | _PendingDocument`. That is the `is_pdf_body` magic-byte check, the
   `_PendingDocument` construction and the `claim_rung_budget("pdf_local", ...)` gate.
3. Add the router the new rung calls:

   ```python
   async def _impersonated_body_outcome(
       response: ImpersonatedResponse, ctx: FetchContext, *, from_status: FetchStatus
   ) -> FetchResult:
   ```

   Same three-way content-type routing as `_resolution_response_outcome`
   (`resolution_source.py:1834-1848`), read in the same order and off the same vocabularies:
   `_HTML_CONTENT_TYPES` to `_classify_html_body`; `is_json_content_type` or
   `_RAW_TEXT_CONTENT_TYPES` to `_raw_body_outcome`; everything else, including an empty
   Content-Type, to `_document_outcome` plus `_finish_document` when it returns a
   `_PendingDocument`.

Four points on that router:

- **`http_status` is the IMPERSONATED response's 200, not the direct 403.** The bytes came with a
  200 and that is the honest record. This diverges from `_rendered_rung`, which passes
  `direct.http_status` deliberately because there the direct GET also got a 200. The consequence
  is that an impersonated rescue's fetch marker line reads `status=ok http=200 route=impersonate`
  with no `failure_class`, so the fact that the direct fetch was refused lives on the escalation
  line's `from_status=blocked`. That is exactly what a Wayback rescue already does (it reports the
  snapshot's own status), so the accounting is consistent. Write it down in the docstring.
- **No meta-refresh hop.** `_resolution_html_outcome` runs the meta-refresh rung after a no-content
  classification; the impersonated router deliberately does not, because following that hop would
  mean deciding which transport dials the target and re-entering the whole hop loop from inside a
  rung. Call `_classify_html_body` directly, and add the FUTURE.md note.
- **A PDF rescue produces `route="pdf_local"`, not `route="impersonate"`.** `_stamped_with_route`
  (`resolution_source.py:1013-1034`) sets `route` to the LAST rung that FIRED, and
  `_document_outcome`'s budget claim opens a `pdf_local` attempt on the page's own context. That
  is the same accounting a meta-refresh hop onto a PDF already produces, and the file's own words
  for it are "the hop got us the bytes, the local read is what the text came from". The
  impersonate attempt is still fully visible on its own escalation line, and the forecaster-facing
  caveat that renders is `pdf_local`'s, which carries the substantive limitation (these are
  query-relevant passages, not the whole document). See open question 1.
- **Do not pass `_aux_ctx(ctx)`.** The Wayback and derived-feed rungs use the child context because
  they fetch a DIFFERENT URL on the page's behalf and must not let that URL's inner rungs hijack
  the page's route. Here the bytes are the cited page's own, and the `pdf_local` attempt SHOULD be
  recorded on the page. Pass `ctx`.

`_classify_html_body` also wants `remaining_wall_s=ctx.rung_budget_s()` so the extractor's optional
precision pass declines under its floor, exactly as `_rendered_rung` passes it at
`resolution_source.py:2143`.

### 3.5 Route, caveat and the marker lines

Nothing to build for any of these; they already exist. Verify rather than add.

- **`route`.** The rung returns a `FetchResult` whose `route` is the default `"direct"`, and
  `_stamped_with_route` supplies `impersonate` because it is the last rung that fired. This is
  what `_rendered_rung` does too; do not set `route` by hand. A fired-and-failed attempt likewise
  gets `route=impersonate` stamped onto the returned `direct` result with no code at all, which is
  the readable `status=blocked route=impersonate` convention.
- **`ROUTE_CAVEATS["impersonate"]`** already reads: "One or more sections below were fetched on a
  retry that presented a different client fingerprint after the host refused ours; the content is
  the page's own." (`resolution_fetch_result.py:267-271`). It needs no change, and
  `tests/resolution_source/test_resolution_source_route_caveats.py:42-50` already passes.
- **`RESOLUTION_SOURCE_FETCH`** gains `route=impersonate` values, which the existing spec already
  documents at `scripts/telemetry/markers.py:816`. No regex change.
- **`RESOLUTION_SOURCE_ESCALATION`** gains `rung=impersonate` lines. The spec captures `rung` as
  `\S+` with no closed-set validation, so no regex change, and
  `tests/test_telemetry_markers.py:1341` already carries a synthetic line with that token. What
  DOES need editing is the spec's prose comment; see section 8.
- **No new `MarkerSpec`.** The two existing lines carry everything: the fetch line says what the
  page ended up as and by which route, and the escalation line says the trigger, the rung, the
  rung's own outcome and the rung's own wall. Adding a spec would archive a redundant column.

### 3.6 Vocabulary, bookkeeping and counts

Five hand edits, and one test in the suite already fails until each is done.

1. **`_RUNG_WALL_SKIP_PHRASE`** (`resolution_source.py:863-871`): add
   `"impersonate": "the impersonated retry"`, inserted after `"meta_refresh"` so dict insertion
   order stays the ladder order. This one dict is load-bearing three ways: `claim_rung_budget`
   indexes it for its log message, `_BUDGET_GATED_RUNGS = tuple(_RUNG_WALL_SKIP_PHRASE)` derives
   from it, and that tuple is what auto-generates the `impersonate_budget_skips` archive key.
   `test_the_budget_gated_rungs_are_the_phrased_rungs` pins the equality.
2. **`RungSkipReason`** (`resolution_fetch_result.py:210-224`): add `"impersonate_unpinnable"` and
   `"impersonate_host_refused"`, each with the same style of one-paragraph comment the existing
   members carry above the Literal. Say for the first that nothing was dialed and that a nonzero
   count means DNS disagreed with the direct fetch's own resolution, and for the second that it is
   the memo doing its job rather than a failure, the same distinction
   `rendered_no_text` draws.
3. **`_rung_counts`** (`resolution_source.py:3321-3424`): add three keys, each with the
   one-sentence reason the neighbours carry.
   - `"impersonate_attempts": fired_by_rung["impersonate"]`, following `rendered_attempts` and
     `wayback_attempts`.
   - `"impersonate_unpinnable_skips": skips_by_reason["impersonate_unpinnable"]`.
   - `"impersonate_host_refused_skips": skips_by_reason["impersonate_host_refused"]`.
   `impersonate_budget_skips` arrives free through the `_BUDGET_GATED_RUNGS` comprehension.
   `test_every_skip_reason_moves_a_counts_key` parametrises over `get_args(RungSkipReason)` and
   fails for any member with no key.
4. **No `impersonate_rescues` key.** Rescues are read off `route=` on the fetch marker, which
   already partitions the population by rung; the only per-rung rescue count in the file is
   `chrome_metric_withholds_rescued`, and that exists because it needs a join the marker cannot
   express. Say so in a comment so nobody adds one.
5. **No new `FetchStatus` and no new `FetchStatusReason`.** Every outcome this rung can produce is
   already in the vocabulary.

---

## 4. Gap-fill v2 integration

**Recommendation: do it, in this same change.** The transport's shape is already exactly right
for sharing, because `curl_options` being session-only forces the interface to be a plain async
function over (url, host map, budget) rather than something that takes a session, which is the
same shape `rendered_fetch.render_page` and `url_context_reader` already have and the same shape
v2's blocked branch can call. Beyond that the cost is four small edits plus one refactor, and the
population is the same federal hosts: v2's driver fetches bls.gov and cdc.gov pages routinely, so
leaving it out means the same 403 stays unrecovered on the surface where a discrepancy actually
moves a forecast.

The work, with its traps:

1. **`PlainFetchResult` has no `http_status`, so v2 cannot express the 403-only narrowing.** Add
   `http_status: int | None = None` to the dataclass (`agentic/fetch_outcomes.py:94-102`) and set
   it in `_non_ok_status_result` (`fetch_outcomes.py:222-244`). Without it a trigger keyed on
   `status == "blocked"` alone would hand the transport a URL v2 itself refused, because v2 reaches
   `blocked` in three places: `_non_ok_status_result` for
   `_RETRYABLE_FETCH_BLOCK_STATUSES = {403, 406, 429}`, `_fetch_plain_url_block` for a Metaculus
   self-reference (`fetch_outcomes.py:196-218`), and `tools._fetch_plain` for a non-public URL
   (`tools.py:335-343`). That is the exact bypass the Tier-1 SSRF tests guard against.
2. **Extract v2's bytes-level classification.** Everything in `_plain_response_outcome`
   (`tools.py:270-315`) after `body = await _read_response_body(...)` is already pure
   bytes-and-content-type work: the declared-PDF and magic-byte branches into
   `local_document.pdf_fetch_result`, the `_body_is_document` branch, the charset-honouring
   `decode_text_body`, `_plain_html_outcome` and `_plain_textual_outcome`. Extract it as
   `_plain_body_outcome(body, content_type, current_url) -> PlainFetchResult` and have both the
   aiohttp path and the impersonated path call it. That is what makes the impersonated body get
   v2's FULL classification rather than a second partial reimplementation, and it is the mirror of
   the Tier-1 extraction in 3.4.
3. **The call site.** In `tools.fetch` (`tools.py:600-612`), between the two existing lines:

   ```python
   plain = await _fetch_plain(url)
   if plain.status == "blocked" and plain.http_status == 403:
       impersonated = await _try_impersonated_fetch(plain)
       if impersonated is not None:
           plain = impersonated
   if plain.status == "blocked":
       return ToolOutcome(...)
   ```

   `_try_impersonated_fetch` is v2's own mapping wrapper, modelled line for line on
   `_try_rendered_fetch` (`tools.py:358-418`): call the transport with
   `host_sems=_FETCH_HOST_SEMAPHORES`, fold every `ImpersonateDeclined` back into `None` because
   this ladder's callers only know `None`, and on a 200 hand the body to `_plain_body_outcome` with
   `method="impersonate"`. Dial `plain.url`, which is already the last hop of v2's own guarded
   redirect loop, the same choice `_try_rendered_fetch` documents.
4. **The provenance tier is mandatory, not optional.** Add `"impersonate": "fetched"` to
   `_METHOD_TO_TIER` (`agentic/provenance.py:62-71`). A method absent from that dict grants NO
   verification tier, so a page the rung really did retrieve would stay untiered and its
   discrepancy would be silently demoted below the briefing, which is the 131.3 failure mode that
   comment describes. This is the single most important line of the v2 integration.
5. **The host gate is v2's own map**, not Tier 1's process-wide one, which the transport's
   `host_sems` parameter already handles.

What v2 does NOT get, deliberately: no rung attempt, no `RungSkipReason`, no `route`, no
escalation marker. v2's telemetry is `method` on the `ToolOutcome` and the findings artifact, and
`method=impersonate` is the whole record it needs.

---

## 5. Dependency change: already landed, with cleanup left

**The brief's premise is stale on this point. Do not redo it.** Commit b0dd940, the current tip of
this branch, already:

- added `"curl-cffi>=0.15.0"` to `[project].dependencies` (`pyproject.toml:35`);
- removed it from `[dependency-groups] dev`;
- removed it from the deptry `DEP004` ignore list, which is now just `["matplotlib"]`
  (`pyproject.toml:346`);
- rewrote the explanatory comment at `pyproject.toml:338-345` to name
  `research/impersonated_fetch.py`.

So there is no `uv add` to run and no lock change to make. `uv.lock` already carries curl-cffi
0.15.0 under the project's own dependencies with `specifier = ">=0.15.0"`, and its
`cp310-abi3-manylinux2014_x86_64` wheel is abi3 for CPython 3.10 and up, so it covers cp312 on
`ubuntu-latest` x86_64 with no source build. macOS arm64, musllinux, aarch64 and the Windows
wheels are all present too.

Every CI and bot job installs the main group: the two `ci.yaml` jobs use `uv sync --dev --frozen`,
the five bot workflows use `uv sync --no-dev --frozen`, and `--no-dev` drops only the dev group. So
curl-cffi now ships to every bot run. Nothing else in the packaging needs to change, and
`make lint_imports` needs no contract edit, because the new module sits inside
`metaculus_bot.research` and the existing contracts are about `metaculus_bot` versus `scripts` and
about the research-to-forecaster direction.

`make audit` (osv-scanner over `uv.lock`) reported "No issues found" across all 223 packages
including curl-cffi 0.15.0 on 2026-09-04, with only the five pre-existing cryptography advisories
filtered. Run it again at the end. **State the limitation wherever the dependency is justified:**
osv-scanner reads Python package versions out of the lockfile and cannot see the libcurl 8.15.0
and BoringSSL binaries VENDORED inside the wheel, and curl upstream is well past 8.15.0, so a
libcurl CVE would not be reported by this gate. That is worth a sentence in the `DEP004` comment
block and in `docs/operations.md`, because the repo's CVE hygiene otherwise leans on `make audit`.

### What IS left: three stale artifacts

1. `.github/workflows/fetch_diagnostic.yaml:48-53` still says curl_cffi "is declared only in
   pyproject's dev group ... so `--with curl_cffi` is still what supplies it here. It stays out of
   the prod dependency set". Rewrite that comment.
2. `.github/workflows/fetch_diagnostic.yaml:61` still passes
   `uv run --frozen --no-dev --with curl_cffi`, now a no-op. Drop `--with curl_cffi`.
3. `scripts/probes/fetch_diagnostic.py` still treats curl_cffi as optional:
   `impersonation_available()` at `:114-116`, the `try: from curl_cffi import requests except
   ImportError` guards at `:167-171` and `:192`, the stale comment above the first, and the
   warning block in `main()` at `:316-318`. With curl-cffi a hard runtime dependency the defensive
   import meets none of `AGENTS.md`'s three allowed justifications for a function-scoped import,
   and it contradicts the fail-fast rule. Hoist `from curl_cffi import requests as curl_requests`
   to module scope, delete `impersonation_available()` and the main() warning, and drop the
   `except ImportError` arms.

---

## 6. Constants to add to `constants.py`

All three go in the existing commented block headed "Resolution-source escalation rungs"
(`metaculus_bot/constants.py:755-854`), following the block's convention that every constant
carries its measurement or its judgment written out in prose. Do not put any of them inline in a
function; `AGENTS.md` puts constants in `constants.py`.

```python
RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S: float = 3.0
```
The rung is one GET against a host that just answered us, so its floor is the meta-refresh hop's
3.0 s (`constants.py:761`) and the derived-feed GET's 3.0 s (`:800`), on the same "0 to 2 s
typical" probe basis as `RESOLUTION_SOURCE_HTTP_TIMEOUT`. Deliberately NOT the browser rung's
12.0 s: no process is launched, no gate is contended process-wide, and no model round trip
happens.

```python
IMPERSONATE_BROWSER_TARGET: str = "chrome146"
```
The concrete curl-cffi impersonation profile, pinned rather than the floating `"chrome"` alias.
Verified in this venv: `"chrome"` resolves through `DEFAULT_CHROME` to `chrome146`
(`.venv/.../requests/impersonate.py:78`, and `normalize_browser_type("chrome") == "chrome146"` at
runtime). The alias is a plain source constant, so a routine curl-cffi bump would silently change
the TLS and HTTP/2 fingerprint and the User-Agent the federal hosts see, which can flip the rung's
success rate with no code change and makes the section 1 measurement non-reproducible. Pinning
makes a fingerprint change a reviewable diff. No `RESOLUTION_SOURCE_` prefix because gap-fill v2
uses the same transport, following `RENDERED_DOM_MAX_CHARS`' precedent.

```python
RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S: float = 0.5
```
Not a new value: MOVE the existing `_MIN_HOP_TIMEOUT_S` from `resolution_source.py:801` into
`constants.py`, keeping its comment verbatim, and have both `_fetch_one_hop` and the new transport
read it. Value-preserving and mechanical, so it satisfies the "timing and deadline code gets
strictly-safer changes only" rule, and it stops the transport from either duplicating the floor or
reaching for a private name in a module it cannot import at module scope.

Reused as-is, not duplicated: `RESOLUTION_SOURCE_HTTP_TIMEOUT` (`:593`),
`RESOLUTION_SOURCE_MAX_RESPONSE_BYTES` (`:596`), `RESOLUTION_SOURCE_PER_URL_MAX_CHARS` (`:597`),
`MAX_REDIRECTS` and `REDIRECT_STATUSES` (`http_fetch.py:102-103`). Note that
`scripts/probes/fetch_diagnostic.py:50` has its own `_IMPERSONATE_TIMEOUT_S = 25.0`, which is
ABOVE the Tier-1 HTTP timeout; it is a diagnostic's own knob and must not be copied into
production.

**No env flag.** The other free rungs have none; only the paid `url_context` rung is flag-gated,
because it spends. The memo, the 403-only trigger and the budget floor are the bounds. See open
question 2 if the operator wants a kill switch anyway.

---

## 7. Tests, test first

Write the tests before the implementation. Load the `test-driven-development` skill. Every test in
this section runs under `make test` with no network and no key.

### 7.1 The containment rule, which is also the money-safety backstop

`tests/conftest.py::_block_native_egress` is autouse and monkeypatches exactly two attributes:
`curl_cffi.requests.Session.request` and `curl_cffi.requests.AsyncSession.request`
(`tests/conftest.py:217-218`, with the classes imported hard at module scope at `:8-9`). Both
replacements append to `native_egress_attempts` and raise
`RuntimeError("Native network egress blocked in tests by _block_native_egress: curl_cffi <METHOD>
<URL>")`, and the fixture FAILS the test at teardown if anything was recorded. Every public entry
lands on `request`: the verb helpers call `self.request(method=..., url=...)` and `stream()` calls
`self.request(..., stream=True)`. `tests/test_egress_guards.py:73-105` pins all four shapes.

Two consequences to hold to:

- **The guard must be the thing that would fail.** No test may reach the real transport. A rung
  that swallowed the `RuntimeError` into a decline would still fail at teardown, which is exactly
  the behaviour we want, and it means the guard cannot be defeated by accident.
- **Patch a seam in the new transport module, not `AsyncSession.request`.** A test that
  re-monkeypatches `AsyncSession.request` overrides the autouse guard for that test and disarms
  the backstop on precisely the code path the backstop exists for. So the rung tests patch
  `resolution_source.fetch_impersonated` (a module attribute, the same convention
  `resolution_source.render_page` and `resolution_source.run_url_context_read` already use), and
  the transport tests patch `impersonated_fetch.AsyncSession` with a fake CLASS, leaving the
  guard's patch of the real class armed underneath. Add `tests/test_egress_guards.py` coverage
  asserting the guard still fires for a session built the way the transport builds one.

### 7.2 The fake for the curl chokepoint

Add to `tests/resolution_source_fakes.py`, beside `FakeSession` / `FakeResponse` / `_fake_render` /
`paid_reader`. Note that `FakeSession` CANNOT intercept this rung, because curl-cffi never goes
through `session.get`.

- `FakeCurlSession`: a class standing in for `AsyncSession`. Records the constructor kwargs
  (`impersonate`, `allow_redirects`, `verify`, `timeout`, and the whole `curl_options` dict, which
  is what the pin assertions read), supports `async with`, and exposes `stream(method, url)` as an
  async context manager yielding a stub response. Track a per-instance and a class-level call log
  so a multi-hop test can assert one session per hop and the pin operand of each.
- The stream stub cannot be a real `curl_cffi.requests.Response`, because stream mode needs
  `queue`, `astream_task` and `quit_now` wired to a live curl handle. It needs `status_code`,
  `headers` (build a genuine `curl_cffi.requests.headers.Headers({...})`, which is
  case-insensitive), `url`, `redirect_url`, `primary_ip`, an async-generator `aiter_content()`, a
  `quit_now` object with `set()` that records the call, and an async `aclose()`.
- A NON-stream fake can use a real `Response()`: verified constructible with no arguments,
  defaulting to `status_code=200`, `content=b""`, `headers=Headers()`, `url=""`,
  `redirect_url=""`. Set `ok` explicitly if anything reads it.

### 7.3 Transport tests, new file `tests/test_impersonated_fetch.py`

Sitting beside `tests/test_http_fetch.py` (the `FilteringResolver`, `build_session`,
`read_body_capped` and `BROWSER_HEADERS` suite) and `tests/test_rendered_fetch.py`. Note that
`tests/test_fetch_hardening.py` is NOT the SSRF suite despite the name: it covers
forecasting-tools' Metaculus-API retry wrapper.

1. **The pin is built from the VETTED address.** Patch `rendered_fetch.resolve_pinned_host` to
   return a known `(host, ip)` and assert the constructor got
   `curl_options[CurlOpt.RESOLVE] == ["www.example.com:443:93.184.216.34"]`. Assert the value is a
   `list`, explicitly, with a comment naming the character-iteration trap: the type hint says
   `str` and an inert pin fails OPEN.
2. **The port comes from the URL.** Parametrise over `https://h/x` (443), `http://h/x` (80) and
   `https://h:8443/x` (8443).
3. **An IPv6 vetted address is bracketed.** `["v6.example.com:443:[2606:2800:220:1:248:1893:25c8:1946]"]`.
   Both bracketed and bare forms work against a `::1` listener (measured), and bracketed is the
   unambiguous one.
4. **The proxy options are always set**, `CurlOpt.PROXY == ""` and `CurlOpt.NOPROXY == "*"`, with
   the test's docstring carrying the `trust_env`-is-dead-code receipt. Add a second test that sets
   `http_proxy` in the environment through monkeypatch and asserts the options are still set, so
   the invariant is pinned against the environment rather than against a clean one.
5. **A private address is refused before any request.** Patch
   `resolution_source.socket.getaddrinfo` to return `10.0.0.5`, assert `ImpersonateUnpinnable`, and
   assert the fake session class was never instantiated. Repeat for a link-local literal
   (`http://169.254.169.254/latest/meta-data/`), a userinfo URL, a non-http scheme, a non-ASCII
   host and a trailing-dot host.
6. **A mixed resolution is refused.** One public and one private address for the same host rejects
   the whole hostname.
7. **A redirect to a private host is refused.** Hop 0 answers 302 with a `Location` on a host that
   resolves privately; assert `ImpersonateHopRefused` carrying `"ssrf_blocked"`, and assert no
   second session was built.
8. **A redirect to metaculus.com is refused** with `"metaculus_self_ref"`, and the refusal ORDER is
   pinned: a URL that is both non-public and a self-reference reports `ssrf_blocked`.
9. **A redirect is re-pinned.** Hop 0 on host A redirects to host B; assert two sessions, each with
   its own single-entry `RESOLVE` operand for its own host, and that the second hop's pin does not
   mention host A.
10. **A relative `Location` is absolutised.** Assert the second hop's URL comes from
    `redirect_url`, and add a case where `redirect_url` is empty and only a relative `location`
    header is present, so the `urljoin` fallback is covered.
11. **The redirect cap is `MAX_REDIRECTS`.** Six hops raises `ImpersonateRedirectLimit`; five
    succeeds. Read the cap from `http_fetch.MAX_REDIRECTS`, never a literal.
12. **The body cap cuts the transfer.** A stream yielding past `max_bytes` raises
    `ImpersonateBodyTooLarge`, and `quit_now.set()` was called BEFORE `aclose()`. That ordering is
    the whole point: `aclose()` alone drains the rest of the body.
13. **A body exactly at the cap succeeds**, so the boundary is `>` and not `>=`, matching
    `read_body_capped`.
14. **The wall bound is the caller's deadline.** With `deadline_monotonic_s` already past, no
    request is made and `ImpersonateTransportError(failure_class="timeout")` is raised. With a
    stream that never yields, the `asyncio.timeout` fires and produces the same failure class.
    Include a comment stating that libcurl's own `timeout=` does not bound a streamed body, with
    the 4.5 s against `timeout=1.0` measurement.
15. **The per-hop timeout is clamped after the gate.** Assert the constructor's `timeout` equals
    `min(per_hop_timeout_s, remaining)` and never exceeds `per_hop_timeout_s`.
16. **The impersonation target is pinned.** Assert the constructor got the exact string
    `"chrome146"` from `IMPERSONATE_BROWSER_TARGET`, and assert it is NOT the alias `"chrome"`,
    with the docstring saying that a curl-cffi bump would otherwise move the fingerprint silently.
17. **`allow_redirects=False` and `verify=certifi.where()`** are on the constructor.
18. **No `BROWSER_HEADERS`.** Assert no `headers` argument was passed, with the coherence reason in
    the docstring.
19. **The pin assertion fails shut.** A stub whose `primary_ip` differs from the vetted address
    raises `ImpersonatePinNotHeld` and returns no body. Add the empty-`primary_ip` case, asserting
    the deferred re-check after the read.
20. **Every libcurl failure maps to a failure class.** Parametrise over `Timeout`, `SSLError`,
    `CertificateVerifyError`, `DNSError`, `ConnectionError`, `IncompleteRead` and a bare
    `RequestException`, asserting the expected token and that `exc` carries the class name. Add one
    test asserting the transport does NOT catch `OSError` broadly, since `RequestException`
    subclasses it.
21. **A non-200 is DATA.** A 403 stub returns an `ImpersonatedResponse` with `status == 403` and
    raises nothing.
22. **Politeness.** Two concurrent calls to the same host serialise on the host gate; two different
    hosts run concurrently. Model on the existing per-host in-flight-peak assertions
    `FakeSession` supports.
23. **One session per hop, each closed.** Assert `close()` or `__aexit__` ran for every session
    built, including on every decline path, so no `_force_timeout` polling task leaks. Assert no
    pending asyncio tasks remain after the call.
24. **Memo behaviour.** `note_impersonation_refused` then `impersonation_refused` is True for
    another URL on the same host and False for a different host; `reset_impersonation_memo` clears
    it.

### 7.4 Rung tests, new file `tests/resolution_source/test_resolution_source_impersonate_rung.py`

Model the structure on `tests/resolution_source/test_resolution_source_wayback_rung.py`.

**First, the package fixture.** A rung that fires on `blocked` will fire inside dozens of existing
tests that deliberately return 403, exactly as the Wayback rung would have. Add an autouse fixture
to `tests/resolution_source/conftest.py` declining it by default, following
`_decline_the_wayback_rung`'s pattern and writing the same kind of docstring:

```python
@pytest.fixture(autouse=True)
def _decline_the_impersonate_rung(monkeypatch):
    monkeypatch.setattr(resolution_source, "_IMPERSONATE_TRIGGER_HTTP_STATUS", frozenset())
```

Emptying a module-level trigger constant declines before the rung looks at anything, so it records
no attempt, claims no route and issues no request. That means `_impersonate_rung_applies` should
read its 403 from a module constant
`_IMPERSONATE_TRIGGER_HTTP_STATUS: frozenset[int] = frozenset({403})` rather than an inline
literal, which is the same shape `_WAYBACK_TRIGGER_STATUSES` has and the same reason. Tests that
exercise the rung restore the module's OWN constant object, imported so the two cannot drift.

Also add `reset_impersonation_memo()` to the conftest's `_reset_shared_gates` fixture (or its own
autouse fixture), before and after the yield, for the same reason the render memo is reset: it
outlives one provider call by design, so without a reset a test inherits another's memo and passes
or fails on order.

Then the tests:

1. **The trigger population.** Parametrise: a `blocked` result with `http_status=403` fires; 406,
   429 and a 301 (the self-reference shape) do not; `error` with `http_status=401` does not;
   `js_wall`, `no_resolving_content`, `not_found`, `stale_data`, `empty_body`,
   `unreadable_document`, `unsupported_type` and `success` do not.
2. **`ssrf_blocked` is excluded, and loudly.** Add to
   `tests/resolution_source/test_resolution_source_third_party_rung_ssrf.py`, in
   `TestBothTriggerSetsExcludeOurOwnRefusal`, that the impersonate trigger cannot fire on
   `ssrf_blocked` (which is trivially true given the 403 test, so assert it through
   `_impersonate_rung_applies` rather than through set membership, and keep the existing
   spelling-still-exists guard). Add the end-to-end case: a private-resolving URL produces
   `ssrf_blocked` with `route == "direct"`, no rung attempts, and the transport never called.
3. **A rescue.** A 403 direct fetch plus a stubbed transport returning HTML with real prose gives
   `status == "success"`, `route == "impersonate"`, `http_status == 200`, one rung attempt with
   `rung="impersonate"`, `from_status="blocked"`, `outcome="success"` and a non-None `wall_s`, and
   `counts["impersonate_attempts"] == 1`.
4. **The rescue renders the caveat.** The published section carries
   `ROUTE_CAVEATS["impersonate"]` verbatim, read from the mapping rather than restated.
5. **A still-403 answer.** The returned result is the DIRECT one with its `failure_class=http_403`
   and `server=akamaighost` intact, `route == "impersonate"`, one attempt with `outcome="blocked"`,
   and the host is memoized.
6. **The memo saves the second URL.** Two cited URLs on one host, the first refused under
   impersonation: the second records `impersonate_host_refused` as a skip, emits NO escalation
   line, and the transport was called exactly once.
7. **A 404 under impersonation does not memoize.** Second URL still tries.
8. **A 200 that classifies as unreadable declines.** The page's final status stays `blocked`, the
   attempt's `outcome` is `js_wall` (or `no_resolving_content`), and the paid rung is still
   reachable for that URL.
9. **Budget skip.** With `rung_budget_s` stubbed below `RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S`,
   the transport is never called, `counts["rung_budget_skips"] == 1` and
   `counts["impersonate_budget_skips"] == 1`, and no escalation line is emitted.
10. **The unpinnable skip.** The transport raising `ImpersonateUnpinnable` leaves ONE attempt
    carrying `skipped_reason="impersonate_unpinnable"`, no escalation line, and
    `counts["impersonate_unpinnable_skips"] == 1`.
11. **It runs on the fast path.** With `fast_path=True` the rung still fires and records NO
    `fast_path` skip, alongside the existing cheap-rung assertion in
    `test_resolution_source_dispatch.py::TestFastPath`.
12. **Ladder position.** A 403 page where impersonation rescues never reaches the Wayback rung
    (restore the real `_WAYBACK_TRIGGER_STATUSES` for this test and assert no archive URL was
    requested) and never reaches the paid reader (arm it with `arm_paid_rung` and assert zero
    calls). A 403 page where impersonation fails DOES reach both.
13. **It dials `direct.url`, not the cited URL,** when the direct fetch followed a redirect before
    being refused, and the rung ATTEMPT stays keyed on the cited URL. Mirror
    `test_resolution_source_rendered_rung.py`'s equivalent.
14. **A landing the guard refuses is a decline, not a terminal result.** When `direct.url` differs
    from the cited URL and fails `_hop_refusal`, the rung returns None with no attempt.
15. **A PDF rescue.** The transport returns `%PDF-` bytes with `application/pdf`; the digest
    publishes, `route == "pdf_local"`, and BOTH escalation lines appear
    (`rung=impersonate` and `rung=pdf_local`). This is the bls.gov `wkstp.pdf` case from section 1
    and it must be covered.
16. **A JSON or CSV rescue** goes through the raw-body path, including the vacuous-body refusal.
17. **Marker output.** Capture logs and assert the exact `RESOLUTION_SOURCE_FETCH` and
    `RESOLUTION_SOURCE_ESCALATION` lines for a rescue and for a failed attempt, then parse them
    with `scripts/telemetry/markers.parse_log_text` and assert the harvested fields, so the lines
    are pinned as the data contract they are.
18. **Bookkeeping guards.** The three existing parametrised tests in
    `test_resolution_source_rung_bookkeeping.py` cover the new vocabulary automatically. Add the
    explicit `_BUDGET_GATED_RUNGS` membership assertion for `"impersonate"` and one asserting
    `_RUNG_WALL_SKIP_PHRASE` has the entry, so a `KeyError` inside `claim_rung_budget` can never
    ship.

### 7.5 Gap-fill v2 tests, in `tests/test_agentic_tools.py`

1. `PlainFetchResult.http_status` is set for a 403, a 406, a 429 and any other non-200.
2. The impersonated retry fires for `blocked` with `http_status == 403` and NOT for `blocked` from
   a self-reference or a non-public URL, which is the SSRF bypass case and deserves its own test
   asserting the transport was never called.
3. A rescue returns `method="impersonate"` with the page's text, and
   `provenance._method_to_tier("impersonate") == "fetched"`. Add a completeness-style test that
   every `method` a `ToolOutcome` can carry for a real retrieval is present in `_METHOD_TO_TIER`,
   so the next method cannot be added untiered.
4. A decline leaves the existing `blocked` outcome byte-identical to today's.
5. The extracted `_plain_body_outcome` gives identical results for a body whichever path supplied
   it: parametrise one HTML, one PDF, one CSV and one image body across both call paths.

### 7.6 Gates

`make test` (the full suite, about 105 s), `make lint`, `make format`, `make typecheck`,
`make deps`, `make lint_imports`, `make audit`. `make all` runs format, lint, deps, lint_imports,
typecheck and the verbose suite. basedpyright must stay at 0 errors. Then push and check the run
with `gh run list --repo No-Stream/metaculus-bot --branch impersonate-rung`; the `--repo` flag is
required because `origin` is the fork and `upstream` is the Metaculus template. CI green is the
gate, not a local green run.

---

## 8. Docs and registry edits

Ten surfaces describe the ladder or the reserved token, and each needs a specific edit. Line
numbers are from 2026-09-04 and will drift as you edit; grep the quoted phrase.

| File and line | What it says now | Edit |
|---|---|---|
| `metaculus_bot/research/resolution_source.py:1-80` (module docstring) | Enumerates the rungs: "Three free rungs sit under Tier 1", "A fourth rung leaves our own aiohttp client" | Add the impersonation rung to the enumeration, with the 4-of-4 diagnostic result and the note that it is the one rung that leaves aiohttp without leaving our address |
| `docs/research.md:869-873` | "The four rungs added on 2026-09-03" | Name the new rung and its date |
| `docs/research.md:992-995` | "Four more rungs shipped on 2026-09-03, tried cheapest first" | Update the count and the ordering paragraph; state that impersonate sits between the direct fetch and Wayback and why |
| `docs/research.md:1176-1184` | "One sentence belongs to a rung that has not shipped: `impersonate` is in the route vocabulary and carries a caveat, but nothing produces that route" | Rewrite: the rung ships, the caveat renders, and the completeness test now guards a live token |
| `docs/research.md:1193-1258` | "Six of the keys count rungs that FIRED" plus the per-rung `*_budget_skips` list and the skip-reason catalogue | Seven keys; add `impersonate_attempts`, `impersonate_budget_skips`, `impersonate_unpinnable_skips`, `impersonate_host_refused_skips`, each with its binding constraint in one clause |
| `docs/research.md` SSRF paragraph (just after :1258) | Describes the aiohttp per-hop re-guard and the rendered rung's different guard | Add the impersonated rung's third shape: libcurl bypasses the filtering resolver, so the rung pre-resolves and pins with `CURLOPT_RESOLVE`, one session per hop, and asserts the connected address |
| `docs/operations.md:1340-1342` | "`impersonate` is reserved in the vocabulary for a rung that is not built" | Rewrite as a live route. Also add the vendored-libcurl note from section 5 wherever `make audit` is described |
| `scripts/telemetry/markers.py:843-889` | The `resolution_source_escalation` spec's prose enumerates the from_status and rung pairs as "disjoint by construction" with no `impersonate` entry | Add `blocked` paired with `impersonate`. Keep the existing sentence that `blocked` never pairs with a browser rung, which stays true |
| `scripts/telemetry/markers.py:816` | The `route` enumeration comment already lists `impersonate` | No change. Verify only |
| `AGENTS.md:146` | "`fetch_ladder_plan_2026-09-03.md`, whose ladder is built except the TLS-impersonation rung" | Drop the exception clause and point at this plan file |
| `AGENTS.md:160-161` and `docs/architecture.md:309-310` | The "Outbound fetch transports (never hand-rolled)" rows | Add `impersonated_fetch.py` (the curl-cffi TLS-impersonating retry) to both |
| `FUTURE.md:2876-2887` | The parked "GitHub-runner egress reputation" item whose step 0 was the diagnostic | Record the diagnostic's result and that the rung shipped; keep the both-403 hosts as the open half |
| `FUTURE.md` (new entries) | | Three: a 200 throttle interstitial as a Tier-1 trigger, needing v2's phrase check ported; an impersonated 200 with no readable text as a browser-rung trigger, needing the dispatcher to escalate on a rung's result; and a meta-refresh hop from an impersonated body |

Nothing under `metaculus_bot/performance_analysis/` or elsewhere in `scripts/` references
`resolution_source_fetch` or `resolution_source_escalation`, so the residual-analysis tooling needs
no change. The only non-test consumers are `markers.py`, `resolution_source.py` and
`resolution_fetch_result.py`.

---

## 9. QA plan

### What free local QA can and cannot prove

The unit and integration suite proves every invariant that does not need a hostile host: the pin
operand, the fail-shut refusals, the per-hop re-guard and re-pin, the body cap, the wall bound, the
pinned impersonation string, the classification parity, the counts and the marker lines.

What it cannot prove is that impersonation actually recovers the four hosts from production egress,
because the operator's laptop is not fingerprint-scored: column A already gets 200 there, so there
is no 403 to recover and a local run of the rung would exercise the happy path of a page that never
needed it.

### The live proof: extend the free diagnostic with a column D

Extend `scripts/probes/fetch_diagnostic.py` with a fourth column that runs the real Tier-1 provider
entry point per URL and prints what the ladder produced:

```python
results = await fetch_resolution_sources([probe_url.url], query="", fast_path=False)
```

Print, per URL, `status`, `route`, and the rung attempts as `rung=<token> outcome=<token>` pairs
read off `FetchResult.rung_attempts`, plus `primary_ip` if you thread it through (or simply confirm
no `ImpersonatePinNotHeld` ERROR line appears in the log). A row reading
`status=ok route=impersonate rung=impersonate outcome=success` on each of the four Akamai federal
URLs, from the runner, is the proof. The verdict paragraph should count them.

Four things to get right in that column:

1. **It is structurally free.** The workflow passes NO secrets, so `GOOGLE_API_KEY` is unset and
   `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED` is unset, which means the one paid rung cannot fire; it
   self-declines on the flag before the key. Keep it that way and say so in the workflow comment.
2. **It costs extra requests.** Column D runs the whole free ladder, so a both-403 host will also
   fetch the archive, repeating column C's request. The probe's `HostPacer` cannot see inside the
   provider, though the provider's own per-host `Semaphore(1)` serialises same-host requests. Budget
   1 to 3 extra requests per URL and state it in the column's docstring, because politeness is a
   stated property of this probe.
3. **Bump the step timeout.** The step is capped at 15 minutes and the job at 20. Column D adds up
   to 10 URLs times a 45 s provider wall in the worst case, so raise the step cap and the job cap,
   and recompute the worst-case sentence in the workflow comment.
4. **It reads production code only.** Do not add a probe-local copy of anything. Column D's whole
   value is that it exercises `fetch_resolution_sources`, so a result it reports is one production
   would get.

### The ask the operator has to approve

Dispatching the diagnostic is FREE (no secrets, no LLM, no publish) but it burns GitHub Actions
minutes and probes federal hosts from the runner IP, so per `AGENTS.md`'s cost gate it still needs
the operator's go. Hand them exactly this, after the branch is pushed:

```
gh workflow run fetch_diagnostic.yaml --repo No-Stream/metaculus-bot --ref impersonate-rung
gh run list --repo No-Stream/metaculus-bot --workflow fetch_diagnostic.yaml --limit 1
```

The `--repo` flag is mandatory: `origin` is the fork, `upstream` is the Metaculus template, no
default repo is set, and a bare `gh workflow` command silently targets upstream. What to read
afterwards: the four Akamai federal rows in column D. What each answer means:

- **All four `route=impersonate status=ok`.** The rung works from production egress. Merge.
- **Column B is 200 and column D is not.** Our rung has a bug the transport tests missed, most
  likely the CAINFO or pin question from section 2.3. Check the run log for `ImpersonatePinNotHeld`
  at ERROR and for a `failure_class=tls` on the escalation line, and fix before merging.
- **Column B is also 403 now.** The hosts changed their scoring since 2026-09-04, or the runner IP
  moved. The rung is still correct and still free; re-price the case with the operator.

**No paid run is part of this verification.** The `make backtest_smoke_test` and the bot workflows
are a final pre-merge check the operator fires once, deliberately, and a rung whose whole
population is a 403 that the free diagnostic can measure directly does not need one.

---

## 10. Open questions for the lead

Only the ones where a decision changes what gets built. Each carries a recommendation, so absent
an answer, build the recommendation.

1. **Should an impersonated PDF rescue render the `impersonate` caveat or the `pdf_local` one?**
   As designed in 3.4 it renders `pdf_local`, because `route` is the last rung that fired and the
   local read is what produced the text, which is the accounting a meta-refresh hop onto a PDF
   already produces. The forecaster then reads "these are query-relevant passages, not the whole
   document" but not "the host refused us and we retried with a different fingerprint". One of the
   four proven URLs is a PDF, so this is not hypothetical. **Recommendation: keep `pdf_local` and
   accept the lost sentence.** The impersonation fact does not change how a forecaster should weight
   the content, unlike a capture's age or a model's mediation, and the alternative is either a
   second route field or a rung that lies about which one produced the text. Changing this means
   letting a result carry more than one route, which is a `FetchResult` and `ROUTE_CAVEATS` change
   with its own review.
2. **Does the rung want an env kill switch?** As designed it has none, matching every other free
   rung; only the paid rung is flag-gated. **Recommendation: no flag.** The 403-only trigger, the
   per-run host memo and the 3 s budget floor bound it, and a new flag is a new thing to get wrong
   in five workflow yamls. If you want one anyway, the shape is
   `RESOLUTION_SOURCE_IMPERSONATE_ENABLED` defaulting to ON in code (unlike the paid rung's
   default-off), which adds one constant, one `env_flag_enabled` call, one skip token and five
   yaml edits.
3. **Ship the gap-fill v2 half in this change, or a follow-up?** Section 4 prices it at four small
   edits plus one refactor of `_plain_response_outcome`'s tail, and it needs the
   `_METHOD_TO_TIER` line or a genuinely retrieved page gets silently demoted.
   **Recommendation: same change.** The transport, the tests and the review context are all here,
   and a v2 rung landing separately means the `PlainFetchResult.http_status` field and the
   classification refactor get reviewed twice.
