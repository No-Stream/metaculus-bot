"""One-shot identity check for the Metaculus API host, run before we send the token.

On 2026-07-21 metaculus.com's DNS was repointed at a GoDaddy parking host (WHOIS
updated that day, GoDaddy nameservers serving parking IPs, a fresh GoDaddy DV
cert for www.metaculus.com issued the same afternoon). The bot's scheduled
GitHub Actions runs kept firing and forecasting-tools attached
``Authorization: Token $METACULUS_TOKEN`` to the question-fetch GET
(``forecasting_tools/helpers/metaculus_api.py`` line 409:
``requests.get(url, params=params, **cls._get_auth_headers())``), leaking the
token to an unknown host before dying on an opaque 404. The real Metaculus
origin was still alive behind Cloudflare — this was an upstream domain incident,
not a code bug.

``verify_metaculus_api_identity`` makes ONE unauthenticated request (no token,
no headers) to the posts list and confirms the host behaves like the real API
before any authenticated call runs. Two jobs:

1. Never send the token to a host we haven't sanity-checked — the preflight
   itself carries no credentials.
2. Fail fast with a diagnostic that names the likely cause (DNS parking/hijack),
   instead of a bare ``HTTPError`` traceback from deep inside forecasting-tools.

The request goes through an isolated ``requests.Session`` with
``trust_env=False``. That is load-bearing, not hygiene: with the default
``trust_env=True``, ``Session.prepare_request`` calls ``get_netrc_auth(url)``
and a matching ``machine www.metaculus.com`` — or a generic ``default`` — entry
in ``~/.netrc`` / ``$NETRC`` would attach ``Authorization: Basic ...`` to this
"unauthenticated" probe (reproduced live on requests 2.34.2), defeating the
whole point on exactly the hijack path this guard exists to cover. Disabling
``trust_env`` also skips proxy-env and ``REQUESTS_CA_BUNDLE`` pickup for this
one call; that is an accepted trade — GHA runners have direct egress, and the
no-credentials-to-an-unknown-host invariant wins over honoring ambient network
config for a single identity probe.

Deliberately NOT retried: this is an identity gate, not a transient-failure
absorber. Retries (with the token attached) belong to ``fetch_hardening``, which
runs only after identity is established. One shot, fail fast.

Signatures observed live 2026-07-21:

- Real API, unauthenticated GET ``/api/posts/?limit=1`` -> 403, ``text/plain``
  body "Permission Error: The API is only available to authenticated users."
- Real API, authenticated -> 200 JSON dict with a ``"results"`` key.
- Parked/hijacked host, same URL -> 404 empty body; ``/api2/...`` paths return
  200 with an HTML lander redirect.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests
from forecasting_tools.helpers.metaculus_client import MetaculusClient

logger = logging.getLogger(__name__)


def _api_base_url() -> str:
    """The API root the bot's own fetches will use.

    Read from ``MetaculusClient`` rather than the deprecated ``MetaculusApi`` shim, which
    on ft 0.2.92 no longer carries ``API_BASE_URL`` at all — this module was written
    against an older ft and raised ``AttributeError`` on import until repointed, and a
    guard that fails at import is a guard that isn't running. Constructing a client is
    cheap and side-effect-free apart from a missing-token warning, and it resolves
    ``METACULUS_API_BASE_URL`` exactly the way every real fetch does, so the host we vet
    is the host we will actually send the token to even under an env override.
    ``tests/test_ft_upgrade_seams.py`` pins the attribute so a future ft bump fails in CI
    instead of at prod startup.
    """
    return MetaculusClient().base_url


# The posts list is the exact endpoint the question fetch hits, so its unauthenticated
# behavior is the identity signature we want to check.
PREFLIGHT_URL = f"{_api_base_url()}/posts/?limit=1"

# The real Metaculus API gates unauthenticated access behind these statuses —
# this is its fingerprint when we send no token.
_AUTH_GATED_STATUSES = frozenset({401, 403})

# Transient edge conditions the real Metaculus front door emits under load
# (see fetch_hardening._RETRYABLE_STATUSES and the 2026-05-19 CDN-403 incident).
# We still abort — we can't confirm identity — but with a throttle-flavored
# message so the operator doesn't chase a phantom hijack.
_TRANSIENT_STATUSES = frozenset({408, 429, 502, 503, 504})

# Cap on the response body echoed into the diagnostic (keeps the log line bounded).
_BODY_PREVIEW_CHARS = 200

# Shared tail appended to hijack-flavored failure messages. Names the likely
# cause and, critically, tells the operator NOT to retry with credentials.
_DIAGNOSTIC_HINT = (
    "looks like DNS parking/hijack or a non-Metaculus host answering www.metaculus.com; "
    "do NOT retry with credentials; check `dig www.metaculus.com` and Metaculus status channels"
)


class MetaculusApiIdentityError(RuntimeError):
    """Raised when the host answering www.metaculus.com doesn't behave like the real API."""


def _parse_json_object(body: str) -> dict[str, Any] | None:
    """Parse ``body`` as JSON, returning the object as a dict or None if it isn't a JSON object."""
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def verify_metaculus_api_identity(timeout: float = 20.0) -> None:
    """Confirm www.metaculus.com is answered by the real Metaculus API before any authed call.

    Sends ONE unauthenticated GET (no token, no headers) to ``PREFLIGHT_URL``
    through an isolated ``trust_env=False`` session (so no netrc/env credential
    is attached). Passes silently on the real API's fingerprint; raises
    ``MetaculusApiIdentityError`` with a diagnostic on anything else. Never
    retries — see module docstring.
    """
    try:
        with requests.Session() as session:
            session.trust_env = False  # do not let ~/.netrc or proxy env inject credentials
            response = session.get(PREFLIGHT_URL, timeout=timeout, allow_redirects=False)
    except requests.RequestException as e:
        raise MetaculusApiIdentityError(
            f"Metaculus API identity preflight could not reach {PREFLIGHT_URL!r} "
            f"({type(e).__name__}: {e}); DNS/TLS/connect failure before any response. "
            "Do NOT retry with credentials; check `dig www.metaculus.com` and Metaculus status channels."
        ) from e

    status = response.status_code
    body_preview = response.text[:_BODY_PREVIEW_CHARS]

    if status in _AUTH_GATED_STATUSES:
        logger.info(f"Metaculus API identity preflight passed ({status=} auth-gated)")
        return

    if status == 200:
        parsed = _parse_json_object(response.text)
        if parsed is not None and "results" in parsed:
            logger.info(f"Metaculus API identity preflight passed ({status=} JSON results payload)")
            return
        raise MetaculusApiIdentityError(
            f"Metaculus API identity preflight got {status=} from {PREFLIGHT_URL!r} but the body is not the "
            f"expected JSON results payload (first {_BODY_PREVIEW_CHARS} chars: {body_preview!r}); {_DIAGNOSTIC_HINT}."
        )

    if status in _TRANSIENT_STATUSES:
        raise MetaculusApiIdentityError(
            f"Metaculus API identity preflight got {status=} from {PREFLIGHT_URL!r} "
            f"(first {_BODY_PREVIEW_CHARS} chars: {body_preview!r}); transient edge throttle/server condition — "
            "not necessarily a hijack; a later retry of the whole run is appropriate; "
            "do NOT retry with credentials now."
        )

    if 500 <= status < 600:
        raise MetaculusApiIdentityError(
            f"Metaculus API identity preflight got {status=} from {PREFLIGHT_URL!r} "
            f"(first {_BODY_PREVIEW_CHARS} chars: {body_preview!r}); cannot verify API identity. "
            "This may be a genuine Metaculus server error rather than a hijack, but the run is useless either way. "
            "Do NOT retry with credentials; check Metaculus status channels."
        )

    # Any other status — 404, a 3xx redirect (allow_redirects=False keeps it a
    # status, not a followed hop), an unexpected 2xx, or a stray 4xx.
    raise MetaculusApiIdentityError(
        f"Metaculus API identity preflight got unexpected {status=} from {PREFLIGHT_URL!r} "
        f"(first {_BODY_PREVIEW_CHARS} chars: {body_preview!r}); {_DIAGNOSTIC_HINT}."
    )
