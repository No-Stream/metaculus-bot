"""Hand-built ``requests.Response`` objects for tests that stub the HTTP transport.

``requests`` has no public constructor for a Response with a body, so a fake has to poke the
private ``_content`` attribute. That poke lives here once, and the tests that answer a
monkeypatched ``requests.get`` or ``HTTPAdapter.send`` (the supply probe, the Mantic client, the
identity preflight, the Mantic end-to-end harness) all build their responses through these two
functions, so a ``requests`` change is chased through one file.

Not named ``test_*`` on purpose: pytest imports it without collecting it. Nothing here opens a socket.
"""

from __future__ import annotations

import json
from typing import Any

import requests


def text_response(
    text: str, *, status: int = 200, url: str = "", request: requests.PreparedRequest | None = None
) -> requests.Response:
    """A Response carrying ``text`` as a UTF-8 body. ``url`` defaults to the request's URL when one is given."""
    response = requests.Response()
    response.status_code = status
    response.encoding = "utf-8"
    response._content = text.encode()
    if request is not None:
        response.request = request
        url = url or request.url or ""
    response.url = url
    return response


def json_response(
    payload: Any, *, status: int = 200, url: str = "", request: requests.PreparedRequest | None = None
) -> requests.Response:
    """A Response whose body is ``payload`` serialized as JSON."""
    return text_response(json.dumps(payload), status=status, url=url, request=request)
