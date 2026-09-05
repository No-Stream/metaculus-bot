"""The transport half of the fetch marker's `failure_class` vocabulary.

``_network_failure_class`` buckets the exception a GET died with into one of six tokens. The
isinstance order is load-bearing: against the installed aiohttp the TLS and DNS connector errors
all subclass ``ClientConnectorError``, so a reorder collapses ``tls`` and ``dns`` into
``connection`` with nothing else failing, and telling a host that refused our TLS from one our
egress IP could not resolve is the measurement the escalation ladder's case rests on. One
hand-written expected token per bucket, so a reorder fails here.
"""

from __future__ import annotations

import ssl

import aiohttp
import pytest
from aiohttp.client_reqrep import ConnectionKey, RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from metaculus_bot.research import resolution_source
from metaculus_bot.research.resolution_source import _network_failure_class, resolution_source_provider
from tests.resolution_source_fakes import FakeSession, _mock_question

_KEY = ConnectionKey("host.example.com", 443, True, True, None, None, None)
_REQUEST = RequestInfo(URL("https://host.example.com/report"), "GET", CIMultiDictProxy(CIMultiDict()))

# What aiohttp's parser says for the trueup.io shape: a `Content-Encoding` it has no decoder for.
# `http_parser.DeflateBuffer` raises `ContentEncodingError`, and `ClientResponse.start` re-raises
# every `HttpProcessingError` as `ClientResponseError(status=<code>)`.
_UNDECODABLE_ENCODING = "Can not decode content-encoding: zstandard (zstd). Please install `backports.zstd`"


def _content_encoding_failure() -> aiohttp.ClientResponseError:
    return aiohttp.ClientResponseError(_REQUEST, (), status=400, message=_UNDECODABLE_ENCODING)


class TestNetworkFailureClass:
    def test_the_specific_connector_errors_are_subclasses_of_the_general_one(self):
        """The premise of the ordering, pinned against the installed aiohttp."""
        for specific in (
            aiohttp.ClientConnectorCertificateError,
            aiohttp.ClientConnectorSSLError,
            aiohttp.ClientConnectorDNSError,
        ):
            assert issubclass(specific, aiohttp.ClientConnectorError)
        # The parser's re-raise is a SIBLING of the payload error, which is why `decode` cannot
        # claim it and why `connection` used to.
        assert not issubclass(aiohttp.ClientResponseError, aiohttp.ClientPayloadError)
        assert not issubclass(aiohttp.ClientResponseError, aiohttp.ClientConnectionError)

    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            pytest.param(aiohttp.ServerTimeoutError("read timed out"), "timeout", id="server_timeout"),
            pytest.param(TimeoutError(), "timeout", id="asyncio_timeout"),
            pytest.param(
                aiohttp.ClientConnectorCertificateError(_KEY, ssl.SSLCertVerificationError("self-signed")),
                "tls",
                id="certificate",
            ),
            pytest.param(aiohttp.ClientConnectorSSLError(_KEY, ssl.SSLError("handshake failure")), "tls", id="ssl"),
            pytest.param(aiohttp.ClientConnectorDNSError(_KEY, OSError("Name or service not known")), "dns", id="dns"),
            pytest.param(
                aiohttp.ClientConnectorError(_KEY, OSError("Connection refused")), "connection", id="connector"
            ),
            pytest.param(aiohttp.ServerDisconnectedError(), "connection", id="server_disconnected"),
            pytest.param(aiohttp.ClientPayloadError("Response payload is not completed"), "decode", id="payload"),
            pytest.param(_content_encoding_failure(), "malformed_response", id="undecodable_content_encoding"),
            pytest.param(
                aiohttp.ClientResponseError(
                    _REQUEST, (), status=400, message="Got more than 8190 bytes when reading Header value is too long."
                ),
                "malformed_response",
                id="oversized_header",
            ),
        ],
    )
    def test_each_bucket_on_a_known_input(self, exc: BaseException, expected: str):
        assert _network_failure_class(exc) == expected

    def test_the_undecodable_encoding_is_not_a_connection_fault(self):
        """The one decode failure this bundle measured (trueup.io, zstd) recorded as `connection`,
        the bucket that reads as the host never answering, when the host answered in full."""
        assert _network_failure_class(_content_encoding_failure()) != "connection"

    async def test_the_marker_line_carries_the_new_token_end_to_end(self, monkeypatch, caplog):
        """The emitter and the parser are pinned against the same string: `malformed_response`
        rides the FETCH line's `failure_class` and `exc` names the aiohttp class."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession({"https://host.example.com/report": _content_encoding_failure()})
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria="See https://host.example.com/report")

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == [
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://host.example.com/report "
            "status=error http=n/a embeds=none failure_class=malformed_response exc=ClientResponseError"
        ]
