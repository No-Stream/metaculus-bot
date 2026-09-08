"""Tests for api_preflight: unauthenticated identity check before we send the token.

Motivated by the 2026-07-21 DNS-parking incident — metaculus.com's DNS was
repointed at a GoDaddy parking host while the bot's scheduled runs kept sending
``Authorization: Token $METACULUS_TOKEN`` to the unknown host. The preflight
makes ONE unauthenticated GET and aborts with a diagnostic if the host doesn't
behave like the real API, so the token is never leaked to a hijacked host. Since
the Mantic season it is generic over the API base URL: ``verify_api_identity``
takes the base, and ``verify_metaculus_api_identity`` is the Metaculus wrapper.

Mocking strategy: patch the HTTP transport (``HTTPAdapter.send``), NOT
``requests.get`` wholesale. The real ``requests.Session`` (with
``trust_env=False``) runs end-to-end — including ``prepare_request``'s netrc
logic — so the credential-leak regression is actually observable here. The
previous wholesale-``get`` patch is exactly why the netrc leak was invisible.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pytest
import requests
from requests.adapters import HTTPAdapter

from metaculus_bot import api_preflight, cli
from metaculus_bot.api_preflight import (
    ApiIdentityError,
    MetaculusApiIdentityError,
    verify_api_identity,
    verify_metaculus_api_identity,
)
from metaculus_bot.constants import MANTIC_API_BASE_URL
from metaculus_bot.performance_analysis import cli as perf_cli

# The live Mantic fingerprint (probed 2026-09-08): the posts list is public and answers the
# same JSON shape an authenticated Metaculus read does.
_MANTIC_POSTS_BODY = '{"next": null, "previous": null, "results": [{"id": 650}]}'
_PARKED_LANDER_HTML = '<html><head><script>window.location.href="/lander"</script></head></html>'


@contextmanager
def _mock_transport(
    status: int | None = None,
    body: str = "",
    exc: BaseException | None = None,
) -> Iterator[dict]:
    """Stub the HTTP transport so the preflight runs against a fake response.

    Only ``HTTPAdapter.send`` is replaced; the real Session (trust_env=False)
    still prepares the request, so ``captured["request"]`` reflects exactly what
    would go on the wire (headers included). Yields a dict populated with the
    prepared request, the send kwargs, and the send count.
    """
    captured: dict = {"send_count": 0}

    def fake_send(self: HTTPAdapter, request: requests.PreparedRequest, **kwargs: object) -> requests.Response:
        captured["request"] = request
        captured["kwargs"] = kwargs
        captured["send_count"] += 1
        if exc is not None:
            raise exc
        assert status is not None  # a non-exception transport stub must supply a status
        response = requests.Response()
        response.status_code = status
        response._content = body.encode()
        response.encoding = "utf-8"
        response.url = request.url or api_preflight.preflight_url()
        response.request = request
        return response

    with patch.object(HTTPAdapter, "send", fake_send):
        yield captured


class TestPassesOnRealApiSignature:
    """The preflight must accept the real Metaculus API's fingerprints without raising."""

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_gated_status_passes(self, status: int) -> None:
        with _mock_transport(status=status, body="Permission Error: ..."):
            verify_metaculus_api_identity()  # no raise

    def test_authenticated_json_results_passes(self) -> None:
        with _mock_transport(status=200, body='{"results": []}'):
            verify_metaculus_api_identity()  # no raise

    def test_success_logs_info_line(self, caplog: pytest.LogCaptureFixture) -> None:
        with _mock_transport(status=403, body="Permission Error"), caplog.at_level("INFO"):
            verify_metaculus_api_identity()
        assert any("preflight passed" in r.message for r in caplog.records)


class TestRaisesOnImposterHost:
    """A host that doesn't behave like the real API must fail fast with a diagnostic."""

    def test_404_empty_body_raises_and_names_status(self) -> None:
        with _mock_transport(status=404, body=""), pytest.raises(ApiIdentityError, match="404"):
            verify_metaculus_api_identity()

    def test_200_html_lander_raises(self) -> None:
        with _mock_transport(status=200, body=_PARKED_LANDER_HTML), pytest.raises(ApiIdentityError):
            verify_metaculus_api_identity()

    def test_diagnostic_names_the_vetted_metaculus_host(self) -> None:
        """The hijack hint tells the operator which host to `dig`: the one the base URL resolves
        to, read at call time exactly as the wrapper vets it."""
        host = urlparse(api_preflight.preflight_url()).hostname
        assert host
        with _mock_transport(status=404, body=""), pytest.raises(ApiIdentityError) as excinfo:
            verify_metaculus_api_identity()
        assert f"dig {host}" in str(excinfo.value)

    def test_200_json_without_results_raises(self) -> None:
        with _mock_transport(status=200, body='{"detail": "nope"}'), pytest.raises(ApiIdentityError):
            verify_metaculus_api_identity()

    def test_302_redirect_raises_and_is_not_followed(self) -> None:
        with (
            _mock_transport(status=302, body="") as captured,
            pytest.raises(ApiIdentityError, match="302"),
        ):
            verify_metaculus_api_identity()
        # allow_redirects=False: the lander redirect must stay visible as a 3xx.
        assert captured["send_count"] == 1

    def test_500_raises_with_server_error_flavor(self) -> None:
        with (
            _mock_transport(status=500, body="Internal Server Error"),
            pytest.raises(ApiIdentityError, match="server error"),
        ):
            verify_metaculus_api_identity()

    def test_connection_error_raises_chained(self) -> None:
        original = requests.ConnectionError("dns down")
        with _mock_transport(exc=original), pytest.raises(ApiIdentityError) as excinfo:
            verify_metaculus_api_identity()
        assert excinfo.value.__cause__ is original

    def test_timeout_raises_chained(self) -> None:
        original = requests.Timeout("slow")
        with _mock_transport(exc=original), pytest.raises(ApiIdentityError) as excinfo:
            verify_metaculus_api_identity()
        assert excinfo.value.__cause__ is original


class TestTransientEdgeStatuses:
    """Transient front-door conditions get a throttle-flavored message, not the hijack hint."""

    @pytest.mark.parametrize("status", [408, 429, 502, 503, 504])
    def test_transient_status_raises_without_hijack_hint(self, status: int) -> None:
        with (
            _mock_transport(status=status, body="Too Many Requests"),
            pytest.raises(ApiIdentityError, match="transient") as excinfo,
        ):
            verify_metaculus_api_identity()
        message = str(excinfo.value)
        # Must not carry the DNS-parking/hijack diagnostic (its distinctive tokens).
        assert "parking" not in message
        assert "dig " not in message
        assert "do NOT retry with credentials" in message


class TestNeverSendsCredentials:
    """The whole point: the preflight request must carry no auth of any kind."""

    def test_no_authorization_header_even_with_netrc(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A generic ``default`` netrc entry must NOT attach Basic auth.

        This is the F1 regression: with the default trust_env=True, requests'
        prepare_request would call get_netrc_auth and inject Authorization. The
        isolated trust_env=False session must suppress it.
        """
        netrc_file = tmp_path / "netrc"
        netrc_file.write_text("default login leaked_user password leaked_pass\n")
        netrc_file.chmod(0o600)
        monkeypatch.setenv("NETRC", str(netrc_file))

        with _mock_transport(status=403, body="Permission Error") as captured:
            verify_metaculus_api_identity()

        request = captured["request"]
        assert "Authorization" not in request.headers
        assert request.method == "GET"
        assert request.url == api_preflight.preflight_url()

    def test_passes_timeout_to_transport(self) -> None:
        with _mock_transport(status=403) as captured:
            verify_metaculus_api_identity(timeout=5.0)
        assert captured["kwargs"].get("timeout") == 5.0


class TestVerifyApiIdentityAgainstMantic:
    """The generic check, called the way the mantic run mode calls it: with Mantic's API base.

    A Mantic run must not depend on Metaculus DNS health, so the wrapper is NOT involved here;
    only the given base URL is vetted, and the diagnostics name its host.
    """

    def test_public_json_results_passes_and_probes_the_mantic_posts_list(self) -> None:
        with _mock_transport(status=200, body=_MANTIC_POSTS_BODY) as captured:
            verify_api_identity(MANTIC_API_BASE_URL)  # no raise
        assert captured["request"].url == f"{MANTIC_API_BASE_URL}/posts/?limit=1"
        assert captured["send_count"] == 1

    def test_sends_no_credentials_even_with_netrc(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        netrc_file = tmp_path / "netrc"
        netrc_file.write_text("default login leaked_user password leaked_pass\n")
        netrc_file.chmod(0o600)
        monkeypatch.setenv("NETRC", str(netrc_file))

        with _mock_transport(status=200, body=_MANTIC_POSTS_BODY) as captured:
            verify_api_identity(MANTIC_API_BASE_URL)

        assert "Authorization" not in captured["request"].headers
        assert captured["request"].method == "GET"

    def test_parked_host_raises_naming_the_mantic_host_not_metaculus(self) -> None:
        with _mock_transport(status=404, body=_PARKED_LANDER_HTML), pytest.raises(ApiIdentityError) as excinfo:
            verify_api_identity(MANTIC_API_BASE_URL)
        message = str(excinfo.value)
        assert "404" in message
        assert "dig competitions.mantic.com" in message
        assert "metaculus" not in message.lower()

    def test_connect_failure_names_the_mantic_host(self) -> None:
        original = requests.ConnectionError("dns down")
        with _mock_transport(exc=original), pytest.raises(ApiIdentityError) as excinfo:
            verify_api_identity(MANTIC_API_BASE_URL)
        assert excinfo.value.__cause__ is original
        assert "dig competitions.mantic.com" in str(excinfo.value)

    def test_200_without_results_raises(self) -> None:
        with _mock_transport(status=200, body='{"detail": "nope"}'), pytest.raises(ApiIdentityError):
            verify_api_identity(MANTIC_API_BASE_URL)

    def test_passes_timeout_to_transport(self) -> None:
        with _mock_transport(status=200, body=_MANTIC_POSTS_BODY) as captured:
            verify_api_identity(MANTIC_API_BASE_URL, timeout=5.0)
        assert captured["kwargs"].get("timeout") == 5.0


class TestExceptionName:
    def test_the_metaculus_era_name_is_the_same_class(self) -> None:
        """One class, two names: callers written when the module vetted only Metaculus keep
        catching the same exception the Mantic path raises."""
        assert MetaculusApiIdentityError is ApiIdentityError


class TestEntryPointWiring:
    """Guard against the import being silently stripped from the entry points (formatter footgun)."""

    def test_cli_imports_preflight(self) -> None:
        assert cli.verify_metaculus_api_identity is verify_metaculus_api_identity

    def test_performance_analysis_cli_imports_preflight(self) -> None:
        assert perf_cli.verify_metaculus_api_identity is verify_metaculus_api_identity


class TestPerformanceCliInvokesPreflight:
    """Invocation/ordering: the preflight actually gates the live pull, and only the live pull."""

    def test_cached_path_does_not_preflight(self) -> None:
        """--cached is a disk read; it must NOT hit the network via the preflight."""
        with (
            patch.object(perf_cli, "verify_metaculus_api_identity") as verify,
            patch.object(perf_cli, "load_dataset", return_value={}),
            patch.object(perf_cli, "generate_report", return_value=""),
        ):
            perf_cli.main(["--cached", "x.json"])
        verify.assert_not_called()

    def test_live_pull_preflights_before_fetch(self) -> None:
        """Live pull must call verify BEFORE build_performance_dataset."""
        manager = MagicMock()
        with (
            patch.object(perf_cli, "verify_metaculus_api_identity") as verify,
            patch.object(perf_cli, "build_performance_dataset", return_value={}) as build,
            patch.object(perf_cli, "save_dataset"),
            patch.object(perf_cli, "generate_report", return_value=""),
        ):
            manager.attach_mock(verify, "verify")
            manager.attach_mock(build, "build")
            perf_cli.main([])

        verify.assert_called_once()
        call_names = [name for name, _, _ in manager.mock_calls]
        assert call_names.index("verify") < call_names.index("build")

    def test_preflight_failure_aborts_before_fetch(self) -> None:
        """If the preflight raises, the token-sending fetch must never run."""
        with (
            patch.object(
                perf_cli,
                "verify_metaculus_api_identity",
                side_effect=ApiIdentityError("hijacked"),
            ),
            patch.object(perf_cli, "build_performance_dataset") as build,
            patch.object(perf_cli, "save_dataset"),
            patch.object(perf_cli, "generate_report", return_value=""),
            pytest.raises(ApiIdentityError),
        ):
            perf_cli.main([])
        build.assert_not_called()
