"""Behavior pins for ``metaculus_bot.http_status``'s shared transient-fetch predicate.

``is_transient_question_fetch_error`` is the BENCHMARK/BACKTEST outer-loop retry
policy shared by ``community_benchmark._fetch_type_with_retries`` and
``backtest.question_prep._fetch_with_retries``. The pins here are the two
directions the retired per-file substring scan misfired on the exception shape
forecasting-tools actually raises (a message-only HTTPError interpolating the
URL, question ids, and the echoed response body): a bare transient-looking digit
in the message must carry no signal, while the anchored ``Status code: NNN``
phrase must.
"""

from __future__ import annotations

import http.client

import pytest
from requests import exceptions as req_exc
from urllib3 import exceptions as ul3_exc

from metaculus_bot.http_status import is_transient_question_fetch_error


class TestStatusKeyedClassification:
    """With a readable status, the status alone decides — message text is ignored."""

    def test_ft_shaped_503_is_retryable(self):
        exc = req_exc.HTTPError(
            "HTTPError. Url: https://www.metaculus.com/api/posts/. Status code: 503. "
            "Response reason: Service Unavailable. Response text: upstream connect error"
        )
        assert is_transient_question_fetch_error(exc) is True

    def test_bare_502_without_status_phrase_is_not_retryable(self):
        # The digits come from a question id and a URL offset, not a status. The old
        # substring scan retried this; the anchored read must not.
        exc = RuntimeError("question 502 failed while paging from offset 5020")
        assert is_transient_question_fetch_error(exc) is False

    def test_500_is_retryable(self):
        # The retired token list omitted 500 even though it is the canonical transient
        # status; the shared predicate treats it like its 5xx siblings.
        exc = req_exc.HTTPError("HTTPError. Url: https://www.metaculus.com/api/posts/. Status code: 500.")
        assert is_transient_question_fetch_error(exc) is True

    def test_403_is_not_retryable(self):
        # No 403 carve-out on this policy — that belongs to the hardened prod fetch
        # path (fetch_hardening), and a readable status suppresses the text fallback
        # even when the body mentions a timeout.
        exc = req_exc.HTTPError("HTTPError. Status code: 403. Response text: request timeout while authorizing")
        assert is_transient_question_fetch_error(exc) is False

    def test_status_code_read_off_a_real_response_object(self):
        class _Response:
            status_code = 429

        exc = req_exc.HTTPError("throttled")
        exc.response = _Response()  # type: ignore[assignment]
        assert is_transient_question_fetch_error(exc) is True


class TestTransportErrors:
    """Transport-level failures are retryable wherever they sit in the cause chain."""

    @pytest.mark.parametrize(
        "transport_exc",
        [
            req_exc.ConnectionError("connection reset"),
            req_exc.Timeout("read timed out"),
            ul3_exc.ProtocolError("connection aborted"),
            http.client.RemoteDisconnected("remote end closed connection"),
        ],
        ids=["connection", "timeout", "protocol", "remote_disconnected"],
    )
    def test_direct_transport_error_is_retryable(self, transport_exc: BaseException):
        assert is_transient_question_fetch_error(transport_exc) is True

    def test_transport_error_behind_a_wrapper_is_retryable(self):
        # ft re-raises a fresh message-only exception chained via ``raise ... from e``;
        # the isinstance walk must see through that wrapper.
        wrapper = RuntimeError("fetch failed")
        wrapper.__cause__ = req_exc.Timeout("read timed out")
        assert is_transient_question_fetch_error(wrapper) is True


class TestStatuslessTextFallback:
    """Only a fully statusless exception falls back to the narrow text check."""

    @pytest.mark.parametrize("message", ["429 too many requests", "request timed out", "socket timeout"])
    def test_throttle_and_timeout_wording_is_retryable(self, message: str):
        assert is_transient_question_fetch_error(RuntimeError(message)) is True

    def test_unrelated_statusless_error_is_not_retryable(self):
        assert is_transient_question_fetch_error(ValueError("totally different error format")) is False
