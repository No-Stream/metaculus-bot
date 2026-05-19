"""Tests for fetch_hardening: bounded retry + timeout for question-list GETs.

Companion to publish_hardening (and its tests in test_wall_clock_abort.py).
fetch_hardening wraps ``MetaculusApi._get_questions_from_api`` so transient
403/429/5xx and connection-level errors don't kill a CI run.

Observed failure (2026-05-19): a single CDN/WAF-style 403 on
``/api/posts/?...&tournaments=summer-futureeval-2026`` — 33s stall, generic
"API only available to authenticated users" body — on a known-good token.
This wrapper would have absorbed it cleanly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests
from forecasting_tools import MetaculusApi

from metaculus_bot import fetch_hardening


def _http_error(status: int) -> requests.HTTPError:
    """Build an HTTPError with the response.status_code populated."""
    response = MagicMock(spec=requests.Response)
    response.status_code = status
    return requests.HTTPError(f"{status} Client Error", response=response)


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    """Make backoff effectively zero so the suite stays snappy."""
    monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_BACKOFF_BASE", 0.0)
    monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_BACKOFF_JITTER", 0.0)
    monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_BACKOFF_BASE", 0.0)
    monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_BACKOFF_JITTER", 0.0)


# ---------------------------------------------------------------------------
# _is_retryable: status-code policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [403, 429, 500, 502, 503, 504])
def test_is_retryable_for_retryable_statuses(status):
    """The observed CDN 403, plus 429 and 5xx, are retryable."""
    assert fetch_hardening._is_retryable(_http_error(status)) is True


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_is_retryable_skips_non_retryable_statuses(status):
    """Real auth/validation errors should not be retried."""
    assert fetch_hardening._is_retryable(_http_error(status)) is False


def test_is_retryable_for_timeout():
    assert fetch_hardening._is_retryable(requests.Timeout("slow")) is True


def test_is_retryable_for_connection_error():
    assert fetch_hardening._is_retryable(requests.ConnectionError("dns down")) is True


def test_is_retryable_for_unrelated_exception():
    """ValueError, RuntimeError, etc. shouldn't be retried."""
    assert fetch_hardening._is_retryable(ValueError("nope")) is False
    assert fetch_hardening._is_retryable(RuntimeError("nope")) is False


def test_is_retryable_handles_httperror_without_response():
    """When the chain has no response anywhere, treat as non-retryable (fail-fast)."""
    err = requests.HTTPError("no response attached")
    err.response = None  # type: ignore[assignment]
    assert fetch_hardening._is_retryable(err) is False


def test_is_retryable_walks_cause_chain_for_wrapped_httperror():
    """forecasting-tools' raise_for_status_with_additional_info wraps the original.

    Real-world pattern from `forecasting_tools/util/misc.py:32`:
        raise requests.HTTPError(error_message) from e
    The outer HTTPError has no .response; the original is in __cause__. The
    retryable check must traverse the chain or it'll mark every real Metaculus
    failure as non-retryable.
    """
    inner = _http_error(403)
    outer = requests.HTTPError("HTTPError. Url: ... Response reason: Forbidden. ...")
    outer.__cause__ = inner
    assert fetch_hardening._is_retryable(outer) is True


def test_is_retryable_walks_cause_chain_skips_non_retryable_inner():
    """If the inner cause is a non-retryable status (401), don't retry."""
    inner = _http_error(401)
    outer = requests.HTTPError("HTTPError. Url: ... Response reason: Unauthorized. ...")
    outer.__cause__ = inner
    assert fetch_hardening._is_retryable(outer) is False


def test_is_retryable_self_referencing_cause_does_not_loop():
    """Defensive: if __cause__ points back at the exception, _is_retryable terminates."""
    err = requests.HTTPError("self-ref")
    err.__cause__ = err
    # Should return False (no retryable response) without infinite-looping.
    assert fetch_hardening._is_retryable(err) is False


# ---------------------------------------------------------------------------
# _summarize_exc: log readability
# ---------------------------------------------------------------------------


def test_summarize_exc_for_direct_httperror():
    assert fetch_hardening._summarize_exc(_http_error(503)) == "HTTP 503"


def test_summarize_exc_walks_cause_chain():
    """Real-world failure: outer HTTPError has no .response, inner has the status.

    Without chain walking, the WARNING log line would say "HTTP ?" for every
    actual Metaculus 403 — defeating the debuggability point.
    """
    inner = _http_error(403)
    outer = requests.HTTPError("HTTPError. Url: ... Response reason: Forbidden. ...")
    outer.__cause__ = inner
    assert fetch_hardening._summarize_exc(outer) == "HTTP 403"


def test_summarize_exc_for_timeout():
    assert fetch_hardening._summarize_exc(requests.Timeout("slow")) == "Timeout"


def test_summarize_exc_for_connection_error():
    assert fetch_hardening._summarize_exc(requests.ConnectionError("dns")) == "ConnectionError"


def test_summarize_exc_self_referencing_cause_does_not_loop():
    err = requests.HTTPError("self-ref")
    err.__cause__ = err
    # Falls through to type-name fallback without infinite-looping.
    assert fetch_hardening._summarize_exc(err) == "HTTPError"


# ---------------------------------------------------------------------------
# _wrap_with_retry: behavior under transient failures
# ---------------------------------------------------------------------------


def test_wrapper_retries_on_403_and_succeeds(monkeypatch, caplog):
    """The 2026-05-19 failure mode: 403, then success. Wrapper recovers."""
    monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_RETRIES", 2)
    monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_RETRIES", 2)

    n_calls = {"n": 0}

    def fake_get(*args, **kwargs):
        n_calls["n"] += 1
        if n_calls["n"] == 1:
            raise _http_error(403)
        return ["question-1", "question-2"]

    wrapped = fetch_hardening._wrap_with_retry("fake", fake_get)
    with caplog.at_level("WARNING"):
        result = wrapped("dummy")

    assert result == ["question-1", "question-2"]
    assert n_calls["n"] == 2
    assert any("FETCH_HARDENING" in r.message and "HTTP 403" in r.message for r in caplog.records)


def test_wrapper_logs_status_for_chained_httperror(monkeypatch, caplog):
    """Production failure mode: forecasting-tools wraps the original HTTPError.

    The outer exception (raised by raise_for_status_with_additional_info) has
    no .response; the original is in __cause__. Wrapper must log "HTTP 403",
    not "HTTP ?", or operators can't tell which retry policy fired.
    """
    monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_RETRIES", 2)
    monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_RETRIES", 2)

    n_calls = {"n": 0}

    def fake_get(*args, **kwargs):
        n_calls["n"] += 1
        if n_calls["n"] == 1:
            inner = _http_error(503)
            outer = requests.HTTPError("HTTPError. Url: ... Response reason: Service Unavailable. ...")
            raise outer from inner
        return []

    wrapped = fetch_hardening._wrap_with_retry("fake", fake_get)
    with caplog.at_level("WARNING"):
        wrapped("dummy")

    assert n_calls["n"] == 2
    assert any("HTTP 503" in r.message for r in caplog.records), "Status code should be extracted from cause chain"


def test_wrapper_retries_on_timeout_and_succeeds(monkeypatch):
    monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_RETRIES", 2)
    monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_RETRIES", 2)

    n_calls = {"n": 0}

    def fake_get(*args, **kwargs):
        n_calls["n"] += 1
        if n_calls["n"] == 1:
            raise requests.Timeout("slow")
        return []

    wrapped = fetch_hardening._wrap_with_retry("fake", fake_get)
    assert wrapped("dummy") == []
    assert n_calls["n"] == 2


def test_wrapper_retries_on_connection_error_and_succeeds(monkeypatch):
    monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_RETRIES", 2)
    monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_RETRIES", 2)

    n_calls = {"n": 0}

    def fake_get(*args, **kwargs):
        n_calls["n"] += 1
        if n_calls["n"] == 1:
            raise requests.ConnectionError("dns")
        return []

    wrapped = fetch_hardening._wrap_with_retry("fake", fake_get)
    assert wrapped("dummy") == []
    assert n_calls["n"] == 2


def test_wrapper_does_not_retry_on_401(monkeypatch):
    """A real auth failure must surface immediately, not wait through retries."""
    monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_RETRIES", 2)
    monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_RETRIES", 2)

    n_calls = {"n": 0}

    def fake_get(*args, **kwargs):
        n_calls["n"] += 1
        raise _http_error(401)

    wrapped = fetch_hardening._wrap_with_retry("fake", fake_get)
    with pytest.raises(requests.HTTPError):
        wrapped("dummy")
    assert n_calls["n"] == 1  # no retry


def test_wrapper_does_not_retry_on_404(monkeypatch):
    """Resource-not-found is permanent; retrying just burns time."""
    monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_RETRIES", 2)
    monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_RETRIES", 2)

    n_calls = {"n": 0}

    def fake_get(*args, **kwargs):
        n_calls["n"] += 1
        raise _http_error(404)

    wrapped = fetch_hardening._wrap_with_retry("fake", fake_get)
    with pytest.raises(requests.HTTPError):
        wrapped("dummy")
    assert n_calls["n"] == 1


def test_wrapper_does_not_retry_on_unrelated_exception(monkeypatch):
    """ValueError from somewhere deep shouldn't get retried."""
    monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_RETRIES", 2)
    monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_RETRIES", 2)

    n_calls = {"n": 0}

    def fake_get(*args, **kwargs):
        n_calls["n"] += 1
        raise ValueError("some other bug")

    wrapped = fetch_hardening._wrap_with_retry("fake", fake_get)
    with pytest.raises(ValueError):
        wrapped("dummy")
    assert n_calls["n"] == 1


def test_wrapper_gives_up_after_retries_exhausted(monkeypatch):
    """Three consecutive 503s: raises the last HTTPError after FETCH_GET_RETRIES + 1 attempts."""
    monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_RETRIES", 2)
    monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_RETRIES", 2)

    n_calls = {"n": 0}

    def fake_get(*args, **kwargs):
        n_calls["n"] += 1
        raise _http_error(503)

    wrapped = fetch_hardening._wrap_with_retry("fake", fake_get)
    with pytest.raises(requests.HTTPError):
        wrapped("dummy")
    assert n_calls["n"] == 3  # initial + 2 retries


def test_wrapper_succeeds_first_try_no_retries(monkeypatch):
    """Happy path: no warnings, single call."""
    monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_RETRIES", 2)
    monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_RETRIES", 2)

    n_calls = {"n": 0}

    def fake_get(*args, **kwargs):
        n_calls["n"] += 1
        return ["q"]

    wrapped = fetch_hardening._wrap_with_retry("fake", fake_get)
    assert wrapped("dummy") == ["q"]
    assert n_calls["n"] == 1


def test_wrapper_zero_retries_fails_fast(monkeypatch):
    """FETCH_GET_RETRIES=0 means a single attempt; failure surfaces immediately."""
    monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_RETRIES", 0)
    monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_RETRIES", 0)

    n_calls = {"n": 0}

    def fake_get(*args, **kwargs):
        n_calls["n"] += 1
        raise _http_error(503)

    wrapped = fetch_hardening._wrap_with_retry("fake", fake_get)
    with pytest.raises(requests.HTTPError):
        wrapped("dummy")
    assert n_calls["n"] == 1


def test_wrapper_passes_args_and_kwargs_through():
    """Wrapper must forward (*args, **kwargs) faithfully."""
    captured = {}

    def fake_get(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "ok"

    wrapped = fetch_hardening._wrap_with_retry("fake", fake_get)
    result = wrapped("a", "b", group_question_mode="single", extra=42)

    assert result == "ok"
    assert captured["args"] == ("a", "b")
    assert captured["kwargs"] == {"group_question_mode": "single", "extra": 42}


# ---------------------------------------------------------------------------
# Global socket-timeout install
# ---------------------------------------------------------------------------


def test_install_get_timeout_default_adds_timeout_when_caller_omits(monkeypatch):
    """Bare `requests.get(url, params=...)` should pick up the installed default."""
    from forecasting_tools.helpers import metaculus_api as ft_api

    captured = {}
    inner_calls = {"n": 0}

    def fake_inner_get(*args, **kwargs):
        inner_calls["n"] += 1
        captured.update(kwargs)
        return MagicMock()

    # Stage the inner ``requests.get`` BEFORE installing the wrapper so the
    # wrapper closes over our fake, not the real network-call. monkeypatch
    # restores ``ft_api.requests.get`` to the real one after the test.
    monkeypatch.setattr(ft_api.requests, "get", fake_inner_get)

    fetch_hardening._install_get_timeout_default(7.5)

    ft_api.requests.get("http://example.com", params={"x": 1})

    assert inner_calls["n"] == 1
    assert captured.get("timeout") == 7.5


def test_install_get_timeout_default_respects_caller_supplied_timeout(monkeypatch):
    """If the caller already passed timeout=, don't clobber it."""
    from forecasting_tools.helpers import metaculus_api as ft_api

    captured = {}

    def fake_inner_get(*args, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(ft_api.requests, "get", fake_inner_get)

    fetch_hardening._install_get_timeout_default(7.5)

    ft_api.requests.get("http://example.com", timeout=2.0)

    assert captured.get("timeout") == 2.0


# ---------------------------------------------------------------------------
# apply_fetch_hardening: idempotency, calling-convention preservation
# ---------------------------------------------------------------------------


def _reset_fetch_hardening_state(monkeypatch):
    """Snapshot + restore patched methods, sentinel, and the global requests.get wrapper.

    apply_fetch_hardening permanently swaps ``ft_api.requests.get`` for a
    timeout-injecting wrapper. monkeypatch.setattr restores the original
    bound view after the test, preventing leakage across tests.
    """
    from forecasting_tools.helpers import metaculus_api as ft_api

    for name in fetch_hardening._PATCHED_METHODS:
        monkeypatch.setattr(MetaculusApi, name, MetaculusApi.__dict__[name])

    monkeypatch.setattr(ft_api.requests, "get", ft_api.requests.get)

    if hasattr(MetaculusApi, fetch_hardening._SENTINEL):
        monkeypatch.setattr(MetaculusApi, fetch_hardening._SENTINEL, False)
        delattr(MetaculusApi, fetch_hardening._SENTINEL)
    else:
        monkeypatch.setattr(MetaculusApi, fetch_hardening._SENTINEL, False, raising=False)
        delattr(MetaculusApi, fetch_hardening._SENTINEL)


def test_apply_fetch_hardening_idempotent(monkeypatch):
    _reset_fetch_hardening_state(monkeypatch)

    fetch_hardening.apply_fetch_hardening()
    after_first = {name: MetaculusApi.__dict__[name].__func__ for name in fetch_hardening._PATCHED_METHODS}

    fetch_hardening.apply_fetch_hardening()
    after_second = {name: MetaculusApi.__dict__[name].__func__ for name in fetch_hardening._PATCHED_METHODS}

    for name in fetch_hardening._PATCHED_METHODS:
        assert after_first[name] is after_second[name]


def test_apply_fetch_hardening_supports_class_and_instance_calls(monkeypatch):
    """Wrapped method must be callable as both ClassName.foo() and ClassName().foo().

    Same regression class as F18 in publish_hardening: setattr(...) of a plain
    function would break instance-level calls (self gets bound as cls).
    fetch_hardening re-wraps as classmethod, same as publish_hardening.
    """
    _reset_fetch_hardening_state(monkeypatch)

    fetch_hardening.apply_fetch_hardening()

    descriptor = MetaculusApi.__dict__[fetch_hardening._PATCHED_METHODS[0]]
    assert isinstance(descriptor, classmethod)


def test_apply_fetch_hardening_logs_summary(monkeypatch, caplog):
    _reset_fetch_hardening_state(monkeypatch)

    with caplog.at_level("INFO"):
        fetch_hardening.apply_fetch_hardening()

    assert any("Fetch hardening applied" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# End-to-end: patched MetaculusApi survives a 403 followed by success
# ---------------------------------------------------------------------------


def test_patched_get_questions_from_api_recovers_from_403(monkeypatch):
    """Simulate the 2026-05-19 incident: first call 403, second succeeds.

    Patches ``requests.get`` on the forecasting-tools module so the real
    `_get_questions_from_api` runs end-to-end through the hardened wrapper.
    """
    from forecasting_tools.helpers import metaculus_api as ft_api

    _reset_fetch_hardening_state(monkeypatch)
    monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_RETRIES", 2)
    monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_RETRIES", 2)
    # Make sleep between random.uniform(2, 3) inside _get_questions_from_api a no-op
    # so the test stays fast.
    monkeypatch.setattr("forecasting_tools.helpers.metaculus_api.time.sleep", lambda *_: None)
    # Token check inside _get_auth_headers reads os.getenv at call time.
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")

    n_calls = {"n": 0}
    captured_kwargs: list[dict] = []

    def fake_get(*args, **kwargs):
        n_calls["n"] += 1
        captured_kwargs.append(dict(kwargs))
        # Don't use spec= here: raise_for_status_with_additional_info accesses
        # response.url / .reason / .text which we want auto-generated by MagicMock.
        response = MagicMock()
        if n_calls["n"] == 1:
            response.status_code = 403
            response.url = "https://www.metaculus.com/api/posts/?dummy"
            response.reason = "Forbidden"
            response.text = "Permission Error: ..."
            response.json.return_value = None
            response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden", response=response)
        else:
            response.status_code = 200
            response.content = b'{"results": []}'
            response.raise_for_status.return_value = None
        return response

    # Stage the inner ``requests.get`` BEFORE apply_fetch_hardening so the
    # global GET-timeout wrapper closes over our fake. This mirrors how the
    # real installation would close over forecasting-tools' real requests.get.
    monkeypatch.setattr(ft_api.requests, "get", fake_get)

    fetch_hardening.apply_fetch_hardening()

    # Drive the real wrapped method. GroupQuestionMode is a Literal type
    # ("exclude" | "unpack_subquestions"), not an enum.
    result = MetaculusApi._get_questions_from_api({"limit": 100, "offset": 0}, "exclude")
    assert result == []
    assert n_calls["n"] == 2  # 403 then success
    # The global GET-timeout wrapper should have injected timeout= on every call.
    for kwargs in captured_kwargs:
        assert kwargs.get("timeout") is not None, "GET-timeout wrapper should inject default timeout"
