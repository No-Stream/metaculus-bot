"""Tests for metaculus_bot.research_providers helpers."""

from __future__ import annotations

import asyncio

import pytest

from metaculus_bot.research_providers import is_asknews_subscription_error


class _FakeForbiddenError(Exception):
    """Stand-in for ``asknews_sdk.errors.ForbiddenError`` — matched by class name."""


# Rename the class attribute so ``type(exc).__name__`` matches the SDK's real
# ForbiddenError class name (the predicate matches on class name substring).
_FakeForbiddenError.__name__ = "ForbiddenError"


class TestIsAsknewsSubscriptionError:
    """is_asknews_subscription_error matches only when BOTH class name and
    message signature match. Narrow matching is load-bearing: we don't want
    generic 403s from unrelated providers to be silenced as off-season noise.
    """

    def test_forbidden_error_with_subscription_code_matches(self) -> None:
        exc = _FakeForbiddenError("403011 - subscription is not currently active")
        assert is_asknews_subscription_error(exc) is True

    def test_forbidden_error_with_subscription_phrase_matches(self) -> None:
        # Alternate message wording — the check accepts either 403011 or the phrase.
        exc = _FakeForbiddenError("subscription is not currently active on this tier")
        assert is_asknews_subscription_error(exc) is True

    def test_forbidden_error_with_unrelated_message_does_not_match(self) -> None:
        # Class name matches but message doesn't: not the subscription-inactive
        # signature. Don't silence — could be a real permission issue.
        exc = _FakeForbiddenError("403000 - rate limit hit")
        assert is_asknews_subscription_error(exc) is False

    def test_generic_runtime_error_with_403_forbidden_does_not_match(self) -> None:
        # Class name doesn't match — must not silence third-party 403s.
        exc = RuntimeError("403 Forbidden")
        assert is_asknews_subscription_error(exc) is False

    def test_permission_error_does_not_match(self) -> None:
        # Built-in PermissionError has "forbidden path" wording but isn't the
        # SDK class name.
        exc = PermissionError("forbidden path /tmp")
        assert is_asknews_subscription_error(exc) is False

    def test_asyncio_timeout_error_does_not_match(self) -> None:
        # Timeout is an operational failure we DO want to alert on, not silence.
        exc = asyncio.TimeoutError()
        assert is_asknews_subscription_error(exc) is False


@pytest.mark.parametrize(
    "exc,expected",
    [
        (_FakeForbiddenError("403011 - subscription is not currently active"), True),
        (_FakeForbiddenError("some other error"), False),
        (RuntimeError("403 Forbidden"), False),
        (PermissionError("forbidden path /tmp"), False),
        (asyncio.TimeoutError(), False),
    ],
)
def test_is_asknews_subscription_error_parametrized(exc: BaseException, expected: bool) -> None:
    """Parametrized smoke test mirroring the class-based tests above."""
    assert is_asknews_subscription_error(exc) is expected
