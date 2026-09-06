"""Tests for metaculus_bot.research.providers helpers."""

from __future__ import annotations

import pytest

from metaculus_bot.research.providers import is_asknews_subscription_error


class ForbiddenError(Exception):
    """Stand-in for ``asknews_sdk.errors.ForbiddenError``."""


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        pytest.param(
            ForbiddenError("403011 - subscription is not currently active"),
            True,
            id="forbidden_subscription_code",
        ),
        pytest.param(
            ForbiddenError("subscription is not currently active on this tier"),
            True,
            id="forbidden_subscription_phrase",
        ),
        pytest.param(
            ForbiddenError("403000 - rate limit hit"),
            False,
            id="forbidden_unrelated_message",
        ),
        pytest.param(RuntimeError("403 Forbidden"), False, id="generic_403"),
        pytest.param(PermissionError("forbidden path /tmp"), False, id="permission_error"),
        pytest.param(TimeoutError(), False, id="timeout_error"),
    ],
)
def test_is_asknews_subscription_error(exc: BaseException, expected: bool) -> None:
    """Match only the AskNews subscription-inactive error signature."""
    assert is_asknews_subscription_error(exc) is expected
