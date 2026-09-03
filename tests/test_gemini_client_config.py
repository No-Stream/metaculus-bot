"""Tests for the shared google-genai client configuration (``research/gemini_client_config``).

Half of these are canaries on the SDK rather than on our code: the reader's whole timeout
arithmetic rests on what ``google.genai`` does with ``HttpRetryOptions``, so an SDK bump
that changed the default retry policy or the backoff formula would otherwise move a
deadline silently.
"""

import pytest
import tenacity
from google.genai import _api_client
from google.genai import types as genai_types

from metaculus_bot.research.gemini_client_config import (
    GEMINI_RETRY_HTTP_STATUS_CODES,
    GEMINI_RETRY_INITIAL_DELAY_S,
    GEMINI_RETRY_MAX_DELAY_S,
    build_gemini_http_options,
    gemini_retry_sleep_allowance_s,
    gemini_thinking_config,
)


class TestSdkRetryCanaries:
    def test_a_bare_client_would_not_retry_at_all(self) -> None:
        """The reason this module exists: ``retry_options=None`` is stop-after-one.

        Not "retry the obvious transients" — nothing, which is how a ``503 UNAVAILABLE``
        that returns in milliseconds cost two production document reads outright.
        """
        default_kwargs = _api_client.retry_args(None)

        assert default_kwargs["stop"].max_attempt_number == 1
        assert "retry" not in default_kwargs, "no retry predicate at all on the default path"

    def test_our_options_reach_the_sdk_retryer_with_the_backoff_we_assume(self) -> None:
        options = build_gemini_http_options(timeout_ms=26_500, attempts=2)

        retry_kwargs = _api_client.retry_args(options.retry_options)

        assert retry_kwargs["stop"].max_attempt_number == 2
        wait = retry_kwargs["wait"]
        assert isinstance(wait, tenacity.wait_exponential_jitter)
        assert wait.initial == GEMINI_RETRY_INITIAL_DELAY_S
        assert wait.max == GEMINI_RETRY_MAX_DELAY_S
        # The two values ``gemini_retry_sleep_allowance_s`` assumes and cannot configure.
        assert wait.exp_base == 2.0
        assert wait.jitter == 1.0


class TestHttpOptionsBuilder:
    def test_timeout_and_attempts_are_carried_verbatim(self) -> None:
        options = build_gemini_http_options(timeout_ms=350_000, attempts=2)

        assert options.timeout == 350_000
        assert options.retry_options is not None
        assert options.retry_options.attempts == 2

    def test_only_transient_statuses_are_retried(self) -> None:
        """A 400, 401, 403 or 404 answers the same however many times we ask, and a retry on
        one of them would spend a second call to learn nothing."""
        codes = build_gemini_http_options(timeout_ms=1_000, attempts=2).retry_options
        assert codes is not None

        assert codes.http_status_codes == list(GEMINI_RETRY_HTTP_STATUS_CODES)
        assert 503 in GEMINI_RETRY_HTTP_STATUS_CODES
        assert not {400, 401, 403, 404} & set(GEMINI_RETRY_HTTP_STATUS_CODES)


class TestSleepAllowance:
    @pytest.mark.parametrize(
        ("attempts", "expected_s"),
        # Worst case per retry is min(1.0 * 2**retry + 1.0, 8.0): 2.0, then 3.0, then 5.0.
        [(1, 0.0), (2, 2.0), (3, 5.0), (4, 10.0)],
    )
    def test_worst_case_totals(self, attempts: int, expected_s: float) -> None:
        assert gemini_retry_sleep_allowance_s(attempts) == expected_s

    def test_a_single_attempt_reserves_nothing(self) -> None:
        # attempts=1 means no retry, so a caller dividing its budget must not lose a slice
        # of it to a sleep that never happens.
        assert gemini_retry_sleep_allowance_s(1) == 0


class TestThinkingConfig:
    def test_lowercase_levels_resolve_through_the_case_insensitive_enum(self) -> None:
        assert gemini_thinking_config("medium").thinking_level == genai_types.ThinkingLevel.MEDIUM
        assert gemini_thinking_config("low").thinking_level == genai_types.ThinkingLevel.LOW

    def test_an_unknown_level_raises_here_rather_than_at_the_api(self) -> None:
        # The SDK's own coercion would not: ``ThinkingLevel("ludicrous")`` warns and hands
        # back a fabricated member, which would reach Google as a made-up level.
        with pytest.raises(ValueError, match="ludicrous"):
            gemini_thinking_config("ludicrous")
