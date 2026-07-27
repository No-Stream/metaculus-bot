import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from litellm.exceptions import APIError, RateLimitError

from metaculus_bot import fallback_openrouter
from metaculus_bot.constants import CREDIT_ALERT_RESUME_DATE
from metaculus_bot.credit_telemetry import DonatedKeyState, reset_donated_key_state_cache
from metaculus_bot.fallback_openrouter import (
    DONATED_KEY_PROVIDERS,
    FallbackOpenRouterLlm,
    build_llm_with_openrouter_fallback,
    get_credit_key_fallback_count,
    get_donated_404_fallback_count,
    get_generic_key_fallback_count,
    is_credit_caused_error,
    is_suppressible_credit_error,
    reset_credit_key_fallback_count,
    reset_donated_404_fallback_count,
    reset_generic_key_fallback_count,
    should_retry_with_general_key,
    should_route_via_donated_key,
)

# The verbatim OpenRouter response that cost the 2026-07-26 tournament run two of
# three forecasters and most of the research stack. Copied character-for-character
# from the run log: HTTP 403 (not the documented 402), the phrase "Key limit
# exceeded (total limit)", a "code":403 field that the old negative rule matched on,
# and a key hash that happens to contain none of 401/402/403/429.
PRODUCTION_KEY_LIMIT_403 = (
    "litellm.APIError: APIError: OpenrouterException - "
    '{"error":{"message":"Key limit exceeded (total limit). Manage it using '
    "https://openrouter.ai/workspaces/default/keys/"
    '8f5af82f134c33c0dbada6e1ce93b780819cc08716001bef5ab4af81791702bd","code":403}}'
)

# A realistic OpenRouter moderation 403. The body carries `reasons` and
# `flagged_input` (up to ~100 chars of OUR OWN PROMPT replayed back), which is why
# a numeric substring match on the whole message is unsafe.
MODERATION_403 = (
    "litellm.APIError: APIError: OpenrouterException - "
    '{"error":{"message":"Your input was flagged for the following reasons: violence",'
    '"code":403,"metadata":{"reasons":["violence"],'
    '"flagged_input":"Will the conflict escalate before Aug 18 2026?",'
    '"provider_name":"OpenAI","model_slug":"openai/gpt-5.6-sol"}}}'
)

# The same moderation 403 whose replayed prompt text contains the token "402" — a
# dollar figure or bill number is entirely ordinary in a forecasting prompt. The
# old bare-"402" substring cue would read this as an empty wallet, bill the
# personal key for a call that will refuse again, and exempt a real moderation
# block from alerting.
MODERATION_403_ECHOING_402 = (
    "litellm.APIError: APIError: OpenrouterException - "
    '{"error":{"message":"Your input was flagged for the following reasons: violence",'
    '"code":403,"metadata":{"reasons":["violence"],'
    '"flagged_input":"Will the $402 billion defense package pass before Aug 18 2026?",'
    '"provider_name":"OpenAI","model_slug":"openai/gpt-5.6-sol"}}}'
)

# The footgun that dictates the cue wording. "limit exceeded" is a substring of
# this, so the short form would classify every rate-limit breach as credit-caused
# and silently exempt it from alerting for the whole suppression window.
FREE_MODEL_RATE_LIMIT_429 = (
    "litellm.RateLimitError: OpenrouterException - "
    '{"error":{"message":"Rate limit exceeded: free-models-per-day","code":429}}'
)


class TestPredicates:
    def test_should_route_via_donated_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # OpenAI / Anthropic always route via the donated key when one is configured.
        assert should_route_via_donated_key("openrouter/openai/gpt-5.1") is True
        assert should_route_via_donated_key("openrouter/anthropic/claude-sonnet-4") is True
        # Google is gated on GEMINI_USE_DONATED_OPENROUTER_KEY. With the toggle on,
        # flash models prefer the donated key — but gemini-3.1-pro is on the
        # DONATED_KEY_BLOCKED_GOOGLE_MODELS blocklist (free-tier BYOK → 429), so it
        # is pinned to the personal key even with the toggle ON.
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "true")
        assert should_route_via_donated_key("openrouter/google/gemini-3.5-flash") is True
        assert should_route_via_donated_key("openrouter/google/gemini-3.1-flash-lite") is True
        assert should_route_via_donated_key("openrouter/google/gemini-3-flash-preview") is True
        assert should_route_via_donated_key("openrouter/google/gemini-3.1-pro-preview") is False
        # Explicit toggle off: ALL Google calls go through the operator's personal key only.
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "false")
        assert should_route_via_donated_key("openrouter/google/gemini-3.1-pro-preview") is False
        assert should_route_via_donated_key("openrouter/google/gemini-3.5-flash") is False
        # Default with the env var unset is ON (donated key with personal fallback):
        # after Metaculus raised the Google rate limits (2026-06-16) the donated key
        # serves most Gemini. A flash model routes donated-first by default; the
        # gemini-3.1-pro slug stays blocklisted (pinned to personal) regardless.
        monkeypatch.delenv("GEMINI_USE_DONATED_OPENROUTER_KEY", raising=False)
        assert should_route_via_donated_key("openrouter/google/gemini-3.5-flash") is True
        assert should_route_via_donated_key("openrouter/google/gemini-3.1-pro-preview") is False
        # Providers NOT covered by the donated key.
        assert should_route_via_donated_key("openrouter/x-ai/grok-4.1-fast") is False
        assert should_route_via_donated_key("openrouter/qwen/qwen3-235b") is False
        # Non-OpenRouter slugs.
        assert should_route_via_donated_key("perplexity/sonar") is False
        # Defensive: bogus inputs return False rather than raising.
        assert should_route_via_donated_key("openrouter/") is False  # parts < 2
        assert should_route_via_donated_key("") is False

    def test_donated_key_providers_set(self) -> None:
        # Pin the membership so any drift surfaces in code review rather than
        # silently changing routing.
        assert DONATED_KEY_PROVIDERS == frozenset({"openai", "anthropic", "google"})

    @pytest.mark.parametrize(
        "message, expected",
        [
            ("HTTP 402 Payment Required", True),
            ("payment required", True),
            ("insufficient credit on key", True),
            ("401 Unauthorized", True),
            ("invalid API key", True),
            ("disabled api key", True),
            # Donated-key allowed-providers 404 quirk → should fall back.
            ("404 No allowed providers are available for the selected model.", True),
            ("no allowed providers", True),
            # Donated-key data-policy / guardrail 404 (added 2026-05-17 for OpenAI
            # native search migration) — donated key's data-collection guardrail
            # blocks OpenAI native search; the personal key has no such block.
            (
                "404 No endpoints available matching your guardrail restrictions and data policy. "
                "Configure: https://openrouter.ai/settings/privacy",
                True,
            ),
            ("matching your guardrail restrictions", True),
            ("data policy", True),
            # 429 rate-limit: textual detection falls back (BYOK quotas are per-key).
            ("429 Too Many Requests", True),
            ("Rate limit exceeded", True),
            # Belt-and-suspenders textual patterns for 429 edge cases.
            ('{"code":429, "message": "rate limited"}', True),
            ("rate-limited upstream by provider", True),
            # Negative: moderation / infrastructure errors do NOT fall back.
            ("403 Forbidden moderation", False),
            ("502 Bad Gateway", False),
            ("503 Service Unavailable", False),
        ],
    )
    def test_should_retry_with_general_key(self, message: str, expected: bool) -> None:
        assert should_retry_with_general_key(Exception(message)) is expected

    def test_litellm_rate_limit_error_triggers_fallback(self) -> None:
        """litellm.RateLimitError (typed 429) triggers fallback — BYOK quotas are independent."""
        from litellm.exceptions import RateLimitError

        exc = RateLimitError(
            message="Rate limit exceeded on openrouter",
            model="openrouter/google/gemini-3.1-pro-preview",
            llm_provider="openrouter",
        )
        assert should_retry_with_general_key(exc) is True

    def test_litellm_service_unavailable_does_not_trigger_fallback(self) -> None:
        """litellm.ServiceUnavailableError (503) does NOT trigger fallback — infrastructure issue."""
        from litellm.exceptions import ServiceUnavailableError

        exc = ServiceUnavailableError(
            message="503 Service Unavailable",
            model="openrouter/openai/gpt-5.1",
            llm_provider="openrouter",
        )
        assert should_retry_with_general_key(exc) is False


class TestFallbackOpenRouterLlm:
    @pytest.mark.asyncio
    async def test_primary_success_no_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.1",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        # Patch the internal primary call point to avoid network.
        monkeypatch.setattr(llm, "_invoke_once_using_primary", AsyncMock(return_value="answer"))

        out = await llm.invoke("hi")
        assert out == "answer"

    @pytest.mark.asyncio
    async def test_fallback_on_402(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FallbackOpenRouterLlm(
            model="openrouter/anthropic/claude-sonnet-4",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("HTTP 402 Payment Required: insufficient credit")),
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        out = await llm.invoke("hi")
        assert out == "ok"

    @pytest.mark.asyncio
    async def test_fallback_on_429_rate_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """429 on primary key falls back to secondary — BYOK quotas are independent."""
        from litellm.exceptions import RateLimitError

        llm = FallbackOpenRouterLlm(
            model="openrouter/google/gemini-3.1-pro-preview",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        exc = RateLimitError(
            message="Rate limit exceeded",
            model="openrouter/google/gemini-3.1-pro-preview",
            llm_provider="openrouter",
        )
        monkeypatch.setattr(llm, "_invoke_once_using_primary", AsyncMock(side_effect=exc))
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="fallback_ok"))

        out = await llm.invoke("hi")
        assert out == "fallback_ok"

    @pytest.mark.asyncio
    async def test_no_fallback_on_403(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.1",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("403 Forbidden moderation")),
        )

        with pytest.raises(Exception):
            await llm.invoke("hi")

    @pytest.mark.asyncio
    async def test_no_fallback_on_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """503 Service Unavailable re-raises without fallback — infrastructure issue, not key-scoped."""
        from litellm.exceptions import ServiceUnavailableError

        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.1",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        exc = ServiceUnavailableError(
            message="503 Service Unavailable",
            model="openrouter/openai/gpt-5.1",
            llm_provider="openrouter",
        )
        monkeypatch.setattr(llm, "_invoke_once_using_primary", AsyncMock(side_effect=exc))

        with pytest.raises(ServiceUnavailableError):
            await llm.invoke("hi")

    @pytest.mark.asyncio
    async def test_no_secondary_key_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.1",
            primary_api_key="special",
            secondary_api_key=None,
            temperature=0,
        )

        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("401 Unauthorized")),
        )

        with pytest.raises(Exception):
            await llm.invoke("hi")


class TestKeyLimitExceeded403:
    """OpenRouter reports a per-key SPEND-CAP breach as HTTP 403 with the text
    "Key limit exceeded (total limit)" — not the 402 its own error docs document.

    litellm has no 403 branch for OpenRouter, so this always arrives as a bare
    ``APIError``, and the old negative rule (``"403" in msg`` → never fall back,
    written for content moderation) matched the ``"code":403`` field in the JSON
    body. Two of three forecasters, native search, the AskNews summarizer, the
    financial classifier, prediction-market keyword extraction, gap-fill v1 and
    gap-fill v2 all died on a run where the operator's funded personal key sat
    idle. These tests pin the classification in both directions: a spend-cap 403
    falls back, a moderation 403 still does not.
    """

    def test_production_message_falls_back_and_is_credit_classified(self) -> None:
        """The exact string from the 2026-07-26 run log. Both predicates must fire:
        the retry decision routes the call to the funded key, and the credit
        classification is what lets cli exempt the expected empty wallet from CI
        alerting. Neither reads the network — both are textual.
        """
        exc = Exception(PRODUCTION_KEY_LIMIT_403)
        assert should_retry_with_general_key(exc) is True
        assert is_credit_caused_error(exc) is True

    @pytest.mark.parametrize(
        "message",
        [
            "Key limit exceeded (total limit).",
            "key limit exceeded (daily limit)",
            'OpenrouterException - {"error":{"message":"Key limit exceeded (monthly limit).","code":403}}',
        ],
    )
    def test_key_limit_variants_all_classify_as_credit(self, message: str) -> None:
        """The cue is the phrase, not the parenthetical, so the daily and monthly
        cap variants are covered without enumerating them.
        """
        assert is_credit_caused_error(Exception(message)) is True
        assert should_retry_with_general_key(Exception(message)) is True

    def test_moderation_403_does_not_fall_back_and_is_not_credit(self) -> None:
        """Negative control. A moderation block is not key-scoped — the personal key
        would refuse the same prompt — so it must keep failing closed.
        """
        exc = Exception(MODERATION_403)
        assert should_retry_with_general_key(exc) is False
        assert is_credit_caused_error(exc) is False

    def test_moderation_403_echoing_402_in_prompt_is_still_moderation(self) -> None:
        """The nasty variant, and a latent defect this fix closes rather than adds.

        OpenRouter replays up to ~100 characters of our own prompt as
        ``flagged_input``, and forecasting prompts are full of dollar figures and
        bill numbers. A bare "402" substring match on the whole message therefore
        reads an ordinary moderation refusal as an empty wallet — which would both
        bill the personal key for a call that refuses again and exempt a real
        moderation block from alerting.
        """
        exc = Exception(MODERATION_403_ECHOING_402)
        assert "402" in str(exc)
        assert should_retry_with_general_key(exc) is False
        assert is_credit_caused_error(exc) is False

    def test_free_model_rate_limit_429_is_not_credit_classified(self) -> None:
        """The footgun regression test: this is why the cue is "key limit exceeded"
        and not the shorter "limit exceeded", which is a substring of
        "Rate limit exceeded: free-models-per-day". Shortening the cue would
        classify every 429 as credit-caused and silently exempt real rate-limit
        breakage from alerting for the whole suppression window.
        """
        exc = Exception(FREE_MODEL_RATE_LIMIT_429)
        assert is_credit_caused_error(exc) is False
        # It still falls back — a 429 is key-scoped (BYOK quotas are per-key) — it
        # just isn't exempt from alerting.
        assert should_retry_with_general_key(exc) is True

    def test_plain_402_still_falls_back_without_a_403_in_the_body(self) -> None:
        """The documented 402 path is untouched by the moderation hardening."""
        exc = Exception('{"error":{"message":"Insufficient credits","code":402}}')
        assert should_retry_with_general_key(exc) is True
        assert is_credit_caused_error(exc) is True


class TestCreditSuppressibility:
    """``is_suppressible_credit_error`` — the drained-vs-revoked discriminator.

    The operator's decision: a genuinely DRAINED donated key must not redden CI
    while they self-fund the season, but a key Metaculus revoked or re-capped to
    zero must stay red. Both produce identical "Key limit exceeded" text, so the
    text cue alone cannot make that call — we probe OpenRouter's free
    ``/auth/key`` endpoint and only DRAINED is exempt.

    Fallback ROUTING is deliberately not gated on the probe: a stale or cached
    balance read must never be able to strand the ensemble on a dry key.
    """

    @pytest.fixture(autouse=True)
    def _no_real_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default every test in this class to "probe was never able to run", so a
        test that forgets to pin a state can't reach the network.
        """
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        reset_donated_key_state_cache()

    def _pin_state(self, monkeypatch: pytest.MonkeyPatch, state: DonatedKeyState) -> MagicMock:
        probe = MagicMock(return_value=state)
        monkeypatch.setattr(fallback_openrouter, "classify_donated_key_state", probe)
        return probe

    def test_drained_key_is_suppressible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._pin_state(monkeypatch, DonatedKeyState.DRAINED)
        assert is_suppressible_credit_error(Exception(PRODUCTION_KEY_LIMIT_403)) is True

    @pytest.mark.parametrize(
        "state",
        [DonatedKeyState.REVOKED, DonatedKeyState.ZEROED, DonatedKeyState.FUNDED, DonatedKeyState.UNKNOWN],
    )
    def test_every_other_state_stays_alertable(self, monkeypatch: pytest.MonkeyPatch, state: DonatedKeyState) -> None:
        """The regression that matters: a revoked key, a key re-capped to zero, a
        key that still has money, and a probe that failed all keep CI red.
        """
        self._pin_state(monkeypatch, state)
        assert is_suppressible_credit_error(Exception(PRODUCTION_KEY_LIMIT_403)) is False

    def test_probe_failure_fails_safe_to_alertable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No donated key configured → the probe can't run → UNKNOWN → still red.
        A broken probe must never be able to silently green a run.
        """
        assert is_suppressible_credit_error(Exception(PRODUCTION_KEY_LIMIT_403)) is False

    def test_documented_402_needs_no_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 402 / insufficient-credit response is unambiguous: the wallet is empty.
        Only the ambiguous spend-cap 403 pays for a probe, so the long-standing 402
        path keeps working with the endpoint unreachable.
        """
        probe = self._pin_state(monkeypatch, DonatedKeyState.UNKNOWN)
        assert is_suppressible_credit_error(Exception("HTTP 402 Payment Required: insufficient credit")) is True
        probe.assert_not_called()

    @pytest.mark.parametrize("message", ["401 Unauthorized", "404 no allowed providers", MODERATION_403])
    def test_non_credit_causes_are_never_suppressible(self, monkeypatch: pytest.MonkeyPatch, message: str) -> None:
        probe = self._pin_state(monkeypatch, DonatedKeyState.DRAINED)
        assert is_suppressible_credit_error(Exception(message)) is False
        probe.assert_not_called()

    def test_reported_429_echoing_402_is_not_exempted_from_alerting(self) -> None:
        """A rate limit must stay alertable even when the body echoes "402".

        The mirror image of the terse-402 case, and the more dangerous direction. A
        reported 429 whose replayed prompt contains "402" and no moderation word
        routes correctly on the status — but while the classification was text-only
        it read the bare digits as an empty wallet, so a genuine rate-limit fallback
        was subtracted from ``alertable`` for the whole suppression window. Same
        failure the synthesis predicted for a too-short credit cue, reached through
        the numeric door instead.
        """
        body = (
            "OpenrouterException - "
            '{"error":{"message":"Rate limited","code":429,'
            '"metadata":{"input":"Will the $402 billion package pass before Aug 18 2026?"}}}'
        )
        exc = RateLimitError(message=body, model="openrouter/openai/gpt-5.6-sol", llm_provider="openrouter")
        assert "402" in str(exc)  # the poisoning cue is present
        assert should_retry_with_general_key(exc) is True  # 429 is key-scoped: still falls back
        assert is_credit_caused_error(exc) is False
        assert is_suppressible_credit_error(exc) is False

    def test_terse_reported_402_is_credit_classified_like_it_routes(self) -> None:
        """Routing and alerting must answer "was this about money?" identically.

        A reported 402 whose prose spells out neither a credit phrase nor the digits
        (litellm renders it as just ``litellm.APIError: wallet empty``) routes to the
        paid key on the reported status. While the classification stayed text-only it
        did NOT count as credit-caused, so the run reddened CI on precisely the
        expected empty wallet the suppression window exists for.
        """
        exc = APIError(status_code=402, message="wallet empty", llm_provider="openrouter", model="openai/gpt-5.6-sol")
        assert "402" not in str(exc)  # the gap only exists because the digits are absent
        assert should_retry_with_general_key(exc) is True
        assert is_credit_caused_error(exc) is True
        assert is_suppressible_credit_error(exc) is True


class TestCreditCauseClassification:
    """``is_credit_caused_error`` is the ONE place the "empty wallet" text cues
    live: ``should_retry_with_general_key`` and the credit-subset counter both go
    through it, so the retry decision and the counter can't drift apart.

    Only these messages are exempt from alerting during the suppression window
    (see ``constants.credit_alerts_active``) — everything else keeps reddening CI.
    """

    @pytest.mark.parametrize(
        "message",
        [
            "HTTP 402 Payment Required",
            "payment required",
            "insufficient credit on key",
            "out of credits",
            "insufficient funds",
            "402 Payment Required: insufficient credit",
        ],
    )
    def test_credit_messages_classified_as_credit_caused(self, message: str) -> None:
        assert is_credit_caused_error(Exception(message)) is True
        # Same messages must still trigger the fallback itself — the exemption
        # changes accounting, never routing.
        assert should_retry_with_general_key(Exception(message)) is True

    @pytest.mark.parametrize(
        "message",
        [
            "401 Unauthorized",
            "invalid API key",
            "disabled api key",
            "404 No allowed providers are available for the selected model.",
            "429 Too Many Requests",
            "Rate limit exceeded",
            "matching your guardrail restrictions",
            "data policy",
            "403 Forbidden moderation",
            "503 Service Unavailable",
        ],
    )
    def test_non_credit_messages_not_classified_as_credit_caused(self, message: str) -> None:
        """The regression that matters: real breakage must never be exempted."""
        assert is_credit_caused_error(Exception(message)) is False


class TestPaidFallbackWarningExitClause:
    """The paid-fallback WARNING must not promise a red CI run it won't deliver.

    ``credit_alerts_active`` is monkeypatched (rather than waiting on the wall
    clock) so both branches stay covered before and after 2026-09-10. The
    ``suppressible`` flag is passed in by the caller, which has already paid for
    the donated-key probe — the note renders the decision, it doesn't re-derive it.
    """

    def test_suppressible_credit_cause_during_window_says_not_alertable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fallback_openrouter, "credit_alerts_active", lambda: False)
        note = fallback_openrouter._fallback_alert_note(suppressible=True)
        assert "NOT counted as alertable" in note
        assert CREDIT_ALERT_RESUME_DATE.isoformat() in note

    def test_credit_cause_after_resume_says_exit_non_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fallback_openrouter, "credit_alerts_active", lambda: True)
        note = fallback_openrouter._fallback_alert_note(suppressible=True)
        assert note == "Run will complete, then exit non-zero to alert."

    def test_non_credit_cause_always_says_exit_non_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Even mid-suppression, a 401 still promises the red run it will cause."""
        monkeypatch.setattr(fallback_openrouter, "credit_alerts_active", lambda: False)
        note = fallback_openrouter._fallback_alert_note(suppressible=False)
        assert note == "Run will complete, then exit non-zero to alert."

    def test_revoked_donated_key_is_not_promised_as_suppressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A credit-SHAPED failure on a revoked key is not suppressible, so the
        note must promise the red run it will actually cause. Before the
        discriminator, the note keyed on the text cue alone and would have claimed
        the run stays green.
        """
        monkeypatch.setattr(fallback_openrouter, "credit_alerts_active", lambda: False)
        note = fallback_openrouter._fallback_alert_note(suppressible=False)
        assert "NOT counted as alertable" not in note


class TestFallbackCounters:
    """Every donated->personal fallback must be counted + logged loudly so silent
    personal-key spend can't accumulate. ``_generic_key_fallback_count`` counts ALL
    fallback causes; ``_donated_404_fallback_count`` (allowed-providers 404) and
    ``_credit_key_fallback_count`` (402/insufficient-credit) are two disjoint
    subsets of that total. cli.py folds the generic counter into its non-zero-exit
    alert and subtracts the credit subset while credit alerting is suppressed.
    """

    def setup_method(self) -> None:
        reset_generic_key_fallback_count()
        reset_donated_404_fallback_count()
        reset_credit_key_fallback_count()

    def teardown_method(self) -> None:
        reset_generic_key_fallback_count()
        reset_donated_404_fallback_count()
        reset_credit_key_fallback_count()

    @pytest.mark.asyncio
    async def test_credit_fallback_bumps_credit_subset_and_still_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A 402 fallback bumps the generic total AND the credit subset, and still
        logs the loud PAID PERSONAL-KEY FALLBACK warning. The suppression only
        removes the event from cli's alertable arithmetic — never from the logs.
        """
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.6-sol",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("HTTP 402 Payment Required: insufficient credit")),
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.fallback_openrouter"):
            assert await llm.invoke("hi") == "ok"

        assert get_generic_key_fallback_count() == 1
        assert get_credit_key_fallback_count() == 1
        assert get_donated_404_fallback_count() == 0
        assert any("PAID PERSONAL-KEY FALLBACK" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "message",
        [
            "401 Unauthorized: invalid api key",
            "429 Too Many Requests: rate limit exceeded",
        ],
    )
    async def test_non_credit_fallback_leaves_credit_subset_at_zero(
        self, monkeypatch: pytest.MonkeyPatch, message: str
    ) -> None:
        """401 / 429 bump only the generic total. If either leaked into the credit
        subset, cli would subtract it and a genuinely broken key would ship green.
        """
        llm = FallbackOpenRouterLlm(
            model="openrouter/anthropic/claude-opus-4.8",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(llm, "_invoke_once_using_primary", AsyncMock(side_effect=Exception(message)))
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        assert await llm.invoke("hi") == "ok"
        assert get_generic_key_fallback_count() == 1
        assert get_credit_key_fallback_count() == 0

    @pytest.mark.asyncio
    async def test_donated_404_leaves_credit_subset_at_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The two subsets are disjoint: a 404 belongs to the 404 subset only, so
        cli's single subtraction can't double-discount it.
        """
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.4",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("404 No allowed providers are available for the selected model.")),
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        assert await llm.invoke("hi") == "ok"
        assert get_generic_key_fallback_count() == 1
        assert get_donated_404_fallback_count() == 1
        assert get_credit_key_fallback_count() == 0

    @pytest.mark.asyncio
    async def test_generic_fallback_bumps_counter_and_logs_every_time(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A 401/402 (non-404) fallback bumps ONLY the generic counter, not the 404 subset,
        and logs a WARNING on EVERY fallback (no once-per-instance suppression)."""
        llm = FallbackOpenRouterLlm(
            model="openrouter/anthropic/claude-opus-4.8",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("HTTP 402 Payment Required: insufficient credit")),
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.fallback_openrouter"):
            assert await llm.invoke("hi") == "ok"
            assert await llm.invoke("hi") == "ok"

        # Generic counter bumped on both fallbacks; 404 subset untouched.
        assert get_generic_key_fallback_count() == 2
        assert get_donated_404_fallback_count() == 0
        # Loud WARNING on every fallback, not just the first.
        paid_warnings = [r for r in caplog.records if "PAID PERSONAL-KEY FALLBACK" in r.getMessage()]
        assert len(paid_warnings) == 2

    @pytest.mark.asyncio
    async def test_key_limit_403_on_drained_key_bumps_credit_subset(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The production failure, end to end through the wrapper: the spend-cap 403
        now falls back to the personal key, and because the probe confirms the
        donated key is genuinely drained, the event lands in the suppressible credit
        subset so cli exits zero on it.
        """
        monkeypatch.setattr(
            fallback_openrouter, "classify_donated_key_state", MagicMock(return_value=DonatedKeyState.DRAINED)
        )
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.6-sol",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(
            llm, "_invoke_once_using_primary", AsyncMock(side_effect=Exception(PRODUCTION_KEY_LIMIT_403))
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.fallback_openrouter"):
            assert await llm.invoke("hi") == "ok"

        assert get_generic_key_fallback_count() == 1
        assert get_credit_key_fallback_count() == 1
        assert get_donated_404_fallback_count() == 0
        # Nothing is silenced by the suppression — the loud warning still fires.
        assert any("PAID PERSONAL-KEY FALLBACK" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_key_limit_403_on_revoked_key_stays_alertable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same error text, revoked key: still falls back (the run should publish if
        it can), but the credit subset stays empty so cli exits non-zero. This is
        the whole reason the discriminator exists — a text cue alone cannot tell
        these two runs apart.
        """
        monkeypatch.setattr(
            fallback_openrouter, "classify_donated_key_state", MagicMock(return_value=DonatedKeyState.REVOKED)
        )
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.6-sol",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(
            llm, "_invoke_once_using_primary", AsyncMock(side_effect=Exception(PRODUCTION_KEY_LIMIT_403))
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        assert await llm.invoke("hi") == "ok"
        assert get_generic_key_fallback_count() == 1
        assert get_credit_key_fallback_count() == 0

    @pytest.mark.asyncio
    async def test_moderation_403_still_raises_and_counts_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Negative control at the wrapper level: a moderation 403 must not reach the
        personal key and must not bump any counter.
        """
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.6-sol",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(
            llm, "_invoke_once_using_primary", AsyncMock(side_effect=Exception(MODERATION_403_ECHOING_402))
        )
        secondary = AsyncMock(return_value="should not be reached")
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", secondary)

        with pytest.raises(Exception, match="flagged"):
            await llm.invoke("hi")

        secondary.assert_not_awaited()
        assert get_generic_key_fallback_count() == 0
        assert get_credit_key_fallback_count() == 0

    @pytest.mark.asyncio
    async def test_donated_404_fallback_bumps_both_counters(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A 404 'no allowed providers' fallback bumps BOTH the generic total and the 404 subset
        (a 404 fallback is still a personal-key fallback)."""
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.4",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("404 No allowed providers are available for the selected model.")),
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.fallback_openrouter"):
            assert await llm.invoke("hi") == "ok"

        assert get_generic_key_fallback_count() == 1
        assert get_donated_404_fallback_count() == 1
        assert any("no allowed providers" in r.getMessage() for r in caplog.records)


class TestBuilder:
    def test_builder_returns_wrapper_when_both_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        llm = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5.1")
        assert isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_plain_when_only_general(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        llm = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5.1")
        # Not wrapper, should be a GeneralLlm
        from forecasting_tools import GeneralLlm as GL

        assert isinstance(llm, GL)
        assert not isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_plain_for_non_donated_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Models from providers not in DONATED_KEY_PROVIDERS get a plain GeneralLlm
        (no donated-key wrapping). Grok via x-ai is the canonical example.
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        llm = build_llm_with_openrouter_fallback("openrouter/x-ai/grok-4.1-fast")
        # Not wrapper, should be a GeneralLlm
        from forecasting_tools import GeneralLlm as GL

        assert isinstance(llm, GL)
        assert not isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_returns_wrapper_for_google_flash_when_donated_toggle_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flash Google models route via the donated wrapper when the toggle is ON.

        Originally added in task #12 (Google in DONATED_KEY_PROVIDERS). After the
        2026-06-16 rate-limit bump the donated key serves the flash models, so a
        flash slug returns a FallbackOpenRouterLlm (donated primary, personal
        fallback). gemini-3.1-pro is handled separately by the blocklist test.
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "true")
        llm = build_llm_with_openrouter_fallback("openrouter/google/gemini-3.5-flash")
        assert isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_plain_for_google_pro_blocklisted_when_toggle_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """gemini-3.1-pro is pinned to the personal key via the blocklist even with
        the donated toggle ON — so the builder returns a plain GeneralLlm (no
        donated attempt, no 429, no fallback-counter bump), while a flash model in
        the same env returns a FallbackOpenRouterLlm.

        Temporary workaround; see DONATED_KEY_BLOCKED_GOOGLE_MODELS
        (``TODO(gemini-3.1-pro-donated)``) and FUTURE.md.
        """
        from forecasting_tools import GeneralLlm as GL

        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "true")
        pro = build_llm_with_openrouter_fallback("openrouter/google/gemini-3.1-pro-preview")
        assert isinstance(pro, GL)
        assert not isinstance(pro, FallbackOpenRouterLlm)
        flash = build_llm_with_openrouter_fallback("openrouter/google/gemini-3.5-flash")
        assert isinstance(flash, FallbackOpenRouterLlm)

    def test_builder_plain_for_google_when_donated_toggle_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the donated-key toggle off, ALL Google calls bypass the donated
        wrapper entirely — the resulting LLM is a plain GeneralLlm using the
        operator's general OpenRouter key.
        """
        from forecasting_tools import GeneralLlm as GL

        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "false")
        llm = build_llm_with_openrouter_fallback("openrouter/google/gemini-3.5-flash")
        assert isinstance(llm, GL)
        assert not isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_returns_wrapper_for_google_flash_when_toggle_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default (env var unset) is ON: flash Google calls prefer the donated wrapper.

        After Metaculus raised the Google rate limits (2026-06-16) the donated key
        serves most Gemini, so with two distinct keys configured a flash slug
        returns a FallbackOpenRouterLlm (donated primary, personal fallback).
        gemini-3.1-pro stays blocklisted (plain GeneralLlm) — see the dedicated
        blocklist test.
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        monkeypatch.delenv("GEMINI_USE_DONATED_OPENROUTER_KEY", raising=False)
        llm = build_llm_with_openrouter_fallback("openrouter/google/gemini-3.5-flash")
        assert isinstance(llm, FallbackOpenRouterLlm)


class TestDeprecationTripwire:
    """Post-submission deprecation tripwire (added 2026-05-17 after the
    2026-05-15 x-ai/grok-4.1-fast deprecation silently 404'd for ~2 days).
    """

    def test_records_deprecation_404(self) -> None:
        """A canonical OpenRouter deprecation 404 is recorded with model + msg."""
        from metaculus_bot.fallback_openrouter import (
            _DEPRECATION_ALERTS,
            _record_deprecation_if_matched,
            clear_deprecation_alerts,
        )

        clear_deprecation_alerts()
        msg = "Grok 4.1 Fast is deprecated. xAI recommends switching to Grok 4.3"
        matched = _record_deprecation_if_matched("x-ai/grok-4.1-fast", msg)
        assert matched is True
        assert _DEPRECATION_ALERTS == [("x-ai/grok-4.1-fast", msg)]
        clear_deprecation_alerts()

    def test_does_not_record_unrelated_error(self) -> None:
        """Plain 401/429/etc. don't match — false positives turn CI red without cause."""
        from metaculus_bot.fallback_openrouter import (
            _DEPRECATION_ALERTS,
            _record_deprecation_if_matched,
            clear_deprecation_alerts,
        )

        clear_deprecation_alerts()
        for msg in (
            "401 Unauthorized: invalid api key",
            "429 Too Many Requests: rate limit exceeded",
            "402 Payment Required: insufficient credit",
            "404 No allowed providers",
            "503 Service Unavailable",
        ):
            matched = _record_deprecation_if_matched("openrouter/openai/gpt-5.5", msg)
            assert matched is False, f"falsely matched: {msg}"
        assert _DEPRECATION_ALERTS == []
        clear_deprecation_alerts()

    def test_check_exits_on_populated_list(self) -> None:
        """Tripwire fires sys.exit(1) when at least one deprecation was seen."""
        from metaculus_bot.fallback_openrouter import (
            _record_deprecation_if_matched,
            check_deprecation_alerts_and_exit,
            clear_deprecation_alerts,
        )

        clear_deprecation_alerts()
        _record_deprecation_if_matched("x-ai/grok-4.1-fast", "Grok 4.1 Fast is deprecated.")
        with pytest.raises(SystemExit) as exc_info:
            check_deprecation_alerts_and_exit()
        assert exc_info.value.code == 1
        clear_deprecation_alerts()

    def test_check_returns_silently_on_empty(self) -> None:
        """No deprecation observed → tripwire is a no-op (run completes cleanly)."""
        from metaculus_bot.fallback_openrouter import (
            check_deprecation_alerts_and_exit,
            clear_deprecation_alerts,
        )

        clear_deprecation_alerts()
        # Must not raise SystemExit
        check_deprecation_alerts_and_exit()


class TestStatusCodeClassification:
    """Numeric routing reads the reported ``status_code``, not digits in the message.

    litellm formats the message as ``APIError: {provider} - {raw body}``. An OpenRouter
    body carries a 64-hex key hash (two independent Monte Carlo estimates put the odds
    of it containing one of 401/402/403/429/502/503 at ~8.8%) and, on a moderation
    refusal, up to ~100 characters of OUR OWN PROMPT in ``flagged_input``. Grepping that
    text for status digits reads coincidences as statuses in both directions: a
    coincidental "429" makes a moderation 403 fall back and bill the personal key for a
    call that will refuse again, and a "402" in a forecasting prompt (a dollar figure, a
    bill number) reads as an empty wallet.

    Every litellm exception carries the real status as an int, so where one is reported
    it is the only numeric evidence consulted. Where none is (a plain ``Exception``),
    the substring cues stay live — see ``TestPredicates`` for those.
    """

    @staticmethod
    def _api_error(status: int, message: str) -> Exception:
        from litellm.exceptions import APIError

        return APIError(status_code=status, message=message, llm_provider="openrouter", model="openai/gpt-5.6-sol")

    # A 403 body from the drained donated key, key hash included verbatim.
    _DRAINED = (
        'APIError: OpenrouterException - {"error":{"message":"Key limit exceeded (total limit). '
        "Manage it using https://openrouter.ai/workspaces/default/keys/"
        '8f5af82f134c33c0dbada6e1ce93b780819cc08716001bef5ab4af81791702bd","code":403}}'
    )

    def test_spend_cap_403_still_falls_back(self) -> None:
        """The load-bearing ordering: a DRAINED-key 403 is credit-caused, so it falls back.

        The credit check runs upstream of the 403 veto; making the veto status-based must
        not let it overtake that.
        """
        assert should_retry_with_general_key(self._api_error(403, self._DRAINED)) is True

    def test_moderation_403_does_not_fall_back(self) -> None:
        """A genuine moderation 403 still refuses to swap keys — both keys would refuse."""
        body = 'APIError: OpenrouterException - {"error":{"message":"Blocked by moderation","code":403}}'
        assert should_retry_with_general_key(self._api_error(403, body)) is False

    def test_prompt_echo_402_does_not_read_as_credit(self) -> None:
        """A moderation 403 replaying a prompt containing "402" is not an empty wallet.

        Deliberately omits every moderation WORD cue, so the text-only veto cannot save
        this case — the reported 403 is the only thing that distinguishes it. Left
        unguarded it bills the personal key for a call that will refuse again AND, being
        classified credit-caused, exempts a real content block from CI alerting.
        """
        body = (
            "APIError: OpenrouterException - "
            '{"error":{"message":"Input blocked by provider policy",'
            '"metadata":{"reasons":["policy"],"flagged input":"...HR 402 authorizes $402 million..."},'
            '"code":403}}'
        )
        exc = self._api_error(403, body)
        assert should_retry_with_general_key(exc) is False

    def test_key_hash_digits_do_not_trigger_fallback(self) -> None:
        """A 403 whose key hash happens to contain "429" must not read as a rate limit.

        ~8.8% of key rotations produce a hash like this. The current donated hash
        contains none of the six statuses, which is luck, not design.
        """
        body = (
            'APIError: OpenrouterException - {"error":{"message":"Blocked by content policy. '
            'Key: https://openrouter.ai/keys/a429b502c401d9f3","code":403}}'
        )
        assert should_retry_with_general_key(self._api_error(403, body)) is False

    def test_status_detected_when_message_has_no_digits(self) -> None:
        """Detection no longer depends on the provider spelling the status in prose."""
        assert should_retry_with_general_key(self._api_error(429, "upstream is busy right now")) is True
        assert should_retry_with_general_key(self._api_error(401, "bad credentials")) is True
        assert should_retry_with_general_key(self._api_error(402, "wallet empty")) is True

    def test_infrastructure_statuses_still_refuse_fallback(self) -> None:
        """5xx are infrastructure, not key-scoped — swapping keys cannot help."""
        assert should_retry_with_general_key(self._api_error(502, "upstream died")) is False
        assert should_retry_with_general_key(self._api_error(503, "upstream unavailable")) is False

    def test_route_scoped_404_text_cues_survive(self) -> None:
        """The 404 family is classified by TEXT, and must stay that way.

        "no allowed providers" and the guardrail / data-policy block are both 404s that
        DO warrant a key swap, while a plain 404 (missing model) does not — so the status
        alone cannot decide, and these cues must not be collapsed into a numeric rule.
        """
        allowed = "404 No allowed providers are available for the selected model."
        guardrail = "404 No endpoints available matching your guardrail restrictions and data policy."
        assert should_retry_with_general_key(self._api_error(404, allowed)) is True
        assert should_retry_with_general_key(self._api_error(404, guardrail)) is True
        assert should_retry_with_general_key(self._api_error(404, "model not found")) is False

    def test_non_llm_exception_keeps_textual_classification(self) -> None:
        """A statusless exception is classified exactly as before (regression guard).

        ``requests``-style errors keep their status on ``.response``, and the suite
        asserts this predicate against plain ``Exception("401 Unauthorized")`` strings.
        Neither reports a ``status_code``, so both must keep reaching the text cues.
        """
        assert should_retry_with_general_key(Exception("401 Unauthorized")) is True
        assert should_retry_with_general_key(Exception("HTTP 402 Payment Required")) is True
        assert should_retry_with_general_key(Exception("429 Too Many Requests")) is True
        assert should_retry_with_general_key(Exception("403 Forbidden moderation")) is False
        assert should_retry_with_general_key(Exception("502 Bad Gateway")) is False


class TestCreditClassificationPrecedence:
    """Explicit credit wording > moderation wording > the reported status.

    The three signals disagree in real bodies, so their precedence is the whole design:

    * Explicit phrases outrank everything. A drained-key 403 must fall back, and its body
      can legitimately carry HTTP boilerplate like "Forbidden" — letting a generic
      moderation WORD veto "Key limit exceeded" would resurrect the exact bug this whole
      change exists to fix.
    * Moderation wording outranks the status. A refusal body replays up to ~100 chars of
      our own prompt, so it can contain any digits at all; where wording and status
      disagree, take the conservative branch and do not bill the paid key.
    * The status decides only what is left, replacing the bare-digit cue that
      ``_is_credit_message`` has to fall back on when no status was reported.
    """

    @staticmethod
    def _api_error(status: int, message: str) -> Exception:
        from litellm.exceptions import APIError

        return APIError(status_code=status, message=message, llm_provider="openrouter", model="openai/gpt-5.6-sol")

    def test_moderation_403_echoing_prompt_402_is_not_credit(self) -> None:
        """403 whose ``flagged_input`` replays a prompt containing "402" → not credit.

        The forecasting prompt is full of dollar figures and bill numbers, so this is the
        realistic shape of the poisoning case.
        """
        body = (
            "APIError: OpenrouterException - "
            '{"error":{"message":"Blocked","metadata":{"reasons":["policy"],'
            '"flagged_input":"...HR 402 authorizes $402 million in FY2026..."},"code":403}}'
        )
        exc = self._api_error(403, body)
        assert is_credit_caused_error(exc) is False
        assert should_retry_with_general_key(exc) is False

    def test_moderation_wording_outranks_a_402_status(self) -> None:
        """Where the status says 402 but the wording says refusal, take the safe branch.

        Trusting the status here would bill the personal key for a call that refuses
        again, and would credit-classify a real content block — exempting it from CI
        alerting for the whole suppression window.
        """
        body = 'APIError: OpenrouterException - {"error":{"message":"flagged for moderation","code":402}}'
        exc = self._api_error(402, body)
        assert is_credit_caused_error(exc) is False
        assert should_retry_with_general_key(exc) is False

    def test_explicit_credit_wording_outranks_http_forbidden_boilerplate(self) -> None:
        """A drained-key 403 rendered as "403 Forbidden" still falls back.

        Regression guard on the precedence order: "forbidden" is a moderation CUE, so if
        the veto were allowed to outrank the explicit phrase, the production dry-key
        failure would stop falling back and the ensemble would strand on the dead key.
        """
        body = "APIError: OpenrouterException - 403 Forbidden: Key limit exceeded (total limit)."
        exc = self._api_error(403, body)
        assert is_credit_caused_error(exc) is True
        assert should_retry_with_general_key(exc) is True
