"""Concurrency invariants for publish hardening: the event loop, and the shared post patch.

Companion to the retry/timeout pins in ``tests/test_wall_clock_abort.py``, which cover
what one publish call does. These cover what publish does to everything ELSE running at
the same time — the two ways the hardening's own machinery leaked into its neighbors.

1. **Event-loop block.** The wrapper runs the POST on a worker thread but waits with
   ``future.result(timeout=...)``, a synchronous block, and its only caller is ft's
   ``async def publish_report_to_metaculus`` (which calls both ``post_*`` methods with no
   await). ft runs every question of a batch under one ``asyncio.gather``, so a pinned
   loop freezes every sibling question's tasks while their wall-clock deadlines keep
   advancing — a forecaster then gets cancelled on time it never got to use and is
   recorded as a soft-deadline drop. Real cost per question is two
   ``_sleep_between_requests`` calls (3.5-4.5s each) plus network, and up to
   ``PUBLISH_POST_TIMEOUT * (PUBLISH_POST_RETRIES + 1)`` per POST if Metaculus stalls.

2. **Lost-update leak on the shared ``requests.post`` patch.** Save/restore of a module
   global is only correct under strict LIFO nesting. The timeout-and-retry path breaks
   that by construction: ``future.cancel()`` returns False on a running future, so a
   timed-out orphan and its retry are both alive in the shared 4-thread pool, both inside
   the context manager. ``fetch_hardening`` already reasoned this out for its GET twin
   and installs once for exactly this reason.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
import requests
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.helpers import metaculus_client as _ft_metaculus_client
from forecasting_tools.helpers.metaculus_client import MetaculusClient

from metaculus_bot import publish_hardening
from metaculus_bot.constants import PUBLISH_POST_RETRIES
from metaculus_bot.http_status import http_status_from_exception
from scripts.telemetry.markers import parse_log_text

# A real Metaculus publish URL: the forced-timeout override is scoped to the Metaculus
# host on purpose (metaculus_client.requests IS the global requests module, so an
# unscoped permanent patch would re-time every other POST in the process).
_METACULUS_POST_URL = "https://www.metaculus.com/api/questions/forecast/"

# One GHA-shaped log line's prefix plus the harvest metadata, so a WARN this module
# actually emitted can be run through the real telemetry parser (the run logs the
# archive harvests are the workflow's `python | tee` output, not caplog's messages).
_LOG_PREFIX = "2026-08-03 12:05:06,123 - metaculus_bot.publish_hardening - WARNING - "
_HARVEST_META = {
    "run_id": "999",
    "workflow": "tournament",
    "artifact": "research-999",
    "run_date": "2026-08-03T12:00:00Z",
    "log_file": "run.log",
}


@pytest.fixture
def pristine_report_publish(monkeypatch: pytest.MonkeyPatch):
    """Restore each report type's publish method (and drop the offload marker) after the test."""
    for report_type in publish_hardening._PATCHED_REPORT_TYPES:
        monkeypatch.setattr(
            report_type,
            publish_hardening._PUBLISH_METHOD,
            report_type.__dict__[publish_hardening._PUBLISH_METHOD],
        )
    return


class TestPublishDoesNotBlockTheEventLoop:
    """A publish in flight must not stop sibling questions' tasks from running."""

    @pytest.mark.asyncio
    async def test_async_publish_seam_yields_to_the_loop(
        self, monkeypatch: pytest.MonkeyPatch, pristine_report_publish: None
    ) -> None:
        # Drive the REAL seam ft calls: report.publish_report_to_metaculus, an async
        # method whose body issues both POSTs synchronously. A heartbeat ticking every
        # 10ms must keep ticking through a 1s publish; before the fix a 2s publish
        # yielded 4 ticks over 2.21s instead of ~221.
        publish_seconds = 1.0
        heartbeat_period = 0.01

        # Stand in for ft's async publish whose body is fully synchronous — the shape
        # that pins the loop. Installed BEFORE the offload so the offload wraps it.
        async def fake_async_publish(self: Any, metaculus_client: Any = None) -> str:
            time.sleep(publish_seconds)  # noqa: ASYNC251  # the blocking body IS what this test pins
            return "published"

        monkeypatch.setattr(BinaryReport, publish_hardening._PUBLISH_METHOD, fake_async_publish)
        publish_hardening.apply_report_publish_offload()

        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(heartbeat_period)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        await asyncio.sleep(heartbeat_period * 2)
        started = time.monotonic()
        result = await BinaryReport.publish_report_to_metaculus(object())  # type: ignore[arg-type]
        elapsed = time.monotonic() - started
        beat.cancel()

        assert result == "published", "the offload must return the wrapped call's value"
        assert elapsed >= publish_seconds * 0.8, "the publish should still have taken its full duration"
        expected_ticks = publish_seconds / heartbeat_period
        # Generous floor (half the ideal tick count) so the assertion is about "the loop
        # kept running", not scheduler precision. Pre-fix this was ~4% of expected.
        assert ticks >= expected_ticks * 0.5, (
            f"event loop was starved during publish: {ticks} heartbeat ticks over {elapsed:.2f}s, "
            f"expected roughly {expected_ticks:.0f}. The publish seam is blocking the loop, so every "
            "sibling question in ft's asyncio.gather is frozen while its deadlines advance."
        )

    @pytest.mark.asyncio
    async def test_offload_propagates_exceptions(
        self, monkeypatch: pytest.MonkeyPatch, pristine_report_publish: None
    ) -> None:
        # A publish failure must still reach ft's caller: moving work to a thread must
        # not swallow the error that tells us a forecast never landed.
        async def failing_publish(self: Any, metaculus_client: Any = None) -> None:
            raise RuntimeError("metaculus rejected the forecast")

        monkeypatch.setattr(BinaryReport, publish_hardening._PUBLISH_METHOD, failing_publish)
        publish_hardening.apply_report_publish_offload()

        with pytest.raises(RuntimeError, match="metaculus rejected the forecast"):
            await BinaryReport.publish_report_to_metaculus(object())  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_offload_forwards_the_client_argument(
        self, monkeypatch: pytest.MonkeyPatch, pristine_report_publish: None
    ) -> None:
        # ft passes its own client instance (forecast_bot.py: publish_report_to_metaculus(
        # metaculus_client=self.metaculus_client)); dropping it would silently build a
        # second client and bypass the bot's own.
        seen: list[Any] = []

        async def recording_publish(self: Any, metaculus_client: Any = None) -> None:
            seen.append(metaculus_client)

        monkeypatch.setattr(BinaryReport, publish_hardening._PUBLISH_METHOD, recording_publish)
        publish_hardening.apply_report_publish_offload()

        sentinel = object()
        await BinaryReport.publish_report_to_metaculus(object(), metaculus_client=sentinel)  # type: ignore[arg-type]
        assert seen == [sentinel]

    def test_offload_is_idempotent(self, monkeypatch: pytest.MonkeyPatch, pristine_report_publish: None) -> None:
        publish_hardening.apply_report_publish_offload()
        first = BinaryReport.__dict__[publish_hardening._PUBLISH_METHOD]
        publish_hardening.apply_report_publish_offload()
        assert BinaryReport.__dict__[publish_hardening._PUBLISH_METHOD] is first

    def test_every_published_report_type_is_offloaded(self) -> None:
        # All three question types the bot forecasts publish through their own report
        # class, so missing one silently leaves that type blocking the loop.
        for question_type in (BinaryQuestion, NumericQuestion, MultipleChoiceQuestion):
            report_type = DataOrganizer.get_report_type_for_question_type(question_type)
            assert report_type in publish_hardening._PATCHED_REPORT_TYPES, (
                f"{report_type.__name__} publishes but is not in _PATCHED_REPORT_TYPES"
            )

    def test_each_report_type_defines_its_own_publish(self) -> None:
        # The offload marker is per-function rather than per-class precisely because
        # these share a base; if one ever stopped defining its own publish, patching it
        # would silently shadow the base for its siblings.
        for report_type in publish_hardening._PATCHED_REPORT_TYPES:
            assert publish_hardening._PUBLISH_METHOD in report_type.__dict__, (
                f"{report_type.__name__} no longer defines {publish_hardening._PUBLISH_METHOD}; "
                "the ft publish seam moved"
            )


class TestSocketTimeoutPatchDoesNotLeak:
    """The forced-timeout patch must survive overlapping publishes."""

    def test_timeout_override_is_installed_once_not_per_call(self) -> None:
        # The structural fix: there is no per-call context manager left to interleave.
        # A save/restore around each call cannot be made safe here, because
        # future.cancel() returns False on a running future, so a timed-out orphan and
        # its retry are both inside the window at once (the module docstring's
        # abandoned-worker path). fetch_hardening reached the same conclusion for GET.
        assert not hasattr(publish_hardening, "_inject_socket_timeout"), (
            "the per-call save/restore context manager must be gone — it leaks the "
            "process-global requests.post under the timeout-orphan-plus-retry interleaving"
        )
        assert hasattr(publish_hardening, "_install_post_timeout_override")

    def test_overlapping_wrapped_publishes_leave_requests_post_intact(self) -> None:
        # The reachable interleaving under the OLD design: worker B enters while A is
        # inside, A exits first and restores what IT saw (the real post), so B's wrapper
        # stays installed forever after B exits. Reproduced at HEAD before the fix
        # (`post is real` -> False), with one wrapper layer accumulating per occurrence.
        # Under the install-once design the wrappers never toggle, so two overlapping
        # publishes leave the module exactly as they found it.
        real_post = _ft_metaculus_client.requests.post
        b_inside = threading.Event()
        calls: list[float | None] = []

        def slow_post_via_module(hold_s: float, wait_for_b: bool) -> None:
            if wait_for_b:
                b_inside.wait(3.0)
            else:
                time.sleep(0.1)
                b_inside.set()
            _ft_metaculus_client.requests.post(_METACULUS_POST_URL, json={})
            time.sleep(hold_s)

        def fake_post(*args: Any, **kwargs: Any) -> None:
            calls.append(kwargs.get("timeout"))

        try:
            _ft_metaculus_client.requests.post = fake_post  # type: ignore[assignment]
            publish_hardening._install_post_timeout_override(20.0)
            wrapped_a = publish_hardening._wrap_with_timeout_retry("a", slow_post_via_module)
            wrapped_b = publish_hardening._wrap_with_timeout_retry("b", slow_post_via_module)
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(wrapped_a, 0.05, True)
                second = pool.submit(wrapped_b, 0.3, False)
                first.result()
                second.result()

            assert calls == [20.0, 20.0], f"both overlapping publishes should carry the forced timeout; got {calls}"
        finally:
            _ft_metaculus_client.requests.post = real_post  # type: ignore[assignment]

        assert _ft_metaculus_client.requests.post is real_post

    def test_timeout_is_forced_on_metaculus_and_left_alone_elsewhere(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Two halves of one invariant. 0.2.92 always passes timeout=self.timeout (30s),
        # so the override must OVERRIDE rather than setdefault or the tighter publish
        # ceiling never applies. But because the install is permanent AND
        # metaculus_client.requests IS the global requests module, it must also be
        # scoped: exa_py, litellm's Databricks path, and huggingface_hub all POST
        # through the same function, and silently lowering their timeout to a publish
        # ceiling can only manufacture failures.
        seen: list[tuple[str, Any]] = []

        def fake_post(url: str, *args: Any, **kwargs: Any) -> None:
            seen.append((url, kwargs.get("timeout")))

        monkeypatch.setattr(_ft_metaculus_client.requests, "post", fake_post)
        publish_hardening._install_post_timeout_override(7.5)
        try:
            _ft_metaculus_client.requests.post(_METACULUS_POST_URL, timeout=30)
            _ft_metaculus_client.requests.post("https://api.exa.ai/search", timeout=120)
            # Keyword form too, so a future ft call style still scopes correctly.
            _ft_metaculus_client.requests.post(url=_METACULUS_POST_URL, timeout=30)
        finally:
            monkeypatch.undo()

        assert seen == [
            (_METACULUS_POST_URL, 7.5),
            ("https://api.exa.ai/search", 120),
            (_METACULUS_POST_URL, 7.5),
        ], f"forced timeout must apply to Metaculus POSTs only; got {seen}"


class TestHardeningStillCoversThePostSeam:
    """Guard against the offload accidentally replacing the timeout/retry layer."""

    def test_post_methods_are_still_patched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in publish_hardening._PATCHED_METHODS:
            monkeypatch.setattr(MetaculusClient, name, MetaculusClient.__dict__[name])
        if hasattr(MetaculusClient, publish_hardening._SENTINEL):
            monkeypatch.setattr(MetaculusClient, publish_hardening._SENTINEL, False)
            delattr(MetaculusClient, publish_hardening._SENTINEL)
        else:
            monkeypatch.setattr(MetaculusClient, publish_hardening._SENTINEL, False, raising=False)
            delattr(MetaculusClient, publish_hardening._SENTINEL)

        real_post = _ft_metaculus_client.requests.post
        before = {name: MetaculusClient.__dict__[name] for name in publish_hardening._PATCHED_METHODS}
        try:
            publish_hardening.apply_publish_hardening()
            for name in publish_hardening._PATCHED_METHODS:
                assert MetaculusClient.__dict__[name] is not before[name], f"{name} must still be wrapped"
            # And the one entry point installs all three layers, so nothing is left to
            # a caller to remember.
            assert _ft_metaculus_client.requests.post is not real_post, "layer 1 must be installed"
            assert getattr(
                BinaryReport.__dict__[publish_hardening._PUBLISH_METHOD],
                publish_hardening._REPORT_SENTINEL,
                False,
            ), "layer 3 must be installed"
        finally:
            _ft_metaculus_client.requests.post = real_post  # type: ignore[assignment]
            for report_type in publish_hardening._PATCHED_REPORT_TYPES:
                raw = report_type.__dict__[publish_hardening._PUBLISH_METHOD]
                original = getattr(raw, "__wrapped__", None)
                if original is not None:
                    setattr(report_type, publish_hardening._PUBLISH_METHOD, original)


class TestPublishAttemptFailureCounter:
    """The retry wrapper's terminal ``raise last_exc`` is the one place a publish
    ATTEMPT failure becomes countable. Before the counter, q45085's 405-closed
    (2026-08-03) burned both attempts and every degradation counter still read
    zero — ``questions_failed_to_publish`` only sees the min-forecasters floor.
    Telemetry only: the raise itself must be untouched."""

    def test_exhausted_retries_bump_the_counter_once_and_reraise(self) -> None:
        def always_405(*_args: Any, **_kwargs: Any) -> None:
            raise requests.HTTPError("Error while posting prediction: Status code: 405")

        wrapped = publish_hardening._wrap_with_timeout_retry("_post_question_prediction", always_405)
        publish_hardening.reset_publish_attempt_failures()
        try:
            with pytest.raises(requests.HTTPError):
                wrapped()
            # One exhausted publish = ONE counted failure, not one per attempt.
            assert publish_hardening.publish_attempt_failures() == 1
        finally:
            publish_hardening.reset_publish_attempt_failures()

    def test_retry_recovery_leaves_the_counter_at_zero(self) -> None:
        attempts: list[int] = []

        def flaky_then_ok(*_args: Any, **_kwargs: Any) -> str:
            attempts.append(1)
            if len(attempts) == 1:
                raise requests.ConnectionError("transient")
            return "ok"

        wrapped = publish_hardening._wrap_with_timeout_retry("post_question_comment", flaky_then_ok)
        publish_hardening.reset_publish_attempt_failures()
        try:
            assert wrapped() == "ok"
            # A retried-then-successful publish is not a failure.
            assert publish_hardening.publish_attempt_failures() == 0
        finally:
            publish_hardening.reset_publish_attempt_failures()

    def test_reset_zeroes_the_module_counter(self) -> None:
        publish_hardening._bump_publish_attempt_failure()
        publish_hardening.reset_publish_attempt_failures()
        assert publish_hardening.publish_attempt_failures() == 0

    def test_concurrent_exhausted_publishes_both_count(self) -> None:
        # Layer 3 runs each report's publish on its own worker thread and ft's gather
        # can have several questions publishing at once, so the increment really is
        # reached concurrently. ``+=`` on a module global is interruptible between
        # bytecodes; unlocked, a lost update would let a two-question publish outage
        # read as one.
        def always_fails(*_args: Any, **_kwargs: Any) -> None:
            raise requests.ConnectionError("dead socket")

        wrapped = publish_hardening._wrap_with_timeout_retry("post_question_comment", always_fails)
        publish_hardening.reset_publish_attempt_failures()
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(wrapped) for _ in range(40)]
                for future in futures:
                    with pytest.raises(requests.ConnectionError):
                        future.result()
            assert publish_hardening.publish_attempt_failures() == 40
        finally:
            publish_hardening.reset_publish_attempt_failures()


class TestNonRetryable4xxIsNotRetried:
    """A 4xx that is not 408/429 is a permanent verdict on THIS request, so a second
    identical POST can only burn wall clock inside the per-question budget. q45085's
    405-closed was retried 4s later and failed identically. 5xx and transport-level
    failures still retry exactly as before."""

    @staticmethod
    def _ft_style_error(status: int) -> requests.HTTPError:
        """Reproduce ft's re-raise shape: ``raise_for_status_with_additional_info``
        builds a BRAND-NEW HTTPError from a message string, so the exception the
        wrapper catches has no ``.response`` and the status lives only in the text."""
        return requests.HTTPError(
            f"HTTPError. Url: https://www.metaculus.com/api/questions/forecast/. Status code: {status}. "
            'Response reason: Method Not Allowed. Response text: {"error":"Question 45085 is already '
            'closed to forecasting !"}. Response JSON: None.'
        )

    def _count_attempts(self, exc: BaseException) -> int:
        attempts: list[int] = []

        def always_raises(*_args: Any, **_kwargs: Any) -> None:
            attempts.append(1)
            raise exc

        wrapped = publish_hardening._wrap_with_timeout_retry("_post_question_prediction", always_raises)
        publish_hardening.reset_publish_attempt_failures()
        try:
            with pytest.raises(type(exc)):
                wrapped()
            # Either way the terminal counter fires exactly once — only the attempt
            # count changes, never the telemetry or the raise.
            assert publish_hardening.publish_attempt_failures() == 1
        finally:
            publish_hardening.reset_publish_attempt_failures()
        return len(attempts)

    def test_405_already_closed_is_attempted_once(self) -> None:
        assert self._count_attempts(self._ft_style_error(405)) == 1

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 405])
    def test_permanent_4xx_statuses_are_attempted_once(self, status: int) -> None:
        assert self._count_attempts(self._ft_style_error(status)) == 1

    @pytest.mark.parametrize("status", [408, 429])
    def test_transient_4xx_statuses_still_retry(self, status: int) -> None:
        assert self._count_attempts(self._ft_style_error(status)) == PUBLISH_POST_RETRIES + 1

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_5xx_still_retries(self, status: int) -> None:
        assert self._count_attempts(self._ft_style_error(status)) == PUBLISH_POST_RETRIES + 1

    def test_statusless_transport_error_still_retries(self) -> None:
        # Unclassifiable means retry: transport flakiness is what the budget is FOR,
        # and refusing to retry something we couldn't read would be the regression.
        assert self._count_attempts(requests.ConnectionError("connection reset by peer")) == PUBLISH_POST_RETRIES + 1

    def test_status_is_read_off_a_real_response_when_present(self) -> None:
        # Belt and braces for a caller that doesn't route through ft's re-raise: a
        # genuine raise_for_status HTTPError carries the response object.
        response = requests.Response()
        response.status_code = 405
        assert http_status_from_exception(requests.HTTPError(response=response)) == 405

    def test_status_is_recovered_through_fts_cause_chain(self) -> None:
        # ft does `raise HTTPError(message) from e`, so the original (which has the
        # response) survives as __cause__ even when the message text changes shape.
        response = requests.Response()
        response.status_code = 403
        original = requests.HTTPError(response=response)
        rewrapped = requests.HTTPError("some rephrased message with no status in it")
        rewrapped.__cause__ = original
        assert http_status_from_exception(rewrapped) == 403

    def test_a_three_digit_number_in_a_payload_echo_is_not_read_as_a_status(self) -> None:
        # The substring trap that bit the OpenRouter credit classifier: OpenRouter
        # replays our own prompt text, and forecasting prompts are full of figures.
        # Only ft's literal "Status code: NNN" phrasing counts.
        echoed = requests.HTTPError('Response text: {"flagged_input":"will the index close above 405 by June"}')
        assert http_status_from_exception(echoed) is None
        assert publish_hardening._is_retryable(echoed) is True

    def test_the_forgone_retry_logs_a_line_the_harvester_ignores(self, caplog: pytest.LogCaptureFixture) -> None:
        """Emitter half of the telemetry pin. The publish_hardening MarkerSpec is
        anchored on the ``attempt N/M`` shape, so this second WARN deliberately carries
        no such clause — appending the text to the attempt line instead would have
        stopped EVERY publish failure from harvesting. test_telemetry_markers pins the
        harvester against a hand-written literal; this drives the real emitter, so the
        two cannot drift apart without a test going red."""

        def always_405(*_args: Any, **_kwargs: Any) -> None:
            raise self._ft_style_error(405)

        wrapped = publish_hardening._wrap_with_timeout_retry("_post_question_prediction", always_405)
        publish_hardening.reset_publish_attempt_failures()
        try:
            with (
                caplog.at_level(logging.WARNING, logger="metaculus_bot.publish_hardening"),
                pytest.raises(requests.HTTPError),
            ):
                wrapped()
        finally:
            publish_hardening.reset_publish_attempt_failures()

        forgone = [message for message in caplog.messages if "not retrying" in message]
        assert len(forgone) == 1, caplog.messages
        assert forgone[0] == (
            "PUBLISH_HARDENING: _post_question_prediction not retrying status 405 "
            "— a second identical POST cannot succeed"
        )
        harvested = parse_log_text(f"{_LOG_PREFIX}{forgone[0]}\n", **_HARVEST_META)
        assert harvested["publish_hardening"] == [], (
            "the forgone-retry WARN must not harvest as a publish attempt — it carries no attempt N/M clause"
        )
        # The attempt WARN that DOES carry the clause still harvests, so the exclusion
        # above is about this line's shape rather than a broken spec.
        attempt = [message for message in caplog.messages if "attempt 1/" in message]
        assert len(attempt) == 1, caplog.messages
        assert parse_log_text(f"{_LOG_PREFIX}{attempt[0]}\n", **_HARVEST_META)["publish_hardening"] != []

    def test_a_retried_status_logs_no_forgone_retry_line(self, caplog: pytest.LogCaptureFixture) -> None:
        # The line is emitted only where a retry was actually given up; on a 429 (and on
        # the final attempt of anything) saying "not retrying" would be noise or a lie.
        def always_429(*_args: Any, **_kwargs: Any) -> None:
            raise self._ft_style_error(429)

        wrapped = publish_hardening._wrap_with_timeout_retry("post_question_comment", always_429)
        publish_hardening.reset_publish_attempt_failures()
        try:
            with (
                caplog.at_level(logging.WARNING, logger="metaculus_bot.publish_hardening"),
                pytest.raises(requests.HTTPError),
            ):
                wrapped()
        finally:
            publish_hardening.reset_publish_attempt_failures()

        assert not any("not retrying" in message for message in caplog.messages), caplog.messages
