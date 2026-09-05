"""The spend gates on the committed probe scripts are the only thing that makes them safe to keep.

Every credit spend in this repo goes through the operator (AGENTS.md, "Cost discipline"), so a
script anyone may run must be structurally incapable of spending. Two probes have a gate worth
pinning, one class each below.

`gemini_verify.py` is the one script whose whole purpose is to spend: three live calls on the
operator's personal Google AI Studio key, which is why it sits in the PAID list. Between a bare
`uv run python scripts/probes/gemini_verify.py` and those three billed calls there is exactly one
guard, the `--i-accept-spend` refusal, and it was asserted nowhere. So this module pins both halves:
the refusal must exit before the client is even constructed, and the accepted path must make three
calls and no more — a fourth call added later would spend more than the flag's own cost estimate,
and the operator's "go" was given against that estimate.

`fetch_diagnostic.py` is on the FREE list, but its column D runs the production escalation ladder,
whose one paid rung (the Gemini `url_context` read) is gated only on an env flag and a key that a
laptop mirroring prod supplies. `main` forces that flag off before any probe runs; the last class
here pins that the paid reader is never invoked even with the flag and a key set and the rung's
403 trigger population deliberately forced, so the free property is structural rather than a matter
of the environment.

Nothing here touches the network. The gemini client is replaced with a fake whose
`generate_content` returns real `google.genai` response objects, so `tests/conftest.py`'s autouse
`_block_network_egress` fixture has nothing to block and the shapes the probe reads
(`model_version`, `usage_metadata`, `grounding_metadata`, `url_context_metadata`) come from the
SDK's own models rather than from a hand-rolled stub that could drift from them. The fetch-diagnostic
class drives the real provider ladder against a fake refused session, so its 403 is produced by the
fetch layer rather than by the egress guard, whose `ssrf_blocked` status is not a url_context trigger
and would make the assertion pass vacuously.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from google.genai import types as genai_types

from metaculus_bot.constants import RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV
from metaculus_bot.research import resolution_source
from metaculus_bot.research.http_fetch import reset_host_semaphores
from metaculus_bot.research.impersonated_fetch import reset_impersonation_memo
from metaculus_bot.research.robots_policy import reset_robots_cache
from scripts.probes import fetch_diagnostic, gemini_verify
from tests.resolution_source_fakes import _URL, _impersonated, arm_paid_rung, paid_reader, refused_page_with_robots


def _probe_response() -> genai_types.GenerateContentResponse:
    """One response carrying every field the probe's three print paths read.

    Deliberately serves BOTH call shapes: `run_grounded_call` reads `grounding_metadata` and
    `run_url_context_call` reads `url_context_metadata`, and one response answering both keeps the
    fake client a pure call counter rather than a second implementation of the probe's own routing.
    """
    return genai_types.GenerateContentResponse(
        model_version="probe-model",
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=11,
            candidates_token_count=7,
            thoughts_token_count=3,
            tool_use_prompt_token_count=5,
            total_token_count=26,
        ),
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(role="model", parts=[genai_types.Part(text="probe answer")]),
                grounding_metadata=genai_types.GroundingMetadata(web_search_queries=["unemployment rate"]),
                url_context_metadata=genai_types.UrlContextMetadata(
                    url_metadata=[
                        genai_types.UrlMetadata(
                            retrieved_url="https://example.test/a",
                            url_retrieval_status=genai_types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS,
                        )
                    ]
                ),
            )
        ],
    )


class _FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_content(self, *, model: str, contents: object, config: object) -> genai_types.GenerateContentResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _probe_response()


class _FakeClient:
    def __init__(self) -> None:
        self.models = _FakeModels()


class TestGeminiVerifyRefusesWithoutTheFlag:
    """A bare invocation must cost nothing, and must not even build a client."""

    def test_it_exits_two_and_never_builds_a_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built: list[str] = []
        monkeypatch.setattr(gemini_verify, "build_probe_client", lambda: built.append("built"))
        monkeypatch.setattr("sys.argv", ["gemini_verify.py"])

        with pytest.raises(SystemExit) as exc:
            gemini_verify.main()

        # Exit 2, not 1: the same code argparse uses for a usage error, because that is what this
        # is — the flag is required and was not given.
        assert exc.value.code == 2
        # The client construction reads GOOGLE_API_KEY. Refusing BEFORE it runs is what makes the
        # gate independent of whether a key happens to be in the environment.
        assert built == [], "the refusal path built a client, so the gate sits after the spend decision"

    def test_the_cost_estimate_prints_before_the_refusal(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The operator's "go" is given against a stated price, so the refusal has to say what the
        # run would cost rather than only that it declined.
        monkeypatch.setattr(gemini_verify, "build_probe_client", lambda: pytest.fail("client built"))
        monkeypatch.setattr("sys.argv", ["gemini_verify.py"])

        with pytest.raises(SystemExit):
            gemini_verify.main()

        out = capsys.readouterr().out
        assert "Estimated cost of this run" in out
        assert "--i-accept-spend" in out


class TestGeminiVerifySpendsExactlyThreeCalls:
    """With the flag, the probe makes the three calls its cost estimate priced and no others."""

    def test_three_calls_on_the_flagged_path(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = _FakeClient()
        monkeypatch.setattr(gemini_verify, "build_probe_client", lambda: client)
        monkeypatch.setattr("sys.argv", ["gemini_verify.py", "--i-accept-spend"])

        gemini_verify.main()

        assert len(client.models.calls) == 3, (
            f"the probe made {len(client.models.calls)} calls; its cost estimate and its place in "
            "AGENTS.md's paid list both say three, and the operator's approval is given against "
            "that number"
        )
        # Call 1 is the grounded search; calls 2 and 3 are the matched robots pair, and the ONLY
        # difference between them is the target host — that is what makes the comparison a control.
        urls = [c["contents"] for c in client.models.calls[1:]]
        assert gemini_verify.ROBOTS_ALLOWED_URL in str(urls[0])
        assert gemini_verify.ROBOTS_DISALLOWED_URL in str(urls[1])
        # Both retrieved on the fake, so the verdict is the both-retrieved branch.
        assert "Google-Extended hypothesis REFUTED" in capsys.readouterr().out

    def test_the_model_under_test_is_the_one_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _FakeClient()
        monkeypatch.setattr(gemini_verify, "build_probe_client", lambda: client)
        monkeypatch.setattr("sys.argv", ["gemini_verify.py", "--i-accept-spend", "--model", "candidate-9"])

        gemini_verify.main()

        assert {c["model"] for c in client.models.calls} == {"candidate-9"}


class TestFetchDiagnosticForcesThePaidRungOff:
    """Column D runs the production ladder, whose last rung is the paid Gemini ``url_context``
    read. ``main`` forces the flag off before probing, so even a laptop with the flag and key set
    cannot spend. The trigger population is forced deliberately: the fetch layer returns a 403, the
    ``blocked`` status the paid rung fires on, rather than the egress guard's ``ssrf_blocked``,
    which the rung does not trigger on and which would make the assertion pass vacuously.
    """

    @pytest.fixture
    def forced_403_ladder(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        """Arm the paid rung, force the 403 population, and neutralise the intervening free rungs.

        The reader spy is armed (flag on, key set, budget granted) so a call would be recorded; the
        cited page answers 403 via ``refused_page_with_robots`` so the ladder reaches the paid gate;
        and the impersonation, browser and Wayback rungs are stubbed to decline without a network
        hop so the only rung whose gate is under test is ``url_context``. Returns the reader's
        recorded calls.
        """
        reset_host_semaphores()
        reset_robots_cache()
        reset_impersonation_memo()

        def _getaddrinfo(host: str, port: Any, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
            del host, port, args, kwargs
            return [(0, 0, 0, "", ("8.8.8.8", 0))]

        monkeypatch.setattr(resolution_source.socket, "getaddrinfo", _getaddrinfo)

        reader, calls = paid_reader()
        arm_paid_rung(monkeypatch, reader, budget_s=30.0)

        session = refused_page_with_robots()
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        # The impersonated retry fires on the 403; make it decline without a network dial. The
        # ``**kwargs`` swallows ``document_max_bytes`` whether or not the caller passes it yet.
        async def _still_refused(url: str, **kwargs: Any) -> Any:
            del kwargs
            await asyncio.sleep(0)  # a real yield point, so the stub schedules like the transport
            return _impersonated(403, url=url)

        monkeypatch.setattr(resolution_source, "fetch_impersonated", _still_refused)

        async def _no_browser(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            await asyncio.sleep(0)  # the browser rung's declined signal, scheduled like the render

        monkeypatch.setattr(resolution_source, "render_page", _no_browser)
        monkeypatch.setattr(resolution_source, "_WAYBACK_TRIGGER_STATUSES", frozenset())
        return calls

    async def test_the_forced_population_would_otherwise_reach_the_paid_reader(
        self, forced_403_ladder: list[dict[str, Any]]
    ) -> None:
        """The control: with the flag left on, the 403 population reaches the paid reader, so the
        guard test below is not passing vacuously on a population that never triggers the rung."""
        result = await fetch_diagnostic.probe_ladder(_URL)

        assert result.status == "success"
        assert result.route == "url_context"
        assert len(forced_403_ladder) == 1

    async def test_main_forcing_the_flag_off_keeps_the_paid_reader_unspent(
        self, forced_403_ladder: list[dict[str, Any]]
    ) -> None:
        """The guard: once ``main`` has forced the flag off, the same 403 population never spends,
        because the rung declines on its flag before it looks for its key or its robots policy."""
        fetch_diagnostic._disable_the_paid_rung()

        assert os.environ[RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV] == "false"
        result = await fetch_diagnostic.probe_ladder(_URL)

        assert result.status == "blocked"
        assert forced_403_ladder == []

    async def test_main_runs_the_force_off_before_any_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The placement: ``main`` disables the paid rung before it opens a session or probes, so
        the force-off cannot be raced by a probe that ran first."""
        order: list[str] = []
        monkeypatch.setenv(RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV, "true")

        def _record_force() -> None:
            order.append("force")
            os.environ[RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV] = "false"

        monkeypatch.setattr(fetch_diagnostic, "_disable_the_paid_rung", _record_force)

        class _NoSession:
            async def __aenter__(self) -> _NoSession:
                await asyncio.sleep(0)
                order.append("session")
                return self

            async def __aexit__(self, *_exc: Any) -> None:
                await asyncio.sleep(0)

        monkeypatch.setattr(fetch_diagnostic, "_get_session", _NoSession)

        async def _no_egress_ip(_session: Any) -> str:
            await asyncio.sleep(0)
            return "203.0.113.7"

        async def _no_probes(_session: Any, _pacer: Any) -> list[Any]:
            await asyncio.sleep(0)
            order.append("probes")
            return []

        monkeypatch.setattr(fetch_diagnostic, "read_egress_ip", _no_egress_ip)
        monkeypatch.setattr(fetch_diagnostic, "run_probes", _no_probes)

        await fetch_diagnostic.main()

        assert order[0] == "force"
        assert order.index("force") < order.index("probes")
        assert os.environ[RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV] == "false"
