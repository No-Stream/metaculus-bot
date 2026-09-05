"""The spend gate on `scripts/probes/gemini_verify.py` is the only thing that makes it safe to keep.

Every credit spend in this repo goes through the operator (AGENTS.md, "Cost discipline"), and
`gemini_verify.py` is the one committed script whose whole purpose is to spend: three live calls on
the operator's personal Google AI Studio key, which is why it sits in the PAID list. Between a bare
`uv run python scripts/probes/gemini_verify.py` and those three billed calls there is exactly one
guard, the `--i-accept-spend` refusal, and it was asserted nowhere. So this module pins both halves:
the refusal must exit before the client is even constructed, and the accepted path must make three
calls and no more — a fourth call added later would spend more than the flag's own cost estimate,
and the operator's "go" was given against that estimate.

Nothing here touches the network. The client is replaced with a fake whose `generate_content`
returns real `google.genai` response objects, so `tests/conftest.py`'s autouse
`_block_network_egress` fixture has nothing to block and the shapes the probe reads
(`model_version`, `usage_metadata`, `grounding_metadata`, `url_context_metadata`) come from the
SDK's own models rather than from a hand-rolled stub that could drift from them.
"""

from __future__ import annotations

import pytest
from google.genai import types as genai_types

from scripts.probes import gemini_verify


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
