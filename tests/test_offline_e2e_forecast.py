"""Offline full-pipeline e2e forecast test — the breaking-dependency tripwire.

WHAT THIS DEFENDS AGAINST
=========================
The litellm 1.92 crash (``acompletion(tools=...)`` eagerly imports a proxy MCP
handler needing ``fastapi``, which we don't install) fired only when the real
call EXECUTED with ``tools=`` — a plain ``import litellm`` smoke test would NOT
catch it, and the agentic gap-fill v2 loop (the only ``tools=`` caller) soft-
failed to "" so nothing went red in CI. This test drives the REAL code paths of
every external dependency on the forecast critical path — research, forecaster
fan-out, aggregation, gap-fill v1 AND v2 — stubbing ONLY the outermost network
boundary (the socket-opening client call). If a future dep upgrade breaks an
import / transform / call-path anywhere in that stack, this test goes red.

DESIGN
======
* **LLM seam — a routing wrapper, not a global mock.** The pipeline makes many
  heterogeneous LLM calls that each need a DIFFERENT valid canned response
  (binary/numeric/MC forecaster blocks, summarizer prose, gap-fill v1 gap JSON,
  native-search prose, the agentic v2 driver's tool calls, the parser salvage).
  A single ``litellm.mock_response`` can't satisfy all of them. Instead we patch
  the two ``acompletion`` chokepoints with a ROUTER that inspects the outgoing
  ``model`` + ``messages`` (the system-prompt text identifies the call type),
  selects the matching canned text, and forwards to the REAL
  ``litellm.acompletion(**kwargs, mock_response=<routed>)``. Forwarding to real
  litellm is load-bearing: it executes all real litellm import/transform/tools-
  path code (catching the fastapi class of bug) while short-circuiting only the
  network. The v2 driver path additionally uses ``mock_tool_calls`` so the loop
  gets a real tool-call-shaped response.

* **Provider seams — stub at the lowest client boundary.** Each research
  provider's external client (AskNews SDK, google-genai Client, aiohttp session,
  Exa client) is stubbed so OUR formatting/parsing code runs for real but no
  socket opens. The autouse network-egress guard in conftest.py is the
  belt-and-suspenders backstop: if a stub is missed, the test trips the guard
  (a clear RuntimeError) rather than spending real money.

PATHS FULLY EXERCISED (all three question types complete end to end)
====================================================================
Binary, numeric, and MC each run the full ``forecast_questions`` →
``_research_and_make_predictions`` pipeline offline and produce a published-shape
``ForecastReport``. Real code executed per question: the whole research fan-out
(AskNews + summarizer, native search, Gemini grounded, financial-data classifier,
prediction-market snapshot across 4 platforms, resolution-source fetch), gap-fill
v1 (analyzer + parallel resolvers), gap-fill v2 (the agentic tool loop, driving
REAL ``litellm.acompletion`` with ``tools=`` — the fastapi tripwire), the
forecaster fan-out through the value-extraction ladder (rung=block), and CDF/MC
post-processing + aggregation. Stacking is prod-disabled (the three
``*_STACKING_ENABLED`` flags default off and are NOT set here), so the median/
skipped aggregation path runs — the production-default path.

Partially exercised: the stacker LLM itself (crux → targeted search → stacker)
is not driven, because prod runs with stacking disabled; the conditional-stacking
mechanism is covered by ``tests/test_conditional_stacking.py``. Rendered-fetch
(headless Chromium) and read_document (Gemini url_context) inside the agentic loop
only fire if the driver requests them; the scripted driver concludes without them,
so those specific rungs are not covered here (they are unit-tested in
``tests/test_agentic_tools.py``).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asknews_sdk
import litellm
import pytest
from forecasting_tools import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.ai_models import general_llm as ft_general_llm
from forecasting_tools.data_models.forecast_report import ForecastReport

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.llm_configs import (
    DISAGREEMENT_ANALYZER_LLM,
    FORECASTER_LLMS,
    PARSER_LLM,
    RESEARCHER_LLM,
    STACKER_LLM,
    SUMMARIZER_LLM,
)
from metaculus_bot.research import gemini_search, prediction_market, resolution_source
from metaculus_bot.research import providers as research_providers
from metaculus_bot.research.agentic import llm as agentic_llm

_NOW = datetime.now(timezone.utc)
_OPEN = _NOW - timedelta(days=30)
_RESOLVE = _NOW + timedelta(days=180)

# A fetchable URL in resolution criteria exercises the resolution-source provider
# (extract → fetch → trafilatura extract). example.com is RFC-2606 reserved; the
# aiohttp session is stubbed so no socket opens and the SSRF-guard getaddrinfo is
# patched to a public IP.
_RESOLUTION_URL = "https://data.example.gov/unemployment-report"


# ---------------------------------------------------------------------------
# Canned forecaster / stacker blocks — parse at value_extraction rung 1 (block).
# Kept byte-for-byte in sync with the STRUCTURED FORECAST schemas
# (metaculus_bot/structured_output_schema.py) so extract_binary/mc/numeric all
# land rung=block. Mirrors tests/pipeline_test_helpers.py's canned reasonings.
# ---------------------------------------------------------------------------

_CANNED_BINARY = """\
## Analysis
The status quo holds; no post-open trigger event has occurred. Base rate is low.

```json
{"question_type": "binary", "posterior_prob": 0.22}
```
"""

_CANNED_NUMERIC = """\
## Analysis
Recent measurements cluster tightly; the distribution is centered near the latest value.
OUTCOME_TYPE: CONTINUOUS

```json
{
  "question_type": "numeric",
  "declared_percentiles": {
    "0.01": 3.0, "0.025": 3.2, "0.05": 3.4, "0.1": 3.6, "0.2": 3.8, "0.4": 4.1, "0.5": 4.3,
    "0.6": 4.5, "0.8": 5.0, "0.9": 5.6, "0.95": 6.2, "0.975": 7.0, "0.99": 7.8
  },
  "outcome_type": "continuous"
}
```
"""

_CANNED_MC = """\
## Analysis
Option A carries institutional momentum; B is the eroding status quo; C is a tail.

```json
{"question_type": "multiple_choice", "option_probs": {"Option A": 0.45, "Option B": 0.40, "Option C": 0.15}}
```
"""

_CANNED_SUMMARY_PROSE = (
    "Newest directly-relevant article: 2026-04-14. The unemployment rate stood at 4.1% in the "
    "most recent release [B: Reuters]. Initial jobless claims trended up modestly."
)

_CANNED_NATIVE_SEARCH_PROSE = (
    "The Bureau of Labor Statistics reported the unemployment rate at 4.1% in April 2026 "
    "[BLS](https://www.bls.gov/news.release/empsit.nr0.htm)."
)

# gap-fill v1 analyzer: a single-gap JSON payload (parse_gap_list).
_CANNED_GAP_ANALYZER = json.dumps(
    {"gaps": [{"gap": "Latest BLS release date", "why_matters": "Anchors the level", "search_query": "BLS release"}]}
)

# Providers that MUST report `ok` in the diagnostics block for these questions.
# Verified empirically (all three question types, INFO logs) to be identical:
# asknews/native_search/gemini_search/resolution_source all land `ok`, while
# financial_data + prediction_market legitimately return `empty` (non-financial
# question, no market matches on the stubbed empty payloads) — so those two are
# NOT asserted `ok`. A future dep break that errors any required provider is
# swallowed into status="errored" by the orchestrator, so this is the direct
# catch for the non-litellm dependency class (google-genai / asknews / aiohttp).
_REQUIRED_OK_PROVIDERS = frozenset({"asknews", "native_search", "gemini_search", "resolution_source"})


# ---------------------------------------------------------------------------
# LLM router — one wrapper for both acompletion chokepoints.
# ---------------------------------------------------------------------------


def _messages_text(kwargs: dict[str, Any]) -> str:
    """Concatenate all message contents so we can sniff the call type from the prompt."""
    parts: list[str] = []
    for msg in kwargs.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # Vision messages carry a list of content parts (text + image_url dicts).
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    return "\n".join(parts)


def _route_general_llm(kwargs: dict[str, Any]) -> str:
    """Pick the canned response for a forecasting-tools GeneralLlm call.

    Routes on prompt content. The base forecaster prompts embed a
    'STRUCTURED FORECAST' block schema and a per-type cue (percentiles / options
    line); the summarizer/native-search/gap-fill-analyzer calls carry their own
    distinctive text. Order matters — most-specific first.
    """
    text = _messages_text(kwargs)
    lower = text.lower()

    # A forecaster (or stacker) call — the ONLY calls carrying the fenced
    # STRUCTURED FORECAST block instruction. Checked FIRST because the base MC /
    # numeric prompts also contain "Intelligence Briefing" (which would otherwise
    # collide with the summarizer signal below). Pick the block by question type:
    # the numeric prompt talks about percentiles; the MC prompt has an
    # "Options (in resolution order):" line; else binary.
    if "STRUCTURED FORECAST" in text:
        if "percentile" in lower:
            return _CANNED_NUMERIC
        if "options (in resolution order)" in lower:
            return _CANNED_MC
        return _CANNED_BINARY

    # gap-fill v1 analyzer: asks for a JSON {"gaps": [...]} list.
    if "research-quality auditor" in lower or '"gaps"' in text:
        return _CANNED_GAP_ANALYZER

    # AskNews summarizer: builds an "intelligence briefing" from raw <research>.
    if "intelligence briefing" in lower or "<research>" in lower:
        return _CANNED_SUMMARY_PROSE

    # Everything else — native / targeted web search, gap-fill resolver,
    # perplexity fallback — carries a "research assistant" framing and no block.
    return _CANNED_NATIVE_SEARCH_PROSE


def _install_llm_router(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch BOTH acompletion chokepoints to route → real litellm + mock_response.

    Chokepoint 1: ``forecasting_tools.ai_models.general_llm.acompletion`` — every
    GeneralLlm call (forecasters, stacker, parser, summarizer, native search).
    Chokepoint 2: ``metaculus_bot.research.agentic.llm.acompletion`` — the raw-
    litellm agentic v2 driver (the ONLY ``tools=`` caller).

    Both forward to the REAL ``litellm.acompletion`` with ``mock_response`` (and,
    for the tools path, ``mock_tool_calls``) added — so all real litellm
    import/transform/tools-gated code executes while the network is short-circuited.
    """
    real_acompletion = litellm.acompletion

    async def general_llm_router(**kwargs: Any) -> Any:
        return await real_acompletion(**kwargs, mock_response=_route_general_llm(kwargs))

    # Agentic driver: the loop needs a tool-call-shaped response. We script it to
    # (1) set_research_plan on the first turn, then (2) conclude — enough to run
    # the loop end to end without any external tool call, while still driving the
    # real litellm ``tools=`` path (the fastapi tripwire) on every step.
    agentic_state = {"step": 0}

    async def agentic_router(**kwargs: Any) -> Any:
        if kwargs.get("tools"):
            step = agentic_state["step"]
            agentic_state["step"] += 1
            if step == 0:
                mock_tool_calls = [
                    {
                        "id": "plan0",
                        "type": "function",
                        "function": {
                            "name": "set_research_plan",
                            "arguments": json.dumps(
                                {"gaps": [{"id": "g1", "question": "Latest authoritative measurement?"}]}
                            ),
                        },
                    }
                ]
            else:
                mock_tool_calls = [
                    {
                        "id": "done0",
                        "type": "function",
                        "function": {
                            "name": "conclude",
                            "arguments": json.dumps(
                                {
                                    "gap_accounting": [
                                        {
                                            "gap_id": "g1",
                                            "actions_taken": "briefing already covers it",
                                            "status": "resolved",
                                        }
                                    ]
                                }
                            ),
                        },
                    }
                ]
            return await real_acompletion(
                **kwargs, mock_response="driving the agentic loop", mock_tool_calls=mock_tool_calls
            )
        # The ghost phase calls with tools=None; return a plain block so
        # _summarize_ghost parses it (telemetry only).
        return await real_acompletion(**kwargs, mock_response=_CANNED_BINARY)

    monkeypatch.setattr(ft_general_llm, "acompletion", general_llm_router)
    monkeypatch.setattr(agentic_llm, "acompletion", agentic_router)
    # drop_params must be True for the agentic wrapper's OpenRouter reasoning_effort
    # handling; a GeneralLlm invoke sets it globally in prod, but assert it here so
    # the agentic path is deterministic even if it runs first.
    monkeypatch.setattr(litellm, "drop_params", True)


# ---------------------------------------------------------------------------
# Provider client stubs — the lowest socket-opening boundary of each provider.
# ---------------------------------------------------------------------------


class _FakeArticle:
    """Duck-typed AskNews article (attribute access used by _format_single_article)."""

    def __init__(self, title: str, url: str) -> None:
        self.eng_title = title
        self.summary = "Unemployment held at 4.1% in the latest monthly print."
        self.language = "en"
        self.pub_date = datetime(2026, 4, 14, 9, 0, tzinfo=timezone.utc)
        self.source_id = "reuters"
        self.article_url = url


class _FakeAskNewsResponse:
    def __init__(self, articles: list[_FakeArticle]) -> None:
        self.as_dicts = articles


class _FakeAskNewsSDK:
    """Async context manager standing in for AsyncAskNewsSDK; .news.search_news is awaited."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.news = MagicMock()
        self.news.search_news = AsyncMock(
            return_value=_FakeAskNewsResponse(
                [_FakeArticle("US unemployment steady at 4.1%", "https://reuters.com/econ/us-unemployment-apr-2026")]
            )
        )

    async def __aenter__(self) -> "_FakeAskNewsSDK":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


def _make_gemini_response() -> Any:
    """Minimal google-genai GenerateContentResponse shape with one grounding chunk.

    Passes the grounded-chunk floor (>=1 chunk) so _format_grounded_response
    returns non-empty text with a Sources section.
    """

    web = SimpleNamespace(uri="https://vertex-redirect/blob", title="BLS Employment Situation", domain="bls.gov")
    chunk = SimpleNamespace(web=web)
    metadata = SimpleNamespace(grounding_chunks=[chunk], grounding_supports=None, web_search_queries=["unemployment"])
    candidate = SimpleNamespace(grounding_metadata=metadata, url_context_metadata=None)
    return SimpleNamespace(
        text="Google Search grounding: the April 2026 unemployment rate was 4.1%.",
        candidates=[candidate],
    )


def _fake_gemini_client() -> MagicMock:
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=_make_gemini_response())
    return client


# --- aiohttp fakes for prediction-market + resolution-source (JSON + HTML) ----


class _FakeContent:
    def __init__(self, resp: "_FakeHttpResponse") -> None:
        self._resp = resp

    async def iter_chunked(self, n: int) -> Any:
        body = self._resp._body
        for i in range(0, len(body), n):
            yield body[i : i + n]


class _FakeHttpResponse:
    def __init__(self, status: int = 200, *, body: bytes = b"{}", content_type: str = "application/json") -> None:
        self.status = status
        self._body = body
        self.headers = {"Content-Type": content_type}
        self.content = _FakeContent(self)

    async def read(self) -> bytes:
        return self._body

    async def text(self) -> str:
        return self._body.decode("utf-8", errors="replace")

    async def json(self) -> Any:
        return json.loads(self._body)

    async def __aenter__(self) -> "_FakeHttpResponse":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


_RESOLUTION_HTML = (
    b"<!doctype html><html><head><title>Unemployment Report</title></head><body>"
    b"<article><h1>April 2026 Employment Situation</h1>"
    b"<p>The Bureau of Labor Statistics reported the seasonally adjusted unemployment "
    b"rate at 4.1 percent for April 2026, unchanged from the prior month. Nonfarm "
    b"payroll employment rose by 175,000. The labor force participation rate held at "
    b"62.7 percent. Analysts had expected a reading near 4.2 percent, so the print was "
    b"modestly stronger than consensus. Wage growth cooled slightly year over year.</p>"
    b"</article></body></html>"
)


class _FakeHttpSession:
    """aiohttp.ClientSession stand-in. Prediction-market JSON hosts return an empty
    but well-formed payload (no matches — the formatter is still exercised); the
    resolution-source host returns an article-shaped HTML body so trafilatura runs.
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.closed = False

    def get(self, url: str, **_kwargs: Any) -> _FakeHttpResponse:
        low = url.lower()
        if "example.gov" in low or "example.com" in low:
            return _FakeHttpResponse(200, body=_RESOLUTION_HTML, content_type="text/html; charset=utf-8")
        # Prediction-market JSON APIs: shape-valid empty results.
        if "manifold" in low:
            return _FakeHttpResponse(200, body=b"[]", content_type="application/json")
        if "predictit" in low:
            return _FakeHttpResponse(200, body=b'{"markets": []}', content_type="application/json")
        if "kalshi" in low and "series" in low:
            return _FakeHttpResponse(200, body=b'{"series": []}', content_type="application/json")
        if "kalshi" in low:
            return _FakeHttpResponse(200, body=b'{"events": [], "cursor": ""}', content_type="application/json")
        # Polymarket public-search + anything else.
        return _FakeHttpResponse(200, body=b'{"events": [], "markets": []}', content_type="application/json")

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> "_FakeHttpSession":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()


def _install_provider_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub every provider's external client at its socket-opening boundary."""
    # AskNews SDK — both the two-phase provider and the agentic tools' search_news
    # do a function-scoped `from asknews_sdk import AsyncAskNewsSDK`, so patching
    # the module attribute covers both call sites.
    monkeypatch.setattr(asknews_sdk, "AsyncAskNewsSDK", _FakeAskNewsSDK)

    # Skip the AskNews provider's real rate-gate sleeps.
    async def _noop_rate_gate() -> None:
        return None

    monkeypatch.setattr(research_providers, "_asknews_rate_gate", _noop_rate_gate)
    # The two-phase AskNews provider sleeps 10.1s twice before its calls; patch
    # asyncio.sleep inside the providers module to keep the test fast (it only
    # affects that module's sleeps, not the event loop).
    real_sleep = research_providers.asyncio.sleep

    async def _fast_sleep(seconds: float) -> None:
        # Collapse the provider's deliberate 10.1s throttle waits; keep 0-sleeps
        # (checkpoints) real so scheduling semantics are unchanged.
        await real_sleep(0)

    monkeypatch.setattr(research_providers.asyncio, "sleep", _fast_sleep)

    # Gemini grounded search — patch build_gemini_client (the public factory the
    # provider calls). NOT _cached_client_for_key: that's an lru_cache-wrapped
    # function conftest's autouse _clear_gemini_client_cache fixture calls
    # .cache_clear() on at teardown, so replacing it with a plain lambda would
    # break teardown. Patching the caller leaves the lru_cache intact.
    monkeypatch.setattr(gemini_search, "build_gemini_client", _fake_gemini_client)

    # Prediction-market + resolution-source aiohttp sessions.
    monkeypatch.setattr(prediction_market, "_get_session", lambda: _FakeHttpSession())
    monkeypatch.setattr(resolution_source, "_get_session", lambda: _FakeHttpSession())
    # resolution_source runs a getaddrinfo SSRF preflight on every URL; example.gov
    # has no real DNS, so return a public IP (mirrors test_resolution_source_provider).
    monkeypatch.setattr(
        resolution_source.socket,
        "getaddrinfo",
        lambda *a, **k: [(0, 0, 0, "", ("8.8.8.8", 0))],
    )
    prediction_market._reset_session_caches()


def _install_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror the prod-workflow env: every provider ENABLED + dummy keys so gates pass.

    Stacking flags are deliberately NOT set — prod runs with stacking disabled, so
    the default-off median/skipped path is what we exercise. (conftest's autouse
    fixture sets the *_STACKING_ENABLED flags on; we delete them here to reproduce
    prod, which routes through the non-stacked aggregation.)
    """
    for flag in (
        "NATIVE_SEARCH_ENABLED",
        "GEMINI_SEARCH_ENABLED",
        "FINANCIAL_DATA_ENABLED",
        "GAP_FILL_ENABLED",
        "GAP_FILL_V2_ENABLED",
        "PREDICTION_MARKETS_ENABLED",
        "RESOLUTION_SOURCE_ENABLED",
    ):
        monkeypatch.setenv(flag, "true")
    # Restore prod's stacking-disabled default (conftest autouse turns these on).
    for flag in ("BINARY_STACKING_ENABLED", "MC_STACKING_ENABLED", "NUMERIC_STACKING_ENABLED"):
        monkeypatch.setenv(flag, "false")
    # Dummy API keys so provider-selection gates pass. AskNews creds make it the
    # primary provider (prod case). GOOGLE_API_KEY for gemini + FRED_API_KEY for
    # financial. Personal OpenRouter key only (donated deleted) so the fallback
    # wrapper stays single-key deterministic and the agentic router is on one key.
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "dummy-client")
    monkeypatch.setenv("ASKNEWS_SECRET", "dummy-secret")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-google")
    monkeypatch.setenv("FRED_API_KEY", "dummy-fred")
    monkeypatch.setenv("EXA_API_KEY", "dummy-exa")
    monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")


def _make_bot() -> TemplateForecaster:
    """Build the REAL production ensemble (llm_configs singletons), stacking-strategy
    CONDITIONAL_STACKING (the code default), min_forecasters=1, is_benchmarking=False.

    is_benchmarking MUST be False: the prediction-market and resolution-source
    providers hard-disable under benchmarking, and gap-fill v2 returns "" — we want
    all of them to run.
    """
    llms: dict[str, Any] = {
        "forecasters": FORECASTER_LLMS,
        "stacker": STACKER_LLM,
        "analyzer": DISAGREEMENT_ANALYZER_LLM,
        "summarizer": SUMMARIZER_LLM,
        "parser": PARSER_LLM,
        "researcher": RESEARCHER_LLM,
    }
    return TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,  # default False; no Metaculus post
        aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
        llms=llms,
        is_benchmarking=False,
        min_forecasters_to_publish=1,
    )


def _binary_question() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will the US unemployment rate exceed 5% by December 2026?",
        id_of_question=70001,
        id_of_post=80001,
        page_url="https://www.metaculus.com/questions/70001/",
        background_info="The US unemployment rate has been between 3.4% and 4.2% for the past year.",
        resolution_criteria=(
            "Resolves YES if BLS reports a seasonally adjusted unemployment rate of 5.0% or higher "
            f"for any month through December 2026. Source: {_RESOLUTION_URL}"
        ),
        fine_print="Uses seasonally adjusted figures from the BLS Employment Situation report.",
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


def _numeric_question() -> NumericQuestion:
    return NumericQuestion(
        question_text="What will the US unemployment rate be in December 2026?",
        id_of_question=70002,
        id_of_post=80002,
        page_url="https://www.metaculus.com/questions/70002/",
        background_info="The US unemployment rate is reported monthly by the BLS.",
        resolution_criteria=(
            "Resolves to the seasonally adjusted U-3 rate published by BLS for December 2026. "
            f"Source: {_RESOLUTION_URL}"
        ),
        fine_print="If revised, uses the initial release value.",
        unit_of_measure="percent",
        lower_bound=0.0,
        upper_bound=20.0,
        open_lower_bound=False,
        open_upper_bound=True,
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


def _mc_question() -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Which economic scenario is most likely for the US in 2026?",
        id_of_question=70003,
        id_of_post=80003,
        page_url="https://www.metaculus.com/questions/70003/",
        options=["Option A", "Option B", "Option C"],
        background_info="Multiple economic scenarios are possible depending on Fed policy.",
        resolution_criteria=(
            f"Resolves to the option best describing the realized outcome by year-end. Source: {_RESOLUTION_URL}"
        ),
        fine_print="Resolution determined by a panel of three economists.",
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


@pytest.fixture
def offline_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the LLM router, all provider stubs, and the prod-mirroring env."""
    _install_env(monkeypatch)
    _install_llm_router(monkeypatch)
    _install_provider_stubs(monkeypatch)


class TestOfflineE2EForecast:
    """Full offline pipeline for each question type, all providers on, no network."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_binary_full_pipeline_offline(self, offline_pipeline: None, caplog: pytest.LogCaptureFixture) -> None:
        bot = _make_bot()
        with caplog.at_level("INFO"):
            reports = await bot.forecast_questions([_binary_question()])

        assert len(reports) == 1
        report = reports[0]
        assert isinstance(report, ForecastReport)
        prediction = report.prediction
        assert isinstance(prediction, float)
        # Binary clamp is [0.02, 0.98]; median-of-3 identical 0.22 blocks = 0.22.
        assert 0.02 <= prediction <= 0.98

        _assert_pipeline_ran(caplog, bot, expect_qtype="binary")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_numeric_full_pipeline_offline(
        self, offline_pipeline: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        bot = _make_bot()
        with caplog.at_level("INFO"):
            reports = await bot.forecast_questions([_numeric_question()])

        assert len(reports) == 1
        report = reports[0]
        assert isinstance(report, ForecastReport)
        # Numeric prediction is a NumericDistribution; its published CDF is 201 points.
        cdf = report.prediction.cdf
        assert len(cdf) == 201
        # Monotonic non-decreasing CDF within [0, 1].
        probs = [pt.percentile for pt in cdf]
        assert probs == sorted(probs)
        assert 0.0 <= probs[0] <= probs[-1] <= 1.0

        _assert_pipeline_ran(caplog, bot, expect_qtype="numeric")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_mc_full_pipeline_offline(self, offline_pipeline: None, caplog: pytest.LogCaptureFixture) -> None:
        bot = _make_bot()
        with caplog.at_level("INFO"):
            reports = await bot.forecast_questions([_mc_question()])

        assert len(reports) == 1
        report = reports[0]
        assert isinstance(report, ForecastReport)
        option_probs = [opt.probability for opt in report.prediction.predicted_options]
        assert len(option_probs) == 3
        assert abs(sum(option_probs) - 1.0) < 1e-6
        assert all(0.0 <= p <= 1.0 for p in option_probs)

        _assert_pipeline_ran(caplog, bot, expect_qtype="multiple_choice")


def _assert_pipeline_ran(caplog: pytest.LogCaptureFixture, bot: TemplateForecaster, *, expect_qtype: str) -> None:
    """Assert (via logs + the bot's degradation counters) that every real code
    path executed offline WITHOUT any swallowed failure.

    The load-bearing signals:
    - alertable_count == 0: the sum of all degradation counters (forecasters
      dropped, publish failures, stacker fallbacks, research-provider failures,
      gap-fill v2 errors). The orchestrator SWALLOWS provider exceptions into
      status="errored" + a counter bump rather than re-raising, so a broken
      provider dep would otherwise pass silently — this is the tripwire for it.
    - EXTRACTION_RUNG rung=block: the forecaster's canned block parsed at rung 1
      (the value-extraction ladder ran for real over real model output).
    - GAP_FILL_V2 ... error=None: the agentic v2 loop EXECUTED and did NOT crash —
      this directly asserts the fastapi class of bug is absent (a dead-on-arrival
      import error would stamp error=<repr> on every question).
    - Provider diagnostics: the required providers returned 'ok' and NONE errored
      (their real formatting code ran end to end).
    """
    # No swallowed degradation: a provider erroring, a forecaster being dropped,
    # the stacker falling back, or gap-fill v2 crashing all bump this.
    assert bot.alertable_count == 0, (
        f"pipeline degraded — a provider errored, a forecaster was dropped, the stacker "
        f"fell back, or gap-fill v2 crashed: alertable_count={bot.alertable_count}"
    )

    messages = [rec.getMessage() for rec in caplog.records]
    text = "\n".join(messages)

    # Forecaster value-extraction landed on rung 1 for the expected question type.
    extraction_lines = [m for m in messages if "EXTRACTION_RUNG:" in m]
    assert extraction_lines, "no EXTRACTION_RUNG telemetry — forecaster extraction did not run"
    assert any(f"qtype={expect_qtype}" in m and "rung=block" in m for m in extraction_lines), (
        f"expected a rung=block extraction for qtype={expect_qtype}; got: {extraction_lines}"
    )

    # Gap-fill v2 executed and did not crash (the fastapi tripwire).
    v2_lines = [m for m in messages if "GAP_FILL_V2:" in m]
    assert v2_lines, "no GAP_FILL_V2 marker — the agentic v2 loop never ran"
    clean_v2_lines = [m for m in v2_lines if "error=None" in m]
    assert clean_v2_lines, f"gap-fill v2 crashed (fastapi-class bug?): {v2_lines}"
    # A crash-free marker (error=None) with tool_calls=0 would still pass the check
    # above even though the driver never issued a tool call — i.e. a driver that
    # stopped sending tools looks identical to a healthy run. Require >=1 tool call
    # so "the v2 loop executed" means it actually drove the tools= path. The
    # agentic_router scripts set_research_plan + conclude, so tool_calls is >=2 here.
    # (?<!dup_) keeps the match off the sibling dup_tool_calls= field.
    tool_call_counts = [
        int(match.group(1)) for m in clean_v2_lines if (match := re.search(r"(?<!dup_)tool_calls=(\d+)", m))
    ]
    assert tool_call_counts, f"GAP_FILL_V2 marker has no tool_calls= field: {clean_v2_lines}"
    assert any(n > 0 for n in tool_call_counts), (
        f"gap-fill v2 ran but issued no tool calls (tool_calls=0) — driver stopped sending tools: {clean_v2_lines}"
    )

    # Provider diagnostics show the stubbed providers ran end to end. Each required
    # provider must report `ok` (its real formatting code produced non-empty text),
    # and NO provider may report `errored` — the direct catch for a swallowed
    # provider-dep break (google-genai / asknews / aiohttp), which the orchestrator
    # turns into status="errored" instead of re-raising.
    assert "Provider diagnostics" in text, "no provider-diagnostics telemetry"
    for provider in _REQUIRED_OK_PROVIDERS:
        assert f"{provider}: ok" in text, (
            f"required provider {provider!r} did not report 'ok' in diagnostics — a provider dep may have broken:\n{text}"
        )
    assert ": errored" not in text, f"a research provider errored (swallowed by the orchestrator):\n{text}"
