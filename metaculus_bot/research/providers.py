"""Research provider strategy abstraction.

`choose_provider_with_name` returns an async callable (and its name) that, given
a `MetaculusQuestion`, returns formatted research.  The selection is governed
by environment variables so the logic lives in one place instead of being in
`TemplateForecaster.run_research`.

Providers receive the full question (not just the text) so they can use
auxiliary fields like `id_of_question` for caching, `resolution_criteria` for
keyword extraction, and `scheduled_resolution_time` for backtest-leakage
defenses (see `prediction_market_provider.py`).
"""

import asyncio
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from forecasting_tools import GeneralLlm, SmartSearcher
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    ASKNEWS_BACKOFF_SECS,
    ASKNEWS_CLIENT_ID_ENV,
    ASKNEWS_MAX_CONCURRENCY,
    ASKNEWS_MAX_RPS,
    ASKNEWS_MAX_TRIES,
    ASKNEWS_SECRET_ENV,
    ASKNEWS_WALL_TIMEOUT,
    EXA_API_KEY_ENV,
    NATIVE_SEARCH_CONTEXT_SIZE,
    NATIVE_SEARCH_DEFAULT_MODEL,
    NATIVE_SEARCH_MAX_RESULTS,
    NATIVE_SEARCH_MAX_TOKENS,
    NATIVE_SEARCH_MODEL_ENV,
    NATIVE_SEARCH_REASONING_EFFORT_DEFAULT,
    NATIVE_SEARCH_REASONING_EFFORT_ENV,
    NATIVE_SEARCH_TIMEOUT,
    NATIVE_SEARCH_VERBOSITY_DEFAULT,
    NATIVE_SEARCH_VERBOSITY_ENV,
    OPENROUTER_API_KEY_ENV,
    PERPLEXITY_API_KEY_ENV,
    PERPLEXITY_RESEARCH_MODEL,
    PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER,
    PERPLEXITY_WALL_TIMEOUT,
    RESEARCH_PROVIDER_ENV,
)
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.llm_retry import invoke_with_transient_retry
from metaculus_bot.prompts import web_research_prompt
from metaculus_bot.research.provider_diagnostics import record_provider_detail
from metaculus_bot.research.raw_log import record_raw_research

ResearchCallable = Callable[[MetaculusQuestion], Awaitable[str]]
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concrete provider helpers
# ---------------------------------------------------------------------------


_ASKNEWS_GLOBAL_SEMAPHORE: asyncio.Semaphore | None = None
_ASKNEWS_RATE_LOCK: asyncio.Lock | None = None
_ASKNEWS_LAST_CALL_TS: float = 0.0


def _get_asknews_rate_lock() -> asyncio.Lock:
    """Get-or-create the process-wide lock guarding the AskNews RPS gate.

    Lazy purely for CONSISTENCY with ``get_asknews_semaphore`` below, which owns the
    identical lifecycle two lines away. Both are process-wide asyncio primitives for
    the same provider; having one built at import and the other lazily was a
    difference with no reason behind it, and the import-time one is the shape that
    can bind to a loop that later dies.

    Honest scope: I could not construct a failing case against THIS gate.
    ``asyncio.Lock`` binds to the running loop when it first creates a future, and a
    lock left HELD at loop close does wedge later loops with "is bound to a different
    event loop" — reproduced in isolation, where a cancelled holder left a waiter
    queued. But driving the real ``_asknews_rate_gate`` the same way, a second
    ``asyncio.run`` succeeds: the cancellation releases the lock before the loop
    closes, so it rebinds cleanly. So this is a latent-shape cleanup, not a fix for an
    observed failure, and it is deliberately not paired with a staleness check —
    detecting a stale binding needs a private ``_get_loop`` probe that reads clean in
    exactly the case that later fails, so the check would be reassuring rather than
    effective.
    """
    global _ASKNEWS_RATE_LOCK  # noqa: PLW0603  # sole lazy-init of the process-wide AskNews lock; see docstring
    if _ASKNEWS_RATE_LOCK is None:
        _ASKNEWS_RATE_LOCK = asyncio.Lock()
    return _ASKNEWS_RATE_LOCK


async def _asknews_rate_gate() -> None:
    global _ASKNEWS_LAST_CALL_TS  # noqa: PLW0603  # process-wide AskNews RPS clock, shared by the provider and agentic tools
    if ASKNEWS_MAX_RPS <= 0:
        return
    min_interval = 1.0 / ASKNEWS_MAX_RPS
    async with _get_asknews_rate_lock():
        now = time.monotonic()
        wait = _ASKNEWS_LAST_CALL_TS + min_interval - now
        if wait > 0:
            await asyncio.sleep(wait)
            now = time.monotonic()
        _ASKNEWS_LAST_CALL_TS = now


async def asknews_rate_gate() -> None:
    """Public seam for the process-wide AskNews RPS gate.

    Delegates at call time so tests that monkeypatch ``_asknews_rate_gate``
    keep intercepting calls routed through this public name (used by
    ``research.agentic.tools``).
    """
    await _asknews_rate_gate()


def get_asknews_semaphore() -> asyncio.Semaphore:
    """Get-or-create the single process-wide AskNews concurrency semaphore.

    Owns the only lazy-init of ``_ASKNEWS_GLOBAL_SEMAPHORE`` so every AskNews
    caller (the two-phase provider here and ``research.agentic.tools``)
    contends on the same throttle.
    """
    global _ASKNEWS_GLOBAL_SEMAPHORE  # noqa: PLW0603  # sole lazy-init of the process-wide AskNews semaphore; see docstring
    if _ASKNEWS_GLOBAL_SEMAPHORE is None:
        _ASKNEWS_GLOBAL_SEMAPHORE = asyncio.Semaphore(max(1, int(ASKNEWS_MAX_CONCURRENCY)))
    return _ASKNEWS_GLOBAL_SEMAPHORE


def is_asknews_subscription_error(exc: BaseException) -> bool:
    """True iff exc is AskNews's 403011 subscription-inactive signature.

    Narrow match (class name AND inner message) so a generic "403 Forbidden"
    from an unrelated provider isn't silenced. SDK raises ForbiddenError with
    code 403011 or "subscription is not currently active" when billing lapses.
    """
    msg = str(exc).lower()
    return "forbiddenerror" in type(exc).__name__.lower() and (
        "403011" in msg or "subscription is not currently active" in msg
    )


# Per-phase AskNews search parameters: (log label, SDK strategy, n_articles).
# HOT is the latest-news sweep, HISTORICAL the deeper knowledge pull.
_ASKNEWS_PHASES: dict[str, tuple[str, str, int]] = {
    "hot": ("HOT", "latest news", 6),
    "historical": ("HIST", "news knowledge", 10),
}

# Extra spacing before each phase, on top of the RPS gate: the vendor still
# returns 429s at our nominal rate, so each phase eats a fixed wait first.
_ASKNEWS_PHASE_WAIT_SEC = 10.1


# Retry predicate: only retry on known transient rate/concurrency errors.
# Text-matched on purpose, unlike the LLM paths that read ``llm_status_code``:
# the AskNews SDK raises its own ``asknews_sdk.errors`` classes carrying a
# ``.code`` (429000 / 429001 / 403011) and never subclasses ``openai.APIError``,
# so a status-based primitive reads None here and would disable this retry.
def _is_asknews_retryable(err: Exception) -> bool:
    msg = str(err).lower()
    return ("429" in msg) or ("rate limit" in msg) or ("concurrency limit" in msg)


async def _asknews_phase(
    sdk: Any,
    question_text: str,
    *,
    phase: str,
    tries: int,
    backoff: float,
    qid: int | None,
) -> tuple[Any, int]:
    """Run one AskNews search phase with its own retry loop; returns ``(articles, attempts_used)``.

    ``attempts_used`` is what lets the HISTORICAL phase spend only the budget HOT
    left over, so a phase that burned retries can't double the wall-clock cost.
    Only transient rate/concurrency errors are retried; anything else raises
    immediately, and an exhausted ladder re-raises the last error.
    """
    label, strategy, n_articles = _ASKNEWS_PHASES[phase]
    last_exc: Exception | None = None
    attempt_used = 0
    for attempt in range(1, tries + 1):
        attempt_used = attempt
        try:
            if phase == "historical":
                logger.info(
                    f"AskNews {label} attempt {attempt}/{tries}: Passing rate gate before historical news call..."
                )
                await _asknews_rate_gate()
                logger.info(f"AskNews {label} attempt {attempt}/{tries}: Calling historical news...")
            else:
                logger.info(f"AskNews {label} attempt {attempt}/{tries}: Calling latest news...")
                await _asknews_rate_gate()
            response = await sdk.news.search_news(
                query=question_text,
                n_articles=n_articles,
                return_type="both",
                strategy=strategy,
            )
            articles = response.as_dicts
            record_raw_research(qid=qid, provider="asknews", phase=phase, payload=articles)
            return articles, attempt_used
        except Exception as e:
            last_exc = e
            if not _is_asknews_retryable(e):
                raise
            if attempt < tries:
                sleep_for = backoff * (10 + 3**attempt)
                await asyncio.sleep(sleep_for)
            else:
                assert last_exc is not None
                raise last_exc  # noqa: B904  # re-raises the exception being handled; `from` would self-reference
    raise AssertionError("unreachable: the retry ladder either returns or raises")


def _asknews_provider() -> ResearchCallable:
    get_asknews_semaphore()

    async def _fetch(question: MetaculusQuestion) -> str:
        # Hard wall-clock timeout around the full provider. AskNews's internal
        # retry loop fails fast on non-retryable errors, but a genuine network
        # hang (connect stall, DNS hang, server not closing the stream) is
        # otherwise unbounded. This backstops that case so a stuck AskNews
        # call can't hold the whole research phase hostage.
        return await asyncio.wait_for(
            _fetch_impl(question.question_text, qid=getattr(question, "id_of_question", None)),
            timeout=ASKNEWS_WALL_TIMEOUT,
        )

    async def _fetch_impl(question_text: str, *, qid: int | None = None) -> str:
        assert _ASKNEWS_GLOBAL_SEMAPHORE is not None
        tries = max(1, int(ASKNEWS_MAX_TRIES))
        backoff = float(ASKNEWS_BACKOFF_SECS)

        async with _ASKNEWS_GLOBAL_SEMAPHORE:
            # Use custom AskNews integration with proper rate limiting between API calls
            from asknews_sdk import (  # noqa: PLC0415  # late import: tests patch asknews_sdk.AsyncAskNewsSDK at source
                AsyncAskNewsSDK,
            )

            client_id = os.getenv(ASKNEWS_CLIENT_ID_ENV)
            secret = os.getenv(ASKNEWS_SECRET_ENV)
            if not client_id or not secret:
                raise ValueError("ASKNEWS_CLIENT_ID and ASKNEWS_SECRET environment variables must be set")

            logger.info(f"AskNews: Using custom integration, client_id={client_id[:8]}...")

            async with AsyncAskNewsSDK(
                client_id=client_id,
                client_secret=secret,
                scopes={"news"},
            ) as sdk:
                # Hack: despite including rate limits in our asknews logic, we still get rate limits; manually massage addl waits to handle
                logger.info(f"AskNews: Waiting {_ASKNEWS_PHASE_WAIT_SEC}s before hot news call...")
                await asyncio.sleep(_ASKNEWS_PHASE_WAIT_SEC)
                hot_articles, hot_attempt_used = await _asknews_phase(
                    sdk, question_text, phase="hot", tries=tries, backoff=backoff, qid=qid
                )
                assert hot_articles is not None

                # Phase 2: HISTORICAL (news knowledge), reuse HOT results; do not re-call HOT on retries
                logger.info(f"AskNews: Waiting {_ASKNEWS_PHASE_WAIT_SEC}s before historical news call...")
                await asyncio.sleep(_ASKNEWS_PHASE_WAIT_SEC)
                historical_articles, _ = await _asknews_phase(
                    sdk,
                    question_text,
                    phase="historical",
                    tries=max(1, tries - (hot_attempt_used - 1)),
                    backoff=backoff,
                    qid=qid,
                )
                assert historical_articles is not None

                logger.info(
                    f"AskNews: Got {len(hot_articles)} hot articles, {len(historical_articles)} historical articles"
                )

                formatted_articles = _format_asknews_dual_sections(
                    hot_articles=hot_articles,
                    historical_articles=historical_articles,
                )
                if not formatted_articles:
                    logger.warning(
                        f"ASKNEWS_NO_ARTICLES: question={qid} "
                        f"hot={len(hot_articles)} historical={len(historical_articles)}"
                    )
                    record_provider_detail(qid, "asknews", {"sources": {"articles": "empty(no_articles)"}})
                    return ""

                logger.info(
                    f"AskNews: Success, got {len(formatted_articles)} chars from {len(hot_articles)} hot + {len(historical_articles)} historical articles"
                )
                return formatted_articles

    return _fetch


def _format_single_article(article: Any) -> str:
    pub_date = article.pub_date.strftime("%B %d, %Y %I:%M %p")
    return (
        f"**{article.eng_title}**\n{article.summary}\n"
        f"Original language: {article.language}\n"
        f"Publish date: {pub_date}\n"
        f"Source:[{article.source_id}]({article.article_url})\n\n"
    )


def _format_asknews_dual_sections(
    hot_articles: list[Any],
    historical_articles: list[Any],
) -> str:
    """Format AskNews articles into two labeled sections: Historical Context and Recent Developments.

    Deduplicates within each list and cross-deduplicates (hot articles that duplicate historical
    URLs are removed). Historical section comes first in the output.

    Both phases empty returns ``""``, NOT a prose "no articles" sentence. The sentence
    defeated every downstream empty guard: the orchestrator's ``has_output`` read chars>0
    and reported ``ok``, the summarizer LLM (whose prompt has no no-data escape) was asked
    to write a briefing from it, and the result rendered under the AskNews header as if it
    were research. Gemini's grounded-chunk floor next door is the pattern — refuse.

    Pure: the ASKNEWS_NO_ARTICLES WARN and the ``lost=articles:...`` registry token
    belong to ``_asknews_provider``, which owns the qid — a formatter writing the
    module-global provider-detail registry raced ``_degraded_to_raw_articles``' write
    for the same key only by accident of ordering.
    """
    hist_deduped = _dedup_articles_by_url(historical_articles) if historical_articles else []
    hot_deduped = _dedup_articles_by_url(hot_articles) if hot_articles else []

    if hist_deduped:
        hist_urls = {_normalize_url_for_dedup(str(a.article_url)) for a in hist_deduped}
        hot_deduped = [a for a in hot_deduped if _normalize_url_for_dedup(str(a.article_url)) not in hist_urls]

    if not hist_deduped and not hot_deduped:
        return ""

    total_before = len(historical_articles) + len(hot_articles)
    total_after = len(hist_deduped) + len(hot_deduped)
    removed = total_before - total_after
    if removed > 0:
        logger.info(f"AskNews URL dedup: {total_before} -> {total_after} (removed {removed} duplicates)")

    formatted_articles = "Here are the relevant news articles:\n\n"

    if hist_deduped:
        sorted_hist = sorted(hist_deduped, key=lambda x: x.pub_date, reverse=True)
        formatted_articles += "## Historical Context & Background\n\n"
        for article in sorted_hist:
            formatted_articles += _format_single_article(article)

    if hot_deduped:
        sorted_hot = sorted(hot_deduped, key=lambda x: x.pub_date, reverse=True)
        formatted_articles += "\n## Recent Developments & Current News\n\n"
        for article in sorted_hot:
            formatted_articles += _format_single_article(article)

    return formatted_articles


def _exa_provider(default_llm: GeneralLlm) -> ResearchCallable:
    async def _fetch(question: MetaculusQuestion) -> str:
        searcher = SmartSearcher(
            # temperature ignored when model is a preconfigured GeneralLlm; None
            # keeps litellm from applying a sampling param on the fallback str path.
            model=default_llm,
            temperature=None,
            num_searches_to_run=2,
            num_sites_per_search=10,
            # 0.2.92 gained SmartSearcher citation controls (include_works_cited_list,
            # use_brackets_around_citations), both defaulting to False. Pin them
            # False explicitly so the research-text shape stays as it is today — no
            # appended works-cited footer, no inline [n] brackets — even if a future
            # upstream default flips them on.
            include_works_cited_list=False,
            use_brackets_around_citations=False,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            " you a question they intend to forecast on. To be a great assistant, you generate"
            " a concise but detailed rundown of the most relevant news, including if the question"
            " would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question.question_text}"
        )
        return await searcher.invoke(prompt)

    return _fetch


def _perplexity_provider(use_open_router: bool = False, is_benchmarking: bool = False) -> ResearchCallable:
    async def _fetch(question: MetaculusQuestion) -> str:
        model_name = PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER if use_open_router else PERPLEXITY_RESEARCH_MODEL
        # temperature=None: 0.2.92's GeneralLlm ctor already defaults temperature to
        # None (it was a hard 0 pre-0.2.92), so this is now redundant-but-explicit —
        # kept to pin provider-default sampling against a future default flip. No top_p.
        # allowed_tries=1 hands the retry budget to the gated wrapper below; left
        # unpinned it inherited forecasting-tools' default of 2 with an un-gated
        # random.uniform(5, 10) tenacity sleep.
        #
        # No explicit api_key, unlike the near-identical builder in
        # ``ResearchOrchestrator._call_perplexity`` which passes
        # ``get_openrouter_api_key(model_name)``. Equivalent TODAY and only by
        # coincidence: perplexity is not in ``DONATED_KEY_PROVIDERS``, so that helper
        # returns ``OPENROUTER_API_KEY``, which litellm also reads straight from the
        # environment. It stops being equivalent the moment perplexity becomes
        # donated-key-eligible, at which point this call site silently keeps billing
        # the personal key while the orchestrator's switches. The real fix is to
        # collapse the two builders (forge flagged the duplication); until then, don't
        # "clean up" this asymmetry by deleting the orchestrator's api_key argument.
        model = GeneralLlm(model=model_name, temperature=None, allowed_tries=1)
        # Exclude prediction markets research when benchmarking to avoid data leakage
        prediction_markets_instruction = (
            "" if is_benchmarking else "In addition to news, consider all relevant prediction markets.\n"
        )
        prompt = (
            "You are an assistant to a superforecaster.\n"
            "Generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.\n"
            f"{prediction_markets_instruction}"
            "Do not produce forecasts yourself. Provide data for the superforecaster.\n\n"
            f"Question:\n{question.question_text}"
        )
        return await invoke_with_transient_retry(
            lambda: model.invoke(prompt),
            wall_timeout=PERPLEXITY_WALL_TIMEOUT,
            label="perplexity_research",
        )

    return _fetch


def build_native_search_llm(
    model_slug: str | None = None,
    *,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
) -> GeneralLlm:
    """Build a GeneralLlm configured for OpenAI native web search via OpenRouter.

    Shared by the native search research provider, the targeted research module,
    and the gap-fill resolver.

    Reasoning effort and verbosity come from the global NATIVE_SEARCH_REASONING_EFFORT
    / NATIVE_SEARCH_VERBOSITY env at call time (so workflow overrides take effect
    without re-importing), UNLESS the caller passes an explicit ``reasoning_effort``
    / ``verbosity`` override — an explicit value always wins over the env read.
    This lets callers like the gap-fill resolver pin their own model/effort
    without perturbing the main native_search provider, which stays on the
    env-driven LOW. An empty string (from either the override or the env)
    disables passing the corresponding kwarg.
    """
    base_model = model_slug or os.getenv(NATIVE_SEARCH_MODEL_ENV, NATIVE_SEARCH_DEFAULT_MODEL)
    model_with_search = f"openrouter/{base_model}"

    kwargs: dict = {
        "model": model_with_search,
        # temperature=None: 0.2.92's GeneralLlm ctor already defaults temperature to
        # None (it was a hard 0 pre-0.2.92), so this is now redundant-but-explicit —
        # kept to pin provider-default sampling against a future default flip. reasoning
        # models defer to provider defaults. top_p left unset.
        "temperature": None,
        "max_tokens": NATIVE_SEARCH_MAX_TOKENS,
        "timeout": NATIVE_SEARCH_TIMEOUT,
        # allowed_tries=1: a malformed-whitespace response from OpenRouter (the
        # 2026-05-20 incident) won't be cured by retrying the same call, and
        # the wall-clock guard at the caller (asyncio.wait_for in _fetch) is
        # bounding the budget. With allowed_tries=1 the worst case is one
        # NATIVE_SEARCH_WALL_TIMEOUT window instead of forecasting-tools'
        # default ``allowed_tries`` multiplied by NATIVE_SEARCH_TIMEOUT (which
        # resets per HTTP request).
        "allowed_tries": 1,
        "plugins": [{"id": "web", "max_results": NATIVE_SEARCH_MAX_RESULTS, "engine": "native"}],
        "web_search_options": {"search_context_size": NATIVE_SEARCH_CONTEXT_SIZE},
    }

    effort = (
        reasoning_effort
        if reasoning_effort is not None
        else os.getenv(NATIVE_SEARCH_REASONING_EFFORT_ENV, NATIVE_SEARCH_REASONING_EFFORT_DEFAULT)
    )
    if effort:
        kwargs["reasoning"] = {"effort": effort}

    # `verbosity` is a top-level OpenRouter / litellm parameter (see litellm
    # `acompletion(... verbosity=...)` and OpenAI gpt-5 transformation). Earlier
    # we tucked it inside `extra_body`; that worked because OpenRouter merges
    # the body, but the canonical form matches the docs and survives any future
    # extra_body validation. GeneralLlm passes unknown kwargs through to
    # litellm by default (`pass_through_unknown_kwargs=True`).
    verbosity_value = (
        verbosity if verbosity is not None else os.getenv(NATIVE_SEARCH_VERBOSITY_ENV, NATIVE_SEARCH_VERBOSITY_DEFAULT)
    )
    if verbosity_value:
        kwargs["verbosity"] = verbosity_value

    # Route through the donated-key wrapper. For openrouter/openai/* slugs this
    # prefers the Metaculus-donated OAI_ANTH_OPENROUTER_KEY (OpenAI now enabled
    # on it as of 2026-05-29) with automatic fallback to the personal
    # OPENROUTER_API_KEY on credential/credit/guardrail errors. Non-donated
    # providers (x-ai, etc.) get a plain GeneralLlm — same as before.

    return build_llm_with_openrouter_fallback(**kwargs)


def _native_search_provider(
    model_slug: str | None = None,
    is_benchmarking: bool = False,
) -> ResearchCallable:
    """Research provider using models with native web search capability via OpenRouter :online suffix."""

    async def _fetch(question: MetaculusQuestion) -> str:
        from metaculus_bot.constants import (  # noqa: PLC0415  # late read: tests patch this constant on the constants module
            NATIVE_SEARCH_WALL_TIMEOUT,
        )

        llm = build_native_search_llm(model_slug)
        prompt = web_research_prompt(
            question.question_text,
            # The MC ballot (None on other types): a searching model can only query candidate
            # names it has been shown (q44952 — zero retrieval on the eventual winner).
            options=getattr(question, "options", None),
            is_benchmarking=is_benchmarking,
            citation_style="markdown",
        )
        logger.info(f"NativeSearch: Calling {llm.model} for research")
        # Wall-clock backstop (now owned by invoke_with_transient_retry): see
        # NATIVE_SEARCH_WALL_TIMEOUT in constants.py for the 2026-05-20 incident
        # that motivated the hard cap. The transient-retry wrapper additionally
        # recovers from instant aiohttp blips (litellm #14895) on this
        # allowed_tries=1 LLM without ever retrying a slow stall (elapsed gate).
        result = await invoke_with_transient_retry(
            lambda: llm.invoke(prompt), wall_timeout=NATIVE_SEARCH_WALL_TIMEOUT, label="native_search"
        )
        logger.info(f"NativeSearch: Got {len(result)} chars from {llm.model}")
        record_raw_research(
            qid=getattr(question, "id_of_question", None),
            provider="native_search",
            payload=result,
        )
        # Strip utm_source=openai from the forecaster-facing text; the raw log
        # above keeps the untouched payload for archival fidelity.
        return _strip_utm_source(result)

    return _fetch


# Public alias for the native search provider (used by tests and external callers)
native_search_provider = _native_search_provider


# ---------------------------------------------------------------------------
# Strategy selector
# ---------------------------------------------------------------------------


def _forced_provider_choice(
    forced_lc: str,
    *,
    default_llm: GeneralLlm | None,
    exa_callback: ResearchCallable | None,
    perplexity_callback: ResearchCallable | None,
    openrouter_callback: ResearchCallable | None,
    is_benchmarking: bool,
) -> tuple[ResearchCallable, str] | None:
    """Resolve an explicit ``RESEARCH_PROVIDER`` override, or None to fall through to auto."""
    if forced_lc == "asknews":
        # Fail fast if creds missing to make misconfig obvious
        if not (os.getenv(ASKNEWS_CLIENT_ID_ENV) and os.getenv(ASKNEWS_SECRET_ENV)):
            raise ValueError("RESEARCH_PROVIDER=asknews requires ASKNEWS_CLIENT_ID and ASKNEWS_SECRET to be set")
        return _asknews_provider(), "asknews"
    if forced_lc == "exa":
        if exa_callback is not None:
            return exa_callback, "exa"
        if default_llm is None:
            raise ValueError("RESEARCH_PROVIDER=exa requires default_llm or exa_callback to be provided")
        return _exa_provider(default_llm), "exa"
    if forced_lc == "perplexity":
        if perplexity_callback is not None:
            return perplexity_callback, "perplexity"
        return _perplexity_provider(use_open_router=False, is_benchmarking=is_benchmarking), "perplexity"
    if forced_lc == "openrouter":
        if openrouter_callback is not None:
            return openrouter_callback, "openrouter"
        return _perplexity_provider(use_open_router=True, is_benchmarking=is_benchmarking), "openrouter"
    # Any other value behaves as auto
    return None


def _auto_provider_choice(
    *,
    default_llm: GeneralLlm | None,
    exa_callback: ResearchCallable | None,
    perplexity_callback: ResearchCallable | None,
    openrouter_callback: ResearchCallable | None,
    is_benchmarking: bool,
) -> tuple[ResearchCallable, str]:
    """First provider whose credentials are present, in the documented priority order."""
    if os.getenv(ASKNEWS_CLIENT_ID_ENV) and os.getenv(ASKNEWS_SECRET_ENV):
        return _asknews_provider(), "asknews"

    if os.getenv(EXA_API_KEY_ENV):
        if exa_callback is not None:
            return exa_callback, "exa"
        if default_llm is None:
            raise ValueError("default_llm must be provided for Exa research provider")
        return _exa_provider(default_llm), "exa"

    if os.getenv(PERPLEXITY_API_KEY_ENV):
        if perplexity_callback is not None:
            return perplexity_callback, "perplexity"
        return _perplexity_provider(use_open_router=False, is_benchmarking=is_benchmarking), "perplexity"

    if os.getenv(OPENROUTER_API_KEY_ENV):
        if openrouter_callback is not None:
            return openrouter_callback, "openrouter"
        return _perplexity_provider(use_open_router=True, is_benchmarking=is_benchmarking), "openrouter"

    async def _empty(_: MetaculusQuestion) -> str:
        return ""

    return _empty, "none"


def choose_provider_with_name(
    default_llm: GeneralLlm | None = None,
    *,
    exa_callback: ResearchCallable | None = None,
    perplexity_callback: ResearchCallable | None = None,
    openrouter_callback: ResearchCallable | None = None,
    is_benchmarking: bool = False,
) -> tuple[ResearchCallable, str]:
    """Return a research coroutine and its provider name.

    Priority order replicates pre-refactor behaviour:
    1. AskNews (ASKNEWS_CLIENT_ID & ASKNEWS_SECRET)
    2. Exa.ai (EXA_API_KEY)
    3. Perplexity (PERPLEXITY_API_KEY)
    4. Perplexity via OpenRouter (OPENROUTER_API_KEY)
    5. Fallback stub that returns an empty string.

    ``RESEARCH_PROVIDER`` forces a specific provider; an unrecognized value falls
    through to the priority order above.
    """
    forced = os.getenv(RESEARCH_PROVIDER_ENV)
    if forced:
        choice = _forced_provider_choice(
            forced.strip().lower(),
            default_llm=default_llm,
            exa_callback=exa_callback,
            perplexity_callback=perplexity_callback,
            openrouter_callback=openrouter_callback,
            is_benchmarking=is_benchmarking,
        )
        if choice is not None:
            return choice

    return _auto_provider_choice(
        default_llm=default_llm,
        exa_callback=exa_callback,
        perplexity_callback=perplexity_callback,
        openrouter_callback=openrouter_callback,
        is_benchmarking=is_benchmarking,
    )


# ---------------------------------------------------------------------------
# URL normalization and dedup helpers (simple, robust, testable)
# ---------------------------------------------------------------------------


# OpenAI native search tags every citation URL with `?utm_source=openai`; it is
# pure tracking noise fanned into every forecaster prompt + the published comment.
# Match the param wherever it sits in the query string, capturing the leading
# separator and an optional trailing `&` so removal keeps the query well-formed.
_UTM_SOURCE_OPENAI_RE = re.compile(r"([?&])utm_source=openai\b(&)?")


def _strip_utm_source(text: str) -> str:
    """Drop ``utm_source=openai`` tracking params from URLs in native-search text.

    Handles the param as the sole query param (``?utm_source=openai`` -> ``), the
    first of several (``?utm_source=openai&a=b`` -> ``?a=b``), or a later one
    (``&utm_source=openai`` -> ``). Other query params are preserved. Operates on
    the free-text research blob (the native-search LLM emits the URLs inline).
    """

    def _repl(match: re.Match[str]) -> str:
        # Keep the leading separator only when another param follows, so the
        # query string stays well-formed (`?a&utm&b` -> `?a&b`, not `?ab`).
        return match.group(1) if match.group(2) else ""

    return _UTM_SOURCE_OPENAI_RE.sub(_repl, text)


def _normalize_url_for_dedup(url: str) -> str:
    """Return a canonicalized URL for deduplication.

    - Lowercase scheme and netloc
    - Drop fragment
    - Remove common tracking params (utm_*, gclid, fbclid, igshid, ref, mc_cid, mc_eid)
    - Sort remaining query params
    - Strip single trailing slash on path
    - Normalize mobile and AMP variants (m. subdomain, trailing /amp)
    """
    if not url:
        return url
    parts = urlsplit(url)
    scheme = (parts.scheme or "").lower()
    netloc = (parts.netloc or "").lower()

    # Normalize mobile subdomain 'm.' to base domain when present
    if netloc.startswith("m."):
        netloc = netloc[2:]

    # Normalize path: drop trailing '/amp' and single trailing slash
    path = parts.path or ""
    if path.endswith("/amp"):
        path = path[:-4]
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    # Normalize query: remove tracking keys; sort remaining
    drop_keys = {"gclid", "fbclid", "igshid", "ref", "mc_cid", "mc_eid"}
    q = []
    for k, raw_value in parse_qsl(parts.query, keep_blank_values=True):
        if k.startswith("utm_") or k in drop_keys:
            continue
        # Normalize trivial trailing slashes in parameter values (e.g., b=2/ -> b=2)
        value = raw_value.rstrip("/") if isinstance(raw_value, str) and raw_value.endswith("/") else raw_value
        q.append((k, value))
    # Sort for canonical order
    q.sort()
    query = urlencode(q, doseq=True)

    # Drop fragment
    fragment = ""

    return urlunsplit((scheme, netloc, path, query, fragment))


def _dedup_articles_by_url(articles: list[Any]) -> list[Any]:
    """Order-preserving deduplication of articles by normalized URL.

    Articles may be objects with attribute `article_url` or dicts with key `article_url`.
    Items without a URL are kept.
    """
    seen: set[str] = set()
    result: list[Any] = []
    for item in articles:
        url = item.get("article_url") if isinstance(item, dict) else getattr(item, "article_url", None)  # type: ignore[unreachable]

        if not url:
            result.append(item)
            continue

        norm = _normalize_url_for_dedup(str(url))
        if norm in seen:
            continue
        seen.add(norm)
        result.append(item)

    # Defensive: ensure not all items were dropped; if so, keep the first
    if not result and articles:
        result.append(articles[0])
    return result
