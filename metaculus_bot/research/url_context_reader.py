"""One Gemini ``url_context`` read, shared by the two surfaces that pay for one.

Gap-fill v2's ``read_document`` tool has spent this call since 2026-07; the Tier-1
resolution-source ladder gained it on 2026-09-03 as its last rung, behind its own flag. Both
want exactly the same thing — hand Gemini a URL, get back what it read, and know whether it
actually retrieved anything — so the call lives here once and each caller supplies its own
model, thinking level, timeout, attempts and telemetry role.

The RETRIEVAL COUNT is the reason this returns a tuple rather than a string, and it is
load-bearing on both surfaces: Gemini answers fluently from parametric memory when every
retrieval failed (Q38195, 2026-07-19 — 30 search queries, 0 grounding chunks, a confident
fabricated contract table with fake ``[primary]`` tags reached forecasters). A caller that
cannot tell those apart cannot honestly label what it renders, so the count comes back and the
caller withholds on zero.

This module makes NO policy decisions: no size gate, no robots pre-check, no flag. Those belong
to the callers, because the two answer them differently — v2 serves a document it already holds
from a local digest, while Tier-1 refuses the read outright when the flag is off.
"""

from __future__ import annotations

from typing import Any

from metaculus_bot.research.gemini_client_config import build_gemini_http_options, gemini_thinking_config
from metaculus_bot.research.gemini_usage import log_gemini_usage
from metaculus_bot.research.url_context_telemetry import extract_url_context_telemetry


def build_document_prompt(ask: str) -> str:
    """Wrap a caller's ask in the three instructions that keep a read checkable.

    Verbatim quotes, because a paraphrase of a document we cannot see is unverifiable. The
    document's own stated dates, because both callers render into evidence whose age decides how
    much it is worth. And a plain "this does not address the ask", because the alternative is a
    fluent answer assembled out of recall — the failure this whole path is guarded against.
    """
    return (
        f"{ask}\n\n"
        "Answer using verbatim quotes from the document whenever possible. Include the document's stated dates. "
        "If the document does not address the ask, say that plainly."
    )


def run_url_context_read(
    url: str,
    ask: str,
    *,
    api_key: str,
    role: str,
    model: str,
    thinking_level: str,
    timeout_ms: int,
    attempts: int,
) -> tuple[str, int, list[str]]:
    """Read ``url`` via Gemini url_context. Returns ``(text, n_successful_retrievals, statuses)``.

    ``statuses`` is every reported ``url_retrieval_status`` name, in the SDK's order, so a
    caller's suppression log line can say WHY nothing was retrieved. A count of zero is the same
    number whether the fetch was refused, timed out, or the tool never ran, and those are
    different problems.

    Synchronous on purpose: the SDK is, both callers run it under ``asyncio.to_thread``, and the
    client-side ``timeout_ms`` is what returns the worker — an ``asyncio.wait_for`` around a
    thread cancels the coroutine and not the thread, so without a client ceiling a hung endpoint
    leaks a worker into the shared pool. ``attempts`` comes with it because the SDK retries
    NOTHING by default (``retry_args(None)`` is ``stop_after_attempt(1)``), which is how two
    production reads died outright on a ``503 UNAVAILABLE``.
    """
    from google import genai  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import
    from google.genai import types as genai_types  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import

    client = genai.Client(
        api_key=api_key,
        http_options=build_gemini_http_options(timeout_ms=timeout_ms, attempts=attempts),
    )
    tools: list[Any] = [{"url_context": {}}]
    config = genai_types.GenerateContentConfig(
        tools=tools,
        # Explicit rather than the model's default: quoting a fetched document back is the least
        # reasoning-heavy Gemini call the bot makes.
        thinking_config=gemini_thinking_config(thinking_level),
    )
    response = client.models.generate_content(
        model=model,
        contents=f"{build_document_prompt(ask)}\n\nURL: {url}",
        config=config,
    )
    log_gemini_usage(response, role=role, model=model)
    _, _, n_url_success, entries = extract_url_context_telemetry(response)
    text = getattr(response, "text", "")
    return (text if isinstance(text, str) else "", n_url_success, [status for status, _url in entries])
