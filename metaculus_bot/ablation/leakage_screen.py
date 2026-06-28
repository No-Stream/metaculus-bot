"""LLM-based leakage screen for the ablation benchmark.

Operates on the *cached full research blob* (first-pass + gap-fill addendum)
written by ``metaculus_bot.ablation.research``. Each verdict is cached on disk
via ``AblationCache.write_leakage_screen``.

The behavior here intentionally diverges from the production screen in
``metaculus_bot.backtest.leakage`` in two places:

1. On detector failure, this module returns ``is_leaked=True`` (drop the
   question) rather than ``False`` (keep it). The ablation prefers losing
   data to admitting suspect data.
2. The detector prompt asks for **structured JSON output** rather than a
   YES/NO prefix. Production's prefix matcher (``response.strip().upper()
   .startswith("YES")``) silently mis-parses common LLM decorations such as
   ``**Answer: YES**`` (markdown bold) — observed live on Q43131, where a
   leaky question went undropped. Structured JSON makes the verdict robust
   to surrounding prose / markdown.

The prompt also explicitly enumerates implication-leak patterns (anchored
comparisons, bracketing ranges, threshold framing, date-specific outcomes
inside the resolution window) — patterns the redactor missed live on Q43151
(ISM PMI) where ``"unchanged from March 52.7%"`` revealed the April value.
"""

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any

import openai
from forecasting_tools import GeneralLlm, MetaculusQuestion

from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.backtest.scoring import GroundTruth
from metaculus_bot.research.provider_diagnostics import PROVIDER_DIAGNOSTICS_HEADER

# Transient exceptions worth a single retry. The detector LLM call can fail in
# two ways that legitimately recover on retry: (1) the LLM provider hiccups
# (``openai.APIError`` is the common base for litellm's connection/timeout/
# rate-limit/service-unavailable wrappers — ``litellm.APIError``, ``litellm.
# APIConnectionError``, ``litellm.RateLimitError``, ``litellm.Timeout`` are
# all subclasses of ``openai.APIError`` but NOT of one another — checked
# inheritance directly, see the litellm exception module) or the asyncio
# call times out (``asyncio.TimeoutError``); (2) the LLM emits malformed/
# unexpected JSON, which ``_extract_is_leaked`` raises as ``ValueError``
# (and ``json.JSONDecodeError``, a ValueError subclass, on a strict-JSON
# parse failure before the regex fallback). Anything outside this set is a
# real bug — schema-parse typo, AttributeError from a refactor, KeyError
# from missing dict access — and must propagate so the operator sees it,
# instead of being silently swallowed as ``detector_failed=True``.
_DETECTOR_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    openai.APIError,  # litellm exceptions (RateLimitError, Timeout, APIConnectionError, ...)
    ValueError,  # includes json.JSONDecodeError + _extract_is_leaked failures
)

DEFAULT_DETECTOR_MODEL: str = "openrouter/z-ai/glm-4.5-air:free"
_EMPTY_BLOB_RESPONSE: str = "<empty research; nothing to leak>"
_FAILED_RESPONSE_SENTINEL: str = "<failed>"

logger: logging.Logger = logging.getLogger(__name__)


LEAKAGE_DETECTOR_PROMPT = """You are checking whether a research report reveals \
the actual outcome of a forecasting question.

Question: {question_text}
Resolution criteria: {resolution_criteria}
Actual resolution: {resolution_string}

The research report is provided in <research> tags. Determine whether the report
contains information that clearly reveals or strongly implies the resolution.

Pay attention to IMPLIED reveals as well as direct statements:
- Sentences that anchor current values via comparison to a known prior value
  (e.g. "X was unchanged from its March reading of 52.7%" reveals current X = 52.7%).
- Tight bracketing ranges that include the resolution
  (e.g. "approximately 17.0-17.5 million" tightly bounds the truth).
- Threshold framings that reveal the side of a comparison
  (e.g. "scored just below the 40% threshold" reveals it was below threshold).
- Date-specific outcomes within the question's resolution window
  (e.g. past-tense claims about events on dates inside the resolution period).

Respond ONLY with a JSON object matching this schema:
{{
  "is_leaked": true | false,
  "explanation": "<brief reason>"
}}

Do not include any prose outside the JSON.

<research>
{research_text}
</research>"""


def _build_detector_llm(model: str) -> GeneralLlm:
    """Construct a detector LLM with a generous token budget for reasoning models.

    Production ``leakage.py:22`` uses ``max_tokens=500`` against ``gpt-5.4-mini``,
    where ``max_tokens`` only bounds the visible answer. The ablation default is
    ``glm-4.5-air:free`` — a reasoning model where ``max_tokens`` is the
    *combined* budget for reasoning + content (Z.AI docs:
    https://docs.z.ai/guides/overview/concept-param). At 500, the model can
    exhaust the budget on reasoning alone for long blobs and emit
    ``content=None`` with ``finish_reason="length"``. We keep reasoning ON
    (it's the whole point of using a reasoning model for this judgment task)
    and raise the ceiling to 32k so reasoning + answer both fit comfortably,
    even on the largest research blobs. Empirically Z.AI's ~14 tok/s throughput
    on the free tier makes the latency cost of a higher ceiling small.

    ``response_format={"type": "json_object"}`` is passed via ``extra_body``
    so OpenRouter forwards it to providers that honor JSON-mode (most
    OpenAI-compatible endpoints, including z-ai). Providers that ignore the
    flag still typically emit JSON; ``_extract_is_leaked`` falls back to
    regex extraction for prose-wrapped JSON.
    """
    return GeneralLlm(
        model=model,
        temperature=0.0,
        max_tokens=32_000,
        extra_body={"response_format": {"type": "json_object"}},
    )


_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _extract_is_leaked(raw_response: str) -> bool:
    """Parse ``is_leaked`` from a JSON-shaped LLM response.

    Tries strict JSON parse first; falls back to regex-extracting the first
    ``{...}`` block from prose-wrapped responses (some providers prepend
    chain-of-thought before honoring JSON-mode).

    Raises ``ValueError`` if no parseable ``is_leaked`` boolean is found —
    the caller treats this as a detector failure and conservatively drops.
    """
    stripped = raw_response.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None

    if not isinstance(payload, dict):
        match = _JSON_OBJECT_RE.search(stripped)
        if match is None:
            raise ValueError(f"no JSON object in response: {stripped!r}")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"failed to parse extracted JSON: {match.group(0)!r}") from exc

    if not isinstance(payload, dict) or "is_leaked" not in payload:
        raise ValueError(f"response missing 'is_leaked' field: {payload!r}")

    is_leaked = payload["is_leaked"]
    # Free-tier providers occasionally emit ``"true"`` / ``"false"`` strings
    # inside JSON when the model is sloppy with quoting. Accept those forms
    # (case-insensitive, plus yes/no) rather than dropping the question to
    # detector_failed=True for what's essentially a type-encoding wobble.
    if isinstance(is_leaked, str):
        normalized = is_leaked.strip().lower()
        if normalized in ("true", "yes"):
            return True
        if normalized in ("false", "no"):
            return False
        raise ValueError(f"'is_leaked' string must be true/false/yes/no; got {is_leaked!r}")
    if not isinstance(is_leaked, bool):
        raise ValueError(f"'is_leaked' must be a boolean, got {type(is_leaked).__name__}: {is_leaked!r}")
    return is_leaked


_DETECTOR_RETRY_BACKOFF_SECONDS: float = 2.0


async def _detect_leakage_structured(
    question: MetaculusQuestion,
    ground_truth: GroundTruth,
    research_text: str,
    detector_llm: Any,
) -> tuple[bool, str]:
    """Run the structured-JSON leakage detector. Returns (is_leaked, raw_response).

    Retries the LLM call ONCE on transient failures (network blip, JSON
    parse error). Z.AI's free-tier glm-4.5-air detector occasionally has
    5-10 minute outages; without retry, a brief blip drops a chunk of
    questions to ``detector_failed=True`` (conservative is_leaked=True),
    where re-screen would have succeeded. The detector is free, so the
    retry is cheap insurance.

    After two failed attempts, raises the last exception; the caller
    treats this as ``detector_failed=True`` and conservatively drops.
    """
    prompt = LEAKAGE_DETECTOR_PROMPT.format(
        question_text=question.question_text,
        resolution_criteria=question.resolution_criteria,
        resolution_string=ground_truth.resolution_string,
        research_text=research_text,
    )
    last_exc: BaseException | None = None
    for attempt in range(2):
        try:
            response = await detector_llm.invoke(prompt)
            is_leaked = _extract_is_leaked(response)
            return is_leaked, response
        except _DETECTOR_TRANSIENT_EXCEPTIONS as exc:
            last_exc = exc
            if attempt < 1:
                logger.info(
                    "Leakage detector attempt %d failed (%s: %s); retrying",
                    attempt + 1,
                    type(exc).__name__,
                    exc,
                )
                await asyncio.sleep(_DETECTOR_RETRY_BACKOFF_SECONDS)
                continue
            break
    assert last_exc is not None
    raise last_exc


def _is_content_free_blob(research_blob: str) -> bool:
    """True iff the blob carries no leakable research content.

    A fully-failed research run no longer returns ``""`` — it returns a
    diagnostics-only blob (just the ``## Provider Diagnostics`` section appended
    by the orchestrator: provider names, statuses, timings; nothing leakable). We
    strip a trailing diagnostics section before the emptiness check so such a blob
    short-circuits the detector just like the old empty-string case did, instead
    of being needlessly screened.
    """
    header_idx = research_blob.find(PROVIDER_DIAGNOSTICS_HEADER)
    without_diagnostics = research_blob if header_idx == -1 else research_blob[:header_idx]
    # Trailing ``---`` separator and surrounding whitespace are not research either.
    return without_diagnostics.strip().rstrip("-").strip() == ""


def _research_blob_sha(research_blob: str) -> str:
    """Short SHA-256 prefix used to detect stale cached verdicts.

    A 16-hex-char prefix (8 bytes / 64 bits) collides at the birthday bound of
    ~4 billion blobs — well above any realistic ablation working set. Stored
    on the verdict so a downstream cache reader can detect that the blob was
    re-pruned between screen runs and conservatively re-screen.
    """
    return hashlib.sha256(research_blob.encode()).hexdigest()[:16]


def _build_verdict(
    is_leaked: bool,
    detector_response: str,
    detector_model: str,
    detector_failed: bool,
    research_blob: str,
) -> dict:
    return {
        "is_leaked": is_leaked,
        "detector_response": detector_response,
        "detector_model": detector_model,
        "detector_failed": detector_failed,
        "screened_at": datetime.now().isoformat(),
        "research_blob_sha": _research_blob_sha(research_blob),
    }


async def screen_research_blob(
    question: MetaculusQuestion,
    ground_truth: GroundTruth,
    research_blob: str,
    cache: AblationCache,
    *,
    detector_llm: Any | None = None,
    detector_model: str = DEFAULT_DETECTOR_MODEL,
    force: bool = False,
) -> dict:
    """Screen a cached research blob for leakage. Caches the verdict per qid.

    Returns a dict shaped like::

        {
            "is_leaked": bool,
            "detector_response": str,        # raw response or "<failed>" sentinel
            "detector_model": str,
            "detector_failed": bool,         # True iff the LLM call raised
            "screened_at": "<ISO datetime>",
        }

    Behavior:
    - Cache hit (and not force): return cached payload unchanged.
    - Cache miss (or force=True): build the detector_llm if not supplied, call
      ``_detect_leakage_structured``, record verdict + raw response. On
      exception (transport failure OR JSON parse failure), set is_leaked=True
      (CONSERVATIVE — drop the question, the production screen keeps it) and
      detector_failed=True.
    - Content-free research blob (empty, or diagnostics-only from a fully-failed
      run): short-circuit without calling the LLM.
    - Always writes the verdict to cache before returning (even on detector
      failure).
    """
    # Async-checkpoint at function entry so cache-hit + empty-blob early returns
    # still satisfy flake8-async ASYNC910. The cooperative yield is cheap.
    await asyncio.sleep(0)

    assert question.id_of_question is not None
    qid = question.id_of_question

    if not force:
        cached = cache.read_leakage_screen(qid)
        if cached is not None:
            return cached

    if _is_content_free_blob(research_blob):
        verdict = _build_verdict(
            is_leaked=False,
            detector_response=_EMPTY_BLOB_RESPONSE,
            detector_model=detector_model,
            detector_failed=False,
            research_blob=research_blob,
        )
        cache.write_leakage_screen(qid, verdict)
        return verdict

    if detector_llm is None:
        detector_llm = _build_detector_llm(detector_model)

    try:
        is_leaked, response_text = await _detect_leakage_structured(question, ground_truth, research_blob, detector_llm)
    except _DETECTOR_TRANSIENT_EXCEPTIONS:
        # Conservative-drop: production returns False (keep) on detector
        # failure; we override to True (drop) for the ablation. Transient
        # transport errors (litellm.APIError + subclasses, asyncio timeouts)
        # and JSON-parse failures (ValueError, json.JSONDecodeError) land
        # here — either way we don't trust the verdict but we know the
        # detector itself isn't buggy. Anything outside this set (KeyError,
        # AttributeError, TypeError from a refactor regression) propagates
        # so the operator sees real bugs instead of losing questions silently
        # to detector_failed=True.
        logger.exception(f"Leakage detector failed for qid {qid}")
        verdict = _build_verdict(
            is_leaked=True,
            detector_response=_FAILED_RESPONSE_SENTINEL,
            detector_model=detector_model,
            detector_failed=True,
            research_blob=research_blob,
        )
    else:
        verdict = _build_verdict(
            is_leaked=is_leaked,
            detector_response=response_text,
            detector_model=detector_model,
            detector_failed=False,
            research_blob=research_blob,
        )
        logger.info(f"Leakage screen Q{qid}: is_leaked={is_leaked}")

    cache.write_leakage_screen(qid, verdict)
    return verdict


async def _screen_under_semaphore(
    question: MetaculusQuestion,
    ground_truth: GroundTruth,
    research_blob: str,
    cache: AblationCache,
    detector_llm: Any,
    detector_model: str,
    force: bool,
    semaphore: asyncio.Semaphore,
) -> tuple[int, dict]:
    assert question.id_of_question is not None
    qid = question.id_of_question
    async with semaphore:
        verdict = await screen_research_blob(
            question,
            ground_truth,
            research_blob,
            cache,
            detector_llm=detector_llm,
            detector_model=detector_model,
            force=force,
        )
    return qid, verdict


async def screen_batch(
    questions: list[MetaculusQuestion],
    ground_truths: dict[int, GroundTruth],
    research_cache_payloads: dict[int, str],
    cache: AblationCache,
    *,
    detector_model: str = DEFAULT_DETECTOR_MODEL,
    force: bool = False,
    concurrency: int = 4,
) -> tuple[list[MetaculusQuestion], dict[int, GroundTruth], dict[int, dict]]:
    """Run ``screen_research_blob`` on a batch under a semaphore.

    Builds the detector_llm once (shared across calls). Skips questions missing
    from ``research_cache_payloads`` — they have no research to screen and are
    excluded from the clean set (no research means we can't trust the
    question).

    Returns ``(clean_questions, clean_ground_truths, all_verdicts)`` where
    ``clean_*`` exclude any qid whose verdict has ``is_leaked=True``, and
    ``all_verdicts`` maps every screened qid to its verdict dict. Logs the
    final leakage rate; warns at >50% (matches production semantics).
    """
    detector_llm = _build_detector_llm(detector_model)
    semaphore = asyncio.Semaphore(concurrency)

    questions_with_research: list[MetaculusQuestion] = []
    questions_without_research: list[MetaculusQuestion] = []
    for question in questions:
        if question.id_of_question in research_cache_payloads:
            questions_with_research.append(question)
        else:
            questions_without_research.append(question)

    for question in questions_without_research:
        logger.warning(
            f"Q{question.id_of_question}: no research blob in cache; skipping leakage screen and excluding from clean set"
        )

    tasks = []
    for q in questions_with_research:
        qid = q.id_of_question
        assert qid is not None
        tasks.append(
            _screen_under_semaphore(
                q,
                ground_truths[qid],
                research_cache_payloads[qid],
                cache,
                detector_llm,
                detector_model,
                force,
                semaphore,
            )
        )
    results = await asyncio.gather(*tasks)

    all_verdicts: dict[int, dict] = {qid: verdict for qid, verdict in results}
    clean_qids: set[int] = {qid for qid, verdict in all_verdicts.items() if not verdict["is_leaked"]}

    clean_questions = [q for q in questions_with_research if q.id_of_question in clean_qids]
    clean_ground_truths = {qid: gt for qid, gt in ground_truths.items() if qid in clean_qids}

    total_screened = len(all_verdicts)
    leaked_count = total_screened - len(clean_qids)
    leakage_rate_pct = (leaked_count / total_screened * 100) if total_screened > 0 else 0.0
    logger.info(
        f"{leaked_count}/{total_screened} questions excluded due to research leakage ({leakage_rate_pct:.0f}% leakage rate)"
    )

    if leakage_rate_pct > 50:
        logger.warning(
            f"High leakage rate: {leakage_rate_pct:.0f}% of questions had research leakage. "
            f"Research provider may be returning resolution information."
        )

    return clean_questions, clean_ground_truths, all_verdicts
