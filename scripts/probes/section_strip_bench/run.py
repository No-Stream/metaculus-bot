"""The paid half: the model call on the personal key, the spend meter, and the per-call runner.

The forecaster call goes straight through ``litellm.acompletion``, whose OpenRouter transformer asks for
usage accounting on every request, so each reply carries the charge OpenRouter made and is read by the
same reader the production ledger uses. The ladder's salvage parser is built through the repo's own
builder, which after ``personal_key_only_environment`` can only reach the personal key, and its charges
arrive through the litellm success-callback ledger. A charge OpenRouter did not report stops the run:
the meter fails shut rather than reading an unmeasurable call as free.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import litellm
import openai
from forecasting_tools import GeneralLlm

from metaculus_bot.config import load_environment
from metaculus_bot.constants import (
    DONATED_OPENROUTER_KEY_ENABLED_ENV,
    OAI_ANTH_OPENROUTER_KEY_ENV,
    OPENROUTER_API_KEY_ENV,
)
from metaculus_bot.credit_telemetry import (
    PERSONAL_KEY_ALIAS,
    TokenCounts,
    _CallUsage,  # the ledger's own usage reader and its record; one reader in the repo, so the bench shares it
    _openrouter_call_usage,
    llm_call_metadata,
    role_spend_rows,
)
from metaculus_bot.exceptions import UnitMismatchError, ValueExtractionError
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from scripts.probes.section_strip_bench.plan import PlanItem
from scripts.probes.section_strip_bench.report import (
    STATUS_API_ERROR,
    STATUS_BUILD_FAILED,
    STATUS_EXTRACTION_FAILED,
    STATUS_SCORED,
    STATUS_SKIPPED_SPEND_CAP,
    STATUS_UNIT_MISMATCH,
    CallRow,
)
from scripts.probes.section_strip_bench.scoring import score_reply

logger = logging.getLogger(__name__)

# The forecaster slots' own ceiling and timeout (llm_configs.REASONING_MODEL_CONFIG); retries are litellm's.
MAX_COMPLETION_TOKENS = 64_000
CALL_TIMEOUT_S = 480.0
CALL_RETRIES = 2
FORECASTER_ROLE = "strip_bench_forecaster"
PARSER_ROLE = "parser"
ERROR_TEXT_LIMIT = 500
# Bad key, no credit, or a model gate on the account (OpenRouter's 18+ attestation answers 403); never a per-call fault.
ACCOUNT_REFUSAL_STATUSES = frozenset({401, 402, 403})


@dataclass(frozen=True)
class ModelReply:
    """One completion: its text, its token counts, and what OpenRouter charged for it (None when unreported)."""

    text: str
    tokens: TokenCounts
    charged_usd: float | None


def charged_usd(usage: _CallUsage) -> float | None:
    """The money one call cost, by the ledger's rule (``record_llm_call_spend``): the cost, plus the
    upstream charge only on a BYOK route, where OpenRouter reports the real bill there and ``cost`` as 0."""
    if usage.cost_usd is None and usage.byok_upstream_usd is None:
        return None
    upstream = (usage.byok_upstream_usd or 0.0) if usage.is_byok else 0.0
    return (usage.cost_usd or 0.0) + upstream


def reply_from_response(response: Any) -> ModelReply:
    usage = _openrouter_call_usage(response)
    return ModelReply(
        text=response.choices[0].message.content or "", tokens=usage.tokens, charged_usd=charged_usd(usage)
    )


ModelCall = Callable[[str], Awaitable[ModelReply]]


def build_model_call(model: str, api_key: str, *, reasoning_effort: str | None) -> ModelCall:
    """One completion per prompt on the personal key: the prompt as the only message, no tools."""
    kwargs: dict[str, Any] = {
        "model": f"openrouter/{model}",
        "api_key": api_key,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "timeout": CALL_TIMEOUT_S,
        "num_retries": CALL_RETRIES,
        "metadata": llm_call_metadata(FORECASTER_ROLE, PERSONAL_KEY_ALIAS),
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    async def _call(prompt: str) -> ModelReply:
        response = await litellm.acompletion(messages=[{"role": "user", "content": prompt}], **kwargs)
        return reply_from_response(response)

    return _call


def personal_key_only_environment() -> str:
    """Load ``.env``, then make the personal key the ONLY OpenRouter key this process can reach.

    The donated key is removed from the environment and the master switch set false before any client
    exists, so ``build_llm_with_openrouter_fallback`` (which the parser salvage rung builds through) can
    only ever hand litellm the personal key.
    """
    load_environment()
    os.environ.pop(OAI_ANTH_OPENROUTER_KEY_ENV, None)
    os.environ[DONATED_OPENROUTER_KEY_ENABLED_ENV] = "false"
    api_key = os.getenv(OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{OPENROUTER_API_KEY_ENV} must be set: this bench spends only the personal key")
    return api_key


def build_parser_llm(parser_model: str) -> GeneralLlm:
    """The ladder's salvage parser: the same cheap tier at low effort, on the personal key."""
    return build_llm_with_openrouter_fallback(
        f"openrouter/{parser_model}",
        role=PARSER_ROLE,
        temperature=None,
        max_tokens=32_000,
        stream=False,
        timeout=300,
        allowed_tries=1,
        reasoning={"effort": "low"},
    )


def parser_ledger_usd() -> float:
    """What the parser salvage calls have been charged so far, off the litellm success-callback ledger."""
    return sum(row.charged_usd or 0.0 for row in role_spend_rows() if row.role == PARSER_ROLE)


@dataclass
class SpendMeter:
    """Measured spend against the cap: forecaster charges as they return, plus the parser ledger.

    ``stop_reason`` is set once the cap is reached or once any reply came back without a charge, since an
    unmeasured call cannot be held under a cap; the runner skips every later call while it is set.
    """

    cap_usd: float
    forecaster_usd: float = 0.0
    unpriced_calls: int = 0
    refusal: str | None = None

    def add(self, reply_charged_usd: float | None) -> None:
        if reply_charged_usd is None:
            self.unpriced_calls += 1
        else:
            self.forecaster_usd += reply_charged_usd

    def refuse(self, error: str) -> None:
        """A 401/402/403 is the account, not the call: every later call would fail the same way, so stop now."""
        self.refusal = error

    @property
    def measured_usd(self) -> float:
        return self.forecaster_usd + parser_ledger_usd()

    @property
    def stop_reason(self) -> str | None:
        if self.refusal is not None:
            return f"the provider refused the key or the model: {self.refusal}"
        if self.unpriced_calls:
            return f"{self.unpriced_calls} reply(ies) carried no charge, so spend cannot be measured against the cap"
        if self.measured_usd >= self.cap_usd:
            return f"measured spend ${self.measured_usd:.4f} reached the cap ${self.cap_usd:.2f}"
        return None


def _row_for(item: PlanItem, status: str) -> CallRow:
    return CallRow(
        question_id=item.question.question_id,
        qtype=item.question.qtype,
        n_options=item.question.n_options,
        arm=item.arm,
        seed=item.seed,
        status=status,
    )


async def _score_into(
    row: CallRow, item: PlanItem, reply: ModelReply, parser_llm: GeneralLlm, *, model_name: str
) -> None:
    """Fill the row from the reply: the score when the ladder and the build succeed, a status otherwise."""
    row.prompt_tokens, row.completion_tokens = reply.tokens.prompt, reply.tokens.completion
    row.reasoning_tokens, row.cached_tokens = reply.tokens.reasoning, reply.tokens.cached
    row.cost_usd, row.rationale = reply.charged_usd, reply.text
    question = item.question
    try:
        scored = await score_reply(
            question.question, question.resolution, reply.text, parser_llm, model_name=model_name
        )
    except ValueExtractionError as exc:
        row.status, row.error = STATUS_EXTRACTION_FAILED, str(exc)[:ERROR_TEXT_LIMIT]
    except UnitMismatchError as exc:
        row.status, row.error = STATUS_UNIT_MISMATCH, str(exc)[:ERROR_TEXT_LIMIT]
    except ValueError as exc:
        row.status, row.error = STATUS_BUILD_FAILED, str(exc)[:ERROR_TEXT_LIMIT]
    else:
        row.status, row.score, row.forecast = STATUS_SCORED, scored.score, scored.forecast
        row.rung, row.block_present = scored.rung, scored.block_present


async def run_item(
    item: PlanItem,
    *,
    call: ModelCall,
    parser_llm: GeneralLlm,
    model_name: str,
    meter: SpendMeter,
    semaphore: asyncio.Semaphore,
) -> CallRow:
    """One call: forecast, extract, score. Every failure mode is a row status, never the end of the run."""
    async with semaphore:
        if (stop_reason := meter.stop_reason) is not None:
            row = _row_for(item, STATUS_SKIPPED_SPEND_CAP)
            row.error = stop_reason
            return row
        started = time.monotonic()
        row = _row_for(item, STATUS_API_ERROR)
        try:
            reply = await call(item.prompt)
        except openai.OpenAIError as exc:  # litellm's whole exception family derives from openai's
            row.error = f"{type(exc).__name__}: {exc}"[:ERROR_TEXT_LIMIT]
            # litellm's own APIError carries status_code without being an APIStatusError, so read it by name.
            if getattr(exc, "status_code", None) in ACCOUNT_REFUSAL_STATUSES:
                meter.refuse(row.error)
        else:
            meter.add(reply.charged_usd)
            await _score_into(row, item, reply, parser_llm, model_name=model_name)
        row.elapsed_s = time.monotonic() - started
        return row


async def run_plan(
    plan: Sequence[PlanItem],
    *,
    call: ModelCall,
    parser_llm: GeneralLlm,
    model_name: str,
    meter: SpendMeter,
    concurrency: int,
    on_row: Callable[[CallRow], None],
) -> list[CallRow]:
    """Run every planned call under the concurrency limit, handing each finished row to ``on_row`` as it lands."""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(
            run_item(item, call=call, parser_llm=parser_llm, model_name=model_name, meter=meter, semaphore=semaphore)
        )
        for item in plan
    ]
    rows: list[CallRow] = []
    for task in asyncio.as_completed(tasks):
        row = await task
        on_row(row)
        rows.append(row)
    return rows
