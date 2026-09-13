# Fix: instant transient `litellm.Timeout` on `allowed_tries=1` LLM calls

**Branch:** `goog-cost` (current; mid-feature — stay on it)
**Status:** implementing (TDD)
**Date:** 2026-06-25

## Problem (root cause)

CI run `test-run-output-200kish-tok-jun-25.md` line 466:
```
GapFill: analyzer failed (Timeout): litellm.Timeout: Connection timed out.
Timeout passed=120.0, time taken=0.001 seconds
```
A "timeout" that fires in **1 ms** is not a real timeout. Traced to litellm 1.80.0
`llms/custom_httpx/http_handler.py:387` — litellm caught an `httpx.TimeoutException`
(the aiohttp transport, default since v1.71.x) and re-wrapped it as `litellm.Timeout`.
Under concurrent async bursts the aiohttp transport raises near-instant connection
failures. (Exact 1 ms signature not matched to a specific GH issue; the broader
"aiohttp under concurrency → spurious instant timeouts" pattern IS documented —
litellm issue #14895 recommends `DISABLE_AIOHTTP_TRANSPORT=true` as the fix.)

**Why no recovery:** forecasting-tools' retry decorator
(`retryable_model.py:39`) is tenacity `stop_after_attempt(allowed_tries)` with **no
exception predicate**. `allowed_tries=1` ⇒ exactly one attempt, zero retries, on any
error. So the four `allowed_tries=1` litellm configs lose all work on a single blip.

## User constraints (load-bearing)

- Retry **fast** failures (sub-second / couple-second) — those are transient.
- **Never** retry a **slow** failure (e.g. a real timeout after minutes) — retrying a
  120s+ stall 3× would "break everything." This is the hard gate.
- 3 retries, backoff ~1s / 10s / 30s.
- Be thorough: fix every site with the same vulnerability.
- Do both: the retry AND the transport flag. Briefly live-test the flag.
- Documentation-grounded, not guessed.

## Round-2 refinements (2026-06-25, post-implementation)

User clarified the elapsed gate is a UNIVERSAL rule, not transient-only:
> "in general, we should only retry if less than 30 seconds has passed. So if we
> try for five minutes or we have a partial output and then we time out, we
> shouldn't fire off another call or that will end up being late for the
> question submission deadline."

Three follow-on changes:
1. **Threshold 10s → 30s.** `TRANSIENT_RETRY_MAX_ELAPSED_S = 30.0`. The gate measures
   the *failed attempt's own duration*, so a 5-min reasoning attempt that then times
   out is never retried (deadline safety); only sub-30s blips retry. Still far below
   every real per-call timeout (120/300/360/420/480/500s).
2. **Gap-fill resolver effort medium → low** (`GAP_FILL_RESOLVER_REASONING_EFFORT`).
   The resolver was the ~5-min critical-path bottleneck; low effort is ~4.5× faster
   (per the native_search v3 bench note in constants.py). Quality to be spot-checked
   from the GHA test-bot run. Update the `test_gap_fill_pass.py` "medium" assertion.
3. **Extend the 30s gate to the `allowed_tries≥2` sites** (6 forecasters, crux
   analyzer, AskNews summarizer). These currently retry on ANY error up to N× via
   forecasting-tools' tenacity — including SLOW timeouts (bounded only by soft
   deadlines, but still wasteful + deadline-risky). forecasting-tools' tenacity has no
   elapsed gate and we can't modify it, so the ONLY way to impose "no retry after 30s"
   is to set those instances to `allowed_tries=1` and wrap their invoke in a gated
   helper. They need a BROAD predicate (retry any non-permanent error — preserving
   their existing valuable fast-retries like empty-model-response), NOT the
   transient-only one, all under the same 30s gate.

→ Add `invoke_with_broad_retry` (or a `predicate=` param on the existing helper):
   same elapsed gate + backoff, but retries any exception EXCEPT clearly-permanent
   ones (`litellm.AuthenticationError`, `BadRequestError`, `NotFoundError`,
   `PermissionDeniedError`, `UnprocessableEntityError`, `ContentPolicyViolationError`,
   `ContextWindowExceededError`). Keep `asyncio.TimeoutError` retryable-by-type but the
   30s gate stops it anyway (it only fires at the wall cap ≫ 30s). NOTE: forecasters
   share `REASONING_MODEL_CONFIG` but are SEPARATE GeneralLlm instances from PARSER_LLM
   — setting forecaster `allowed_tries=1` does NOT touch the parser (verified
   llm_configs.py:54-90 vs :119). Per-instance override: `{**REASONING_MODEL_CONFIG,
   "allowed_tries": 1}`.

## Design — elapsed-gated transient retry (original, 10s→now 30s)

## Design — elapsed-gated transient retry

New module `metaculus_bot/llm_retry.py`:

```
TRANSIENT_RETRY_EXCEPTIONS = (litellm.Timeout, litellm.APIConnectionError,
                              litellm.InternalServerError, litellm.ServiceUnavailableError)
# RateLimitError deliberately excluded — handled by FallbackOpenRouterLlm key-swap + AskNews backoff.

invoke_with_transient_retry(make_awaitable, *, wall_timeout, label,
                            backoffs=(1.,10.,30.), max_elapsed_s=TRANSIENT_RETRY_MAX_ELAPSED_S=10.):
  for attempt in 0..len(backoffs):
    start = monotonic()
    try: return await asyncio.wait_for(make_awaitable(), timeout=wall_timeout)
    except Exception as e:
      elapsed = monotonic() - start
      if last_attempt or not is_fast_transient(e, elapsed): raise
      log warning; await asyncio.sleep(backoffs[attempt])
```

**The elapsed gate is the safety mechanism.** `is_fast_transient` returns True only if
`elapsed < max_elapsed_s` AND `isinstance(e, TRANSIENT_RETRY_EXCEPTIONS)`. A real
wall-clock `asyncio.TimeoutError` (fires at `wall_timeout`, elapsed ≈ wall_timeout ≫ 10s)
is therefore NEVER retried. A 120s `litellm.Timeout` is NEVER retried. Only the
sub-second blips retry. `max_elapsed_s=10` sits well above "couple seconds" and far
below any real timeout (120/300/360/420s). Worst-case added latency on all-fast-fail:
~41s backoff + final attempt up to wall_timeout. Bounded.

**`make_awaitable` is a factory** (not a coroutine) — fresh awaitable per attempt
(coroutines are single-await).

**Avoid double-retry:** every wrapped site keeps `allowed_tries=1` so the inner
tenacity is a no-op and our gated wrapper is the SOLE retry layer. The financial
classifier (currently default `allowed_tries=2`) must be set to `allowed_tries=1`
when wrapped, else inner tenacity would retry slow failures (the exact thing we're
avoiding).

**Stacker compatibility:** stacker is `allowed_tries=1` *by design* (prefers
cross-provider fallback over retrying a stalled provider). The gated retry only fires
on FAST blips (<10s) — NOT the "stalled for minutes" case — so it composes correctly:
fast blip → cheap same-provider retry; slow stall → no retry → existing
STACKER_FALLBACK_LLM path. Design intent preserved.

## Call sites to wrap (from parallel sweep)

| Site | File | allowed_tries | wall guard | action |
|---|---|---|---|---|
| gap-fill resolver | `targeted.py:_resolve_single_gap` (~236) | 1 | NATIVE_SEARCH_WALL_TIMEOUT=420 | wrap |
| gap-fill analyzer | `targeted.py:_run_analyzer` (~189) | 1 | GAP_FILL_ANALYZER_WALL_TIMEOUT=135 | wrap |
| native_search provider | `providers.py:_fetch` (~406) | 1 | NATIVE_SEARCH_WALL_TIMEOUT=420 | wrap |
| targeted search | `targeted.py:run_targeted_search` (~101) | 1 | NATIVE_SEARCH_WALL_TIMEOUT=420 | wrap |
| stacker primary+fallback | `aggregation_pipeline.py` (~294,321) / `stacking.py` | 1 | STACKER_SOFT_DEADLINE=500 / 300 | wrap |
| financial classifier | `financial_data.py:_classify_financial_question` (~73) | 2→1 | NONE today | wrap + set tries=1 + adds the missing wall cap |

Already protected (tries≥2, recover from instant blip): 6 forecasters, parser,
crux analyzer, AskNews summarizer. **Pre-existing observation (not in scope):** those
tries≥2 sites WILL retry a slow timeout (no elapsed gate) up to N×, bounded by their
soft deadlines. Flag to user; separate change.

Different SDK (not this bug): `gemini_search.py` (google-genai direct), AskNews (own
retry loop). Noted, not touched.

## Transport flag

`os.environ.setdefault("DISABLE_AIOHTTP_TRANSPORT", "true")` set BEFORE any litellm
import (top of package `__init__` or main entry) so it applies to CI, local runs, and
backtests. Also add `DISABLE_AIOHTTP_TRANSPORT: 'true'` to the workflow yamls for
visibility. Documented env var, default False, reversible.

## Validation

- TDD unit tests for the helper (fast-blip retries; slow-timeout does NOT; backoff
  sequence; exception-type filter; success-after-retry; wall-clock still bounds).
- Existing suites: test_gap_fill_pass, test_targeted_research, test_native_search_provider,
  test_financial_data_provider, test_stacking, test_aggregation_pipeline must stay green.
- `make test`, `make lint`, `make typecheck`.
- Live flag test (PAID, user-run or approved): tiny script — set flag, fire 2 concurrent
  cheap OpenRouter calls, assert success + report active transport. Surface cmd + cost.
