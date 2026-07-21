# Smoke-run cost decomposition — 2026-07-17 (Zambia Q44229/44240)

One-question gap-fill-v2 smoke, log at `/tmp/v2-smoke.log`, window **11:08:45–11:18:23 PT**
(18:08–18:18 UTC). Measured spend: **$3.34 donated** (`limit_remaining` 95.56 → 92.22) +
**$0.19 personal** (`usage` 23.41 → 23.60). Roster: 2026-07-15 xhigh bump
(sol/5.5/fable/opus at `effort=xhigh`, grok high, gemini-pro default).

## What the auth/key fields actually mean (verified against live pulls)

`GET /api/v1/auth/key` on the donated key at 21:05 UTC returned
`limit=850, limit_remaining=91.125161523, usage=4.158858877, byok_usage=754.7159796`.
Identity check: `850 − 4.158858877 − 754.7159796 = 91.125161523` — exact to 9 decimals. So:

- **`limit`** — cumulative credits ever provisioned on the key ($850 granted by Metaculus
  to date), not a monthly cap (`limit_reset=null`).
- **`usage`** — lifetime spend billed as *native OpenRouter credits* only.
- **`byok_usage`** — lifetime spend routed through BYOK provider integrations (Metaculus
  attached provider-native OpenAI/Anthropic/Google keys to the donated account). With
  `include_byok_in_limit: true`, BYOK spend draws down the limit too.
- **`limit_remaining`** = `limit − usage − byok_usage` (when `include_byok_in_limit`).

That resolves the "inconsistent" triple (limit 850 / remaining ~92 / usage 4.16): nearly all
donated-key traffic is BYOK-routed, so `usage` is effectively frozen at $4.16 while
`byok_usage` does all the moving. This is also exactly why `CREDIT_SPEND` reported
`run_delta_usd=0.00` — it diffed `usage`. (Fixed in `credit_telemetry.py`: delta now comes
from `limit_remaining` when present, `usage` otherwise.)

The personal key is the opposite: `limit=null` (uncapped), spend lands in `usage`
(`usage_daily=0.194954` — matching the smoke's $0.19 exactly).

## Attribution — the confound is clean

- **No GHA overlap.** The smoke ran 18:08–18:18 UTC; the nearest tournament runs were
  17:33Z and 18:38Z, both ~30 s no-ops (skip-already-forecasted; a real forecast takes
  ~6 min). The apparent 11:08 collision was a UTC-vs-PT coincidence.
- **UTC-day reconciliation (donated).** `byok_usage_daily` at 21:05 UTC = **$7.41** =
  smoke ($3.34) + the one real tournament run of the day (Q44466 at 09:23Z, **$4.07**
  by subtraction). Zero unexplained spend.
- **UTC-day reconciliation (personal).** `usage_daily = 0.194954` — the smoke was the
  *only* personal-key spend all day.
- **TS-anchor session: ~$0 on both keys today.** Its commit (21ebeb7, 12:11 PT) is
  "replay-validated" (offline replay), and the day's ledger leaves it no room. High
  confidence the $3.34/$0.19 are entirely the smoke's.
- Bonus prod datapoint: between my two API pulls, the 21:11Z run forecast Q44642 for
  **$4.66** donated (`limit_remaining` 91.13 → 86.47 = `byok_usage_daily` 7.41 → 12.07).

## Component decomposition

No per-call token counts exist (forecasting-tools cost tracking unsupported for every model
in the roster), so rows are bottom-up estimates from OpenRouter pricing + observed output
sizes + wall-clock (a proxy for reasoning tokens), constrained to sum to the exact key
totals. Individual rows ±40%; the key-level split is exact.

| Component | Model (key) | Est. USD | Confidence / basis |
|---|---|---:|---|
| Forecaster: gpt-5.6-sol xhigh | donated | ~0.85 | low-med — 341 s wall, 12.9k-char answer, $5/$30 per M, xhigh reasoning dominates |
| Forecaster: gpt-5.5 xhigh | donated | ~0.55 | low-med — 209 s, 11.1k chars, $5/$30 |
| Forecaster: claude-fable-5 xhigh | donated | ~0.45 | low-med — 56 s, 6.7k chars, priciest rates ($10/$50) |
| Forecaster: claude-opus-4.8 xhigh | donated | ~0.20 | low-med — 46 s, 4.6k chars, $5/$25 |
| Forecaster: gemini-3.1-pro-preview | personal (pinned) | ~0.11 | med — $2/$12, 86 s |
| Forecaster: grok-4.5 high | personal (x-ai 404s on donated) | ~0.08 | med — $2/$6, 107 s; the two personal rows must sum to 0.19 and do |
| AskNews summarizer (sol low) | donated | ~0.25 | med — briefing output ~33k chars (~8k tok) at $30/M out |
| Native-search primary (sol low + web tool) | donated | ~0.20 | low — search-tool billing is opaque; 102 s, 8.8k chars |
| Gap-fill v1 (terra analyzer + 4 parallel sol web searches) | donated | ~0.55 | low — analyzer cheap (~0.05); 4 sol search calls carry it |
| Gap-fill v2 driver (gpt-5.6-luna medium) | donated | ~0.30 | low-med — 7 steps / 15 tool calls of accumulating context, but $1/$6 pricing keeps it modest |
| Misc (financial classifier gpt-5.4-mini; parser LLM **unused** — all extractions `rung=block`) | donated | ~0.01 | high |
| Gemini grounded search + v2 reader (google-genai) | GOOGLE_API_KEY | 0.00 marginal | high — free tier, off OpenRouter entirely |
| Stacking chain | — | 0.00 | certain — spread 0.343 > 0.15 but numeric stacking disabled in prod flags; no stacker/crux/targeted-search calls |
| **Total** | | **~3.53** | donated 3.34 exact + personal 0.19 exact |

Roughly: **forecaster fan-out ~$2.25 (64%)**, research stack ~$1.30 (36%), of which
gap-fill (v1+v2) ~$0.85. The v2 driver itself is a small line item — luna is cheap; the
expensive research is the v1 sol web-search fan-out it currently runs *alongside*.

## Is ~$3/question the new normal? Burn rate vs. balance

Yes — and it's the cheap end. Three same-day datapoints on the xhigh roster:

| Question | Kind | Donated cost |
|---|---|---:|
| Zambia smoke (44240) | numeric, no stacker | $3.34 (+0.19 personal) |
| Q44466 (09:23Z tournament) | numeric | ~$4.07 |
| Q44642 (21:11Z tournament) | — | ~$4.66 |

Call it **~$3.5–4.5/question donated** for research-heavy questions post-xhigh-bump
(2026-07-15). A stacker-triggered binary would add ~$0.5–1 (fable xhigh + sol targeted
search).

Burn: `byok_usage_daily` $12.07 today, `byok_usage_weekly` $46.07,
`byok_usage_monthly` $179.00 over 16.5 days of July (≈ $10.9/day — and July's first half
was mostly the *cheaper* pre-xhigh roster). At $11–12/day the projected quarterly burn is
**~$1,000/quarter**, against **$86.47 remaining** at last pull.

**That is ~7–10 days of runway.** A top-up request to Metaculus is warranted now, not at
the $50 floor — the CREDIT_FLOOR_BREACH alarm would fire in ~3 days at this cadence. Of
the $850 provisioned lifetime, ~$759 is consumed.

Levers if a top-up stalls: xhigh → high on the two OpenAI forecasters (sol's 341 s wall
suggests the largest reasoning-token bill in the ensemble), or trim the v1 gap-fill sol
fan-out once v2 replaces it (~$0.75/question combined).
