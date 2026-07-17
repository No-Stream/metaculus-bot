# Exa-alive confirmation replay: arm_terra_low_exa_alive vs arm_terra_low

Same frozen brief (user_brief.md, byte-identical), same driver (openai/gpt-5.6-terra,
effort=low), same LoopConfig (14 calls / 540s / 90s conclude threshold). Only delta:
`.env` now carries a valid `EXA_API_KEY`, so `search_web` works instead of 401ing.

## Pass criteria (all met)

- **search_web returns real results**: 4/4 calls `status: ok` with dated ECZ / ZambiaLII
  results; zero 401s or `status: error` anywhere in the transcript.
- **No deadline_hit**: `deadline_hit=false`, `concluded_early=true`. The loop exhausted
  the 14-call tool budget and concluded cleanly when told "you must conclude now"
  (anytime-output path, working as designed).
- **Findings detached + cited**: all 9 findings carry Source + verbatim Quote + Date;
  no likelihood language (ghost forecast stays confined to ghost.json, never published).
- **wall_s**: 52.6 loop / 54.3 outer — well under 300.
- **No unhandled errors**: exit 0; only the two known-benign `structured_output_schema`
  WARNINGs (ghost parser probing binary/MC schemas against a numeric payload — identical
  lines appear in the original arm).

## What Exa changed

| metric | terra_low (Exa 401) | terra_low_exa_alive |
|---|---|---|
| steps | 5 | 6 |
| tool_calls | 11 | 14 |
| search_web (ok/total) | 0/2 | 4/4 |
| fetches (rendered) | 6 (0) | 7 (1) |
| findings | 7 | 9 |
| pending_leads | 3 | 3 |
| lint_rejections | 0 | 0 |
| loop wall_s | 30.3 | 52.6 |
| est cost | $0.36 | $0.52 |

- **Tool allocation**: with search alive the driver leaned into it — 4 searches vs 2
  (both of which had 401'd), and it spent the freed budget on follow-up fetches of
  search hits (ECZ 2021 stats page, ZambiaLII) rather than concluding at 11 calls.
- **Findings count/quality**: 7 → 9. The two extra findings are exactly what the
  original arm listed as unresolved pending leads: the ECZ final-register total
  (8,786,300 voters, from the ECZ homepage via search) and the Constitutional Court
  Lungu-eligibility judgment ([2024] ZMCC 27, via ZambiaLII search hit). Search also
  surfaced the 2021 election-statistics page, which the driver fetched (the one
  rendered/Chromium escalation).
- **Pending leads shifted from infrastructure to substance**: the original arm's leads
  included "Exa searches failed"; the new arm's leads are genuine research gaps
  (verbatim official 2021 percentage, final 2026 candidate list, post-nomination poll).
- **Ghost forecast barely moved**: median 56.0 → 55.1 — the extra evidence was
  confirmatory, not directional, which is what you'd want from a corrections-oriented
  research pass on a well-briefed question.

Verdict: PASS. Exa being alive improves the loop exactly along the intended axis
(resolving leads that fetch-only couldn't) with no behavioral regressions.
