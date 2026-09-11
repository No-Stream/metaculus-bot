# Fall merge completion and smoke readiness

The recovered fetch-ladder migration is merged into `mantic-competition`, and the existing
known-API and page-digest implementations are wired into production. The reviewed code commit is
`8ec0f2cf253d7412ba5145b5548498d269c68441`. All local implementation and verification work is
complete; the change is ready for the operator-controlled push, remote CI and paid smoke.

## Delivered

- Completed ladder steps 4–6: derived-feed reuse, complete-read cache, shared throttle verdict,
  digest seat, duplicate-path removal, and caller telemetry. The approved separate host-semaphore
  scopes remain in place. Merge: `d41e43e`.
- Bound the real page digest to the policies (`55cf837`). Tests cover grounded tail passages,
  cache reuse, remaining-wall BM25 fallback, full-document reads and preserved PDF page labels.
- Bound known-API URL translation and the three explicit tools (`8ec0f2c`). Two-series FRED URLs
  return both series and canonical links or decline; API lookups use the current remaining wall.
  Explicit tools and rung zero share the gap-fill question's session and Kalshi counter.
- Fixed both findings from the integrated review: the free diagnostic uses deterministic BM25
  (`a86c7f9`), and successful explicit API results earn fetched evidence tiers from canonical
  backend links (`e913dd0`). Ordinary `fetch(url)` retains its requested-URL-only evidence rule.
- Removed the clean, merged Linux ladder worktree. The `fall/ladder` branch is retained.

## Verification

The ladder branch passed 9,854 tests before merge. The merged integration suite passed 1,312
tests. Final independent review of `d41e43e..8ec0f2c` passed with no remaining verified material
bug, including 221 focused tests and four fresh-process import permutations. The requested
`/forge` command was unavailable; independent source review and executable regressions supplied
that gate.

At the final code commit, credential-free Ruff, formatting, basedpyright, deptry, all six import
contracts, and `uv build` passed. Both the source distribution and wheel were built. Gitleaks
8.30.1 scanned `main..HEAD` (197 commits), with zero findings. OSV 2.3.8 scanned 228 packages with
zero unignored vulnerabilities; the existing five cryptography advisory exceptions remain.
The scanned lockfile SHA-256 is
`a35a81e919e88febf067d711a725e41bc7bb60c7adb33473e4902916367eacc2`.

The full credential-free coverage gate passed: **10,136 passed, 39 skipped, 5 live deselected**,
with 93% total coverage, in 418.85 seconds. HEAD stayed at `8ec0f2c` throughout that run.
The final follow-up added one tool-description clause, its presence test, and documentation;
all 221 agentic-tool tests, Ruff, formatting and a rebuilt wheel/source distribution passed.
All non-comment lines in the seven touched workflow files were verified unchanged. Logs are
`/tmp/fall-final-cov.log`, `/tmp/fall-final-static.log`, `/tmp/fall-final-build.log`, and
`/tmp/fall-api-description-green.log` on this machine.

A focused command that interleaved files from `tests/resolution_source` with a root test file
created duplicate pytest directory collectors and omitted package fixtures on the second visit.
Grouping the files passed all 63 tests. No test was disabled or given a network exemption; the
whole-suite gate uses normal root discovery.

## Archive replay and its limits

The full available artifact archive was synced. The replay evaluated 964 eligible outcomes under
each of four policies; 386 outcomes lacked the direct status needed for replay, and 57 expired
artifacts were unavailable. This verifies reconstructed trigger/order selection, not live response
bytes, rescue success, or timing. The two apparent lost outcomes under gap-fill policies were
historical resolution-source paid-reader attempts, a rung those policies deliberately exclude;
they are not two lost successful rescues. See `fetch_ladder_replay_2026-09-10.md` for the counts,
policy comparison, omissions and JSON artifact path.

## Operator-controlled smoke

No push, paid run, workflow dispatch, publication, or merge to `main` was performed. Remote CI
has not run for the local commits. After an approved push and remote CI, the one-question smoke is:

```sh
gh workflow run test_bot_basic.yaml --repo No-Stream/metaculus-bot --ref mantic-competition
```

The documented estimate is about $2.60, with actual cost dependent on the research calls. It
publishes a Metaculus forecast/comment and requires separate per-run approval. The Mantic
dispatcher enablement remains a separate, approved action after the production merge.
