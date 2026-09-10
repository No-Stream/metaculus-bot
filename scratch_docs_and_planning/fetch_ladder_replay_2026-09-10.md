Fetch-ladder archive replay. Every rung is stubbed to record its attempt and DECLINE, so the
replayed status always equals the reconstructed direct status and the replayed route is the last
rung consulted, not one that served bytes. What this measures is the rung SEQUENCE each preset
would attempt per archived outcome, and the archived rescues its dispatcher no longer attempts.

Presets found on metaculus_bot.research.fetch_ladder.policy: GAP_FILL_DIRECT_POLICY, GAP_FILL_DOCUMENT_POLICY, GAP_FILL_FETCH_POLICY, RESOLUTION_SOURCE_POLICY

Records read:
  fetcher: 129 records, 9 naming the rung that produced them, 129 replayable, 124 carrying a fidelity note
  loop: 1221 records, 1221 naming the rung that produced them, 835 replayable, 398 carrying a fidelity note

Recorded but NOT replayed (the archive does not carry the direct status behind them):
  loop/ok/document                           156
  loop/ok/rendered                            80
  loop/error/document                         67
  loop/ok/cache                               50
  loop/ok/digest_local                        20
  loop/ok/impersonate                         10
  loop/ok/pdf_local                            3

=== GAP_FILL_DIRECT_POLICY === 964 replayed, 13 distinct cells, 2 lost rescues, 2 undecidable (the rescue overwrote the direct http_status its trigger reads)
  archived              route       replayed              route            n  rung sequence / example host
  no_resolving_content  url_context blocked               direct           2  (none) [scottaaronson.blog]  <- archived rescue never attempted
  success               wayback     blocked               direct           1  (none) [www.bls.gov]  <- rescue verdict undecidable: the direct http_status is lost
  success               impersonate blocked               direct           1  (none) [results.cik.bg]  <- rescue verdict undecidable: the direct http_status is lost
  ok                    plain       success               direct         542  (none) [manifold.markets]
  blocked               plain       blocked               direct         169  (none) [www.metaculus.com]
  error                 plain       error                 direct          84  (none) [pubmed.ncbi.nlm.nih.gov]
  success               direct      success               direct          75  (none) [scottaaronson.blog]
  error                 plain       not_found             direct          29  (none) [population.un.org]
  blocked               direct      blocked               direct          25  (none) [www.bls.gov]
  js_wall               direct      js_wall               direct          12  (none) [www.ocearch.org]
  empty                 empty       js_wall               direct          11  (none) [robotracker.app]
  error                 direct      error                 direct           9  (none) [ballotpedia.org]
  not_found             direct      not_found             direct           4  (none) [au.finance.yahoo.com]

=== GAP_FILL_DOCUMENT_POLICY === 964 replayed, 15 distinct cells, 2 lost rescues, 2 undecidable (the rescue overwrote the direct http_status its trigger reads)
  archived              route       replayed              route            n  rung sequence / example host
  no_resolving_content  url_context blocked               direct           1  (none) [scottaaronson.blog]  <- archived rescue never attempted
  no_resolving_content  url_context blocked               impersonate      1  impersonate [www.sagaftra.org]  <- archived rescue never attempted
  success               wayback     blocked               direct           1  (none) [www.bls.gov]  <- rescue verdict undecidable: the direct http_status is lost
  success               impersonate blocked               direct           1  (none) [results.cik.bg]  <- rescue verdict undecidable: the direct http_status is lost
  ok                    plain       success               direct         542  (none) [manifold.markets]
  blocked               plain       blocked               impersonate    126  impersonate [www.imf.org]
  error                 plain       error                 direct          84  (none) [pubmed.ncbi.nlm.nih.gov]
  success               direct      success               direct          75  (none) [scottaaronson.blog]
  blocked               plain       blocked               direct          43  (none) [www.metaculus.com]
  error                 plain       not_found             direct          29  (none) [population.un.org]
  blocked               direct      blocked               impersonate     25  impersonate [www.bls.gov]
  js_wall               direct      js_wall               rendered        12  rendered [www.ocearch.org]
  empty                 empty       js_wall               rendered        11  rendered [robotracker.app]
  error                 direct      error                 direct           9  (none) [ballotpedia.org]
  not_found             direct      not_found             direct           4  (none) [au.finance.yahoo.com]

=== GAP_FILL_FETCH_POLICY === 964 replayed, 15 distinct cells, 2 lost rescues, 1 undecidable (the rescue overwrote the direct http_status its trigger reads)
  archived              route       replayed              route            n  rung sequence / example host
  no_resolving_content  url_context blocked               wayback          1  wayback [scottaaronson.blog]  <- archived rescue never attempted
  no_resolving_content  url_context blocked               wayback          1  impersonate -> wayback [www.sagaftra.org]  <- archived rescue never attempted
  success               impersonate blocked               wayback          1  wayback [results.cik.bg]  <- rescue verdict undecidable: the direct http_status is lost
  ok                    plain       success               direct         542  (none) [manifold.markets]
  blocked               plain       blocked               wayback        126  impersonate -> wayback [www.imf.org]
  error                 plain       error                 wayback         84  wayback [pubmed.ncbi.nlm.nih.gov]
  success               direct      success               direct          75  (none) [scottaaronson.blog]
  blocked               plain       blocked               wayback         43  wayback [www.metaculus.com]
  error                 plain       not_found             wayback         29  wayback [population.un.org]
  blocked               direct      blocked               wayback         25  impersonate -> wayback [www.bls.gov]
  js_wall               direct      js_wall               rendered        12  derived_api -> rendered [www.ocearch.org]
  empty                 empty       js_wall               rendered        11  derived_api -> rendered [robotracker.app]
  error                 direct      error                 wayback          9  wayback [ballotpedia.org]
  not_found             direct      not_found             wayback          4  wayback [au.finance.yahoo.com]
  success               wayback     blocked               wayback          1  wayback [www.bls.gov]

=== RESOLUTION_SOURCE_POLICY === 964 replayed, 16 distinct cells, 0 lost rescues, 1 undecidable (the rescue overwrote the direct http_status its trigger reads)
  archived              route       replayed              route            n  rung sequence / example host
  success               impersonate blocked               url_context      1  wayback -> url_context [results.cik.bg]  <- rescue verdict undecidable: the direct http_status is lost
  ok                    plain       success               direct         542  (none) [manifold.markets]
  blocked               plain       blocked               url_context    126  impersonate -> wayback -> url_context [www.imf.org]
  error                 plain       error                 url_context     84  wayback -> url_context [pubmed.ncbi.nlm.nih.gov]
  success               direct      success               direct          75  (none) [scottaaronson.blog]
  error                 plain       not_found             wayback         29  wayback [population.un.org]
  blocked               direct      blocked               url_context     25  impersonate -> wayback -> url_context [www.bls.gov]
  blocked               plain       blocked               wayback         22  wayback [www.metaculus.com]
  blocked               plain       blocked               url_context     21  wayback -> url_context [query1.finance.yahoo.com]
  js_wall               direct      js_wall               url_context     12  derived_api -> rendered -> url_context [www.ocearch.org]
  empty                 empty       js_wall               url_context     11  derived_api -> rendered -> url_context [robotracker.app]
  error                 direct      error                 url_context      9  wayback -> url_context [ballotpedia.org]
  not_found             direct      not_found             wayback          4  wayback [au.finance.yahoo.com]
  no_resolving_content  url_context blocked               url_context      1  wayback -> url_context [scottaaronson.blog]
  success               wayback     blocked               url_context      1  wayback -> url_context [www.bls.gov]
  no_resolving_content  url_context blocked               url_context      1  impersonate -> wayback -> url_context [www.sagaftra.org]

## Scope and interpretation

This is a local replay at ladder commit `4b2a368d58acfeb25ee8c8ee7101e297a70b2f57`.
Every escalation rung was stubbed to record its pure trigger and decline. It proves selection and
order; it does not prove that a rescue would serve bytes, that a body would classify the same way,
or that a live key, robots check, memo, cap, or deadline would admit the rung.

The input is the persisted artifact archive only. The manifest has 247 entries and all 247 have
`latest_source=artifact`; historical comment backfill records were not consumed. The raw view has
99 files and 837 provider records; its `resolution_source` filter yields 96 records and 129 URL
outcomes. The `by_qid` view has 247 files and 338 records; 121 records carry a `gap_fill_v2`
transcript, yielding 1,221 tool outcomes. The sync reported 57 expired old-log artifacts; those
are absent from this replay and are not silently imputed.

| Policy / caller | Old rows | New rows | Status changes | Route changes | Derived consulted | Lost | Undecidable |
|---|---:|---:|---:|---:|---:|---:|---:|
| `GAP_FILL_DIRECT_POLICY` | 964 | 964 | 586 | 839 | 0 | 2 | 2 |
| `GAP_FILL_DOCUMENT_POLICY` | 964 | 964 | 586 | 876 | 0 | 2 | 2 |
| `GAP_FILL_FETCH_POLICY` | 964 | 964 | 586 | 888 | 23 | 2 | 1 |
| `RESOLUTION_SOURCE_POLICY` | 964 | 964 | 586 | 887 | 23 | 0 | 1 |

The 586 status changes are expected archive-vocabulary and direct-status reconstruction: 542
`ok -> success`, 29 `error -> not_found` from disclosed HTTP 404s, 11 `empty -> js_wall`, two
rescued `no_resolving_content -> blocked`, and two rescued `success -> blocked`. Every replayed
status equals its reconstructed direct status, with zero status mismatches and zero parse failures.

The intentional archive gaps are 50 cached outcomes, 80 rendered outcomes, 67 document errors,
156 successful document reads, 20 local digests, 10 impersonated rescues, and three local PDF
reads. Their transcripts name the rescue method but omit the direct status, so the tool correctly
counts 386 outcomes without replaying them. No replayable row carries `throttled`; the throttle
status inverse exists, but this corpus supplies no throttle row. The 23 derived rows per applicable
policy are the expected `derived_api -> rendered` selections. After those intentional mappings,
policy differences, and documented rescue gaps, no unexpected replay mismatch remains.
