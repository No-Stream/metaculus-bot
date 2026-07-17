# Packet A

Replay of the same gap-fill v2 research task (identical system prompt, user
brief, tools, and budgets) by an anonymized driver model.

## Per-run stats

| stat | value |
|---|---|
| steps | 6 |
| tool_calls | 11 |
| per-tool counts | conclude=1, fetch=5, read_document=1, record_findings=1, search_web=3 |
| dup_tool_calls | 0 |
| deadline_hit | False |
| concluded_early | True |
| wall_s (loop) | 68.3 |
| findings | 4 |
| pending_leads | 2 |
| lint_rejections | 0 |
| llm calls | 7 |
| prompt tokens | 145746 |
| completion tokens | 2445 |
| est. cost USD | 0.1604 |


## Findings artifact (what the forecasting panel would receive)

## Agentic Research Findings

### economic perceptions
Claim: (A) The same UCT/CSSR survey reports that 41% said their personal economic conditions had worsened over the preceding 12 months and 38% said they had improved.
Source: https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26
Quote:
> When asked about whether their economic conditions were better or worse than 12 months ago, respondents were split, with 41% reporting that things had gotten worse, while 38% reported that their personal conditions had improved.
Date: March 2026 publication; survey June-November 2025
Retrieved how: Fetched the University of Cape Town/CSSR page directly.

### historical election results
Claim: (A) The Electoral Commission of Zambia's page reporting the 2016 presidential results lists Hakainde Hichilema with 1,760,347 votes, and states that Edgar Lungu secured over fifty per cent of total valid votes cast.
Source: https://www.elections.org.zm/?p=2881
Quote:
> Hakainde Hichilema UPND: 1,760,347 ... President Edgar Lungu was re-elected into the office of President of Zambia after securing over fifty-per cent of total valid votes cast with 1,860,877 votes.
Date: August 15, 2016 result announcement (page retrieved July 17, 2026)
Retrieved how: Fetched the ECZ page directly.

### polling and timing
Claim: (A) The University of Cape Town/CSSR briefing paper says the Zambia Elections Research Network survey ran from June to November 2025 and reported 55% planning to vote for the president, 13% for an opposition party, and 32% undecided or refusing to answer; this fieldwork predates the question opening date of June 30, 2026.
Source: https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26
Quote:
> It ran from June to November 2025. ... When asked how they planned to vote in the upcoming elections, 55% of respondents reported planning to vote for the president, 13% intended for an opposition party and – notably – 32% didn’t know or refused to answer the question.
Date: March 2026 publication; fieldwork June-November 2025
Retrieved how: Fetched the University of Cape Town/CSSR page directly.

### resolution and timing
Claim: (A) The Electoral Commission of Zambia timetable lists the first ballot for Thursday, August 13, 2026, declaration of election results on Monday, August 17, 2026, and publication of election results from August 17 through August 24, 2026.
Source: https://www.elections.org.zm/wp-content/uploads/2026-Elections-Timetable-14-August-2025-30-days.pdf
Quote:
> Poll Date - First Ballot ... Thursday, August 13, 2026 ... Declaration of Election Results ... Monday, August 17, 2026 ... Publication of Election Results ... Monday, August 17, 2026 to Monday, August 24, 2026
Date: ECZ timetable fetched July 17, 2026; schedule milestones dated August 13-24, 2026
Retrieved how: Fetched the ECZ PDF directly; document extraction returned the timetable text.

Pending leads:
- Direct verification of the ECZ 2021 final presidential results remains unresolved: the supplied briefing cites 2,852,348 votes/59.02%, while a targeted document extraction from the AU report returned a conflicting unquoted figure (2,810,757/59.38%); do not treat the latter as verified.
- No direct first-round vote-share prediction market or transparent post-May-2026 probability poll was found. The available Polymarket market concerns winner, including a possible second round, and is not a term match.

## Ghost forecast (driver's own private forecast after research freeze)

qtype=numeric, parsed=median=56.0

Ghost structured block:
```json
{
  "question_type": "numeric",
  "declared_percentiles": {
    "0.01": 38.0,
    "0.025": 41.0,
    "0.05": 44.0,
    "0.10": 47.0,
    "0.20": 50.0,
    "0.40": 54.0,
    "0.50": 56.0,
    "0.60": 58.0,
    "0.80": 62.0,
    "0.90": 65.0,
    "0.95": 67.0,
    "0.975": 69.0,
    "0.99": 71.0
  },
  "outcome_type": "continuous"
}
```

## Full tool-call trace

### Step 1 (assistant)
- TOOL CALL: fetch({"url": "https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: plain; [budget: 521s remaining, 4/14 tool calls used]
  RESULT PREVIEW: links: / - https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26#main-content / - https://uct.ac.za / - https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26 / - https://uct.ac.za/ / - https://uct.ac.za/students/prospective-students-welcome/welcome-uct / - https://uct.ac.za/applicant
- TOOL CALL: fetch({"url": "https://www.elections.org.zm/wp-content/uploads/2026-Elections-Timetable-14-August-2025-30-days.pdf", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: document; [budget: 521s remaining, 4/14 tool calls used]
  RESULT PREVIEW: The provided document **does not address** what percentage of votes Hakainde Hichilema will receive in the first round of Zambia's 2026 presidential elections.  / The document is instead a planning schedule published by the Electoral Commission of Zambia titled **"REVISED 2026 GENERAL ELECTION - ROA
- TOOL CALL: fetch({"url": "https://au.int/sites/default/files/newsevents/reports/40695-rp-FINAL_REPORT_OF_THE_AUEOM_TO_ZAMBIA_2021_GENERAL_ELECTIONS_-_5_NOV.-1.pdf", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: document; [budget: 521s remaining, 4/14 tool calls used]
  RESULT PREVIEW: The provided document does not address your question [1.1].  / Based on the URL and document title, the source is the **"Final Report of the African Union Election Observation Mission (AUEOM) to the Zambia 2021 General Elections"** [1.1]. The document's stated dates include the election date of **12
- TOOL CALL: fetch({"url": "https://www.elections.org.zm/?p=2881", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: plain; [budget: 521s remaining, 4/14 tool calls used]
  RESULT PREVIEW: links: / - https://www.elections.org.zm/about-ecz/contact-electoral-commission-of-zambia/ / - https://www.elections.org.zm/ / - https://web.facebook.com/electoralcommissionofzambia / - https://twitter.com/ZambiaElections / - https://www.linkedin.com/company/electoral-commission-of-zambia / - https:/

### Step 2 (assistant)
- TOOL CALL: search_web({"query": "site:elections.org.zm 2021 presidential election Hichilema 2,852,348 59.02", "end_published_date": null})
  RESULT: tool: search_web; status: error; method: search; [budget: 518s remaining, 7/14 tool calls used]
  RESULT PREVIEW: Exa search failed: ValueError: Request failed with status code 401: {"requestId":"96f14cde7afe3470f566da32bf5bf461","error":"Invalid API key","tag":"INVALID_API_KEY"} / [budget: 518s remaining, 7/14 tool calls used]
- TOOL CALL: search_web({"query": "Zambia 2026 presidential election official candidate list 14 ECZ May 2026", "end_published_date": "2026-07-17"})
  RESULT: tool: search_web; status: error; method: search; [budget: 518s remaining, 7/14 tool calls used]
  RESULT PREVIEW: Exa search failed: ValueError: Request failed with status code 401: {"requestId":"e75a973a13d955ff565eb745ed1e6b51","error":"Invalid API key","tag":"INVALID_API_KEY"} / [budget: 518s remaining, 7/14 tool calls used]
- TOOL CALL: search_web({"query": "Zambia 2026 election Hichilema poll July 2026 vote intention", "end_published_date": "2026-07-17"})
  RESULT: tool: search_web; status: error; method: search; [budget: 518s remaining, 7/14 tool calls used]
  RESULT PREVIEW: Exa search failed: ValueError: Request failed with status code 401: {"requestId":"3adefc9640be63d5dbf71793f4a02fd2","error":"Invalid API key","tag":"INVALID_API_KEY"} / [budget: 518s remaining, 7/14 tool calls used]

### Step 3 (assistant)
- TOOL CALL: record_findings (4 finding(s) banked)
  RESULT: tool: record_findings; status: ok; method: internal; [budget: 511s remaining, 8/14 tool calls used]
  RESULT PREVIEW: Recorded 4 finding(s). / [budget: 511s remaining, 8/14 tool calls used]

### Step 4 (assistant)
- TOOL CALL: read_document({"url": "https://au.int/sites/default/files/newsevents/reports/40695-rp-FINAL_REPORT_OF_THE_AUEOM_TO_ZAMBIA_2021_GENERAL_ELECTIONS_-_5_NOV.-1.pdf", "ask": "Find the official/reported final presidential election results for Zambia's 2021 election. State Hakainde Hichilema's vote count and percentage, and quote the exact passage containing both."})
  RESULT: tool: read_document; status: ok; method: document; [budget: 486s remaining, 9/14 tool calls used]
  RESULT PREVIEW: Due to technical limitations, the provided PDF URL is currently inaccessible via the automated browsing tool (returning a connection/block error) [1.1, 3.1]. Consequently, it is not possible to extract or verify verbatim quotes directly from this specific document.  / However, based on the official 

### Step 5 (assistant)
- TOOL CALL: fetch({"url": "https://www.elections.org.zm/wp-content/uploads/2021-VOTER-REGISTRATION-STATISTICS-SUMMARY.pdf", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: document; [budget: 476s remaining, 10/14 tool calls used]
  RESULT PREVIEW: This document does not address the question of what percentage of votes Hakainde Hichilema will receive in the first round of Zambia's 2026 presidential elections.  / Instead, the document is titled **"2021 VOTER REGISTRATION STATISTICS"**. It contains voter registration data and comparisons with th

### Step 6 (assistant)
- TOOL CALL: conclude (final_findings=0, pending_leads=2)
  RESULT: tool: conclude; status: ok; method: internal; [budget: 471s remaining, 11/14 tool calls used]
  RESULT PREVIEW: Concluded with 0 final finding(s) and 2 pending lead(s). / [budget: 471s remaining, 11/14 tool calls used]

### (GHOST PROMPT injected)

### Step 7 (assistant)
Assistant text: ```json
{
  "question_type": "numeric",
  "declared_percentiles": {
    "0.01": 38.0,
    "0.025": 41.0,
    "0.05": 44.0,
    "0.10": 47.0,
    "0.20": 50.0,
    "0.40": 54.0,
    "0.50": 56.0,
    "0.60": 58.0,
    "0.80": 62.0,
    "0.90": 65.0,
    "0.95": 67.0,
    "0.975": 69.0,
    "0.99": 71.0
  },
  "outcome_type": "continuous"
}
```

