# Packet D

Replay of the same gap-fill v2 research task (identical system prompt, user
brief, tools, and budgets) by an anonymized driver model.

## Per-run stats

| stat | value |
|---|---|
| steps | 5 |
| tool_calls | 11 |
| per-tool counts | conclude=1, fetch=6, record_findings=2, search_web=2 |
| dup_tool_calls | 0 |
| deadline_hit | False |
| concluded_early | True |
| wall_s (loop) | 30.3 |
| findings | 7 |
| pending_leads | 3 |
| lint_rejections | 0 |
| llm calls | 6 |
| prompt tokens | 131495 |
| completion tokens | 2261 |
| est. cost USD | 0.3627 |


## Findings artifact (what the forecasting panel would receive)

## Agentic Research Findings

### ⚠ Corrections to the briefing
The verified findings below contradict the research briefing and supersede the corresponding briefing content.

Claim: Discrepancy: the briefing’s Polymarket snapshot says approximately 85% and about $72,400 volume as of July 17, 2026; the fetched market page displayed 86% and $74,586 volume.
Source: https://polymarket.com/pl/event/zambia-presidential-election-winner-20260604205247322/will-hakainde-hichilema-win-the-2026-zambian-presidential-elections-20260604205342549
Quote:
> “Hakainde Hichilema 86%” and “$74,586 Wol.”
Date: 2026-07-17
Retrieved how: Fetched Polymarket market page

### economic indicators
Claim: (B) Reuters reported annual inflation of 6.5% in June 2026 and an IMF projection of 4.3% economic growth for 2026, following 3.8% in 2025.
Source: https://www.marketscreener.com/news/zambia-election-will-put-hichilemas-economic-record-to-the-test-ce7f5ed9d18cf723
Quote:
> “Annual inflation slowed to 6.5% in June, its lowest level in more than eight years”; “The International Monetary Fund projects Zambia's economy to grow 4.3% this year from 3.8% last year.”
Date: 2026-07-09
Retrieved how: Fetched Reuters article republished by MarketScreener

### historical result
Claim: (A) ECZ's 2016 presidential-results page reports 1,760,347 votes for Hakainde Hichilema; the page lists 1,860,877 votes for Edgar Lungu and the other candidates' totals.
Source: https://www.elections.org.zm/?p=2881
Quote:
> “The full results are as follows; - Edgar Chagwa C. Lungu PF: 1,860,877 - Hakainde Hichilema UPND: 1,760,347”
Date: 2016-08-15
Retrieved how: Fetched ECZ results page

### polling
Claim: (B) The ZEREN survey ran from June to November 2025, before the question opened on June 30, 2026; 55% said they planned to vote for the president, 13% for an opposition party, and 32% did not know or refused.
Source: https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26
Quote:
> “It ran from June to November 2025.” “When asked how they planned to vote in the upcoming elections, 55% of respondents reported planning to vote for the president, 13% intended for an opposition party and – notably – 32% didn’t know or refused to answer the question.”
Date: 2026-03
Retrieved how: Fetched University of Cape Town briefing-paper page

Claim: (B) Reuters reported on July 9, 2026 that the ZEREN survey finding that 55% planned to vote for Hichilema was collected before Brian Mundubile launched his campaign.
Source: https://www.marketscreener.com/news/zambia-election-will-put-hichilemas-economic-record-to-the-test-ce7f5ed9d18cf723
Quote:
> “A survey by the Zambia Election Research Network late last year found 51% of respondents expected a free and fair election and 55% planned to vote for Hichilema, although it was conducted before Mundubile launched his campaign.”
Date: 2026-07-09
Retrieved how: Fetched Reuters article republished by MarketScreener

### prediction market terms
Claim: (C) As fetched on July 17, 2026, Polymarket displayed Hichilema at 86%, Mundubile at 10%, and $74,586 volume; its rules concern the listed candidate who wins the next presidential election, include a potential second round, and use a March 31, 2027 fallback date.
Source: https://polymarket.com/pl/event/zambia-presidential-election-winner-20260604205247322/will-hakainde-hichilema-win-the-2026-zambian-presidential-elections-20260604205342549
Quote:
> “Hakainde Hichilema 86%”; “Brian Mundubile 10%”; “$74,586 Wol.” “This market will resolve according to the listed candidate who wins the next Zambian presidential election.” “This market includes any potential second round.” “If the results are not known definitively by March 31, 2027, 11:59 PM ET, this market will resolve to ‘Other’.”
Date: 2026-07-17
Retrieved how: Fetched Polymarket market page

### resolution timing
Claim: (A) The ECZ revised 2026 roadmap sets the first-ballot poll date as Thursday, August 13, 2026; it lists declaration of election results on Monday, August 17, 2026, and publication from August 17 to August 24, 2026.
Source: https://www.elections.org.zm/wp-content/uploads/2026-Elections-Timetable-14-August-2025-30-days.pdf
Quote:
> “POLL DATE - FIRST BALLOT: Thursday, August 13, 2026”; “Declaration of Election Results: Monday, August 17, 2026”; “Publication of Election Results: Monday, August 17, 2026 to Monday, August 24, 2026.”
Date: 2025-08-14
Retrieved how: Fetched ECZ roadmap PDF

Pending leads:
- An ECZ page or gazette confirming the final 2026 presidential candidate list and an ECZ publication of the final voter register total were not verified from primary material; Exa search was unavailable due to an API authentication error.
- No post-June 30, 2026 nationwide probability poll with disclosed methodology was located in the supplied sources or verified searches.
- The 2021 AU observation-report PDF did not yield extractable result tables through the fetch tool; the briefing’s 59.02% historical figure remains a lead for direct table verification.

## Ghost forecast (driver's own private forecast after research freeze)

qtype=numeric, parsed=median=56.0

Ghost structured block:
```json
{
  "question_type": "numeric",
  "declared_percentiles": {
    "0.01": 42.0,
    "0.025": 44.0,
    "0.05": 46.0,
    "0.1": 48.5,
    "0.2": 51.0,
    "0.4": 54.0,
    "0.5": 56.0,
    "0.6": 57.5,
    "0.8": 61.0,
    "0.9": 64.0,
    "0.95": 66.5,
    "0.975": 69.0,
    "0.99": 72.0
  },
  "outcome_type": "continuous"
}
```

## Full tool-call trace

### Step 1 (assistant)
- TOOL CALL: fetch({"url": "https://www.elections.org.zm/wp-content/uploads/2026-Elections-Timetable-14-August-2025-30-days.pdf", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: document; [budget: 527s remaining, 4/14 tool calls used]
  RESULT PREVIEW: The provided document does not address or contain any information regarding the percentage of votes Hakainde Hichilema (or any other candidate) will receive in the 2026 presidential elections.  / Instead, the document is the **"ELECTORAL COMMISSION OF ZAMBIA REVISED 2026 GENERAL ELECTION - ROADMAP"*
- TOOL CALL: fetch({"url": "https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: plain; [budget: 527s remaining, 4/14 tool calls used]
  RESULT PREVIEW: links: / - https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26#main-content / - https://uct.ac.za / - https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26 / - https://uct.ac.za/ / - https://uct.ac.za/students/prospective-students-welcome/welcome-uct / - https://uct.ac.za/applicant
- TOOL CALL: fetch({"url": "https://au.int/sites/default/files/newsevents/reports/40695-rp-FINAL_REPORT_OF_THE_AUEOM_TO_ZAMBIA_2021_GENERAL_ELECTIONS_-_5_NOV.-1.pdf", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: document; [budget: 527s remaining, 4/14 tool calls used]
  RESULT PREVIEW: The provided document does not address this question.  / As indicated by the URL, the document is the **"Final Report of the AUEOM [African Union Election Observation Mission] to Zambia 2021 General Elections,"** which is dated **5 November 2021** (covering the elections held on **12 August 2021**).
- TOOL CALL: fetch({"url": "https://www.marketscreener.com/news/zambia-election-will-put-hichilemas-economic-record-to-the-test-ce7f5ed9d18cf723", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: plain; [budget: 527s remaining, 4/14 tool calls used]
  RESULT PREVIEW: links: / - https://www.marketscreener.com/ / - https://www.marketscreener.com/login/forgot-password / - https://www.marketscreener.com/services/solutions/?utmzb_campaign=abo++menu-top_abo+s-abonner+&utmzb_content=++++&utmzb_source=menu-top&utmzb_medium=url_declenchement / - https://www.marketscreene

### Step 2 (assistant)
- TOOL CALL: record_findings (4 finding(s) banked)
  RESULT: tool: record_findings; status: ok; method: internal; [budget: 521s remaining, 5/14 tool calls used]
  RESULT PREVIEW: Recorded 4 finding(s). / [budget: 521s remaining, 5/14 tool calls used]

### Step 3 (assistant)
- TOOL CALL: fetch({"url": "https://www.elections.org.zm/?p=2881", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: plain; [budget: 516s remaining, 9/14 tool calls used]
  RESULT PREVIEW: links: / - https://www.elections.org.zm/about-ecz/contact-electoral-commission-of-zambia/ / - https://www.elections.org.zm/ / - https://web.facebook.com/electoralcommissionofzambia / - https://twitter.com/ZambiaElections / - https://www.linkedin.com/company/electoral-commission-of-zambia / - https:/
- TOOL CALL: fetch({"url": "https://polymarket.com/pl/event/zambia-presidential-election-winner-20260604205247322/will-hakainde-hichilema-win-the-2026-zambian-presidential-elections-20260604205342549", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: plain; [budget: 516s remaining, 9/14 tool calls used]
  RESULT PREVIEW: links: / - https://polymarket.com/pl/event/zambia-presidential-election-winner-20260604205247322/will-hakainde-hichilema-win-the-2026-zambian-presidential-elections-20260604205342549#__pm_main / - https://polymarket.com/pl / - https://polymarket.com/pl/sports/world-cup / - https://polymarket.com/pl/
- TOOL CALL: search_web({"query": "site:elections.org.zm 2026 presidential nominations Hichilema candidates July 2026", "end_published_date": "2026-07-17"})
  RESULT: tool: search_web; status: error; method: search; [budget: 516s remaining, 9/14 tool calls used]
  RESULT PREVIEW: Exa search failed: ValueError: Request failed with status code 401: {"requestId":"afedc2deb9f688165dc34e970e9577d5","error":"Invalid API key","tag":"INVALID_API_KEY"} / [budget: 516s remaining, 9/14 tool calls used]
- TOOL CALL: search_web({"query": "site:elections.org.zm 2026 final register 8,786,300 voters", "end_published_date": "2026-07-17"})
  RESULT: tool: search_web; status: error; method: search; [budget: 516s remaining, 9/14 tool calls used]
  RESULT PREVIEW: Exa search failed: ValueError: Request failed with status code 401: {"requestId":"1a8d0016a8c76242385c5d14506d5d81","error":"Invalid API key","tag":"INVALID_API_KEY"} / [budget: 516s remaining, 9/14 tool calls used]

### Step 4 (assistant)
- TOOL CALL: record_findings (3 finding(s) banked)
  RESULT: tool: record_findings; status: ok; method: internal; [budget: 512s remaining, 10/14 tool calls used]
  RESULT PREVIEW: Recorded 3 finding(s). / [budget: 512s remaining, 10/14 tool calls used]

### Step 5 (assistant)
- TOOL CALL: conclude (final_findings=0, pending_leads=3)
  RESULT: tool: conclude; status: ok; method: internal; [budget: 509s remaining, 11/14 tool calls used]
  RESULT PREVIEW: Concluded with 0 final finding(s) and 3 pending lead(s). / [budget: 509s remaining, 11/14 tool calls used]

### (GHOST PROMPT injected)

### Step 6 (assistant)
Assistant text: ```json
{
  "question_type": "numeric",
  "declared_percentiles": {
    "0.01": 42.0,
    "0.025": 44.0,
    "0.05": 46.0,
    "0.1": 48.5,
    "0.2": 51.0,
    "0.4": 54.0,
    "0.5": 56.0,
    "0.6": 57.5,
    "0.8": 61.0,
    "0.9": 64.0,
    "0.95": 66.5,
    "0.975": 69.0,
    "0.99": 72.0
  },
  "outcome_type": "continuous"
}
```

