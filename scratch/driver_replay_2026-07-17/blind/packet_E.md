# Packet E

Replay of the same gap-fill v2 research task (identical system prompt, user
brief, tools, and budgets) by an anonymized driver model.

## Per-run stats

| stat | value |
|---|---|
| steps | 5 |
| tool_calls | 13 |
| per-tool counts | conclude=1, fetch=7, record_findings=2, search_news=2, search_web=1 |
| dup_tool_calls | 0 |
| deadline_hit | False |
| concluded_early | True |
| wall_s (loop) | 41.7 |
| findings | 6 |
| pending_leads | 3 |
| lint_rejections | 0 |
| llm calls | 6 |
| prompt tokens | 146460 |
| completion tokens | 2522 |
| est. cost USD | 0.404 |


## Findings artifact (what the forecasting panel would receive)

## Agentic Research Findings

### candidate field
Claim: (C) Lusaka Times reported on May 22, 2026 that 14 candidates successfully filed presidential nominations, and named Hakainde Hichilema among them. This is pre-window context (before June 30, 2026), and is a local-reporting account rather than an ECZ publication.
Source: https://www.lusakatimes.com/2026/05/22/ecz-closes-presidential-nominations-with-14-candidates-on-ballot/
Quote:
> “The Electoral Commission of Zambia (ECZ) has closed presidential nominations for the 2026 General Election with 14 candidates successfully filing in ahead of the August 13 polls.” “Among those who successfully filed in is incumbent President Hakainde Hichilema of the United Party for National Development (UPND)”.
Date: 2026-05-22
Retrieved how: Fetched Lusaka Times report.

### historical reference class
Claim: (A) ECZ’s 2016 results page lists Hakainde Hichilema with 1,760,347 votes and Edgar Lungu with 1,860,877 votes; the page states Lungu secured over fifty percent of total valid votes cast.
Source: https://www.elections.org.zm/?p=2881
Quote:
> “President Edgar Lungu was re-elected into the office of President of Zambia after securing over fifty-per cent of total valid votes cast with 1,860,877 votes.” “Hakainde Hichilema UPND: 1,760,347”.
Date: 2016-08-15
Retrieved how: Fetched ECZ 2016 presidential-results news page.

### polling
Claim: (B) The University of Cape Town/CSSR briefing paper says the Zambia Elections Research Network survey ran June–November 2025 and records 55% planning to vote for the president, 13% for an opposition party, and 32% who did not know or refused. The survey period pre-dates the June 30, 2026 question open date.
Source: https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26
Quote:
> “It ran from June to November 2025.” “When asked how they planned to vote in the upcoming elections, 55% of respondents reported planning to vote for the president, 13% intended for an opposition party and – notably – 32% didn’t know or refused to answer the question.”
Date: 2026-03
Retrieved how: Fetched University of Cape Town/CSSR publication page.

### prediction market terms
Claim: (C) Polymarket’s Zambia winner market displayed Hakainde Hichilema at 86%, Brian Mundubile at 10%, and $74,586 volume when fetched. Its rules concern the listed candidate who wins the next presidential election, include a possible second round, and use a March 31, 2027 results deadline; those terms differ from this question’s first-round vote-share metric and September 1, 2026 availability condition.
Source: https://polymarket.com/pl/event/zambia-presidential-election-winner-20260604205247322/will-hakainde-hichilema-win-the-2026-zambian-presidential-elections-20260604205342549
Quote:
> “Hakainde Hichilema 86%”; “Brian Mundubile 10%”; “$74,586 Wol.” “This market will resolve according to the listed candidate who wins the next Zambian presidential election. This market includes any potential second round. If the results are not known definitively by March 31, 2027, 11:59 PM ET, this market will resolve to ‘Other’.”
Date: 2026-07-17
Retrieved how: Fetched Polymarket market page.

### resolution state
Claim: (A) The ECZ homepage fetched on July 17, 2026 displayed administrative notices dated May 8 and May 5, 2026; the retrieved homepage content did not display a 2026 presidential first-round result.
Source: https://www.elections.org.zm/
Quote:
> “SCHEDULE OF PROVINCIAL PRE-PROCESSING OF PRESIDENTIAL SUPPORTERS IN THE PROVINCIAL CENTRES FROM MONDAY 11TH TO FRIDAY 15TH MAY, 2026. May 8, 2026”; “Pursuant to Section 71 of the Public Procurement Act, No. 8 of 2020, the Commission has concluded the evaluation and Procurement Committee approval… after Five (5) working days from the date of May 5, 2026”.
Date: 2026-07-17
Retrieved how: Fetched ECZ homepage.

### resolution timing
Claim: (A) The ECZ revised 2026 roadmap schedules the first-ballot poll for Thursday, August 13, 2026; receipt and verification of results for August 15–17; declaration on August 17; and publication for August 17–24. These are scheduled dates, not official results. (Inside question window: the poll and result milestones are after the June 30, 2026 open date.)
Source: https://www.elections.org.zm/wp-content/uploads/2026-Elections-Timetable-14-August-2025-30-days.pdf
Quote:
> “POLL DATE - FIRST BALLOT: Thursday, August 13, 2026”; “Receipt and Verification of Results: Saturday, August 15, 2026 to Monday, August 17, 2026”; “Declaration of Election Results: Monday, August 17, 2026”; “Publication of Election Results: Monday, August 17, 2026 to Monday, August 24, 2026”.
Date: 2025-08-14
Retrieved how: Fetched ECZ revised 2026 general-election roadmap PDF.

Pending leads:
- An ECZ primary publication confirming the final presidential candidate list was not located in the fetched ECZ homepage material; the 14-candidate figure was verified only through a local report.
- The fetched African Union 2021 report was not text-extracted to a page/quotation containing the 59.02% result; pursue a directly searchable official 2021 ECZ results document if that historical figure is material.
- No transparently documented nationwide vote-intention survey conducted after the June 30, 2026 question opening date was identified in the research performed.

## Ghost forecast (driver's own private forecast after research freeze)

qtype=numeric, parsed=median=57.0

Ghost structured block:
```json
{
  "question_type": "numeric",
  "declared_percentiles": {
    "0.01": 35.0,
    "0.025": 39.0,
    "0.05": 42.0,
    "0.1": 45.0,
    "0.2": 49.0,
    "0.4": 54.0,
    "0.5": 57.0,
    "0.6": 59.0,
    "0.8": 64.0,
    "0.9": 68.0,
    "0.95": 71.0,
    "0.975": 74.0,
    "0.99": 78.0
  },
  "outcome_type": "continuous"
}
```

## Full tool-call trace

### Step 1 (assistant)
- TOOL CALL: fetch({"url": "https://www.elections.org.zm/wp-content/uploads/2026-Elections-Timetable-14-August-2025-30-days.pdf", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: document; [budget: 529s remaining, 5/14 tool calls used]
  RESULT PREVIEW: The provided document **does not address** what percentage of votes Hakainde Hichilema will receive in the first round of Zambia's 2026 presidential elections.  / The document is an administrative timeline titled the **"ELECTORAL COMMISSION OF ZAMBIA REVISED 2026 GENERAL ELECTION - ROADMAP"**. It ou
- TOOL CALL: fetch({"url": "https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: plain; [budget: 529s remaining, 5/14 tool calls used]
  RESULT PREVIEW: links: / - https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26#main-content / - https://uct.ac.za / - https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26 / - https://uct.ac.za/ / - https://uct.ac.za/students/prospective-students-welcome/welcome-uct / - https://uct.ac.za/applicant
- TOOL CALL: fetch({"url": "https://www.elections.org.zm/?p=2881", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: plain; [budget: 529s remaining, 5/14 tool calls used]
  RESULT PREVIEW: links: / - https://www.elections.org.zm/about-ecz/contact-electoral-commission-of-zambia/ / - https://www.elections.org.zm/ / - https://web.facebook.com/electoralcommissionofzambia / - https://twitter.com/ZambiaElections / - https://www.linkedin.com/company/electoral-commission-of-zambia / - https:/
- TOOL CALL: fetch({"url": "https://polymarket.com/pl/event/zambia-presidential-election-winner-20260604205247322/will-hakainde-hichilema-win-the-2026-zambian-presidential-elections-20260604205342549", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: plain; [budget: 529s remaining, 5/14 tool calls used]
  RESULT PREVIEW: links: / - https://polymarket.com/pl/event/zambia-presidential-election-winner-20260604205247322/will-hakainde-hichilema-win-the-2026-zambian-presidential-elections-20260604205342549#__pm_main / - https://polymarket.com/pl / - https://polymarket.com/pl/sports/world-cup / - https://polymarket.com/pl/
- TOOL CALL: search_web({"query": "site:elections.org.zm 2026 presidential nominations 14 candidates Hakainde Hichilema May 2026", "end_published_date": "2026-07-17"})
  RESULT: tool: search_web; status: error; method: search; [budget: 529s remaining, 5/14 tool calls used]
  RESULT PREVIEW: Exa search failed: ValueError: Request failed with status code 401: {"requestId":"768973c3d18b678a510af3ecc1744683","error":"Invalid API key","tag":"INVALID_API_KEY"} / [budget: 529s remaining, 5/14 tool calls used]

### Step 2 (assistant)
- TOOL CALL: record_findings (4 finding(s) banked)
  RESULT: tool: record_findings; status: ok; method: internal; [budget: 521s remaining, 6/14 tool calls used]
  RESULT PREVIEW: Recorded 4 finding(s). / [budget: 521s remaining, 6/14 tool calls used]

### Step 3 (assistant)
- TOOL CALL: fetch({"url": "https://www.lusakatimes.com/2026/05/22/ecz-closes-presidential-nominations-with-14-candidates-on-ballot/", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: plain; [budget: 507s remaining, 11/14 tool calls used]
  RESULT PREVIEW: links: / - https://www.lusakatimes.com/ / - https://www.lusakatimes.com/zambiancolumn/ / - https://www.lusakatimes.com/economy/ / - https://www.lusakatimes.com/other-news/ / - https://www.lusakatimes.com/headlines/ / - https://www.lusakatimes.com/health/ / - https://www.lusakatimes.com/photo-gallery
- TOOL CALL: fetch({"url": "https://www.elections.org.zm/", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: plain; [budget: 507s remaining, 11/14 tool calls used]
  RESULT PREVIEW: links: / - https://www.elections.org.zm/about-ecz/contact-electoral-commission-of-zambia/ / - https://www.elections.org.zm/ / - https://web.facebook.com/electoralcommissionofzambia / - https://twitter.com/ZambiaElections / - https://www.linkedin.com/company/electoral-commission-of-zambia / - https:/
- TOOL CALL: fetch({"url": "https://au.int/sites/default/files/newsevents/reports/40695-rp-FINAL_REPORT_OF_THE_AUEOM_TO_ZAMBIA_2021_GENERAL_ELECTIONS_-_5_NOV.-1.pdf", "start_char": 0})
  RESULT: tool: fetch; status: ok; method: document; [budget: 507s remaining, 11/14 tool calls used]
  RESULT PREVIEW: Based on the provided URL, the document is the **"Final Report of the African Union Election Observation Mission (AUEOM) to the 12 August 2021 General Elections in the Republic of Zambia"** (issued in November 2021) [1.1].  / Because this document is a retrospective report detailing the events, obse
- TOOL CALL: search_news({"query": "Zambia ECZ 14 presidential candidates nominations May 2026 Hichilema"})
  RESULT: tool: search_news; status: ok; method: news; [budget: 507s remaining, 11/14 tool calls used]
  RESULT PREVIEW: ### Constitutional Court Dismisses Challenge to Zambia's President Hichilema 2026 Candidacy / Date: 2026-06-20T05:30:43+00:00 / Source: Zambia: News Diggers! / URL: https://diggers.news/courts/2026/06/20/concourt-dismisses-hhs-eligibility-case/ / Summary: The Constitutional Court of Zambia has dismi
- TOOL CALL: search_news({"query": "Zambia election July 2026 Hichilema poll vote intention survey"})
  RESULT: tool: search_news; status: ok; method: news; [budget: 507s remaining, 11/14 tool calls used]
  RESULT PREVIEW: ### Zambia 2026 Election: Hichilema's Economic Record Under Scrutiny Amid Opposition Challenge / Date: 2026-07-11T08:00:44+00:00 / Source: Sunday World / URL: https://sundayworld.co.za/ / Summary: Zambia is preparing for an August 2026 election where incumbent President Hakainde Hichilema is favored

### Step 4 (assistant)
- TOOL CALL: record_findings (2 finding(s) banked)
  RESULT: tool: record_findings; status: ok; method: internal; [budget: 500s remaining, 12/14 tool calls used]
  RESULT PREVIEW: Recorded 2 finding(s). / [budget: 500s remaining, 12/14 tool calls used]

### Step 5 (assistant)
- TOOL CALL: conclude (final_findings=0, pending_leads=3)
  RESULT: tool: conclude; status: ok; method: internal; [budget: 498s remaining, 13/14 tool calls used]
  RESULT PREVIEW: Concluded with 0 final finding(s) and 3 pending lead(s). / [budget: 498s remaining, 13/14 tool calls used]

### (GHOST PROMPT injected)

### Step 6 (assistant)
Assistant text: ```json
{
  "question_type": "numeric",
  "declared_percentiles": {
    "0.01": 35.0,
    "0.025": 39.0,
    "0.05": 42.0,
    "0.1": 45.0,
    "0.2": 49.0,
    "0.4": 54.0,
    "0.5": 57.0,
    "0.6": 59.0,
    "0.8": 64.0,
    "0.9": 68.0,
    "0.95": 71.0,
    "0.975": 74.0,
    "0.99": 78.0
  },
  "outcome_type": "continuous"
}
```

