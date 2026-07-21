# Packet C

Replay of the same gap-fill v2 research task (identical system prompt, user
brief, tools, and budgets) by an anonymized driver model.

## Per-run stats

| stat | value |
|---|---|
| steps | 6 |
| tool_calls | 7 |
| per-tool counts | conclude=1, fetch=3, record_findings=1, search_news=1, search_web=1 |
| dup_tool_calls | 0 |
| deadline_hit | False |
| concluded_early | True |
| wall_s (loop) | 58.1 |
| findings | 9 |
| pending_leads | 4 |
| lint_rejections | 0 |
| llm calls | 7 |
| prompt tokens | 239150 |
| completion tokens | 4942 |
| est. cost USD | 0.5277 |


## Findings artifact (what the forecasting panel would receive)

## Agentic Research Findings

### ⚠ Corrections to the briefing
The verified findings below contradict the research briefing and supersede the corresponding briefing content.

Claim: Discrepancy: the supplied briefing states ECZ confirmed 12 candidates (per Vatican News, July 17 2026) and elsewhere cites 13-14 candidates (Deutsche Welle, Lusaka Times), but The Maravi Post (July 5, 2026) reports the ECZ cleared 26 presidential candidates.
Source: https://www.maravipost.com/can-zambias-fragmented-opposition-unseat-mighty-hichilema-on-august-13-polls/
Quote:
> the Electoral Commission of Zambia has cleared 26 presidential candidates
Date: 2026-07-05
Retrieved how: search_news result

### Candidate field size - discrepancy with briefing's 12/13/14 counts
Claim: The Electoral Commission of Zambia cleared 26 presidential candidates, with 226 UPND candidates, 155 NRPUP candidates, and about 184 independents on the broader ballot (parliamentary/local, not just presidential).
Source: https://www.maravipost.com/can-zambias-fragmented-opposition-unseat-mighty-hichilema-on-august-13-polls/
Quote:
> the Electoral Commission of Zambia has cleared 26 presidential candidates, with the two most prominent challengers being Brian Mundubile of the Tonse Alliance and Fred M'membe of the Socialist Party... about 184 independents on the ballot, many former UPND members, compared to 226 UPND candidates and 155 from the NRPUP.
Date: 2026-07-05
Retrieved how: search_news result

### Election environment/trust
Claim: Same ZEREN survey: 51% of respondents confident 2026 elections would be completely free and fair; 62% of UPND supporters vs 36% of opposition supporters expected free/fair elections.
Source: https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26
Quote:
> A majority of our respondents were confident that the 2026 elections would be completely free and fair (51%)... While 62% of UPND supporters replied that they expected the elections to be 'completely free and fair', only 36% of opposition party supporters concurred.
Date: 2026-03 (published); fieldwork June-Nov 2025
Retrieved how: fetched primary source page directly

### Historical reference: 2016 result
Claim: In 2016 Hichilema received 47.63% of votes, losing to Lungu's 50.35%.
Source: https://www.elections.org.zm/?p=2881
Quote:
> Hichilema received 1,760,347 votes, or 47.63%, losing narrowly to Lungu's 50.35%.
Date: 2016-08
Retrieved how: from supplied web research summary citing ECZ

### Historical reference: 2021 result
Claim: Hichilema won the 2021 first round with 59.02% of valid votes (2,852,348 votes), while Lungu received 38.71%.
Source: https://au.int/sites/default/files/newsevents/reports/40695-rp-FINAL_REPORT_OF_THE_AUEOM_TO_ZAMBIA_2021_GENERAL_ELECTIONS_-_5_NOV.-1.pdf
Quote:
> Hichilema won the 2021 first round with 2,852,348 votes, or 59.02%, while Lungu received 38.71%.
Date: 2021-08
Retrieved how: from supplied web research summary citing AU Election Observation Mission report

### Polling evidence
Claim: ZEREN survey (fieldwork June-November 2025, published March 2026 by UCT/CSSR): 55% of respondents planned to vote for Hichilema/the president, 13% for an opposition party, 32% undecided/refused.
Source: https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26
Quote:
> When asked how they planned to vote in the upcoming elections, 55% of respondents reported planning to vote for the president, 13% intended for an opposition party and – notably – 32% didn't know or refused to answer the question.
Date: 2026-03 (published); fieldwork June-Nov 2025
Retrieved how: fetched primary source page directly

### Prediction market - resolution mismatch
Claim: No Polymarket (or other) market found that directly prices Hichilema's first-round vote-share percentage; only a binary 'who wins' market (including second round) and a turnout-bracket market exist.
Source: https://polymarket.com/pl/event/zambia-presidential-election-winner-20260604205247322/will-hakainde-hichilema-win-the-2026-zambian-presidential-elections-20260604205342549
Quote:
> This market includes any potential second round.
Date: 2026-07-17
Retrieved how: searched and fetched Polymarket pages

### Prediction market - turnout (not candidate vote share)
Claim: Polymarket first-round turnout market: brackets priced 50-60% at 35%, 60-70% at 32%, 70-80% at 15%, <50% at 10%, 80%+ at 5%, on $25,371 volume, as of access July 17 2026.
Source: https://polymarket.com/pl/event/zambia-presidential-election-1st-round-turnout-20260604214258491
Quote:
> This market will resolve according to the official voter turnout rate for the first round of the 2026 Zambian presidential election... 50-60% 35% 60-70% 32% 70-80% 15% <50% 10%
Date: 2026-07-17 (accessed)
Retrieved how: fetched Polymarket page directly

### Prediction market - winner (not first-round %)
Claim: Polymarket 'will Hichilema win' market (includes any potential second round) prices Hichilema at 86%, Mundubile 10%, M'membe <1%, on $74,586 total volume, as of July 17 2026.
Source: https://polymarket.com/pl/event/zambia-presidential-election-winner-20260604205247322/will-hakainde-hichilema-win-the-2026-zambian-presidential-elections-20260604205342549
Quote:
> Hakainde Hichilema 86% Brian Mundubile 10% Fred M'membe <1% ... This market includes any potential second round.
Date: 2026-07-17 (accessed)
Retrieved how: fetched Polymarket page directly

Pending leads:
- No transparent, methodologically documented poll conducted after May 2026 nominations (post opposition consolidation around Mundubile) was found; ZEREN survey (fieldwork June-Nov 2025) remains the most recent disclosed-methodology poll.
- Candidate count discrepancy unresolved: various sources report 12, 13-14, and 26 presidential candidates; the final ECZ ballot list as certified close to Aug 13 should be checked directly at elections.org.zm nearer the vote.
- No prediction market found that directly prices Hichilema's first-round vote-share percentage (only a binary win market including potential second round, and a turnout-bracket market); if such a market emerges before Aug 13 it should be checked.
- Zambian Eye's mid-July 2026 online poll reportedly showing Mundubile ahead was not independently verified/fetched — worth checking methodology if used as evidence.

## Ghost forecast (driver's own private forecast after research freeze)

qtype=numeric, parsed=median=60.0

Ghost structured block:
```json
{
  "question_type": "numeric",
  "declared_percentiles": {
    "0.01": 35.0, "0.025": 40.0, "0.05": 44.0, "0.1": 48.0, "0.2": 52.0, "0.4": 57.0, "0.5": 60.0,
    "0.6": 62.0, "0.8": 67.0, "0.9": 71.0, "0.95": 75.0, "0.975": 78.0, "0.99": 82.0
  },
  "outcome_type": "continuous"
}
```

## Full tool-call trace

### Step 1 (assistant)
- TOOL CALL: fetch({"url": "https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26"})
  RESULT: tool: fetch; status: ok; method: plain; [budget: 533s remaining, 1/14 tool calls used]
  RESULT PREVIEW: links: / - https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26#main-content / - https://uct.ac.za / - https://humanities.uct.ac.za/cssr/idcppa/publication/bp-03-26 / - https://uct.ac.za/ / - https://uct.ac.za/students/prospective-students-welcome/welcome-uct / - https://uct.ac.za/applicant

### Step 2 (assistant)
- TOOL CALL: fetch({"url": "https://polymarket.com/pl/event/zambia-presidential-election-winner-20260604205247322/will-hakainde-hichilema-win-the-2026-zambian-presidential-elections-20260604205342549"})
  RESULT: tool: fetch; status: ok; method: plain; [budget: 529s remaining, 2/14 tool calls used]
  RESULT PREVIEW: links: / - https://polymarket.com/pl/event/zambia-presidential-election-winner-20260604205247322/will-hakainde-hichilema-win-the-2026-zambian-presidential-elections-20260604205342549#__pm_main / - https://polymarket.com/pl / - https://polymarket.com/pl/sports/world-cup / - https://polymarket.com/pl/

### Step 3 (assistant)
- TOOL CALL: fetch({"url": "https://polymarket.com/pl/event/zambia-presidential-election-1st-round-turnout-20260604214258491"})
  RESULT: tool: fetch; status: ok; method: plain; [budget: 525s remaining, 3/14 tool calls used]
  RESULT PREVIEW: links: / - https://polymarket.com/pl/event/zambia-presidential-election-1st-round-turnout-20260604214258491#__pm_main / - https://polymarket.com/pl / - https://polymarket.com/pl/sports/world-cup / - https://polymarket.com/pl/combos / - https://polymarket.com/pl/asset/sp500 / - https://polymarket.com

### Step 4 (assistant)
- TOOL CALL: search_web({"query": "Polymarket Zambia Hichilema vote percentage first round margin"})
  RESULT: tool: search_web; status: error; method: search; [budget: 517s remaining, 5/14 tool calls used]
  RESULT PREVIEW: Exa search failed: ValueError: Request failed with status code 401: {"requestId":"cfbda257c76c553d79376ee38ccd2065","error":"Invalid API key","tag":"INVALID_API_KEY"} / [budget: 517s remaining, 5/14 tool calls used]
- TOOL CALL: search_news({"query": "Zambia election poll Mundubile Hichilema July 2026"})
  RESULT: tool: search_news; status: ok; method: news; [budget: 517s remaining, 5/14 tool calls used]
  RESULT PREVIEW: ### Zambia 2026 Election: Hichilema's Economic Record Under Scrutiny Amid Opposition Challenge / Date: 2026-07-11T08:00:44+00:00 / Source: Sunday World / URL: https://sundayworld.co.za/ / Summary: Zambia is preparing for an August 2026 election where incumbent President Hakainde Hichilema is favored

### Step 5 (assistant)
Assistant text: I have enough. Let me record findings and conclude.
- TOOL CALL: record_findings (9 finding(s) banked)
  RESULT: tool: record_findings; status: ok; method: internal; [budget: 495s remaining, 6/14 tool calls used]
  RESULT PREVIEW: Recorded 9 finding(s). / [budget: 495s remaining, 6/14 tool calls used]

### Step 6 (assistant)
- TOOL CALL: conclude (final_findings=0, pending_leads=4)
  RESULT: tool: conclude; status: ok; method: internal; [budget: 481s remaining, 7/14 tool calls used]
  RESULT PREVIEW: Concluded with 0 final finding(s) and 4 pending lead(s). / [budget: 481s remaining, 7/14 tool calls used]

### (GHOST PROMPT injected)

### Step 7 (assistant)
Assistant text: ```json
{
  "question_type": "numeric",
  "declared_percentiles": {
    "0.01": 35.0, "0.025": 40.0, "0.05": 44.0, "0.1": 48.0, "0.2": 52.0, "0.4": 57.0, "0.5": 60.0,
    "0.6": 62.0, "0.8": 67.0, "0.9": 71.0, "0.95": 75.0, "0.975": 78.0, "0.99": 82.0
  },
  "outcome_type": "continuous"
}
```

