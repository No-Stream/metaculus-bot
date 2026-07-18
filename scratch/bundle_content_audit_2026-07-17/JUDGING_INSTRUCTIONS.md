# Bundle content audit — judging instructions (shared by all judges)

## Why you're doing this

Our Metaculus forecasting bot assembles a research bundle (~11k tokens mean) that all six
forecaster LLMs read for every question. A token audit found the three news-ish sections
(AskNews briefing, Native Search, Gemini grounded) share 40-47% of their quantitative facts,
and the AskNews briefing alone is 44% of the bundle. The operator will NOT cut content on
token counts alone — he wants a careful content-level judgment of what each section actually
contributes before any trimming. Your judgment is the evidence that decision rests on.

## Your input

Each packet (`packets/q<qid>.md`) contains the question title, type, description, resolution
criteria, and fine print, followed by the full archived research bundle split into sections
delimited by `<<<SECTION: name | N chars>>>` ... `<<<END SECTION: name>>>` markers.

Sections you may see: `asknews` (primary news briefing, LLM-summarized), `native_search`
(OpenAI web-search report), `gemini_search` (Google-grounded report), `financial_data`
(FRED/yfinance numbers), `prediction_market` (Polymarket/Kalshi/Manifold/PredictIt odds
table), `resolution_source` (fetched text of the resolution URL), `gap_fill_v1` (second-pass
targeted searches on identified factual gaps), `diagnostics` (provider telemetry). Ignore
`preamble` (it's just the bundle title line).

## The taxonomy

For EACH section in EACH question, partition the content (by rough share of its text) into:

- **(a) uniquely load-bearing** — facts, numbers, dates, or analysis that appear ONLY in this
  section and that a competent forecaster would plausibly use: they anchor the base rate,
  pin the current status, bear directly on the resolution criteria, or supply a decision-
  relevant caveat. Judge against THIS question's resolution criteria, not general interest.
- **(b) corroborative duplication** — content whose facts also appear in ANOTHER section of
  the same bundle. This has real value (independent retrieval paths confirming the same
  fact reduce hallucination risk) but is not unique. Cross-section duplication goes here,
  in BOTH sections that carry it — the synthesis dedups later, don't try to pick a "owner".
- **(c) pure repetition / padding** — content with no marginal value even as corroboration:
  repetition WITHIN the same section, boilerplate ("Credibility: High", generic
  methodology/hedging paragraphs, "the research does not specify..." filler), vague
  padding sentences, formatting overhead, restatements of the question itself, stale
  articles about a different event that merely share keywords with the question.

Percentages are rough (nearest 5-10%) and must sum to ~100 per section. Judge by content
share of the section's text, not by line count precision.

## Required output per question

Write (append) to your assigned file in `judgments/` a block of this exact shape:

```
## Q<qid> — <short title> (<type>, <bundle chars> chars)

| Section | chars | a_unique% | b_corrob% | c_padding% | one-line note |
|---|---:|---:|---:|---:|---|
| asknews | ... | ... | ... | ... | ... |
| ... every section present ... |

**Padding exhibits (category c, verbatim quotes):**
1. [section] "..." — why it's padding
2. [section] "..." — why
3. [section] "..." — why

**Unique-value callouts:** which specific facts were single-source (category a) and which
section carried them — 2-4 bullets. Flag especially anything only ONE section got right or
got at all (e.g. the actual current status, a poll crosstab, a market price).

**Cut-first verdict:** if forced to drop ONE entire section from this question's bundle,
which one and why. What forecast-relevant information would be lost?

**Halving test:** for the two LARGEST sections in this bundle: if each were cut to half its
length by an intelligent prioritizer, what load-bearing content (if any) would plausibly be
lost? Answer per section in 1-2 sentences.
```

## Specific hypotheses to check (from the token audit — confirm or refute, don't assume)

1. The AskNews briefing's tail (historical-articles-derived content, later subsections) is
   largely redundant with its own head (hot-articles bottom line). True in your questions?
2. Native Search and Gemini restate the same headline numbers AskNews already carries,
   differing mainly in phrasing. How much genuinely NEW fact content does each add?
3. gap_fill_v1 re-searches things the bundle already answers, vs. filling real gaps.
4. The prediction_market section is high-value-per-token (prior audit found it's the section
   forecasters cite most). Does its content hold up as decision-relevant here?
5. Boilerplate density: AskNews briefings carry per-claim "Source:/Credibility:" apparatus —
   is that apparatus load-bearing (forecasters need source quality) or padding?

## Ground rules

- FREE task: read local files only. No web access, no API calls, no code execution needed.
- Judge as a forecaster-consumer: "would removing this text change or degrade a careful
  forecast on THIS question's resolution criteria?"
- Quote verbatim for every category-(c) exhibit — the operator wants receipts.
- Don't grade the writing style; grade information content.
- If a section is entirely absent from a packet, skip it (don't invent rows).
- After your last question, add a final `## Judge summary` section: per-section averages
  across your questions, your overall cut-first pick, and any pattern you saw repeatedly
  (with question ids).
