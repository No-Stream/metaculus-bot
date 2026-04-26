# Future Ideas

Ideas for improving the forecasting bot, roughly ordered by expected impact and feasibility.

## Near-term (worth exploring soon)

### ~~Supervisor agent for high-disagreement questions~~ DONE

Implemented as conditional stacking (`AggregationStrategy.CONDITIONAL_STACKING`).

### ~~Financial data tool access (yFinance, FRED)~~ DONE

Implemented as `financial_data_provider.py`.

### Second-pass web search + scrape pipeline

Our first-pass research (AskNews API dump, Grok native search) is a black box — we can't
control what gets fetched, can't parse PDFs or JS-heavy pages, and can't follow up on gaps.
A second pass with full-control scraping would address this.

**Three use cases for the second pass:**

1. **Gap-filling**: After initial research + forecasting, identify information gaps or
   unanswered questions from the first pass and run targeted searches to fill them.
2. **Resolution source reading**: Many questions include specific resolution source URLs
   in the fine print. Directly scraping the authoritative source (e.g., a government
   dataset, a specific report) gives ground-truth current state instead of secondhand
   summaries.
3. **Reopening inaccessible sources**: The first-pass research often surfaces URLs that
   the bot can't open (PDFs, paywalled content, JS-rendered pages). The forecasting
   prompt should instruct models to flag interesting sources they couldn't access, so
   the second pass can scrape them with more sophisticated tools.

**Tool candidates**: Olostep (cheaper, PAYG) or Firecrawl (pricier but the industry norm).
Both handle PDFs, JS rendering, and give full control over what gets fetched.

**Architecture**: Runs after the initial research phase, before (or as input to) forecasting.
Could also feed into the stacking pass for high-disagreement questions. Moderate effort.

### Separate outside/inside view stages

Currently both phases happen in a single prompt. Making them separate LLM calls means the
inside view genuinely adjusts FROM an explicit base rate rather than constructing both
together. This could help with the arithmetic-override problem (models computing correct
probabilities then ignoring them).

Smingers uses this architecture with cross-pollination: outside view from Model A feeds
into inside view for Model B, introducing diversity.

We could prototype this by having a first pass produce only a base rate + reference class,
then feeding that explicitly to the second pass. Moderate-to-high effort.

### LLM-based forecast self-evaluation

After each forecast, run a cheap model to assess: research relevance, factual accuracy,
reasoning soundness, date/chronology correctness, resolution criteria interpretation.
Flag potential issues before submission.

Smingers found this invaluable for catching date confusion, hallucinated sources, and
reasoning failures. Implementation: easy (structured eval prompt + cheap model call).

## Medium-term (requires more exploration)

### Mixture model parameterization for numeric questions

Instead of asking LLMs for 11 percentiles (which they find unnatural), ask them to
parameterize a mixture of distributions: specify 2-3 components with means, stds, and
weights. This naturally produces smoother, better-shaped CDFs.

Mantic uses this approach and reports good results. The LLM selects components capturing
different scenarios, and the final prediction is a weighted combination.

Would require changes to the numeric prompt, parsing, and CDF construction pipeline.

### Aggregation strategy improvements

Ideas from analysis (lower priority since prompt changes address the bigger issues):

- Trimmed mean (drop highest + lowest, mean of middle): robustness of median with
  better signal preservation. With 6 models, could drop top and bottom, mean of 4.
- Post-aggregation shrinkage toward 50% (~15-20%): corrects the NO-bias found in
  our analysis. Formula: `adjusted = pred * 0.82 + 0.5 * 0.18`.
- Spread-aware aggregation: widen uncertainty when models disagree rather than just
  picking the middle.
- Weighted aggregation by historical model performance (per question type).

Need more data (more resolved questions) to confidently evaluate these.

### Domain-aware CDF spread tuning

Our PIT analysis found non-financial questions are 22% too wide while financial questions
are well-calibrated. The pipeline could use the forecastability classification (now output
by the prompt) to apply different tail-widening parameters.

Could also use the FORECASTABILITY tag to adjust smoothing, tail mass allocation, or
post-hoc CDF scaling.

## Longer-term (significant R&D)

### Agentic deep research (ReAct loop)

Move from one-shot research to an iterative research agent that can: execute search queries,
evaluate results, identify gaps, execute follow-up queries, run code for analysis, and
synthesize findings. Smingers is moving this direction.

Main blocker: API costs. The canned Grok native search works adequately for most questions.
Agentic research would be most valuable for complex questions where a single search
doesn't surface the right information.

Could prototype with selective activation (only for questions where initial research
scores poorly on a relevance check).

### Prediction market integration (read, not anchor)

Direct API access to Polymarket, Kalshi, Metaculus community predictions (where available)
as one research input. Currently markets show up in web search results inconsistently.
Structured access would be more reliable.

Important: should be presented as one noisy input, not an anchor. The prompt already says
"not beholden to them" and our analysis found this works reasonably well.
