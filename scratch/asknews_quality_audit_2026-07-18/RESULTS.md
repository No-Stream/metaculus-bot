# AskNews pipeline quality audit — 2026-07-18

**Question:** the 2026-07-17 content audit (`scratch/bundle_content_audit_2026-07-17/RESULTS.md`)
found the AskNews section — the biggest bundle section at 44% of tokens — to be 57% padding
AND stale or directionally wrong in 5/10 judged questions (q44219, q44255, q44512, q44551,
q44555) while smaller sections were right. This audit answers: where does that failure live
(fetch vs summarize vs the AskNews index itself), what exactly is the padding, and what
specific prompt/param reforms would fix it.

**Data, all free:** the schema-2 archive records for all 64 prod bundles
(`backtests/research_archive/by_qid/`), the extracted AskNews sections of the five failures
(`sections/` here), deterministic corpus screens (`date_profile.py`,
`padding_decomposition.py` here), the content-audit judgments, the one archived raw
pre-summarization AskNews pull (`scratch/research_role_audit_2026-07-17/inputs/asknews_raw.md`),
and git history of the summarizer prompt/config. No LLM or research API calls.

**One structural caveat up front:** the archive stores only the *post-summarization*
research text (`provider_results[].text` is empty; `chars`/`latency` only). Raw
pre-summarization article sets are not archived for any of the 64 bundles. FETCH-vs-SUMMARIZE
classification therefore rests on the briefing's own internal dates and content — which turns
out to be decisive anyway, because in four of five cases the briefing *visibly contains* the
fresh or correct material it then mishandles.

---

## 0. Config timeline (which pipeline produced each failure)

The summarizer prompt and model changed mid-sample. Mapping run timestamps to first-parent
main state:

| Failure | Run (UTC) | Main state | Summarizer model | Window-stamping / PRE-WINDOW rule | No-forecast rule |
|---|---|---|---|---|---|
| q44219 | Jul 01 | `caca770` | **gpt-5.4-mini** low | absent | absent |
| q44255 | Jul 01 | `caca770` | **gpt-5.4-mini** low | absent | absent |
| q44512 | Jul 10 | `caca770` | **gpt-5.4-mini** low | absent | absent |
| q44551 | Jul 10 | `caca770` | **gpt-5.4-mini** low | absent | absent |
| q44555 | Jul 13 | `f084bf7` | gpt-5.6-sol low | present | present |

(`f0b65ca` added the sol migration + no-forecast rule on Jul 9; `30bca2f` added
window-stamping Jul 8; both reached main in `642b027`, merged Jul 11 09:37. The Jul-10 runs
predate the merge.)

This matters: **4 of the 5 failures ran on the old mini summarizer with no no-forecast rule
and no window-stamping.** The rule works — verdict-language screen over all 64 sections:
6/44 mini-era briefings contain explicit verdict language ("most likely outcome/lab/state",
"best answer is"), **0/20 sol-era** do, and "Bottom line" blocks dropped 44/44 → 10/20.
Three of the five failure *verdicts* (44219 "Most likely lab: OpenAI", 44255 "most likely
outcome is Yes", 44551 "points most strongly to Arizona") are that already-patched surface.
The reforms below target the residual: the padding machine, the missing recency/supersession
logic, and the silent-staleness problem that survive into the sol era (q44555 is the proof —
sol-era, all current rules active, still the worst section in the sample).

Countervailing regression: sol-era sections are **24% longer** (mean 22.4k chars vs 18.0k
mini-era), and the one measured replay (`research_role_audit_2026-07-17`) shows the
"summarizer" *expanding* its input — 17,858 chars of raw articles → 27,111 chars of briefing
(+52%). The Jul-18 removal of "a longer, thorough summary is better" (`5cfb6cd`) should pull
this back some; unvalidated.

---

## 1. FETCH vs SUMMARIZE vs UPSTREAM — the five failures

Method: date-profile every date mention in each briefing against the run timestamp
(`date_profile.py` output below), plus close reading of each extracted section against the
judged failure.

| qid | Failure (from content audit) | Freshest cited dates | On-topic evidence age | Classification |
|---|---|---|---|---|
| 44219 | "OpenAI most likely" off stale April leaderboard data; never sees July Anthropic lead | Jun 30, Jun 29 (1-2d old) | **67d** (Apr 25 leaderboard article) | **FETCH primary, SUMMARIZE aggravator, part-UPSTREAM** |
| 44255 | Concludes "Yes" for before-July-4; auto-enactment lands ~Jul 10 | Jun 30 (1d); median cited age **2d** | fresh | **SUMMARIZE (clean)** |
| 44512 | Bottom line mislabels 25/28 swimming-only golds as team totals | Jul 10 (0d, McKeown withdrawal) | fresh | **SUMMARIZE (clean)** |
| 44551 | Arizona lean; never names Death Valley (the Ogimet favorite) | Jul 10 (0d) | fresh but off-crux | **UPSTREAM primary, SUMMARIZE aggravator** |
| 44555 | 32.7k chars of obsolete Platner chronology; question resolves on replacement nominee | Jul 13 (0d, incl. the withdrawal) | fresh | **SUMMARIZE primary, FETCH secondary** |

Split: **SUMMARIZE implicated in 5/5 (primary in 3), FETCH primary in 1 + secondary in 1,
UPSTREAM primary in 1.** The headline correction to the "staleness" framing: **the raw
article sets were mostly fresh.** The corpus screen agrees (48% of cited dates within 30
days of run, <2% older than 180d). What the judges read as staleness is *salience
inversion* — stale on-topic facts outranking fresh adjacent facts in the briefing's
attention, plus (mini-era) a verdict block anchored on the stale ones.

### Receipts per case

**q44219 — FETCH primary.** The set contains fresh articles (Microsoft Agent Confidence
survey, Jun 29; BeInCrypto Claude Science/GeneBench, Jun 30) but they are topically
adjacent; the only *leaderboard* articles are Apr 25 (the-decoder), Mar 8 (DEV), Mar 17
(infobae) and a Jun 23 Medium piece that names no lab. The hot phase (48h window) delivered
freshness without relevance; the historical phase delivered relevance without freshness.
The summarizer then compounded it — "Key facts repeated because they are central: GPT-5.5:
60 points" (its own flagged repeat of the April datapoint) and a "Practical forecasting
conclusion: OpenAI" verdict (pre-no-forecast-rule). Part-UPSTREAM because "current live
leaderboard standing" is not a *news article* — a news index can only carry it if an outlet
wrote it up within the crawl window; native_search and gap_fill got it by reading the live
mirror, which AskNews structurally cannot.

**q44255 — SUMMARIZE, the cleanest case.** Median cited age 2 days; the briefing itself
carries every input to the correct answer: presented Jun 29-30, "10 days excluding Sundays",
Trump refusing to sign, Speaker Johnson "won't veto". It even notices the gap — "The
research does not provide a direct confirmation of the exact date the 10-day window
expires" — then waves it away ("the research strongly implies enactment before July 4") and
concludes **Yes**. Nobody did the arithmetic Jun 29 + 10 non-Sundays ≈ Jul 10 > Jul 4.
The fetch was excellent. The no-forecast rule now suppresses the *verdict*, but nothing in
the current prompt would make the summarizer do the deadline math whose absence caused it.

**q44512 — SUMMARIZE, internal-contradiction miss.** The briefing *contains its own
correction*: the Perth Now block deep in the section states "179 total medals including
**67 gold** at Birmingham 2022... nearly half of Australia's 2022 gold medals (25) were won
in the pool." The Bottom line nonetheless asserts "25 at Birmingham 2022 and 28 at Gold
Coast 2018" as team totals. The prompt's rule 6 ("Flags any contradictions between
sources") produced flags for a McKeown age discrepancy (24 vs 25) and a name typo
("McKeon") — trivia — while missing the load-bearing 25-vs-67 collision sitting in its own
output. Contradiction-flagging without a materiality/supersession criterion surfaces noise
and misses signal.

**q44551 — UPSTREAM primary.** The question resolves on a station-level Ogimet ranking;
the decisive fact (Death Valley/Furnace Creek is climatologically the US hot station,
Jul 31 normal high 118°F) is not news and will ~never be in a 48h/60d news index. The
articles fetched are what a news index has: regional heat stories, seasonal outlooks, heat-
safety tips. The SUMMARIZE aggravator: the briefing converts *article-count frequency* into
a lean — "Arizona has the most repeated high-end temperature signals in the research" —
which is exactly backwards for a station-level question (repetition of Phoenix coverage is
media-attention data, not temperature data). No summarizer prompt makes a news index carry
climatology; the fix is disclosure (reform R4) so the section stops out-shouting the
sections that do carry the crux.

**q44555 — SUMMARIZE primary (the sol-era failure).** The raw set was fresh through run
day and *included* the Jul 10 Platner withdrawal; the briefing's own opener correctly
notes "older Collins–Platner polling may have limited relevance to the eventual replacement
nominee" — then spends the majority of 32.7k chars (the sample's largest section) on the
obsolete Collins-vs-Platner poll chronology and scandal dossier, tagged with **57
[PRE-WINDOW] + 48 [SINGLE-SOURCE]** repetitions (15.7% of section chars in bracket tags
alone). This is the comprehensiveness mandate working as written: "Extracts ALL facts...",
"Be COMPREHENSIVE — do not omit relevant details", "Omit only information that is clearly
irrelevant" — obsolete polls are not *clearly irrelevant*, so in they go, at full length.
FETCH secondary: the replacement-matchup polls (PPP, Z-to-A — published Jul 11-12, in
native/gemini/gap_fill) didn't make the 6-hot/10-historical budget; withdrawal-drama
coverage outcompeted them in the relevance ranking.

---

## 2. Padding decomposition (what the 57% consists of)

Deterministic screens over all 64 schema-2 AskNews sections (`padding_decomposition.py`),
mechanical categories only; the semantic remainder is taken from the judged exhibits:

| Component | Mean % of section chars | Median | p90 | Notes |
|---|---:|---:|---:|---|
| **Tail restatement** (Key facts / Key Quantitative / Expert Opinions / Contradictions / credibility-notes / takeaways subsections in the last 60%) | **24.1** | 19.0 | 53.2 | The "tail restates head" finding, now measured. Every judged exhibit of self-repeat lives here. |
| **Per-fact apparatus lines** (`Source:`/`Credibility:`/`Date:` label lines) | 5.6 | 5.2 | 10.2 | Worst: q44255 at **31.6%** — 31 Source: + 31 Credibility: lines for ~30 facts. |
| **Inline bracket tags** ([PRE-WINDOW…]/[SINGLE-SOURCE]/tier tags) | 1.1 | 0.0 | 4.1 | Mean is low but concentrated: q44555 = **15.7%**. Partially fixed by `eaa7721` (first-occurrence-full). |
| **Editorial implication blocks** ("Implication for forecasting" / "Relevance" / "Interpretation" per article) | 2.3 | 0.0 | 9.1 | q44219 = 13.4%. Not asked for by the prompt; a per-article editorial habit. |
| **Chatbot offers** ("If you want, I can…") | 0.3 | 0.1 | 0.8 | Small but reads as evidence to 6 forecasters. |
| **Total mechanical** | **~33** | 30.1 | 58.1 | |
| **Topical-drift article content** (semantic; judged) | **~20-25** | — | — | Microsoft agent survey + tokens/sec specs (44219), arXiv finances + crypto-mining incident (44225), heat-safety tips + non-weather local column (44551), obsolete Platner dossier (44555), intraday-quote chronicle (44773). |

The two together reproduce the judges' ~57%. Supporting stat: **26% of bold-number mentions
corpus-wide are repeats of a number already stated in the same section** (33-48% in the
failure cases) — fact-level self-duplication, invisible to 12-gram screens.

So the padding is, in order: (1) the self-summary tail (~24pp), (2) keyword-adjacent
articles that share the question's vocabulary but not its resolution criteria (~20-25pp),
(3) the per-fact source/credibility apparatus (~6pp), (4) per-article editorializing +
offers (~3pp). Cross-article restatement of the same fact runs through (1) and (2).

---

## 3. Fetch parameters (verified, `metaculus_bot/research/providers.py:106-228`)

- HOT: `search_news(query=question_text, n_articles=6, return_type="both",
  strategy="latest news")` — the strategy pins the crawl window to ~48h.
- HISTORICAL: `n_articles=10, strategy="news knowledge"` — relevance search over ~60 days.
- Query is the **raw question_text only** — resolution criteria and fine print never reach
  AskNews. (The summarizer gets them; the fetch doesn't.)
- No `categories`, no `similarity_score_threshold` override (default 0.5), no
  `start/end_timestamp`. URL-dedup across phases; historical rendered FIRST under
  "Historical Context & Background", hot second under "Recent Developments" — and the
  summarizer prompt's rule 7 explicitly preserves that ordering, so the freshest material
  is structurally buried at 47% (mini-era) to 57% (sol-era) of the way into the section.

**Is HISTORICAL the staleness vector?** Half-true, and the wrong lever alone. In 44219 and
44555 the stale/on-topic matter is historical-sourced and holds 10/16 of the article
budget. But the fresh drift is HOT-sourced: 44219's Jun 29-30 padding blocks (Microsoft
survey, GeneBench) *are* the hot articles. A blind hot-boost buys more fresh drift, not
more fresh signal. Also hot's 48h window misses decisive events 3-13 days old (Platner
withdrew Jul 10; a Jul-13 hot call can't see it — historical is what caught it). The fetch
fix worth making is **query enrichment** (append resolution-criteria key terms so the
relevance ranking optimizes for the right thing), not a ratio flip. The burial fix is
prompt-side ordering, which is free.

---

## 4. Prompt-structure findings (current `asknews_summarizer_prompt`, prompts.py:251)

1. **No recency weighting or supersession rule anywhere.** The prompt says date every fact
   and *flag* contradictions; it never says newer supersedes older, never says state which
   of two conflicting claims governs, and never asks for date arithmetic. This is the
   direct cause of 44255 and the enabler of 44219/44512.
2. **The comprehensiveness mandate is the padding engine.** "Extracts ALL facts...",
   "Be COMPREHENSIVE", "Omit only information that is clearly irrelevant" — an inclusion
   default over 16 articles of keyword-matched news. The measured result is a "summary"
   longer than its input (+52% in the one replay). The Jul-18 length-nudge removal trims
   the explicit ask but leaves the ALL/COMPREHENSIVE framing intact.
3. **Rule 7 ("Maintains the section structure (Historical Context vs Recent Developments)
   if present") mandates burial.** The raw format puts historical first; rule 7 makes the
   briefing mirror it, so the newest evidence starts at ~50% depth. Nothing asks for
   load-bearing-first ordering.
4. **Rule 5 ("Notes the date, source, and credibility of each piece of information")
   invites the apparatus.** Per-fact 3-line Date/Source/Credibility blocks (~6%,
   p90 10%, worst 32%) plus a standalone "Source-by-source credibility notes" subsection
   that duplicates the inline tags in 5/5 failure sections. The tier-tag instruction
   (`_SOURCE_TIER_TAG_INSTRUCTION`, `2b582d1`) is the compact replacement for exactly
   this — but the old rule-5 wording still coexists with it, so briefings emit both.
5. **Nothing forbids the self-summary tail** (Key facts / takeaways / credibility recap
   subsections = 24% of section chars) or per-article "Implication for forecasting"
   editorializing, and (pre-`eaa7721`) the full 75-char PRE-WINDOW tag repeated up to 57×.
6. **Silent staleness:** only 4/64 briefings state the age of their evidence in the
   opening 1,500 chars. A briefing built on 67-day-old leaderboard data reads exactly as
   authoritative as one built on yesterday's — that's what let 44219's stale lean out-shout
   two fresh sections.

---

## 5. Reform proposals (ranked; NOT implemented)

All prompt diffs target `asknews_summarizer_prompt` (`metaculus_bot/prompts.py:251`)
unless noted. Prevention matrix uses the five failures as run *plus* the already-shipped
no-forecast/window-stamping rules as baseline (i.e., credit only what today's prompt does
NOT already fix).

### R1 — Resolution-first prioritized briefing with a compression contract (biggest lever)

**Diff:** replace briefing-shape rules 1 and 7 and the "Be COMPREHENSIVE" bullet with:

- Open with a mandated `LATEST STATE (as of <newest on-topic article date>)` block: the
  3-8 facts most directly bearing on the resolution criteria, newest first, each dated.
  If no article directly addresses the resolution quantity/criteria, say so in one
  sentence (see R4).
- Then supporting context in descending load-bearing order (NOT the raw Historical/Recent
  article order).
- State each fact once. Do NOT append summary, key-facts, takeaway, or source-credibility
  recap sections. Do NOT add per-article "Implication/Relevance" commentary.
- Target ≤900 words. Compress, don't delete: secondary-but-relevant facts get one line
  each; screened-out topical-only articles get at most a one-line list at the end.

**Expected effect:** 44555 prevented as judged (obsolete chronology compressed to a
base-rate line; live replacement matchup leads); 44219/44551 padding+burial fixed and the
stale/off-crux lean demoted from headline to dated line-item; 44512 partial (single-
statement rule raises the odds the 25-vs-67 collision is confronted); 44255 no (needs R2).
Token effect: implements the content audit's "halve AskNews" (~$60/qtr: ~$25 input across
7 readers + ~half the summarizer's own output bill), likely overshoots halving given the
57% padding share.
**Risk:** moderate — the content audit warns AskNews's trimmed core is only lightly
duplicated (2020 precedent in 44555, sector context in 44453); a hard cap could drop it.
The "compress, don't delete" clause + replay validation is the mitigation. This is the one
reform that must be replay-validated before shipping.

### R2 — Supersession + deadline-arithmetic rule (fixes the wrong-direction cases)

**Diff:** add one CRITICAL RULES bullet:

- When two claims conflict and the conflict is time-ordered, the NEWER claim supersedes —
  present both with dates and state explicitly which governs now. Never blend or average
  conflicting numbers. When the question turns on a deadline, window, or cutoff date,
  compute the relevant calendar arithmetic explicitly against the resolution date and show
  the count (this is a calendar fact, not a forecast).

**Expected effect:** 44255 directly prevented (Jun 29 + 10 non-Sundays ≈ Jul 10 > Jul 4
becomes a mandatory computation — the fact the briefing conceded it lacked); 44512
largely prevented (25-swimming vs 67-total is a conflict the rule forces it to resolve
rather than trivia-flag); 44219 improved (Mar→Apr→Jun leaderboard readings become an
explicit supersession chain ending "newest available = Jun 23, unnamed lab"); 44555
reinforced (pre-withdrawal polls explicitly marked superseded, complementing R1); 44551
neutral.
**Risk:** low. Smallest diff, highest correctness-per-line. Watch one seam: the arithmetic
must stay on the evidence side of the no-forecast rule (computing "window expires ~Jul 10"
is a fact; "therefore No" would be a forecast).

### R3 — Per-article relevance gate keyed to resolution criteria

**Diff:** add before the extraction instructions:

- First classify each article: does it bear on THIS question's resolution criteria (not
  merely share its keywords/entities)? Articles that fail get at most one line in a
  "Screened out (topical only): …" list with a 5-word reason each. Never build extraction
  blocks for them.

**Expected effect:** kills the ~20-25pp semantic-drift padding (Microsoft survey, arXiv
finances, crypto incident, heat-safety tips, non-weather column). Prevents none of the
five directional errors alone, but shrinks the salience inversion that powered 44219/44555
and cuts the biggest share of what R1's budget would otherwise have to squeeze. Overlaps
R1 — if R1 ships in full, R3 is its screening step; ship R3 alone if R1 is judged too
aggressive.
**Risk:** low-moderate: a lazy screen could drop genuinely relevant background; the
mandatory screened-out list is the cheap audit trail (grep-able in run_logs).

### R4 — Mandatory evidence-age + coverage disclosure (the UPSTREAM mitigation)

**Diff:** fold into R1's LATEST STATE block (or add standalone):

- State the publication date of the newest article that DIRECTLY addresses the resolution
  criteria (not merely the topic). If none does, or the newest is older than ~2 weeks,
  say prominently: "No article in this set directly reports <resolution quantity>; the
  newest on-topic evidence is from <date>."

**Expected effect:** 44219's core harm neutralized — the briefing would have opened with
"newest leaderboard snapshot here is Apr 25 (67 days old)", letting the six forecasters
correctly prefer native_search/gap_fill's July data instead of reading an 18k-char section
as current; 44551 similarly discloses that no article addresses the Ogimet station ranking.
This is the only reform that helps the UPSTREAM cases, where no fetch/prompt change can put
non-news facts into a news index.
**Risk:** near-zero. Also gives the ops side a grep-able staleness signal (run_logs) —
cheap telemetry for free.

### R5 — Fetch query enrichment (+optional 8/8 rebalance) — weakest evidence, cheapest param lever

**Diff (`providers.py`):** build the AskNews query as `question_text` + a distilled clause
of resolution-criteria key terms (deterministic: title + the criteria's proper nouns /
quantity phrases; NO extra LLM call). Optionally rebalance `n_articles` 6 hot / 10
historical → 8/8.

**Expected effect:** aims the relevance ranking at what actually resolves the question —
in 44555 the replacement-matchup polls (which the fuzzy news ranking lost to
scandal-coverage volume) become the highest-similarity hits; marginal for 44219 (a current-
standings article may simply not exist in the index — R4 covers that); neutral elsewhere.
The 8/8 rebalance is genuinely two-sided per §3 (hot supplied the drift in 44219) — treat
it as an experiment arm, not a recommendation.
**Risk:** moderate-uncertain: query changes shift the whole retrieval distribution and are
the one reform NOT testable by replay (needs live AskNews calls to compare article sets).
Ship last, behind its own before/after eyeball.

**Also recommended (implementation hygiene, not a reform):** archive the raw
pre-summarization AskNews text (e.g. in `provider_results[].details`) — today it's thrown
away, which is why FETCH/SUMMARIZE attribution had to be forensic and why summarizer
replays require fresh paid pulls. One field, negligible size (~18k chars/q), makes every
future audit and replay free.

### Prevention matrix

| Reform | 44219 stale lean | 44255 deadline math | 44512 gold mislabel | 44551 Arizona lean | 44555 obsolete burial |
|---|---|---|---|---|---|
| R1 prioritize+budget | partial (demoted) | no | partial | partial (padding) | **yes** |
| R2 supersession+math | improved | **yes** | **likely** | no | reinforces |
| R3 relevance gate | partial (drift cut) | no | no | partial (drift cut) | partial |
| R4 age disclosure | **yes (neutralized)** | no | no | **yes (disclosed)** | supports |
| R5 query enrichment | marginal | no | no | no | likely (polls fetched) |
| *(already shipped: no-forecast rule, `eaa7721` tag fix)* | verdict gone | verdict gone | — | verdict gone | tags −15.7pp |

Top-3 to ship: **R2 (smallest diff, kills the wrong-direction class), R1+R3 as one
prioritization rewrite (kills the padding + burial class, delivers the ~$60/qtr), R4
(kills the silent-staleness class, only mitigation for UPSTREAM).** R5 is the optional
param experiment behind them.

---

## 6. Validation plan + cost

Replay-eval per the `scratch/research_role_audit_2026-07-17/` pattern: same inputs, prompt
variants, blinded judge.

- **Inputs:** raw pre-summarization pulls are not archived (see §0 caveat), so inputs are
  1 archived pull (Zambia q44229) + fresh AskNews pulls for ~9 more questions — ideally
  re-pulling the five failure questions themselves plus 4 controls. 2 calls/question
  within the existing AskNews subscription quota (no marginal $).
- **Arms:** current prompt (control), R2-only, R1+R3+R4, all-in. 10 questions × 4 arms
  × 1 sol-low summarizer call ≈ **40 calls ≈ $8-9** (measured $0.217/call; the R1 arms
  will be cheaper on output).
- **Judging:** blinded packets, judge-as-subagent (free), forced rankings + a targeted
  probe per known failure ("does the briefing state who leads NOW / do the deadline math /
  reconcile 25-vs-67?").
- **Watch item:** confirm on the R1 arm that the lightly-duplicated unique core (q44555's
  2020 precedent analog) survives the budget.

**Total marginal cost ≈ $10-15 OpenRouter** (worst case with retries), zero Metaculus
publishing, AskNews within subscription. R5, if pursued, needs a separate live A/B of
article sets (another ~20 AskNews calls, no LLM cost, eyeball comparison).
