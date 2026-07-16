# Classification rubric — gap-fill v2 step-zero miss audit

You are classifying WHY a forecasting-bot miss happened, using a dossier that contains:
the question (title, description, resolution criteria, fine print), the resolution, the
bot's published forecast + per-model forecasts, the research bundle the forecaster LLMs
were given, and the per-model forecaster rationales.

## The taxonomy (assign exactly ONE primary bucket per question)

- **A. Missing fact** — a decisive, publicly available fact never appeared in the research
  bundle (e.g. the vote had already been scheduled, the data release was already out, an
  event on a tracking list had already occurred). The forecast would plausibly have been
  materially different with that fact.
- **B. Stale fact** — the bundle HAD the fact but in an outdated form; fresher
  primary-source state existed before the question closed and would have changed the read.
- **C. Misread resolution mechanics** — forecasters (or the research synthesis)
  misunderstood the resolution criteria / fine print: counting events outside the
  resolution window, wrong metric (GAAP vs non-GAAP, 3-day vs 5-day gross), wrong source,
  wrong threshold semantics, wrong entity. The information was there; the mechanics were
  misapplied.
- **D. Hallucinated/misattributed research** — the bundle or a rationale asserts something
  the underlying source does not support, and the forecast leaned on it.
- **E. Judgment** — the research was adequate; the models weighed it badly (overconfidence,
  base-rate neglect, herding on prediction markets, refusing to trust their own arithmetic,
  bad tail width on numerics). ALSO use E (note "pipeline" in the justification) when
  research was adequate but a pipeline/formatting failure distorted the submitted forecast
  (e.g. mass piled at an open bound the model couldn't express beyond).
- **F. Genuinely surprising resolution** — a low-probability outcome occurred; the forecast
  was reasonable ex ante. Reserve F for cases where you'd defend the ex-ante forecast even
  knowing everything in the bundle plus anything cheaply findable at forecast time.

A–D are "research-comprehension" failures (addressable by a better agentic research stage);
E–F are not.

## How to decide (work through in order)

1. Read the question + resolution criteria + fine print CAREFULLY. What exactly resolves it?
2. Read the resolution + the published forecast. What was the miss, quantitatively?
3. Find the decisive consideration: what fact or judgment, had it been different, flips this
   from a miss to a hit?
4. Check the bundle: was that decisive fact present? Present-but-stale? Absent? Misstated?
5. Check 2–3 forecaster rationales: did they use the bundle correctly? Did they misread the
   resolution mechanics? Did they assert facts not in the bundle?
6. Only if research was fine and mechanics were understood: was the weighing bad (E), or was
   this a reasonable forecast beaten by a surprising world (F)?

Rules of thumb:
- Bucket the PRIMARY driver. If a stale bundle AND bad judgment both contributed, ask which
  one moves the forecast more; note the secondary bucket.
- A requires the missing fact to have plausibly existed publicly BEFORE the forecast was
  submitted (check the submission date vs. what the fact is). If you cannot establish from
  the dossier whether such a fact existed pre-submission, say so explicitly and lower your
  confidence — do NOT invent world events. Your training data ends before most of these
  resolutions; judge from the dossier only.
- Herding on prediction markets that were themselves wrong = E (the info was there).
- "Research couldn't have known" (future event, genuine tail) = F, not A.
- For numerics: if the bundle's central estimate was near the truth but the submitted
  distribution wasn't, that is E (judgment or pipeline), not a research failure.

## Output format (per question, strict)

### qid <QID> — <short title>
- **Cohort**: miss | control
- **Miss summary**: one sentence — what we said vs what happened (with numbers).
- **Decisive consideration**: one sentence.
- **Bundle check**: what the bundle had/lacked on that point (quote a short key phrase).
- **Rationale check**: what the models did with it (name 1–2 models, quote briefly).
- **PRIMARY bucket**: A|B|C|D|E|F
- **Secondary bucket** (optional): letter or "none"
- **Justification**: 1–2 sentences. For A–D name the SPECIFIC fact/misreading.
- **Confidence**: high | medium | low (+ what would change the call)
- **v2-addressable**: yes (A–D) | no (E–F)

## Control questions (cohort=control)

Same reading, different deliverable: these scored WELL. Ask: would the same critical reading
have flagged a research failure here anyway? Classify as:
- **clean** — bundle adequate, mechanics understood, forecast well-founded; or
- **latent A/B/C/D** — the bundle had a gap/staleness/misread of the same kind we flag in
  misses, but it didn't end up biting (i.e. the buckets might be partly hindsight).
Use the same output format; put "clean" or "latent <letter>" in PRIMARY bucket.
