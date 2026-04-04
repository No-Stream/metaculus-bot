from datetime import datetime

from forecasting_tools import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    clean_indents,
)

__all__ = [
    "binary_prompt",
    "disagreement_crux_prompt",
    "multiple_choice_prompt",
    "numeric_prompt",
    "stacking_binary_prompt",
    "stacking_multiple_choice_prompt",
    "stacking_numeric_prompt",
    "targeted_search_prompt",
]


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def binary_prompt(question: BinaryQuestion, research: str) -> str:
    """
    Return the forecasting prompt for binary questions.
    """

    return clean_indents(
        f"""
            You are a senior forecaster preparing a public report for expert peers.
            You will be judged based on the accuracy _and calibration_ of your forecast with the Metaculus peer score (log score).
            You should consider current prediction markets when possible but not be beholden to them.

            Your Metaculus question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {_today_str()}.
            Reproduce the following analysis template in your answer:

            ── Analysis Template ──

            PHASE 0: PRELIMINARY CHECK

            0) Resolution check
               • Does the research already contain evidence that the resolution condition has been met (or is now impossible to meet)? If so, assign a near-extreme probability (≥95% or ≤5%), briefly explain why, and skip to the final answer. Do not perform full reference-class analysis for questions whose answers are already deterministic from current evidence.

            PHASE 1: OUTSIDE VIEW (anchor on historical context above)

            1) Source analysis (focus on historical context section)
               • Briefly summarize the main sources from the briefing; include date, credibility, and scope.
               • Separate facts from opinions. Exercise healthy skepticism: only weight opinions strongly when they come from identifiable experts or credentialed entities. Internet sources mix fact and opinion freely.

            2) Reference class and quantitative base rate
               • List plausible reference classes for this question and evaluate suitability.
               • State the outside-view base rate(s) and how you combine them into a baseline probability.
               • Attempt an explicit calculation if the data supports it: historical frequency, rate extrapolation, z-score, or probability union (for "at least one of N" questions, compute 1 - product of (1-p_i)). A rough quantitative estimate from data is more reliable than an intuitive guess.

            3) Timeframe reasoning
               • How long until resolution? If the timeline were halved/doubled, how would the probability shift and why?

            ── Now consider the recent developments above ──

            PHASE 2: INSIDE VIEW UPDATE (update from your base rate using current news)

            4) Evidence weighting (current news items classified as Strong/Moderate/Weak)
               • Classify key evidence using this rubric:
                 - Strong: multiple independent sources; clear causal mechanisms; strong precedent
                 - Moderate: one good source; indirect links; weak precedent
                 - Weak: anecdotes; speculative logic; volatile indicators

            5) Competing cases and red-teaming
               • Strongest Bear Case (No): most compelling, evidence-based argument for No.
               • Strongest Bull Case (Yes): most compelling, evidence-based argument for Yes.
               • Red-team both: attack assumptions, data gaps, and causal claims.

            6) Final rationale and calibration — integrate outside→inside view
               • Explicitly state: "My base rate was X%. After considering current evidence, I'm moving to Y% because..."
               • Odds check: translate your probability to odds (e.g., 90% = 9:1, 99% = 99:1). Does this feel right? How would a ±10% shift resonate with your analysis?
               • Small-delta check: would a ±10% change still be coherent with the rationale? Why?
               • Trajectory check: consider whether the "status quo" means "nothing changes" or "the current trajectory reaches its natural conclusion" (e.g., a deadline arriving, a trend continuing, a process completing). Justify predictions that diverge from the most likely trajectory.
               • Quantitative anchor: if you computed a probability from data (z-score, historical frequency, regression, etc.), state that number and explain how your final answer relates to it. If you're adjusting away from a data-derived probability, name the specific reason.

            ── Brief checklist (keep concise) ───────────────────────────────
            • Paraphrase the resolution criteria (<30 words).
            • Bait-and-switch check: does your reasoning address the EXACT question and resolution criteria, not a related-but-different question? This is a common and costly error.
            • State the outside-view base rate you anchored to.
            • Consistency line: "X out of 100 times, [criteria] happens." Sensible?
            • Top 3-5 evidence items + quick factual validity check.
            • Blind-spot scenario most likely to make this forecast wrong; direction of impact.
            • Trajectory check sanity check: does your prediction align with the most likely trajectory?

            [The last thing you write MUST BE your final answer as an INTEGER percentage. "Probability: ZZ%"]
            An example response is: "Probability: 50%"
            """
    )


def multiple_choice_prompt(question: MultipleChoiceQuestion, research: str) -> str:
    return clean_indents(
        f"""
        You are a **senior forecaster** preparing a rigorous public report for expert peers.
        Your accuracy and *calibration* will be scored with Metaculus' log-score, so avoid
        over-confidence and make sure your probabilities sum to **100%**.
        Please consider news, research, and prediction markets, but you are not beholden to them.

        ── Question ──────────────────────────────────────────────────────────
        {question.question_text}

        • Options (in resolution order): {question.options}

        

        ── Context ───────────────────────────────────────────────────────────
        {question.background_info}

        {question.resolution_criteria}
        {question.fine_print}

        ── Intelligence Briefing (assistant research) ────────────────────────
        {research}

        Today's date: {_today_str()}
        Reproduce the following analysis template in your answer:

        ── Analysis Template ──

        PHASE 1: OUTSIDE VIEW (anchor on historical context above)

        (1) Source analysis (focus on historical context section)
            • Summarize key sources; note recency, credibility, and scope.
            • Separate facts from opinions. Exercise healthy skepticism: only weight opinions strongly when they come from identifiable experts or credentialed entities. Internet sources mix fact and opinion freely.

        (2) Reference class (outside view) analysis
            • Candidate reference classes and suitability.
            • Outside-view distribution over options; discuss the historical rate of upsets/unexpected outcomes in this domain and how that affects the distribution.

        (3) Timeframe reasoning
            • Time to resolution; describe how halving/doubling the timeline might reshape the distribution.

        ── Now consider the recent developments above ──

        PHASE 2: INSIDE VIEW UPDATE (update from your base rate using current news)

        (4) Evidence weighting (current news items classified as Strong/Moderate/Weak)
            • Apply the rubric:
              - Strong: multiple independent sources; clear causality; strong precedent
              - Moderate: one good source; indirect links; weak precedent
              - Weak: anecdotes; speculative logic; volatile indicators

        (5) Strongest pro case for the currently most-likely option
            • Use weighted evidence and explicit causal chains.

        (6) Red-team critique
            • Attack assumptions in (5); highlight hidden premises and data that could flip the conclusion.

        (7) Unexpected scenario(s)
            • Plausible but overlooked pathways for a different option to win; justify residual mass on tails.

        (8) Final rationale and calibration — integrate outside→inside view
            • Explicitly state: "My base rate was X%. After considering current evidence, I'm moving to Y% because..."
            • Odds check: translate your probability to odds (e.g., 90% = 9:1, 99% = 99:1). Does this feel right? How would a ±10% shift resonate with your analysis?
            • Small-delta check: would ±10% on the leading options remain coherent with your reasoning?
            • Blind-spot consideration: if the resolution is unexpected, what would likely be the reason, and how should that affect confidence spreads?
            Remember:
            • Good forecasters leave a little probability on most options and avoid overconfidence.
            • Use integers 1%-99% (no 0 % or 100 %).
            • They must sum to 100 %.

        ── Brief checklist (keep concise) ───────────────────────────────────
        • Paraphrase options & resolution criteria (<30 words).
        • Bait-and-switch check: does your reasoning address the EXACT question and resolution criteria, not a related-but-different question? This is a common and costly error.
        • State the outside-view distribution used as anchor.
        • Consistency line: "Most likely: __; least likely: __; coherent with rationale?"
        • Top 3-5 evidence items + quick factual validity check.
        • Blind-spot statement; trajectory check sanity check.

        [**CRITICAL**: You MUST assign a probability (1-99%) to EVERY single option listed above.
        Even if an option seems very unlikely, assign it at least 1%. Never skip any option.]

        ── Final answer (must be last lines, one line per option, all options included, in same order, nothing after) ──
        Option_A: NN%
        Option_B: NN%
        …
        Option_N: NN%
        """
    )


def numeric_prompt(
    question: NumericQuestion,
    research: str,
    lower_bound_message: str,
    upper_bound_message: str,
) -> str:
    unit_str = question.unit_of_measure or "unknown units, assume unitless (e.g. raw count)"
    return clean_indents(
        f"""
        You are a **senior forecaster** writing a public report for expert peers.
        You will be scored with Metaculus' log-score, so accuracy **and** calibration
        (especially the width of your prediction interval) are critical.
        Calibration guidance: For volatile quantities (financial markets, novel events, short-horizon
        relative returns), produce wide, diffuse distributions — these are fundamentally hard to predict.
        For stable, well-measured indicators with recent data (economic indices, demographic measures,
        climate data), anchor tightly to recent observations with historically-appropriate variance.
        Do not over-hedge on quantities you can actually predict well.
        Given the mathematics of log score, penalties for overconfident narrow intervals are severe,
        but penalties for overly wide intervals on predictable quantities also accumulate.
        Please consider news, research, and prediction markets, but you are not beholden to them.

        ── Question ──
        {question.question_text}

        ── Context ──
        {question.background_info}

        {question.resolution_criteria}
        {question.fine_print}

        ── Units & Bounds (must follow) ──
        • Base units for output values: {unit_str}
        • Allowed range (in base units): [{getattr(question, "lower_bound", "???")}, {getattr(question, "upper_bound", "???")}]
        • Note: allowed range is suggestive of units! If needed, you may use it to infer units.
        • All 11 percentiles you output must be numeric values in the base unit and fall within that range.
        • If your reasoning uses billions/millions/thousands, convert to base unit numerically (e.g., 350B → 350000000000). No suffixes or scientific notation, just numbers.

        ── Scoring Rule ──
        Metaculus continuous questions use a log density score: score = ln f(x*), where f is your forecasted PDF evaluated at the realized value x*. A uniform 0.01 floor is added to every PDF to avoid −∞; excluding the truth yields ln(0.01) ≈ -4.605, while sharp accuracy is rewarded (e.g., f(x*) = 10 → +2.303). Probability mass below/above the bounds is scored as a binary event;  PDF sharpness is capped (about 0.01 ≤ f ≤ ~35), so spiky tricks don't pay. This is a proper scoring rule—to maximize expected score, report your true uncertainty and resist overconfident, narrow shapes.

        ── Intelligence Briefing (assistant research) ────────────────────────
        {research}

        Today's date: {_today_str()}

        {lower_bound_message}
        {upper_bound_message}

        Reproduce the following analysis template in your answer:

        -- Analysis Template ──

        PHASE 1: OUTSIDE VIEW (anchor on historical context above)

        (1) Source analysis and data anchor
            - Summarize key sources; note recency, credibility, and scope.
            - Separate facts from opinions. Exercise healthy skepticism: only weight opinions strongly when they come from identifiable experts or credentialed entities. Internet sources mix fact and opinion freely.
            - Critical: what is the most recent authoritative measurement or data point for this quantity? Your prediction should be centered near this value unless you have strong, specific evidence for departure.

        (2) Outside view and quantitative modeling
            - Candidate reference classes and suitability.
            - State the outside view range and how you anchor to it.
            - If the data supports it, perform an explicit quantitative estimate: extrapolate recent trends, compute historical mean and variance, or fit a simple model. A rough calculation from data is more reliable than an intuitive range estimate.

        (3) Timeframe and dynamics
            - Time to resolution; describe how halving or doubling the timeline might shift percentiles.
            - Status-quo outcome: what value is implied if current conditions simply persist.
            - Trend continuation: extrapolate historical data to the closing date.

        (4) Expert and market priors
            - Cite ranges or point forecasts from specialists, prediction markets, or peers.

        ── Now consider the recent developments above ──

        PHASE 2: INSIDE VIEW UPDATE (update from your base rate using current news)

        (5) Evidence weighting for inside view adjustments (current news items classified as Strong/Moderate/Weak)
            - Strong: multiple independent sources, clear causal links, strong precedent
            - Moderate: one good source, indirect links, weak precedent
            - Weak: anecdotes, speculative logic, volatile indicators

        (6) Tail scenarios
            - Coherent pathway for unusually low results.
            - Coherent pathway for unusually high results.

        (7) Red team and final rationale — integrate outside→inside view
            - Challenge assumptions and data quality.
            - Explicitly state: "My base rate was X%. After considering current evidence, I'm moving to Y% because..."
            - Odds check: translate your probability to odds (e.g., 90% = 9:1, 99% = 99:1). Does this feel right? How would a ±10% shift resonate with your analysis?
            - Small delta check: would +/- 10 percent on key percentiles still fit the reasoning
            - Trajectory check: consider whether "status quo" means "nothing changes" or "the current trajectory reaches its natural conclusion." Justify deviations from the most likely trajectory.
            - Quantitative anchor: if you derived a central estimate or range from data, state it and explain how your final percentiles relate to it.

        (8) Calibration and distribution shaping
            - Think in ranges, not single points.
            - Keep 2.5% and 97.5% far apart to allow for unknown unknowns.
            - Ensure strictly increasing percentiles.
            - Avoid scientific notation.
            - Respect the explicit bounds above.

        (9) Outcome type classification
            Determine whether the resolution value for this question will always be a whole integer
            (e.g. counts, rankings, number of events, number of countries) or can be any real number
            (e.g. temperatures, percentages, dollar amounts, ratios).
            Output exactly one of:
            OUTCOME_TYPE: DISCRETE
            OUTCOME_TYPE: CONTINUOUS

        (9b) Forecastability classification
            How inherently predictable is this quantity on the given time horizon?
            - HIGH: stable indicator with recent data, low historical variance
              (e.g., monthly unemployment rate, home price index, CO2 concentration)
            - MEDIUM: event-based or moderately variable
              (e.g., election results, quarterly earnings, box office)
            - LOW: volatile or near-random on this horizon
              (e.g., 2-week stock/futures returns, financial spreads, novel metrics)
            Output exactly one of:
            FORECASTABILITY: HIGH
            FORECASTABILITY: MEDIUM
            FORECASTABILITY: LOW
            For LOW forecastability, your IQR should span a large fraction of the allowed range.
            For HIGH, your IQR can be as narrow as the historical data justifies.

        (10) Brief checklist
            - Units: what are the units of the output values and why? Incorrect units can cause severe penalties in log score.
            - Paraphrase the resolution criteria and units in less than 30 words.
            - Bait-and-switch check: does your reasoning address the EXACT question and resolution criteria, not a related-but-different question? This is a common and costly error.
            - State the outside view baseline used.
            - Consistency line about which percentile corresponds to the status quo or trend.
            - Top 3 to 5 evidence items plus a quick factual validity check.
            - Blind spot scenario and expected effect on tails.
            - Trajectory check sanity check: does your prediction align with the most likely trajectory?
            - Forecastability check: does your interval width match the forecastability classification?
            - Remember: log score penalizes both overconfident narrow intervals AND overly wide intervals on predictable quantities.

        Prediction:
        [Reminders:
        - Floating point numbers in the base unit
        - Must be last lines, nothing after
        - STRICTLY INCREASING percentiles meaning e.g. p20 > p10 and not equal.)
        Example:]

        Percentile 2.5: 1.2
        Percentile 5: 10.1
        Percentile 10: 12.3
        Percentile 20: 23.4
        Percentile 40: 34.5
        Percentile 50: 45.6
        Percentile 60: 56.7
        Percentile 80: 67.8
        Percentile 90: 78.9
        Percentile 95: 89.0
        Percentile 97.5: 123.4
        """
    )


def stacking_binary_prompt(question: BinaryQuestion, research: str, base_predictions: list[str]) -> str:
    """Return the stacking prompt for binary questions that takes multiple model predictions as input."""
    predictions_text = "\n".join([f"Model {i + 1} Analysis:\n{pred}\n" for i, pred in enumerate(base_predictions)])

    return clean_indents(
        f"""
        You are a senior meta-forecaster specializing in combining predictions from multiple expert models.
        You will be judged based on the accuracy and calibration of your final forecast using the Metaculus peer score (log score).
        
        Your task is to synthesize multiple expert analyses into a single, well-calibrated probability.
        
        Your Metaculus question is:
        {question.question_text}
        
        Question background:
        {question.background_info}
        
        This question's outcome will be determined by the specific criteria below:
        {question.resolution_criteria}
        
        {question.fine_print}
        
        Your research assistant provided this context:
        {research}
        
        Today is {_today_str()}.
        
        ── Multiple Expert Analyses ──
        {predictions_text}
        
        ── Meta-Analysis Framework ──
        1) Model agreement analysis
           • Where do the models agree? What shared evidence drives consensus?
           • Where do they disagree? What causes divergent reasoning?
           • Are disagreements due to different evidence weighting or different evidence sources?
        
        2) Evidence synthesis
           • Which evidence appears most frequently across analyses? Is this justified?
           • What unique evidence does each model bring? How credible is it?
           • Are there systematic biases visible across models (overconfidence, anchoring, etc.)?
        
        3) Reasoning quality assessment
           • Which models demonstrate strongest analytical rigor?
           • Which models best incorporate reference class reasoning?
           • Which models show appropriate uncertainty calibration?
        
        4) Meta-level adjustments
           • Should I weight models equally or give more weight to better-reasoned analyses?
           • Are there blind spots that all models missed?
           • How should I account for model correlation vs independence?
        
        5) Final synthesis
           • What probability best integrates all the evidence and reasoning?
           • Does this probability appropriately reflect the uncertainty in the question?
           • Sanity check: does this probability make sense given the base rate and evidence?
        
        The last thing you write MUST BE your final answer as an INTEGER percentage. "Probability: ZZ%"
        An example response is: "Probability: 50%"
        """
    )


def stacking_multiple_choice_prompt(
    question: MultipleChoiceQuestion, research: str, base_predictions: list[str]
) -> str:
    """Return the stacking prompt for multiple choice questions."""
    predictions_text = "\n".join([f"Model {i + 1} Analysis:\n{pred}\n" for i, pred in enumerate(base_predictions)])

    return clean_indents(
        f"""
        You are a senior meta-forecaster specializing in combining predictions from multiple expert models.
        Your accuracy and calibration will be scored with Metaculus' log-score, so avoid over-confidence 
        and make sure your probabilities sum to **100%**.
        
        ── Question ──────────────────────────────────────────────────────────
        {question.question_text}
        
        • Options (in resolution order): {question.options}
        
        ── Context ───────────────────────────────────────────────────────────
        {question.background_info}
        
        {question.resolution_criteria}
        {question.fine_print}
        
        ── Intelligence Briefing ────────────────────────────────
        {research}
        
        Today's date: {_today_str()}
        
        ── Multiple Expert Analyses ──
        {predictions_text}
        
        ── Meta-Analysis Framework ──
        1) Model agreement analysis
           • Which options show consensus vs divergence across models?
           • What shared reasoning drives agreement on likely/unlikely options?
           • Where models disagree, what drives the different assessments?
        
        2) Evidence synthesis across models
           • What evidence appears consistently? Is this justified by source quality?
           • What unique insights does each model contribute?
           • Are there systematic biases (overconfidence on favorites, neglect of tails)?
        
        3) Probability distribution analysis
           • Which models show appropriate uncertainty (avoid 0%/100%)?
           • How do the models differ in their tail probability allocation?
           • Are there systematic patterns in how models distribute probability?
        
        4) Reasoning quality assessment
           • Which analyses demonstrate strongest logical coherence?
           • Which models best incorporate reference class reasoning?
           • Which show most appropriate calibration for this question type?
        
        5) Meta-level synthesis
           • Should models be weighted equally or by reasoning quality?
           • Are there overlooked scenarios that all models missed?
           • How should I account for correlation vs independence in model errors?
        
        6) Final distribution calibration
           • What probability distribution best synthesizes all analyses?
           • Does my distribution appropriately reflect uncertainty?
           • Are my tail probabilities justified given the evidence?
        
        **CRITICAL**: You MUST assign a probability (1-99%) to EVERY single option listed above.
        Even if an option seems very unlikely, assign it at least 1%. Never skip any option.
        
        ── Final answer (must be last lines, one line per option, all options included, in same order, nothing after) ──
        Option_A: NN%
        Option_B: NN%
        …
        Option_N: NN%
        """
    )


def stacking_numeric_prompt(
    question: NumericQuestion,
    research: str,
    base_predictions: list[str],
    lower_bound_message: str,
    upper_bound_message: str,
) -> str:
    """Return the stacking prompt for numeric questions."""
    predictions_text = "\n".join([f"Model {i + 1} Analysis:\n{pred}\n" for i, pred in enumerate(base_predictions)])

    return clean_indents(
        f"""
        You are a senior meta-forecaster specializing in combining predictions from multiple expert models.
        You will be scored with Metaculus' log-score, so accuracy **and** calibration 
        (especially the width of your 90/10 interval) are critical.
        
        ── Question ──────────────────────────────────────────────────────────
        {question.question_text}
        
        ── Context ───────────────────────────────────────────────────────────
        {question.background_info}
        
        {question.resolution_criteria}
        {question.fine_print}
        
        Units: {question.unit_of_measure or "Not stated: infer if possible"}
        
        ── Units & Bounds (must follow) ─────────────────────────────────────
        • Base unit for output values: {question.unit_of_measure or "base unit"}
        • Allowed range (base units): [{getattr(question, "lower_bound", "???")}, {getattr(question, "upper_bound", "???")}]
        • All 11 percentiles you output must be numeric values in the base unit and fall within that range.
        • If your reasoning uses B/M/k, convert to base unit numerically (e.g., 350B → 350000000000). No suffixes.

        ── Scoring Rule ──
        Metaculus continuous questions use a log density score: score = ln f(x*), where f is your forecasted PDF evaluated at the realized value x*. A uniform 0.01 floor is added to every PDF to avoid −∞; excluding the truth yields ln(0.01) ≈ -4.605, while sharp accuracy is rewarded (e.g., f(x*) = 10 → +2.303). Probability mass below/above the bounds is scored as a binary event;  PDF sharpness is capped (about 0.01 ≤ f ≤ ~35), so spiky tricks don't pay. This is a proper scoring rule—to maximize expected score, report your true uncertainty and resist overconfident, narrow shapes.
        
        ── Intelligence Briefing ────────────────────────────────
        {research}
        
        Today's date: {_today_str()}
        
        {lower_bound_message}
        {upper_bound_message}
        
        ── Multiple Expert Analyses ──
        {predictions_text}
        
        ── Meta-Analysis Framework ──
        1) Distribution comparison
           • Compare the central tendencies (medians) across models - what explains differences?
           • Compare uncertainty ranges (90% intervals) - which models show appropriate calibration?
           • Are there systematic patterns in how models approach this forecasting problem?
        
        2) Evidence synthesis
           • What evidence/approaches appear across multiple analyses?
           • What unique insights or data does each model contribute?
           • Which models demonstrate strongest analytical rigor for this question type?
        
        3) Calibration assessment
           • Which models show appropriate uncertainty given the available evidence?
           • Are any models systematically overconfident (too narrow ranges)?
           • Which uncertainty ranges seem most justified by the evidence quality?
        
        4) Reference class integration
           • How do models differ in their reference class selection?
           • Which outside view approaches seem most appropriate?
           • Should I favor models with stronger reference class reasoning?
        
        5) Meta-level synthesis
           • Should I weight models equally or by reasoning quality?
           • Are there blind spots or scenarios all models missed?
           • How should I account for correlation vs independence in model approaches?
        
        6) Final distribution calibration
           • What percentiles best synthesize all the evidence and reasoning?
           • Does my final distribution appropriately reflect epistemic uncertainty?
           • Are my tails justified given the potential for unknown unknowns?
        
        Remember: Think in ranges, not points. Keep 2.5th and 97.5th percentiles appropriately wide.
        Ensure strictly increasing percentiles and respect the bounds above.

        OUTPUT FORMAT, floating point numbers
        Must be last lines, nothing after, STRICTLY INCREASING percentiles meaning e.g. p20 > p10 and not equal.

        Percentile 2.5: [value]
        Percentile 5: [value]
        Percentile 10: [value]
        Percentile 20: [value]
        Percentile 40: [value]
        Percentile 50: [value]
        Percentile 60: [value]
        Percentile 80: [value]
        Percentile 90: [value]
        Percentile 95: [value]
        Percentile 97.5: [value]
        """
    )


def disagreement_crux_prompt(question_text: str, base_predictions: list[str]) -> str:
    """Prompt for a cheap model to extract the core factual disagreement between forecaster analyses."""
    predictions_text = "\n".join([f"Forecaster {i + 1} Analysis:\n{pred}\n" for i, pred in enumerate(base_predictions)])

    return clean_indents(
        f"""
        Multiple forecasters analyzed the same question and produced significantly different predictions.

        Question:
        {question_text}

        ── Forecaster Analyses ──
        {predictions_text}

        Read the analyses above. They disagree. Identify the core factual question(s) driving
        the disagreement — what specific facts, data points, or events do the forecasters
        interpret differently or assume differently about?

        Output ONLY the factual question(s), in 1-3 sentences. Do not forecast, do not give
        opinions, do not explain your reasoning.
        """
    )


def targeted_search_prompt(crux: str, question_text: str, *, is_benchmarking: bool = False) -> str:
    """Prompt for Grok with native search to resolve a specific factual disagreement."""
    benchmarking_warning = (
        "\n\nIMPORTANT: This is a benchmarking run. DO NOT search for or include prediction "
        "market odds, forecasts, or betting lines — this would constitute data leakage."
        if is_benchmarking
        else ""
    )
    return clean_indents(
        f"""
        Search the web for current, factual information to resolve this specific question:
        {crux}

        This is for forecasting the following question:
        {question_text}

        Focus on: recent official data, primary sources, quantitative evidence, confirmed
        timelines, and resolution-relevant facts. Include inline citations [source](url)
        for all claims.{benchmarking_warning}
        """
    )
