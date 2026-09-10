"""Paired section-strip bench: how much do the two gap-fill sections move a cheap forecaster?

Every archived research bundle ends with two appended gap-fill sections, v1 (``## Targeted Gap-Fill
(second pass)``, about $0.76 a question) and v2 (``## Agentic Research Findings``, about $0.36), and
whether v1 stays for the fall turns on whether its text changes forecasts. For every resolved question
where both ran, this bench forecasts four arms of the same frozen bundle with ONE cheap model (``full``
as published, ``minus_v1``, ``minus_v2``, ``minus_both``), several replicates each, scores each forecast
against the resolution with the repo's scoring functions, and reports the paired full-minus-arm deltas.
No research runs: the bundles come off ``backtests/research_archive/latest/``, the question metadata off
the residual round's tagged dataset, and the only network calls are completions on the OPERATOR'S
PERSONAL OpenRouter key (the donated key is removed from the process environment before any client is
built). The default model, Meta's Muse Spark 1.3 Contributor tier, may train on prompts; the operator
accepted that for this open-source repo, and the OpenRouter account's privacy setting must allow
training providers or every call fails on data policy. Runs are gated like ``gemini_verify.py``: a bare
invocation prints the plan and the estimate and refuses, ``--dry-run`` prints them and exits clean,
``--i-accept-spend`` runs under ``--max-spend-usd``, and ``--rescore <run dir>`` rebuilds the results
from ``calls.jsonl`` offline.

    uv run python -m scripts.probes.section_strip_bench --dry-run
    uv run python -m scripts.probes.section_strip_bench --i-accept-spend

Modules: ``bundle`` (the archive cut into arms, the question per pair), ``scoring`` (ladder, CDF build,
log scores), ``plan`` (prompts and the estimate), ``run`` (the paid call and the spend meter), ``report``
(rows, paired deltas, Markdown), ``cli``.
"""
