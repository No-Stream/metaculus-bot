"""Forecaster-facing presentation for resolution-source fetch results.

This module shapes already-classified results into bounded page text and the
markdown section consumed by the forecasting prompt. It has no fetch, ladder,
deadline, marker, or provider responsibilities; those remain in
``resolution_source``. The one-way dependency on result vocabulary keeps the
rendered status and route tokens identical to the fetcher's contracts.
"""

from __future__ import annotations

from datetime import datetime

from metaculus_bot.constants import (
    RESOLUTION_SOURCE_DATAWRAPPER_MAX_CHARTS,
    RESOLUTION_SOURCE_DATAWRAPPER_PER_DATASET_MAX_CHARS,
    RESOLUTION_SOURCE_MIN_SECTION_CHARS,
    RESOLUTION_SOURCE_PER_URL_MAX_CHARS,
    RESOLUTION_SOURCE_TOTAL_MAX_CHARS,
)
from metaculus_bot.research.resolution_body_text import _truncate_csv_middle, _truncate_with_marker
from metaculus_bot.research.resolution_fetch_result import (
    ROUTE_CAVEATS,
    FetchResult,
    _render_fetch_failures,
)


def _unreadable_embed_disclosure(providers: list[str]) -> str:
    """The one-line note a rendered page carries when it hides figures in an embed.

    Forecaster-facing and deliberately plain: the section it sits in is captioned
    "primary grading evidence", so a page whose resolving numbers are NOT in the
    text has to say so or the caveat overstates what was retrieved. No count of
    embeds — one embed can be referenced by both a container div and a loader
    script, and an overstated count in evidence prose is its own small fabrication.
    """
    return (
        f"[This page displays data through {', '.join(providers)} embed(s) that this fetch cannot read — "
        f"any figures shown inside them are NOT in the page text below.]"
    )


def _page_text_with_leads(extracted: str, url: str, providers: list[str], chart_block: str = "") -> str:
    """Per-URL-capped page text, LED by the chart-data block and the embed disclosure.

    Both leads lead (exactly like the Tier-2 dataset lead) because every truncator
    on this text is head-preserving, so anything at the tail is the first thing a
    later trim discards. As a trailer the disclosure survived the per-URL truncation
    here but not the aggregate `_budgeted_success_sections` cut, which re-truncates
    an over-budget body through `_truncate_with_marker` — on prod constants (5 x
    6000 per-URL against an 18000 total) a fourth Infogram page rendered under the
    "primary grading evidence" caption with the disclosure gone and only a generic
    truncation marker left, which is the q44554/44556 failure the disclosure exists
    to prevent. Leading it also puts the caveat ahead of the text it qualifies,
    which is why the wording says "below".

    Chart data goes ABOVE the disclosure: on a page whose prose carries none of
    the resolving figures (q43949) it is the only resolving content in the section,
    so it must be the last thing any trim reaches, and the disclosure then still
    sits immediately above the prose it qualifies.

    Both leads are budgeted out of the cap rather than added on top, so the per-URL
    bound the section budget relies on still holds — including in the pathological
    case where the leads alone exceed the cap (a test can tune the cap below the
    chart block's own).
    """
    leads = [lead for lead in (chart_block, _unreadable_embed_disclosure(providers) if providers else "") if lead]
    if not leads:
        return _truncate_with_marker(extracted, RESOLUTION_SOURCE_PER_URL_MAX_CHARS, url)
    return _lead_then_capped_body("\n\n".join(leads), extracted, url)


def _lead_then_capped_body(lead: str, body: str, url: str) -> str:
    """A provenance lead, then as much of ``body`` as the per-URL cap leaves, inside the bound.

    The one arithmetic every rung that serves an artifact under a lead uses: the chart-data /
    embed leads here, the derived feed, the archived capture, the model's reading. The lead LEADS
    because every truncator on this text is head-preserving, so anything at the tail is the first
    thing a later trim discards, and a lead trimmed off leaves the artifact passed off as the live
    page (the q44554/44556 failure). Its cost comes OUT of the cap rather than on top of it, so the
    per-URL bound ``_budgeted_success_sections`` relies on still holds — including the
    pathological case where the lead alone exceeds the cap, where the lead itself is truncated
    (a bare lead there busts the bound the aggregate budget assumes). A blank body renders the
    lead alone for the same reason.
    """
    body_cap = RESOLUTION_SOURCE_PER_URL_MAX_CHARS - len(lead) - 2
    if body_cap <= 0 or not body.strip():
        return _truncate_with_marker(lead, RESOLUTION_SOURCE_PER_URL_MAX_CHARS, url)
    return f"{lead}\n\n{_truncate_with_marker(body, body_cap, url)}"


def _budgeted_success_sections(
    successes: list[FetchResult], fetched_iso: str
) -> tuple[list[str], list[FetchResult], int]:
    """Render the success sections inside the two partitioned budgets.

    Returns ``(sections, kept, dropped)``: the rendered sections, the results they were rendered
    FROM in the same order, and how many successes the budget dropped outright. ``kept`` exists
    for the route caveats, which describe an artifact a forecaster can see and so must be
    computed over what renders rather than over every success.

    Cited pages and Tier-2 datasets draw on separate allowances, so a chart's rows can
    never evict the page text the section exists to serve.

    A remainder under ``RESOLUTION_SOURCE_MIN_SECTION_CHARS`` drops the section rather than
    rendering into it: below the truncation marker's own length the truncator degrades to a bare
    slice, so a rescued section landing on a sliver rendered its provenance lead cut mid-word with
    no marker while the caveat block above, computed over ``kept``, promised a complete disclosure.
    """
    sections: list[str] = []
    kept: list[FetchResult] = []
    page_remaining = RESOLUTION_SOURCE_TOTAL_MAX_CHARS
    dataset_remaining = RESOLUTION_SOURCE_DATAWRAPPER_MAX_CHARTS * RESOLUTION_SOURCE_DATAWRAPPER_PER_DATASET_MAX_CHARS
    dropped = 0
    for r in successes:
        # Cheap per-section budget accounting on the text body only. Section
        # overhead (URL heading + fetched-date line) is negligible relative to
        # the RESOLUTION_SOURCE_TOTAL_MAX_CHARS total budget; if the caller
        # tightens it dramatically for a test, we still cut the text
        # conservatively.
        is_dataset = r.chart_id is not None
        remaining = dataset_remaining if is_dataset else page_remaining
        if remaining < RESOLUTION_SOURCE_MIN_SECTION_CHARS:
            dropped += 1
            continue
        body = r.text
        if len(body) > remaining:
            # Through the marker-emitting truncator, not a bare slice. A bare slice cut
            # mid-sentence AND could eat the per-URL `[truncated at N chars ...]` marker the
            # fetch already appended at the end — leaving an already-truncated page rendering
            # as complete. Reachable on prod constants (5 x 6000 per-URL against an 18000
            # total). The CSV variant keeps both ends, which is what makes a dataset's newest
            # rows survive whichever direction it runs.
            body = (_truncate_csv_middle if is_dataset else _truncate_with_marker)(body, remaining, r.url)
        if is_dataset:
            dataset_remaining -= len(body)
        else:
            page_remaining -= len(body)
        sections.append(f"### {r.url}\n(fetched {fetched_iso})\n\n{body}")
        kept.append(r)
    return sections, kept, dropped


def _route_caveats(rendered: list[FetchResult]) -> list[str]:
    """One sentence per non-direct route present in the sections that RENDER.

    Computed over the successes the section budget KEPT rather than over every result, because a
    caveat describes an artifact a forecaster can see: a rung that fired and failed left the direct
    route's own outcome, which the failure notice already names, and a success the aggregate budget
    dropped has no section below for the sentence to describe (on prod constants, 5 x 6000 per-URL
    pages against an 18000 total, a rendered page cited last was disclosed and then omitted). Order
    comes from ``ROUTE_CAVEATS``' own insertion order, so it is stable across questions rather than
    following fetch order.

    Empty for an all-direct question, which is the overwhelming majority and the case whose
    rendered section has to stay byte-identical to what it was before the ladder existed.
    """
    return [caveat for route, caveat in ROUTE_CAVEATS.items() if any(r.route == route for r in rendered)]


def format_resolution_sections(results: list[FetchResult], fetched_at: datetime) -> str:
    """Render fetch results as a markdown body block (orchestrator adds the ``##`` header).

    Returns ``""`` only when no URLs were attempted (empty ``results``). When
    URLs were attempted:

    - ALL failed (403 / JS wall / error / etc.) → a one-line notice naming the
      unreachable domains and their statuses, so forecasters learn the resolving
      page was never seen instead of silently getting nothing (the qid 44211
      failure: the CBP dashboard 403'd and no one in the pipeline knew).
    - SOME succeeded → the success sections as before, plus a terse trailing
      note about any that failed.

    Enforces ``RESOLUTION_SOURCE_TOTAL_MAX_CHARS`` across CITED-page success
    sections: later sections are trimmed (or dropped) once the budget is spent.
    Tier-2 dataset sections (``chart_id`` set) budget against their OWN allowance
    (``MAX_CHARTS x PER_DATASET_MAX_CHARS``) — the two classes are partitioned so
    a chart's rows can never evict the cited page text the section exists to
    serve, while a dataset still renders adjacent to its parent page. Per-URL
    truncation is the caller's responsibility (already applied in ``_fetch_one``
    and the hop); these caps cover the aggregate section length. When one or
    more sections are dropped entirely (budget spent before them), a final line
    names the dropped count so downstream readers can tell the snapshot is partial.

    Failure wording is partitioned the same way: a Datawrapper dataset is not a
    CITED resolution source, and its most common non-success — ``stale_data``,
    the freshness guard refusing to serve months-old data as live — is not a
    fetch failure at all, so datasets never ride the "cited resolution source(s)
    yielded no usable content" notices and get their own withheld line instead.
    """
    if not results:
        return ""

    successes = [r for r in results if r.status == "success"]
    cited_failures = [r for r in results if r.status != "success" and r.chart_id is None]
    dataset_nonsuccesses = [r for r in results if r.status != "success" and r.chart_id is not None]

    def _dataset_withheld_note() -> str:
        n = len(dataset_nonsuccesses)
        statuses = ", ".join(sorted({r.status for r in dataset_nonsuccesses}))
        # Wording covers every non-success a dataset can carry, not just
        # `stale_data`: a body that is empty or not row-shaped is withheld under
        # the same rule (nothing may be passed off as the chart's live series).
        return (
            f"[{n} embedded chart dataset(s) not served ({statuses}) — withheld rather than "
            f"passed off as the live series; the cited page text is unaffected.]"
        )

    if not successes:
        n = len(cited_failures)
        # "yielded no usable content", not "could not be fetched / was unreachable":
        # `no_resolving_content` and `empty_body` are pages that ANSWERED 200 and carried
        # nothing, and telling a forecaster the source was unreachable misstates the null
        # they have to weigh — "the tracker was down" and "the tracker has no reading" are
        # different pieces of evidence. The per-domain status token says which it was.
        notice = (
            f"[{n} resolution source(s) yielded no usable content: {_render_fetch_failures(cited_failures)}] — "
            f"nothing from the cited resolving page(s) is in this bundle; weight other evidence accordingly."
        )
        if dataset_nonsuccesses:
            notice += "\n\n" + _dataset_withheld_note()
        return notice

    fetched_iso = fetched_at.strftime("%Y-%m-%d")
    sections, kept, dropped = _budgeted_success_sections(successes, fetched_iso)
    caveat = "\n".join(
        [
            f"Snapshot of the cited resolution source(s) as of {fetched_iso} — primary grading evidence.",
            *_route_caveats(kept),
        ]
    )

    rendered = caveat + "\n\n" + "\n\n".join(sections)
    if dropped:
        rendered += f"\n\n[{dropped} additional source(s) omitted — section budget]"
    if cited_failures:
        rendered += (
            f"\n\n[Note: {len(cited_failures)} other cited resolution source(s) yielded no usable content: "
            f"{_render_fetch_failures(cited_failures)} — weight accordingly.]"
        )
    if dataset_nonsuccesses:
        rendered += "\n\n" + _dataset_withheld_note()
    return rendered
