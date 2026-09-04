"""The per-URL fetch-outcome vocabulary for the resolution-source provider.

One responsibility: what a single fetch attempt is allowed to SAY. That is the
``FetchStatus`` set, the one HTTP-status table both the Tier-1 page fetch and the
Tier-2 Datawrapper CDN hop read, the ``FetchResult`` record with its
success-implies-content invariant, the content-vacuity rule that hands a 200
carrying nothing a failure status instead, the one token an outcome is REPORTED
as (``ok`` or the verbatim status, shared by the diagnostics map and the run-log
marker), and the two pure reductions over a finished result list (the
provider-diagnostics source map and the compact failure line the section
renders).

Split out of ``research.resolution_source`` so the status contract — the thing
the section renderer, the diagnostics block and the run-log telemetry all key on
— can be read and tested without the network layer around it. Every status
string here is a telemetry token: changing one is a breaking change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

from metaculus_bot.research.http_fetch import MAX_UNDECODABLE_CHAR_RATIO, DatawrapperChartRef

# `stale_data` is Tier-2-only: the Datawrapper hop reached a dataset whose
# Last-Modified is outside the freshness window — older than the bound, missing,
# unparseable, or implausibly far in the FUTURE — withheld rather than served as live.
#
# `empty_body` is the 200-with-nothing-in-it case: a body that is empty or
# whitespace-only carries no information, so calling it `success` published an
# empty "primary grading evidence" section, suppressed the all-failed notice for
# every sibling URL, and reported `ok` to provider diagnostics. It is a FAILURE
# status for the same reason the HTML branch treats an empty extraction as
# `js_wall`: content is what makes a fetch a success.
#
# `no_resolving_content` is the 200 whose extracted text carries no content worth
# grading against — page chrome, and nothing else. Two ways a page earns it, told
# apart by `FetchResult.status_reason`:
#
#   `embed_shell` — the page's numbers exist but live inside a third-party data
#   embed (Infogram / Flourish / Tableau) our extractor cannot read, and what
#   trafilatura returned is the chrome around it. Shipped for qids 44554/44556,
#   whose tracker page rendered 2.9k chars of forecast background as "primary
#   grading evidence" with zero polling numbers in it.
#
#   `thin_page` — the extraction is under the same shell floor with no named embed
#   provider anywhere in the raw HTML. The 2026-09-01 round found five content-free
#   `success` renders and the embed-gated verdict reached none of them: q45088's
#   127-char SPA tab list (`Nationwide / Midwest / Northeast / …`) and q45215's
#   385 chars of Kazakh region names both published under the "primary grading
#   evidence" caption with nothing resolving in them. The floor is what the
#   verdict rests on either way, so gating it on a named provider was withholding
#   one shape of chrome and publishing the other.
#
# Distinct from `empty_body` (nothing was there at all) and from `js_wall` (the
# page needs JS to assemble ANY content, i.e. under the much lower JS-wall floor):
# here the page answered with prose-shaped chrome, which is why — like `blocked` /
# `js_wall` — it is a Tier-2 ESCALATION SEAM rather than a refusal.
#
# `unreadable_document` is a document we DID read the bytes of and could not turn
# into text: a scan with no text layer, an encrypted file, or a malformed one
# (`FetchResult.status_reason` says which). Deliberately NOT `unsupported_type`,
# which keeps meaning "a content type we do not read at all" — the two answer
# different questions, and a paid document read is only ever worth spending on
# this one.
# `ungrounded` is the PAID reader answering without having retrieved anything: Gemini's
# url_context tool reported zero successful retrievals, so whatever text came back is recall
# rather than a read of the page, and it is DISCARDED rather than rendered. Its own token because
# it is the one failure that cost money, and because it says something no other status does —
# the host answered a third-party fetcher's request with nothing while refusing ours. Mirrors the
# grounded-chunk floor `gemini_search` applies and the identical guard on gap-fill v2's
# `read_document`; the receipt for why is Q38195 (2026-07-19), where 30 search queries and 0
# grounding chunks produced a confident fabricated table with fake `[primary]` tags.
FetchStatus = Literal[
    "success",
    "blocked",
    "not_found",
    "js_wall",
    "error",
    "unsupported_type",
    "ssrf_blocked",
    "stale_data",
    "empty_body",
    "no_resolving_content",
    "unreadable_document",
    "ungrounded",
]

# Which rule produced a status that has more than one rule behind it. A telemetry token
# like every `FetchStatus`: it rides the `RESOLUTION_SOURCE_FETCH` marker as `reason=`,
# which is what separates the embed-gated population (queryable since 2026-08) from the
# generalised thin-page one, so a later "how often does the floor withhold a page
# nothing else would have caught?" cut is a query rather than a re-derivation.
#
# `embed_shell` / `thin_page` belong to `no_resolving_content`; `no_text_layer` /
# `encrypted` / `malformed` to `unreadable_document`, where the split is what says
# whether a paid document read could ever help (only `no_text_layer` — the other two
# are bytes no reader gets text out of).
#
# The document rung adds three. `no_matching_passage` is a `success` we DID read whose BM25
# selection matched no query term: the digest renders its header, its outline and the "no
# passage matched" sentence, which in the run log and the archive was byte-identical to a
# document that handed the forecasters the resolving paragraph — on the one surface whose
# stated contract is that `success` means CONTENT. `budget_skipped` and `parse_contention`
# belong to the `unsupported_type` a held-but-unparsed document earns, and say which rule
# declined: the question ran out of wall, or every parse slot was taken. Without them a
# skipped document is indistinguishable from a body that was never a document at all.
#
# `renderer_unavailable` is the browser rung declining before it rendered anything: Playwright
# missing or broken, a host that will not pin to a public IP, a browser error, or a URL a
# browser already read to nothing this run. It rides a rung attempt's `skipped_reason` rather
# than a result's `status_reason` — nothing was rendered, so nothing about the page changed —
# and it is here so the whole reason vocabulary has one home.
FetchStatusReason = Literal[
    "embed_shell",
    "thin_page",
    "no_text_layer",
    "encrypted",
    "malformed",
    "no_matching_passage",
    "budget_skipped",
    "parse_contention",
    "renderer_unavailable",
]

# Which rung of the escalation ladder produced a result. The vocabulary is pinned to the
# `route=` group of the `resolution_source_fetch` marker spec
# (`scripts/telemetry/markers.py`) — adding a token is fine, renaming one is a breaking
# telemetry change. `direct` is the plain fetch and is the only value that does NOT ride
# the marker, so every line the archive already holds stays byte-identical.
FetchRoute = Literal[
    "direct",
    "meta_refresh",
    "impersonate",
    "pdf_local",
    "derived_api",
    "rendered",
    "wayback",
    "url_context",
]

# HTTP status -> FetchStatus for non-OK terminal responses — the ONE table both the
# Tier-1 page fetch (_resolution_status_outcome) and the Tier-2 Datawrapper CDN hop
# (_datawrapper_hop_status) read, so a future addition (451, 503, ...) cannot land on
# one surface only. 3xx is deliberately absent: Tier-1 vets redirects upstream via
# _resolution_redirect_outcome, and the CDN hop intentionally maps a 3xx to `error`.
_NON_OK_FETCH_STATUS: dict[int, FetchStatus] = {
    403: "blocked",
    406: "blocked",
    429: "blocked",
    404: "not_found",
    410: "not_found",
}


@dataclass
class RungAttempt:
    """One escalation-rung attempt on one URL: what triggered it, and what it cost.

    Carried on the result rather than logged where it happens, because the
    ``RESOLUTION_SOURCE_ESCALATION`` marker names the question and the question id only
    exists at the provider's per-question aggregation point (same reason the fetch
    marker is emitted there).

    ``from_status`` is the status the DIRECT route would have returned — the trigger
    population — so the marker answers "how often does this rung fire" without a join.
    ``url`` is the URL the rung was invoked ON, which for a meta-refresh hop is the stub
    rather than the page it led to: the stub is the URL the QUESTION cited and the one
    every earlier fetch record is filed under, so it is what a "which cited sources need
    this rung" cut has to key on.
    ``wall_s`` is None until the attempt finishes: a rung whose cost is a local parse
    knows it immediately, while the meta-refresh hop is only over once the followed
    request comes back, which happens a layer above where the attempt is created.

    ``skipped_reason`` marks an attempt that never ran (no wall budget left). Those are
    NOT escalation lines — the marker means "a rung fired" — so they ride the provider's
    ``details["counts"]`` instead, where a zero renders nothing but survives into the
    archive.
    """

    rung: FetchRoute
    from_status: FetchStatus
    url: str
    started_at: float
    wall_s: float | None = None
    skipped_reason: str = ""

    def finish(self, now: float) -> None:
        """Stamp the elapsed wall-clock, unless the rung already measured its own."""
        if self.wall_s is None:
            self.wall_s = max(0.0, now - self.started_at)


@dataclass
class FetchResult:
    url: str
    status: FetchStatus
    text: str  # extracted + truncated; "" unless status == "success"
    http_status: int | None
    content_type: str | None
    # Charts seen in a fetched page's raw HTML (set on Tier-1 HTML results,
    # including js_wall pages — a JS-walled page still exposes its embeds).
    datawrapper_charts: list[DatawrapperChartRef] = field(default_factory=list)
    # Data-embed providers the page references that we have NO route to
    # (`unreadable_data_embed_providers`). Set on Tier-1 HTML results. Drives both
    # the `no_resolving_content` verdict and, on a page that DID carry prose, the
    # disclosure appended to its rendered text.
    unreadable_embeds: list[str] = field(default_factory=list)
    # Which rule produced the status, where the status alone is ambiguous. Set on
    # `no_resolving_content` (`embed_shell` / `thin_page`) and `unreadable_document`
    # (`no_text_layer` / `encrypted` / `malformed`); None everywhere else.
    status_reason: FetchStatusReason | None = None
    # Which rung of the ladder produced this result, and the per-rung attempts behind
    # it. `direct` plus an empty list is the plain fetch, which is the overwhelming
    # majority and renders no extra telemetry at all.
    route: FetchRoute = "direct"
    rung_attempts: list[RungAttempt] = field(default_factory=list)
    # Provenance for Tier-2 dataset results (None on ordinary page fetches).
    chart_id: str | None = None
    chart_title: str | None = None
    parent_url: str | None = None
    data_last_modified: str | None = None  # ISO-8601; None when the header was missing/unparseable

    def __post_init__(self) -> None:
        """Enforce the ``text`` invariant the field comment states.

        A `success` carrying blank text is the defect this guard exists for: the
        JSON/text/CSV branch used to ship an empty 200 body as `success`, which
        rendered an empty section under the "primary grading evidence" caveat,
        suppressed the all-failed notice, and told provider diagnostics `ok`.
        Every construction site now decides vacuity BEFORE it picks a status
        (:func:`vacuous_body_status`), so a blank success can only come from a
        future edit — and it should crash the provider (the orchestrator turns a
        provider exception into one `errored` provider result) rather than quietly
        publish a hole in the grading evidence.
        """
        if self.status == "success" and not self.text.strip():
            raise ValueError(f"FetchResult(status='success') with blank text for {self.url}")


# Delimiters a Datawrapper "Get the data" export uses. Checked on the HEADER
# line only: a dataset's first row names its columns, so a header with no
# delimiter at all is not the CSV this route is supposed to serve.
_CSV_DELIMITERS: tuple[str, ...] = (",", ";", "\t", "|")


def looks_like_csv_rows(text: str) -> bool:
    """True when ``text`` is shaped like a dataset: a delimited header + ≥1 row.

    The precondition for asserting a dataset is LIVE. The Tier-2 lead stamps
    ``Dataset published <ts>`` off the ``Last-Modified`` header alone, so without
    this an empty or soft-404 CDN body renders under an authoritative freshness
    claim — the same shape as a venue's manufactured $0.50 price.

    Markup is rejected outright (a body whose first non-blank character is ``<``
    is an error page, not a dataset) because an HTML error page can easily carry
    a comma somewhere and pass the delimiter test.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    header = lines[0].lstrip()
    if header.startswith("<"):
        return False
    return any(delimiter in header for delimiter in _CSV_DELIMITERS)


def vacuous_body_status(text: str, undecodable_ratio: float, *, require_csv_rows: bool) -> FetchStatus | None:
    """The failure status a 200 body earns for carrying no usable content, else None.

    The one place "does this body carry information?" is decided, so every
    ``FetchResult(status="success")`` on a raw-body branch is predicated on
    content rather than on the status line. Three ways a 200 carries nothing:

    - it could not be DECODED (``undecodable_ratio`` above the bound) — mojibake
      like ``0�.�4�2�`` type-checks as text and rendered as grading evidence;
    - it is empty or whitespace-only;
    - (Tier-2 datasets only) it is not row-shaped, so nothing may claim it is the
      chart's live series.
    """
    if undecodable_ratio > MAX_UNDECODABLE_CHAR_RATIO:
        return "unsupported_type"
    if not text.strip():
        return "empty_body"
    if require_csv_rows and not looks_like_csv_rows(text):
        return "unsupported_type"
    return None


def fetch_outcome_token(result: FetchResult) -> str:
    """The telemetry token for one fetch: ``"ok"`` for a success, else the verbatim status.

    Shared by the provider-diagnostics source map below and the per-URL
    ``RESOLUTION_SOURCE_FETCH`` run-log marker, so the two can never disagree about
    what a fetch outcome is called. ``"ok"`` (not ``"success"``) because the
    diagnostics formatter recognizes that prefix as "this source contributed".
    """
    return "ok" if result.status == "success" else result.status


def _fetch_result_sources(results: list[FetchResult]) -> dict[str, str]:
    """Per-URL outcome map for provider diagnostics: ``{domain: "ok" | <FetchStatus>}``.

    A fetched URL normalizes to ``"ok"`` (the shared "contributed" token the
    diagnostics formatter recognizes); every other ``FetchStatus``
    (``blocked`` / ``js_wall`` / ``not_found`` / ``error`` / ``unsupported_type`` /
    ``ssrf_blocked`` / ``empty_body`` / ``no_resolving_content`` /
    ``unreadable_document``) is kept verbatim
    as the loss token so the reason survives into the ``lost=`` segment. Duplicate
    domains are disambiguated with a ``#N`` suffix so no per-URL outcome is silently
    overwritten.

    Tier-2 dataset results are keyed ``datawrapper:<chart_id>`` — they are hop
    artifacts, not cited sources, and every dataset URL shares one CDN netloc, so
    domain keys would collapse them into ``static.dwcdn.net#N`` noise. Their
    ``stale_data`` verdict maps to the benign ``"none"``: that is the freshness
    guard REFUSING to serve months-old data as live, i.e. the feature working as
    designed, and reporting it in ``lost=`` would dress a by-design withhold as a
    lost cited source. A genuinely failed hop (``error``/``blocked``/``not_found``/
    ``empty_body``/``unsupported_type``) keeps its verbatim loss token — that is
    real signal about the CDN, and it is the reason the content check runs BEFORE
    the freshness guard: an empty CDN body must not borrow ``stale_data``'s
    by-design amnesty.
    """
    sources: dict[str, str] = {}
    for r in results:
        if r.chart_id is not None:
            key = f"datawrapper:{r.chart_id}"
        else:
            try:
                key = urlparse(r.url).netloc or r.url
            except ValueError:
                key = r.url
        if key in sources:
            n = 2
            while f"{key}#{n}" in sources:
                n += 1
            key = f"{key}#{n}"
        if r.chart_id is not None and r.status == "stale_data":
            sources[key] = "none"
        else:
            sources[key] = fetch_outcome_token(r)
    return sources


def _render_fetch_failures(failures: list[FetchResult]) -> str:
    """Render failed fetches as a compact ``"domain: status, domain: status"`` list."""
    parts: list[str] = []
    for r in failures:
        try:
            domain = urlparse(r.url).netloc or r.url
        except ValueError:
            domain = r.url
        parts.append(f"{domain}: {r.status}")
    return ", ".join(parts)
