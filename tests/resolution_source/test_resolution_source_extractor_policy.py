"""The extractor publication policy: which extraction of a 200 reaches forecasters, if any.

A page can clear the 400-char chrome floor with nothing but navigation, so the floor alone
is not the content check. ``_extract_page_text`` scores the default extraction with
``content_share`` (chars in table rows or in lines of at least the cutoff, over all chars),
re-extracts under ``favor_precision`` when the default is chrome-shaped, and publishes the
precision text only if it clears the floor and the same metric; otherwise the page is
withheld as ``thin_page`` so the rendered rung still fires. Calibrated 2026-09-03 on 118
bodies, receipt ``scratch/fetch_ladder_2026-09-03/chrome_calibration.md``.

Real trafilatura runs on the menu-tree and price-table fixtures. The congress.gov flip, where
the default extraction swaps a bill-status card for a member dropdown and precision restores
the card, depends on trafilatura's readability fallback and does not reproduce on a synthetic
page, so the rescue tests fake the extractor at the seam the module resolves it on.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from metaculus_bot.research import resolution_source
from metaculus_bot.research.provider_diagnostics import pop_provider_detail
from metaculus_bot.research.rendered_fetch import RenderedPage
from metaculus_bot.research.resolution_source import (
    _WAYBACK_TRIGGER_STATUSES,
    FetchContext,
    _fetch_one,
    _rendered_rung_applies,
    _rung_counts,
    content_share,
    looks_like_page_chrome,
    resolution_source_provider,
)
from metaculus_bot.research.wayback import wayback_snapshot_url
from tests.resolution_source_fakes import (
    _JS_SHELL,
    FakeResponse,
    FakeSession,
    _escape_config,
    _mock_question,
    _snapshot_url,
)

_URL = "https://stats.example.com/labour-force"
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _document(inner: str) -> bytes:
    return (
        "<!doctype html><html><head><title>Labour Force, Australia</title></head><body>"
        f"<nav>Home | Statistics</nav><main>{inner}</main><footer>&copy; 2026</footer></body></html>"
    ).encode()


def _release_listing() -> str:
    """The abs.gov.au shape: a release archive, one 48-char listing line per month."""
    items = "".join(f"<li>Labour Force, Australia, {month} 2026 Archive release</li>" for month in _MONTHS * 3)
    return f"<h1>Labour Force, Australia</h1><ul>{items}</ul>"


# Extracts to about 2,000 chars of 48-char lines: well over the floor and nothing but a menu.
_MENU_TREE = _document(_release_listing())

# The yahoo history shape: a table of short cells. Every row is a `| ... |` line to trafilatura.
_PRICE_TABLE = _document(
    "<h1>Historical prices</h1><table><tr><th>Date</th><th>Close</th><th>Volume</th></tr>"
    + "".join(
        f"<tr><td>2026-0{(i % 9) + 1}-{(i % 28) + 1:02d}</td><td>{7000 + i * 3}.{i % 10}1</td><td>{i % 5}</td></tr>"
        for i in range(45)
    )
    + "</table>"
)

# What the fake extractor hands back for the congress.gov shape.
_MEMBER_DROPDOWN = "Sponsors/Cosponsors\n" + "\n".join(f"Member {i:03d} [D-ST] (113th-119th)" for i in range(400))
_STATUS_CARD = (
    "H.R.2913 - Ukraine Support Act\n"
    "Latest Action: Senate - 07/20/2026 Read the second time and placed on Senate Legislative Calendar "
    "under General Orders. Calendar No. 412.\n"
    "Tracker: This bill has the status Passed House. Here are the steps for Status of Legislation: "
    "Introduced, Passed House, Passed Senate, To President, Became Law.\n"
    "Sponsor: Rep. Example, Someone [D-XX-1] (Introduced 04/16/2026). Committees: House - Foreign Affairs; "
    "Financial Services; Judiciary; Ways and Means; Oversight and Government Reform.\n"
    "Committee Meetings: 05/07/26 10:00AM. Committee Reports: H. Rept. 119-142. Roll Call Votes: 2 total."
)

# Well over the 400-char floor once extracted, so the rendered DOM tests the rung and the
# metric rather than the floor.
_RENDERED_PROSE = (
    "The seasonally adjusted unemployment rate held at 4.2 percent in July 2026, with employment "
    "rising by 24,500 persons and the participation rate steady at 67.0 percent. Monthly hours "
    "worked rose 0.3 percent, and underemployment eased to 6.1 percent. The trend series shows "
    "unemployment flat over the year to date after the two increases recorded in late 2025. "
    "Full-time employment rose by 31,200 and part-time employment fell by 6,700, while the "
    "employment-to-population ratio edged up to 64.3 percent. The number of unemployed persons "
    "fell by 4,100 to 604,900, and the youth unemployment rate was 9.6 percent, down 0.4 points "
    "on the month in seasonally adjusted terms."
)


def _fake_extractor(default: str | None, precision: str | None):
    def _extract(body: bytes | str, url: str, *, favor_precision: bool = False) -> str | None:
        del body, url
        return precision if favor_precision else default

    return _extract


class TestContentShare:
    """The metric, pinned on hand-computable strings."""

    def test_a_line_at_the_cutoff_counts_and_one_under_it_does_not(self):
        cutoff = resolution_source.RESOLUTION_SOURCE_CONTENT_LINE_MIN_CHARS
        assert content_share("x" * cutoff) == 1.0
        assert content_share("x" * (cutoff - 1)) == 0.0

    def test_a_table_row_counts_whatever_its_length(self):
        assert content_share("| 2026-01-01 | 7000.01 |") == 1.0
        assert content_share("   | a | b |") == 1.0

    def test_the_share_is_by_characters_not_by_lines(self):
        text = "a" * 60 + "\n" + "b" * 20 + "\n" + "c" * 20
        assert content_share(text) == pytest.approx(0.6)

    def test_blank_lines_and_surrounding_whitespace_are_dropped_first(self):
        text = "   \n  " + "a" * 60 + "   \n\n\t" + "b" * 40 + " \n"
        assert content_share(text) == pytest.approx(0.6)

    def test_empty_text_is_zero(self):
        assert content_share("") == 0.0
        assert content_share("  \n ") == 0.0

    def test_the_threshold_sits_between_the_calibrated_chrome_and_content_extremes(self):
        """The receipt's margin: navigation chrome tops out at 0.329 (the kasa homepage, an
        ambiguous menu plus ticker) and the thinnest labelled content is 0.431 (the
        wastewaterscan dashboard). A threshold outside that band re-litigates the study."""
        assert 0.329 < resolution_source.RESOLUTION_SOURCE_CONTENT_SHARE_MIN < 0.431


class TestChromeShapedPages:
    async def test_a_menu_tree_over_the_floor_is_withheld_as_a_thin_page(self):
        """P3-2 repro (c): abs.gov.au's release archive extracts 11,725 chars of listing lines
        and published as `success`. The extraction clears the floor by a wide margin, which
        is what makes this the metric's case and not the floor's."""
        extracted = resolution_source._extract_main_text(_MENU_TREE, _URL)
        assert extracted is not None
        assert not looks_like_page_chrome(extracted)
        assert content_share(extracted) < resolution_source.RESOLUTION_SOURCE_CONTENT_SHARE_MIN
        session = FakeSession({_URL: FakeResponse(200, body=_MENU_TREE)})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "no_resolving_content"
        assert result.status_reason == "thin_page"
        assert result.text == ""
        assert result.http_status == 200
        assert result.chrome_metric_withheld is True
        assert result.precision_rescued is False

    async def test_a_withheld_menu_tree_still_earns_the_rendered_rung(self, monkeypatch):
        """The withhold keeps `thin_page` on purpose: uk.finance.yahoo's direct body is a
        1,191-char menu plus one quote line, and its render is the full 23,991-char price
        table. A withhold under any other reason would strand the page one rung short."""
        calls: list[str] = []

        async def _render(
            url: str,
            *,
            memo_scope: str,
            host_gate,
            goto_timeout_ms: int,
            deadline_monotonic_s: float | None = None,
            harvest_json: bool = False,
        ) -> RenderedPage:
            del host_gate, goto_timeout_ms, harvest_json, memo_scope, deadline_monotonic_s
            calls.append(url)
            await asyncio.sleep(0)
            return RenderedPage(url=url, content_type="text/html", html=_document(f"<p>{_RENDERED_PROSE}</p>").decode())

        monkeypatch.setattr(resolution_source, "render_page", _render)
        direct_only = await resolution_source._classify_html_body(_MENU_TREE, _URL, "text/html", http_status=200)
        assert _rendered_rung_applies(direct_only.result)
        session = FakeSession({_URL: FakeResponse(200, body=_MENU_TREE)})

        result = await _fetch_one(session, _URL, {})

        assert calls == [_URL]
        assert result.status == "success"
        assert result.route == "rendered"
        assert "unemployment rate held at 4.2 percent" in result.text

    async def test_a_dropdown_is_rescued_by_the_precision_fallback(self, monkeypatch):
        """P3-2 repro (a): congress.gov's default extraction is a 54,393-char member dropdown
        with none of `Latest Action` / `Passed House` in it; under `favor_precision` the same
        bytes extract to the 2,411-char status card. The card clears the floor and the
        metric, so it is what publishes."""
        monkeypatch.setattr(resolution_source, "_extract_main_text", _fake_extractor(_MEMBER_DROPDOWN, _STATUS_CARD))
        assert not looks_like_page_chrome(_MEMBER_DROPDOWN)
        session = FakeSession({_URL: FakeResponse(200, body=_document("<p>irrelevant</p>"))})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        assert result.text.startswith("H.R.2913 - Ukraine Support Act")
        assert "Passed House" in result.text
        assert "Member 000" not in result.text
        assert result.precision_rescued is True
        assert result.chrome_metric_withheld is False

    @pytest.mark.parametrize(
        "precision_text",
        [
            pytest.param("\n".join(f"Menu item {i}" for i in range(60)), id="precision_is_still_chrome"),
            pytest.param(
                "H.R.2913 - Ukraine Support Act. Latest Action: Passed House.", id="precision_under_the_floor"
            ),
            pytest.param(None, id="precision_extracts_nothing"),
        ],
    )
    async def test_a_page_whose_precision_fallback_also_fails_is_withheld(self, monkeypatch, precision_text):
        """Both extractions have to fail before the page is withheld, and the withhold is the
        `thin_page` reason so the rendered rung still gets its turn."""
        monkeypatch.setattr(resolution_source, "_extract_main_text", _fake_extractor(_MEMBER_DROPDOWN, precision_text))
        session = FakeSession({_URL: FakeResponse(200, body=_document("<p>irrelevant</p>"))})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "no_resolving_content"
        assert result.status_reason == "thin_page"
        assert result.text == ""
        assert result.chrome_metric_withheld is True
        assert result.precision_rescued is False
        assert _rendered_rung_applies(result)

    async def test_an_article_of_long_lines_publishes_under_the_default_extractor(self, article_html, monkeypatch):
        """The common case pays for no second parse: the precision pass runs only once the
        default extraction has failed the metric."""
        calls: list[bool] = []
        real = resolution_source._extract_main_text

        def _spy(body: bytes | str, url: str, *, favor_precision: bool = False) -> str | None:
            calls.append(favor_precision)
            return real(body, url, favor_precision=favor_precision)

        monkeypatch.setattr(resolution_source, "_extract_main_text", _spy)
        session = FakeSession({_URL: FakeResponse(200, body=article_html)})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        assert "3.2 percent annual" in result.text
        assert calls == [False]
        assert result.precision_rescued is False
        assert result.chrome_metric_withheld is False

    async def test_a_table_of_short_cells_publishes(self):
        """A price-history table is rows of 10-char cells, i.e. short lines by any line-length
        rule. Table rows count as content whatever their length, which is what keeps the
        yahoo history tables and the tracxn funding tables on the published side."""
        session = FakeSession({_URL: FakeResponse(200, body=_PRICE_TABLE)})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        assert "| 2026-01-01 | 7000.01 | 0 |" in result.text
        assert result.precision_rescued is False

    async def test_chart_data_still_rescues_a_chrome_shaped_page(self):
        """Chart data counts as content on every page, and a menu tree carrying a Highcharts
        config is a page whose numbers we DID read. The chart block publishes ALONE: the menu
        text failed the line-shape metric, which is the policy's definition of chrome, and the
        same text is withheld one branch over when no chart block is present. Riding along it
        filled the per-URL cap with up to 6,000 chars of navigation under the primary grading
        evidence caption and ate the aggregate budget sibling pages could use. The withhold is
        still counted, so the archive sees the metric fire on a page that published."""
        config = {
            "xAxis": [{"categories": ["2024", "2025", "2026"]}],
            "series": [{"name": "Unemployed persons", "data": [612_300, 598_100, 604_900]}],
        }
        page = _document(
            f'{_release_listing()}<div class="charts-highchart" data-chart="{_escape_config(config)}"></div>'
        )
        session = FakeSession({_URL: FakeResponse(200, body=page)})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        assert "Unemployed persons" in result.text
        assert "604,900" in result.text or "604900" in result.text
        assert "Archive release" not in result.text
        assert result.chrome_metric_withheld is True
        assert _rung_counts([result])["chrome_metric_withholds"] == 1


class TestExtractorPolicyCounts:
    """Both decisions ride `details["counts"]`, additive, so no archived token moves."""

    async def test_a_withhold_and_a_rescue_each_move_their_own_key(self, monkeypatch):
        menu_url = "https://menu.example.com/releases"
        card_url = "https://bill.example.com/hr2913"
        session = FakeSession(
            {
                menu_url: FakeResponse(200, body=_MENU_TREE),
                card_url: FakeResponse(200, body=_document("<p>irrelevant</p>")),
            }
        )
        withheld = await _fetch_one(session, menu_url, {})
        monkeypatch.setattr(resolution_source, "_extract_main_text", _fake_extractor(_MEMBER_DROPDOWN, _STATUS_CARD))
        rescued = await _fetch_one(session, card_url, {})

        counts = _rung_counts([withheld, rescued])

        assert counts["chrome_metric_withholds"] == 1
        assert counts["precision_fallback_rescues"] == 1

    async def test_the_keys_reach_the_provider_detail(self, monkeypatch):
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession({_URL: FakeResponse(200, body=_MENU_TREE)})
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria=f"Resolves per {_URL} on release.")

        await resolution_source_provider(is_benchmarking=False)(q)

        detail = pop_provider_detail(q.id_of_question, "resolution_source")
        assert detail["sources"] == {"stats.example.com": "no_resolving_content"}
        assert detail["counts"]["chrome_metric_withholds"] == 1
        assert detail["counts"]["precision_fallback_rescues"] == 0


class TestThePolicyTravelsWithEveryRoute:
    """The policy is not the direct route's alone, and both other routes that reach it were
    unpinned: a rendered DOM re-enters the same classification, and a served archive capture has
    to carry the extractor's own decision out of it."""

    async def test_a_rendered_dom_that_is_another_menu_tree_is_withheld_too(self, monkeypatch):
        """The P3-3 shape, and the defect the policy was adopted for: the browser answered, and
        what it answered with was navigation.

        The rendered DOM goes through ``_classify_html_body`` exactly as a fetched body does, so
        the metric withholds it and the direct ``js_wall`` stands. Scoping the metric to the direct
        route would publish this menu under the primary-grading-evidence caption."""

        async def _render(
            url: str,
            *,
            memo_scope: str,
            host_gate,
            goto_timeout_ms: int,
            deadline_monotonic_s: float | None = None,
            harvest_json: bool = False,
        ) -> RenderedPage:
            del memo_scope, host_gate, goto_timeout_ms, deadline_monotonic_s, harvest_json
            await asyncio.sleep(0)
            return RenderedPage(url=url, content_type="text/html", html=_MENU_TREE.decode())

        monkeypatch.setattr(resolution_source, "render_page", _render)
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "js_wall"
        assert result.text == ""
        assert "Archive release" not in result.text
        # The rung fired and its own outcome is the withhold, which is what makes
        # `route=rendered status=js_wall` readable as "we tried the browser and this is the answer".
        assert result.route == "rendered"
        assert [(a.rung, a.outcome) for a in result.rung_attempts] == [("rendered", "no_resolving_content")]

    async def test_a_precision_rescue_inside_an_archived_capture_is_still_counted(self, monkeypatch):
        """A capture is classified by the same path a live page is, so the extractor's decision
        belongs to the result the rung serves. Without the carry, a rescue inside an archived body
        publishes its text while `precision_fallback_rescues` reads zero, which is the count the
        policy's own calibration is read back on."""
        monkeypatch.setattr(resolution_source, "_WAYBACK_TRIGGER_STATUSES", _WAYBACK_TRIGGER_STATUSES)
        monkeypatch.setattr(resolution_source, "_extract_main_text", _fake_extractor(_MEMBER_DROPDOWN, _STATUS_CARD))
        # A past year, so a rung reading its own clock instead of the fetch's would ask the archive
        # for a URL no handler serves (the same reason `TestWaybackRung._NOW` is dated back).
        now = datetime(2025, 9, 4, tzinfo=UTC)
        snapshot = _snapshot_url(_URL, captured=now - timedelta(days=3))
        session = FakeSession(
            {
                _URL: FakeResponse(403, body=b"denied", content_type="text/html"),
                wayback_snapshot_url(_URL, now=now): FakeResponse(302, headers={"Location": snapshot}),
                snapshot: FakeResponse(200, body=_document("<p>irrelevant</p>"), content_type="text/html"),
            }
        )

        result = await _fetch_one(session, _URL, {}, FetchContext(now=now))

        assert result.status == "success"
        assert result.route == "wayback"
        assert "Passed House" in result.text
        assert result.precision_rescued is True
        assert _rung_counts([result])["precision_fallback_rescues"] == 1
