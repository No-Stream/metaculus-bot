from datetime import datetime, timezone

from metaculus_bot.research_providers import (
    _dedup_articles_by_url,
    _format_asknews_dual_sections,
    _normalize_url_for_dedup,
)


class DummyArticle:
    def __init__(self, url: str, title: str = "", pub_date: datetime | None = None) -> None:
        self.article_url = url
        self.eng_title = title
        self.pub_date = pub_date or datetime.now(tz=timezone.utc)
        self.summary = f"Summary for {title}"
        self.language = "en"
        self.source_id = "test-source"


def test_normalize_url_basic_tracking_and_slashes() -> None:
    u1 = "https://EXAMPLE.com/path/?utm_source=x&utm_medium=y&gclid=123&a=1&b=2#frag"
    u2 = "https://example.com/path?a=1&b=2/"
    n1 = _normalize_url_for_dedup(u1)
    n2 = _normalize_url_for_dedup(u2)
    assert n1 == n2


def test_normalize_url_mobile_and_amp() -> None:
    u1 = "https://m.news.com/article/amp"
    u2 = "https://news.com/article"
    assert _normalize_url_for_dedup(u1) == _normalize_url_for_dedup(u2)


def test_dedup_articles_by_url_preserves_order_and_keeps_non_url_items() -> None:
    items = [
        DummyArticle("https://site.com/a?utm_campaign=z"),
        DummyArticle("https://site.com/a/"),  # duplicate of first after normalization
        {"article_url": "https://m.other.com/b/amp"},
        DummyArticle("https://other.com/b"),  # duplicate of previous after normalization
        {"no_url": True},  # should be kept
    ]
    deduped = _dedup_articles_by_url(items)
    # Expect first occurrence of each URL + the item without URL
    assert len(deduped) == 3
    assert isinstance(deduped[0], DummyArticle)
    assert isinstance(deduped[1], dict)
    assert isinstance(deduped[2], dict) and deduped[2].get("no_url")


# ---------------------------------------------------------------------------
# Tests for dual research stream formatting
# ---------------------------------------------------------------------------


class TestFormatAsknewsDualSections:
    def test_both_sections_present_with_correct_headers(self) -> None:
        hist = [DummyArticle("https://example.com/old", title="Old Event")]
        hot = [DummyArticle("https://example.com/new", title="New Event")]

        result = _format_asknews_dual_sections(hot_articles=hot, historical_articles=hist)

        assert "## Historical Context & Background" in result
        assert "## Recent Developments & Current News" in result
        assert result.index("Historical Context") < result.index("Recent Developments")

    def test_historical_only(self) -> None:
        hist = [DummyArticle("https://example.com/old", title="Old Event")]

        result = _format_asknews_dual_sections(hot_articles=[], historical_articles=hist)

        assert "## Historical Context & Background" in result
        assert "## Recent Developments & Current News" not in result
        assert "Old Event" in result

    def test_hot_only(self) -> None:
        hot = [DummyArticle("https://example.com/new", title="New Event")]

        result = _format_asknews_dual_sections(hot_articles=hot, historical_articles=[])

        assert "## Historical Context & Background" not in result
        assert "## Recent Developments & Current News" in result
        assert "New Event" in result

    def test_empty_lists_returns_no_articles_message(self) -> None:
        result = _format_asknews_dual_sections(hot_articles=[], historical_articles=[])

        assert "No articles were found" in result

    def test_cross_set_dedup_removes_hot_duplicates_of_historical(self) -> None:
        shared_url = "https://example.com/shared-article"
        hist = [DummyArticle(shared_url, title="Historical Version")]
        hot = [
            DummyArticle(shared_url, title="Hot Version"),
            DummyArticle("https://example.com/unique-hot", title="Unique Hot"),
        ]

        result = _format_asknews_dual_sections(hot_articles=hot, historical_articles=hist)

        assert "Historical Version" in result
        assert "Hot Version" not in result
        assert "Unique Hot" in result

    def test_within_set_dedup(self) -> None:
        hist = [
            DummyArticle("https://example.com/a", title="First A"),
            DummyArticle("https://example.com/a/", title="Duplicate A"),
        ]
        hot = [
            DummyArticle("https://example.com/b", title="First B"),
            DummyArticle("https://m.example.com/b", title="Mobile Duplicate B"),
        ]

        result = _format_asknews_dual_sections(hot_articles=hot, historical_articles=hist)

        assert "First A" in result
        assert "Duplicate A" not in result
        assert "First B" in result
        assert "Mobile Duplicate B" not in result

    def test_articles_sorted_by_date_within_sections(self) -> None:
        older = datetime(2025, 1, 1, tzinfo=timezone.utc)
        newer = datetime(2025, 6, 1, tzinfo=timezone.utc)

        hist = [
            DummyArticle("https://example.com/old", title="Older Hist", pub_date=older),
            DummyArticle("https://example.com/new", title="Newer Hist", pub_date=newer),
        ]

        result = _format_asknews_dual_sections(hot_articles=[], historical_articles=hist)

        newer_pos = result.index("Newer Hist")
        older_pos = result.index("Older Hist")
        assert newer_pos < older_pos, "Newer articles should appear first (reverse chronological)"

    def test_article_fields_are_formatted(self) -> None:
        article = DummyArticle("https://example.com/story", title="Big Story")

        result = _format_asknews_dual_sections(hot_articles=[article], historical_articles=[])

        assert "**Big Story**" in result
        assert "Summary for Big Story" in result
        assert "Original language: en" in result
        assert "Source:[test-source](https://example.com/story)" in result
