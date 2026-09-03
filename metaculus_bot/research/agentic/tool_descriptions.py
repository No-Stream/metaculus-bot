"""Driver-facing text for the four agentic tools: descriptions + JSON parameter schemas.

This is what the gap-fill v2 driver LLM reads when it decides which tool to call, so the
wording is behavioral rather than documentation: the descriptions steer
search_news-vs-search_web routing, promise that ``fetch`` handles PDFs / JS pages / images
so the driver does not avoid a URL by format, and spell out the ``start_char`` pagination
contract. Split out of ``tools.py`` so the prose sits in one place and the module holding
the fetch ladder stays readable; ``build_gap_fill_tools`` (``tools.py``) pairs each
description + schema with its handler.
"""

from __future__ import annotations

SEARCH_NEWS_DESCRIPTION = (
    "Search recent and historical NEWS coverage (AskNews). Use for: events,\n"
    "announcements, things that happened, ongoing-situation updates. Query with a\n"
    "short natural-language phrase, not keywords. Returns a digest of matching\n"
    "articles with dates and URLs. Use search_web instead for: reports, datasets,\n"
    "official documents, niche/technical facts, or anything where the best source\n"
    "is not a news article.\n"
    'Example: search_news(query="Nauru parliament treaty ratification vote")'
)

SEARCH_WEB_DESCRIPTION = (
    "Semantic web search (Exa). Use for: official documents, datasets, reports,\n"
    "organizational pages, technical/niche facts, finding a primary source you\n"
    "believe exists. Returns results with URLs and relevant excerpts. Follow up\n"
    "promising results with fetch(url) — excerpts are often not enough to verify\n"
    "a claim. Use search_news instead for event/news coverage.\n"
    'Example: search_web(query="IAEA safeguards report Iran enrichment June 2026 pdf")'
)

FETCH_DESCRIPTION = (
    "Fetch a URL and return its main content as concise markdown, plus a list of\n"
    "outbound links. Handles ordinary pages, JavaScript-heavy pages, PDFs, and\n"
    "images automatically (the result's `method` field tells you how it was\n"
    "read) — do NOT avoid a URL because of its format. A PDF is read here, in\n"
    "full text, with `method=pdf_local`, and paginates exactly like a long HTML\n"
    "page. Content over the size cap is truncated, ending with `[truncated at N\n"
    "of M chars — call again with start_char=N]`; pass start_char to read the\n"
    "next window (continuations are served from cache — they are cheap and do not\n"
    "refetch). Links in the result are leads you can fetch next.\n"
    "Use read_document instead only when you need a specific question answered\n"
    "from inside a long/complex document.\n"
    "Do NOT fetch metaculus.com URLs — the question brief already reflects them.\n"
    'Example: fetch(url="https://www.ons.gov.uk/releases/gdpquarterly")\n'
    'Example: fetch(url="https://example.gov/long-report", start_char=12000)'
)

READ_DOCUMENT_DESCRIPTION = (
    "Ask a specific question of a specific document, and get back the passages of\n"
    "it that bear on your `ask`, quoted verbatim with page numbers where the\n"
    "document has pages (`method=digest_local`). Use it for targeted extraction\n"
    "from a long or complex document — a 200-page report, a filing, a big CSV —\n"
    "and for a URL where fetch returned status=blocked/js_wall/error and you\n"
    "still need the content: this tool fetches the page itself, and where it\n"
    "cannot, a model reads the URL for you instead (`method=document`, slower).\n"
    "A digest with no matching passage means the document does not discuss what\n"
    "you asked, not that the read failed — try a different ask or another source.\n"
    "Always pass a precise `ask`: it is what selects the passages.\n"
    'Example: read_document(url="https://example.gov/report-q2.pdf",\n'
    '                       ask="What is the reported unemployment rate for May 2026, and what revision to April is stated?")'
)

_SEARCH_NEWS_PARAMETERS = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}

_SEARCH_WEB_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "end_published_date": {"type": ["string", "null"]},
    },
    "required": ["query"],
    "additionalProperties": False,
}

_FETCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {"type": "string"},
        "start_char": {"type": "integer", "minimum": 0},
    },
    "required": ["url"],
    "additionalProperties": False,
}

_READ_DOCUMENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {"type": "string"},
        "ask": {"type": "string"},
    },
    "required": ["url", "ask"],
    "additionalProperties": False,
}
