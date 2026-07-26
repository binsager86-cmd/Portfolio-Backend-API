from app.api.v1 import news


class ExistingArticle:
    def __init__(self):
        self.title = "Old title"
        self.summary = "Old summary"
        self.category = "company_announcement"
        self.url = "https://ifsahdocs.boursakuwait.com.kw/DiscAssets/2026_17577/Boursa_HTML_en.html"
        self.related_symbols = None
        self.impact = "informational"
        self.language = "en"
        self.attachments_json = None
        self.fetched_at = None


def test_rss_item_maps_direct_document_url_and_category(monkeypatch):
    monkeypatch.setattr(
        news,
        "_resolve_boursa_document_url",
        lambda url: "https://ifsahdocs.boursakuwait.com.kw/DiscAssets/2026_17577/disclosure.pdf",
    )
    xml = """<?xml version="1.0" encoding="utf-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>AINS - Disclosure of Material Information - contribution to the Kuwait Emergency Response Fund</title>
        <type>145</type>
        <description></description>
        <link>https://www.boursakuwait.com.kw/en/news/view#BK51841</link>
        <url>https://ifsahdocs.boursakuwait.com.kw/DiscAssets/2026_17577/Boursa_HTML_en.html</url>
        <pubDate>Sun, 26 Jul 2026 14:48:13 +0300</pubDate>
        <security>303</security>
      </item>
    </channel></rss>"""

    raw_items = news._parse_rss_xml(xml)
    mapped = news._map_item(raw_items[0], "E")

    assert mapped["id"] == "BK51841"
    assert mapped["category"] == "regulatory"
    assert mapped["relatedSymbols"] == ["AINS"]
    assert mapped["url"].endswith("/disclosure.pdf")
    assert mapped["attachments"] == [{"type": "pdf", "url": mapped["url"]}]


def test_rss_type_143_is_financial(monkeypatch):
    monkeypatch.setattr(news, "_resolve_boursa_document_url", lambda url: url)
    item = {
        "__source": "rss",
        "title": "BOURSA - Boursa Kuwait's BoD Meeting for approving the Financial Information",
        "type": "143",
        "description": "",
        "link": "https://www.boursakuwait.com.kw/en/news/view#BK51833",
        "url": "https://ifsahdocs.boursakuwait.com.kw/DiscAssets/2026_17572/Boursa_HTML_en.html",
        "pubDate": "Sun, 26 Jul 2026 13:43:56 +0300",
    }

    mapped = news._map_item(item, "E")

    assert mapped["category"] == "financial"
    assert mapped["impact"] == "medium"
    assert mapped["relatedSymbols"] == ["BOURSA"]


def test_existing_article_refreshes_direct_document_metadata():
    row = ExistingArticle()
    changed = news._update_existing_article(row, {
        "title": "AINS - Disclosure of Material Information",
        "summary": "Updated summary",
        "category": "regulatory",
        "url": "https://ifsahdocs.boursakuwait.com.kw/DiscAssets/2026_17577/disclosure.pdf",
        "relatedSymbols": ["AINS"],
        "impact": "medium",
        "language": "en",
        "attachments": [{"type": "pdf", "url": "https://ifsahdocs.boursakuwait.com.kw/DiscAssets/2026_17577/disclosure.pdf"}],
    })

    assert changed is True
    assert row.category == "regulatory"
    assert row.url.endswith("/disclosure.pdf")
    assert row.related_symbols == "AINS"
    assert row.attachments_json is not None
    assert row.fetched_at is not None