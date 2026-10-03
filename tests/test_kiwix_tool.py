"""Tests for tools/kiwix_tool.py: parsing kiwix-serve pages for Open WebUI."""

import functools
import threading
from collections.abc import Iterator
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import kiwix_tool as kt
import pytest

ARTICLE = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Photosynthesis</title></head>
<body><nav>Main menu</nav><div id="mw-content-text">
<div class="hatnote">For other uses, see Photosynthesis (disambiguation).</div>
<table class="infobox"><tr><th>Process</th><td>Light — chemical energy</td></tr></table>
<p>Photosynthesis (/ˌfoʊtəˈsɪnθəsɪs/) converts light energy<sup class="reference">[1]</sup>
into <a href="./Chemical_energy">chemical energy</a>.</p>
<div class="mw-heading mw-heading2"><h2 id="Overview">Overview</h2></div>
<ul><li>Oxygenic</li><li>Anoxygenic</li></ul>
<div class="mw-heading mw-heading2"><h2 id="References">References</h2></div>
<ol class="references"><li>A cited source</li></ol>
<p>Text after the references</p>
</div></body></html>"""


def _search_page(count: int) -> bytes:
    """Build a kiwix-serve search page with ``count`` results and its page chrome."""
    items = "".join(
        f'<li><a href="/content/wiki/Topic_{i}">\n  Topic {i}\n</a><cite>snippet</cite></li>'
        for i in range(count)
    )
    return (
        '<html><body><a href="/skin/x">Home</a><div class="results"><ul>'
        f'{items}</ul></div><div class="footer"><a href="?start=10">Next</a></div></body></html>'
    ).encode()


def test_search_results_lists_only_result_links_up_to_limit():
    """Search parsing keeps result links in order, trims titles, and stops at ten."""
    results = kt._search_results(_search_page(12))

    assert len(results) == 10
    assert results[0] == ("Topic 0", "/content/wiki/Topic_0")
    assert results[-1] == ("Topic 9", "/content/wiki/Topic_9")


def test_article_text_keeps_prose_and_drops_noise():
    """Article text has headings and unicode intact, without citations or reference lists."""
    text = kt._article_text(ARTICLE.encode())

    assert text.splitlines() == [
        "# Photosynthesis",
        "Process: Light — chemical energy",
        "Photosynthesis (/ˌfoʊtəˈsɪnθəsɪs/) converts light energy into chemical energy.",
        "## Overview",
        "Oxygenic",
        "Anoxygenic",
    ]


class _KiwixHandler(SimpleHTTPRequestHandler):
    """Serve files as kiwix-serve serves articles: text/html with no charset."""

    extensions_map = {"": "text/html"}

    def log_message(self, *_: object) -> None:
        """Keep test output quiet."""


@pytest.fixture
def tools(tmp_path: Path) -> Iterator[kt.Tools]:
    """Return a Tools instance pointed at a local fake kiwix-serve."""
    article = tmp_path / "content" / "wiki" / "Photosynthesis"
    article.parent.mkdir(parents=True)
    article.write_bytes(ARTICLE.encode())
    handler = functools.partial(_KiwixHandler, directory=str(tmp_path))
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    instance = kt.Tools()
    instance.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    yield instance
    server.shutdown()


def test_read_articles_decodes_utf8_without_a_charset_header(tools):
    """An article served as bare text/html keeps its non-ASCII characters."""
    result = tools.read_articles(["content/wiki/Photosynthesis"])

    assert result.startswith("--- START OF ARTICLE: /content/wiki/Photosynthesis ---\n")
    assert "Light — chemical energy" in result
    assert "/ˌfoʊtəˈsɪnθəsɪs/" in result


def test_read_articles_truncates_long_articles_and_reports_failures(tools, monkeypatch):
    """Long text is cut with a marker, and a missing article yields a failure line."""
    monkeypatch.setattr(kt, "_MAX_ARTICLE_CHARS", 20)

    result = tools.read_articles(["/content/wiki/Photosynthesis", "/content/wiki/Missing"])

    assert "# Photosynthesis\nPro\n[article truncated]" in result
    assert "--- FAILED TO FETCH: /content/wiki/Missing" in result
