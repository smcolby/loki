"""Open WebUI tool definition for searching the local Kiwix server.

Paste the contents of this file into the Open WebUI tool editor
(Workspace → Tools → +) to expose offline Kiwix archives to the LLM.

The tool communicates with the Kiwix service over the Docker internal network
using the ``kiwix-serve`` hostname and its fixed internal container port (8080),
which is independent of the host port configured in config.yaml.

The ``Tools`` methods use reST ``:param:`` docstrings because Open WebUI builds
each parameter's description for the model from those lines.

Dependencies
------------
This file requires ``beautifulsoup4`` and ``requests``, both of which ship in
the Open WebUI image.
"""

import re
import urllib.parse

import requests
from bs4 import BeautifulSoup, Tag

_MAX_SEARCH_RESULTS = 10
_MAX_ARTICLES_PER_CALL = 3
_MAX_ARTICLE_CHARS = 30000

# Elements that carry no article prose: citation markers, reference lists, navigation
_NOISE_SELECTORS = (
    "script",
    "style",
    "sup.reference",
    ".mw-cite-backlink",
    ".mw-editsection",
    ".reflist",
    ".references",
    ".refbegin",
    ".navbox",
    ".navbox-styles",
    ".hatnote",
    ".noprint",
)

# Trailing sections that list sources and links rather than content
_END_SECTIONS = re.compile(
    r"^#+ (See also|Notes|References|Citations|Sources|Bibliography|Further reading"
    r"|External links)$"
)

_BLOCK_TAGS = ("p", "li", "dd", "dt", "tr", "caption", "h2", "h3", "h4", "h5", "h6")


def _search_results(html: bytes) -> list[tuple[str, str]]:
    """Return ``(title, path)`` pairs from a kiwix-serve search results page.

    Parameters
    ----------
    html : bytes
        Raw search page, decoded through its own charset declaration.

    Returns
    -------
    list of tuple of str
        Up to ``_MAX_SEARCH_RESULTS`` titles with their article paths.
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for link in soup.select(".results li > a[href]"):
        title = " ".join(link.get_text().split())
        href = link["href"]
        if title and isinstance(href, str):
            results.append((title, href))
    return results[:_MAX_SEARCH_RESULTS]


def _article_text(html: bytes) -> str:
    """Extract an article's prose as plain text with Markdown-style headings.

    Citation markers, reference lists, and navigation boxes are dropped, and the
    text stops at the first trailing section such as "References".

    Parameters
    ----------
    html : bytes
        Raw article page, decoded through its own charset declaration.

    Returns
    -------
    str
        Article text, one block per line.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    content = soup.find(id="mw-content-text")
    body = content if isinstance(content, Tag) else soup
    for element in body.select(", ".join(_NOISE_SELECTORS)):
        element.decompose()

    # Turn source line breaks into spaces so only block ends below start new lines
    for string in body.find_all(string=True):
        if "\n" in string:
            string.replace_with(string.replace("\n", " "))

    # Mark headings and end every block element with a line break
    for heading in body.find_all(["h2", "h3", "h4", "h5", "h6"]):
        if isinstance(heading, Tag) and heading.name:
            heading.insert(0, "#" * int(heading.name[1]) + " ")
    for block in body.find_all(_BLOCK_TAGS):
        if isinstance(block, Tag):
            block.append("\n")

    # Separate table cells so infobox labels do not run into their values
    for cell in body.find_all(["th", "td"]):
        if isinstance(cell, Tag):
            cell.append(": " if cell.name == "th" and cell.find_next_sibling("td") else " ")

    lines = []
    for raw in body.get_text().splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        if _END_SECTIONS.match(line):
            break
        lines.append(line)
    text = "\n".join(lines)
    return f"# {title}\n{text}" if title else text


class Tools:
    """Open WebUI tool for querying the local Kiwix offline knowledge server.

    Communicates with the kiwix-serve container over the Docker internal
    network. The base URL uses the Docker service name and fixed internal
    container port, independent of the host port set in config.yaml.
    """

    def __init__(self) -> None:
        self.base_url = "http://kiwix-serve:8080"

    def search_article_titles(self, query: str) -> str:
        """Search the offline Wikipedia archive and list matching article titles and paths.

        Call this first, then pass the most relevant paths to read_articles.
        If the exact topic is not listed, pick the broadest relevant parent article.

        :param query: The specific, disambiguated topic or entity to search for.
        :return: Matching titles and paths, or an error message.
        """
        params = urllib.parse.urlencode({"pattern": query, "pageLength": _MAX_SEARCH_RESULTS})
        try:
            response = requests.get(f"{self.base_url}/search?{params}", timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            return f"Search failed: {exc}"

        results = _search_results(response.content)
        if not results:
            return f"No articles found for {query!r}. Try a broader or differently worded query."
        listing = "\n".join(f"Title: {title} | Path: {path}" for title, path in results)
        return (
            "Found the following articles. Pass the exact Path of up to "
            f"{_MAX_ARTICLES_PER_CALL} of them to read_articles:\n{listing}"
        )

    def read_articles(self, paths: list[str]) -> str:
        """Read the text of up to three articles from the offline Wikipedia archive.

        Pass the exact paths returned by search_article_titles, such as
        "/content/wikipedia_en_all_nopic_2025-12/Photosynthesis".

        :param paths: Exact article paths from search_article_titles.
        :return: Each article's text between start and end markers.
        """
        articles = []
        for path in paths[:_MAX_ARTICLES_PER_CALL]:
            path = path if path.startswith("/") else "/" + path
            try:
                response = requests.get(f"{self.base_url}{path}", timeout=15)
                response.raise_for_status()
            except requests.exceptions.RequestException as exc:
                articles.append(f"--- FAILED TO FETCH: {path} ({exc}) ---")
                continue

            text = _article_text(response.content)
            if len(text) > _MAX_ARTICLE_CHARS:
                text = text[:_MAX_ARTICLE_CHARS] + "\n[article truncated]"
            articles.append(f"--- START OF ARTICLE: {path} ---\n{text}\n--- END OF ARTICLE ---")
        return "\n\n".join(articles)
