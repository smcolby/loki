"""Open WebUI tool definition for searching the local Kiwix server.

Paste the contents of this file into the Open WebUI tool editor
(Admin Panel → Tools → +) to expose offline Kiwix archives to the LLM.

The tool communicates with the Kiwix service over the Docker internal network
using the ``kiwix-serve`` hostname and its fixed internal container port (8080),
which is independent of the host port configured in config.yaml.

Dependencies
------------
This file requires ``beautifulsoup4``, which must be available in the Open WebUI
Python environment. Install it via the Open WebUI requirements or by adding it to
the container image if it is not already present.
"""

import urllib.parse

import requests
try:
    from bs4 import BeautifulSoup
except ImportError as exc:
    raise ImportError(
        "The kiwix_tool requires 'beautifulsoup4'. "
        "Install it in the Open WebUI environment: pip install beautifulsoup4"
    ) from exc

_MAX_SEARCH_RESULTS = 10
_MAX_ARTICLES_PER_CALL = 3
_MAX_ARTICLE_CHARS = 8000


class Tools:
    """Open WebUI tool for querying the local Kiwix offline knowledge server.

    Communicates with the kiwix-serve container over the Docker internal
    network. The base URL uses the Docker service name and fixed internal
    container port, independent of the host port set in config.yaml.
    """

    def __init__(self) -> None:
        self.base_url = "http://kiwix-serve:8080"

    def search_article_titles(self, query: str) -> str:
        """Search the local Kiwix database and return matching article titles and paths.

        Use this tool first to find the correct specific article path before reading it.
        If the exact topic is not listed, select the broadest relevant parent article to read.

        Parameters
        ----------
        query : str
            The specific, disambiguated topic or entity to search for.

        Returns
        -------
        str
            A newline-separated list of up to ``_MAX_SEARCH_RESULTS`` matching
            titles and paths, or an error message.
        """
        safe_query = urllib.parse.quote(query)
        url = f"{self.base_url}/search?pattern={safe_query}"

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                title = a_tag.get_text(strip=True)

                # Exclude Kiwix UI chrome and pagination links from results
                if title and not href.startswith(("?", "/search", "/skin", "/catalog")):
                    results.append(f"Title: {title} | Path: {href}")

            if not results:
                print("No matches found in the local database.")
                return "No results found for that query."

            print("Successfully retrieved search result titles.")

            formatted_results = "\n".join(results[:_MAX_SEARCH_RESULTS])
            return f"Found the following articles. Use the read_articles tool with the exact Path to read up to {_MAX_ARTICLES_PER_CALL} of them:\n{formatted_results}"

        except requests.exceptions.RequestException as e:
            print(f"Failed to connect to the local server: {e}.")
            return "Search failed due to a network error."
    def read_articles(self, paths: list[str]) -> str:
        """Fetch the full text of up to ``_MAX_ARTICLES_PER_CALL`` articles from the local Kiwix database.

        Pass the exact paths returned by the search_article_titles tool.

        Parameters
        ----------
        paths : list of str
            URL paths to read (e.g. ``["/content/A/Article_1.html"]``).
            Only the first ``_MAX_ARTICLES_PER_CALL`` entries are processed.

        Returns
        -------
        str
            Concatenated article text, with each article wrapped in start/end markers.
        """
        # Respect the configured per-call article limit
        path_list = paths[:_MAX_ARTICLES_PER_CALL]
        combined_text = []

        for path in path_list:
            if not path.startswith("/"):
                path = "/" + path

            url = f"{self.base_url}{path}"

            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()

                clean_text = " ".join(soup.stripped_strings)
                combined_text.append(
                    f"--- START OF ARTICLE: {path} ---\n{clean_text[:_MAX_ARTICLE_CHARS]}\n--- END OF ARTICLE ---"
                )

            except requests.exceptions.RequestException as e:
                combined_text.append(f"--- FAILED TO FETCH: {path} ({e}) ---")

        print(f"Successfully extracted {len(path_list)} articles.")
        return "\n\n".join(combined_text)
