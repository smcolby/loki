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
from bs4 import BeautifulSoup


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
            A newline-separated list of matching titles and paths, or an error message.
        """
        safe_query = urllib.parse.quote(query)
        url = f"{self.base_url}/search?pattern={safe_query}"

        try:
            # Send a network request to retrieve the search results page.
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            # Extract links from the search results.
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                title = a_tag.get_text(strip=True)

                # Filter out standard Kiwix UI links and pagination.
                if title and not href.startswith(("?", "/search", "/skin", "/catalog")):
                    results.append(f"Title: {title} | Path: {href}")

            if not results:
                print("No matches found in the local database.")
                return "No results found for that query."

            print("Successfully retrieved search result titles.")

            # Return the top results for the model to evaluate.
            formatted_results = "\n".join(results[:10])
            return f"Found the following articles. Use the read_articles tool with the exact Path to read up to 3 of them:\n{formatted_results}"

        except requests.exceptions.RequestException as e:
            print(f"Failed to connect to the local server: {e}.")
            return "Search failed due to a network error."

    def read_articles(self, paths: list[str]) -> str:
        """Fetch the full text of up to 3 specific articles from the local Kiwix database.

        Pass the exact paths returned by the search_article_titles tool.

        Parameters
        ----------
        paths : list of str
            URL paths to read (e.g. ``["/content/A/Article_1.html"]``).
            Only the first 3 entries are processed.

        Returns
        -------
        str
            Concatenated article text, with each article wrapped in start/end markers.
        """
        # Limit to a batch of 3 articles.
        path_list = paths[:3]
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
                    f"--- START OF ARTICLE: {path} ---\n{clean_text[:8000]}\n--- END OF ARTICLE ---"
                )

            except requests.exceptions.RequestException as e:
                combined_text.append(f"--- FAILED TO FETCH: {path} ({e}) ---")

        print(f"Successfully extracted {len(path_list)} articles.")
        return "\n\n".join(combined_text)
