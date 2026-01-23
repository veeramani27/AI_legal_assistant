#
# 5. TOOLS AND BINDINGS
#


import os

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_core.tools import tool
from legal_modules.setup import llm

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

MAX_RESULTS = 5
MAX_CONTENT_CHARS = 3000


def google_search_and_fetch(query: str, max_results: int = MAX_RESULTS):
    """
    LangGraph Tool:
    - Performs Google search
    - Fetches page content for each result
    - Returns structured data for LLM consumption
    """

    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        raise ValueError("GOOGLE_API_KEY and GOOGLE_CSE_ID must be set")

    # Setup and perform Google search
    search_url = "https://www.googleapis.com/customsearch/v1"
    search_params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": min(max_results, 10),
    }

    search_response = requests.get(search_url, params=search_params, timeout=10)
    search_response.raise_for_status()
    search_data = search_response.json()

    # Fetch page content for each result and format it
    results = []

    for item in search_data.get("items", []):
        url = item.get("link")
        result = {
            "title": item.get("title"),
            "url": url,
            "snippet": item.get("snippet"),
            "content": None,
            "error": None,
        }

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            page_response = requests.get(url, headers=headers, timeout=10)
            page_response.raise_for_status()

            soup = BeautifulSoup(page_response.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = " ".join(soup.stripped_strings)
            result["content"] = text[:MAX_CONTENT_CHARS]

        except Exception as e:
            result["error"] = str(e)

        results.append(result)

    return {
        "query": query,
        "results": results,
        "result_count": len(results),
    }


@tool
def web_search_tool(query: str) -> dict:
    """
    Search the web and return full page content for each result.

    Params:
    - query (str): The search query

    Returns:
    dict: A dictionary containing the search results
    """
    return google_search_and_fetch("site:indiankanoon.org " + query)


websearch_llm = llm.bind_tools([web_search_tool])
