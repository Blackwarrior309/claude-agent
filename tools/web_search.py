"""
Web search and scraping tools using requests + BeautifulSoup.
"""
import logging
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MicroBusinessBot/1.0; research-only)",
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
}


def search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo and return titles + snippets."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}&kl=de-de"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        for result in soup.select(".result")[:max_results]:
            title_el = result.select_one(".result__title")
            snippet_el = result.select_one(".result__snippet")
            link_el = result.select_one(".result__url")

            if title_el:
                results.append({
                    "title": title_el.get_text(strip=True),
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                    "url": link_el.get_text(strip=True) if link_el else "",
                })
        return results
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []


def fetch_page_text(url: str, max_chars: int = 3000) -> Optional[str]:
    """Fetch and extract main text content from a URL."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:max_chars]
    except Exception as e:
        logger.warning(f"Page fetch failed for {url}: {e}")
        return None


def get_tool_definitions() -> list[dict]:
    return [
        {
            "name": "web_search",
            "description": "Search the web for information using DuckDuckGo",
            "schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            "fn": lambda query, max_results=5: search_duckduckgo(query, max_results),
        },
        {
            "name": "fetch_url",
            "description": "Fetch and extract text content from a URL",
            "schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 3000},
                },
                "required": ["url"],
            },
            "fn": lambda url, max_chars=3000: fetch_page_text(url, max_chars),
        },
    ]
