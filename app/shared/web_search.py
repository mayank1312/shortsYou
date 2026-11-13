import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from app.core.configuration import settings


logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"

REQUEST_TIMEOUT_SECONDS = 15.0

RESULTS_PER_QUERY = 5

SNIPPET_CHARS = 280


def _search_one(query: str) -> list[dict[str, str]]:

    try:

        response = httpx.post(
            TAVILY_URL,
            headers={
                "Authorization": f"Bearer {settings.TAVILY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "topic": "news",
                "time_range": "month",
                "search_depth": "basic",
                "max_results": RESULTS_PER_QUERY,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        results: list[dict[str, Any]] = response.json().get(
            "results", []
        )

    except Exception:

        # a failed search must never fail the whole forecast
        logger.exception("Tavily search failed | query=%s", query)

        return []

    return [
        {
            "title": str(result.get("title", "")).strip(),
            "snippet": str(result.get("content", "")).strip()[
                :SNIPPET_CHARS
            ],
        }
        for result in results
        if result.get("title")
    ]


def search_many(queries: list[str]) -> dict[str, list[dict[str, str]]]:
    """
    One search per query, run in parallel. Returns {query: results}.
    Returns {} when no Tavily key is configured, so callers can simply
    carry on without web data.
    """

    if not settings.TAVILY_API_KEY or not queries:
        return {}

    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        results = list(pool.map(_search_one, queries))

    return dict(zip(queries, results))