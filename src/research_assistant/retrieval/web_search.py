"""General web search client.

Chapter 5.3 treats Tavily and Serper as interchangeable: both exist to give
an agent clean, structured search results instead of a page of HTML built
for a human. This module supports either, picking whichever key is
configured, so swapping providers is a change to ``.env`` rather than a
change to any agent's code.

Tavily authenticates with an ``Authorization: Bearer`` header; Serper uses
an ``X-API-KEY`` header. Both take a JSON POST body.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config import MissingAPIKeyError, get_settings
from ..schemas import CandidateSource
from .base import RetrievalError, condense, post_json

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
SERPER_SEARCH_URL = "https://google.serper.dev/search"


def parse_tavily_response(
    payload: Dict[str, Any], max_results: Optional[int] = None
) -> List[CandidateSource]:
    """Turn a Tavily /search response into candidate sources."""
    if not isinstance(payload, dict) or "results" not in payload:
        raise RetrievalError(
            "Tavily response did not contain a 'results' list; the API shape may have changed. "
            f"Top-level keys seen: {sorted(payload)[:10] if isinstance(payload, dict) else type(payload)}"
        )
    return _from_generic_results(
        payload.get("results") or [],
        title_key="title",
        url_key="url",
        snippet_key="content",
        max_results=max_results,
    )


def parse_serper_response(
    payload: Dict[str, Any], max_results: Optional[int] = None
) -> List[CandidateSource]:
    """Turn a Serper /search response into candidate sources."""
    if not isinstance(payload, dict) or "organic" not in payload:
        raise RetrievalError(
            "Serper response did not contain an 'organic' list; the API shape may have changed. "
            f"Top-level keys seen: {sorted(payload)[:10] if isinstance(payload, dict) else type(payload)}"
        )
    return _from_generic_results(
        payload.get("organic") or [],
        title_key="title",
        url_key="link",
        snippet_key="snippet",
        max_results=max_results,
    )


def _from_generic_results(
    items: List[Any],
    *,
    title_key: str,
    url_key: str,
    snippet_key: str,
    max_results: Optional[int],
) -> List[CandidateSource]:
    sources: List[CandidateSource] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = condense(item.get(title_key), limit=300)
        url = item.get(url_key)
        if not title or not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        note = condense(item.get(snippet_key)) or "No snippet returned by the search provider."
        sources.append(
            CandidateSource(
                title=title,
                source_url=url,
                source_type="web page",
                relevance_note=note,
            )
        )
        if max_results is not None and len(sources) >= max_results:
            break
    return sources


async def search_web(query: str, max_results: int = 8) -> List[CandidateSource]:
    """Search the general web using whichever provider is configured.

    Raises MissingAPIKeyError if neither provider has a key, rather than
    silently returning nothing -- an agent that believes it searched and
    found nothing is worse than one told plainly that it never searched.
    """
    settings = get_settings()
    provider = settings.search_provider

    if provider == "tavily":
        payload = await post_json(
            TAVILY_SEARCH_URL,
            json_body={
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
        )
        return parse_tavily_response(payload, max_results=max_results)

    if provider == "serper":
        payload = await post_json(
            SERPER_SEARCH_URL,
            json_body={"q": query, "num": max_results},
            headers={"X-API-KEY": settings.serper_api_key or ""},
        )
        return parse_serper_response(payload, max_results=max_results)

    raise MissingAPIKeyError(
        "No web search provider is configured. Set TAVILY_API_KEY (free tier at "
        "https://tavily.com/) or SERPER_API_KEY (https://serper.dev/) in your .env file."
    )
