"""Shared plumbing for the three retrieval clients.

Chapter 4.3 of the book gives the Search Agent three retrieval tools: a
general web search API, arXiv, and Semantic Scholar. Each one speaks a
different protocol, but they share the same concerns -- timeouts, polite
rate limiting, turning a failure into a message a human can act on -- so
those live here rather than being written three times.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Mapping, Optional

import httpx

DEFAULT_TIMEOUT_SECONDS = 20.0
USER_AGENT = "research-assistant/0.1 (multi-agent research assistant; educational project)"


class RetrievalError(RuntimeError):
    """A retrieval call failed in a way the caller should surface, not swallow."""


class RateLimiter:
    """Enforces a minimum interval between requests to one API.

    Both academic APIs this project uses ask callers to pace themselves:
    arXiv asks for roughly three seconds between requests, and Semantic
    Scholar grants one request per second to an API key holder. Respecting
    that in code is cheaper than getting throttled and misreading the
    resulting errors as "no results found" -- which, as Chapter 3.4 notes,
    is exactly the kind of silent failure that makes an agent look like it
    searched when it did not.
    """

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = min_interval_seconds
        self._lock = asyncio.Lock()
        self._last_call: Optional[float] = None

    async def wait(self) -> None:
        async with self._lock:
            if self._last_call is not None:
                elapsed = time.monotonic() - self._last_call
                remaining = self._min_interval - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)
            self._last_call = time.monotonic()


def condense(text: Optional[str], limit: int = 280) -> str:
    """Collapse whitespace and truncate on a word boundary.

    Used to turn a long abstract into a short, honest relevance note. It
    never paraphrases or embellishes -- it only shortens, so the note stays
    something the source actually says.
    """
    if not text:
        return ""
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed[:limit].rsplit(" ", 1)[0].rstrip(",;:.")
    return f"{cut}..."


def _describe_http_error(exc: httpx.HTTPStatusError) -> str:
    status = exc.response.status_code
    body = exc.response.text[:200].replace("\n", " ") if exc.response.text else ""
    hint = ""
    if status in (401, 403):
        hint = (
            " This is usually a missing/invalid API key, or -- if you are running inside a "
            "sandboxed environment -- an outbound network policy blocking the host. Check both."
        )
    elif status == 429:
        hint = " Rate limited. Slow down, or get an API key for a higher guaranteed rate."
    return f"HTTP {status} from {exc.request.url.host}: {body}{hint}"


async def get_text(
    url: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
    headers: Optional[Mapping[str, str]] = None,
    rate_limiter: Optional[RateLimiter] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """GET a URL and return the raw response body as text."""
    if rate_limiter is not None:
        await rate_limiter.wait()
    merged_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url, params=params, headers=merged_headers)
            response.raise_for_status()
            return response.text
    except httpx.HTTPStatusError as exc:
        raise RetrievalError(_describe_http_error(exc)) from exc
    except httpx.HTTPError as exc:
        raise RetrievalError(f"Could not reach {url}: {exc}") from exc


async def post_json(
    url: str,
    *,
    json_body: Mapping[str, Any],
    headers: Optional[Mapping[str, str]] = None,
    rate_limiter: Optional[RateLimiter] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """POST a JSON body and return the decoded JSON response."""
    if rate_limiter is not None:
        await rate_limiter.wait()
    merged_headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json", **(headers or {})}
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.post(url, json=dict(json_body), headers=merged_headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        raise RetrievalError(_describe_http_error(exc)) from exc
    except httpx.HTTPError as exc:
        raise RetrievalError(f"Could not reach {url}: {exc}") from exc


async def get_json(
    url: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
    headers: Optional[Mapping[str, str]] = None,
    rate_limiter: Optional[RateLimiter] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """GET a URL and return the decoded JSON response."""
    if rate_limiter is not None:
        await rate_limiter.wait()
    merged_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url, params=params, headers=merged_headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        raise RetrievalError(_describe_http_error(exc)) from exc
    except httpx.HTTPError as exc:
        raise RetrievalError(f"Could not reach {url}: {exc}") from exc
