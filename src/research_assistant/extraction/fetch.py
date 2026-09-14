"""Fetching a candidate source and turning it into a SourceDocument.

One extra behaviour is worth explaining. A Search Agent result for an arXiv
paper points at the abstract page, which contains the abstract but not the
paper. Extracting claims from an abstract alone would mean the system
routinely "reads" papers it has only skimmed the summary of. So arXiv
abstract links are rewritten to the PDF before fetching, which is the
difference between quoting a paper and quoting its blurb.
"""
from __future__ import annotations

import re
from typing import Optional

import httpx

from ..schemas import CandidateSource
from .documents import SourceDocument
from .parsers import ParseError, extract_html_text, extract_pdf_text

DEFAULT_TIMEOUT_SECONDS = 40.0
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024  # a 25MB paper is already unusual

USER_AGENT = "research-assistant/0.1 (multi-agent research assistant; educational project)"

_ARXIV_ABS = re.compile(r"^https?://(?:www\.)?arxiv\.org/abs/(?P<id>[^/?#]+)", re.IGNORECASE)


class FetchError(RuntimeError):
    """A source could not be downloaded."""


def resolve_fetch_url(url: str) -> str:
    """Prefer the full text over a landing page where the mapping is known."""
    match = _ARXIV_ABS.match(url.strip())
    if match:
        return f"https://arxiv.org/pdf/{match.group('id')}"
    return url


def _looks_like_pdf(content_type: str, data: bytes, url: str) -> bool:
    if "application/pdf" in content_type.lower():
        return True
    if data[:5] == b"%PDF-":
        return True
    return url.lower().split("?")[0].endswith(".pdf")


async def fetch_source(
    source: CandidateSource | str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> SourceDocument:
    """Download one source and extract its text.

    Raises FetchError if the source cannot be downloaded and ParseError if
    it downloads but cannot be read. The two are kept distinct because they
    mean different things to the agent: the first is worth retrying or
    skipping, the second means this source will never be usable.
    """
    if isinstance(source, CandidateSource):
        original_url, fallback_title = source.source_url, source.title
    else:
        original_url, fallback_title = source, ""

    fetch_url = resolve_fetch_url(original_url)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(fetch_url, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            data = response.content
            content_type = response.headers.get("content-type", "")
    except httpx.HTTPStatusError as exc:
        raise FetchError(
            f"HTTP {exc.response.status_code} fetching {fetch_url}. The source may be "
            "paywalled, removed, or blocking automated access."
        ) from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"Could not download {fetch_url}: {exc}") from exc

    if len(data) > MAX_DOWNLOAD_BYTES:
        raise FetchError(
            f"{fetch_url} is {len(data) // (1024 * 1024)}MB, above the "
            f"{MAX_DOWNLOAD_BYTES // (1024 * 1024)}MB limit."
        )

    if _looks_like_pdf(content_type, data, fetch_url):
        text, page_count = extract_pdf_text(data)
        return SourceDocument(
            source_url=original_url,
            title=fallback_title,
            text=text,
            content_kind="pdf",
            page_count=page_count,
        )

    try:
        html = data.decode(response.encoding or "utf-8", errors="replace")
    except (LookupError, UnicodeDecodeError):
        html = data.decode("utf-8", errors="replace")

    text, title = extract_html_text(html, url=fetch_url)
    return SourceDocument(
        source_url=original_url,
        title=title or fallback_title,
        text=text,
        content_kind="html",
    )
