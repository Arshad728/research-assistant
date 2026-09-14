"""Semantic Scholar retrieval client.

Semantic Scholar covers published work across fields and, usefully for
judging how established a claim is (Chapter 3.4), reports how often each
paper has been cited.

An API key is passed in the ``x-api-key`` header when one is configured.
Phase 1.2 found that this is worth having: the keyless pool is shared
globally and heavily throttles the ``/paper/search`` endpoint specifically,
which is the one endpoint this client depends on.

A deliberate honesty choice about ``source_type``: this module never labels
a paper "peer-reviewed," because nothing in the metadata actually proves
peer review happened. It reports what the metadata does support -- journal
article, conference paper, or preprint -- and leaves any stronger claim to
a human or to the verification step.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..schemas import CandidateSource
from .base import RateLimiter, RetrievalError, condense, get_json

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

REQUESTED_FIELDS = (
    "title,abstract,url,year,venue,citationCount,externalIds,openAccessPdf,publicationTypes"
)

# A configured API key guarantees one request per second; without one the
# shared pool is slower and far less reliable, so pace either way.
_RATE_LIMITER = RateLimiter(1.0)


def _pick_url(paper: Dict[str, Any]) -> Optional[str]:
    """Choose the best link for a paper, preferring one a reader can open.

    An arXiv link is preferred over the Semantic Scholar landing page when
    the paper has one. That ordering is deliberate and was wrong in an
    earlier version: deduplication canonicalises arXiv identifiers, so the
    same preprint arriving from arXiv directly and from Semantic Scholar's
    record of it only collapses into one source if both routes produce an
    arXiv URL. Preferring the S2 page meant the same paper was read twice
    and could be cited twice as though it were two sources -- which is worse
    than wasteful, because the stopping rule counts distinct sources as
    corroboration.
    """
    open_access = paper.get("openAccessPdf") or {}
    external = paper.get("externalIds") or {}

    candidates = []
    if isinstance(external, dict) and external.get("ArXiv"):
        candidates.append(f"https://arxiv.org/abs/{external['ArXiv']}")
    candidates.append(paper.get("url"))
    if isinstance(open_access, dict):
        candidates.append(open_access.get("url"))
    if isinstance(external, dict) and external.get("DOI"):
        candidates.append(f"https://doi.org/{external['DOI']}")

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
            return candidate
    return None


def _classify(paper: Dict[str, Any]) -> str:
    publication_types = paper.get("publicationTypes") or []
    if isinstance(publication_types, list):
        if "JournalArticle" in publication_types:
            return "journal article"
        if "Conference" in publication_types:
            return "conference paper"
        if "Review" in publication_types:
            return "review article"
    external = paper.get("externalIds") or {}
    if isinstance(external, dict) and external.get("ArXiv") and not paper.get("venue"):
        return "preprint"
    return "academic paper"


def parse_semantic_scholar_response(
    payload: Dict[str, Any], max_results: Optional[int] = None
) -> List[CandidateSource]:
    """Turn a /paper/search response into candidate sources."""
    if not isinstance(payload, dict) or "data" not in payload:
        raise RetrievalError(
            "Semantic Scholar response did not contain a 'data' list; the API shape may have "
            f"changed. Top-level keys seen: {sorted(payload)[:10] if isinstance(payload, dict) else type(payload)}"
        )

    papers = payload.get("data") or []
    sources: List[CandidateSource] = []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        title = condense(paper.get("title"), limit=300)
        url = _pick_url(paper)
        if not title or not url:
            continue

        facts = []
        venue = paper.get("venue")
        if venue:
            facts.append(str(venue))
        year = paper.get("year")
        if year:
            facts.append(str(year))
        citations = paper.get("citationCount")
        if isinstance(citations, int):
            facts.append(f"cited {citations} times")

        prefix = " | ".join(facts)
        abstract = condense(paper.get("abstract"))
        if prefix and abstract:
            note = f"{prefix}. {abstract}"
        else:
            note = prefix or abstract or "No abstract available from Semantic Scholar."

        sources.append(
            CandidateSource(
                title=title,
                source_url=url,
                source_type=_classify(paper),
                relevance_note=note,
            )
        )
        if max_results is not None and len(sources) >= max_results:
            break

    return sources


async def search_semantic_scholar(query: str, max_results: int = 10) -> List[CandidateSource]:
    """Search Semantic Scholar for published work matching a free-text query."""
    settings = get_settings()
    headers = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    params = {"query": query, "limit": max_results, "fields": REQUESTED_FIELDS}
    payload = await get_json(
        SEMANTIC_SCHOLAR_SEARCH_URL,
        params=params,
        headers=headers,
        rate_limiter=_RATE_LIMITER,
    )
    return parse_semantic_scholar_response(payload, max_results=max_results)
