"""Retrieval clients for the Search Agent's three tools (Chapter 4.3)."""

from .arxiv import search_arxiv
from .base import RetrievalError
from .semantic_scholar import search_semantic_scholar
from .web_search import search_web

__all__ = [
    "RetrievalError",
    "search_arxiv",
    "search_semantic_scholar",
    "search_web",
]
