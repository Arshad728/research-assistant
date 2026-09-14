"""The Search Agent's three tools, exposed to the Claude Agent SDK.

This module is deliberately thin. All of the real work -- HTTP, parsing,
rate limiting -- lives in ``research_assistant.retrieval``; everything here
does is adapt those functions into the shape the SDK expects, so that the
retrieval code stays testable without ever starting an agent.

Results are handed back as JSON rather than prose. That is the Chapter 3.3
idea applied to the agent's own tools: the agent receives a list of records
with known fields, not a paragraph it has to re-interpret.

Failures are returned as readable error text with ``is_error`` set, rather
than raised. A raised exception would end the agent's turn; an error it can
read is something it can react to -- by trying a different tool, or a
different query -- which is exactly the observation loop described in
Chapter 2.3.
"""
from __future__ import annotations

import contextvars
import json
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from claude_agent_sdk import create_sdk_mcp_server, tool

from ..config import MissingAPIKeyError
from ..retrieval import search_arxiv, search_semantic_scholar, search_web
from ..retrieval.base import RetrievalError
from ..schemas import CandidateSource
from ..search.planner import SearchPlanner

SEARCH_TOOL_SERVER_NAME = "research-retrieval"

_SCHEMA = {"query": str, "max_results": int}

# Queries already run during the current Search Agent run, held per-task so
# that concurrent runs cannot see each other's history.
#
# Chapter 4.3 requires the agent to "reformulate rather than repeat," and an
# adversarial review of an earlier version pointed out that this was only
# ever an instruction in the system prompt: the repeat-detection code existed
# but nothing on the live path called it. A model asked not to repeat itself
# usually will not. Enforcing it here makes "usually" into "cannot", which is
# the distinction the rest of this project keeps insisting on.
_QUERY_HISTORY: contextvars.ContextVar[Optional[Dict[str, Set]]] = contextvars.ContextVar(
    "search_query_history", default=None
)


def start_query_history() -> contextvars.Token:
    """Begin recording queries for one run. Returns a token for resetting."""
    return _QUERY_HISTORY.set({"fingerprints": set(), "queries": set()})


def reset_query_history(token: contextvars.Token) -> None:
    _QUERY_HISTORY.reset(token)


def _already_run(tool_name: str, query: str) -> bool:
    """Has this query, in substance, already been run by this tool?"""
    history = _QUERY_HISTORY.get()
    if history is None:
        return False
    fingerprint = (tool_name, SearchPlanner._fingerprint(query))
    if fingerprint in history["fingerprints"]:
        return True
    history["fingerprints"].add(fingerprint)
    history["queries"].add(query)
    return False


def _ok(sources: List[CandidateSource], *, source_name: str) -> Dict[str, Any]:
    payload = {
        "source": source_name,
        "result_count": len(sources),
        "results": [source.model_dump() for source in sources],
    }
    if not sources:
        payload["note"] = (
            "No results matched this query. Consider rephrasing it, broadening it, "
            "or trying a different tool -- do not simply repeat the same query."
        )
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}]}


def _failed(message: str) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": f"SEARCH ERROR: {message}"}], "is_error": True}


async def _run(
    search_fn: Callable[..., Awaitable[List[CandidateSource]]],
    args: Dict[str, Any],
    *,
    source_name: str,
    default_max: int,
) -> Dict[str, Any]:
    query = (args.get("query") or "").strip()
    if not query:
        return _failed("No query was provided.")
    if _already_run(source_name, query):
        return _failed(
            f"This search has already been run against {source_name} in this session, and "
            "would return the same results. Reformulate it -- different keywords, a "
            "different angle, or a different tool -- rather than repeating it."
        )
    max_results = args.get("max_results") or default_max
    try:
        max_results = max(1, min(int(max_results), 25))
    except (TypeError, ValueError):
        max_results = default_max

    try:
        sources = await search_fn(query, max_results=max_results)
    except MissingAPIKeyError as exc:
        return _failed(str(exc))
    except RetrievalError as exc:
        return _failed(str(exc))
    except Exception as exc:  # noqa: BLE001 - a tool must not kill the agent's turn
        return _failed(f"Unexpected {type(exc).__name__}: {exc}")
    return _ok(sources, source_name=source_name)


@tool(
    "search_arxiv",
    "Search arXiv for preprints. Best for recent computer science, physics, and "
    "mathematics work that may not be formally published yet.",
    _SCHEMA,
)
async def search_arxiv_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    return await _run(search_arxiv, args, source_name="arxiv", default_max=10)


@tool(
    "search_semantic_scholar",
    "Search Semantic Scholar for published academic papers across all fields. "
    "Returns citation counts, which help judge how established a finding is.",
    _SCHEMA,
)
async def search_semantic_scholar_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    return await _run(search_semantic_scholar, args, source_name="semantic_scholar", default_max=10)


@tool(
    "search_web",
    "Search the general web. Best for current events, industry reports, government "
    "or organisational sources, and anything not confined to academic literature.",
    _SCHEMA,
)
async def search_web_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    return await _run(search_web, args, source_name="web", default_max=8)


SEARCH_TOOLS = [search_arxiv_tool, search_semantic_scholar_tool, search_web_tool]

# The fully-qualified names the SDK uses for permissioning these tools.
ALLOWED_SEARCH_TOOL_NAMES = [
    f"mcp__{SEARCH_TOOL_SERVER_NAME}__search_arxiv",
    f"mcp__{SEARCH_TOOL_SERVER_NAME}__search_semantic_scholar",
    f"mcp__{SEARCH_TOOL_SERVER_NAME}__search_web",
]


def create_search_tool_server():
    """Build the in-process MCP server holding the Search Agent's tools."""
    return create_sdk_mcp_server(
        name=SEARCH_TOOL_SERVER_NAME,
        version="0.1.0",
        tools=SEARCH_TOOLS,
    )
