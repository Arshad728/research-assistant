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
# the distinction the rest of this project keeps insisting on. It works the
# same way regardless of which provider is driving the search, since both
# the Claude tool wrappers and Gemini's ``run_search_tool`` call through
# ``_fetch_sources`` below.
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


async def _fetch_sources(
    search_fn: Callable[..., Awaitable[List[CandidateSource]]],
    args: Dict[str, Any],
    *,
    tool_name: str,
    default_max: int,
):
    """Validate args, refuse a repeat, clamp max_results, and call a retrieval function.

    Returns the sources on success, or a plain error string on failure --
    deliberately not a raised exception, so a bad query, a repeat, or a down
    endpoint is something the calling agent can read and react to (Chapter
    2.3's observation loop) instead of something that ends its turn. Shared
    by both ``_run`` below (the Claude MCP tool envelope) and
    ``run_search_tool`` (used by the Search Agent's Gemini function-calling
    loop), so a query clamps, repeats and fails identically no matter which
    model asked for it.
    """
    query = (args.get("query") or "").strip()
    if not query:
        return "No query was provided."
    if _already_run(tool_name, query):
        return (
            f"This search has already been run against {tool_name} in this session, and "
            "would return the same results. Reformulate it -- different keywords, a "
            "different angle, or a different tool -- rather than repeating it."
        )
    max_results = args.get("max_results") or default_max
    try:
        max_results = max(1, min(int(max_results), 25))
    except (TypeError, ValueError):
        max_results = default_max

    try:
        return await search_fn(query, max_results=max_results)
    except MissingAPIKeyError as exc:
        return str(exc)
    except RetrievalError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - a tool must not kill the agent's turn
        return f"Unexpected {type(exc).__name__}: {exc}"


async def _run(
    search_fn: Callable[..., Awaitable[List[CandidateSource]]],
    args: Dict[str, Any],
    *,
    source_name: str,
    default_max: int,
) -> Dict[str, Any]:
    result = await _fetch_sources(search_fn, args, tool_name=source_name, default_max=default_max)
    if isinstance(result, str):
        return _failed(result)
    return _ok(result, source_name=source_name)


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


# Tool names, descriptions and retrieval functions in one place, so the
# Claude tool wrappers above and any other caller describe the same three
# tools the same way. Descriptions are reused verbatim as the Gemini
# function-calling declarations in ``agents/search_agent.py``, so the two
# providers are told about the same tools in the same words.
TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "search_arxiv": {
        "fn": search_arxiv,
        "default_max": 10,
        "description": (
            "Search arXiv for preprints. Best for recent computer science, physics, and "
            "mathematics work that may not be formally published yet."
        ),
    },
    "search_semantic_scholar": {
        "fn": search_semantic_scholar,
        "default_max": 10,
        "description": (
            "Search Semantic Scholar for published academic papers across all fields. "
            "Returns citation counts, which help judge how established a finding is."
        ),
    },
    "search_web": {
        "fn": search_web,
        "default_max": 8,
        "description": (
            "Search the general web. Best for current events, industry reports, government "
            "or organisational sources, and anything not confined to academic literature."
        ),
    },
}


async def run_search_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Run one of the three retrieval tools by name, for callers outside the
    Claude MCP envelope -- currently the Search Agent's Gemini function-calling
    loop. Shares ``_fetch_sources`` with the Claude tool wrappers above, so a
    query clamps and fails exactly the same way no matter which model asked.

    Returns ``{"result_count": ..., "results": [...]}`` (plus a ``"note"`` if
    nothing matched), or ``{"error": "..."}`` -- never raises, for the same
    reason ``_fetch_sources`` does not.
    """
    spec = TOOL_SPECS.get(name)
    if spec is None:
        return {"error": f"Unknown tool {name!r}."}

    result = await _fetch_sources(spec["fn"], args, tool_name=name, default_max=spec["default_max"])
    if isinstance(result, str):
        return {"error": result}

    payload: Dict[str, Any] = {
        "result_count": len(result),
        "results": [source.model_dump() for source in result],
    }
    if not result:
        payload["note"] = (
            "No results matched this query. Consider rephrasing it, broadening it, "
            "or trying a different tool -- do not simply repeat the same query."
        )
    return payload
