"""Tests for the SDK tool wrappers around the retrieval clients.

The behaviour that matters here is not "does it search" -- that is tested a
layer down -- but "what does the agent see." A tool that raises would end
the agent's turn; these tests pin down that every failure instead comes
back as readable text the agent can act on.
"""
import json

import httpx
import respx

from research_assistant.retrieval.arxiv import ARXIV_API_URL
from research_assistant.tools.search_tools import (
    ALLOWED_SEARCH_TOOL_NAMES,
    SEARCH_TOOLS,
    create_search_tool_server,
    reset_query_history,
    search_arxiv_tool,
    search_web_tool,
    start_query_history,
)

from .fixtures import ARXIV_ATOM_RESPONSE

EMPTY_ARXIV_FEED = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'


def _text(result):
    return result["content"][0]["text"]


@respx.mock
async def test_successful_search_returns_structured_json():
    respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(200, text=ARXIV_ATOM_RESPONSE))

    result = await search_arxiv_tool.handler({"query": "four day week", "max_results": 5})

    assert not result.get("is_error")
    payload = json.loads(_text(result))
    assert payload["source"] == "arxiv"
    assert payload["result_count"] == 2
    assert set(payload["results"][0]) == {
        "title",
        "source_url",
        "source_type",
        "relevance_note",
    }


@respx.mock
async def test_empty_result_set_tells_the_agent_not_to_repeat_itself():
    respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(200, text=EMPTY_ARXIV_FEED))

    payload = json.loads(_text(await search_arxiv_tool.handler({"query": "nonsense"})))

    assert payload["result_count"] == 0
    assert "do not simply repeat" in payload["note"]


async def test_missing_query_is_an_error_not_a_crash():
    result = await search_arxiv_tool.handler({"query": "   "})
    assert result["is_error"] is True
    assert "No query" in _text(result)


async def test_missing_api_key_comes_back_as_readable_text(no_api_keys):
    result = await search_web_tool.handler({"query": "four day week"})
    assert result["is_error"] is True
    assert "TAVILY_API_KEY" in _text(result)


@respx.mock
async def test_http_failure_comes_back_as_readable_text():
    respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(403, text="denied"))

    result = await search_arxiv_tool.handler({"query": "four day week"})

    assert result["is_error"] is True
    assert "SEARCH ERROR" in _text(result)
    assert "HTTP 403" in _text(result)


@respx.mock
async def test_max_results_is_clamped_to_a_sane_range():
    route = respx.get(ARXIV_API_URL).mock(
        return_value=httpx.Response(200, text=ARXIV_ATOM_RESPONSE)
    )

    await search_arxiv_tool.handler({"query": "x", "max_results": 9999})
    assert route.calls.last.request.url.params["max_results"] == "25"

    await search_arxiv_tool.handler({"query": "x", "max_results": 0})
    assert route.calls.last.request.url.params["max_results"] == "10"


class TestRepeatedQueriesAreRefused:
    """The invariant the documentation claims, now actually enforced.

    A review found that the repeat-detection code existed but nothing on the
    live path called it: "the agent does not repeat queries" was a sentence
    in a system prompt, not a guarantee. These tests pin the guarantee.
    """

    @respx.mock
    async def test_the_same_query_twice_is_refused(self):
        route = respx.get(ARXIV_API_URL).mock(
            return_value=httpx.Response(200, text=ARXIV_ATOM_RESPONSE)
        )
        token = start_query_history()
        try:
            first = await search_arxiv_tool.handler({"query": "four day week"})
            second = await search_arxiv_tool.handler({"query": "four day week"})
        finally:
            reset_query_history(token)

        assert not first.get("is_error")
        assert second["is_error"] is True
        assert "already been run" in _text(second)
        assert route.call_count == 1, "the repeated query still hit the API"

    @respx.mock
    async def test_a_reordered_query_counts_as_the_same_query(self):
        respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(200, text=ARXIV_ATOM_RESPONSE))
        token = start_query_history()
        try:
            await search_arxiv_tool.handler({"query": "four day week productivity"})
            again = await search_arxiv_tool.handler({"query": "productivity four-day week"})
        finally:
            reset_query_history(token)

        assert again["is_error"] is True

    @respx.mock
    async def test_a_genuinely_different_query_is_allowed(self):
        route = respx.get(ARXIV_API_URL).mock(
            return_value=httpx.Response(200, text=ARXIV_ATOM_RESPONSE)
        )
        token = start_query_history()
        try:
            await search_arxiv_tool.handler({"query": "four day week productivity"})
            second = await search_arxiv_tool.handler({"query": "reduced hours output manufacturing"})
        finally:
            reset_query_history(token)

        assert not second.get("is_error")
        assert route.call_count == 2

    @respx.mock
    async def test_the_same_query_to_a_different_tool_is_allowed(self):
        # Searching arXiv and the web for the same thing is sensible; only
        # repeating it against the same source is wasted effort.
        respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(200, text=ARXIV_ATOM_RESPONSE))
        token = start_query_history()
        try:
            first = await search_arxiv_tool.handler({"query": "four day week"})
            second = await search_web_tool.handler({"query": "four day week"})
        finally:
            reset_query_history(token)

        assert not first.get("is_error")
        # The web tool fails for a missing key, not for being a repeat.
        assert "already been run" not in _text(second)

    @respx.mock
    async def test_history_does_not_leak_between_runs(self):
        respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(200, text=ARXIV_ATOM_RESPONSE))

        for _ in range(2):
            token = start_query_history()
            try:
                result = await search_arxiv_tool.handler({"query": "four day week"})
            finally:
                reset_query_history(token)
            assert not result.get("is_error")

    @respx.mock
    async def test_without_a_run_in_progress_nothing_is_refused(self):
        respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(200, text=ARXIV_ATOM_RESPONSE))

        first = await search_arxiv_tool.handler({"query": "standalone use"})
        second = await search_arxiv_tool.handler({"query": "standalone use"})

        assert not first.get("is_error")
        assert not second.get("is_error")


def test_tool_server_exposes_exactly_the_three_search_tools():
    server = create_search_tool_server()
    assert server["type"] == "sdk"
    assert server["name"] == "research-retrieval"

    tool_names = {t.name for t in SEARCH_TOOLS}
    assert tool_names == {"search_arxiv", "search_semantic_scholar", "search_web"}
    assert set(ALLOWED_SEARCH_TOOL_NAMES) == {
        f"mcp__research-retrieval__{name}" for name in tool_names
    }
