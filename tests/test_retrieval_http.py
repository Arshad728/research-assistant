"""Tests for the HTTP half of each retrieval client.

These use respx to intercept requests, so nothing leaves the machine. What
they check is the part most likely to break silently against a real API:
the URL, the query parameters, and above all the authentication header
names, which differ for every provider and produce an unhelpful 401 when
wrong.
"""
import httpx
import pytest
import respx

from research_assistant import config
from research_assistant.config import MissingAPIKeyError
from research_assistant.retrieval.arxiv import ARXIV_API_URL, search_arxiv
from research_assistant.retrieval.base import RetrievalError
from research_assistant.retrieval.semantic_scholar import (
    SEMANTIC_SCHOLAR_SEARCH_URL,
    search_semantic_scholar,
)
from research_assistant.retrieval.web_search import (
    SERPER_SEARCH_URL,
    TAVILY_SEARCH_URL,
    search_web,
)

from .fixtures import ARXIV_ATOM_RESPONSE, SEMANTIC_SCHOLAR_RESPONSE, SERPER_RESPONSE, TAVILY_RESPONSE


@respx.mock
async def test_arxiv_request_is_built_correctly():
    route = respx.get(ARXIV_API_URL).mock(
        return_value=httpx.Response(200, text=ARXIV_ATOM_RESPONSE)
    )
    sources = await search_arxiv("four day work week", max_results=5)

    assert len(sources) == 2
    request = route.calls.last.request
    assert request.url.params["search_query"] == "all:four day work week"
    assert request.url.params["max_results"] == "5"
    assert request.url.params["sortBy"] == "relevance"
    assert "research-assistant" in request.headers["user-agent"]


@respx.mock
async def test_semantic_scholar_sends_api_key_header_when_configured(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2-test-key")
    config.get_settings.cache_clear()

    route = respx.get(SEMANTIC_SCHOLAR_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=SEMANTIC_SCHOLAR_RESPONSE)
    )
    await search_semantic_scholar("four day work week", max_results=3)

    request = route.calls.last.request
    assert request.headers["x-api-key"] == "s2-test-key"
    assert request.url.params["query"] == "four day work week"
    assert request.url.params["limit"] == "3"
    assert "citationCount" in request.url.params["fields"]


@respx.mock
async def test_semantic_scholar_omits_api_key_header_when_absent(no_api_keys):
    route = respx.get(SEMANTIC_SCHOLAR_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=SEMANTIC_SCHOLAR_RESPONSE)
    )
    await search_semantic_scholar("anything")

    assert "x-api-key" not in route.calls.last.request.headers


@respx.mock
async def test_tavily_uses_bearer_authorization_header(monkeypatch, no_api_keys):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    config.get_settings.cache_clear()

    route = respx.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=TAVILY_RESPONSE)
    )
    sources = await search_web("four day work week", max_results=4)

    assert len(sources) == 2
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer tvly-test-key"
    assert b'"query": "four day work week"' in request.content or b'"query":"four day work week"' in request.content


@respx.mock
async def test_serper_uses_x_api_key_header(monkeypatch, no_api_keys):
    monkeypatch.setenv("SERPER_API_KEY", "serper-test-key")
    config.get_settings.cache_clear()

    route = respx.post(SERPER_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=SERPER_RESPONSE)
    )
    sources = await search_web("four day work week")

    assert len(sources) == 1
    assert route.calls.last.request.headers["x-api-key"] == "serper-test-key"


async def test_web_search_without_any_provider_says_so_plainly(no_api_keys):
    with pytest.raises(MissingAPIKeyError, match="TAVILY_API_KEY"):
        await search_web("anything")


@respx.mock
async def test_tavily_is_preferred_when_both_keys_are_set(monkeypatch, no_api_keys):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.setenv("SERPER_API_KEY", "serper-test-key")
    config.get_settings.cache_clear()

    tavily = respx.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=TAVILY_RESPONSE)
    )
    serper = respx.post(SERPER_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=SERPER_RESPONSE)
    )
    await search_web("anything")

    assert tavily.called
    assert not serper.called


@respx.mock
async def test_forbidden_response_mentions_both_likely_causes():
    respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(403, text="denied"))
    with pytest.raises(RetrievalError) as excinfo:
        await search_arxiv("anything")

    message = str(excinfo.value)
    assert "HTTP 403" in message
    assert "API key" in message
    assert "network policy" in message


@respx.mock
async def test_rate_limited_response_says_so(no_api_keys):
    respx.get(SEMANTIC_SCHOLAR_SEARCH_URL).mock(
        return_value=httpx.Response(429, text="slow down")
    )
    with pytest.raises(RetrievalError, match="Rate limited"):
        await search_semantic_scholar("anything")


@respx.mock
async def test_connection_failure_is_reported_not_swallowed():
    respx.get(ARXIV_API_URL).mock(side_effect=httpx.ConnectError("no route to host"))
    with pytest.raises(RetrievalError, match="Could not reach"):
        await search_arxiv("anything")
