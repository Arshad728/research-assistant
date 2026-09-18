"""Tests for the Search Agent's Gemini function-calling path.

The Claude path is exercised live only -- it needs the real SDK's own
tool-use loop and a network-reachable model. What's tested here offline is
the hand-rolled loop in ``_find_sources_gemini`` (Gemini's client just
returns function calls and expects the caller to run them, unlike the
Claude SDK which runs its own loop internally), by replacing the real
``google.genai.Client`` with a fake one that plays back a scripted
conversation. That checks the loop's plumbing -- accumulating turns,
dispatching function calls through ``run_search_tool``, stopping once the
model replies with plain text -- without any network or a real API key.

``run_search_tool`` itself, the piece shared with the Claude MCP tool
wrappers, is tested directly too: it is what should make a query behave the
same way (clamping, error text) no matter which provider is asking.
"""
import json

import httpx
import pytest
import respx
from google.genai import types

from research_assistant.agents.search_agent import SearchAgent
from research_assistant.config import MissingAPIKeyError
from research_assistant.retrieval.arxiv import ARXIV_API_URL
from research_assistant.tools.search_tools import TOOL_SPECS, run_search_tool

from .fixtures import ARXIV_ATOM_RESPONSE


def _response(parts):
    """Build a real GenerateContentResponse, so .text/.function_calls behave
    exactly as they would against a live reply."""
    content = types.Content(role="model", parts=parts)
    candidate = types.Candidate(content=content, finish_reason="STOP")
    return types.GenerateContentResponse(candidates=[candidate])


class FakeModels:
    """Stands in for ``client.models``, playing back a fixed list of replies."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        return self._responses.pop(0)


class FakeGeminiClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


class TestToolSpecs:
    def test_covers_exactly_the_three_search_tools(self):
        assert set(TOOL_SPECS.keys()) == {"search_arxiv", "search_semantic_scholar", "search_web"}

    def test_every_spec_has_a_callable_and_a_description(self):
        for spec in TOOL_SPECS.values():
            assert callable(spec["fn"])
            assert spec["description"]


class TestRunSearchTool:
    @respx.mock
    async def test_matches_the_claude_tool_wrapper_s_underlying_data(self):
        respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(200, text=ARXIV_ATOM_RESPONSE))

        result = await run_search_tool("search_arxiv", {"query": "four day week", "max_results": 5})

        assert result["result_count"] == 2
        assert result["results"][0]["source_url"] == "http://arxiv.org/abs/2401.01234v1"

    async def test_a_missing_query_is_an_error_not_a_crash(self):
        result = await run_search_tool("search_arxiv", {"query": "   "})
        assert result == {"error": "No query was provided."}

    async def test_an_unrecognised_tool_name_is_a_readable_error(self):
        result = await run_search_tool("search_bing", {"query": "x"})
        assert "Unknown tool" in result["error"]

    async def test_missing_api_key_comes_back_as_readable_text(self, no_api_keys):
        result = await run_search_tool("search_web", {"query": "four day week"})
        assert "TAVILY_API_KEY" in result["error"]


class TestFindSourcesGemini:
    @respx.mock
    async def test_drives_the_tool_loop_and_reconciles_the_shortlist(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

        tool_call = _response(
            [
                types.Part(
                    function_call=types.FunctionCall(
                        name="search_arxiv",
                        args={"query": "four day work week", "max_results": 5},
                    )
                )
            ]
        )
        final_reply = _response(
            [
                types.Part(
                    text=json.dumps(
                        [
                            {
                                "source_url": "http://arxiv.org/abs/2401.01234v1",
                                "relevance_note": "primary trial result",
                            }
                        ]
                    )
                )
            ]
        )
        fake_client = FakeGeminiClient([tool_call, final_reply])
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: fake_client)
        respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(200, text=ARXIV_ATOM_RESPONSE))

        agent = SearchAgent(provider="gemini", max_rounds=2)
        outcome = await agent.find_sources("four day work week")

        assert len(fake_client.models.calls) == 2
        # Both arXiv entries were retrieved by the tool call...
        assert outcome.retrieved_count == 2
        # ...but only the one URL the model actually named survives reconciliation.
        assert outcome.shortlisted_count == 1
        assert outcome.sources[0].source_url == "http://arxiv.org/abs/2401.01234v1"
        assert not outcome.dropped_unknown_urls

    async def test_needs_a_gemini_key(self, no_api_keys):
        agent = SearchAgent(provider="gemini")
        with pytest.raises(MissingAPIKeyError, match="GEMINI_API_KEY"):
            await agent.find_sources("anything")

    def test_defaults_to_the_claude_provider(self):
        assert SearchAgent().provider == "claude"
