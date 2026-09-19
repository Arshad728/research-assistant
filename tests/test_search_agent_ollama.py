"""Tests for the Search Agent's Ollama function-calling path.

Ollama's native Python client returns ``tool_calls`` on the assistant
message the same way Groq's OpenAI-compatible client does, so this mirrors
``test_search_agent_groq.py``'s approach: a fake ``ollama.Client`` plays
back a scripted conversation (one tool call, then a final plain-text
reply), which checks the hand-rolled loop's plumbing without any network or
a running Ollama server.

Two real shape differences from Groq's client, confirmed by introspecting
the installed ``ollama`` package rather than assumed, and deliberately
exercised here rather than copied blind from the Groq test file:

* A native ``Message.ToolCall`` has no ``id`` field at all (unlike Groq's
  OpenAI-style tool calls), so the fakes below don't have one either, and
  the loop correlates a tool's result back by ``tool_name`` instead of a
  ``tool_call_id``.
* ``function.arguments`` arrives already parsed as a dict, not a JSON
  string, so the fakes hand back a dict directly and the loop needs no
  ``json.loads`` step to read it.

``TestToolSpecs`` and ``TestRunSearchTool`` are provider-agnostic and are
already covered in ``test_search_agent_gemini.py``; this file only adds the
Ollama-specific loop and host-selection tests. There is no "needs a key"
test here, unlike the Groq and Gemini files -- Ollama needs no API key at
all, only a local server, which is exactly what the host-selection tests
below check instead.
"""
import json

import pytest

from research_assistant.agents.extraction_agent import DEFAULT_OLLAMA_HOST
from research_assistant.agents.search_agent import SearchAgent
from research_assistant.retrieval.arxiv import ARXIV_API_URL
import httpx
import respx

from .fixtures import ARXIV_ATOM_RESPONSE


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    """Mirrors ollama._types.Message.ToolCall -- no `id` field, unlike Groq's."""

    def __init__(self, name, arguments):
        self.function = FakeFunction(name, arguments)

    def model_dump(self):
        return {"function": {"name": self.function.name, "arguments": self.function.arguments}}


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChatResponse:
    """Mirrors ollama._types.ChatResponse: `.message`, not `.choices[0].message`."""

    def __init__(self, message):
        self.message = message


class FakeOllamaClient:
    """Stands in for ollama.Client, playing back fixed replies from .chat()."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, *, model, messages, tools, options):
        self.calls.append(
            {"model": model, "messages": [dict(m) for m in messages], "tools": tools, "options": options}
        )
        return self._responses.pop(0)


class TestFindSourcesOllama:
    @respx.mock
    async def test_drives_the_tool_loop_and_reconciles_the_shortlist(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)

        tool_call_reply = FakeChatResponse(
            FakeMessage(
                content=None,
                tool_calls=[
                    FakeToolCall(
                        "search_arxiv",
                        {"query": "four day work week", "max_results": 5},
                    )
                ],
            )
        )
        final_reply = FakeChatResponse(
            FakeMessage(
                content=json.dumps(
                    [
                        {
                            "source_url": "http://arxiv.org/abs/2401.01234v1",
                            "relevance_note": "primary trial result",
                        }
                    ]
                ),
                tool_calls=None,
            )
        )
        fake_client = FakeOllamaClient([tool_call_reply, final_reply])
        monkeypatch.setattr("ollama.Client", lambda **kwargs: fake_client)
        respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(200, text=ARXIV_ATOM_RESPONSE))

        agent = SearchAgent(provider="ollama", max_rounds=2)
        outcome = await agent.find_sources("four day work week")

        assert len(fake_client.calls) == 2
        # Both arXiv entries were retrieved by the tool call...
        assert outcome.retrieved_count == 2
        # ...but only the one URL the model actually named survives reconciliation.
        assert outcome.shortlisted_count == 1
        assert outcome.sources[0].source_url == "http://arxiv.org/abs/2401.01234v1"
        assert not outcome.dropped_unknown_urls

        # The arguments the fake tool call carried were already a dict -- no
        # json.loads step was needed to read them, unlike Groq's.
        first_call_tools = fake_client.calls[0]["tools"]
        assert first_call_tools[0]["function"]["name"] == "search_arxiv"

        # The tool result was sent back keyed by tool_name, not a tool_call_id
        # (which doesn't exist on this client's tool calls at all).
        tool_messages = [m for m in fake_client.calls[1]["messages"] if m["role"] == "tool"]
        assert tool_messages[0]["tool_name"] == "search_arxiv"
        assert "tool_call_id" not in tool_messages[0]

    @respx.mock
    async def test_needs_no_api_key(self, no_api_keys, monkeypatch):
        """Unlike every other provider, no key is required at all -- just a reachable server."""
        monkeypatch.delenv("OLLAMA_HOST", raising=False)

        final_reply = FakeChatResponse(FakeMessage(content="[]", tool_calls=None))
        fake_client = FakeOllamaClient([final_reply])
        monkeypatch.setattr("ollama.Client", lambda **kwargs: fake_client)

        agent = SearchAgent(provider="ollama", max_rounds=1)
        outcome = await agent.find_sources("anything")

        assert outcome.sources == []

    async def test_defaults_to_the_default_host_when_unset(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)

        captured = {}

        def fake_client_ctor(**kwargs):
            captured.update(kwargs)
            return FakeOllamaClient([FakeChatResponse(FakeMessage(content="[]", tool_calls=None))])

        monkeypatch.setattr("ollama.Client", fake_client_ctor)

        await SearchAgent(provider="ollama", max_rounds=1).find_sources("anything")

        assert captured["host"] == DEFAULT_OLLAMA_HOST

    async def test_uses_ollama_host_when_set(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://example.internal:11434")

        captured = {}

        def fake_client_ctor(**kwargs):
            captured.update(kwargs)
            return FakeOllamaClient([FakeChatResponse(FakeMessage(content="[]", tool_calls=None))])

        monkeypatch.setattr("ollama.Client", fake_client_ctor)

        await SearchAgent(provider="ollama", max_rounds=1).find_sources("anything")

        assert captured["host"] == "http://example.internal:11434"

    def test_defaults_to_the_claude_provider(self):
        assert SearchAgent().provider == "claude"
