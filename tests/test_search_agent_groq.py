"""Tests for the Search Agent's Groq function-calling path.

Groq's client is OpenAI-compatible: the assistant message carries
``tool_calls`` and the caller is expected to run them and send back a
``role: "tool"`` message per call, rather than the model running its own
loop internally the way the Claude SDK does. This mirrors
``test_search_agent_gemini.py`` -- a fake ``groq.Groq`` client plays back a
scripted conversation (one tool call, then a final plain-text reply), which
checks the hand-rolled loop's plumbing (accumulating messages, dispatching
tool calls through ``run_search_tool``, stopping once the model replies with
no further calls) without any network or a real API key.

``TestToolSpecs`` and ``TestRunSearchTool`` are provider-agnostic and are
already covered in ``test_search_agent_gemini.py``; this file only adds the
Groq-specific loop and provider-selection tests.
"""
import json

import pytest

from research_assistant.agents.search_agent import SearchAgent
from research_assistant.config import MissingAPIKeyError
from research_assistant.retrieval.arxiv import ARXIV_API_URL
import httpx
import respx

from .fixtures import ARXIV_ATOM_RESPONSE


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.type = "function"
        self.function = FakeFunction(name, arguments)

    def model_dump(self):
        return {
            "id": self.id,
            "type": self.type,
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeCompletionResponse:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]


class FakeCompletions:
    """Stands in for ``client.chat.completions``, playing back fixed replies."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, *, model, messages, tools):
        self.calls.append({"model": model, "messages": [dict(m) for m in messages], "tools": tools})
        return self._responses.pop(0)


class FakeChat:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)


class FakeGroqClient:
    def __init__(self, responses):
        self.chat = FakeChat(responses)


class TestFindSourcesGroq:
    @respx.mock
    async def test_drives_the_tool_loop_and_reconciles_the_shortlist(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")

        tool_call_reply = FakeCompletionResponse(
            FakeMessage(
                content=None,
                tool_calls=[
                    FakeToolCall(
                        "call_1",
                        "search_arxiv",
                        json.dumps({"query": "four day work week", "max_results": 5}),
                    )
                ],
            )
        )
        final_reply = FakeCompletionResponse(
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
        fake_client = FakeGroqClient([tool_call_reply, final_reply])
        monkeypatch.setattr("groq.Groq", lambda **kwargs: fake_client)
        respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(200, text=ARXIV_ATOM_RESPONSE))

        agent = SearchAgent(provider="groq", max_rounds=2)
        outcome = await agent.find_sources("four day work week")

        assert len(fake_client.chat.completions.calls) == 2
        # Both arXiv entries were retrieved by the tool call...
        assert outcome.retrieved_count == 2
        # ...but only the one URL the model actually named survives reconciliation.
        assert outcome.shortlisted_count == 1
        assert outcome.sources[0].source_url == "http://arxiv.org/abs/2401.01234v1"
        assert not outcome.dropped_unknown_urls

        # The tool result was sent back as a "tool" message keyed to the call id.
        tool_messages = [m for m in fake_client.chat.completions.calls[1]["messages"] if m["role"] == "tool"]
        assert tool_messages[0]["tool_call_id"] == "call_1"

    async def test_needs_a_groq_key(self, no_api_keys):
        agent = SearchAgent(provider="groq")
        with pytest.raises(MissingAPIKeyError, match="GROQ_API_KEY"):
            await agent.find_sources("anything")

    def test_defaults_to_the_claude_provider(self):
        assert SearchAgent().provider == "claude"
