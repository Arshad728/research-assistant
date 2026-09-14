"""Tests for the parts of the Search Agent that do not need a language model.

Running the agent itself needs an API key and network access. What can be
tested here without either is the piece that matters most for correctness:
recovering the real retrieved sources out of tool results, which is what the
final shortlist is rebuilt from.
"""
import json

from research_assistant.agents.search_agent import (
    SEARCH_AGENT_SYSTEM_PROMPT,
    SearchOutcome,
    _sources_from_tool_result,
)
from research_assistant.schemas import CandidateSource


class FakeToolResultBlock:
    """Stands in for the SDK's ToolResultBlock, which carries tool output."""

    def __init__(self, content):
        self.content = content


TOOL_PAYLOAD = {
    "source": "arxiv",
    "result_count": 2,
    "results": [
        {
            "title": "Shorter Working Weeks and Measured Output",
            "source_url": "http://arxiv.org/abs/2401.01234v1",
            "source_type": "preprint",
            "relevance_note": "A. Researcher | 2024 | econ.GN.",
        },
        {
            "title": "Scheduling Effects on Knowledge Work",
            "source_url": "http://arxiv.org/abs/2402.05678v2",
            "source_type": "preprint",
            "relevance_note": "C. First et al. | 2024 | cs.CY.",
        },
    ],
}


class TestRecoveringToolResults:
    def test_reads_sources_from_a_list_of_text_blocks(self):
        block = FakeToolResultBlock([{"type": "text", "text": json.dumps(TOOL_PAYLOAD)}])
        sources = _sources_from_tool_result(block)

        assert len(sources) == 2
        assert all(isinstance(s, CandidateSource) for s in sources)
        assert sources[0].source_url == "http://arxiv.org/abs/2401.01234v1"

    def test_reads_sources_from_plain_string_content(self):
        block = FakeToolResultBlock(json.dumps(TOOL_PAYLOAD))
        assert len(_sources_from_tool_result(block)) == 2

    def test_ignores_an_error_result(self):
        block = FakeToolResultBlock([{"type": "text", "text": "SEARCH ERROR: HTTP 403 ..."}])
        assert _sources_from_tool_result(block) == []

    def test_ignores_a_malformed_record_without_losing_the_good_ones(self):
        payload = {
            "results": [
                TOOL_PAYLOAD["results"][0],
                {"title": "No URL here", "source_type": "preprint", "relevance_note": "x"},
                {"title": "Bad URL", "source_url": "not-a-url", "source_type": "x", "relevance_note": "y"},
            ]
        }
        block = FakeToolResultBlock([{"type": "text", "text": json.dumps(payload)}])
        sources = _sources_from_tool_result(block)

        assert len(sources) == 1
        assert sources[0].title == "Shorter Working Weeks and Measured Output"

    def test_handles_empty_or_missing_content(self):
        assert _sources_from_tool_result(FakeToolResultBlock(None)) == []
        assert _sources_from_tool_result(FakeToolResultBlock([])) == []


class TestSystemPrompt:
    def test_forbids_inventing_urls(self):
        assert "MUST be one a tool actually returned" in SEARCH_AGENT_SYSTEM_PROMPT

    def test_forbids_repeating_a_query(self):
        assert "Never re-run a query" in SEARCH_AGENT_SYSTEM_PROMPT

    def test_keeps_the_agent_in_its_lane(self):
        # Chapter 4.3: the Search Agent finds sources; it does not draw conclusions.
        assert "you do not answer the question" in SEARCH_AGENT_SYSTEM_PROMPT


class TestSearchOutcome:
    def test_reports_how_much_survived(self):
        outcome = SearchOutcome(
            query="four day week",
            sources=[CandidateSource(**TOOL_PAYLOAD["results"][0])],
            retrieved_count=9,
            dropped_unknown_urls=["https://example.net/invented"],
        )
        assert outcome.shortlisted_count == 1
        assert outcome.retrieved_count == 9
        assert len(outcome.dropped_unknown_urls) == 1
