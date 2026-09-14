"""Phase 2.3, offline half: the whole search path wired together.

Chapter 6.3's retrieval testing step asks for the Search Agent to be run on
sample queries and the results checked for relevance. Half of that needs the
real internet and a real model. The other half -- do the three tools, the
planner, the deduplicator and the reconciler actually fit together, and does
a sensible shortlist come out the other end -- can be checked completely
offline, which is what this file does.

The one thing simulated rather than run is the model's reply. That is
deliberate: the point of the reconciler is that the system does not have to
trust that reply, so testing against a hand-written one (including a
deliberately bad one) exercises more than a live run would.
"""
import json

import httpx
import pytest
import respx

from research_assistant.retrieval.arxiv import ARXIV_API_URL
from research_assistant.retrieval.semantic_scholar import SEMANTIC_SCHOLAR_SEARCH_URL
from research_assistant.retrieval.web_search import TAVILY_SEARCH_URL
from research_assistant.search.planner import SearchPlanner, deduplicate
from research_assistant.search.reconcile import reconcile_shortlist
from research_assistant.tools.search_tools import (
    search_arxiv_tool,
    search_semantic_scholar_tool,
    search_web_tool,
)

QUERY = "effect of a four day work week on productivity"

# An arXiv preprint that also appears in Semantic Scholar under the same
# arXiv id -- the realistic case that makes deduplication necessary.
ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.01234v1</id>
    <published>2024-01-02T18:00:00Z</published>
    <title>Shorter Working Weeks and Measured Output: A Multi-Site Trial</title>
    <summary>A multi-site trial of a four-day working week measuring productivity
      and total weekly output across eleven organisations.</summary>
    <author><name>A. Researcher</name></author>
    <arxiv:primary_category term="econ.GN" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""

S2_PAYLOAD = {
    "total": 2,
    "data": [
        {
            "title": "Shorter Working Weeks and Measured Output: A Multi-Site Trial",
            "abstract": "A multi-site trial of a four-day working week measuring productivity.",
            "url": None,
            "year": 2024,
            "venue": "",
            "citationCount": 12,
            "externalIds": {"ArXiv": "2401.01234"},
        },
        {
            "title": "The Four-Day Week: Assessing Global Trials",
            "abstract": "Assesses productivity outcomes from four-day week trials worldwide.",
            "url": "https://www.semanticscholar.org/paper/abc123",
            "year": 2025,
            "venue": "Journal of Work Studies",
            "citationCount": 143,
            "publicationTypes": ["JournalArticle"],
        },
    ],
}

TAVILY_PAYLOAD = {
    "query": QUERY,
    "results": [
        {
            "title": "What 61 companies learned from a four-day week trial",
            "url": "https://example.org/four-day-week-report",
            "content": "Report on a coordinated four-day week trial covering productivity, "
            "revenue and staff turnover across 61 organisations.",
            "score": 0.96,
        },
        {
            "title": "Opinion: the four-day week is a passing fad",
            "url": "https://example.com/opinion-column",
            "content": "A columnist argues the four-day week will not last.",
            "score": 0.44,
        },
    ],
}


@pytest.fixture
def all_apis_mocked(monkeypatch):
    from research_assistant import config

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    config.get_settings.cache_clear()

    with respx.mock:
        respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(200, text=ARXIV_FEED))
        respx.get(SEMANTIC_SCHOLAR_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=S2_PAYLOAD)
        )
        respx.post(TAVILY_SEARCH_URL).mock(return_value=httpx.Response(200, json=TAVILY_PAYLOAD))
        yield


async def _run_all_tools() -> list:
    """Call all three tools the way the agent would, and recover their results."""
    from research_assistant.agents.search_agent import _sources_from_tool_result

    class Block:
        def __init__(self, content):
            self.content = content

    collected = []
    for tool in (search_arxiv_tool, search_semantic_scholar_tool, search_web_tool):
        result = await tool.handler({"query": QUERY, "max_results": 5})
        assert not result.get("is_error"), result["content"][0]["text"]
        collected.extend(_sources_from_tool_result(Block(result["content"])))
    return collected


async def test_all_three_tools_return_usable_sources(all_apis_mocked):
    retrieved = await _run_all_tools()
    assert len(retrieved) == 5  # 1 arXiv + 2 Semantic Scholar + 2 web
    assert {s.source_type for s in retrieved} == {"preprint", "journal article", "web page"}


async def test_the_same_paper_from_two_apis_is_counted_once(all_apis_mocked):
    retrieved = deduplicate(await _run_all_tools())
    assert len(retrieved) == 4
    arxiv_titles = [s.title for s in retrieved if "Shorter Working Weeks" in s.title]
    assert len(arxiv_titles) == 1


async def test_planner_judges_the_combined_results_adequate(all_apis_mocked):
    planner = SearchPlanner(original_query=QUERY)
    assessment = planner.register_attempt(QUERY, "all_tools", deduplicate(await _run_all_tools()))

    assert assessment.verdict == "adequate"
    assert assessment.distinct_domains >= 2
    assert planner.should_continue() is False


async def test_a_realistic_model_selection_produces_a_clean_shortlist(all_apis_mocked):
    retrieved = deduplicate(await _run_all_tools())

    # What a well-behaved Search Agent would reply: the three substantive
    # sources kept, the opinion column dropped as commentary rather than
    # evidence.
    model_reply = json.dumps(
        [
            {
                "source_url": "https://www.semanticscholar.org/paper/abc123",
                "relevance_note": "Cross-country assessment of four-day week trials.",
            },
            {
                "source_url": "http://arxiv.org/abs/2401.01234v1",
                "relevance_note": "Multi-site trial measuring output directly.",
            },
            {
                "source_url": "https://example.org/four-day-week-report",
                "relevance_note": "Large coordinated trial with reported outcomes.",
            },
        ]
    )

    shortlist = reconcile_shortlist(model_reply, retrieved, max_items=10)

    assert len(shortlist.sources) == 3
    assert not shortlist.dropped_unknown_urls
    assert not shortlist.used_fallback
    assert "example.com/opinion-column" not in {s.source_url for s in shortlist.sources}
    # Notes are the model's judgement; titles and types come from the tools.
    assert shortlist.sources[0].source_type == "journal article"
    assert shortlist.sources[0].relevance_note.startswith("Cross-country")


async def test_an_invented_url_never_reaches_the_shortlist(all_apis_mocked):
    retrieved = deduplicate(await _run_all_tools())

    model_reply = json.dumps(
        [
            {"source_url": "https://www.semanticscholar.org/paper/abc123", "relevance_note": "real"},
            {
                "source_url": "https://journals.example.net/2024/four-day-week-metastudy",
                "relevance_note": "a plausible-looking paper that was never retrieved",
            },
        ]
    )

    shortlist = reconcile_shortlist(model_reply, retrieved)

    assert len(shortlist.sources) == 1
    assert shortlist.dropped_unknown_urls == [
        "https://journals.example.net/2024/four-day-week-metastudy"
    ]


async def test_a_failing_tool_does_not_stop_the_others(monkeypatch):
    from research_assistant import config
    from research_assistant.agents.search_agent import _sources_from_tool_result

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    config.get_settings.cache_clear()

    class Block:
        def __init__(self, content):
            self.content = content

    with respx.mock:
        respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(503, text="unavailable"))
        respx.post(TAVILY_SEARCH_URL).mock(return_value=httpx.Response(200, json=TAVILY_PAYLOAD))

        failed = await search_arxiv_tool.handler({"query": QUERY})
        succeeded = await search_web_tool.handler({"query": QUERY})

    assert failed["is_error"] is True
    assert not succeeded.get("is_error")
    assert len(_sources_from_tool_result(Block(succeeded["content"]))) == 2
