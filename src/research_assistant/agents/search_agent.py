"""The Search Agent (Chapter 4.3).

This module is the thin layer where the language model actually enters the
picture. Everything it depends on has been built and tested separately: the
three retrieval tools (Phase 2.1), and the planner and reconciler that keep
the loop honest (Phase 2.2).

The division of labour is deliberate and is the same one Decision Record
0001 argued for:

* The **model** decides which tool to reach for, how to phrase each query,
  how to rephrase after a thin round, and which retrieved sources are worth
  passing on. That is judgement, and it is what a language model is for.
* The **code** remembers what has already been tried, refuses a repeated
  query, decides when a result set is thin, caps the number of rounds, and
  rebuilds the final shortlist from the tool records rather than from the
  model's prose. That is bookkeeping, and it behaves the same way every run.

Running this needs a real ANTHROPIC_API_KEY and outbound access to the
retrieval APIs. Everything underneath it does not, which is why the test
suite can cover the behaviour that matters without either.
"""
from __future__ import annotations

import json
from typing import List, Optional

from pydantic import BaseModel, Field

from ..schemas import CandidateSource
from ..search.planner import SearchPlanner, deduplicate
from ..search.reconcile import Shortlist, reconcile_shortlist
from ..tools.search_tools import (
    ALLOWED_SEARCH_TOOL_NAMES,
    SEARCH_TOOL_SERVER_NAME,
    create_search_tool_server,
)

SEARCH_AGENT_SYSTEM_PROMPT = """\
You are the Search Agent in a multi-agent research system. Your only job is to
find real, relevant candidate sources for a research question. You do not
summarise findings, you do not answer the question, and you never write prose
conclusions -- later agents do that.

You have three tools:
- search_arxiv: preprints, strongest for recent computer science, physics, maths.
- search_semantic_scholar: published academic work across all fields, with
  citation counts that indicate how established a result is.
- search_web: everything else -- current events, industry and government
  reports, organisations' own publications.

How to work:
1. Decide which tool or tools fit the question. Most questions deserve more
   than one; an academic question is rarely answered well by web search alone,
   and a question about current practice is rarely answered well by papers
   alone.
2. Run a search. Read what came back.
3. If a round is thin or off-topic, REFORMULATE - change the wording, change
   the keywords, or switch tools. Never re-run a query you have already run;
   it will return the same thing.
4. Judge the results. Drop anything clearly irrelevant, or that is commentary
   about research rather than research itself, when better material exists.

When you have finished searching, reply with ONLY a JSON array of the sources
worth passing on, in this exact form:

[
  {"source_url": "<the exact URL as returned by the tool>",
   "relevance_note": "<one sentence on why this source bears on the question>"}
]

Rules for that final reply:
- Every source_url MUST be one a tool actually returned to you in this
  conversation. Do not adjust, shorten, or reconstruct URLs, and never include
  a source you did not retrieve. Anything else is discarded.
- Order them best-first.
- If nothing relevant was found, reply with an empty array: []
"""


class SearchOutcome(BaseModel):
    """Everything one Search Agent run produced, including how it got there."""

    query: str
    sources: List[CandidateSource] = Field(default_factory=list)
    retrieved_count: int = 0
    queries_tried: List[str] = Field(default_factory=list)
    dropped_unknown_urls: List[str] = Field(default_factory=list)
    stop_reason: str = ""
    used_fallback: bool = False

    @property
    def shortlisted_count(self) -> int:
        return len(self.sources)


class SearchAgent:
    """Turns a research sub-task into a shortlist of real candidate sources."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        max_rounds: int = 3,
        max_shortlist: int = 10,
    ) -> None:
        self.model = model
        self.max_rounds = max_rounds
        self.max_shortlist = max_shortlist

    def _build_options(self):
        # Imported here so that importing this module never requires the SDK
        # to be importable in environments that only run the unit tests.
        from claude_agent_sdk import ClaudeAgentOptions

        return ClaudeAgentOptions(
            system_prompt=SEARCH_AGENT_SYSTEM_PROMPT,
            mcp_servers={SEARCH_TOOL_SERVER_NAME: create_search_tool_server()},
            allowed_tools=list(ALLOWED_SEARCH_TOOL_NAMES),
            permission_mode="bypassPermissions",
            model=self.model,
            max_turns=self.max_rounds * 3,
        )

    async def find_sources(self, query: str) -> SearchOutcome:
        """Search for sources answering ``query`` and return a checked shortlist."""
        from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, TextBlock, ToolResultBlock

        planner = SearchPlanner(original_query=query, max_rounds=self.max_rounds)
        retrieved: List[CandidateSource] = []
        reply_text: List[str] = []

        prompt = (
            f"Research question: {query}\n\n"
            f"Find the most relevant real sources. You may search up to {self.max_rounds} "
            "times. Reply with the JSON array described in your instructions when done."
        )

        from ..tools.search_tools import reset_query_history, start_query_history

        # Repeated queries are refused for the duration of this run. See the
        # note in search_tools.py: the instruction not to repeat lives in the
        # prompt, but the guarantee lives here.
        history_token = start_query_history()
        try:
            async with ClaudeSDKClient(options=self._build_options()) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    for block in getattr(message, "content", []) or []:
                        if isinstance(block, ToolResultBlock):
                            retrieved.extend(_sources_from_tool_result(block))
                        elif isinstance(message, AssistantMessage) and isinstance(block, TextBlock):
                            reply_text.append(block.text)
        finally:
            reset_query_history(history_token)

        retrieved = deduplicate(retrieved)
        shortlist: Shortlist = reconcile_shortlist(
            "\n".join(reply_text), retrieved, max_items=self.max_shortlist
        )

        planner.register_attempt(query, "agent", shortlist.sources)

        return SearchOutcome(
            query=query,
            sources=shortlist.sources,
            retrieved_count=len(retrieved),
            queries_tried=planner.tried_queries(),
            dropped_unknown_urls=shortlist.dropped_unknown_urls,
            stop_reason=planner.stop_reason(),
            used_fallback=shortlist.used_fallback,
        )


def _sources_from_tool_result(block) -> List[CandidateSource]:
    """Recover the structured results a search tool returned.

    Tool results arrive as the JSON text the tool produced. Parsing it back
    is what lets the agent's final shortlist be rebuilt from real retrievals
    instead of from the model's description of them.
    """
    content = getattr(block, "content", None)
    texts: List[str] = []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)

    sources: List[CandidateSource] = []
    for text in texts:
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        for record in payload.get("results") or []:
            if not isinstance(record, dict):
                continue
            try:
                sources.append(CandidateSource(**record))
            except Exception:  # noqa: BLE001 - a malformed record is skipped, not fatal
                continue
    return sources
