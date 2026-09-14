"""The test query set (Phase 6.1).

Chapter 7.1 suggests picking "a handful of test queries spanning different
topics and difficulty levels." The set below is chosen so that failures are
informative rather than merely disappointing: each query stresses a
different part of the system, and knowing which ones fail says something
specific about what is wrong.

A query that returns nothing is not automatically a failure. Two of these
are included precisely because a well-behaved system should struggle with
them, and should say so rather than inventing an answer. An evaluation that
only uses easy questions measures nothing except whether the easy path
works.
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel

Difficulty = Literal["straightforward", "moderate", "hard", "should_find_little"]


class EvalQuery(BaseModel):
    """One evaluation query, and what it is meant to test."""

    id: str
    question: str
    difficulty: Difficulty
    tests: str
    expect_findings: bool = True


TEST_QUERIES: List[EvalQuery] = [
    EvalQuery(
        id="four_day_week",
        question="What does current research say about the effect of a four-day work week "
        "on productivity?",
        difficulty="straightforward",
        tests="The ordinary case: a well-studied question with both academic and "
        "general-web coverage. If this fails, something is broken rather than hard.",
    ),
    EvalQuery(
        id="rag_hallucination",
        question="How effective is retrieval-augmented generation at reducing hallucination "
        "in large language models?",
        difficulty="straightforward",
        tests="Heavy arXiv coverage and almost no general-web coverage. Exercises the "
        "academic retrieval path in isolation.",
    ),
    EvalQuery(
        id="remote_work_wages",
        question="Does remote work reduce wage growth for employees who never return to "
        "the office?",
        difficulty="moderate",
        tests="A question where the honest answer is mixed. Tests whether the Writer "
        "Agent reports disagreement instead of averaging it away.",
    ),
    EvalQuery(
        id="ai_regulation_transparency",
        question="What transparency reporting does current AI regulation require of model "
        "developers?",
        difficulty="moderate",
        tests="Moves quickly and lives in policy documents rather than papers. Tests "
        "whether web retrieval carries a query that academic search cannot.",
    ),
    EvalQuery(
        id="microdosing_productivity",
        question="What is the measured effect of microdosing psilocybin on software "
        "engineers' debugging accuracy?",
        difficulty="should_find_little",
        tests="Deliberately over-specific: plausible-sounding, almost certainly unstudied. "
        "A system that produces a confident report here is fabricating. The correct "
        "outcome is few or no verified findings and a report that says so.",
        expect_findings=False,
    ),
    EvalQuery(
        id="nonexistent_framework",
        question="What are the documented performance characteristics of the Zephyrine "
        "consensus protocol in distributed databases?",
        difficulty="should_find_little",
        tests="Names something that does not exist. Tests the strongest failure mode in "
        "the system: whether it will invent sources for an invented subject.",
        expect_findings=False,
    ),
]


def query_by_id(query_id: str) -> EvalQuery:
    for query in TEST_QUERIES:
        if query.id == query_id:
            return query
    raise KeyError(f"No test query with id {query_id!r}")
