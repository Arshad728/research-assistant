"""Tests for the Phase 1.3 shared schema.

The first two tests parse the exact JSON examples printed in Chapter 4 of
the book, so a change here that breaks compatibility with the book is
caught immediately, not discovered later while building an agent.
"""
import pytest
from pydantic import ValidationError

from research_assistant.schemas import CandidateSource, SharedState, VerifiedFinding

# Copied verbatim from Chapter 4, "How the agents communicate."
BOOK_CANDIDATE_SOURCE_EXAMPLE = {
    "title": "The Four-Day Week: Assessing Global Trials",
    "source_url": "https://example.org/four-day-week-trial-2025",
    "source_type": "peer-reviewed study",
    "relevance_note": (
        "Reports a controlled trial measuring output before and after "
        "switching to a four-day schedule."
    ),
}

BOOK_VERIFIED_FINDING_EXAMPLE = {
    "claim": (
        "Output per hour rose by approximately 8% after the switch, while "
        "total weekly output stayed roughly level."
    ),
    "supporting_snippet": (
        "...hourly productivity increased by 8% during the trial period, "
        "while total weekly output remained statistically unchanged..."
    ),
    "source_url": "https://example.org/four-day-week-trial-2025",
    "verified": True,
}


def test_candidate_source_matches_book_example():
    source = CandidateSource(**BOOK_CANDIDATE_SOURCE_EXAMPLE)
    assert source.title == "The Four-Day Week: Assessing Global Trials"
    # round-trips back to the same shape
    assert CandidateSource(**source.model_dump()) == source


def test_verified_finding_matches_book_example():
    finding = VerifiedFinding(**BOOK_VERIFIED_FINDING_EXAMPLE)
    assert finding.verified is True
    assert finding.source_url == "https://example.org/four-day-week-trial-2025"


def test_verified_finding_defaults_to_verified_true():
    finding = VerifiedFinding(
        claim="x",
        supporting_snippet="y",
        source_url="https://example.org/z",
    )
    assert finding.verified is True


@pytest.mark.parametrize("model_cls, extra_fields", [
    (CandidateSource, {"title": "t", "source_type": "blog post", "relevance_note": "n"}),
    (VerifiedFinding, {"claim": "c", "supporting_snippet": "s"}),
])
def test_rejects_non_url_source(model_cls, extra_fields):
    with pytest.raises(ValidationError):
        model_cls(source_url="not-a-url", **extra_fields)


def test_shared_state_accumulates_search_rounds():
    state = SharedState(original_query="four-day work week productivity")
    state.add_search_round([CandidateSource(**BOOK_CANDIDATE_SOURCE_EXAMPLE)])
    assert state.search_rounds_completed == 1
    assert len(state.candidate_sources) == 1

    state.add_search_round([CandidateSource(**BOOK_CANDIDATE_SOURCE_EXAMPLE)])
    assert state.search_rounds_completed == 2
    assert len(state.candidate_sources) == 2


def test_distinct_sources_backing_findings_counts_unique_urls():
    state = SharedState(original_query="four-day work week productivity")
    same_source_twice = [
        VerifiedFinding(**BOOK_VERIFIED_FINDING_EXAMPLE),
        VerifiedFinding(
            claim="A second, different claim from the same paper.",
            supporting_snippet="...a different snippet from the same source...",
            source_url=BOOK_VERIFIED_FINDING_EXAMPLE["source_url"],
        ),
    ]
    state.add_verified_findings(same_source_twice)
    assert state.distinct_sources_backing_findings() == 1

    state.add_verified_findings([
        VerifiedFinding(
            claim="A claim from a second, independent source.",
            supporting_snippet="...snippet from a different paper...",
            source_url="https://example.org/a-different-study-2025",
        )
    ])
    assert state.distinct_sources_backing_findings() == 2
