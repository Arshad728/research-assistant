"""Structured data shapes for agent hand-offs.

These classes are the concrete form of the JSON examples in Chapter 4 of the
book ("How the agents communicate: structured hand-offs in this system") and
the shared-state idea from Chapter 3.2. Every later phase imports from here
rather than each agent inventing its own shape for the data it passes along.

A note on "confidence": Chapter 6's Phase 1.3 table describes the shared
schema loosely as "claim, snippet, source, and confidence." Chapter 4's own
worked example is more concrete and uses a boolean ``verified`` flag instead
of a graded score, because the Extraction and Verification Agent's whole
design is a binary gate: a claim is either matched back to its source
closely enough to trust, or it is flagged/discarded (Chapter 4.4). This
module follows Chapter 4: ``verified`` is the confidence signal for v1. A
graded, numeric confidence score is a reasonable future extension (Chapter
8.1) but is not needed for the architecture as designed.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


def _looks_like_url(value: str) -> str:
    if not (value.startswith("http://") or value.startswith("https://")):
        raise ValueError(f"expected an http(s) URL, got: {value!r}")
    return value


class CandidateSource(BaseModel):
    """One entry in the Search Agent's shortlist (Chapter 4, first hand-off).

    Produced by the Search Agent, consumed by the Extraction and
    Verification Agent, by way of the Orchestrator.
    """

    title: str
    source_url: str
    source_type: str = Field(
        description='e.g. "peer-reviewed study", "preprint", "news article", "blog post"'
    )
    relevance_note: str = Field(
        description="A short, human-readable reason this source looks relevant to the query."
    )

    @field_validator("source_url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        return _looks_like_url(value)


class VerifiedFinding(BaseModel):
    """One verified fact (Chapter 4, second hand-off).

    Produced by the Extraction and Verification Agent, consumed by the
    Writer Agent, by way of the Orchestrator. ``verified`` should only ever
    be True here -- a claim that fails the check against
    ``supporting_snippet`` is meant to be discarded before it is ever
    constructed as a VerifiedFinding, not represented with verified=False
    (Chapter 4.4).
    """

    claim: str
    supporting_snippet: str = Field(
        description="The exact snippet of source text the claim rests on."
    )
    source_url: str
    verified: bool = True

    @field_validator("source_url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        return _looks_like_url(value)


class SharedState(BaseModel):
    """The Orchestrator's running record for one query (Chapters 3.2 and 4.6).

    Deliberately simple in Phase 1.3: enough structure for every later agent
    to read and update the same record, without yet building the
    sequencing/stopping logic that uses it -- that is Phase 5.
    """

    original_query: str
    candidate_sources: List[CandidateSource] = Field(default_factory=list)
    verified_findings: List[VerifiedFinding] = Field(default_factory=list)
    search_rounds_completed: int = 0
    notes: List[str] = Field(default_factory=list)

    def add_search_round(self, sources: List[CandidateSource]) -> None:
        """Record one round of the Search Agent's results (Chapter 4.7)."""
        self.candidate_sources.extend(sources)
        self.search_rounds_completed += 1

    def add_verified_findings(self, findings: List[VerifiedFinding]) -> None:
        """Record output from the Extraction and Verification Agent."""
        self.verified_findings.extend(findings)

    def distinct_sources_backing_findings(self) -> int:
        """How many distinct sources the current verified findings draw from.

        Chapter 4.7's worked example uses exactly this number ("four solid,
        verified findings ... drawn from three different sources") as part of
        deciding whether to keep searching -- Phase 5.2 builds on this.
        """
        return len({finding.source_url for finding in self.verified_findings})
