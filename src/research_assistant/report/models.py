"""The shape of a finished report (Phase 4.1).

Chapter 4.5 sets the structure: "an introduction, findings organized by
theme, a conclusion, and a reference list, with every claim in the body
linked back to the source that supports it."

Two decisions here are worth explaining.

The report is a structured object, not a string of Markdown. Markdown and
PDF are both rendered *from* this object, which means the two formats can
never disagree, and -- more importantly -- the validation in validate.py
can inspect the report's parts rather than trying to parse prose back into
structure.

Citations are numbered per *source*, not per finding. Three findings drawn
from one paper share one reference number, which is how a reader expects a
reference list to behave. The Writer Agent works in finding numbers instead
(``[F1]``), and the conversion happens here, where it can be checked.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from ..schemas import CandidateSource, VerifiedFinding

FINDING_MARKER = re.compile(r"\[F(\d+)\]")
CITATION_MARKER = re.compile(r"\[(\d+)\]")


class Citation(BaseModel):
    """One entry in the reference list."""

    number: int
    source_url: str
    title: str = ""
    source_type: str = ""

    def to_markdown(self) -> str:
        label = self.title or self.source_url
        kind = f" ({self.source_type})" if self.source_type else ""
        return f"[{self.number}] {label}{kind}. <{self.source_url}>"


class ReportSection(BaseModel):
    """One themed section of the body."""

    heading: str
    paragraphs: List[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """A finished, cited report."""

    question: str
    introduction: str = ""
    sections: List[ReportSection] = Field(default_factory=list)
    conclusion: str = ""
    citations: List[Citation] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)

    def body_paragraphs(self) -> List[str]:
        """Every paragraph of prose in the report, in order."""
        paragraphs = [self.introduction] if self.introduction else []
        for section in self.sections:
            paragraphs.extend(p for p in section.paragraphs if p.strip())
        if self.conclusion:
            paragraphs.append(self.conclusion)
        return paragraphs

    def full_text(self) -> str:
        return "\n\n".join(self.body_paragraphs())

    def cited_numbers(self) -> List[int]:
        """Every citation number referenced anywhere in the body."""
        return sorted({int(n) for n in CITATION_MARKER.findall(self.full_text())})

    def to_markdown(self) -> str:
        lines: List[str] = [f"# {self.question}", ""]

        if self.introduction:
            lines.extend([self.introduction, ""])

        for section in self.sections:
            lines.extend([f"## {section.heading}", ""])
            for paragraph in section.paragraphs:
                lines.extend([paragraph, ""])

        if self.conclusion:
            lines.extend(["## Conclusion", "", self.conclusion, ""])

        if self.citations:
            lines.extend(["## References", ""])
            lines.extend(citation.to_markdown() for citation in self.citations)
            lines.append("")

        if self.caveats:
            lines.extend(["## Verification notes", ""])
            lines.extend(f"- {caveat}" for caveat in self.caveats)
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


def build_citations(
    findings: Sequence[VerifiedFinding],
    sources: Optional[Sequence[CandidateSource]] = None,
) -> tuple[List[Citation], Dict[int, int]]:
    """Number the sources, and map each finding to its source's number.

    Returns the reference list and a mapping from one-based finding number
    to citation number, which is what turns the Writer Agent's ``[F3]``
    into the reader's ``[2]``.
    """
    titles = {source.source_url: source for source in (sources or [])}

    citations: List[Citation] = []
    number_for_url: Dict[str, int] = {}
    finding_to_citation: Dict[int, int] = {}

    for index, finding in enumerate(findings, start=1):
        url = finding.source_url
        if url not in number_for_url:
            number_for_url[url] = len(citations) + 1
            source = titles.get(url)
            citations.append(
                Citation(
                    number=number_for_url[url],
                    source_url=url,
                    title=source.title if source else "",
                    source_type=source.source_type if source else "",
                )
            )
        finding_to_citation[index] = number_for_url[url]

    return citations, finding_to_citation


def convert_finding_markers(text: str, finding_to_citation: Dict[int, int]) -> tuple[str, List[int]]:
    """Rewrite ``[F3]`` markers as reader-facing ``[2]`` citation numbers.

    Any marker pointing at a finding that does not exist is left in place
    rather than quietly deleted, and reported back to the caller. Silently
    removing it would turn "this sentence cites evidence that was never
    given to the writer" into "this sentence has no citation," which is the
    same problem wearing a disguise.
    """
    unknown: List[int] = []

    def replace(match: re.Match) -> str:
        finding_number = int(match.group(1))
        citation_number = finding_to_citation.get(finding_number)
        if citation_number is None:
            unknown.append(finding_number)
            return match.group(0)
        return f"[{citation_number}]"

    return FINDING_MARKER.sub(replace, text), unknown
