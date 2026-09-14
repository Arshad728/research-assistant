"""The Writer Agent (Chapter 4.5).

The last agent in the pipeline, and the only one that produces prose. It
receives verified findings, groups them by theme, and drafts a report in
which every factual sentence carries a citation.

The loop it runs is worth describing, because it is the reason this agent
does not need to be trusted. The model drafts; ``validate_report`` checks
properties that can be checked mechanically; if anything failed, the
specific failures are handed back to the model as revision instructions and
it drafts again. After the revision budget runs out, whatever still fails is
attached to the report as visible verification notes rather than quietly
shipped.

That last part is the important one. A report that silently omits a problem
looks exactly like a report that had none. Chapter 7.3's warning -- that a
well-written report "will read fluently and assuredly regardless of how much
or how little evidence actually supported it" -- applies to this agent's own
output, so the honest thing is to print what could not be confirmed on the
document itself.
"""
from __future__ import annotations

from typing import Awaitable, Callable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from ..report.models import ResearchReport, build_citations, convert_finding_markers
from ..report.synthesis import (
    WRITER_SYSTEM_PROMPT,
    build_writer_prompt,
    parse_report_reply,
)
from ..report.validate import ReportValidation, validate_report
from ..schemas import CandidateSource, VerifiedFinding

CompletionFn = Callable[[str, str], Awaitable[str]]


class WriterOutcome(BaseModel):
    """A finished report plus an account of how it got there."""

    report: ResearchReport
    validation: ReportValidation
    revisions_used: int = 0
    draft_failed: bool = Field(
        default=False,
        description="True when no usable draft could be parsed from the model at all.",
    )

    @property
    def passed_validation(self) -> bool:
        return self.validation.passed


class WriterAgent:
    """Turns verified findings into a structured, cited report."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        complete: Optional[CompletionFn] = None,
        max_revisions: int = 1,
    ) -> None:
        self.model = model
        self.max_revisions = max_revisions
        if complete is not None:
            self._complete = complete
        else:
            from .extraction_agent import complete_with_sdk

            self._complete = lambda system, user: complete_with_sdk(system, user, model=model)

    def _assemble(
        self,
        question: str,
        drafted: dict,
        findings: Sequence[VerifiedFinding],
        sources: Optional[Sequence[CandidateSource]],
    ) -> ResearchReport:
        """Turn the model's draft into a report with a real reference list."""
        citations, finding_to_citation = build_citations(findings, sources)

        def convert(text: str) -> str:
            converted, _ = convert_finding_markers(text, finding_to_citation)
            return converted

        sections = [
            {
                "heading": section["heading"],
                "paragraphs": [convert(p) for p in section["paragraphs"]],
            }
            for section in drafted["sections"]
        ]

        report = ResearchReport(
            question=question,
            introduction=convert(drafted["introduction"]),
            sections=sections,
            conclusion=convert(drafted["conclusion"]),
            citations=citations,
        )

        # Drop reference entries the finished prose never actually cites, so
        # the reference list describes the report rather than the research.
        used = set(report.cited_numbers())
        if used:
            report.citations = [c for c in citations if c.number in used]
        return report

    async def write_report(
        self,
        question: str,
        findings: Sequence[VerifiedFinding],
        sources: Optional[Sequence[CandidateSource]] = None,
    ) -> WriterOutcome:
        """Draft, check, revise if needed, and return the report with its verdict."""
        if not findings:
            empty = ResearchReport(
                question=question,
                introduction=(
                    "No verified findings were available for this question, so no report "
                    "could be written. This is a statement about the evidence gathered, "
                    "not about the question."
                ),
                caveats=["No sources produced a claim that passed verification."],
            )
            return WriterOutcome(
                report=empty,
                validation=ReportValidation(issues=[]),
                draft_failed=True,
            )

        base_prompt = build_writer_prompt(question, findings)
        prompt = base_prompt
        report: Optional[ResearchReport] = None
        validation = ReportValidation(issues=[])
        revisions_used = 0

        for attempt in range(self.max_revisions + 1):
            reply = await self._complete(WRITER_SYSTEM_PROMPT, prompt)
            drafted = parse_report_reply(reply)

            if drafted is None:
                if attempt >= self.max_revisions:
                    break
                revisions_used += 1
                prompt = (
                    f"{base_prompt}\n\nYour previous reply could not be read as the required "
                    "JSON object. Reply with the JSON object only, and nothing else."
                )
                continue

            report = self._assemble(question, drafted, findings, sources)
            validation = validate_report(report, findings)
            if validation.passed or attempt >= self.max_revisions:
                break

            revisions_used += 1
            prompt = f"{base_prompt}\n\n{validation.feedback_for_model()}"

        if report is None:
            failed = ResearchReport(
                question=question,
                introduction=(
                    "The writer could not produce a usable draft for this question. The "
                    "verified findings are unaffected and can be written up by hand or by "
                    "re-running this step."
                ),
                caveats=["The Writer Agent's reply could not be read as a structured report."],
            )
            return WriterOutcome(
                report=failed,
                validation=ReportValidation(issues=[]),
                revisions_used=revisions_used,
                draft_failed=True,
            )

        # Anything still unresolved is printed on the report itself.
        report.caveats = validation.as_caveats()
        return WriterOutcome(report=report, validation=validation, revisions_used=revisions_used)


def save_report(report: ResearchReport, stem: str) -> Tuple[str, str]:
    """Write the report as Markdown and PDF. Returns both paths."""
    from ..report.render_pdf import render_report_pdf

    markdown_path = f"{stem}.md"
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(report.to_markdown())

    pdf_path = render_report_pdf(report, f"{stem}.pdf")
    return markdown_path, pdf_path
