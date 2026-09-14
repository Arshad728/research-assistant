"""Checking a finished report before anyone reads it (Phase 4.3).

The Writer Agent is the one agent in the system with an incentive to be
smooth. Chapter 4.5 puts it plainly: keeping the writer downstream of
verification "means the Writer Agent's only inputs are already-checked
findings, so a fluent writing style can never paper over an unverified
claim." That holds for the claims it is *given*. It does not, on its own,
stop a writer from adding a detail that no finding contained, attaching a
citation to the wrong source, or stating a figure it inferred rather than
read.

So the same pattern used in Phase 2 and Phase 3 applies once more: the
model writes, and code checks the properties that can be checked
mechanically. The strongest of these is the numeric check, which asks
whether every meaningful figure in the finished prose actually appears in
the evidence the writer was handed. A report is allowed to be eloquent. It
is not allowed to introduce a statistic from nowhere.
"""
from __future__ import annotations

import re
from typing import List, Literal, Sequence, Set

from pydantic import BaseModel, Field

from ..extraction.verify import numbers_in
from ..schemas import VerifiedFinding
from .models import CITATION_MARKER, FINDING_MARKER, ResearchReport

Severity = Literal["error", "warning"]

# A number followed by a percent sign always matters, however small.
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


class ReportIssue(BaseModel):
    check: str
    severity: Severity
    detail: str


class ReportValidation(BaseModel):
    issues: List[ReportIssue] = Field(default_factory=list)

    @property
    def errors(self) -> List[ReportIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> List[ReportIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def feedback_for_model(self) -> str:
        """The revision instructions sent back to the writer after a failed check."""
        if self.passed:
            return ""
        lines = ["Your draft has problems that must be fixed:"]
        lines.extend(f"- {issue.detail}" for issue in self.errors)
        lines.append(
            "Rewrite the report to fix every point above. Do not invent new evidence; "
            "use only the findings you were given, and cite them with [F<number>] markers."
        )
        return "\n".join(lines)

    def as_caveats(self) -> List[str]:
        """Remaining problems, phrased for a reader rather than for the model."""
        return [f"{issue.check}: {issue.detail}" for issue in self.issues]


def significant_numbers(text: str) -> Set[float]:
    """Numbers in a piece of prose that ought to be traceable to evidence.

    Small whole numbers are excluded, because a report legitimately says
    things like "three themes emerged" or "two of the four studies," and
    demanding a citation for those would produce constant false alarms
    without catching anything real. Percentages are never excluded, however
    small, because "output rose 8%" is exactly the kind of figure that must
    come from somewhere.

    This deliberately lets a fabricated small integer through. The
    finding-level check in Phase 3.3 is stricter and runs first; this is a
    second net, not the only one.
    """
    values = set()
    percent_values = {float(match) for match in _PERCENT.findall(text or "")}
    for value in numbers_in(text):
        if value in percent_values or not value.is_integer() or value >= 10:
            values.add(value)
    return values


def evidence_numbers(findings: Sequence[VerifiedFinding]) -> Set[float]:
    """Every number appearing anywhere in the verified evidence."""
    pool: Set[float] = set()
    for finding in findings:
        pool |= numbers_in(finding.claim)
        pool |= numbers_in(finding.supporting_snippet)
    return pool


def validate_report(
    report: ResearchReport, findings: Sequence[VerifiedFinding]
) -> ReportValidation:
    """Run every structural and factual check on a finished report."""
    issues: List[ReportIssue] = []
    body = report.full_text()

    # 1. No leftover finding markers the numbering step could not resolve.
    leftover = sorted({int(n) for n in FINDING_MARKER.findall(body)})
    if leftover:
        issues.append(
            ReportIssue(
                check="unresolved_markers",
                severity="error",
                detail=(
                    "The report cites finding(s) "
                    + ", ".join(f"F{n}" for n in leftover)
                    + f", but only {len(findings)} finding(s) were provided."
                ),
            )
        )

    # 2. Every citation number used must exist in the reference list.
    available = {citation.number for citation in report.citations}
    dangling = sorted(set(report.cited_numbers()) - available)
    if dangling:
        issues.append(
            ReportIssue(
                check="dangling_citations",
                severity="error",
                detail=(
                    "The report uses citation marker(s) "
                    + ", ".join(f"[{n}]" for n in dangling)
                    + " which have no matching entry in the reference list."
                ),
            )
        )

    # 3. Every reference must be a source that actually produced a finding.
    finding_urls = {finding.source_url for finding in findings}
    ungrounded = [c for c in report.citations if c.source_url not in finding_urls]
    if ungrounded:
        issues.append(
            ReportIssue(
                check="ungrounded_citations",
                severity="error",
                detail=(
                    "The reference list includes source(s) that produced no verified finding: "
                    + ", ".join(c.source_url for c in ungrounded)
                ),
            )
        )

    # 4. Every themed section must cite something.
    for section in report.sections:
        section_text = " ".join(section.paragraphs)
        if not CITATION_MARKER.search(section_text) and not FINDING_MARKER.search(section_text):
            issues.append(
                ReportIssue(
                    check="uncited_section",
                    severity="error",
                    detail=f'The section "{section.heading}" contains no citation at all.',
                )
            )

    # 5. Any paragraph stating a figure must say where the figure came from.
    for paragraph in report.body_paragraphs():
        if significant_numbers(paragraph) and not CITATION_MARKER.search(paragraph):
            issues.append(
                ReportIssue(
                    check="uncited_figure",
                    severity="error",
                    detail=(
                        "A paragraph states a figure with no citation: "
                        f'"{paragraph.strip()[:120]}..."'
                    ),
                )
            )

    # 6. Every meaningful figure must appear somewhere in the evidence.
    supported = evidence_numbers(findings)
    unsupported = sorted(significant_numbers(body) - supported)
    if unsupported:
        issues.append(
            ReportIssue(
                check="unsupported_figures",
                severity="error",
                detail=(
                    "The report states "
                    + ", ".join(_format(n) for n in unsupported)
                    + ", which appear in none of the verified findings."
                ),
            )
        )

    # 7. Advisory: evidence gathered but never used.
    cited_urls = {
        citation.source_url for citation in report.citations
        if citation.number in set(report.cited_numbers())
    }
    unused = sorted(finding_urls - cited_urls)
    if unused:
        issues.append(
            ReportIssue(
                check="unused_evidence",
                severity="warning",
                detail=(
                    f"{len(unused)} source(s) produced verified findings that the report never "
                    "cites. That may be a reasonable editorial choice, or evidence being "
                    "quietly dropped."
                ),
            )
        )

    # 8. Advisory: a report with no body at all.
    if not report.sections:
        issues.append(
            ReportIssue(
                check="empty_body",
                severity="error",
                detail="The report has no themed sections.",
            )
        )

    return ReportValidation(issues=issues)


def _format(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)
